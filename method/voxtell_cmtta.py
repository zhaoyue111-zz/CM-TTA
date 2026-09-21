"""Original CM-TTA adapted to VoxTell's frozen 3-D network.

The only model-specific changes are the VoxTell text-embedding interface and
the extension of CAC, entropy, and soft Dice from 2-D to 3-D tensors. LSPM and
DSPU follow CM-TTA equations (3)--(8): one optimizer update is made per full
case, after losses from all of that case's patches have been accumulated.
"""

from __future__ import annotations

import warnings
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


EPS = 1e-8
TDC_DECODER_PAIRS = ((0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3))


def avg_entropy(
    probabilities: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """Official CM-TTA entropy ``-p log(p)`` over valid 3-D voxels."""
    probabilities = probabilities.float()
    safe_eps = max(float(eps), float(torch.finfo(probabilities.dtype).eps))
    probabilities = probabilities.clamp(safe_eps, 1.0 - safe_eps)
    entropy = -probabilities * probabilities.log()
    if valid_mask is None:
        valid_mask = torch.ones_like(probabilities)
    else:
        valid_mask = valid_mask.to(device=probabilities.device, dtype=entropy.dtype)
        if valid_mask.ndim == probabilities.ndim - 1:
            valid_mask = valid_mask.unsqueeze(1)
        if tuple(valid_mask.shape) != tuple(probabilities.shape):
            raise ValueError(
                "valid_mask must match probability shape (apart from a singleton channel), "
                f"got {tuple(valid_mask.shape)} vs {tuple(probabilities.shape)}"
            )
    entropy = (entropy * valid_mask).flatten(start_dim=1).sum(dim=1)
    mass = valid_mask.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)
    return (entropy / mass).sum()


def _broadcast_valid_mask(
    reference: torch.Tensor, valid_mask: Optional[torch.Tensor]
) -> torch.Tensor:
    if valid_mask is None:
        return torch.ones_like(reference)
    valid_mask = valid_mask.to(device=reference.device, dtype=reference.dtype)
    if valid_mask.ndim == reference.ndim - 1:
        valid_mask = valid_mask.unsqueeze(1)
    if tuple(valid_mask.shape) != tuple(reference.shape):
        raise ValueError(
            "valid_mask must match tensor shape (apart from a singleton channel), "
            f"got {tuple(valid_mask.shape)} vs {tuple(reference.shape)}"
        )
    return valid_mask


def masked_dice_components(
    predictions: torch.Tensor,
    pseudo_label: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> dict[str, torch.Tensor]:
    """Return per-view masked Dice statistics before the final case reduction."""
    if predictions.ndim != 5 or pseudo_label.ndim != 5:
        raise ValueError(
            "Expected predictions (V,1,D,H,W) and pseudo_label (1,1,D,H,W), "
            f"got {tuple(predictions.shape)} and {tuple(pseudo_label.shape)}"
        )
    target = pseudo_label.expand(predictions.shape[0], *pseudo_label.shape[1:])
    valid_mask = _broadcast_valid_mask(predictions, valid_mask)
    pred_flat = predictions.float().flatten(start_dim=1)
    target_flat = target.float().flatten(start_dim=1)
    valid_flat = valid_mask.float().flatten(start_dim=1)
    pred_flat = pred_flat * valid_flat
    target_flat = target_flat * valid_flat
    return {
        "intersection": (pred_flat * target_flat).sum(dim=1),
        "prediction_mass": pred_flat.sum(dim=1),
        "pseudo_mass": target_flat.sum(dim=1),
    }


def soft_dice_loss(
    predictions: torch.Tensor,
    pseudo_label: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """DSPU soft Dice over every view, with a detached soft pseudo-label."""
    components = masked_dice_components(predictions, pseudo_label, valid_mask)
    dice = 1.0 - 2.0 * components["intersection"] / (
        components["prediction_mass"] + components["pseudo_mass"] + EPS
    )
    return dice.mean()


def masked_entropy_components(
    probabilities: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return entropy sum and valid mass for exact case-level reduction."""
    probabilities = probabilities.float()
    safe_eps = max(float(eps), float(torch.finfo(probabilities.dtype).eps))
    probabilities = probabilities.clamp(safe_eps, 1.0 - safe_eps)
    entropy = -probabilities * probabilities.log()
    valid_mask = _broadcast_valid_mask(probabilities, valid_mask).float()
    return (
        (entropy * valid_mask).flatten(start_dim=1).sum(dim=1),
        valid_mask.flatten(start_dim=1).sum(dim=1),
    )


def cac_from_features(
    vision_features: torch.Tensor,
    text_features: torch.Tensor,
    logits: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    feature_spatial_shape: Optional[tuple[int, int, int]] = None,
) -> torch.Tensor:
    """Compute CM-TTA's hard-threshold per-token cosine CAC in 3-D.

    The released VoxTell projection hook exposes visual tokens as ``(S,B,C)``
    and logits as ``(B,N,D,H,W)``.  The 5-D feature form is accepted for small
    test doubles and older checkpoints as well.  ``valid_mask`` is in the
    logits/input order ``(B,D,H,W)``.
    """
    if vision_features.ndim not in (3, 5):
        raise ValueError(
            "Expected projected 3-D features (S,B,C) or (B,H,W,D,C), "
            f"got {tuple(vision_features.shape)}"
        )
    if text_features.ndim != 3:
        raise ValueError(f"Expected text features (N,B,C), got {text_features.shape}")
    if logits.ndim != 5:
        raise ValueError(f"Expected logits (B,N,D,H,W), got {logits.shape}")
    components = cac_components_from_features(
        vision_features,
        text_features,
        logits,
        valid_mask=valid_mask,
        feature_spatial_shape=feature_spatial_shape,
    )
    return cac_from_components(
        components["foreground_sum"],
        components["foreground_mass"],
        components["background_sum"],
        components["background_mass"],
    )


def tdc_patch_components(
    decoder_outputs: list[torch.Tensor] | tuple[torch.Tensor, ...],
    valid_mask: torch.Tensor,
    threshold: float = 0.5,
) -> dict[str, torch.Tensor]:
    """Accumulate one patch's TDC pair statistics on the D5 grid.

    VoxTell returns decoder logits in ``[D5,D4,D3,D2,D1]`` order.  TDC uses
    only the first four outputs; lower-resolution outputs are resized to D5
    before hard thresholding.  The returned statistics are intentionally
    additive so callers can perform one exact case-level reduction.
    """
    if not isinstance(decoder_outputs, (list, tuple)) or len(decoder_outputs) < 4:
        raise ValueError("TDC requires decoder outputs [D5,D4,D3,D2,D1]")
    if not 0.0 <= float(threshold) <= 1.0:
        raise ValueError(f"TDC threshold must be in [0, 1], got {threshold}")
    # VoxTell decoder logits and input/valid masks are both in (D,H,W) order.
    # Do not infer or permute axes from shapes: non-cubic dimensions are part
    # of the interface contract, not evidence for a different layout.
    reference = decoder_outputs[0]
    if reference.ndim != 5 or reference.shape[1] != 1:
        raise ValueError(
            "TDC expects VoxTell decoder logits with shape (B,1,D,H,W), "
            f"got {tuple(reference.shape)}"
        )
    batch = reference.shape[0]
    spatial_shape = tuple(int(size) for size in reference.shape[2:])
    valid = valid_mask
    if valid.ndim == 4:
        valid = valid.unsqueeze(1)
    if valid.ndim != 5 or valid.shape[0] != batch:
        raise ValueError(
            "TDC valid_mask must have shape (B,D,H,W) or (B,1,D,H,W), "
            f"got {tuple(valid.shape)}"
        )
    valid = valid.to(device=reference.device, dtype=torch.bool)
    if tuple(valid.shape[2:]) != spatial_shape:
        valid = F.interpolate(valid.float(), size=spatial_shape, mode="nearest").bool()

    masks = []
    finite = torch.ones(batch, dtype=torch.bool, device=reference.device)
    for level, logits in enumerate(decoder_outputs[:4]):
        if not torch.is_tensor(logits) or logits.ndim != 5:
            raise ValueError(f"TDC decoder D{5 - level} must be 5-D logits")
        if logits.shape[:2] != reference.shape[:2]:
            raise ValueError("TDC decoder outputs must agree in batch and prompt dimensions")
        logits = logits.float()
        finite &= torch.isfinite(logits).flatten(start_dim=1).all(dim=1)
        logits = torch.nan_to_num(logits, nan=0.0, posinf=20.0, neginf=-20.0)
        if level and tuple(logits.shape[2:]) != spatial_shape:
            logits = F.interpolate(
                logits, size=spatial_shape, mode="trilinear", align_corners=False
            )
        masks.append((torch.sigmoid(logits) >= float(threshold)) & valid)

    intersection = reference.new_zeros((batch, len(TDC_DECODER_PAIRS)), dtype=torch.float32)
    count1 = intersection.clone()
    count2 = intersection.clone()
    for pair_index, (left, right) in enumerate(TDC_DECODER_PAIRS):
        mask_left = masks[left].flatten(start_dim=1)
        mask_right = masks[right].flatten(start_dim=1)
        count1[:, pair_index] = mask_left.sum(dim=1).float()
        count2[:, pair_index] = mask_right.sum(dim=1).float()
        intersection[:, pair_index] = (mask_left & mask_right).sum(dim=1).float()
    return {
        "intersection": intersection,
        "count1": count1,
        "count2": count2,
        "finite": finite,
    }


def tdc_from_components(
    intersection: torch.Tensor,
    count1: torch.Tensor,
    count2: torch.Tensor,
    finite: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute case-level pair Dice and TDC from globally summed statistics."""
    pair_nonempty = (count1 + count2) > 0
    pair_dice = 2.0 * intersection / (count1 + count2).clamp_min(1.0)
    pair_valid = pair_nonempty & finite.unsqueeze(1)
    pair_count = pair_valid.sum(dim=1)
    tdc = (pair_dice * pair_valid.float()).sum(dim=1) / pair_count.clamp_min(1)
    tdc = torch.where(pair_count > 0, tdc, torch.zeros_like(tdc))
    return tdc, pair_dice, pair_valid


def decoder_consistency_probabilities(
    decoder_outputs: list[torch.Tensor] | tuple[torch.Tensor, ...],
    valid_mask: Optional[torch.Tensor],
    bg_threshold: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Build the detached D2--D5 consistency masks on the D5 grid.

    The VoxTell decoder API and ordinary segmentation logits both use
    ``(B,1,D,H,W)``.  The decoder grid is therefore kept in that same order;
    only spatial resizing is allowed.
    """
    if not isinstance(decoder_outputs, (list, tuple)) or len(decoder_outputs) < 4:
        raise ValueError("Decoder consistency requires [D5,D4,D3,D2,D1] outputs")
    if not 0.0 <= float(bg_threshold) <= 1.0:
        raise ValueError(f"bg_threshold must be in [0, 1], got {bg_threshold}")
    reference = decoder_outputs[0]
    if reference.ndim != 5 or reference.shape[1] != 1:
        raise ValueError(
            "Decoder consistency expects D5 logits with shape (B,1,D,H,W), "
            f"got {tuple(reference.shape)}"
        )
    batch = reference.shape[0]
    d5_shape = tuple(int(size) for size in reference.shape[2:])
    if valid_mask is None:
        valid_d5 = torch.ones(
            (batch, 1, *d5_shape), device=reference.device, dtype=torch.bool
        )
    else:
        valid = valid_mask
        if valid.ndim == 4:
            valid = valid.unsqueeze(1)
        if valid.ndim != 5 or valid.shape[0] != batch or valid.shape[1] != 1:
            raise ValueError(
                "Decoder consistency valid_mask must have shape (B,D,H,W) or "
                f"(B,1,D,H,W), got {tuple(valid.shape)}"
            )
        valid_d5 = valid.to(device=reference.device, dtype=torch.float32)
        if tuple(valid_d5.shape[2:]) != d5_shape:
            valid_d5 = F.interpolate(valid_d5, size=d5_shape, mode="nearest")
        valid_d5 = valid_d5.bool()

    probabilities = []
    for level, logits in enumerate(decoder_outputs[:4]):
        if not torch.is_tensor(logits) or logits.ndim != 5:
            raise ValueError(f"D{5 - level} decoder output must be 5-D")
        if logits.shape[:2] != reference.shape[:2]:
            raise ValueError("D2--D5 decoder outputs disagree in batch/channel shape")
        logits = torch.nan_to_num(logits.float(), nan=0.0, posinf=20.0, neginf=-20.0)
        if level and tuple(logits.shape[2:]) != d5_shape:
            logits = F.interpolate(logits, size=d5_shape, mode="trilinear", align_corners=False)
        probabilities.append(torch.sigmoid(logits))
    stacked = torch.cat(probabilities, dim=1)  # (B,4,D,H,W), D5,D4,D3,D2
    p5 = stacked[:, :1]
    lower = stacked[:, 1:]
    votes = (lower > 0.5).sum(dim=1, keepdim=True)
    fg = (p5 > 0.5) & (votes >= 2) & valid_d5
    bg = (p5 < 0.5) & (lower.amax(dim=1, keepdim=True) < float(bg_threshold)) & valid_d5
    known = fg | bg
    ambiguous = valid_d5 & ~known
    missing = valid_d5 & (p5 < 0.5) & (votes >= 1)
    return {
        "p5": p5.detach(),
        "probabilities": stacked.detach(),
        "valid": valid_d5.detach(),
        "fg": fg.detach(),
        "bg": bg.detach(),
        "amb": ambiguous.detach(),
        "miss": missing.detach(),
    }


def decoder_grid_to_input_order(
    tensor: torch.Tensor,
    target_spatial_shape: tuple[int, int, int],
    mode: str = "nearest",
) -> torch.Tensor:
    """Resize a D5-grid tensor already in input/logit (D,H,W) order."""
    if tensor.ndim != 5:
        raise ValueError(f"Expected (B,1,D,H,W) D5 tensor, got {tuple(tensor.shape)}")
    converted = tensor
    if tuple(converted.shape[2:]) != tuple(target_spatial_shape):
        if mode in ("linear", "bilinear", "bicubic", "trilinear"):
            converted = F.interpolate(converted, size=target_spatial_shape, mode=mode, align_corners=False)
        else:
            converted = F.interpolate(converted, size=target_spatial_shape, mode=mode)
    return converted


def masked_tversky_loss_from_components(
    true_positive: torch.Tensor,
    false_positive: torch.Tensor,
    false_negative: torch.Tensor,
    alpha: float = 0.3,
    beta: float = 0.7,
    valid_mass: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-view Tversky losses and their equal-view mean.

    Components are already case-level sums with shape ``(num_views,)``.
    Empty views contribute an explicit zero instead of a NaN.
    """
    if not (true_positive.shape == false_positive.shape == false_negative.shape):
        raise ValueError("Tversky components must have identical shapes")
    if true_positive.ndim != 1:
        raise ValueError("Tversky components must have shape (num_views,)")
    if valid_mass is None:
        valid = (true_positive + false_positive + false_negative) > 0.0
    else:
        if valid_mass.shape != true_positive.shape:
            raise ValueError("Tversky valid_mass must match component shape")
        valid = valid_mass > 0.0
    denominator = true_positive + float(alpha) * false_positive + float(beta) * false_negative + EPS
    per_view = 1.0 - (true_positive + EPS) / denominator
    per_view = torch.where(valid, per_view, torch.zeros_like(per_view))
    return per_view.mean(), per_view


def masked_balanced_bce_from_components(
    foreground_sum: torch.Tensor,
    background_sum: torch.Tensor,
    foreground_mass: torch.Tensor | float,
    background_mass: torch.Tensor | float,
) -> torch.Tensor:
    """Equal-weight foreground/background BCE means with empty-safe terms."""
    terms = []
    if float(foreground_mass) > 0.0:
        terms.append(foreground_sum / torch.as_tensor(foreground_mass, device=foreground_sum.device).clamp_min(1.0))
    if float(background_mass) > 0.0:
        terms.append(background_sum / torch.as_tensor(background_mass, device=background_sum.device).clamp_min(1.0))
    if not terms:
        return foreground_sum.new_zeros(())
    return sum(terms) / len(terms)


def check_voxtell_decoder_d5_alignment(
    model: nn.Module,
    images: torch.Tensor,
    text_embedding: torch.Tensor,
    atol: float = 1e-5,
    rtol: float = 1e-5,
) -> dict[str, float | tuple[int, ...]]:
    """Validate the real VoxTell normal/D5 decoder interface.

    This is an opt-in diagnostic because it performs the normal forward in
    addition to ``return_decoder_outputs=True``.  The decoder return type and
    tensor ranks are checked before selecting D5 at index zero.
    """
    with torch.no_grad():
        normal = model(images, text_embedding, return_decoder_outputs=False)
        decoder_outputs = model(
            images, text_embedding, return_decoder_outputs=True
        )
    if isinstance(normal, (list, tuple)):
        if not normal or not torch.is_tensor(normal[0]):
            raise RuntimeError("VoxTell normal forward returned no logits tensor")
        normal = normal[0]
    if not isinstance(decoder_outputs, (list, tuple)) or len(decoder_outputs) < 4:
        raise RuntimeError(
            "VoxTell decoder diagnostic expected [D5,D4,D3,D2,D1] or a compatible sequence"
        )
    if not all(torch.is_tensor(output) and output.ndim == 5 for output in decoder_outputs[:4]):
        raise RuntimeError("VoxTell D2--D5 decoder outputs must all be 5-D tensors")
    d5 = decoder_outputs[0]
    if tuple(normal.shape) != tuple(d5.shape):
        raise RuntimeError(
            "VoxTell normal logits and D5 logits have different shapes: "
            f"normal={tuple(normal.shape)} d5={tuple(d5.shape)}"
        )
    difference = (normal.float() - d5.float()).abs()
    max_abs = float(difference.max().cpu())
    mean_abs = float(difference.mean().cpu())
    if not torch.allclose(normal, d5, atol=atol, rtol=rtol):
        raise RuntimeError(
            "VoxTell normal logits and D5 logits are not numerically aligned: "
            f"max_abs={max_abs:.6g} mean_abs={mean_abs:.6g}"
        )
    return {
        "normal_shape": tuple(int(size) for size in normal.shape),
        "d5_shape": tuple(int(size) for size in d5.shape),
        "max_abs_error": max_abs,
        "mean_abs_error": mean_abs,
    }


def _average_tie_rank_1d(values: torch.Tensor, descending: bool) -> torch.Tensor:
    """Match CM-SFDA's zero-based average rank for tied view scores."""
    ranked = -values if descending else values
    ranks = torch.empty_like(ranked, dtype=torch.float32)
    offset = 0
    for unique_value in torch.unique(ranked, sorted=True):
        tied = ranked == unique_value
        count = int(tied.sum())
        ranks[tied] = offset + 0.5 * (count - 1)
        offset += count
    return ranks


def _canonical_valid_mask(
    valid_mask: Optional[torch.Tensor],
    batch: int,
    source_spatial_shape: tuple[int, int, int],
    target_spatial_shape: tuple[int, int, int],
    device: torch.device,
    permute_source_to_target: bool = False,
) -> torch.Tensor:
    """Return a ``(B,*target_spatial_shape)`` mask in feature-grid order."""
    if valid_mask is None:
        return torch.ones((batch, *target_spatial_shape), device=device, dtype=torch.float32)
    mask = torch.as_tensor(valid_mask, device=device).float()
    if mask.ndim == 5 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 4 or mask.shape[0] != batch:
        raise ValueError(
            "valid_mask must have shape (B,D,H,W) or (B,1,D,H,W), "
            f"got {tuple(mask.shape)}"
        )
    mask_shape = tuple(int(size) for size in mask.shape[1:])
    if permute_source_to_target and mask_shape == source_spatial_shape:
        # The external VoxTell API uses (D,H,W), while its projected-memory
        # grid uses (H,W,D).  Permute before resizing when the grids differ.
        mask = mask.permute(0, 2, 3, 1).contiguous()
        if tuple(mask.shape[1:]) == target_spatial_shape:
            return mask
    elif mask_shape == target_spatial_shape:
        return mask
    if tuple(mask.shape[1:]) != target_spatial_shape:
        mask = F.interpolate(
            mask.unsqueeze(1), size=target_spatial_shape, mode="nearest"
        ).squeeze(1)
    return mask


def cac_components_from_features(
    vision_features: torch.Tensor,
    text_features: torch.Tensor,
    logits: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    feature_spatial_shape: Optional[tuple[int, int, int]] = None,
) -> dict[str, torch.Tensor]:
    """Accumulate CM-TTA's per-token similarity-map statistics.

    CM-TTA normalizes every visual token and the text feature first, computes
    their cosine map, and only then averages the map inside the hard
    foreground/background regions.  The returned similarity sums and masses
    can therefore be added across patches and reduced exactly once at case
    level.  The hard partition is detached by construction and cannot carry
    a gradient to the segmentation probabilities.
    """
    if text_features.ndim != 3:
        raise ValueError(f"Expected text features (N,B,C), got {text_features.shape}")
    if logits.ndim != 5:
        raise ValueError(f"Expected logits (B,N,D,H,W) or (B,N,H,W,D), got {logits.shape}")
    batch = logits.shape[0]
    if vision_features.ndim == 3:
        tokens, feature_batch, channels = vision_features.shape
        if feature_batch != batch:
            raise ValueError("Vision features and logits have different batch sizes")
        logits_spatial_shape = tuple(int(size) for size in logits.shape[2:])
        if feature_spatial_shape is None:
            output_spatial_shape = logits_spatial_shape
        else:
            output_spatial_shape = tuple(int(size) for size in feature_spatial_shape)
        if tokens != int(np.prod(output_spatial_shape)):
            raise ValueError(
                "Projected visual token count does not match logits spatial size: "
                f"{tokens} vs {output_spatial_shape}"
            )
        vision = vision_features.permute(1, 2, 0).reshape(batch, channels, *output_spatial_shape)
        mask = _canonical_valid_mask(
            valid_mask,
            batch,
            logits_spatial_shape,
            output_spatial_shape,
            logits.device,
            permute_source_to_target=feature_spatial_shape is not None,
        )
        probability = torch.sigmoid(logits[:, 0].float())
        if feature_spatial_shape is not None:
            # Flattened VoxTell memory tokens are ordered (H,W,D), whereas
            # segmentation logits are returned in (D,H,W) order.
            probability = probability.permute(0, 2, 3, 1).contiguous()
    elif vision_features.ndim == 5:
        if vision_features.shape[0] != batch:
            raise ValueError("Vision features and logits have different batch sizes")
        vision = vision_features.permute(0, 4, 1, 2, 3).float()
        output_spatial_shape = tuple(int(size) for size in vision.shape[2:])
        logits_spatial_shape = tuple(int(size) for size in logits.shape[2:])
        probability = torch.sigmoid(logits[:, 0].float()).permute(0, 2, 3, 1)
        mask = _canonical_valid_mask(
            valid_mask,
            batch,
            logits_spatial_shape,
            output_spatial_shape,
            logits.device,
            permute_source_to_target=True,
        )
    else:
        raise ValueError(f"Expected projected 3-D features, got {tuple(vision_features.shape)}")

    if probability.shape[1:] != vision.shape[2:]:
        probability = F.interpolate(
            probability.unsqueeze(1), size=vision.shape[2:], mode="trilinear", align_corners=False
        ).squeeze(1)
        mask = F.interpolate(
            mask.unsqueeze(1), size=vision.shape[2:], mode="nearest"
        ).squeeze(1)
    mask = mask.to(dtype=probability.dtype)
    # CM-TTA uses probability only for a hard region partition.  In
    # particular, the probability value itself must not weight visual tokens.
    foreground_mask = (probability.detach() > 0.5).to(dtype=probability.dtype) * mask
    background_mask = (probability.detach() <= 0.5).to(dtype=probability.dtype) * mask
    visual_tokens = F.normalize(vision.float(), dim=1)
    text = text_features[0].float()
    if text.shape[0] != batch or text.shape[1] != vision.shape[1]:
        raise ValueError(
            "Projected text and visual feature batch/channel dimensions must agree: "
            f"text={tuple(text.shape)}, visual={(batch, vision.shape[1])}"
        )
    text = F.normalize(text, dim=1).view(batch, vision.shape[1], 1, 1, 1)
    similarity_map = (visual_tokens * text).sum(dim=1)
    return {
        "foreground_sum": (similarity_map * foreground_mask).sum(dim=(1, 2, 3)),
        "foreground_mass": foreground_mask.sum(dim=(1, 2, 3)),
        "background_sum": (similarity_map * background_mask).sum(dim=(1, 2, 3)),
        "background_mass": background_mask.sum(dim=(1, 2, 3)),
    }


def cac_from_components(
    foreground_sum: torch.Tensor,
    foreground_mass: torch.Tensor,
    background_sum: torch.Tensor,
    background_mass: torch.Tensor,
) -> torch.Tensor:
    """Finish CAC from globally accumulated similarity-map sums and masses."""
    if foreground_sum.ndim != 1 or background_sum.ndim != 1:
        raise ValueError("CAC similarity sums must have shape (B,)")
    if foreground_sum.shape != background_sum.shape:
        raise ValueError("Foreground/background CAC statistics must have matching shapes")
    foreground = foreground_sum / foreground_mass.clamp_min(1.0)
    background = background_sum / background_mass.clamp_min(1.0)
    return foreground - background


def select_cac_view(
    cac_scores: torch.Tensor,
    probabilities: torch.Tensor,
    selection_p: float,
) -> tuple[int, torch.Tensor]:
    """Select views by the official CAC-rank plus entropy-rank rule."""
    if cac_scores.ndim != 1 or probabilities.ndim < 2:
        raise ValueError("Expected one CAC score and one probability map per view")
    if cac_scores.shape[0] != probabilities.shape[0]:
        raise ValueError("CAC scores and probabilities must agree in view count")
    if not 0.0 < float(selection_p) <= 1.0:
        raise ValueError("selection_p must be in (0, 1]")
    probabilities = probabilities.float().clamp(EPS, 1.0 - EPS)
    entropy = -(probabilities * probabilities.log())
    entropy = entropy.flatten(start_dim=1).mean(dim=1)
    return select_cac_view_from_entropy(cac_scores, entropy, selection_p)


def select_cac_view_from_entropy(
    cac_scores: torch.Tensor,
    entropy_scores: torch.Tensor,
    selection_p: float,
) -> tuple[int, torch.Tensor]:
    """Apply CM-TTA's combined CAC/entropy rank to case-level statistics."""
    if cac_scores.ndim != 1 or entropy_scores.ndim != 1:
        raise ValueError("CAC and entropy scores must have shape (num_views,)")
    if cac_scores.shape != entropy_scores.shape:
        raise ValueError("CAC and entropy scores must agree in view count")
    if not 0.0 < float(selection_p) <= 1.0:
        raise ValueError("selection_p must be in (0, 1]")
    entropy_rank = entropy_scores.argsort().argsort().float()
    cac_rank = (-cac_scores).argsort().argsort().float()
    combined_rank = entropy_rank + cac_rank
    num_views = cac_scores.numel()
    num_selected = max(1, int(num_views * float(selection_p)))
    if num_selected != 1:
        raise ValueError(
            "VoxTell CM-TTA requires exactly one selected view; "
            f"selection_p={selection_p} with num_views={num_views} "
            f"would select {num_selected} views"
        )
    selected_indices = torch.argsort(combined_rank, descending=False)[:1]
    return int(selected_indices[0].item()), selected_indices


class ShortPromptMemory:
    """FIFO memory of recent short prompt deltas and their quality scores."""

    def __init__(self, max_length: int):
        self.max_length = int(max_length)
        if self.max_length < 1:
            raise ValueError("short memory length must be positive")
        self.deltas: deque[torch.Tensor] = deque(maxlen=self.max_length)
        self.qualities: deque[float] = deque(maxlen=self.max_length)

    @property
    def cacs(self):
        """Compatibility view for checkpoints/tools written before TDC LSPM."""
        return self.qualities

    @property
    def contexts(self):
        """Compatibility view; memory entries are prompt deltas."""
        return self.deltas

    def __len__(self) -> int:
        return len(self.deltas)

    def weighted_delta(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.deltas:
            raise RuntimeError("Cannot fuse an empty short-delta memory")
        scores = torch.tensor(list(self.qualities), device=device, dtype=torch.float32)
        weights = torch.softmax(scores, dim=0)
        result = torch.zeros_like(self.deltas[0], device=device, dtype=dtype)
        for weight, delta in zip(weights, self.deltas):
            result = result + weight.to(dtype) * delta.to(device=device, dtype=dtype)
        return result

    def weighted_ctx(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        """Compatibility alias for callers that use the old ctx name."""
        return self.weighted_delta(device, dtype)

    def append_delta(self, delta: torch.Tensor, quality: float) -> None:
        self.deltas.append(delta.detach().cpu().clone())
        self.qualities.append(float(quality))

    def append(self, ctx: torch.Tensor, quality: float) -> None:
        """Compatibility alias; ``ctx`` is stored as a delta."""
        self.append_delta(ctx, quality)

    def state_dict(self) -> dict:
        return {
            "max_length": self.max_length,
            "deltas": [delta.clone() for delta in self.deltas],
            "qualities": list(self.qualities),
            # Keep the old key so older analysis scripts can still inspect it.
            "cacs": list(self.qualities),
        }

    def load_state_dict(self, state: dict) -> None:
        self.max_length = int(state["max_length"])
        if self.max_length < 1:
            raise ValueError("short memory length must be positive")
        if "ctxs" in state or "prompts" in state:
            raise ValueError(
                "Legacy random-ctx memory is incompatible with ctx_delta checkpoints"
            )
        contexts = state.get("deltas", [])
        qualities = state.get("qualities", state.get("cacs", []))
        if len(contexts) != len(qualities):
            raise ValueError("short memory deltas and quality scores must have equal lengths")
        self.deltas = deque(maxlen=self.max_length)
        self.qualities = deque(maxlen=self.max_length)
        for delta, quality in zip(contexts, qualities):
            self.append_delta(delta, float(quality))


class VoxTellCMTTA:
    """CM-TTA/LSPM/DSPU with one trainable FP32 token-tuning delta."""

    def __init__(
        self,
        model: nn.Module,
        initial_ctx: Optional[torch.Tensor],
        device,
        args,
        qwen_text_encoder: Optional[nn.Module] = None,
        qwen_tokenizer=None,
        text_prompt: str = "liver",
        n_ctx: Optional[int] = None,
        formatted_text_prompt: Optional[str] = None,
    ):
        self.device = torch.device(device)
        self.args = args
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
            parameter.grad = None
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("VoxTell model must be completely frozen")
        self.qwen_text_encoder = qwen_text_encoder
        if self.qwen_text_encoder is not None:
            self.qwen_text_encoder.eval()
            for parameter in self.qwen_text_encoder.parameters():
                parameter.requires_grad_(False)
                parameter.grad = None
            if any(parameter.requires_grad for parameter in self.qwen_text_encoder.parameters()):
                raise RuntimeError("Qwen text encoder must be completely frozen")

        self.text_prompt = str(text_prompt)
        self.n_ctx = 1 if n_ctx is None else int(n_ctx)
        if self.n_ctx < 1:
            raise ValueError("n_ctx must be positive")
        self._fixed_token_embeddings = None
        self._fixed_attention_mask = None
        self._liver_token_indices = None

        if self.qwen_text_encoder is not None:
            if qwen_tokenizer is None:
                raise ValueError("qwen_tokenizer is required with qwen_text_encoder")
            self._initialize_qwen_context(
                qwen_tokenizer, initial_ctx, formatted_text_prompt
            )
        else:
            if initial_ctx is None:
                raise ValueError(
                    "initial_ctx is required only for the test/identity text encoder"
                )
            initial_ctx = initial_ctx.detach()
            if initial_ctx.ndim == 3:
                if initial_ctx.shape[0] != 1:
                    raise ValueError(
                        "Expected one context batch, "
                        f"got {tuple(initial_ctx.shape)}"
                    )
                initial_ctx = initial_ctx[0]
            if initial_ctx.ndim != 2 or initial_ctx.shape[0] != self.n_ctx:
                raise ValueError(
                    "Expected context with shape (n_ctx,D) or (1,n_ctx,D), "
                    f"got {tuple(initial_ctx.shape)}"
                )
            initial_ctx = initial_ctx.to(self.device, dtype=torch.float32)
            self.ctx_delta = nn.Parameter(initial_ctx.clone())
            self.initial_ctx_delta = initial_ctx.clone()
        self.long_delta: Optional[torch.Tensor] = None
        self.short_delta: Optional[torch.Tensor] = None

        self.lr = float(args.lr)
        self.ema_momentum = float(args.ema_momentum)
        self.w_cac = float(args.w_cac)
        self.w_entropy = float(args.w_entropy)
        self.pseudo_update_mode = str(getattr(args, "pseudo_update_mode", "original"))
        if self.pseudo_update_mode not in ("original", "decoder_masked"):
            raise ValueError(
                "pseudo_update_mode must be 'original' or 'decoder_masked'"
            )
        self.bg_threshold = float(getattr(args, "bg_threshold", 0.1))
        self.tversky_alpha = float(getattr(args, "tversky_alpha", 0.3))
        self.tversky_beta = float(getattr(args, "tversky_beta", 0.7))
        self.tversky_weight = float(getattr(args, "tversky_weight", 1.0))
        self.amb_weight = float(getattr(args, "amb_weight", 0.05))
        # Optional validation only.  It performs one additional ordinary
        # forward beside the decoder-output forward and is therefore off by
        # default for the normal low-memory adaptation path.
        self.decoder_alignment_check = bool(
            getattr(args, "decoder_alignment_check", False)
        )
        if not 0.0 <= self.bg_threshold <= 1.0:
            raise ValueError("bg_threshold must be in [0, 1]")
        if self.tversky_alpha < 0.0 or self.tversky_beta < 0.0:
            raise ValueError("tversky_alpha and tversky_beta must be non-negative")
        if self.tversky_weight < 0.0:
            raise ValueError("tversky_weight must be non-negative")
        if self.amb_weight < 0.0:
            raise ValueError("amb_weight must be non-negative")
        self.num_aug_views = int(args.num_aug_views)  # K; total views are K+1.
        self.selection_p = float(args.selection_p)
        self.view_selection_metric = str(getattr(args, "view_selection_metric", "cac"))
        self.use_entropy_rank = bool(getattr(args, "use_entropy_rank", True))
        if self.view_selection_metric not in ("cac", "tdc"):
            raise ValueError("view_selection_metric must be 'cac' or 'tdc'")
        if self.num_aug_views < 1:
            raise ValueError("num_aug_views must be at least 1")
        if not 0.0 < self.selection_p <= 1.0:
            raise ValueError("selection_p must be in (0, 1]")

        self.optimizer = torch.optim.Adam([self.ctx_delta], lr=self.lr)
        amp_enabled = self.device.type == "cuda"
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=amp_enabled, init_scale=float(args.amp_init_scale)
            )
        else:
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=amp_enabled, init_scale=float(args.amp_init_scale)
            )
        self.optimizer_step_count = 0

        memory_length = getattr(args, "short_memory_length", getattr(args, "prompt_memory_size", 16))
        self.short_memory = ShortPromptMemory(int(memory_length))
        self.view_batch_size = max(1, int(getattr(args, "view_batch_size", 1)))
        # VoxTell's hook normally returns (B,H,W,D,C).  Keep the configured
        # grid as a fallback for wrappers that flatten it to (S,B,C).
        base_model = getattr(self.model, "_orig_mod", self.model)
        decoder_configs = getattr(base_model, "DECODER_CONFIGS", None)
        selected_decoder_layer = getattr(base_model, "selected_decoder_layer", None)
        self.feature_spatial_shape = None
        if isinstance(decoder_configs, dict) and selected_decoder_layer in decoder_configs:
            shape = decoder_configs[selected_decoder_layer].get("shape")
            if shape is not None:
                depth, height, width = (int(value) for value in shape)
                # The flattened hook follows VoxTell's (H,W,D) ordering.
                self.feature_spatial_shape = (height, width, depth)
        self._vision_features = None
        self._text_features = None
        self._hooks = [
            self.model.project_bottleneck_embed.register_forward_hook(
                self._capture("vision")
            ),
            self.model.project_text_embed.register_forward_hook(
                self._capture("text")
            ),
        ]
        self.last_trace = {}
        self.last_view_selection = {}
        self._last_pseudo_diagnostics = {}
        self._last_decoder_pseudo_cache = None
        self._print_trainable_parameters()

    def reset_case_adaptation_state(self) -> None:
        """Reset prompt/memory/optimizer state for selector-only evaluation."""
        self.ctx_delta.data.copy_(self.initial_ctx_delta)
        self.ctx_delta.grad = None
        self.short_delta = None
        self.long_delta = None
        self.short_memory = ShortPromptMemory(self.short_memory.max_length)
        self.optimizer.state.clear()
        self.optimizer.zero_grad(set_to_none=True)
        self.optimizer_step_count = 0
        self.last_trace = {}
        self.last_view_selection = {}

    def _capture(self, name):
        def hook(_module, _inputs, output):
            if name == "vision":
                self._vision_features = output
            else:
                self._text_features = output

        return hook

    def _initialize_qwen_context(
        self,
        tokenizer,
        initial_ctx: Optional[torch.Tensor],
        formatted_text_prompt: Optional[str],
    ) -> None:
        """Build fixed Qwen inputs and zero-initialized token deltas."""
        if self.text_prompt != "liver":
            raise ValueError('VoxTell CM-TTA fixes the text prompt to "liver"')
        if formatted_text_prompt is None:
            try:
                from voxtell.utils.text_embedding import wrap_with_instruction
            except ModuleNotFoundError as error:
                raise RuntimeError(
                    "VoxTell text utilities are required to construct the fixed Qwen prompt"
                ) from error
            formatted_prompt = wrap_with_instruction([self.text_prompt])[0]
        else:
            # This explicit form is retained for small test doubles; the
            # production path always takes VoxTell's wrapper above.
            formatted_prompt = formatted_text_prompt
        max_length = int(getattr(self.args, "max_text_length", 8192))
        try:
            tokenized = tokenizer(
                [formatted_prompt],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
                return_offsets_mapping=True,
            )
        except (TypeError, ValueError):
            # Slow tokenizers do not expose offset mappings. The token-id
            # subsequence fallback below still locates the fixed liver token.
            tokenized = tokenizer(
                [formatted_prompt],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
        if "input_ids" not in tokenized or "attention_mask" not in tokenized:
            raise ValueError("Qwen tokenizer must return input_ids and attention_mask")
        input_ids = torch.as_tensor(tokenized["input_ids"], dtype=torch.long)
        attention_mask = torch.as_tensor(tokenized["attention_mask"], dtype=torch.bool)
        if input_ids.ndim != 2 or input_ids.shape[0] != 1:
            raise ValueError(f"Expected one tokenized prompt, got {tuple(input_ids.shape)}")

        self.qwen_text_encoder = self.qwen_text_encoder.to(self.device).eval()
        token_embedding = self.qwen_text_encoder.get_input_embeddings()
        if token_embedding is None:
            raise ValueError("Qwen text encoder does not expose get_input_embeddings()")
        with torch.no_grad():
            fixed_embeddings = token_embedding(input_ids.to(self.device)).detach()
        self._fixed_token_embeddings = fixed_embeddings
        self._fixed_attention_mask = attention_mask.to(self.device)

        query_start = formatted_prompt.rfind(self.text_prompt)
        query_end = query_start + len(self.text_prompt)
        offsets = tokenized.get("offset_mapping")
        liver_indices = []
        if offsets is not None and query_start >= 0:
            for index, (start, end) in enumerate(torch.as_tensor(offsets)[0].tolist()):
                if end > start and start < query_end and end > query_start:
                    liver_indices.append(index)
        if not liver_indices:
            active_indices = torch.nonzero(attention_mask[0], as_tuple=False).flatten().tolist()
            active_ids = input_ids[0, active_indices].tolist()
            query_token_candidates = []
            for query in (self.text_prompt, " " + self.text_prompt):
                query_tokens = tokenizer(
                    [query], add_special_tokens=False, return_tensors="pt"
                )["input_ids"][0].tolist()
                if query_tokens and query_tokens not in query_token_candidates:
                    query_token_candidates.append(query_tokens)
            for query_tokens in query_token_candidates:
                for start in range(max(0, len(active_ids) - len(query_tokens) + 1)):
                    if active_ids[start:start + len(query_tokens)] == query_tokens:
                        liver_indices = active_indices[start:start + len(query_tokens)]
                        break
                if liver_indices:
                    break
        if not liver_indices:
            raise ValueError(
                "Could not locate all fixed 'liver' tokens in the wrapped Qwen prompt"
            )

        embedding_dim = int(token_embedding.embedding_dim)
        if initial_ctx is not None and torch.as_tensor(initial_ctx).numel():
            if torch.as_tensor(initial_ctx).detach().abs().max().item() != 0.0:
                raise ValueError(
                    "Non-zero initial ctx is incompatible with zero-initialized ctx_delta"
                )
        self._liver_token_indices = torch.tensor(
            liver_indices, device=self.device, dtype=torch.long
        )
        self.n_ctx = len(liver_indices)
        delta = torch.zeros(
            (self.n_ctx, embedding_dim), device=self.device, dtype=torch.float32
        )
        self.ctx_delta = nn.Parameter(delta)
        self.initial_ctx_delta = delta.clone()

    def _encode_ctx(self, ctx_delta: torch.Tensor) -> torch.Tensor:
        """Encode fixed wrapped ``liver`` tokens plus position-local deltas."""
        if self.qwen_text_encoder is None:
            return ctx_delta.unsqueeze(0) if ctx_delta.ndim == 2 else ctx_delta
        if ctx_delta.ndim == 2:
            ctx_delta = ctx_delta.unsqueeze(0)
        if tuple(ctx_delta.shape[:2]) != (1, self.n_ctx):
            raise ValueError(
                f"Expected ctx_delta shape (1,{self.n_ctx},D), got {tuple(ctx_delta.shape)}"
            )
        fixed = self._fixed_token_embeddings
        fixed_mask = self._fixed_attention_mask
        indices = self._liver_token_indices
        if fixed is None or fixed_mask is None or indices is None:
            raise RuntimeError("Qwen context inputs were not initialized")
        ctx_delta = ctx_delta.to(device=fixed.device, dtype=fixed.dtype)
        delta_embeddings = torch.zeros_like(fixed)
        delta_embeddings = delta_embeddings.index_copy(1, indices, ctx_delta)
        inputs_embeds = fixed + delta_embeddings
        attention_mask = fixed_mask
        outputs = self.qwen_text_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        try:
            from voxtell.utils.text_embedding import last_token_pool
        except ModuleNotFoundError:
            # The fallback is only for the lightweight test double; production
            # VoxTell always uses its own pooling helper above.
            left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
            if left_padding:
                pooled = hidden_states[:, -1]
            else:
                last_index = attention_mask.sum(dim=1) - 1
                pooled = hidden_states[
                    torch.arange(hidden_states.shape[0], device=hidden_states.device),
                    last_index,
                ]
        else:
            pooled = last_token_pool(hidden_states, attention_mask)
        return pooled.unsqueeze(1)

    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    @property
    def ctx(self) -> torch.Tensor:
        """Compatibility alias for the sole trainable prompt delta."""
        return self.ctx_delta

    @property
    def initial_ctx(self) -> torch.Tensor:
        """Compatibility alias for the initial prompt delta."""
        return self.initial_ctx_delta

    @property
    def short_ctx(self) -> Optional[torch.Tensor]:
        """Compatibility alias for the short prompt delta."""
        return self.short_delta

    @short_ctx.setter
    def short_ctx(self, value: Optional[torch.Tensor]) -> None:
        self.short_delta = value

    @property
    def long_ctx(self) -> Optional[torch.Tensor]:
        """Compatibility alias for the long prompt delta."""
        return self.long_delta

    @long_ctx.setter
    def long_ctx(self, value: Optional[torch.Tensor]) -> None:
        self.long_delta = value

    @property
    def optimizer_parameters(self):
        return [parameter for group in self.optimizer.param_groups for parameter in group["params"]]

    def _print_trainable_parameters(self) -> None:
        trainable = [("ctx_delta", self.ctx_delta)]
        unexpected = [
            name
            for name, parameter in trainable
            if not parameter.requires_grad
        ]
        optimizer_parameters = self.optimizer_parameters
        if unexpected or len(optimizer_parameters) != 1 or optimizer_parameters[0] is not self.ctx_delta:
            raise RuntimeError("Only ctx_delta may be trainable and optimized")
        print(
            "[VoxTell-CM-TTA] trainable parameters: "
            f"ctx_delta shape={tuple(self.ctx_delta.shape)}, dtype={self.ctx_delta.dtype}, "
            f"numel={self.ctx_delta.numel()}"
        )

    def _check_case_gradients(self, allow_nonfinite: bool = False) -> None:
        if self.ctx_delta.grad is None:
            raise RuntimeError("ctx_delta did not receive a gradient during case adaptation")
        if self.ctx_delta.grad.norm() == 0:
            raise RuntimeError("ctx_delta gradient is zero during case adaptation")
        if not allow_nonfinite and not torch.isfinite(self.ctx_delta.grad).all():
            raise RuntimeError("ctx_delta gradient is zero or non-finite during case adaptation")
        frozen_modules = [("VoxTell", self.model)]
        if self.qwen_text_encoder is not None:
            frozen_modules.append(("Qwen", self.qwen_text_encoder))
        leaked = [
            f"{module_name}.{name}"
            for module_name, module in frozen_modules
            for name, parameter in module.named_parameters()
            if parameter.grad is not None
        ]
        if leaked:
            raise RuntimeError(f"Frozen model parameters received gradients: {leaked[:5]}")

    def _text_input(self, text_features: torch.Tensor, batch_size: int) -> torch.Tensor:
        return text_features.expand(batch_size, -1, -1).unsqueeze(2)

    def _forward(self, images: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        text_features = self._encode_ctx(ctx)
        logits = self.model(images, self._text_input(text_features, images.shape[0]))
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if logits.ndim != 5:
            raise ValueError(f"VoxTell must return (B,N,D,H,W) logits, got {logits.shape}")
        return logits

    def _forward_decoder_outputs(
        self, images: torch.Tensor, ctx: torch.Tensor
    ) -> list[torch.Tensor]:
        """Run the analysis-only VoxTell decoder-output interface for TDC."""
        text_features = self._encode_ctx(ctx)
        try:
            outputs = self.model(
                images,
                self._text_input(text_features, images.shape[0]),
                return_decoder_outputs=True,
            )
        except TypeError as error:
            raise RuntimeError(
                "TDC requires VoxTell forward(return_decoder_outputs=True)"
            ) from error
        if not isinstance(outputs, (list, tuple)) or len(outputs) < 4:
            raise RuntimeError(
                "VoxTell return_decoder_outputs=True must return [D5,D4,D3,D2,D1]"
            )
        if not all(
            torch.is_tensor(output)
            and output.ndim == 5
            and output.shape[0] == images.shape[0]
            and output.shape[1] == images.shape[1]
            for output in outputs[:4]
        ):
            raise RuntimeError(
                "VoxTell D2--D5 decoder outputs must be 5-D (B,N,D,H,W) tensors"
            )
        return list(outputs)

    def _cac_components(
        self, logits: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        if self._vision_features is None or self._text_features is None:
            raise RuntimeError("VoxTell CAC feature hooks did not capture a forward pass")
        return cac_components_from_features(
            self._vision_features,
            self._text_features,
            logits,
            valid_mask=valid_mask,
            feature_spatial_shape=self.feature_spatial_shape,
        )

    def _cac(
        self, logits: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        if self._vision_features is None or self._text_features is None:
            raise RuntimeError("VoxTell CAC feature hooks did not capture a forward pass")
        return cac_from_features(
            self._vision_features,
            self._text_features,
            logits,
            valid_mask=valid_mask,
            feature_spatial_shape=self.feature_spatial_shape,
        )

    @staticmethod
    def _add_components(
        accumulator: Optional[dict[str, torch.Tensor]],
        components: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        if accumulator is None:
            return {key: value for key, value in components.items()}
        return {
            key: accumulator[key] + components[key]
            for key in components
        }

    def _case_cac(
        self,
        ctx: torch.Tensor,
        patches: Iterable[torch.Tensor],
        valid_masks: Optional[Iterable[torch.Tensor]] = None,
    ) -> float:
        patches = list(patches)
        if valid_masks is None:
            valid_masks = [None] * len(patches)
        else:
            valid_masks = list(valid_masks)
        if len(valid_masks) != len(patches):
            raise ValueError("patches and valid_masks must have equal lengths")
        accumulator = None
        with torch.no_grad():
            for patch, valid_mask in zip(patches, valid_masks):
                patch = patch.unsqueeze(0).to(self.device, non_blocking=True)
                if valid_mask is None:
                    valid_mask = torch.ones((1, *patch.shape[-3:]), device=self.device)
                else:
                    valid_mask = valid_mask.unsqueeze(0).to(self.device, non_blocking=True)
                logits = self._forward(patch, ctx)
                components = self._cac_components(logits, valid_mask)
                accumulator = self._add_components(accumulator, components)
        if accumulator is None:
            raise ValueError("A complete case must contain at least one patch")
        score = cac_from_components(
            accumulator["foreground_sum"],
            accumulator["foreground_mass"],
            accumulator["background_sum"],
            accumulator["background_mass"],
        )
        return float(score[0].detach().cpu())

    def _case_tdc(
        self,
        ctx: torch.Tensor,
        patches: Iterable[torch.Tensor],
        valid_masks: Optional[Iterable[torch.Tensor]] = None,
    ) -> float:
        """Compute one exact case-level D5--D2 TDC score for a prompt."""
        patches = list(patches)
        if valid_masks is None:
            valid_masks = [None] * len(patches)
        else:
            valid_masks = list(valid_masks)
        if len(valid_masks) != len(patches):
            raise ValueError("patches and valid_masks must have equal lengths")
        accumulator = None
        finite = None
        with torch.no_grad():
            for patch, valid_mask in zip(patches, valid_masks):
                patch = patch.unsqueeze(0).to(self.device, non_blocking=True)
                if valid_mask is None:
                    valid_mask = torch.ones(
                        (1, *patch.shape[-3:]), device=self.device
                    )
                else:
                    if valid_mask.ndim == 3:
                        valid_mask = valid_mask.unsqueeze(0)
                    valid_mask = valid_mask.to(self.device, non_blocking=True)
                decoder_outputs = self._forward_decoder_outputs(patch, ctx)
                local = tdc_patch_components(decoder_outputs, valid_mask)
                if accumulator is None:
                    accumulator = {
                        key: local[key].clone()
                        for key in ("intersection", "count1", "count2")
                    }
                    finite = local["finite"].clone()
                else:
                    for key in ("intersection", "count1", "count2"):
                        accumulator[key].add_(local[key])
                    finite &= local["finite"]
        if accumulator is None or finite is None:
            raise ValueError("A complete case must contain at least one patch")
        score, _pair_dice, _pair_valid = tdc_from_components(
            accumulator["intersection"],
            accumulator["count1"],
            accumulator["count2"],
            finite,
        )
        return float(score[0].detach().cpu())

    def _case_quality(
        self,
        ctx: torch.Tensor,
        patches: Iterable[torch.Tensor],
        valid_masks: Optional[Iterable[torch.Tensor]] = None,
    ) -> float:
        """Return the primary quality used by both LSPM and view selection."""
        if self.view_selection_metric == "tdc":
            return self._case_tdc(ctx, patches, valid_masks)
        return self._case_cac(ctx, patches, valid_masks)

    @staticmethod
    def _sample_intensity_params(num_views: int) -> list[dict[str, float]]:
        params = [{"scale": 1.0, "offset": 0.0}]
        for _ in range(num_views):
            params.append(
                {
                    "scale": float(torch.empty(()).uniform_(0.85, 1.15)),
                    "offset": float(torch.empty(()).uniform_(-0.15, 0.15)),
                }
            )
        return params

    @staticmethod
    def _make_views(
        patch: torch.Tensor,
        params: list[dict[str, float]],
        valid_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        return VoxTellCMTTA._make_view_batch(
            patch, params, valid_mask, 0, len(params)
        )

    @staticmethod
    def _make_view_batch(
        patch: torch.Tensor,
        params: list[dict[str, float]],
        valid_mask: Optional[torch.Tensor],
        start: int,
        end: int,
    ) -> torch.Tensor:
        if not 0 <= start < end <= len(params):
            raise ValueError("invalid view batch range")
        views = []
        for view_index in range(start, end):
            param = params[view_index]
            view = patch if view_index == 0 else patch * param["scale"] + param["offset"]
            views.append(view)
        views = torch.stack(views, dim=0).contiguous()
        if valid_mask is not None:
            if valid_mask.ndim == 4 and valid_mask.shape[0] == 1:
                valid_mask = valid_mask[0]
            if valid_mask.ndim != 3 or tuple(valid_mask.shape) != tuple(patch.shape[-3:]):
                raise ValueError(
                    "valid_mask must have shape (D,H,W) for a patch, "
                    f"got {tuple(valid_mask.shape)}"
                )
            # Re-mask after intensity augmentation so padding never acquires
            # an offset/noise-like foreground value.
            views = views * valid_mask.to(device=views.device, dtype=views.dtype).unsqueeze(0)
        return views

    def _dynamic_short_ctx(
        self,
        patches: list[torch.Tensor],
        valid_masks: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, float, float, float]:
        current = self.ctx_delta
        current_quality = self._case_quality(current, patches, valid_masks)
        if len(self.short_memory) == 0:
            short = current
            return short, current_quality, current_quality, 0.0

        historical = self.short_memory.weighted_delta(self.device, current.dtype)
        historical_quality = self._case_quality(historical, patches, valid_masks)
        weights = torch.softmax(
            torch.tensor([historical_quality, current_quality], device=self.device), dim=0
        )
        weight_historical = float(weights[0].detach().cpu())
        short = weight_historical * historical + (1.0 - weight_historical) * current
        return short, current_quality, historical_quality, weight_historical

    def _select_case_view(
        self,
        patches: list[torch.Tensor],
        params: list[dict[str, float]],
        short_ctx: torch.Tensor,
        valid_masks: Optional[list[torch.Tensor]] = None,
        collect_tdc: bool = False,
    ) -> tuple[int, torch.Tensor]:
        if valid_masks is None:
            valid_masks = [None] * len(patches)
        if len(valid_masks) != len(patches):
            raise ValueError("patches and valid_masks must have equal lengths")
        accumulator = None
        collect_tdc = collect_tdc or self.view_selection_metric == "tdc"
        tdc_accumulator = None
        tdc_finite = None
        with torch.no_grad():
            for patch, valid_mask in zip(patches, valid_masks):
                if valid_mask is None:
                    valid_mask = torch.ones(patch.shape[-3:])
                elif valid_mask.ndim == 4 and valid_mask.shape[0] == 1:
                    valid_mask = valid_mask[0]
                total_views = len(params)
                for start in range(0, total_views, self.view_batch_size):
                    end = min(total_views, start + self.view_batch_size)
                    view_batch = self._make_view_batch(
                        patch, params, valid_mask, start, end
                    ).to(self.device, non_blocking=True)
                    input_mask_batch = valid_mask.unsqueeze(0).to(
                        self.device, non_blocking=True
                    ).expand(end - start, -1, -1, -1)
                    if collect_tdc:
                        decoder_outputs = self._forward_decoder_outputs(
                            view_batch, short_ctx.detach()
                        )
                        logits = decoder_outputs[0]
                        local_tdc = tdc_patch_components(
                            decoder_outputs, input_mask_batch
                        )
                    else:
                        logits = self._forward(view_batch, short_ctx.detach())
                        local_tdc = None
                    components = self._cac_components(logits, input_mask_batch)
                    probabilities = torch.sigmoid(logits[:, :1])
                    local_entropy_sum, local_entropy_mass = masked_entropy_components(
                        probabilities, input_mask_batch
                    )
                    if accumulator is None:
                        # Keep a real global view axis.  Adding successive
                        # chunks directly would align their local index 0s
                        # and collapse all views when batch_size=1.
                        accumulator = {
                            key: torch.zeros(
                                (total_views, *value.shape[1:]),
                                device=value.device,
                                dtype=value.dtype,
                            )
                            for key, value in components.items()
                        }
                        entropy_sum = torch.zeros(
                            (total_views,),
                            device=local_entropy_sum.device,
                            dtype=local_entropy_sum.dtype,
                        )
                        entropy_mass = torch.zeros_like(entropy_sum)
                        if collect_tdc:
                            tdc_accumulator = {
                                key: torch.zeros(
                                    (total_views, *value.shape[1:]),
                                    device=value.device,
                                    dtype=value.dtype,
                                )
                                for key, value in local_tdc.items()
                                if key != "finite"
                            }
                            tdc_finite = torch.ones(
                                (total_views,),
                                device=local_tdc["intersection"].device,
                                dtype=torch.bool,
                            )
                    for key, value in components.items():
                        accumulator[key][start:end].add_(value)
                    if local_tdc is not None:
                        for key in ("intersection", "count1", "count2"):
                            tdc_accumulator[key][start:end].add_(local_tdc[key])
                        tdc_finite[start:end] &= local_tdc["finite"]
                    entropy_sum[start:end].add_(local_entropy_sum)
                    entropy_mass[start:end].add_(local_entropy_mass)
        if accumulator is None:
            raise ValueError("A complete case must contain at least one patch")
        foreground_counts = accumulator["foreground_mass"].detach().cpu().tolist()
        background_counts = accumulator["background_mass"].detach().cpu().tolist()
        count_report = ", ".join(
            f"view={view}: fg={float(fg):.0f}, bg={float(bg):.0f}"
            for view, (fg, bg) in enumerate(zip(foreground_counts, background_counts))
        )
        print(
            "[VoxTell-CM-TTA] case CAC valid bottleneck tokens: "
            f"case={self.optimizer_step_count + 1}; {count_report}"
        )
        scores = cac_from_components(
            accumulator["foreground_sum"],
            accumulator["foreground_mass"],
            accumulator["background_sum"],
            accumulator["background_mass"],
        ).detach()
        if entropy_sum is None or entropy_mass is None:
            raise RuntimeError("Case entropy accumulation produced no statistics")
        entropy_scores = (entropy_sum / entropy_mass.clamp_min(1.0)).detach()
        tdc_scores = None
        if collect_tdc:
            if tdc_accumulator is None or tdc_finite is None:
                raise RuntimeError("TDC case accumulation produced no statistics")
            tdc_scores, _pair_dice, _pair_valid = tdc_from_components(
                tdc_accumulator["intersection"],
                tdc_accumulator["count1"],
                tdc_accumulator["count2"],
                tdc_finite,
            )
            if self.view_selection_metric == "tdc":
                selection_scores = tdc_scores.detach()
            else:
                selection_scores = scores
        else:
            selection_scores = scores
        cac_rank = (-scores).argsort().argsort().float()
        tdc_rank = None
        if tdc_scores is not None:
            tdc_rank = _average_tie_rank_1d(tdc_scores, descending=True)
        cac_entropy_rank = entropy_scores.argsort().argsort().float()
        tdc_entropy_rank = (
            _average_tie_rank_1d(entropy_scores, descending=False)
            if tdc_scores is not None
            else None
        )
        cac_combined_rank = cac_rank + cac_entropy_rank
        tdc_combined_rank = (
            None
            if tdc_rank is None or tdc_entropy_rank is None
            else tdc_rank + tdc_entropy_rank
        )
        if self.view_selection_metric == "tdc":
            # CM-SFDA uses zero-based average ranks for ties in both TDC and
            # entropy; retain the existing argsort ranks for CAC compatibility.
            entropy_rank = tdc_entropy_rank
            quality_rank = tdc_rank
            quality_entropy_rank = tdc_combined_rank
        else:
            entropy_rank = cac_entropy_rank
            quality_rank = cac_rank
            quality_entropy_rank = cac_combined_rank
        combined_rank = quality_rank if not self.use_entropy_rank else quality_entropy_rank
        if self.view_selection_metric == "tdc" or not self.use_entropy_rank:
            num_selected = max(1, int(len(params) * self.selection_p))
            if num_selected != 1:
                raise ValueError(
                    "VoxTell CM-TTA requires exactly one selected view; "
                    f"selection_p={self.selection_p} with num_views={len(params)} "
                    f"would select {num_selected} views"
                )
            selected = int(torch.argsort(combined_rank, stable=True)[0].item())
        else:
            selected, _ = select_cac_view_from_entropy(
                selection_scores, entropy_scores, self.selection_p
            )
        self.last_view_selection = {
            "cac": scores.cpu().tolist(),
            "tdc": None if tdc_scores is None else tdc_scores.cpu().tolist(),
            "selection_metric": self.view_selection_metric,
            "selection_scores": selection_scores.cpu().tolist(),
            "entropy": entropy_scores.cpu().tolist(),
            "cac_rank": cac_rank.cpu().tolist(),
            "tdc_rank": None if tdc_rank is None else tdc_rank.cpu().tolist(),
            "entropy_rank": entropy_rank.cpu().tolist(),
            "cac_entropy_rank": cac_entropy_rank.cpu().tolist(),
            "tdc_entropy_rank": (
                None if tdc_entropy_rank is None else tdc_entropy_rank.cpu().tolist()
            ),
            "cac_combined_rank": cac_combined_rank.cpu().tolist(),
            "tdc_combined_rank": (
                None if tdc_combined_rank is None else tdc_combined_rank.cpu().tolist()
            ),
            "combined_rank": combined_rank.cpu().tolist(),
            "selected_view": selected,
        }
        # Keep the historical return value (CAC scores) for callers/results;
        # LSPM uses last_view_selection["selection_scores"] below so TDC mode
        # can independently use its primary quality.
        return selected, scores

    def _forward_case_supervision_stats(
        self,
        patches: list[torch.Tensor],
        valid_masks: list[torch.Tensor],
        params: list[dict[str, float]],
        selected_view: int,
        short_ctx_value: torch.Tensor,
        short_current_weight: float,
        long_ctx: torch.Tensor,
        autocast_enabled: bool,
    ) -> dict[str, torch.Tensor]:
        """Collect global Dice/entropy statistics without retaining graphs."""
        total_views = len(params)
        dice_stats = None
        entropy_sum = None
        entropy_mass = None
        with torch.no_grad():
            for patch, valid_mask in zip(patches, valid_masks):
                selected = self._make_view_batch(
                    patch, params, valid_mask, selected_view, selected_view + 1
                ).to(self.device, non_blocking=True)
                input_mask = valid_mask.unsqueeze(0).to(
                    self.device, non_blocking=True
                )
                with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                    pseudo_logits = self._forward(selected, long_ctx)
                    pseudo_label = torch.sigmoid(pseudo_logits[:, :1]).detach()
                for start in range(0, total_views, self.view_batch_size):
                    end = min(total_views, start + self.view_batch_size)
                    view_batch = self._make_view_batch(
                        patch, params, valid_mask, start, end
                    ).to(self.device, non_blocking=True)
                    input_mask_batch = valid_mask.unsqueeze(0).to(
                        self.device, non_blocking=True
                    ).expand(end - start, -1, -1, -1)
                    with torch.autocast(
                        device_type=self.device.type, enabled=autocast_enabled
                    ):
                        student_ctx = short_ctx_value + short_current_weight * (
                            self.ctx_delta - self.ctx_delta.detach()
                        )
                        student_logits = self._forward(view_batch, student_ctx)
                        probabilities = torch.sigmoid(student_logits[:, :1])
                        local_dice = masked_dice_components(
                            probabilities,
                            pseudo_label,
                            input_mask_batch.unsqueeze(1),
                        )
                        local_entropy, local_mass = masked_entropy_components(
                            probabilities[
                                max(0, selected_view - start) :
                                max(0, selected_view - start) + 1
                            ]
                            if start <= selected_view < end
                            else probabilities[:1],
                            input_mask_batch[
                                max(0, selected_view - start) :
                                max(0, selected_view - start) + 1
                            ]
                            if start <= selected_view < end
                            else input_mask_batch[:1],
                        )
                    if dice_stats is None:
                        dice_stats = {
                            key: torch.zeros(
                                (total_views, *value.shape[1:]),
                                device=value.device,
                                dtype=value.dtype,
                            )
                            for key, value in local_dice.items()
                        }
                    for key, value in local_dice.items():
                        dice_stats[key][start:end].add_(value)
                    if start <= selected_view < end:
                        if entropy_sum is None:
                            entropy_sum = torch.zeros_like(local_entropy[0])
                            entropy_mass = torch.zeros_like(local_mass[0])
                        entropy_sum.add_(local_entropy[0])
                        entropy_mass.add_(local_mass[0])
        if dice_stats is None or entropy_sum is None or entropy_mass is None:
            raise RuntimeError("Case supervision accumulation produced no statistics")
        return {
            **dice_stats,
            "entropy_sum": entropy_sum,
            "entropy_mass": entropy_mass,
        }

    def _selected_prompt_region_diagnostics(
        self,
        patches: list[torch.Tensor],
        valid_masks: list[torch.Tensor],
        params: list[dict[str, float]],
        selected_view: int,
        ctx: torch.Tensor,
        pseudo_cache: list[dict[str, torch.Tensor]],
        autocast_enabled: bool,
        prefix: str,
    ) -> dict[str, Optional[float]]:
        """Measure one selected view against a fixed teacher mask set."""
        region_sums = {
            name: 0.0
            for name in ("fg", "bg", "amb", "amb_anchor", "miss")
        }
        region_counts = {name: 0.0 for name in region_sums}
        foreground_volume = 0.0
        with torch.no_grad():
            for patch, valid_mask, cached in zip(patches, valid_masks, pseudo_cache):
                spatial_shape = tuple(int(size) for size in patch.shape[1:])
                input_valid = valid_mask.unsqueeze(0).unsqueeze(1).to(
                    self.device, non_blocking=True
                ).bool()
                input_regions = {
                    name: decoder_grid_to_input_order(
                        cached[name].to(self.device).float(), spatial_shape, mode="nearest"
                    ).bool() & input_valid
                    for name in region_sums
                }
                selected = self._make_view_batch(
                    patch, params, valid_mask, selected_view, selected_view + 1
                ).to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self.device.type, enabled=autocast_enabled
                ):
                    probabilities = torch.sigmoid(
                        self._forward(selected, ctx)[:, :1]
                    ).float()
                foreground_volume += float(
                    ((probabilities > 0.5) & input_valid).sum().cpu()
                )
                for name, mask in input_regions.items():
                    region_sums[name] += float((probabilities * mask.float()).sum().cpu())
                    region_counts[name] += float(mask.sum().cpu())

        diagnostics: dict[str, Optional[float]] = {
            f"{prefix}_foreground_volume": foreground_volume,
        }
        for name in region_sums:
            count = region_counts[name]
            diagnostics[f"{prefix}_{name}_mean_probability"] = (
                region_sums[name] / count if count > 0.0 else None
            )
        return diagnostics

    def _backward_case_decoder_masked_supervision(
        self,
        patches: list[torch.Tensor],
        valid_masks: list[torch.Tensor],
        params: list[dict[str, float]],
        selected_view: int,
        short_ctx_value: torch.Tensor,
        short_current_weight: float,
        long_ctx: torch.Tensor,
        autocast_enabled: bool,
        ctx_delta_before: Optional[torch.Tensor] = None,
    ) -> tuple[float, float]:
        """Backpropagate D2--D5 masked BCE/Tversky and entropy case-wise.

        The first pass keeps only detached decoder masks/targets and additive
        scalar statistics.  The second pass replays student chunks and
        backpropagates the exact derivatives of the case-level reductions,
        so no patch graph is retained and the result is not a patch average.
        """
        total_views = len(params)
        scalar_keys = (
            "bce_fg_sum", "bce_bg_sum", "bce_fg_count", "bce_bg_count",
            "amb_bce_sum", "amb_bce_count",
            "tversky_tp", "tversky_fp", "tversky_fn", "tversky_mass",
            "entropy_sum", "entropy_mass",
        )
        stats = {
            key: torch.zeros(
                (total_views,) if key.startswith("tversky_") else (),
                device=self.device,
                dtype=torch.float32,
            )
            for key in scalar_keys
        }
        pseudo_cache = []
        region_counts = {
            name: 0.0
            for name in ("fg", "bg", "amb", "amb_anchor", "miss", "valid")
        }
        teacher_sums = {
            name: 0.0
            for name in ("fg", "bg", "amb", "amb_anchor", "miss")
        }
        decoder_alignment_max = 0.0
        decoder_alignment_sum = 0.0
        decoder_alignment_mass = 0.0
        aligned_student_sums = {
            name: 0.0
            for name in ("fg", "bg", "amb", "amb_anchor", "miss")
        }
        if ctx_delta_before is None:
            ctx_delta_before = self.ctx_delta.detach().clone()
        else:
            ctx_delta_before = ctx_delta_before.detach().clone()

        with torch.no_grad():
            for patch, valid_mask in zip(patches, valid_masks):
                selected = self._make_view_batch(
                    patch, params, valid_mask, selected_view, selected_view + 1
                ).to(self.device, non_blocking=True)
                decoder_input_mask = valid_mask.unsqueeze(0).to(
                    self.device, non_blocking=True
                )
                with torch.autocast(
                    device_type=self.device.type, enabled=autocast_enabled
                ):
                    decoder_outputs = self._forward_decoder_outputs(selected, long_ctx)
                    decoder_masks = decoder_consistency_probabilities(
                        decoder_outputs,
                        decoder_input_mask,
                        bg_threshold=self.bg_threshold,
                    )
                    if self.decoder_alignment_check:
                        # This diagnostic intentionally uses the same selected
                        # image and the same long context for both interfaces.
                        # It validates the real VoxTell contract without
                        # changing the training graph or loss.
                        normal_logits = self._forward(selected, long_ctx)[:, :1]
                        d5_logits = decoder_outputs[0][:, :1]
                        if tuple(normal_logits.shape) != tuple(d5_logits.shape):
                            raise RuntimeError(
                                "VoxTell normal logits and D5 logits have different "
                                f"shapes: normal={tuple(normal_logits.shape)} "
                                f"d5={tuple(d5_logits.shape)}"
                            )
                        difference = (normal_logits.float() - d5_logits.float()).abs()
                        decoder_alignment_max = max(
                            decoder_alignment_max, float(difference.max().cpu())
                        )
                        decoder_alignment_sum += float(difference.sum().cpu())
                        decoder_alignment_mass += float(difference.numel())
                        if not torch.allclose(
                            normal_logits, d5_logits, atol=1e-5, rtol=1e-5
                        ):
                            raise RuntimeError(
                                "VoxTell normal logits and D5 logits are not aligned: "
                                f"max_abs={decoder_alignment_max:.6g}"
                            )
                        aligned_student_probabilities = torch.sigmoid(
                            normal_logits.float()
                        )
                p5_d5 = decoder_masks["p5"]
                fg_d5 = decoder_masks["fg"]
                bg_d5 = decoder_masks["bg"]
                amb_d5 = decoder_masks["amb"]
                miss_d5 = decoder_masks["miss"]
                amb_anchor_d5 = amb_d5 & ~miss_d5
                if bool((amb_anchor_d5 & miss_d5).any()):
                    raise RuntimeError("M_amb_anchor must be disjoint from M_miss")
                valid_d5 = decoder_masks["valid"]
                for name, mask in (
                    ("fg", fg_d5), ("bg", bg_d5), ("amb", amb_d5),
                    ("amb_anchor", amb_anchor_d5), ("miss", miss_d5),
                    ("valid", valid_d5),
                ):
                    region_counts[name] += float(mask.sum().cpu())
                for name, mask in (
                    ("fg", fg_d5),
                    ("bg", bg_d5),
                    ("amb", amb_d5),
                    ("amb_anchor", amb_anchor_d5),
                    ("miss", miss_d5),
                ):
                    teacher_sums[name] += float((p5_d5 * mask.float()).sum().cpu())
                    if self.decoder_alignment_check:
                        aligned_student_sums[name] += float(
                            (aligned_student_probabilities * mask.float()).sum().cpu()
                        )

                # Save only detached CPU tensors.  The teacher decoder output
                # is reused through these masks/targets; no second teacher
                # forward is needed in the differentiable replay.
                pseudo_cache.append(
                    {
                        key: value.cpu()
                        for key, value in {
                            "target": p5_d5,
                            "fg": fg_d5,
                            "bg": bg_d5,
                            "amb": amb_d5,
                            "amb_anchor": amb_anchor_d5,
                            "miss": miss_d5,
                            "valid": valid_d5,
                        }.items()
                    }
                )

                for start in range(0, total_views, self.view_batch_size):
                    end = min(total_views, start + self.view_batch_size)
                    view_batch = self._make_view_batch(
                        patch, params, valid_mask, start, end
                    ).to(self.device, non_blocking=True)
                    with torch.autocast(
                        device_type=self.device.type, enabled=autocast_enabled
                    ):
                        student_ctx = short_ctx_value + short_current_weight * (
                            self.ctx_delta - self.ctx_delta.detach()
                        )
                        student_logits = self._forward(view_batch, student_ctx)
                        student_probabilities = torch.sigmoid(student_logits[:, :1])
                        spatial_shape = tuple(int(size) for size in student_logits.shape[2:])
                        target_input = decoder_grid_to_input_order(
                            p5_d5, spatial_shape, mode="trilinear"
                        )
                        fg_input = decoder_grid_to_input_order(
                            fg_d5.float(), spatial_shape, mode="nearest"
                        ).bool()
                        bg_input = decoder_grid_to_input_order(
                            bg_d5.float(), spatial_shape, mode="nearest"
                        ).bool()
                        valid_input = valid_mask.unsqueeze(0).unsqueeze(1).to(
                            self.device
                        ).bool()
                        fg_input = fg_input & valid_input
                        bg_input = bg_input & valid_input
                        region_mask = fg_input | bg_input
                        target_batch = target_input.expand(end - start, -1, -1, -1, -1)
                        fg_batch = fg_input.expand(end - start, -1, -1, -1, -1)
                        bg_batch = bg_input.expand(end - start, -1, -1, -1, -1)
                        known_batch = region_mask.expand(end - start, -1, -1, -1, -1)
                        bce_map = F.binary_cross_entropy_with_logits(
                            student_logits[:, :1].float(), target_batch.float(), reduction="none"
                        )
                        stats["bce_fg_sum"].add_((bce_map * fg_batch.float()).sum())
                        stats["bce_bg_sum"].add_((bce_map * bg_batch.float()).sum())
                        stats["bce_fg_count"].add_(fg_batch.float().sum())
                        stats["bce_bg_count"].add_(bg_batch.float().sum())
                        amb_anchor_input = decoder_grid_to_input_order(
                            amb_anchor_d5.float(), spatial_shape, mode="nearest"
                        ).bool() & valid_input
                        amb_anchor_batch = amb_anchor_input.expand(
                            end - start, -1, -1, -1, -1
                        )
                        amb_bce_map = F.binary_cross_entropy_with_logits(
                            student_logits[:, :1].float(),
                            target_batch.float(),
                            reduction="none",
                        )
                        stats["amb_bce_sum"].add_(
                            (amb_bce_map * amb_anchor_batch.float()).sum()
                        )
                        stats["amb_bce_count"].add_(amb_anchor_batch.float().sum())
                        probabilities = student_probabilities.float()
                        target_float = target_batch.float()
                        stats["tversky_tp"][start:end].add_(
                            (probabilities * target_float * known_batch)
                            .flatten(start_dim=1)
                            .sum(dim=1)
                        )
                        stats["tversky_fp"][start:end].add_(
                            (probabilities * (1.0 - target_float) * known_batch)
                            .flatten(start_dim=1)
                            .sum(dim=1)
                        )
                        stats["tversky_fn"][start:end].add_(
                            ((1.0 - probabilities) * target_float * known_batch)
                            .flatten(start_dim=1)
                            .sum(dim=1)
                        )
                        stats["tversky_mass"][start:end].add_(
                            known_batch.flatten(start_dim=1).sum(dim=1)
                        )
                        if start <= selected_view < end:
                            selected_index = selected_view - start
                            selected_probability = probabilities[selected_index:selected_index + 1]
                            entropy_sum, entropy_mass = masked_entropy_components(
                                selected_probability,
                                valid_mask.unsqueeze(0).unsqueeze(1),
                            )
                            stats["entropy_sum"].add_(entropy_sum[0])
                            stats["entropy_mass"].add_(entropy_mass[0])

        # The diagnostic before-volume and regional student probabilities use
        # the exact same context type as the post-update diagnostic below:
        # the raw token-tuning delta, never the short-context bridge.
        before_student_diagnostics = self._selected_prompt_region_diagnostics(
            patches,
            valid_masks,
            params,
            selected_view,
            ctx_delta_before,
            pseudo_cache,
            autocast_enabled,
            "student_before",
        )
        self._last_decoder_pseudo_cache = pseudo_cache

        # Build the exact scalar case reductions from detached statistics and
        # obtain coefficients for the graph replay below.
        bce_inputs = {}
        if float(stats["bce_fg_count"]) > 0.0:
            bce_inputs["bce_fg_sum"] = stats["bce_fg_sum"].detach().requires_grad_(True)
        if float(stats["bce_bg_count"]) > 0.0:
            bce_inputs["bce_bg_sum"] = stats["bce_bg_sum"].detach().requires_grad_(True)
        global_bce = masked_balanced_bce_from_components(
            bce_inputs.get("bce_fg_sum", stats["bce_fg_sum"].detach()),
            bce_inputs.get("bce_bg_sum", stats["bce_bg_sum"].detach()),
            stats["bce_fg_count"].detach(),
            stats["bce_bg_count"].detach(),
        )
        if bce_inputs:
            bce_derivatives = dict(zip(
                bce_inputs,
                torch.autograd.grad(global_bce, tuple(bce_inputs.values())),
            ))
        else:
            global_bce = torch.zeros((), device=self.device)
            bce_derivatives = {}

        amb_bce_input = None
        if float(stats["amb_bce_count"]) > 0.0:
            amb_bce_input = stats["amb_bce_sum"].detach().requires_grad_(True)
            global_amb_loss = amb_bce_input / stats["amb_bce_count"].detach().clamp_min(1.0)
            amb_derivative = torch.autograd.grad(global_amb_loss, amb_bce_input)[0]
        else:
            global_amb_loss = torch.zeros((), device=self.device)
            amb_derivative = None

        tv_inputs = tuple(
            stats[key].detach().requires_grad_(True)
            for key in ("tversky_tp", "tversky_fp", "tversky_fn")
        )
        tversky_valid = stats["tversky_mass"].detach() > 0.0
        if bool(tversky_valid.any()):
            global_tversky, per_view_tversky = masked_tversky_loss_from_components(
                tv_inputs[0],
                tv_inputs[1],
                tv_inputs[2],
                alpha=self.tversky_alpha,
                beta=self.tversky_beta,
                valid_mass=stats["tversky_mass"].detach(),
            )
            raw_tv_derivatives = torch.autograd.grad(global_tversky, tv_inputs)
            tv_derivatives = tuple(
                derivative * tversky_valid.to(derivative.dtype)
                for derivative in raw_tv_derivatives
            )
        else:
            global_tversky = torch.zeros((), device=self.device)
            tv_derivatives = (None, None, None)

        entropy_sum = stats["entropy_sum"].detach().requires_grad_(True)
        entropy_mass = stats["entropy_mass"].detach().requires_grad_(True)
        global_entropy = entropy_sum / entropy_mass.clamp_min(1.0)
        entropy_derivatives = torch.autograd.grad(global_entropy, (entropy_sum, entropy_mass))

        for patch, valid_mask, cached in zip(patches, valid_masks, pseudo_cache):
            input_mask = valid_mask.unsqueeze(0).unsqueeze(1).to(self.device, non_blocking=True).bool()
            for start in range(0, total_views, self.view_batch_size):
                end = min(total_views, start + self.view_batch_size)
                view_batch = self._make_view_batch(
                    patch, params, valid_mask, start, end
                ).to(self.device, non_blocking=True)
                with torch.autocast(
                    device_type=self.device.type, enabled=autocast_enabled
                ):
                    student_ctx = short_ctx_value + short_current_weight * (
                        self.ctx_delta - self.ctx_delta.detach()
                    )
                    student_logits = self._forward(view_batch, student_ctx)[:, :1]
                    spatial_shape = tuple(int(size) for size in student_logits.shape[2:])
                    target_input = decoder_grid_to_input_order(
                        cached["target"].to(self.device), spatial_shape, mode="trilinear"
                    )
                    fg_input = decoder_grid_to_input_order(
                        cached["fg"].to(self.device).float(), spatial_shape, mode="nearest"
                    ).bool() & input_mask
                    bg_input = decoder_grid_to_input_order(
                        cached["bg"].to(self.device).float(), spatial_shape, mode="nearest"
                    ).bool() & input_mask
                    amb_anchor_input = decoder_grid_to_input_order(
                        cached["amb_anchor"].to(self.device).float(),
                        spatial_shape,
                        mode="nearest",
                    ).bool() & input_mask
                    known_input = fg_input | bg_input
                    target_batch = target_input.expand(end - start, -1, -1, -1, -1)
                    fg_batch = fg_input.expand(end - start, -1, -1, -1, -1)
                    bg_batch = bg_input.expand(end - start, -1, -1, -1, -1)
                    known_batch = known_input.expand(end - start, -1, -1, -1, -1)
                    differentiable = []
                    bce_map = F.binary_cross_entropy_with_logits(
                        student_logits.float(), target_batch.float(), reduction="none"
                    )
                    if "bce_fg_sum" in bce_derivatives:
                        differentiable.append(
                            ((bce_map * fg_batch.float()).sum(), bce_derivatives["bce_fg_sum"])
                        )
                    if "bce_bg_sum" in bce_derivatives:
                        differentiable.append(
                            ((bce_map * bg_batch.float()).sum(), bce_derivatives["bce_bg_sum"])
                        )
                    if amb_derivative is not None:
                        amb_anchor_batch = amb_anchor_input.expand(
                            end - start, -1, -1, -1, -1
                        )
                        amb_bce_map = F.binary_cross_entropy_with_logits(
                            student_logits.float(),
                            target_batch.float(),
                            reduction="none",
                        )
                        differentiable.append(
                            (
                                (amb_bce_map * amb_anchor_batch.float()).sum(),
                                amb_derivative * self.amb_weight,
                            )
                        )
                    if bool(tversky_valid[start:end].any()):
                        probabilities = torch.sigmoid(student_logits.float())
                        target_float = target_batch.float()
                        local_tv = (
                            (probabilities * target_float * known_batch)
                            .flatten(start_dim=1)
                            .sum(dim=1),
                            (probabilities * (1.0 - target_float) * known_batch)
                            .flatten(start_dim=1)
                            .sum(dim=1),
                            ((1.0 - probabilities) * target_float * known_batch)
                            .flatten(start_dim=1)
                            .sum(dim=1),
                        )
                        differentiable.extend(
                            (tensor, self.tversky_weight * derivative[start:end])
                            for tensor, derivative in zip(local_tv, tv_derivatives)
                        )
                    if start <= selected_view < end:
                        selected_index = selected_view - start
                        probabilities = torch.sigmoid(student_logits.float())
                        local_entropy_sum, _ = masked_entropy_components(
                            probabilities[selected_index:selected_index + 1],
                            input_mask,
                        )
                        differentiable.append(
                            (local_entropy_sum[0], entropy_derivatives[0] * self.w_entropy)
                        )
                if differentiable:
                    local_objective = sum(
                        (tensor * derivative).sum()
                        for tensor, derivative in differentiable
                    )
                    self.scaler.scale(local_objective).backward()

        valid_count = max(region_counts["valid"], 1.0)

        def _mean_or_none(total: float, count: float) -> Optional[float]:
            return total / count if count > 0.0 else None

        diagnostics = {
            "pseudo_update_mode": self.pseudo_update_mode,
            "M_fg_count": int(region_counts["fg"]),
            "M_bg_count": int(region_counts["bg"]),
            "M_amb_count": int(region_counts["amb"]),
            "M_amb_anchor_count": int(region_counts["amb_anchor"]),
            "M_miss_count": int(region_counts["miss"]),
            "M_fg_fraction": region_counts["fg"] / valid_count,
            "M_bg_fraction": region_counts["bg"] / valid_count,
            "M_amb_fraction": region_counts["amb"] / valid_count,
            "M_amb_anchor_fraction": region_counts["amb_anchor"] / valid_count,
            "M_miss_fraction": region_counts["miss"] / valid_count,
            "teacher_fg_mean_probability": _mean_or_none(
                teacher_sums["fg"], region_counts["fg"]
            ),
            "teacher_bg_mean_probability": _mean_or_none(
                teacher_sums["bg"], region_counts["bg"]
            ),
            "teacher_amb_mean_probability": _mean_or_none(
                teacher_sums["amb"], region_counts["amb"]
            ),
            "teacher_amb_anchor_mean_probability": _mean_or_none(
                teacher_sums["amb_anchor"], region_counts["amb_anchor"]
            ),
            "teacher_miss_mean_probability": _mean_or_none(
                teacher_sums["miss"], region_counts["miss"]
            ),
            "pseudo_bce": float(global_bce.detach().cpu()),
            "bce_loss": float(global_bce.detach().cpu()),
            "pseudo_tversky": float(global_tversky.detach().cpu()),
            "tversky_loss": float((self.tversky_weight * global_tversky).detach().cpu()),
            "amb_loss": float(global_amb_loss.detach().cpu()),
            "weighted_amb_loss": float(
                (self.amb_weight * global_amb_loss).detach().cpu()
            ),
            "amb_weight": self.amb_weight,
            **before_student_diagnostics,
        }
        if self.decoder_alignment_check:
            same_context = torch.allclose(long_ctx, ctx_delta_before, atol=1e-7, rtol=1e-7)
            diagnostics.update(
                {
                    "decoder_alignment_max_abs_error": decoder_alignment_max,
                    "decoder_alignment_mean_abs_error": (
                        decoder_alignment_sum / decoder_alignment_mass
                        if decoder_alignment_mass > 0.0
                        else 0.0
                    ),
                    "decoder_alignment_same_context": bool(same_context),
                }
            )
            for name in ("fg", "bg", "amb", "miss"):
                count = region_counts[name]
                teacher_mean = _mean_or_none(teacher_sums[name], count)
                aligned_mean = _mean_or_none(aligned_student_sums[name], count)
                diagnostics[f"aligned_student_{name}_mean_probability"] = aligned_mean
                if teacher_mean is not None and aligned_mean is not None and not np.isclose(
                    teacher_mean, aligned_mean, atol=1e-5, rtol=1e-5
                ):
                    raise RuntimeError(
                        "VoxTell teacher/student region probabilities are not aligned "
                        f"for {name}: teacher={teacher_mean:.6g} "
                        f"student={aligned_mean:.6g}"
                    )
                before_mean = diagnostics.get(
                    f"student_before_{name}_mean_probability"
                )
                if (
                    same_context
                    and teacher_mean is not None
                    and before_mean is not None
                    and not np.isclose(teacher_mean, before_mean, atol=1e-5, rtol=1e-5)
                ):
                    raise RuntimeError(
                        "VoxTell same-context teacher/student region probabilities "
                        f"are not aligned for {name}: teacher={teacher_mean:.6g} "
                        f"student_before={before_mean:.6g}"
                    )
        diagnostics.update(
            {
                "teacher_mean_fg_probability": diagnostics["teacher_fg_mean_probability"],
                "teacher_mean_bg_probability": diagnostics["teacher_bg_mean_probability"],
                "teacher_mean_amb_probability": diagnostics["teacher_amb_mean_probability"],
                "teacher_mean_amb_anchor_probability": diagnostics[
                    "teacher_amb_anchor_mean_probability"
                ],
                "teacher_mean_miss_probability": diagnostics["teacher_miss_mean_probability"],
                # Compatibility aliases now explicitly refer to the raw
                # pre-update ctx-delta diagnostic, not the short bridge.
                "student_fg_mean_probability": diagnostics[
                    "student_before_fg_mean_probability"
                ],
                "student_bg_mean_probability": diagnostics[
                    "student_before_bg_mean_probability"
                ],
                "student_amb_mean_probability": diagnostics[
                    "student_before_amb_mean_probability"
                ],
                "student_miss_mean_probability": diagnostics[
                    "student_before_miss_mean_probability"
                ],
                "prediction_foreground_volume_before": diagnostics[
                    "student_before_foreground_volume"
                ],
            }
        )
        self._last_pseudo_diagnostics = diagnostics
        pseudo_loss = (
            global_bce
            + self.tversky_weight * global_tversky
            + self.amb_weight * global_amb_loss
        )
        return float(pseudo_loss.detach().cpu()), float(global_entropy.detach().cpu())

    def _backward_case_supervision(
        self,
        patches: list[torch.Tensor],
        valid_masks: list[torch.Tensor],
        params: list[dict[str, float]],
        selected_view: int,
        short_ctx_value: torch.Tensor,
        short_current_weight: float,
        long_ctx: torch.Tensor,
        autocast_enabled: bool,
        ctx_delta_before: Optional[torch.Tensor] = None,
    ) -> tuple[float, float]:
        """Backpropagate case-level Dice and entropy with one patch graph."""
        if self.pseudo_update_mode == "decoder_masked":
            return self._backward_case_decoder_masked_supervision(
                patches,
                valid_masks,
                params,
                selected_view,
                short_ctx_value,
                short_current_weight,
                long_ctx,
                autocast_enabled,
                ctx_delta_before,
            )
        self._last_pseudo_diagnostics = {"pseudo_update_mode": "original"}
        stats = self._forward_case_supervision_stats(
            patches,
            valid_masks,
            params,
            selected_view,
            short_ctx_value,
            short_current_weight,
            long_ctx,
            autocast_enabled,
        )
        global_dice_inputs = tuple(
            stats[key].detach().requires_grad_(True)
            for key in ("intersection", "prediction_mass", "pseudo_mass")
        )
        global_dice = (
            1.0
            - 2.0 * global_dice_inputs[0]
            / (global_dice_inputs[1] + global_dice_inputs[2] + EPS)
        ).mean()
        dice_derivatives = torch.autograd.grad(global_dice, global_dice_inputs)
        global_entropy_sum = stats["entropy_sum"].detach().requires_grad_(True)
        global_entropy_mass = stats["entropy_mass"].detach().requires_grad_(True)
        global_entropy = global_entropy_sum / global_entropy_mass.clamp_min(1.0)
        entropy_derivatives = torch.autograd.grad(
            global_entropy, (global_entropy_sum, global_entropy_mass)
        )

        total_views = len(params)
        for patch, valid_mask in zip(patches, valid_masks):
            selected = self._make_view_batch(
                patch, params, valid_mask, selected_view, selected_view + 1
            ).to(self.device, non_blocking=True)
            input_mask = valid_mask.unsqueeze(0).to(
                self.device, non_blocking=True
            )
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type, enabled=autocast_enabled
            ):
                pseudo_logits = self._forward(selected, long_ctx)
                pseudo_label = torch.sigmoid(pseudo_logits[:, :1]).detach()
            for start in range(0, total_views, self.view_batch_size):
                end = min(total_views, start + self.view_batch_size)
                view_batch = self._make_view_batch(
                    patch, params, valid_mask, start, end
                ).to(self.device, non_blocking=True)
                input_mask_batch = valid_mask.unsqueeze(0).to(
                    self.device, non_blocking=True
                ).expand(end - start, -1, -1, -1)
                with torch.autocast(
                    device_type=self.device.type, enabled=autocast_enabled
                ):
                    student_ctx = short_ctx_value + short_current_weight * (
                        self.ctx_delta - self.ctx_delta.detach()
                    )
                    student_logits = self._forward(view_batch, student_ctx)
                    probabilities = torch.sigmoid(student_logits[:, :1])
                    local_dice = masked_dice_components(
                        probabilities,
                        pseudo_label,
                        input_mask_batch.unsqueeze(1),
                    )
                    local_tensors = [
                        local_dice["intersection"],
                        local_dice["prediction_mass"],
                    ]
                    local_gradients = [
                        dice_derivatives[0][start:end],
                        dice_derivatives[1][start:end],
                    ]
                    if start <= selected_view < end:
                        selected_index = selected_view - start
                        local_entropy_sum, local_entropy_mass = masked_entropy_components(
                            probabilities[selected_index:selected_index + 1],
                            input_mask_batch[selected_index:selected_index + 1],
                        )
                        local_tensors.append(local_entropy_sum)
                        local_gradients.append(
                            entropy_derivatives[0] * self.w_entropy
                        )
                differentiable = [
                    (tensor, gradient)
                    for tensor, gradient in zip(local_tensors, local_gradients)
                    if tensor.requires_grad
                ]
                if differentiable:
                    local_objective = sum(
                        (tensor * gradient).sum() for tensor, gradient in differentiable
                    )
                    self.scaler.scale(local_objective).backward()
        return float(global_dice.detach().cpu()), float(global_entropy.detach().cpu())

    def _backward_case_cac(
        self,
        patches: list[torch.Tensor],
        valid_masks: list[torch.Tensor],
        params: list[dict[str, float]],
        selected_view: int,
        short_ctx_value: torch.Tensor,
        short_current_weight: float,
        autocast_enabled: bool,
    ) -> float:
        """Backpropagate one exact global selected-view CAC with low memory."""
        # First collect the exact case-level CAC inputs without autograd.  A
        # later pass applies the global CAC derivative patch by patch, so no
        # collection of patch graphs is needed.
        case_components = None
        with torch.no_grad():
            for patch, valid_mask in zip(patches, valid_masks):
                selected = self._make_view_batch(
                    patch, params, valid_mask, selected_view, selected_view + 1
                ).to(self.device, non_blocking=True)
                input_mask = valid_mask.unsqueeze(0).to(
                    self.device, non_blocking=True
                )
                with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                    student_ctx = short_ctx_value + short_current_weight * (
                        self.ctx_delta - self.ctx_delta.detach()
                    )
                    selected_logits = self._forward(selected, student_ctx)
                    components = self._cac_components(selected_logits, input_mask)
                    case_components = self._add_components(case_components, components)
        if case_components is None:
            raise RuntimeError("Selected-view case CAC accumulation produced no statistics")

        # Differentiate the one global similarity-map CAC expression with
        # respect to its four aggregate inputs.  Each derivative is then supplied to a
        # one-patch autograd graph below; this is exact chain-rule gradient
        # accumulation and has the same result as retaining every patch graph.
        global_inputs = tuple(
            case_components[key].detach().requires_grad_(True)
            for key in (
                "foreground_sum",
                "foreground_mass",
                "background_sum",
                "background_mass",
            )
        )
        case_cac_graph = cac_from_components(*global_inputs)
        global_derivatives = torch.autograd.grad(case_cac_graph[0], global_inputs)
        cac_loss = -case_cac_graph[0].detach()

        # Backpropagate the derivative of w_cac * (-case_cac), one selected
        # view/patch at a time.  The current graph is released every iteration.
        for patch, valid_mask in zip(patches, valid_masks):
            selected = self._make_view_batch(
                patch, params, valid_mask, selected_view, selected_view + 1
            ).to(self.device, non_blocking=True)
            input_mask = valid_mask.unsqueeze(0).to(
                self.device, non_blocking=True
            )
            with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                student_ctx = short_ctx_value + short_current_weight * (
                    self.ctx_delta - self.ctx_delta.detach()
                )
                selected_logits = self._forward(selected, student_ctx)
                components = self._cac_components(selected_logits, input_mask)
            local_inputs = (
                components["foreground_sum"],
                components["foreground_mass"],
                components["background_sum"],
                components["background_mass"],
            )
            local_tensors = []
            local_gradients = []
            for local, derivative in zip(local_inputs, global_derivatives):
                if local.requires_grad:
                    local_tensors.append(local)
                    local_gradients.append(derivative * (-self.w_cac))
            if local_tensors:
                local_objective = sum(
                    (tensor * gradient).sum()
                    for tensor, gradient in zip(local_tensors, local_gradients)
                )
                self.scaler.scale(local_objective).backward()
        return float(cac_loss.cpu())

    def prepare_case(
        self,
        patches: list[torch.Tensor],
        valid_masks: Optional[list[torch.Tensor]] = None,
    ) -> dict:
        """Prepare one case without selecting views or updating any parameter."""
        if not patches:
            raise ValueError("prepare_case received no patches")
        if valid_masks is None:
            valid_masks = [torch.ones(patch.shape[-3:]) for patch in patches]
        if len(valid_masks) != len(patches):
            raise ValueError("patches and valid_masks must have equal lengths")
        patches = [patch.float().contiguous() for patch in patches]
        normalized_masks = []
        for mask, patch in zip(valid_masks, patches):
            if mask.ndim == 4 and mask.shape[0] == 1:
                mask = mask[0]
            if mask.ndim != 3 or tuple(mask.shape) != tuple(patch.shape[-3:]):
                raise ValueError(
                    "Each valid mask must have shape (D,H,W) matching its patch, "
                    f"got {tuple(mask.shape)} vs {tuple(patch.shape[-3:])}"
                )
            normalized_masks.append(mask.float().contiguous())
        valid_masks = normalized_masks
        params = self._sample_intensity_params(self.num_aug_views)
        short_ctx, current_quality, historical_quality, weight_historical = self._dynamic_short_ctx(
            patches, valid_masks
        )

        if self.long_delta is None:
            long_ctx = short_ctx.detach().clone()
        else:
            long_ctx = self.ema_momentum * self.long_delta + (1.0 - self.ema_momentum) * short_ctx.detach()
        long_ctx = long_ctx.detach()
        return {
            "patches": patches,
            "valid_masks": valid_masks,
            "params": params,
            "short_ctx": short_ctx,
            "current_quality": current_quality,
            "historical_quality": historical_quality,
            # Compatibility fields retained for existing result consumers.
            "current_cac": current_quality,
            "historical_cac": historical_quality,
            "weight_historical": weight_historical,
            "long_ctx": long_ctx,
        }

    def adapt_case(
        self,
        patches: list[torch.Tensor],
        valid_masks: Optional[list[torch.Tensor]] = None,
        prepared_case: Optional[dict] = None,
    ) -> dict:
        """Adapt once on one complete case, aggregating all patch gradients."""
        if prepared_case is None:
            prepared_case = self.prepare_case(patches, valid_masks)
        patches = prepared_case["patches"]
        valid_masks = prepared_case["valid_masks"]
        params = prepared_case["params"]
        short_ctx = prepared_case["short_ctx"]
        current_quality = prepared_case.get("current_quality", prepared_case["current_cac"])
        historical_quality = prepared_case.get(
            "historical_quality", prepared_case["historical_cac"]
        )
        weight_historical = prepared_case["weight_historical"]
        long_ctx = prepared_case["long_ctx"]

        selected_view, selection_scores = self._select_case_view(
            patches, params, short_ctx, valid_masks
        )
        short_snapshot = short_ctx.detach().clone()
        short_ctx_value = short_snapshot
        short_current_weight = 1.0 - weight_historical
        # Diagnostics must compare the exact raw prompt delta before and
        # after this case update.  In particular, the pre-update volume must
        # not use the short-context bridge used by the training objective.
        ctx_delta_before = self.ctx_delta.detach().clone()
        self.optimizer.zero_grad(set_to_none=True)
        sums = {"soft_dice": 0.0, "cac_loss": 0.0, "entropy_loss": 0.0, "loss": 0.0}
        autocast_enabled = self.device.type == "cuda"
        soft_dice, entropy_loss = self._backward_case_supervision(
            patches,
            valid_masks,
            params,
            selected_view,
            short_ctx_value,
            short_current_weight,
            long_ctx,
            autocast_enabled,
            ctx_delta_before,
        )
        sums["soft_dice"] = soft_dice
        if self.pseudo_update_mode == "decoder_masked":
            # Keep the legacy key for result consumers, while explicitly
            # exposing that this value is the replacement pseudo loss.
            sums["pseudo_loss"] = soft_dice
            sums.update(self._last_pseudo_diagnostics)
            sums["pseudo_loss_type"] = (
                "masked_balanced_bce_tversky_amb_anchor"
                if self.amb_weight > 0
                else "masked_balanced_bce_tversky"
            )
            sums["legacy_soft_dice_field_is_pseudo_loss"] = True
        sums["entropy_loss"] = entropy_loss
        sums["loss"] = soft_dice + self.w_entropy * entropy_loss

        cac_loss = self._backward_case_cac(
            patches,
            valid_masks,
            params,
            selected_view,
            short_ctx_value,
            short_current_weight,
            autocast_enabled,
        )
        sums["cac_loss"] = float(cac_loss)
        sums["loss"] += self.w_cac * sums["cac_loss"]
        sums["total_loss"] = sums["loss"]

        # Check that backward produced a real ctx gradient before handing it
        # to GradScaler.  Non-finite values are allowed to reach the scaler:
        # an AMP overflow is a recoverable skipped step, not a protocol error.
        self._check_case_gradients(allow_nonfinite=True)
        scale_before_step = float(self.scaler.get_scale())
        self.scaler.unscale_(self.optimizer)
        if torch.isfinite(self.ctx_delta.grad).all():
            self._check_case_gradients()
        else:
            warnings.warn(
                "GradScaler detected a non-finite ctx gradient; this case's "
                "optimizer step may be skipped",
                RuntimeWarning,
                stacklevel=2,
            )
        torch.nn.utils.clip_grad_norm_([self.ctx_delta], float(self.args.grad_clip))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after_step = float(self.scaler.get_scale())
        optimizer_step_skipped = scale_after_step < scale_before_step
        if optimizer_step_skipped:
            warnings.warn(
                "GradScaler skipped the ctx optimizer step after AMP overflow",
                RuntimeWarning,
                stacklevel=2,
            )
        self.optimizer_step_count += 1

        if self.pseudo_update_mode == "decoder_masked":
            post_diagnostics = self._selected_prompt_region_diagnostics(
                patches,
                valid_masks,
                params,
                selected_view,
                self.ctx_delta.detach(),
                self._last_decoder_pseudo_cache,
                autocast_enabled,
                "student_after",
            )
            sums.update(post_diagnostics)
            sums["prediction_foreground_volume_after"] = post_diagnostics[
                "student_after_foreground_volume"
            ]
            self._last_decoder_pseudo_cache = None

        self.short_delta = short_snapshot
        self.long_delta = long_ctx
        selected_cac = float(selection_scores[selected_view].detach().cpu())
        # Keep compatibility with callers/tests that provide a custom view
        # selector without populating ``last_view_selection``.  The real
        # selector always stores the configured primary (CAC or TDC) quality.
        selection_quality = self.last_view_selection.get("selection_scores")
        if selection_quality is None:
            selected_quality = selected_cac
        else:
            selected_quality = float(selection_quality[selected_view])
        # Store the actual short prompt paired with the configured primary
        # case/view quality.  Entropy is never used by LSPM.
        self.short_memory.append_delta(self.short_delta, selected_quality)
        self.last_trace = {
            "selected_view": selected_view,
            "pseudo_source_view": selected_view,
            "selected_view_scale": params[selected_view]["scale"],
            "selected_view_offset": params[selected_view]["offset"],
            "view_params": [dict(param) for param in params],
            "view_selection": dict(self.last_view_selection),
            "num_views": 1 + self.num_aug_views,
            "num_patches": len(patches),
            "optimizer_steps_for_case": 1,
            "optimizer_step_skipped": optimizer_step_skipped,
            "current_quality": current_quality,
            "historical_quality": historical_quality,
            "current_cac": current_quality,
            "historical_cac": historical_quality,
            "historical_weight": weight_historical,
            "selected_quality": selected_quality,
            "selected_cac": selected_cac,
            **sums,
        }
        return dict(self.last_trace)

    def state_dict(self) -> dict:
        return {
            "ctx_delta": self.ctx_delta.detach().cpu(),
            "initial_ctx_delta": self.initial_ctx_delta.detach().cpu(),
            "short_delta": None if self.short_delta is None else self.short_delta.cpu(),
            "long_delta": None if self.long_delta is None else self.long_delta.cpu(),
            "ctx_delta_memory": self.short_memory.state_dict(),
            "text_prompt": self.text_prompt,
            "n_ctx": self.n_ctx,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "optimizer_step_count": self.optimizer_step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        if "ctx" in state or "initial_ctx" in state or "ctx_memory" in state:
            raise ValueError(
                "Legacy random-ctx checkpoint is incompatible with ctx_delta format"
            )
        required = {
            "ctx_delta",
            "initial_ctx_delta",
            "short_delta",
            "long_delta",
            "ctx_delta_memory",
            "text_prompt",
            "n_ctx",
            "optimizer",
            "scaler",
        }
        missing = sorted(required.difference(state))
        if missing:
            raise ValueError(
                "ctx_delta checkpoint is incomplete; missing keys: "
                + ", ".join(missing)
            )
        if state["text_prompt"] != self.text_prompt or int(state["n_ctx"]) != self.n_ctx:
            raise ValueError(
                "ctx_delta checkpoint prompt/token count does not match the current "
                "fixed VoxTell tokenizer"
            )
        delta = state["ctx_delta"].to(self.device)
        if tuple(delta.shape) != tuple(self.ctx_delta.shape):
            raise ValueError(
                "ctx_delta shape does not match the current tokenizer-derived liver token count: "
                f"checkpoint={tuple(delta.shape)}, current={tuple(self.ctx_delta.shape)}"
            )
        self.ctx_delta.data.copy_(delta)
        self.initial_ctx_delta.copy_(state["initial_ctx_delta"].to(self.device))
        self.short_delta = (
            None if state["short_delta"] is None else state["short_delta"].to(self.device)
        )
        self.long_delta = (
            None if state["long_delta"] is None else state["long_delta"].to(self.device)
        )
        self.short_memory.load_state_dict(state["ctx_delta_memory"])
        self.optimizer.load_state_dict(state["optimizer"])
        for optimizer_state in self.optimizer.state.values():
            for key, value in optimizer_state.items():
                if torch.is_tensor(value):
                    optimizer_state[key] = value.to(self.device)
        self.scaler.load_state_dict(state.get("scaler", {}))
        self.optimizer_step_count = int(state.get("optimizer_step_count", 0))


def save_cmtta_checkpoint(path: str, adapter: VoxTellCMTTA, args, history: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "voxtell-cmtta-lspm-dspu-ctx-delta-v1",
            "adapter": adapter.state_dict(),
            "args": vars(args),
            "history": history,
        },
        path,
    )


def load_cmtta_checkpoint(path: str, adapter: VoxTellCMTTA) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") == "voxtell-cmtta-lspm-dspu-ctx-v1":
        raise ValueError(
            "Legacy random-ctx checkpoint is incompatible with ctx_delta format"
        )
    if checkpoint.get("format") != "voxtell-cmtta-lspm-dspu-ctx-delta-v1":
        raise ValueError(f"Unsupported CM-TTA checkpoint format: {checkpoint.get('format')}")
    adapter.load_state_dict(checkpoint["adapter"])
    return checkpoint
