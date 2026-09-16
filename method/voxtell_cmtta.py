"""Original CM-TTA adapted to VoxTell's frozen 3-D network.

The only model-specific changes are the VoxTell text-embedding interface and
the extension of CAC, entropy, and soft Dice from 2-D to 3-D tensors. LSPM and
DSPU follow CM-TTA equations (3)--(8): one optimizer update is made per full
case, after losses from all of that case's patches have been accumulated.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


EPS = 1e-8


def avg_entropy(
    probabilities: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    eps: float = EPS,
) -> torch.Tensor:
    """CM-TTA binary entropy, averaged over valid 3-D voxels."""
    probabilities = probabilities.float()
    safe_eps = max(float(eps), float(torch.finfo(probabilities.dtype).eps))
    probabilities = probabilities.clamp(safe_eps, 1.0 - safe_eps)
    entropy = -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    )
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


# Compatibility for callers of the earlier local name.
binary_entropy = avg_entropy


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
    entropy = -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    )
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
    """Compute CM-TTA's soft foreground/background CAC in 3-D.

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
        logits,
        valid_mask=valid_mask,
        feature_spatial_shape=feature_spatial_shape,
    )
    text = text_features[0].float()
    if text.shape[0] != components["foreground_sum"].shape[0]:
        raise ValueError("Projected text and visual feature batch dimensions must agree")
    return cac_from_components(
        components["foreground_sum"],
        components["foreground_mass"],
        components["background_sum"],
        components["background_mass"],
        text,
    )


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
    logits: torch.Tensor,
    valid_mask: Optional[torch.Tensor] = None,
    feature_spatial_shape: Optional[tuple[int, int, int]] = None,
) -> dict[str, torch.Tensor]:
    """Accumulate unnormalized visual evidence for Eq. (1).

    No voxel-wise cosine is computed here.  The returned sums can be added
    across patches and normalized/cosined exactly once at case level.
    """
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
    foreground_probability = probability * mask
    background_probability = (1.0 - probability) * mask
    return {
        "foreground_sum": (vision * foreground_probability.unsqueeze(1)).sum(dim=(2, 3, 4)),
        "foreground_mass": foreground_probability.sum(dim=(1, 2, 3)),
        "background_sum": (vision * background_probability.unsqueeze(1)).sum(dim=(2, 3, 4)),
        "background_mass": background_probability.sum(dim=(1, 2, 3)),
    }


def cac_from_components(
    foreground_sum: torch.Tensor,
    foreground_mass: torch.Tensor,
    background_sum: torch.Tensor,
    background_mass: torch.Tensor,
    text_features: torch.Tensor,
) -> torch.Tensor:
    """Finish CAC after sums from all valid voxels have been accumulated."""
    if text_features.ndim != 2:
        raise ValueError(f"Expected text features (B,C), got {tuple(text_features.shape)}")
    if foreground_sum.shape != background_sum.shape or foreground_sum.shape != text_features.shape:
        raise ValueError("CAC feature sums and text features must have matching (B,C) shapes")
    foreground = foreground_sum / (foreground_mass.unsqueeze(1) + EPS)
    background = background_sum / (background_mass.unsqueeze(1) + EPS)
    text = F.normalize(text_features.float(), dim=1)
    foreground = F.normalize(foreground.float(), dim=1)
    background = F.normalize(background.float(), dim=1)
    return (foreground * text).sum(dim=1) - (background * text).sum(dim=1)


def select_cac_view(
    cac_scores: torch.Tensor,
    probabilities: torch.Tensor,
    selection_p: float,
) -> tuple[int, torch.Tensor]:
    """Select the highest-CAC view, as specified by CM-TTA Eq. (2)."""
    if cac_scores.ndim != 1 or probabilities.ndim < 2:
        raise ValueError("Expected one CAC score and one probability map per view")
    if cac_scores.shape[0] != probabilities.shape[0]:
        raise ValueError("CAC scores and probabilities must agree in view count")
    if not 0.0 < float(selection_p) <= 1.0:
        raise ValueError("selection_p must be in (0, 1]")
    del probabilities
    del selection_p  # retained for CLI/API compatibility; CAC selects one view
    selected = cac_scores.argmax().reshape(1)
    return int(selected[0].item()), selected


class ShortPromptMemory:
    """FIFO memory M_i containing recent short prompts and their CAC scores."""

    def __init__(self, max_length: int):
        self.max_length = int(max_length)
        if self.max_length < 1:
            raise ValueError("short memory length must be positive")
        self.prompts: deque[torch.Tensor] = deque(maxlen=self.max_length)
        self.cacs: deque[float] = deque(maxlen=self.max_length)

    def __len__(self) -> int:
        return len(self.prompts)

    def weighted_prompt(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.prompts:
            raise RuntimeError("Cannot fuse an empty short prompt memory")
        scores = torch.tensor(list(self.cacs), device=device, dtype=torch.float32)
        weights = torch.softmax(scores, dim=0)
        result = torch.zeros_like(self.prompts[0], device=device, dtype=dtype)
        for weight, prompt in zip(weights, self.prompts):
            result = result + weight.to(dtype) * prompt.to(device=device, dtype=dtype)
        return result

    def append(self, prompt: torch.Tensor, cac: float) -> None:
        self.prompts.append(prompt.detach().cpu().clone())
        self.cacs.append(float(cac))

    def state_dict(self) -> dict:
        return {
            "max_length": self.max_length,
            "prompts": [prompt.clone() for prompt in self.prompts],
            "cacs": list(self.cacs),
        }

    def load_state_dict(self, state: dict) -> None:
        self.max_length = int(state["max_length"])
        if self.max_length < 1:
            raise ValueError("short memory length must be positive")
        prompts = state.get("prompts", [])
        cacs = state.get("cacs", [])
        if len(prompts) != len(cacs):
            raise ValueError("short memory prompts and CAC scores must have equal lengths")
        self.prompts = deque(maxlen=self.max_length)
        self.cacs = deque(maxlen=self.max_length)
        for prompt, cac in zip(prompts, cacs):
            self.append(prompt, float(cac))


class VoxTellCMTTA:
    """CM-TTA/LSPM/DSPU with one trainable FP32 soft prompt."""

    def __init__(
        self,
        model: nn.Module,
        initial_prompt: torch.Tensor,
        device,
        args,
        qwen_text_encoder: Optional[nn.Module] = None,
    ):
        self.device = torch.device(device)
        self.model = model.to(self.device).eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)
        if any(parameter.requires_grad for parameter in self.model.parameters()):
            raise RuntimeError("VoxTell model must be completely frozen")
        self.qwen_text_encoder = qwen_text_encoder
        if self.qwen_text_encoder is not None:
            self.qwen_text_encoder.eval()
            for parameter in self.qwen_text_encoder.parameters():
                parameter.requires_grad_(False)
            if any(parameter.requires_grad for parameter in self.qwen_text_encoder.parameters()):
                raise RuntimeError("Qwen text encoder must be completely frozen")

        if initial_prompt.ndim == 2:
            initial_prompt = initial_prompt.unsqueeze(1)
        if tuple(initial_prompt.shape[:2]) != (1, 1):
            raise ValueError(
                "Expected one prompt with shape (1,1,D) or (1,D), "
                f"got {tuple(initial_prompt.shape)}"
            )
        initial_prompt = initial_prompt.detach().to(self.device, dtype=torch.float32)
        self.soft_prompt = nn.Parameter(initial_prompt.clone())
        self.initial_prompt = initial_prompt.clone()
        self.long_prompt: Optional[torch.Tensor] = None
        self.short_prompt: Optional[torch.Tensor] = None

        self.args = args
        self.lr = float(args.lr)
        self.ema_momentum = float(args.ema_momentum)
        self.w_cac = float(args.w_cac)
        self.w_entropy = float(args.w_entropy)
        self.num_aug_views = int(args.num_aug_views)  # K; total views are K+1.
        self.selection_p = float(args.selection_p)
        if self.num_aug_views < 1:
            raise ValueError("num_aug_views must be at least 1")
        if not 0.0 < self.selection_p <= 1.0:
            raise ValueError("selection_p must be in (0, 1]")

        self.optimizer = torch.optim.Adam([self.soft_prompt], lr=self.lr)
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

    def _capture(self, name):
        def hook(_module, _inputs, output):
            if name == "vision":
                self._vision_features = output
            else:
                self._text_features = output

        return hook

    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    @property
    def optimizer_parameters(self):
        return [parameter for group in self.optimizer.param_groups for parameter in group["params"]]

    def _text_input(self, prompt: torch.Tensor, batch_size: int) -> torch.Tensor:
        return prompt.expand(batch_size, -1, -1).unsqueeze(2)

    def _forward(self, images: torch.Tensor, prompt: torch.Tensor) -> torch.Tensor:
        logits = self.model(images, self._text_input(prompt, images.shape[0]))
        if isinstance(logits, (list, tuple)):
            logits = logits[0]
        if logits.ndim != 5:
            raise ValueError(f"VoxTell must return (B,N,D,H,W) logits, got {logits.shape}")
        return logits

    def _cac_components(
        self, logits: torch.Tensor, valid_mask: Optional[torch.Tensor] = None
    ) -> dict[str, torch.Tensor]:
        if self._vision_features is None or self._text_features is None:
            raise RuntimeError("VoxTell CAC feature hooks did not capture a forward pass")
        return cac_components_from_features(
            self._vision_features,
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
        prompt: torch.Tensor,
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
        text_sum = None
        with torch.no_grad():
            for patch, valid_mask in zip(patches, valid_masks):
                patch = patch.unsqueeze(0).to(self.device, non_blocking=True)
                if valid_mask is None:
                    valid_mask = torch.ones((1, *patch.shape[-3:]), device=self.device)
                else:
                    valid_mask = valid_mask.unsqueeze(0).to(self.device, non_blocking=True)
                logits = self._forward(patch, prompt)
                components = self._cac_components(logits, valid_mask)
                accumulator = self._add_components(accumulator, components)
                text = self._text_features[0].float()
                text_sum = text if text_sum is None else text_sum + text
        if accumulator is None:
            raise ValueError("A complete case must contain at least one patch")
        text = text_sum / len(patches)
        score = cac_from_components(
            accumulator["foreground_sum"],
            accumulator["foreground_mass"],
            accumulator["background_sum"],
            accumulator["background_mass"],
            text,
        )
        return float(score[0].detach().cpu())

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

    def _dynamic_short_prompt(
        self,
        patches: list[torch.Tensor],
        valid_masks: Optional[list[torch.Tensor]] = None,
    ) -> tuple[torch.Tensor, float, float, float]:
        current = self.soft_prompt
        current_cac = self._case_cac(current, patches, valid_masks)
        if len(self.short_memory) == 0:
            short = current
            return short, current_cac, current_cac, 0.0

        historical = self.short_memory.weighted_prompt(self.device, current.dtype)
        historical_cac = self._case_cac(historical, patches, valid_masks)
        weights = torch.softmax(
            torch.tensor([historical_cac, current_cac], device=self.device), dim=0
        )
        weight_historical = float(weights[0].detach().cpu())
        short = weight_historical * historical + (1.0 - weight_historical) * current
        return short, current_cac, historical_cac, weight_historical

    def _select_case_view(
        self,
        patches: list[torch.Tensor],
        params: list[dict[str, float]],
        short: torch.Tensor,
        valid_masks: Optional[list[torch.Tensor]] = None,
    ) -> tuple[int, torch.Tensor]:
        if valid_masks is None:
            valid_masks = [None] * len(patches)
        if len(valid_masks) != len(patches):
            raise ValueError("patches and valid_masks must have equal lengths")
        accumulator = None
        text_sum = None
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
                    logits = self._forward(view_batch, short.detach())
                    components = self._cac_components(logits, input_mask_batch)
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
                    for key, value in components.items():
                        accumulator[key][start:end].add_(value)
                    text = self._text_features[0].float()
                    if text_sum is None:
                        text_sum = torch.zeros(
                            (total_views, text.shape[1]),
                            device=text.device,
                            dtype=text.dtype,
                        )
                    text_sum[start:end] += text
        if accumulator is None:
            raise ValueError("A complete case must contain at least one patch")
        scores = cac_from_components(
            accumulator["foreground_sum"],
            accumulator["foreground_mass"],
            accumulator["background_sum"],
            accumulator["background_mass"],
            text_sum,
        ).detach()
        selected, _ = select_cac_view(
            scores, torch.ones((scores.shape[0], 1), device=scores.device), self.selection_p
        )
        return selected, scores

    def _forward_case_supervision_stats(
        self,
        patches: list[torch.Tensor],
        valid_masks: list[torch.Tensor],
        params: list[dict[str, float]],
        selected_view: int,
        short_value: torch.Tensor,
        short_current_weight: float,
        long_prompt: torch.Tensor,
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
                    pseudo_logits = self._forward(selected, long_prompt)
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
                        student_prompt = short_value + short_current_weight * (
                            self.soft_prompt - self.soft_prompt.detach()
                        )
                        student_logits = self._forward(view_batch, student_prompt)
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

    def _backward_case_supervision(
        self,
        patches: list[torch.Tensor],
        valid_masks: list[torch.Tensor],
        params: list[dict[str, float]],
        selected_view: int,
        short_value: torch.Tensor,
        short_current_weight: float,
        long_prompt: torch.Tensor,
        autocast_enabled: bool,
    ) -> tuple[float, float]:
        """Backpropagate case-level Dice and entropy with one patch graph."""
        stats = self._forward_case_supervision_stats(
            patches,
            valid_masks,
            params,
            selected_view,
            short_value,
            short_current_weight,
            long_prompt,
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

        scale = float(self.scaler.get_scale())
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
                pseudo_logits = self._forward(selected, long_prompt)
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
                    student_prompt = short_value + short_current_weight * (
                        self.soft_prompt - self.soft_prompt.detach()
                    )
                    student_logits = self._forward(view_batch, student_prompt)
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
                        dice_derivatives[0][start:end] * scale,
                        dice_derivatives[1][start:end] * scale,
                    ]
                    if start <= selected_view < end:
                        selected_index = selected_view - start
                        local_entropy_sum, local_entropy_mass = masked_entropy_components(
                            probabilities[selected_index:selected_index + 1],
                            input_mask_batch[selected_index:selected_index + 1],
                        )
                        local_tensors.append(local_entropy_sum)
                        local_gradients.append(entropy_derivatives[0] * scale)
                differentiable = [
                    (tensor, gradient)
                    for tensor, gradient in zip(local_tensors, local_gradients)
                    if tensor.requires_grad
                ]
                if differentiable:
                    local_objective = sum(
                        (tensor * gradient).sum() for tensor, gradient in differentiable
                    )
                    local_objective.backward()
        return float(global_dice.detach().cpu()), float(global_entropy.detach().cpu())

    def _backward_case_cac(
        self,
        patches: list[torch.Tensor],
        valid_masks: list[torch.Tensor],
        params: list[dict[str, float]],
        selected_view: int,
        short_value: torch.Tensor,
        short_current_weight: float,
        autocast_enabled: bool,
    ) -> float:
        """Backpropagate one exact global selected-view CAC with low memory."""
        # First collect the exact case-level CAC inputs without autograd.  A
        # later pass applies the global CAC derivative patch by patch, so no
        # collection of patch graphs is needed.
        case_components = None
        case_text_sum = None
        with torch.no_grad():
            for patch, valid_mask in zip(patches, valid_masks):
                selected = self._make_view_batch(
                    patch, params, valid_mask, selected_view, selected_view + 1
                ).to(self.device, non_blocking=True)
                input_mask = valid_mask.unsqueeze(0).to(
                    self.device, non_blocking=True
                )
                with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                    student_prompt = short_value + short_current_weight * (
                        self.soft_prompt - self.soft_prompt.detach()
                    )
                    selected_logits = self._forward(selected, student_prompt)
                    components = self._cac_components(selected_logits, input_mask)
                    case_components = self._add_components(case_components, components)
                    text = self._text_features[0].float()
                    case_text_sum = text if case_text_sum is None else case_text_sum + text
        if case_components is None or case_text_sum is None:
            raise RuntimeError("Selected-view case CAC accumulation produced no statistics")

        # Differentiate the one global pooled-CAC expression with respect to
        # its five aggregate inputs.  Each derivative is then supplied to a
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
        global_text = (case_text_sum / len(patches)).detach().requires_grad_(True)
        case_cac_graph = cac_from_components(*global_inputs, global_text)
        global_derivatives = torch.autograd.grad(
            case_cac_graph[0], (*global_inputs, global_text)
        )
        cac_loss = -case_cac_graph[0].detach()

        # Backpropagate the derivative of w_cac * (-case_cac), one selected
        # view/patch at a time.  The current graph is released every iteration.
        scale = -float(self.scaler.get_scale()) * self.w_cac
        for patch, valid_mask in zip(patches, valid_masks):
            selected = self._make_view_batch(
                patch, params, valid_mask, selected_view, selected_view + 1
            ).to(self.device, non_blocking=True)
            input_mask = valid_mask.unsqueeze(0).to(
                self.device, non_blocking=True
            )
            with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                student_prompt = short_value + short_current_weight * (
                    self.soft_prompt - self.soft_prompt.detach()
                )
                selected_logits = self._forward(selected, student_prompt)
                components = self._cac_components(selected_logits, input_mask)
                local_text = self._text_features[0].float()
            local_inputs = (
                components["foreground_sum"],
                components["foreground_mass"],
                components["background_sum"],
                components["background_mass"],
                local_text,
            )
            local_tensors = []
            local_gradients = []
            for local, derivative in zip(local_inputs, global_derivatives):
                if local.requires_grad:
                    local_tensors.append(local)
                    text_factor = 1.0 / len(patches) if local is local_text else 1.0
                    local_gradients.append(derivative * (scale * text_factor))
            if local_tensors:
                local_objective = sum(
                    (tensor * gradient).sum()
                    for tensor, gradient in zip(local_tensors, local_gradients)
                )
                local_objective.backward()
        return float(cac_loss.cpu())

    def adapt_case(
        self,
        patches: list[torch.Tensor],
        valid_masks: Optional[list[torch.Tensor]] = None,
    ) -> dict:
        """Adapt once on one complete case, aggregating all patch gradients."""
        if not patches:
            raise ValueError("adapt_case received no patches")
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
        short, current_cac, historical_cac, weight_historical = self._dynamic_short_prompt(
            patches, valid_masks
        )

        if self.long_prompt is None:
            long = short.detach().clone()
        else:
            long = self.ema_momentum * self.long_prompt + (1.0 - self.ema_momentum) * short.detach()
        long = long.detach()

        selected_view, selection_scores = self._select_case_view(
            patches, params, short, valid_masks
        )
        short_snapshot = short.detach().clone()
        short_value = short_snapshot
        short_current_weight = 1.0 - weight_historical
        self.optimizer.zero_grad(set_to_none=True)
        sums = {"soft_dice": 0.0, "cac_loss": 0.0, "entropy_loss": 0.0, "loss": 0.0}
        autocast_enabled = self.device.type == "cuda"
        soft_dice, entropy_loss = self._backward_case_supervision(
            patches,
            valid_masks,
            params,
            selected_view,
            short_value,
            short_current_weight,
            long,
            autocast_enabled,
        )
        sums["soft_dice"] = soft_dice
        sums["entropy_loss"] = entropy_loss
        sums["loss"] = soft_dice + self.w_entropy * entropy_loss

        cac_loss = self._backward_case_cac(
            patches,
            valid_masks,
            params,
            selected_view,
            short_value,
            short_current_weight,
            autocast_enabled,
        )
        sums["cac_loss"] = float(cac_loss)
        sums["loss"] += self.w_cac * sums["cac_loss"]

        self.scaler.unscale_(self.optimizer)
        torch.nn.utils.clip_grad_norm_([self.soft_prompt], float(self.args.grad_clip))
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer_step_count += 1

        self.short_prompt = short_snapshot
        self.long_prompt = long
        selected_cac = float(selection_scores[selected_view].detach().cpu())
        # Store the actual short prompt used for this case, paired with its
        # selected-view CAC.  The deque itself supplies FIFO eviction.
        self.short_memory.append(self.short_prompt, selected_cac)
        self.last_trace = {
            "selected_view": selected_view,
            "pseudo_source_view": selected_view,
            "num_views": 1 + self.num_aug_views,
            "num_patches": len(patches),
            "optimizer_steps_for_case": 1,
            "current_cac": current_cac,
            "historical_cac": historical_cac,
            "historical_weight": weight_historical,
            "selected_cac": selected_cac,
            **sums,
        }
        return dict(self.last_trace)

    def state_dict(self) -> dict:
        return {
            "soft_prompt": self.soft_prompt.detach().cpu(),
            "initial_prompt": self.initial_prompt.detach().cpu(),
            "short_prompt": None if self.short_prompt is None else self.short_prompt.cpu(),
            "long_prompt": None if self.long_prompt is None else self.long_prompt.cpu(),
            "short_memory": self.short_memory.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "optimizer_step_count": self.optimizer_step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        self.soft_prompt.data.copy_(state["soft_prompt"].to(self.device))
        self.initial_prompt.copy_(state.get("initial_prompt", state["soft_prompt"]).to(self.device))
        self.short_prompt = (
            None if state.get("short_prompt") is None else state["short_prompt"].to(self.device)
        )
        self.long_prompt = (
            None if state.get("long_prompt") is None else state["long_prompt"].to(self.device)
        )
        self.short_memory.load_state_dict(state["short_memory"])
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
            "format": "voxtell-cmtta-lspm-dspu-v1",
            "adapter": adapter.state_dict(),
            "args": vars(args),
            "history": history,
        },
        path,
    )


def load_cmtta_checkpoint(path: str, adapter: VoxTellCMTTA) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "voxtell-cmtta-lspm-dspu-v1":
        raise ValueError(f"Unsupported CM-TTA checkpoint format: {checkpoint.get('format')}")
    adapter.load_state_dict(checkpoint["adapter"])
    return checkpoint
