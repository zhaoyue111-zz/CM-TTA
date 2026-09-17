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
    """FIFO memory M_i containing recent short ctx tensors and CAC scores."""

    def __init__(self, max_length: int):
        self.max_length = int(max_length)
        if self.max_length < 1:
            raise ValueError("short memory length must be positive")
        self.contexts: deque[torch.Tensor] = deque(maxlen=self.max_length)
        self.cacs: deque[float] = deque(maxlen=self.max_length)

    def __len__(self) -> int:
        return len(self.contexts)

    def weighted_ctx(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if not self.contexts:
            raise RuntimeError("Cannot fuse an empty short ctx memory")
        scores = torch.tensor(list(self.cacs), device=device, dtype=torch.float32)
        weights = torch.softmax(scores, dim=0)
        result = torch.zeros_like(self.contexts[0], device=device, dtype=dtype)
        for weight, ctx in zip(weights, self.contexts):
            result = result + weight.to(dtype) * ctx.to(device=device, dtype=dtype)
        return result

    def append(self, ctx: torch.Tensor, cac: float) -> None:
        self.contexts.append(ctx.detach().cpu().clone())
        self.cacs.append(float(cac))

    def state_dict(self) -> dict:
        return {
            "max_length": self.max_length,
            "ctxs": [ctx.clone() for ctx in self.contexts],
            "cacs": list(self.cacs),
        }

    def load_state_dict(self, state: dict) -> None:
        self.max_length = int(state["max_length"])
        if self.max_length < 1:
            raise ValueError("short memory length must be positive")
        contexts = state.get("ctxs", state.get("prompts", []))
        cacs = state.get("cacs", [])
        if len(contexts) != len(cacs):
            raise ValueError("short memory ctxs and CAC scores must have equal lengths")
        self.contexts = deque(maxlen=self.max_length)
        self.cacs = deque(maxlen=self.max_length)
        for ctx, cac in zip(contexts, cacs):
            self.append(ctx, float(cac))


class VoxTellCMTTA:
    """CM-TTA/LSPM/DSPU with one trainable FP32 text context."""

    def __init__(
        self,
        model: nn.Module,
        initial_ctx: Optional[torch.Tensor],
        device,
        args,
        qwen_text_encoder: Optional[nn.Module] = None,
        qwen_tokenizer=None,
        text_prompt: str = "liver",
        n_ctx: int = 1,
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
        self.n_ctx = int(n_ctx)
        if self.n_ctx < 1:
            raise ValueError("n_ctx must be positive")
        self._fixed_token_embeddings = None
        self._fixed_attention_mask = None
        self._ctx_insert_index = None

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
            self.ctx = nn.Parameter(initial_ctx.clone())
            self.initial_ctx = initial_ctx.clone()
        self.long_ctx: Optional[torch.Tensor] = None
        self.short_ctx: Optional[torch.Tensor] = None

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

        self.optimizer = torch.optim.Adam([self.ctx], lr=self.lr)
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
        self._print_trainable_parameters()

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
        """Build fixed Qwen inputs and a CoOp-style context parameter."""
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
        offsets = tokenized.get("offset_mapping")
        insert_index = None
        if offsets is not None and query_start >= 0:
            for index, (start, end) in enumerate(torch.as_tensor(offsets)[0].tolist()):
                if end > start and end > query_start:
                    insert_index = index
                    break
        if insert_index is None:
            active_indices = torch.nonzero(attention_mask[0], as_tuple=False).flatten().tolist()
            query_tokens = tokenizer(
                [self.text_prompt], add_special_tokens=False, return_tensors="pt"
            )["input_ids"][0].tolist()
            active_ids = input_ids[0, active_indices].tolist()
            for start in range(max(1, len(active_ids) - len(query_tokens) + 1)):
                if active_ids[start:start + len(query_tokens)] == query_tokens:
                    insert_index = active_indices[start]
                    break
            if insert_index is None:
                if len(active_indices) < 2:
                    raise ValueError("Could not locate the fixed liver token in Qwen input")
                insert_index = active_indices[-2]

        embedding_dim = int(token_embedding.embedding_dim)
        if initial_ctx is None:
            ctx = torch.empty(
                (self.n_ctx, embedding_dim),
                device=self.device,
                dtype=torch.float32,
            )
            nn.init.normal_(ctx, std=0.02)
        else:
            ctx = initial_ctx.detach()
            if ctx.ndim == 3 and ctx.shape[0] == 1:
                ctx = ctx[0]
            elif ctx.ndim != 2:
                raise ValueError(
                    "Expected initial ctx with shape (n_ctx,D) or (1,n_ctx,D), "
                    f"got {tuple(ctx.shape)}"
                )
            if tuple(ctx.shape) != (self.n_ctx, embedding_dim):
                raise ValueError(
                    "Expected initial ctx with shape "
                    f"({self.n_ctx},{embedding_dim}), got {tuple(ctx.shape)}"
                )
            ctx = ctx.to(self.device, dtype=torch.float32)
        self.ctx = nn.Parameter(ctx.clone())
        self.initial_ctx = ctx.clone()
        self._ctx_insert_index = int(insert_index)

    def _encode_ctx(self, ctx: torch.Tensor) -> torch.Tensor:
        """Encode ctx + fixed wrapped ``liver`` tokens into VoxTell text features."""
        if self.qwen_text_encoder is None:
            return ctx.unsqueeze(0) if ctx.ndim == 2 else ctx
        if ctx.ndim == 2:
            ctx = ctx.unsqueeze(0)
        if tuple(ctx.shape[:2]) != (1, self.n_ctx):
            raise ValueError(
                f"Expected ctx shape (1,{self.n_ctx},D), got {tuple(ctx.shape)}"
            )
        fixed = self._fixed_token_embeddings
        fixed_mask = self._fixed_attention_mask
        index = self._ctx_insert_index
        if fixed is None or fixed_mask is None or index is None:
            raise RuntimeError("Qwen context inputs were not initialized")
        ctx = ctx.to(device=fixed.device, dtype=fixed.dtype)
        context_mask = torch.ones(
            (1, self.n_ctx), device=fixed_mask.device, dtype=fixed_mask.dtype
        )
        inputs_embeds = torch.cat(
            [fixed[:, :index], ctx, fixed[:, index:]], dim=1
        )
        attention_mask = torch.cat(
            [fixed_mask[:, :index], context_mask, fixed_mask[:, index:]], dim=1
        )
        outputs = self.qwen_text_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )
        hidden_states = outputs.last_hidden_state if hasattr(outputs, "last_hidden_state") else outputs[0]
        last_index = attention_mask.shape[1] - 1 - torch.flip(
            attention_mask, dims=(1,)
        ).float().argmax(dim=1)
        pooled = hidden_states[torch.arange(hidden_states.shape[0], device=hidden_states.device), last_index]
        return pooled.unsqueeze(1)

    def close(self) -> None:
        for handle in self._hooks:
            handle.remove()
        self._hooks.clear()

    @property
    def optimizer_parameters(self):
        return [parameter for group in self.optimizer.param_groups for parameter in group["params"]]

    def _print_trainable_parameters(self) -> None:
        trainable = [("ctx", self.ctx)]
        unexpected = [
            name
            for name, parameter in trainable
            if not parameter.requires_grad
        ]
        optimizer_parameters = self.optimizer_parameters
        if unexpected or len(optimizer_parameters) != 1 or optimizer_parameters[0] is not self.ctx:
            raise RuntimeError("Only ctx may be trainable and optimized")
        print(
            "[VoxTell-CM-TTA] trainable parameters: "
            f"ctx shape={tuple(self.ctx.shape)}, dtype={self.ctx.dtype}, "
            f"numel={self.ctx.numel()}"
        )

    def _check_case_gradients(self, allow_nonfinite: bool = False) -> None:
        if self.ctx.grad is None:
            raise RuntimeError("ctx did not receive a gradient during case adaptation")
        if self.ctx.grad.norm() == 0:
            raise RuntimeError("ctx gradient is zero during case adaptation")
        if not allow_nonfinite and not torch.isfinite(self.ctx.grad).all():
            raise RuntimeError("ctx gradient is zero or non-finite during case adaptation")
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
        current = self.ctx
        current_cac = self._case_cac(current, patches, valid_masks)
        if len(self.short_memory) == 0:
            short = current
            return short, current_cac, current_cac, 0.0

        historical = self.short_memory.weighted_ctx(self.device, current.dtype)
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
        short_ctx: torch.Tensor,
        valid_masks: Optional[list[torch.Tensor]] = None,
    ) -> tuple[int, torch.Tensor]:
        if valid_masks is None:
            valid_masks = [None] * len(patches)
        if len(valid_masks) != len(patches):
            raise ValueError("patches and valid_masks must have equal lengths")
        accumulator = None
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
                    logits = self._forward(view_batch, short_ctx.detach())
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
                            self.ctx - self.ctx.detach()
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
    ) -> tuple[float, float]:
        """Backpropagate case-level Dice and entropy with one patch graph."""
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
                        self.ctx - self.ctx.detach()
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
                        self.ctx - self.ctx.detach()
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
                    self.ctx - self.ctx.detach()
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
        short_ctx, current_cac, historical_cac, weight_historical = self._dynamic_short_ctx(
            patches, valid_masks
        )

        if self.long_ctx is None:
            long_ctx = short_ctx.detach().clone()
        else:
            long_ctx = self.ema_momentum * self.long_ctx + (1.0 - self.ema_momentum) * short_ctx.detach()
        long_ctx = long_ctx.detach()

        selected_view, selection_scores = self._select_case_view(
            patches, params, short_ctx, valid_masks
        )
        short_snapshot = short_ctx.detach().clone()
        short_ctx_value = short_snapshot
        short_current_weight = 1.0 - weight_historical
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
        )
        sums["soft_dice"] = soft_dice
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

        # Check that backward produced a real ctx gradient before handing it
        # to GradScaler.  Non-finite values are allowed to reach the scaler:
        # an AMP overflow is a recoverable skipped step, not a protocol error.
        self._check_case_gradients(allow_nonfinite=True)
        scale_before_step = float(self.scaler.get_scale())
        self.scaler.unscale_(self.optimizer)
        if torch.isfinite(self.ctx.grad).all():
            self._check_case_gradients()
        else:
            warnings.warn(
                "GradScaler detected a non-finite ctx gradient; this case's "
                "optimizer step may be skipped",
                RuntimeWarning,
                stacklevel=2,
            )
        torch.nn.utils.clip_grad_norm_([self.ctx], float(self.args.grad_clip))
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

        self.short_ctx = short_snapshot
        self.long_ctx = long_ctx
        selected_cac = float(selection_scores[selected_view].detach().cpu())
        # Store the actual short prompt used for this case, paired with its
        # selected-view CAC.  The deque itself supplies FIFO eviction.
        self.short_memory.append(self.short_ctx, selected_cac)
        self.last_trace = {
            "selected_view": selected_view,
            "pseudo_source_view": selected_view,
            "num_views": 1 + self.num_aug_views,
            "num_patches": len(patches),
            "optimizer_steps_for_case": 1,
            "optimizer_step_skipped": optimizer_step_skipped,
            "current_cac": current_cac,
            "historical_cac": historical_cac,
            "historical_weight": weight_historical,
            "selected_cac": selected_cac,
            **sums,
        }
        return dict(self.last_trace)

    def state_dict(self) -> dict:
        return {
            "ctx": self.ctx.detach().cpu(),
            "initial_ctx": self.initial_ctx.detach().cpu(),
            "short_ctx": None if self.short_ctx is None else self.short_ctx.cpu(),
            "long_ctx": None if self.long_ctx is None else self.long_ctx.cpu(),
            "ctx_memory": self.short_memory.state_dict(),
            "text_prompt": self.text_prompt,
            "n_ctx": self.n_ctx,
            "optimizer": self.optimizer.state_dict(),
            "scaler": self.scaler.state_dict(),
            "optimizer_step_count": self.optimizer_step_count,
        }

    def load_state_dict(self, state: dict) -> None:
        if "ctx" not in state:
            raise ValueError(
                "Checkpoint does not contain ctx; final text-embedding checkpoints "
                "cannot be loaded by the ctx-based adapter"
            )
        self.ctx.data.copy_(state["ctx"].to(self.device))
        self.initial_ctx.copy_(state.get("initial_ctx", state["ctx"]).to(self.device))
        self.short_ctx = (
            None if state.get("short_ctx") is None else state["short_ctx"].to(self.device)
        )
        self.long_ctx = (
            None if state.get("long_ctx") is None else state["long_ctx"].to(self.device)
        )
        self.short_memory.load_state_dict(state["ctx_memory"])
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
            "format": "voxtell-cmtta-lspm-dspu-ctx-v1",
            "adapter": adapter.state_dict(),
            "args": vars(args),
            "history": history,
        },
        path,
    )


def load_cmtta_checkpoint(path: str, adapter: VoxTellCMTTA) -> dict:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "voxtell-cmtta-lspm-dspu-ctx-v1":
        raise ValueError(f"Unsupported CM-TTA checkpoint format: {checkpoint.get('format')}")
    adapter.load_state_dict(checkpoint["adapter"])
    return checkpoint
