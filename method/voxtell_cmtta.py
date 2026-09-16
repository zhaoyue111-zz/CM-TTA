"""Original CM-TTA adapted to VoxTell's frozen 3-D network.

The only model-specific changes are the VoxTell text-embedding interface and
the extension of CAC, entropy, and soft Dice from 2-D to 3-D tensors. LSPM and
DSPU follow CM-TTA equations (3)--(8): one optimizer update is made per full
case, after losses from all of that case's patches have been accumulated.
"""

from __future__ import annotations

from collections import deque
from typing import Iterable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


EPS = 1e-8


def avg_entropy(probabilities: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """CM-TTA's binary pixel/voxel entropy, flattened over 3-D space."""
    probabilities = probabilities.float().clamp(eps, 1.0 - eps)
    entropy = -(
        probabilities * probabilities.log()
        + (1.0 - probabilities) * (1.0 - probabilities).log()
    )
    return entropy.flatten(start_dim=1).mean(dim=1).sum()


# Compatibility for callers of the earlier local name.
binary_entropy = avg_entropy


def soft_dice_loss(predictions: torch.Tensor, pseudo_label: torch.Tensor) -> torch.Tensor:
    """DSPU soft Dice over every view, with a detached soft pseudo-label."""
    if predictions.ndim != 5 or pseudo_label.ndim != 5:
        raise ValueError(
            "Expected predictions (V,1,D,H,W) and pseudo_label (1,1,D,H,W), "
            f"got {tuple(predictions.shape)} and {tuple(pseudo_label.shape)}"
        )
    target = pseudo_label.expand(predictions.shape[0], *pseudo_label.shape[1:])
    pred_flat = predictions.float().flatten(start_dim=1)
    target_flat = target.float().flatten(start_dim=1)
    numerator = 2.0 * (pred_flat * target_flat).sum(dim=1)
    denominator = pred_flat.sum(dim=1) + target_flat.sum(dim=1) + EPS
    return (1.0 - numerator / denominator).mean()


def cac_from_features(
    vision_features: torch.Tensor,
    text_features: torch.Tensor,
    logits: torch.Tensor,
) -> torch.Tensor:
    """Compute CM-TTA's soft foreground/background CAC in 3-D.

    The released VoxTell projection hook exposes visual tokens as ``(S,B,C)``
    and logits as ``(B,N,H,W,D)``.  The 5-D feature form is accepted for small
    test doubles and older checkpoints as well.
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
    if vision_features.ndim == 3:
        tokens, batch, channels = vision_features.shape
        if batch != logits.shape[0]:
            raise ValueError("Vision features and logits have different batch sizes")
        spatial_shape = tuple(int(size) for size in logits.shape[2:])
        if tokens != int(np.prod(spatial_shape)):
            raise ValueError(
                "Projected visual token count does not match logits spatial size: "
                f"{tokens} vs {spatial_shape}"
            )
        vision = vision_features.permute(1, 2, 0).reshape(batch, channels, *spatial_shape)
        probability = torch.sigmoid(logits[:, 0].float())
    else:
        if vision_features.shape[0] != logits.shape[0]:
            raise ValueError("Vision features and logits have different batch sizes")
        vision = vision_features.permute(0, 4, 1, 2, 3).float()
        probability = torch.sigmoid(logits[:, 0].float()).permute(0, 2, 3, 1)

    vision = F.normalize(vision, dim=1)
    text = F.normalize(text_features[0].float(), dim=1)
    if text.shape[0] != vision.shape[0] or text.shape[1] != vision.shape[1]:
        raise ValueError("Projected text and visual feature dimensions must agree")
    similarity = (vision * text[:, :, None, None, None]).sum(dim=1)

    if probability.shape[1:] != similarity.shape[1:]:
        probability = F.interpolate(
            probability.unsqueeze(1),
            size=similarity.shape[1:],
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)

    # Eq. (1) of CM-TTA uses the soft prediction itself as the foreground
    # evidence and (1 - P) as the background evidence.
    background_probability = 1.0 - probability
    foreground_mass = probability.sum(dim=(1, 2, 3)) + EPS
    background_mass = background_probability.sum(dim=(1, 2, 3)) + EPS
    foreground_similarity = (similarity * probability).sum(dim=(1, 2, 3)) / foreground_mass
    background_similarity = (similarity * background_probability).sum(dim=(1, 2, 3)) / background_mass
    return foreground_similarity - background_similarity


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
        self.weight_decay = float(args.weight_decay)
        self.ema_momentum = float(args.ema_momentum)
        self.w_cac = float(args.w_cac)
        self.w_entropy = float(args.w_entropy)
        self.num_aug_views = int(args.num_aug_views)  # K; total views are K+1.
        self.selection_p = float(args.selection_p)
        if self.num_aug_views < 1:
            raise ValueError("num_aug_views must be at least 1")
        if not 0.0 < self.selection_p <= 1.0:
            raise ValueError("selection_p must be in (0, 1]")

        self.optimizer = torch.optim.AdamW(
            [self.soft_prompt], lr=self.lr, weight_decay=self.weight_decay
        )
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

    def _cac(self, logits: torch.Tensor) -> torch.Tensor:
        if self._vision_features is None or self._text_features is None:
            raise RuntimeError("VoxTell CAC feature hooks did not capture a forward pass")
        return cac_from_features(self._vision_features, self._text_features, logits)

    def _case_cac(self, prompt: torch.Tensor, patches: Iterable[torch.Tensor]) -> float:
        scores = []
        with torch.no_grad():
            for patch in patches:
                patch = patch.unsqueeze(0).to(self.device, non_blocking=True)
                logits = self._forward(patch, prompt)
                scores.append(float(self._cac(logits)[0].detach().cpu()))
        if not scores:
            raise ValueError("A complete case must contain at least one patch")
        return float(np.mean(scores))

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
    def _make_views(patch: torch.Tensor, params: list[dict[str, float]]) -> torch.Tensor:
        views = [patch]
        for param in params[1:]:
            # Intensity-only transforms preserve the voxel correspondence used
            # by DSPU. No spatial transform is applied to a 3-D volume.
            views.append(patch * param["scale"] + param["offset"])
        return torch.stack(views, dim=0).contiguous()

    def _dynamic_short_prompt(self, patches: list[torch.Tensor]) -> tuple[torch.Tensor, float, float, float]:
        current = self.soft_prompt
        current_cac = self._case_cac(current, patches)
        if len(self.short_memory) == 0:
            short = current
            return short, current_cac, current_cac, 0.0

        historical = self.short_memory.weighted_prompt(self.device, current.dtype)
        historical_cac = self._case_cac(historical, patches)
        weights = torch.softmax(
            torch.tensor([historical_cac, current_cac], device=self.device), dim=0
        )
        weight_historical = float(weights[0].detach().cpu())
        short = weight_historical * historical + (1.0 - weight_historical) * current
        return short, current_cac, historical_cac, weight_historical

    def _select_case_view(
        self, patches: list[torch.Tensor], params: list[dict[str, float]], short: torch.Tensor
    ) -> tuple[int, torch.Tensor]:
        per_patch = []
        per_patch_probabilities = []
        with torch.no_grad():
            for patch in patches:
                views = self._make_views(patch, params).to(self.device, non_blocking=True)
                logits = self._forward(views, short.detach())
                per_patch.append(self._cac(logits).detach())
                per_patch_probabilities.append(torch.sigmoid(logits[:, :1]).squeeze(1).detach())
        scores = torch.stack(per_patch, dim=0).mean(dim=0)
        probabilities = torch.stack(per_patch_probabilities, dim=0).mean(dim=0)
        selected, _ = select_cac_view(scores, probabilities, self.selection_p)
        return selected, scores

    def adapt_case(self, patches: list[torch.Tensor]) -> dict:
        """Adapt once on one complete case, aggregating all patch gradients."""
        if not patches:
            raise ValueError("adapt_case received no patches")
        patches = [patch.float().contiguous() for patch in patches]
        params = self._sample_intensity_params(self.num_aug_views)
        short, current_cac, historical_cac, weight_historical = self._dynamic_short_prompt(patches)

        if self.long_prompt is None:
            long = short.detach().clone()
        else:
            long = self.ema_momentum * self.long_prompt + (1.0 - self.ema_momentum) * short.detach()
        long = long.detach()

        selected_view, selection_scores = self._select_case_view(patches, params, short)
        short_snapshot = short.detach().clone()
        self.optimizer.zero_grad(set_to_none=True)
        sums = {"soft_dice": 0.0, "cac_loss": 0.0, "entropy_loss": 0.0, "loss": 0.0}
        autocast_enabled = self.device.type == "cuda"
        for patch in patches:
            views = self._make_views(patch, params).to(self.device, non_blocking=True)
            selected = views[selected_view:selected_view + 1]
            with torch.no_grad(), torch.autocast(
                device_type=self.device.type, enabled=autocast_enabled
            ):
                long_logits = self._forward(selected, long)
                pseudo_label = torch.sigmoid(long_logits[:, :1]).detach()

            with torch.autocast(device_type=self.device.type, enabled=autocast_enabled):
                student_logits = self._forward(views, short)
                student_probabilities = torch.sigmoid(student_logits[:, :1])
                student_cac = self._cac(student_logits)
                soft_dice = soft_dice_loss(student_probabilities, pseudo_label)
                cac_loss = -student_cac[selected_view]
                entropy_loss = avg_entropy(student_probabilities[selected_view])
                loss = soft_dice + self.w_cac * cac_loss + self.w_entropy * entropy_loss

            # Gradient accumulation is one aggregated case loss; optimizer.step
            # remains outside this loop and executes exactly once per case.
            self.scaler.scale(loss / len(patches)).backward()
            for name, value in (
                ("soft_dice", soft_dice),
                ("cac_loss", cac_loss),
                ("entropy_loss", entropy_loss),
                ("loss", loss),
            ):
                sums[name] += float(value.detach().cpu()) / len(patches)

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
