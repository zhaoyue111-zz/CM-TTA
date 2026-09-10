"""VoxTell SFDA with one trainable soft prompt and CAC/TSE quality.

The Qwen text encoder is used only before adaptation to create the initial
embedding. During adaptation the only optimizer parameter is the free
``soft_prompt_embedding`` tensor; there is no Teacher VoxTell network.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn

try:
    from .semantic_quality import (
        QUALITY_MODES,
        SemanticPrototypeMemory,
        average_tie_ranks,
        compute_semantic_quality,
        extract_case_seed_statistics,
        load_quality_config,
        summarize_similarity_distribution,
    )
except ImportError:  # Compatibility with direct ``from sfda_voxtell import ...``.
    from semantic_quality import (  # type: ignore
        QUALITY_MODES,
        SemanticPrototypeMemory,
        average_tie_ranks,
        compute_semantic_quality,
        extract_case_seed_statistics,
        load_quality_config,
        summarize_similarity_distribution,
    )


def masked_segmentation_loss(logits, pseudo, valid):
    """CM-TTA confidence-masked pseudo-label BCE + Dice loss."""
    valid = valid.float()
    normalizer = valid.sum().clamp_min(1.0)
    bce = (
        F.binary_cross_entropy_with_logits(logits, pseudo, reduction="none") * valid
    ).sum() / normalizer
    prob = torch.sigmoid(logits.float())
    intersection = (prob * pseudo * valid).sum()
    dice = (2 * intersection + 1) / (
        (prob * valid).sum() + (pseudo * valid).sum() + 1
    )
    return bce + 1 - dice, bce.detach(), dice.detach()


def entropy_loss(logits, valid):
    """CM-TTA binary entropy, restricted to confident voxels."""
    prob = torch.sigmoid(logits.float()).clamp(1e-6, 1 - 1e-6)
    entropy = -(prob * prob.log() + (1 - prob) * (1 - prob).log())
    return (entropy * valid).sum() / valid.sum().clamp_min(1.0)


def _binary_view_entropy(prob):
    prob = prob.float().clamp(1e-6, 1 - 1e-6)
    entropy = -(prob * prob.log() + (1 - prob) * (1 - prob).log())
    return entropy.flatten(start_dim=1).mean(dim=1)


def compute_cac_score(vision_features, text_features, logits, fg_threshold=0.5):
    """Compute CM-TTA foreground/background contrast for VoxTell 3-D tensors.

    The CM-TTA formula is unchanged: mean foreground cosine similarity minus
    mean background cosine similarity. VoxTell's only shape adaptation is
    converting logits ``(B,N,D,H,W)`` to ``(B,H,W,D)`` to match projected
    bottleneck features ``(B,H,W,D,C)``.
    """
    if vision_features.ndim != 5:
        raise ValueError(
            "Expected 3-D bottleneck features (B,H,W,D,C), "
            f"got {tuple(vision_features.shape)}"
        )
    if text_features.ndim != 3:
        raise ValueError(
            "Expected projected text features (N,B,C), "
            f"got {tuple(text_features.shape)}"
        )
    if logits.ndim != 5:
        raise ValueError(
            "Expected VoxTell logits (B,N,D,H,W), got "
            f"{tuple(logits.shape)}"
        )
    if text_features.shape[1] != vision_features.shape[0]:
        raise ValueError("Projected text batch and bottleneck batch must agree")
    if text_features.shape[2] != vision_features.shape[4]:
        raise ValueError("Projected text and bottleneck channel dimensions must agree")

    vision = F.normalize(vision_features.permute(0, 4, 1, 2, 3).float(), dim=1)
    text = F.normalize(text_features[0].float(), dim=1)
    similarity = (vision * text[:, :, None, None, None]).sum(dim=1)

    # Necessary VoxTell axis adaptation only: D,H,W -> H,W,D.
    seg_prob = torch.sigmoid(logits[:, 0].float()).permute(0, 2, 3, 1)
    if seg_prob.shape[1:] != similarity.shape[1:]:
        seg_prob = F.interpolate(
            seg_prob.unsqueeze(1),
            size=similarity.shape[1:],
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)
    fg_mask = (seg_prob > fg_threshold).float()
    bg_mask = 1.0 - fg_mask
    fg_count = fg_mask.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)
    bg_count = bg_mask.flatten(start_dim=1).sum(dim=1).clamp_min(1.0)
    fg_sim = (similarity * fg_mask).flatten(start_dim=1).sum(dim=1) / fg_count
    bg_sim = (similarity * bg_mask).flatten(start_dim=1).sum(dim=1) / bg_count
    return fg_sim - bg_sim


def select_cac_views(cac_scores, probabilities, selection_p):
    """Rank views using quality + entropy with average ranks for exact ties."""
    if cac_scores.ndim != 2:
        raise ValueError(f"Expected CAC scores (B,V), got {tuple(cac_scores.shape)}")
    if probabilities.ndim < 3:
        raise ValueError(
            f"Expected probabilities (B,V,...), got {tuple(probabilities.shape)}"
        )
    if cac_scores.shape[:2] != probabilities.shape[:2]:
        raise ValueError("CAC scores and probabilities must agree in batch/view dimensions")
    if not 0 < float(selection_p) <= 1:
        raise ValueError(f"selection_p must be in (0, 1], got {selection_p}")

    batch_size, num_views = cac_scores.shape
    entropy = _binary_view_entropy(
        probabilities.reshape(batch_size * num_views, *probabilities.shape[2:])
    ).view(batch_size, num_views)
    entropy_ranks = average_tie_ranks(entropy, descending=False)
    cac_ranks = average_tie_ranks(cac_scores, descending=True)
    combined_ranks = entropy_ranks + cac_ranks
    keep = max(1, int(num_views * float(selection_p)))
    return torch.argsort(combined_ranks, dim=1, stable=True)[:, :keep]


def cac_loss(cac_scores):
    """CM-TTA CAC loss: maximize foreground/background concept contrast."""
    return -cac_scores.mean()


class VoxTellPromptSFDA:
    """Offline SFDA that updates exactly one free soft prompt tensor."""

    def __init__(
        self,
        model: nn.Module,
        initial_soft_prompt: torch.Tensor,
        device,
        args,
        qwen_text_encoder: Optional[nn.Module] = None,
    ):
        self.model = model.to(device)
        self.model.eval()
        for parameter in self.model.parameters():
            parameter.requires_grad_(False)

        self.qwen_text_encoder = qwen_text_encoder
        if self.qwen_text_encoder is not None:
            self.qwen_text_encoder.eval()
            for parameter in self.qwen_text_encoder.parameters():
                parameter.requires_grad_(False)
        self._assert_frozen_modules("initialization")

        if initial_soft_prompt.ndim == 2:
            initial_soft_prompt = initial_soft_prompt.unsqueeze(1)
        if initial_soft_prompt.ndim != 3 or initial_soft_prompt.shape[0] != 1:
            raise ValueError(
                "Expected one prompt embedding with shape (1, 1, D) or (1, D), "
                f"got {tuple(initial_soft_prompt.shape)}"
            )
        if initial_soft_prompt.shape[1] != 1:
            raise ValueError(
                "方案 A only supports one soft prompt vector per prompt; "
                f"received {initial_soft_prompt.shape[1]} vectors"
            )
        self.embedding_dim = int(initial_soft_prompt.shape[-1])
        # Keep the tiny trainable state in FP32 even when Qwen emitted an
        # FP16/BF16 embedding. Autocast converts it at VoxTell operations, while
        # AdamW and the leaf gradient retain a numerically stable master copy.
        initial_soft_prompt = initial_soft_prompt.detach().to(
            device=device, dtype=torch.float32
        )
        self.soft_prompt_embedding = nn.Parameter(initial_soft_prompt.clone())
        self.initial_soft_prompt = initial_soft_prompt.clone()
        self.teacher_soft_prompt = initial_soft_prompt.clone()
        self.device, self.args = device, args
        self.quality_mode = str(getattr(args, "quality_mode", "cac")).lower()
        if self.quality_mode not in QUALITY_MODES:
            raise ValueError(
                f"quality_mode must be one of {QUALITY_MODES}, got {self.quality_mode!r}"
            )
        if self.quality_mode != "cac" and float(getattr(args, "w_quality", 0.0)) != 0.0:
            raise ValueError(
                "Semantic quality is evaluation-only in this version; use w_quality=0"
            )
        default_quality_config = Path(__file__).resolve().parents[1] / "configs" / "tse.json"
        self.quality_config = load_quality_config(
            getattr(args, "quality_config", default_quality_config)
        )
        self.prototype_memory = SemanticPrototypeMemory(
            self.quality_config["num_prototypes"]
        )
        self.prototype_diagnostics = {}
        self.optimizer = torch.optim.AdamW(
            [self.soft_prompt_embedding], lr=args.lr, weight_decay=args.weight_decay
        )
        self._assert_optimizer_only_soft_prompt()
        trainable_parameter_count = sum(
            parameter.numel() for parameter in self.optimizer_parameters
        )
        print(
            "Trainable parameters: soft_prompt_embedding "
            f"{trainable_parameter_count} (embedding_dim={self.embedding_dim}, "
            f"dtype={self.soft_prompt_embedding.dtype})"
        )
        assert trainable_parameter_count == self.embedding_dim, (
            "For a single prompt, trainable parameter count must equal the "
            f"embedding dimension ({self.embedding_dim}), got {trainable_parameter_count}"
        )
        amp_init_scale = float(getattr(args, "amp_init_scale", 1024.0))
        if amp_init_scale <= 0:
            raise ValueError(f"amp_init_scale must be positive, got {amp_init_scale}")
        if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
            self.scaler = torch.amp.GradScaler(
                "cuda", enabled=device.type == "cuda", init_scale=amp_init_scale
            )
        else:  # PyTorch 2.0 compatibility.
            self.scaler = torch.cuda.amp.GradScaler(
                enabled=device.type == "cuda", init_scale=amp_init_scale
            )
        self._consecutive_amp_skips = 0
        self.record_soft_prompt_grad_norm = bool(
            getattr(args, "record_soft_prompt_grad_norm", False)
        )

        # These hooks reuse the projected VoxTell bottleneck and fixed-prompt
        # text representation for both CAC and TSE; no extra decoder is run.
        hook_model = getattr(self.model, "_orig_mod", self.model)
        hook_model = getattr(hook_model, "module", hook_model)
        vision_layer = self.quality_config["feature_layer"]
        text_layer = self.quality_config["text_feature_layer"]
        missing = [name for name in (vision_layer, text_layer) if not hasattr(hook_model, name)]
        if missing:
            raise AttributeError(
                "VoxTell network is missing configured quality feature module(s): "
                + ", ".join(missing)
            )
        self._cac_features = {}
        self._hook_handles = [
            getattr(hook_model, vision_layer).register_forward_hook(self._capture("vision")),
            getattr(hook_model, text_layer).register_forward_hook(self._capture("text")),
        ]

    @property
    def optimizer_parameters(self):
        return [
            parameter
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        ]

    def _assert_frozen_modules(self, stage):
        trainable_network = [
            name
            for name, parameter in self.model.named_parameters()
            if parameter.requires_grad
        ]
        if trainable_network:
            raise RuntimeError(
                f"VoxTell network parameters must be frozen at {stage}; "
                f"found trainable: {trainable_network[:5]}"
            )
        network_grads = [
            name for name, parameter in self.model.named_parameters() if parameter.grad is not None
        ]
        if network_grads:
            raise RuntimeError(
                f"VoxTell parameter gradients detected at {stage}: {network_grads[:5]}"
            )
        if self.qwen_text_encoder is not None:
            trainable_qwen = [
                name
                for name, parameter in self.qwen_text_encoder.named_parameters()
                if parameter.requires_grad
            ]
            if trainable_qwen:
                raise RuntimeError(
                    "Qwen text encoder must be frozen and outside adaptation; "
                    f"found trainable: {trainable_qwen[:5]}"
                )
            qwen_grads = [
                name
                for name, parameter in self.qwen_text_encoder.named_parameters()
                if parameter.grad is not None
            ]
            if qwen_grads:
                raise RuntimeError(
                    f"Qwen text encoder gradients detected at {stage}: {qwen_grads[:5]}"
                )

    def _assert_optimizer_only_soft_prompt(self):
        parameters = self.optimizer_parameters
        if len(parameters) != 1 or parameters[0] is not self.soft_prompt_embedding:
            raise RuntimeError(
                "Optimizer must contain exactly soft_prompt_embedding and no "
                "VoxTell/Qwen parameter"
            )

    def _assert_no_forbidden_grads(self, stage):
        self._assert_frozen_modules(stage)
        self._assert_optimizer_only_soft_prompt()

    def _capture(self, key):
        def hook(_module, _inputs, output):
            self._cac_features[key] = output

        return hook

    def close(self):
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()

    def _text(self, soft_prompt, batch_size):
        return soft_prompt.expand(batch_size, -1, -1).unsqueeze(2)

    @staticmethod
    def _make_extra_views(base, count):
        views = []
        for _ in range(count):
            result = base.clone()
            if torch.rand(()) < 0.8:
                result = result * torch.empty((), device=result.device).uniform_(0.85, 1.15)
            if torch.rand(()) < 0.8:
                result = result + torch.empty((), device=result.device).uniform_(-0.15, 0.15)
            if torch.rand(()) < 0.5:
                result = result + torch.randn_like(result) * 0.05
            views.append(result.contiguous())
        return views

    @staticmethod
    def _normalize_case_ids(case_ids, batch_size):
        if case_ids is None:
            return [f"anonymous-{index}" for index in range(batch_size)]
        if isinstance(case_ids, str):
            case_ids = [case_ids]
        case_ids = [str(case_id) for case_id in case_ids]
        if len(case_ids) != batch_size:
            raise ValueError("Number of case identifiers must match batch size")
        return case_ids

    def _semantic_quality(self, vision_features, logits, case_ids):
        positive, negative, valid = self.prototype_memory.leave_one_out(
            case_ids, vision_features.device
        )
        return compute_semantic_quality(
            vision_features,
            logits,
            positive,
            negative,
            valid,
            self.quality_config,
            stop_evidence_gradient=bool(
                self.quality_config.get("stop_evidence_gradient", True)
            ),
        )

    def _quality_scores(self, vision_features, text_features, logits, case_ids):
        cac = compute_cac_score(vision_features, text_features, logits)
        semantic = None
        if self.quality_mode != "cac":
            semantic = self._semantic_quality(vision_features, logits, case_ids)
            score = semantic[self.quality_mode]
        else:
            score = cac
        return score, cac, semantic

    @staticmethod
    def _make_deterministic_seed_views(base, count):
        """Create fixed aligned intensity views without consuming any RNG state."""
        views = [base]
        scales = (0.90, 1.10, 0.95, 1.05)
        offsets = (-0.05, 0.05, 0.025, -0.025)
        for index in range(1, int(count)):
            position = (index - 1) % len(scales)
            cycle = (index - 1) // len(scales)
            attenuation = 1.0 / (cycle + 1)
            scale = 1.0 + (scales[position] - 1.0) * attenuation
            offset = offsets[position] * attenuation
            views.append((base * scale + offset).contiguous())
        return torch.stack(views, dim=1)

    @staticmethod
    def _new_case_diagnostic(histogram_bins):
        return {
            "patches": 0,
            "fg_count": 0.0,
            "bg_count": 0.0,
            "valid_count": 0.0,
            "similarity_count": 0.0,
            "similarity_sum": 0.0,
            "similarity_square_sum": 0.0,
            "similarity_min": float("inf"),
            "similarity_max": float("-inf"),
            "similarity_histogram": torch.zeros(histogram_bins, dtype=torch.float64),
        }

    @staticmethod
    def _accumulate_case_diagnostic(target, statistics):
        target["patches"] += int(statistics["fg_count"].shape[0])
        for name in (
            "fg_count",
            "bg_count",
            "valid_count",
            "similarity_count",
            "similarity_sum",
            "similarity_square_sum",
        ):
            target[name] += float(statistics[name].detach().sum().cpu())
        target["similarity_min"] = min(
            target["similarity_min"],
            float(statistics["similarity_min"].detach().min().cpu()),
        )
        target["similarity_max"] = max(
            target["similarity_max"],
            float(statistics["similarity_max"].detach().max().cpu()),
        )
        target["similarity_histogram"] += statistics[
            "similarity_histogram"
        ].detach().double().cpu().sum(dim=0)

    def _finalize_prototype_diagnostics(self, raw_diagnostics):
        cases = {}
        global_raw = self._new_case_diagnostic(
            int(self.quality_config["similarity_histogram_bins"])
        )
        for case_id in sorted(raw_diagnostics):
            raw = raw_diagnostics[case_id]
            valid_count = max(raw["valid_count"], float(self.quality_config["epsilon"]))
            positive, negative, valid = self.prototype_memory.leave_one_out(
                [case_id], "cpu"
            )
            del positive, negative
            cases[case_id] = {
                "patches": raw["patches"],
                "foreground_seed_count": raw["fg_count"],
                "background_seed_count": raw["bg_count"],
                "foreground_seed_fraction": raw["fg_count"] / valid_count,
                "background_seed_fraction": raw["bg_count"] / valid_count,
                "total_seed_fraction": (raw["fg_count"] + raw["bg_count"]) / valid_count,
                "valid_feature_voxels": raw["valid_count"],
                "loo_prototype_valid": bool(valid.all()),
                "similarity_distribution": summarize_similarity_distribution(
                    raw["similarity_histogram"],
                    raw["similarity_count"],
                    raw["similarity_sum"],
                    raw["similarity_square_sum"],
                    raw["similarity_min"],
                    raw["similarity_max"],
                ),
            }
            for name in (
                "patches",
                "fg_count",
                "bg_count",
                "valid_count",
                "similarity_count",
                "similarity_sum",
                "similarity_square_sum",
            ):
                global_raw[name] += raw[name]
            global_raw["similarity_min"] = min(
                global_raw["similarity_min"], raw["similarity_min"]
            )
            global_raw["similarity_max"] = max(
                global_raw["similarity_max"], raw["similarity_max"]
            )
            global_raw["similarity_histogram"] += raw["similarity_histogram"]

        epsilon = float(self.quality_config["epsilon"])
        if global_raw["fg_count"] <= 0 or global_raw["bg_count"] <= 0:
            raise RuntimeError(
                "TSE prototype construction failed: global foreground/background "
                f"seed counts are {global_raw['fg_count']}/{global_raw['bg_count']}. "
                "Adjust seed thresholds after inspecting similarity diagnostics."
            )
        valid_count = max(global_raw["valid_count"], epsilon)
        dataset = {
            "cases": len(cases),
            "patches": global_raw["patches"],
            "foreground_seed_count": global_raw["fg_count"],
            "background_seed_count": global_raw["bg_count"],
            "foreground_seed_fraction": global_raw["fg_count"] / valid_count,
            "background_seed_fraction": global_raw["bg_count"] / valid_count,
            "total_seed_fraction": (
                global_raw["fg_count"] + global_raw["bg_count"]
            ) / valid_count,
            "valid_foreground_cases": sum(
                int(item["foreground_seed_count"] > 0) for item in cases.values()
            ),
            "valid_background_cases": sum(
                int(item["background_seed_count"] > 0) for item in cases.values()
            ),
            "valid_loo_cases": sum(
                int(item["loo_prototype_valid"]) for item in cases.values()
            ),
            "similarity_distribution": summarize_similarity_distribution(
                global_raw["similarity_histogram"],
                global_raw["similarity_count"],
                global_raw["similarity_sum"],
                global_raw["similarity_square_sum"],
                global_raw["similarity_min"],
                global_raw["similarity_max"],
            ),
        }
        if dataset["valid_loo_cases"] == 0:
            raise RuntimeError(
                "TSE prototype construction produced no valid leave-one-case-out "
                "prototype; at least two cases with foreground and background seeds are required"
            )
        return {"cases": cases, "dataset": dataset}

    def build_prototype_memory(self, loader, force=False):
        """Build prototypes from deterministic non-overlapping full-case patches."""
        if len(self.prototype_memory) and not force:
            return
        try:
            from data.sfda_voxtell import (
                extract_volume_patch,
                load_preprocessed_image,
                nonoverlapping_patch_locations,
                pad_to_patch_grid,
            )
        except ImportError as error:
            raise ImportError("Could not import deterministic target-volume utilities") from error
        dataset = getattr(loader, "dataset", None)
        entries = getattr(dataset, "entries", None)
        patch_size = getattr(dataset, "patch_size", None)
        if entries is None or patch_size is None:
            raise TypeError(
                "Prototype construction requires a VoxTellTargetDataset with entries/patch_size"
            )
        self.prototype_memory.clear()
        seed_views = int(self.quality_config["seed_views"])
        patch_batch_size = int(self.quality_config["prototype_patch_batch_size"])
        histogram_bins = int(self.quality_config["similarity_histogram_bins"])
        rank = dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        world_size = dist.get_world_size() if dist.is_available() and dist.is_initialized() else 1
        raw_diagnostics = {}
        for entry_index, (image_path, label_path) in enumerate(entries):
            if label_path is not None:
                raise RuntimeError("Prototype construction must receive unlabeled train entries")
            if entry_index % world_size != rank:
                continue
            case_id = image_path.name
            volume = load_preprocessed_image(image_path)
            padded, valid_mask, _ = pad_to_patch_grid(volume, patch_size)
            locations = nonoverlapping_patch_locations(padded.shape[-3:], patch_size)
            case_diagnostic = self._new_case_diagnostic(histogram_bins)
            for start in range(0, len(locations), patch_batch_size):
                batch_locations = locations[start:start + patch_batch_size]
                patches = torch.stack(
                    [extract_volume_patch(padded, location, patch_size) for location in batch_locations]
                ).to(self.device, non_blocking=True)
                valid_patches = torch.stack(
                    [extract_volume_patch(valid_mask, location, patch_size) for location in batch_locations]
                ).to(self.device, non_blocking=True)
                views = self._make_deterministic_seed_views(patches, seed_views)
                flat_views = views.reshape(
                    patches.shape[0] * seed_views, *patches.shape[1:]
                )
                self._cac_features.clear()
                with torch.no_grad(), torch.autocast(
                    device_type=self.device.type, enabled=self.device.type == "cuda"
                ):
                    prompt = self._text(self.initial_soft_prompt, flat_views.shape[0])
                    logits = self.model(flat_views, prompt)
                statistics = extract_case_seed_statistics(
                    self._cac_features["vision"],
                    self._cac_features["text"],
                    logits,
                    patches.shape[0],
                    seed_views,
                    self.quality_config,
                    valid_masks=valid_patches,
                )
                self.prototype_memory.add([case_id] * patches.shape[0], statistics)
                self._accumulate_case_diagnostic(case_diagnostic, statistics)
            raw_diagnostics[case_id] = case_diagnostic
        self.prototype_memory.synchronize()
        if not len(self.prototype_memory):
            raise RuntimeError("No target cases were available for prototype construction")
        if world_size > 1:
            gathered = [None for _ in range(world_size)]
            dist.all_gather_object(gathered, raw_diagnostics)
            raw_diagnostics = {
                case_id: diagnostic
                for rank_diagnostics in gathered
                for case_id, diagnostic in rank_diagnostics.items()
            }
        self.prototype_diagnostics = self._finalize_prototype_diagnostics(raw_diagnostics)
        if rank == 0:
            for case_id, diagnostic in self.prototype_diagnostics["cases"].items():
                print(f"TSE seeds {case_id}: {json.dumps(diagnostic, ensure_ascii=False)}")
            print(
                "TSE prototype dataset: "
                + json.dumps(self.prototype_diagnostics["dataset"], ensure_ascii=False)
            )
            output_dir = Path(getattr(self.args, "output_dir", "."))
            output_dir.mkdir(parents=True, exist_ok=True)
            (output_dir / "prototype_diagnostics.json").write_text(
                json.dumps(self.prototype_diagnostics, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )

    def _select_views(self, views, case_ids):
        """Rank every view without retaining a backward graph."""
        batch_size, num_views = views.shape[:2]
        flat_views = views.reshape(batch_size * num_views, *views.shape[2:])
        self._cac_features.clear()
        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            selection_prompt = self._text(
                self.soft_prompt_embedding.detach(), flat_views.shape[0]
            )
            flat_logits = self.model(flat_views, selection_prompt)
            probabilities = torch.sigmoid(flat_logits.float()).view(
                batch_size, num_views, *flat_logits.shape[1:]
            )
            flat_case_ids = [
                case_id for case_id in case_ids for _ in range(num_views)
            ]
            scores, _, _ = self._quality_scores(
                self._cac_features["vision"],
                self._cac_features["text"],
                flat_logits,
                flat_case_ids,
            )
            scores = scores.view(batch_size, num_views)
            return select_cac_views(scores, probabilities, self.args.selection_p)

    def adapt_batch(self, weak, strong, case_ids=None):
        weak = weak.to(self.device, non_blocking=True)
        strong = strong.to(self.device, non_blocking=True)
        batch_size = weak.shape[0]
        case_ids = self._normalize_case_ids(case_ids, batch_size)
        if self.quality_mode != "cac" and not len(self.prototype_memory):
            raise RuntimeError("TSE mode requires build_prototype_memory() before adaptation")
        num_aug_views = max(1, int(getattr(self.args, "num_aug_views", 9)))

        with torch.no_grad(), torch.autocast(
            device_type=self.device.type, enabled=self.device.type == "cuda"
        ):
            teacher_soft_prompt = self._text(self.teacher_soft_prompt, batch_size)
            teacher_prob = torch.sigmoid(self.model(weak, teacher_soft_prompt).float())
            confidence = torch.maximum(teacher_prob, 1 - teacher_prob)
            pseudo = (teacher_prob >= 0.5).float()
            valid = (confidence >= self.args.confidence_threshold).float()

        # Intensity-only views keep teacher pseudo-labels voxel-aligned.
        extra = self._make_extra_views(weak, max(0, num_aug_views - 2))
        views = (
            torch.stack([weak, strong, *extra], dim=1)
            if num_aug_views > 1
            else weak[:, None]
        )
        views = views[:, :num_aug_views]

        # View ranking does not need gradients. Only rerun the selected views
        # with autograd so the 3-D network does not retain activations for all
        # augmented views at once.
        selected = self._select_views(views, case_ids)
        gather_shape = (batch_size, selected.shape[1]) + (1,) * (views.ndim - 2)
        selected_views = views.gather(
            1,
            selected.view(*gather_shape).expand(
                batch_size, selected.shape[1], *views.shape[2:]
            ),
        ).contiguous()
        flat_selected_views = selected_views.view(
            batch_size * selected.shape[1], *views.shape[2:]
        )

        self._cac_features.clear()
        self.optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=self.device.type, enabled=self.device.type == "cuda"):
            student_soft_prompt = self._text(
                self.soft_prompt_embedding, flat_selected_views.shape[0]
            )
            flat_logits = self.model(flat_selected_views, student_soft_prompt)
            selected_case_ids = [
                case_id for case_id in case_ids for _ in range(selected.shape[1])
            ]
            selected_quality, selected_cac_flat, semantic = self._quality_scores(
                self._cac_features["vision"],
                self._cac_features["text"],
                flat_logits,
                selected_case_ids,
            )
            selected_quality = selected_quality.view(batch_size, selected.shape[1])
            selected_cac = selected_cac_flat.view(batch_size, selected.shape[1])
            selected_logits = flat_logits.view(
                batch_size, selected.shape[1], *flat_logits.shape[1:]
            )
            selected_pseudo = pseudo[:, None].expand_as(selected_logits)
            selected_valid = valid[:, None].expand_as(selected_logits)
            segmentation, bce, dice = masked_segmentation_loss(
                selected_logits, selected_pseudo, selected_valid
            )
            entropy = entropy_loss(selected_logits, selected_valid)
            loss_cac = cac_loss(selected_cac)
            loss_quality = -selected_quality.mean()
            quality_weight = (
                self.args.w_cac
                if self.quality_mode == "cac"
                else float(getattr(self.args, "w_quality", 0.0))
            )
            loss = (
                self.args.w_seg * segmentation
                + self.args.w_entropy * entropy
                + quality_weight * loss_quality
            )

        loss_terms = {
            "loss": loss,
            "segmentation": segmentation,
            "entropy": entropy,
            "cac_loss": loss_cac,
            "quality_loss": loss_quality,
        }
        nonfinite_terms = [
            name for name, value in loss_terms.items() if not torch.isfinite(value).all()
        ]
        if nonfinite_terms:
            details = ", ".join(
                f"{name}={float(value.detach().cpu())}" for name, value in loss_terms.items()
            )
            raise RuntimeError(
                "Non-finite forward loss term(s): "
                f"{', '.join(nonfinite_terms)} ({details})"
            )

        self.scaler.scale(loss).backward()
        self._assert_no_forbidden_grads("after backward")
        self.scaler.unscale_(self.optimizer)
        prompt_grad = self.soft_prompt_embedding.grad
        if prompt_grad is None:
            raise RuntimeError("soft_prompt_embedding gradient is missing")
        grad_is_finite = bool(torch.isfinite(prompt_grad).all())
        if not grad_is_finite and not self.scaler.is_enabled():
            raise RuntimeError(
                "soft_prompt_embedding gradient is non-finite while AMP is disabled"
            )
        soft_prompt_grad_norm = (
            float(prompt_grad.detach().norm().cpu()) if grad_is_finite else 0.0
        )
        if grad_is_finite:
            torch.nn.utils.clip_grad_norm_(
                [self.soft_prompt_embedding], self.args.grad_clip
            )
        scale_before = float(self.scaler.get_scale())
        self.scaler.step(self.optimizer)
        self.scaler.update()
        scale_after = float(self.scaler.get_scale())
        update_skipped = not grad_is_finite
        if update_skipped:
            self._consecutive_amp_skips += 1
            self.optimizer.zero_grad(set_to_none=True)
            print(
                "AMP skipped one optimizer step after a non-finite prompt gradient; "
                f"scale={scale_before:g}->{scale_after:g}, "
                f"loss={float(loss.detach().cpu()):.6g}, "
                f"seg={float(segmentation.detach().cpu()):.6g}, "
                f"entropy={float(entropy.detach().cpu()):.6g}, "
                f"cac_loss={float(loss_cac.detach().cpu()):.6g}"
            )
            if self._consecutive_amp_skips >= 8:
                raise RuntimeError(
                    "AMP encountered non-finite prompt gradients for 8 consecutive "
                    "batches; this is persistent numerical instability rather than "
                    "a recoverable loss-scale overflow"
                )
        else:
            self._consecutive_amp_skips = 0
        self._assert_no_forbidden_grads("after optimizer step")

        if not update_skipped:
            with torch.no_grad():
                self.teacher_soft_prompt.mul_(self.args.ema_momentum).add_(
                    self.soft_prompt_embedding.detach(),
                    alpha=1 - self.args.ema_momentum,
                )
        values = {
            "loss": float(loss.detach().cpu()),
            "bce": float(bce.cpu()),
            "dice": float(dice.cpu()),
            "entropy": float(entropy.detach().cpu()),
            "cac": float(selected_cac.detach().mean().cpu()),
            "cac_loss": float(loss_cac.detach().cpu()),
            "quality": float(selected_quality.detach().mean().cpu()),
            "quality_loss": float(loss_quality.detach().cpu()),
            "coverage": float(valid.mean().cpu()),
            "selected_views": float(selected.shape[1]),
            "update_skipped": float(update_skipped),
        }
        if self.record_soft_prompt_grad_norm:
            values["soft_prompt_grad_norm"] = soft_prompt_grad_norm
        if semantic is not None:
            values.update(
                purity=float(semantic["purity"].detach().mean().cpu()),
                completeness=float(semantic["completeness"].detach().mean().cpu()),
                tse=float(semantic["tse"].detach().mean().cpu()),
                prototype_valid=float(
                    semantic["prototype_valid"].detach().float().mean().cpu()
                ),
            )
        return values

    def fit(self, loader, epoch_end_callback=None):
        history = []
        if self.quality_mode != "cac":
            self.build_prototype_memory(loader)
        metric_names = [
            "loss", "bce", "dice", "entropy", "cac", "cac_loss", "quality",
            "quality_loss", "coverage",
            "selected_views", "update_skipped"
        ]
        if self.quality_mode != "cac":
            metric_names.extend(("purity", "completeness", "tse", "prototype_valid"))
        if self.record_soft_prompt_grad_norm:
            metric_names.append("soft_prompt_grad_norm")
        for epoch in range(1, self.args.epochs + 1):
            totals = {key: 0.0 for key in metric_names}
            for step, batch in enumerate(loader, start=1):
                # The optional third item is only an image identifier; target
                # labels are never consumed during train_cases adaptation.
                weak, strong = batch[:2]
                case_ids = batch[2] if len(batch) > 2 else None
                values = self.adapt_batch(weak, strong, case_ids)
                for key in totals:
                    totals[key] += values[key]
                if step % self.args.print_freq == 0 or step == len(loader):
                    print(
                        f"epoch {epoch}/{self.args.epochs} step {step}/{len(loader)} "
                        f"loss={totals['loss']/step:.4f} "
                        f"{self.quality_mode}={totals['quality']/step:.4f} "
                        f"coverage={totals['coverage']/step:.3f} "
                        f"skipped={int(totals['update_skipped'])}"
                    )
            row = {key: value / max(1, len(loader)) for key, value in totals.items()}
            row["epoch"] = epoch
            history.append(row)
            print(json.dumps(row, ensure_ascii=False))
            if epoch_end_callback is not None:
                epoch_end_callback(epoch, row, history)
        return history


def save_sfda_checkpoint(path, adapter: VoxTellPromptSFDA, args, history):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "voxtell-sfda-prompt-tse-v4",
            "soft_prompt_embedding": adapter.soft_prompt_embedding.detach().cpu(),
            "initial_soft_prompt": adapter.initial_soft_prompt.detach().cpu(),
            "teacher_soft_prompt": adapter.teacher_soft_prompt.detach().cpu(),
            # Legacy aliases preserve loading compatibility for v1/v2 tools.
            "prompt_embedding": adapter.soft_prompt_embedding.detach().cpu(),
            "teacher_prompt": adapter.teacher_soft_prompt.detach().cpu(),
            "optimizer": adapter.optimizer.state_dict(),
            "scaler": adapter.scaler.state_dict(),
            "prototype_memory": adapter.prototype_memory.state_dict(),
            "prototype_diagnostics": adapter.prototype_diagnostics,
            "quality_config": adapter.quality_config,
            "history": history,
            "args": vars(args),
        },
        path,
    )


def load_sfda_checkpoint(path, adapter: VoxTellPromptSFDA):
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if checkpoint.get("format") not in {
        "voxtell-sfda-prompt-v1",
        "voxtell-sfda-prompt-cac-v2",
        "voxtell-sfda-soft-prompt-cac-v3",
        "voxtell-sfda-prompt-tse-v4",
    }:
        raise ValueError(f"Unsupported checkpoint format: {checkpoint.get('format')}")
    soft_prompt = checkpoint.get("soft_prompt_embedding", checkpoint.get("prompt_embedding"))
    if soft_prompt is None:
        raise KeyError("Checkpoint does not contain soft_prompt_embedding/prompt_embedding")
    teacher_soft_prompt = checkpoint.get(
        "teacher_soft_prompt", checkpoint.get("teacher_prompt", soft_prompt)
    )
    adapter.soft_prompt_embedding.data.copy_(soft_prompt.to(adapter.device))
    if "initial_soft_prompt" in checkpoint:
        adapter.initial_soft_prompt.copy_(checkpoint["initial_soft_prompt"].to(adapter.device))
    adapter.teacher_soft_prompt.copy_(teacher_soft_prompt.to(adapter.device))
    if "prototype_memory" in checkpoint:
        adapter.prototype_memory.load_state_dict(checkpoint["prototype_memory"])
    if "prototype_diagnostics" in checkpoint:
        adapter.prototype_diagnostics = checkpoint["prototype_diagnostics"]
    if "optimizer" in checkpoint:
        adapter.optimizer.load_state_dict(checkpoint["optimizer"])
        for state in adapter.optimizer.state.values():
            for key, value in state.items():
                if torch.is_tensor(value):
                    state[key] = value.to(
                        device=adapter.device,
                        dtype=(
                            adapter.soft_prompt_embedding.dtype
                            if value.is_floating_point()
                            else value.dtype
                        ),
                    )
    if "scaler" in checkpoint:
        adapter.scaler.load_state_dict(checkpoint["scaler"])
    adapter._assert_no_forbidden_grads("after checkpoint load")
    return checkpoint
