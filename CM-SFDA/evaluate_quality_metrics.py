"""Full-volume, GT-only audit of VoxTell CAC and target semantic evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.sfda_voxtell import (
    load_preprocessed_labeled_case,
    make_target_loader,
    read_image_entries,
)
from method.semantic_quality import (
    average_tie_ranks,
    compute_saaf_quality,
    text_similarity_map,
)
from method.sfda_voxtell import VoxTellPromptSFDA
from run_sfda_voxtell import build_predictor, seed_everything


QUALITY_NAMES = (
    "confidence", "entropy", "consistency", "cac", "purity", "coverage", "saaf",
    "completeness", "tse",
)
SELECTION_NAMES = ("cac", "saaf", "purity", "coverage", "cac_entropy", "saaf_entropy")


def _fuse_logits_then_sigmoid(logits, denominator):
    """Fuse Gaussian-weighted logits first, then apply sigmoid exactly once."""
    denominator = denominator.clamp_min(torch.finfo(logits.dtype).eps)
    fused_logits = logits / denominator
    return fused_logits, torch.sigmoid(fused_logits)


def _rankdata(values):
    """Tie-aware one-based average ranks."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return ranks


def spearman(values, dice):
    """Return None for too few/constant valid observations."""
    values = np.asarray(values, dtype=np.float64)
    dice = np.asarray(dice, dtype=np.float64)
    finite = np.isfinite(values) & np.isfinite(dice)
    if int(finite.sum()) < 2:
        return None
    first, second = _rankdata(values[finite]), _rankdata(dice[finite])
    if first.std() == 0 or second.std() == 0:
        return None
    return float(np.corrcoef(first, second)[0, 1])


def binary_auroc(scores, target):
    """Tie-aware binary AUROC, or None when either class is absent."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    positive, negative = int(target.sum()), int((~target).sum())
    if positive == 0 or negative == 0:
        return None
    ranks = _rankdata(scores)
    return float((ranks[target].sum() - positive * (positive + 1) / 2) / (positive * negative))


def binary_auprc(scores, target):
    """Threshold-grouped average precision, or None for an empty target."""
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    positive = int(target.sum())
    if positive == 0:
        return None
    order = np.argsort(-scores, kind="mergesort")
    scores, target = scores[order], target[order]
    ends = np.r_[np.flatnonzero(np.diff(scores)), len(scores) - 1]
    true_positive = np.cumsum(target)[ends]
    false_positive = 1 + ends - true_positive
    recall = true_positive / positive
    precision = true_positive / (true_positive + false_positive)
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def _case_seed(base_seed, case_id):
    digest = hashlib.sha256(str(case_id).encode("utf-8")).digest()
    return (int(base_seed) + int.from_bytes(digest[:4], "little")) % (2**32)


def deterministic_view_parameters(num_views, seed):
    """Generate fixed intensity parameters shared by every metric."""
    if int(num_views) < 1:
        raise ValueError("num_aug_views must be positive")
    generator = np.random.default_rng(int(seed))
    parameters = [(1.0, 0.0)]
    parameters.extend(
        (float(generator.uniform(0.85, 1.15)), float(generator.uniform(-0.15, 0.15)))
        for _ in range(int(num_views) - 1)
    )
    return parameters


def _resize_patch_map(values, patch_size):
    if tuple(values.shape[-3:]) == tuple(patch_size):
        return values
    return F.interpolate(values.unsqueeze(1), size=patch_size, mode="trilinear", align_corners=False).squeeze(1)


def _align_target_to_volume(target, volume):
    """Align GT spatially with nearest-neighbor semantics only."""
    target = target.float()
    if tuple(target.shape[-3:]) != tuple(volume.shape[-3:]):
        target = F.interpolate(
            target.unsqueeze(0), size=tuple(volume.shape[-3:]), mode="nearest"
        ).squeeze(0)
    return target


def _add_spatial_patch(accumulator, patch, location):
    patch_size = patch.shape[-3:]
    slices = tuple(slice(int(start), int(start) + int(size)) for start, size in zip(location, patch_size))
    accumulator[(..., *slices)] += patch


def _training_style_views(volume, num_views, seed):
    """Reproduce dataset strong augmentation and adapter extra-view sampling."""
    generator = torch.Generator(device=volume.device)
    generator.manual_seed(int(seed) % (2**63 - 1))

    def augment(base):
        result = base.clone()
        if bool(torch.rand((), generator=generator, device=volume.device) < 0.8):
            scale = torch.empty((), device=volume.device).uniform_(
                0.85, 1.15, generator=generator
            )
            result = result * scale
        if bool(torch.rand((), generator=generator, device=volume.device) < 0.8):
            offset = torch.empty((), device=volume.device).uniform_(
                -0.15, 0.15, generator=generator
            )
            result = result + offset
        if bool(torch.rand((), generator=generator, device=volume.device) < 0.5):
            result = result + torch.randn(
                result.shape, generator=generator, device=volume.device
            ) * 0.05
        return result.contiguous()

    views = [volume, augment(volume)]
    while len(views) < int(num_views):
        views.append(augment(volume))
    return torch.stack(views[: int(num_views)], dim=0)


def _voxtell_sliding_window_views(adapter, volume, num_views, seed, case_id=None, target=None):
    """Run the exact VoxTell sliding-window/padding/Gaussian logit pipeline.

    VoxTell's predictor pads with ``pad_nd_image``, obtains slicers through its
    private slicer helper, weights every patch with nnU-Net's Gaussian map and
    only then divides the accumulated logits. Evidence and text similarity use
    the same Gaussian numerator/denominator. Candidate view selection is done
    independently for every sliding-window patch, matching training. When
    ``target`` is supplied, each patch also records candidate-view GT Dice,
    patch-best Dice, selected Dice and the non-negative patch-oracle gap.
    Full-volume selected maps are reported by Dice only; they are never compared
    with a fixed-view oracle.
    """
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian

    predictor = getattr(adapter, "predictor", None)
    if predictor is None:
        raise RuntimeError("Evaluation adapter must expose the VoxTell predictor")
    padded, revert_padding = pad_nd_image(
        volume, predictor.patch_size, "constant", {"value": 0}, True, None
    )
    slicers = predictor._internal_get_sliding_window_slicers(padded.shape[1:])
    gaussian = compute_gaussian(
        tuple(predictor.patch_size),
        sigma_scale=1.0 / 8,
        value_scaling_factor=10,
        device=adapter.device,
    ).float()
    padded_shape = tuple(int(value) for value in padded.shape[1:])
    valid_crop = revert_padding[1:]
    padded_valid = torch.zeros_like(padded, dtype=torch.float32)
    padded_valid[(..., *valid_crop)] = 1.0
    # These are disk-backed so the audit does not keep 9 logits, 9 evidence
    # maps, 9 similarity maps and multiple selected volumes in RAM. The
    # numerical accumulation order and dtype remain unchanged.
    target_padded = None
    if target is not None:
        target = _align_target_to_volume(target, volume)
        target_padded, _ = pad_nd_image(
            target, predictor.patch_size, "constant", {"value": 0}, True, None
        )
        if tuple(target_padded.shape[1:]) != padded_shape:
            raise ValueError(
                f"Padded target/volume shape mismatch: {target_padded.shape} vs {padded.shape}"
            )
    gaussian_sum = np.zeros(padded_shape, dtype=np.float32)
    patch_records = []
    selected_names = SELECTION_NAMES
    case_key = "validation-case" if case_id is None else str(case_id)
    prototype_valid = 0
    prototype_queries = 0
    with tempfile.TemporaryDirectory(prefix="voxtell_quality_") as cache_dir:
        def mmap(name, shape):
            return np.memmap(Path(cache_dir) / f"{name}.bin", mode="w+", dtype=np.float32, shape=shape)

        logit_sum = mmap("logits", (num_views, *padded_shape))
        evidence_sum = mmap("evidence", (num_views, *padded_shape))
        similarity_sum = mmap("similarity", (num_views, *padded_shape))
        selected_logit_sum = {name: mmap(f"selected_{name}", padded_shape) for name in selected_names}

        for patch_index, slicer in enumerate(slicers):
            patch = padded[slicer].to(adapter.device, non_blocking=True)
            valid_patch = padded_valid[slicer].to(adapter.device, non_blocking=True)
            views = _training_style_views(patch, num_views, seed + patch_index)
            valid_views = valid_patch.expand(num_views, -1, -1, -1, -1)
            adapter._cac_features.clear()
            with torch.no_grad(), torch.autocast(
                device_type=adapter.device.type, enabled=adapter.device.type == "cuda"
            ):
                prompt = adapter._text(adapter.initial_soft_prompt, num_views)
                logits = adapter.model(views, prompt)
            flat_case_ids = [case_key] * num_views
            current_text_features = adapter._cac_features["text"]
            _quality, cac, semantic = adapter._quality_scores(
                adapter._cac_features["vision"],
                adapter._cac_features["text"],
                logits,
                flat_case_ids,
                spatial_valid=valid_views,
            )
            # SAAF evidence is audited for both modes.  Legacy purity/TSE
            # details are intentionally not reused: they are prototype-based,
            # whereas this audit always measures the frozen ``liver`` anchor.
            if adapter.quality_metric == "saaf":
                saaf_details = semantic
            else:
                saaf_details = adapter._saaf_quality(
                    adapter._cac_features["vision"], logits, valid_views
                )
            if saaf_details is None:
                raise RuntimeError("SAAF audit did not produce anchor evidence details")
            semantic = saaf_details
            probability = torch.sigmoid(logits[:, 0].float())
            evidence = semantic["evidence"]
            similarity = text_similarity_map(
                adapter._cac_features["vision"], current_text_features
            )
            evidence = _resize_patch_map(evidence.permute(0, 3, 1, 2), tuple(predictor.patch_size))
            similarity = _resize_patch_map(similarity.permute(0, 3, 1, 2), tuple(predictor.patch_size))
            probability = _resize_patch_map(probability, tuple(predictor.patch_size))
            cac = cac.detach().view(1, num_views)
            semantic_purity = semantic["purity"].detach().view(1, num_views)
            semantic_coverage = semantic["coverage"].detach().view(1, num_views)
            semantic_saaf = semantic["saaf"].detach().view(1, num_views)
            clipped = probability.clamp(1e-6, 1 - 1e-6).view(1, num_views, *probability.shape[-3:])
            patch_metrics = {
                "confidence": torch.maximum(clipped, 1 - clipped).flatten(start_dim=2).mean(dim=2),
                "entropy": -(clipped * clipped.log() + (1 - clipped) * (1 - clipped).log()).flatten(start_dim=2).mean(dim=2),
                "consistency": _soft_consistency(probability).view(1, num_views),
                "cac": cac,
                "purity": semantic_purity,
                "coverage": semantic_coverage,
                "saaf": semantic_saaf,
                # Compatibility aliases used by older audit consumers.
                "completeness": semantic_coverage,
                "tse": semantic_saaf,
            }
            selected_indices = _selection_indices(patch_metrics)
            patch_slices = slicer[1:]
            weight = gaussian.detach().cpu()
            logits_cpu = logits[:, 0].float().detach().cpu()
            evidence_cpu = evidence.detach().cpu()
            similarity_cpu = similarity.detach().cpu()
            # Keep the original torch float32 accumulation order while using
            # writable memmap-backed storage instead of resident tensors.
            torch.from_numpy(logit_sum[(..., *patch_slices)]).add_(logits_cpu * weight)
            torch.from_numpy(evidence_sum[(..., *patch_slices)]).add_(evidence_cpu * weight)
            torch.from_numpy(similarity_sum[(..., *patch_slices)]).add_(similarity_cpu * weight)
            torch.from_numpy(gaussian_sum[patch_slices]).add_(weight)
            for name in selected_names:
                torch.from_numpy(selected_logit_sum[name][patch_slices]).add_(
                    logits_cpu[selected_indices[name]] * weight
                )
            patch_record = {
                "patch_index": patch_index,
                "location": [[int(s.start), int(s.stop)] for s in patch_slices],
                "quality": {name: [float(v) for v in patch_metrics[name].view(-1).detach().cpu()] for name in QUALITY_NAMES},
                "selected_view": selected_indices,
                "diagnostics": [
                    {
                        "case_id": case_key,
                        "view_id": int(view_index),
                        "entropy": float(patch_metrics["entropy"].view(-1)[view_index].cpu()),
                        "cac": float(cac.view(-1)[view_index].cpu()),
                        "saaf": float(semantic["saaf"].view(-1)[view_index].cpu()),
                        "purity": float(semantic["purity"].view(-1)[view_index].cpu()),
                        "coverage": float(semantic["coverage"].view(-1)[view_index].cpu()),
                        "mu_fg": float(semantic["mu_fg"].view(-1)[view_index].cpu()),
                        "mu_bg": float(semantic["mu_bg"].view(-1)[view_index].cpu()),
                        "attention_mad": float(semantic["attention_mad"].view(-1)[view_index].cpu()),
                        "mask_ratio": float(semantic["mask_ratio"].view(-1)[view_index].cpu()),
                        "evidence_sum": float(semantic["evidence_sum"].view(-1)[view_index].cpu()),
                        "valid": bool(semantic["valid"].view(-1)[view_index].cpu()),
                        "invalid_reason": semantic["invalid_reason"][view_index],
                        "selected": any(index == view_index for index in selected_indices.values()),
                    }
                    for view_index in range(num_views)
                ],
            }
            if target_padded is not None:
                # Dice is deliberately evaluated on CPU: fused probabilities
                # and padded GT must be on the same device.
                probability_cpu = probability.detach().cpu()
                target_patch_cpu = target_padded[slicer].detach().cpu()
                candidate_dice = _dice_per_view(probability_cpu, target_patch_cpu)
                best_dice = float(candidate_dice.max().detach().cpu())
                patch_record["candidate_dice"] = [float(v) for v in candidate_dice.detach().cpu()]
                patch_record["best_dice"] = best_dice
                patch_record["selection"] = {
                    name: dict(zip(
                        ("selected_dice", "patch_best_dice", "gap_to_patch_oracle"),
                        _patch_oracle_stats(candidate_dice, index),
                    ))
                    for name, index in selected_indices.items()
                }
            patch_records.append(patch_record)
            prototype_valid += int(semantic["valid"].sum().cpu())
            prototype_queries += int(semantic["valid"].numel())

        if not bool((gaussian_sum > 0).all()):
            raise RuntimeError("VoxTell sliding-window inference left uncovered voxels")
        crop = valid_crop
        denominator_np = np.maximum(np.asarray(gaussian_sum[crop]), np.finfo(np.float32).eps)
        fused_logits_np = np.asarray(logit_sum[(..., *crop)], dtype=np.float32).copy()
        fused_logits, probability = _fuse_logits_then_sigmoid(
            torch.from_numpy(fused_logits_np), torch.from_numpy(denominator_np.copy())
        )
        evidence_original = torch.from_numpy(
            (np.asarray(evidence_sum[(0, *crop)], dtype=np.float32) / denominator_np).copy()
        )
        full_metrics = {name: [] for name in QUALITY_NAMES}
        full_consistency = _soft_consistency(probability)
        for view_index in range(num_views):
            evidence_view = torch.from_numpy(
                (np.asarray(evidence_sum[(view_index, *crop)], dtype=np.float32) / denominator_np).copy()
            )
            similarity_view = torch.from_numpy(
                (np.asarray(similarity_sum[(view_index, *crop)], dtype=np.float32) / denominator_np).copy()
            )
            view_probability = probability[view_index:view_index + 1]
            clipped = view_probability.clamp(1e-6, 1 - 1e-6)
            saaf = compute_saaf_quality(
                view_probability,
                evidence_view[None],
                epsilon=adapter.quality_config["epsilon"],
                min_mass=adapter.quality_config.get("saaf_min_mass", 1e-6),
            )
            full_metrics["confidence"].append(float(torch.maximum(view_probability, 1 - view_probability).mean()))
            full_metrics["entropy"].append(float(-(clipped * clipped.log() + (1 - clipped) * (1 - clipped).log()).mean()))
            full_metrics["consistency"].append(float(full_consistency[view_index]))
            full_metrics["cac"].append(float(_cac_from_full_maps(view_probability, similarity_view[None])[0]))
            full_metrics["purity"].append(float(saaf["purity"][0]))
            full_metrics["coverage"].append(float(saaf["coverage"][0]))
            full_metrics["saaf"].append(float(saaf["saaf"][0]))
            full_metrics["completeness"].append(float(saaf["coverage"][0]))
            full_metrics["tse"].append(float(saaf["saaf"][0]))
        selected_dice = {}
        for name, value in selected_logit_sum.items():
            selected_np = (np.asarray(value[crop], dtype=np.float32) / denominator_np).copy()
            if target is not None:
                selected_probability = torch.sigmoid(torch.from_numpy(selected_np))
                selected_dice[name] = float(
                    _dice_per_view(selected_probability[None], target.float())[0]
                )
        return {
            "probability": probability,
            "evidence": evidence_original,
            "full_metrics": {name: torch.tensor(values, dtype=torch.float32) for name, values in full_metrics.items()},
            "selected_dice": selected_dice,
            "patch_records": patch_records,
            "prototype_valid_fraction": prototype_valid / prototype_queries if prototype_queries else 0.0,
            "locations": slicers,
        }


def _dice_per_view(probability, target):
    prediction = probability >= 0.5
    target = target.bool().expand_as(prediction)
    intersection = (prediction & target).flatten(start_dim=1).sum(dim=1).float()
    denominator = (
        prediction.flatten(start_dim=1).sum(dim=1) + target.flatten(start_dim=1).sum(dim=1)
    ).float()
    return torch.where(denominator > 0, 2 * intersection / denominator, torch.ones_like(denominator))


def _patch_oracle_stats(candidate_dice, selected_index):
    """Return selected Dice, patch-best Dice and a non-negative patch gap."""
    values = torch.as_tensor(candidate_dice, dtype=torch.float32).reshape(-1)
    if values.numel() == 0:
        raise ValueError("candidate_dice must contain at least one view")
    best = values.max()
    selected = values[int(selected_index)]
    gap = torch.clamp(best - selected, min=0.0)
    return float(selected), float(best), float(gap)


def _soft_consistency(probability):
    consensus = probability.mean(dim=0, keepdim=True).expand_as(probability)
    intersection = (probability * consensus).flatten(start_dim=1).sum(dim=1)
    denominator = probability.flatten(start_dim=1).sum(dim=1) + consensus.flatten(start_dim=1).sum(dim=1)
    return (2 * intersection + 1e-6) / (denominator + 1e-6)


def _cac_from_full_maps(probability, similarity):
    foreground = (probability > 0.5).float()
    background = 1.0 - foreground
    fg_count = foreground.flatten(start_dim=1).sum(dim=1).clamp_min(1)
    bg_count = background.flatten(start_dim=1).sum(dim=1).clamp_min(1)
    return (foreground * similarity).flatten(start_dim=1).sum(dim=1) / fg_count - (
        background * similarity
    ).flatten(start_dim=1).sum(dim=1) / bg_count


def _selection_indices(metrics):
    result = {name: int(metrics[name].argmax()) for name in ("cac", "saaf", "purity", "coverage")}
    entropy_rank = average_tie_ranks(metrics["entropy"].view(1, -1), descending=False)
    for quality in ("cac", "saaf"):
        quality_rank = average_tie_ranks(metrics[quality].view(1, -1), descending=True)
        combined = entropy_rank + quality_rank
        result[f"{quality}_entropy"] = int(torch.argsort(combined, dim=1, stable=True)[0, 0])
    return result


def evidence_localization_metrics(evidence, target, threshold):
    target_np = target.detach().cpu().numpy().astype(bool).reshape(-1)
    evidence_np = evidence.detach().cpu().numpy().astype(np.float64).reshape(-1)
    if not target_np.any():
        return {"valid": False, "reason": "empty_gt", "dice": None, "auroc": None, "auprc": None}
    binary = evidence_np >= float(threshold)
    denominator = int(binary.sum()) + int(target_np.sum())
    dice = 2 * int(np.logical_and(binary, target_np).sum()) / max(1, denominator)
    auroc = binary_auroc(evidence_np, target_np)
    return {
        "valid": auroc is not None,
        "reason": None if auroc is not None else "auroc_requires_two_classes",
        "dice": float(dice),
        "auroc": auroc,
        "auprc": binary_auprc(evidence_np, target_np),
    }


def _tensor_statistics(values):
    values = values.detach().float().cpu().reshape(-1)
    quantiles = torch.quantile(values, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95]))
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles": {
            name: float(value)
            for name, value in zip(("q05", "q25", "q50", "q75", "q95"), quantiles)
        },
    }


def save_visualization(path, image, target, prediction, evidence):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image, target = image.detach().cpu(), target.squeeze(0).detach().cpu()
    foreground_per_slice = target.flatten(start_dim=1).sum(dim=1)
    slice_index = int(foreground_per_slice.argmax()) if bool(target.any()) else target.shape[0] // 2
    panels = (
        (image[0, slice_index], "image", "gray", None, None),
        (target[slice_index], "GT", "gray", 0, 1),
        (prediction[slice_index].cpu(), "prediction", "gray", 0, 1),
        (evidence[slice_index].cpu(), "evidence", "magma", 0, 1),
    )
    figure, axes = plt.subplots(1, 4, figsize=(14, 4))
    for axis, (values, title, cmap, minimum, maximum) in zip(axes, panels):
        axis.imshow(values, cmap=cmap, vmin=minimum, vmax=maximum)
        axis.set_title(title)
        axis.axis("off")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(figure)


def parse_args():
    parser = argparse.ArgumentParser(description="Full-volume VoxTell quality/Dice audit")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--voxtell_root", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--prompt", default="prostate")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quality_config", default=str(Path(__file__).parent / "configs" / "tse.json"))
    parser.add_argument(
        "--quality_mode",
        choices=("cac", "purity", "completeness", "tse"),
        default="cac",
        help="Legacy semantic mode; SAAF is selected with --quality_metric saaf",
    )
    parser.add_argument(
        "--quality_metric", choices=("cac", "saaf"), default="saaf",
        help="Metric used by the adapter view-selection path",
    )
    parser.add_argument("--w_quality", type=float, default=0.0)
    parser.add_argument("--num_aug_views", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--visualize_cases", type=int, default=5)
    parser.add_argument("--output", default="results_/voxtell_sfda/quality_audit.json")
    parser.add_argument("--seed", type=int, default=1377)
    return parser.parse_args()


def summarize_quality_audit(
    cases,
    mixed_views,
    selected_dice,
    patch_selected_dice,
    patch_best_dice,
    patch_oracle_gap,
    best_fixed_view_dice,
    patch_spearman_case,
    invalid_evidence_cases,
    args,
    prototype_diagnostics,
):
    """Build the audit summary with separate patch-oracle and fixed-view stats."""
    def case_selection_means(field, name):
        values = [
            case["selections"][name][field]
            for case in cases
            if case.get("selections", {}).get(name, {}).get(field) is not None
        ]
        return float(np.mean(values)) if values else None

    macro_spearman = {}
    macro_valid_cases = {}
    for name in QUALITY_NAMES:
        valid = [case["within_case_spearman"][name] for case in cases
                 if case["within_case_spearman"][name] is not None]
        macro_spearman[name] = float(np.mean(valid)) if valid else None
        macro_valid_cases[name] = len(valid)
    global_spearman = {
        name: spearman([row[name] for row in mixed_views], [row["dice"] for row in mixed_views])
        for name in QUALITY_NAMES
    }
    patch_spearman_macro = {}
    patch_spearman_valid_cases = {}
    patch_spearman_valid_patches = {}
    for name in QUALITY_NAMES:
        case_values = [value for value in patch_spearman_case[name] if value is not None]
        patch_spearman_macro[name] = float(np.mean(case_values)) if case_values else None
        patch_spearman_valid_cases[name] = len(case_values)
        patch_spearman_valid_patches[name] = sum(
            case["patch_spearman"][name]["valid_patches"] for case in cases
        )
    valid_localization = [
        case["evidence_localization_original_view"] for case in cases
        if case["evidence_localization_original_view"]["valid"]
    ]
    return {
        "protocol": {
            "quality_mode": args.quality_mode,
            "quality_metric": getattr(args, "quality_metric", "cac"),
            "w_quality": args.w_quality,
            "w_cac": getattr(args, "w_cac", 0.0),
            "gt_usage": "offline metrics/visualization only",
            "views": "identical per-patch seeded training-style scale/offset/noise views for every metric",
            "inference": "VoxTell-native padding, sliding windows and Gaussian-weighted logit fusion before sigmoid",
            "oracle": "patchwise candidate-view GT Dice; fixed-view Dice is reported separately and never used for patch gap",
        },
        "prototype_diagnostics": prototype_diagnostics,
        "within_case_spearman_macro": macro_spearman,
        "within_case_spearman_valid_cases": macro_valid_cases,
        "global_mixed_view_spearman": global_spearman,
        "patch_spearman_macro": patch_spearman_macro,
        "patch_spearman_valid_cases": patch_spearman_valid_cases,
        "patch_spearman_valid_patches": patch_spearman_valid_patches,
        "selected_view_mean_dice": {
            name: float(np.mean(values)) if values else None for name, values in selected_dice.items()
        },
        "selected_view_valid_cases": {name: len(values) for name, values in selected_dice.items()},
        "patch_selected_dice_mean": {
            name: case_selection_means("patch_selected_dice", name) for name in SELECTION_NAMES
        },
        "patch_selected_valid_patches": {name: len(values) for name, values in patch_selected_dice.items()},
        "patch_oracle_best_dice_mean": (
            float(np.mean([
                case["selections"][SELECTION_NAMES[0]]["patch_best_dice"]
                for case in cases
                if case.get("selections", {}).get(SELECTION_NAMES[0], {}).get("patch_best_dice") is not None
            ]))
            if any(case.get("selections", {}).get(SELECTION_NAMES[0], {}).get("patch_best_dice") is not None for case in cases)
            else None
        ),
        "patch_oracle_valid_patches": len(patch_best_dice),
        "patch_oracle_gap_mean": {
            name: case_selection_means("patch_gap_to_oracle", name) for name in SELECTION_NAMES
        },
        "patch_oracle_gap_valid_patches": {name: len(values) for name, values in patch_oracle_gap.items()},
        "best_fixed_view_mean_dice": float(np.mean(best_fixed_view_dice)) if best_fixed_view_dice else None,
        "best_fixed_view_valid_cases": len(best_fixed_view_dice),
        "valid_cases": len(cases),
        "valid_patches": len(patch_best_dice),
        "evidence_localization_macro": {
            name: float(np.mean([row[name] for row in valid_localization])) if valid_localization else None
            for name in ("dice", "auroc", "auprc")
        },
        "evidence_valid_cases": len(valid_localization),
        "evidence_invalid_cases": invalid_evidence_cases,
        "cases": cases,
    }


def main():
    args = parse_args()
    if args.w_quality != 0:
        raise ValueError("Quality audit is evaluation-only; use --w_quality 0")
    seed_everything(args.seed)
    device = torch.device(args.device)
    predictor = build_predictor(args.model_dir, device, args.voxtell_root)
    with torch.no_grad():
        initial_prompt = predictor.embed_text_prompts([args.prompt]).detach()
        text_anchor = predictor.embed_text_prompts(["liver"]).detach()
    output = Path(args.output)
    adapter_args = argparse.Namespace(
        **vars(args),
        output_dir=str(output.parent),
        lr=0.0,
        weight_decay=0.0,
        amp_init_scale=1024.0,
        record_soft_prompt_grad_norm=False,
    )
    adapter = VoxTellPromptSFDA(
        predictor.network,
        initial_prompt,
        device,
        adapter_args,
        text_anchor=text_anchor,
    )
    adapter.predictor = predictor
    train_loader = make_target_loader(args.data_dir, tuple(predictor.patch_size), args.batch_size, args.num_workers)
    if args.quality_metric != "saaf" and args.quality_mode != "cac":
        adapter.build_prototype_memory(train_loader)

    cases = []
    mixed_views = []
    selected_dice = {name: [] for name in SELECTION_NAMES}
    patch_selected_dice = {name: [] for name in SELECTION_NAMES}
    patch_best_dice = []
    patch_oracle_gap = {name: [] for name in SELECTION_NAMES}
    best_fixed_view_dice = []
    patch_spearman_case = {name: [] for name in QUALITY_NAMES}
    invalid_evidence_cases = []
    try:
        test_entries = read_image_entries(args.data_dir, "test")
        for case_index, (image_path, label_path) in enumerate(test_entries):
            volume, target = load_preprocessed_labeled_case(image_path, label_path)
            target = _align_target_to_volume(target, volume)
            inference = _voxtell_sliding_window_views(
                adapter,
                volume,
                args.num_aug_views,
                _case_seed(args.seed, image_path.name),
                case_id=image_path.name,
                target=target,
            )
            probability, evidence, metrics = (
                inference["probability"], inference["evidence"], inference["full_metrics"]
            )
            dice = _dice_per_view(probability, target.float())
            fixed_index, fixed_dice = int(dice.argmax()), float(dice.max())
            best_fixed_view_dice.append(fixed_dice)
            per_view = []
            for view_index in range(args.num_aug_views):
                row = {
                    "case": image_path.name,
                    "view": view_index,
                    "dice": float(dice[view_index]),
                    **{name: float(values[view_index]) for name, values in metrics.items()},
                }
                per_view.append(row)
                mixed_views.append(row)
            correlations = {
                name: spearman([row[name] for row in per_view], [row["dice"] for row in per_view])
                for name in QUALITY_NAMES
            }
            patch_spearman = {}
            for name in QUALITY_NAMES:
                values = [
                    spearman(patch["quality"][name], patch["candidate_dice"])
                    for patch in inference["patch_records"]
                    if "candidate_dice" in patch
                ]
                patch_spearman[name] = values
                valid_patch_correlations = [v for v in values if v is not None]
                patch_spearman_case[name].append(
                    float(np.mean(valid_patch_correlations)) if valid_patch_correlations else None
                )
            for patch in inference["patch_records"]:
                if "candidate_dice" in patch:
                    patch["spearman"] = {
                        name: spearman(patch["quality"][name], patch["candidate_dice"])
                        for name in QUALITY_NAMES
                    }
            selections = {}
            patch_best_dice.extend(
                patch["best_dice"] for patch in inference["patch_records"] if "best_dice" in patch
            )
            for name, selected in inference["selected_dice"].items():
                selected_dice[name].append(selected)
                patch_values = [
                    patch["selection"][name]["selected_dice"]
                    for patch in inference["patch_records"] if "selection" in patch
                ]
                patch_best_values = [
                    patch["selection"][name]["patch_best_dice"]
                    for patch in inference["patch_records"] if "selection" in patch
                ]
                patch_gap_values = [
                    max(0.0, patch["selection"][name]["gap_to_patch_oracle"])
                    for patch in inference["patch_records"] if "selection" in patch
                ]
                patch_selected_dice[name].extend(patch_values)
                patch_oracle_gap[name].extend(patch_gap_values)
                selections[name] = {
                    "view": "patchwise",
                    "full_volume_dice": selected,
                    "patch_selected_dice": float(np.mean(patch_values)) if patch_values else None,
                    "patch_best_dice": float(np.mean(patch_best_values)) if patch_best_values else None,
                    "patch_gap_to_oracle": float(np.mean(patch_gap_values)) if patch_gap_values else None,
                    "valid_patches": len(patch_values),
                }
            localization = evidence_localization_metrics(
                evidence, target.float(), adapter.quality_config["evidence_threshold"]
            )
            if not localization["valid"]:
                invalid_evidence_cases.append({"case": image_path.name, "reason": localization["reason"]})
            cases.append(
                {
                    "case": image_path.name,
                    "views": per_view,
                    "within_case_spearman": correlations,
                    "patch_spearman": {
                        name: {
                            "mean": float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else None,
                            "valid_patches": sum(v is not None for v in values),
                        }
                        for name, values in patch_spearman.items()
                    },
                    "best_fixed_view": fixed_index,
                    "best_fixed_view_dice": fixed_dice,
                    "patch_count": len(inference["patch_records"]),
                    "patches": inference["patch_records"],
                    "selections": selections,
                    "evidence_localization_original_view": localization,
                    "evidence_statistics_original_view": _tensor_statistics(evidence),
                    "prototype_valid_fraction": inference["prototype_valid_fraction"],
                    "sliding_window_patches": len(inference["locations"]),
                }
            )
            if case_index < args.visualize_cases:
                save_visualization(
                    output.parent / "quality_visualizations" / f"{image_path.stem}.png",
                    volume,
                    target,
                    probability[0] >= 0.5,
                    evidence,
                )
    finally:
        adapter.close()

    result = summarize_quality_audit(
        cases,
        mixed_views,
        selected_dice,
        patch_selected_dice,
        patch_best_dice,
        patch_oracle_gap,
        best_fixed_view_dice,
        patch_spearman_case,
        invalid_evidence_cases,
        args,
        adapter.prototype_diagnostics,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
