"""Full-volume, GT-only audit of VoxTell CAC and target semantic evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.sfda_voxtell import (
    extract_volume_patch,
    load_preprocessed_labeled_case,
    make_target_loader,
    pad_to_patch_grid,
    read_image_entries,
    sliding_window_locations,
)
from method.semantic_quality import (
    average_tie_ranks,
    compute_tse_components,
    text_similarity_map,
)
from method.sfda_voxtell import VoxTellPromptSFDA
from run_sfda_voxtell import build_predictor, seed_everything


QUALITY_NAMES = (
    "confidence",
    "entropy",
    "consistency",
    "cac",
    "purity",
    "completeness",
    "tse",
)
SELECTION_NAMES = ("cac", "tse", "purity", "completeness", "cac_entropy", "tse_entropy")


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


def _add_spatial_patch(accumulator, patch, location):
    patch_size = patch.shape[-3:]
    slices = tuple(slice(int(start), int(start) + int(size)) for start, size in zip(location, patch_size))
    accumulator[(..., *slices)] += patch


def infer_full_volume_views(adapter, volume, patch_size, num_views, seed, case_id=None):
    """Sliding-window inference and coverage-normalized full-volume fusion."""
    padded, _, original_shape = pad_to_patch_grid(volume, patch_size)
    padded_shape = tuple(int(value) for value in padded.shape[-3:])
    locations = sliding_window_locations(
        padded_shape, patch_size, float(adapter.quality_config["evaluation_overlap"])
    )
    patch_batch_size = int(adapter.quality_config["evaluation_patch_batch_size"])
    parameters = deterministic_view_parameters(num_views, seed)
    accumulators = {
        name: torch.zeros((num_views, *padded_shape), dtype=torch.float32)
        for name in ("probability", "evidence", "text_similarity")
    }
    coverage = torch.zeros(padded_shape, dtype=torch.float32)
    prototype_valid = 0
    prototype_queries = 0
    for start in range(0, len(locations), patch_batch_size):
        batch_locations = locations[start : start + patch_batch_size]
        patches = torch.stack([extract_volume_patch(padded, location, patch_size) for location in batch_locations]).to(
            adapter.device, non_blocking=True
        )
        views = torch.stack([patches * scale + offset for scale, offset in parameters], dim=1)
        flat_views = views.reshape(patches.shape[0] * num_views, *patches.shape[1:])
        adapter._cac_features.clear()
        with torch.no_grad(), torch.autocast(
            device_type=adapter.device.type, enabled=adapter.device.type == "cuda"
        ):
            prompt = adapter._text(adapter.initial_soft_prompt, flat_views.shape[0])
            logits = adapter.model(flat_views, prompt)
        query_case_id = "validation-case" if case_id is None else str(case_id)
        semantic = adapter._semantic_quality(
            adapter._cac_features["vision"], logits, [query_case_id] * flat_views.shape[0]
        )
        probability = _resize_patch_map(torch.sigmoid(logits[:, 0].float()), patch_size)
        evidence = _resize_patch_map(semantic["evidence"].permute(0, 3, 1, 2), patch_size)
        similarity = _resize_patch_map(
            text_similarity_map(adapter._cac_features["vision"], adapter._cac_features["text"]).permute(0, 3, 1, 2),
            patch_size,
        )
        maps = {
            "probability": probability.view(patches.shape[0], num_views, *patch_size),
            "evidence": evidence.view(patches.shape[0], num_views, *patch_size),
            "text_similarity": similarity.view(patches.shape[0], num_views, *patch_size),
        }
        prototype_valid += int(semantic["prototype_valid"].sum().cpu())
        prototype_queries += int(semantic["prototype_valid"].numel())
        for patch_index, location in enumerate(batch_locations):
            for name in accumulators:
                _add_spatial_patch(accumulators[name], maps[name][patch_index].detach().cpu(), location)
            patch_slices = tuple(
                slice(int(position), int(position) + int(size))
                for position, size in zip(location, patch_size)
            )
            coverage[patch_slices] += 1
    if not bool((coverage > 0).all()):
        raise RuntimeError("Sliding-window inference left uncovered voxels")
    crop = tuple(slice(0, size) for size in original_shape)
    fused = {
        name: (values / coverage.clamp_min(1))[(..., *crop)]
        for name, values in accumulators.items()
    }
    fused["prototype_valid_fraction"] = prototype_valid / prototype_queries if prototype_queries else 0.0
    fused["locations"] = locations
    return fused


def _dice_per_view(probability, target):
    prediction = probability >= 0.5
    target = target.bool().expand_as(prediction)
    intersection = (prediction & target).flatten(start_dim=1).sum(dim=1).float()
    denominator = (
        prediction.flatten(start_dim=1).sum(dim=1) + target.flatten(start_dim=1).sum(dim=1)
    ).float()
    return torch.where(denominator > 0, 2 * intersection / denominator, torch.ones_like(denominator))


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
    result = {name: int(metrics[name].argmax()) for name in ("cac", "tse", "purity", "completeness")}
    entropy_rank = average_tie_ranks(metrics["entropy"].view(1, -1), descending=False)
    for quality in ("cac", "tse"):
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
    parser.add_argument("--quality_mode", choices=("tse",), default="tse")
    parser.add_argument("--w_quality", type=float, default=0.0)
    parser.add_argument("--num_aug_views", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--visualize_cases", type=int, default=5)
    parser.add_argument("--output", default="results/voxtell_sfda/quality_audit.json")
    parser.add_argument("--seed", type=int, default=1377)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quality_mode != "tse" or args.w_quality != 0:
        raise ValueError("Quality audit requires --quality_mode tse --w_quality 0")
    seed_everything(args.seed)
    device = torch.device(args.device)
    predictor = build_predictor(args.model_dir, device, args.voxtell_root)
    with torch.no_grad():
        initial_prompt = predictor.embed_text_prompts([args.prompt]).detach()
    output = Path(args.output)
    adapter_args = argparse.Namespace(
        **vars(args),
        output_dir=str(output.parent),
        lr=0.0,
        weight_decay=0.0,
        amp_init_scale=1024.0,
        record_soft_prompt_grad_norm=False,
    )
    adapter = VoxTellPromptSFDA(predictor.network, initial_prompt, device, adapter_args)
    train_loader = make_target_loader(args.data_dir, tuple(predictor.patch_size), args.batch_size, args.num_workers)
    adapter.build_prototype_memory(train_loader)

    cases = []
    mixed_views = []
    selected_dice = {name: [] for name in SELECTION_NAMES}
    invalid_evidence_cases = []
    try:
        test_entries = read_image_entries(args.data_dir, "test")
        for case_index, (image_path, label_path) in enumerate(test_entries):
            volume, target = load_preprocessed_labeled_case(image_path, label_path)
            inference = infer_full_volume_views(
                adapter,
                volume,
                tuple(int(value) for value in predictor.patch_size),
                args.num_aug_views,
                _case_seed(args.seed, image_path.name),
                case_id=image_path.name,
            )
            probability, evidence, similarity = (
                inference["probability"], inference["evidence"], inference["text_similarity"]
            )
            purity, completeness, tse = compute_tse_components(
                probability, evidence, adapter.quality_config["epsilon"]
            )
            clipped = probability.clamp(1e-6, 1 - 1e-6)
            metrics = {
                "confidence": torch.maximum(probability, 1 - probability).flatten(start_dim=1).mean(dim=1),
                "entropy": -(
                    clipped * clipped.log() + (1 - clipped) * (1 - clipped).log()
                ).flatten(start_dim=1).mean(dim=1),
                "consistency": _soft_consistency(probability),
                "cac": _cac_from_full_maps(probability, similarity),
                "purity": purity,
                "completeness": completeness,
                "tse": tse,
            }
            selection_indices = _selection_indices(metrics)
            dice = _dice_per_view(probability, target.float())
            oracle_index, oracle_dice = int(dice.argmax()), float(dice.max())
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
            selections = {}
            for name, selected_index in selection_indices.items():
                selected = float(dice[selected_index])
                selected_dice[name].append(selected)
                selections[name] = {
                    "view": selected_index,
                    "selected_dice": selected,
                    "oracle_best_view": oracle_index,
                    "oracle_best_dice": oracle_dice,
                    "gap_to_oracle": oracle_dice - selected,
                }
            localization = evidence_localization_metrics(
                evidence[0], target.float(), adapter.quality_config["evidence_threshold"]
            )
            if not localization["valid"]:
                invalid_evidence_cases.append({"case": image_path.name, "reason": localization["reason"]})
            cases.append(
                {
                    "case": image_path.name,
                    "views": per_view,
                    "within_case_spearman": correlations,
                    "oracle_best_view": oracle_index,
                    "oracle_best_dice": oracle_dice,
                    "selections": selections,
                    "evidence_localization_original_view": localization,
                    "evidence_statistics_original_view": _tensor_statistics(evidence[0]),
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
                    evidence[0],
                )
    finally:
        adapter.close()

    macro_spearman = {}
    macro_valid_cases = {}
    for name in QUALITY_NAMES:
        valid = [case["within_case_spearman"][name] for case in cases if case["within_case_spearman"][name] is not None]
        macro_spearman[name] = float(np.mean(valid)) if valid else None
        macro_valid_cases[name] = len(valid)
    global_spearman = {
        name: spearman([row[name] for row in mixed_views], [row["dice"] for row in mixed_views])
        for name in QUALITY_NAMES
    }
    valid_localization = [
        case["evidence_localization_original_view"]
        for case in cases
        if case["evidence_localization_original_view"]["valid"]
    ]
    result = {
        "protocol": {
            "quality_mode": args.quality_mode,
            "w_quality": args.w_quality,
            "gt_usage": "offline metrics/visualization only",
            "views": "identical deterministic intensity views for every metric",
            "inference": "complete preprocessed volume with coverage-normalized sliding-window fusion",
        },
        "prototype_diagnostics": adapter.prototype_diagnostics,
        "within_case_spearman_macro": macro_spearman,
        "within_case_spearman_valid_cases": macro_valid_cases,
        "global_mixed_view_spearman": global_spearman,
        "selected_view_mean_dice": {
            name: float(np.mean(values)) if values else None for name, values in selected_dice.items()
        },
        "evidence_localization_macro": {
            name: float(np.mean([row[name] for row in valid_localization])) if valid_localization else None
            for name in ("dice", "auroc", "auprc")
        },
        "evidence_valid_cases": len(valid_localization),
        "evidence_invalid_cases": invalid_evidence_cases,
        "cases": cases,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
