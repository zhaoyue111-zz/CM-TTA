"""Full-volume, GT-only audit of VoxTell CAC, TRA and semantic evidence."""

from __future__ import annotations

import argparse
import csv
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
from method.sfda_voxtell import VoxTellPromptSFDA, load_sfda_checkpoint
from run_sfda_voxtell import binary_segmentation_metrics, build_predictor, seed_everything


QUALITY_NAMES = (
    "confidence", "entropy", "consistency", "cac", "purity", "coverage", "saaf",
    "completeness", "tse",
)
TRA_NAMES = ("tra_teacher", "tra_original")
TRA_GT_NAMES = ("dice", "recall", "precision")
TRA_COMPARISON_NAMES = ("cac", *TRA_NAMES)
SELECTION_NAMES = ("cac", "saaf", "purity", "coverage", "cac_entropy", "saaf_entropy")
STATISTICS_MAX_QUANTILE_SAMPLES = 1_000_000


def _fuse_logits_then_sigmoid(logits, denominator):
    """Fuse Gaussian-weighted logits first, then apply sigmoid exactly once."""
    denominator = denominator.clamp_min(torch.finfo(logits.dtype).eps)
    fused_logits = logits / denominator
    return fused_logits, torch.sigmoid(fused_logits)


def _fuse_evidence_with_valid_weights(evidence_sum, evidence_weight_sum):
    """Fuse only evidence from valid patch/views and report spatial coverage."""
    evidence_sum = np.asarray(evidence_sum, dtype=np.float32)
    evidence_weight_sum = np.asarray(evidence_weight_sum, dtype=np.float32)
    covered = evidence_weight_sum > 0
    fused = evidence_sum / np.maximum(evidence_weight_sum, np.finfo(np.float32).eps)
    return fused, covered


def _full_volume_selection_valid(selected_weight_sum, valid_patch_count):
    """A selected full-volume map is valid only when every voxel is covered."""
    weight = np.asarray(selected_weight_sum, dtype=np.float32)
    return bool(valid_patch_count > 0 and np.isfinite(weight).all() and (weight > 0).all())


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


def compute_tra_score(similarity_map, probability_map):
    """Spearman TRA between a prompt-similarity map and final foreground probability.

    Constant maps and maps with fewer than two finite paired voxels are
    undefined and return ``None``. Ground truth is deliberately not an input.
    """
    similarity_map = torch.as_tensor(similarity_map).detach().float().cpu()
    probability_map = torch.as_tensor(probability_map).detach().float().cpu()
    if tuple(similarity_map.shape) != tuple(probability_map.shape):
        raise ValueError(
            "TRA similarity/probability maps must have equal shapes; got "
            f"{tuple(similarity_map.shape)} and {tuple(probability_map.shape)}"
        )
    return spearman(similarity_map.reshape(-1).numpy(), probability_map.reshape(-1).numpy())


@torch.no_grad()
def _project_prompt_features(adapter, prompt, batch_size):
    """Use VoxTell's native frozen text projection on (tokens,batch,channels)."""
    network = getattr(adapter.model, "_orig_mod", adapter.model)
    network = getattr(network, "module", network)
    if not hasattr(network, "project_text_embed"):
        raise AttributeError("VoxTell network is missing project_text_embed")
    text = adapter._text(prompt.detach(), batch_size).squeeze(2)
    text = text.permute(1, 0, 2).contiguous()
    with torch.autocast(
        device_type=adapter.device.type,
        enabled=adapter.device.type == "cuda",
    ):
        projected = network.project_text_embed(text)
    if not torch.is_tensor(projected) or projected.ndim != 3:
        shape = getattr(projected, "shape", None)
        raise ValueError(f"Expected projected prompt features (N,B,C), got {shape}")
    if projected.shape[1] != batch_size:
        raise ValueError(
            "Projected prompt batch does not match view batch: "
            f"{projected.shape[1]} vs {batch_size}"
        )
    return projected.detach()


def summarize_tra_cac(view_rows, checkpoint_loaded):
    """Compare CAC/TRA against full-volume Dice, recall and precision."""
    if not checkpoint_loaded:
        raise ValueError("TRA/CAC comparison requires an adapted CM-SFDA checkpoint")
    cases = {}
    for row in view_rows:
        cases.setdefault(row["case"], []).append(row)

    within_case = {}
    within_valid_cases = {}
    within_valid_views = {}
    global_spearman = {}
    global_valid_views = {}
    selected_values = {
        name: {metric: [] for metric in TRA_GT_NAMES}
        for name in TRA_COMPARISON_NAMES
    }
    selected_valid_cases = {name: 0 for name in TRA_COMPARISON_NAMES}
    for score_name in TRA_COMPARISON_NAMES:
        within_case[score_name] = {}
        within_valid_cases[score_name] = {}
        within_valid_views[score_name] = {}
        global_spearman[score_name] = {}
        global_valid_views[score_name] = {}
        for gt_name in TRA_GT_NAMES:
            case_correlations = []
            n_within_views = 0
            for rows in cases.values():
                valid_rows = [
                    row for row in rows
                    if row.get(f"{score_name}_valid", False)
                    and row.get(score_name) is not None
                    and np.isfinite(row[score_name])
                    and np.isfinite(row[gt_name])
                ]
                n_within_views += len(valid_rows)
                correlation = spearman(
                    [row[score_name] for row in valid_rows],
                    [row[gt_name] for row in valid_rows],
                )
                if correlation is not None:
                    case_correlations.append(correlation)
            within_case[score_name][gt_name] = (
                float(np.mean(case_correlations)) if case_correlations else None
            )
            within_valid_cases[score_name][gt_name] = len(case_correlations)
            within_valid_views[score_name][gt_name] = n_within_views

            valid_rows = [
                row for row in view_rows
                if row.get(f"{score_name}_valid", False)
                and row.get(score_name) is not None
                and np.isfinite(row[score_name])
                and np.isfinite(row[gt_name])
            ]
            global_spearman[score_name][gt_name] = spearman(
                [row[score_name] for row in valid_rows],
                [row[gt_name] for row in valid_rows],
            )
            global_valid_views[score_name][gt_name] = len(valid_rows)

            selected_rows = [row for row in view_rows if row.get(f"selected_{score_name}", False)]
            selected_values[score_name][gt_name].extend(
                float(row[gt_name]) for row in selected_rows if np.isfinite(row[gt_name])
            )
        selected_valid_cases[score_name] = len(
            [row for row in view_rows if row.get(f"selected_{score_name}", False)]
        )

    return {
        "protocol": {
            "probability_prompt": "checkpoint soft_prompt_embedding (standard adapted inference)",
            "tra_teacher_prompt": "detached checkpoint EMA teacher_soft_prompt",
            "tra_original_prompt": "checkpoint initial_soft_prompt",
            "similarity": "voxelwise cosine between frozen project_bottleneck_embed output and native project_text_embed output",
            "correlation": "full-volume finite-pair Spearman(similarity, final foreground probability); constant maps are invalid",
            "view_selection": "per-case argmax score over valid views; CAC uses finite scores, TRA uses finite defined correlations",
            "gt_usage": "GT from train/test splits is accessed only in this offline audit for Dice/recall/precision evaluation and correlations; neither prompt maps, predictions nor selection use GT",
            "checkpoint_loaded": True,
        },
        "within_case_spearman_macro": within_case,
        "within_case_valid_cases": within_valid_cases,
        "within_case_valid_views": within_valid_views,
        "global_mixed_view_spearman": global_spearman,
        "global_mixed_view_valid_views": global_valid_views,
        "selected_view_mean_gt_metrics": {
            name: {
                metric: float(np.mean(values)) if values else None
                for metric, values in metrics.items()
            }
            for name, metrics in selected_values.items()
        },
        "selected_view_valid_cases": selected_valid_cases,
        "view_rows": len(view_rows),
        "cases": len(cases),
        "split_case_counts": {
            split: len({row["case"] for row in view_rows if row.get("split") == split})
            for split in ("train", "test")
        },
    }


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


def _training_style_view_specs(num_views, seed):
    """Sample one reproducible set of train-style transforms per case/view."""
    num_views = int(num_views)
    if num_views < 1:
        raise ValueError("num_views must be at least one")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) % (2**63 - 1))
    specs = [(1.0, 0.0, None)]
    for _ in range(1, num_views):
        scale = 1.0
        if bool(torch.rand((), generator=generator) < 0.8):
            scale = float(torch.empty(()).uniform_(0.85, 1.15, generator=generator))
        offset = 0.0
        if bool(torch.rand((), generator=generator) < 0.8):
            offset = float(torch.empty(()).uniform_(-0.15, 0.15, generator=generator))
        noise_seed = None
        if bool(torch.rand((), generator=generator) < 0.5):
            noise_seed = int(torch.randint(0, 2**31 - 1, (), generator=generator))
        specs.append((scale, offset, noise_seed))
    return specs


def _write_case_global_views(volume, num_views, seed, destination):
    """Materialize case-wide views once, so every patch sees the same transform.

    ``destination`` is normally a disk-backed array with shape
    ``(num_views, C, D, H, W)``. Noise is sampled over the full preprocessed
    volume and therefore remains spatially aligned across overlapping patches.
    """
    base = volume.detach().to(device="cpu", dtype=torch.float32).contiguous()
    expected_shape = (int(num_views), *tuple(base.shape))
    if tuple(destination.shape) != expected_shape:
        raise ValueError(f"Expected case-view storage {expected_shape}, got {destination.shape}")
    specs = _training_style_view_specs(num_views, seed)
    for view_index, (scale, offset, noise_seed) in enumerate(specs):
        view = base * scale + offset
        if noise_seed is not None:
            noise_generator = torch.Generator(device="cpu")
            noise_generator.manual_seed(noise_seed)
            view = view + torch.randn(view.shape, generator=noise_generator) * 0.05
        destination[view_index] = view.contiguous().numpy()
    if hasattr(destination, "flush"):
        destination.flush()
    return specs


def _crop_case_global_views(case_views, slicer):
    """Crop all case-wide views at one predictor sliding-window location."""
    patch_views = np.array(case_views[(slice(None), *slicer)], dtype=np.float32, copy=True)
    return torch.from_numpy(patch_views)


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
    valid_patch_counts = np.zeros(num_views, dtype=np.int64)
    patch_count = len(slicers)
    with tempfile.TemporaryDirectory(prefix="voxtell_quality_") as cache_dir:
        def mmap(name, shape):
            return np.memmap(Path(cache_dir) / f"{name}.bin", mode="w+", dtype=np.float32, shape=shape)

        logit_sum = mmap("logits", (num_views, *padded_shape))
        case_views = mmap("case_views", (num_views, *tuple(padded.shape)))
        _write_case_global_views(padded, num_views, seed, case_views)
        evidence_sum = mmap("evidence", (num_views, *padded_shape))
        evidence_weight_sum = mmap("evidence_weight", (num_views, *padded_shape))
        similarity_sum = mmap("similarity", (num_views, *padded_shape))
        tra_teacher_similarity_sum = mmap("tra_teacher_similarity", (num_views, *padded_shape))
        tra_original_similarity_sum = mmap("tra_original_similarity", (num_views, *padded_shape))
        selected_logit_sum = {name: mmap(f"selected_{name}", padded_shape) for name in selected_names}
        selected_weight_sum = {name: mmap(f"selected_weight_{name}", padded_shape) for name in selected_names}
        selection_skipped = {name: False for name in selected_names}
        selection_valid_patch_counts = {name: 0 for name in selected_names}

        for patch_index, slicer in enumerate(slicers):
            patch = padded[slicer].to(adapter.device, non_blocking=True)
            valid_patch = padded_valid[slicer].to(adapter.device, non_blocking=True)
            views = _crop_case_global_views(case_views, slicer).to(
                adapter.device, non_blocking=True
            )
            valid_views = valid_patch.expand(num_views, -1, -1, -1, -1)
            adapter._cac_features.clear()
            with torch.no_grad(), torch.autocast(
                device_type=adapter.device.type, enabled=adapter.device.type == "cuda"
            ):
                # The final probability map follows standard VoxTell SFDA
                # inference and uses the adapted student prompt from checkpoint.
                prompt = adapter._text(adapter.soft_prompt_embedding.detach(), num_views)
                try:
                    model_output = adapter.model(views, prompt, return_diagnostics=True)
                except TypeError as error:
                    raise RuntimeError(
                        "SAAF audit requires VoxTell forward(return_diagnostics=True)"
                    ) from error
                if (
                    isinstance(model_output, tuple)
                    and len(model_output) == 2
                    and isinstance(model_output[1], dict)
                ):
                    logits, model_diagnostics = model_output
                    native_cross = model_diagnostics.get("cross_attention")
                    if native_cross is None:
                        raise RuntimeError("VoxTell diagnostics do not contain cross_attention")
                else:
                    raise RuntimeError(
                        "VoxTell diagnostics must return (logits, diagnostics)"
                    )
            flat_case_ids = [case_key] * num_views
            vision_features = adapter._cac_features["vision"]
            current_text_features = adapter._cac_features["text"]
            _quality, cac, semantic = adapter._quality_scores(
                vision_features,
                adapter._cac_features["text"],
                logits,
                flat_case_ids,
                native_cross_attention=native_cross,
                spatial_valid=valid_views,
            )
            # SAAF evidence is audited for both modes.  Legacy purity/TSE
            # details are intentionally not reused: they are prototype-based;
            # this audit always measures the same forward's prompt response.
            if adapter.quality_metric == "saaf":
                saaf_details = semantic
            else:
                saaf_details = adapter._saaf_quality(
                    vision_features,
                    logits,
                    native_cross,
                    valid_views,
                )
            if saaf_details is None:
                raise RuntimeError("SAAF audit did not produce final-layer evidence details")
            semantic = saaf_details
            probability = torch.sigmoid(logits[:, 0].float())
            evidence = semantic["evidence"]
            similarity = text_similarity_map(
                vision_features, current_text_features
            )
            teacher_text_features = _project_prompt_features(
                adapter, adapter.teacher_soft_prompt.detach(), num_views
            )
            original_text_features = _project_prompt_features(
                adapter, adapter.initial_soft_prompt.detach(), num_views
            )
            tra_teacher_similarity = text_similarity_map(
                vision_features, teacher_text_features
            )
            tra_original_similarity = text_similarity_map(
                vision_features, original_text_features
            )
            evidence = _resize_patch_map(evidence.permute(0, 3, 1, 2), tuple(predictor.patch_size))
            similarity = _resize_patch_map(similarity.permute(0, 3, 1, 2), tuple(predictor.patch_size))
            tra_teacher_similarity = _resize_patch_map(
                tra_teacher_similarity.permute(0, 3, 1, 2), tuple(predictor.patch_size)
            )
            tra_original_similarity = _resize_patch_map(
                tra_original_similarity.permute(0, 3, 1, 2), tuple(predictor.patch_size)
            )
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
            selected_indices, patch_skipped = _selection_indices(
                patch_metrics, semantic["valid"]
            )
            for name, skipped in patch_skipped.items():
                if not skipped:
                    selection_valid_patch_counts[name] += 1
            patch_slices = slicer[1:]
            weight = gaussian.detach().cpu()
            logits_cpu = logits[:, 0].float().detach().cpu()
            evidence_cpu = evidence.detach().cpu()
            similarity_cpu = similarity.detach().cpu()
            tra_teacher_similarity_cpu = tra_teacher_similarity.detach().cpu()
            tra_original_similarity_cpu = tra_original_similarity.detach().cpu()
            # Keep the original torch float32 accumulation order while using
            # writable memmap-backed storage instead of resident tensors.
            torch.from_numpy(logit_sum[(..., *patch_slices)]).add_(logits_cpu * weight)
            semantic_valid = semantic["valid"].detach().cpu().reshape(-1).bool()
            for view_index in range(num_views):
                if bool(semantic_valid[view_index]):
                    valid_patch_counts[view_index] += 1
                    torch.from_numpy(evidence_sum[(view_index, *patch_slices)]).add_(
                        evidence_cpu[view_index] * weight
                    )
                    torch.from_numpy(evidence_weight_sum[(view_index, *patch_slices)]).add_(weight)
            torch.from_numpy(similarity_sum[(..., *patch_slices)]).add_(similarity_cpu * weight)
            torch.from_numpy(tra_teacher_similarity_sum[(..., *patch_slices)]).add_(
                tra_teacher_similarity_cpu * weight
            )
            torch.from_numpy(tra_original_similarity_sum[(..., *patch_slices)]).add_(
                tra_original_similarity_cpu * weight
            )
            torch.from_numpy(gaussian_sum[patch_slices]).add_(weight)
            for name in selected_names:
                if patch_skipped[name]:
                    continue
                with torch.no_grad():
                    torch.from_numpy(selected_logit_sum[name][patch_slices]).add_(
                        logits_cpu[selected_indices[name]] * weight
                    )
                    torch.from_numpy(selected_weight_sum[name][patch_slices]).add_(weight)
            patch_record = {
                "patch_index": patch_index,
                "location": [[int(s.start), int(s.stop)] for s in patch_slices],
                "quality": {name: [float(v) for v in patch_metrics[name].view(-1).detach().cpu()] for name in QUALITY_NAMES},
                "selected_view": selected_indices,
                "selection_skipped": patch_skipped,
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
                        "selected_by": [
                            name for name, index in selected_indices.items()
                            if index == view_index
                        ],
                        "selected": view_index == selected_indices.get(
                            "saaf" if adapter.quality_metric == "saaf" else "cac"
                        ),
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
        evidence_original_np, _ = _fuse_evidence_with_valid_weights(
            evidence_sum[(0, *crop)], evidence_weight_sum[(0, *crop)]
        )
        evidence_original = torch.from_numpy(evidence_original_np.copy())
        full_metrics = {name: [] for name in QUALITY_NAMES}
        full_metrics.update({name: [] for name in TRA_NAMES})
        full_consistency = _soft_consistency(probability)
        full_view_valid = torch.zeros(num_views, dtype=torch.bool)
        view_invalid_reasons = [None for _ in range(num_views)]
        evidence_coverage = np.zeros(num_views, dtype=np.float32)
        for view_index in range(num_views):
            evidence_view_np, evidence_covered = _fuse_evidence_with_valid_weights(
                evidence_sum[(view_index, *crop)], evidence_weight_sum[(view_index, *crop)]
            )
            evidence_coverage[view_index] = float(evidence_covered.mean())
            evidence_view = torch.from_numpy(evidence_view_np.copy())
            similarity_view = torch.from_numpy(
                (np.asarray(similarity_sum[(view_index, *crop)], dtype=np.float32) / denominator_np).copy()
            )
            tra_teacher_similarity_view = torch.from_numpy(
                (
                    np.asarray(tra_teacher_similarity_sum[(view_index, *crop)], dtype=np.float32)
                    / denominator_np
                ).copy()
            )
            tra_original_similarity_view = torch.from_numpy(
                (
                    np.asarray(tra_original_similarity_sum[(view_index, *crop)], dtype=np.float32)
                    / denominator_np
                ).copy()
            )
            view_probability = probability[view_index:view_index + 1]
            clipped = view_probability.clamp(1e-6, 1 - 1e-6)
            saaf = compute_saaf_quality(
                view_probability,
                evidence_view[None],
                epsilon=adapter.quality_config["epsilon"],
                min_mass=adapter.quality_config.get("saaf_min_mass", 1e-6),
            )
            if not bool(evidence_covered.all()):
                view_invalid_reasons[view_index] = "incomplete_evidence_spatial_coverage"
            elif not bool(saaf["valid"][0]):
                view_invalid_reasons[view_index] = "saaf_quality_invalid"
            else:
                full_view_valid[view_index] = True
            full_metrics["confidence"].append(float(torch.maximum(view_probability, 1 - view_probability).mean()))
            full_metrics["entropy"].append(float(-(clipped * clipped.log() + (1 - clipped) * (1 - clipped).log()).mean()))
            full_metrics["consistency"].append(float(full_consistency[view_index]))
            full_metrics["cac"].append(float(_cac_from_full_maps(view_probability, similarity_view[None])[0]))
            full_metrics["purity"].append(float(saaf["purity"][0]))
            full_metrics["coverage"].append(float(saaf["coverage"][0]))
            full_metrics["saaf"].append(float(saaf["saaf"][0]))
            full_metrics["completeness"].append(float(saaf["coverage"][0]))
            full_metrics["tse"].append(float(saaf["saaf"][0]))
            tra_teacher = compute_tra_score(
                tra_teacher_similarity_view, view_probability[0]
            )
            tra_original = compute_tra_score(
                tra_original_similarity_view, view_probability[0]
            )
            full_metrics["tra_teacher"].append(
                float(tra_teacher) if tra_teacher is not None else float("nan")
            )
            full_metrics["tra_original"].append(
                float(tra_original) if tra_original is not None else float("nan")
            )
        selected_dice = {}
        selection_diagnostics = {}
        for name, value in selected_logit_sum.items():
            selection_skipped[name] = selection_valid_patch_counts[name] == 0
            selected_weight = np.asarray(selected_weight_sum[name][crop], dtype=np.float32)
            selected_covered = selected_weight > 0
            selected_coverage = float(selected_covered.mean())
            method_valid = bool(
                (not selection_skipped[name])
                and _full_volume_selection_valid(selected_weight, selection_valid_patch_counts[name])
            )
            method_reason = None
            if selection_skipped[name]:
                method_reason = "no_valid_selected_patch"
            elif not bool(selected_covered.all()):
                method_reason = "incomplete_selected_spatial_coverage"
            method_denominator = np.maximum(
                selected_weight,
                np.finfo(np.float32).eps,
            )
            selected_np = (np.asarray(value[crop], dtype=np.float32) / method_denominator).copy()
            if target is not None and method_valid:
                selected_probability = torch.sigmoid(torch.from_numpy(selected_np))
                selected_dice[name] = float(
                    _dice_per_view(selected_probability[None], target.float())[0]
                )
            else:
                selected_dice[name] = None
            selection_diagnostics[name] = {
                "valid_patch_count": int(selection_valid_patch_counts[name]),
                "valid_patch_fraction": float(selection_valid_patch_counts[name] / max(1, patch_count)),
                "spatial_coverage_fraction": selected_coverage,
                "full_volume_valid": method_valid,
                "invalid_reason": method_reason,
            }
        view_diagnostics = {}
        for view_index in range(num_views):
            view_diagnostics[str(view_index)] = {
                "valid_patch_count": int(valid_patch_counts[view_index]),
                "valid_patch_fraction": float(valid_patch_counts[view_index] / max(1, patch_count)),
                "spatial_coverage_fraction": float(evidence_coverage[view_index]),
                "full_view_valid": bool(full_view_valid[view_index]),
                "invalid_reason": view_invalid_reasons[view_index],
            }
        return {
            "probability": probability,
            "evidence": evidence_original,
            "full_metrics": {name: torch.tensor(values, dtype=torch.float32) for name, values in full_metrics.items()},
            "selected_dice": selected_dice,
            "selection_skipped": selection_skipped,
            "selection_valid_patch_counts": selection_valid_patch_counts,
            "selection_diagnostics": selection_diagnostics,
            "patch_records": patch_records,
            "prototype_valid_fraction": prototype_valid / prototype_queries if prototype_queries else 0.0,
            "view_valid": full_view_valid,
            "view_diagnostics": view_diagnostics,
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


def _ground_truth_metrics_per_view(probability, target):
    """Compute binary GT metrics after prediction maps have been finalized."""
    target_tensor = target.detach().cpu()
    if target_tensor.ndim == 4 and target_tensor.shape[0] == 1:
        target_tensor = target_tensor[0]
    elif target_tensor.ndim == 5 and target_tensor.shape[:2] == (1, 1):
        target_tensor = target_tensor[0, 0]
    target_array = target_tensor.numpy().astype(bool)
    rows = []
    for view_probability in probability:
        prediction = view_probability.detach().cpu().numpy() >= 0.5
        metrics = binary_segmentation_metrics(prediction, target_array)
        rows.append({name: float(metrics[name]) for name in TRA_GT_NAMES})
    return rows


def _best_finite_view(scores, valid=None):
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    mask = np.isfinite(scores)
    if valid is not None:
        valid = np.asarray(valid, dtype=bool).reshape(-1)
        if valid.shape != mask.shape:
            raise ValueError("View score validity mask has the wrong shape")
        mask &= valid
    if not mask.any():
        return None
    return int(np.argmax(np.where(mask, scores, -np.inf)))


def _labeled_audit_entries(data_dir):
    """Pair both image splits with GT for this offline-only evaluation pass."""
    root = Path(data_dir)
    entries = []
    for split in ("train", "test"):
        for image_path, _split_label in read_image_entries(root, split):
            # Training loaders intentionally expose no labels. The offline
            # auditor pairs them here, after adaptation/prototype inputs have
            # been separated from GT, solely for final metric computation.
            label_path = root / "labels" / image_path.parent.name / image_path.name
            if not label_path.is_file():
                raise FileNotFoundError(
                    f"Offline GT audit requires a matching label: {label_path}"
                )
            entries.append((image_path, label_path, split))
    entries.sort(key=lambda entry: (entry[0].parent.name, entry[0].name))
    return entries


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


def _selection_indices(metrics, saaf_valid=None, valid_mask=None):
    """Select views with method-specific validity.

    CAC and CAC+entropy are legacy baselines and always rank every view.  SAAF,
    purity, coverage and SAAF+entropy rank only valid SAAF views.  When a mask is
    supplied the second return value is a per-method skip dictionary.
    """
    if saaf_valid is not None and valid_mask is not None:
        raise ValueError("Pass only one of saaf_valid or valid_mask")
    legacy_return = saaf_valid is None and valid_mask is None
    saaf_valid = valid_mask if valid_mask is not None else saaf_valid
    names = ("cac", "saaf", "purity", "coverage")
    if saaf_valid is None:
        valid = torch.ones_like(metrics["saaf"], dtype=torch.bool)
    else:
        valid = torch.as_tensor(saaf_valid, device=metrics["saaf"].device).bool().reshape(-1)
        if valid.numel() != metrics["saaf"].numel():
            raise ValueError("saaf_valid must contain one flag per candidate view")
    def flat(name):
        return torch.as_tensor(metrics[name]).reshape(-1)

    result = {}
    skipped = {name: False for name in SELECTION_NAMES}
    # Legacy CAC family: no SAAF mask is ever applied.
    result["cac"] = int(flat("cac").argmax())
    cac_entropy = flat("entropy")
    cac_rank = average_tie_ranks(flat("cac").view(1, -1), descending=True)
    entropy_rank = average_tie_ranks(cac_entropy.view(1, -1), descending=False)
    result["cac_entropy"] = int(torch.argsort(cac_rank + entropy_rank, dim=1, stable=True)[0, 0])

    # SAAF family: invalid quality is -inf and invalid entropy is +inf.
    if bool(valid.any()):
        masked_entropy = flat("entropy").masked_fill(~valid, float("inf"))
        masked_entropy_rank = average_tie_ranks(masked_entropy.view(1, -1), descending=False)
        for name in ("saaf", "purity", "coverage"):
            masked_quality = flat(name).masked_fill(~valid, float("-inf"))
            result[name] = int(masked_quality.argmax())
        saaf_quality = flat("saaf").masked_fill(~valid, float("-inf"))
        saaf_rank = average_tie_ranks(saaf_quality.view(1, -1), descending=True)
        combined = saaf_rank + masked_entropy_rank
        combined = combined.masked_fill(~valid.view(1, -1), float("inf"))
        result["saaf_entropy"] = int(torch.argsort(combined, dim=1, stable=True)[0, 0])
    else:
        for name in ("saaf", "purity", "coverage", "saaf_entropy"):
            skipped[name] = True
    return result if legacy_return else (result, skipped)


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


def _tensor_statistics(values, max_quantile_samples=STATISTICS_MAX_QUANTILE_SAMPLES):
    values = values.detach().float().cpu().reshape(-1)
    if values.numel() == 0:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "quantiles": {name: None for name in ("q05", "q25", "q50", "q75", "q95")},
            "quantile_sample_count": 0,
            "quantiles_exact": True,
        }
    max_quantile_samples = max(1, int(max_quantile_samples))
    exact_quantiles = values.numel() <= max_quantile_samples
    if exact_quantiles:
        quantile_values = values
    else:
        # Exact torch.quantile sorts/copies its input and can exceed PyTorch's
        # tensor-size limit for full 3-D volumes. Use a bounded, deterministic
        # sample for quantiles while retaining exact full-volume reductions below.
        generator = torch.Generator(device="cpu")
        generator.manual_seed(0)
        indices = torch.randint(
            values.numel(), (max_quantile_samples,), generator=generator
        )
        quantile_values = values.index_select(0, indices)
    quantiles = torch.quantile(
        quantile_values, torch.tensor([0.05, 0.25, 0.5, 0.75, 0.95])
    )
    return {
        "mean": float(values.mean()),
        "std": float(values.std(unbiased=False)),
        "min": float(values.min()),
        "max": float(values.max()),
        "quantiles": {
            name: float(value)
            for name, value in zip(("q05", "q25", "q50", "q75", "q95"), quantiles)
        },
        "quantile_sample_count": int(quantile_values.numel()),
        "quantiles_exact": bool(exact_quantiles),
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
    parser.add_argument(
        "--checkpoint", required=True,
        help="Required adapted CM-SFDA checkpoint supplying student, EMA teacher and original prompts",
    )
    parser.add_argument("--prompt", default="liver")
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
    parser.add_argument(
        "--views_csv", default=None,
        help="Per-case/view CAC and TRA table (defaults next to --output)",
    )
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
    within_valid_views = {}
    for name in QUALITY_NAMES:
        valid = [case["within_case_spearman"][name] for case in cases
                 if case["within_case_spearman"][name] is not None]
        macro_spearman[name] = float(np.mean(valid)) if valid else None
        macro_valid_cases[name] = len(valid)
        within_valid_views[name] = sum(
            int(case.get("within_case_valid_views", {}).get(name, 0)) for case in cases
        )
    global_spearman = {
        name: spearman(
            [row[name] for row in mixed_views if row.get("valid_metrics", {}).get(name, True)],
            [row["dice"] for row in mixed_views if row.get("valid_metrics", {}).get(name, True)],
        )
        for name in QUALITY_NAMES
    }
    global_valid_views = {
        name: sum(1 for row in mixed_views if row.get("valid_metrics", {}).get(name, True))
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
        "within_case_spearman_valid_views": within_valid_views,
        "global_mixed_view_spearman": global_spearman,
        "global_mixed_view_valid_views": global_valid_views,
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
    checkpoint_path = Path(args.checkpoint).expanduser()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Adapted CM-SFDA checkpoint is required for TRA evaluation: {checkpoint_path}"
        )
    if args.w_quality != 0:
        raise ValueError("Quality audit is evaluation-only; use --w_quality 0")
    if args.quality_metric == "saaf" and str(args.prompt).lower() != "liver":
        raise ValueError("This SAAF audit is configured for prompt 'liver'; set --prompt liver")
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
    adapter = VoxTellPromptSFDA(
        predictor.network,
        initial_prompt,
        device,
        adapter_args,
    )
    adapter.predictor = predictor
    load_sfda_checkpoint(str(checkpoint_path), adapter)
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
    tra_cac_view_rows = []
    try:
        audit_entries = _labeled_audit_entries(args.data_dir)
        for case_index, (image_path, label_path, split_name) in enumerate(audit_entries):
            print(f"[{case_index + 1}/{len(audit_entries)}] {image_path.name}", flush=True)
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
            view_valid = inference.get("view_valid", torch.ones(args.num_aug_views, dtype=torch.bool))
            comparison_scores = {
                name: metrics[name].detach().cpu().numpy().reshape(-1)
                for name in TRA_COMPARISON_NAMES
            }
            selected_full_views = {
                name: _best_finite_view(scores)
                for name, scores in comparison_scores.items()
            }
            # GT is first read by metric computation after every view selection
            # decision above is fixed from CAC/TRA scores alone.
            dice = _dice_per_view(probability, target.float())
            gt_metrics = _ground_truth_metrics_per_view(probability, target)
            fixed_index, fixed_dice = int(dice.argmax()), float(dice.max())
            best_fixed_view_dice.append(fixed_dice)
            per_view = []
            for view_index in range(args.num_aug_views):
                finite_scores = {}
                for name in TRA_NAMES:
                    score = float(comparison_scores[name][view_index])
                    finite_scores[name] = score if np.isfinite(score) else None
                row = {
                    "case": image_path.name,
                    "view": view_index,
                    "dice": float(dice[view_index]),
                    **{
                        name: (
                            float(values[view_index])
                            if np.isfinite(float(values[view_index]))
                            else None
                        )
                        for name, values in metrics.items()
                        if name not in TRA_NAMES
                    },
                    **gt_metrics[view_index],
                    **finite_scores,
                }
                valid_semantic = bool(view_valid[view_index])
                row["valid_metrics"] = {
                    name: (valid_semantic if name in ("saaf", "purity", "coverage", "completeness", "tse") else True)
                    for name in QUALITY_NAMES
                }
                per_view.append(row)
                mixed_views.append(row)
                comparison_row = {
                    "case": image_path.name,
                    "split": split_name,
                    "view": int(view_index),
                    "cac": float(comparison_scores["cac"][view_index])
                    if np.isfinite(comparison_scores["cac"][view_index]) else None,
                    **finite_scores,
                    **gt_metrics[view_index],
                }
                for name in TRA_COMPARISON_NAMES:
                    comparison_row[f"{name}_valid"] = comparison_row[name] is not None
                    comparison_row[f"selected_{name}"] = (
                        selected_full_views[name] == view_index
                    )
                tra_cac_view_rows.append(comparison_row)
            correlations = {
                name: spearman(
                    [row[name] for row in per_view if row["valid_metrics"][name]],
                    [row["dice"] for row in per_view if row["valid_metrics"][name]],
                )
                for name in QUALITY_NAMES
            }
            valid_view_counts = {
                name: sum(1 for row in per_view if row["valid_metrics"][name])
                for name in QUALITY_NAMES
            }
            patch_spearman = {}
            for name in QUALITY_NAMES:
                values = [
                    spearman(
                        [value for value, valid in zip(
                            patch["quality"][name],
                            [d["valid"] for d in patch.get("diagnostics", [])],
                        ) if valid or name not in ("saaf", "purity", "coverage", "completeness", "tse")],
                        [dice_value for dice_value, valid in zip(
                            patch["candidate_dice"],
                            [d["valid"] for d in patch.get("diagnostics", [])],
                        ) if valid or name not in ("saaf", "purity", "coverage", "completeness", "tse")],
                    )
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
                    patch_valid = [d["valid"] for d in patch.get("diagnostics", [])]
                    patch["spearman"] = {
                        name: spearman(
                            [value for value, valid in zip(patch["quality"][name], patch_valid)
                             if valid or name not in ("saaf", "purity", "coverage", "completeness", "tse")],
                            [value for value, valid in zip(patch["candidate_dice"], patch_valid)
                             if valid or name not in ("saaf", "purity", "coverage", "completeness", "tse")],
                        )
                        for name in QUALITY_NAMES
                    }
            selections = {}
            patch_best_dice.extend(
                patch["best_dice"] for patch in inference["patch_records"] if "best_dice" in patch
            )
            for name, selected in inference["selected_dice"].items():
                if selected is not None:
                    selected_dice[name].append(selected)
                patch_values = [
                    patch["selection"][name]["selected_dice"]
                    for patch in inference["patch_records"]
                    if name in patch.get("selection", {})
                ]
                patch_best_values = [
                    patch["selection"][name]["patch_best_dice"]
                    for patch in inference["patch_records"]
                    if name in patch.get("selection", {})
                ]
                patch_gap_values = [
                    max(0.0, patch["selection"][name]["gap_to_patch_oracle"])
                    for patch in inference["patch_records"]
                    if name in patch.get("selection", {})
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
                    "full_volume_valid": selected is not None,
                    "invalid_reason": inference.get("selection_diagnostics", {}).get(name, {}).get("invalid_reason"),
                }
            # Preserve diagnostics for methods whose selected full-volume map
            # was invalid and therefore is represented by a None Dice.
            for name, diagnostic in inference.get("selection_diagnostics", {}).items():
                selections.setdefault(
                    name,
                    {
                        "view": "patchwise",
                        "full_volume_dice": None,
                        "patch_selected_dice": None,
                        "patch_best_dice": None,
                        "patch_gap_to_oracle": None,
                        "valid_patches": int(diagnostic.get("valid_patch_count", 0)),
                        "full_volume_valid": bool(diagnostic.get("full_volume_valid", False)),
                        "invalid_reason": diagnostic.get("invalid_reason"),
                    },
                )
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
                    "within_case_valid_views": valid_view_counts,
                    "tra_cac_within_case_spearman": {
                        name: {
                            gt_name: spearman(
                                [row[name] for row in tra_cac_view_rows
                                 if row["case"] == image_path.name and row[f"{name}_valid"]],
                                [row[gt_name] for row in tra_cac_view_rows
                                 if row["case"] == image_path.name and row[f"{name}_valid"]],
                            )
                            for gt_name in TRA_GT_NAMES
                        }
                        for name in TRA_COMPARISON_NAMES
                    },
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
                    "tra_cac_selected_views": selected_full_views,
                    "evidence_localization_original_view": localization,
                    "evidence_statistics_original_view": _tensor_statistics(evidence),
                    "prototype_valid_fraction": inference["prototype_valid_fraction"],
                    "sliding_window_patches": len(inference["locations"]),
                    "view_diagnostics": inference.get("view_diagnostics", {}),
                    "selection_diagnostics": inference.get("selection_diagnostics", {}),
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
    views_csv = (
        Path(args.views_csv)
        if args.views_csv
        else output.with_name(f"{output.stem}_tra_cac_views.csv")
    )
    views_csv.parent.mkdir(parents=True, exist_ok=True)
    csv_fields = (
        "case", "split", "view", "cac", "tra_teacher", "tra_original",
        "dice", "recall", "precision",
        "cac_valid", "tra_teacher_valid", "tra_original_valid",
        "selected_cac", "selected_tra_teacher", "selected_tra_original",
    )
    with views_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows(tra_cac_view_rows)
    result["tra_cac_comparison"] = summarize_tra_cac(
        tra_cac_view_rows, checkpoint_loaded=True
    )
    result["tra_cac_comparison"]["protocol"]["num_aug_views"] = int(args.num_aug_views)
    result["tra_cac_comparison"]["protocol"]["checkpoint_path"] = args.checkpoint
    result["tra_cac_comparison"]["protocol"]["view_generation"] = (
        "the case seed generates all training-style scale/offset/noise views once over the full padded volume; every sliding-window patch crops those same aligned views"
    )
    result["tra_cac_views_csv"] = str(views_csv)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "cases"}, indent=2))


if __name__ == "__main__":
    main()
