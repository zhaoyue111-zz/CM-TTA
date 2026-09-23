#!/usr/bin/env python3
"""Offline VoxTell attention--prediction diagnostics for CM-TTA.

This module deliberately delegates preprocessing, augmentation/TDC selection,
and probability inference to the existing CM-TTA/VoxTell implementations.  It
does not adapt ``ctx_delta`` or model parameters.  The attention map is the
last transformer decoder layer, the fixed organ query, averaged over all
heads.  It is reconstructed with the predictor's own sliding-window slicers
and Gaussian weights.

The VoxTell multi-output predictor returns ``[D5, D4, D3, D2, D1]``.  The
diagnostic uses D5 for the probability map.  D1--D4, when described in
metadata, mean patch-upsampled-then-sliding-window-fused full-volume outputs,
not raw low-resolution probability maps.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import average_precision_score, roc_auc_score


STAT_FIELDS = (
    "voxel_count", "mean", "std", "min", "p05", "p25", "median", "p75", "p95", "max"
)
REGIONS = ("valid", "gt_foreground", "gt_background_valid", "TP", "FN", "FP", "TN")
SCORE_NAMES = ("probability", "raw_attention", "rank_attention", "residual")
DECODER_COUNT = 5


def _finite_float(value: Any) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def json_shape(shape: Sequence[int] | None) -> str:
    return json.dumps([int(v) for v in shape]) if shape is not None else ""


def stats(values: np.ndarray | Iterable[float]) -> dict[str, float | int]:
    values = np.asarray(values, dtype=np.float32)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"voxel_count": 0, **{field: float("nan") for field in STAT_FIELDS[1:]}}
    quantiles = np.percentile(values, [5, 25, 50, 75, 95]).astype(np.float32)
    return {
        "voxel_count": int(values.size),
        "mean": float(np.mean(values, dtype=np.float32)),
        "std": float(np.std(values, dtype=np.float32)),
        "min": float(np.min(values)),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(np.max(values)),
    }


def percentile_rank(values: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Percentile rank calculated independently over finite valid voxels."""
    values = np.asarray(values, dtype=np.float32)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if values.shape != valid_mask.shape:
        raise ValueError(f"rank shape mismatch: {values.shape} vs {valid_mask.shape}")
    result = np.full(values.shape, np.nan, dtype=np.float32)
    selected = valid_mask & np.isfinite(values)
    if not np.any(selected):
        return result
    flat = values[selected].astype(np.float64, copy=False)
    if flat.size == 1:
        result[selected] = 0.5
    else:
        result[selected] = ((rankdata(flat, method="average") - 1.0) / (flat.size - 1)).astype(np.float32)
    return result


def attention_tokens_to_3d(tokens: np.ndarray, memory_shape_dhw: Sequence[int]) -> np.ndarray:
    """Restore VoxTell's flattened ``(H,W,D)`` token order to ``(D,H,W)``."""
    tokens = np.asarray(tokens, dtype=np.float32)
    shape = tuple(int(v) for v in memory_shape_dhw)
    if tokens.ndim != 1:
        raise ValueError(f"attention tokens must be 1-D, got {tokens.shape}")
    if len(shape) != 3 or int(np.prod(shape)) != tokens.size:
        raise ValueError(f"attention token count {tokens.size} != memory shape product {shape}")
    # Model code rearranges (B,C,D,H,W) -> (B,H,W,D,C) -> (H*W*D,B,C).
    return tokens.reshape(shape[1], shape[2], shape[0]).transpose(2, 0, 1).astype(np.float32, copy=False)


def apply_intensity_view(
    data: torch.Tensor, params: Mapping[str, float], valid_mask: torch.Tensor | None = None
) -> torch.Tensor:
    """Apply the same affine view operation as ``VoxTellCMTTA``."""
    view = data if float(params.get("scale", 1.0)) == 1.0 and float(params.get("offset", 0.0)) == 0.0 else data * float(params["scale"]) + float(params["offset"])
    if valid_mask is not None:
        view = view * valid_mask.to(device=view.device, dtype=view.dtype)
    return view.contiguous()


def inverse_view_map(view_map: np.ndarray, params: Mapping[str, float]) -> np.ndarray:
    """Return a map in the original crop coordinates.

    Current CM-TTA views are intensity-only, so their spatial inverse is the
    identity.  Probabilities/attention must not be intensity-inverted.
    """
    del params
    return np.asarray(view_map, dtype=np.float32).copy()


def _slice_shape(slicer: Sequence[slice]) -> tuple[int, int, int]:
    return tuple(int(s.stop - s.start) for s in slicer[1:])  # type: ignore[union-attr]


def aggregate_attention_patches(
    records: Sequence[Mapping[str, Any]],
    slicers: Sequence[Sequence[slice]],
    padded_shape_dhw: Sequence[int],
    revert_padding: Sequence[slice],
    gaussian: np.ndarray,
) -> np.ndarray:
    """Gaussian-fuse captured patch attention using predictor slicers."""
    padded_shape = tuple(int(v) for v in padded_shape_dhw)
    gaussian = np.asarray(gaussian, dtype=np.float32)
    if gaussian.shape != _slice_shape(slicers[0]):
        raise ValueError(f"Gaussian shape {gaussian.shape} != patch shape {_slice_shape(slicers[0])}")
    if len(records) != len(slicers):
        raise AssertionError(f"captured {len(records)} attention patches for {len(slicers)} slicers")
    numerator = np.zeros(padded_shape, dtype=np.float32)
    denominator = np.zeros(padded_shape, dtype=np.float32)
    for record, slicer in zip(records, slicers):
        weights = np.asarray(record["weights"], dtype=np.float32)
        if weights.ndim != 4 or weights.shape[0] != 1 or weights.shape[2] < 1:
            raise AssertionError(f"expected (B,heads,query,tokens), got {weights.shape}")
        token_map = attention_tokens_to_3d(weights[0, :, 0, :].mean(axis=0), record["memory_shape_dhw"])
        tile_shape = _slice_shape(slicer)
        if token_map.shape == tile_shape:
            tile = token_map
        else:
            tile = F.interpolate(
                torch.from_numpy(token_map)[None, None], size=tile_shape,
                mode="trilinear", align_corners=False,
            )[0, 0].numpy().astype(np.float32, copy=False)
        spatial = tuple(slicer[1:])
        weighted = tile * gaussian
        numerator[spatial] += weighted
        denominator[spatial] += gaussian
    if not np.all(np.isfinite(denominator)) or np.any(denominator <= 0):
        raise AssertionError("attention Gaussian fusion has uncovered or non-finite voxels")
    fused = numerator / denominator
    fused = fused[tuple(revert_padding)]
    if not np.all(np.isfinite(fused)):
        raise AssertionError("fused attention contains non-finite values")
    return fused.astype(np.float32, copy=False)


def crop_gt_to_inference_region(
    gt_full: np.ndarray,
    valid_inference_mask: np.ndarray,
    bbox: Sequence[Sequence[int]],
    case: str,
) -> np.ndarray:
    gt_full = np.asarray(gt_full, dtype=bool)
    valid_inference_mask = np.asarray(valid_inference_mask, dtype=bool)
    if gt_full.shape != valid_inference_mask.shape:
        raise AssertionError(
            f"{case}: GT/mask shape mismatch before crop: {gt_full.shape} vs {valid_inference_mask.shape}"
        )
    outside_count = int(np.count_nonzero(gt_full & ~valid_inference_mask))
    if outside_count:
        raise ValueError(
            f"{case}: {outside_count} GT foreground voxels lie outside predictor crop bbox"
        )
    slices = tuple(slice(int(lo), int(hi)) for lo, hi in bbox)
    cropped = gt_full[slices]
    if not np.all(valid_inference_mask[slices]):
        raise AssertionError(f"{case}: predictor crop bbox contains invalid inference voxels")
    return cropped.astype(bool, copy=False)


def confusion_regions(
    probability: np.ndarray, gt: np.ndarray, valid_mask: np.ndarray, threshold: float = 0.5
) -> dict[str, np.ndarray]:
    probability = np.asarray(probability, dtype=np.float32)
    gt = np.asarray(gt, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not (probability.shape == gt.shape == valid_mask.shape):
        raise AssertionError(f"probability/gt/valid shape mismatch: {probability.shape}, {gt.shape}, {valid_mask.shape}")
    predicted = probability >= np.float32(threshold)
    return {
        "valid": valid_mask,
        "gt_foreground": valid_mask & gt,
        "gt_background_valid": valid_mask & ~gt,
        "TP": valid_mask & gt & predicted,
        "FN": valid_mask & gt & ~predicted,
        "FP": valid_mask & ~gt & predicted,
        "TN": valid_mask & ~gt & ~predicted,
    }


def assert_same_embedding(reference: torch.Tensor, candidate: torch.Tensor) -> None:
    if reference is candidate:
        return
    if reference.shape != candidate.shape or reference.dtype != candidate.dtype:
        raise AssertionError(
            f"embedding shape/dtype mismatch: {tuple(reference.shape)}/{reference.dtype} vs "
            f"{tuple(candidate.shape)}/{candidate.dtype}"
        )
    if not torch.equal(reference, candidate):
        raise AssertionError("TDC and probability/attention do not use identical text embeddings")


@contextlib.contextmanager
def fixed_embedding_for_selector(adapter: Any, embedding: torch.Tensor):
    """Make the existing selector consume a pre-encoded embedding unchanged.

    ``VoxTellCMTTA._select_case_view`` normally receives a context delta and
    encodes it internally. The diagnostic already owns the required fixed
    ``short_ctx`` embedding, so temporarily redirect only this selector call
    to that tensor. No model, parameter, optimizer, or training method is
    changed.
    """
    original_encode = adapter._encode_ctx

    def return_fixed(_ctx_delta: torch.Tensor) -> torch.Tensor:
        assert_same_embedding(embedding, embedding)
        return embedding

    adapter._encode_ctx = return_fixed
    try:
        yield
    finally:
        adapter._encode_ctx = original_encode


def selected_view_from_tdc(tdc_scores: Sequence[float]) -> tuple[int, list[int]]:
    scores = np.asarray(tdc_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.all(np.isfinite(scores)):
        raise AssertionError(f"TDC scores must be a non-empty finite vector, got {scores}")
    expected = int(np.argmax(scores))
    order = np.argsort(-scores, kind="stable")
    ranks = np.empty(scores.size, dtype=np.int64)
    ranks[order] = np.arange(scores.size, dtype=np.int64)
    return expected, ranks.tolist()


def _metric_samples(
    y_true: np.ndarray,
    score: np.ndarray,
    max_negative_voxels: int | None = None,
    seed: int = 20260923,
) -> tuple[np.ndarray, np.ndarray, dict[str, int | bool]]:
    y_true = np.asarray(y_true, dtype=bool)
    score = np.asarray(score, dtype=np.float32)
    finite = np.isfinite(score)
    y_true, score = y_true[finite], score[finite]
    positive = np.flatnonzero(y_true)
    negative = np.flatnonzero(~y_true)
    metadata: dict[str, int | bool] = {
        "raw_positive_count": int(positive.size),
        "raw_negative_count": int(negative.size),
        "negative_sampling_enabled": bool(max_negative_voxels is not None),
    }
    if max_negative_voxels is not None:
        if int(max_negative_voxels) < 1:
            raise ValueError("metric_max_negative_voxels must be positive when provided")
        if negative.size > int(max_negative_voxels):
            rng = np.random.default_rng(int(seed))
            negative = np.sort(rng.choice(negative, size=int(max_negative_voxels), replace=False))
    selected = np.concatenate((positive, negative))
    metadata.update({
        "actual_positive_count": int(positive.size),
        "actual_negative_count": int(negative.size),
        "actual_sample_count": int(selected.size),
        "negative_sampled": bool(negative.size < metadata["raw_negative_count"]),
    })
    return y_true[selected], score[selected], metadata


def safe_auc_pr(
    y_true: np.ndarray,
    score: np.ndarray,
    max_negative_voxels: int | None = None,
    seed: int = 20260923,
) -> tuple[float, float]:
    y_true, score, _ = _metric_samples(y_true, score, max_negative_voxels, seed)
    if y_true.size == 0 or np.unique(y_true).size < 2:
        return float("nan"), float("nan")
    try:
        return float(roc_auc_score(y_true, score)), float(average_precision_score(y_true, score))
    except ValueError:
        return float("nan"), float("nan")


def balanced_indices(gt: np.ndarray, valid_mask: np.ndarray, seed: int = 20260923) -> np.ndarray:
    gt = np.asarray(gt, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    foreground = np.flatnonzero(valid_mask & gt)
    background = np.flatnonzero(valid_mask & ~gt)
    if foreground.size == 0 or background.size == 0:
        return np.unique(np.concatenate((foreground, background)))
    n = min(foreground.size, background.size)
    rng = np.random.default_rng(int(seed))
    sampled_background = rng.choice(background, size=n, replace=False)
    return np.concatenate((foreground, sampled_background)).astype(np.int64, copy=False)


def _correlations(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(x) & np.isfinite(y)
    x, y = np.asarray(x)[finite], np.asarray(y)[finite]
    if x.size < 2 or np.std(x) == 0 or np.std(y) == 0:
        return float("nan"), float("nan")
    pearson = float(np.corrcoef(x.astype(np.float64), y.astype(np.float64))[0, 1])
    spearman = float(spearmanr(x, y).statistic)
    return pearson, spearman


def compute_case_metrics(
    probability: np.ndarray,
    raw_attention: np.ndarray,
    gt: np.ndarray,
    valid_mask: np.ndarray,
    *,
    case: str = "",
    prompt: str = "liver",
    selected_view: int = -1,
    selected_view_tdc: float = float("nan"),
    threshold: float = 0.5,
    balanced_seed: int = 20260923,
    metric_max_negative_voxels: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, np.ndarray]]:
    probability = np.asarray(probability, dtype=np.float32)
    raw_attention = np.asarray(raw_attention, dtype=np.float32)
    gt = np.asarray(gt, dtype=bool)
    valid_mask = np.asarray(valid_mask, dtype=bool)
    if not (probability.shape == raw_attention.shape == gt.shape == valid_mask.shape):
        raise AssertionError(f"attention.shape == probability.shape == gt.shape required, got {probability.shape}, {raw_attention.shape}, {gt.shape}")
    if not np.all(np.isfinite(probability[valid_mask])) or not np.all(np.isfinite(raw_attention[valid_mask])):
        raise AssertionError("valid probability/attention voxels must be finite")
    rank_probability = percentile_rank(probability, valid_mask)
    rank_attention = percentile_rank(raw_attention, valid_mask)
    residual = (rank_attention - rank_probability).astype(np.float32)
    regions = confusion_regions(probability, gt, valid_mask, threshold)
    score_maps = {
        "probability": probability,
        "raw_attention": raw_attention,
        "rank_attention": rank_attention,
        "residual": residual,
    }
    common = {
        "case": case, "prompt": prompt, "selected_view": int(selected_view),
        "selected_view_tdc": _finite_float(selected_view_tdc), "threshold": float(threshold),
        "probability_shape": json_shape(probability.shape), "attention_shape": json_shape(raw_attention.shape),
        "gt_shape": json_shape(gt.shape), "valid_mask_source": "predictor_crop_to_nonzero_bbox_padding_removed",
        "probability_interpolation": "none_same_size", "gt_interpolation": "nearest",
    }
    rows: list[dict[str, Any]] = []
    for region in REGIONS:
        mask = regions[region]
        row = dict(common, region=region)
        row["valid_voxel_count"] = int(valid_mask.sum())
        for score_name, score_map in score_maps.items():
            for key, value in stats(score_map[mask]).items():
                row[f"{score_name}_{key}"] = value
        rows.append(row)

    valid_values = valid_mask & np.isfinite(probability) & np.isfinite(raw_attention)
    pearson, spearman = _correlations(raw_attention[valid_values], probability[valid_values])
    indices = balanced_indices(gt, valid_values, balanced_seed)
    flat_attention, flat_probability = raw_attention.ravel(), probability.ravel()
    balanced_pearson, balanced_spearman = _correlations(flat_attention[indices], flat_probability[indices]) if indices.size else (float("nan"), float("nan"))
    metrics: dict[str, Any] = dict(common)
    metrics.update({
        "balanced_seed": int(balanced_seed), "valid_voxel_count": int(valid_values.sum()),
        "gt_foreground_voxel_count": int((valid_mask & gt).sum()),
        "gt_background_valid_voxel_count": int((valid_mask & ~gt).sum()),
        "balanced_sample_count": int(indices.size),
        "pearson_attention_probability": pearson,
        "spearman_attention_probability": spearman,
        "pearson_attention_probability_balanced": balanced_pearson,
        "spearman_attention_probability_balanced": balanced_spearman,
    })
    # Classification metrics use percentile-rank attention. Raw attention is
    # retained for descriptive region statistics only and is not interpreted
    # as a cross-case comparable global intensity.
    score_for_metrics = {"attention": rank_attention, "probability": probability, "R": residual}
    sampling_metadata: dict[str, dict[str, int | bool]] = {}
    for group_name, positive_region, negative_region in (
        ("background_FN_TN", "FN", "TN"),
        ("foreground_TP_FP", "TP", "FP"),
        ("gt_foreground_background", "gt_foreground", "gt_background_valid"),
    ):
        selection = regions[positive_region] | regions[negative_region]
        labels = regions[positive_region][selection]
        for score_name, score_map in score_for_metrics.items():
            auc, ap = safe_auc_pr(
                labels,
                score_map[selection],
                max_negative_voxels=metric_max_negative_voxels,
                seed=balanced_seed + sum(ord(char) for char in f"{group_name}:{score_name}"),
            )
            metrics[f"{group_name}_{score_name}_auroc"] = auc
            metrics[f"{group_name}_{score_name}_auprc"] = ap
            _, _, sample_metadata = _metric_samples(
                labels,
                score_map[selection],
                max_negative_voxels=metric_max_negative_voxels,
                seed=balanced_seed + sum(ord(char) for char in f"{group_name}:{score_name}"),
            )
            sampling_metadata[f"{group_name}_{score_name}"] = sample_metadata
            for field, value in sample_metadata.items():
                metrics[f"{group_name}_{score_name}_{field}"] = value
    metrics["metric_max_negative_voxels"] = (
        "" if metric_max_negative_voxels is None else int(metric_max_negative_voxels)
    )
    metrics["metric_negative_sampling"] = "negative_only" if metric_max_negative_voxels is not None else "none"
    metrics["metric_sampling_seed"] = int(balanced_seed)
    return rows, metrics, {"probability": probability, "raw_attention": raw_attention, "rank_attention": rank_attention, "residual": residual, **regions}


class AttentionCapture(contextlib.AbstractContextManager):
    """Temporarily request per-head weights from the final cross-attention."""

    def __init__(self, model: torch.nn.Module):
        self.model = getattr(model, "_orig_mod", model)
        self.records: list[dict[str, Any]] = []
        self.memory_shapes: list[tuple[int, int, int]] = []
        self._handles: list[Any] = []
        self._original_forward = None

    def __enter__(self):
        decoder = self.model.transformer_decoder
        layer = decoder.layers[-1]
        attention = layer.multihead_attn
        self._original_forward = attention.forward

        def forward_with_heads(*args, **kwargs):
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = False
            return self._original_forward(*args, **kwargs)

        attention.forward = forward_with_heads

        def capture_memory(_module, _inputs, output):
            if output.ndim != 5:
                raise AssertionError(f"projected memory must be (B,H,W,D,C), got {tuple(output.shape)}")
            self.memory_shapes.append(tuple(int(v) for v in (output.shape[3], output.shape[1], output.shape[2])))

        def capture_attention(_module, _inputs, output):
            if not isinstance(output, (tuple, list)) or len(output) != 2 or output[1] is None:
                raise AssertionError("cross-attention did not return weights")
            weights = output[1].detach().float().cpu().numpy()
            if weights.ndim != 4:
                raise AssertionError(f"expected per-head attention weights, got {weights.shape}")
            if weights.shape[2] != 1:
                raise AssertionError(
                    "fixed query=0 is only valid for one organ query; "
                    f"captured {weights.shape[2]} queries"
                )
            if not self.memory_shapes:
                raise AssertionError("attention was captured before projected memory")
            memory_shape = self.memory_shapes[-1]
            if weights.shape[-1] != int(np.prod(memory_shape)):
                raise AssertionError(f"attention token count {weights.shape[-1]} != memory product {memory_shape}")
            self.records.append({"weights": weights, "memory_shape_dhw": memory_shape})

        self._handles.append(self.model.project_bottleneck_embed.register_forward_hook(capture_memory))
        self._handles.append(attention.register_forward_hook(capture_attention))
        return self

    def __exit__(self, exc_type, exc, traceback):
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
        if self._original_forward is not None:
            self.model.transformer_decoder.layers[-1].multihead_attn.forward = self._original_forward
        return False


def resolve_label_value(label_map: np.ndarray, explicit_value: int | None, case: str) -> int:
    values = sorted(int(v) for v in np.unique(label_map) if int(v) != 0)
    if explicit_value is not None:
        if int(explicit_value) not in values:
            raise ValueError(f"{case}: requested --label-value {explicit_value}, existing non-zero labels are {values}")
        return int(explicit_value)
    if len(values) == 1:
        return values[0]
    if len(values) > 1:
        raise ValueError(f"{case}: multiple non-zero GT labels {values}; pass --label-value explicitly")
    raise ValueError(f"{case}: no non-zero GT label found; pass --label-value explicitly if intentional")


def build_adapter(predictor, args):
    from method.voxtell_cmtta import VoxTellCMTTA

    adapter_args = argparse.Namespace(
        lr=0.0, amp_init_scale=32.0, ema_momentum=0.9, w_cac=1.0, w_entropy=0.0,
        pseudo_update_mode="original", bg_threshold=0.1, tversky_alpha=0.3,
        tversky_beta=0.7, tversky_weight=1.0, amb_weight=0.0,
        num_aug_views=args.num_aug_views, selection_p=args.selection_p,
        view_selection_metric="tdc", use_entropy_rank=False,
        short_memory_length=1, view_batch_size=args.view_batch_size,
        grad_clip=1.0, max_text_length=getattr(predictor, "max_text_length", 8192),
        decoder_alignment_check=False,
    )
    return VoxTellCMTTA(
        model=predictor.network, initial_ctx=None, device=predictor.device,
        args=adapter_args, qwen_text_encoder=predictor.text_backbone,
        qwen_tokenizer=predictor.tokenizer, text_prompt=args.prompt,
    )


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _numeric_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    ignored = {"case", "prompt", "probability_shape", "attention_shape", "gt_shape", "valid_mask_source", "tdc_scores_json", "view_params_json"}
    fields = [key for key in rows[0] if key not in ignored and isinstance(rows[0].get(key), (int, float, np.integer, np.floating))]
    result = []
    for field in fields:
        values = np.asarray([float(row[field]) for row in rows], dtype=np.float64)
        values = values[np.isfinite(values)]
        result.append({
            "summary_type": "metric", "region": "", "metric": field,
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "median": float(np.median(values)) if values.size else float("nan"),
            "std": float(np.std(values)) if values.size else float("nan"),
            "valid_case_count": int(values.size),
        })
    return result


def _region_summary(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Summarize region statistics case-wise over valid crop-space regions."""
    result = []
    for region in REGIONS:
        group = [row for row in rows if row.get("region") == region]
        if not group:
            continue
        fields = [
            f"{score_name}_{stat_name}"
            for score_name in SCORE_NAMES
            for stat_name in STAT_FIELDS
            if f"{score_name}_{stat_name}" in group[0]
        ]
        for field in fields:
            values = np.asarray([float(row[field]) for row in group], dtype=np.float64)
            values = values[np.isfinite(values)]
            result.append({
                "summary_type": "region_stat", "region": region, "metric": field,
                "mean": float(np.mean(values)) if values.size else float("nan"),
                "median": float(np.median(values)) if values.size else float("nan"),
                "std": float(np.std(values)) if values.size else float("nan"),
                "valid_case_count": int(values.size),
            })
    return result


def _json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.integer, torch.Tensor)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    return value


def _conclusions(metrics_rows: Sequence[Mapping[str, Any]], region_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    del region_rows  # Raw attention region means remain descriptive CSV fields only.

    def metric_summary(group: str, score: str, metric: str) -> dict[str, Any]:
        field = f"{group}_{score}_{metric}"
        values = np.asarray(
            [float(row[field]) for row in metrics_rows if math.isfinite(float(row.get(field, "nan")))],
            dtype=np.float64,
        )
        return {
            "mean": float(np.mean(values)) if values.size else float("nan"),
            "median": float(np.median(values)) if values.size else float("nan"),
            "valid_case_count": int(values.size),
            "greater_than_0_5_case_count": int(np.count_nonzero(values > 0.5)),
        }

    def group_summary(group: str) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for score in ("attention", "R"):
            result[score] = {
                "auroc": metric_summary(group, score, "auroc"),
                "auprc": metric_summary(group, score, "auprc"),
            }
        for metric in ("auroc", "auprc"):
            r_values = np.asarray(
                [float(row[f"{group}_R_{metric}"]) for row in metrics_rows if math.isfinite(float(row.get(f"{group}_R_{metric}", "nan")))],
                dtype=np.float64,
            )
            attention_values = np.asarray(
                [float(row[f"{group}_attention_{metric}"]) for row in metrics_rows if math.isfinite(float(row.get(f"{group}_R_{metric}", "nan"))) and math.isfinite(float(row.get(f"{group}_attention_{metric}", "nan")))],
                dtype=np.float64,
            )
            probability_values = np.asarray(
                [float(row[f"{group}_probability_{metric}"]) for row in metrics_rows if math.isfinite(float(row.get(f"{group}_R_{metric}", "nan"))) and math.isfinite(float(row.get(f"{group}_probability_{metric}", "nan")))],
                dtype=np.float64,
            )
            result[f"R_vs_attention_{metric}"] = {
                "mean_delta": float(np.mean(r_values - attention_values)) if r_values.size == attention_values.size and r_values.size else float("nan"),
                "R_better_case_count": int(np.count_nonzero(r_values > attention_values)) if r_values.size == attention_values.size else 0,
                "valid_case_count": int(min(r_values.size, attention_values.size)),
            }
            result[f"R_vs_probability_{metric}"] = {
                "mean_delta": float(np.mean(r_values - probability_values)) if r_values.size == probability_values.size and r_values.size else float("nan"),
                "R_better_case_count": int(np.count_nonzero(r_values > probability_values)) if r_values.size == probability_values.size else 0,
                "valid_case_count": int(min(r_values.size, probability_values.size)),
            }
        return result

    pearson = _finite_case_mean(metrics_rows, "pearson_attention_probability")
    spearman = _finite_case_mean(metrics_rows, "spearman_attention_probability")
    return {
        "FN_vs_TN": group_summary("background_FN_TN"),
        "TP_vs_FP": group_summary("foreground_TP_FP"),
        "attention_probability_similarity": {"pearson_mean": pearson, "spearman_mean": spearman},
        "raw_attention_note": "raw attention 仅作病例内描述性统计，不解释为跨病例可直接比较的全局强度。",
        "interpretation": "指标按病例汇总；R/attention/probability 的比较不由区域均值单独判定有效性。",
    }


def _finite_case_mean(rows: Sequence[Mapping[str, Any]], field: str) -> float:
    values = [float(row[field]) for row in rows if math.isfinite(float(row.get(field, "nan")))]
    return float(np.mean(values)) if values else float("nan")


def _save_maps(directory: Path, maps: Mapping[str, np.ndarray]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    for name in ("probability", "raw_attention", "rank_attention", "residual", "TP", "FN", "FP", "TN", "valid"):
        np.save(directory / f"{name}.npy", np.asarray(maps[name]))


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--split-file", default=None)
    parser.add_argument("--voxtell-root", default="/data/zy/VoxTell_from_disk")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--text-model", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--prompt", default="liver")
    parser.add_argument("--label-value", type=int, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--num-aug-views", type=int, default=9)
    parser.add_argument("--view-batch-size", type=int, default=1)
    parser.add_argument("--selection-p", type=float, default=0.1)
    parser.add_argument("--metric-max-negative-voxels", type=int, default=None,
                        help="optional cap for negative voxels in AUROC/AUPRC; default uses all voxels")
    parser.add_argument("--balanced-seed", type=int, default=20260923)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save-case", action="append", default=None, help="case basename to save maps for; repeat as needed")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--save-maps", action="store_true")
    parser.add_argument("--all-layers-on-device", action="store_true", help="faster but less safe for return_all_layers=True")
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.save_maps and not args.save_case:
        raise ValueError("--save-maps requires at least one --save-case")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    root = Path(args.voxtell_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from acvl_utils.cropping_and_padding.padding import pad_nd_image
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    from nnunetv2.inference.sliding_window_prediction import compute_gaussian
    from data.voxtell_p0 import load_ras_image, make_case_patches, read_test_entries
    from voxtell.inference.decoder_probability_distribution import decoder_metadata
    from voxtell.inference.predictor_multiclass import VoxTellPredictor

    device = torch.device(args.device)
    entries = read_test_entries(args.data_dir, args.split_file)
    if args.limit is not None:
        entries = entries[:args.limit]
    if not entries:
        raise RuntimeError("no image/label pairs found")
    reader = NibabelIOWithReorient()
    label_values: dict[str, int] = {}
    for image_path, label_path in entries:
        case = image_path.name.removesuffix(".nii.gz")
        label_array = reader.read_images([str(label_path)])[0]
        label_values[case] = resolve_label_value(np.rint(label_array[0]).astype(np.int64), args.label_value, case)
        del label_array

    predictor = VoxTellPredictor(
        str(args.model_dir), device=device, text_encoding_model=args.text_model,
        return_all_layers_on_cpu=not args.all_layers_on_device,
    )
    adapter = build_adapter(predictor, args)
    ctx_snapshot = adapter.ctx_delta.detach().clone()
    region_rows: list[dict[str, Any]] = []
    metric_rows: list[dict[str, Any]] = []
    view_rows: list[dict[str, Any]] = []
    for index, (image_path, label_path) in enumerate(entries, 1):
        case = image_path.name.removesuffix(".nii.gz")
        image = load_ras_image(str(image_path))
        data, bbox, original_shape = predictor.preprocess(image)
        patches, valid_masks, _locations, crop_shape = make_case_patches(data, predictor.patch_size)
        params = adapter._sample_intensity_params(args.num_aug_views)
        short_ctx = adapter._encode_ctx(adapter.ctx_delta.detach()).detach()
        with fixed_embedding_for_selector(adapter, short_ctx):
            selected_view, _ = adapter._select_case_view(
                patches, params, short_ctx, valid_masks, collect_tdc=True
            )
        if not torch.equal(adapter.ctx_delta.detach(), ctx_snapshot):
            raise AssertionError("offline diagnostic changed ctx_delta")
        selection = dict(adapter.last_view_selection)
        tdc_scores = selection.get("tdc") or []
        expected_view, tdc_ranks = selected_view_from_tdc(tdc_scores)
        if int(selected_view) != expected_view:
            raise AssertionError(
                f"TDC selection mismatch for {case}: selected={selected_view}, "
                f"argmax_tdc={expected_view}, scores={tdc_scores}"
            )
        if selection.get("tdc_rank") is not None and list(selection["tdc_rank"]) != tdc_ranks:
            raise AssertionError(
                f"TDC rank mismatch for {case}: selector={selection['tdc_rank']}, expected={tdc_ranks}"
            )
        view_rows.extend({
            "case": case, "prompt": args.prompt, "view": view, "tdc_score": float(score),
            "tdc_rank": int(tdc_ranks[view]), "selected_view": int(selected_view),
            "scale": float(params[view]["scale"]), "offset": float(params[view]["offset"]),
        } for view, score in enumerate(tdc_scores))
        selected_param = params[selected_view]
        selected_data = apply_intensity_view(data.float(), selected_param)
        diagnostic_embedding = short_ctx
        assert_same_embedding(short_ctx, diagnostic_embedding)
        # Crop space has no predictor padding. The only valid mask is therefore
        # all crop voxels; predictor padding is removed by its public API.
        full_valid_mask = predictor.get_last_valid_inference_mask()
        crop_slices = tuple(slice(int(lo), int(hi)) for lo, hi in bbox)
        valid_mask = full_valid_mask[crop_slices].astype(bool, copy=False)
        if valid_mask.shape != tuple(int(v) for v in crop_shape) or not np.all(valid_mask):
            raise AssertionError("predictor valid inference mask does not exactly cover the preprocessed crop")
        with AttentionCapture(predictor.network) as capture:
            returned_logits = predictor.predict_sliding_window_return_logits(
                selected_data, diagnostic_embedding, return_all_layers=True
            )
        if not isinstance(returned_logits, (list, tuple)) or len(returned_logits) != DECODER_COUNT:
            raise AssertionError("return_all_layers=True must return five decoder outputs")
        d5_logits = returned_logits[0][0].detach().float().cpu()
        probability = torch.sigmoid(d5_logits).numpy().astype(np.float32, copy=False)
        if probability.shape != tuple(crop_shape):
            raise AssertionError(f"D5 probability shape {probability.shape} != crop shape {crop_shape}")
        padded_data, _revert = pad_nd_image(selected_data, predictor.patch_size, "constant", {"value": 0}, True, None)
        slicers = predictor._internal_get_sliding_window_slicers(padded_data.shape[1:])
        if len(capture.records) != len(slicers):
            raise AssertionError(
                f"captured {len(capture.records)} attention patches for {len(slicers)} sliding-window slicers"
            )
        gaussian = compute_gaussian(tuple(predictor.patch_size), sigma_scale=1.0 / 8, value_scaling_factor=10, device="cpu").numpy().astype(np.float32)
        raw_attention = aggregate_attention_patches(capture.records, slicers, padded_data.shape[1:], _revert[1:], gaussian)
        raw_attention = inverse_view_map(raw_attention, selected_param)
        if raw_attention.shape != probability.shape:
            raise AssertionError(f"attention/probability mismatch: {raw_attention.shape} vs {probability.shape}")

        label_full = reader.read_images([str(label_path)])[0][0]
        gt_full = np.rint(label_full).astype(np.int64) == label_values[case]
        gt = crop_gt_to_inference_region(gt_full, full_valid_mask, bbox, case)
        if gt.shape != probability.shape:
            gt = F.interpolate(torch.from_numpy(gt.astype(np.float32))[None, None], size=probability.shape, mode="nearest")[0, 0].numpy().astype(bool)
        if not (raw_attention.shape == probability.shape == gt.shape == valid_mask.shape):
            raise AssertionError("final attention/probability/GT/valid shapes must match")
        rows, metrics, maps = compute_case_metrics(
            probability, raw_attention, gt, valid_mask, case=case, prompt=args.prompt,
            selected_view=selected_view, selected_view_tdc=tdc_scores[selected_view],
            threshold=args.threshold, balanced_seed=args.balanced_seed,
            metric_max_negative_voxels=args.metric_max_negative_voxels,
        )
        observed_shapes: list[Sequence[int] | None] = [None] * DECODER_COUNT
        for info in getattr(predictor, "decoder_output_metadata", []):
            list_index = int(info["model_output_list_index"])
            observed_shapes[list_index] = info.get("observed_raw_patch_shape")
        metadata = decoder_metadata(
            [shape if shape is not None else tuple(int(v) for v in returned_logits[i].shape[2:])
             for i, shape in enumerate(observed_shapes)], probability.shape
        )
        decoder_semantics = {
            item["decoder_stage"]: (
                "final_output_sliding_window_gaussian_fused_full_volume"
                if item["is_final_output"]
                else "patch_upsampled_then_sliding_window_gaussian_fused_full_volume"
            )
            for item in metadata
        }
        metrics.update({
            "decoder_stage": "D5", "is_final_output": True,
            "raw_shape": json_shape(metadata[-1]["raw_shape"]), "aligned_shape": json_shape(probability.shape),
            "attention_memory_shape": json_shape(capture.records[0]["memory_shape_dhw"]),
            "attention_layer": "last_transformer_decoder_layer", "attention_head_aggregation": "mean_all_heads",
            "attention_query": "current_organ_query", "attention_semantics": "patch-local softmax cross-attention, upsampled and Gaussian fused",
            "decoder_metadata_json": json.dumps(metadata), "decoder_semantics_json": json.dumps(decoder_semantics),
            "view_count": len(params), "tdc_scores_json": json.dumps(tdc_scores),
            "tdc_ranks_json": json.dumps(tdc_ranks), "view_params_json": json.dumps(params),
        })
        for row in rows:
            row.update({
                "decoder_stage": "D5", "is_final_output": True, "raw_shape": json_shape(metadata[-1]["raw_shape"]),
                "aligned_shape": json_shape(probability.shape), "threshold": args.threshold,
                "attention_memory_shape": json_shape(capture.records[0]["memory_shape_dhw"]),
                "attention_layer": "last_transformer_decoder_layer", "attention_head_aggregation": "mean_all_heads",
                "attention_query": "current_organ_query", "attention_semantics": "patch-local softmax cross-attention, upsampled and Gaussian fused",
                "decoder_metadata_json": json.dumps(metadata), "decoder_semantics_json": json.dumps(decoder_semantics),
                "view_count": len(params), "tdc_scores_json": json.dumps(tdc_scores),
                "tdc_ranks_json": json.dumps(tdc_ranks), "view_params_json": json.dumps(params),
            })
        region_rows.extend(rows)
        metric_rows.append(metrics)
        if args.save_maps and case in {Path(name).name.removesuffix(".nii.gz") for name in args.save_case}:
            _save_maps(Path(args.output_dir) / "maps" / case, maps)
        del returned_logits, probability, raw_attention, gt, label_full, image, data, selected_data, patches, valid_masks
        print(f"[{index}/{len(entries)}] {case}: selected_view={selected_view}, tdc={tdc_scores[selected_view]:.6f}")

    if not torch.equal(adapter.ctx_delta.detach(), ctx_snapshot):
        raise AssertionError("offline diagnostic changed ctx_delta")
    adapter.close()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_csv(output_dir / "attention_probability_region_stats_per_case.csv", region_rows)
    _write_csv(output_dir / "attention_probability_metrics_per_case.csv", metric_rows)
    summary = _numeric_summary(metric_rows) + _region_summary(region_rows)
    _write_csv(output_dir / "attention_probability_summary.csv", summary, ("summary_type", "region", "metric", "mean", "median", "std", "valid_case_count"))
    diagnostics = {
        "config": vars(args), "view_scores": view_rows, "metric_summary": summary,
        "conclusions": _conclusions(metric_rows, region_rows),
        "notes": [
            "D5 is model return list index 0 and final output; D1 is index 4.",
            "D1-D4 semantics are patch-upsampled then sliding-window Gaussian-fused full-volume probabilities, not raw low-resolution maps.",
            "Summary statistics are computed from per-case rows before case-level aggregation.",
            "Crop-outside voxels are rejected if they contain GT foreground; crop-space statistics use gt_background_valid only.",
        ],
    }
    (output_dir / "attention_probability_diagnostics.json").write_text(json.dumps(_json_safe(diagnostics), ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved diagnostics to {output_dir}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    return run(parse_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
