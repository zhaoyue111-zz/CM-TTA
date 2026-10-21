from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from method.voxtell_cmtta import tdc_from_components, tdc_patch_components
from prototype_diagnostic.extract_features import (
    D5_PREHEAD_MODULE,
    SlidingD5FeatureExtractor,
    apply_intensity_view,
    deterministic_view_params,
    prepare_case,
)
from prototype_diagnostic.utils import case_name


def load_view_params_replay(
    results_json: str | Path | None,
) -> tuple[dict[str, list[dict[str, float]]], list[str]]:
    """Load augmentation parameters only; adapted-run metrics are ignored."""
    if results_json is None:
        return {}, []
    path = Path(results_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a cases list")

    views_by_case: dict[str, list[dict[str, float]]] = {}
    order: list[str] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("Every results row must be a mapping")
        raw_name = row.get("basename", row.get("case_name", row.get("case")))
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise ValueError("Every results row must contain a non-empty case basename")
        name = case_name(raw_name)
        if name in order:
            raise ValueError(f"Duplicate case in results.json: {name}")
        order.append(name)
        trace = row.get("adaptation_trace")
        params = trace.get("view_params") if isinstance(trace, dict) else None
        if isinstance(params, list) and params:
            normalized = []
            for param in params:
                if not isinstance(param, dict) or not {"scale", "offset"} <= set(param):
                    raise ValueError(f"Invalid view params for {name}")
                normalized.append(
                    {"scale": float(param["scale"]), "offset": float(param["offset"])}
                )
            views_by_case[name] = normalized
    return views_by_case, order


def resolve_view_params(
    case_names: Iterable[str],
    replay_views: dict[str, list[dict[str, float]]],
    *,
    num_views: int,
    seed: int,
) -> dict[str, list[dict[str, float]]]:
    resolved = {}
    for raw_name in case_names:
        name = case_name(raw_name)
        params = replay_views.get(name)
        if params is None:
            params = deterministic_view_params(name, num_views, seed)
        if len(params) != num_views:
            raise ValueError(
                f"Replayed {name} has {len(params)} views, expected --num_views={num_views}"
            )
        if not params or params[0] != {"scale": 1.0, "offset": 0.0}:
            raise ValueError(f"{name} view 0 must be scale=1 and offset=0")
        resolved[name] = params
    return resolved


def compute_source_tdc_for_case(
    prepared,
    params: list[dict[str, float]],
    extractor: SlidingD5FeatureExtractor,
    text_embedding: torch.Tensor,
) -> dict[str, object]:
    """Recompute full-volume D5-D2 TDC for every view of one source case."""
    view_rows = []
    for view_index, param in enumerate(params):
        viewed = apply_intensity_view(prepared.data, param, view_index)
        extracted = extractor.extract(
            viewed,
            text_embedding,
            capture_feature=False,
            return_decoder_logits=True,
        )
        if extracted.decoder_logits is None or len(extracted.decoder_logits) < 4:
            raise RuntimeError("Source TDC extraction did not return D5-D2 logits")
        decoder_outputs = [value[None, None] for value in extracted.decoder_logits[:4]]
        valid = torch.ones_like(decoder_outputs[0][:, 0], dtype=torch.bool)
        components = tdc_patch_components(decoder_outputs, valid)
        tdc, pair_dice, pair_valid = tdc_from_components(**components)
        view_rows.append(
            {
                "view_index": int(view_index),
                "scale": float(param["scale"]),
                "offset": float(param["offset"]),
                "tdc": float(tdc[0]),
                "pair_dice": pair_dice[0].detach().cpu().tolist(),
                "pair_valid": pair_valid[0].detach().cpu().tolist(),
            }
        )
    scores = [float(row["tdc"]) for row in view_rows]
    if not scores:
        raise ValueError("At least one source TDC view is required")
    return {
        "case_name": prepared.image_path.name,
        "view_count": len(view_rows),
        "views": view_rows,
        "original_tdc": scores[0],
        "mean_tdc": float(np.mean(scores)),
        "max_tdc": float(np.max(scores)),
        "memory_ranking_metric": "original_tdc",
        "memory_ranking_value": scores[0],
    }


def recompute_source_tdc(
    predictor,
    text_embedding: torch.Tensor,
    entries_by_name: dict[str, tuple[Path, Path]],
    case_names: Iterable[str],
    view_params_by_case: dict[str, list[dict[str, float]]],
) -> dict[str, dict[str, object]]:
    extractor = SlidingD5FeatureExtractor(predictor)
    records: dict[str, dict[str, object]] = {}
    names = [case_name(name) for name in case_names]
    for index, name in enumerate(names, start=1):
        prepared = prepare_case(predictor, entries_by_name[name][0])
        record = compute_source_tdc_for_case(
            prepared, view_params_by_case[name], extractor, text_embedding
        )
        records[name] = record
        print(
            f"[prototype-diagnostic] source TDC {index}/{len(names)} {name}: "
            f"original={record['original_tdc']:.6f} "
            f"mean={record['mean_tdc']:.6f} max={record['max_tdc']:.6f}"
        )
    return records


def select_memory_cases(
    available_case_names: Iterable[str],
    manual_cases: list[str] | None,
    source_tdc_by_case: dict[str, dict[str, object]],
    top_ratio: float,
) -> list[str]:
    available = [case_name(name) for name in available_case_names]
    available_set = set(available)
    if manual_cases:
        selected = [case_name(name) for name in manual_cases]
        if len(selected) != len(set(selected)):
            raise ValueError("--memory_cases contains duplicates")
    else:
        if not 0.0 < float(top_ratio) <= 1.0:
            raise ValueError("--memory_top_ratio must be in (0, 1]")
        candidates = list(available)
        missing_tdc = [name for name in candidates if name not in source_tdc_by_case]
        if missing_tdc:
            raise ValueError(
                "Source TDC was not recomputed for: "
                + ", ".join(missing_tdc[:5])
            )
        order_index = {name: index for index, name in enumerate(available)}
        candidates.sort(
            key=lambda name: (
                -float(source_tdc_by_case[name]["memory_ranking_value"]),
                order_index[name],
            )
        )
        count = max(1, int(math.ceil(len(candidates) * float(top_ratio))))
        selected = candidates[:count]

    unknown = [name for name in selected if name not in available_set]
    if unknown:
        raise ValueError("Unknown memory cases: " + ", ".join(unknown))
    return selected


def _component_regions(mask: np.ndarray, min_region_voxels: int):
    structure = ndimage.generate_binary_structure(3, 3)
    labels, count = ndimage.label(np.asarray(mask, dtype=bool), structure=structure)
    regions = []
    for component_id in range(1, int(count) + 1):
        region = labels == component_id
        size = int(region.sum())
        if size >= min_region_voxels:
            regions.append((component_id, region, size))
    return regions


def _eroded_core(region: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return region
    core = ndimage.binary_erosion(
        region,
        structure=ndimage.generate_binary_structure(3, 3),
        iterations=int(iterations),
        border_value=0,
    )
    # Tiny but otherwise valid regions should not silently disappear.
    return core if core.any() else region


def aggregate_normalized_view_features(features: Iterable[torch.Tensor]) -> torch.Tensor:
    """Normalize each voxel in every view before equal-weight view aggregation."""
    normalized = []
    expected_shape = None
    for feature in features:
        if feature.ndim != 4:
            raise ValueError(f"Expected (C,D,H,W) feature, got {tuple(feature.shape)}")
        if expected_shape is None:
            expected_shape = tuple(feature.shape)
        elif tuple(feature.shape) != expected_shape:
            raise ValueError("All feature views must have the same shape")
        normalized.append(F.normalize(feature.float(), dim=0))
    if not normalized:
        raise ValueError("At least one feature view is required")
    return torch.stack(normalized, dim=0).mean(dim=0)


def region_prototype(feature_mean: torch.Tensor, core: np.ndarray) -> torch.Tensor:
    core_tensor = torch.as_tensor(core, dtype=torch.bool)
    if tuple(core_tensor.shape) != tuple(feature_mean.shape[1:]) or not bool(core_tensor.any()):
        raise ValueError("Prototype core must be non-empty and match feature spatial shape")
    return F.normalize(feature_mean[:, core_tensor].mean(dim=1).float(), dim=0)


def build_prototype_bank(
    predictor,
    text_embedding: torch.Tensor,
    entries_by_name: dict[str, tuple[Path, Path]],
    memory_cases: list[str],
    source_tdc_by_case: dict[str, dict[str, object]],
    view_params_by_case: dict[str, list[dict[str, float]]],
    output_path: Path,
    *,
    num_views: int,
    view_support: float,
    min_region_voxels: int,
    core_erosion: int,
) -> dict[str, object]:
    if not 0.0 < float(view_support) <= 1.0:
        raise ValueError("--view_support must be in (0, 1]")
    if min_region_voxels < 1 or core_erosion < 0:
        raise ValueError("Region size must be positive and erosion non-negative")

    extractor = SlidingD5FeatureExtractor(predictor)
    prototypes: list[dict[str, object]] = []
    case_summaries = []
    feature_channels = None
    for case_index, name in enumerate(memory_cases, start=1):
        image_path, _label_path = entries_by_name[name]
        prepared = prepare_case(predictor, image_path)
        params = view_params_by_case[name]

        support_sum = torch.zeros(tuple(prepared.data.shape[-3:]), dtype=torch.int16)
        normalized_feature_sum = None
        for view_index, param in enumerate(params):
            viewed = apply_intensity_view(prepared.data, param, view_index)
            extracted = extractor.extract(viewed, text_embedding)
            support_sum += (extracted.probability > 0.5).to(torch.int16)
            if extracted.feature is None:
                raise RuntimeError("D5 feature hook returned no feature")
            normalized = F.normalize(extracted.feature.float(), dim=0)
            if normalized_feature_sum is None:
                normalized_feature_sum = torch.zeros_like(normalized)
            normalized_feature_sum.add_(normalized)
        if normalized_feature_sum is None:
            raise RuntimeError(f"No feature views extracted for {name}")
        feature_mean = normalized_feature_sum / float(len(params))
        feature_channels = int(feature_mean.shape[0])
        support = support_sum.float() / float(len(params))
        consensus = support.numpy() >= float(view_support)
        regions = _component_regions(consensus, int(min_region_voxels))
        retained_voxels = 0
        for region_id, region, region_size in regions:
            core = _eroded_core(region, int(core_erosion))
            feature = region_prototype(feature_mean, core)
            if not bool(torch.isfinite(feature).all()) or float(feature.norm()) == 0.0:
                continue
            retained_voxels += region_size
            prototypes.append(
                {
                    "feature": feature.cpu(),
                    "case_name": name,
                    "region_id": int(region_id),
                    "region_size": int(region_size),
                    "core_size": int(core.sum()),
                    "tdc": source_tdc_by_case[name]["memory_ranking_value"],
                    "tdc_original": source_tdc_by_case[name]["original_tdc"],
                    "tdc_mean": source_tdc_by_case[name]["mean_tdc"],
                    "tdc_max": source_tdc_by_case[name]["max_tdc"],
                    "view_support_mean": float(support.numpy()[region].mean()),
                }
            )
        case_summaries.append(
            {
                "case_name": name,
                "tdc_original": source_tdc_by_case[name]["original_tdc"],
                "tdc_mean": source_tdc_by_case[name]["mean_tdc"],
                "tdc_max": source_tdc_by_case[name]["max_tdc"],
                "memory_ranking_metric": "original_tdc",
                "memory_ranking_value": source_tdc_by_case[name][
                    "memory_ranking_value"
                ],
                "num_views": len(params),
                "consensus_foreground_voxels": int(consensus.sum()),
                "retained_region_count": len(regions),
                "retained_region_voxels": retained_voxels,
            }
        )
        print(
            f"[prototype-diagnostic] bank {case_index}/{len(memory_cases)} {name}: "
            f"regions={len(regions)} consensus_voxels={int(consensus.sum())}"
        )

    if not prototypes:
        raise RuntimeError(
            "Prototype bank is empty; lower --view_support/--min_region_voxels "
            "or choose memory cases with foreground predictions"
        )
    bank = {
        "format_version": 1,
        "feature_source": D5_PREHEAD_MODULE,
        "feature_aggregation": (
            "per-view voxel-wise channel L2 normalization, equal view mean, "
            "region mean, final L2 normalization"
        ),
        "tdc_source": "same frozen source VoxTell, prompt, and view parameters",
        "memory_ranking_metric": "original_tdc",
        "feature_channels": feature_channels,
        "memory_cases": list(memory_cases),
        "view_support": float(view_support),
        "min_region_voxels": int(min_region_voxels),
        "core_erosion": int(core_erosion),
        "num_views": int(num_views),
        "case_summaries": case_summaries,
        "prototypes": prototypes,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(bank, output_path)
    print(
        f"[prototype-diagnostic] saved {len(prototypes)} prototypes "
        f"with shape ({len(prototypes)}, {feature_channels}) to {output_path}"
    )
    return bank
