from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from scipy import ndimage

from prototype_diagnostic.extract_features import (
    D5_PREHEAD_MODULE,
    SlidingD5FeatureExtractor,
    apply_intensity_view,
    deterministic_view_params,
    prepare_case,
)
from prototype_diagnostic.utils import case_name


def load_results_metadata(
    results_json: str | Path | None,
) -> tuple[dict[str, float], dict[str, list[dict[str, float]]], list[str]]:
    if results_json is None:
        return {}, {}, []
    path = Path(results_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(rows, list):
        raise ValueError(f"{path} must contain a cases list")

    tdc_by_case: dict[str, float] = {}
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
        selection = trace.get("view_selection") if isinstance(trace, dict) else None
        tdc = None
        if isinstance(selection, dict):
            values = selection.get("tdc")
            if isinstance(values, list) and values and values[0] is not None:
                # Case-level screening uses the unaugmented view's TDC, not the
                # oracle/selected augmentation and never any GT measurement.
                tdc = float(values[0])
        for key in ("tdc", "TDC", "case_tdc"):
            if tdc is None and row.get(key) is not None:
                tdc = float(row[key])
        if tdc is not None:
            if not np.isfinite(tdc):
                raise ValueError(f"Non-finite TDC for {name}")
            tdc_by_case[name] = tdc

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
    return tdc_by_case, views_by_case, order


def select_memory_cases(
    available_case_names: Iterable[str],
    manual_cases: list[str] | None,
    tdc_by_case: dict[str, float],
    top_ratio: float,
    results_order: list[str],
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
        candidates = [name for name in results_order if name in available_set]
        missing_tdc = [name for name in candidates if name not in tdc_by_case]
        if missing_tdc:
            raise ValueError(
                "TDC screening requested but results.json lacks case-level TDC for: "
                + ", ".join(missing_tdc[:5])
            )
        if not candidates:
            raise ValueError("No split cases with TDC values were found in results.json")
        order_index = {name: index for index, name in enumerate(results_order)}
        candidates.sort(key=lambda name: (-tdc_by_case[name], order_index[name]))
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


def build_prototype_bank(
    predictor,
    text_embedding: torch.Tensor,
    entries_by_name: dict[str, tuple[Path, Path]],
    memory_cases: list[str],
    tdc_by_case: dict[str, float],
    replay_views: dict[str, list[dict[str, float]]],
    output_path: Path,
    *,
    num_views: int,
    view_support: float,
    min_region_voxels: int,
    core_erosion: int,
    seed: int,
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
        params = replay_views.get(name)
        if params is None:
            params = deterministic_view_params(name, num_views, seed)
        elif len(params) != num_views:
            raise ValueError(
                f"Replayed {name} has {len(params)} views, expected --num_views={num_views}"
            )

        support_sum = torch.zeros(tuple(prepared.data.shape[-3:]), dtype=torch.int16)
        feature_sum = None
        for view_index, param in enumerate(params):
            viewed = apply_intensity_view(prepared.data, param, view_index)
            extracted = extractor.extract(viewed, text_embedding)
            support_sum += (extracted.probability > 0.5).to(torch.int16)
            if feature_sum is None:
                feature_sum = torch.zeros_like(extracted.feature, dtype=torch.float32)
            feature_sum.add_(extracted.feature)
        if feature_sum is None:
            raise RuntimeError(f"No views extracted for {name}")
        feature_mean = feature_sum / float(len(params))
        feature_channels = int(feature_mean.shape[0])
        support = support_sum.float() / float(len(params))
        consensus = support.numpy() >= float(view_support)
        regions = _component_regions(consensus, int(min_region_voxels))
        retained_voxels = 0
        for region_id, region, region_size in regions:
            core = _eroded_core(region, int(core_erosion))
            core_tensor = torch.from_numpy(core)
            feature = feature_mean[:, core_tensor].mean(dim=1)
            feature = F.normalize(feature.float(), dim=0)
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
                    "tdc": tdc_by_case.get(name),
                    "view_support_mean": float(support.numpy()[region].mean()),
                }
            )
        case_summaries.append(
            {
                "case_name": name,
                "tdc": tdc_by_case.get(name),
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
