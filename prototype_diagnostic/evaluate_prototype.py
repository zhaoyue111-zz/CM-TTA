from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from prototype_diagnostic.extract_features import SlidingD5FeatureExtractor, prepare_case
from prototype_diagnostic.utils import (
    bbox_slices,
    binary_ranking_metrics,
    distribution_stats,
    foreground_mask,
    load_ras_image,
    load_ras_label_values,
    restore_crop,
    save_case_figure,
    save_float_nifti_model_space,
    top_fraction_enrichment,
    top_fraction_mask,
    write_json,
)


def prototype_similarity_maps(
    feature: torch.Tensor,
    prototypes: list[dict[str, object]],
    query_case: str,
    topk: int = 5,
    voxel_chunk_size: int = 262_144,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, int]]:
    if feature.ndim != 4:
        raise ValueError(f"Expected feature (C,D,H,W), got {tuple(feature.shape)}")
    if topk < 1 or voxel_chunk_size < 1:
        raise ValueError("topk and voxel_chunk_size must be positive")
    external = [item for item in prototypes if item["case_name"] != query_case]
    excluded = len(prototypes) - len(external)
    if not external:
        raise RuntimeError(
            f"Leave-one-case-out leaves no prototypes for query {query_case}"
        )
    matrix = torch.stack(
        [torch.as_tensor(item["feature"], dtype=torch.float32) for item in external]
    )
    matrix = F.normalize(matrix, dim=1)
    if matrix.shape[1] != feature.shape[0]:
        raise ValueError(
            f"Prototype channels {matrix.shape[1]} != feature channels {feature.shape[0]}"
        )
    voxels = feature.float().reshape(feature.shape[0], -1).T
    actual_topk = min(int(topk), int(matrix.shape[0]))
    max_parts = []
    topk_parts = []
    for start in range(0, voxels.shape[0], int(voxel_chunk_size)):
        chunk = F.normalize(voxels[start : start + voxel_chunk_size], dim=1)
        similarities = chunk @ matrix.T
        max_parts.append(similarities.max(dim=1).values)
        topk_parts.append(similarities.topk(actual_topk, dim=1).values.mean(dim=1))
    spatial = tuple(int(v) for v in feature.shape[1:])
    return (
        torch.cat(max_parts).reshape(spatial),
        torch.cat(topk_parts).reshape(spatial),
        {
            "bank_prototype_count": len(prototypes),
            "excluded_self_prototype_count": excluded,
            "matched_prototype_count": len(external),
            "requested_topk": int(topk),
            "actual_topk": actual_topk,
        },
    )


def _map_metrics(name: str, values: np.ndarray, target: np.ndarray) -> dict[str, object]:
    target = np.asarray(target, dtype=bool)
    foreground = distribution_stats(values[target])
    background = distribution_stats(values[~target])
    ranking = binary_ranking_metrics(values, target)
    gap = None
    if foreground["mean"] is not None and background["mean"] is not None:
        gap = float(foreground["mean"] - background["mean"])
    result: dict[str, object] = {
        "gt_foreground": foreground,
        "gt_background": background,
        "foreground_background_mean_gap": gap,
        **ranking,
    }
    if name.startswith("proto_"):
        result["rank_enrichment"] = top_fraction_enrichment(values, target)
    return result


def evaluate_query_case(
    predictor,
    text_embedding: torch.Tensor,
    image_path: Path,
    label_path: Path,
    bank: dict[str, object],
    output_dir: Path,
    *,
    topk: int,
    voxel_chunk_size: int,
    foreground_labels: list[int] | tuple[int, ...] = (1,),
) -> dict[str, object]:
    name = image_path.name
    prepared = prepare_case(predictor, image_path)
    extractor = SlidingD5FeatureExtractor(predictor)
    extracted = extractor.extract(prepared.data, text_embedding)
    if extracted.feature is None:
        raise RuntimeError("D5 feature hook returned no query feature")
    similarity_max, similarity_topk, match_info = prototype_similarity_maps(
        extracted.feature,
        bank["prototypes"],
        name,
        topk=topk,
        voxel_chunk_size=voxel_chunk_size,
    )

    target_full = foreground_mask(
        load_ras_label_values(label_path), foreground_labels
    )
    if target_full.shape != prepared.original_shape:
        raise ValueError(
            f"GT/source shape mismatch for {name}: {target_full.shape} vs "
            f"{prepared.original_shape}"
        )
    crop = bbox_slices(prepared.bbox)
    target = target_full[crop]
    outside_crop_foreground = int(target_full.sum()) - int(target.sum())
    if outside_crop_foreground:
        raise RuntimeError(
            f"GT foreground falls outside VoxTell's nonzero image crop for {name}: "
            f"{outside_crop_foreground} voxels"
        )
    probability = extracted.probability.numpy()
    sim_max = similarity_max.numpy()
    sim_topk = similarity_topk.numpy()
    expected = tuple(int(v) for v in prepared.data.shape[-3:])
    if not (target.shape == probability.shape == sim_max.shape == sim_topk.shape == expected):
        raise RuntimeError(
            f"Spatial alignment failure for {name}: target={target.shape}, "
            f"prob={probability.shape}, max={sim_max.shape}, topk={sim_topk.shape}, "
            f"data={expected}"
        )

    metrics: dict[str, object] = {
        "case_name": name,
        "foreground_labels": [int(value) for value in foreground_labels],
        "roi_shape": [int(value) for value in prepared.original_shape],
        "preprocessed_shape": [int(value) for value in prepared.preprocessed_shape],
        "bbox": [[int(value) for value in bounds] for bounds in prepared.bbox],
        "sliding_patch_count": int(extracted.sliding_patch_count),
        "gt_tumor_voxels": int(target_full.sum()),
        "tumor_prevalence": float(target_full.mean()),
        "gt_foreground_voxels": int(target.sum()),
        "gt_foreground_voxels_outside_preprocessed_crop": outside_crop_foreground,
        "evaluated_voxels": int(target.size),
        "pred_fg_voxels": int((probability > 0.5).sum()),
        "zero_foreground_case": bool((probability > 0.5).sum() == 0),
        **match_info,
        "proto_max": _map_metrics("proto_max", sim_max, target),
        "proto_topk": _map_metrics("proto_topk", sim_topk, target),
        "seg_probability": _map_metrics("seg_probability", probability, target),
    }

    full_probability = restore_crop(
        probability, prepared.bbox, prepared.original_shape, fill_value=0.0
    )
    full_max = restore_crop(sim_max, prepared.bbox, prepared.original_shape, fill_value=0.0)
    full_topk = restore_crop(
        sim_topk, prepared.bbox, prepared.original_shape, fill_value=0.0
    )
    top_one_crop = top_fraction_mask(sim_max, 0.01)
    top_one_full = restore_crop(
        top_one_crop, prepared.bbox, prepared.original_shape, fill_value=False
    )

    case_dir = output_dir / name.removesuffix(".nii.gz")
    case_dir.mkdir(parents=True, exist_ok=True)
    save_float_nifti_model_space(
        full_probability, image_path, case_dir / "seg_prob.nii.gz"
    )
    save_float_nifti_model_space(
        full_max, image_path, case_dir / "proto_sim_max.nii.gz"
    )
    save_float_nifti_model_space(
        full_topk, image_path, case_dir / "proto_sim_topk.nii.gz"
    )
    image_full = np.squeeze(load_ras_image(str(image_path)))
    slice_index = save_case_figure(
        case_dir / "diagnostic.png",
        image_full,
        target_full,
        full_probability,
        full_max,
        full_topk,
        top_one_full,
    )
    metrics["representative_axial_slice"] = slice_index
    metrics["evaluation_space"] = (
        "official VoxTell preprocessed space; outputs restored to liver cuboid ROI"
    )
    write_json(case_dir / "metrics.json", metrics)
    return metrics


def _aggregate(rows: list[dict[str, object]], path: tuple[str, ...]) -> dict[str, object]:
    values = []
    for row in rows:
        value: object = row
        for key in path:
            value = value[key] if isinstance(value, dict) else None
        if value is not None and np.isfinite(float(value)):
            values.append(float(value))
    if not values:
        return {"count": 0, "mean": None, "std": None, "median": None}
    array = np.asarray(values, dtype=np.float64)
    return {
        "count": len(values),
        "mean": float(array.mean()),
        "std": float(array.std()),
        "median": float(np.median(array)),
    }


def summarize_query_groups(rows: list[dict[str, object]]) -> dict[str, object]:
    def summarize(group_rows: list[dict[str, object]]) -> dict[str, object]:
        paths = {
            "proto_max_auroc": ("proto_max", "auroc"),
            "proto_max_auprc": ("proto_max", "auprc"),
            "proto_max_foreground_background_gap": (
                "proto_max",
                "foreground_background_mean_gap",
            ),
            "proto_topk_auroc": ("proto_topk", "auroc"),
            "proto_topk_auprc": ("proto_topk", "auprc"),
            "proto_topk_foreground_background_gap": (
                "proto_topk",
                "foreground_background_mean_gap",
            ),
            "seg_probability_auroc": ("seg_probability", "auroc"),
            "seg_probability_auprc": ("seg_probability", "auprc"),
            "seg_probability_foreground_background_gap": (
                "seg_probability",
                "foreground_background_mean_gap",
            ),
        }
        return {
            "case_count": len(group_rows),
            "case_names": [row["case_name"] for row in group_rows],
            "metrics": {name: _aggregate(group_rows, path) for name, path in paths.items()},
        }

    zero = [row for row in rows if row["zero_foreground_case"]]
    normal = [row for row in rows if not row["zero_foreground_case"]]
    return {
        "all_query_cases": summarize(rows),
        "normal_foreground_cases": summarize(normal),
        "zero_foreground_cases": summarize(zero),
        "group_definition": {
            "normal_foreground_cases": "pred_fg_voxels > 0",
            "zero_foreground_cases": "pred_fg_voxels == 0",
        },
    }
