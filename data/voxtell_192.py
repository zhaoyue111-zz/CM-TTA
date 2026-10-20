"""Manifest-driven offline data path for whole-case 192^3 VoxTell runs.

The adaptation path in :mod:`run_voxtell_192_comparison` only consumes the
arrays written here.  In particular it never crops, normalizes, or resizes a
case online.  Ground truth is intentionally kept out of the patch-building
API so it cannot accidentally become an adapter input.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import torch
import torch.nn.functional as F


TARGET_SHAPE = (192, 192, 192)


@dataclass(frozen=True)
class SourceCase:
    case_id: str
    image_path: Path
    gt_path: Path


@dataclass(frozen=True)
class PreprocessedCase:
    case_id: str
    artifact_path: Path
    image_path: Path
    gt_path: Path


def _manifest_rows(payload: object, path: Path) -> list[dict]:
    rows = payload.get("cases") if isinstance(payload, dict) else payload
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{path} must be a non-empty list or contain a cases list")
    if not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"Every case in {path} must be an object")
    return rows


def _resolve_manifest_path(value: object, manifest_path: Path, field: str) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError(f"Manifest field {field!r} must be a non-empty path string")
    path = Path(value).expanduser()
    return path if path.is_absolute() else (manifest_path.parent / path).resolve()


def load_source_manifest(path: str | Path, *, require_files: bool = True) -> list[SourceCase]:
    """Read ``case_id``, ``image_path``, and ``gt_path`` without layout assumptions."""
    manifest_path = Path(path).expanduser().resolve()
    rows = _manifest_rows(json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path)
    cases: list[SourceCase] = []
    seen: set[str] = set()
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every source case must have a non-empty case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id in source manifest: {case_id}")
        seen.add(case_id)
        image_path = _resolve_manifest_path(row.get("image_path"), manifest_path, "image_path")
        gt_path = _resolve_manifest_path(row.get("gt_path"), manifest_path, "gt_path")
        if require_files:
            for name, file_path in (("image", image_path), ("GT", gt_path)):
                if not file_path.is_file():
                    raise FileNotFoundError(f"{name} file for {case_id} does not exist: {file_path}")
        cases.append(SourceCase(case_id, image_path, gt_path))
    return cases


def load_preprocessed_manifest(
    path: str | Path, *, require_files: bool = True
) -> list[PreprocessedCase]:
    manifest_path = Path(path).expanduser().resolve()
    rows = _manifest_rows(json.loads(manifest_path.read_text(encoding="utf-8")), manifest_path)
    cases: list[PreprocessedCase] = []
    seen: set[str] = set()
    for row in rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("Every preprocessed case must have a non-empty case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate case_id in preprocessed manifest: {case_id}")
        seen.add(case_id)
        artifact = _resolve_manifest_path(
            row.get("preprocessed_path"), manifest_path, "preprocessed_path"
        )
        image_path = _resolve_manifest_path(row.get("image_path"), manifest_path, "image_path")
        gt_path = _resolve_manifest_path(row.get("gt_path"), manifest_path, "gt_path")
        if require_files:
            for name, file_path in (
                ("preprocessed artifact", artifact),
                ("source image", image_path),
                ("source GT", gt_path),
            ):
                if not file_path.is_file():
                    raise FileNotFoundError(f"{name} for {case_id} does not exist: {file_path}")
        cases.append(PreprocessedCase(case_id, artifact, image_path, gt_path))
    return cases


def bbox_slices(bbox: Sequence[Sequence[int]]) -> tuple[slice, slice, slice]:
    normalized = tuple(tuple(int(value) for value in axis) for axis in bbox)
    if len(normalized) != 3 or any(len(axis) != 2 for axis in normalized):
        raise ValueError(f"bbox must contain three [start, end] pairs, got {bbox}")
    if any(start < 0 or end <= start for start, end in normalized):
        raise ValueError(f"bbox contains an invalid interval: {bbox}")
    return tuple(slice(start, end) for start, end in normalized)


def crop_with_bbox(array: np.ndarray, bbox: Sequence[Sequence[int]]) -> np.ndarray:
    array = np.asarray(array)
    if array.ndim == 4 and array.shape[0] == 1:
        return array[(slice(None), *bbox_slices(bbox))]
    if array.ndim == 3:
        return array[bbox_slices(bbox)]
    raise ValueError(f"Expected (D,H,W) or (1,D,H,W), got {array.shape}")


def resize_image_and_gt(
    cropped_image: np.ndarray,
    cropped_gt: np.ndarray,
    target_shape: Sequence[int] = TARGET_SHAPE,
) -> tuple[np.ndarray, np.ndarray]:
    """Resize image with trilinear and binary GT with nearest interpolation."""
    target = tuple(int(value) for value in target_shape)
    if target != TARGET_SHAPE:
        raise ValueError(f"This experiment requires target_shape={TARGET_SHAPE}, got {target}")
    image = torch.as_tensor(np.asarray(cropped_image), dtype=torch.float32)
    gt = torch.as_tensor(np.asarray(cropped_gt), dtype=torch.float32)
    if image.ndim == 3:
        image = image.unsqueeze(0)
    if gt.ndim == 3:
        gt = gt.unsqueeze(0)
    if image.ndim != 4 or gt.ndim != 4 or image.shape[0] != 1 or gt.shape[0] != 1:
        raise ValueError("Cropped image and GT must have shape (1,D,H,W) or (D,H,W)")
    if tuple(image.shape) != tuple(gt.shape):
        raise ValueError(f"Cropped image/GT mismatch: {tuple(image.shape)} vs {tuple(gt.shape)}")
    image_192 = F.interpolate(
        image.unsqueeze(0), size=target, mode="trilinear", align_corners=False
    )[0].contiguous()
    gt_192 = F.interpolate(gt.unsqueeze(0), size=target, mode="nearest")[0]
    gt_192 = (gt_192 >= 0.5).to(torch.uint8).contiguous()
    return image_192.numpy().astype(np.float32, copy=False), gt_192.numpy().astype(np.uint8, copy=False)


def voxtell_crop_and_normalize(image: np.ndarray) -> tuple[np.ndarray, list[list[int]], tuple[int, int, int]]:
    """Apply the exact crop/normalization operations used by VoxTellPredictor."""
    try:
        from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
        from nnunetv2.preprocessing.normalization.default_normalization_schemes import (
            ZScoreNormalization,
        )
    except ModuleNotFoundError as error:
        raise RuntimeError("VoxTell preprocessing requires nnunetv2") from error
    image = np.asarray(image)
    if image.ndim == 3:
        image = image[None]
    if image.ndim != 4 or image.shape[0] != 1:
        raise ValueError(f"Expected one-channel RAS image, got {image.shape}")
    image = image.astype(np.float32, copy=True)
    original_shape = tuple(int(value) for value in image.shape[1:])
    cropped, _, bbox = crop_to_nonzero(image, None)
    normalized = ZScoreNormalization(intensityproperties={}).run(cropped, None)
    bbox_list = [[int(axis.start), int(axis.stop)] for axis in bbox_slices(bbox)]
    return normalized.astype(np.float32, copy=False), bbox_list, original_shape


def read_ras_case(
    image_path: str | Path,
    gt_path: str | Path,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Read image and GT through VoxTell's nnUNet RAS reader and check geometry."""
    try:
        import nibabel as nib
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    except ModuleNotFoundError as error:
        raise RuntimeError("RAS NIfTI loading requires nibabel and nnunetv2") from error
    image_path = Path(image_path)
    gt_path = Path(gt_path)
    image_nifti = nib.load(str(image_path))
    gt_nifti = nib.load(str(gt_path))
    if tuple(image_nifti.shape[:3]) != tuple(gt_nifti.shape[:3]):
        raise ValueError(f"Image/GT disk shape mismatch: {image_nifti.shape} vs {gt_nifti.shape}")
    if not np.allclose(image_nifti.affine, gt_nifti.affine):
        raise ValueError(f"Image/GT affine mismatch: {image_path} vs {gt_path}")
    reader = NibabelIOWithReorient()
    image, properties = reader.read_images([str(image_path)])
    gt, gt_properties = reader.read_images([str(gt_path)])
    if tuple(image.shape) != tuple(gt.shape):
        raise ValueError(f"RAS image/GT shape mismatch: {image.shape} vs {gt.shape}")
    if not np.allclose(
        properties["nibabel_stuff"]["reoriented_affine"],
        gt_properties["nibabel_stuff"]["reoriented_affine"],
    ):
        raise ValueError("RAS image/GT reoriented affine mismatch")
    return image, gt, properties


def foreground_count_metadata(
    case_id: str,
    original_gt: np.ndarray,
    cropped_gt: np.ndarray,
    gt_192: np.ndarray,
) -> dict[str, int | bool]:
    """Validate that a positive foreground survives crop and nearest resize."""
    original_count = int(np.count_nonzero(original_gt))
    cropped_count = int(np.count_nonzero(cropped_gt))
    resized_count = int(np.count_nonzero(gt_192))
    erased_by_crop = original_count > 0 and cropped_count == 0
    erased_by_resize = original_count > 0 and resized_count == 0
    if erased_by_crop:
        raise ValueError(
            f"Case {case_id}: original GT has {original_count} foreground voxels "
            "but the image-derived crop erased all foreground"
        )
    if erased_by_resize:
        raise ValueError(
            f"Case {case_id}: original GT has {original_count} foreground voxels "
            "but nearest resizing to 192^3 erased all foreground"
        )
    return {
        "original_foreground_voxel_count": original_count,
        "cropped_foreground_voxel_count": cropped_count,
        "gt_192_foreground_voxel_count": resized_count,
        "foreground_erased_by_crop": erased_by_crop,
        "foreground_erased_by_resize": erased_by_resize,
    }


def preprocess_source_case(
    case: SourceCase,
    output_dir: str | Path,
    foreground_labels: Iterable[int],
) -> dict:
    labels = tuple(sorted({int(value) for value in foreground_labels}))
    if not labels:
        raise ValueError("At least one foreground label value is required")
    image, raw_gt, properties = read_ras_case(case.image_path, case.gt_path)
    normalized_crop, bbox, original_shape = voxtell_crop_and_normalize(image)
    binary_gt = np.isin(raw_gt, labels).astype(np.uint8)
    gt_crop = crop_with_bbox(binary_gt, bbox)
    if tuple(gt_crop.shape) != tuple(normalized_crop.shape):
        raise RuntimeError("The image bbox did not produce an aligned GT crop")
    image_192, gt_192 = resize_image_and_gt(normalized_crop, gt_crop)
    if image_192.shape != (1, *TARGET_SHAPE) or image_192.dtype != np.float32:
        raise RuntimeError("Offline image is not float32 (1,192,192,192)")
    if gt_192.shape != (1, *TARGET_SHAPE) or gt_192.dtype != np.uint8:
        raise RuntimeError("Offline GT is not uint8 (1,192,192,192)")
    if not set(np.unique(gt_192)).issubset({0, 1}):
        raise RuntimeError("Nearest-resized GT is not binary")
    foreground_metadata = foreground_count_metadata(
        case.case_id, binary_gt, gt_crop, gt_192
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    artifact_path = output_dir / f"{case.case_id}.npz"
    np.savez_compressed(
        artifact_path,
        image_192=image_192,
        gt_192=gt_192,
        gt_cropped_native=gt_crop.astype(np.uint8, copy=False),
        image_cropped_native=normalized_crop.astype(np.float32, copy=False),
        original_shape=np.asarray(original_shape, dtype=np.int64),
        bbox=np.asarray(bbox, dtype=np.int64),
        case_id=np.asarray(case.case_id),
        image_path=np.asarray(str(case.image_path.resolve())),
        gt_path=np.asarray(str(case.gt_path.resolve())),
        foreground_labels=np.asarray(labels, dtype=np.int64),
        original_foreground_voxel_count=np.asarray(
            foreground_metadata["original_foreground_voxel_count"], dtype=np.int64
        ),
        cropped_foreground_voxel_count=np.asarray(
            foreground_metadata["cropped_foreground_voxel_count"], dtype=np.int64
        ),
        gt_192_foreground_voxel_count=np.asarray(
            foreground_metadata["gt_192_foreground_voxel_count"], dtype=np.int64
        ),
        foreground_erased_by_crop=np.asarray(
            foreground_metadata["foreground_erased_by_crop"], dtype=np.bool_
        ),
        foreground_erased_by_resize=np.asarray(
            foreground_metadata["foreground_erased_by_resize"], dtype=np.bool_
        ),
        original_affine=np.asarray(
            properties["nibabel_stuff"]["original_affine"], dtype=np.float64
        ),
        reoriented_affine=np.asarray(
            properties["nibabel_stuff"]["reoriented_affine"], dtype=np.float64
        ),
    )
    return {
        "case_id": case.case_id,
        "preprocessed_path": str(artifact_path.resolve()),
        "image_path": str(case.image_path.resolve()),
        "gt_path": str(case.gt_path.resolve()),
        "original_shape": list(original_shape),
        "bbox": bbox,
        "cropped_shape": list(normalized_crop.shape[-3:]),
        "target_shape": list(TARGET_SHAPE),
        "foreground_labels": list(labels),
        **foreground_metadata,
    }


def load_case_artifact(case: PreprocessedCase | str | Path) -> dict:
    artifact_path = case.artifact_path if isinstance(case, PreprocessedCase) else Path(case)
    with np.load(artifact_path, allow_pickle=False) as archive:
        result = {key: archive[key] for key in archive.files}
    result["case_id"] = str(result["case_id"].item())
    result["image_path"] = Path(str(result["image_path"].item()))
    result["gt_path"] = Path(str(result["gt_path"].item()))
    result["original_shape"] = tuple(int(value) for value in result["original_shape"])
    result["bbox"] = [[int(value) for value in axis] for axis in result["bbox"]]
    required_foreground_fields = (
        "original_foreground_voxel_count",
        "cropped_foreground_voxel_count",
        "gt_192_foreground_voxel_count",
        "foreground_erased_by_crop",
        "foreground_erased_by_resize",
        "gt_cropped_native",
    )
    missing = [field for field in required_foreground_fields if field not in result]
    if missing:
        raise ValueError(
            f"Offline artifact {artifact_path} lacks foreground safeguards {missing}; "
            "rerun preprocess_voxtell_192.py"
        )
    for field in required_foreground_fields[:3]:
        result[field] = int(result[field].item())
    for field in required_foreground_fields[3:5]:
        result[field] = bool(result[field].item())
    image_192 = result["image_192"]
    gt_192 = result["gt_192"]
    if image_192.shape != (1, *TARGET_SHAPE) or image_192.dtype != np.float32:
        raise ValueError(f"Invalid offline image in {artifact_path}: {image_192.shape}/{image_192.dtype}")
    if gt_192.shape != (1, *TARGET_SHAPE) or gt_192.dtype != np.uint8:
        raise ValueError(f"Invalid offline GT in {artifact_path}: {gt_192.shape}/{gt_192.dtype}")
    if not set(np.unique(gt_192)).issubset({0, 1}):
        raise ValueError(f"Offline GT is not binary in {artifact_path}")
    expected_crop = tuple(end - start for start, end in result["bbox"])
    if tuple(result["image_cropped_native"].shape[-3:]) != expected_crop:
        raise ValueError(f"Offline native crop/bbox mismatch in {artifact_path}")
    cropped_gt = result["gt_cropped_native"]
    if tuple(cropped_gt.shape[-3:]) != expected_crop:
        raise ValueError(f"Offline native GT crop/bbox mismatch in {artifact_path}")
    if cropped_gt.dtype != np.uint8 or not set(np.unique(cropped_gt)).issubset({0, 1}):
        raise ValueError(f"Offline native cropped GT is not binary uint8 in {artifact_path}")
    observed_cropped = int(np.count_nonzero(cropped_gt))
    observed_192 = int(np.count_nonzero(gt_192))
    if observed_cropped != result["cropped_foreground_voxel_count"]:
        raise ValueError(f"Offline cropped foreground count mismatch in {artifact_path}")
    if observed_192 != result["gt_192_foreground_voxel_count"]:
        raise ValueError(f"Offline 192^3 foreground count mismatch in {artifact_path}")
    original_count = result["original_foreground_voxel_count"]
    if original_count < 0:
        raise ValueError(f"Offline original foreground count is negative in {artifact_path}")
    expected_erased_by_crop = original_count > 0 and observed_cropped == 0
    expected_erased_by_resize = original_count > 0 and observed_192 == 0
    if expected_erased_by_crop:
        raise ValueError(
            f"Case {result['case_id']}: offline image crop erased all foreground"
        )
    if expected_erased_by_resize:
        raise ValueError(
            f"Case {result['case_id']}: offline nearest resize erased all foreground"
        )
    if (
        expected_erased_by_crop != result["foreground_erased_by_crop"]
        or expected_erased_by_resize != result["foreground_erased_by_resize"]
    ):
        raise ValueError(f"Offline foreground-erasure flags mismatch in {artifact_path}")
    return result


def make_single_spatial_patch(image_192: np.ndarray | torch.Tensor):
    """Build the only legal adaptation layout for this experiment."""
    image = torch.as_tensor(image_192, dtype=torch.float32).contiguous()
    if tuple(image.shape) != (1, *TARGET_SHAPE):
        raise ValueError(f"Expected offline image shape (1,{TARGET_SHAPE}), got {tuple(image.shape)}")
    patches = [image]
    valid_masks = [torch.ones(TARGET_SHAPE, dtype=torch.float32)]
    locations = [(0, 0, 0)]
    assert len(patches) == len(valid_masks) == len(locations) == 1
    return patches, valid_masks, locations


def restore_probability_to_original(
    probability_192: np.ndarray | torch.Tensor,
    bbox: Sequence[Sequence[int]],
    original_shape: Sequence[int],
) -> np.ndarray:
    """Trilinearly restore soft probability, then insert it into full RAS space."""
    probability = torch.as_tensor(probability_192, dtype=torch.float32)
    while probability.ndim > 3 and probability.shape[0] == 1:
        probability = probability[0]
    if tuple(probability.shape) != TARGET_SHAPE:
        raise ValueError(f"Expected 192^3 probability, got {tuple(probability.shape)}")
    crop_shape = tuple(end - start for start, end in (
        (int(axis[0]), int(axis[1])) for axis in bbox
    ))
    restored_crop = F.interpolate(
        probability[None, None], size=crop_shape, mode="trilinear", align_corners=False
    )[0, 0].cpu().numpy()
    full = np.zeros(tuple(int(value) for value in original_shape), dtype=np.float32)
    full[bbox_slices(bbox)] = restored_crop
    return full
