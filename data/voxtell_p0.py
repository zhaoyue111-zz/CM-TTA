"""VoxTell/P0 data utilities for the root CM-TTA path.

Only the balanced split's test cases are exposed. Labels are deliberately
loaded by the evaluation caller, never by the adaptation data path.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F

try:
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
    from nnunetv2.preprocessing.normalization.default_normalization_schemes import (
        ZScoreNormalization,
    )
except ModuleNotFoundError as error:  # Keep split validation importable.
    NibabelIOWithReorient = None
    crop_to_nonzero = None
    ZScoreNormalization = None
    _NNUNET_IMPORT_ERROR = error
else:
    _NNUNET_IMPORT_ERROR = None


P0 = "P0"
DEFAULT_SPLIT = Path("worst_zeroshot_split_p0") / "balanced_zeroshot_split.json"


def read_balanced_split(data_dir: str, split_file: Optional[str] = None) -> dict:
    root = Path(data_dir)
    path = Path(split_file).expanduser() if split_file else root / DEFAULT_SPLIT
    if not path.exists():
        raise FileNotFoundError(f"P0 split file does not exist: {path}")
    split = json.loads(path.read_text(encoding="utf-8"))
    if "test_cases" not in split:
        raise KeyError(f"{path} must contain test_cases")
    return split


def _case_name(case) -> str:
    if isinstance(case, dict):
        case = case.get("case_name", case.get("name", case.get("case")))
    if case is None:
        raise ValueError("Each test case must be a filename or case mapping")
    name = Path(str(case)).name
    return name if name.endswith(".nii.gz") else f"{name}.nii.gz"


def read_test_entries(
    data_dir: str, split_file: Optional[str] = None
) -> List[Tuple[Path, Path]]:
    root = Path(data_dir)
    split = read_balanced_split(data_dir, split_file)
    image_dir = root / "images" / P0
    label_dir = root / "labels" / P0
    entries = []
    for case in split["test_cases"]:
        name = _case_name(case)
        image_path = image_dir / name
        label_path = label_dir / name
        if not image_path.exists():
            raise FileNotFoundError(f"P0 test image does not exist: {image_path}")
        if not label_path.exists():
            raise FileNotFoundError(f"P0 test label does not exist: {label_path}")
        entries.append((image_path, label_path))
    if not entries:
        raise ValueError("The P0 test split is empty")
    return entries


def load_ras_image(path: str) -> np.ndarray:
    if _NNUNET_IMPORT_ERROR is not None:
        raise RuntimeError("VoxTell/P0 loading requires nnunetv2") from _NNUNET_IMPORT_ERROR
    return NibabelIOWithReorient().read_images([str(path)])[0]


def load_ras_label(path: str) -> np.ndarray:
    if _NNUNET_IMPORT_ERROR is not None:
        raise RuntimeError("VoxTell/P0 loading requires nnunetv2") from _NNUNET_IMPORT_ERROR
    return (NibabelIOWithReorient().read_images([str(path)])[0] > 0).astype(np.uint8)


def preprocess_for_voxtell(path: Path, predictor) -> Tuple[torch.Tensor, Tuple, Tuple[int, ...]]:
    """Use VoxTell's official RAS, nonzero-crop and z-score preprocessing."""
    image = load_ras_image(str(path))
    return predictor.preprocess(image)


def pad_to_patch_grid(
    volume: torch.Tensor, patch_size
) -> Tuple[torch.Tensor, torch.Tensor, Tuple[int, ...]]:
    """Right-pad (C,D,H,W) and return a matching valid voxel mask."""
    if volume.ndim != 4 or len(patch_size) != 3:
        raise ValueError(f"Expected volume (C,D,H,W), got {tuple(volume.shape)}")
    patch_size = tuple(int(value) for value in patch_size)
    original_shape = tuple(int(value) for value in volume.shape[-3:])
    target_shape = tuple(
        max(patch, ((size + patch - 1) // patch) * patch)
        for size, patch in zip(original_shape, patch_size)
    )
    padding = []
    for current, target in zip(reversed(original_shape), reversed(target_shape)):
        padding.extend((0, target - current))
    padded = F.pad(volume, padding, value=0.0).contiguous()
    valid = torch.zeros((1, *target_shape), dtype=torch.float32, device=volume.device)
    valid[(slice(None), *(slice(0, size) for size in original_shape))] = 1.0
    return padded, valid, original_shape


def nonoverlapping_patch_locations(volume_shape, patch_size):
    volume_shape = tuple(int(value) for value in volume_shape)
    patch_size = tuple(int(value) for value in patch_size)
    if any(size % patch for size, patch in zip(volume_shape, patch_size)):
        raise ValueError("Volume shape must be divisible by patch size")
    return [
        (d, h, w)
        for d in range(0, volume_shape[0], patch_size[0])
        for h in range(0, volume_shape[1], patch_size[1])
        for w in range(0, volume_shape[2], patch_size[2])
    ]


def extract_volume_patch(volume: torch.Tensor, location, patch_size) -> torch.Tensor:
    slices = tuple(
        slice(int(start), int(start) + int(size))
        for start, size in zip(location, patch_size)
    )
    return volume[(slice(None), *slices)].contiguous()


def make_case_patches(volume: torch.Tensor, patch_size):
    padded, valid, original_shape = pad_to_patch_grid(volume, patch_size)
    locations = nonoverlapping_patch_locations(padded.shape[-3:], patch_size)
    patches = [extract_volume_patch(padded, location, patch_size) for location in locations]
    valid_masks = [extract_volume_patch(valid, location, patch_size)[0] for location in locations]
    return patches, valid_masks, locations, original_shape
