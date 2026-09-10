"""P0 target-domain data for VoxTell SFDA.

The adaptation split is intentionally fixed to the P0 protocol.  In
particular, this module never discovers cases from a CSV or from a directory
listing: both train and test cases must come from the required split JSON.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

try:
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    from nnunetv2.preprocessing.cropping.cropping import crop_to_nonzero
    from nnunetv2.preprocessing.normalization.default_normalization_schemes import (
        ZScoreNormalization,
    )
    _NNUNET_IMPORT_ERROR = None
except ModuleNotFoundError as error:  # Keep split/label-safety checks importable.
    NibabelIOWithReorient = None
    crop_to_nonzero = None
    ZScoreNormalization = None
    _NNUNET_IMPORT_ERROR = error


P0 = "P0"
P0_SPLIT_RELATIVE = Path("worst_zeroshot_split_p0") / "worst_zeroshot_split.json"


def read_worst_zeroshot_split(data_dir: str):
    """Read the required P0 train/test split JSON."""
    root = Path(data_dir)
    split_path = root / P0_SPLIT_RELATIVE
    if not split_path.exists():
        raise FileNotFoundError(f"Split file does not exist: {split_path}")
    with split_path.open("r", encoding="utf-8") as file:
        split = json.load(file)
    if "train_cases" not in split or "test_cases" not in split:
        raise KeyError(f"{split_path} must contain train_cases and test_cases")
    return split


def _case_filename(case) -> str:
    """Normalize a JSON case entry to ``<case_name>.nii.gz``."""
    if isinstance(case, dict):
        case = case.get("case_name", case.get("name", case.get("case")))
    if case is None:
        raise ValueError("Each split case must be a string or a mapping with a case_name/name field")
    case_name = Path(str(case)).name
    return case_name if case_name.endswith(".nii.gz") else f"{case_name}.nii.gz"


def read_image_entries(data_dir: str, split: str = "train") -> List[Tuple[Path, Optional[Path]]]:
    """Build P0 image/label paths from the required split JSON.

    ``split='train'`` returns ``(image_path, None)`` so the adaptation dataset
    cannot accidentally read target labels. ``split='test'`` returns matching
    label paths for final evaluation.
    """
    if split not in {"train", "test"}:
        raise ValueError(f"split must be 'train' or 'test', got {split!r}")
    root = Path(data_dir)
    split_data = read_worst_zeroshot_split(data_dir)
    case_names = split_data["train_cases" if split == "train" else "test_cases"]
    image_dir = root / "images" / P0
    label_dir = root / "labels" / P0
    entries = []
    for case in case_names:
        case_name = _case_filename(case)
        image_path = image_dir / case_name
        label_path = label_dir / case_name
        if not image_path.exists():
            raise FileNotFoundError(f"Image listed in split does not exist: {image_path}")
        if split == "test" and not label_path.exists():
            raise FileNotFoundError(f"Label listed for test case does not exist: {label_path}")
        entries.append((image_path, label_path if split == "test" else None))
    if not entries:
        raise ValueError(f"Split '{split}' in the split JSON contains no cases")
    return entries


class VoxTellTargetDataset(Dataset):
    """Returns aligned weak/strong crops; no target masks are read here."""

    def __init__(self, entries, patch_size=(192, 192, 192)):
        if _NNUNET_IMPORT_ERROR is not None:
            raise RuntimeError(
                "nnunetv2 is required for VoxTellTargetDataset image loading; "
                "install CM-SFDA/requirements.txt"
            ) from _NNUNET_IMPORT_ERROR
        self.entries = entries
        self.patch_size = tuple(int(x) for x in patch_size)
        self.reader = NibabelIOWithReorient()
        self.normalization = ZScoreNormalization(intensityproperties={})

    def __len__(self):
        return len(self.entries)

    def _load(self, path: Path):
        return load_preprocessed_image(path, self.reader, self.normalization)

    def _random_crop(self, image):
        pad = []
        for current, target in zip(reversed(image.shape[1:]), reversed(self.patch_size)):
            missing = max(0, target - current)
            pad.extend([missing // 2, missing - missing // 2])
        if any(pad):
            image = torch.nn.functional.pad(image, pad, value=0)
        starts = []
        for current, target in zip(image.shape[1:], self.patch_size):
            maximum = current - target
            starts.append(np.random.randint(0, maximum + 1) if maximum > 0 else 0)
        slices = tuple(slice(s, s + t) for s, t in zip(starts, self.patch_size))
        return image[(slice(None), *slices)].contiguous()

    @staticmethod
    def _strong_augment(image):
        """Apply aligned intensity perturbations only.

        Spatial flips would require transforming the teacher pseudo-label as
        well.  Intensity-only perturbations keep the CAC/consistency targets
        voxel-aligned while still producing distinct views.
        """
        result = image.clone()
        if torch.rand(()) < 0.8:
            result = result * torch.empty((), device=result.device).uniform_(0.85, 1.15)
        if torch.rand(()) < 0.8:
            result = result + torch.empty((), device=result.device).uniform_(-0.15, 0.15)
        if torch.rand(()) < 0.5:
            result = result + torch.randn_like(result) * 0.05
        return result.contiguous()

    def __getitem__(self, index):
        image_path, _ = self.entries[index]
        crop = self._random_crop(self._load(image_path))
        return crop.clone(), self._strong_augment(crop), image_path.name


def make_target_loader(data_dir, patch_size=(192, 192, 192), batch_size=1, num_workers=0):
    dataset = VoxTellTargetDataset(read_image_entries(data_dir, "train"), patch_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers,
                      pin_memory=torch.cuda.is_available())


def load_preprocessed_image(path, reader=None, normalization=None):
    """Load one complete target case exactly as the adaptation dataset does."""
    if _NNUNET_IMPORT_ERROR is not None:
        raise RuntimeError("nnunetv2 is required to preprocess target images") from _NNUNET_IMPORT_ERROR
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image does not exist: {path}")
    reader = reader or NibabelIOWithReorient()
    normalization = normalization or ZScoreNormalization(intensityproperties={})
    image, _ = reader.read_images([str(path)])
    image = image.astype(np.float32, copy=False)
    image, _, _ = crop_to_nonzero(image, None)
    image = normalization.run(image, None)
    return torch.from_numpy(image.copy())


def load_preprocessed_labeled_case(image_path, label_path):
    """Preprocess a complete image and align its GT using the image nonzero bbox.

    This helper is for the offline audit only. Prototype construction calls
    :func:`load_preprocessed_image` and never receives a label path.
    """
    if _NNUNET_IMPORT_ERROR is not None:
        raise RuntimeError("nnunetv2 is required to preprocess validation cases") from _NNUNET_IMPORT_ERROR
    reader = NibabelIOWithReorient()
    image, _ = reader.read_images([str(image_path)])
    label, _ = reader.read_images([str(label_path)])
    image = image.astype(np.float32, copy=False)
    label = (label > 0).astype(np.uint8, copy=False)
    image, _, bbox = crop_to_nonzero(image, None)
    slices = tuple(slice(int(bounds[0]), int(bounds[1])) for bounds in bbox)
    label = label[(slice(None), *slices)]
    image = ZScoreNormalization(intensityproperties={}).run(image, None)
    if tuple(image.shape[1:]) != tuple(label.shape[1:]):
        raise ValueError(
            f"Preprocessed image/label shape mismatch: {image.shape} vs {label.shape}"
        )
    return torch.from_numpy(image.copy()), torch.from_numpy(label.copy()).float()


def pad_to_patch_grid(volume, patch_size):
    """Right-pad a volume to a non-overlapping patch grid and return a valid mask."""
    patch_size = tuple(int(value) for value in patch_size)
    if volume.ndim != 4 or len(patch_size) != 3:
        raise ValueError("Expected volume (C,D,H,W) and a 3-D patch size")
    original_shape = tuple(int(value) for value in volume.shape[-3:])
    target_shape = tuple(
        max(patch, ((size + patch - 1) // patch) * patch)
        for size, patch in zip(original_shape, patch_size)
    )
    padding = []
    for current, target in zip(reversed(original_shape), reversed(target_shape)):
        padding.extend((0, target - current))
    padded = F.pad(volume, padding, value=0.0)
    valid = torch.zeros((1, *target_shape), dtype=torch.float32)
    valid[(slice(None), *(slice(0, size) for size in original_shape))] = 1.0
    return padded.contiguous(), valid, original_shape


def nonoverlapping_patch_locations(volume_shape, patch_size):
    """Return deterministic locations that cover a patch-grid-padded volume once."""
    volume_shape = tuple(int(value) for value in volume_shape)
    patch_size = tuple(int(value) for value in patch_size)
    if any(size % patch for size, patch in zip(volume_shape, patch_size)):
        raise ValueError("nonoverlapping_patch_locations requires a patch-grid-padded shape")
    return [
        (depth, height, width)
        for depth in range(0, volume_shape[0], patch_size[0])
        for height in range(0, volume_shape[1], patch_size[1])
        for width in range(0, volume_shape[2], patch_size[2])
    ]


def sliding_window_locations(volume_shape, patch_size, overlap=0.5):
    """Return deterministic overlapping locations including every volume boundary."""
    volume_shape = tuple(int(value) for value in volume_shape)
    patch_size = tuple(int(value) for value in patch_size)
    if not 0 <= float(overlap) < 1:
        raise ValueError("overlap must be in [0, 1)")

    def axis_starts(size, patch):
        if size <= patch:
            return [0]
        step = max(1, int(round(patch * (1.0 - float(overlap)))))
        starts = list(range(0, size - patch + 1, step))
        if starts[-1] != size - patch:
            starts.append(size - patch)
        return starts

    axes = [axis_starts(size, patch) for size, patch in zip(volume_shape, patch_size)]
    return [(depth, height, width) for depth in axes[0] for height in axes[1] for width in axes[2]]


def extract_volume_patch(volume, location, patch_size):
    slices = tuple(
        slice(int(start), int(start) + int(size))
        for start, size in zip(location, patch_size)
    )
    return volume[(slice(None), *slices)].contiguous()


def fuse_volume_patches(patches, locations, volume_shape):
    """Coverage-normalized fusion for scalar or multi-view spatial patches.

    ``patches`` has shape ``(P,...,D,H,W)`` and the returned tensor has shape
    ``(...,*volume_shape)``. Uniform coverage normalization prevents overlap
    from changing the scale of predictions or evidence.
    """
    if not torch.is_tensor(patches) or patches.ndim < 4:
        raise ValueError("Expected patches (P,...,D,H,W)")
    if patches.shape[0] != len(locations):
        raise ValueError("Patch count and location count must agree")
    volume_shape = tuple(int(value) for value in volume_shape)
    patch_size = tuple(int(value) for value in patches.shape[-3:])
    leading_shape = tuple(int(value) for value in patches.shape[1:-3])
    accumulator = patches.new_zeros((*leading_shape, *volume_shape))
    coverage = patches.new_zeros(volume_shape)
    for patch, location in zip(patches, locations):
        slices = tuple(
            slice(int(start), int(start) + int(size))
            for start, size in zip(location, patch_size)
        )
        accumulator[(..., *slices)] += patch
        coverage[slices] += 1
    if not bool((coverage > 0).all()):
        raise RuntimeError("Sliding-window locations did not cover the complete volume")
    return accumulator / coverage.clamp_min(1)


def load_ras_image(path: str):
    if _NNUNET_IMPORT_ERROR is not None:
        raise RuntimeError("nnunetv2 is required to load RAS images") from _NNUNET_IMPORT_ERROR
    return NibabelIOWithReorient().read_images([path])[0]


def load_ras_label(path: str):
    if _NNUNET_IMPORT_ERROR is not None:
        raise RuntimeError("nnunetv2 is required to load RAS labels") from _NNUNET_IMPORT_ERROR
    return (NibabelIOWithReorient().read_images([path])[0] > 0).astype(np.uint8)
