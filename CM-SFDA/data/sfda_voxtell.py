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
        if not path.exists():
            raise FileNotFoundError(f"Image does not exist: {path}")
        image, _ = self.reader.read_images([str(path)])
        image = image.astype(np.float32, copy=False)
        image, _, _ = crop_to_nonzero(image, None)
        image = self.normalization.run(image, None)
        return torch.from_numpy(image.copy())

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


def load_ras_image(path: str):
    if _NNUNET_IMPORT_ERROR is not None:
        raise RuntimeError("nnunetv2 is required to load RAS images") from _NNUNET_IMPORT_ERROR
    return NibabelIOWithReorient().read_images([path])[0]


def load_ras_label(path: str):
    if _NNUNET_IMPORT_ERROR is not None:
        raise RuntimeError("nnunetv2 is required to load RAS labels") from _NNUNET_IMPORT_ERROR
    return (NibabelIOWithReorient().read_images([path])[0] > 0).astype(np.uint8)
