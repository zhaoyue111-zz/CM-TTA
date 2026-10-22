from __future__ import annotations

import csv
import json
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import nibabel as nib
import numpy as np
import torch
from nibabel.orientations import axcodes2ornt, io_orientation, ornt_transform
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from sklearn.metrics import average_precision_score, roc_auc_score


PERCENTILES = (5, 25, 50, 75, 95)
RANK_FRACTIONS = (0.001, 0.005, 0.01, 0.02, 0.05, 0.10)


def _split_case_name(case: object) -> str:
    if isinstance(case, Mapping):
        case = case.get("case_name", case.get("name", case.get("case")))
    if case is None:
        raise ValueError("Each test case must be a filename or case mapping")
    return case_name(str(case))


def resolve_image_label_dirs(data_dir: str | Path) -> tuple[Path, Path]:
    """Resolve the flat ROI layout, retaining the existing legacy P0 fallback."""
    root = Path(data_dir).expanduser()
    candidates = (
        (root / "images" / "P0", root / "labels" / "P0"),
        (root / "images", root / "labels"),
    )
    for image_dir, label_dir in candidates:
        if image_dir.is_dir() and label_dir.is_dir():
            return image_dir, label_dir
    searched = ", ".join(f"({image}, {label})" for image, label in candidates)
    raise FileNotFoundError(f"No compatible image/label layout found; checked {searched}")


def read_test_entries(
    data_dir: str | Path, split_file: str | Path | None = None
) -> list[tuple[Path, Path]]:
    """Read paired ROI images/labels, optionally restricted by a test split."""
    image_dir, label_dir = resolve_image_label_dirs(data_dir)
    if split_file is None:
        names = [path.name for path in sorted(image_dir.glob("*.nii.gz"))]
    else:
        split_path = Path(split_file).expanduser()
        if not split_path.exists():
            raise FileNotFoundError(f"Split file does not exist: {split_path}")
        payload = json.loads(split_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping) or not isinstance(
            payload.get("test_cases"), list
        ):
            raise ValueError(f"{split_path} must contain a test_cases list")
        names = [_split_case_name(raw_case) for raw_case in payload["test_cases"]]
    entries: list[tuple[Path, Path]] = []
    for name in names:
        image_path = image_dir / name
        label_path = label_dir / name
        if not image_path.is_file():
            raise FileNotFoundError(f"Test image does not exist: {image_path}")
        if not label_path.is_file():
            raise FileNotFoundError(f"Test label does not exist: {label_path}")
        entries.append((image_path, label_path))
    if not entries:
        raise ValueError("No paired liver ROI cases were found")
    return entries


def load_ras_image(path: str | Path) -> np.ndarray:
    return NibabelIOWithReorient().read_images([str(path)])[0]


def load_ras_label_values(path: str | Path) -> np.ndarray:
    """Load integer label IDs in the same RAS/model-space order as the image."""
    label = np.squeeze(NibabelIOWithReorient().read_images([str(path)])[0])
    if label.ndim != 3:
        raise ValueError(f"Expected a 3-D label at {path}, got {label.shape}")
    rounded = np.rint(label)
    if not np.allclose(label, rounded):
        raise ValueError(f"Label contains non-integer values: {path}")
    return rounded.astype(np.int64, copy=False)


def foreground_mask(label: np.ndarray, foreground_labels: Sequence[int]) -> np.ndarray:
    labels = tuple(int(value) for value in foreground_labels)
    if not labels:
        raise ValueError("foreground_labels must contain at least one integer")
    return np.isin(np.asarray(label), labels)


def seed_everything(seed: int) -> None:
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def case_name(value: str | Path) -> str:
    name = Path(str(value)).name
    return name if name.endswith(".nii.gz") else f"{name}.nii.gz"


def bbox_slices(bbox: Sequence[Sequence[int]]) -> tuple[slice, slice, slice]:
    if len(bbox) != 3:
        raise ValueError(f"Expected a 3-D crop bounding box, got {bbox!r}")
    return tuple(slice(int(bounds[0]), int(bounds[1])) for bounds in bbox)


def restore_crop(
    cropped: np.ndarray,
    bbox: Sequence[Sequence[int]],
    original_shape: Sequence[int],
    fill_value: float = 0.0,
) -> np.ndarray:
    cropped = np.asarray(cropped)
    full = np.full(tuple(int(v) for v in original_shape), fill_value, dtype=cropped.dtype)
    slices = bbox_slices(bbox)
    expected = tuple(s.stop - s.start for s in slices)
    if cropped.shape != expected:
        raise ValueError(f"Crop shape {cropped.shape} does not match bbox shape {expected}")
    full[slices] = cropped
    return full


def save_float_nifti_model_space(
    volume: np.ndarray, source_image: Path, output_path: Path
) -> None:
    """Save a float D,H,W nnUNet/RAS-reader array in source NIfTI geometry."""
    source = nib.load(str(source_image))
    original_affine = source.affine
    canonical = source.as_reoriented(io_orientation(original_affine))
    model_volume = np.asarray(volume, dtype=np.float32)
    expected = tuple(int(v) for v in canonical.shape[::-1])
    if model_volume.shape != expected:
        raise ValueError(
            f"Model-space map shape {model_volume.shape} does not match source "
            f"model shape {expected}"
        )

    canonical_array = model_volume.transpose((2, 1, 0))
    mapped = nib.Nifti1Image(canonical_array, affine=canonical.affine)
    from_canonical = ornt_transform(axcodes2ornt("RAS"), io_orientation(original_affine))
    restored = mapped.as_reoriented(from_canonical)
    if restored.shape != source.shape or not np.allclose(restored.affine, original_affine):
        raise RuntimeError(
            f"Failed to restore source geometry for {source_image.name}: "
            f"shape {restored.shape}/{source.shape}"
        )
    restored.set_data_dtype(np.float32)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(restored, str(output_path))


def distribution_stats(values: np.ndarray) -> dict[str, float | int | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "median": None,
            "p5": None,
            "p25": None,
            "p75": None,
            "p95": None,
            "max": None,
        }
    quantiles = np.percentile(values, PERCENTILES)
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "median": float(quantiles[2]),
        "p5": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(values.max()),
    }


def binary_ranking_metrics(
    scores: np.ndarray, target: np.ndarray
) -> dict[str, float | None]:
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    target = np.asarray(target, dtype=bool).reshape(-1)
    finite = np.isfinite(scores)
    scores = scores[finite]
    target = target[finite]
    positives = int(target.sum())
    negatives = int((~target).sum())
    if positives == 0 or negatives == 0:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": float(roc_auc_score(target, scores)),
        "auprc": float(average_precision_score(target, scores)),
    }


def top_fraction_mask(scores: np.ndarray, fraction: float) -> np.ndarray:
    flat = np.asarray(scores).reshape(-1)
    if flat.size == 0 or not 0.0 < fraction <= 1.0:
        raise ValueError("fraction must be in (0, 1] and scores must be non-empty")
    count = max(1, int(math.ceil(flat.size * fraction)))
    if count == flat.size:
        return np.ones_like(scores, dtype=bool)
    indices = np.argpartition(flat, flat.size - count)[flat.size - count :]
    mask = np.zeros(flat.size, dtype=bool)
    mask[indices] = True
    return mask.reshape(np.asarray(scores).shape)


def top_fraction_enrichment(
    scores: np.ndarray,
    target: np.ndarray,
    fractions: Iterable[float] = RANK_FRACTIONS,
) -> dict[str, float | int | None]:
    target = np.asarray(target, dtype=bool)
    target_count = int(target.sum())
    prevalence = float(target_count / target.size) if target.size else 0.0
    result: dict[str, float | int | None] = {}
    for fraction in fractions:
        selected = top_fraction_mask(scores, float(fraction))
        true_positive = int(np.logical_and(selected, target).sum())
        prefix = f"top_{fraction * 100:g}pct"
        result[f"{prefix}_voxel_count"] = int(selected.sum())
        result[f"{prefix}_gt_foreground_precision"] = float(
            true_positive / max(1, int(selected.sum()))
        )
        result[f"{prefix}_gt_foreground_recall"] = (
            float(true_positive / target_count) if target_count else None
        )
        precision = float(true_positive / max(1, int(selected.sum())))
        result[f"{prefix}_enrichment_factor"] = (
            float(precision / prevalence) if prevalence > 0.0 else None
        )
    return result


def flatten_mapping(value: Mapping[str, object], prefix: str = "") -> dict[str, object]:
    flat: dict[str, object] = {}
    for key, item in value.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(item, Mapping):
            flat.update(flatten_mapping(item, name))
        elif isinstance(item, (list, tuple, dict)):
            flat[name] = json.dumps(item, ensure_ascii=False, separators=(",", ":"))
        else:
            flat[name] = item
    return flat


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flattened = [flatten_mapping(row) for row in rows]
    fields: list[str] = []
    for row in flattened:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(flattened)


def json_ready(value):
    if isinstance(value, Mapping):
        return {str(k): json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(json_ready(value), indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def save_case_figure(
    output_path: Path,
    image: np.ndarray,
    target: np.ndarray,
    probability: np.ndarray,
    similarity_max: np.ndarray,
    similarity_topk: np.ndarray,
    top_one_percent: np.ndarray,
) -> int:
    os.environ.setdefault(
        "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cmtta_matplotlib")
    )
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    image = np.asarray(image).squeeze()
    target = np.asarray(target, dtype=bool)
    if target.any():
        slice_index = int(target.sum(axis=(1, 2)).argmax())
    else:
        slice_index = int(np.nanmax(similarity_max, axis=(1, 2)).argmax())

    panels = (
        (image[slice_index], "image", "gray", None, None),
        (target[slice_index], "GT", "gray", 0.0, 1.0),
        (probability[slice_index], "segmentation probability", "magma", 0.0, 1.0),
        (similarity_max[slice_index], "prototype max similarity", "coolwarm", -1.0, 1.0),
        (similarity_topk[slice_index], "prototype top-k mean", "coolwarm", -1.0, 1.0),
    )
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    for axis, (array, title, cmap, vmin, vmax) in zip(axes.flat[:5], panels):
        handle = axis.imshow(array, cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.axis("off")
        if title not in {"image", "GT"}:
            fig.colorbar(handle, ax=axis, fraction=0.046, pad=0.04)

    overlay_axis = axes.flat[5]
    overlay_axis.imshow(image[slice_index], cmap="gray")
    overlay = np.ma.masked_where(~top_one_percent[slice_index], top_one_percent[slice_index])
    overlay_axis.imshow(overlay, cmap="autumn", alpha=0.55, vmin=0, vmax=1)
    if target[slice_index].any():
        overlay_axis.contour(target[slice_index].astype(float), levels=[0.5], colors="cyan")
    overlay_axis.set_title("similarity top 1% (orange), GT contour (cyan)")
    overlay_axis.axis("off")
    fig.suptitle(f"model-space axial slice D={slice_index}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return slice_index
