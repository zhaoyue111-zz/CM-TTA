"""Run original CM-TTA/LSPM/DSPU on complete VoxTell 3-D P0 cases."""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

from data.voxtell_p0 import (
    load_ras_image,
    load_ras_label,
    make_case_patches,
    read_test_entries,
)
from method.voxtell_cmtta import VoxTellCMTTA, load_cmtta_checkpoint, save_cmtta_checkpoint


DEFAULT_VOXTELL_ROOT = Path("/data/zy/VoxTell_from_disk")
DEFAULT_QWEN = Path(
    "/home/SENSETIME/yangtingting/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-Embedding-4B/snapshots/"
    "5cf2132abc99cad020ac570b19d031efec650f2b"
)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def binary_metrics(prediction: np.ndarray, target: np.ndarray) -> dict[str, float]:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/label shape mismatch: {prediction.shape} vs {target.shape}")
    intersection = np.logical_and(prediction, target).sum()
    union = np.logical_or(prediction, target).sum()
    if prediction.sum() == 0 and target.sum() == 0:
        return {"Dice": 1.0, "mIoU": 1.0}
    return {
        "Dice": float(2.0 * intersection / max(1, prediction.sum() + target.sum())),
        "mIoU": float(intersection / max(1, union)),
    }


def save_prediction_nifti(prediction: np.ndarray, image_path: Path, output_path: Path) -> None:
    """Save prediction using the same reader/writer orientation as VoxTell.

    ``load_ras_image`` returns the nnUNet model array and discards reader
    properties.  Re-reading the source here gives ``NibabelIOWithReorient``
    the properties it needs to restore the original NIfTI shape, affine, and
    orientation; no model-space axis order is assumed in this script.
    """
    from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    import nibabel as nib

    reader = NibabelIOWithReorient()
    source_data, properties = reader.read_images([str(image_path)])
    prediction = np.asarray(prediction)
    if prediction.ndim != 3:
        raise ValueError(
            f"Prediction must be a 3-D model-space array, got {prediction.shape}"
        )
    if tuple(source_data.shape[1:]) != tuple(prediction.shape):
        raise ValueError(
            f"Prediction/source shape mismatch for {image_path.name}: "
            f"model-space {prediction.shape} vs reader model shape {source_data.shape[1:]}"
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    reader.write_seg(prediction.astype(np.uint8, copy=False), str(output_path), properties)

    # Fail early if a future reader/writer change silently alters the source
    # image geometry.  The data values are checked through the same reader in
    # tests; these checks cover the on-disk NIfTI geometry at runtime.
    source_nifti = nib.load(str(image_path))
    saved_nifti = nib.load(str(output_path))
    if tuple(saved_nifti.shape[:3]) != tuple(source_nifti.shape[:3]):
        raise RuntimeError(
            f"Saved prediction shape does not match source {image_path.name}: "
            f"{saved_nifti.shape[:3]} vs {source_nifti.shape[:3]}"
        )
    if not np.allclose(saved_nifti.affine, source_nifti.affine):
        raise RuntimeError(
            f"Saved prediction affine does not match source {image_path.name}"
        )


def check_prediction_nifti_geometry(
    prediction_path: Path, image_path: Path, label_path: Path
) -> None:
    """Verify that prediction, image, and GT share the case geometry."""
    import nibabel as nib

    prediction = nib.load(str(prediction_path))
    image = nib.load(str(image_path))
    label = nib.load(str(label_path))
    if tuple(image.shape[:3]) != tuple(label.shape[:3]):
        raise ValueError(
            f"Image/GT shape mismatch: {image_path.name} {image.shape[:3]} vs "
            f"{label_path.name} {label.shape[:3]}"
        )
    if not np.allclose(image.affine, label.affine):
        raise ValueError(f"Image/GT affine mismatch for {image_path.name}")
    if tuple(prediction.shape[:3]) != tuple(image.shape[:3]):
        raise ValueError(
            f"Prediction/image shape mismatch: {prediction.shape[:3]} vs {image.shape[:3]}"
        )
    if not np.allclose(prediction.affine, image.affine):
        raise ValueError(f"Prediction/image affine mismatch for {image_path.name}")


def build_predictor(args):
    root = Path(args.voxtell_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from voxtell.inference.predictor import VoxTellPredictor
    except ModuleNotFoundError as error:
        if error.name == "transformers":
            raise RuntimeError("VoxTell requires transformers and its local dependencies") from error
        raise
    return VoxTellPredictor(
        model_dir=str(Path(args.model_dir).expanduser().resolve()),
        device=args.device,
        text_encoding_model=args.text_model,
    )


def evaluate_case(
    predictor,
    image_path: Path,
    label_path: Path,
    data: torch.Tensor,
    bbox,
    original_shape,
    text_feature: torch.Tensor,
    output_dir: Path,
) -> dict[str, float | str]:
    with torch.no_grad():
        logits = predictor.predict_sliding_window_return_logits(
            data, text_feature.to(predictor.device)
        ).float().cpu()
    cropped_prediction = (torch.sigmoid(logits) > 0.5).numpy().astype(np.uint8)
    prediction = insert_crop_into_image(
        np.zeros((cropped_prediction.shape[0], *original_shape), dtype=np.uint8),
        cropped_prediction,
        bbox,
    )[0]
    target = np.squeeze(load_ras_label(str(label_path)))
    metrics = binary_metrics(prediction, target)
    stem = image_path.name[:-7] if image_path.name.endswith(".nii.gz") else image_path.stem
    prediction_path = output_dir / f"{stem}.nii.gz"
    save_prediction_nifti(prediction, image_path, prediction_path)
    check_prediction_nifti_geometry(prediction_path, image_path, label_path)
    return {"basename": image_path.name, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Original CM-TTA with VoxTell on P0")
    parser.add_argument("--data_dir", default="/data/zy/CT_MRI_DATA_3D")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--voxtell_root", default=str(DEFAULT_VOXTELL_ROOT))
    parser.add_argument("--model_dir", default=str(DEFAULT_VOXTELL_ROOT / "model"))
    parser.add_argument("--text_model", default=str(DEFAULT_QWEN))
    parser.add_argument("--prompt", default="liver")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results_/voxtell_cmtta_p0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=1377)

    # CM-TTA uses the paper's one-step Adam configuration.
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_momentum", type=float, default=0.99)
    parser.add_argument("--amp_init_scale", type=float, default=1024.0)
    parser.add_argument("--short_memory_length", type=int, default=16)
    parser.add_argument("--selection_p", type=float, default=0.1)
    parser.add_argument("--num_aug_views", type=int, default=9)
    parser.add_argument("--view_batch_size", type=int, default=1)
    parser.add_argument("--w_cac", type=float, default=1.0)
    parser.add_argument("--w_entropy", type=float, default=0.1)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument(
        "--tta_steps",
        type=int,
        default=1,
        help="Compatibility argument; a complete 3-D case is exactly one TTA step",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.tta_steps != 1:
        raise ValueError("VoxTell CM-TTA uses exactly one TTA step per complete 3-D case")
    if args.print_freq < 1:
        raise ValueError("--print_freq must be positive")
    if args.prompt != "liver":
        raise ValueError('VoxTell CM-TTA fixes the original text prompt to "liver"')
    seed_everything(args.seed)
    args.device = torch.device(args.device)
    if args.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    entries = read_test_entries(args.data_dir, args.split_file)
    predictor = build_predictor(args)
    predictor.text_backbone.requires_grad_(False)

    adapter = VoxTellCMTTA(
        predictor.network,
        None,
        args.device,
        args,
        qwen_text_encoder=predictor.text_backbone,
        qwen_tokenizer=predictor.tokenizer,
        text_prompt=args.prompt,
    )
    if args.checkpoint:
        load_cmtta_checkpoint(args.checkpoint, adapter)
    predictor.network = adapter.model

    case_rows = []
    history = []
    predictions_dir = output_dir / "predictions"
    try:
        for case_index, (image_path, label_path) in enumerate(entries, start=1):
            image = load_ras_image(str(image_path))
            data, bbox, original_shape = predictor.preprocess(image)
            patches, valid_masks, _locations, _padded_shape = make_case_patches(
                data, predictor.patch_size
            )

            # One complete case is one adaptation time step.  adapt_case sums
            # all patch losses and performs exactly one optimizer/LSPM update.
            trace = adapter.adapt_case(patches, valid_masks)
            with torch.no_grad():
                text_feature = adapter._encode_ctx(adapter.ctx_delta.detach())
            row = evaluate_case(
                predictor,
                image_path,
                label_path,
                data,
                bbox,
                original_shape,
                text_feature,
                predictions_dir,
            )
            row["adaptation_quality"] = trace["selected_cac"]
            case_rows.append(row)
            history.append({"case": image_path.name, **trace})
            if case_index % args.print_freq == 0 or case_index == len(entries):
                print(
                    f"case {case_index}/{len(entries)} {image_path.name} "
                    f"patches={len(patches)} loss={trace['loss']:.4f} "
                    f"view={trace['selected_view']} Dice={row['Dice']:.4f}"
                )
    finally:
        adapter.close()

    average = {
        "Dice": float(np.mean([row["Dice"] for row in case_rows])),
        "mIoU": float(np.mean([row["mIoU"] for row in case_rows])),
    }
    (output_dir / "results.json").write_text(
        json.dumps({"cases": case_rows, "average": average}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_cmtta_checkpoint(str(output_dir / "last.pt"), adapter, args, history)
    print(f"Results: Dice={average['Dice']:.4f} mIoU={average['mIoU']:.4f}")


if __name__ == "__main__":
    main()
