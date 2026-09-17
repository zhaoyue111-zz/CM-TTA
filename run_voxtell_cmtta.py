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
    prediction = predict_case(
        predictor, data, bbox, original_shape, text_feature
    )
    target = np.squeeze(load_ras_label(str(label_path)))
    metrics = binary_metrics(prediction, target)
    stem = image_path.name[:-7] if image_path.name.endswith(".nii.gz") else image_path.stem
    prediction_path = output_dir / f"{stem}.nii.gz"
    save_prediction_nifti(prediction, image_path, prediction_path)
    check_prediction_nifti_geometry(prediction_path, image_path, label_path)
    return {"basename": image_path.name, **metrics}


def predict_case(
    predictor,
    data: torch.Tensor,
    bbox,
    original_shape,
    text_feature: torch.Tensor,
) -> np.ndarray:
    """Run one text-conditioned case prediction in model space."""
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
    return prediction


def predict_nonoverlap_case(
    adapter,
    patches: list[torch.Tensor],
    valid_masks: list[torch.Tensor],
    locations,
    data_shape,
    bbox,
    original_shape,
    zero_shot_text: torch.Tensor,
) -> np.ndarray:
    """Predict zero-shot with the exact non-overlap adaptation patch layout."""
    if not patches:
        raise ValueError("Non-overlap diagnostic received no patches")
    if len(patches) != len(valid_masks) or len(patches) != len(locations):
        raise ValueError("patches, valid_masks, and locations must have equal lengths")

    patch_size = tuple(int(size) for size in patches[0].shape[-3:])
    padded_shape = tuple(
        max(int(location[axis]) + patch_size[axis] for location in locations)
        for axis in range(3)
    )
    padded_prediction = np.zeros(padded_shape, dtype=np.uint8)
    write_count = np.zeros(padded_shape, dtype=np.uint8)
    text_input = adapter._text_input(zero_shot_text.to(adapter.device), 1)
    autocast_enabled = adapter.device.type == "cuda"

    with torch.no_grad():
        for patch, valid_mask, location in zip(patches, valid_masks, locations):
            patch_batch = patch.unsqueeze(0).to(adapter.device, non_blocking=True)
            with torch.autocast(
                device_type=adapter.device.type, enabled=autocast_enabled
            ):
                logits = adapter.model(patch_batch, text_input)
            if isinstance(logits, (list, tuple)):
                logits = logits[0]
            probabilities = torch.sigmoid(logits[0, 0]).float().cpu().numpy()
            valid = valid_mask.detach().cpu().numpy() > 0
            patch_prediction = (probabilities > 0.5) & valid
            slices = tuple(
                slice(int(start), int(start) + patch_size[axis])
                for axis, start in enumerate(location)
            )
            padded_prediction[slices] = patch_prediction.astype(np.uint8)
            write_count[slices] += 1

    if not np.all(write_count == 1):
        missing = int(np.count_nonzero(write_count == 0))
        repeated = int(np.count_nonzero(write_count > 1))
        raise RuntimeError(
            "Non-overlap patch layout did not cover each padded voxel exactly once: "
            f"missing={missing}, repeated={repeated}"
        )
    crop_slices = tuple(slice(0, int(size)) for size in data_shape)
    cropped_prediction = padded_prediction[crop_slices]
    return insert_crop_into_image(
        np.zeros((1, *original_shape), dtype=np.uint8),
        cropped_prediction[None],
        bbox,
    )[0]


def evaluate_all_view_metrics(
    predictor,
    label_path: Path,
    data: torch.Tensor,
    bbox,
    original_shape,
    text_feature: torch.Tensor,
    view_params: list[dict[str, float]],
    view_selection: dict[str, list[float] | int],
) -> list[dict[str, float | bool | int]]:
    """Evaluate every selected-case view against GT and attach rank diagnostics."""
    target = np.squeeze(load_ras_label(str(label_path)))
    metrics = []
    selected_view = int(view_selection["selected_view"])
    for view_index, param in enumerate(view_params):
        scale = float(param["scale"])
        offset = float(param["offset"])
        view_data = data if scale == 1.0 and offset == 0.0 else data * scale + offset
        prediction = predict_case(
            predictor, view_data, bbox, original_shape, text_feature
        )
        binary = binary_metrics(prediction, target)
        metrics.append(
            {
                "view": view_index,
                "Dice": binary["Dice"],
                "mIoU": binary["mIoU"],
                "CAC": float(view_selection["cac"][view_index]),
                "entropy": float(view_selection["entropy"][view_index]),
                "CAC_rank": float(view_selection["cac_rank"][view_index]),
                "entropy_rank": float(view_selection["entropy_rank"][view_index]),
                "combined_rank": float(view_selection["combined_rank"][view_index]),
                "selected": view_index == selected_view,
            }
        )
    return metrics


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

            # Diagnostic only: compare the official zero-shot sliding-window
            # path with the exact non-overlap adaptation patch layout before
            # any ctx/prompt update or view augmentation.
            with torch.no_grad():
                zero_ctx = torch.zeros_like(adapter.ctx_delta.detach())
                zero_shot_text = adapter._encode_ctx(zero_ctx).detach()
            zero_shot_sliding_prediction = predict_case(
                predictor,
                data,
                bbox,
                original_shape,
                zero_shot_text,
            )
            zero_shot_nonoverlap_prediction = predict_nonoverlap_case(
                adapter,
                patches,
                valid_masks,
                _locations,
                data.shape[-3:],
                bbox,
                original_shape,
                zero_shot_text,
            )
            zero_shot_target = np.squeeze(load_ras_label(str(label_path)))
            zero_shot_sliding_dice = binary_metrics(
                zero_shot_sliding_prediction, zero_shot_target
            )["Dice"]
            zero_shot_nonoverlap_dice = binary_metrics(
                zero_shot_nonoverlap_prediction, zero_shot_target
            )["Dice"]
            zero_shot_patch_gap = zero_shot_nonoverlap_dice - zero_shot_sliding_dice
            print(
                f"case {case_index}/{len(entries)} {image_path.name} "
                f"zero_shot_sliding_Dice={zero_shot_sliding_dice:.4f} "
                f"zero_shot_nonoverlap_Dice={zero_shot_nonoverlap_dice:.4f} "
                f"diff={zero_shot_patch_gap:.4f}"
            )

            # One complete case is one adaptation time step.  adapt_case sums
            # all patch losses and performs exactly one optimizer/LSPM update.
            with torch.no_grad():
                prompt_embedding_before = adapter._encode_ctx(
                    adapter.ctx_delta.detach()
                ).detach()
            trace = adapter.adapt_case(patches, valid_masks)
            with torch.no_grad():
                prompt_embedding_after = adapter._encode_ctx(
                    adapter.ctx_delta.detach()
                ).detach()
                prompt_embedding_change = prompt_embedding_after - prompt_embedding_before
                prompt_embedding_change_rate = float(
                    prompt_embedding_change.norm()
                    / prompt_embedding_before.norm().clamp_min(torch.finfo(torch.float32).eps)
                )
                text_feature = prompt_embedding_after
            view_metrics = evaluate_all_view_metrics(
                predictor,
                label_path,
                data,
                bbox,
                original_shape,
                text_feature,
                trace["view_params"],
                trace["view_selection"],
            )
            selected_view_dice = view_metrics[trace["selected_view"]]["Dice"]
            row_view_metrics = view_metrics
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
            row["view_metrics"] = row_view_metrics
            row["zero_shot_sliding_dice"] = zero_shot_sliding_dice
            row["zero_shot_nonoverlap_dice"] = zero_shot_nonoverlap_dice
            row["zero_shot_patch_gap"] = zero_shot_patch_gap
            case_rows.append(row)
            history.append(
                {
                    "case": image_path.name,
                    "zero_shot_sliding_dice": zero_shot_sliding_dice,
                    "zero_shot_nonoverlap_dice": zero_shot_nonoverlap_dice,
                    "zero_shot_patch_gap": zero_shot_patch_gap,
                    "prompt_embedding_change_rate": prompt_embedding_change_rate,
                    "view_metrics": view_metrics,
                    **trace,
                }
            )
            print(
                f"case {case_index}/{len(entries)} {image_path.name} "
                f"prompt_embedding_change="
                f"{prompt_embedding_change_rate:.6e} "
                f"({prompt_embedding_change_rate * 100.0:.4f}%)"
            )
            print(
                f"case {case_index}/{len(entries)} {image_path.name} "
                f"selected_view={trace['selected_view']} "
                f"selected_view_GT_Dice={selected_view_dice:.4f}"
            )
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
