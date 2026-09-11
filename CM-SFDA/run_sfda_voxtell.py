"""Train/evaluate VoxTell under an offline source-free protocol."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch

from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

from data.sfda_voxtell import load_ras_image, load_ras_label, make_target_loader, read_image_entries
from method.sfda_voxtell import VoxTellPromptSFDA, load_sfda_checkpoint, save_sfda_checkpoint


DEFAULT_VOXTELL_ROOT = os.environ.get("VOXTELL_ROOT", r"D:\pythonCode\VoxTell_from_disk")
DEFAULT_MODEL_DIR = os.environ.get("VOXTELL_MODEL_DIR", str(Path(DEFAULT_VOXTELL_ROOT) / "model"))
EVALUATION_METRICS = ("dice", "iou", "recall", "precision")


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_predictor(model_dir, device, voxtell_root):
    root_path = Path(voxtell_root).expanduser().resolve()
    model_path = Path(model_dir).expanduser().resolve()
    if not root_path.is_dir():
        raise FileNotFoundError(
            f"VoxTell root does not exist: {root_path}. "
            "Set it with --voxtell_root."
        )
    if not model_path.exists():
        raise FileNotFoundError(
            f"VoxTell model directory/file does not exist: {model_path}. "
            "Set it with --model_dir."
        )
    root = str(root_path)
    if root not in sys.path:
        sys.path.insert(0, root)
    try:
        from voxtell.inference.predictor import VoxTellPredictor
    except ImportError as error:
        raise ImportError(
            f"Could not import VoxTell from {root_path}; check --voxtell_root"
        ) from error
    return VoxTellPredictor(model_dir=str(model_path), device=device)


def binary_segmentation_metrics(pred, target):
    """Return per-volume binary metrics with foreground as the positive class.

    IoU is computed for the foreground class only. When both masks are empty
    all metrics are 1; otherwise an undefined precision/recall is 0.
    """
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/label shape mismatch: {pred.shape} vs {target.shape}")

    tp = int(np.logical_and(pred, target).sum())
    fp = int(np.logical_and(pred, ~target).sum())
    fn = int(np.logical_and(~pred, target).sum())
    if tp + fp + fn == 0:
        return {name: 1.0 for name in EVALUATION_METRICS}

    dice_denominator = 2 * tp + fp + fn
    foreground_iou_denominator = tp + fp + fn
    foreground_iou = tp / foreground_iou_denominator
    return {
        "dice": float(2 * tp / dice_denominator),
        "iou": float(foreground_iou),
        "recall": float(tp / (tp + fn)) if tp + fn else 0.0,
        "precision": float(tp / (tp + fp)) if tp + fp else 0.0,
    }


def dice_score(pred, target):
    """Compatibility wrapper for consumers that only need Dice."""
    return binary_segmentation_metrics(pred, target)["dice"]


def _surface_metrics(pred, target, spacing):
    """Return ASSD/HD95 when scipy and valid voxel spacing are available."""
    if spacing is None or len(spacing) < 3:
        return None, None
    try:
        from scipy.ndimage import binary_erosion, distance_transform_edt
    except ImportError:
        return None, None
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/label shape mismatch: {pred.shape} vs {target.shape}")
    if not pred.any() and not target.any():
        return 0.0, 0.0
    if not pred.any() or not target.any():
        return None, None
    structure = np.ones((3, 3, 3), dtype=bool)
    pred_surface = pred ^ binary_erosion(pred, structure=structure, border_value=0)
    target_surface = target ^ binary_erosion(target, structure=structure, border_value=0)
    spacing = tuple(float(value) for value in spacing[:3])
    pred_to_target = distance_transform_edt(~target, sampling=spacing)[pred_surface]
    target_to_pred = distance_transform_edt(~pred, sampling=spacing)[target_surface]
    distances = np.concatenate([pred_to_target, target_to_pred])
    return float(distances.mean()), float(np.percentile(distances, 95))


def _native_spacing(spacing):
    """Convert nibabel/NumPy spacing scalars to JSON-safe Python floats."""
    return tuple(float(value) for value in spacing[:3])


def _canonical_source_geometry(image_path):
    """Get RAS affine/spacing for a NIfTI prediction, if nibabel is available."""
    try:
        import nibabel as nib
    except ImportError:
        return None, None, None
    source = nib.as_closest_canonical(nib.load(str(image_path)))
    spacing = _native_spacing(source.header.get_zooms())
    shape = tuple(int(value) for value in source.shape[:3])
    return source.affine, spacing, shape


def _save_prediction_nifti(prediction, image_path, output_path):
    try:
        import nibabel as nib
    except ImportError:
        return False
    source = nib.as_closest_canonical(nib.load(str(image_path)))
    if tuple(source.shape[:3]) != tuple(prediction.shape):
        return False
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    nib.save(nib.Nifti1Image(prediction.astype(np.uint8), source.affine, header), str(output_path))
    return True


def _write_json(path, payload):
    """Atomically write JSON so completed epochs survive later interruption."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary_path.replace(path)


def _epoch_metrics_payload(epoch_records):
    return {
        "metric_definitions": {
            "dice": "foreground Dice coefficient",
            "iou": "foreground IoU: TP / (TP + FP + FN)",
            "recall": "foreground recall",
            "precision": "foreground precision",
            "average": "macro average over test cases",
        },
        "epochs": epoch_records,
    }


def should_evaluate_epoch(epoch, total_epochs, interval):
    """Evaluate at the requested interval and always at the final epoch."""
    if interval <= 0:
        raise ValueError(f"eval_interval must be positive, got {interval}")
    return epoch % interval == 0 or epoch == total_epochs


def evaluate(
    predictor,
    entries,
    soft_prompt_embedding,
    output_dir=None,
    save_predictions=False,
    include_surface_metrics=False,
):
    rows = []
    if save_predictions:
        if output_dir is None:
            raise ValueError("output_dir is required when save_predictions=True")
        output_dir.mkdir(parents=True, exist_ok=True)
    predictor.network.eval()
    for image_path, label_path in entries:
        if label_path is None:
            continue
        image = load_ras_image(str(image_path))
        data, bbox, original_shape = predictor.preprocess(image)
        with torch.inference_mode():
            logits = predictor.predict_sliding_window_return_logits(
                data, soft_prompt_embedding.to(predictor.device)
            ).float().cpu()
        prediction = (torch.sigmoid(logits) > 0.5).numpy().astype(np.uint8)
        prediction = insert_crop_into_image(
            np.zeros((prediction.shape[0], *original_shape), dtype=np.uint8), prediction, bbox
        )[0]
        target = np.squeeze(load_ras_label(str(label_path)))
        if prediction.shape != target.shape:
            raise ValueError(
                f"Prediction/label shape mismatch for {image_path.name}: "
                f"{prediction.shape} vs {target.shape}"
            )
        row = {
            "basename": image_path.name,
            **binary_segmentation_metrics(prediction, target),
        }
        if include_surface_metrics:
            _, spacing, _ = _canonical_source_geometry(image_path)
            assd, hd95 = _surface_metrics(prediction, target, spacing)
            if assd is not None:
                row.update(
                    {"assd": assd, "hd95": hd95, "spacing": list(spacing)}
                )
        rows.append(row)
        if save_predictions:
            output_stem = (
                image_path.name[:-7]
                if image_path.name.endswith(".nii.gz")
                else image_path.stem
            )
            np.save(output_dir / f"{output_stem}.npy", prediction.astype(np.uint8))
            _save_prediction_nifti(
                prediction, image_path, output_dir / f"{output_stem}.nii.gz"
            )
    if not rows:
        print("No labels in test_cases; quantitative evaluation skipped.")
        return {"cases": [], "average": {name: 0.0 for name in EVALUATION_METRICS}}
    average = {
        name: float(np.mean([row[name] for row in rows]))
        for name in EVALUATION_METRICS
    }
    for name in ("assd", "hd95"):
        values = [row[name] for row in rows if name in row]
        if values:
            average[name] = float(np.mean(values))
    result = {"cases": rows, "average": average}
    if save_predictions:
        _write_json(output_dir / "results_.json", result)
    print(
        "evaluation "
        + " ".join(f"{name}={average[name]:.4f}" for name in EVALUATION_METRICS)
    )
    return result


def parse_args():
    parser = argparse.ArgumentParser(description="VoxTell offline source-free domain adaptation")
    parser.add_argument("--data_dir", required=True)
    evaluation_mode = parser.add_mutually_exclusive_group()
    evaluation_mode.add_argument(
        "--no_eval", action="store_true",
        help="Skip evaluation on test_cases after adaptation",
    )
    evaluation_mode.add_argument(
        "--eval_only", action="store_true",
        help="Load --checkpoint and run evaluation without further adaptation",
    )
    parser.add_argument("--voxtell_root", default=DEFAULT_VOXTELL_ROOT)
    parser.add_argument("--model_dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--prompt", default="liver")
    parser.add_argument("--output_dir", default="results_/voxtell_sfda")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="Evaluate the test split every N epochs and at the final epoch")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--ema_momentum", type=float, default=0.99)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--selection_p", type=float, default=0.1,
                        help="Fraction of augmented views retained by quality+entropy ranking")
    parser.add_argument("--num_aug_views", type=int, default=9,
                        help="Number of aligned views used for quality selection")
    parser.add_argument("--w_seg", type=float, default=1.0)
    parser.add_argument("--w_entropy", type=float, default=0.01)
    parser.add_argument("--w_cac", "--w_contrast", dest="w_cac", type=float, default=1.0,
                        help="Weight of the CAC contrast loss")
    parser.add_argument(
        "--quality_mode", default="cac",
        choices=("cac", "purity", "completeness", "tse"),
        help="Pseudo-label quality used for view selection and optional quality loss",
    )
    parser.add_argument(
        "--quality_metric", default="cac", choices=("cac", "saaf"),
        help="View-selection metric; SAAF uses frozen text anchor 'liver' and w_quality must be 0",
    )
    parser.add_argument(
        "--quality_config",
        default=str(Path(__file__).resolve().parent / "configs" / "tse.json"),
        help="JSON containing TSE feature layers, seed thresholds and temperature",
    )
    parser.add_argument(
        "--w_quality", type=float, default=0.0,
        help="Weight of -quality for purity/completeness/TSE modes (0 disables it)",
    )
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--amp_init_scale", type=float, default=1024.0,
                        help="Initial CUDA AMP gradient scale")
    parser.add_argument("--record_soft_prompt_grad_norm", action="store_true",
                        help="Record the soft prompt gradient norm after each update")
    parser.add_argument("--print_freq", type=int, default=10)
    parser.add_argument("--seed", type=int, default=1377)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.quality_metric == "saaf" and str(args.prompt).lower() != "liver":
        raise ValueError("SAAF uses the fixed prompt 'liver'; set --prompt liver")
    if args.eval_only and not args.checkpoint:
        raise ValueError("--eval_only requires --checkpoint")
    if args.eval_interval <= 0:
        raise ValueError(f"--eval_interval must be positive, got {args.eval_interval}")
    seed_everything(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(json.dumps(vars(args), indent=2, default=str), encoding="utf-8")

    predictor = build_predictor(args.model_dir, device, args.voxtell_root)
    qwen_text_encoder = next(
        (
            getattr(predictor, name)
            for name in ("qwen_text_encoder", "text_encoder", "qwen_model")
            if isinstance(getattr(predictor, name, None), torch.nn.Module)
        ),
        None,
    )
    # Qwen is intentionally called once, without autograd, to initialize the
    # free soft prompt. It is not retained in the adaptation optimizer.
    with torch.no_grad():
        initial_soft_prompt = predictor.embed_text_prompts([args.prompt]).detach()
        text_anchor = predictor.embed_text_prompts(["liver"]).detach()
    print(
        f"VoxTell: {args.model_dir}\n"
        "Adaptation: soft_prompt_embedding only\n"
        f"Prompt initialization: Qwen encoder ({args.prompt})\n"
        f"Pseudo-label quality: {args.quality_metric}"
    )
    adapter = VoxTellPromptSFDA(
        predictor.network,
        initial_soft_prompt,
        device,
        args,
        qwen_text_encoder=qwen_text_encoder,
        text_anchor=text_anchor,
    )
    checkpoint = None
    if args.checkpoint:
        checkpoint = load_sfda_checkpoint(args.checkpoint, adapter)
    test_entries = (
        read_image_entries(args.data_dir, "test") if not args.no_eval else None
    )
    epoch_records = []
    epoch_metrics_path = output_dir / "epoch_metrics.json"
    last_path = output_dir / "last.pt"
    predictor.network = adapter.model

    def evaluate_epoch(epoch, _training_row, training_history):
        if not should_evaluate_epoch(epoch, args.epochs, args.eval_interval):
            return
        # Preserve the completed epoch before potentially expensive evaluation.
        save_sfda_checkpoint(str(last_path), adapter, args, training_history)
        evaluation = evaluate(
            predictor,
            test_entries,
            adapter.soft_prompt_embedding.detach(),
            output_dir / "predictions",
            save_predictions=epoch == args.epochs,
            include_surface_metrics=epoch == args.epochs,
        )
        epoch_records.append({"epoch": int(epoch), **evaluation})
        _write_json(epoch_metrics_path, _epoch_metrics_payload(epoch_records))

    if not args.eval_only:
        loader = make_target_loader(
            args.data_dir,
            tuple(predictor.patch_size),
            args.batch_size,
            args.num_workers,
        )
        history = adapter.fit(
            loader,
            epoch_end_callback=evaluate_epoch if not args.no_eval else None,
        )
        save_sfda_checkpoint(str(last_path), adapter, args, history)
    else:
        history = checkpoint.get("history", [])
        checkpoint_epoch = int(history[-1].get("epoch", 0)) if history else 0
        evaluation = evaluate(
            predictor,
            test_entries,
            adapter.soft_prompt_embedding.detach(),
            output_dir / "predictions",
            save_predictions=True,
            include_surface_metrics=True,
        )
        epoch_records.append({"epoch": checkpoint_epoch, **evaluation})
        _write_json(epoch_metrics_path, _epoch_metrics_payload(epoch_records))
    adapter.close()


if __name__ == "__main__":
    main()
