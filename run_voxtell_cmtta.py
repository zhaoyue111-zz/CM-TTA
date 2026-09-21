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


def binary_diagnostic_metrics(
    prediction: np.ndarray, target: np.ndarray
) -> dict[str, float]:
    """Optional smoke-test metrics; never used for training or selection."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape:
        raise ValueError(f"Prediction/label shape mismatch: {prediction.shape} vs {target.shape}")
    prediction_mass = int(prediction.sum())
    target_mass = int(target.sum())
    true_positive = int(np.logical_and(prediction, target).sum())
    return {
        "Dice": binary_metrics(prediction, target)["Dice"],
        "Precision": float(true_positive / prediction_mass) if prediction_mass else 1.0,
        "Recall": float(true_positive / target_mass) if target_mass else (1.0 if prediction_mass == 0 else 0.0),
        "prediction_foreground_volume": float(prediction_mass),
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
    include_diagnostic_metrics: bool = False,
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
    row = {"basename": image_path.name, **metrics}
    if include_diagnostic_metrics:
        row.update(binary_diagnostic_metrics(prediction, target))
    return row


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


def evaluate_view_gt_metrics_before_adaptation(
    predictor,
    label_path: Path,
    data: torch.Tensor,
    bbox,
    original_shape,
    text_feature: torch.Tensor,
    view_params: list[dict[str, float]],
) -> list[dict[str, float | int]]:
    """Evaluate every view with the exact pre-adaptation selection embedding."""
    target = np.squeeze(load_ras_label(str(label_path)))
    metrics = []
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
                "GT_Dice_before_adaptation": binary["Dice"],
                "mIoU_before_adaptation": binary["mIoU"],
            }
        )
    return metrics


def attach_view_selection_metrics(
    view_gt_metrics: list[dict[str, float | int]],
    view_selection: dict[str, list[float] | int | None],
) -> list[dict[str, float | bool | int | str | None]]:
    """Attach post-selection diagnostics without recomputing GT predictions."""
    selected_view = int(view_selection["selected_view"])
    metrics = []
    for base in view_gt_metrics:
        view_index = int(base["view"])
        metrics.append(
            {
                **base,
                "CAC": float(view_selection["cac"][view_index]),
                "TDC": (
                    None
                    if view_selection["tdc"] is None
                    else float(view_selection["tdc"][view_index])
                ),
                "entropy": float(view_selection["entropy"][view_index]),
                "CAC_rank": float(view_selection["cac_rank"][view_index]),
                "TDC_rank": (
                    None
                    if view_selection["tdc_rank"] is None
                    else float(view_selection["tdc_rank"][view_index])
                ),
                "entropy_rank": float(view_selection["entropy_rank"][view_index]),
                "CAC_entropy_rank": float(
                    view_selection["cac_entropy_rank"][view_index]
                ),
                "TDC_entropy_rank": (
                    None
                    if view_selection["tdc_entropy_rank"] is None
                    else float(view_selection["tdc_entropy_rank"][view_index])
                ),
                "CAC_combined_rank": float(
                    view_selection["cac_combined_rank"][view_index]
                ),
                "TDC_combined_rank": (
                    None
                    if view_selection["tdc_combined_rank"] is None
                    else float(view_selection["tdc_combined_rank"][view_index])
                ),
                "combined_rank": float(view_selection["combined_rank"][view_index]),
                "selection_metric": view_selection["selection_metric"],
                "selected": view_index == selected_view,
            }
        )
    return metrics


def selector_only_case_report(
    view_metrics: list[dict[str, float | bool | int | str | None]],
    view_selection: dict[str, list[float] | int | None],
) -> dict:
    """Evaluate all four top-1 selectors on one shared frozen-view result."""
    if not view_metrics:
        raise ValueError("Selector-only evaluation requires at least one view")
    if view_selection["tdc"] is None:
        raise RuntimeError("Selector-only evaluation requires TDC statistics")
    dice = np.asarray(
        [float(metric["GT_Dice_before_adaptation"]) for metric in view_metrics],
        dtype=np.float64,
    )
    if not np.isfinite(dice).all():
        raise RuntimeError("Selector-only GT Dice contains a non-finite value")
    oracle_view = int(np.argmax(dice))
    oracle_dice = float(dice[oracle_view])
    rank_values = {
        "cac_only": view_selection["cac_rank"],
        "tdc_only": view_selection["tdc_rank"],
        "cac_entropy": view_selection["cac_combined_rank"],
        "tdc_entropy": view_selection["tdc_combined_rank"],
    }
    ranks = {}
    for name, values in rank_values.items():
        if values is None:
            raise RuntimeError(f"Selector-only statistics missing {name} ranks")
        rank = np.asarray(values, dtype=np.float64)
        if rank.shape != dice.shape:
            raise RuntimeError(
                f"{name} rank count {rank.size} does not match view count {dice.size}"
            )
        if not np.isfinite(rank).all():
            raise RuntimeError(f"Selector-only {name} ranks contain non-finite values")
        ranks[name] = rank
    selectors = {}
    for name, rank in ranks.items():
        selected_view = int(np.argsort(rank, kind="stable")[0])
        selected_dice = float(dice[selected_view])
        selectors[name] = {
            "selected_view": selected_view,
            "selected_GT_Dice": selected_dice,
            "oracle_best_view": oracle_view,
            "oracle_GT_Dice": oracle_dice,
            "regret": oracle_dice - selected_dice,
            "exact_top1_hit": selected_view == oracle_view,
        }
    return {"oracle_best_view": oracle_view, "oracle_GT_Dice": oracle_dice, "selectors": selectors}


def summarize_selector_only(reports: list[dict]) -> dict[str, dict[str, float | int]]:
    """Summarize hit rate and Dice regret for selector-only reports."""
    methods = ("cac_only", "tdc_only", "cac_entropy", "tdc_entropy")
    summary = {}
    count = len(reports)
    for method in methods:
        rows = [report["selectors"][method] for report in reports]
        regrets = np.asarray([float(row["regret"]) for row in rows], dtype=np.float64)
        hits = sum(bool(row["exact_top1_hit"]) for row in rows)
        summary[method] = {
            "exact_top1_hit_count": int(hits),
            "exact_top1_hit_rate": float(hits / max(1, count)),
            "mean_regret": float(regrets.mean()) if count else 0.0,
            "max_regret": float(regrets.max()) if count else 0.0,
        }
    return summary


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
    parser.add_argument(
        "--view_selection_metric",
        choices=("cac", "tdc"),
        default="cac",
        help="View quality used for rank fusion; default preserves CM-TTA CAC.",
    )
    entropy_group = parser.add_mutually_exclusive_group()
    entropy_group.add_argument(
        "--use_entropy_rank",
        dest="use_entropy_rank",
        action="store_true",
        help="Include entropy rank in view selection (default).",
    )
    entropy_group.add_argument(
        "--no_entropy_rank",
        dest="use_entropy_rank",
        action="store_false",
        help="Use only the selected CAC/TDC quality rank.",
    )
    parser.set_defaults(use_entropy_rank=True)
    parser.add_argument(
        "--selector_only_eval",
        action="store_true",
        help="Evaluate CAC/TDC selectors without any adaptation or optimizer update.",
    )
    parser.add_argument("--num_aug_views", type=int, default=9)
    parser.add_argument("--view_batch_size", type=int, default=1)
    parser.add_argument("--w_cac", type=float, default=1.0)
    parser.add_argument("--w_entropy", type=float, default=0.1)
    parser.add_argument(
        "--pseudo_update_mode",
        choices=("original", "decoder_masked"),
        default="original",
        help="Pseudo-label supervision; original preserves CM-TTA soft Dice.",
    )
    parser.add_argument("--bg_threshold", type=float, default=0.1)
    parser.add_argument("--tversky_alpha", type=float, default=0.3)
    parser.add_argument("--tversky_beta", type=float, default=0.7)
    parser.add_argument("--tversky_weight", type=float, default=1.0)
    parser.add_argument(
        "--decoder_alignment_check",
        action="store_true",
        help=(
            "Validate ordinary logits against decoder D5 and teacher/student "
            "region probabilities during decoder-masked adaptation."
        ),
    )
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
    selector_reports = []
    predictions_dir = output_dir / "predictions"
    try:
        for case_index, (image_path, label_path) in enumerate(entries, start=1):
            if args.selector_only_eval:
                adapter.reset_case_adaptation_state()
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
            zero_shot_sliding_miou = binary_metrics(
                zero_shot_sliding_prediction, zero_shot_target
            )["mIoU"]
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

            # Prepare the exact short prompt and augmentation parameters that
            # adapt_case() will use for view selection, without selecting a
            # view or updating ctx.  All GT view diagnostics are computed
            # from this pre-adaptation embedding.
            prepared_case = adapter.prepare_case(patches, valid_masks)
            with torch.no_grad():
                selection_text_feature_before = adapter._encode_ctx(
                    prepared_case["short_ctx"].detach()
                ).detach()
            view_gt_metrics_before = evaluate_view_gt_metrics_before_adaptation(
                predictor,
                label_path,
                data,
                bbox,
                original_shape,
                selection_text_feature_before,
                prepared_case["params"],
            )

            if args.selector_only_eval:
                # Compute both quality families from this one shared frozen
                # view batch.  No backward, optimizer, or prompt update occurs.
                adapter._select_case_view(
                    patches,
                    prepared_case["params"],
                    prepared_case["short_ctx"],
                    prepared_case["valid_masks"],
                    collect_tdc=True,
                )
                selection = dict(adapter.last_view_selection)
                view_metrics = attach_view_selection_metrics(
                    view_gt_metrics_before, selection
                )
                selector_report = selector_only_case_report(view_metrics, selection)
                selector_reports.append(selector_report)
                row = {
                    "basename": image_path.name,
                    "Dice": zero_shot_sliding_dice,
                    "mIoU": zero_shot_sliding_miou,
                    "selected_view": None,
                    "selected_view_GT_Dice_before_adaptation": None,
                    "Dice_change_from_before_adaptation": 0.0,
                    "selector_only_eval": True,
                    "view_metrics": view_metrics,
                    "selector_results": selector_report["selectors"],
                    "zero_shot_sliding_dice": zero_shot_sliding_dice,
                    "zero_shot_nonoverlap_dice": zero_shot_nonoverlap_dice,
                    "zero_shot_patch_gap": zero_shot_patch_gap,
                }
                case_rows.append(row)
                for view_metric in view_metrics:
                    print(
                        f"case {case_index}/{len(entries)} {image_path.name} "
                        f"view={view_metric['view']} "
                        f"GT_Dice_before_adaptation="
                        f"{view_metric['GT_Dice_before_adaptation']:.4f} "
                        f"CAC={view_metric['CAC']:.6f} "
                        f"TDC={view_metric['TDC']} "
                        f"entropy={view_metric['entropy']:.6f} "
                        f"CAC_rank={view_metric['CAC_rank']:.1f} "
                        f"TDC_rank={view_metric['TDC_rank']} "
                        f"entropy_rank={view_metric['entropy_rank']:.1f} "
                        f"combined_rank={view_metric['combined_rank']:.1f} "
                        f"selected={view_metric['selected']}"
                    )
                print(
                    f"case {case_index}/{len(entries)} {image_path.name} "
                    f"selector_only={json.dumps(selector_report['selectors'], sort_keys=True)}"
                )
                history.append(
                    {
                        "case": image_path.name,
                        "selector_only_eval": True,
                        "view_metrics": view_metrics,
                        "selector_results": selector_report["selectors"],
                        "zero_shot_sliding_dice": zero_shot_sliding_dice,
                        "zero_shot_nonoverlap_dice": zero_shot_nonoverlap_dice,
                        "zero_shot_patch_gap": zero_shot_patch_gap,
                        **selection,
                    }
                )
                continue

            # One complete case is one adaptation time step.  adapt_case sums
            # all patch losses and performs exactly one optimizer/LSPM update.
            with torch.no_grad():
                prompt_embedding_before = adapter._encode_ctx(
                    adapter.ctx_delta.detach()
                ).detach()
            trace = adapter.adapt_case(
                patches, valid_masks, prepared_case=prepared_case
            )
            if len(view_gt_metrics_before) != trace["num_views"]:
                raise RuntimeError(
                    "Pre-adaptation GT view diagnostics do not match the case view count: "
                    f"{len(view_gt_metrics_before)} vs {trace['num_views']}"
                )
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
            view_metrics = attach_view_selection_metrics(
                view_gt_metrics_before, trace["view_selection"]
            )
            selected_view_dice = view_metrics[trace["selected_view"]][
                "GT_Dice_before_adaptation"
            ]
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
                include_diagnostic_metrics=args.decoder_alignment_check,
            )
            row["adaptation_quality"] = trace["selected_cac"]
            row["selected_view"] = trace["selected_view"]
            row["selected_view_GT_Dice_before_adaptation"] = selected_view_dice
            row["Dice_change_from_before_adaptation"] = (
                row["Dice"] - zero_shot_sliding_dice
            )
            row["view_metrics"] = row_view_metrics
            row["zero_shot_sliding_dice"] = zero_shot_sliding_dice
            row["zero_shot_nonoverlap_dice"] = zero_shot_nonoverlap_dice
            row["zero_shot_patch_gap"] = zero_shot_patch_gap
            # Keep the per-case pseudo-update diagnostics in results.json as
            # well as in the checkpoint history, including mask statistics and
            # the replacement BCE/Tversky terms when enabled.
            row["adaptation_trace"] = trace
            case_rows.append(row)
            for view_metric in view_metrics:
                print(
                    f"case {case_index}/{len(entries)} {image_path.name} "
                    f"view={view_metric['view']} "
                    f"GT_Dice_before_adaptation="
                    f"{view_metric['GT_Dice_before_adaptation']:.4f} "
                    f"CAC={view_metric['CAC']:.6f} "
                    f"TDC={view_metric['TDC']} "
                    f"entropy={view_metric['entropy']:.6f} "
                    f"TDC_rank={view_metric['TDC_rank']} "
                    f"entropy_rank={view_metric['entropy_rank']:.1f} "
                    f"combined_rank={view_metric['combined_rank']:.1f} "
                    f"selection_metric={view_metric['selection_metric']} "
                    f"selected={view_metric['selected']}"
                )
            print(
                f"case {case_index}/{len(entries)} {image_path.name} "
                "GT_Dice_before_adaptation_vector="
                + ",".join(
                    f"{metric['GT_Dice_before_adaptation']:.6f}"
                    for metric in view_metrics
                )
            )
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
            if args.decoder_alignment_check:
                before_diag = binary_diagnostic_metrics(
                    zero_shot_sliding_prediction, zero_shot_target
                )
                region_diag = {
                    region: {
                        "teacher": trace.get(f"teacher_{region}_mean_probability"),
                        "student_before": trace.get(
                            f"student_before_{region}_mean_probability"
                        ),
                    }
                    for region in ("fg", "bg", "amb", "miss")
                }
                print(
                    f"case {case_index}/{len(entries)} {image_path.name} "
                    "decoder_alignment="
                    f"max={trace['decoder_alignment_max_abs_error']:.6e} "
                    f"mean={trace['decoder_alignment_mean_abs_error']:.6e}"
                )
                print(
                    f"case {case_index}/{len(entries)} {image_path.name} "
                    f"teacher_student_regions={json.dumps(region_diag, sort_keys=True)}"
                )
                print(
                    f"case {case_index}/{len(entries)} {image_path.name} "
                    f"pseudo_loss={trace.get('pseudo_loss')} "
                    f"bce={trace.get('bce_loss')} "
                    f"tversky={trace.get('tversky_loss')} "
                    f"total={trace.get('total_loss')}"
                )
                print(
                    f"case {case_index}/{len(entries)} {image_path.name} "
                    f"before_Dice={before_diag['Dice']:.6f} "
                    f"before_Precision={before_diag['Precision']:.6f} "
                    f"before_Recall={before_diag['Recall']:.6f} "
                    f"before_volume={before_diag['prediction_foreground_volume']:.0f} "
                    f"after_Dice={row['Dice']:.6f} "
                    f"after_Precision={row['Precision']:.6f} "
                    f"after_Recall={row['Recall']:.6f} "
                    f"after_volume={row['prediction_foreground_volume']:.0f}"
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
    output = {"cases": case_rows, "average": average}
    if args.selector_only_eval:
        selector_summary = summarize_selector_only(selector_reports)
        output["selector_only_summary"] = selector_summary
        print(f"Selector-only summary: {json.dumps(selector_summary, sort_keys=True)}")
    (output_dir / "results.json").write_text(
        json.dumps(output, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    save_cmtta_checkpoint(str(output_dir / "last.pt"), adapter, args, history)
    print(f"Results: Dice={average['Dice']:.4f} mIoU={average['mIoU']:.4f}")


if __name__ == "__main__":
    main()
