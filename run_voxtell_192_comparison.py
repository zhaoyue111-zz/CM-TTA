"""Run four isolated original-CM-TTA selectors on offline whole-case 192^3 data."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch

from data.voxtell_192 import (
    TARGET_SHAPE,
    bbox_slices,
    load_case_artifact,
    load_preprocessed_manifest,
    make_single_spatial_patch,
    restore_probability_to_original,
)
from method.voxtell_cmtta import VoxTellCMTTA
from run_voxtell_cmtta import binary_metrics, save_prediction_nifti, seed_everything


SELECTOR_CONFIGS = (
    {"name": "CAC", "slug": "cac", "view_selection_metric": "cac", "use_entropy_rank": False},
    {"name": "CAC+Entropy", "slug": "cac_entropy", "view_selection_metric": "cac", "use_entropy_rank": True},
    {"name": "TDC", "slug": "tdc", "view_selection_metric": "tdc", "use_entropy_rank": False},
    {"name": "TDC+Entropy", "slug": "tdc_entropy", "view_selection_metric": "tdc", "use_entropy_rank": True},
)

FORCED_ORIGINAL_CMTTA = {
    "pseudo_update_mode": "original",
    "pseudo_teacher_inference": "patch",
    "pseudo_view_weighting": "uniform",
    "pseudo_spatial_weighting": "none",
    "alignment_loss": "cac",
    "w_cac": 1.0,
    "w_entropy": 0.1,
    "response_weight": 0.0,
    "use_lora": False,
    "use_d4_local_distill": False,
    "pseudo_component_filter": "none",
    "pseudo_region_refinement": "none",
    "use_text_local_response_distill": False,
    "num_aug_views": 9,
}

DISABLED_EXPERIMENTAL_PATHS = (
    "multi_layer_supervision",
    "bce_or_tversky_pseudo_loss",
    "response_distillation",
    "text_rank_alignment",
    "region_correction",
    "sliding_window_pseudo_teacher",
    "non_original_pseudo_update_modes",
)

SUMMARY_CASE_FIELDS = (
    "original_foreground_voxel_count",
    "cropped_foreground_voxel_count",
    "gt_192_foreground_voxel_count",
    "all_views_max_Dice",
    "all_views_max_mIoU",
    "selected_view_Dice",
    "selected_view_mIoU",
    "pre_student_192_Dice",
    "pre_student_192_mIoU",
    "post_student_192_Dice",
    "post_student_192_mIoU",
    "post_teacher_original_sliding_Dice",
    "post_teacher_original_sliding_mIoU",
    "post_teacher_192_native_Dice",
    "post_teacher_192_native_mIoU",
    "post_teacher_192_to_original_Dice",
    "post_teacher_192_to_original_mIoU",
    "student_192_Dice_change",
    "student_192_mIoU_change",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Four isolated selector runs on offline whole-case VoxTell 192^3 data"
    )
    parser.add_argument("--preprocessed-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--voxtell-root", default="../VoxTell_from_disk")
    parser.add_argument("--model-dir", default=None)
    parser.add_argument("--text-model", default=None)
    parser.add_argument("--prompt", default="liver cancer")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=1377)
    parser.add_argument("--view-batch-size", type=int, default=1)
    parser.add_argument("--view-params-manifest", default=None)
    parser.add_argument("--foreground-labels", type=int, nargs="+", required=True)
    parser.add_argument("--lr", type=float, default=5e-3)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--ema-momentum", type=float, default=0.99)
    parser.add_argument("--amp-init-scale", type=float, default=1024.0)
    parser.add_argument("--short-memory-length", type=int, default=16)
    parser.add_argument("--fail-if-output-exists", action="store_true")
    return parser.parse_args()


def forced_adapter_args(base: argparse.Namespace, selector: dict, max_text_length: int) -> SimpleNamespace:
    """Create an explicit baseline-only adapter namespace with no experiment knobs."""
    values = {
        "lr": float(base.lr),
        "grad_clip": float(base.grad_clip),
        "ema_momentum": float(base.ema_momentum),
        "amp_init_scale": float(base.amp_init_scale),
        "short_memory_length": int(base.short_memory_length),
        "selection_p": 0.1,
        "view_batch_size": int(base.view_batch_size),
        "view_selection_metric": selector["view_selection_metric"],
        "use_entropy_rank": bool(selector["use_entropy_rank"]),
        "max_text_length": int(max_text_length),
        "gradient_conflict_diagnostics": False,
        "decoder_alignment_check": False,
        "response_diagnostics": False,
        "response_export_dir": None,
        "response_export_cases": None,
        "text_rank_diagnostics": False,
        "d4_local_diagnostics": False,
    }
    values.update(FORCED_ORIGINAL_CMTTA)
    return SimpleNamespace(**values)


def generate_view_params_manifest(
    case_ids: list[str], seed: int, path: str | Path, num_aug_views: int = 9
) -> dict:
    """Materialize a private-RNG augmentation stream shared by all four runs."""
    if len(case_ids) != len(set(case_ids)) or not case_ids:
        raise ValueError("case_ids must be non-empty and unique")
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    rows = []
    for case_id in case_ids:
        params = [{"scale": 1.0, "offset": 0.0}]
        for _ in range(int(num_aug_views)):
            scale = float(torch.empty(()).uniform_(0.85, 1.15, generator=generator))
            offset = float(torch.empty(()).uniform_(-0.15, 0.15, generator=generator))
            params.append({"scale": scale, "offset": offset})
        rows.append({"case_id": case_id, "view_params": params})
    payload = {
        "format": "voxtell-whole-case-192-view-params-v1",
        "seed": int(seed),
        "num_aug_views": int(num_aug_views),
        "case_order": list(case_ids),
        "cases": rows,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def load_view_params_manifest(
    path: str | Path, case_ids: list[str], num_aug_views: int = 9
) -> dict[str, list[dict[str, float]]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if payload.get("case_order") != case_ids:
        raise ValueError("Shared view_params manifest case order does not match input order")
    if int(payload.get("num_aug_views", -1)) != int(num_aug_views):
        raise ValueError("Shared view_params manifest has the wrong augmentation count")
    rows = payload.get("cases")
    if not isinstance(rows, list) or len(rows) != len(case_ids):
        raise ValueError("Shared view_params manifest case count does not match")
    result = {}
    for expected_id, row in zip(case_ids, rows):
        if row.get("case_id") != expected_id:
            raise ValueError("Shared view_params rows are not in the declared case order")
        params = row.get("view_params")
        if not isinstance(params, list) or len(params) != 1 + num_aug_views:
            raise ValueError(f"Invalid view_params count for {expected_id}")
        normalized = []
        for param in params:
            if not isinstance(param, dict) or set(param) != {"scale", "offset"}:
                raise ValueError(f"Invalid view_params entry for {expected_id}")
            scale, offset = float(param["scale"]), float(param["offset"])
            if not np.isfinite(scale) or not np.isfinite(offset):
                raise ValueError(f"Non-finite view parameter for {expected_id}")
            normalized.append({"scale": scale, "offset": offset})
        if normalized[0] != {"scale": 1.0, "offset": 0.0}:
            raise ValueError(f"View zero must be unaugmented for {expected_id}")
        result[expected_id] = normalized
    return result


def adapter_initial_state(adapter: VoxTellCMTTA) -> dict:
    ctx = adapter.ctx_delta.detach().float().cpu().contiguous()
    digest = hashlib.sha256(ctx.numpy().tobytes()).hexdigest()
    return {
        "ctx_delta_sha256": digest,
        "ctx_delta_norm": float(ctx.norm()),
        "ctx_delta_shape": list(ctx.shape),
        "long_delta_is_none": adapter.long_delta is None,
        "short_delta_is_none": adapter.short_delta is None,
        "short_memory_size": len(adapter.short_memory),
        "optimizer_state_size": len(adapter.optimizer.state),
        "optimizer_step_count": int(adapter.optimizer_step_count),
    }


def assert_fresh_adapter(adapter: VoxTellCMTTA, reference: dict | None = None) -> dict:
    state = adapter_initial_state(adapter)
    if not (
        state["long_delta_is_none"]
        and state["short_delta_is_none"]
        and state["short_memory_size"] == 0
        and state["optimizer_state_size"] == 0
        and state["optimizer_step_count"] == 0
    ):
        raise RuntimeError("Selector run did not start from a fresh adapter state")
    if adapter.use_lora or len(adapter.trainable_parameters) != 1:
        raise RuntimeError("Whole-case baseline must train only ctx_delta")
    if reference is not None and state != reference:
        raise RuntimeError("Selector adapters did not start from identical prompt state")
    return state


def frozen_model_version_signature(model) -> tuple:
    """Cheap guard that detects any in-place parameter or buffer mutation."""
    tensors = list(model.named_parameters()) + list(model.named_buffers())
    return tuple(
        (name, id(tensor), int(tensor._version), tuple(tensor.shape), str(tensor.dtype))
        for name, tensor in tensors
    )


def prepare_adaptation_case(adapter: VoxTellCMTTA, image_192, view_params: list[dict]):
    """Prepare one case without accepting a GT argument by construction."""
    patches, valid_masks, locations = make_single_spatial_patch(image_192)
    prepared = adapter.prepare_case(
        patches,
        valid_masks,
        view_params_override=view_params,
        patch_locations=locations,
    )
    if len(prepared["patches"]) != 1 or prepared["patch_locations"] != [(0, 0, 0)]:
        raise RuntimeError("Whole-case 192 experiment produced more than one spatial patch")
    return prepared


def adapt_prepared_case(adapter: VoxTellCMTTA, prepared: dict, case_id: str) -> dict:
    """Run adaptation without accepting or forwarding any GT object."""
    before_steps = int(adapter.optimizer_step_count)
    trace = adapter.adapt_case(
        prepared["patches"],
        prepared["valid_masks"],
        prepared_case=prepared,
        case_basename=case_id,
        adaptation_step=before_steps + 1,
        patch_locations=prepared["patch_locations"],
    )
    step_delta = int(adapter.optimizer_step_count) - before_steps
    successful_steps = int(trace.get("successful_optimizer_steps_for_case", 0))
    if (
        step_delta != 1
        or int(trace["optimizer_steps_for_case"]) != 1
        or successful_steps != 1
        or bool(trace["optimizer_step_skipped"])
    ):
        raise RuntimeError(
            f"Case {case_id} did not complete exactly one successful optimizer step"
        )
    if int(trace.get("spatial_patch_count", trace["num_patches"])) != 1:
        raise RuntimeError(f"Case {case_id} did not use exactly one spatial patch")
    if adapter.long_delta is None:
        raise RuntimeError("Original LSPM/EMA update did not produce adapter.long_delta")
    return trace


def official_window_count(predictor, data: torch.Tensor) -> int:
    """Count positions through VoxTell's own sliding-window slicer."""
    if not np.isclose(float(predictor.tile_step_size), 0.5):
        raise AssertionError(
            f"Official VoxTell evaluation requires 50% overlap, got {predictor.tile_step_size}"
        )
    patch_size = tuple(int(value) for value in predictor.patch_size)
    if tuple(data.shape[-3:]) == TARGET_SHAPE and patch_size != TARGET_SHAPE:
        raise AssertionError(f"VoxTell patch_size must be {TARGET_SHAPE}, got {patch_size}")
    from acvl_utils.cropping_and_padding.padding import pad_nd_image

    padded, _ = pad_nd_image(
        data, patch_size, "constant", {"value": 0}, True, None
    )
    return len(predictor._internal_get_sliding_window_slicers(tuple(padded.shape[1:])))


def predict_official_probability(
    predictor, data: np.ndarray | torch.Tensor, text_feature: torch.Tensor, *, assert_one_window: bool = False
) -> np.ndarray:
    tensor = torch.as_tensor(data, dtype=torch.float32).contiguous()
    if tensor.ndim != 4 or tensor.shape[0] != 1:
        raise ValueError(f"Official inference expects (1,D,H,W), got {tuple(tensor.shape)}")
    count = official_window_count(predictor, tensor)
    if assert_one_window and count != 1:
        raise AssertionError(f"Offline 192^3 input generated {count} official windows, expected one")
    with torch.no_grad():
        logits = predictor.predict_sliding_window_return_logits(
            tensor, text_feature.to(predictor.device)
        ).float().cpu()
    while logits.ndim > 3 and logits.shape[0] == 1:
        logits = logits[0]
    if tuple(logits.shape) != tuple(tensor.shape[-3:]):
        raise ValueError(f"Official logits/input mismatch: {tuple(logits.shape)} vs {tuple(tensor.shape[-3:])}")
    return torch.sigmoid(logits).numpy().astype(np.float32, copy=False)


def load_original_binary_gt(gt_path: Path, foreground_labels: list[int]) -> np.ndarray:
    """Evaluation-only RAS GT loading; this function is never called by adaptation."""
    try:
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
    except ModuleNotFoundError as error:
        raise RuntimeError("Original-space evaluation requires nnunetv2") from error
    raw = NibabelIOWithReorient().read_images([str(gt_path)])[0]
    return np.isin(raw[0], foreground_labels).astype(np.uint8)


def evaluate_probability(probability: np.ndarray, target: np.ndarray) -> dict[str, float]:
    """Threshold only at the evaluation boundary."""
    if int(np.count_nonzero(target)) == 0:
        raise ValueError(
            "Empty evaluation GT is not allowed in the whole-case 192 experiment"
        )
    return binary_metrics(np.asarray(probability) >= 0.5, np.asarray(target) > 0)


def evaluate_selection_context_views(
    predictor,
    image_192: np.ndarray,
    gt_192: np.ndarray,
    selection_text_feature: torch.Tensor,
    view_params: list[dict[str, float]],
) -> list[dict]:
    """Evaluate all views with the exact prepared short context used by selection."""
    rows = []
    for view_index, param in enumerate(view_params):
        view = image_192 if view_index == 0 else (
            image_192 * float(param["scale"]) + float(param["offset"])
        )
        probability = predict_official_probability(
            predictor, view, selection_text_feature, assert_one_window=True
        )
        metrics = evaluate_probability(probability, gt_192[0])
        rows.append(
            {
                "view_index": view_index,
                "scale": float(param["scale"]),
                "offset": float(param["offset"]),
                "selection_context_192_Dice": metrics["Dice"],
                "selection_context_192_mIoU": metrics["mIoU"],
            }
        )
    return rows


def summarize_selection_context_views(
    view_metrics: list[dict], selected_view: int
) -> dict[str, float | int | bool]:
    """Compute oracle/selection diagnostics from short-context view metrics only."""
    dice_values = np.asarray(
        [row["selection_context_192_Dice"] for row in view_metrics],
        dtype=np.float64,
    )
    miou_values = np.asarray(
        [row["selection_context_192_mIoU"] for row in view_metrics],
        dtype=np.float64,
    )
    if not 0 <= int(selected_view) < len(view_metrics):
        raise IndexError("selected_view is outside the selection-context metrics")
    oracle_best_view = int(np.argmax(dice_values))
    return {
        "oracle_best_view": oracle_best_view,
        "all_views_max_Dice": float(dice_values.max()),
        "all_views_max_mIoU": float(miou_values.max()),
        "selected_view_Dice": float(dice_values[selected_view]),
        "selected_view_mIoU": float(miou_values[selected_view]),
        "selected_view_hits_oracle": int(selected_view) == oracle_best_view,
        "selected_view_Dice_regret_to_oracle": float(
            dice_values.max() - dice_values[selected_view]
        ),
    }


def summarize_cases(cases: list[dict]) -> dict:
    """Recomputable case-wise macro summary; no voxel-weighted aggregation."""
    if not cases:
        raise ValueError("Cannot summarize an empty case list")
    summary = {"case_count": len(cases)}
    for field in SUMMARY_CASE_FIELDS:
        summary[f"mean_{field}"] = float(np.mean([float(row[field]) for row in cases]))
    hits = sum(bool(row["selected_view_hits_oracle"]) for row in cases)
    summary.update(
        {
            "selected_view_oracle_hit_count": int(hits),
            "selected_view_oracle_hit_rate": float(hits / len(cases)),
            "mean_selected_view_Dice_regret_to_oracle": float(
                np.mean([float(row["selected_view_Dice_regret_to_oracle"]) for row in cases])
            ),
            "foreground_erased_by_resize_case_count": int(
                sum(bool(row["foreground_erased_by_resize"]) for row in cases)
            ),
            "foreground_erased_by_resize_case_rate": float(
                np.mean([bool(row["foreground_erased_by_resize"]) for row in cases])
            ),
        }
    )
    return summary


def _csv_value(value):
    return json.dumps(value, separators=(",", ":")) if isinstance(value, (dict, list, tuple)) else value


def write_cases_csv(path: Path, rows: list[dict]) -> None:
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key)) for key in fields})


def build_predictor(args):
    root = Path(args.voxtell_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from voxtell.inference.predictor import VoxTellPredictor

    model_dir = Path(args.model_dir).expanduser().resolve() if args.model_dir else root / "model"
    text_model = Path(args.text_model).expanduser().resolve() if args.text_model else model_dir / "Qwen3-Embedding-4B"
    return VoxTellPredictor(
        model_dir=str(model_dir), device=args.device, text_encoding_model=str(text_model)
    )


def run_configuration(
    base_args: argparse.Namespace,
    selector: dict,
    predictor,
    entries,
    shared_view_params: dict[str, list[dict[str, float]]],
    output_dir: Path,
    reference_initial_state: dict | None,
) -> tuple[list[dict], dict, dict]:
    adapter_args = forced_adapter_args(base_args, selector, predictor.max_text_length)
    adapter = VoxTellCMTTA(
        predictor.network,
        None,
        base_args.device,
        adapter_args,
        qwen_text_encoder=predictor.text_backbone,
        qwen_tokenizer=predictor.tokenizer,
        text_prompt=base_args.prompt,
    )
    predictor.network = adapter.model
    initial_state = assert_fresh_adapter(adapter, reference_initial_state)
    config_dir = output_dir / selector["slug"]
    predictions_dir = config_dir / "predictions"
    predictions_dir.mkdir(parents=True, exist_ok=True)
    config_payload = {
        **vars(base_args),
        **vars(adapter_args),
        "selector_name": selector["name"],
        "output_dir": str(config_dir),
        "trainable_parameters": ["ctx_delta"],
        "disabled_experimental_paths": list(DISABLED_EXPERIMENTAL_PATHS),
    }
    (config_dir / "args.json").write_text(
        json.dumps(config_payload, indent=2, default=str), encoding="utf-8"
    )
    case_rows = []
    try:
        for execution_order, entry in enumerate(entries, start=1):
            artifact = load_case_artifact(entry)
            if artifact["case_id"] != entry.case_id:
                raise ValueError(f"Artifact case_id mismatch for {entry.case_id}")
            if artifact["image_path"].resolve() != entry.image_path.resolve() or artifact["gt_path"].resolve() != entry.gt_path.resolve():
                raise ValueError(f"Artifact source paths mismatch for {entry.case_id}")
            artifact_labels = sorted(int(value) for value in artifact["foreground_labels"])
            if artifact_labels != list(base_args.foreground_labels):
                raise ValueError(
                    f"Foreground labels differ from offline preprocessing for {entry.case_id}: "
                    f"artifact={artifact_labels}, CLI={base_args.foreground_labels}"
                )
            original_gt = load_original_binary_gt(
                entry.gt_path, base_args.foreground_labels
            )
            if tuple(original_gt.shape) != artifact["original_shape"]:
                raise ValueError(f"Original GT/recovery shape mismatch for {entry.case_id}")
            observed_original_foreground = int(np.count_nonzero(original_gt))
            if observed_original_foreground != artifact["original_foreground_voxel_count"]:
                raise ValueError(
                    f"Original foreground count mismatch for {entry.case_id}: "
                    f"artifact={artifact['original_foreground_voxel_count']}, "
                    f"observed={observed_original_foreground}"
                )
            if observed_original_foreground == 0:
                raise ValueError(
                    f"Case {entry.case_id}: empty original GT is excluded from metrics"
                )
            params = shared_view_params[entry.case_id]
            prepared = prepare_adaptation_case(adapter, artifact["image_192"], params)

            # Explicit text state 1: raw student ctx_delta immediately before step.
            pre_student_delta = adapter.ctx_delta.detach().clone()
            with torch.no_grad():
                pre_student_text = adapter._encode_ctx(pre_student_delta).detach()
                selection_text = adapter._encode_ctx(
                    prepared["short_ctx"].detach()
                ).detach()

            # Evaluation-only GT. It is not present in either adapter call above/below.
            gt_192 = artifact["gt_192"]
            view_metrics = evaluate_selection_context_views(
                predictor, artifact["image_192"], gt_192, selection_text, params
            )
            pre_student_probability = predict_official_probability(
                predictor,
                artifact["image_192"],
                pre_student_text,
                assert_one_window=True,
            )
            pre_student_metrics = evaluate_probability(
                pre_student_probability, gt_192[0]
            )
            trace = adapt_prepared_case(adapter, prepared, entry.case_id)

            # Explicit text state 2: student ctx_delta after optimizer.step.
            post_student_delta = adapter.ctx_delta.detach().clone()
            with torch.no_grad():
                post_student_text = adapter._encode_ctx(post_student_delta).detach()

            # Explicit text state 3: EMA long teacher after original LSPM update.
            post_teacher_delta = adapter.long_delta.detach().clone()
            with torch.no_grad():
                post_teacher_text = adapter._encode_ctx(post_teacher_delta).detach()

            selected_view = int(trace["selected_view"])
            selection = trace["view_selection"]
            selected_score = float(selection["selection_scores"][selected_view])
            selection_outcomes = summarize_selection_context_views(
                view_metrics, selected_view
            )

            post_student_probability = predict_official_probability(
                predictor, artifact["image_192"], post_student_text, assert_one_window=True
            )
            post_student_metrics = evaluate_probability(post_student_probability, gt_192[0])

            # Teacher B: official interface on 192^3, exactly one window.
            post_teacher_192_probability = predict_official_probability(
                predictor, artifact["image_192"], post_teacher_text, assert_one_window=True
            )
            post_teacher_192_metrics = evaluate_probability(post_teacher_192_probability, gt_192[0])

            # Teacher A: official 50%-overlap/Gaussian path on the saved native crop.
            native_probability = predict_official_probability(
                predictor, artifact["image_cropped_native"], post_teacher_text
            )
            original_sliding_probability = np.zeros(artifact["original_shape"], dtype=np.float32)
            original_sliding_probability[bbox_slices(artifact["bbox"])] = native_probability
            original_sliding_metrics = evaluate_probability(original_sliding_probability, original_gt)

            # Teacher C: restore soft B probability first; threshold only afterwards.
            probability_192_to_original = restore_probability_to_original(
                post_teacher_192_probability, artifact["bbox"], artifact["original_shape"]
            )
            restored_metrics = evaluate_probability(probability_192_to_original, original_gt)

            row = {
                "case_id": entry.case_id,
                "execution_order": execution_order,
                "original_foreground_voxel_count": artifact[
                    "original_foreground_voxel_count"
                ],
                "cropped_foreground_voxel_count": artifact[
                    "cropped_foreground_voxel_count"
                ],
                "gt_192_foreground_voxel_count": artifact[
                    "gt_192_foreground_voxel_count"
                ],
                "foreground_erased_by_resize": artifact[
                    "foreground_erased_by_resize"
                ],
                "view_params": [dict(param) for param in params],
                "selection_context_view_metrics_192": view_metrics,
                "selection_prompt_source": "prepared_case.short_ctx",
                "selected_view": selected_view,
                "selected_view_selection_metric": selector["view_selection_metric"],
                "selected_view_score": selected_score,
                **selection_outcomes,
                "pre_student_192_Dice": pre_student_metrics["Dice"],
                "pre_student_192_mIoU": pre_student_metrics["mIoU"],
                "post_student_192_Dice": post_student_metrics["Dice"],
                "post_student_192_mIoU": post_student_metrics["mIoU"],
                "student_192_Dice_change": post_student_metrics["Dice"] - pre_student_metrics["Dice"],
                "student_192_mIoU_change": post_student_metrics["mIoU"] - pre_student_metrics["mIoU"],
                "post_teacher_original_sliding_Dice": original_sliding_metrics["Dice"],
                "post_teacher_original_sliding_mIoU": original_sliding_metrics["mIoU"],
                "post_teacher_192_native_Dice": post_teacher_192_metrics["Dice"],
                "post_teacher_192_native_mIoU": post_teacher_192_metrics["mIoU"],
                "post_teacher_192_to_original_Dice": restored_metrics["Dice"],
                "post_teacher_192_to_original_mIoU": restored_metrics["mIoU"],
                "optimizer_steps_for_case": int(trace["optimizer_steps_for_case"]),
                "successful_optimizer_steps_for_case": int(
                    trace["successful_optimizer_steps_for_case"]
                ),
                "spatial_patch_count": int(trace.get("spatial_patch_count", trace["num_patches"])),
                "amp_optimizer_step_skipped": bool(trace["optimizer_step_skipped"]),
                "pre_student_text_state": "adapter.ctx_delta before optimizer.step",
                "pre_student_prompt_source": "adapter.ctx_delta before optimizer.step",
                "post_student_text_state": "adapter.ctx_delta after optimizer.step",
                "post_teacher_text_state": "adapter.long_delta after original LSPM/EMA update",
                "post_teacher_uses_adapter_long_delta": True,
                "official_192_window_count": official_window_count(
                    predictor, torch.as_tensor(artifact["image_192"])
                ),
                "post_teacher_original_sliding_window_count": official_window_count(
                    predictor, torch.as_tensor(artifact["image_cropped_native"])
                ),
                "adaptation_trace": trace,
            }
            if (
                row["successful_optimizer_steps_for_case"] != 1
                or row["amp_optimizer_step_skipped"]
                or row["spatial_patch_count"] != 1
            ):
                raise RuntimeError(f"Protocol assertion failed for {entry.case_id}")
            case_rows.append(row)

            stem = entry.case_id
            save_prediction_nifti(
                (original_sliding_probability >= 0.5).astype(np.uint8),
                entry.image_path,
                predictions_dir / f"{stem}_post_teacher_original_sliding.nii.gz",
            )
            save_prediction_nifti(
                (probability_192_to_original >= 0.5).astype(np.uint8),
                entry.image_path,
                predictions_dir / f"{stem}_post_teacher_192_to_original.nii.gz",
            )
            np.save(
                predictions_dir / f"{stem}_post_teacher_192_native_probability.npy",
                post_teacher_192_probability.astype(np.float32, copy=False),
            )
    finally:
        adapter.close()

    summary = summarize_cases(case_rows)
    results = {
        "configuration": dict(selector),
        "forced_original_cmtta": dict(FORCED_ORIGINAL_CMTTA),
        "trainable_parameters": ["ctx_delta"],
        "disabled_experimental_paths": list(DISABLED_EXPERIMENTAL_PATHS),
        "frozen_voxtell_model_reused_unchanged_across_selectors": True,
        "initial_state": initial_state,
        "case_order": [entry.case_id for entry in entries],
        "cases": case_rows,
        "summary": summary,
    }
    (config_dir / "results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (config_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_cases_csv(config_dir / "cases.csv", case_rows)
    return case_rows, summary, initial_state


def write_combined_outputs(output_dir: Path, per_config: dict[str, dict]) -> None:
    names = [config["name"] for config in SELECTOR_CONFIGS]
    case_order = per_config[names[0]]["case_order"]
    for name in names[1:]:
        if per_config[name]["case_order"] != case_order:
            raise RuntimeError("Selector runs did not preserve identical case order")
    comparison_fields = (
        "selected_view",
        "selected_view_Dice",
        "selected_view_mIoU",
        "all_views_max_Dice",
        "all_views_max_mIoU",
        "pre_student_192_Dice",
        "pre_student_192_mIoU",
        "post_student_192_Dice",
        "post_student_192_mIoU",
        "post_teacher_original_sliding_Dice",
        "post_teacher_original_sliding_mIoU",
        "post_teacher_192_native_Dice",
        "post_teacher_192_native_mIoU",
        "post_teacher_192_to_original_Dice",
        "post_teacher_192_to_original_mIoU",
    )
    final_average = {
        name: {
            "final_metric_definition": "post_teacher_original_sliding",
            "mean_Dice": per_config[name]["summary"]["mean_post_teacher_original_sliding_Dice"],
            "mean_mIoU": per_config[name]["summary"]["mean_post_teacher_original_sliding_mIoU"],
        }
        for name in names
    }
    combined_rows = []
    for index, case_id in enumerate(case_order):
        row = {"case_id": case_id, "execution_order": index + 1}
        for name in names:
            case = per_config[name]["cases"][index]
            if case["case_id"] != case_id:
                raise RuntimeError("Combined comparison case alignment failed")
            for field in comparison_fields:
                row[f"{name}_{field}"] = case[field]
            row[f"{name}_final_mean_Dice_all_test_cases"] = final_average[name]["mean_Dice"]
            row[f"{name}_final_mean_mIoU_all_test_cases"] = final_average[name]["mean_mIoU"]
        combined_rows.append(row)
    payload = {
        "case_order": case_order,
        "cases": combined_rows,
        "final_average_all_test_cases": final_average,
        "summaries": {name: per_config[name]["summary"] for name in names},
    }
    (output_dir / "combined_selector_comparison.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    write_cases_csv(output_dir / "combined_selector_comparison.csv", combined_rows)


def main() -> None:
    args = parse_args()
    if not args.prompt.strip():
        raise ValueError("--prompt must be non-empty")
    if args.view_batch_size < 1:
        raise ValueError("--view-batch-size must be positive")
    args.foreground_labels = sorted(set(args.foreground_labels))
    args.device = torch.device(args.device)
    if args.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(args.device)
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.fail_if_output_exists and output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing output directory: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_preprocessed_manifest(args.preprocessed_manifest)
    case_ids = [entry.case_id for entry in entries]
    view_manifest_path = (
        Path(args.view_params_manifest).expanduser().resolve()
        if args.view_params_manifest
        else output_dir / "shared_view_params.json"
    )
    if args.view_params_manifest:
        shared_view_params = load_view_params_manifest(view_manifest_path, case_ids)
    else:
        generate_view_params_manifest(case_ids, args.seed, view_manifest_path)
        shared_view_params = load_view_params_manifest(view_manifest_path, case_ids)
    args.resolved_view_params_manifest = str(view_manifest_path)

    seed_everything(args.seed)
    predictor = build_predictor(args)
    predictor.network = predictor.network.to(args.device).eval()
    for parameter in predictor.network.parameters():
        parameter.requires_grad_(False)
    predictor.text_backbone.requires_grad_(False)
    initial_model_signature = frozen_model_version_signature(predictor.network)
    per_config = {}
    reference_initial_state = None
    for selector in SELECTOR_CONFIGS:
        cases, summary, initial_state = run_configuration(
            args,
            selector,
            predictor,
            entries,
            shared_view_params,
            output_dir,
            reference_initial_state,
        )
        if reference_initial_state is None:
            reference_initial_state = initial_state
        if frozen_model_version_signature(predictor.network) != initial_model_signature:
            raise RuntimeError(
                f"Frozen VoxTell model state changed during selector run {selector['name']}"
            )
        per_config[selector["name"]] = {
            "case_order": case_ids,
            "cases": cases,
            "summary": summary,
        }
    write_combined_outputs(output_dir, per_config)
    print(f"Completed four isolated selector runs in {output_dir}")


if __name__ == "__main__":
    main()
