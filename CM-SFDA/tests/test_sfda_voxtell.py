"""Weight-free checks for the VoxTell SFDA adapter.

These tests use a tiny fake network and never load VoxTell/Qwen weights.
"""

from __future__ import annotations

import json
import inspect
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

CM_SFDA = Path(__file__).resolve().parents[1]
if str(CM_SFDA) not in sys.path:
    sys.path.insert(0, str(CM_SFDA))
if str(CM_SFDA / "method") not in sys.path:
    sys.path.insert(0, str(CM_SFDA / "method"))

from sfda_voxtell import (  # noqa: E402
    ShortPromptQualityMemory,
    VoxTellPromptSFDA,
    apply_recall_recovery,
    cac_loss,
    compute_cac_score,
    compute_tdc_consensus,
    tdc_from_components,
    entropy_loss,
    load_sfda_checkpoint,
    masked_segmentation_loss,
    save_sfda_checkpoint,
    select_cac_views,
    select_tdc_views,
    TDC_PAIR_NAMES,
)
from semantic_quality import (  # noqa: E402
    SemanticPrototypeMemory,
    attention_evidence_map,
    average_tie_ranks,
    compute_saaf_quality,
    compute_tse_components,
    final_decoder_attention,
)
from data.sfda_voxtell import (  # noqa: E402
    fuse_volume_patches,
    nonoverlapping_patch_locations,
    pad_to_patch_grid,
    sliding_window_locations,
)
import data.sfda_voxtell as sfda_data  # noqa: E402
from run_sfda_voxtell import (  # noqa: E402
    _native_spacing,
    binary_segmentation_metrics,
    parse_args,
    should_evaluate_epoch,
)
from evaluate_quality_metrics import (  # noqa: E402
    _fuse_logits_then_sigmoid,
    _fuse_evidence_with_valid_weights,
    _full_volume_selection_valid,
    _ground_truth_metrics_per_view,
    _labeled_audit_entries,
    _patch_oracle_stats,
    _project_prompt_features,
    _tensor_statistics,
    _write_case_global_views,
    _crop_case_global_views,
    _soft_consistency,
    _selection_indices,
    binary_auprc,
    binary_auroc,
    evidence_localization_metrics,
    parse_args as parse_quality_args,
    compute_tra_score,
    summarize_quality_audit,
    summarize_tra_cac,
    spearman,
)


class _VisionProjection(nn.Module):
    def forward(self, features):
        # B,C,D,H,W -> B,H,W,D,C, matching the documented VoxTell hook shape.
        return features.permute(0, 3, 4, 2, 1)


class _TextProjection(nn.Module):
    def forward(self, prompt):
        # B,1,1,C -> 1,B,C, matching the documented VoxTell hook shape.
        return prompt[:, 0, 0, :].unsqueeze(0)


class _TinyVoxTell(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Conv3d(1, 4, kernel_size=1, bias=False)
        self.project_bottleneck_embed = _VisionProjection()
        self.project_text_embed = _TextProjection()
        self.forward_calls = []
        self.prompt_calls = []
        self.image_calls = []
        self.outputs = []

    def forward(self, image, prompt):
        self.forward_calls.append((image.shape[0], torch.is_grad_enabled()))
        self.prompt_calls.append(prompt.detach().clone())
        self.image_calls.append(image.detach().clone())
        features = self.image_encoder(image)
        self.project_bottleneck_embed(features)
        text = self.project_text_embed(prompt)
        bias = text[0, :, 0].view(image.shape[0], 1, 1, 1, 1)
        logits = features[:, :1] + bias
        self.outputs.append(logits.detach().clone())
        return logits


class _TinyTDCVoxTell(_TinyVoxTell):
    def __init__(self):
        super().__init__()
        self.decoder_return_shapes = []
        self.all_negative_decoders = False

    def forward(self, image, prompt, return_decoder_outputs=False):
        logits = super().forward(image, prompt)
        if not return_decoder_outputs:
            return logits
        if self.all_negative_decoders:
            logits = torch.full_like(logits, -10.0)
        size = tuple(int(value) for value in logits.shape[2:])
        half = tuple(max(1, value // 2) for value in size)
        quarter = tuple(max(1, value // 4) for value in size)
        eighth = tuple(max(1, value // 8) for value in size)
        outputs = [
            logits,
            F.interpolate(logits, size=half, mode="trilinear", align_corners=False),
            F.interpolate(logits, size=quarter, mode="trilinear", align_corners=False),
            F.interpolate(logits, size=eighth, mode="trilinear", align_corners=False),
            F.interpolate(
                logits,
                size=tuple(max(1, value // 16) for value in size),
                mode="trilinear",
                align_corners=False,
            ),
        ]
        self.decoder_return_shapes.append([tuple(output.shape) for output in outputs])
        return outputs


class _TinyAttentionVoxTell(nn.Module):
    """VoxTell-shaped model exposing native attention from multiple layers."""

    def __init__(self):
        super().__init__()
        self.text_embedding_dim = 4
        self.image_encoder = nn.Conv3d(1, 4, kernel_size=1, bias=False)
        self.project_bottleneck_embed = _VisionProjection()
        self.project_text_embed = nn.Linear(4, 4, bias=False)
        self.prompt_calls = []
        self.uniform_attention = False
        self.forward_outputs = []

    def forward(self, image, prompt, return_diagnostics=False):
        self.prompt_calls.append(prompt.detach().clone())
        features = self.image_encoder(image)
        vision = self.project_bottleneck_embed(features)
        text = self.project_text_embed(prompt[:, 0, 0, :].unsqueeze(0))
        bias = text[0, :, 0].view(image.shape[0], 1, 1, 1, 1)
        logits = features[:, :1] + bias
        logits = F.interpolate(logits, scale_factor=2, mode="trilinear", align_corners=False)
        self.forward_outputs.append(logits.detach().clone())
        if not return_diagnostics:
            return logits
        spatial_count = int(np.prod(vision.shape[1:-1]))
        position = torch.arange(spatial_count, device=image.device, dtype=torch.float32)
        prompt_shift = prompt[:, 0, 0, 0].float() * 2.0
        center_first = torch.full_like(prompt_shift, 3.0) - prompt_shift
        center_last = torch.full_like(prompt_shift, spatial_count * 0.8) + prompt_shift
        if self.uniform_attention:
            first = torch.full(
                (image.shape[0], 1, spatial_count), 1.0 / spatial_count,
                device=image.device,
            )
            last = first.clone()
        else:
            first_logits = -torch.abs(position[None, None] - center_first[:, None, None]) / 5.0
            last_logits = -torch.abs(position[None, None] - center_last[:, None, None]) / 5.0
            first = first_logits.softmax(dim=-1)
            last = last_logits.softmax(dim=-1)
        return logits, {"cross_attention": [first, last]}


class _OverflowScaler:
    """CPU test double for one recoverable CUDA AMP overflow."""

    def __init__(self):
        self.current_scale = 1024.0

    def scale(self, loss):
        return loss

    def unscale_(self, optimizer):
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                parameter.grad.fill_(float("inf"))

    def is_enabled(self):
        return True

    def get_scale(self):
        return self.current_scale

    def step(self, _optimizer):
        return None

    def update(self):
        self.current_scale /= 2


def _args(**overrides):
    values = dict(
        lr=0.05,
        weight_decay=0.0,
        ema_momentum=0.9,
        confidence_threshold=0.5,
        selection_p=0.5,
        num_aug_views=3,
        w_seg=1.0,
        w_entropy=0.01,
        w_cac=1.0,
        w_quality=0.0,
        enable_recall_recovery=False,
        recovery_teacher_low=0.4,
        recovery_view_threshold=0.5,
        recovery_min_view_votes=2,
        quality_mode="cac",
        quality_config=str(CM_SFDA / "configs" / "tse.json"),
        grad_clip=1.0,
        record_soft_prompt_grad_norm=True,
        epochs=1,
        print_freq=100,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_adapter():
    model = _TinyVoxTell()
    qwen = nn.Linear(4, 4)
    adapter = VoxTellPromptSFDA(
        model,
        torch.ones(1, 1, 4),
        torch.device("cpu"),
        _args(),
        qwen_text_encoder=qwen,
    )
    return adapter, model, qwen


class SoftPromptOnlyTests(unittest.TestCase):
    def test_full_volume_statistics_bound_quantile_work_deterministically(self):
        values = torch.arange(10_000, dtype=torch.float32)
        first = _tensor_statistics(values, max_quantile_samples=256)
        second = _tensor_statistics(values, max_quantile_samples=256)
        self.assertAlmostEqual(first["mean"], 4999.5)
        self.assertAlmostEqual(first["std"], float(values.std(unbiased=False)), places=3)
        self.assertFalse(first["quantiles_exact"])
        self.assertEqual(first["quantile_sample_count"], 256)
        self.assertEqual(first["quantiles"], second["quantiles"])

    def test_case_global_views_are_deterministic_and_overlap_consistently(self):
        volume = torch.arange(1 * 4 * 5 * 6, dtype=torch.float32).reshape(1, 4, 5, 6)
        stored = np.empty((4, *volume.shape), dtype=np.float32)
        repeated = np.empty_like(stored)
        _write_case_global_views(volume, 4, seed=731, destination=stored)
        _write_case_global_views(volume, 4, seed=731, destination=repeated)
        np.testing.assert_array_equal(stored, repeated)

        first = _crop_case_global_views(
            stored,
            (slice(None), slice(0, 3), slice(0, 3), slice(0, 4)),
        )
        second = _crop_case_global_views(
            stored,
            (slice(None), slice(1, 4), slice(2, 5), slice(2, 6)),
        )
        # The patches overlap at full-volume coordinates D=1:3, H=2:3, W=2:4.
        torch.testing.assert_close(first[:, :, 1:3, 2:3, 2:4], second[:, :, 0:2, 0:1, 0:2])
        self.assertTrue(torch.any(first[1:] != first[:1]))

    def test_quality_audit_requires_an_adapted_checkpoint(self):
        with mock.patch.object(
            sys,
            "argv",
            [
                "evaluate_quality_metrics.py", "--data_dir", "/tmp/data",
                "--voxtell_root", "/tmp/voxtell", "--model_dir", "/tmp/model",
            ],
        ):
            with self.assertRaises(SystemExit) as error:
                parse_quality_args()
        self.assertEqual(error.exception.code, 2)

    def test_tra_uses_native_text_projection_and_is_detached(self):
        model = _TinyAttentionVoxTell()
        adapter = VoxTellPromptSFDA(
            model, torch.ones(1, 1, 4), torch.device("cpu"), _args()
        )
        try:
            prompt = torch.arange(4, dtype=torch.float32).view(1, 1, 4).requires_grad_()
            projected = _project_prompt_features(adapter, prompt, batch_size=3)
            native_input = adapter._text(prompt.detach(), 3).squeeze(2)
            native_input = native_input.permute(1, 0, 2).contiguous()
            expected = model.project_text_embed(native_input)
            self.assertEqual(tuple(projected.shape), (1, 3, 4))
            self.assertTrue(torch.allclose(projected, expected))
            self.assertFalse(projected.requires_grad)
        finally:
            adapter.close()

    def test_tra_correlation_handles_finite_and_constant_maps(self):
        similarity = torch.tensor([0.1, 0.2, float("nan"), 0.4])
        probability = torch.tensor([0.1, 0.3, 0.5, 0.8])
        self.assertAlmostEqual(compute_tra_score(similarity, probability), 1.0)
        self.assertIsNone(compute_tra_score(torch.ones(4), probability))
        self.assertIsNone(compute_tra_score(similarity, torch.ones(4)))
        with self.assertRaises(ValueError):
            compute_tra_score(torch.ones(3), torch.ones(4))

    def test_tra_cac_summary_excludes_invalid_scores_and_reports_gt_selection(self):
        rows = [
            {
                "case": "a", "view": 0, "cac": 0.1, "cac_valid": True,
                "tra_teacher": 0.2, "tra_teacher_valid": True,
                "tra_original": 0.4, "tra_original_valid": True,
                "dice": 0.1, "recall": 0.2, "precision": 0.3,
                "selected_cac": False, "selected_tra_teacher": False,
                "selected_tra_original": True,
            },
            {
                "case": "a", "view": 1, "cac": 0.3, "cac_valid": True,
                "tra_teacher": None, "tra_teacher_valid": False,
                "tra_original": 0.2, "tra_original_valid": True,
                "dice": 0.3, "recall": 0.6, "precision": 0.8,
                "selected_cac": True, "selected_tra_teacher": False,
                "selected_tra_original": False,
            },
            {
                "case": "a", "view": 2, "cac": 0.2, "cac_valid": True,
                "tra_teacher": 0.6, "tra_teacher_valid": True,
                "tra_original": 0.1, "tra_original_valid": True,
                "dice": 0.2, "recall": 0.4, "precision": 0.6,
                "selected_cac": False, "selected_tra_teacher": True,
                "selected_tra_original": False,
            },
            {
                "case": "b", "view": 0, "cac": 0.5, "cac_valid": True,
                "tra_teacher": None, "tra_teacher_valid": False,
                "tra_original": 0.7, "tra_original_valid": True,
                "dice": 0.8, "recall": 0.7, "precision": 0.9,
                "selected_cac": True, "selected_tra_teacher": False,
                "selected_tra_original": True,
            },
        ]
        summary = summarize_tra_cac(rows, checkpoint_loaded=True)
        self.assertEqual(summary["global_mixed_view_valid_views"]["cac"]["dice"], 4)
        self.assertEqual(summary["global_mixed_view_valid_views"]["tra_teacher"]["dice"], 2)
        self.assertEqual(summary["within_case_valid_cases"]["tra_teacher"]["dice"], 1)
        self.assertAlmostEqual(
            summary["selected_view_mean_gt_metrics"]["tra_teacher"]["dice"], 0.2
        )
        self.assertEqual(summary["selected_view_valid_cases"]["tra_teacher"], 1)
        self.assertIn("EMA", summary["protocol"]["tra_teacher_prompt"])
        with self.assertRaisesRegex(ValueError, "requires an adapted"):
            summarize_tra_cac(rows, checkpoint_loaded=False)

    def test_ground_truth_metrics_are_per_view_and_use_only_final_masks(self):
        target = torch.zeros(1, 2, 2, 2)
        target[0, 0, 0, 0] = 1
        probability = torch.zeros(2, 2, 2, 2)
        probability[0, 0, 0, 0] = 0.9
        probability[1, 1, 1, 1] = 0.9
        values = _ground_truth_metrics_per_view(probability, target)
        self.assertEqual(values[0], {"dice": 1.0, "recall": 1.0, "precision": 1.0})
        self.assertEqual(values[1]["dice"], 0.0)
        self.assertEqual(values[1]["recall"], 0.0)
        self.assertEqual(values[1]["precision"], 0.0)

    def test_offline_gt_audit_pairs_train_and_test_without_changing_loader(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            train_image = root / "images" / "P0" / "train.nii.gz"
            test_image = root / "images" / "P0" / "test.nii.gz"
            train_label = root / "labels" / "P0" / train_image.name
            test_label = root / "labels" / "P0" / test_image.name
            train_label.parent.mkdir(parents=True)
            train_label.write_bytes(b"offline-only")
            test_label.write_bytes(b"offline-only")
            with mock.patch(
                "evaluate_quality_metrics.read_image_entries",
                side_effect=[[(train_image, None)], [(test_image, test_label)]],
            ):
                entries = _labeled_audit_entries(root)
        self.assertEqual(
            [(entry[0].name, entry[2]) for entry in entries],
            [("test.nii.gz", "test"), ("train.nii.gz", "train")],
        )

    def test_quality_audit_entry_uses_current_sliding_window_signature(self):
        source = inspect.getsource(__import__("evaluate_quality_metrics").main)
        signature = inspect.signature(__import__("evaluate_quality_metrics")._voxtell_sliding_window_views)
        self.assertIn("_voxtell_sliding_window_views(", source)
        self.assertNotIn("infer_full_volume_views", source)
        self.assertNotIn("patch_size", list(signature.parameters))

    def test_gaussian_fusion_applies_sigmoid_after_logit_fusion(self):
        logits = torch.tensor([[[[0.0, 2.0]]]])
        denominator = torch.tensor([[[2.0, 2.0]]])
        fused_logits, probability = _fuse_logits_then_sigmoid(logits, denominator)
        self.assertTrue(torch.allclose(fused_logits, torch.tensor([[[[0.0, 1.0]]]])))
        self.assertTrue(torch.allclose(probability, torch.sigmoid(fused_logits)))
        self.assertFalse(torch.allclose(probability, torch.sigmoid(logits) / denominator))

    def test_patch_consistency_returns_one_score_per_view(self):
        probability = torch.tensor(
            [
                [[[0.9, 0.1], [0.8, 0.2]]],
                [[[0.1, 0.9], [0.2, 0.8]]],
            ],
            dtype=torch.float32,
        )
        consistency = _soft_consistency(probability).view(1, probability.shape[0])
        self.assertEqual(tuple(consistency.shape), (1, 2))

    def test_cac_launcher_explicitly_disables_cac_loss(self):
        launcher = (CM_SFDA / "train_cac.sh").read_text(encoding="utf-8")
        self.assertIn("--w_cac 0", launcher)
        self.assertIn("SCRIPT_DIR=", launcher)
        self.assertIn('"$SCRIPT_DIR/run_sfda_voxtell.py"', launcher)
        self.assertIn('"$SCRIPT_DIR/configs/tse.json"', launcher)
        tse_launcher = (CM_SFDA / "train_tse.sh").read_text(encoding="utf-8")
        self.assertIn('"$SCRIPT_DIR/configs/tse.json"', tse_launcher)
        legacy = (CM_SFDA / "train.sh").read_text(encoding="utf-8")
        self.assertIn("deprecated", legacy)
        self.assertIn("exit 2", legacy)

    def test_patch_oracle_gap_is_nonnegative_and_fixed_view_is_separate(self):
        selected, best, gap = _patch_oracle_stats([0.2, 0.8, 0.4], selected_index=0)
        self.assertAlmostEqual(selected, 0.2)
        self.assertAlmostEqual(best, 0.8)
        self.assertAlmostEqual(gap, 0.6)
        self.assertEqual(_patch_oracle_stats([0.2, 0.8], selected_index=1)[2], 0.0)
        source = inspect.getsource(__import__("evaluate_quality_metrics").summarize_quality_audit)
        self.assertIn("patch_oracle_gap_mean", source)
        self.assertIn("best_fixed_view_mean_dice", source)
        self.assertNotIn('"gap_to_oracle":', source)

    def test_quality_audit_summary_contains_patch_and_fixed_view_fields(self):
        quality_names = __import__("evaluate_quality_metrics").QUALITY_NAMES
        selection_names = __import__("evaluate_quality_metrics").SELECTION_NAMES
        case = {
            "within_case_spearman": {name: None for name in quality_names},
            "patch_spearman": {name: {"valid_patches": 0, "mean": None} for name in quality_names},
            "evidence_localization_original_view": {"valid": False},
        }
        args = SimpleNamespace(quality_mode="tse", w_quality=0.0, w_cac=0.0)
        result = summarize_quality_audit(
            [case], [], {name: [0.5] for name in selection_names},
            {name: [0.4] for name in selection_names}, [0.7],
            {name: [0.3] for name in selection_names}, [0.6],
            {name: [None] for name in quality_names}, [], args, {},
        )
        for field in (
            "selected_view_mean_dice", "patch_selected_dice_mean",
            "patch_oracle_best_dice_mean", "patch_oracle_gap_mean",
            "best_fixed_view_mean_dice", "valid_cases", "valid_patches",
        ):
            self.assertIn(field, result)
        self.assertEqual(result["valid_cases"], 1)
        self.assertEqual(result["valid_patches"], 1)

    def test_evaluation_interval_and_final_epoch(self):
        selected = [
            epoch
            for epoch in range(1, 13)
            if should_evaluate_epoch(epoch, total_epochs=12, interval=5)
        ]
        self.assertEqual(selected, [5, 10, 12])
        with self.assertRaises(ValueError):
            should_evaluate_epoch(1, total_epochs=5, interval=0)

    def test_fit_calls_epoch_callback_after_every_epoch(self):
        adapter, _, _ = _make_adapter()
        adapter.args.epochs = 2
        callbacks = []
        batch = (
            torch.randn(1, 1, 2, 2, 2),
            torch.randn(1, 1, 2, 2, 2),
        )
        try:
            history = adapter.fit(
                [batch],
                epoch_end_callback=lambda epoch, row, rows: callbacks.append(
                    (epoch, row["epoch"], len(rows))
                ),
            )
            self.assertEqual(callbacks, [(1, 1, 1), (2, 2, 2)])
            self.assertEqual(len(history), 2)
        finally:
            adapter.close()

    def test_binary_segmentation_metrics(self):
        metrics = binary_segmentation_metrics(
            np.array([1, 1, 0, 0]),
            np.array([1, 0, 1, 0]),
        )
        self.assertAlmostEqual(metrics["dice"], 0.5)
        self.assertAlmostEqual(metrics["iou"], 1 / 3)
        self.assertAlmostEqual(metrics["recall"], 0.5)
        self.assertAlmostEqual(metrics["precision"], 0.5)
        self.assertEqual(
            binary_segmentation_metrics(np.zeros(2), np.zeros(2)),
            {"dice": 1.0, "iou": 1.0, "recall": 1.0, "precision": 1.0},
        )

    def test_numpy_spacing_is_json_serializable(self):
        spacing = _native_spacing(
            torch.tensor([1.25, 2.5, 3.75], dtype=torch.float32).numpy()
        )
        encoded = json.dumps({"spacing": list(spacing)})
        self.assertEqual(json.loads(encoded)["spacing"], [1.25, 2.5, 3.75])

    def test_low_precision_initial_prompt_uses_fp32_trainable_state(self):
        model = _TinyVoxTell()
        adapter = VoxTellPromptSFDA(
            model,
            torch.ones(1, 1, 4, dtype=torch.float16),
            torch.device("cpu"),
            _args(),
        )
        try:
            self.assertEqual(adapter.soft_prompt_embedding.dtype, torch.float32)
            self.assertEqual(adapter.teacher_soft_prompt.dtype, torch.float32)
        finally:
            adapter.close()

    def test_only_soft_prompt_is_trainable_and_gradient_is_nonempty(self):
        adapter, model, qwen = _make_adapter()
        try:
            self.assertEqual(sum(p.numel() for p in adapter.optimizer_parameters), 4)
            self.assertEqual(len(adapter.optimizer_parameters), 1)
            self.assertIs(adapter.optimizer_parameters[0], adapter.soft_prompt_embedding)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertTrue(all(not p.requires_grad for p in qwen.parameters()))

            result = adapter.adapt_batch(
                torch.randn(1, 1, 2, 2, 2), torch.randn(1, 1, 2, 2, 2)
            )
            self.assertGreater(result["soft_prompt_grad_norm"], 0.0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(all(p.grad is None for p in qwen.parameters()))
            self.assertEqual(
                model.forward_calls,
                [(1, False), (3, False), (1, True)],
            )
            self.assertEqual(result["update_skipped"], 0.0)
        finally:
            adapter.close()

    def test_amp_overflow_skips_update_without_corrupting_prompts(self):
        adapter, _, _ = _make_adapter()
        try:
            soft_prompt_before = adapter.soft_prompt_embedding.detach().clone()
            teacher_prompt_before = adapter.teacher_soft_prompt.detach().clone()
            adapter.scaler = _OverflowScaler()
            result = adapter.adapt_batch(
                torch.randn(1, 1, 2, 2, 2), torch.randn(1, 1, 2, 2, 2)
            )
            self.assertEqual(result["update_skipped"], 1.0)
            self.assertEqual(result["soft_prompt_grad_norm"], 0.0)
            self.assertTrue(torch.equal(soft_prompt_before, adapter.soft_prompt_embedding))
            self.assertTrue(torch.equal(teacher_prompt_before, adapter.teacher_soft_prompt))
        finally:
            adapter.close()

    def test_cac_view_selection_and_losses_have_expected_shapes(self):
        vision = torch.randn(2, 2, 2, 2, 4)
        text = torch.randn(1, 2, 4)
        logits = torch.randn(2, 1, 2, 2, 2, requires_grad=True)
        scores = compute_cac_score(vision, text, logits)
        self.assertEqual(scores.shape, (2,))
        view_scores = scores[:, None].expand(2, 4)
        probabilities = torch.sigmoid(torch.randn(2, 4, 1, 2, 2, 2))
        selected = select_cac_views(view_scores, probabilities, 0.5)
        self.assertEqual(selected.shape, (2, 2))
        valid = torch.ones_like(logits)
        segmentation, bce, dice = masked_segmentation_loss(logits, (logits > 0).float(), valid)
        self.assertEqual(segmentation.ndim, 0)
        self.assertEqual(bce.ndim, 0)
        self.assertEqual(dice.ndim, 0)
        self.assertEqual(entropy_loss(logits, valid).ndim, 0)
        self.assertEqual(cac_loss(scores).ndim, 0)

    def test_false_positive_without_evidence_lowers_purity(self):
        evidence = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        supported_prediction = torch.tensor([[1.0, 1.0, 0.0, 0.0]])
        false_positive_prediction = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        supported_purity, _, _ = compute_tse_components(
            supported_prediction, evidence
        )
        false_positive_purity, _, _ = compute_tse_components(
            false_positive_prediction, evidence
        )
        self.assertLess(false_positive_purity.item(), supported_purity.item())

    def test_removing_strong_evidence_region_lowers_completeness(self):
        evidence = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        complete_prediction = torch.tensor([[1.0, 1.0, 1.0, 0.0]])
        incomplete_prediction = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        _, complete_coverage, _ = compute_tse_components(
            complete_prediction, evidence
        )
        _, incomplete_coverage, _ = compute_tse_components(
            incomplete_prediction, evidence
        )
        self.assertLess(incomplete_coverage.item(), complete_coverage.item())

    def test_leave_one_case_out_excludes_current_case(self):
        memory = SemanticPrototypeMemory()
        statistics = {
            "fg_sum": torch.tensor([[10.0, 0.0], [0.0, 4.0]]),
            "fg_count": torch.tensor([10.0, 4.0]),
            "bg_sum": torch.tensor([[0.0, 6.0], [8.0, 0.0]]),
            "bg_count": torch.tensor([6.0, 8.0]),
        }
        memory.add(["case-a", "case-b"], statistics)
        positive, negative, valid = memory.leave_one_out(["case-a"], "cpu")
        self.assertEqual(memory.contributors(excluding="case-a"), ["case-b"])
        self.assertTrue(torch.allclose(positive[0, 0], torch.tensor([0.0, 1.0])))
        self.assertTrue(torch.allclose(negative[0, 0], torch.tensor([1.0, 0.0])))
        self.assertTrue(valid.all())

    def test_empty_prediction_or_evidence_is_finite(self):
        zeros = torch.zeros(2, 8)
        ones = torch.ones(2, 8)
        for probability, evidence in ((zeros, ones), (ones, zeros), (zeros, zeros)):
            values = compute_tse_components(probability, evidence)
            self.assertTrue(all(torch.isfinite(value).all() for value in values))
            self.assertTrue(all(torch.equal(value, torch.zeros_like(value)) for value in values))

    def test_saaf_is_high_when_attention_evidence_is_inside_mask(self):
        mask = torch.tensor([[0.95, 0.9, 0.05, 0.05]])
        evidence = torch.tensor([[3.0, 2.0, 0.0, 0.0]])
        result = compute_saaf_quality(mask, evidence)
        self.assertTrue(bool(result["valid"].item()))
        self.assertGreater(float(result["saaf"].item()), 0.8)

    def test_saaf_coverage_drops_when_evidence_is_outside_mask(self):
        mask = torch.tensor([[0.95, 0.05, 0.05, 0.05]])
        inside = compute_saaf_quality(mask, torch.tensor([[3.0, 0.0, 0.0, 0.0]]))
        outside = compute_saaf_quality(mask, torch.tensor([[0.0, 3.0, 3.0, 0.0]]))
        self.assertLess(float(outside["coverage"].item()), float(inside["coverage"].item()))
        self.assertLess(float(outside["saaf"].item()), float(inside["saaf"].item()))

    def test_saaf_purity_drops_for_mask_without_evidence(self):
        evidence = torch.tensor([[3.0, 0.0, 0.0, 0.0]])
        tight = compute_saaf_quality(torch.tensor([[0.95, 0.05, 0.05, 0.05]]), evidence)
        broad = compute_saaf_quality(torch.tensor([[0.95, 0.95, 0.95, 0.05]]), evidence)
        self.assertLess(float(broad["purity"].item()), float(tight["purity"].item()))
        self.assertLess(float(broad["saaf"].item()), float(tight["saaf"].item()))

    def test_uniform_attention_is_invalid(self):
        attention = torch.full((1, 2, 1, 8), 1.0 / 8.0)
        _, valid, mad, reasons = attention_evidence_map(
            attention, (2, 2, 2), mad_threshold=1e-5
        )
        self.assertFalse(bool(valid.item()))
        self.assertEqual(reasons[0], "attention_mad_below_threshold")
        self.assertTrue(torch.isfinite(mad).all())

    def test_final_decoder_attention_extracts_native_last_layer(self):
        first = torch.full((2, 1, 32), 1.0 / 32)
        last = torch.zeros_like(first)
        last[..., 7] = 1.0
        result = final_decoder_attention([first, last])
        self.assertEqual(tuple(result.shape), (2, 1, 1, 32))
        self.assertTrue(torch.equal(result[:, 0], last))

    def test_saaf_quality_aligns_finer_logits_to_coarse_evidence(self):
        args = _args(
            quality_metric="saaf", quality_mode="cac", selection_p=0.1,
            num_aug_views=3, batch_size=1,
        )
        adapter = VoxTellPromptSFDA(
            _TinyAttentionVoxTell(), torch.ones(1, 1, 4), torch.device("cpu"), args,
        )
        try:
            image = torch.randn(1, 1, 2, 4, 4)
            adapter._cac_features.clear()
            with torch.no_grad():
                logits, diagnostics = adapter.model(
                    image,
                    adapter._text(adapter.initial_soft_prompt, 1),
                    return_diagnostics=True,
                )
            result = adapter._saaf_quality(
                adapter._cac_features["vision"], logits,
                diagnostics["cross_attention"],
            )
            self.assertEqual(tuple(result["evidence"].shape), (1, 4, 4, 2))
            self.assertTrue(torch.isfinite(result["saaf"]).all())
            last_evidence, _, _, _ = attention_evidence_map(
                final_decoder_attention(diagnostics["cross_attention"]),
                (4, 4, 2),
                mad_threshold=adapter.quality_config["attention_mad_threshold"],
            )
            first_evidence, _, _, _ = attention_evidence_map(
                diagnostics["cross_attention"][0].unsqueeze(1),
                (4, 4, 2),
                mad_threshold=adapter.quality_config["attention_mad_threshold"],
            )
            self.assertTrue(torch.allclose(result["evidence"], last_evidence))
            self.assertFalse(torch.allclose(last_evidence, first_evidence))
        finally:
            adapter.close()

    def test_saaf_selection_uses_updated_student_prompt(self):
        args = _args(
            quality_metric="saaf", quality_mode="cac", selection_p=0.1,
            num_aug_views=3, batch_size=1,
        )
        model = _TinyAttentionVoxTell()
        adapter = VoxTellPromptSFDA(
            model, torch.ones(1, 1, 4), torch.device("cpu"), args
        )
        try:
            with torch.no_grad():
                adapter.soft_prompt_embedding.fill_(2.25)
            views = torch.randn(1, 3, 1, 2, 4, 4)
            selected = adapter._select_views(views, ["case-current-prompt"])
            self.assertEqual(tuple(selected.shape), (1, 1))
            evaluated_prompt = model.prompt_calls[-1]
            self.assertTrue(
                torch.equal(
                    evaluated_prompt,
                    adapter.soft_prompt_embedding.detach().expand(3, -1, -1).unsqueeze(2),
                )
            )
            self.assertNotEqual(
                float(evaluated_prompt[0, 0, 0, 0]),
                float(adapter.initial_soft_prompt[0, 0, 0]),
            )
        finally:
            adapter.close()

    def test_tdc_respects_d5_first_order_and_aligns_decoder_sizes(self):
        d5 = torch.full((1, 1, 4, 4, 4), 10.0)
        d4 = torch.full((1, 1, 2, 2, 2), 10.0)
        d3 = torch.full((1, 1, 1, 1, 1), 10.0)
        d2 = torch.full((1, 1, 1, 1, 1), 10.0)
        d1 = torch.full((1, 1, 1, 1, 1), -10.0)
        result = compute_tdc_consensus([d5, d4, d3, d2, d1])

        self.assertEqual(result["spatial_shape"], (4, 4, 4))
        self.assertEqual(result["decoder_output_count"], 5)
        self.assertEqual(result["tdc_decoder_count"], 4)
        self.assertEqual(TDC_PAIR_NAMES, (
            "dice_d5_d4", "dice_d5_d3", "dice_d5_d2",
            "dice_d4_d3", "dice_d4_d2", "dice_d3_d2",
        ))
        self.assertEqual(tuple(result["pair_dice"].shape), (1, 6))
        self.assertEqual(int(result["valid_pair_count"].item()), 6)
        self.assertTrue(torch.allclose(result["pair_dice"], torch.ones(1, 6)))
        self.assertAlmostEqual(float(result["tdc"].item()), 1.0)

        # A differently ordered list would use the 1x1x1 stage as the target;
        # this guards the VoxTell forward order [D5,D4,D3,D2].
        reversed_result = compute_tdc_consensus([d2, d3, d4, d5, d1])
        self.assertEqual(reversed_result["spatial_shape"], (1, 1, 1))

    def test_tdc_empty_mask_pair_rules_and_finite_outputs(self):
        empty = torch.full((1, 1, 2, 2, 2), -100.0)
        all_empty = compute_tdc_consensus([empty, empty.clone(), empty.clone(), empty.clone()])
        self.assertFalse(bool(all_empty["valid"].item()))
        self.assertEqual(all_empty["invalid_reason"], ["all_decoder_masks_empty"])
        self.assertEqual(int(all_empty["valid_pair_count"].item()), 0)
        self.assertTrue(torch.isfinite(all_empty["tdc"]).all())
        self.assertTrue(torch.isfinite(all_empty["pair_dice"]).all())

        nonempty = torch.full_like(empty, 100.0)
        one_nonempty = compute_tdc_consensus(
            [nonempty, empty, empty.clone(), empty.clone()]
        )
        self.assertTrue(bool(one_nonempty["valid"].item()))
        # Three foreground/background pairs score zero; the three empty/empty
        # pairs are omitted rather than counted as perfect Dice.
        self.assertEqual(int(one_nonempty["valid_pair_count"].item()), 3)
        self.assertEqual(float(one_nonempty["tdc"].item()), 0.0)
        self.assertEqual(
            one_nonempty["pair_valid"].tolist(),
            [[True, True, True, False, False, False]],
        )

    def test_tdc_valid_mask_excludes_padding_on_decoder_grid(self):
        # Decoder order is (H,W,D), while the input mask is (D,H,W). Only one
        # D plane is valid, so every all-positive pair must count 2*3*1 voxels.
        decoder = [torch.full((1, 1, 2, 3, 4), 10.0) for _ in range(4)]
        valid = torch.zeros(1, 2, 3, 4, dtype=torch.bool)
        valid[:, 0, :, :] = True
        result = compute_tdc_consensus(decoder, valid_mask=valid)
        self.assertTrue(torch.all(result["count1"] == 12))
        self.assertTrue(torch.all(result["count2"] == 12))
        self.assertTrue(torch.all(result["intersection"] == 12))
        self.assertAlmostEqual(float(result["tdc"].item()), 1.0)

    def test_tdc_case_reduction_is_not_patch_average(self):
        # One patch has pair Dice 1 and the other pair Dice 0. Global reduction
        # must use global intersection/count, not average the two patch scores.
        first_intersection = torch.full((1, 6), 2.0)
        first_count = torch.full((1, 6), 2.0)
        second_intersection = torch.zeros(1, 6)
        second_count = torch.full((1, 6), 8.0)
        global_tdc, *_ = tdc_from_components(
            first_intersection + second_intersection,
            first_count + second_count,
            first_count + second_count,
            torch.ones(1, dtype=torch.bool),
        )
        self.assertAlmostEqual(float(global_tdc.item()), 0.2)
        self.assertNotAlmostEqual(float(global_tdc.item()), 0.5)

    def test_tdc_quality_only_selection_and_tdc_entropy_selection(self):
        scores = torch.tensor([[0.9, 0.8, 0.1]])
        probabilities = torch.stack(
            [torch.full((1, 2, 2, 2), 0.5),
             torch.full((1, 2, 2, 2), 0.0),
             torch.full((1, 2, 2, 2), 0.9)], dim=1
        )
        valid = torch.ones_like(scores, dtype=torch.bool)
        quality_only, _ = select_tdc_views(
            scores, probabilities, 1.0 / 3.0, valid, use_entropy_rank=False
        )
        fused, _ = select_tdc_views(
            scores, probabilities, 1.0 / 3.0, valid, use_entropy_rank=True
        )
        self.assertEqual(quality_only.tolist(), [[0]])
        self.assertEqual(fused.tolist(), [[1]])

    def test_lspm_memory_and_case_quality_follow_tdc_metric(self):
        memory = ShortPromptQualityMemory(2)
        memory.append(torch.ones(1, 1, 4), 0.1)
        memory.append(torch.full((1, 1, 4), 3.0), 0.9)
        self.assertEqual(len(memory), 2)
        adapter, _, _ = _make_adapter()
        try:
            adapter.quality_metric = "tdc"
            adapter._case_tdc_quality = lambda prompt, patches, masks: float(prompt.mean())
            adapter._case_cac_quality = lambda prompt, patches, masks: 99.0
            self.assertEqual(adapter._case_quality(torch.ones(1, 1, 4), [], []), 1.0)
            adapter.quality_metric = "cac"
            self.assertEqual(adapter._case_quality(torch.ones(1, 1, 4), [], []), 99.0)
        finally:
            adapter.close()

    def test_case_adapter_uses_ten_views_and_one_step_without_reset(self):
        args = _args(
            num_aug_views=9,
            selection_p=0.1,
            use_entropy_rank=False,
            short_memory_length=2,
        )
        adapter = VoxTellPromptSFDA(
            _TinyVoxTell(), torch.ones(1, 1, 4), torch.device("cpu"), args
        )
        try:
            patches = [torch.randn(1, 2, 2, 2), torch.randn(1, 2, 2, 2)]
            masks = [torch.ones(2, 2, 2), torch.ones(2, 2, 2)]
            result = adapter.adapt_case_patches(patches, masks, "case-a", epoch=1)
            self.assertEqual(len(result["view_selection"]["view_params"]), 10)
            self.assertEqual(result["optimizer_steps_for_case"], 1)
            self.assertEqual(result["pseudo_label_refreshes"], 1)
            self.assertEqual(adapter.optimizer_step_count, 1)
            self.assertEqual(len(adapter.short_memory), 1)
            expected_loss = (
                args.w_seg * result["segmentation"]
                + args.w_entropy * result["entropy"]
                + args.w_cac * result["quality_loss"]
            )
            self.assertAlmostEqual(result["loss"], expected_loss, places=5)
            adapter.args.epochs = 0
            adapter.fit([])
            self.assertEqual(len(adapter.short_memory), 1)
        finally:
            adapter.close()

    def test_case_reuses_pseudo_label_for_k_steps_and_updates_soft_prompt_each_time(self):
        import sfda_voxtell

        args = _args(
            num_aug_views=1,
            selection_p=0.5,
            pseudo_label_refresh_steps=3,
            w_cac=1.0,
        )
        model = _TinyVoxTell()
        adapter = VoxTellPromptSFDA(
            model, torch.ones(1, 1, 4), torch.device("cpu"), args
        )
        snapshots = []
        original_step = adapter.optimizer.step

        def capture_step(*step_args, **step_kwargs):
            result = original_step(*step_args, **step_kwargs)
            snapshots.append(adapter.soft_prompt_embedding.detach().clone())
            return result

        adapter.optimizer.step = capture_step
        try:
            with mock.patch.object(
                sfda_voxtell,
                "masked_segmentation_loss",
                wraps=masked_segmentation_loss,
            ) as segmentation_spy:
                result = adapter.adapt_case_patches(
                    [torch.randn(1, 2, 2, 2)],
                    [torch.ones(2, 2, 2)],
                    "refresh-case",
                )
            self.assertEqual(result["pseudo_label_refreshes"], 1)
            self.assertEqual(result["optimizer_steps_for_case"], 3)
            self.assertEqual(adapter.optimizer_step_count, 3)
            self.assertEqual(len(segmentation_spy.call_args_list), 3)
            pseudo_labels = [
                call.args[1].detach().clone()
                for call in segmentation_spy.call_args_list
            ]
            valid_masks = [
                call.args[2].detach().clone()
                for call in segmentation_spy.call_args_list
            ]
            self.assertTrue(all(torch.equal(pseudo_labels[0], value) for value in pseudo_labels[1:]))
            self.assertTrue(all(torch.equal(valid_masks[0], value) for value in valid_masks[1:]))
            self.assertEqual(len(snapshots), 3)
            self.assertFalse(torch.equal(snapshots[0], snapshots[1]))
            self.assertFalse(torch.equal(snapshots[1], snapshots[2]))
            student_prompts = model.prompt_calls[-3:]
            self.assertEqual(len(student_prompts), 3)
            self.assertFalse(torch.equal(student_prompts[0], student_prompts[1]))
            self.assertFalse(torch.equal(student_prompts[1], student_prompts[2]))
        finally:
            adapter.close()

    def test_case_tdc_view_chunking_matches_full_view_forward(self):
        params = [
            {"scale": 1.0, "offset": 0.0},
            *[
                {"scale": 1.0 + 0.01 * index, "offset": 0.01 * index}
                for index in range(1, 10)
            ],
        ]
        patches = [torch.randn(1, 2, 2, 2), torch.randn(1, 2, 2, 2)]
        valid_masks = [torch.ones(2, 2, 2), torch.ones(2, 2, 2)]
        torch.manual_seed(321)
        reference = _TinyTDCVoxTell()
        reference_state = reference.state_dict()
        baseline_tdc = None
        baseline_selected = None
        for view_batch_size in (1, 3, 10):
            model = _TinyTDCVoxTell()
            model.load_state_dict(reference_state)
            adapter = VoxTellPromptSFDA(
                model,
                torch.ones(1, 1, 4),
                torch.device("cpu"),
                _args(
                    quality_metric="tdc",
                    num_aug_views=9,
                    selection_p=0.1,
                    use_entropy_rank=False,
                    view_batch_size=view_batch_size,
                ),
            )
            try:
                selected, details = adapter._select_case_views(
                    patches,
                    valid_masks,
                    adapter.soft_prompt_embedding.detach(),
                    "chunked-case",
                    params,
                )
                tdc = torch.tensor(details["tdc"])
                if baseline_tdc is None:
                    baseline_tdc = tdc
                    baseline_selected = selected
                else:
                    self.assertTrue(torch.equal(selected, baseline_selected))
                    self.assertTrue(torch.equal(tdc, baseline_tdc))
                self.assertLessEqual(max(batch_size for batch_size, _ in model.forward_calls), view_batch_size)
            finally:
                adapter.close()

    def test_case_teacher_uses_selected_view_and_prompt_bridge_uses_current_weight(self):
        args = _args(
            num_aug_views=1,
            selection_p=0.5,
            w_cac=0.0,
            w_entropy=0.0,
            confidence_threshold=0.5,
        )
        model = _TinyVoxTell()
        adapter = VoxTellPromptSFDA(
            model, torch.ones(1, 1, 4), torch.device("cpu"), args
        )
        try:
            short_prompt = torch.full((1, 1, 4), 2.0)
            with torch.no_grad():
                adapter.soft_prompt_embedding.fill_(1.0)
            adapter._prepare_lspm = lambda _patches, _masks: {
                "short_prompt": short_prompt,
                "long_prompt": torch.full((1, 1, 4), 3.0),
                "current_quality": 0.0,
                "historical_quality": 0.0,
                "historical_weight": 0.25,
            }
            adapter._sample_case_view_params = lambda _count: [
                {"scale": 1.0, "offset": 0.0},
                {"scale": 1.0, "offset": 5.0},
            ]
            adapter._select_case_views = lambda *_args: (
                torch.tensor([[1]], dtype=torch.long),
                {
                    "selected_views": [1],
                    "selection_metric": "cac",
                    "use_entropy_rank": False,
                    "cac": [0.0, 1.0],
                    "tdc": None,
                    "entropy": [0.0, 0.0],
                    "quality_rank": [1.0, 0.0],
                    "combined_rank": [1.0, 0.0],
                },
            )
            patch = torch.zeros(1, 2, 2, 2)
            mask = torch.ones(2, 2, 2)
            adapter.adapt_case_patches([patch], [mask], "selected-teacher")

            # Selection forwards happen first. The final two forwards are the
            # long-prompt teacher and the gradient-carrying student on view 1.
            self.assertTrue(torch.equal(model.image_calls[-2], model.image_calls[-1]))
            self.assertTrue(torch.allclose(model.image_calls[-1], torch.full_like(model.image_calls[-1], 5.0)))
            self.assertTrue(torch.allclose(model.prompt_calls[-2], torch.full_like(model.prompt_calls[-2], 3.0)))
            # The value is the short prompt, while its gradient bridge is
            # exactly (1 - historical_weight) = 0.75.
            expected_student = torch.full_like(model.prompt_calls[-1], 2.0)
            self.assertTrue(torch.allclose(model.prompt_calls[-1], expected_student))
        finally:
            adapter.close()

    def test_case_lspm_bridge_scales_soft_prompt_gradient(self):
        args = _args(
            num_aug_views=1,
            selection_p=0.5,
            w_cac=0.0,
            w_entropy=0.0,
            confidence_threshold=0.5,
        )
        torch.manual_seed(123)
        reference = _TinyVoxTell()
        reference_state = reference.state_dict()

        def run(historical_weight):
            model = _TinyVoxTell()
            model.load_state_dict(reference_state)
            adapter = VoxTellPromptSFDA(
                model, torch.ones(1, 1, 4), torch.device("cpu"), args
            )
            captured = []
            handle = adapter.soft_prompt_embedding.register_hook(
                lambda gradient: captured.append(gradient.detach().clone())
            )
            adapter._prepare_lspm = lambda _patches, _masks: {
                "short_prompt": torch.full((1, 1, 4), 2.0),
                "long_prompt": torch.full((1, 1, 4), 3.0),
                "current_quality": 0.0,
                "historical_quality": 0.0,
                "historical_weight": historical_weight,
            }
            adapter._sample_case_view_params = lambda _count: [
                {"scale": 1.0, "offset": 0.0},
                {"scale": 1.0, "offset": 5.0},
            ]
            adapter._select_case_views = lambda *_args: (
                torch.tensor([[1]], dtype=torch.long),
                {
                    "selected_views": [1],
                    "selection_metric": "cac",
                    "use_entropy_rank": False,
                    "cac": [0.0, 1.0],
                    "tdc": None,
                    "entropy": [0.0, 0.0],
                    "quality_rank": [1.0, 0.0],
                    "combined_rank": [1.0, 0.0],
                },
            )
            try:
                adapter.adapt_case_patches(
                    [torch.zeros(1, 2, 2, 2)],
                    [torch.ones(2, 2, 2)],
                    "bridge-gradient",
                )
                self.assertEqual(len(captured), 1)
                return captured[0]
            finally:
                handle.remove()
                adapter.close()

        full_bridge = run(0.0)
        weighted_bridge = run(0.25)
        self.assertTrue(torch.allclose(weighted_bridge, full_bridge * 0.75, atol=1e-6))

    def test_case_tdc_invalid_views_are_excluded_and_all_invalid_falls_back(self):
        args = _args(
            quality_metric="tdc",
            num_aug_views=2,
            selection_p=1.0 / 3.0,
            use_entropy_rank=False,
        )
        adapter = VoxTellPromptSFDA(
            _TinyTDCVoxTell(), torch.ones(1, 1, 4), torch.device("cpu"), args
        )
        try:
            patch = torch.zeros(1, 2, 2, 2)
            masks = [torch.zeros(2, 2, 2)]
            params = [
                {"scale": 1.0, "offset": 0.0},
                {"scale": 1.0, "offset": 1.0},
                {"scale": 1.0, "offset": 2.0},
            ]
            adapter._case_tdc_quality = lambda *_args: 0.0
            selected, details = adapter._select_case_views(
                [patch], masks, adapter.soft_prompt_embedding.detach(), "invalid", params
            )
            self.assertEqual(selected.tolist(), [[0]])
            self.assertEqual(details["tdc_valid"], [False, False, False])
            self.assertEqual(details["combined_rank"], [float("inf")] * 3)
        finally:
            adapter.close()

    def test_case_tdc_mode_uses_differentiable_cac_for_prompt_update(self):
        args = _args(
            quality_metric="tdc",
            num_aug_views=1,
            selection_p=0.5,
            use_entropy_rank=False,
            w_seg=0.0,
            w_entropy=0.0,
            w_cac=0.7,
        )
        model = _TinyTDCVoxTell()
        adapter = VoxTellPromptSFDA(
            model, torch.ones(1, 1, 4), torch.device("cpu"), args
        )
        captured = []
        hook = adapter.soft_prompt_embedding.register_hook(
            lambda gradient: captured.append(gradient.detach().clone())
        )
        try:
            result = adapter.adapt_case_patches(
                [torch.randn(1, 2, 2, 2)],
                [torch.ones(2, 2, 2)],
                "tdc-cac-gradient",
            )
            self.assertTrue(np.isfinite(result["cac"]))
            self.assertAlmostEqual(result["cac_loss"], -result["cac"], places=5)
            selected = result["selected_view"]
            self.assertAlmostEqual(
                result["quality"], result["view_selection"]["tdc"][selected], places=5
            )
            self.assertEqual(len(captured), 1)
            self.assertGreater(float(captured[0].norm()), 0.0)
        finally:
            hook.remove()
            adapter.close()

    def test_case_tdc_ranking_drops_invalid_views_from_top_k(self):
        import sfda_voxtell

        args = _args(
            quality_metric="tdc",
            num_aug_views=2,
            selection_p=1.0,
            use_entropy_rank=False,
            view_batch_size=10,
        )
        adapter = VoxTellPromptSFDA(
            _TinyTDCVoxTell(), torch.ones(1, 1, 4), torch.device("cpu"), args
        )
        try:
            def fake_tdc(_decoder_outputs, valid_mask=None):
                del valid_mask
                intersection = torch.zeros(3, 6)
                count1 = torch.zeros(3, 6)
                count2 = torch.zeros(3, 6)
                intersection[0].fill_(2.0)
                count1[0].fill_(2.0)
                count2[0].fill_(2.0)
                intersection[2].fill_(1.0)
                count1[2].fill_(5.0)
                count2[2].fill_(5.0)
                return {
                    "intersection": intersection,
                    "count1": count1,
                    "count2": count2,
                    "finite": torch.ones(3, dtype=torch.bool),
                }

            with mock.patch.object(
                sfda_voxtell, "compute_tdc_consensus", side_effect=fake_tdc
            ):
                selected, details = adapter._select_case_views(
                    [torch.zeros(1, 2, 2, 2)],
                    [torch.ones(2, 2, 2)],
                    adapter.soft_prompt_embedding.detach(),
                    "partial-invalid",
                    [
                        {"scale": 1.0, "offset": 0.0},
                        {"scale": 1.0, "offset": 1.0},
                        {"scale": 1.0, "offset": 2.0},
                    ],
                )
            self.assertEqual(details["tdc_valid"], [True, False, True])
            self.assertEqual(selected.tolist(), [[0, 2]])
        finally:
            adapter.close()

    def test_tdc_selection_uses_quality_plus_entropy_rank_fusion(self):
        scores = torch.tensor([[0.9, 0.8, 0.1]])
        probabilities = torch.stack(
            [
                torch.full((1, 2, 2, 2), 0.5),
                torch.full((1, 2, 2, 2), 0.0),
                torch.full((1, 2, 2, 2), 0.9),
            ],
            dim=1,
        )
        selected, fallback = select_tdc_views(
            scores, probabilities, 1.0 / 3.0, torch.ones_like(scores, dtype=torch.bool)
        )
        # View 0 wins on TDC but loses entropy rank; rank sums select view 1.
        self.assertEqual(selected.tolist(), [[1]])
        self.assertFalse(bool(fallback.item()))

        tied_selected, _ = select_tdc_views(
            torch.ones(1, 3),
            torch.full((1, 3, 1, 2, 2, 2), 0.25),
            1.0 / 3.0,
            torch.ones(1, 3, dtype=torch.bool),
        )
        self.assertEqual(tied_selected.tolist(), [[0]])

    def test_tdc_selects_with_updated_prompt_and_logs_native_decoder_outputs(self):
        args = _args(
            quality_metric="tdc", quality_mode="cac", selection_p=0.34,
            num_aug_views=3, batch_size=1,
        )
        model = _TinyTDCVoxTell()
        adapter = VoxTellPromptSFDA(
            model, torch.ones(1, 1, 4), torch.device("cpu"), args
        )
        try:
            with torch.no_grad():
                adapter.soft_prompt_embedding.fill_(2.25)
            views = torch.randn(1, 3, 1, 4, 4, 4)
            selected = adapter._select_views(views, ["case-current-tdc-prompt"])

            self.assertEqual(tuple(selected.shape), (1, 1))
            self.assertEqual(model.forward_calls, [(3, False)])
            self.assertEqual(len(model.decoder_return_shapes), 1)
            shapes = model.decoder_return_shapes[0]
            self.assertEqual([shape[2:] for shape in shapes], [
                (4, 4, 4), (2, 2, 2), (1, 1, 1), (1, 1, 1), (1, 1, 1)
            ])
            evaluated_prompt = model.prompt_calls[-1]
            self.assertTrue(torch.equal(
                evaluated_prompt,
                adapter.soft_prompt_embedding.detach().expand(3, -1, -1).unsqueeze(2),
            ))
            self.assertEqual(len(adapter.quality_diagnostics), 3)
            row = adapter.quality_diagnostics[0]
            self.assertIn("tdc", row)
            self.assertTrue(all(name in row for name in TDC_PAIR_NAMES))
            self.assertIn("valid", row)
            self.assertIn("selected", row)
            self.assertEqual(row["decoder_output_count"], 5)
            self.assertEqual(row["tdc_decoder_count"], 4)
        finally:
            adapter.close()

    def test_all_invalid_tdc_falls_back_to_view_zero(self):
        args = _args(
            quality_metric="tdc", quality_mode="cac", selection_p=0.34,
            num_aug_views=3, batch_size=1,
        )
        model = _TinyTDCVoxTell()
        model.all_negative_decoders = True
        adapter = VoxTellPromptSFDA(
            model, torch.ones(1, 1, 4), torch.device("cpu"), args
        )
        try:
            views = torch.randn(1, 3, 1, 4, 4, 4)
            selected = adapter._select_views(views, ["case-empty-tdc"])
            self.assertEqual(selected.tolist(), [[0]])
            self.assertTrue(all(not row["valid"] for row in adapter.quality_diagnostics))
            self.assertTrue(all(row["selection_fallback"] for row in adapter.quality_diagnostics))
            self.assertTrue(adapter.quality_diagnostics[0]["selected"])
            self.assertEqual(
                {row["invalid_reason"] for row in adapter.quality_diagnostics},
                {"all_decoder_masks_empty"},
            )
            result = adapter.adapt_batch(
                torch.randn(1, 1, 4, 4, 4),
                torch.randn(1, 1, 4, 4, 4),
                ["case-empty-tdc"],
            )
            self.assertEqual(result["update_skipped"], 0.0)
        finally:
            adapter.close()

    def test_tdc_fit_writes_per_view_diagnostics(self):
        args = _args(
            quality_metric="tdc", quality_mode="cac", selection_p=0.34,
            num_aug_views=3, batch_size=1,
        )
        with tempfile.TemporaryDirectory() as output_dir:
            args.output_dir = output_dir
            adapter = VoxTellPromptSFDA(
                _TinyTDCVoxTell(), torch.ones(1, 1, 4), torch.device("cpu"), args
            )
            try:
                weak = torch.zeros(1, 1, 4, 4, 4)
                adapter.fit([(weak, weak.clone(), ["case-tdc-diagnostics"])])
                json_path = Path(output_dir) / "tdc_diagnostics.json"
                csv_path = Path(output_dir) / "tdc_diagnostics.csv"
                self.assertTrue(json_path.is_file())
                self.assertTrue(csv_path.is_file())
                rows = json.loads(json_path.read_text(encoding="utf-8"))
                self.assertEqual(len(rows), 3)
                self.assertTrue(all(all(name in row for name in TDC_PAIR_NAMES) for row in rows))
                self.assertIn("selected", rows[0])
                self.assertIn("valid", rows[0])
            finally:
                adapter.close()

    def test_tdc_does_not_change_teacher_thresholds_or_add_tdc_loss(self):
        import sfda_voxtell

        model = _TinyTDCVoxTell()
        args = _args(
            confidence_threshold=0.7, quality_metric="tdc", w_cac=1.0,
            selection_p=0.34, num_aug_views=3,
        )
        adapter = VoxTellPromptSFDA(
            model, torch.ones(1, 1, 4), torch.device("cpu"), args
        )
        try:
            with torch.no_grad():
                adapter.teacher_soft_prompt.fill_(0.65)
                adapter.soft_prompt_embedding.fill_(-0.35)
            teacher_before = adapter.teacher_soft_prompt.detach().clone()
            with mock.patch.object(
                sfda_voxtell,
                "masked_segmentation_loss",
                wraps=masked_segmentation_loss,
            ) as segmentation_spy:
                result = adapter.adapt_batch(
                    torch.zeros(1, 1, 4, 4, 4),
                    torch.ones(1, 1, 4, 4, 4),
                    ["case-tdc-loss-path"],
                )
            self.assertTrue(torch.equal(
                model.prompt_calls[0],
                teacher_before.expand(1, -1, -1).unsqueeze(2),
            ))
            teacher_probability = torch.sigmoid(model.outputs[0].float())
            expected_pseudo = (teacher_probability >= 0.5).float()[:, None]
            expected_valid = (
                torch.maximum(teacher_probability, 1 - teacher_probability) >= 0.7
            ).float()[:, None]
            actual_logits, actual_pseudo, actual_valid = segmentation_spy.call_args.args[:3]
            self.assertTrue(torch.equal(actual_pseudo, expected_pseudo))
            self.assertTrue(torch.equal(actual_valid, expected_valid))
            self.assertEqual(actual_logits.shape, expected_pseudo.shape)
            self.assertAlmostEqual(result["quality_loss"], -result["cac"], places=5)
            expected_loss = (
                result["bce"] + 1.0 - result["dice"]
                + args.w_entropy * result["entropy"]
                + args.w_cac * result["quality_loss"]
            )
            self.assertAlmostEqual(result["loss"], expected_loss, places=5)
        finally:
            adapter.close()

    def test_recall_recovery_uses_multi_view_votes_and_has_no_gradient(self):
        teacher_prob = torch.tensor(
            [0.39, 0.40, 0.41, 0.45, 0.49, 0.50, 0.80], dtype=torch.float32
        ).view(1, 1, 1, 1, 7)
        # The selected view can be below 0.5 as long as enough of all views
        # support the voxel. A vote count of one is insufficient.
        vote_count = torch.tensor(
            [3, 2, 1, 2, 1, 9, 9], dtype=torch.float32
        ).view_as(teacher_prob).requires_grad_()
        teacher_confidence = torch.maximum(teacher_prob, 1 - teacher_prob)
        selected_pseudo = (teacher_prob >= 0.5).float().unsqueeze(1)
        selected_valid = (teacher_confidence >= 0.7).float().unsqueeze(1)

        recovered_pseudo, recovered_valid, candidate, count, ratio = apply_recall_recovery(
            teacher_prob,
            vote_count,
            selected_pseudo,
            selected_valid,
            teacher_low=0.4,
            min_view_votes=2,
        )
        self.assertEqual(
            candidate.flatten().tolist(),
            [False, True, False, True, False, False, False],
        )
        self.assertEqual(
            recovered_pseudo.flatten().tolist(), [0, 1, 0, 1, 0, 1, 1]
        )
        self.assertEqual(
            recovered_valid.flatten().tolist(), [0, 1, 0, 1, 0, 0, 1]
        )
        self.assertEqual(float(count.item()), 2.0)
        self.assertAlmostEqual(float(ratio.item()), 1.0, places=5)
        self.assertFalse(candidate.requires_grad)
        self.assertFalse(recovered_pseudo.requires_grad)
        self.assertFalse(recovered_valid.requires_grad)
        self.assertIsNone(vote_count.grad)

    def test_recall_recovery_uses_votes_when_selected_view_is_below_threshold(self):
        import sfda_voxtell

        model = _TinyVoxTell()
        with torch.no_grad():
            model.image_encoder.weight.zero_()
            model.image_encoder.weight[0, 0, 0, 0, 0] = 1.0
        args = _args(
            quality_metric="cac",
            num_aug_views=3,
            selection_p=0.34,
            confidence_threshold=0.7,
            enable_recall_recovery=True,
            recovery_teacher_low=0.4,
            recovery_view_threshold=0.5,
            recovery_min_view_votes=2,
            w_cac=0.0,
        )
        adapter = VoxTellPromptSFDA(
            model, torch.ones(1, 1, 4), torch.device("cpu"), args
        )
        try:
            teacher_values = torch.tensor(
                [0.39, 0.41, 0.45, 0.49, 0.50, 0.80], dtype=torch.float32
            )
            weak = torch.logit(teacher_values).view(1, 1, 1, 1, 6)
            strong_prob = torch.tensor(
                [0.9, 0.9, 0.9, 0.1, 0.9, 0.9], dtype=torch.float32
            )
            extra_prob = torch.tensor(
                [0.1, 0.1, 0.9, 0.9, 0.9, 0.9], dtype=torch.float32
            )
            strong = torch.logit(strong_prob).view_as(weak)
            extra = torch.logit(extra_prob).view_as(weak)
            with torch.no_grad():
                adapter.teacher_soft_prompt.zero_()
                adapter.soft_prompt_embedding.zero_()
            adapter._make_extra_views = lambda _weak, count: [extra][:count]

            def choose_original(scores, probabilities, selection_p):
                votes = adapter._last_recovery_vote_count
                self.assertIsNotNone(votes)
                self.assertFalse(votes.requires_grad)
                self.assertEqual(votes.flatten().tolist(), [1, 1, 2, 1, 3, 3])
                return torch.zeros((1, 1), dtype=torch.long)

            with mock.patch(
                "sfda_voxtell.masked_segmentation_loss",
                wraps=masked_segmentation_loss,
            ) as segmentation_spy, mock.patch.object(
                sfda_voxtell, "select_cac_views", side_effect=choose_original
            ):
                result = adapter.adapt_batch(weak, strong, ["case-recovery"])

            # The selected/original view predicts this candidate voxel below
            # 0.5; its promotion is supported by the other augmented views.
            self.assertLess(float(torch.sigmoid(model.outputs[-1]).flatten()[2]), 0.5)
            _, actual_pseudo, actual_valid = segmentation_spy.call_args.args[:3]
            self.assertEqual(actual_pseudo.flatten().tolist(), [0, 0, 1, 0, 1, 1])
            self.assertEqual(actual_valid.flatten().tolist(), [0, 0, 1, 0, 0, 1])
            self.assertEqual(result["candidate_voxels"], 1.0)
            self.assertAlmostEqual(result["candidate_ratio"], 0.5, places=5)
            self.assertEqual(result["update_skipped"], 0.0)
            self.assertIsNone(adapter._last_recovery_vote_count)
        finally:
            adapter.close()

    def test_recall_recovery_cli_is_opt_in_with_configurable_thresholds(self):
        with mock.patch.object(sys, "argv", ["run_sfda_voxtell.py", "--data_dir", "/tmp/data"]):
            defaults = parse_args()
        self.assertFalse(defaults.enable_recall_recovery)
        self.assertEqual(defaults.recovery_teacher_low, 0.4)
        self.assertEqual(defaults.recovery_view_threshold, 0.5)
        self.assertEqual(defaults.recovery_min_view_votes, 2)

        with mock.patch.object(
            sys,
            "argv",
            [
                "run_sfda_voxtell.py", "--data_dir", "/tmp/data",
                "--enable_recall_recovery", "--recovery_teacher_low", "0.45",
                "--recovery_view_threshold", "0.6", "--recovery_min_view_votes", "3",
            ],
        ):
            configured = parse_args()
        self.assertTrue(configured.enable_recall_recovery)
        self.assertEqual(configured.recovery_teacher_low, 0.45)
        self.assertEqual(configured.recovery_view_threshold, 0.6)
        self.assertEqual(configured.recovery_min_view_votes, 3)

    def test_all_invalid_saaf_falls_back_to_original_and_updates(self):
        args = _args(
            quality_metric="saaf", quality_mode="cac", selection_p=0.1,
            num_aug_views=3, batch_size=1, confidence_threshold=0.7,
        )
        model = _TinyAttentionVoxTell()
        model.uniform_attention = True
        adapter = VoxTellPromptSFDA(
            model, torch.ones(1, 1, 4), torch.device("cpu"), args
        )
        try:
            views = torch.stack(
                [torch.zeros(1, 2, 4, 4), torch.ones(1, 2, 4, 4), torch.full((1, 2, 4, 4), 2.0)],
                dim=0,
            ).unsqueeze(0)
            selected = adapter._select_views(views, ["case-invalid"])
            self.assertEqual(selected.tolist(), [[0]])
            self.assertTrue(all(row["selection_fallback"] for row in adapter.quality_diagnostics))
            self.assertTrue(all(not row["valid"] for row in adapter.quality_diagnostics))
            self.assertTrue(adapter.quality_diagnostics[0]["selected"])

            adapter.quality_diagnostics.clear()
            calls_before_adapt = len(model.prompt_calls)
            result = adapter.adapt_batch(
                torch.randn(1, 1, 2, 4, 4),
                torch.randn(1, 1, 2, 4, 4),
                ["case-invalid"],
            )
            self.assertEqual(result["update_skipped"], 0.0)
            self.assertNotIn("quality_invalid", result)
            self.assertEqual(len(model.prompt_calls), calls_before_adapt + 3)
            self.assertTrue(np.isfinite(result["loss"]))
        finally:
            adapter.close()

    def test_teacher_pseudo_label_prompt_thresholds_and_loss_path_unchanged(self):
        import sfda_voxtell

        model = _TinyVoxTell()
        args = _args(confidence_threshold=0.7, quality_metric="cac", w_cac=0.0)
        adapter = VoxTellPromptSFDA(
            model, torch.ones(1, 1, 4), torch.device("cpu"), args
        )
        try:
            with torch.no_grad():
                adapter.teacher_soft_prompt.fill_(0.65)
                adapter.soft_prompt_embedding.fill_(-0.35)
            teacher_prompt_before = adapter.teacher_soft_prompt.detach().clone()
            with mock.patch.object(
                sfda_voxtell,
                "masked_segmentation_loss",
                wraps=masked_segmentation_loss,
            ) as segmentation_spy:
                result = adapter.adapt_batch(
                    torch.zeros(1, 1, 2, 2, 2),
                    torch.ones(1, 1, 2, 2, 2),
                    ["case-teacher"],
                )
            self.assertTrue(
                torch.equal(
                    model.prompt_calls[0],
                    teacher_prompt_before.expand(1, -1, -1).unsqueeze(2),
                )
            )
            teacher_probability = torch.sigmoid(model.outputs[0].float())
            expected_pseudo = (teacher_probability >= 0.5).float()[:, None]
            expected_valid = (
                torch.maximum(teacher_probability, 1 - teacher_probability) >= 0.7
            ).float()[:, None]
            actual_logits, actual_pseudo, actual_valid = segmentation_spy.call_args.args[:3]
            self.assertTrue(torch.equal(actual_pseudo, expected_pseudo))
            self.assertTrue(torch.equal(actual_valid, expected_valid))
            self.assertEqual(actual_logits.shape, expected_pseudo.shape)
            self.assertNotIn("candidate_voxels", result)
            self.assertNotIn("candidate_ratio", result)
        finally:
            adapter.close()

    def test_invalid_saaf_views_are_not_selected(self):
        metrics = {
            name: torch.tensor([[0.1, 0.9, 0.8]])
            for name in ("cac", "saaf", "purity", "coverage")
        }
        metrics["entropy"] = torch.tensor([[0.1, 0.2, 0.3]])
        selected, skipped = _selection_indices(metrics, torch.tensor([False, True, False]))
        self.assertFalse(skipped["saaf"])
        self.assertEqual(selected["saaf"], 1)
        self.assertEqual(selected["cac"], 1)
        selected, skipped = _selection_indices(metrics, torch.tensor([False, False, False]))
        self.assertTrue(skipped["saaf"])
        self.assertIn("cac", selected)
        probabilities = torch.full((1, 3, 1, 2, 2, 2), 0.5)
        selected = select_cac_views(
            metrics["saaf"], probabilities, 0.1,
            valid_mask=torch.tensor([[False, True, False]]),
        )
        self.assertTrue(torch.equal(selected, torch.tensor([[1]])))
        with self.assertRaises(ValueError):
            select_cac_views(
                metrics["saaf"], probabilities, 0.1,
                valid_mask=torch.tensor([[False, False, False]]),
            )

    def test_invalid_saaf_lowest_entropy_is_never_selected(self):
        # View 0 has the lowest entropy but is invalid; rank fusion must still
        # select the valid view only.
        scores = torch.tensor([[0.1, 0.2]])
        probabilities = torch.stack(
            [torch.zeros(1, 1, 2, 2, 2), torch.full((1, 1, 2, 2, 2), 0.5)], dim=1
        )
        selected = select_cac_views(
            scores, probabilities, 0.5, valid_mask=torch.tensor([[False, True]])
        )
        self.assertEqual(selected.tolist(), [[1]])

    def test_selection_keeps_only_available_valid_views_when_keep_is_larger(self):
        scores = torch.tensor([[0.1, 0.9, 0.2]])
        probabilities = torch.full((1, 3, 1, 2, 2, 2), 0.5)
        selected = select_cac_views(
            scores, probabilities, 1.0, valid_mask=torch.tensor([[False, True, False]])
        )
        self.assertEqual(selected.tolist(), [[1]])

    def test_cac_selection_is_independent_of_saaf_mask(self):
        metrics = {
            "cac": torch.tensor([0.9, 0.1, 0.2]),
            "saaf": torch.tensor([0.1, 0.8, 0.7]),
            "purity": torch.tensor([0.1, 0.8, 0.7]),
            "coverage": torch.tensor([0.1, 0.8, 0.7]),
            "entropy": torch.tensor([0.3, 0.2, 0.1]),
        }
        all_selected, _ = _selection_indices(metrics, torch.tensor([True, True, True]))
        masked_selected, _ = _selection_indices(metrics, torch.tensor([False, True, True]))
        self.assertEqual(all_selected["cac"], masked_selected["cac"])
        self.assertEqual(all_selected["cac_entropy"], masked_selected["cac_entropy"])

    def test_saaf_spearman_excludes_invalid_views(self):
        # The invalid outlier must not influence a SAAF correlation.  With the
        # two valid observations left, this is a perfect monotonic relation.
        values = [0.1, 0.2, 100.0]
        dice = [0.1, 0.2, 0.0]
        valid = [True, True, False]
        self.assertAlmostEqual(
            spearman([v for v, ok in zip(values, valid) if ok],
                     [d for d, ok in zip(dice, valid) if ok]),
            1.0,
        )

    def test_invalid_patch_hole_marks_selected_volume_invalid(self):
        evidence_sum = np.ones((2, 2, 2), dtype=np.float32)
        evidence_weight = np.ones((2, 2, 2), dtype=np.float32)
        evidence_weight[0, 0, 0] = 0.0
        fused, covered = _fuse_evidence_with_valid_weights(evidence_sum, evidence_weight)
        self.assertFalse(bool(covered.all()))
        self.assertFalse(_full_volume_selection_valid(evidence_weight, 1))
        # The uncovered voxel is not treated as a valid selected prediction.
        self.assertIsNotNone(fused)

    def test_invalid_patch_with_overlapping_valid_patch_keeps_volume_valid(self):
        # Two overlapping patches provide complete effective evidence coverage.
        evidence_sum = np.array([[[1.0, 1.0], [1.0, 1.0]]], dtype=np.float32)
        evidence_weight = np.array([[[1.0, 2.0], [1.0, 2.0]]], dtype=np.float32)
        _, covered = _fuse_evidence_with_valid_weights(evidence_sum, evidence_weight)
        self.assertTrue(bool(covered.all()))
        self.assertTrue(_full_volume_selection_valid(evidence_weight, 1))

    def test_invalid_evidence_does_not_dilute_valid_evidence(self):
        evidence_sum = np.array([[[2.0, 2.0]]], dtype=np.float32)
        evidence_weight = np.array([[[1.0, 1.0]]], dtype=np.float32)
        fused, _ = _fuse_evidence_with_valid_weights(evidence_sum, evidence_weight)
        self.assertTrue(np.allclose(fused, 2.0))

    def test_cac_full_volume_coverage_is_independent_of_saaf_patch_validity(self):
        # CAC accumulates every sliding-window patch, so its own denominator
        # remains complete even if the evidence denominator has a hole.
        cac_weight = np.ones((2, 2, 2), dtype=np.float32)
        saaf_weight = cac_weight.copy()
        saaf_weight[0, 0, 0] = 0.0
        self.assertTrue(_full_volume_selection_valid(cac_weight, 1))
        self.assertFalse(_full_volume_selection_valid(saaf_weight, 1))

    def test_saaf_rejects_batch_size_greater_than_one(self):
        with self.assertRaisesRegex(ValueError, "batch_size=1"):
            VoxTellPromptSFDA(
                _TinyAttentionVoxTell(),
                torch.ones(1, 1, 4),
                torch.device("cpu"),
                _args(quality_metric="saaf", batch_size=2, selection_p=0.1),
            )

    def test_tse_adapter_uses_cross_case_prototypes_without_nan(self):
        model = _TinyVoxTell()
        adapter = VoxTellPromptSFDA(
            model,
            torch.ones(1, 1, 4),
            torch.device("cpu"),
            _args(
                quality_mode="tse",
                w_quality=0.0,
                w_seg=1.0,
                w_entropy=0.0,
            ),
        )
        statistics = {
            "fg_sum": torch.tensor(
                [[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]
            ),
            "fg_count": torch.tensor([1.0, 2.0]),
            "bg_sum": torch.tensor(
                [[0.0, 1.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0]]
            ),
            "bg_count": torch.tensor([1.0, 2.0]),
        }
        adapter.prototype_memory.add(["case-a", "case-b"], statistics)
        try:
            prompt_before = adapter.soft_prompt_embedding.detach().clone()
            result = adapter.adapt_batch(
                torch.randn(1, 1, 2, 2, 2),
                torch.randn(1, 1, 2, 2, 2),
                ["case-a"],
            )
            for name in ("purity", "completeness", "tse", "quality_loss"):
                self.assertTrue(np.isfinite(result[name]))
            self.assertEqual(result["prototype_valid"], 1.0)
            self.assertGreater(result["soft_prompt_grad_norm"], 0.0)
            self.assertFalse(torch.equal(prompt_before, adapter.soft_prompt_embedding))
        finally:
            adapter.close()

    def test_empty_seed_prototypes_produce_invalid_but_finite_evidence(self):
        memory = SemanticPrototypeMemory()
        memory.add(
            ["case-a", "case-b"],
            {
                "fg_sum": torch.zeros(2, 4),
                "fg_count": torch.zeros(2),
                "bg_sum": torch.zeros(2, 4),
                "bg_count": torch.zeros(2),
            },
        )
        _, _, valid = memory.leave_one_out(["case-a"], "cpu")
        self.assertFalse(valid.any())

    def test_equal_quality_scores_receive_tied_ranks(self):
        ranks = average_tie_ranks(torch.tensor([[0.0, 0.0, 1.0, 1.0]]), descending=True)
        self.assertEqual(ranks.tolist(), [[2.5, 2.5, 0.5, 0.5]])

    def test_deterministic_nonoverlap_covers_complete_case_once(self):
        volume = torch.arange(27, dtype=torch.float32).view(1, 3, 3, 3)
        padded, valid, original_shape = pad_to_patch_grid(volume, (2, 2, 2))
        locations = nonoverlapping_patch_locations(padded.shape[-3:], (2, 2, 2))
        patches = torch.stack([
            padded[(slice(None), slice(d, d + 2), slice(h, h + 2), slice(w, w + 2))]
            for d, h, w in locations
        ])
        fused = fuse_volume_patches(patches, locations, padded.shape[-3:])
        self.assertEqual(original_shape, (3, 3, 3))
        self.assertTrue(torch.equal(fused[..., :3, :3, :3], volume))
        self.assertEqual(float(valid.sum()), 27.0)

    def test_overlapping_sliding_locations_have_full_coverage(self):
        locations = sliding_window_locations((5, 6, 7), (3, 3, 3), overlap=0.5)
        coverage = torch.zeros(5, 6, 7)
        for d, h, w in locations:
            coverage[d:d + 3, h:h + 3, w:w + 3] += 1
        self.assertTrue(torch.all(coverage > 0))

    def test_overlapping_fusion_is_coverage_normalized(self):
        locations = [
            (d, h, w)
            for d in (0, 1)
            for h in (0, 1)
            for w in (0, 1)
        ]
        patches = torch.stack(
            [torch.full((1, 2, 2, 2), 2.0 if i == 0 else 4.0)
             for i in range(len(locations))]
        )
        fused = fuse_volume_patches(patches, locations, (3, 3, 3))
        self.assertAlmostEqual(float(fused[0, 1, 1, 1]), 3.75)
        self.assertAlmostEqual(float(fused[0, 0, 0, 0]), 2.0)
        self.assertAlmostEqual(float(fused[0, 2, 2, 2]), 4.0)

    def test_evidence_localization_metrics_mark_empty_gt_invalid(self):
        evidence = torch.tensor([[[0.1, 0.9]]])
        empty_target = torch.zeros_like(evidence)
        result = evidence_localization_metrics(evidence, empty_target, 0.5)
        self.assertFalse(result["valid"])
        self.assertEqual(result["reason"], "empty_gt")
        self.assertIsNone(binary_auroc([0.1, 0.2], [False, False]))
        self.assertIsNone(binary_auprc([0.1, 0.2], [False, False]))

    def test_evidence_ranking_metrics_are_perfect_for_separated_scores(self):
        self.assertAlmostEqual(binary_auroc([0.1, 0.9], [False, True]), 1.0)
        self.assertAlmostEqual(binary_auprc([0.1, 0.9], [False, True]), 1.0)

    def test_prototype_build_uses_all_deterministic_case_patches(self):
        adapter = VoxTellPromptSFDA(
            _TinyVoxTell(), torch.ones(1, 1, 4), torch.device("cpu"), _args(quality_mode="tse")
        )
        adapter.quality_config.update(
            foreground_probability_threshold=0.0,
            background_probability_threshold=1.0,
            foreground_text_similarity_threshold=-1.0,
            background_text_similarity_threshold=1.0,
            probability_stability_threshold=1.0,
            similarity_stability_threshold=1.0,
            prototype_patch_batch_size=2,
        )
        adapter.args.output_dir = tempfile.mkdtemp()
        entries = [(Path("case-a.nii.gz"), None), (Path("case-b.nii.gz"), None)]
        loader = SimpleNamespace(
            dataset=SimpleNamespace(entries=entries, patch_size=(2, 2, 2))
        )
        try:
            with mock.patch.object(
                sfda_data,
                "load_preprocessed_image",
                return_value=torch.zeros(1, 3, 3, 3),
            ):
                adapter.build_prototype_memory(loader)
            # 3^3 is padded to 4^3, hence 8 non-overlapping patches per case.
            self.assertEqual(adapter.prototype_diagnostics["cases"]["case-a.nii.gz"]["patches"], 8)
            self.assertEqual(adapter.prototype_diagnostics["cases"]["case-b.nii.gz"]["patches"], 8)
            self.assertEqual(adapter.prototype_diagnostics["dataset"]["cases"], 2)
            self.assertGreater(adapter.prototype_diagnostics["dataset"]["foreground_seed_count"], 0)
            self.assertGreater(adapter.prototype_diagnostics["dataset"]["background_seed_count"], 0)
        finally:
            adapter.close()

    def test_checkpoint_restores_soft_prompt_teacher_and_optimizer(self):
        adapter, _, _ = _make_adapter()
        restored, _, _ = _make_adapter()
        try:
            adapter.prototype_memory.add(
                ["case-a"],
                {
                    "fg_sum": torch.ones(1, 4),
                    "fg_count": torch.ones(1),
                    "bg_sum": -torch.ones(1, 4),
                    "bg_count": torch.ones(1),
                },
            )
            adapter.adapt_batch(
                torch.randn(1, 1, 2, 2, 2), torch.randn(1, 1, 2, 2, 2)
            )
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "checkpoint.pt"
                save_sfda_checkpoint(path, adapter, adapter.args, [{"epoch": 1}])
                checkpoint = load_sfda_checkpoint(path, restored)
                self.assertEqual(checkpoint["format"], "voxtell-sfda-prompt-tse-v5")
                self.assertIn("soft_prompt_embedding", checkpoint)
                self.assertIn("teacher_soft_prompt", checkpoint)
                self.assertNotIn("text_anchor", checkpoint)
                self.assertIn("prototype_memory", checkpoint)
                self.assertEqual(restored.prototype_memory.contributors(), ["case-a"])
                self.assertTrue(torch.equal(adapter.soft_prompt_embedding, restored.soft_prompt_embedding))
                self.assertTrue(torch.equal(adapter.teacher_soft_prompt, restored.teacher_soft_prompt))
                self.assertEqual(adapter.optimizer.state_dict().keys(), restored.optimizer.state_dict().keys())
                for key, state in adapter.optimizer.state_dict()["state"].items():
                    for state_key, value in state.items():
                        other = restored.optimizer.state_dict()["state"][key][state_key]
                        self.assertTrue(torch.equal(value, other))
        finally:
            adapter.close()
            restored.close()

    def test_training_dataset_uses_no_label(self):
        # The dataset stores train entries as (image, None) and __getitem__
        # only dereferences the first item. Use a missing label path to make
        # accidental label access fail immediately.
        from data.sfda_voxtell import VoxTellTargetDataset  # noqa: WPS433

        dataset = VoxTellTargetDataset.__new__(VoxTellTargetDataset)
        dataset.entries = [(Path("missing-image.nii.gz"), Path("missing-label.nii.gz"))]
        dataset.patch_size = (2, 2, 2)
        dataset._load = lambda _path: torch.zeros(1, 2, 2, 2)
        weak, strong, image_name = dataset[0]
        self.assertEqual(tuple(weak.shape), (1, 2, 2, 2))
        self.assertEqual(tuple(strong.shape), (1, 2, 2, 2))
        self.assertEqual(image_name, "missing-image.nii.gz")


if __name__ == "__main__":
    unittest.main()
