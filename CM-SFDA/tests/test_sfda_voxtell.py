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
    VoxTellPromptSFDA,
    cac_loss,
    compute_cac_score,
    entropy_loss,
    load_sfda_checkpoint,
    masked_segmentation_loss,
    save_sfda_checkpoint,
    select_cac_views,
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
    should_evaluate_epoch,
)
from evaluate_quality_metrics import (  # noqa: E402
    _fuse_logits_then_sigmoid,
    _fuse_evidence_with_valid_weights,
    _full_volume_selection_valid,
    _patch_oracle_stats,
    _soft_consistency,
    _selection_indices,
    binary_auprc,
    binary_auroc,
    evidence_localization_metrics,
    summarize_quality_audit,
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
        self.outputs = []

    def forward(self, image, prompt):
        self.forward_calls.append((image.shape[0], torch.is_grad_enabled()))
        self.prompt_calls.append(prompt.detach().clone())
        features = self.image_encoder(image)
        self.project_bottleneck_embed(features)
        text = self.project_text_embed(prompt)
        bias = text[0, :, 0].view(image.shape[0], 1, 1, 1, 1)
        logits = features[:, :1] + bias
        self.outputs.append(logits.detach().clone())
        return logits


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
                adapter.adapt_batch(
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
