import copy
import os
import sys
import types
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data.voxtell_p0 import make_case_patches, pad_to_patch_grid
from method.voxtell_cmtta import (
    ShortPromptMemory,
    VoxTellCMTTA,
    avg_entropy,
    case_soft_dice_from_components,
    cac_from_features,
    cac_from_components,
    cac_components_from_features,
    decoder_consistency_probabilities,
    decoder_grid_to_input_order,
    check_voxtell_decoder_d5_alignment,
    masked_balanced_bce_from_components,
    masked_tversky_loss_from_components,
    masked_dice_components,
    masked_entropy_components,
    select_cac_view,
    soft_dice_loss,
    tdc_from_components,
    tdc_patch_components,
)
from run_voxtell_cmtta import (
    binary_diagnostic_metrics,
    case_metric_changes,
    check_prediction_nifti_geometry,
    compute_tdc_sliding_comparisons,
    evaluate_case,
    macro_average_case_metrics,
    patch_teacher_probability,
    predict_case_soft_probability,
    save_prediction_nifti,
    selected_view_data,
    selector_only_case_report,
    sliding_teacher_patch_labels,
    stitch_patch_probabilities,
    summarize_gradient_conflict_diagnostics,
    summarize_selector_only,
    zero_shot_diagnostic_fields,
)


class TinyVoxTell(nn.Module):
    """Small frozen VoxTell-shaped network for protocol tests."""

    def __init__(self, text_dim=2):
        super().__init__()
        self.project_bottleneck_embed = nn.Linear(1, 2, bias=False)
        self.project_text_embed = nn.Linear(text_dim, 2, bias=False)

    def forward(self, image, text_embedding, return_decoder_outputs=False):
        batch, _, depth, height, width = image.shape
        if batch == 1:
            self.last_long_input = image.detach().clone()
        else:
            self.last_student_input = image.detach().clone()
        visual = image[:, 0].permute(1, 2, 3, 0).reshape(depth * height * width, batch, 1)
        self.project_bottleneck_embed(visual)
        text = text_embedding.squeeze(2).permute(1, 0, 2)
        projected_text = self.project_text_embed(text)
        # Keep the logits in the same (D,H,W) order as this test network's
        # projected visual token grid.
        prompt_bias = projected_text.mean(dim=-1).transpose(0, 1).view(batch, 1, 1, 1, 1)
        logits = image[:, :1] + prompt_bias
        if return_decoder_outputs:
            return [logits, logits, logits, logits, logits]
        return logits


class TinyQwenTokenizer:
    """Minimal tokenizer exposing the Hugging Face fields used by the adapter."""

    def __call__(self, texts, **kwargs):
        if kwargs.get("add_special_tokens") is False:
            return {"input_ids": torch.tensor([[11]]), "attention_mask": torch.tensor([[1]])}
        return {
            "input_ids": torch.tensor([[1, 10, 11, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1]]),
            "offset_mapping": torch.tensor([[[0, 0], [0, 7], [7, 12], [0, 0]]]),
        }


class MultiTokenQwenTokenizer:
    """Tokenizer double where ``liver`` is represented by two tokens."""

    def __call__(self, texts, **kwargs):
        del texts
        if kwargs.get("add_special_tokens") is False:
            return {"input_ids": torch.tensor([[11, 12]]), "attention_mask": torch.tensor([[1, 1]])}
        return {
            "input_ids": torch.tensor([[1, 10, 11, 12, 2]]),
            "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
            "offset_mapping": torch.tensor(
                [[[0, 0], [0, 7], [7, 9], [9, 12], [0, 0]]]
            ),
        }


class TinyQwen(nn.Module):
    """Frozen differentiable text encoder used to test ctx input learning."""

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(32, 4)

    def get_input_embeddings(self):
        return self.token_embedding

    def forward(self, inputs_embeds=None, attention_mask=None, input_ids=None):
        if inputs_embeds is None:
            inputs_embeds = self.token_embedding(input_ids)
        self.last_inputs_embeds = inputs_embeds
        del attention_mask
        hidden = inputs_embeds + inputs_embeds.mean(dim=1, keepdim=True)
        return types.SimpleNamespace(last_hidden_state=hidden)


class TinySlidingPredictor:
    """Sliding-window double returning logits in the input D,H,W order."""

    def __init__(self):
        self.device = torch.device("cpu")
        self.calls = []

    def predict_sliding_window_return_logits(self, data, text_feature):
        self.calls.append((tuple(data.shape), tuple(text_feature.shape)))
        return data[0].unsqueeze(0)


def make_args(**overrides):
    values = dict(
        lr=0.05,
        ema_momentum=0.9,
        w_cac=1.0,
        w_entropy=0.1,
        num_aug_views=2,
        selection_p=0.1,
        amp_init_scale=32.0,
        grad_clip=1.0,
        short_memory_length=2,
        amb_weight=0.05,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


class VoxTellCMTTATest(unittest.TestCase):
    def test_binary_diagnostic_metrics_returns_confusion_and_all_scores(self):
        prediction = np.array([1, 1, 0, 0], dtype=np.uint8)
        target = np.array([1, 0, 1, 0], dtype=np.uint8)
        metrics = binary_diagnostic_metrics(prediction, target)
        self.assertEqual(
            {key: metrics[key] for key in ("TP", "FP", "FN", "TN")},
            {"TP": 1, "FP": 1, "FN": 1, "TN": 1},
        )
        self.assertAlmostEqual(metrics["Dice"], 0.5)
        self.assertAlmostEqual(metrics["mIoU"], 1.0 / 3.0)
        self.assertAlmostEqual(metrics["Precision"], 0.5)
        self.assertAlmostEqual(metrics["Recall"], 0.5)
        self.assertEqual(metrics["prediction_foreground_volume"], 2.0)
        self.assertEqual(metrics["target_foreground_volume"], 2.0)
        self.assertAlmostEqual(
            metrics["Dice"],
            2.0 * metrics["Precision"] * metrics["Recall"]
            / (metrics["Precision"] + metrics["Recall"]),
        )

    def test_binary_diagnostic_metrics_empty_cases_are_finite_and_compatible(self):
        cases = (
            (
                np.zeros(4, dtype=np.uint8),
                np.zeros(4, dtype=np.uint8),
                {"TP": 0, "FP": 0, "FN": 0, "TN": 4,
                 "Dice": 1.0, "mIoU": 1.0, "Precision": 1.0, "Recall": 1.0},
            ),
            (
                np.zeros(4, dtype=np.uint8),
                np.array([1, 1, 0, 0], dtype=np.uint8),
                {"TP": 0, "FP": 0, "FN": 2, "TN": 2,
                 "Dice": 0.0, "mIoU": 0.0, "Precision": 1.0, "Recall": 0.0},
            ),
            (
                np.array([1, 1, 0, 0], dtype=np.uint8),
                np.zeros(4, dtype=np.uint8),
                {"TP": 0, "FP": 2, "FN": 0, "TN": 2,
                 "Dice": 0.0, "mIoU": 0.0, "Precision": 0.0, "Recall": 1.0},
            ),
        )
        for prediction, target, expected in cases:
            metrics = binary_diagnostic_metrics(prediction, target)
            for key, value in expected.items():
                self.assertEqual(metrics[key], value)
            self.assertTrue(
                all(np.isfinite(float(metrics[key])) for key in (
                    "Dice", "mIoU", "Precision", "Recall"
                ))
            )

    def test_evaluate_case_always_records_full_case_diagnostics(self):
        prediction = np.array(
            [[[1, 0], [1, 0]], [[0, 0], [0, 0]]], dtype=np.uint8
        )
        target = np.array(
            [[[1, 1], [0, 0]], [[0, 0], [0, 0]]], dtype=np.uint8
        )
        with patch("run_voxtell_cmtta.predict_case", return_value=prediction), \
             patch("run_voxtell_cmtta.load_ras_label", return_value=target), \
             patch("run_voxtell_cmtta.save_prediction_nifti"), \
             patch("run_voxtell_cmtta.check_prediction_nifti_geometry"):
            row = evaluate_case(
                predictor=None,
                image_path=Path("case.nii.gz"),
                label_path=Path("case_label.nii.gz"),
                data=None,
                bbox=None,
                original_shape=target.shape,
                text_feature=None,
                output_dir=Path("unused"),
                include_diagnostic_metrics=False,
            )
        self.assertEqual(row["TP"], 1)
        self.assertEqual(row["FP"], 1)
        self.assertEqual(row["FN"], 1)
        self.assertEqual(row["TN"], 5)
        self.assertEqual(row["prediction_foreground_volume"], 2.0)
        self.assertEqual(row["target_foreground_volume"], 2.0)
        self.assertIn("Precision", row)
        self.assertIn("Recall", row)

    def test_zero_shot_aliases_changes_and_macro_average_are_case_level(self):
        zero = binary_diagnostic_metrics(
            np.array([1, 0, 0, 0], dtype=np.uint8),
            np.array([1, 1, 0, 0], dtype=np.uint8),
        )
        adapted = binary_diagnostic_metrics(
            np.array([1, 1, 1, 0], dtype=np.uint8),
            np.array([1, 1, 0, 0], dtype=np.uint8),
        )
        zero_fields = zero_shot_diagnostic_fields(zero)
        self.assertEqual(zero_fields["zero_shot_Dice"], zero_fields["zero_shot_sliding_dice"])
        self.assertEqual(zero_fields["zero_shot_mIoU"], zero_fields["zero_shot_sliding_miou"])
        changes = case_metric_changes(adapted, zero)
        self.assertGreater(changes["Dice_change_from_before_adaptation"], 0.0)
        self.assertGreater(changes["mIoU_change_from_before_adaptation"], 0.0)
        self.assertGreater(changes["prediction_foreground_volume_change"], 0.0)
        self.assertAlmostEqual(changes["prediction_foreground_volume_change_ratio"], 2.0)
        zero_empty = binary_diagnostic_metrics(
            np.zeros(2, dtype=np.uint8), np.zeros(2, dtype=np.uint8)
        )
        self.assertIsNone(case_metric_changes(adapted, zero_empty)["prediction_foreground_volume_change_ratio"])

        row_one = {
            "Dice": 0.5, "mIoU": 0.25, "Precision": 0.5, "Recall": 0.5,
            "zero_shot_Dice": 0.25, "zero_shot_mIoU": 0.1,
            "zero_shot_Precision": 0.2, "zero_shot_Recall": 0.3,
            "Dice_change_from_before_adaptation": 0.25,
            "mIoU_change_from_before_adaptation": 0.15,
            "Precision_change_from_before_adaptation": 0.3,
            "Recall_change_from_before_adaptation": 0.2,
        }
        row_two = {key: value * 0.5 for key, value in row_one.items()}
        average = macro_average_case_metrics([row_one, row_two])
        self.assertAlmostEqual(average["Dice"], 0.375)
        self.assertAlmostEqual(average["zero_shot_Dice"], 0.1875)
        self.assertAlmostEqual(average["Dice_change_from_before_adaptation"], 0.1875)
        self.assertAlmostEqual(average["Dice_change"], 0.1875)
        self.assertAlmostEqual(average["Precision_change"], 0.225)

    def test_selected_view_data_reuses_params_and_view_zero_is_identity(self):
        data = torch.arange(8, dtype=torch.float32).reshape(1, 2, 2, 2)
        params = [
            {"scale": 1.0, "offset": 0.0},
            {"scale": 2.0, "offset": -0.5},
        ]
        original, original_param = selected_view_data(data, params, 0)
        selected, selected_param = selected_view_data(data, params, 1)
        self.assertIs(original, data)
        self.assertEqual(original_param, params[0])
        self.assertTrue(torch.equal(selected, data * 2.0 - 0.5))
        self.assertEqual(selected_param, params[1])

    def test_tdc_sliding_comparisons_use_trace_view_four_sliding_calls_and_no_step(self):
        adapter = VoxTellCMTTA(
            TinyVoxTell(),
            torch.zeros(1, 1, 2),
            "cpu",
            make_args(
                view_selection_metric="tdc",
                num_aug_views=1,
                view_batch_size=1,
                w_cac=0.0,
                w_entropy=0.0,
            ),
        )
        try:
            patch_tensor = torch.zeros(1, 2, 2, 2)
            valid = torch.ones(2, 2, 2)
            prepared = adapter.prepare_case([patch_tensor], [valid])
            with torch.no_grad():
                pre_text = adapter._encode_ctx(adapter.ctx_delta.detach()).detach()
            trace = adapter.adapt_case(
                [patch_tensor], [valid], prepared_case=prepared
            )
            steps_before_eval = adapter.optimizer_step_count
            target = np.zeros((2, 2, 2), dtype=np.uint8)
            predictions = [
                np.zeros_like(target),
                np.ones_like(target),
                np.zeros_like(target),
                np.ones_like(target),
            ]
            with patch("run_voxtell_cmtta.predict_case", side_effect=predictions) as mocked:
                result = compute_tdc_sliding_comparisons(
                    predictor=object(),
                    data=patch_tensor,
                    bbox=None,
                    original_shape=target.shape,
                    target=target,
                    pre_text_feature=pre_text,
                    post_text_feature=pre_text,
                    params=prepared["params"],
                    selected_view=trace["selected_view"],
                )
            self.assertEqual(mocked.call_count, 4)
            self.assertEqual(result["selected_view"], trace["selected_view"])
            self.assertEqual(
                result["selected_param"],
                prepared["params"][trace["selected_view"]],
            )
            self.assertEqual(
                result["predictions"]["pre_original_sliding"].shape,
                target.shape,
            )
            self.assertIn("post_selected_minus_pre_selected", result["differences"])
            self.assertEqual(adapter.optimizer_step_count, steps_before_eval)
        finally:
            adapter.close()

    def test_tdc_sliding_differences_and_macro_average(self):
        target = np.array([1, 1, 0, 0], dtype=np.uint8)
        predictions = [
            np.array([1, 0, 0, 0], dtype=np.uint8),
            np.array([1, 1, 0, 0], dtype=np.uint8),
            np.array([1, 1, 1, 0], dtype=np.uint8),
            np.array([1, 1, 1, 1], dtype=np.uint8),
        ]
        metrics = [binary_diagnostic_metrics(prediction, target) for prediction in predictions]
        expected_delta = metrics[3]["Dice"] - metrics[1]["Dice"]
        row = {
            "Dice": metrics[2]["Dice"],
            "mIoU": metrics[2]["mIoU"],
            "Precision": metrics[2]["Precision"],
            "Recall": metrics[2]["Recall"],
            "zero_shot_Dice": metrics[0]["Dice"],
            "zero_shot_mIoU": metrics[0]["mIoU"],
            "zero_shot_Precision": metrics[0]["Precision"],
            "zero_shot_Recall": metrics[0]["Recall"],
            "Dice_change_from_before_adaptation": 0.0,
            "mIoU_change_from_before_adaptation": 0.0,
            "Precision_change_from_before_adaptation": 0.0,
            "Recall_change_from_before_adaptation": 0.0,
            "pre_original_sliding": metrics[0],
            "pre_selected_sliding": metrics[1],
            "post_original_sliding": metrics[2],
            "post_selected_sliding": metrics[3],
            "tdc_sliding_differences": {
                "post_selected_minus_pre_selected": {
                    metric: metrics[3][metric] - metrics[1][metric]
                    for metric in ("Dice", "mIoU", "Precision", "Recall")
                },
                "post_original_minus_pre_original": {
                    metric: metrics[2][metric] - metrics[0][metric]
                    for metric in ("Dice", "mIoU", "Precision", "Recall")
                },
                "post_selected_minus_post_original": {
                    metric: metrics[3][metric] - metrics[2][metric]
                    for metric in ("Dice", "mIoU", "Precision", "Recall")
                },
            },
        }
        average = macro_average_case_metrics([row, row])
        self.assertAlmostEqual(
            average["tdc_sliding_comparisons"]["post_selected_sliding"]["Dice"],
            metrics[3]["Dice"],
        )
        self.assertAlmostEqual(
            average["tdc_sliding_differences"]["post_selected_minus_pre_selected"]["Dice"],
            expected_delta,
        )

    def test_decoder_alignment_diagnostic_uses_d5_index_zero_and_dhw(self):
        model = TinyVoxTell()
        result = check_voxtell_decoder_d5_alignment(
            model,
            torch.zeros(1, 1, 3, 5, 7),
            torch.ones(1, 1, 1, 2),
        )
        self.assertEqual(result["normal_shape"], (1, 1, 3, 5, 7))
        self.assertEqual(result["d5_shape"], (1, 1, 3, 5, 7))
        self.assertAlmostEqual(result["max_abs_error"], 0.0, places=7)
        self.assertAlmostEqual(result["mean_abs_error"], 0.0, places=7)

    def test_decoder_consistency_keeps_non_cubic_dhw_coordinates(self):
        # Real VoxTell decoder order is (D,H,W)=(3,5,7); all axes differ.
        # Keep every voxel strictly above the 0.5 sigmoid threshold while
        # retaining a distinct value at every (d,h,w) coordinate.
        d5 = (torch.arange(3 * 5 * 7, dtype=torch.float32) + 1.0).reshape(
            1, 1, 3, 5, 7
        )
        d4 = torch.full((1, 1, 2, 3, 4), 4.0)
        d3 = torch.full((1, 1, 2, 3, 4), 4.0)
        d2 = torch.full((1, 1, 2, 3, 4), 4.0)
        valid = torch.ones(1, 1, 3, 5, 7)
        valid[0, 0, 2, 4, 6] = 0.0  # one known (d,h,w) voxel is padding
        coordinate_copy = decoder_grid_to_input_order(d5, (3, 5, 7))
        self.assertTrue(torch.equal(coordinate_copy, d5))
        self.assertEqual(float(coordinate_copy[0, 0, 2, 4, 6]), float(d5[0, 0, 2, 4, 6]))
        masks = decoder_consistency_probabilities([d5, d4, d3, d2], valid)
        self.assertEqual(tuple(masks["fg"].shape), (1, 1, 3, 5, 7))
        self.assertEqual(tuple(masks["probabilities"].shape), (1, 4, 3, 5, 7))
        self.assertEqual(int(masks["fg"].sum()), 104)
        self.assertFalse(bool(masks["fg"][0, 0, 2, 4, 6]))
        self.assertEqual(int(masks["bg"].sum()), 0)
        self.assertEqual(int(masks["amb"].sum()), 0)
        converted = decoder_grid_to_input_order(masks["fg"].float(), (3, 5, 7))
        self.assertFalse(bool(converted[0, 0, 2, 4, 6]))
        self.assertEqual(int(converted.sum()), 104)

    def test_decoder_masked_same_context_path_aligns_teacher_d5_and_student(self):
        model = TinyVoxTell()
        with torch.no_grad():
            model.project_text_embed.weight.fill_(1.0)
        adapter = VoxTellCMTTA(
            model,
            torch.ones(1, 1, 2),
            "cpu",
            make_args(
                pseudo_update_mode="decoder_masked",
                decoder_alignment_check=True,
                num_aug_views=1,
                view_batch_size=1,
                w_cac=0.0,
                w_entropy=0.0,
            ),
        )
        try:
            trace = adapter.adapt_case(
                [torch.zeros(1, 3, 5, 7)],
                [torch.ones(3, 5, 7)],
            )
            self.assertAlmostEqual(
                trace["decoder_alignment_max_abs_error"], 0.0, places=7
            )
            self.assertAlmostEqual(
                trace["decoder_alignment_mean_abs_error"], 0.0, places=7
            )
            self.assertTrue(trace["decoder_alignment_same_context"])
            self.assertAlmostEqual(
                trace["teacher_fg_mean_probability"],
                trace["aligned_student_fg_mean_probability"],
                places=6,
            )
            self.assertEqual(trace["optimizer_steps_for_case"], 1)
        finally:
            adapter.close()

    def test_decoder_masks_handle_fg_only_bg_only_and_ambiguous_regions(self):
        fg_outputs = [torch.full((1, 1, 2, 2, 3), 4.0) for _ in range(4)]
        bg_outputs = [torch.full((1, 1, 2, 2, 3), -4.0) for _ in range(4)]
        ambiguous_outputs = [torch.zeros(1, 1, 2, 2, 3) for _ in range(4)]
        valid = torch.ones(1, 1, 2, 2, 3)
        fg = decoder_consistency_probabilities(fg_outputs, valid)
        bg = decoder_consistency_probabilities(bg_outputs, valid)
        ambiguous = decoder_consistency_probabilities(ambiguous_outputs, valid)
        self.assertEqual(int(fg["fg"].sum()), 12)
        self.assertEqual(int(fg["bg"].sum()), 0)
        self.assertEqual(int(bg["fg"].sum()), 0)
        self.assertEqual(int(bg["bg"].sum()), 12)
        self.assertEqual(int(ambiguous["amb"].sum()), 12)

    def test_amb_anchor_is_ambiguous_without_miss_and_disjoint_from_miss(self):
        d5 = torch.full((1, 1, 1, 1, 3), -4.0)
        d4 = torch.full_like(d5, -4.0)
        d3 = torch.full_like(d5, -4.0)
        d2 = torch.full_like(d5, -4.0)
        # Voxel 0 is an ambiguous miss (one lower decoder votes foreground).
        d4[..., 0] = 4.0
        # Voxel 1 is ambiguous but has no lower-decoder foreground vote.
        d4[..., 1] = 0.0
        d3[..., 1] = 0.0
        d2[..., 1] = 0.0
        masks = decoder_consistency_probabilities(
            [d5, d4, d3, d2], torch.ones(1, 1, 1, 1, 3)
        )
        amb_anchor = masks["amb"] & ~masks["miss"]
        self.assertEqual(int((amb_anchor & masks["miss"]).sum()), 0)
        self.assertTrue(bool(amb_anchor[0, 0, 0, 0, 1]))
        self.assertTrue(bool(masks["miss"][0, 0, 0, 0, 0]))
        self.assertFalse(bool(amb_anchor[0, 0, 0, 0, 0]))

    def test_amb_bce_manual_gradient_excludes_miss_and_handles_empty_mask(self):
        logits = torch.tensor([[[[[0.0, 1.0, -1.0]]]]], requires_grad=True)
        target = torch.tensor([[[[[0.2, 0.8, 0.4]]]]])
        amb_anchor = torch.tensor([[[[[1.0, 1.0, 0.0]]]]])
        miss = torch.tensor([[[[[0.0, 0.0, 1.0]]]]])
        self.assertEqual(int((amb_anchor.bool() & miss.bool()).sum()), 0)
        expected = (
            F.binary_cross_entropy_with_logits(logits, target, reduction="none")
            * amb_anchor
        ).sum() / amb_anchor.sum().clamp_min(1.0)
        loss = expected
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertTrue(torch.isfinite(logits.grad).all())
        self.assertEqual(float(logits.grad[0, 0, 0, 0, 2]), 0.0)
        self.assertGreater(float(logits.grad[0, 0, 0, 0, 0].abs()), 0.0)
        self.assertGreater(float(logits.grad[0, 0, 0, 0, 1].abs()), 0.0)

        empty_logits = torch.zeros(1, 1, 1, 1, 2, requires_grad=True)
        empty_mask = torch.zeros_like(empty_logits)
        empty_loss = (
            F.binary_cross_entropy_with_logits(
                empty_logits, torch.zeros_like(empty_logits), reduction="none"
            )
            * empty_mask
        ).sum() / empty_mask.sum().clamp_min(1.0)
        empty_loss.backward()
        self.assertEqual(float(empty_loss.detach()), 0.0)
        self.assertTrue(torch.isfinite(empty_loss))
        self.assertTrue(torch.equal(empty_logits.grad, torch.zeros_like(empty_logits)))

    def test_decoder_masked_amb_weight_zero_is_baseline_and_positive_adds_ctx_gradient(self):
        params = [
            {"scale": 1.0, "offset": 1.0},
            {"scale": 1.0, "offset": -1.0},
        ]
        patch = torch.zeros(1, 1, 1, 2)
        valid = torch.ones(1, 1, 2)

        def run(amb_weight):
            model = TinyVoxTell()
            with torch.no_grad():
                model.project_text_embed.weight.fill_(1.0)
            adapter = VoxTellCMTTA(
                model,
                torch.zeros(1, 1, 2),
                "cpu",
                make_args(
                    pseudo_update_mode="decoder_masked",
                    num_aug_views=1,
                    view_batch_size=1,
                    w_cac=0.0,
                    w_entropy=0.0,
                    amb_weight=amb_weight,
                ),
            )
            try:
                adapter.optimizer.zero_grad(set_to_none=True)
                ctx = adapter.ctx.detach().clone()
                pseudo_loss, _ = adapter._backward_case_supervision(
                    [patch], [valid], params, 1, ctx, 1.0, ctx, False, ctx
                )
                return (
                    pseudo_loss,
                    adapter.ctx.grad.detach().clone(),
                    dict(adapter._last_pseudo_diagnostics),
                )
            finally:
                adapter.close()

        baseline_loss, baseline_grad, baseline_diag = run(0.0)
        anchored_loss, anchored_grad, anchored_diag = run(0.05)
        self.assertAlmostEqual(baseline_diag["weighted_amb_loss"], 0.0, places=7)
        self.assertAlmostEqual(
            baseline_loss,
            baseline_diag["bce_loss"] + baseline_diag["tversky_loss"],
            places=6,
        )
        self.assertEqual(float(baseline_grad.norm()), 0.0)
        self.assertGreater(anchored_diag["M_amb_anchor_count"], 0)
        self.assertGreater(anchored_diag["weighted_amb_loss"], 0.0)
        self.assertGreater(float(anchored_grad.norm()), 0.0)
        self.assertTrue(torch.isfinite(anchored_grad).all())
        for key in (
            "M_amb_anchor_count",
            "M_amb_anchor_fraction",
            "teacher_amb_anchor_mean_probability",
            "student_before_amb_anchor_mean_probability",
            "amb_loss",
            "weighted_amb_loss",
            "amb_weight",
        ):
            self.assertIn(key, anchored_diag)

    def test_decoder_masked_amb_replay_is_invariant_to_view_chunk_size(self):
        params = [
            {"scale": 1.0, "offset": 1.0},
            {"scale": 1.0, "offset": -1.0},
        ]
        patch = torch.zeros(1, 1, 1, 2)
        valid = torch.ones(1, 1, 2)
        template = TinyVoxTell()
        with torch.no_grad():
            template.project_text_embed.weight.fill_(1.0)
        state = copy.deepcopy(template.state_dict())

        def run(view_batch_size):
            model = TinyVoxTell()
            model.load_state_dict(state)
            adapter = VoxTellCMTTA(
                model,
                torch.zeros(1, 1, 2),
                "cpu",
                make_args(
                    pseudo_update_mode="decoder_masked",
                    num_aug_views=1,
                    view_batch_size=view_batch_size,
                    w_cac=0.0,
                    w_entropy=0.0,
                    amb_weight=0.05,
                ),
            )
            try:
                ctx = adapter.ctx.detach().clone()
                pseudo_loss, _ = adapter._backward_case_supervision(
                    [patch], [valid], params, 1, ctx, 1.0, ctx, False, ctx
                )
                return (
                    pseudo_loss,
                    adapter.ctx.grad.detach().clone(),
                    dict(adapter._last_pseudo_diagnostics),
                )
            finally:
                adapter.close()

        loss_one, grad_one, diag_one = run(1)
        loss_two, grad_two, diag_two = run(2)
        self.assertAlmostEqual(loss_one, loss_two, places=6)
        self.assertTrue(torch.allclose(grad_one, grad_two, atol=1e-6, rtol=1e-6))
        for key in ("amb_loss", "weighted_amb_loss", "pseudo_loss"):
            if key in diag_one:
                self.assertAlmostEqual(diag_one[key], diag_two[key], places=6)

    def test_original_pseudo_update_mode_does_not_use_amb_loss(self):
        model = TinyVoxTell()
        adapter = VoxTellCMTTA(
            model,
            torch.zeros(1, 1, 2),
            "cpu",
            make_args(
                pseudo_update_mode="original",
                amb_weight=0.05,
                num_aug_views=1,
                w_cac=0.0,
                w_entropy=0.0,
            ),
        )
        try:
            trace = adapter.adapt_case(
                [torch.zeros(1, 1, 1, 2)], [torch.ones(1, 1, 2)]
            )
            self.assertEqual(trace["optimizer_steps_for_case"], 1)
            self.assertNotIn("amb_loss", trace)
            self.assertNotIn("weighted_amb_loss", trace)
        finally:
            adapter.close()

    def test_gradient_conflict_diagnostics_is_disabled_by_default(self):
        adapter = VoxTellCMTTA(
            TinyVoxTell(), torch.zeros(1, 1, 2), "cpu", make_args()
        )
        try:
            self.assertFalse(adapter.gradient_conflict_diagnostics)
            trace = adapter.adapt_case([torch.zeros(1, 1, 1, 2)])
            self.assertNotIn("gradient_conflict_diagnostics", trace)
        finally:
            adapter.close()

    def test_gradient_conflict_diagnostics_does_not_change_update(self):
        torch.manual_seed(17)
        base_model = TinyVoxTell()
        patches = [
            torch.tensor([[[[-1.0, -0.2], [0.3, 0.8]], [[0.1, -0.4], [0.6, -0.7]]]])
        ]
        valid = [torch.ones(2, 2, 2)]
        params = [
            {"scale": 1.0, "offset": 0.0},
            {"scale": 1.1, "offset": -0.05},
        ]

        def build(enabled):
            return VoxTellCMTTA(
                copy.deepcopy(base_model),
                torch.zeros(1, 1, 2),
                "cpu",
                make_args(
                    num_aug_views=1,
                    w_entropy=0.1,
                    w_cac=0.25,
                    gradient_conflict_diagnostics=enabled,
                ),
            )

        def prepared(adapter):
            short = adapter.ctx.detach().clone()
            return {
                "patches": [patch.clone() for patch in patches],
                "valid_masks": [mask.clone() for mask in valid],
                "params": copy.deepcopy(params),
                "short_ctx": short,
                "current_quality": 0.0,
                "historical_quality": 0.0,
                "current_cac": 0.0,
                "historical_cac": 0.0,
                "weight_historical": 0.0,
                "long_ctx": short.clone(),
            }

        disabled = build(False)
        enabled = build(True)
        try:
            trace_disabled = disabled.adapt_case(
                patches, valid, prepared_case=prepared(disabled)
            )
            trace_enabled = enabled.adapt_case(
                patches, valid, prepared_case=prepared(enabled)
            )
            self.assertTrue(torch.equal(disabled.ctx.detach(), enabled.ctx.detach()))
            self.assertEqual(
                disabled.optimizer_step_count, enabled.optimizer_step_count
            )
            self.assertEqual(trace_disabled["selected_view"], trace_enabled["selected_view"])
            self.assertAlmostEqual(trace_disabled["loss"], trace_enabled["loss"], places=7)
            diagnostic = trace_enabled["gradient_conflict_diagnostics"]
            self.assertTrue(diagnostic["enabled"])
            self.assertIsNotNone(diagnostic["gradient_reconstruction_relative_error"])
            self.assertLess(
                diagnostic["gradient_reconstruction_relative_error"], 1e-6
            )
        finally:
            disabled.close()
            enabled.close()

    def test_gradient_conflict_cosine_and_zero_gradient_are_safe(self):
        same = torch.tensor([1.0, 2.0])
        opposite = -same
        zero = torch.zeros_like(same)
        same_cosine = VoxTellCMTTA._gradient_cosine(same, same)
        opposite_cosine = VoxTellCMTTA._gradient_cosine(same, opposite)
        self.assertAlmostEqual(same_cosine, 1.0)
        self.assertAlmostEqual(opposite_cosine, -1.0)
        self.assertFalse(same_cosine < 0.0)
        self.assertTrue(opposite_cosine < 0.0)
        self.assertIsNone(VoxTellCMTTA._gradient_cosine(same, zero))

        adapter = VoxTellCMTTA(
            TinyVoxTell(),
            torch.zeros(1, 1, 2),
            "cpu",
            make_args(
                num_aug_views=1,
                w_entropy=0.0,
                w_cac=0.0,
                gradient_conflict_diagnostics=True,
            ),
        )
        try:
            trace = adapter.adapt_case(
                [torch.tensor([[[[-1.0, 0.5], [0.4, -0.7]], [[0.1, 0.2], [-0.3, 0.8]]]])]
            )
            diagnostic = trace["gradient_conflict_diagnostics"]
            self.assertIsNone(diagnostic["cos_pseudo_entropy"])
            self.assertIsNone(diagnostic["cos_pseudo_cac"])
            self.assertIsNone(diagnostic["cos_entropy_cac"])
            self.assertTrue(np.isfinite(diagnostic["pseudo_gradient_norm"]))
        finally:
            adapter.close()

    def test_gradient_conflict_summary_reports_valid_and_conflicting_cases(self):
        rows = [
            {
                "gradient_conflict_diagnostics": {
                    "enabled": True,
                    "cos_pseudo_entropy": 0.5,
                    "cos_pseudo_cac": -0.2,
                    "cos_entropy_cac": 0.1,
                    "pseudo_entropy_conflict": False,
                    "pseudo_cac_conflict": True,
                }
            },
            {
                "gradient_conflict_diagnostics": {
                    "enabled": True,
                    "cos_pseudo_entropy": None,
                    "cos_pseudo_cac": 0.4,
                    "cos_entropy_cac": None,
                    "pseudo_entropy_conflict": None,
                    "pseudo_cac_conflict": False,
                }
            },
        ]
        summary = summarize_gradient_conflict_diagnostics(rows)
        self.assertEqual(summary["enabled_case_count"], 2)
        self.assertEqual(summary["valid_case_count"], 1)
        self.assertEqual(summary["conflict_case_count"]["pseudo_cac_conflict"], 1)
        self.assertAlmostEqual(summary["mean_cos_pseudo_cac"], 0.1)

    def test_tdc_softmax_view_weights_follow_formula_and_are_normalized(self):
        adapter = VoxTellCMTTA(
            TinyVoxTell(),
            torch.zeros(1, 1, 2),
            "cpu",
            make_args(
                view_selection_metric="tdc",
                pseudo_view_weighting="tdc_softmax",
                tdc_softmax_temperature=0.02,
            ),
        )
        try:
            adapter.last_view_selection = {"tdc": [0.1, 0.4, 0.2]}
            weights = adapter._pseudo_view_weights(3, 1)
            scores = torch.tensor([0.1, 0.4, 0.2])
            expected = torch.softmax((scores - scores.max()) / 0.02, dim=0)
            self.assertTrue(torch.allclose(weights, expected, atol=1e-7, rtol=1e-7))
            self.assertTrue(torch.isfinite(weights).all())
            self.assertTrue((weights >= 0).all())
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)
            self.assertGreater(float(weights[1]), float(weights[0]))
            self.assertGreater(float(weights[1]), float(weights[2]))
            adapter.last_view_selection = {"tdc": [0.2, 0.2, 0.2]}
            equal_weights = adapter._pseudo_view_weights(3, 0)
            self.assertTrue(torch.allclose(equal_weights, torch.full((3,), 1 / 3)))
        finally:
            adapter.close()

    def test_tdc_softmax_configuration_and_scores_fail_explicitly(self):
        with self.assertRaisesRegex(ValueError, "requires view_selection_metric"):
            VoxTellCMTTA(
                TinyVoxTell(),
                torch.zeros(1, 1, 2),
                "cpu",
                make_args(pseudo_view_weighting="tdc_softmax"),
            )
        with self.assertRaisesRegex(ValueError, "requires pseudo_update_mode"):
            VoxTellCMTTA(
                TinyVoxTell(),
                torch.zeros(1, 1, 2),
                "cpu",
                make_args(
                    view_selection_metric="tdc",
                    pseudo_view_weighting="tdc_softmax",
                    pseudo_update_mode="decoder_masked",
                ),
            )
        with self.assertRaisesRegex(ValueError, "temperature"):
            VoxTellCMTTA(
                TinyVoxTell(),
                torch.zeros(1, 1, 2),
                "cpu",
                make_args(
                    view_selection_metric="tdc",
                    pseudo_view_weighting="tdc_softmax",
                    tdc_softmax_temperature=0.0,
                ),
            )
        adapter = VoxTellCMTTA(
            TinyVoxTell(),
            torch.zeros(1, 1, 2),
            "cpu",
            make_args(view_selection_metric="tdc", pseudo_view_weighting="tdc_softmax"),
        )
        try:
            with self.assertRaisesRegex(ValueError, "requires TDC scores"):
                adapter._pseudo_view_weights(2, 0)
            adapter.last_view_selection = {"tdc": [0.1, float("nan")]}
            with self.assertRaisesRegex(ValueError, "finite TDC scores"):
                adapter._pseudo_view_weights(2, 0)
        finally:
            adapter.close()

    def test_case_soft_dice_weighting_is_view_level_not_patch_level(self):
        intersection = torch.tensor([2.0, 1.0])
        prediction_mass = torch.tensor([4.0, 2.0])
        pseudo_mass = torch.tensor([2.0, 4.0])
        weights = torch.tensor([0.8, 0.2])
        per_view = 1.0 - 2.0 * intersection / (
            prediction_mass + pseudo_mass + 1e-8
        )
        expected = (weights * per_view).sum()
        self.assertAlmostEqual(
            float(
                case_soft_dice_from_components(
                    intersection, prediction_mass, pseudo_mass, weights
                )
            ),
            float(expected),
            places=7,
        )
        self.assertAlmostEqual(
            float(case_soft_dice_from_components(
                intersection, prediction_mass, pseudo_mass
            )),
            float(per_view.mean()),
            places=7,
        )

    def test_tdc_softmax_case_is_chunk_invariant_and_keeps_teacher_selection(self):
        torch.manual_seed(23)
        base_model = TinyVoxTell()
        patches = [
            torch.tensor([[[[-1.0, -0.2], [0.3, 0.8]], [[0.1, -0.4], [0.6, -0.7]]]]),
            torch.tensor([[[[0.2, -0.5], [0.7, 0.1]], [[-0.6, 0.4], [0.9, -0.3]]]]),
        ]
        masks = [torch.ones(2, 2, 2) for _ in patches]
        params = [
            {"scale": 1.0, "offset": 0.0},
            {"scale": 1.1, "offset": -0.05},
            {"scale": 0.9, "offset": 0.04},
        ]

        def build(view_batch_size):
            return VoxTellCMTTA(
                copy.deepcopy(base_model),
                torch.zeros(1, 1, 2),
                "cpu",
                make_args(
                    view_selection_metric="tdc",
                    pseudo_update_mode="original",
                    pseudo_view_weighting="tdc_softmax",
                    tdc_softmax_temperature=0.02,
                    view_batch_size=view_batch_size,
                    num_aug_views=2,
                    w_cac=0.0,
                    w_entropy=0.0,
                    gradient_conflict_diagnostics=True,
                ),
            )

        def prepared(adapter):
            short = adapter.ctx.detach().clone()
            return {
                "patches": [patch.clone() for patch in patches],
                "valid_masks": [mask.clone() for mask in masks],
                "params": copy.deepcopy(params),
                "short_ctx": short,
                "current_quality": 0.0,
                "historical_quality": 0.0,
                "current_cac": 0.0,
                "historical_cac": 0.0,
                "weight_historical": 0.0,
                "long_ctx": short.clone(),
            }

        one = build(1)
        two = build(2)
        try:
            trace_one = one.adapt_case(patches, masks, prepared_case=prepared(one))
            trace_two = two.adapt_case(patches, masks, prepared_case=prepared(two))
            for trace in (trace_one, trace_two):
                self.assertEqual(trace["optimizer_steps_for_case"], 1)
                self.assertEqual(trace["selected_view"], trace["pseudo_source_view"])
                self.assertEqual(trace["pseudo_view_weighting"], "tdc_softmax")
                self.assertAlmostEqual(trace["pseudo_view_weight_sum"], 1.0, places=6)
                self.assertTrue(np.isfinite(trace["soft_dice"]))
                self.assertLess(
                    trace["gradient_conflict_diagnostics"][
                        "gradient_reconstruction_relative_error"
                    ],
                    1e-6,
                )
            self.assertEqual(trace_one["selected_view"], trace_two["selected_view"])
            self.assertTrue(
                np.allclose(
                    trace_one["pseudo_view_weights"],
                    trace_two["pseudo_view_weights"],
                    atol=1e-7,
                    rtol=1e-7,
                )
            )
            self.assertAlmostEqual(trace_one["soft_dice"], trace_two["soft_dice"], places=6)
            self.assertTrue(torch.allclose(one.ctx.detach(), two.ctx.detach(), atol=1e-6))
        finally:
            one.close()
            two.close()

    def test_sliding_teacher_labels_preserve_non_cubic_coordinates_and_padding(self):
        predictor = TinySlidingPredictor()
        data = torch.arange(3 * 5 * 7, dtype=torch.float32).reshape(1, 3, 5, 7)
        patch_size = (2, 3, 4)
        patches, valid_masks, locations, _original_shape = make_case_patches(
            data, patch_size
        )
        params = [
            {"scale": 1.0, "offset": 0.0},
            {"scale": 1.2, "offset": -0.3},
        ]
        labels, info = sliding_teacher_patch_labels(
            predictor,
            data,
            torch.zeros(1, 1, 2),
            params,
            1,
            patches,
            valid_masks,
            locations,
        )
        self.assertEqual(info["selected_view"], 1)
        self.assertEqual(info["padded_shape"], (4, 6, 8))
        self.assertEqual(info["alignment_max_abs_error"], 0.0)
        self.assertEqual(info["alignment_mean_abs_error"], 0.0)
        stitched, valid_crop = stitch_patch_probabilities(
            labels, valid_masks, locations, data.shape[-3:]
        )
        expected = torch.sigmoid(data[0] * 1.2 - 0.3)
        self.assertTrue(torch.equal(valid_crop, torch.ones(3, 5, 7, dtype=torch.bool)))
        self.assertTrue(torch.allclose(stitched, expected, atol=1e-7, rtol=1e-7))
        self.assertEqual(len(predictor.calls), 1)

    def test_original_sliding_teacher_is_used_after_selected_view_and_one_step(self):
        predictor = TinySlidingPredictor()
        adapter = VoxTellCMTTA(
            TinyVoxTell(),
            torch.zeros(1, 1, 2),
            "cpu",
            make_args(
                pseudo_teacher_inference="sliding",
                num_aug_views=1,
                w_cac=0.0,
                w_entropy=0.0,
            ),
        )
        data = torch.arange(2 * 2 * 4, dtype=torch.float32).reshape(1, 2, 2, 4) / 10.0
        patch_size = (2, 2, 2)
        patches, valid_masks, locations, _ = make_case_patches(data, patch_size)
        params = [
            {"scale": 1.0, "offset": 0.0},
            {"scale": 1.1, "offset": -0.05},
        ]
        short = adapter.ctx.detach().clone()
        prepared = {
            "patches": patches,
            "valid_masks": valid_masks,
            "params": params,
            "short_ctx": short,
            "current_quality": 0.0,
            "historical_quality": 0.0,
            "current_cac": 0.0,
            "historical_cac": 0.0,
            "weight_historical": 0.0,
            "long_ctx": short.clone(),
        }
        calls = []

        def provider(selected_view, long_ctx):
            calls.append((selected_view, long_ctx.detach().clone()))
            labels, _ = sliding_teacher_patch_labels(
                predictor,
                data,
                adapter._encode_ctx(long_ctx).detach(),
                params,
                selected_view,
                patches,
                valid_masks,
                locations,
            )
            return labels

        try:
            trace = adapter.adapt_case(
                patches,
                valid_masks,
                prepared_case=prepared,
                teacher_pseudo_provider=provider,
            )
            self.assertEqual(trace["pseudo_teacher_inference"], "sliding")
            self.assertEqual(trace["selected_view"], calls[0][0])
            self.assertEqual(trace["pseudo_source_view"], trace["selected_view"])
            self.assertEqual(trace["optimizer_steps_for_case"], 1)
            self.assertEqual(adapter.optimizer_step_count, 1)
            self.assertEqual(len(predictor.calls), 1)
        finally:
            adapter.close()

    def test_sliding_teacher_option_is_rejected_for_decoder_masked(self):
        with self.assertRaisesRegex(ValueError, "only with pseudo_update_mode='original'"):
            VoxTellCMTTA(
                TinyVoxTell(),
                torch.zeros(1, 1, 2),
                "cpu",
                make_args(
                    pseudo_update_mode="decoder_masked",
                    pseudo_teacher_inference="sliding",
                ),
            )

    def test_masked_balanced_bce_gives_equal_fg_bg_weight(self):
        fg_sum = torch.tensor(4.0, requires_grad=True)
        bg_sum = torch.tensor(12.0, requires_grad=True)
        result = masked_balanced_bce_from_components(fg_sum, bg_sum, 2.0, 6.0)
        self.assertAlmostEqual(float(result.detach()), 2.0, places=6)
        result.backward()
        self.assertAlmostEqual(float(fg_sum.grad), 0.25, places=6)
        self.assertAlmostEqual(float(bg_sum.grad), 1.0 / 12.0, places=6)

    def test_masked_tversky_is_per_view_case_mean_not_merged_case_ratio(self):
        tp = torch.tensor([8.0, 1.0])
        fp = torch.tensor([0.0, 9.0])
        fn = torch.tensor([2.0, 1.0])
        mean_loss, per_view = masked_tversky_loss_from_components(
            tp, fp, fn, alpha=0.3, beta=0.7,
        )
        expected_per_view = 1.0 - tp / (tp + 0.3 * fp + 0.7 * fn + 1e-8)
        merged = 1.0 - tp.sum() / (
            tp.sum() + 0.3 * fp.sum() + 0.7 * fn.sum() + 1e-8
        )
        self.assertTrue(torch.allclose(per_view, expected_per_view))
        self.assertAlmostEqual(float(mean_loss), float(expected_per_view.mean()), places=6)
        self.assertNotAlmostEqual(float(mean_loss), float(merged), places=4)

    def test_empty_tversky_has_zero_loss_and_zero_gradient(self):
        tp = torch.zeros(2, requires_grad=True)
        fp = torch.zeros(2, requires_grad=True)
        fn = torch.zeros(2, requires_grad=True)
        loss, per_view = masked_tversky_loss_from_components(
            tp, fp, fn, valid_mass=torch.zeros(2)
        )
        self.assertEqual(float(loss.detach()), 0.0)
        self.assertTrue(torch.equal(per_view, torch.zeros(2)))
        loss.backward()
        self.assertTrue(torch.equal(tp.grad, torch.zeros_like(tp)))
        self.assertTrue(torch.equal(fp.grad, torch.zeros_like(fp)))
        self.assertTrue(torch.equal(fn.grad, torch.zeros_like(fn)))

    def test_ambiguous_region_has_no_bce_or_tversky_gradient(self):
        logits = torch.zeros(1, 1, 2, 2, 2, requires_grad=True)
        probability = torch.sigmoid(logits)
        target = torch.full_like(probability, 0.7)
        ambiguous = torch.ones_like(probability)
        known = torch.zeros_like(probability)
        bce = (F.binary_cross_entropy_with_logits(logits, target, reduction="none") * known).sum()
        tp = (probability * target * known).sum().view(1)
        fp = (probability * (1.0 - target) * known).sum().view(1)
        fn = ((1.0 - probability) * target * known).sum().view(1)
        tversky, _ = masked_tversky_loss_from_components(
            tp, fp, fn, valid_mass=known.sum().view(1)
        )
        (bce + tversky + ambiguous.sum() * 0.0).backward()
        self.assertTrue(torch.equal(logits.grad, torch.zeros_like(logits)))

    def test_decoder_masked_pseudo_update_has_safe_empty_regions_and_one_step(self):
        adapter = VoxTellCMTTA(
            TinyVoxTell(),
            torch.zeros(1, 1, 2),
            "cpu",
            make_args(
                pseudo_update_mode="decoder_masked",
                num_aug_views=1,
                view_batch_size=1,
            ),
        )
        try:
            patch = torch.zeros(1, 2, 2, 2)
            trace = adapter.adapt_case([patch], [torch.ones(2, 2, 2)])
            self.assertEqual(trace["optimizer_steps_for_case"], 1)
            self.assertEqual(trace["pseudo_update_mode"], "decoder_masked")
            self.assertEqual(trace["M_fg_count"], 0)
            self.assertEqual(trace["M_bg_count"], 0)
            self.assertEqual(trace["M_amb_count"], 8)
            self.assertTrue(np.isfinite(trace["pseudo_bce"]))
            self.assertTrue(np.isfinite(trace["pseudo_tversky"]))
            self.assertTrue(np.isfinite(trace["loss"]))
        finally:
            adapter.close()

    def test_decoder_masked_loss_uses_soft_d5_target_on_known_region(self):
        model = TinyVoxTell()
        with torch.no_grad():
            model.project_text_embed.weight.fill_(1.0)
        adapter = VoxTellCMTTA(
            model,
            torch.ones(1, 1, 2),
            "cpu",
            make_args(
                pseudo_update_mode="decoder_masked",
                num_aug_views=1,
                view_batch_size=1,
                w_cac=0.0,
                w_entropy=0.0,
            ),
        )
        try:
            before = adapter.ctx_delta.detach().clone()
            original_diagnostics = adapter._selected_prompt_region_diagnostics
            diagnostic_contexts = {}

            def capture_diagnostics(
                patches, valid_masks, params, selected_view, ctx, cache,
                autocast_enabled, prefix,
            ):
                diagnostic_contexts[prefix] = ctx.detach().clone()
                return original_diagnostics(
                    patches, valid_masks, params, selected_view, ctx, cache,
                    autocast_enabled, prefix,
                )

            adapter._selected_prompt_region_diagnostics = capture_diagnostics
            trace = adapter.adapt_case(
                [torch.zeros(1, 2, 2, 2)], [torch.ones(2, 2, 2)]
            )
            self.assertEqual(trace["M_fg_count"], 8)
            self.assertEqual(trace["M_bg_count"], 0)
            self.assertEqual(trace["M_amb_count"], 0)
            self.assertGreater(trace["bce_loss"], 0.0)
            self.assertGreater(trace["tversky_loss"], 0.0)
            self.assertTrue(torch.isfinite(adapter.ctx_delta).all())
            self.assertFalse(torch.equal(before, adapter.ctx_delta.detach()))
            self.assertTrue(
                trace["pseudo_loss_type"].startswith("masked_balanced_bce_tversky")
            )
            self.assertAlmostEqual(
                trace["pseudo_loss"],
                trace["bce_loss"]
                + trace["tversky_loss"]
                + trace["weighted_amb_loss"],
                places=6,
            )
            self.assertEqual(trace["amb_weight"], 0.05)
            self.assertEqual(trace["M_amb_anchor_count"], 0)
            self.assertIn("entropy_loss", trace)
            self.assertIn("cac_loss", trace)
            self.assertIn("total_loss", trace)
            self.assertTrue(torch.equal(diagnostic_contexts["student_before"], before))
            self.assertTrue(
                torch.equal(
                    diagnostic_contexts["student_after"], adapter.ctx_delta.detach()
                )
            )
            self.assertIn("student_before_foreground_volume", trace)
            self.assertIn("student_after_foreground_volume", trace)
        finally:
            adapter.close()

    def test_decoder_masked_multiview_replay_matches_view_batch_size_one(self):
        template = TinyVoxTell()
        with torch.no_grad():
            template.project_text_embed.weight.fill_(1.0)
        template_state = copy.deepcopy(template.state_dict())

        def make_adapter(view_batch_size):
            model = TinyVoxTell()
            model.load_state_dict(template_state)
            return VoxTellCMTTA(
                model,
                torch.ones(1, 1, 2),
                "cpu",
                make_args(
                    pseudo_update_mode="decoder_masked",
                    num_aug_views=2,
                    view_batch_size=view_batch_size,
                    w_cac=0.0,
                    w_entropy=0.0,
                ),
            )

        patches = [torch.zeros(1, 2, 2, 2)]
        valid_masks = [torch.ones(2, 2, 2)]
        batch_one = make_adapter(1)
        batch_two = make_adapter(2)
        try:
            # Prepare once so both runs use identical sampled intensity views,
            # selected-view inputs, short/long contexts, and quality values.
            prepared = batch_one.prepare_case(patches, valid_masks)
            prepared_one = copy.deepcopy(prepared)
            prepared_two = copy.deepcopy(prepared)
            ctx_before_one = batch_one.ctx_delta.detach().clone()
            ctx_before_two = batch_two.ctx_delta.detach().clone()
            trace_one = batch_one.adapt_case(
                patches, valid_masks, prepared_case=prepared_one
            )
            trace_two = batch_two.adapt_case(
                patches, valid_masks, prepared_case=prepared_two
            )
            self.assertEqual(trace_one["optimizer_steps_for_case"], 1)
            self.assertEqual(trace_two["optimizer_steps_for_case"], 1)
            for adapter, ctx_before in (
                (batch_one, ctx_before_one),
                (batch_two, ctx_before_two),
            ):
                self.assertIsNotNone(adapter.ctx_delta.grad)
                self.assertTrue(torch.isfinite(adapter.ctx_delta.grad).all())
                self.assertGreater(float(adapter.ctx_delta.grad.norm()), 0.0)
                self.assertFalse(torch.equal(ctx_before, adapter.ctx_delta.detach()))
            for key in ("pseudo_loss", "bce_loss", "tversky_loss"):
                self.assertTrue(np.isfinite(trace_one[key]))
                self.assertTrue(np.isfinite(trace_two[key]))
                self.assertAlmostEqual(trace_one[key], trace_two[key], places=6)
            self.assertTrue(
                torch.allclose(
                    batch_one.ctx_delta.detach(),
                    batch_two.ctx_delta.detach(),
                    atol=1e-6,
                    rtol=1e-6,
                )
            )
        finally:
            batch_one.close()
            batch_two.close()

    def test_prediction_roundtrips_with_voxtell_reader_writer_geometry(self):
        import nibabel as nib
        from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient

        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            image_path = temp_dir / "image.nii.gz"
            label_path = temp_dir / "label.nii.gz"
            output_path = temp_dir / "prediction.nii.gz"
            source_data = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
            affine = np.array(
                [[-1.0, 0.0, 0.0, 10.0], [0.0, 2.0, 0.0, 20.0], [0.0, 0.0, 3.0, 30.0], [0, 0, 0, 1]],
                dtype=np.float64,
            )
            nib.save(nib.Nifti1Image(source_data, affine), str(image_path))
            nib.save(nib.Nifti1Image((source_data > 0).astype(np.uint8), affine), str(label_path))

            model_space_prediction = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
            save_prediction_nifti(model_space_prediction, image_path, output_path)

            saved = nib.load(str(output_path))
            source = nib.load(str(image_path))
            self.assertEqual(tuple(saved.shape), tuple(source.shape))
            self.assertTrue(np.allclose(saved.affine, source.affine))
            check_prediction_nifti_geometry(output_path, image_path, label_path)
            restored, _ = NibabelIOWithReorient().read_images([str(output_path)])
            self.assertTrue(np.array_equal(restored[0], model_space_prediction))

    def test_only_ctx_and_qwen_are_frozen(self):
        model = TinyVoxTell(text_dim=4)
        qwen = TinyQwen()
        adapter = VoxTellCMTTA(
            model,
            None,
            "cpu",
            make_args(),
            qwen,
            TinyQwenTokenizer(),
            text_prompt="liver",
            n_ctx=2,
            formatted_text_prompt="prefix liver",
        )
        try:
            self.assertEqual(adapter.optimizer_parameters, [adapter.ctx_delta])
            self.assertTrue(adapter.ctx_delta.requires_grad)
            self.assertEqual(tuple(adapter.ctx_delta.shape), (1, 4))
            self.assertEqual(tuple(adapter._encode_ctx(adapter.ctx_delta).shape), (1, 1, 4))
            self.assertEqual(adapter._liver_token_indices.tolist(), [2])
            self.assertTrue(torch.equal(adapter._fixed_attention_mask, torch.tensor([[1, 1, 1, 1]], dtype=torch.bool)))
            self.assertTrue(torch.equal(adapter.ctx_delta, torch.zeros_like(adapter.ctx_delta)))
            self.assertTrue(
                torch.equal(
                    adapter._fixed_token_embeddings[0, 2],
                    qwen.token_embedding.weight[11],
                )
            )
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertTrue(all(not p.requires_grad for p in adapter.qwen_text_encoder.parameters()))
            self.assertEqual(adapter.ctx_delta.dtype, torch.float32)
            self.assertIsInstance(adapter.optimizer, torch.optim.Adam)
            self.assertEqual(adapter.optimizer.param_groups[0]["lr"], 0.05)
            self.assertEqual(adapter.optimizer.param_groups[0]["weight_decay"], 0.0)

            adapter._encode_ctx(adapter.ctx_delta).sum().backward()
            self.assertIsNotNone(adapter.ctx_delta.grad)
            self.assertGreater(float(adapter.ctx_delta.grad.norm()), 0.0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(all(p.grad is None for p in qwen.parameters()))
            adapter.optimizer.zero_grad(set_to_none=True)
            ctx_before = adapter.ctx_delta.detach().clone()
            trace = adapter.adapt_case([torch.zeros(1, 2, 2, 2)])
            self.assertEqual(trace["optimizer_steps_for_case"], 1)
            self.assertFalse(torch.equal(ctx_before, adapter.ctx_delta.detach()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(all(p.grad is None for p in qwen.parameters()))
        finally:
            adapter.close()

    def test_amp_skipped_step_does_not_require_ctx_value_change(self):
        class SkippedScaler:
            def __init__(self):
                self.scale_value = 8.0

            def scale(self, objective):
                return objective

            def unscale_(self, optimizer):
                del optimizer

            def step(self, optimizer):
                del optimizer  # emulate GradScaler skipping an overflowed step

            def update(self):
                self.scale_value /= 2.0

            def get_scale(self):
                return self.scale_value

        adapter = VoxTellCMTTA(
            TinyVoxTell(), torch.zeros(1, 1, 2), "cpu", make_args()
        )
        adapter.scaler = SkippedScaler()
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                trace = adapter.adapt_case([torch.zeros(1, 2, 2, 2)])
            self.assertTrue(trace["optimizer_step_skipped"])
            self.assertEqual(trace["optimizer_steps_for_case"], 1)
            self.assertTrue(any("skipped" in str(item.message) for item in caught))
        finally:
            adapter.close()

    def test_zero_delta_matches_native_text_embedding_and_logits(self):
        model = TinyVoxTell(text_dim=4)
        qwen = TinyQwen()
        tokenizer = TinyQwenTokenizer()
        adapter = VoxTellCMTTA(
            model,
            None,
            "cpu",
            make_args(),
            qwen,
            tokenizer,
            text_prompt="liver",
            formatted_text_prompt="prefix liver",
        )
        try:
            tokenized = tokenizer(
                ["prefix liver"],
                padding=True,
                truncation=True,
                max_length=8192,
                return_tensors="pt",
            )
            with torch.no_grad():
                native = qwen(
                    input_ids=tokenized["input_ids"],
                    attention_mask=tokenized["attention_mask"].bool(),
                ).last_hidden_state[:, -1].unsqueeze(1)
            adapted = adapter._encode_ctx(adapter.ctx_delta)
            self.assertTrue(torch.allclose(adapted, native, atol=1e-7))
            image = torch.randn(1, 1, 2, 2, 2)
            native_logits = model(image, adapter._text_input(native, 1))
            adapted_logits = adapter._forward(image, adapter.ctx_delta)
            self.assertTrue(torch.allclose(adapted_logits, native_logits, atol=1e-7))
            self.assertEqual(tuple(qwen.last_inputs_embeds.shape), (1, 4, 4))
            self.assertEqual(tuple(adapter._fixed_attention_mask.shape), (1, 4))
            self.assertEqual(tuple(adapter._liver_token_indices.tolist()), (2,))
        finally:
            adapter.close()

    def test_multi_token_liver_delta_replaces_all_query_tokens_without_insertion(self):
        qwen = TinyQwen()
        adapter = VoxTellCMTTA(
            TinyVoxTell(text_dim=4),
            None,
            "cpu",
            make_args(),
            qwen,
            MultiTokenQwenTokenizer(),
            text_prompt="liver",
            n_ctx=1,  # Deliberately ignored for Qwen; tokenizer determines the count.
            formatted_text_prompt="prefix liver",
        )
        try:
            self.assertEqual(tuple(adapter.ctx_delta.shape), (2, 4))
            self.assertEqual(adapter._liver_token_indices.tolist(), [2, 3])
            self.assertEqual(tuple(adapter._fixed_attention_mask.shape), (1, 5))
            original = adapter._fixed_token_embeddings.clone()
            zero = adapter._encode_ctx(adapter.ctx_delta)
            self.assertTrue(torch.equal(qwen.last_inputs_embeds, original))
            adapter.ctx_delta.data.copy_(torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0]]))
            adapter._encode_ctx(adapter.ctx_delta)
            updated = qwen.last_inputs_embeds.detach()
            self.assertEqual(tuple(updated.shape), tuple(original.shape))
            self.assertTrue(torch.equal(updated[0, 0], original[0, 0]))
            self.assertTrue(torch.equal(updated[0, 1], original[0, 1]))
            self.assertTrue(torch.allclose(updated[0, 2], original[0, 2] + adapter.ctx_delta[0]))
            self.assertTrue(torch.allclose(updated[0, 3], original[0, 3] + adapter.ctx_delta[1]))
            self.assertTrue(torch.equal(updated[0, 4], original[0, 4]))
            self.assertTrue(torch.allclose(zero, adapter._encode_ctx(torch.zeros_like(adapter.ctx_delta)), atol=1e-7))
        finally:
            adapter.close()

    def test_legacy_random_ctx_checkpoint_is_rejected(self):
        adapter = VoxTellCMTTA(TinyVoxTell(), torch.zeros(1, 1, 2), "cpu", make_args())
        try:
            with self.assertRaisesRegex(ValueError, "Legacy random-ctx"):
                adapter.load_state_dict({"ctx": torch.zeros(1, 2)})
        finally:
            adapter.close()

    def test_soft_dice_is_averaged_over_all_views(self):
        pseudo = torch.ones(1, 1, 2, 2, 2)
        predictions = torch.stack(
            [
                torch.ones(1, 2, 2, 2),
                torch.zeros(1, 2, 2, 2),
                torch.full((1, 2, 2, 2), 0.5),
            ]
        )
        result = soft_dice_loss(predictions, pseudo)
        expected = torch.tensor((0.0 + 1.0 + (1.0 - 2.0 / 3.0)) / 3.0)
        self.assertTrue(torch.allclose(result, expected, atol=1e-6))

    def test_cac_selection_uses_combined_cac_entropy_rank_and_official_entropy(self):
        probabilities = torch.full((3, 1, 2, 2, 2), 0.5)
        selected, selected_indices = select_cac_view(
            torch.tensor([0.1, 0.9, 0.2]), probabilities.squeeze(1), 0.1
        )
        self.assertEqual(selected, 1)
        self.assertEqual(selected_indices.tolist(), [1])
        self.assertAlmostEqual(float(avg_entropy(probabilities[0])), 0.346573, places=5)

        # The CAC winner is view 0, but official rank fusion selects view 1:
        # CAC ranks are [0,1,2], entropy ranks are [2,0,1].
        rank_probabilities = torch.tensor(
            [[[[[0.5]]]], [[[[0.99]]]], [[[[0.2]]]]]
        )
        selected, selected_indices = select_cac_view(
            torch.tensor([0.9, 0.8, 0.1]), rank_probabilities, 0.1
        )
        self.assertEqual(selected, 1)
        self.assertEqual(selected_indices.tolist(), [1])

        with self.assertRaisesRegex(ValueError, "exactly one selected view"):
            select_cac_view(
                torch.tensor([0.1, 0.2, 0.3]),
                torch.full((3, 1, 1, 1, 1), 0.5),
                0.7,
            )

    def test_tdc_empty_pair_rules_and_case_level_reduction(self):
        intersection = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        count1 = torch.tensor([[2.0, 2.0, 0.0, 0.0, 0.0, 0.0]])
        count2 = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        tdc, pair_dice, pair_valid = tdc_from_components(
            intersection, count1, count2, torch.tensor([True])
        )
        self.assertTrue(torch.allclose(pair_dice[0, :2], torch.tensor([1.0, 0.0])))
        self.assertTrue(torch.equal(pair_valid[0, :2], torch.tensor([True, True])))
        self.assertTrue(torch.allclose(tdc, torch.tensor([0.5])))
        self.assertFalse(pair_valid[0, 2:].any())

    def test_tdc_fixed_hwd_conversion_excludes_single_axis_padding(self):
        # Input/valid-mask and decoder order are both (D,H,W)=(3,3,4),
        # with only the last D plane padded.
        volume = torch.zeros(1, 2, 3, 4)
        patches, valid_masks, locations, data_shape = make_case_patches(
            volume, (3, 3, 4)
        )
        self.assertEqual(tuple(valid_masks[0].shape), (3, 3, 4))
        self.assertEqual(tuple(data_shape), (2, 3, 4))
        self.assertEqual(locations, [(0, 0, 0)])

        decoder_mask = torch.full((1, 1, 3, 3, 4), -20.0)
        decoder_mask[:, :, -1] = 20.0  # only the padded D5 D-plane is foreground
        result = tdc_patch_components(
            [decoder_mask, decoder_mask, decoder_mask, decoder_mask],
            valid_masks[0].unsqueeze(0),
        )
        self.assertTrue(torch.equal(result["count1"], torch.zeros_like(result["count1"])))
        self.assertFalse(result["finite"].logical_not().any())
        tdc, _pair_dice, pair_valid = tdc_from_components(
            result["intersection"],
            result["count1"],
            result["count2"],
            result["finite"],
        )
        self.assertTrue(torch.allclose(tdc, torch.zeros_like(tdc)))
        self.assertFalse(pair_valid.any())

    def test_tdc_selection_uses_decoder_outputs_and_case_statistics(self):
        adapter = VoxTellCMTTA(
            TinyVoxTell(),
            torch.ones(1, 1, 2),
            "cpu",
            make_args(view_selection_metric="tdc", num_aug_views=2, view_batch_size=1),
        )
        try:
            patch = torch.zeros(1, 2, 2, 2)
            valid = torch.ones(2, 2, 2)
            params = [
                {"scale": 1.0, "offset": 0.0},
                {"scale": 0.9, "offset": 0.1},
                {"scale": 1.1, "offset": -0.1},
            ]
            selected, cac_scores = adapter._select_case_view(
                [patch, patch + 0.1], params, adapter.ctx.detach(), [valid, valid]
            )
            details = adapter.last_view_selection
            self.assertEqual(cac_scores.shape, (3,))
            self.assertEqual(details["selection_metric"], "tdc")
            self.assertEqual(len(details["tdc"]), 3)
            self.assertEqual(len(details["combined_rank"]), 3)
            self.assertEqual(details["selected_view"], selected)
        finally:
            adapter.close()

    def test_quality_only_rank_does_not_add_entropy(self):
        patch = torch.zeros(1, 2, 2, 2)
        valid = torch.ones(2, 2, 2)
        params = [
            {"scale": 1.0, "offset": 0.0},
            {"scale": 0.9, "offset": 0.1},
            {"scale": 1.1, "offset": -0.1},
        ]
        for metric, rank_key in (("cac", "cac_rank"), ("tdc", "tdc_rank")):
            adapter = VoxTellCMTTA(
                TinyVoxTell(),
                torch.ones(1, 1, 2),
                "cpu",
                make_args(
                    view_selection_metric=metric,
                    use_entropy_rank=False,
                    num_aug_views=2,
                    view_batch_size=1,
                ),
            )
            try:
                adapter._select_case_view(
                    [patch], params, adapter.ctx.detach(), [valid]
                )
                details = adapter.last_view_selection
                self.assertTrue(
                    torch.equal(
                        torch.tensor(details["combined_rank"]),
                        torch.tensor(details[rank_key]),
                    )
                )
            finally:
                adapter.close()

    def test_selector_only_view_selection_does_not_update_prompt_or_optimizer(self):
        adapter = VoxTellCMTTA(
            TinyVoxTell(),
            torch.ones(1, 1, 2),
            "cpu",
            make_args(num_aug_views=2, view_batch_size=1),
        )
        try:
            patch = torch.zeros(1, 2, 2, 2)
            valid = torch.ones(2, 2, 2)
            params = [
                {"scale": 1.0, "offset": 0.0},
                {"scale": 0.9, "offset": 0.1},
                {"scale": 1.1, "offset": -0.1},
            ]
            before = adapter.ctx_delta.detach().clone()
            adapter._select_case_view(
                [patch], params, adapter.ctx_delta.detach(), [valid], collect_tdc=True
            )
            self.assertTrue(torch.equal(adapter.ctx_delta.detach(), before))
            self.assertEqual(adapter.optimizer_step_count, 0)
            self.assertIsNone(adapter.ctx_delta.grad)
        finally:
            adapter.close()

    def test_selector_only_report_has_four_shared_top1_selectors(self):
        view_metrics = [
            {"view": 0, "GT_Dice_before_adaptation": 0.4},
            {"view": 1, "GT_Dice_before_adaptation": 0.9},
            {"view": 2, "GT_Dice_before_adaptation": 0.6},
        ]
        selection = {
            "cac": [0.8, 0.2, 0.5],
            "tdc": [0.1, 0.9, 0.4],
            "cac_rank": [0.0, 2.0, 1.0],
            "tdc_rank": [0.0, 2.0, 1.0],
            "cac_entropy_rank": [1.0, 0.0, 2.0],
            "tdc_entropy_rank": [1.0, 0.0, 2.0],
            "cac_combined_rank": [1.0, 2.0, 3.0],
            "tdc_combined_rank": [1.0, 2.0, 3.0],
            "selected_view": 0,
        }
        report = selector_only_case_report(view_metrics, selection)
        self.assertEqual(
            set(report["selectors"]),
            {"cac_only", "tdc_only", "cac_entropy", "tdc_entropy"},
        )
        for result in report["selectors"].values():
            self.assertEqual(result["oracle_best_view"], 1)
            self.assertIn("regret", result)
        self.assertEqual(report["selectors"]["cac_only"]["selected_view"], 0)
        self.assertEqual(report["selectors"]["tdc_only"]["selected_view"], 0)
        self.assertEqual(report["selectors"]["cac_entropy"]["selected_view"], 0)
        self.assertEqual(report["selectors"]["tdc_entropy"]["selected_view"], 0)
        summary = summarize_selector_only([report])
        self.assertEqual(set(summary), set(report["selectors"]))

    def test_selector_only_combined_rank_beats_entropy_only(self):
        view_metrics = [
            {"view": 0, "GT_Dice_before_adaptation": 0.6},
            {"view": 1, "GT_Dice_before_adaptation": 0.9},
            {"view": 2, "GT_Dice_before_adaptation": 0.5},
        ]
        selection = {
            "tdc": [0.1, 0.2, 0.3],
            "cac_rank": [0.0, 2.0, 1.0],
            "tdc_rank": [0.0, 2.0, 1.0],
            "cac_entropy_rank": [1.0, 0.0, 2.0],
            "tdc_entropy_rank": [1.0, 0.0, 2.0],
            "cac_combined_rank": [1.0, 2.0, 3.0],
            "tdc_combined_rank": [1.0, 2.0, 3.0],
        }
        report = selector_only_case_report(view_metrics, selection)
        # Entropy alone would choose view 1; quality+entropy chooses view 0.
        self.assertEqual(report["selectors"]["cac_entropy"]["selected_view"], 0)
        self.assertEqual(report["selectors"]["tdc_entropy"]["selected_view"], 0)

    def test_cac_matches_source_similarity_map_definition(self):
        vision = torch.tensor(
            [
                [[2.0, 0.0]],
                [[0.0, 2.0]],
                [[1.0, -1.0]],
                [[1.0, -1.0]],
            ]
        )
        text = torch.tensor([[[1.0, 1.0]]])
        logits = torch.tensor([[[[[20.0, 20.0], [-20.0, -20.0]]]]])
        result = cac_from_features(vision, text, logits)
        expected = torch.tensor([(2.0 ** -0.5) - 0.0])
        self.assertTrue(torch.allclose(result, expected, atol=1e-5))

    def test_cac_uses_hard_half_probability_masks_for_similarity_map(self):
        vision = torch.tensor(
            [
                [[1.0, 0.0]],
                [[0.0, 1.0]],
                [[1.0, 1.0]],
                [[2.0, 0.0]],
            ]
        )
        probabilities = torch.tensor([0.9, 0.6, 0.4, 0.1]).reshape(1, 1, 1, 4)
        logits = torch.logit(probabilities).unsqueeze(1)
        text = torch.tensor([[[1.0, 0.0]]])
        components = cac_components_from_features(vision, text, logits)
        self.assertTrue(
            torch.equal(components["foreground_mass"], torch.tensor([2.0]))
        )
        self.assertTrue(
            torch.equal(components["background_mass"], torch.tensor([2.0]))
        )
        self.assertTrue(
            torch.allclose(components["foreground_sum"], torch.tensor([1.0]))
        )
        self.assertTrue(
            torch.allclose(
                components["background_sum"], torch.tensor([1.0 + 2.0 ** -0.5])
            )
        )

        all_foreground = cac_components_from_features(
            vision, text, torch.full_like(logits, 20.0)
        )
        all_background = cac_components_from_features(
            vision, text, torch.full_like(logits, -20.0)
        )
        self.assertTrue(torch.isfinite(cac_from_components(
            all_foreground["foreground_sum"],
            all_foreground["foreground_mass"],
            all_foreground["background_sum"],
            all_foreground["background_mass"],
        )).all())
        self.assertTrue(torch.isfinite(cac_from_components(
            all_background["foreground_sum"],
            all_background["foreground_mass"],
            all_background["background_sum"],
            all_background["background_mass"],
        )).all())

    def test_case_cac_uses_global_sums_not_patch_cac_mean(self):
        first = cac_from_components(
            torch.tensor([10.0]), torch.tensor([10.0]),
            torch.tensor([0.0]), torch.tensor([10.0]),
        )
        second = cac_from_components(
            torch.tensor([0.0]), torch.tensor([1.0]),
            torch.tensor([0.0]), torch.tensor([1.0]),
        )
        global_score = cac_from_components(
            torch.tensor([10.0]), torch.tensor([11.0]),
            torch.tensor([0.0]), torch.tensor([11.0]),
        )
        self.assertFalse(torch.allclose(global_score, (first + second) / 2.0))

    def test_case_cac_is_invariant_to_patch_partition_and_empty_padding(self):
        volume = torch.tensor(
            [[[[0.0, 0.2], [0.4, 0.6]], [[0.8, 1.0], [0.3, 0.1]]]]
        )
        model = TinyVoxTell()
        with torch.no_grad():
            model.project_text_embed.weight.copy_(torch.eye(2))
        adapter = VoxTellCMTTA(
            model, torch.zeros(1, 1, 2), "cpu", make_args()
        )
        try:
            full_mask = torch.ones(2, 2, 2)
            full_score = adapter._case_cac(
                adapter.ctx.detach(), [volume], [full_mask]
            )
            split_patches = [volume[:, :1], volume[:, 1:]]
            split_masks = [full_mask[:1], full_mask[1:]]
            split_score = adapter._case_cac(
                adapter.ctx.detach(), split_patches, split_masks
            )
            padded_score = adapter._case_cac(
                adapter.ctx.detach(),
                split_patches + [torch.full_like(volume, 99.0)],
                split_masks + [torch.zeros_like(full_mask)],
            )
        finally:
            adapter.close()
        self.assertAlmostEqual(full_score, split_score, places=6)
        self.assertAlmostEqual(full_score, padded_score, places=6)

    def test_padding_mask_excludes_padding_and_augmentation_keeps_it_zero(self):
        volume = torch.ones(1, 3, 4, 5)
        padded, valid, _ = pad_to_patch_grid(volume, (2, 3, 3))
        self.assertEqual(tuple(padded.shape), (1, 4, 6, 6))
        self.assertEqual(tuple(valid.shape), (1, 4, 6, 6))
        self.assertEqual(float(valid[:, :3, :4, :5].sum()), 3 * 4 * 5)
        patches, valid_masks, _locations, _original = make_case_patches(volume, (2, 3, 3))
        self.assertEqual(len(patches), len(valid_masks))
        augmented = VoxTellCMTTA._make_views(
            patches[-1], [{"scale": 1.0, "offset": 0.0}, {"scale": 1.0, "offset": 10.0}],
            valid_masks[-1],
        )
        self.assertTrue(torch.equal(augmented * (1.0 - valid_masks[-1]), torch.zeros_like(augmented)))

    def test_padding_is_excluded_from_cac_and_soft_dice(self):
        vision = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]]])
        text = torch.tensor([[[1.0, 0.0]]])
        logits_a = torch.tensor([[[[[20.0, -20.0]]]]])
        logits_b = torch.tensor([[[[[20.0, 20.0]]]]])
        valid = torch.tensor([[[[1.0, 0.0]]]])
        cac_a = cac_from_features(vision, text, logits_a, valid_mask=valid)
        cac_b = cac_from_features(vision, text, logits_b, valid_mask=valid)
        self.assertTrue(torch.allclose(cac_a, cac_b, atol=1e-5))

        prediction_a = torch.tensor([[[[[0.8, 0.0]]]]])
        prediction_b = torch.tensor([[[[[0.8, 1.0]]]]])
        pseudo_a = torch.tensor([[[[[0.8, 0.0]]]]])
        pseudo_b = torch.tensor([[[[[0.8, 0.2]]]]])
        dice_a = soft_dice_loss(prediction_a, pseudo_a, valid_mask=valid)
        dice_b = soft_dice_loss(prediction_b, pseudo_b, valid_mask=valid)
        self.assertTrue(torch.allclose(dice_a, dice_b, atol=1e-6))

        entropy_a = avg_entropy(prediction_a, valid_mask=valid)
        entropy_b = avg_entropy(prediction_b, valid_mask=valid)
        self.assertTrue(torch.allclose(entropy_a, entropy_b, atol=1e-6))

    def test_select_case_view_keeps_global_view_axis_for_every_chunk_size(self):
        patch = torch.tensor(
            [[[[0.0, 0.2, 0.4], [0.6, 0.8, 1.0]],
              [[1.0, 0.8, 0.6], [0.4, 0.2, 0.0]]]]
        )
        valid_mask = torch.ones(2, 2, 3)
        patches = [patch, patch + 0.1]
        valid_masks = [valid_mask, valid_mask]
        params = [{"scale": 1.0, "offset": 0.0}]
        params.extend(
            {"scale": 0.8 + 0.04 * index, "offset": -0.1 + 0.02 * index}
            for index in range(9)
        )
        results = []
        for view_batch_size in (1, 3, 10):
            model = TinyVoxTell()
            with torch.no_grad():
                model.project_bottleneck_embed.weight.copy_(torch.tensor([[1.0], [0.5]]))
                model.project_text_embed.weight.copy_(torch.eye(2))
            adapter = VoxTellCMTTA(
                model,
                torch.ones(1, 1, 2),
                "cpu",
                make_args(num_aug_views=9, view_batch_size=view_batch_size),
            )
            try:
                selected, scores = adapter._select_case_view(
                    patches, params, adapter.ctx.detach(), valid_masks
                )
                results.append((selected, scores))
            finally:
                adapter.close()

        self.assertEqual(tuple(results[0][1].shape), (10,))
        self.assertEqual(results[0][0], results[1][0])
        self.assertEqual(results[0][0], results[2][0])
        self.assertTrue(torch.allclose(results[0][1], results[1][1], atol=1e-6))
        self.assertTrue(torch.allclose(results[0][1], results[2][1], atol=1e-6))

    def test_two_stage_global_cac_gradient_matches_full_graph(self):
        patches = [
            torch.tensor([[[[0.0, 0.2], [0.4, 0.6]], [[0.8, 1.0], [0.3, 0.1]]]]),
            torch.tensor([[[[0.1, 0.3], [0.5, 0.7]], [[0.9, 0.6], [0.2, 0.0]]]]),
        ]
        valid_masks = [
            torch.tensor([[[1.0, 1.0], [1.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]]),
            torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]]),
        ]
        params = [
            {"scale": 1.0, "offset": 0.0},
            {"scale": 1.1, "offset": -0.05},
        ]
        model = TinyVoxTell()
        # Make foreground/background token similarities different while the
        # prompt-text path remains differentiable.
        model.project_bottleneck_embed = nn.Linear(1, 2, bias=True)
        with torch.no_grad():
            model.project_bottleneck_embed.weight.copy_(torch.tensor([[1.0], [0.0]]))
            model.project_bottleneck_embed.bias.copy_(torch.tensor([0.0, 1.0]))
            model.project_text_embed.weight.copy_(torch.eye(2))
        adapter = VoxTellCMTTA(
            model,
            torch.ones(1, 1, 2),
            "cpu",
            make_args(num_aug_views=1, view_batch_size=1),
        )

        def direct_cac_gradient():
            adapter.optimizer.zero_grad(set_to_none=True)
            components_sum = None
            for patch, valid_mask in zip(patches, valid_masks):
                selected = adapter._make_view_batch(
                    patch, params, valid_mask, 1, 2
                )
                student_prompt = adapter.ctx.detach().clone() + (
                    adapter.ctx - adapter.ctx.detach()
                )
                logits = adapter._forward(selected, student_prompt)
                components = adapter._cac_components(
                    logits, valid_mask.unsqueeze(0)
                )
                components_sum = adapter._add_components(components_sum, components)
            case_cac = cac_from_components(
                components_sum["foreground_sum"],
                components_sum["foreground_mass"],
                components_sum["background_sum"],
                components_sum["background_mass"],
            )
            (-adapter.w_cac * case_cac[0]).backward()
            return adapter.ctx.grad.detach().clone()

        try:
            direct_gradient = direct_cac_gradient()

            adapter.optimizer.zero_grad(set_to_none=True)
            adapter._backward_case_cac(
                patches,
                valid_masks,
                params,
                1,
                adapter.ctx.detach().clone(),
                1.0,
                False,
            )
            two_stage_gradient = adapter.ctx.grad.detach().clone()
        finally:
            adapter.close()

        self.assertTrue(torch.allclose(two_stage_gradient, direct_gradient, atol=1e-6))
        self.assertGreater(float(direct_gradient.norm()), 0.0)

        # Hard CAC regions must be insensitive to probability perturbations.
        logits = torch.tensor(
            [[[[[2.0, -2.0, 2.0, -2.0]]]]],
            requires_grad=True,
        )
        vision = torch.randn(4, 1, 2)
        text = torch.tensor([[[1.0, 0.0]]], requires_grad=True)
        components = cac_components_from_features(vision, text, logits)
        (-cac_from_components(
            components["foreground_sum"], components["foreground_mass"],
            components["background_sum"], components["background_mass"],
        )).backward()
        self.assertIsNone(logits.grad)
        self.assertIsNotNone(text.grad)
        self.assertGreater(float(text.grad.norm()), 0.0)

    def test_case_dice_entropy_and_gradient_are_patch_partition_invariant(self):
        full_patch = torch.tensor(
            [[[[0.0, 0.2], [0.4, 0.6]], [[0.8, 1.0], [0.3, 0.1]],
              [[0.1, 0.3], [0.5, 0.7]], [[0.9, 0.6], [0.2, 0.0]]]]
        )
        full_mask = torch.ones(4, 2, 2)
        params = [
            {"scale": 1.0, "offset": 0.0},
            {"scale": 1.1, "offset": -0.05},
        ]

        def make_adapter():
            model = TinyVoxTell()
            with torch.no_grad():
                model.project_text_embed.weight.copy_(torch.eye(2))
            return VoxTellCMTTA(
                model,
                torch.ones(1, 1, 2),
                "cpu",
                make_args(num_aug_views=1, view_batch_size=1),
            )

        def run(adapter, patches, masks):
            adapter.optimizer.zero_grad(set_to_none=True)
            dice, entropy = adapter._backward_case_supervision(
                patches,
                masks,
                params,
                1,
                adapter.ctx.detach().clone(),
                1.0,
                adapter.ctx.detach().clone(),
                False,
            )
            return dice, entropy, adapter.ctx.grad.detach().clone()

        def direct_run(adapter, patches, masks):
            adapter.optimizer.zero_grad(set_to_none=True)
            dice_stats = None
            entropy_sum = None
            entropy_mass = None
            for patch, mask in zip(patches, masks):
                selected = adapter._make_view_batch(patch, params, mask, 1, 2)
                with torch.no_grad():
                    pseudo = torch.sigmoid(
                        adapter._forward(selected, adapter.ctx.detach())[:, :1]
                    )
                views = adapter._make_view_batch(patch, params, mask, 0, 2)
                prompt = adapter.ctx.detach().clone() + (
                    adapter.ctx - adapter.ctx.detach()
                )
                probabilities = torch.sigmoid(adapter._forward(views, prompt)[:, :1])
                local_mask = mask.unsqueeze(0).expand(2, -1, -1, -1)
                local_dice = masked_dice_components(
                    probabilities, pseudo, local_mask.unsqueeze(1)
                )
                dice_stats = adapter._add_components(dice_stats, local_dice)
                local_entropy, local_mass = masked_entropy_components(
                    probabilities[1:2], local_mask[1:2]
                )
                entropy_sum = local_entropy[0] if entropy_sum is None else entropy_sum + local_entropy[0]
                entropy_mass = local_mass[0] if entropy_mass is None else entropy_mass + local_mass[0]
            dice = 1.0 - 2.0 * dice_stats["intersection"] / (
                dice_stats["prediction_mass"] + dice_stats["pseudo_mass"] + 1e-8
            )
            dice = dice.mean()
            entropy = entropy_sum / entropy_mass.clamp_min(1.0)
            (dice + adapter.w_entropy * entropy).backward()
            return float(dice.detach()), float(entropy.detach()), adapter.ctx.grad.detach().clone()

        unsplit = make_adapter()
        split = make_adapter()
        direct = make_adapter()
        try:
            full_result = run(unsplit, [full_patch], [full_mask])
            direct_result = direct_run(direct, [full_patch], [full_mask])
            split_result = run(
                split,
                [full_patch[:, :2], full_patch[:, 2:]],
                [full_mask[:2], full_mask[2:]],
            )
        finally:
            unsplit.close()
            split.close()
            direct.close()

        self.assertAlmostEqual(full_result[0], split_result[0], places=6)
        self.assertAlmostEqual(full_result[1], split_result[1], places=6)
        self.assertTrue(torch.allclose(full_result[2], split_result[2], atol=1e-6))
        self.assertAlmostEqual(full_result[0], direct_result[0], places=6)
        self.assertAlmostEqual(full_result[1], direct_result[1], places=6)
        self.assertTrue(torch.allclose(full_result[2], direct_result[2], atol=1e-6))

        with_padding = make_adapter()
        try:
            padded_result = run(
                with_padding,
                [full_patch[:, :2], full_patch[:, 2:], torch.full_like(full_patch, 99.0)],
                [full_mask[:2], full_mask[2:], torch.zeros_like(full_mask)],
            )
        finally:
            with_padding.close()
        self.assertAlmostEqual(full_result[0], padded_result[0], places=6)
        self.assertAlmostEqual(full_result[1], padded_result[1], places=6)
        self.assertTrue(torch.allclose(full_result[2], padded_result[2], atol=1e-6))

        zero_weight = make_adapter()
        zero_weight.w_entropy = 0.0
        zero_direct = make_adapter()
        zero_direct.w_entropy = 0.0
        try:
            zero_result = run(zero_weight, [full_patch], [full_mask])
            zero_direct_result = direct_run(zero_direct, [full_patch], [full_mask])
        finally:
            zero_weight.close()
            zero_direct.close()
        self.assertAlmostEqual(zero_result[0], zero_direct_result[0], places=6)
        self.assertAlmostEqual(zero_result[1], zero_direct_result[1], places=6)
        self.assertTrue(torch.allclose(zero_result[2], zero_direct_result[2], atol=1e-6))

    def test_selected_view_is_pseudo_label_source_and_case_has_one_step(self):
        model = TinyVoxTell()
        adapter = VoxTellCMTTA(
            model, torch.zeros(1, 1, 2), "cpu", make_args(view_batch_size=3)
        )
        try:
            # Make selection deterministic while retaining the real DSPU path.
            adapter._case_cac = lambda *_args: 0.0
            adapter._select_case_view = lambda *_args: (
                1,
                torch.tensor([0.0, 1.0, 0.0]),
            )
            patch = torch.zeros(1, 2, 2, 2)
            trace = adapter.adapt_case([patch, patch + 0.1])
            self.assertEqual(trace["selected_view"], 1)
            self.assertEqual(trace["pseudo_source_view"], 1)
            self.assertEqual(trace["num_views"], 3)
            self.assertEqual(trace["num_patches"], 2)
            self.assertEqual(trace["optimizer_steps_for_case"], 1)
            self.assertEqual(adapter.optimizer_step_count, 1)
            self.assertEqual(len(adapter.short_memory), 1)
            # The long-prompt forward is immediately followed by the all-view
            # student forward.  Its image must be student view 1.
            self.assertTrue(torch.equal(model.last_long_input, model.last_student_input[1:2]))
            # A second case exercises the non-leaf short-prompt fusion while
            # still allowing only one update for that case.
            trace2 = adapter.adapt_case([patch])
            self.assertEqual(trace2["optimizer_steps_for_case"], 1)
            self.assertEqual(adapter.optimizer_step_count, 2)
        finally:
            adapter.close()

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA AMP is unavailable")
    def test_cuda_amp_adapt_case_initializes_scaler_and_steps_once(self):
        device = torch.device("cuda")
        patches = [
            torch.tensor(
                [[[[0.0, 0.2], [0.4, 0.6]], [[0.8, 1.0], [0.3, 0.1]]]],
                device=device,
            ),
            torch.tensor(
                [[[[0.1, 0.3], [0.5, 0.7]], [[0.9, 0.6], [0.2, 0.0]]]],
                device=device,
            ),
        ]
        valid_masks = [
            torch.tensor([[[1.0, 1.0], [1.0, 0.0]], [[1.0, 1.0], [1.0, 1.0]]], device=device),
            torch.tensor([[[1.0, 1.0], [1.0, 1.0]], [[1.0, 0.0], [1.0, 1.0]]], device=device),
        ]
        params = [
            {"scale": 1.0, "offset": 0.0},
            {"scale": 1.1, "offset": -0.05},
        ]
        prompt = torch.zeros(1, 1, 2, device=device)

        def make_adapter():
            model = TinyVoxTell().to(device)
            with torch.no_grad():
                model.project_bottleneck_embed.weight.copy_(torch.tensor([[1.0], [0.0]], device=device))
                model.project_text_embed.weight.copy_(torch.eye(2, device=device))
            return VoxTellCMTTA(
                model, prompt, device, make_args(num_aug_views=1, view_batch_size=1)
            )

        def direct_gradient(adapter):
            adapter.optimizer.zero_grad(set_to_none=True)
            short_value = adapter.ctx.detach().clone()
            long_prompt = short_value.clone()
            dice_stats = None
            entropy_sum = None
            entropy_mass = None
            cac_components = None
            with torch.autocast(device_type="cuda", enabled=True):
                student_prompt = short_value + (
                    adapter.ctx - adapter.ctx.detach()
                )
                for patch, valid_mask in zip(patches, valid_masks):
                    selected = adapter._make_view_batch(
                        patch, params, valid_mask, 1, 2
                    ).to(device)
                    with torch.no_grad():
                        pseudo = torch.sigmoid(
                            adapter._forward(selected, long_prompt)[:, :1]
                        ).detach()
                    views = adapter._make_view_batch(
                        patch, params, valid_mask, 0, 2
                    ).to(device)
                    logits = adapter._forward(views, student_prompt)
                    probabilities = torch.sigmoid(logits[:, :1])
                    local_mask = valid_mask.unsqueeze(0).expand(2, -1, -1, -1)
                    dice_stats = adapter._add_components(
                        dice_stats,
                        masked_dice_components(
                            probabilities, pseudo, local_mask.unsqueeze(1)
                        ),
                    )
                    local_entropy, local_mass = masked_entropy_components(
                        probabilities[1:2], local_mask[1:2]
                    )
                    entropy_sum = (
                        local_entropy[0]
                        if entropy_sum is None
                        else entropy_sum + local_entropy[0]
                    )
                    entropy_mass = (
                        local_mass[0]
                        if entropy_mass is None
                        else entropy_mass + local_mass[0]
                    )
                    selected_logits = adapter._forward(selected, student_prompt)
                    local_cac = adapter._cac_components(
                        selected_logits,
                        valid_mask.unsqueeze(0),
                    )
                    cac_components = adapter._add_components(cac_components, local_cac)
                dice = (
                    1.0
                    - 2.0 * dice_stats["intersection"]
                    / (
                        dice_stats["prediction_mass"]
                        + dice_stats["pseudo_mass"]
                        + 1e-8
                    )
                ).mean()
                entropy = entropy_sum / entropy_mass.clamp_min(1.0)
                case_cac = cac_from_components(
                    cac_components["foreground_sum"],
                    cac_components["foreground_mass"],
                    cac_components["background_sum"],
                    cac_components["background_mass"],
                )[0]
                objective = dice + adapter.w_entropy * entropy - adapter.w_cac * case_cac
                direct_components = torch.autograd.grad(
                    (dice, adapter.w_entropy * entropy, -adapter.w_cac * case_cac),
                    adapter.ctx,
                    retain_graph=True,
                )
            adapter.scaler.scale(objective).backward()
            scale = float(adapter.scaler.get_scale())
            scaled_gradient = adapter.ctx.grad.detach().float().clone()
            return {
                "scaled": scaled_gradient,
                "unscaled": scaled_gradient / scale,
                "scale": scale,
                "components": {
                    name: gradient.detach().float().clone()
                    for name, gradient in zip(
                        ("dice", "entropy", "cac"), direct_components
                    )
                },
            }

        def two_stage_component_gradients():
            """Split replay gradients only when the aggregate check fails."""
            adapter = make_adapter()
            try:
                short_value = adapter.ctx.detach().clone()
                original_entropy_weight = adapter.w_entropy
                component_gradients = {}

                def replay_gradient():
                    scale = float(adapter.scaler.get_scale())
                    gradient = adapter.ctx.grad.detach().float().clone()
                    return gradient / scale

                adapter.optimizer.zero_grad(set_to_none=True)
                adapter.w_entropy = 0.0
                adapter._backward_case_supervision(
                    patches, valid_masks, params, 1, short_value, 1.0,
                    short_value, True
                )
                component_gradients["dice"] = replay_gradient()

                adapter.optimizer.zero_grad(set_to_none=True)
                adapter.w_entropy = original_entropy_weight
                adapter._backward_case_supervision(
                    patches, valid_masks, params, 1, short_value, 1.0,
                    short_value, True
                )
                supervision_gradient = replay_gradient()
                component_gradients["entropy"] = (
                    supervision_gradient - component_gradients["dice"]
                )

                adapter.optimizer.zero_grad(set_to_none=True)
                adapter._backward_case_cac(
                    patches, valid_masks, params, 1, short_value, 1.0, True
                )
                component_gradients["cac"] = replay_gradient()
                return component_gradients
            finally:
                adapter.close()

        direct_adapter = make_adapter()
        two_stage_adapter = make_adapter()
        try:
            direct_gradient_value = direct_gradient(direct_adapter)
            two_stage_adapter.optimizer.zero_grad(set_to_none=True)
            short_value = two_stage_adapter.ctx.detach().clone()
            two_stage_adapter._backward_case_supervision(
                patches,
                valid_masks,
                params,
                1,
                short_value,
                1.0,
                short_value,
                True,
            )
            two_stage_adapter._backward_case_cac(
                patches,
                valid_masks,
                params,
                1,
                short_value,
                1.0,
                True,
            )
            two_stage_scale = float(two_stage_adapter.scaler.get_scale())
            two_stage_scaled_gradient = (
                two_stage_adapter.ctx.grad.detach().float().clone()
            )
            two_stage_gradient_value = two_stage_scaled_gradient / two_stage_scale
        finally:
            direct_adapter.close()
            two_stage_adapter.close()

        self.assertAlmostEqual(
            direct_gradient_value["scale"],
            two_stage_scale,
            places=6,
            msg=(
                "direct and two-stage GradScaler scales differ: "
                f"direct_scale={direct_gradient_value['scale']} "
                f"two_stage_scale={two_stage_scale}"
            ),
        )

        scaled_difference = (
            two_stage_scaled_gradient - direct_gradient_value["scaled"]
        ).abs()
        unscaled_difference = (
            two_stage_gradient_value - direct_gradient_value["unscaled"]
        ).abs()
        max_abs_diff = float(unscaled_difference.max().cpu())
        max_relative_diff = float(
            (
                unscaled_difference
                / direct_gradient_value["unscaled"].abs().clamp_min(1e-12)
            ).max().cpu()
        )

        if not torch.allclose(
            two_stage_gradient_value,
            direct_gradient_value["unscaled"],
            atol=2e-3,
            rtol=2e-3,
        ):
            replay_components = two_stage_component_gradients()
            component_differences = {
                name: float(
                    (replay_components[name] - direct_gradient_value["components"][name])
                    .abs()
                    .max()
                    .cpu()
                )
                for name in ("dice", "entropy", "cac")
            }
            self.fail(
                "Unscaled direct/two-stage AMP gradients differ. "
                f"direct_gradient={direct_gradient_value['unscaled']} "
                f"two_stage_gradient={two_stage_gradient_value} "
                f"max_abs_diff={max_abs_diff} "
                f"max_relative_diff={max_relative_diff} "
                f"direct_scale={direct_gradient_value['scale']} "
                f"two_stage_scale={two_stage_scale} "
                f"scaled_max_abs_diff={float(scaled_difference.max().cpu())} "
                f"component_max_abs_diff={component_differences}"
            )

        adapter = make_adapter()
        try:
            trace = adapter.adapt_case(patches)
            self.assertEqual(trace["optimizer_steps_for_case"], 1)
            self.assertEqual(adapter.optimizer_step_count, 1)
            self.assertIn("scale", adapter.scaler.state_dict())
        finally:
            adapter.close()

    def test_real_voxtell_qwen_zero_delta_text_and_logits(self):
        if os.environ.get("RUN_REAL_VOXTELL_TESTS") != "1":
            self.skipTest("set RUN_REAL_VOXTELL_TESTS=1 to run the real VoxTell/Qwen test")
        from run_voxtell_cmtta import DEFAULT_QWEN, DEFAULT_VOXTELL_ROOT

        root = Path(os.environ.get("VOXTELL_ROOT", str(DEFAULT_VOXTELL_ROOT)))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from voxtell.inference.predictor import VoxTellPredictor
        except (ImportError, ModuleNotFoundError) as error:
            self.skipTest(f"real VoxTell dependencies unavailable: {error}")

        model_dir = Path(os.environ.get("VOXTELL_MODEL_DIR", str(root / "model")))
        text_model = Path(os.environ.get("VOXTELL_TEXT_MODEL", str(DEFAULT_QWEN)))
        if not model_dir.exists() or not text_model.exists():
            self.skipTest("real VoxTell model or Qwen weights are unavailable")
        device = torch.device(
            os.environ.get("VOXTELL_TEST_DEVICE", "cuda:0" if torch.cuda.is_available() else "cpu")
        )
        predictor = VoxTellPredictor(
            model_dir=str(model_dir), device=device, text_encoding_model=str(text_model)
        )
        native = predictor.embed_text_prompts(["liver"]).detach().to(device)
        # This opt-in check only verifies zero-delta text/logit equivalence.
        # A full 192^3, ten-view adapt_case is unnecessary and memory-intensive
        # for this test.
        args = make_args()
        args.max_text_length = predictor.max_text_length
        adapter = VoxTellCMTTA(
            predictor.network,
            None,
            device,
            args,
            qwen_text_encoder=predictor.text_backbone,
            qwen_tokenizer=predictor.tokenizer,
            text_prompt="liver",
        )
        try:
            with torch.no_grad():
                zero_text = adapter._encode_ctx(adapter.ctx_delta.detach())
            self.assertTrue(torch.allclose(zero_text, native, atol=2e-5, rtol=2e-5))

            patch = torch.zeros((1, *predictor.patch_size))
            model_patch = patch.unsqueeze(0).to(device)
            with torch.no_grad():
                native_logits = predictor.network(model_patch, native.unsqueeze(2))
                adapted_logits = adapter._forward(model_patch, adapter.ctx_delta.detach())
                if isinstance(native_logits, (list, tuple)):
                    native_logits = native_logits[0]
            self.assertTrue(torch.allclose(native_logits, adapted_logits, atol=2e-4, rtol=2e-4))
            print(
                "[real VoxTell] zero ctx_delta norm: "
                f"{float(adapter.ctx_delta.detach().norm().cpu()):.8g}"
            )
        finally:
            adapter.close()

    def test_short_memory_is_fifo_and_cac_weighted(self):
        memory = ShortPromptMemory(2)
        memory.append(torch.tensor([1.0]), 0.0)
        memory.append(torch.tensor([3.0]), 2.0)
        fused = memory.weighted_ctx(torch.device("cpu"), torch.float32)
        self.assertGreater(float(fused), 2.5)
        memory.append(torch.tensor([5.0]), 4.0)
        self.assertEqual(len(memory), 2)
        self.assertEqual([float(x) for x in memory.contexts], [3.0, 5.0])

    def test_lspm_primary_quality_switches_between_case_cac_and_case_tdc(self):
        for metric, expected in (("cac", 3.0), ("tdc", 7.0)):
            adapter = VoxTellCMTTA(
                TinyVoxTell(),
                torch.zeros(1, 1, 2),
                "cpu",
                make_args(view_selection_metric=metric),
            )
            try:
                adapter._case_cac = lambda *_args: 3.0
                adapter._case_tdc = lambda *_args: 7.0
                short, current, historical, _weight = adapter._dynamic_short_ctx(
                    [torch.zeros(1, 2, 2, 2)], [torch.ones(2, 2, 2)]
                )
                self.assertEqual(current, expected)
                self.assertEqual(historical, expected)
                self.assertEqual(tuple(short.shape), (1, 2))
            finally:
                adapter.close()

    def test_checkpoint_contains_lspm_optimizer_and_scaler_state(self):
        adapter = VoxTellCMTTA(TinyVoxTell(), torch.zeros(1, 1, 2), "cpu", make_args())
        try:
            state = adapter.state_dict()
            self.assertTrue(
                set(("ctx_delta", "initial_ctx_delta", "short_delta", "long_delta", "ctx_delta_memory", "optimizer", "scaler"))
                <= set(state)
            )
            self.assertIn("deltas", state["ctx_delta_memory"])
            self.assertIn("cacs", state["ctx_delta_memory"])
        finally:
            adapter.close()


if __name__ == "__main__":
    unittest.main()
