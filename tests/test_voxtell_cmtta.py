import os
import sys
import types
import unittest
import warnings
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
from torch import nn

from data.voxtell_p0 import make_case_patches, pad_to_patch_grid
from method.voxtell_cmtta import (
    ShortPromptMemory,
    VoxTellCMTTA,
    avg_entropy,
    cac_from_features,
    cac_from_components,
    cac_components_from_features,
    masked_dice_components,
    masked_entropy_components,
    select_cac_view,
    soft_dice_loss,
    tdc_from_components,
    tdc_patch_components,
)
from run_voxtell_cmtta import check_prediction_nifti_geometry, save_prediction_nifti


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
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


class VoxTellCMTTATest(unittest.TestCase):
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
            adapter.scaler.scale(objective).backward()
            return adapter.ctx.grad.detach().clone()

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
            two_stage_gradient_value = two_stage_adapter.ctx.grad.detach().clone()
        finally:
            direct_adapter.close()
            two_stage_adapter.close()

        self.assertTrue(
            torch.allclose(
                two_stage_gradient_value,
                direct_gradient_value,
                atol=2e-3,
                rtol=2e-3,
            )
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
