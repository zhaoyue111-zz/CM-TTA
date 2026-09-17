import types
import unittest
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
)
from run_voxtell_cmtta import save_prediction_nifti


class TinyVoxTell(nn.Module):
    """Small frozen VoxTell-shaped network for protocol tests."""

    def __init__(self, text_dim=2):
        super().__init__()
        self.project_bottleneck_embed = nn.Linear(1, 2, bias=False)
        self.project_text_embed = nn.Linear(text_dim, 2, bias=False)

    def forward(self, image, text_embedding):
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
        return image[:, :1] + prompt_bias


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


class TinyQwen(nn.Module):
    """Frozen differentiable text encoder used to test ctx input learning."""

    def __init__(self):
        super().__init__()
        self.token_embedding = nn.Embedding(32, 4)

    def get_input_embeddings(self):
        return self.token_embedding

    def forward(self, inputs_embeds, attention_mask):
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
    def test_prediction_is_transposed_back_to_nifti_axis_order(self):
        import nibabel as nib

        with TemporaryDirectory() as temp_dir:
            temp_dir = Path(temp_dir)
            image_path = temp_dir / "image.nii.gz"
            output_path = temp_dir / "prediction.nii.gz"
            source_data = np.arange(4 * 3 * 2, dtype=np.float32).reshape(4, 3, 2)
            affine = np.diag([1.0, 2.0, 3.0, 1.0])
            nib.save(nib.Nifti1Image(source_data, affine), str(image_path))

            model_space_prediction = np.arange(2 * 3 * 4, dtype=np.uint8).reshape(2, 3, 4)
            save_prediction_nifti(model_space_prediction, image_path, output_path)

            saved = nib.as_closest_canonical(nib.load(str(output_path)))
            self.assertEqual(tuple(saved.shape), (4, 3, 2))
            self.assertTrue(
                np.array_equal(saved.get_fdata(), model_space_prediction.transpose(2, 1, 0))
            )
            self.assertTrue(np.allclose(saved.affine, affine))

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
            self.assertEqual(adapter.optimizer_parameters, [adapter.ctx])
            self.assertTrue(adapter.ctx.requires_grad)
            self.assertEqual(tuple(adapter.ctx.shape), (2, 4))
            self.assertEqual(tuple(adapter._encode_ctx(adapter.ctx).shape), (1, 1, 4))
            self.assertEqual(adapter._ctx_insert_index, 2)
            self.assertTrue(
                torch.equal(
                    adapter._fixed_token_embeddings[0, 2],
                    qwen.token_embedding.weight[11],
                )
            )
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertTrue(all(not p.requires_grad for p in adapter.qwen_text_encoder.parameters()))
            self.assertEqual(adapter.ctx.dtype, torch.float32)
            self.assertIsInstance(adapter.optimizer, torch.optim.Adam)
            self.assertEqual(adapter.optimizer.param_groups[0]["lr"], 0.05)
            self.assertEqual(adapter.optimizer.param_groups[0]["weight_decay"], 0.0)

            adapter._encode_ctx(adapter.ctx).sum().backward()
            self.assertIsNotNone(adapter.ctx.grad)
            self.assertGreater(float(adapter.ctx.grad.norm()), 0.0)
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(all(p.grad is None for p in qwen.parameters()))
            adapter.optimizer.zero_grad(set_to_none=True)
            ctx_before = adapter.ctx.detach().clone()
            trace = adapter.adapt_case([torch.zeros(1, 2, 2, 2)])
            self.assertEqual(trace["optimizer_steps_for_case"], 1)
            self.assertFalse(torch.equal(ctx_before, adapter.ctx.detach()))
            self.assertTrue(all(p.grad is None for p in model.parameters()))
            self.assertTrue(all(p.grad is None for p in qwen.parameters()))
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

    def test_cac_selects_highest_score_and_entropy_is_binary(self):
        probabilities = torch.full((3, 1, 2, 2, 2), 0.5)
        selected, selected_indices = select_cac_view(
            torch.tensor([0.1, 0.9, 0.2]), probabilities.squeeze(1), 0.1
        )
        self.assertEqual(selected, 1)
        self.assertEqual(selected_indices.tolist(), [1])
        self.assertAlmostEqual(float(avg_entropy(probabilities[0])), 0.693147, places=5)

    def test_cac_pools_features_before_cosine(self):
        # The foreground average is [1, 1] and background average [1, -1].
        # Cosine-after-pooling therefore gives 1 - 0 = 1.  Averaging voxel
        # cosines would incorrectly produce a different result.
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
        self.assertTrue(torch.allclose(result, torch.ones(1), atol=1e-5))

    def test_cac_uses_hard_half_probability_masks_for_feature_pooling(self):
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
        components = cac_components_from_features(vision, logits)
        self.assertTrue(
            torch.equal(components["foreground_mass"], torch.tensor([2.0]))
        )
        self.assertTrue(
            torch.equal(components["background_mass"], torch.tensor([2.0]))
        )
        self.assertTrue(
            torch.allclose(components["foreground_sum"], torch.tensor([[1.0, 1.0]]))
        )
        self.assertTrue(
            torch.allclose(components["background_sum"], torch.tensor([[3.0, 1.0]]))
        )

        all_foreground = cac_components_from_features(
            vision, torch.full_like(logits, 20.0)
        )
        all_background = cac_components_from_features(
            vision, torch.full_like(logits, -20.0)
        )
        self.assertTrue(torch.isfinite(cac_from_components(
            all_foreground["foreground_sum"],
            all_foreground["foreground_mass"],
            all_foreground["background_sum"],
            all_foreground["background_mass"],
            torch.tensor([[1.0, 0.0]]),
        )).all())
        self.assertTrue(torch.isfinite(cac_from_components(
            all_background["foreground_sum"],
            all_background["foreground_mass"],
            all_background["background_sum"],
            all_background["background_mass"],
            torch.tensor([[1.0, 0.0]]),
        )).all())

    def test_case_cac_uses_global_sums_not_patch_cac_mean(self):
        text = torch.tensor([[1.0, 0.0]])
        first = cac_from_components(
            torch.tensor([[10.0, 0.0]]), torch.tensor([10.0]),
            torch.tensor([[0.0, 10.0]]), torch.tensor([10.0]), text
        )
        second = cac_from_components(
            torch.tensor([[0.0, 1.0]]), torch.tensor([1.0]),
            torch.tensor([[0.0, 1.0]]), torch.tensor([1.0]), text
        )
        global_score = cac_from_components(
            torch.tensor([[10.0, 1.0]]), torch.tensor([11.0]),
            torch.tensor([[0.0, 11.0]]), torch.tensor([11.0]), text
        )
        self.assertFalse(torch.allclose(global_score, (first + second) / 2.0))

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
        # Make pooled foreground/background directions different so both the
        # probability-weighted visual path and the prompt-text path contribute
        # a non-zero gradient.
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

        def direct_cac_gradient(detach_components=False, detach_text=False):
            adapter.optimizer.zero_grad(set_to_none=True)
            components_sum = None
            text_sum = None
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
                if detach_components:
                    components = {key: value.detach() for key, value in components.items()}
                components_sum = adapter._add_components(components_sum, components)
                text = adapter._text_features[0].float()
                text_sum = text if text_sum is None else text_sum + text
            if detach_text:
                text_sum = text_sum.detach()
            case_cac = cac_from_components(
                components_sum["foreground_sum"],
                components_sum["foreground_mass"],
                components_sum["background_sum"],
                components_sum["background_mass"],
                text_sum / len(patches),
            )
            if case_cac[0].requires_grad:
                (-adapter.w_cac * case_cac[0]).backward()
            else:
                return torch.zeros_like(adapter.ctx)
            return adapter.ctx.grad.detach().clone()

        try:
            direct_gradient = direct_cac_gradient()
            visual_probability_gradient = direct_cac_gradient(detach_text=True)
            text_gradient = direct_cac_gradient(detach_components=True)

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
        self.assertEqual(float(visual_probability_gradient.norm()), 0.0)
        self.assertGreater(float(text_gradient.norm()), 0.0)

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
            text_sum = None
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
                    text = adapter._text_features[0].float()
                    text_sum = text if text_sum is None else text_sum + text
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
                    text_sum / len(patches),
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
                set(("ctx", "initial_ctx", "short_ctx", "long_ctx", "ctx_memory", "optimizer", "scaler"))
                <= set(state)
            )
            self.assertIn("ctxs", state["ctx_memory"])
            self.assertIn("cacs", state["ctx_memory"])
        finally:
            adapter.close()


if __name__ == "__main__":
    unittest.main()
