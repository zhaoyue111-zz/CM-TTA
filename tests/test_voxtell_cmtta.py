import types
import unittest

import torch
from torch import nn

from data.voxtell_p0 import make_case_patches, pad_to_patch_grid
from method.voxtell_cmtta import (
    ShortPromptMemory,
    VoxTellCMTTA,
    avg_entropy,
    cac_from_features,
    cac_from_components,
    select_cac_view,
    soft_dice_loss,
)


class TinyVoxTell(nn.Module):
    """Small frozen VoxTell-shaped network for protocol tests."""

    def __init__(self):
        super().__init__()
        self.project_bottleneck_embed = nn.Linear(1, 2, bias=False)
        self.project_text_embed = nn.Linear(2, 2, bias=False)

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
    def test_only_soft_prompt_and_qwen_are_frozen(self):
        model = TinyVoxTell()
        qwen = nn.Linear(2, 2)
        adapter = VoxTellCMTTA(model, torch.ones(1, 1, 2), "cpu", make_args(), qwen)
        try:
            self.assertEqual(adapter.optimizer_parameters, [adapter.soft_prompt])
            self.assertTrue(adapter.soft_prompt.requires_grad)
            self.assertTrue(all(not p.requires_grad for p in model.parameters()))
            self.assertTrue(all(not p.requires_grad for p in qwen.parameters()))
            self.assertEqual(adapter.soft_prompt.dtype, torch.float32)
            self.assertIsInstance(adapter.optimizer, torch.optim.Adam)
            self.assertEqual(adapter.optimizer.param_groups[0]["lr"], 0.05)
            self.assertEqual(adapter.optimizer.param_groups[0]["weight_decay"], 0.0)
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
                    patches, params, adapter.soft_prompt.detach(), valid_masks
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
                student_prompt = adapter.soft_prompt.detach().clone() + (
                    adapter.soft_prompt - adapter.soft_prompt.detach()
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
            (-adapter.w_cac * case_cac[0]).backward()
            return adapter.soft_prompt.grad.detach().clone()

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
                adapter.soft_prompt.detach().clone(),
                1.0,
                False,
            )
            two_stage_gradient = adapter.soft_prompt.grad.detach().clone()
        finally:
            adapter.close()

        self.assertTrue(torch.allclose(two_stage_gradient, direct_gradient, atol=1e-6))
        self.assertGreater(float(direct_gradient.norm()), 0.0)
        self.assertGreater(float(visual_probability_gradient.norm()), 0.0)
        self.assertGreater(float(text_gradient.norm()), 0.0)

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

    def test_short_memory_is_fifo_and_cac_weighted(self):
        memory = ShortPromptMemory(2)
        memory.append(torch.tensor([1.0]), 0.0)
        memory.append(torch.tensor([3.0]), 2.0)
        fused = memory.weighted_prompt(torch.device("cpu"), torch.float32)
        self.assertGreater(float(fused), 2.5)
        memory.append(torch.tensor([5.0]), 4.0)
        self.assertEqual(len(memory), 2)
        self.assertEqual([float(x) for x in memory.prompts], [3.0, 5.0])

    def test_checkpoint_contains_lspm_optimizer_and_scaler_state(self):
        adapter = VoxTellCMTTA(TinyVoxTell(), torch.zeros(1, 1, 2), "cpu", make_args())
        try:
            state = adapter.state_dict()
            self.assertTrue(
                set(("soft_prompt", "short_prompt", "long_prompt", "short_memory", "optimizer", "scaler"))
                <= set(state)
            )
            self.assertIn("prompts", state["short_memory"])
            self.assertIn("cacs", state["short_memory"])
        finally:
            adapter.close()


if __name__ == "__main__":
    unittest.main()
