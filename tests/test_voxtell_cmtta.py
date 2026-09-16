import types
import unittest

import torch
from torch import nn

from method.voxtell_cmtta import (
    ShortPromptMemory,
    VoxTellCMTTA,
    avg_entropy,
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
        weight_decay=0.0,
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

    def test_selected_view_is_pseudo_label_source_and_case_has_one_step(self):
        model = TinyVoxTell()
        adapter = VoxTellCMTTA(model, torch.zeros(1, 1, 2), "cpu", make_args())
        try:
            # Make selection deterministic while retaining the real DSPU path.
            adapter._case_cac = lambda _prompt, _patches: 0.0
            adapter._select_case_view = lambda _patches, _params, _short: (
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
