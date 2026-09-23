import sys
import unittest
from pathlib import Path

import numpy as np
import torch

from attention_probability_diagnostic import (
    aggregate_attention_patches,
    apply_intensity_view,
    attention_tokens_to_3d,
    balanced_indices,
    compute_case_metrics,
    confusion_regions,
    inverse_view_map,
    percentile_rank,
    resolve_label_value,
    safe_auc_pr,
)


class AttentionProbabilityDiagnosticTest(unittest.TestCase):
    def test_attention_tokens_restore_voxtell_hwd_order(self):
        tokens = np.arange(2 * 3 * 4, dtype=np.float32)
        restored = attention_tokens_to_3d(tokens, (4, 2, 3))
        self.assertEqual(restored.shape, (4, 2, 3))
        self.assertEqual(restored[2, 1, 2], 1 * 3 * 4 + 2 * 4 + 2)

    def test_sliding_attention_fusion_and_padding_removal(self):
        slicers = []
        records = []
        for depth in (0, 1):
            for height in (0, 1):
                for width in (0, 1):
                    slicers.append((slice(None), slice(depth, depth + 2), slice(height, height + 2), slice(width, width + 2)))
                    value = 1.0 if (depth, height, width) == (0, 0, 0) else 3.0
                    records.append({"weights": np.full((1, 2, 1, 8), value, dtype=np.float32), "memory_shape_dhw": (2, 2, 2)})
        result = aggregate_attention_patches(
            records, slicers, (3, 3, 3), (slice(0, 3), slice(0, 3), slice(0, 3)), np.ones((2, 2, 2), dtype=np.float32)
        )
        self.assertEqual(result.shape, (3, 3, 3))
        self.assertAlmostEqual(float(result[0, 0, 0]), 1.0)
        self.assertAlmostEqual(float(result[1, 1, 1]), 2.75)
        self.assertAlmostEqual(float(result[2, 2, 2]), 3.0)

    def test_intensity_view_inverse_preserves_map_coordinates(self):
        data = torch.ones(1, 2, 2, 2)
        viewed = apply_intensity_view(data, {"scale": 2.0, "offset": 0.5})
        self.assertTrue(torch.allclose(viewed, torch.full_like(viewed, 2.5)))
        probability = np.arange(8, dtype=np.float32).reshape(2, 2, 2)
        restored = inverse_view_map(probability, {"scale": 2.0, "offset": 0.5})
        np.testing.assert_array_equal(restored, probability)

    def test_confusion_regions_and_crop_outside_not_tn(self):
        probability = np.array([[[0.8, 0.2, 0.0, 0.0]]], dtype=np.float32)
        gt = np.array([[[1, 1, 0, 0]]], dtype=bool)
        valid = np.array([[[1, 1, 1, 0]]], dtype=bool)
        regions = confusion_regions(probability, gt, valid)
        self.assertEqual(int(regions["TP"].sum()), 1)
        self.assertEqual(int(regions["FN"].sum()), 1)
        self.assertEqual(int(regions["TN"].sum()), 1)
        _, metrics, _ = compute_case_metrics(probability, np.zeros_like(probability), gt, valid)
        self.assertEqual(metrics["gt_background_valid_voxel_count"], 1)

    def test_rank_and_residual_are_case_valid_only(self):
        values = np.array([[[0.1, 0.9, 0.0]]], dtype=np.float32)
        valid = np.array([[[1, 1, 0]]], dtype=bool)
        rank = percentile_rank(values, valid)
        np.testing.assert_allclose(rank[valid], [0.0, 1.0])
        self.assertTrue(np.isnan(rank[~valid]).all())
        _, metrics, maps = compute_case_metrics(values, values[:, :, ::-1].copy(), valid, valid)
        self.assertIn("residual", maps)
        self.assertTrue(np.isfinite(metrics["pearson_attention_probability"]))

    def test_empty_fp_or_fn_auc_pr_is_nan(self):
        auc, ap = safe_auc_pr(np.array([True, True]), np.array([0.1, 0.2], dtype=np.float32))
        self.assertTrue(np.isnan(auc))
        self.assertTrue(np.isnan(ap))
        auc, ap = safe_auc_pr(np.array([], dtype=bool), np.array([], dtype=np.float32))
        self.assertTrue(np.isnan(auc))
        self.assertTrue(np.isnan(ap))

    def test_balanced_sampling_is_reproducible(self):
        gt = np.array([True, False, False, False, False])
        valid = np.ones_like(gt, dtype=bool)
        first = balanced_indices(gt, valid, seed=123)
        second = balanced_indices(gt, valid, seed=123)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.size, 2)

    def test_label_selection_rejects_multiple_nonzero_values(self):
        self.assertEqual(resolve_label_value(np.array([0, 4, 4]), None, "one"), 4)
        with self.assertRaisesRegex(ValueError, r"\[4, 9\].*--label-value"):
            resolve_label_value(np.array([0, 4, 9]), None, "ambiguous")

    def test_real_voxtell_decoder_order_matches_metadata(self):
        root = Path("/data/zy/VoxTell_from_disk")
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        try:
            from dynamic_network_architectures.building_blocks.residual import BasicBlockD
            from torch import nn
            from voxtell.model.voxtell_model import VoxTellModel
        except ModuleNotFoundError as exc:
            self.skipTest(f"full VoxTell model dependencies unavailable: {exc}")

        old_configs = VoxTellModel.DECODER_CONFIGS
        VoxTellModel.DECODER_CONFIGS = {i: {"channels": 2, "shape": (64 // (2 ** i),) * 3} for i in range(6)}
        try:
            model = VoxTellModel(
                input_channels=1, n_stages=6, features_per_stage=[2] * 6,
                conv_op=nn.Conv3d, kernel_sizes=[3] * 6, strides=[1, 2, 2, 2, 2, 2],
                n_blocks_per_stage=[1] * 6, n_conv_per_stage_decoder=[1] * 5,
                conv_bias=False, norm_op=nn.InstanceNorm3d,
                norm_op_kwargs={"eps": 1e-5, "affine": True}, dropout_op=None,
                dropout_op_kwargs=None, nonlin=nn.LeakyReLU,
                nonlin_kwargs={"inplace": True}, deep_supervision=True,
                block=BasicBlockD, num_maskformer_stages=5, query_dim=8,
                decoder_layer=5, text_embedding_dim=4, num_heads=1,
                project_to_decoder_hidden_dim=4,
            ).eval()
            with torch.no_grad():
                outputs = model(torch.randn(1, 1, 64, 64, 64), torch.randn(1, 1, 4), return_decoder_outputs=True)
            shapes = [tuple(output.shape[2:]) for output in outputs]
            metadata = model.decoder.get_output_metadata(observed_output_shapes=shapes, patch_spatial_shape=(64, 64, 64))
            self.assertEqual(len(outputs), 5)
            self.assertEqual(metadata[-1]["model_output_list_index"], 0)
            self.assertTrue(metadata[-1]["is_final_output"])
            self.assertEqual(metadata[0]["model_output_list_index"], 4)
            self.assertEqual(shapes[0], tuple(metadata[-1]["observed_raw_patch_shape"]))
            self.assertEqual(shapes[4], tuple(metadata[0]["observed_raw_patch_shape"]))
            self.assertGreater(np.prod(shapes[0]), np.prod(shapes[4]))
        finally:
            VoxTellModel.DECODER_CONFIGS = old_configs


if __name__ == "__main__":
    unittest.main()
