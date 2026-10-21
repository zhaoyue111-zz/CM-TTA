from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import nibabel as nib
import numpy as np
import torch
from nnunetv2.imageio.nibabel_reader_writer import NibabelIOWithReorient
from nnunetv2.inference.sliding_window_prediction import compute_steps_for_sliding_window
from torch import nn

from prototype_diagnostic.build_bank import (
    _component_regions,
    _eroded_core,
    aggregate_normalized_view_features,
    compute_source_tdc_for_case,
    load_view_params_replay,
    region_prototype,
    select_memory_cases,
)
from prototype_diagnostic.evaluate_prototype import prototype_similarity_maps
from prototype_diagnostic.extract_features import ExtractedVolume, SlidingD5FeatureExtractor
from prototype_diagnostic.utils import (
    binary_ranking_metrics,
    foreground_mask,
    read_test_entries,
    restore_crop,
    save_float_nifti_model_space,
    top_fraction_enrichment,
)


class _TinyNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.stages = nn.ModuleList([nn.Conv3d(1, 2, 1, bias=False)])
        with torch.no_grad():
            self.decoder.stages[0].weight.copy_(
                torch.tensor([[[[[1.0]]]], [[[[2.0]]]]])
            )

    def forward(self, image, _text_embedding, return_decoder_outputs=False):
        feature = self.decoder.stages[-1](image)
        logits = feature[:, :1]
        return [logits, logits, logits, logits] if return_decoder_outputs else logits


class _TinyPredictor:
    def __init__(self):
        self.device = torch.device("cpu")
        self.patch_size = (2, 2, 2)
        self.tile_step_size = 0.5
        self.network = _TinyNetwork().eval()

    def _internal_get_sliding_window_slicers(self, image_size):
        steps = compute_steps_for_sliding_window(
            image_size, self.patch_size, self.tile_step_size
        )
        return [
            (slice(None), slice(d, d + 2), slice(h, h + 2), slice(w, w + 2))
            for d in steps[0]
            for h in steps[1]
            for w in steps[2]
        ]


class PrototypeDiagnosticTest(unittest.TestCase):
    def test_sliding_hook_returns_aligned_prehead_feature(self):
        predictor = _TinyPredictor()
        extractor = SlidingD5FeatureExtractor(predictor, print_shapes=False)
        data = torch.arange(27, dtype=torch.float32).reshape(1, 3, 3, 3)
        result = extractor.extract(data, torch.zeros(1, 1, 2))
        self.assertEqual(tuple(result.feature.shape), (2, 3, 3, 3))
        self.assertTrue(torch.allclose(result.feature[0], data[0], atol=1e-5))
        self.assertTrue(torch.allclose(result.feature[1], data[0] * 2, atol=1e-5))
        self.assertTrue(torch.allclose(result.logits, data[0], atol=1e-5))

        decoder_result = extractor.extract(
            data,
            torch.zeros(1, 1, 2),
            capture_feature=False,
            return_decoder_logits=True,
        )
        self.assertIsNone(decoder_result.feature)
        self.assertEqual(len(decoder_result.decoder_logits), 4)
        for logits in decoder_result.decoder_logits:
            self.assertTrue(torch.allclose(logits, data[0], atol=1e-5))

    def test_connected_components_and_erosion_are_three_dimensional(self):
        mask = np.zeros((7, 7, 7), dtype=bool)
        mask[1:6, 1:6, 1:6] = True
        mask[0, 0, 0] = True
        regions = _component_regions(mask, min_region_voxels=2)
        self.assertEqual(len(regions), 1)
        _, region, size = regions[0]
        self.assertEqual(size, 126)  # diagonal corner joins under 26-connectivity
        core = _eroded_core(region, 1)
        self.assertLess(int(core.sum()), size)
        self.assertGreater(int(core.sum()), 0)

    def test_similarity_is_leave_one_case_out(self):
        feature = torch.tensor(
            [
                [[[1.0, 0.0]]],
                [[[0.0, 1.0]]],
            ]
        )
        prototypes = [
            {"case_name": "query.nii.gz", "feature": torch.tensor([1.0, 0.0])},
            {"case_name": "other.nii.gz", "feature": torch.tensor([0.0, 1.0])},
        ]
        maximum, topk, info = prototype_similarity_maps(
            feature, prototypes, "query.nii.gz", topk=5, voxel_chunk_size=1
        )
        self.assertEqual(info["excluded_self_prototype_count"], 1)
        self.assertEqual(info["actual_topk"], 1)
        self.assertTrue(torch.allclose(maximum, torch.tensor([[[0.0, 1.0]]])))
        self.assertTrue(torch.equal(maximum, topk))

    def test_results_json_only_replays_views_and_source_tdc_controls_ranking(self):
        with TemporaryDirectory() as tmp:
            path = Path(tmp) / "results.json"
            path.write_text(
                json.dumps(
                    {
                        "cases": [
                            {
                                "basename": "a.nii.gz",
                                "adaptation_trace": {
                                    "view_selection": {"tdc": [0.6, 0.9]},
                                    "view_params": [
                                        {"scale": 1.0, "offset": 0.0},
                                        {"scale": 1.1, "offset": 0.1},
                                    ],
                                },
                            },
                            {
                                "basename": "b.nii.gz",
                                "adaptation_trace": {
                                    "view_selection": {"tdc": [0.8, 0.2]}
                                },
                            },
                        ]
                    }
                ),
                encoding="utf-8",
            )
            views, order = load_view_params_replay(path)
            source_tdc = {
                "a.nii.gz": {"memory_ranking_value": 0.9},
                "b.nii.gz": {"memory_ranking_value": 0.1},
            }
            selected = select_memory_cases(
                ["a.nii.gz", "b.nii.gz"], None, source_tdc, 0.5
            )
        self.assertEqual(order, ["a.nii.gz", "b.nii.gz"])
        self.assertEqual(selected, ["a.nii.gz"])
        self.assertEqual(len(views["a.nii.gz"]), 2)

    def test_source_tdc_is_recomputed_for_every_view(self):
        class StubExtractor:
            def __init__(self):
                self.calls = 0

            def extract(self, _image, _text, **kwargs):
                self.calls += 1
                self.assert_flags = kwargs
                logits = tuple(torch.ones(2, 2, 2) for _ in range(4))
                return ExtractedVolume(
                    feature=None,
                    logits=logits[0],
                    probability=torch.sigmoid(logits[0]),
                    decoder_logits=logits,
                )

        extractor = StubExtractor()
        prepared = SimpleNamespace(
            image_path=Path("case.nii.gz"), data=torch.zeros(1, 2, 2, 2)
        )
        params = [
            {"scale": 1.0, "offset": 0.0},
            {"scale": 1.1, "offset": 0.0},
            {"scale": 0.9, "offset": 0.1},
        ]
        result = compute_source_tdc_for_case(
            prepared, params, extractor, torch.zeros(1, 1, 2)
        )
        self.assertEqual(extractor.calls, len(params))
        self.assertEqual(result["view_count"], len(params))
        self.assertEqual(len(result["views"]), len(params))
        self.assertEqual(result["original_tdc"], 1.0)
        self.assertEqual(result["mean_tdc"], 1.0)
        self.assertEqual(result["max_tdc"], 1.0)
        self.assertEqual(result["memory_ranking_value"], result["original_tdc"])
        self.assertEqual(
            extractor.assert_flags,
            {"capture_feature": False, "return_decoder_logits": True},
        )

    def test_voxel_normalized_region_prototype_has_unit_norm(self):
        first = torch.zeros(2, 1, 1, 2)
        second = torch.zeros_like(first)
        first[0] = 100.0
        second[1] = 0.01
        feature_mean = aggregate_normalized_view_features([first, second])
        prototype = region_prototype(
            feature_mean, np.ones((1, 1, 2), dtype=bool)
        )
        self.assertTrue(torch.allclose(prototype.norm(), torch.tensor(1.0)))
        self.assertTrue(
            torch.allclose(
                prototype, torch.tensor([2**-0.5, 2**-0.5]), atol=1e-6
            )
        )

    def test_foreground_labels_support_multiple_explicit_ids(self):
        label = np.array([[[0, 1, 2, 3]]])
        selected = foreground_mask(label, [1, 3])
        self.assertTrue(np.array_equal(selected, np.array([[[False, True, False, True]]])))

    def test_flat_layout_reader_is_local_and_independent(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "images").mkdir()
            (root / "labels").mkdir()
            (root / "images" / "case_a.nii.gz").touch()
            (root / "labels" / "case_a.nii.gz").touch()
            split = root / "split.json"
            split.write_text(
                json.dumps({"test_cases": [{"case_name": "case_a"}]}),
                encoding="utf-8",
            )
            entries = read_test_entries(root, split)
        self.assertEqual(entries[0][0], root / "images" / "case_a.nii.gz")
        self.assertEqual(entries[0][1], root / "labels" / "case_a.nii.gz")

    def test_metrics_and_crop_restoration(self):
        target = np.array([0, 0, 1, 1], dtype=bool)
        scores = np.array([0.1, 0.2, 0.8, 0.9])
        ranking = binary_ranking_metrics(scores, target)
        enrichment = top_fraction_enrichment(scores, target, fractions=(0.5,))
        self.assertEqual(ranking["auroc"], 1.0)
        self.assertEqual(ranking["auprc"], 1.0)
        self.assertEqual(enrichment["top_50pct_gt_foreground_precision"], 1.0)
        restored = restore_crop(
            np.ones((2, 2, 2), dtype=np.float32),
            ((1, 3), (1, 3), (1, 3)),
            (4, 4, 4),
        )
        self.assertEqual(restored.shape, (4, 4, 4))
        self.assertEqual(float(restored.sum()), 8.0)

    def test_float_nifti_round_trip_preserves_model_space_and_geometry(self):
        with TemporaryDirectory() as tmp:
            source_path = Path(tmp) / "source.nii.gz"
            output_path = Path(tmp) / "map.nii.gz"
            source_array = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
            affine = np.array(
                [[-2.0, 0.0, 0.0, 10.0], [0.0, 3.0, 0.0, -4.0],
                 [0.0, 0.0, 4.0, 2.0], [0.0, 0.0, 0.0, 1.0]]
            )
            nib.save(nib.Nifti1Image(source_array, affine), source_path)
            reader = NibabelIOWithReorient()
            model_array, _ = reader.read_images([str(source_path)])

            save_float_nifti_model_space(model_array[0], source_path, output_path)

            round_trip, _ = reader.read_images([str(output_path)])
            self.assertTrue(np.array_equal(round_trip[0], model_array[0]))
            saved = nib.load(output_path)
            source = nib.load(source_path)
            self.assertEqual(saved.shape, source.shape)
            self.assertTrue(np.allclose(saved.affine, source.affine))


if __name__ == "__main__":
    unittest.main()
