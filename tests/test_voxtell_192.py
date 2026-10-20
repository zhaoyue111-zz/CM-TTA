import json
import types
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import nibabel as nib
import numpy as np
import torch

from data.voxtell_192 import (
    TARGET_SHAPE,
    PreprocessedCase,
    SourceCase,
    bbox_slices,
    crop_with_bbox,
    load_case_artifact,
    load_preprocessed_directory,
    load_preprocessed_manifest,
    load_source_directory,
    make_single_spatial_patch,
    preprocess_source_case,
    restore_probability_to_original,
)
from run_voxtell_192_comparison import (
    FORCED_ORIGINAL_CMTTA,
    SELECTOR_CONFIGS,
    adapt_prepared_case,
    adapter_initial_state,
    assert_fresh_adapter,
    generate_view_params_manifest,
    frozen_model_version_signature,
    evaluate_selection_context_views,
    load_view_params_manifest,
    official_window_count,
    prepare_adaptation_case,
    summarize_cases,
    summarize_selection_context_views,
    write_combined_outputs,
)
from method.voxtell_cmtta import VoxTellCMTTA


class _OneWindowPredictor:
    patch_size = TARGET_SHAPE
    tile_step_size = 0.5

    @staticmethod
    def _internal_get_sliding_window_slicers(image_size):
        assert tuple(image_size) == TARGET_SHAPE
        return [(slice(None), slice(0, 192), slice(0, 192), slice(0, 192))]


class _SpyAdapter:
    def __init__(self):
        self.optimizer_step_count = 0
        self.prepare_calls = []
        self.adapt_calls = []

    def prepare_case(self, patches, valid_masks, **kwargs):
        self.prepare_calls.append((patches, valid_masks, kwargs))
        return {
            "patches": patches,
            "valid_masks": valid_masks,
            "params": kwargs["view_params_override"],
            "patch_locations": kwargs["patch_locations"],
        }

    def adapt_case(self, patches, valid_masks, **kwargs):
        self.adapt_calls.append((patches, valid_masks, kwargs))
        self.optimizer_step_count += 1
        self.long_delta = torch.ones(1)
        return {
            "optimizer_steps_for_case": 1,
            "successful_optimizer_steps_for_case": 1,
            "optimizer_step_skipped": False,
            "spatial_patch_count": len(patches),
            "num_patches": len(patches),
        }


class _FreshAdapter:
    def __init__(self):
        self.ctx_delta = torch.nn.Parameter(torch.zeros(2, 3))
        self.long_delta = None
        self.short_delta = None
        self.short_memory = []
        self.optimizer = types.SimpleNamespace(state={})
        self.optimizer_step_count = 0
        self.use_lora = False
        self.trainable_parameters = [self.ctx_delta]


class _PromptTokenizer:
    def __call__(self, _texts, **kwargs):
        if kwargs.get("add_special_tokens") is False:
            return {"input_ids": torch.tensor([[11, 12]])}
        return {
            "input_ids": torch.tensor([[1, 10, 11, 12, 2]]),
            "attention_mask": torch.ones((1, 5), dtype=torch.long),
            "offset_mapping": torch.tensor(
                [[[0, 0], [0, 6], [7, 12], [13, 19], [20, 26]]]
            ),
        }


class _PromptEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding = torch.nn.Embedding(32, 4)

    def get_input_embeddings(self):
        return self.embedding


def _summary_case(offset=0.0):
    row = {
        "selected_view_hits_oracle": offset == 0.0,
        "selected_view_Dice_regret_to_oracle": 0.1 + offset,
    }
    fields = (
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
    row.update({field: index / 20.0 + offset for index, field in enumerate(fields)})
    row["foreground_erased_by_resize"] = False
    return row


def test_source_directory_pairs_images_with_same_named_labels_without_manifest():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        images_dir = root / "images"
        labels_dir = root / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        for filename in ("0002.nii.gz", "0001.nii.gz"):
            (images_dir / filename).touch()
            (labels_dir / filename).touch()

        cases = load_source_directory(images_dir)

        assert [case.case_id for case in cases] == ["0001", "0002"]
        assert [case.image_path for case in cases] == [
            images_dir / "0001.nii.gz",
            images_dir / "0002.nii.gz",
        ]
        assert [case.gt_path for case in cases] == [
            labels_dir / "0001.nii.gz",
            labels_dir / "0002.nii.gz",
        ]


def test_source_directory_reports_missing_same_named_label():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        images_dir = root / "images"
        labels_dir = root / "labels"
        images_dir.mkdir()
        labels_dir.mkdir()
        (images_dir / "missing-label.nii.gz").touch()

        try:
            load_source_directory(images_dir)
        except FileNotFoundError as error:
            assert str(labels_dir / "missing-label.nii.gz") in str(error)
        else:
            raise AssertionError("Directory discovery accepted an image without its GT")


def test_preprocessed_directory_discovers_artifacts_without_manifest():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        source_image = root / "images" / "0001.nii.gz"
        source_gt = root / "labels" / "0001.nii.gz"
        source_image.parent.mkdir()
        source_gt.parent.mkdir()
        source_image.touch()
        source_gt.touch()
        artifact_dir = root / "offline" / "cases"
        artifact_dir.mkdir(parents=True)
        artifact_path = artifact_dir / "0001.npz"
        np.savez(
            artifact_path,
            case_id=np.asarray("0001"),
            image_path=np.asarray(str(source_image)),
            gt_path=np.asarray(str(source_gt)),
        )

        entries = load_preprocessed_directory(root / "offline")

        assert len(entries) == 1
        assert entries[0].case_id == "0001"
        assert entries[0].artifact_path == artifact_path
        assert entries[0].image_path == source_image
        assert entries[0].gt_path == source_gt


def test_offline_preprocessing_is_exact_192_binary_and_records_geometry():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        image = np.zeros((9, 8, 7), dtype=np.float32)
        image[2:8, 1:7, 2:6] = np.arange(6 * 6 * 4, dtype=np.float32).reshape(6, 6, 4) + 1
        gt = np.zeros_like(image, dtype=np.uint8)
        gt[3:7, 2:6, 3:5] = 7
        affine = np.array(
            [[1.2, 0, 0, 11], [0, 1.3, 0, -4], [0, 0, 2.5, 7], [0, 0, 0, 1]],
            dtype=np.float64,
        )
        image_path = root / "image.nii.gz"
        gt_path = root / "gt.nii.gz"
        nib.save(nib.Nifti1Image(image, affine), image_path)
        nib.save(nib.Nifti1Image(gt, affine), gt_path)
        row = preprocess_source_case(
            SourceCase("case-a", image_path, gt_path), root / "offline", [7]
        )
        artifact = load_case_artifact(row["preprocessed_path"])
        assert artifact["image_192"].shape == (1, 192, 192, 192)
        assert artifact["image_192"].dtype == np.float32
        assert artifact["gt_192"].shape == (1, 192, 192, 192)
        assert artifact["gt_192"].dtype == np.uint8
        assert set(np.unique(artifact["gt_192"])) == {0, 1}
        assert artifact["original_foreground_voxel_count"] == int(
            np.count_nonzero(gt == 7)
        )
        assert artifact["cropped_foreground_voxel_count"] > 0
        assert artifact["gt_192_foreground_voxel_count"] > 0
        assert artifact["foreground_erased_by_resize"] is False
        assert np.allclose(artifact["original_affine"], affine)
        assert tuple(row["cropped_shape"]) == tuple(
            stop - start for start, stop in artifact["bbox"]
        )


def test_image_bbox_crops_gt_and_restores_in_identical_coordinates():
    original = np.zeros((7, 8, 9), dtype=np.uint8)
    original[2:5, 1:7, 3:8] = 1
    bbox = [[1, 6], [1, 7], [2, 9]]
    cropped = crop_with_bbox(original, bbox)
    restored = np.zeros_like(original)
    restored[bbox_slices(bbox)] = cropped
    expected = np.zeros_like(original)
    expected[bbox_slices(bbox)] = original[bbox_slices(bbox)]
    assert np.array_equal(restored, expected)


def test_192_input_has_one_spatial_patch_and_one_official_window():
    image = np.zeros((1, 192, 192, 192), dtype=np.float32)
    patches, masks, locations = make_single_spatial_patch(image)
    assert len(patches) == len(masks) == len(locations) == 1
    assert locations == [(0, 0, 0)]
    assert torch.all(masks[0] == 1)
    assert official_window_count(_OneWindowPredictor(), patches[0]) == 1


def test_one_optimizer_step_and_gt_never_enters_adapter_calls():
    adapter = _SpyAdapter()
    image = np.zeros((1, 192, 192, 192), dtype=np.float32)
    params = [{"scale": 1.0, "offset": 0.0}] + [
        {"scale": 1.0 + index * 0.01, "offset": -index * 0.01}
        for index in range(1, 10)
    ]
    gt_sentinel = object()
    prepared = prepare_adaptation_case(adapter, image, params)
    trace = adapt_prepared_case(adapter, prepared, "case-a")
    assert trace["optimizer_steps_for_case"] == 1
    assert trace["successful_optimizer_steps_for_case"] == 1
    assert adapter.optimizer_step_count == 1
    for call in adapter.prepare_calls + adapter.adapt_calls:
        positional, masks, kwargs = call
        assert all(value is not gt_sentinel for value in positional)
        assert all(value is not gt_sentinel for value in masks)
        assert all(value is not gt_sentinel for value in kwargs.values())


def test_selection_metrics_use_short_ctx_when_raw_ctx_delta_differs():
    image = np.zeros((1, 1, 1, 1), dtype=np.float32)
    gt = np.ones((1, 1, 1, 1), dtype=np.uint8)
    params = [
        {"scale": 1.0, "offset": 0.0},
        {"scale": 1.0, "offset": 1.0},
    ]
    raw_ctx_text = torch.tensor([[[1.0]]])
    short_ctx_text = torch.tensor([[[2.0]]])

    def fake_probability(_predictor, view, text_feature, **_kwargs):
        is_augmented = float(np.asarray(view).mean()) > 0.5
        uses_short = float(text_feature.item()) == 2.0
        foreground = is_augmented if uses_short else not is_augmented
        return np.ones((1, 1, 1), dtype=np.float32) if foreground else np.zeros(
            (1, 1, 1), dtype=np.float32
        )

    with patch(
        "run_voxtell_192_comparison.predict_official_probability",
        side_effect=fake_probability,
    ):
        short_metrics = evaluate_selection_context_views(
            object(), image, gt, short_ctx_text, params
        )
        raw_metrics = evaluate_selection_context_views(
            object(), image, gt, raw_ctx_text, params
        )
    short_outcomes = summarize_selection_context_views(short_metrics, selected_view=1)
    raw_outcomes = summarize_selection_context_views(raw_metrics, selected_view=1)
    assert short_outcomes["oracle_best_view"] == 1
    assert short_outcomes["selected_view_Dice"] == 1.0
    assert short_outcomes["selected_view_Dice_regret_to_oracle"] == 0.0
    assert raw_outcomes["oracle_best_view"] == 0
    assert raw_outcomes["selected_view_Dice"] == 0.0


def test_preprocessing_fails_when_positive_gt_is_erased_by_192_resize():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        image = np.ones((4, 4, 4), dtype=np.float32)
        gt = np.zeros_like(image, dtype=np.uint8)
        gt[1, 1, 1] = 9
        image_path = root / "image.nii.gz"
        gt_path = root / "gt.nii.gz"
        nib.save(nib.Nifti1Image(image, np.eye(4)), image_path)
        nib.save(nib.Nifti1Image(gt, np.eye(4)), gt_path)
        empty_image = np.zeros((1, *TARGET_SHAPE), dtype=np.float32)
        empty_gt = np.zeros((1, *TARGET_SHAPE), dtype=np.uint8)
        try:
            with patch(
                "data.voxtell_192.resize_image_and_gt",
                return_value=(empty_image, empty_gt),
            ):
                preprocess_source_case(
                    SourceCase("tiny-tumor", image_path, gt_path),
                    root / "offline",
                    [9],
                )
        except ValueError as error:
            assert "tiny-tumor" in str(error)
            assert "erased all foreground" in str(error)
        else:
            raise AssertionError("Preprocessing accepted a foreground-erased GT")


def test_amp_skipped_step_does_not_commit_success_count_or_lspm_state():
    class SkippingScaler:
        def __init__(self):
            self.scale = 1024.0

        def step(self, _optimizer):
            return None

        def update(self):
            self.scale = 512.0

        def get_scale(self):
            return self.scale

    class FakeOptimizer:
        def __init__(self):
            self.zero_grad_calls = 0

        def zero_grad(self, set_to_none=False):
            assert set_to_none is True
            self.zero_grad_calls += 1

    short_delta = torch.tensor([1.0])
    long_delta = torch.tensor([2.0])
    memory = ["unchanged"]
    dummy = types.SimpleNamespace(
        scaler=SkippingScaler(),
        optimizer=FakeOptimizer(),
        optimizer_step_count=3,
        short_delta=short_delta,
        long_delta=long_delta,
        short_memory=memory,
    )
    try:
        VoxTellCMTTA._perform_successful_optimizer_step(dummy, 1024.0)
    except RuntimeError as error:
        assert "not adapted" in str(error)
    else:
        raise AssertionError("AMP-skipped optimizer step did not fail")
    assert dummy.optimizer_step_count == 3
    assert dummy.short_delta is short_delta
    assert dummy.long_delta is long_delta
    assert dummy.short_memory is memory and dummy.short_memory == ["unchanged"]
    assert dummy.optimizer.zero_grad_calls == 1


def test_four_configs_share_order_and_params_but_adapter_states_are_isolated():
    assert [(row["view_selection_metric"], row["use_entropy_rank"]) for row in SELECTOR_CONFIGS] == [
        ("cac", False),
        ("cac", True),
        ("tdc", False),
        ("tdc", True),
    ]
    assert FORCED_ORIGINAL_CMTTA["pseudo_update_mode"] == "original"
    assert FORCED_ORIGINAL_CMTTA["alignment_loss"] == "cac"
    assert FORCED_ORIGINAL_CMTTA["response_weight"] == 0.0
    assert FORCED_ORIGINAL_CMTTA["use_lora"] is False
    with TemporaryDirectory() as tmp:
        path = Path(tmp) / "views.json"
        case_order = ["b", "a", "c"]
        generate_view_params_manifest(case_order, 123, path)
        first = load_view_params_manifest(path, case_order)
        second = load_view_params_manifest(path, case_order)
        assert first == second
        assert json.loads(path.read_text())["case_order"] == case_order

    adapters = [_FreshAdapter() for _ in SELECTOR_CONFIGS]
    reference = assert_fresh_adapter(adapters[0])
    for adapter in adapters[1:]:
        assert_fresh_adapter(adapter, reference)
    adapters[0].ctx_delta.data.add_(1)
    adapters[0].long_delta = torch.ones(1)
    adapters[0].optimizer.state["changed"] = True
    assert torch.all(adapters[1].ctx_delta == 0)
    assert adapters[1].long_delta is None
    assert adapters[1].optimizer.state == {}
    assert adapter_initial_state(adapters[1]) == reference

    model = torch.nn.Linear(2, 2)
    model_signature = frozen_model_version_signature(model)
    with torch.no_grad():
        model.weight.add_(1)
    assert frozen_model_version_signature(model) != model_signature


def test_qwen_context_accepts_liver_cancer_and_tracks_all_prompt_tokens():
    dummy = types.SimpleNamespace(
        text_prompt="liver cancer",
        args=types.SimpleNamespace(max_text_length=64),
        device=torch.device("cpu"),
        qwen_text_encoder=_PromptEncoder(),
    )
    VoxTellCMTTA._initialize_qwen_context(
        dummy,
        _PromptTokenizer(),
        initial_ctx=None,
        formatted_text_prompt="prefix liver cancer suffix",
    )
    assert dummy.n_ctx == 2
    assert dummy._liver_token_indices.tolist() == [2, 3]
    assert tuple(dummy.ctx_delta.shape) == (2, 4)


def test_summary_is_recomputed_as_case_macro_average():
    rows = [_summary_case(0.0), _summary_case(0.2)]
    summary = summarize_cases(rows)
    assert summary["case_count"] == 2
    assert summary["selected_view_oracle_hit_count"] == 1
    assert summary["selected_view_oracle_hit_rate"] == 0.5
    assert np.isclose(
        summary["mean_all_views_max_Dice"],
        np.mean([row["all_views_max_Dice"] for row in rows]),
    )
    assert np.isclose(summary["mean_selected_view_Dice_regret_to_oracle"], 0.2)
    assert summary["foreground_erased_by_resize_case_count"] == 0


def test_combined_outputs_align_cases_and_include_each_final_macro_average():
    case_order = ["case-2", "case-1"]
    per_config = {}
    for config_index, config in enumerate(SELECTOR_CONFIGS):
        cases = []
        for case_index, case_id in enumerate(case_order):
            row = _summary_case(config_index * 0.01 + case_index * 0.02)
            row.update({"case_id": case_id, "selected_view": case_index})
            cases.append(row)
        per_config[config["name"]] = {
            "case_order": list(case_order),
            "cases": cases,
            "summary": summarize_cases(cases),
        }
    with TemporaryDirectory() as tmp:
        output = Path(tmp)
        write_combined_outputs(output, per_config)
        payload = json.loads(
            (output / "combined_selector_comparison.json").read_text()
        )
        assert payload["case_order"] == case_order
        assert [row["case_id"] for row in payload["cases"]] == case_order
        assert set(payload["final_average_all_test_cases"]) == {
            config["name"] for config in SELECTOR_CONFIGS
        }
        csv_text = (output / "combined_selector_comparison.csv").read_text()
        assert "CAC_final_mean_Dice_all_test_cases" in csv_text
        assert "TDC+Entropy_final_mean_mIoU_all_test_cases" in csv_text


def test_soft_192_probability_restores_shape_bbox_and_preserves_soft_interpolation():
    probability = np.zeros(TARGET_SHAPE, dtype=np.float32)
    probability[96:] = 1.0
    bbox = [[2, 7], [3, 9], [1, 8]]
    restored = restore_probability_to_original(probability, bbox, (10, 12, 11))
    assert restored.shape == (10, 12, 11)
    outside = restored.copy()
    outside[bbox_slices(bbox)] = 0
    assert np.count_nonzero(outside) == 0
    crop = restored[bbox_slices(bbox)]
    assert crop.shape == (5, 6, 7)
    assert np.any((crop > 0.0) & (crop < 1.0))


def test_preprocessed_manifest_preserves_explicit_case_order_and_paths():
    with TemporaryDirectory() as tmp:
        root = Path(tmp)
        rows = []
        for case_id in ("case-z", "case-a"):
            artifact = root / f"{case_id}.npz"
            image = root / f"{case_id}-image.nii.gz"
            gt = root / f"{case_id}-gt.nii.gz"
            artifact.touch()
            image.touch()
            gt.touch()
            rows.append(
                {
                    "case_id": case_id,
                    "preprocessed_path": artifact.name,
                    "image_path": image.name,
                    "gt_path": gt.name,
                }
            )
        manifest = root / "manifest.json"
        manifest.write_text(json.dumps({"cases": rows}), encoding="utf-8")
        entries = load_preprocessed_manifest(manifest)
        assert [entry.case_id for entry in entries] == ["case-z", "case-a"]
        assert all(isinstance(entry, PreprocessedCase) for entry in entries)


def load_tests(_loader, _tests, _pattern):
    """Let the repository's unittest-only VoxTell environment run pytest-style tests."""
    suite = unittest.TestSuite()
    for name, value in sorted(globals().items()):
        if name.startswith("test_") and callable(value):
            suite.addTest(unittest.FunctionTestCase(value, description=name))
    return suite


if __name__ == "__main__":
    unittest.main()
