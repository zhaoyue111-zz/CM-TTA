# VoxTell D5 foreground-prototype diagnostic

This directory is an inference-only experiment. It does not construct CM-TTA,
`ctx_delta`, a loss, or an optimizer, and it never changes model weights or pseudo
labels. Ground truth is loaded only after the prototype bank has been completed.

## What is extracted

The real VoxTell decoder computes its final D5 logits as:

```python
x = decoder.stages[-1](...)
seg_pred = torch.einsum("b c h w d, b n c -> b n h w d", x, final_mask_embedding)
```

The hook target is therefore `network.decoder.stages[-1]`. Its output is the final
full-resolution `(B, C, D, H, W)` voxel feature immediately before the D5
segmentation einsum. It already receives text-conditioned information propagated
from preceding decoder-stage fusion, while remaining upstream of the final
foreground logit projection.

For every case/view, `SlidingD5FeatureExtractor` uses the predictor's own patch size,
50% overlap slicers and nnUNet Gaussian (`sigma_scale=1/8`, scale 10). Logits and all
feature channels are fused using the same spatial Gaussian denominator. Padding is
removed using `pad_nd_image`'s returned slices; no spatial reshape is used.

## Prototype bank

Memory cases are chosen either explicitly with `--memory_cases` or by descending
source-model TDC. TDC is always recomputed here with the same frozen VoxTell, fixed
prompt and views used for feature extraction. For every view, D5-D2 logits are
restored over the complete preprocessed case with the same sliding-window Gaussian
fusion, then reduced by CM-TTA's `tdc_patch_components`/`tdc_from_components`.
`source_tdc.json` records each case's original-view, view-mean and view-maximum TDC;
memory ranking defaults to `original_tdc`.

`--results_json` is optional and is used **only** to replay
`adaptation_trace.view_params`. Its saved TDC and adapted model/ctx state are ignored.
Cases without replay parameters use the original CM-TTA intensity sampler under a
deterministic per-case seed.

For each memory case, hard foreground masks from all views are combined into a
voxel support fraction. Voxels at or above `--view_support` are split with 26-connected
3-D components. Small components are dropped, optional erosion defines a core, and
each view's D5 feature is first L2-normalized along the channel dimension at every
voxel. Normalized view features are averaged, the core is spatially averaged, and
the final region vector is L2-normalized again. No label is read or consulted.

Each saved prototype contains `feature`, `case_name`, `region_id`, `region_size`,
`tdc`, and `view_support_mean` (plus `core_size`).

## Example

Run from the CM-TTA repository root. The existing results file below is optional and
only makes this diagnostic reuse exactly the same augmentation parameters:

```bash
python -m prototype_diagnostic.run_diagnostic \
  --data_dir /data/hilab/datasets/WCH_MRI145/HBP \
  --split_file /data/hilab/datasets/WCH_MRI145/HBP/split.json \
  --voxtell_root /home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk \
  --model_dir /home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/model \
  --text_model /home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/model/Qwen3-Embedding-4B \
  --prompt "liver cancer" \
  --foreground_labels 1 \
  --results_json outputs/HBP_nonoverlap_original/tdc/results.json \
  --memory_top_ratio 0.2 \
  --num_views 10 \
  --view_support 0.8 \
  --min_region_voxels 64 \
  --core_erosion 1 \
  --topk 5 \
  --device cuda:0 \
  --text_device cpu \
  --output_dir prototype_diagnostic/outputs/HBP_tdc_prototypes \
  --fail_if_output_exists
```

To choose memory cases manually, pass, for example,
`--memory_cases case_a.nii.gz case_b.nii.gz`. Omitting `--results_json` generates a
deterministic view set locally. Source TDC is still reported for every split case.
`--query_cases` can restrict evaluation; otherwise split test cases are evaluated in
their original order. `--foreground_labels` accepts one or more integer IDs (for
example `--foreground_labels 1 3`) and is used only after bank construction.

## Outputs and interpretation

- `prototype_bank.pt`: region prototypes and bank provenance.
- `source_tdc.json`: source-state per-view TDC plus original/mean/max summaries and
  the exact ranking value.
- `per_case_metrics.csv`: flattened per-case distributions, AUROC/AUPRC, and rank
  enrichment.
- `summary.json`: all cases plus separate `pred_fg_voxels > 0` and
  `pred_fg_voxels == 0` groups.
- `cases/<case>/`: `metrics.json`, `diagnostic.png`, `seg_prob.nii.gz`,
  `proto_sim_max.nii.gz`, and `proto_sim_topk.nii.gz`.

Similarity evaluation is restricted to the preprocessed nonzero crop, where D5
features exist. Saved NIfTIs are restored to source shape/affine; voxels outside that
crop are zero. If a query is also a memory case, every prototype from that same case
is excluded before matching.

The recovery hypothesis is supported when zero-foreground cases still show:

- prototype AUROC clearly above 0.5 and useful AUPRC relative to foreground prevalence;
- positive foreground-minus-background similarity gaps;
- high GT precision/recall within the top 0.1%-10% similarity voxels;
- materially stronger prototype metrics than segmentation-probability metrics.

It is weakened or rejected when prototype AUROC remains near/below 0.5, gaps are
non-positive, and top-ranked similarity voxels do not enrich GT foreground—especially
when this persists in the `zero_foreground_cases` group.
