# VoxTell D5 foreground-prototype diagnostic

This directory is an offline SFDA diagnostic over pre-cropped liver cuboid ROIs.
The protocol is:

`liver cuboid ROI -> frozen VoxTell sliding inference -> sliding-volume TDC -> reliable-case region prototype bank -> leave-one-case-out prototype retrieval -> GT-only offline evaluation`

It does not construct LoRA, EMA, CM-TTA, `ctx_delta`, LSPM, a loss, or an optimizer,
and it never changes model weights or pseudo labels. The fixed prompt is
`"liver cancer"`. Ground truth is loaded only after the prototype bank has been
completed.

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

Input cases are read directly from `<data_dir>/images/*.nii.gz` with same-named files
in `<data_dir>/labels/`. Although each input is already a liver cuboid ROI, VoxTell
still uses its official `predictor.preprocess()`, including `crop_to_nonzero` and
Z-score normalization. All features, TDC values, and prototypes are computed in this
official preprocessed space, then aligned to the original liver cuboid through the
saved bounding box. The diagnostic does not run a new liver segmentation step.

For every case/view, `SlidingD5FeatureExtractor` uses the predictor's own patch size,
50% overlap slicers and nnUNet Gaussian (`sigma_scale=1/8`, scale 10). Logits and all
feature channels are fused using the same spatial Gaussian denominator. Padding is
removed using `pad_nd_image`'s returned slices; no spatial reshape is used.

## Prototype bank

Memory cases are chosen either explicitly with `--memory_cases` or by descending
source-model TDC. TDC is always recomputed here with the same frozen VoxTell, fixed
prompt and views used for feature extraction. For every view, D5-D2 logits are
restored over the complete preprocessed volume with the same sliding-window Gaussian fusion, then
reduced by `tdc_patch_components`/`tdc_from_components`. This is recorded as
`tdc_type="sliding_volume_tdc"`; it is not CM-TTA non-overlap patch-accumulation TDC.
`source_tdc.json` records each case's original-view, view-mean and view-maximum TDC;
memory ranking defaults to `original_tdc`.

By default, views are generated from a fixed per-case seed. `--results_json` remains
optional and is used **only** to replay
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

Run from the CM-TTA repository root:

```bash
python -m prototype_diagnostic.run_diagnostic \
  --data_dir /path/to/liver_cuboid_roi \
  --voxtell_root /home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk \
  --model_dir /home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/model \
  --text_model /home/ubuntu/zy/pythoncode/sfda/VoxTell_from_disk/model/Qwen3-Embedding-4B \
  --prompt "liver cancer" \
  --foreground_labels 1 \
  --memory_top_ratio 0.2 \
  --num_views 10 \
  --view_support 0.8 \
  --min_region_voxels 64 \
  --core_erosion 1 \
  --topk 5 \
  --device cuda:0 \
  --text_device cpu \
  --output_dir prototype_diagnostic/outputs/liver_roi_prototypes \
  --fail_if_output_exists
```

To choose memory cases manually, pass, for example,
`--memory_cases case_a.nii.gz case_b.nii.gz`. `--split_file` can optionally restrict
the discovered flat-layout cases. `--query_cases` can restrict evaluation; otherwise
all paired ROIs are evaluated. `--foreground_labels` accepts one or more integer IDs
(for example `--foreground_labels 1 3`) and is used only after bank construction.

## Outputs and interpretation

- `prototype_bank.pt`: region prototypes and bank provenance.
- `source_tdc.json`: source-state per-view TDC plus original/mean/max summaries and
  the exact ranking value.
- `bank_region_quality.csv`: post-hoc GT intersection, purity, and GT coverage for
  every already-created prototype region. These values never affect bank selection.
- `per_case_metrics.csv`: flattened per-case distributions, AUROC/AUPRC, and rank
  enrichment, plus ROI shape, sliding patch count, tumor voxels/prevalence, and the
  zero-foreground flag.
- `summary.json`: all cases plus separate `pred_fg_voxels > 0` and
  `pred_fg_voxels == 0` groups.
- `cases/<case>/`: `metrics.json`, `diagnostic.png`, `seg_prob.nii.gz`,
  `proto_sim_max.nii.gz`, and `proto_sim_topk.nii.gz`.

Similarity evaluation runs in official VoxTell preprocessed space. Saved maps are
restored through the preprocessing bbox to the original liver cuboid shape and affine.
If a query is also a memory case, every prototype from that same case is excluded
before matching.

The recovery hypothesis is supported when zero-foreground cases still show:

- prototype AUROC clearly above 0.5 and useful AUPRC relative to foreground prevalence;
- positive foreground-minus-background similarity gaps;
- high GT precision/recall within the top 0.1%-10% similarity voxels;
- materially stronger prototype metrics than segmentation-probability metrics.

It is weakened or rejected when prototype AUROC remains near/below 0.5, gaps are
non-positive, and top-ranked similarity voxels do not enrich GT foreground—especially
when this persists in the `zero_foreground_cases` group.
