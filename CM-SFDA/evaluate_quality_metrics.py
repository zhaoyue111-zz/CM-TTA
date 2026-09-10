"""Offline GT-only audit of pseudo-label quality metrics for VoxTell.

This script never adapts the model.  It builds TSE prototypes from unlabeled
P0 train cases, then evaluates augmented P0 test-case patches. Ground truth is
used only after inference to calculate Dice and correlations.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from data.sfda_voxtell import load_ras_label, make_target_loader, read_image_entries
from method.sfda_voxtell import VoxTellPromptSFDA, compute_cac_score
from run_sfda_voxtell import build_predictor, seed_everything


def _rankdata(values):
    """Tie-aware ranks without requiring scipy."""
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1) + 1
        start = end
    return ranks


def spearman(values, dice):
    if len(values) < 2:
        return 0.0
    first, second = _rankdata(values), _rankdata(dice)
    if first.std() == 0 or second.std() == 0:
        return 0.0
    return float(np.corrcoef(first, second)[0, 1])


def _crop_by_bbox(array, bbox):
    if bbox is None:
        return array
    slices = tuple(slice(int(bounds[0]), int(bounds[1])) for bounds in bbox)
    return array[slices]


def _center_patch(tensor, patch_size, pad_value=0.0):
    padding = []
    for current, target in zip(reversed(tensor.shape[-3:]), reversed(patch_size)):
        missing = max(0, int(target) - int(current))
        padding.extend((missing // 2, missing - missing // 2))
    if any(padding):
        tensor = F.pad(tensor, padding, value=pad_value)
    starts = [
        max(0, (int(current) - int(target)) // 2)
        for current, target in zip(tensor.shape[-3:], patch_size)
    ]
    slices = tuple(slice(start, start + int(size)) for start, size in zip(starts, patch_size))
    return tensor[(..., *slices)].contiguous()


def _case_patch(predictor, image_path, label_path):
    # Use VoxTell's own preprocessing for the image. Apply its nonzero bbox to
    # the RAS label, then nearest-resize the label to the preprocessed grid.
    from data.sfda_voxtell import load_ras_image

    image = load_ras_image(str(image_path))
    data, bbox, _ = predictor.preprocess(image)
    data = torch.as_tensor(data).float()
    if data.ndim == 3:
        data = data.unsqueeze(0)
    target = np.squeeze(load_ras_label(str(label_path)))
    target = torch.from_numpy(_crop_by_bbox(target, bbox).copy()).float()[None, None]
    target = F.interpolate(target, size=data.shape[-3:], mode="nearest")[0]
    patch_size = tuple(int(value) for value in predictor.patch_size)
    return _center_patch(data, patch_size), _center_patch(target, patch_size)


def _soft_dice_per_view(probability):
    mean_probability = probability.mean(dim=0, keepdim=True).expand_as(probability)
    intersection = (probability * mean_probability).flatten(start_dim=1).sum(dim=1)
    denominator = (
        probability.flatten(start_dim=1).sum(dim=1)
        + mean_probability.flatten(start_dim=1).sum(dim=1)
    )
    return (2 * intersection + 1e-6) / (denominator + 1e-6)


def _dice_per_view(probability, target):
    prediction = probability >= 0.5
    target = target.bool().expand_as(prediction)
    intersection = (prediction & target).flatten(start_dim=1).sum(dim=1).float()
    denominator = (
        prediction.flatten(start_dim=1).sum(dim=1)
        + target.flatten(start_dim=1).sum(dim=1)
    ).float()
    return torch.where(denominator > 0, 2 * intersection / denominator, torch.ones_like(denominator))


def parse_args():
    parser = argparse.ArgumentParser(description="Offline VoxTell quality/Dice correlation audit")
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--voxtell_root", required=True)
    parser.add_argument("--model_dir", required=True)
    parser.add_argument("--prompt", default="prostate")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--quality_config", default=str(Path(__file__).parent / "configs" / "tse.json"))
    parser.add_argument("--num_aug_views", type=int, default=9)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--output", default="results/voxtell_sfda/quality_audit.json")
    parser.add_argument("--seed", type=int, default=1377)
    return parser.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device(args.device)
    predictor = build_predictor(args.model_dir, device, args.voxtell_root)
    with torch.no_grad():
        initial_prompt = predictor.embed_text_prompts([args.prompt]).detach()
    # Supply only adapter fields needed by construction; this script performs
    # no optimization or teacher/model update.
    adapter_args = argparse.Namespace(
        **vars(args),
        quality_mode="tse",
        lr=0.0,
        weight_decay=0.0,
        amp_init_scale=1024.0,
        record_soft_prompt_grad_norm=False,
    )
    adapter = VoxTellPromptSFDA(predictor.network, initial_prompt, device, adapter_args)
    train_loader = make_target_loader(
        args.data_dir,
        tuple(predictor.patch_size),
        args.batch_size,
        args.num_workers,
    )
    adapter.build_prototype_memory(train_loader)

    rows = []
    selected_dice = {name: [] for name in ("confidence", "entropy", "consistency", "cac", "tse")}
    try:
        for image_path, label_path in read_image_entries(args.data_dir, "test"):
            image, target = _case_patch(predictor, image_path, label_path)
            image = image.unsqueeze(0).to(device)
            views = torch.stack(
                [image, *adapter._make_extra_views(image, args.num_aug_views - 1)], dim=1
            )[0]
            adapter._cac_features.clear()
            with torch.no_grad(), torch.autocast(device_type=device.type, enabled=device.type == "cuda"):
                prompt = adapter._text(adapter.initial_soft_prompt, views.shape[0])
                logits = adapter.model(views, prompt)
            probability = torch.sigmoid(logits[:, 0].float())
            if tuple(target.shape[-3:]) != tuple(probability.shape[-3:]):
                target = F.interpolate(target[None], size=probability.shape[-3:], mode="nearest")[0]
            confidence = torch.maximum(probability, 1 - probability).flatten(start_dim=1).mean(dim=1)
            clipped = probability.clamp(1e-6, 1 - 1e-6)
            entropy = -(
                clipped * clipped.log() + (1 - clipped) * (1 - clipped).log()
            ).flatten(start_dim=1).mean(dim=1)
            consistency = _soft_dice_per_view(probability)
            cac = compute_cac_score(
                adapter._cac_features["vision"], adapter._cac_features["text"], logits
            )
            semantic = adapter._semantic_quality(
                adapter._cac_features["vision"], logits, [image_path.name] * views.shape[0]
            )
            dice = _dice_per_view(probability, target.to(device))
            metrics = {
                "confidence": confidence,
                "entropy": entropy,
                "consistency": consistency,
                "cac": cac,
                "tse": semantic["tse"],
            }
            for view_index in range(views.shape[0]):
                rows.append(
                    {
                        "case": image_path.name,
                        "view": int(view_index),
                        "dice": float(dice[view_index].cpu()),
                        **{
                            name: float(value[view_index].detach().cpu())
                            for name, value in metrics.items()
                        },
                    }
                )
            for name, values in metrics.items():
                index = int(values.argmin() if name == "entropy" else values.argmax())
                selected_dice[name].append(float(dice[index].cpu()))
    finally:
        adapter.close()

    dice_values = [row["dice"] for row in rows]
    result = {
        "protocol": "GT is used only for this offline audit; adaptation/prototypes are label-free",
        "correlations": {
            name: spearman([row[name] for row in rows], dice_values)
            for name in selected_dice
        },
        "selected_view_mean_dice": {
            name: float(np.mean(values)) if values else 0.0
            for name, values in selected_dice.items()
        },
        "views": rows,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key != "views"}, indent=2))


if __name__ == "__main__":
    main()
