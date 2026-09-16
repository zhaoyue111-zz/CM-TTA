"""Run the root CM-TTA protocol with the 3-D VoxTell model on P0 test cases.

Only the balanced split's ``test_cases`` are used for adaptation and
evaluation.  Target labels are loaded after adaptation, solely for metrics.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch

from acvl_utils.cropping_and_padding.bounding_boxes import insert_crop_into_image

from data.voxtell_p0 import (
    load_ras_image,
    load_ras_label,
    make_case_adaptation_patches,
    read_image_entries,
)
from method.voxtell_cmtta import (
    VoxTellCMTTA,
    load_cmtta_checkpoint,
    save_cmtta_checkpoint,
)


DEFAULT_VOXTELL_ROOT = Path("/data/zy/VoxTell_from_disk")
DEFAULT_QWEN = Path(
    "/home/SENSETIME/yangtingting/.cache/huggingface/hub/"
    "models--Qwen--Qwen3-Embedding-4B/snapshots/"
    "5cf2132abc99cad020ac570b19d031efec650f2b"
)


class PromptMemory:
    """Quality-weighted prompt snapshots used by online CM-TTA."""

    def __init__(self, max_size: int, fusion_momentum: float):
        self.max_size = int(max_size)
        self.fusion_momentum = float(fusion_momentum)
        self.prompts = []
        self.scores = []

    def __len__(self):
        return len(self.prompts)

    def fuse_into(self, prompt: torch.Tensor) -> torch.Tensor:
        if not self.prompts:
            return prompt
        scores = np.asarray(self.scores, dtype=np.float64)
        weights = np.exp(scores - scores.max())
        weights /= weights.sum()
        history = torch.zeros_like(self.prompts[0], device=prompt.device)
        for weight, saved in zip(weights, self.prompts):
            history.add_(saved.to(prompt.device, dtype=prompt.dtype), alpha=float(weight))
        return (1.0 - self.fusion_momentum) * prompt + self.fusion_momentum * history

    def push(self, prompt: torch.Tensor, score: float) -> None:
        saved = prompt.detach().cpu().clone()
        if saved.ndim == 3:
            saved = saved[0]
        if len(self.prompts) < self.max_size:
            self.prompts.append(saved)
            self.scores.append(float(score))
            return
        worst = int(np.argmin(self.scores))
        if float(score) > self.scores[worst]:
            self.prompts[worst] = saved
            self.scores[worst] = float(score)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_strong_view(image: torch.Tensor) -> torch.Tensor:
    """Aligned 3-D intensity augmentation; spatial geometry stays unchanged."""
    result = image.clone()
    if torch.rand(()) < 0.8:
        result = result * torch.empty((), device=image.device).uniform_(0.85, 1.15)
    if torch.rand(()) < 0.8:
        result = result + torch.empty((), device=image.device).uniform_(-0.15, 0.15)
    if torch.rand(()) < 0.5:
        result = result + torch.randn_like(result) * 0.05
    return result.contiguous()


def binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if pred.shape != target.shape:
        raise ValueError(f"Prediction/label shape mismatch: {pred.shape} vs {target.shape}")
    intersection = np.logical_and(pred, target).sum()
    union = np.logical_or(pred, target).sum()
    pred_sum = pred.sum()
    target_sum = target.sum()
    if pred_sum == 0 and target_sum == 0:
        return {"Dice": 1.0, "mIoU": 1.0}
    return {
        "Dice": float(2.0 * intersection / max(1, pred_sum + target_sum)),
        "mIoU": float(intersection / max(1, union)),
    }


def save_prediction_nifti(prediction: np.ndarray, image_path: Path, output_path: Path) -> None:
    import nibabel as nib

    source = nib.as_closest_canonical(nib.load(str(image_path)))
    if tuple(source.shape[:3]) != tuple(prediction.shape):
        raise ValueError(
            f"Prediction/source shape mismatch for {image_path.name}: "
            f"{prediction.shape} vs {source.shape[:3]}"
        )
    header = source.header.copy()
    header.set_data_dtype(np.uint8)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    nib.save(nib.Nifti1Image(prediction.astype(np.uint8), source.affine, header), str(output_path))


def build_predictor(args):
    root = Path(args.voxtell_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    try:
        from voxtell.inference.predictor import VoxTellPredictor
    except ModuleNotFoundError as error:
        if error.name == "transformers":
            raise RuntimeError(
                "VoxTell requires transformers; install the root requirements "
                "before running the P0 CM-TTA entry"
            ) from error
        raise

    return VoxTellPredictor(
        model_dir=str(Path(args.model_dir).expanduser().resolve()),
        device=args.device,
        text_encoding_model=args.text_model,
    )


def evaluate_case(predictor, image_path: Path, label_path: Path, data, bbox, original_shape,
                  prompt: torch.Tensor, output_dir: Path) -> dict:
    with torch.no_grad():
        logits = predictor.predict_sliding_window_return_logits(
            data, prompt.to(predictor.device)
        ).float().cpu()
    cropped_prediction = (torch.sigmoid(logits) > 0.5).numpy().astype(np.uint8)
    prediction = insert_crop_into_image(
        np.zeros((cropped_prediction.shape[0], *original_shape), dtype=np.uint8),
        cropped_prediction,
        bbox,
    )[0]
    target = np.squeeze(load_ras_label(str(label_path)))
    metrics = binary_metrics(prediction, target)
    stem = image_path.name[:-7] if image_path.name.endswith(".nii.gz") else image_path.stem
    save_prediction_nifti(prediction, image_path, output_dir / f"{stem}.nii.gz")
    return {"basename": image_path.name, **metrics}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Root CM-TTA with VoxTell on P0 test cases")
    parser.add_argument("--data_dir", default="/data/zy/CT_MRI_DATA_3D")
    parser.add_argument("--split_file", default=None)
    parser.add_argument("--voxtell_root", default=str(DEFAULT_VOXTELL_ROOT))
    parser.add_argument("--model_dir", default=str(DEFAULT_VOXTELL_ROOT / "model"))
    parser.add_argument("--text_model", default=str(DEFAULT_QWEN))
    parser.add_argument("--prompt", default="liver")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output_dir", default="results_/voxtell_cmtta_p0")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--seed", type=int, default=1377)
    parser.add_argument("--tta_steps", type=int, default=1)
    parser.add_argument("--selection_p", type=float, default=0.1)
    parser.add_argument("--num_aug_views", type=int, default=9)
    parser.add_argument("--confidence_threshold", type=float, default=0.7)
    parser.add_argument("--w_seg", type=float, default=1.0)
    parser.add_argument("--w_entropy", type=float, default=0.1)
    parser.add_argument("--w_cac", type=float, default=1.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--ema_momentum", type=float, default=0.99)
    parser.add_argument("--amp_init_scale", type=float, default=1024.0)
    parser.add_argument("--print_freq", type=int, default=1)
    parser.add_argument("--online", type=int, default=1)
    parser.add_argument("--use_prompt_memory", type=int, default=1)
    parser.add_argument("--prompt_memory_size", type=int, default=16)
    parser.add_argument("--prompt_fusion_momentum", type=float, default=0.3)
    parser.add_argument("--use_aug_param_memory", type=int, default=1)
    parser.add_argument("--aug_param_memory_size", type=int, default=32)
    parser.add_argument("--n_memory_params", type=int, default=3)
    parser.add_argument("--quality_metric", choices=("cac", "saaf", "tdc"), default="cac")
    parser.add_argument("--quality_mode", choices=("cac", "purity", "completeness", "tse"), default="cac")
    parser.add_argument(
        "--quality_config",
        default=str(Path(__file__).resolve().parent / "configs" / "tse.json"),
    )
    parser.add_argument("--w_quality", type=float, default=0.0)
    parser.add_argument("--enable_recall_recovery", action="store_true")
    parser.add_argument("--recovery_teacher_low", type=float, default=0.4)
    parser.add_argument("--recovery_view_threshold", type=float, default=0.5)
    parser.add_argument("--recovery_min_view_votes", type=int, default=2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.online:
        raise ValueError("This P0 protocol requires --online 1 for cross-case adaptation")
    if args.tta_steps < 1:
        raise ValueError("--tta_steps must be positive")
    if args.quality_metric != "cac" or args.quality_mode != "cac":
        raise ValueError("The root P0 CM-TTA entry currently enables CAC only")
    seed_everything(args.seed)
    args.device = torch.device(args.device)
    if args.device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        torch.cuda.set_device(args.device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "args.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    entries = read_image_entries(args.data_dir, "test", args.split_file)
    predictor = build_predictor(args)
    with torch.no_grad():
        initial_prompt = predictor.embed_text_prompts([args.prompt]).detach()
    qwen_text_encoder = getattr(predictor, "text_backbone", None)
    if qwen_text_encoder is not None:
        qwen_text_encoder.requires_grad_(False)
    adapter = VoxTellCMTTA(
        predictor.network,
        initial_prompt,
        args.device,
        args,
        qwen_text_encoder=qwen_text_encoder,
    )
    if args.checkpoint:
        load_cmtta_checkpoint(args.checkpoint, adapter)
    predictor.network = adapter.model

    prompt_memory = PromptMemory(args.prompt_memory_size, args.prompt_fusion_momentum)
    case_rows = []
    history = []
    predictions_dir = output_dir / "predictions"
    try:
        for case_index, (image_path, label_path) in enumerate(entries, start=1):
            image = load_ras_image(str(image_path))
            data, bbox, original_shape = predictor.preprocess(image)
            patches, _locations, _ = make_case_adaptation_patches(data, predictor.patch_size)

            if args.use_prompt_memory and len(prompt_memory):
                with torch.no_grad():
                    adapter.soft_prompt_embedding.data.copy_(
                        prompt_memory.fuse_into(adapter.soft_prompt_embedding.data)
                    )
            case_values = []
            for patch_index, patch in enumerate(patches, start=1):
                weak = patch.unsqueeze(0)
                strong = make_strong_view(weak)
                for _ in range(args.tta_steps):
                    values = adapter.adapt_batch(weak, strong, [image_path.name])
                    case_values.append(values)
                if patch_index % args.print_freq == 0 or patch_index == len(patches):
                    print(
                        f"case {case_index}/{len(entries)} {image_path.name} "
                        f"patch {patch_index}/{len(patches)} "
                        f"loss={case_values[-1]['loss']:.4f} "
                        f"cac={case_values[-1]['cac']:.4f}"
                    )

            case_quality = float(np.mean([value["quality"] for value in case_values]))
            if args.use_prompt_memory:
                prompt_memory.push(adapter.soft_prompt_embedding, case_quality)
            row = evaluate_case(
                predictor, image_path, label_path, data, bbox, original_shape,
                adapter.soft_prompt_embedding.detach(), predictions_dir
            )
            row["adaptation_quality"] = case_quality
            case_rows.append(row)
            history.append({"case": image_path.name, "patches": len(patches), **case_values[-1]})
            print(
                f"evaluated {image_path.name}: Dice={row['Dice']:.4f} "
                f"mIoU={row['mIoU']:.4f}"
            )
    finally:
        adapter.close()

    average = {
        "Dice": float(np.mean([row["Dice"] for row in case_rows])),
        "mIoU": float(np.mean([row["mIoU"] for row in case_rows])),
    }
    results = {"cases": case_rows, "average": average}
    (output_dir / "results.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    save_cmtta_checkpoint(str(output_dir / "last.pt"), adapter, args, history)
    print(f"Results: Dice={average['Dice']:.4f} mIoU={average['mIoU']:.4f}")


if __name__ == "__main__":
    main()
