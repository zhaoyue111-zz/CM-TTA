from __future__ import annotations

import hashlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import torch
from acvl_utils.cropping_and_padding.padding import pad_nd_image
from nnunetv2.inference.sliding_window_prediction import compute_gaussian

from data.voxtell_p0 import load_ras_image
from method.voxtell_cmtta import VoxTellCMTTA


D5_PREHEAD_MODULE = "network.decoder.stages[-1]"


@dataclass
class PreparedCase:
    image_path: Path
    data: torch.Tensor
    bbox: Sequence[Sequence[int]]
    original_shape: tuple[int, int, int]


@dataclass
class ExtractedVolume:
    feature: torch.Tensor
    logits: torch.Tensor
    probability: torch.Tensor


def build_predictor(
    voxtell_root: str,
    model_dir: str,
    text_model: str,
    device: torch.device,
):
    root = Path(voxtell_root).expanduser().resolve()
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from voxtell.inference.predictor import VoxTellPredictor

    predictor = VoxTellPredictor(
        model_dir=str(Path(model_dir).expanduser().resolve()),
        device=device,
        text_encoding_model=str(Path(text_model).expanduser().resolve()),
    )
    predictor.network = predictor.network.to(device).eval()
    predictor.network.requires_grad_(False)
    predictor.text_backbone.requires_grad_(False)
    return predictor


def _resolve_dtype(name: str, device: torch.device) -> torch.dtype | None:
    if name == "auto":
        return torch.float16 if device.type == "cuda" else None
    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def embed_prompt(
    predictor,
    prompt: str,
    text_device: torch.device,
    text_dtype: str = "auto",
) -> torch.Tensor:
    """Encode one prompt with VoxTell's tokenizer, wrapper and pooling helper."""
    if not prompt.strip():
        raise ValueError("prompt must not be empty")
    if text_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA text_device requested but CUDA is unavailable")
    from voxtell.utils.text_embedding import last_token_pool, wrap_with_instruction

    dtype = _resolve_dtype(text_dtype, text_device)
    backbone = predictor.text_backbone.to(device=text_device, dtype=dtype).eval()
    wrapped = wrap_with_instruction([prompt])
    tokens = predictor.tokenizer(
        wrapped,
        padding=True,
        truncation=True,
        max_length=predictor.max_text_length,
        return_tensors="pt",
    )
    tokens = {key: value.to(text_device) for key, value in tokens.items()}
    with torch.inference_mode():
        hidden = backbone(**tokens).last_hidden_state
        embedding = last_token_pool(hidden, tokens["attention_mask"])
        embedding = embedding.reshape(1, 1, -1).detach().to(predictor.device)
    predictor.text_backbone = backbone.to("cpu")
    if text_device.type == "cuda":
        torch.cuda.empty_cache()
    return embedding


def prepare_case(predictor, image_path: Path) -> PreparedCase:
    image = load_ras_image(str(image_path))
    data, bbox, original_shape = predictor.preprocess(image)
    return PreparedCase(
        image_path=Path(image_path),
        data=data,
        bbox=bbox,
        original_shape=tuple(int(v) for v in original_shape),
    )


def deterministic_view_params(case: str, num_views: int, seed: int) -> list[dict[str, float]]:
    """Reuse CM-TTA's intensity-view sampler with a stable per-case RNG seed."""
    if num_views < 1:
        raise ValueError("num_views must be positive")
    digest = hashlib.sha256(f"{seed}:{case}".encode("utf-8")).digest()
    case_seed = int.from_bytes(digest[:8], "little") % (2**31)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(case_seed)
        return VoxTellCMTTA._sample_intensity_params(num_views - 1)


def apply_intensity_view(
    data: torch.Tensor, param: dict[str, float], view_index: int
) -> torch.Tensor:
    scale = float(param["scale"])
    offset = float(param["offset"])
    if view_index == 0:
        if scale != 1.0 or offset != 0.0:
            raise ValueError("View 0 must be the unaugmented scale=1, offset=0 view")
        return data
    return data * scale + offset


class SlidingD5FeatureExtractor:
    """Gaussian-fuse final D5 pre-head features and logits over a full case."""

    def __init__(self, predictor, print_shapes: bool = True):
        self.predictor = predictor
        self.device = torch.device(predictor.device)
        network = predictor.network
        if not hasattr(network, "decoder") or not hasattr(network.decoder, "stages"):
            raise TypeError("VoxTell network does not expose decoder.stages")
        if len(network.decoder.stages) < 1:
            raise ValueError("VoxTell decoder has no stages")
        if hasattr(network.decoder, "get_output_metadata"):
            metadata = network.decoder.get_output_metadata()
            final = [item for item in metadata if item.get("is_final_output")]
            if (
                len(final) != 1
                or final[0].get("head_type") != "final_mask_einsum"
                or int(final[0].get("internal_stage_index", -1))
                != len(network.decoder.stages) - 1
            ):
                raise RuntimeError(
                    "VoxTell decoder metadata no longer maps its final D5 output "
                    "to stages[-1] followed by final_mask_einsum"
                )
        self.hook_module = network.decoder.stages[-1]
        self.print_shapes = bool(print_shapes)
        self._printed = False

    def extract(
        self, input_image: torch.Tensor, text_embedding: torch.Tensor
    ) -> ExtractedVolume:
        if input_image.ndim != 4:
            raise ValueError(f"Expected input (C,D,H,W), got {tuple(input_image.shape)}")
        if text_embedding.ndim != 3 or text_embedding.shape[1] != 1:
            raise ValueError("Prototype diagnostics require exactly one text prompt")

        network = self.predictor.network.to(self.device).eval()
        padded, revert_padding = pad_nd_image(
            input_image,
            self.predictor.patch_size,
            "constant",
            {"value": 0},
            True,
            None,
        )
        slicers = self.predictor._internal_get_sliding_window_slicers(padded.shape[1:])
        padded_shape = tuple(int(v) for v in padded.shape[1:])
        gaussian = compute_gaussian(
            tuple(self.predictor.patch_size),
            sigma_scale=1.0 / 8,
            value_scaling_factor=10,
            device=torch.device("cpu"),
        ).float()
        logits_sum = torch.zeros(padded_shape, dtype=torch.float32)
        denominator = torch.zeros(padded_shape, dtype=torch.float32)
        feature_sum: torch.Tensor | None = None
        captured: list[torch.Tensor] = []

        def capture(_module, _inputs, output):
            if not torch.is_tensor(output) or output.ndim != 5:
                raise RuntimeError(
                    f"{D5_PREHEAD_MODULE} must return (B,C,D,H,W), got {type(output)}"
                )
            captured.append(output.detach())

        handle = self.hook_module.register_forward_hook(capture)
        try:
            for tile_index, slicer in enumerate(slicers):
                tile = padded[slicer][None].to(self.device, non_blocking=True)
                captured.clear()
                with torch.inference_mode(), torch.autocast(
                    device_type=self.device.type,
                    enabled=self.device.type == "cuda",
                ):
                    output = network(tile, text_embedding.to(self.device))
                if isinstance(output, (list, tuple)):
                    output = output[0]
                if not torch.is_tensor(output) or output.ndim != 5:
                    raise RuntimeError("VoxTell forward did not return 5-D D5 logits")
                if len(captured) != 1:
                    raise RuntimeError(
                        f"Expected one {D5_PREHEAD_MODULE} activation per tile, got {len(captured)}"
                    )
                patch_feature = captured[0][0].float().cpu()
                patch_logits = output[0, 0].float().cpu()
                tile_shape = tuple(int(s.stop - s.start) for s in slicer[1:])
                if tuple(patch_feature.shape[1:]) != tile_shape:
                    raise RuntimeError(
                        "D5 pre-head feature is not full-resolution/aligned with the tile: "
                        f"feature={tuple(patch_feature.shape)} tile={tile_shape}"
                    )
                if tuple(patch_logits.shape) != tile_shape:
                    raise RuntimeError(
                        f"D5 logits shape {tuple(patch_logits.shape)} != tile {tile_shape}"
                    )
                if feature_sum is None:
                    feature_sum = torch.zeros(
                        (int(patch_feature.shape[0]), *padded_shape), dtype=torch.float32
                    )
                feature_sum[(slice(None), *slicer[1:])] += patch_feature * gaussian.unsqueeze(0)
                logits_sum[slicer[1:]] += patch_logits * gaussian
                denominator[slicer[1:]] += gaussian

                if self.print_shapes and not self._printed and tile_index == 0:
                    print(
                        "[prototype-diagnostic] first tensors: "
                        f"preprocessed={tuple(input_image.shape)} "
                        f"padded={tuple(padded.shape)} tile={tuple(tile.shape)} "
                        f"D5_prehead={tuple(captured[0].shape)} "
                        f"D5_logits={tuple(output.shape)}"
                    )
        finally:
            handle.remove()

        if feature_sum is None or not slicers:
            raise RuntimeError("Sliding-window inference produced no tiles")
        if bool((denominator <= 0).any()):
            raise RuntimeError("Sliding-window Gaussian denominator contains uncovered voxels")
        feature_sum.div_(denominator.unsqueeze(0))
        logits_sum.div_(denominator)
        crop = tuple(revert_padding[1:])
        feature = feature_sum[(slice(None), *crop)].contiguous()
        logits = logits_sum[crop].contiguous()
        probability = torch.sigmoid(logits)
        if self.print_shapes and not self._printed:
            print(
                "[prototype-diagnostic] first fused tensors: "
                f"feature={tuple(feature.shape)} logits={tuple(logits.shape)} "
                f"probability={tuple(probability.shape)} tiles={len(slicers)}"
            )
            self._printed = True
        return ExtractedVolume(feature=feature, logits=logits, probability=probability)
