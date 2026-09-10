"""Cross-case target semantic evidence (TSE) for VoxTell SFDA.

The prototype bank stores sums/counts rather than already averaged vectors so
that a case can be removed exactly when constructing leave-one-case-out (LOO)
prototypes.  The leading prototype axis is kept even though v1 uses K=1.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F


QUALITY_MODES = ("cac", "purity", "completeness", "tse")


def load_quality_config(path):
    """Load and validate the JSON configuration used by the quality module."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    required = {
        "feature_layer",
        "text_feature_layer",
        "foreground_probability_threshold",
        "background_probability_threshold",
        "foreground_text_similarity_threshold",
        "background_text_similarity_threshold",
        "probability_stability_threshold",
        "similarity_stability_threshold",
        "temperature",
        "epsilon",
        "seed_views",
        "num_prototypes",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise KeyError(f"Quality config {path} is missing: {', '.join(missing)}")
    if int(config["num_prototypes"]) != 1:
        raise ValueError("TSE v1 implements one foreground/background prototype")
    if int(config["seed_views"]) < 2:
        raise ValueError("seed_views must be at least 2 to measure stability")
    if float(config["temperature"]) <= 0 or float(config["epsilon"]) <= 0:
        raise ValueError("temperature and epsilon must be positive")
    if not 0 <= float(config["background_probability_threshold"]) <= 1:
        raise ValueError("background_probability_threshold must be in [0, 1]")
    if not 0 <= float(config["foreground_probability_threshold"]) <= 1:
        raise ValueError("foreground_probability_threshold must be in [0, 1]")
    if float(config["background_probability_threshold"]) > float(
        config["foreground_probability_threshold"]
    ):
        raise ValueError("background probability threshold must not exceed foreground")
    for name in (
        "foreground_text_similarity_threshold",
        "background_text_similarity_threshold",
    ):
        if not -1 <= float(config[name]) <= 1:
            raise ValueError(f"{name} must be in [-1, 1]")
    for name in ("probability_stability_threshold", "similarity_stability_threshold"):
        if float(config[name]) < 0:
            raise ValueError(f"{name} must be non-negative")
    return config


def _vision_channels_first(vision_features):
    if not torch.is_tensor(vision_features) or vision_features.ndim != 5:
        shape = getattr(vision_features, "shape", None)
        raise ValueError(f"Expected vision features (B,H,W,D,C), got {shape}")
    return F.normalize(vision_features.float(), dim=-1)


def text_similarity_map(vision_features, text_features):
    """Return cosine target-text similarity on the VoxTell feature grid."""
    vision = _vision_channels_first(vision_features)
    if not torch.is_tensor(text_features) or text_features.ndim != 3:
        shape = getattr(text_features, "shape", None)
        raise ValueError(f"Expected text features (N,B,C), got {shape}")
    if text_features.shape[1] != vision.shape[0] or text_features.shape[2] != vision.shape[-1]:
        raise ValueError("Text and vision feature batch/channel dimensions must agree")
    # VoxTell currently supplies one fixed target prompt. Mean pooling keeps the
    # helper well-defined if its projection exposes more than one prompt token.
    text = F.normalize(text_features.float().mean(dim=0), dim=-1)
    return (vision * text[:, None, None, None, :]).sum(dim=-1)


def probability_on_feature_grid(logits, spatial_shape):
    """Convert VoxTell (B,N,D,H,W) logits to feature-grid (B,H,W,D) probabilities."""
    if not torch.is_tensor(logits) or logits.ndim != 5:
        shape = getattr(logits, "shape", None)
        raise ValueError(f"Expected VoxTell logits (B,N,D,H,W), got {shape}")
    probability = torch.sigmoid(logits[:, 0].float()).permute(0, 2, 3, 1)
    if tuple(probability.shape[1:]) != tuple(spatial_shape):
        probability = F.interpolate(
            probability.unsqueeze(1),
            size=tuple(spatial_shape),
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)
    # Out-of-place clamp is required when Q is used as a differentiable loss;
    # mutating SigmoidBackward's output would invalidate autograd versioning.
    return probability.clamp(0.0, 1.0)


def extract_case_seed_statistics(
    vision_features,
    text_features,
    logits,
    batch_size,
    num_views,
    config,
):
    """Aggregate stable foreground/background seed features for each case.

    All current augmentations are intensity-only, hence they are already in the
    weak-view coordinate system. A future spatial augmentation must be inverted
    before calling this function.
    """
    expected = int(batch_size) * int(num_views)
    if vision_features.shape[0] != expected or logits.shape[0] != expected:
        raise ValueError("Flattened seed-view batch does not match batch_size*num_views")
    channels = vision_features.shape[-1]
    spatial = vision_features.shape[1:-1]
    vision = _vision_channels_first(vision_features).view(
        batch_size, num_views, *spatial, channels
    )
    similarity = text_similarity_map(vision_features, text_features).view(
        batch_size, num_views, *spatial
    )
    probability = probability_on_feature_grid(logits, spatial).view(
        batch_size, num_views, *spatial
    )

    mean_probability = probability.mean(dim=1)
    mean_similarity = similarity.mean(dim=1)
    probability_stable = probability.std(dim=1, unbiased=False) <= float(
        config["probability_stability_threshold"]
    )
    similarity_stable = similarity.std(dim=1, unbiased=False) <= float(
        config["similarity_stability_threshold"]
    )
    stable = probability_stable & similarity_stable
    foreground = (
        stable
        & (mean_probability >= float(config["foreground_probability_threshold"]))
        & (mean_similarity >= float(config["foreground_text_similarity_threshold"]))
    )
    background = (
        stable
        & (mean_probability <= float(config["background_probability_threshold"]))
        & (mean_similarity <= float(config["background_text_similarity_threshold"]))
    )

    # Average aligned view features first; each selected voxel contributes one
    # unit regardless of the configured number of seed views.
    mean_features = F.normalize(vision.mean(dim=1), dim=-1)
    flat_features = mean_features.reshape(batch_size, -1, channels)
    foreground_flat = foreground.reshape(batch_size, -1).float()
    background_flat = background.reshape(batch_size, -1).float()
    fg_sum = torch.einsum("bmc,bm->bc", flat_features, foreground_flat)
    bg_sum = torch.einsum("bmc,bm->bc", flat_features, background_flat)
    return {
        "fg_sum": fg_sum,
        "fg_count": foreground_flat.sum(dim=1),
        "bg_sum": bg_sum,
        "bg_count": background_flat.sum(dim=1),
    }


class SemanticPrototypeMemory:
    """CPU-backed per-case statistics with exact LOO prototype queries."""

    def __init__(self, num_prototypes=1):
        if int(num_prototypes) != 1:
            raise ValueError("TSE v1 supports num_prototypes=1")
        self.num_prototypes = int(num_prototypes)
        self.case_statistics = {}

    def __len__(self):
        return len(self.case_statistics)

    def clear(self):
        self.case_statistics.clear()

    def add(self, case_ids, statistics):
        """Add detached seed sums/counts, merging repeated crops by case id."""
        if len(case_ids) != statistics["fg_sum"].shape[0]:
            raise ValueError("case_ids and seed statistics batch must agree")
        for index, case_id in enumerate(case_ids):
            key = str(case_id)
            item = {
                "fg_sum": statistics["fg_sum"][index].detach().float().cpu().unsqueeze(0),
                "fg_count": statistics["fg_count"][index].detach().float().cpu().view(1),
                "bg_sum": statistics["bg_sum"][index].detach().float().cpu().unsqueeze(0),
                "bg_count": statistics["bg_count"][index].detach().float().cpu().view(1),
            }
            if key not in self.case_statistics:
                self.case_statistics[key] = item
            else:
                for name in item:
                    self.case_statistics[key][name].add_(item[name])

    def synchronize(self):
        """Merge per-case contributions across DDP ranks, if initialized."""
        if not (dist.is_available() and dist.is_initialized()):
            return
        gathered = [None for _ in range(dist.get_world_size())]
        dist.all_gather_object(gathered, self.state_dict())
        self.clear()
        for state in gathered:
            for case_id, statistics in state["case_statistics"].items():
                batch_statistics = {
                    name: value.squeeze(0) if name.endswith("_sum") else value.squeeze(0)
                    for name, value in statistics.items()
                }
                self.add([case_id], {
                    "fg_sum": batch_statistics["fg_sum"].unsqueeze(0),
                    "fg_count": batch_statistics["fg_count"].view(1),
                    "bg_sum": batch_statistics["bg_sum"].unsqueeze(0),
                    "bg_count": batch_statistics["bg_count"].view(1),
                })

    def contributors(self, excluding=None):
        excluded = None if excluding is None else str(excluding)
        return [key for key in self.case_statistics if key != excluded]

    def leave_one_out(self, case_ids, device):
        """Return (positive, negative, valid) without any queried case's seeds."""
        if not self.case_statistics:
            raise RuntimeError("Semantic prototype memory has not been built")
        example = next(iter(self.case_statistics.values()))
        channels = int(example["fg_sum"].shape[-1])
        positives, negatives, validity = [], [], []
        epsilon = torch.finfo(torch.float32).eps
        for case_id in case_ids:
            contributors = self.contributors(excluding=case_id)
            fg_sum = torch.zeros(self.num_prototypes, channels)
            bg_sum = torch.zeros_like(fg_sum)
            fg_count = torch.zeros(self.num_prototypes)
            bg_count = torch.zeros_like(fg_count)
            for contributor in contributors:
                item = self.case_statistics[contributor]
                fg_sum += item["fg_sum"]
                bg_sum += item["bg_sum"]
                fg_count += item["fg_count"]
                bg_count += item["bg_count"]
            valid = (fg_count > 0) & (bg_count > 0)
            positive = fg_sum / fg_count.clamp_min(epsilon).unsqueeze(-1)
            negative = bg_sum / bg_count.clamp_min(epsilon).unsqueeze(-1)
            positives.append(F.normalize(positive, dim=-1))
            negatives.append(F.normalize(negative, dim=-1))
            validity.append(valid)
        return (
            torch.stack(positives).to(device=device),
            torch.stack(negatives).to(device=device),
            torch.stack(validity).to(device=device),
        )

    def state_dict(self):
        return {
            "num_prototypes": self.num_prototypes,
            "case_statistics": {
                case_id: {name: value.clone() for name, value in statistics.items()}
                for case_id, statistics in self.case_statistics.items()
            },
        }

    def load_state_dict(self, state):
        if int(state.get("num_prototypes", 1)) != self.num_prototypes:
            raise ValueError("Checkpoint prototype count does not match configuration")
        self.case_statistics = {
            str(case_id): {
                name: value.detach().float().cpu().clone()
                for name, value in statistics.items()
            }
            for case_id, statistics in state.get("case_statistics", {}).items()
        }


def semantic_evidence_map(
    vision_features,
    positive_prototypes,
    negative_prototypes,
    prototype_valid,
    temperature,
    stop_gradient=True,
):
    """Compute E(v) using cosine similarity to positive/negative prototypes."""
    features = _vision_channels_first(vision_features)
    batch_size, *spatial, channels = features.shape
    flattened = features.reshape(batch_size, -1, channels)
    if positive_prototypes.ndim == 2:
        positive_prototypes = positive_prototypes.unsqueeze(1)
        negative_prototypes = negative_prototypes.unsqueeze(1)
        prototype_valid = prototype_valid.unsqueeze(1)
    positive = F.normalize(positive_prototypes.detach().float(), dim=-1)
    negative = F.normalize(negative_prototypes.detach().float(), dim=-1)
    positive_similarity = torch.einsum("bmc,bkc->bmk", flattened, positive)
    negative_similarity = torch.einsum("bmc,bkc->bmk", flattened, negative)
    floor = torch.finfo(positive_similarity.dtype).min
    valid = prototype_valid.bool()
    difference = positive_similarity - negative_similarity
    difference = difference.masked_fill(~valid[:, None, :], floor).amax(dim=-1)
    any_valid = valid.any(dim=-1)
    evidence = torch.sigmoid(difference / float(temperature))
    evidence = torch.where(any_valid[:, None], evidence, torch.zeros_like(evidence))
    evidence = torch.nan_to_num(evidence, nan=0.0, posinf=1.0, neginf=0.0)
    evidence = evidence.view(batch_size, *spatial)
    return evidence.detach() if stop_gradient else evidence


def compute_tse_components(probability, evidence, epsilon=1e-6):
    """Compute semantic purity, coverage/completeness and their harmonic mean."""
    if probability.shape != evidence.shape:
        raise ValueError(
            f"Probability/evidence shape mismatch: {probability.shape} vs {evidence.shape}"
        )
    probability = torch.nan_to_num(probability.float(), nan=0.0).clamp(0.0, 1.0)
    evidence = torch.nan_to_num(evidence.float(), nan=0.0).clamp(0.0, 1.0)
    product = (probability * evidence).flatten(start_dim=1).sum(dim=1)
    probability_mass = probability.flatten(start_dim=1).sum(dim=1)
    evidence_mass = evidence.flatten(start_dim=1).sum(dim=1)
    purity = product / (probability_mass + float(epsilon))
    completeness = product / (evidence_mass + float(epsilon))
    tse = 2 * purity * completeness / (purity + completeness + float(epsilon))
    return tuple(
        torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0)
        for value in (purity, completeness, tse)
    )


def compute_semantic_quality(
    vision_features,
    logits,
    positive_prototypes,
    negative_prototypes,
    prototype_valid,
    config,
    stop_evidence_gradient=True,
):
    spatial = vision_features.shape[1:-1]
    probability = probability_on_feature_grid(logits, spatial)
    evidence = semantic_evidence_map(
        vision_features,
        positive_prototypes,
        negative_prototypes,
        prototype_valid,
        config["temperature"],
        stop_gradient=stop_evidence_gradient,
    )
    purity, completeness, tse = compute_tse_components(
        probability, evidence, config["epsilon"]
    )
    return {
        "purity": purity,
        "completeness": completeness,
        "tse": tse,
        "evidence": evidence,
        "probability": probability,
        "prototype_valid": prototype_valid.any(dim=-1),
    }
