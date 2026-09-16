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
        "prototype_patch_batch_size",
        "evaluation_patch_batch_size",
        "evaluation_overlap",
        "evidence_threshold",
        "similarity_histogram_bins",
        "attention_mad_threshold",
        "attention_min_valid_positions",
        "saaf_min_mass",
    }
    missing = sorted(required.difference(config))
    if missing:
        raise KeyError(f"Quality config {path} is missing: {', '.join(missing)}")
    if int(config["num_prototypes"]) != 1:
        raise ValueError("TSE v1 implements one foreground/background prototype")
    if int(config["seed_views"]) < 2:
        raise ValueError("seed_views must be at least 2 to measure stability")
    if int(config["prototype_patch_batch_size"]) < 1 or int(
        config["evaluation_patch_batch_size"]
    ) < 1:
        raise ValueError("patch batch sizes must be positive")
    if not 0 <= float(config["evaluation_overlap"]) < 1:
        raise ValueError("evaluation_overlap must be in [0, 1)")
    if not 0 <= float(config["evidence_threshold"]) <= 1:
        raise ValueError("evidence_threshold must be in [0, 1]")
    if int(config["similarity_histogram_bins"]) < 2:
        raise ValueError("similarity_histogram_bins must be at least 2")
    if float(config["attention_mad_threshold"]) < 0 or int(config["attention_min_valid_positions"]) < 2:
        raise ValueError("attention MAD threshold must be non-negative and valid positions >= 2")
    if float(config["saaf_min_mass"]) <= 0:
        raise ValueError("saaf_min_mass must be positive")
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


def final_decoder_attention(cross_attention):
    """Return VoxTell's native last-layer cross-attention as (B,1,T,S).

    VoxTell currently returns one already head-averaged ``(B,T,S)`` tensor
    per decoder layer.  A four-dimensional per-head result is also accepted
    for compatible forks, without recomputing query/key projections.
    """
    if isinstance(cross_attention, (list, tuple)):
        if not cross_attention:
            raise ValueError("VoxTell returned an empty cross_attention list")
        attention = cross_attention[-1]
    else:
        attention = cross_attention
    if not torch.is_tensor(attention):
        raise TypeError("VoxTell cross_attention must contain tensors")
    if attention.ndim == 3:
        attention = attention.unsqueeze(1)
    if attention.ndim != 4:
        raise ValueError(
            "Expected native final cross-attention (B,T,S) or (B,H,T,S), "
            f"got {tuple(attention.shape)}"
        )
    return attention


def attention_evidence_map(
    attention,
    spatial_shape,
    spatial_valid=None,
    text_token_mask=None,
    eps=1e-6,
    mad_threshold=1e-3,
    min_valid_positions=2,
):
    """Robust per-head/token log-attention evidence and validity diagnostics."""
    if attention.ndim != 4:
        raise ValueError(f"Expected attention (B,heads,tokens,spatial), got {tuple(attention.shape)}")
    batch_size, heads, tokens, spatial = attention.shape
    spatial_shape = tuple(int(value) for value in spatial_shape)
    if int(torch.tensor(spatial_shape).prod()) != spatial:
        raise ValueError("Attention spatial dimension does not match spatial_shape")
    finite_attention = torch.isfinite(attention)
    if spatial_valid is None:
        spatial_valid = torch.ones((batch_size, spatial), dtype=torch.bool, device=attention.device)
    else:
        spatial_valid = torch.as_tensor(spatial_valid, device=attention.device).bool().reshape(batch_size, spatial)
    if text_token_mask is None:
        text_token_mask = torch.ones((batch_size, tokens), dtype=torch.bool, device=attention.device)
    else:
        text_token_mask = torch.as_tensor(text_token_mask, device=attention.device).bool()
    log_attention = torch.log(attention.float().clamp_min(float(eps)))
    evidence = torch.zeros_like(log_attention)
    mad_values = torch.zeros((batch_size, heads, tokens), dtype=log_attention.dtype, device=log_attention.device)
    valid_views = torch.ones(batch_size, dtype=torch.bool, device=attention.device)
    invalid_reasons = [None for _ in range(batch_size)]
    for batch_index in range(batch_size):
        for head_index in range(heads):
            for token_index in range(tokens):
                mask = spatial_valid[batch_index] & finite_attention[batch_index, head_index, token_index]
                if not bool(text_token_mask[batch_index, token_index]):
                    continue
                if int(mask.sum()) < int(min_valid_positions):
                    valid_views[batch_index] = False
                    invalid_reasons[batch_index] = "insufficient_valid_spatial_positions"
                    continue
                values = log_attention[batch_index, head_index, token_index, mask]
                median = torch.median(values)
                mad = torch.median(torch.abs(values - median))
                mad_values[batch_index, head_index, token_index] = mad
                if not bool(torch.isfinite(median) & torch.isfinite(mad)):
                    valid_views[batch_index] = False
                    invalid_reasons[batch_index] = "nonfinite_attention_statistics"
                    continue
                if float(mad) < float(mad_threshold):
                    valid_views[batch_index] = False
                    invalid_reasons[batch_index] = "attention_mad_below_threshold"
                    continue
                z = (log_attention[batch_index, head_index, token_index] - median) / (1.4826 * mad + float(eps))
                evidence[batch_index, head_index, token_index] = torch.where(mask, z.relu(), torch.zeros_like(z))
    valid_views &= text_token_mask.any(dim=1)
    token_count = text_token_mask.sum(dim=1).clamp_min(1).view(batch_size, 1, 1)
    evidence = evidence * text_token_mask[:, None, :, None].float()
    evidence = evidence.sum(dim=(1, 2)) / (heads * token_count.squeeze(-1).squeeze(-1).float()).view(batch_size, 1)
    evidence = evidence.view(batch_size, *spatial_shape)
    evidence = torch.nan_to_num(evidence, nan=0.0, posinf=0.0, neginf=0.0)
    evidence = torch.where(valid_views.view(batch_size, *([1] * len(spatial_shape))), evidence, torch.zeros_like(evidence))
    mad_summary = mad_values.masked_fill(~text_token_mask[:, None, :], float("nan"))
    attention_mad = torch.nanmean(mad_summary, dim=(1, 2))
    attention_mad = torch.nan_to_num(attention_mad, nan=0.0, posinf=0.0, neginf=0.0)
    for index in range(batch_size):
        if not bool(valid_views[index]) and invalid_reasons[index] is None:
            invalid_reasons[index] = "invalid_attention"
    return evidence, valid_views, attention_mad, invalid_reasons


def compute_saaf_quality(probability, evidence, epsilon=1e-6, min_mass=1e-6):
    """Compute SAAF purity/coverage F-score with explicit invalid cases."""
    if probability.shape != evidence.shape:
        raise ValueError(f"Probability/evidence shape mismatch: {probability.shape} vs {evidence.shape}")
    probability = torch.nan_to_num(probability.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp(0.0, 1.0)
    evidence = torch.nan_to_num(evidence.float(), nan=0.0, posinf=0.0, neginf=0.0).clamp_min(0.0)
    mask_mass = probability.flatten(start_dim=1).sum(dim=1)
    background_mass = (1.0 - probability).flatten(start_dim=1).sum(dim=1)
    evidence_sum = evidence.flatten(start_dim=1).sum(dim=1)
    valid = (
        torch.isfinite(mask_mass)
        & torch.isfinite(background_mass)
        & torch.isfinite(evidence_sum)
        & (mask_mass > float(min_mass))
        & (background_mass > float(min_mass))
        & (evidence_sum > float(min_mass))
    )
    product = (probability * evidence).flatten(start_dim=1).sum(dim=1)
    mu_fg = product / (mask_mass + float(epsilon))
    mu_bg = ((1.0 - probability) * evidence).flatten(start_dim=1).sum(dim=1) / (background_mass + float(epsilon))
    purity = mu_fg / (mu_fg + mu_bg + float(epsilon))
    coverage = product / (evidence_sum + float(epsilon))
    saaf = 2 * purity * coverage / (purity + coverage + float(epsilon))
    outputs = {
        "purity": purity,
        "coverage": coverage,
        "saaf": saaf,
        "mu_fg": mu_fg,
        "mu_bg": mu_bg,
        "mask_ratio": probability.mean(dim=tuple(range(1, probability.ndim))),
        "evidence_sum": evidence_sum,
        "valid": valid,
    }
    for name, value in outputs.items():
        if torch.is_tensor(value):
            outputs[name] = torch.where(valid, torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0), torch.zeros_like(value))
    invalid_reasons = []
    for index in range(probability.shape[0]):
        if bool(valid[index]):
            invalid_reasons.append(None)
        elif float(evidence_sum[index]) <= float(min_mass):
            invalid_reasons.append("evidence_mass_too_small")
        elif float(mask_mass[index]) <= float(min_mass):
            invalid_reasons.append("empty_soft_mask")
        elif float(background_mass[index]) <= float(min_mass):
            invalid_reasons.append("near_all_foreground_mask")
        else:
            invalid_reasons.append("nonfinite_saaf_inputs")
    outputs["invalid_reason"] = invalid_reasons
    return outputs


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


def mask_on_feature_grid(mask, spatial_shape):
    """Map a (B,1,D,H,W) valid-region mask to VoxTell's (B,H,W,D) grid."""
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError(f"Expected valid mask (B,1,D,H,W), got {tuple(mask.shape)}")
    mapped = mask[:, 0].float().permute(0, 2, 3, 1)
    if tuple(mapped.shape[1:]) != tuple(spatial_shape):
        mapped = F.interpolate(
            mapped.unsqueeze(1),
            size=tuple(spatial_shape),
            mode="trilinear",
            align_corners=False,
        ).squeeze(1)
    return mapped.clamp(0.0, 1.0)


def extract_case_seed_statistics(
    vision_features,
    text_features,
    logits,
    batch_size,
    num_views,
    config,
    valid_masks=None,
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
    if valid_masks is None:
        valid_weights = torch.ones_like(mean_probability)
    else:
        if valid_masks.shape[0] != batch_size:
            raise ValueError("valid_masks batch must agree with seed patch batch")
        valid_weights = mask_on_feature_grid(valid_masks, spatial)
    valid = valid_weights > 0
    foreground = (
        stable
        & valid
        & (mean_probability >= float(config["foreground_probability_threshold"]))
        & (mean_similarity >= float(config["foreground_text_similarity_threshold"]))
    )
    background = (
        stable
        & valid
        & (mean_probability <= float(config["background_probability_threshold"]))
        & (mean_similarity <= float(config["background_text_similarity_threshold"]))
    )

    # Average aligned view features first; each selected voxel contributes one
    # unit regardless of the configured number of seed views.
    mean_features = F.normalize(vision.mean(dim=1), dim=-1)
    flat_features = mean_features.reshape(batch_size, -1, channels)
    # Keep seed mass bounded by the valid (unpadded) support.  The explicit
    # minimum is defensive for low-precision interpolation/broadcasting and
    # guarantees that diagnostic seed ratios cannot exceed one.
    weights_flat = valid_weights.reshape(batch_size, -1).clamp(0.0, 1.0)
    foreground_flat = torch.minimum(
        foreground.reshape(batch_size, -1).float() * weights_flat,
        weights_flat,
    )
    background_flat = torch.minimum(
        background.reshape(batch_size, -1).float() * weights_flat,
        weights_flat,
    )
    flat_similarity = mean_similarity.reshape(batch_size, -1)
    histogram_bins = int(config["similarity_histogram_bins"])
    diagnostic_values = {
        "similarity_sum": [],
        "similarity_square_sum": [],
        "similarity_min": [],
        "similarity_max": [],
        "similarity_histogram": [],
    }
    for index in range(batch_size):
        values = flat_similarity[index][weights_flat[index] > 0].float()
        if values.numel():
            diagnostic_values["similarity_sum"].append(values.sum())
            diagnostic_values["similarity_square_sum"].append(values.square().sum())
            diagnostic_values["similarity_min"].append(values.min())
            diagnostic_values["similarity_max"].append(values.max())
            diagnostic_values["similarity_histogram"].append(
                torch.histc(values, bins=histogram_bins, min=-1.0, max=1.0)
            )
        else:
            diagnostic_values["similarity_sum"].append(values.new_zeros(()))
            diagnostic_values["similarity_square_sum"].append(values.new_zeros(()))
            diagnostic_values["similarity_min"].append(values.new_tensor(float("inf")))
            diagnostic_values["similarity_max"].append(values.new_tensor(float("-inf")))
            diagnostic_values["similarity_histogram"].append(
                values.new_zeros(histogram_bins)
            )
    fg_sum = torch.einsum("bmc,bm->bc", flat_features, foreground_flat)
    bg_sum = torch.einsum("bmc,bm->bc", flat_features, background_flat)
    return {
        "fg_sum": fg_sum,
        "fg_count": foreground_flat.sum(dim=1),
        "bg_sum": bg_sum,
        "bg_count": background_flat.sum(dim=1),
        "valid_count": weights_flat.sum(dim=1),
        "similarity_count": (weights_flat > 0).sum(dim=1).float(),
        **{
            name: torch.stack(values)
            for name, values in diagnostic_values.items()
        },
    }


def average_tie_ranks(values, descending=False):
    """Return zero-based average ranks; equal values receive exactly equal ranks."""
    if values.ndim != 2:
        raise ValueError(f"Expected rank values (B,V), got {tuple(values.shape)}")
    ranked_values = -values if descending else values
    ranks = torch.empty_like(ranked_values, dtype=torch.float32)
    for batch_index in range(ranked_values.shape[0]):
        row = ranked_values[batch_index]
        offset = 0
        for unique_value in torch.unique(row, sorted=True):
            tied = row == unique_value
            count = int(tied.sum())
            average_rank = offset + 0.5 * (count - 1)
            ranks[batch_index, tied] = float(average_rank)
            offset += count
    return ranks


def summarize_similarity_distribution(
    histogram,
    count,
    value_sum,
    square_sum,
    minimum,
    maximum,
):
    """Produce JSON-safe similarity moments, approximate quantiles and histogram."""
    histogram = torch.as_tensor(histogram).detach().double().cpu()
    count = float(count)
    bins = int(histogram.numel())
    edges = torch.linspace(-1.0, 1.0, bins + 1, dtype=torch.float64)
    empty_quantiles = {
        "q05": None,
        "q25": None,
        "q50": None,
        "q75": None,
        "q95": None,
    }
    if count <= 0:
        return {
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "quantiles": empty_quantiles,
            "histogram": histogram.long().tolist(),
            "bin_edges": edges.tolist(),
        }
    mean = float(value_sum) / count
    variance = max(0.0, float(square_sum) / count - mean * mean)
    cumulative = histogram.cumsum(0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    quantiles = {}
    for name, fraction in (
        ("q05", 0.05),
        ("q25", 0.25),
        ("q50", 0.50),
        ("q75", 0.75),
        ("q95", 0.95),
    ):
        target = torch.tensor(fraction * count, dtype=cumulative.dtype)
        index = int(torch.searchsorted(cumulative, target).clamp(max=bins - 1))
        quantiles[name] = float(centers[index])
    return {
        "count": int(round(count)),
        "mean": mean,
        "std": variance ** 0.5,
        "min": float(minimum),
        "max": float(maximum),
        "quantiles": quantiles,
        "histogram": histogram.long().tolist(),
        "bin_edges": edges.tolist(),
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
