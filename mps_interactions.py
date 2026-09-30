"""Raw local interaction decomposition for discrete Born-MPS models.

For a configuration x and a non-empty feature subset S,

    h_x(S) = -log p_S(x_S)

is the subset surprisal under the exact MPS marginal. Raw interaction terms are
the Moebius transform

    I_x(S) = h_x(S) - sum_{T proper non-empty subset of S} I_x(T).

Consequently, if all orders are included,

    -log p(x) = sum_{S non-empty} I_x(S).

This module intentionally implements only the raw decomposition used by the
paper experiments. No entropy-centered baseline is required.
"""

from __future__ import annotations

import itertools
import math
from typing import Dict, Iterable, Mapping, Optional, Tuple

import torch

from mps import MPS, MPSShapeError


Subset = Tuple[int, ...]


def _canonical_subset(sites: Iterable[int], num_sites: int) -> Subset:
    subset = tuple(sorted(int(site) for site in sites))
    if len(set(subset)) != len(subset):
        raise ValueError(f"subset contains duplicate sites: {subset}")
    for site in subset:
        if site < 0 or site >= num_sites:
            raise IndexError(f"site {site} out of range [0, {num_sites})")
    return subset


@torch.no_grad()
def subset_log_prob(
    model: MPS,
    configuration: torch.Tensor,
    sites: Iterable[int],
    *,
    log_z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return log p_S(x_S) for an arbitrary subset S.

    Unselected physical indices are summed out exactly through the MPS
    double-layer transfer contraction. Selected sites are fixed to the values in
    configuration. Per-site rescaling keeps the contraction numerically stable.

    The empty subset has probability one and therefore log probability zero.
    """
    if configuration.dim() != 1:
        raise MPSShapeError(
            "configuration must be 1-D with shape (num_sites,), "
            f"got {tuple(configuration.shape)}"
        )
    if len(configuration) != model.num_sites:
        raise MPSShapeError(
            f"expected {model.num_sites} sites, got {len(configuration)}"
        )

    device = model.site_tensors[0].device
    configuration = configuration.to(device=device, dtype=torch.long)
    model._validate_configurations(configuration.unsqueeze(0))

    subset = _canonical_subset(sites, model.num_sites)
    if not subset:
        return torch.zeros((), dtype=torch.float64, device=device)

    selected = set(subset)
    env = torch.ones(1, 1, dtype=model.dtype, device=device)
    log_scale = torch.zeros((), dtype=torch.float64, device=device)

    for site, tensor in enumerate(model.site_tensors):
        if site in selected:
            value = int(configuration[site].item())
            matrix = tensor[:, value, :]
            env = matrix.conj().transpose(0, 1) @ env @ matrix
        else:
            matrices = model._as_matrices(tensor)
            contracted = torch.matmul(env, matrices)
            matrices_dagger = matrices.conj().transpose(1, 2)
            env = torch.matmul(matrices_dagger, contracted).sum(dim=0)

        scale = env.abs().max().clamp_min(model._numerical_floor)
        env = env / scale
        log_scale = log_scale + scale.double().log()

    numerator = env.squeeze().real.clamp_min(model._numerical_floor)
    log_numerator = numerator.double().log() + log_scale
    if log_z is None:
        log_z = model.log_norm()
    return log_numerator - log_z.double()


@torch.no_grad()
def subset_surprisal(
    model: MPS,
    configuration: torch.Tensor,
    sites: Iterable[int],
    *,
    log_z: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return h_x(S) = -log p_S(x_S)."""
    return -subset_log_prob(
        model,
        configuration,
        sites,
        log_z=log_z,
    )


@torch.no_grad()
def raw_interactions(
    model: MPS,
    configuration: torch.Tensor,
    *,
    max_order: Optional[int] = None,
) -> Dict[Subset, float]:
    """Compute all raw Moebius interaction terms up to max_order.

    Results are returned in increasing interaction order. max_order=None
    computes the complete decomposition through order model.num_sites.
    """
    if configuration.dim() != 1:
        raise MPSShapeError(
            "configuration must be 1-D with shape (num_sites,), "
            f"got {tuple(configuration.shape)}"
        )
    if max_order is None:
        max_order = model.num_sites
    if max_order < 1 or max_order > model.num_sites:
        raise ValueError(
            f"max_order must lie in [1, {model.num_sites}], got {max_order}"
        )

    log_z = model.log_norm()
    interactions: Dict[Subset, float] = {}

    for order in range(1, max_order + 1):
        for subset in itertools.combinations(range(model.num_sites), order):
            h_value = float(
                subset_surprisal(
                    model,
                    configuration,
                    subset,
                    log_z=log_z,
                ).item()
            )

            lower_order_sum = 0.0
            for lower_order in range(1, order):
                for proper_subset in itertools.combinations(subset, lower_order):
                    lower_order_sum += interactions[proper_subset]

            interactions[subset] = h_value - lower_order_sum

    return interactions


def order_fidelity_curve(
    anomaly_score: float,
    interactions: Mapping[Subset, float],
    *,
    max_order: Optional[int] = None,
    eps: float = 1e-12,
) -> list[dict]:
    """Return reconstruction residual c_m as interaction order increases.

    c_m = |A(x) - sum_{1 <= |S| <= m} I_x(S)| / A(x)
    """
    score = float(anomaly_score)
    denominator = max(abs(score), eps)

    if not interactions:
        raise ValueError("interactions must not be empty")

    available_max = max(len(subset) for subset in interactions)
    if max_order is None:
        max_order = available_max
    if max_order < 1 or max_order > available_max:
        raise ValueError(
            f"max_order must lie in [1, {available_max}], got {max_order}"
        )

    cumulative = 0.0
    curve = []
    for order in range(1, max_order + 1):
        order_sum = sum(
            value
            for subset, value in interactions.items()
            if len(subset) == order
        )
        cumulative += order_sum
        residual = score - cumulative
        curve.append({
            "order": order,
            "order_contribution": order_sum,
            "reconstruction": cumulative,
            "signed_residual": residual,
            "c_m": abs(residual) / denominator,
        })
    return curve


def sparsity_curve(
    anomaly_score: float,
    interactions: Mapping[Subset, float],
    *,
    eps: float = 1e-12,
) -> list[dict]:
    """Reconstruct the score with top-k positive and top-k negative terms.

    Positive interactions are sorted from largest to smallest. Negative
    interactions are sorted from most negative to least negative. At step k,
    up to k terms from each sign are retained. num_interactions records the
    actual explanation size when one sign has fewer than k terms.
    """
    score = float(anomaly_score)
    denominator = max(abs(score), eps)

    positive = sorted(
        (float(value) for value in interactions.values() if value > 0.0),
        reverse=True,
    )
    negative = sorted(
        (float(value) for value in interactions.values() if value < 0.0)
    )

    max_k = max(len(positive), len(negative), 0)
    curve = [{
        "k_per_sign": 0,
        "num_interactions": 0,
        "positive_terms": 0,
        "negative_terms": 0,
        "reconstruction": 0.0,
        "signed_residual": score,
        "c_k": abs(score) / denominator,
    }]

    positive_sum = 0.0
    negative_sum = 0.0

    for k in range(1, max_k + 1):
        if k <= len(positive):
            positive_sum += positive[k - 1]
        if k <= len(negative):
            negative_sum += negative[k - 1]

        reconstruction = positive_sum + negative_sum
        residual = score - reconstruction
        curve.append({
            "k_per_sign": k,
            "num_interactions": min(k, len(positive)) + min(k, len(negative)),
            "positive_terms": min(k, len(positive)),
            "negative_terms": min(k, len(negative)),
            "reconstruction": reconstruction,
            "signed_residual": residual,
            "c_k": abs(residual) / denominator,
        })

    return curve


def interaction_count(num_features: int, max_order: int) -> int:
    """Number of non-empty subsets up to max_order."""
    if max_order < 1 or max_order > num_features:
        raise ValueError(
            f"max_order must lie in [1, {num_features}], got {max_order}"
        )
    return sum(
        math.comb(num_features, order)
        for order in range(1, max_order + 1)
    )
