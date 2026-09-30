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


def _prepare_configuration(model: MPS, configuration: torch.Tensor) -> torch.Tensor:
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
    return configuration


def _advance_transfer(
    model: MPS,
    env: torch.Tensor,
    log_scale: torch.Tensor,
    tensor: torch.Tensor,
    *,
    fixed_value: Optional[int],
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance one site in the double-layer probability contraction."""
    if fixed_value is None:
        matrices = model._as_matrices(tensor)
        contracted = torch.matmul(env, matrices)
        matrices_dagger = matrices.conj().transpose(1, 2)
        next_env = torch.matmul(matrices_dagger, contracted).sum(dim=0)
    else:
        matrix = tensor[:, fixed_value, :]
        next_env = matrix.conj().transpose(0, 1) @ env @ matrix

    scale = next_env.abs().max().clamp_min(model._numerical_floor)
    next_env = next_env / scale
    next_log_scale = log_scale + scale.double().log()
    return next_env, next_log_scale


def _log_scalar_contraction(
    model: MPS,
    env: torch.Tensor,
    log_scale: torch.Tensor,
) -> torch.Tensor:
    value = env.squeeze().real.clamp_min(model._numerical_floor)
    return value.double().log() + log_scale


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
    configuration = _prepare_configuration(model, configuration)
    subset = _canonical_subset(sites, model.num_sites)
    if not subset:
        return torch.zeros(
            (),
            dtype=torch.float64,
            device=model.site_tensors[0].device,
        )

    selected = set(subset)
    env = torch.ones(
        1,
        1,
        dtype=model.dtype,
        device=model.site_tensors[0].device,
    )
    log_scale = torch.zeros(
        (),
        dtype=torch.float64,
        device=env.device,
    )

    for site, tensor in enumerate(model.site_tensors):
        fixed_value = (
            int(configuration[site].item())
            if site in selected
            else None
        )
        env, log_scale = _advance_transfer(
            model,
            env,
            log_scale,
            tensor,
            fixed_value=fixed_value,
        )

    log_numerator = _log_scalar_contraction(model, env, log_scale)
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
def subset_surprisals_up_to_order(
    model: MPS,
    configuration: torch.Tensor,
    *,
    max_order: int,
) -> Dict[Subset, float]:
    """Compute every h_x(S) up to max_order in one branched chain pass.

    Each prefix contraction is reused by its marginalized and fixed-value child
    branches. This avoids contracting the whole MPS independently for every
    subset and is substantially faster for the paper experiments.
    """
    configuration = _prepare_configuration(model, configuration)
    if max_order < 1 or max_order > model.num_sites:
        raise ValueError(
            f"max_order must lie in [1, {model.num_sites}], got {max_order}"
        )

    device = model.site_tensors[0].device
    states: Dict[Subset, tuple[torch.Tensor, torch.Tensor]] = {
        (): (
            torch.ones(1, 1, dtype=model.dtype, device=device),
            torch.zeros((), dtype=torch.float64, device=device),
        )
    }

    for site, tensor in enumerate(model.site_tensors):
        next_states: Dict[Subset, tuple[torch.Tensor, torch.Tensor]] = {}
        fixed_value = int(configuration[site].item())

        for subset, (env, log_scale) in states.items():
            marginal_env, marginal_scale = _advance_transfer(
                model,
                env,
                log_scale,
                tensor,
                fixed_value=None,
            )
            next_states[subset] = (marginal_env, marginal_scale)

            if len(subset) < max_order:
                fixed_env, fixed_scale = _advance_transfer(
                    model,
                    env,
                    log_scale,
                    tensor,
                    fixed_value=fixed_value,
                )
                next_states[subset + (site,)] = (fixed_env, fixed_scale)

        states = next_states

    log_z = model.log_norm().double()
    surprisals: Dict[Subset, float] = {}

    for subset, (env, log_scale) in states.items():
        if not subset:
            continue
        log_numerator = _log_scalar_contraction(model, env, log_scale)
        surprisals[subset] = float((log_z - log_numerator).item())

    return surprisals


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
    if max_order is None:
        max_order = model.num_sites
    if max_order < 1 or max_order > model.num_sites:
        raise ValueError(
            f"max_order must lie in [1, {model.num_sites}], got {max_order}"
        )

    surprisals = subset_surprisals_up_to_order(
        model,
        configuration,
        max_order=max_order,
    )
    interactions: Dict[Subset, float] = {}

    for order in range(1, max_order + 1):
        for subset in itertools.combinations(range(model.num_sites), order):
            h_value = surprisals[subset]

            lower_order_terms = []
            for lower_order in range(1, order):
                for proper_subset in itertools.combinations(subset, lower_order):
                    lower_order_terms.append(interactions[proper_subset])

            interactions[subset] = h_value - math.fsum(lower_order_terms)

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

    order_sums = []
    curve = []
    for order in range(1, max_order + 1):
        order_sum = math.fsum(
            value
            for subset, value in interactions.items()
            if len(subset) == order
        )
        order_sums.append(order_sum)
        cumulative = math.fsum(order_sums)
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

    for k in range(1, max_k + 1):
        positive_sum = math.fsum(positive[:k])
        negative_sum = math.fsum(negative[:k])
        reconstruction = math.fsum((positive_sum, negative_sum))
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
