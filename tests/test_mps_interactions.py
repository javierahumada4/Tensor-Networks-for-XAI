import itertools

import torch

from mps import MPS
from mps_interactions import (
    interaction_count,
    order_fidelity_curve,
    raw_interactions,
    sparsity_curve,
    subset_log_prob,
)


def _correlated_three_site_mps() -> MPS:
    torch.manual_seed(7)
    model = MPS(
        num_sites=3,
        physical_dims=[2, 2, 2],
        dtype=torch.float64,
    )
    model.site_tensors[0].data = torch.randn(1, 2, 2, dtype=torch.float64)
    model.site_tensors[1].data = torch.randn(2, 2, 2, dtype=torch.float64)
    model.site_tensors[2].data = torch.randn(2, 2, 1, dtype=torch.float64)
    model.normalize_state()
    return model


def _all_binary_three_site() -> torch.Tensor:
    return torch.tensor(
        list(itertools.product(range(2), repeat=3)),
        dtype=torch.long,
    )


def test_subset_log_prob_matches_bruteforce_marginalization() -> None:
    model = _correlated_three_site_mps()
    configurations = _all_binary_three_site()
    probabilities = model.log_prob(configurations).exp()

    x = torch.tensor([1, 0, 1], dtype=torch.long)

    for subset in [(0,), (1,), (2,), (0, 2), (0, 1), (0, 1, 2)]:
        mask = torch.ones(len(configurations), dtype=torch.bool)
        for site in subset:
            mask &= configurations[:, site] == x[site]

        expected = probabilities[mask].sum().log()
        actual = subset_log_prob(model, x, subset)

        torch.testing.assert_close(
            actual,
            expected,
            rtol=1e-10,
            atol=1e-10,
        )


def test_empty_subset_has_probability_one() -> None:
    model = _correlated_three_site_mps()
    x = torch.tensor([1, 0, 1], dtype=torch.long)

    torch.testing.assert_close(
        subset_log_prob(model, x, ()),
        torch.tensor(0.0, dtype=torch.float64),
    )


def test_complete_raw_interactions_reconstruct_full_nll() -> None:
    model = _correlated_three_site_mps()
    x = torch.tensor([1, 0, 1], dtype=torch.long)

    score = float(model.anomaly_score(x.unsqueeze(0))[0].item())
    interactions = raw_interactions(model, x, max_order=3)

    assert len(interactions) == 7
    assert abs(sum(interactions.values()) - score) < 1e-9

    curve = order_fidelity_curve(score, interactions)
    assert abs(curve[-1]["reconstruction"] - score) < 1e-9
    assert curve[-1]["c_m"] < 1e-10


def test_first_order_residual_is_previous_correlation_share() -> None:
    model = _correlated_three_site_mps()
    x = torch.tensor([1, 0, 1], dtype=torch.long)

    score = float(model.anomaly_score(x.unsqueeze(0))[0].item())
    interactions = raw_interactions(model, x, max_order=3)
    curve = order_fidelity_curve(score, interactions)

    singles = sum(
        value for subset, value in interactions.items() if len(subset) == 1
    )
    expected_c1 = abs(score - singles) / score

    assert abs(curve[0]["c_m"] - expected_c1) < 1e-12


def test_product_distribution_has_no_higher_order_interactions() -> None:
    data = torch.tensor(
        [
            [0, 0, 0],
            [0, 1, 0],
            [1, 0, 1],
            [1, 1, 1],
            [0, 0, 1],
            [1, 1, 0],
        ],
        dtype=torch.long,
    )
    model = MPS.from_empirical_frequencies(
        data,
        physical_dims=[2, 2, 2],
        dtype=torch.float64,
        pseudocount=1e-6,
    )
    x = torch.tensor([1, 0, 1], dtype=torch.long)

    interactions = raw_interactions(model, x, max_order=3)

    higher = [
        abs(value)
        for subset, value in interactions.items()
        if len(subset) > 1
    ]
    assert max(higher) < 1e-10


def test_sparsity_curve_reaches_exact_score_with_complete_interactions() -> None:
    model = _correlated_three_site_mps()
    x = torch.tensor([1, 0, 1], dtype=torch.long)

    score = float(model.anomaly_score(x.unsqueeze(0))[0].item())
    interactions = raw_interactions(model, x, max_order=3)
    curve = sparsity_curve(score, interactions)

    assert curve[0]["num_interactions"] == 0
    assert abs(curve[0]["c_k"] - 1.0) < 1e-12
    assert curve[-1]["num_interactions"] == len(interactions)
    assert curve[-1]["c_k"] < 1e-10


def test_interaction_count() -> None:
    assert interaction_count(6, 1) == 6
    assert interaction_count(6, 2) == 21
    assert interaction_count(6, 3) == 41
    assert interaction_count(6, 6) == 63
