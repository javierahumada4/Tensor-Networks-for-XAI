import dataclasses

import torch

from dmrg_trainer import DMRGConfig, dmrg_train
from mps import MPS


def _all_binary_configurations() -> torch.Tensor:
    return torch.tensor(
        [[0, 0], [0, 1], [1, 0], [1, 1]],
        dtype=torch.long,
    )


def test_empirical_initialization_matches_product_of_marginals() -> None:
    data = torch.tensor(
        [[0, 0], [0, 1], [1, 1], [0, 1]],
        dtype=torch.long,
    )

    model = MPS.from_empirical_frequencies(
        data,
        physical_dims=[2, 2],
        dtype=torch.float64,
        pseudocount=0.0,
    )

    assert model.bond_dims == [1]

    configurations = _all_binary_configurations()
    probabilities = model.log_prob(configurations).exp()

    p0 = torch.tensor([3 / 4, 1 / 4], dtype=torch.float64)
    p1 = torch.tensor([1 / 4, 3 / 4], dtype=torch.float64)
    expected = torch.tensor(
        [
            p0[0] * p1[0],
            p0[0] * p1[1],
            p0[1] * p1[0],
            p0[1] * p1[1],
        ],
        dtype=torch.float64,
    )

    torch.testing.assert_close(probabilities, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(
        probabilities.sum(),
        torch.tensor(1.0, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_empirical_initialization_pseudocount_gives_unseen_states_support() -> None:
    data = torch.tensor(
        [[0, 0], [0, 1], [0, 1], [0, 0]],
        dtype=torch.long,
    )

    model = MPS.from_empirical_frequencies(
        data,
        physical_dims=[3, 2],
        dtype=torch.float64,
        pseudocount=1.0,
    )

    unseen = torch.tensor([[2, 0], [2, 1]], dtype=torch.long)
    unseen_probabilities = model.log_prob(unseen).exp()

    assert torch.all(unseen_probabilities > 0)

    all_configurations = torch.tensor(
        [[i, j] for i in range(3) for j in range(2)],
        dtype=torch.long,
    )
    total_probability = model.log_prob(all_configurations).exp().sum()
    torch.testing.assert_close(
        total_probability,
        torch.tensor(1.0, dtype=torch.float64),
        rtol=1e-12,
        atol=1e-12,
    )


def test_truncation_rank_is_selected_from_discarded_weight() -> None:
    model = MPS(num_sites=2, physical_dims=[2, 2], dtype=torch.float64)
    singular_values = torch.tensor([4.0, 3.0, 1.0], dtype=torch.float64)

    # Total weight = 26. Keeping two values discards 1/26 ~= 0.03846.
    assert model._truncation_rank(
        singular_values, max_bond_dim=None, epsilon_trunc=0.04
    ) == 2

    # A 3% tolerance requires retaining all three singular values.
    assert model._truncation_rank(
        singular_values, max_bond_dim=None, epsilon_trunc=0.03
    ) == 3


def test_truncation_rank_respects_hard_cap_when_tolerance_cannot_be_met() -> None:
    model = MPS(num_sites=2, physical_dims=[2, 2], dtype=torch.float64)
    singular_values = torch.tensor([4.0, 3.0, 1.0], dtype=torch.float64)

    rank = model._truncation_rank(
        singular_values,
        max_bond_dim=2,
        epsilon_trunc=0.01,
    )

    assert rank == 2

    weights = singular_values.square()
    discarded_weight = weights[rank:].sum() / weights.sum()
    assert discarded_weight > 0.01


def test_split_and_truncate_meets_discarded_weight_tolerance() -> None:
    model = MPS(num_sites=2, physical_dims=[2, 2], dtype=torch.float64)

    # Across the two-site cut, singular values are exactly [4, 1].
    merged = torch.diag(
        torch.tensor([4.0, 1.0], dtype=torch.float64)
    ).reshape(1, 2, 2, 1)

    kept = model.split_and_truncate(
        0,
        merged,
        direction="right",
        max_bond_dim=2,
        epsilon_trunc=0.06,
    )

    # Keeping only sigma=4 discards 1 / (16 + 1) ~= 0.05882.
    assert len(kept) == 1
    assert model.bond_dims == [1]

    total_weight = merged.square().sum()
    kept_weight = kept.square().sum()
    discarded_weight = 1.0 - kept_weight / total_weight

    assert discarded_weight <= 0.06


def test_zero_truncation_tolerance_keeps_full_available_rank() -> None:
    model = MPS(num_sites=2, physical_dims=[2, 2], dtype=torch.float64)
    merged = torch.diag(
        torch.tensor([4.0, 1.0], dtype=torch.float64)
    ).reshape(1, 2, 2, 1)

    kept = model.split_and_truncate(
        0,
        merged,
        direction="right",
        max_bond_dim=2,
        epsilon_trunc=0.0,
    )

    assert len(kept) == 2
    assert model.bond_dims == [2]


def test_checkpoint_roundtrip_preserves_variable_bond_dimensions(tmp_path) -> None:
    model = MPS(num_sites=2, physical_dims=[2, 2], dtype=torch.float64)
    merged = torch.diag(
        torch.tensor([2.0, 1.0], dtype=torch.float64)
    ).reshape(1, 2, 2, 1)
    model.split_and_truncate(
        0,
        merged,
        direction="right",
        max_bond_dim=2,
        epsilon_trunc=0.0,
    )
    model.normalize_state()

    path = tmp_path / "model.pt"
    model.save(str(path))
    loaded = MPS.load(str(path))

    assert loaded.bond_dims == model.bond_dims
    configurations = _all_binary_configurations()
    torch.testing.assert_close(
        loaded.log_prob(configurations),
        model.log_prob(configurations),
        rtol=1e-12,
        atol=1e-12,
    )


def test_trainer_uses_fixed_max_bond_dimension() -> None:
    data = torch.tensor(
        [[0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1], [0, 0], [1, 1]],
        dtype=torch.long,
    )
    model = MPS.from_empirical_frequencies(
        data,
        physical_dims=[2, 2],
        dtype=torch.float64,
        pseudocount=1e-6,
    )

    config = DMRGConfig(
        num_descent_steps=1,
        max_bond_dim=2,
        epsilon_trunc=0.0,
        lr=1e-3,
        num_loops=2,
        batch_size=len(data),
        patience=10,
        early_stopping_patience=0,
        metric_for_stopping="train_nll",
        seed=0,
    )

    history = dmrg_train(model, data, config=config)

    assert history
    assert all(record["max_bond_dim"] == 2 for record in history)
    assert all(record["epsilon_trunc"] == 0.0 for record in history)
    assert all(max(record["bond_dims"]) <= 2 for record in history)

    config_fields = {field.name for field in dataclasses.fields(DMRGConfig)}
    assert "epsilon_trunc" in config_fields
    assert "init_bond_cap" not in config_fields
    assert "bond_growth_factor" not in config_fields
    assert "grow_confirm_loops" not in config_fields
