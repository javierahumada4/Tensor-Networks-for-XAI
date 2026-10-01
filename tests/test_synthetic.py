import json

import numpy as np

from data_artifacts import load_encoded_bundle
from prepare_synthetic import (
    CASES,
    _generate_anomalies,
    _generate_normals,
    prepare_synthetic_suite,
)


def _parity(values: np.ndarray) -> np.ndarray:
    return np.bitwise_xor.reduce(values, axis=1)


def test_parity_anomalies_always_violate_planted_relation() -> None:
    for case in ["pairwise", "parity3", "parity4"]:
        order = CASES[case]["planted_order"]
        rng = np.random.default_rng(7)
        x = _generate_anomalies(case, 1000, rng=rng)

        expected = _parity(x[:, : order - 1])
        observed = x[:, order - 1]

        assert np.all(observed == 1 - expected)


def test_normal_parity_violation_rate_matches_noise() -> None:
    noise = 0.03
    for case in ["pairwise", "parity3", "parity4"]:
        order = CASES[case]["planted_order"]
        rng = np.random.default_rng(11)
        x = _generate_normals(
            case,
            50000,
            noise=noise,
            rng=rng,
        )
        expected = _parity(x[:, : order - 1])
        observed = x[:, order - 1]
        violation_rate = np.mean(observed != expected)

        assert abs(violation_rate - noise) < 0.005


def test_marginal_anomalies_use_rare_state() -> None:
    rng = np.random.default_rng(5)
    x = _generate_anomalies("marginal", 500, rng=rng)

    assert np.all(x[:, 0] == 1)
    assert set(np.unique(x[:, 1])).issubset({0, 1})


def test_synthetic_suite_persists_one_class_artifacts(tmp_path) -> None:
    root = tmp_path / "synthetic"
    suite = prepare_synthetic_suite(
        root,
        n_normal=1000,
        n_anomaly=100,
        noise=0.02,
        seed=123,
    )

    assert set(suite["cases"]) == set(CASES)

    for case, spec in CASES.items():
        data = load_encoded_bundle(root / case)
        truth = json.loads(
            (root / case / "synthetic_truth.json").read_text()
        )

        assert data.train.y.sum().item() == 0
        assert data.val.y.sum().item() == 0
        assert data.test.y.sum().item() == 100
        assert truth["planted_order"] == spec["planted_order"]
        assert len(data.feature_names) == spec["num_features"]
