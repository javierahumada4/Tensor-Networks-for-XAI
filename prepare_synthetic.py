"""Generate synthetic one-class datasets with known anomaly interaction order.

Cases:
- marginal: one rare marginal value, no dependency required (order 1)
- pairwise: rare violation of X1 = X0 (order 2)
- parity3: rare violation of X2 = X0 XOR X1 (order 3)
- parity4: rare violation of X3 = X0 XOR X1 XOR X2 (order 4)

For parity cases, every proper subset of the planted variables has uniform
marginals in the population. Only the full planted subset carries the parity
dependency. A small noise probability keeps anomalous configurations inside
the support of the normal distribution.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from data_artifacts import OneClassSplitConfig, prepare_one_class_artifacts


CASES = {
    "marginal": {"num_features": 2, "planted_order": 1},
    "pairwise": {"num_features": 2, "planted_order": 2},
    "parity3": {"num_features": 3, "planted_order": 3},
    "parity4": {"num_features": 4, "planted_order": 4},
}


def _binary_frame(values: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        values.astype(np.int64),
        columns=[f"x{i}" for i in range(values.shape[1])],
    )


def _generate_normals(
    case: str,
    n: int,
    *,
    noise: float,
    rng: np.random.Generator,
) -> np.ndarray:
    spec = CASES[case]
    d = spec["num_features"]

    if case == "marginal":
        x = np.empty((n, 2), dtype=np.int64)
        x[:, 0] = rng.binomial(1, noise, size=n)
        x[:, 1] = rng.integers(0, 2, size=n)
        return x

    x = rng.integers(0, 2, size=(n, d), dtype=np.int64)
    planted_order = spec["planted_order"]
    parent_count = planted_order - 1
    parity = np.bitwise_xor.reduce(x[:, :parent_count], axis=1)
    flips = rng.binomial(1, noise, size=n).astype(np.int64)
    x[:, planted_order - 1] = parity ^ flips
    return x


def _generate_anomalies(
    case: str,
    n: int,
    *,
    rng: np.random.Generator,
) -> np.ndarray:
    spec = CASES[case]
    d = spec["num_features"]

    if case == "marginal":
        x = np.empty((n, 2), dtype=np.int64)
        x[:, 0] = 1
        x[:, 1] = rng.integers(0, 2, size=n)
        return x

    x = rng.integers(0, 2, size=(n, d), dtype=np.int64)
    planted_order = spec["planted_order"]
    parent_count = planted_order - 1
    parity = np.bitwise_xor.reduce(x[:, :parent_count], axis=1)
    x[:, planted_order - 1] = 1 - parity
    return x


def prepare_synthetic_suite(
    output_root: Path,
    *,
    n_normal: int = 20000,
    n_anomaly: int = 1000,
    noise: float = 0.02,
    seed: int = 123,
) -> dict:
    if n_normal < 100:
        raise ValueError("n_normal must be >= 100")
    if n_anomaly < 1:
        raise ValueError("n_anomaly must be >= 1")
    if not (0.0 < noise < 0.5):
        raise ValueError("noise must satisfy 0 < noise < 0.5")

    output_root.mkdir(parents=True, exist_ok=True)
    suite = {
        "format_version": 1,
        "seed": seed,
        "n_normal": n_normal,
        "n_anomaly": n_anomaly,
        "noise": noise,
        "cases": {},
    }

    for case_index, (case, spec) in enumerate(CASES.items()):
        rng = np.random.default_rng(seed + 1000 * case_index)

        normal = _generate_normals(
            case,
            n_normal,
            noise=noise,
            rng=rng,
        )
        anomaly = _generate_anomalies(
            case,
            n_anomaly,
            rng=rng,
        )

        X = np.concatenate([normal, anomaly], axis=0)
        y = pd.Series(
            np.concatenate([
                np.zeros(n_normal, dtype=np.int64),
                np.ones(n_anomaly, dtype=np.int64),
            ])
        )
        frame = _binary_frame(X)
        columns = list(frame.columns)

        bundle = prepare_one_class_artifacts(
            frame,
            y,
            output_root / case,
            normal_label=0,
            dataset_name=case,
            split_config=OneClassSplitConfig(
                train_fraction=0.70,
                val_fraction=0.15,
                seed=seed,
            ),
            categorical_columns=columns,
            n_bins=8,
            max_categories=4,
        )

        truth = {
            "case": case,
            "planted_order": spec["planted_order"],
            "num_features": spec["num_features"],
            "normal_noise_probability": noise,
            "anomaly_definition": (
                "rare marginal state"
                if case == "marginal"
                else "violation of planted parity relation"
            ),
            "expected_population_structure": (
                "independent product distribution"
                if case == "marginal"
                else (
                    "all proper subsets of planted variables are uniform; "
                    "dependency first appears at planted order"
                )
            ),
        }
        (bundle.root / "synthetic_truth.json").write_text(
            json.dumps(truth, indent=2),
            encoding="utf-8",
        )

        suite["cases"][case] = {
            **truth,
            "counts": bundle.manifest["counts"],
            "physical_dims": bundle.physical_dims,
        }

    (output_root / "synthetic_suite_manifest.json").write_text(
        json.dumps(suite, indent=2),
        encoding="utf-8",
    )
    return suite


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "output_root",
        type=Path,
        nargs="?",
        default=Path("artifacts/synthetic"),
    )
    parser.add_argument("--n-normal", type=int, default=20000)
    parser.add_argument("--n-anomaly", type=int, default=1000)
    parser.add_argument("--noise", type=float, default=0.02)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    suite = prepare_synthetic_suite(
        args.output_root,
        n_normal=args.n_normal,
        n_anomaly=args.n_anomaly,
        noise=args.noise,
        seed=args.seed,
    )
    print(json.dumps(suite, indent=2))


if __name__ == "__main__":
    main()
