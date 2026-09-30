"""Run the two raw-interaction explanation experiments.

Experiments
-----------
1. Explanation fidelity vs interaction order:
       c_m = |A(x) - sum_{1 <= |S| <= m} I_x(S)| / A(x)

2. Explanation sparsity vs number of interactions:
   retain the top-k positive and top-k negative raw interactions and measure the
   same relative reconstruction residual.

Samples are drawn uniformly from labeled test anomalies with a fixed seed.
Selection is independent of anomaly score. For computational reproducibility,
the automatic maximum interaction order is the largest order whose cumulative
number of subsets does not exceed a fixed per-sample budget.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

import numpy as np
import torch

from data_artifacts import load_encoded_bundle
from mps import MPS
from mps_interactions import (
    interaction_count,
    order_fidelity_curve,
    raw_interactions,
    sparsity_curve,
)


def choose_max_order(num_features: int, max_subsets_per_sample: int) -> int:
    """Largest order whose cumulative subset count fits the budget."""
    if max_subsets_per_sample < num_features:
        raise ValueError(
            "max_subsets_per_sample must allow at least all first-order terms"
        )
    selected = 1
    for order in range(1, num_features + 1):
        count = interaction_count(num_features, order)
        if count > max_subsets_per_sample:
            break
        selected = order
    return selected


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), q))


def summarize_fidelity(rows: list[dict]) -> list[dict]:
    grouped: dict[int, list[float]] = {}
    for row in rows:
        grouped.setdefault(int(row["order"]), []).append(float(row["c_m"]))

    summary = []
    for order in sorted(grouped):
        values = grouped[order]
        summary.append({
            "order": order,
            "n": len(values),
            "mean_c_m": mean(values),
            "median_c_m": median(values),
            "q25_c_m": _quantile(values, 0.25),
            "q75_c_m": _quantile(values, 0.75),
            "fraction_le_0_10": sum(v <= 0.10 for v in values) / len(values),
            "fraction_le_0_05": sum(v <= 0.05 for v in values) / len(values),
        })
    return summary


def summarize_sparsity(rows: list[dict]) -> list[dict]:
    """Aggregate by symmetric k-per-sign budget.

    k_per_sign is shared across samples. num_interactions_mean records the actual
    number of terms retained because one sign can run out before the other.
    """
    grouped: dict[int, list[dict]] = {}
    for row in rows:
        grouped.setdefault(int(row["k_per_sign"]), []).append(row)

    summary = []
    for k in sorted(grouped):
        current = grouped[k]
        values = [float(row["c_k"]) for row in current]
        sizes = [int(row["num_interactions"]) for row in current]
        summary.append({
            "k_per_sign": k,
            "max_interaction_budget": 2 * k,
            "n": len(values),
            "mean_num_interactions": mean(sizes),
            "median_num_interactions": median(sizes),
            "mean_c_k": mean(values),
            "median_c_k": median(values),
            "q25_c_k": _quantile(values, 0.25),
            "q75_c_k": _quantile(values, 0.75),
            "fraction_le_0_10": sum(v <= 0.10 for v in values) / len(values),
            "fraction_le_0_05": sum(v <= 0.05 for v in values) / len(values),
        })
    return summary


def stable_threshold(curve: list[dict], key: str, threshold: float, x_key: str):
    """First point after which every remaining residual stays below threshold."""
    values = [float(row[key]) for row in curve]
    for index in range(len(values)):
        if all(value <= threshold for value in values[index:]):
            return curve[index][x_key]
    return None


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("dataset")
    parser.add_argument("model_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-order", type=int, default=0)
    parser.add_argument("--max-subsets-per-sample", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    if args.max_samples < 1:
        raise ValueError("max_samples must be >= 1")

    data = load_encoded_bundle(args.dataset_root / args.dataset)
    model = MPS.load(str(args.model_dir / "model.pt"), map_location="cpu")
    model.eval()

    num_features = model.num_sites
    if num_features != len(data.feature_names):
        raise ValueError("model and encoded dataset feature counts disagree")

    if args.max_order == 0:
        max_order = choose_max_order(
            num_features,
            args.max_subsets_per_sample,
        )
    else:
        max_order = args.max_order
        if max_order < 1 or max_order > num_features:
            raise ValueError(
                f"max_order must lie in [1, {num_features}], got {max_order}"
            )

    subset_count = interaction_count(num_features, max_order)
    if subset_count > args.max_subsets_per_sample and args.max_order == 0:
        raise AssertionError("automatic order exceeded subset budget")

    anomaly_positions = torch.nonzero(
        data.test.y == 1,
        as_tuple=False,
    ).flatten().numpy()

    if len(anomaly_positions) == 0:
        raise ValueError("test split contains no labeled anomalies")

    rng = np.random.default_rng(args.seed)
    sample_count = min(args.max_samples, len(anomaly_positions))
    if sample_count == len(anomaly_positions):
        selected_positions = np.sort(anomaly_positions)
    else:
        selected_positions = np.sort(
            rng.choice(
                anomaly_positions,
                size=sample_count,
                replace=False,
            )
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    fidelity_rows: list[dict] = []
    sparsity_rows: list[dict] = []
    sample_rows: list[dict] = []

    for sample_index, test_position in enumerate(selected_positions, start=1):
        x = data.test.x[int(test_position)]
        row_index = int(data.test.row_indices[int(test_position)].item())
        score = float(model.anomaly_score(x.unsqueeze(0))[0].item())

        interactions = raw_interactions(
            model,
            x,
            max_order=max_order,
        )

        fidelity = order_fidelity_curve(
            score,
            interactions,
            max_order=max_order,
        )
        sparsity = sparsity_curve(score, interactions)

        for row in fidelity:
            fidelity_rows.append({
                "dataset": args.dataset,
                "sample_index": sample_index,
                "test_position": int(test_position),
                "row_index": row_index,
                **row,
            })

        for row in sparsity:
            sparsity_rows.append({
                "dataset": args.dataset,
                "sample_index": sample_index,
                "test_position": int(test_position),
                "row_index": row_index,
                **row,
            })

        sample_rows.append({
            "dataset": args.dataset,
            "sample_index": sample_index,
            "test_position": int(test_position),
            "row_index": row_index,
            "anomaly_score": score,
            "max_order": max_order,
            "num_interactions_computed": len(interactions),
            "positive_interactions": sum(v > 0 for v in interactions.values()),
            "negative_interactions": sum(v < 0 for v in interactions.values()),
            "c_at_max_order": fidelity[-1]["c_m"],
            "stable_order_10pct": stable_threshold(
                fidelity, "c_m", 0.10, "order"
            ),
            "stable_order_5pct": stable_threshold(
                fidelity, "c_m", 0.05, "order"
            ),
            "stable_terms_10pct": stable_threshold(
                sparsity, "c_k", 0.10, "num_interactions"
            ),
            "stable_terms_5pct": stable_threshold(
                sparsity, "c_k", 0.05, "num_interactions"
            ),
        })

        print(
            f"[{sample_index}/{sample_count}] "
            f"row={row_index} score={score:.6f} "
            f"interactions={len(interactions)} "
            f"c_mmax={fidelity[-1]['c_m']:.6g}",
            flush=True,
        )

    fidelity_summary = summarize_fidelity(fidelity_rows)
    sparsity_summary = summarize_sparsity(sparsity_rows)

    write_csv(args.output_dir / "samples.csv", sample_rows)
    write_csv(args.output_dir / "fidelity_per_sample.csv", fidelity_rows)
    write_csv(args.output_dir / "fidelity_summary.csv", fidelity_summary)
    write_csv(args.output_dir / "sparsity_per_sample.csv", sparsity_rows)
    write_csv(args.output_dir / "sparsity_summary.csv", sparsity_summary)

    metadata = {
        "dataset": args.dataset,
        "seed": args.seed,
        "sampling": "uniform without replacement from labeled test anomalies",
        "available_anomalies": int(len(anomaly_positions)),
        "sample_count": int(sample_count),
        "num_features": num_features,
        "max_order": max_order,
        "full_decomposition": max_order == num_features,
        "interaction_subsets_per_sample": subset_count,
        "max_subsets_per_sample": args.max_subsets_per_sample,
        "feature_names": data.feature_names,
        "physical_dims": data.physical_dims,
        "experiment_1": (
            "Explanation fidelity vs interaction order using raw interactions "
            "and c_m only"
        ),
        "experiment_2": (
            "Explanation sparsity vs top-k positive and top-k negative raw "
            "interactions using c_k"
        ),
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
