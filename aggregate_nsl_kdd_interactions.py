"""Aggregate sharded NSL-KDD interaction experiment outputs."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd


SPARSITY_K_CHECKPOINTS = [
    0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000
]


def q(values, p):
    return float(np.quantile(np.asarray(values, dtype=float), p))


def summarize_fidelity(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for order, g in frame.groupby("order", sort=True):
        vals = g["c_m"].astype(float).tolist()
        rows.append({
            "order": int(order),
            "n": len(vals),
            "mean_c_m": mean(vals),
            "median_c_m": median(vals),
            "q25_c_m": q(vals, 0.25),
            "q75_c_m": q(vals, 0.75),
            "fraction_le_0_10": sum(v <= 0.10 for v in vals) / len(vals),
            "fraction_le_0_05": sum(v <= 0.05 for v in vals) / len(vals),
        })
    return pd.DataFrame(rows)


def summarize_sparsity(frame: pd.DataFrame) -> pd.DataFrame:
    available = sorted(set(int(x) for x in frame["k_per_sign"].unique()))
    wanted = sorted(set(k for k in SPARSITY_K_CHECKPOINTS if k in available))
    if available and available[-1] not in wanted:
        wanted.append(available[-1])

    rows = []
    for k in wanted:
        g = frame[frame["k_per_sign"] == k]
        vals = g["c_k"].astype(float).tolist()
        sizes = g["num_interactions"].astype(int).tolist()
        rows.append({
            "k_per_sign": int(k),
            "max_interaction_budget": 2 * int(k),
            "n": len(vals),
            "mean_num_interactions": mean(sizes),
            "median_num_interactions": median(sizes),
            "mean_c_k": mean(vals),
            "median_c_k": median(vals),
            "q25_c_k": q(vals, 0.25),
            "q75_c_k": q(vals, 0.75),
            "fraction_le_0_10": sum(v <= 0.10 for v in vals) / len(vals),
            "fraction_le_0_05": sum(v <= 0.05 for v in vals) / len(vals),
        })
    return pd.DataFrame(rows)


def summarize_families(fidelity: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (family, order), g in fidelity.groupby(["family", "order"], sort=True):
        vals = g["c_m"].astype(float).tolist()
        rows.append({
            "family": family,
            "order": int(order),
            "n": len(vals),
            "median_c_m": median(vals),
            "q25_c_m": q(vals, 0.25),
            "q75_c_m": q(vals, 0.75),
            "fraction_le_0_10": sum(v <= 0.10 for v in vals) / len(vals),
            "fraction_le_0_05": sum(v <= 0.05 for v in vals) / len(vals),
        })
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("shards_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    sample_files = sorted(args.shards_root.glob("**/samples.csv"))
    fidelity_files = sorted(args.shards_root.glob("**/fidelity_per_sample.csv"))
    sparsity_files = sorted(args.shards_root.glob("**/sparsity_per_sample.csv"))
    metadata_files = sorted(args.shards_root.glob("**/metadata.json"))

    if not sample_files or not fidelity_files or not sparsity_files:
        raise RuntimeError("missing shard CSV outputs")

    samples = pd.concat([pd.read_csv(p) for p in sample_files], ignore_index=True)
    fidelity = pd.concat([pd.read_csv(p) for p in fidelity_files], ignore_index=True)
    sparsity = pd.concat([pd.read_csv(p) for p in sparsity_files], ignore_index=True)

    samples = samples.sort_values("sample_index").reset_index(drop=True)
    fidelity = fidelity.sort_values(["sample_index", "order"]).reset_index(drop=True)
    sparsity = sparsity.sort_values(["sample_index", "k_per_sign"]).reset_index(drop=True)

    if samples["sample_index"].nunique() != len(samples):
        raise RuntimeError("duplicate sample indices across shards")
    if len(samples) != 100:
        raise RuntimeError(f"expected 100 sampled anomalies, got {len(samples)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples.to_csv(args.output_dir / "samples.csv", index=False)
    fidelity.to_csv(args.output_dir / "fidelity_per_sample.csv", index=False)
    sparsity.to_csv(args.output_dir / "sparsity_per_sample.csv", index=False)

    fidelity_summary = summarize_fidelity(fidelity)
    sparsity_summary = summarize_sparsity(sparsity)
    family_summary = summarize_families(fidelity)

    fidelity_summary.to_csv(args.output_dir / "fidelity_summary.csv", index=False)
    sparsity_summary.to_csv(args.output_dir / "sparsity_summary.csv", index=False)
    family_summary.to_csv(args.output_dir / "fidelity_by_family.csv", index=False)

    metadata = [json.loads(p.read_text()) for p in metadata_files]
    families = (
        samples["family"].value_counts().sort_index().astype(int).to_dict()
    )

    order3 = fidelity_summary[fidelity_summary["order"] == 3].iloc[0]
    report = {
        "dataset": "NSL-KDD",
        "model": "refactored D=16 paper variant from workflow run 36754543152",
        "samples": int(len(samples)),
        "seed": 123,
        "features": 40,
        "max_order": 3,
        "interactions_per_sample": 10700,
        "full_decomposition": False,
        "families": families,
        "experiment_1": {
            "metric": "c_m = |A(x)-sum_{1<=|S|<=m} I_x(S)| / |A(x)|",
            "order3_median_c_m": float(order3["median_c_m"]),
            "order3_fraction_le_10pct": float(order3["fraction_le_0_10"]),
            "order3_fraction_le_5pct": float(order3["fraction_le_0_05"]),
        },
        "experiment_2": {
            "metric": (
                "top-k positive + top-k negative raw interactions; residual "
                "measured against the full anomaly NLL"
            ),
            "checkpoint_budgets": [
                int(x) for x in sparsity_summary["max_interaction_budget"].tolist()
            ],
        },
        "limitation": (
            "The 40-feature NSL-KDD model has 2^40-1 non-empty subsets. "
            "This run computes every subset through order 3 (10,700 terms per "
            "sample) but is not a complete interaction decomposition, so the "
            "order-3 and maximum sparsity residuals need not converge to zero."
        ),
        "shards": metadata,
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
