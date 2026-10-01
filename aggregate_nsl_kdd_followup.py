"""Aggregate NSL-KDD order-4 and residual-aware greedy follow-up experiments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median

import numpy as np
import pandas as pd


CHECKPOINTS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


def q(values, p):
    return float(np.quantile(np.asarray(values, dtype=float), p))


def summarize_order(frame: pd.DataFrame) -> pd.DataFrame:
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


def summarize_greedy(curves: pd.DataFrame) -> pd.DataFrame:
    rows = []
    grouped = {int(k): g for k, g in curves.groupby("sample_index")}
    for budget in CHECKPOINTS:
        vals = []
        actual = []
        for _, g in grouped.items():
            g = g.sort_values("num_interactions")
            eligible = g[g["num_interactions"] <= budget]
            row = eligible.iloc[-1] if len(eligible) else g.iloc[0]
            vals.append(float(row["c_k"]))
            actual.append(int(row["num_interactions"]))
        rows.append({
            "interaction_budget": budget,
            "n": len(vals),
            "mean_terms_used": mean(actual),
            "median_terms_used": median(actual),
            "mean_c_k": mean(vals),
            "median_c_k": median(vals),
            "q25_c_k": q(vals, 0.25),
            "q75_c_k": q(vals, 0.75),
            "fraction_le_0_10": sum(v <= 0.10 for v in vals) / len(vals),
            "fraction_le_0_05": sum(v <= 0.05 for v in vals) / len(vals),
            "fraction_le_0_01": sum(v <= 0.01 for v in vals) / len(vals),
        })
    return pd.DataFrame(rows)


def threshold_stats(samples: pd.DataFrame, column: str) -> dict:
    achieved = samples[column].notna()
    values = samples.loc[achieved, column].astype(float).tolist()
    return {
        "achieved_fraction": float(achieved.mean()),
        "achieved_count": int(achieved.sum()),
        "median_terms_among_achieved": float(median(values)) if values else None,
        "q25_terms_among_achieved": q(values, 0.25) if values else None,
        "q75_terms_among_achieved": q(values, 0.75) if values else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("order4_root", type=Path)
    parser.add_argument("greedy_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    order_sample_files = sorted(args.order4_root.glob("**/samples.csv"))
    order_curve_files = sorted(args.order4_root.glob("**/fidelity_per_sample.csv"))
    greedy_sample_files = sorted(args.greedy_root.glob("**/samples.csv"))
    greedy_curve_files = sorted(args.greedy_root.glob("**/greedy_per_sample.csv"))
    if not all((order_sample_files, order_curve_files, greedy_sample_files, greedy_curve_files)):
        raise RuntimeError("missing follow-up shard outputs")

    order_samples = pd.concat([pd.read_csv(p) for p in order_sample_files], ignore_index=True)
    order_curves = pd.concat([pd.read_csv(p) for p in order_curve_files], ignore_index=True)
    greedy_samples = pd.concat([pd.read_csv(p) for p in greedy_sample_files], ignore_index=True)
    greedy_curves = pd.concat([pd.read_csv(p) for p in greedy_curve_files], ignore_index=True)

    if len(order_samples) != 20 or order_samples["sample_index"].nunique() != 20:
        raise RuntimeError(f"expected 20 unique order-4 samples, got {len(order_samples)}")
    if len(greedy_samples) != 100 or greedy_samples["sample_index"].nunique() != 100:
        raise RuntimeError(f"expected 100 unique greedy samples, got {len(greedy_samples)}")

    order_samples = order_samples.sort_values("sample_index").reset_index(drop=True)
    order_curves = order_curves.sort_values(["sample_index", "order"]).reset_index(drop=True)
    greedy_samples = greedy_samples.sort_values("sample_index").reset_index(drop=True)
    greedy_curves = greedy_curves.sort_values(
        ["sample_index", "num_interactions"]
    ).reset_index(drop=True)

    order_summary = summarize_order(order_curves)
    greedy_summary = summarize_greedy(greedy_curves)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    order_samples.to_csv(args.output_dir / "order4_samples.csv", index=False)
    order_curves.to_csv(args.output_dir / "order4_fidelity_per_sample.csv", index=False)
    order_summary.to_csv(args.output_dir / "order4_fidelity_summary.csv", index=False)
    greedy_samples.to_csv(args.output_dir / "greedy_samples.csv", index=False)
    greedy_curves.to_csv(args.output_dir / "greedy_per_sample.csv", index=False)
    greedy_summary.to_csv(args.output_dir / "greedy_sparsity_summary.csv", index=False)

    o3 = order_summary[order_summary["order"] == 3].iloc[0]
    o4 = order_summary[order_summary["order"] == 4].iloc[0]
    final_c = greedy_samples["final_c_k"].astype(float).tolist()

    report = {
        "dataset": "NSL-KDD",
        "model": "validated refactored D=16 paper variant",
        "order4": {
            "samples": 20,
            "interactions_per_sample": 102090,
            "median_c3": float(o3["median_c_m"]),
            "median_c4": float(o4["median_c_m"]),
            "median_delta_c4_minus_c3": float(
                median((order_samples["c4"] - order_samples["c3"]).astype(float))
            ),
            "fraction_c4_better_than_c3": float(
                (order_samples["c4"] < order_samples["c3"]).mean()
            ),
            "fraction_c4_le_10pct": float(o4["fraction_le_0_10"]),
            "fraction_c4_le_5pct": float(o4["fraction_le_0_05"]),
            "full_decomposition": False,
        },
        "greedy": {
            "samples": 100,
            "interaction_pool": "all 10,700 raw interactions through order 3",
            "rule": (
                "choose at each step the remaining term that most reduces "
                "absolute reconstruction residual"
            ),
            "to_10pct": threshold_stats(greedy_samples, "terms_to_10pct"),
            "to_5pct": threshold_stats(greedy_samples, "terms_to_5pct"),
            "to_1pct": threshold_stats(greedy_samples, "terms_to_1pct"),
            "median_final_c_k": float(median(final_c)),
            "fraction_final_le_10pct": sum(v <= 0.10 for v in final_c) / len(final_c),
            "fraction_final_le_5pct": sum(v <= 0.05 for v in final_c) / len(final_c),
            "fraction_final_le_1pct": sum(v <= 0.01 for v in final_c) / len(final_c),
            "median_terms_selected": float(median(
                greedy_samples["terms_selected"].astype(float).tolist()
            )),
        },
        "interpretation_guardrail": (
            "Order-4 remains a truncated decomposition of a 40-feature model. "
            "Greedy sparsity is residual-aware and therefore measures existence "
            "of a compact reconstruction under this heuristic, not a unique or "
            "causal explanation."
        ),
    }
    (args.output_dir / "report.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
