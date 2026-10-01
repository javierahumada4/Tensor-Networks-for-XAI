"""Aggregate NSL-KDD greedy stability controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def summary_stats(g, column):
    v = g[column].dropna().astype(float)
    return {
        "median": float(v.median()) if len(v) else None,
        "q25": float(v.quantile(0.25)) if len(v) else None,
        "q75": float(v.quantile(0.75)) if len(v) else None,
        "mean": float(v.mean()) if len(v) else None,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("shards_root", type=Path)
    p.add_argument("output_dir", type=Path)
    args = p.parse_args()

    control_files = sorted(args.shards_root.glob("**/stability_controls.csv"))
    shuffle_files = sorted(args.shards_root.glob("**/target_shuffle.csv"))
    if not control_files or not shuffle_files:
        raise RuntimeError("missing stability shard outputs")

    controls = pd.concat([pd.read_csv(x) for x in control_files], ignore_index=True)
    shuffle = pd.concat([pd.read_csv(x) for x in shuffle_files], ignore_index=True)
    if controls["sample_index"].nunique() != 100 or shuffle["sample_index"].nunique() != 100:
        raise RuntimeError("expected 100 unique base anomalies")

    rows = []
    metrics = [
        "own_fidelity_x",
        "interaction_jaccard",
        "feature_jaccard",
        "sign_agreement_shared",
        "cross_fidelity_x_to_neighbor",
    ]
    for (method, budget), g in controls.groupby(["method", "budget"], sort=True):
        row = {"method": method, "budget": int(budget), "n": len(g)}
        for metric in metrics:
            s = summary_stats(g, metric)
            for stat, value in s.items():
                row[f"{metric}_{stat}"] = value
        row["fraction_own_le_5pct"] = float((g["own_fidelity_x"] <= 0.05).mean())
        row["fraction_cross_le_5pct"] = float((g["cross_fidelity_x_to_neighbor"] <= 0.05).mean())
        rows.append(row)
    summary = pd.DataFrame(rows)

    shuffle_summary = {}
    for pct in [10, 5, 1]:
        tc = f"true_terms_to_{pct}pct"
        sc = f"shuffle_terms_to_{pct}pct"
        true_ok = shuffle[tc].notna()
        shuf_ok = shuffle[sc].notna()
        shuffle_summary[f"threshold_{pct}pct"] = {
            "true_achieved_fraction": float(true_ok.mean()),
            "shuffle_achieved_fraction": float(shuf_ok.mean()),
            "true_median_terms": float(shuffle.loc[true_ok, tc].median()) if true_ok.any() else None,
            "shuffle_median_terms": float(shuffle.loc[shuf_ok, sc].median()) if shuf_ok.any() else None,
        }

    neighbor = controls.drop_duplicates("sample_index")
    report = {
        "dataset": "NSL-KDD",
        "samples": 100,
        "neighbor_control": {
            "median_hamming_distance": float(neighbor["hamming_distance"].median()),
            "q25_hamming_distance": float(neighbor["hamming_distance"].quantile(0.25)),
            "q75_hamming_distance": float(neighbor["hamming_distance"].quantile(0.75)),
            "median_score_relative_difference": float(neighbor["score_relative_difference"].median()),
        },
        "budget_10": {},
        "target_shuffle": shuffle_summary,
        "interpretation": (
            "High own fidelity alone is insufficient. Stable explanations should "
            "also show interaction/feature overlap and reasonable cross-fidelity "
            "for nearby same-family attacks. If shuffled targets are reconstructed "
            "with similar sparsity, sparse greedy fidelity is largely a subset-sum control effect."
        ),
    }
    b10 = summary[summary["budget"] == 10]
    for _, row in b10.iterrows():
        report["budget_10"][row["method"]] = {
            "median_own_fidelity": float(row["own_fidelity_x_median"]),
            "median_interaction_jaccard": float(row["interaction_jaccard_median"]),
            "median_feature_jaccard": float(row["feature_jaccard_median"]),
            "median_cross_fidelity": float(row["cross_fidelity_x_to_neighbor_median"]),
            "fraction_own_le_5pct": float(row["fraction_own_le_5pct"]),
            "fraction_cross_le_5pct": float(row["fraction_cross_le_5pct"]),
        }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    controls.to_csv(args.output_dir / "stability_controls.csv", index=False)
    shuffle.to_csv(args.output_dir / "target_shuffle.csv", index=False)
    summary.to_csv(args.output_dir / "stability_summary.csv", index=False)
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
