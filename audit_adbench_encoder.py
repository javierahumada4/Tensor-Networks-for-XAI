"""Audit corrected ADBench encoding before retraining the MPS models.

The script evaluates only the independent empirical-marginal model on the
persistent encoded splits. This isolates changes caused by the encoder from
changes caused by MPS optimization.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from data_artifacts import load_encoded_bundle


DATASETS = [
    "annthyroid",
    "cardio",
    "cover",
    "mammography",
    "shuttle",
    "vowels",
]


def independent_nll(train_x, test_x, physical_dims, pseudocount=1e-6):
    score = torch.zeros(len(test_x), dtype=torch.float64)
    for site, dim in enumerate(physical_dims):
        counts = torch.bincount(
            train_x[:, site].long(),
            minlength=dim,
        ).double()
        probabilities = counts + pseudocount
        probabilities /= probabilities.sum()
        score -= torch.log(probabilities[test_x[:, site].long()])
    return score.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--previous-results", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    previous = {}
    if args.previous_results and args.previous_results.exists():
        with args.previous_results.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                previous[row["dataset"]] = {
                    "independent_auroc": float(row["independent_auroc"]),
                    "independent_auprc": float(row["independent_auprc"]),
                }

    results = []
    for dataset in DATASETS:
        data = load_encoded_bundle(args.dataset_root / dataset)
        scores = independent_nll(
            data.train.x,
            data.test.x,
            data.physical_dims,
        )
        labels = data.test.y.numpy().astype(np.int64)
        auroc = float(roc_auc_score(labels, scores))
        auprc = float(average_precision_score(labels, scores))

        kind_counts = {}
        for spec in data.encoder.specs:
            kind_counts[spec.kind] = kind_counts.get(spec.kind, 0) + 1

        row = {
            "dataset": dataset,
            "test_samples": len(labels),
            "test_anomalies": int(labels.sum()),
            "features": len(data.feature_names),
            "physical_dim_sum": int(sum(data.physical_dims)),
            "max_physical_dim": int(max(data.physical_dims)),
            "feature_kinds": kind_counts,
            "independent_auroc": auroc,
            "independent_auprc": auprc,
        }
        if dataset in previous:
            row["previous_independent_auroc"] = previous[dataset]["independent_auroc"]
            row["previous_independent_auprc"] = previous[dataset]["independent_auprc"]
            row["delta_auroc"] = auroc - row["previous_independent_auroc"]
            row["delta_auprc"] = auprc - row["previous_independent_auprc"]

        results.append(row)
        print(json.dumps(row), flush=True)

    report = {
        "purpose": "encoder-only ADBench regression after fixing degenerate quantile binning",
        "results": results,
        "macro": {
            "independent_auroc": float(np.mean([r["independent_auroc"] for r in results])),
            "independent_auprc": float(np.mean([r["independent_auprc"] for r in results])),
        },
    }
    if all("delta_auroc" in row for row in results):
        report["macro"]["delta_auroc_vs_previous"] = float(
            np.mean([r["delta_auroc"] for r in results])
        )
        report["macro"]["delta_auprc_vs_previous"] = float(
            np.mean([r["delta_auprc"] for r in results])
        )

    (args.output_dir / "audit.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    fields = [
        "dataset", "test_samples", "test_anomalies", "features",
        "physical_dim_sum", "max_physical_dim",
        "independent_auroc", "independent_auprc",
        "previous_independent_auroc", "previous_independent_auprc",
        "delta_auroc", "delta_auprc",
    ]
    with (args.output_dir / "audit.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
