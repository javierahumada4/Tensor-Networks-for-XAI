"""Evaluate the frozen final Born-MPS models on the six ADBench datasets.

Evaluation is threshold-free: AUROC and average precision (AUPRC) are computed
from the held-out test scores. The independent empirical-marginal initializer is
reported as a required baseline, using exactly the same encoded split and
pseudocount as the trained MPS.

Hyperparameters are never selected from test labels.
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


def independent_nll(
    train_x: torch.Tensor,
    test_x: torch.Tensor,
    physical_dims: list[int],
    *,
    pseudocount: float = 1e-6,
) -> torch.Tensor:
    """NLL under the product of smoothed empirical one-site marginals."""
    scores = torch.zeros(len(test_x), dtype=torch.float64)
    for site, physical_dim in enumerate(physical_dims):
        counts = torch.bincount(
            train_x[:, site].long(),
            minlength=physical_dim,
        ).to(dtype=torch.float64)
        probabilities = counts + float(pseudocount)
        probabilities /= probabilities.sum()
        scores -= torch.log(probabilities[test_x[:, site].long()])
    return scores


def score_summary(scores: np.ndarray, labels: np.ndarray) -> dict:
    normal = scores[labels == 0]
    anomaly = scores[labels == 1]
    return {
        "normal_mean": float(np.mean(normal)),
        "normal_median": float(np.median(normal)),
        "anomaly_mean": float(np.mean(anomaly)),
        "anomaly_median": float(np.median(anomaly)),
    }


def evaluate_dataset(
    dataset_root: Path,
    model_root: Path,
    dataset: str,
) -> dict:
    data = load_encoded_bundle(dataset_root / dataset)
    output_dir = model_root / dataset

    score_payload = torch.load(
        output_dir / "test_scores.pt",
        map_location="cpu",
        weights_only=True,
    )
    summary = json.loads((output_dir / "summary.json").read_text())

    labels = score_payload["y"].cpu().numpy().astype(np.int64)
    trained_scores = score_payload["nll"].cpu().numpy().astype(np.float64)

    if len(labels) != len(data.test.y):
        raise ValueError(f"{dataset}: score/test length mismatch")
    if not np.array_equal(labels, data.test.y.cpu().numpy()):
        raise ValueError(f"{dataset}: labels in model artifact differ from data bundle")
    if not torch.equal(
        score_payload["row_indices"].cpu(),
        data.test.row_indices.cpu(),
    ):
        raise ValueError(f"{dataset}: row indices differ from frozen test split")
    if not np.isfinite(trained_scores).all():
        raise ValueError(f"{dataset}: non-finite trained NLL scores")

    baseline_scores = independent_nll(
        data.train.x,
        data.test.x,
        data.physical_dims,
        pseudocount=1e-6,
    ).numpy()

    trained_auroc = float(roc_auc_score(labels, trained_scores))
    trained_auprc = float(average_precision_score(labels, trained_scores))
    baseline_auroc = float(roc_auc_score(labels, baseline_scores))
    baseline_auprc = float(average_precision_score(labels, baseline_scores))

    return {
        "dataset": dataset,
        "test_samples": int(len(labels)),
        "test_anomalies": int(labels.sum()),
        "prevalence": float(labels.mean()),
        "trained_auroc": trained_auroc,
        "trained_auprc": trained_auprc,
        "independent_auroc": baseline_auroc,
        "independent_auprc": baseline_auprc,
        "delta_auroc": trained_auroc - baseline_auroc,
        "delta_auprc": trained_auprc - baseline_auprc,
        "trained_scores": score_summary(trained_scores, labels),
        "independent_scores": score_summary(baseline_scores, labels),
        "initial_val_nll": float(summary["initial"]["val_nll"]),
        "restored_val_nll": float(summary["restored_model"]["val_nll"]),
        "relative_val_nll_improvement": (
            float(summary["initial"]["val_nll"])
            - float(summary["restored_model"]["val_nll"])
        ) / abs(float(summary["initial"]["val_nll"])),
        "best_loop": int(summary["best_record"]["loop"]),
        "num_history_records": int(summary["num_history_records"]),
        "bond_dims": summary["restored_model"]["bond_dims"],
        "total_skipped_nan": int(summary["total_skipped_nan"]),
        "elapsed_s": float(summary["elapsed_s"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("model_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    results = [
        evaluate_dataset(args.dataset_root, args.model_root, dataset)
        for dataset in DATASETS
    ]

    aggregate = {
        "macro_trained_auroc": float(np.mean([r["trained_auroc"] for r in results])),
        "macro_trained_auprc": float(np.mean([r["trained_auprc"] for r in results])),
        "macro_independent_auroc": float(np.mean([r["independent_auroc"] for r in results])),
        "macro_independent_auprc": float(np.mean([r["independent_auprc"] for r in results])),
        "macro_delta_auroc": float(np.mean([r["delta_auroc"] for r in results])),
        "macro_delta_auprc": float(np.mean([r["delta_auprc"] for r in results])),
        "datasets_trained_auroc_better": sum(r["delta_auroc"] > 0 for r in results),
        "datasets_trained_auprc_better": sum(r["delta_auprc"] > 0 for r in results),
    }

    report = {
        "evaluation": (
            "held-out test; higher NLL is the anomaly score; no test labels "
            "were used for model or hyperparameter selection"
        ),
        "baseline": (
            "product of smoothed empirical one-site marginals used to initialize "
            "the MPS"
        ),
        "results": results,
        "aggregate": aggregate,
    }

    (args.output_dir / "evaluation.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )

    flat_fields = [
        "dataset",
        "test_samples",
        "test_anomalies",
        "prevalence",
        "trained_auroc",
        "trained_auprc",
        "independent_auroc",
        "independent_auprc",
        "delta_auroc",
        "delta_auprc",
        "initial_val_nll",
        "restored_val_nll",
        "relative_val_nll_improvement",
        "best_loop",
        "num_history_records",
        "total_skipped_nan",
        "elapsed_s",
    ]
    with (args.output_dir / "evaluation.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for result in results:
            writer.writerow({key: result[key] for key in flat_fields})

    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
