"""Compare regenerated ADBench metrics with the invalid pre-fix run.

The legacy values below are retained only to quantify the impact of the encoder
bug. They must not be used as paper results. Per-dataset values are the rounded
numbers reported from the pre-fix evaluation; macro values preserve the
aggregate numbers reported by that evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


INVALID_PREFX = {
    "annthyroid": {"auroc": 0.490, "auprc": 0.388},
    "cardio": {"auroc": 0.891, "auprc": 0.897},
    "cover": {"auroc": 0.543, "auprc": 0.337},
    "mammography": {"auroc": 0.848, "auprc": 0.562},
    "shuttle": {"auroc": 0.995, "auprc": 0.994},
    "vowels": {"auroc": 0.778, "auprc": 0.388},
}
INVALID_PREFX_MACRO = {"auroc": 0.7574, "auprc": 0.5942}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evaluation_json", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    report = json.loads(args.evaluation_json.read_text(encoding="utf-8"))
    rows = []
    by_name = {row["dataset"]: row for row in report["results"]}

    missing = sorted(set(INVALID_PREFX) - set(by_name))
    if missing:
        raise ValueError(f"regenerated evaluation is missing datasets: {missing}")

    for dataset, old in INVALID_PREFX.items():
        new = by_name[dataset]
        rows.append(
            {
                "dataset": dataset,
                "invalid_prefx_auroc": old["auroc"],
                "regenerated_auroc": float(new["trained_auroc"]),
                "delta_auroc": float(new["trained_auroc"]) - old["auroc"],
                "invalid_prefx_auprc": old["auprc"],
                "regenerated_auprc": float(new["trained_auprc"]),
                "delta_auprc": float(new["trained_auprc"]) - old["auprc"],
            }
        )

    aggregate = report["aggregate"]
    macro = {
        "invalid_prefx_auroc": INVALID_PREFX_MACRO["auroc"],
        "regenerated_auroc": float(aggregate["macro_trained_auroc"]),
        "delta_auroc": (
            float(aggregate["macro_trained_auroc"])
            - INVALID_PREFX_MACRO["auroc"]
        ),
        "invalid_prefx_auprc": INVALID_PREFX_MACRO["auprc"],
        "regenerated_auprc": float(aggregate["macro_trained_auprc"]),
        "delta_auprc": (
            float(aggregate["macro_trained_auprc"])
            - INVALID_PREFX_MACRO["auprc"]
        ),
    }

    comparison = {
        "warning": (
            "The pre-fix metrics are invalid scientific results because their "
            "encoded ADBench bundles were produced by the defective encoder. "
            "They are included only as a diagnostic reference to measure the "
            "effect of regeneration."
        ),
        "legacy_values_precision": (
            "Per-dataset pre-fix values are rounded to three decimals; macro "
            "pre-fix values come from the original aggregate report."
        ),
        "results": rows,
        "macro": macro,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "regeneration_comparison.json").write_text(
        json.dumps(comparison, indent=2),
        encoding="utf-8",
    )

    fields = [
        "dataset",
        "invalid_prefx_auroc",
        "regenerated_auroc",
        "delta_auroc",
        "invalid_prefx_auprc",
        "regenerated_auprc",
        "delta_auprc",
    ]
    with (args.output_dir / "regeneration_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)

    print(json.dumps(comparison, indent=2))


if __name__ == "__main__":
    main()
