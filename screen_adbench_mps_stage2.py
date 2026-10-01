"""Second-stage MPS hyperparameter screening after epsilon=1e-6 unlocked rank growth.

This stage varies D_max and learning rate at epsilon_trunc=1e-6, and probes
smaller truncation tolerances (including no tolerance-based truncation) because
stage 1 selected the smallest epsilon that was tested.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from statistics import mean, median

from screen_adbench_mps import DATASETS, run_candidate


CANDIDATES = [
    {"name": "D8_e6", "max_bond_dim": 8, "epsilon_trunc": 1e-6, "lr": 8e-4},
    {"name": "D16_e6", "max_bond_dim": 16, "epsilon_trunc": 1e-6, "lr": 8e-4},
    {"name": "D32_e6", "max_bond_dim": 32, "epsilon_trunc": 1e-6, "lr": 8e-4},
    {"name": "D64_e6", "max_bond_dim": 64, "epsilon_trunc": 1e-6, "lr": 8e-4},
    {"name": "lr3e-4_e6", "max_bond_dim": 32, "epsilon_trunc": 1e-6, "lr": 3e-4},
    {"name": "lr2e-3_e6", "max_bond_dim": 32, "epsilon_trunc": 1e-6, "lr": 2e-3},
    {"name": "eps1e-7", "max_bond_dim": 32, "epsilon_trunc": 1e-7, "lr": 8e-4},
    {"name": "eps1e-8", "max_bond_dim": 32, "epsilon_trunc": 1e-8, "lr": 8e-4},
    {"name": "eps0", "max_bond_dim": 32, "epsilon_trunc": 0.0, "lr": 8e-4},
]


def finite(value):
    return math.isfinite(float(value))


def select(results):
    by_dataset = {}
    for row in results:
        by_dataset.setdefault(row["dataset"], []).append(row)

    ranks = {c["name"]: [] for c in CANDIDATES}
    improvements = {c["name"]: [] for c in CANDIDATES}
    runtimes = {c["name"]: [] for c in CANDIDATES}
    skipped = {c["name"]: 0 for c in CANDIDATES}
    dataset_rankings = {}

    for dataset, rows in by_dataset.items():
        ordered = sorted(
            rows,
            key=lambda r: r["best_val_nll"] if finite(r["best_val_nll"]) else float("inf"),
        )
        ranking = {}
        for rank, row in enumerate(ordered, 1):
            ranking[row["candidate"]] = rank
            ranks[row["candidate"]].append(rank)
            improvements[row["candidate"]].append(row["relative_val_improvement"])
            runtimes[row["candidate"]].append(row["elapsed_s"])
            skipped[row["candidate"]] += row["num_skipped_nan"]
        dataset_rankings[dataset] = ranking

    aggregate = []
    for candidate in CANDIDATES:
        name = candidate["name"]
        aggregate.append({
            **candidate,
            "mean_rank": mean(ranks[name]),
            "median_rank": median(ranks[name]),
            "mean_relative_val_improvement": mean(improvements[name]),
            "median_relative_val_improvement": median(improvements[name]),
            "median_runtime_s": median(runtimes[name]),
            "total_skipped_nan": skipped[name],
        })

    aggregate.sort(key=lambda row: (
        row["total_skipped_nan"] > 0,
        row["mean_rank"],
        -row["mean_relative_val_improvement"],
        row["median_runtime_s"],
    ))
    return {
        "selection_rule": (
            "prefer zero non-finite updates; then lowest mean within-dataset "
            "validation-NLL rank; then highest mean relative validation-NLL "
            "improvement; then lower median runtime"
        ),
        "selected": aggregate[0],
        "aggregate": aggregate,
        "dataset_rankings": dataset_rankings,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--batches-per-loop", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    histories = args.output_dir / "histories"
    histories.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(DATASETS) * len(CANDIDATES)
    counter = 0
    for dataset in DATASETS:
        for candidate in CANDIDATES:
            counter += 1
            print(
                f"[{counter}/{total}] {dataset}/{candidate['name']} "
                f"D={candidate['max_bond_dim']} "
                f"eps={candidate['epsilon_trunc']:.0e} "
                f"lr={candidate['lr']:.1e}",
                flush=True,
            )
            result = run_candidate(
                args.dataset_root,
                dataset,
                candidate,
                batch_size=args.batch_size,
                loops=args.loops,
                batches_per_loop=args.batches_per_loop,
                seed=args.seed,
            )
            history = result.pop("history")
            (histories / f"{dataset}__{candidate['name']}.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )
            results.append(result)
            print(
                f"  best_val={result['best_val_nll']:.6f} "
                f"imp={100*result['relative_val_improvement']:.3f}% "
                f"time={result['elapsed_s']:.2f}s "
                f"bonds={result['restored_bond_dims']}",
                flush=True,
            )

    selection = select(results)
    (args.output_dir / "screening_results.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2), encoding="utf-8"
    )

    fields = list(results[0].keys())
    with (args.output_dir / "screening_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(results)

    print("\nSelected stage-2 configuration:", flush=True)
    print(json.dumps(selection["selected"], indent=2), flush=True)
    print("\nAggregate ranking:", flush=True)
    print(json.dumps(selection["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
