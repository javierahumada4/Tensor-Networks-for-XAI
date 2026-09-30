"""Longer confirmation screen for common positive-epsilon MPS hyperparameters.

Stage 1 identified epsilon_trunc=1e-6 as the point where empirical-product
initialization can reliably grow correlated ranks. Stage 2 showed no consistent
benefit from smaller positive epsilons. This stage therefore fixes epsilon=1e-6
and confirms the most plausible D_max / learning-rate combinations with more
updates.

Selection minimizes mean relative validation-NLL regret across datasets:
    regret(c,d) = (L(c,d) - min_c L(c,d)) / |min_c L(c,d)|
Only normal validation NLL is used.
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
    {"name": "D8_lr8e-4", "max_bond_dim": 8, "epsilon_trunc": 1e-6, "lr": 8e-4},
    {"name": "D16_lr8e-4", "max_bond_dim": 16, "epsilon_trunc": 1e-6, "lr": 8e-4},
    {"name": "D32_lr8e-4", "max_bond_dim": 32, "epsilon_trunc": 1e-6, "lr": 8e-4},
    {"name": "D64_lr8e-4", "max_bond_dim": 64, "epsilon_trunc": 1e-6, "lr": 8e-4},
    {"name": "D16_lr2e-3", "max_bond_dim": 16, "epsilon_trunc": 1e-6, "lr": 2e-3},
    {"name": "D32_lr2e-3", "max_bond_dim": 32, "epsilon_trunc": 1e-6, "lr": 2e-3},
]


def select_by_relative_regret(results):
    by_dataset = {}
    for row in results:
        by_dataset.setdefault(row["dataset"], []).append(row)

    regrets = {c["name"]: [] for c in CANDIDATES}
    improvements = {c["name"]: [] for c in CANDIDATES}
    runtimes = {c["name"]: [] for c in CANDIDATES}
    skipped = {c["name"]: 0 for c in CANDIDATES}
    dataset_details = {}

    for dataset, rows in by_dataset.items():
        finite_rows = [
            r for r in rows if math.isfinite(float(r["best_val_nll"]))
        ]
        if not finite_rows:
            raise RuntimeError(f"all candidates failed for {dataset}")

        best = min(float(r["best_val_nll"]) for r in finite_rows)
        details = {}
        for row in rows:
            name = row["candidate"]
            value = float(row["best_val_nll"])
            if math.isfinite(value):
                regret = (value - best) / max(abs(best), 1e-12)
            else:
                regret = float("inf")
            regrets[name].append(regret)
            improvements[name].append(row["relative_val_improvement"])
            runtimes[name].append(row["elapsed_s"])
            skipped[name] += row["num_skipped_nan"]
            details[name] = {
                "best_val_nll": value,
                "relative_regret": regret,
            }
        dataset_details[dataset] = {
            "best_val_nll": best,
            "candidates": details,
        }

    aggregate = []
    for candidate in CANDIDATES:
        name = candidate["name"]
        finite_regrets = [r for r in regrets[name] if math.isfinite(r)]
        aggregate.append({
            **candidate,
            "mean_relative_regret": mean(finite_regrets)
                if len(finite_regrets) == len(DATASETS) else float("inf"),
            "median_relative_regret": median(finite_regrets)
                if finite_regrets else float("inf"),
            "max_relative_regret": max(finite_regrets)
                if finite_regrets else float("inf"),
            "mean_relative_val_improvement": mean(improvements[name]),
            "median_runtime_s": median(runtimes[name]),
            "total_skipped_nan": skipped[name],
        })

    aggregate.sort(key=lambda row: (
        row["total_skipped_nan"] > 0,
        row["mean_relative_regret"],
        row["max_relative_regret"],
        row["median_runtime_s"],
    ))

    return {
        "selection_rule": (
            "positive epsilon only; prefer zero non-finite updates; minimize "
            "mean relative validation-NLL regret across datasets; then minimize "
            "worst-dataset regret; then lower median runtime"
        ),
        "selected": aggregate[0],
        "aggregate": aggregate,
        "dataset_details": dataset_details,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--batches-per-loop", type=int, default=24)
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
                f"D={candidate['max_bond_dim']} eps=1e-6 "
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
                json.dumps(history, indent=2), encoding="utf-8"
            )
            results.append(result)
            print(
                f"  val={result['best_val_nll']:.6f} "
                f"imp={100*result['relative_val_improvement']:.3f}% "
                f"time={result['elapsed_s']:.1f}s "
                f"bonds={result['restored_bond_dims']}",
                flush=True,
            )

    selection = select_by_relative_regret(results)

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

    print("\nSelected confirmation configuration:", flush=True)
    print(json.dumps(selection["selected"], indent=2), flush=True)
    print("\nAggregate:", flush=True)
    print(json.dumps(selection["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
