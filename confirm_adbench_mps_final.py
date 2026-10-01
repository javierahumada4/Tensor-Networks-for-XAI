"""Final focused hyperparameter confirmation before freezing paper models.

This experiment resolves the only remaining ambiguity from the previous
screening stages: whether disabling discarded-weight truncation (epsilon=0)
provides a reproducible validation-NLL benefit over epsilon=1e-6.

The two strongest positive-epsilon configurations from the prior confirmation
are compared with otherwise matched epsilon=0 variants. Model selection uses
normal validation NLL only.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean

from screen_adbench_mps import DATASETS, run_candidate


CANDIDATES = [
    {
        "name": "D16_e6_lr2e-3",
        "max_bond_dim": 16,
        "epsilon_trunc": 1e-6,
        "lr": 2e-3,
    },
    {
        "name": "D16_e0_lr2e-3",
        "max_bond_dim": 16,
        "epsilon_trunc": 0.0,
        "lr": 2e-3,
    },
    {
        "name": "D32_e6_lr8e-4",
        "max_bond_dim": 32,
        "epsilon_trunc": 1e-6,
        "lr": 8e-4,
    },
    {
        "name": "D32_e0_lr8e-4",
        "max_bond_dim": 32,
        "epsilon_trunc": 0.0,
        "lr": 8e-4,
    },
]


def select(results: list[dict]) -> dict:
    by_dataset: dict[str, list[dict]] = {}
    for row in results:
        by_dataset.setdefault(row["dataset"], []).append(row)

    regrets = {candidate["name"]: [] for candidate in CANDIDATES}
    runtimes = {candidate["name"]: [] for candidate in CANDIDATES}
    improvements = {candidate["name"]: [] for candidate in CANDIDATES}
    details = {}

    for dataset, rows in by_dataset.items():
        finite_rows = [
            row for row in rows
            if float(row["best_val_nll"]) < float("inf")
        ]
        if not finite_rows:
            raise RuntimeError(f"all candidates failed for {dataset}")

        best = min(float(row["best_val_nll"]) for row in finite_rows)
        details[dataset] = {}

        for row in rows:
            value = float(row["best_val_nll"])
            regret = (
                (value - best) / max(abs(best), 1e-12)
                if value < float("inf")
                else float("inf")
            )
            name = row["candidate"]
            regrets[name].append(regret)
            runtimes[name].append(float(row["elapsed_s"]))
            improvements[name].append(float(row["relative_val_improvement"]))
            details[dataset][name] = {
                "best_val_nll": value,
                "relative_regret": regret,
                "relative_val_improvement": row["relative_val_improvement"],
                "best_loop": row["best_loop"],
                "bond_dims": row["restored_bond_dims"],
                "max_discarded_weight": row["max_discarded_weight"],
                "num_skipped_nan": row["num_skipped_nan"],
            }

    aggregate = []
    for candidate in CANDIDATES:
        name = candidate["name"]
        aggregate.append({
            **candidate,
            "mean_relative_regret": mean(regrets[name]),
            "max_relative_regret": max(regrets[name]),
            "mean_relative_val_improvement": mean(improvements[name]),
            "mean_runtime_s": mean(runtimes[name]),
            "datasets_won": sum(
                1 for dataset in details
                if details[dataset][name]["relative_regret"] == 0.0
            ),
        })

    aggregate.sort(
        key=lambda row: (
            row["mean_relative_regret"],
            row["max_relative_regret"],
            -row["datasets_won"],
            row["mean_runtime_s"],
        )
    )

    return {
        "selection_rule": (
            "minimize mean relative normal-validation NLL regret; then "
            "worst-dataset regret; then number of dataset wins; then runtime"
        ),
        "selected": aggregate[0],
        "aggregate": aggregate,
        "dataset_details": details,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--loops", type=int, default=10)
    parser.add_argument("--batches-per-loop", type=int, default=16)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    histories = args.output_dir / "histories"
    histories.mkdir(exist_ok=True)

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
                f"  val={result['best_val_nll']:.6f} "
                f"imp={100 * result['relative_val_improvement']:.2f}% "
                f"time={result['elapsed_s']:.1f}s "
                f"bonds={result['restored_bond_dims']}",
                flush=True,
            )

    selection = select(results)
    (args.output_dir / "results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )

    print("SELECTED", json.dumps(selection["selected"], indent=2), flush=True)
    print("AGGREGATE", json.dumps(selection["aggregate"], indent=2), flush=True)


if __name__ == "__main__":
    main()
