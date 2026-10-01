"""Screen common MPS hyperparameters on the six prepared ADBench datasets.

The screening is deliberately small and one-factor-at-a-time around a central
configuration. Every candidate sees the same deterministic minibatch sequence
within a dataset. Model selection uses normal validation NLL only.

This is a screening stage, not the final training run: batches_per_loop is
capped to keep the search affordable. After selecting one common configuration,
final models are trained with full epochs in train_adbench_final.py.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from statistics import mean, median

import torch

from data_artifacts import load_encoded_bundle
from dmrg_trainer import DMRGConfig, dmrg_train
from mps import MPS


DATASETS = [
    "annthyroid",
    "cardio",
    "cover",
    "mammography",
    "shuttle",
    "vowels",
]

CENTRAL = {
    "max_bond_dim": 32,
    "epsilon_trunc": 1e-5,
    "lr": 8e-4,
}

CANDIDATES = [
    {"name": "D8", "max_bond_dim": 8, "epsilon_trunc": 1e-5, "lr": 8e-4},
    {"name": "D32", **CENTRAL},
    {"name": "D64", "max_bond_dim": 64, "epsilon_trunc": 1e-5, "lr": 8e-4},
    {"name": "eps1e-4", "max_bond_dim": 32, "epsilon_trunc": 1e-4, "lr": 8e-4},
    {"name": "eps1e-6", "max_bond_dim": 32, "epsilon_trunc": 1e-6, "lr": 8e-4},
    {"name": "lr3e-4", "max_bond_dim": 32, "epsilon_trunc": 1e-5, "lr": 3e-4},
    {"name": "lr2e-3", "max_bond_dim": 32, "epsilon_trunc": 1e-5, "lr": 2e-3},
]


def _finite(value: float) -> bool:
    return math.isfinite(float(value))


def run_candidate(
    dataset_root: Path,
    dataset_name: str,
    candidate: dict,
    *,
    batch_size: int,
    loops: int,
    batches_per_loop: int,
    seed: int,
) -> dict:
    data = load_encoded_bundle(dataset_root / dataset_name)

    model = MPS.from_empirical_frequencies(
        data.train.x,
        physical_dims=data.physical_dims,
        dtype=torch.float64,
        pseudocount=1e-6,
    )

    init_train_nll = float(model.nll(data.train.x, batch_size=batch_size))
    init_val_nll = float(model.nll(data.val.x, batch_size=batch_size))

    config = DMRGConfig(
        num_descent_steps=1,
        max_bond_dim=candidate["max_bond_dim"],
        epsilon_trunc=candidate["epsilon_trunc"],
        lr=candidate["lr"],
        num_loops=loops,
        batch_size=batch_size,
        # Prevent the LR scheduler from changing the candidate during screening.
        patience=loops + 10,
        lr_shrink=0.5,
        lr_min=1e-12,
        early_stopping_patience=0,
        abort_after_dead_loops=2,
        batches_per_loop=batches_per_loop,
        metric_for_stopping="val_nll",
        seed=seed,
    )

    start = time.perf_counter()
    history = dmrg_train(model, data.train.x, data.val.x, config=config)
    elapsed = time.perf_counter() - start

    val_values = [
        float(record["val_nll"])
        for record in history
        if "val_nll" in record and _finite(record["val_nll"])
    ]
    train_values = [
        float(record["train_nll"])
        for record in history
        if _finite(record["train_nll"])
    ]

    best_val_nll = min(val_values) if val_values else float("inf")
    best_train_nll = min(train_values) if train_values else float("inf")
    best_loop = (
        min(
            (record for record in history if _finite(record.get("val_nll", float("inf")))),
            key=lambda record: record["val_nll"],
        )["loop"]
        if val_values
        else None
    )

    skipped = sum(int(record.get("num_skipped_nan", 0)) for record in history)
    max_discarded = max(
        (float(record.get("max_discarded_weight", 0.0)) for record in history),
        default=0.0,
    )

    relative_improvement = (
        (init_val_nll - best_val_nll) / abs(init_val_nll)
        if _finite(best_val_nll) and init_val_nll != 0
        else float("-inf")
    )

    return {
        "dataset": dataset_name,
        "candidate": candidate["name"],
        "max_bond_dim": candidate["max_bond_dim"],
        "epsilon_trunc": candidate["epsilon_trunc"],
        "lr": candidate["lr"],
        "init_train_nll": init_train_nll,
        "init_val_nll": init_val_nll,
        "best_train_nll": best_train_nll,
        "best_val_nll": best_val_nll,
        "best_loop": best_loop,
        "relative_val_improvement": relative_improvement,
        "elapsed_s": elapsed,
        "num_history_records": len(history),
        "num_skipped_nan": skipped,
        "max_discarded_weight": max_discarded,
        "restored_bond_dims": list(model.bond_dims),
        "history": history,
    }


def select_common_configuration(results: list[dict]) -> dict:
    by_dataset: dict[str, list[dict]] = {}
    for result in results:
        by_dataset.setdefault(result["dataset"], []).append(result)

    rank_accumulator = {candidate["name"]: [] for candidate in CANDIDATES}
    improvement_accumulator = {candidate["name"]: [] for candidate in CANDIDATES}
    runtime_accumulator = {candidate["name"]: [] for candidate in CANDIDATES}
    failures = {candidate["name"]: 0 for candidate in CANDIDATES}

    dataset_rankings = {}
    for dataset, rows in by_dataset.items():
        valid = [row for row in rows if _finite(row["best_val_nll"])]
        invalid = [row for row in rows if not _finite(row["best_val_nll"])]
        valid.sort(key=lambda row: row["best_val_nll"])

        ranking = {}
        for rank, row in enumerate(valid, start=1):
            ranking[row["candidate"]] = rank
            rank_accumulator[row["candidate"]].append(rank)
            improvement_accumulator[row["candidate"]].append(
                row["relative_val_improvement"]
            )
            runtime_accumulator[row["candidate"]].append(row["elapsed_s"])

        penalty = len(CANDIDATES) + 1
        for row in invalid:
            ranking[row["candidate"]] = penalty
            rank_accumulator[row["candidate"]].append(penalty)
            failures[row["candidate"]] += 1

        dataset_rankings[dataset] = ranking

    aggregate = []
    for candidate in CANDIDATES:
        name = candidate["name"]
        aggregate.append(
            {
                **candidate,
                "mean_rank": mean(rank_accumulator[name]),
                "median_rank": median(rank_accumulator[name]),
                "mean_relative_val_improvement": mean(
                    improvement_accumulator[name]
                )
                if improvement_accumulator[name]
                else float("-inf"),
                "median_runtime_s": median(runtime_accumulator[name])
                if runtime_accumulator[name]
                else float("inf"),
                "failures": failures[name],
            }
        )

    aggregate.sort(
        key=lambda row: (
            row["failures"],
            row["mean_rank"],
            -row["mean_relative_val_improvement"],
            row["median_runtime_s"],
        )
    )
    selected = aggregate[0]

    return {
        "selection_rule": (
            "lowest failures, then lowest mean within-dataset validation-NLL "
            "rank, then highest mean relative validation-NLL improvement, "
            "then lower median runtime"
        ),
        "selected": selected,
        "aggregate": aggregate,
        "dataset_rankings": dataset_rankings,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--loops", type=int, default=5)
    parser.add_argument("--batches-per-loop", type=int, default=8)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    histories_dir = args.output_dir / "histories"
    histories_dir.mkdir(parents=True, exist_ok=True)

    results = []
    total = len(DATASETS) * len(CANDIDATES)
    done = 0

    for dataset in DATASETS:
        for candidate in CANDIDATES:
            done += 1
            print(
                f"[{done}/{total}] {dataset} / {candidate['name']} "
                f"(D={candidate['max_bond_dim']}, "
                f"eps={candidate['epsilon_trunc']:.0e}, "
                f"lr={candidate['lr']:.1e})",
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
            (histories_dir / f"{dataset}__{candidate['name']}.json").write_text(
                json.dumps(history, indent=2),
                encoding="utf-8",
            )
            results.append(result)
            print(
                f"  init_val={result['init_val_nll']:.6f} "
                f"best_val={result['best_val_nll']:.6f} "
                f"improvement={100*result['relative_val_improvement']:.3f}% "
                f"time={result['elapsed_s']:.1f}s "
                f"bonds={result['restored_bond_dims']}",
                flush=True,
            )

    selection = select_common_configuration(results)

    (args.output_dir / "screening_results.json").write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    (args.output_dir / "selection.json").write_text(
        json.dumps(selection, indent=2),
        encoding="utf-8",
    )

    fields = [
        "dataset", "candidate", "max_bond_dim", "epsilon_trunc", "lr",
        "init_train_nll", "init_val_nll", "best_train_nll", "best_val_nll",
        "best_loop", "relative_val_improvement", "elapsed_s",
        "num_history_records", "num_skipped_nan", "max_discarded_weight",
        "restored_bond_dims",
    ]
    with (args.output_dir / "screening_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow(result)

    print("\nSelected common configuration:", flush=True)
    print(json.dumps(selection["selected"], indent=2), flush=True)


if __name__ == "__main__":
    main()
