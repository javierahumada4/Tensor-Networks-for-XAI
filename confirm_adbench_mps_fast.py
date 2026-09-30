"""Focused confirmation of the four competitive common MPS configurations."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from statistics import mean

from screen_adbench_mps import DATASETS, run_candidate


CANDIDATES = [
    {"name": "D16_lr8e-4", "max_bond_dim": 16, "epsilon_trunc": 1e-6, "lr": 8e-4},
    {"name": "D16_lr2e-3", "max_bond_dim": 16, "epsilon_trunc": 1e-6, "lr": 2e-3},
    {"name": "D32_lr8e-4", "max_bond_dim": 32, "epsilon_trunc": 1e-6, "lr": 8e-4},
    {"name": "D32_lr2e-3", "max_bond_dim": 32, "epsilon_trunc": 1e-6, "lr": 2e-3},
]


def select(results):
    by_dataset = {}
    for row in results:
        by_dataset.setdefault(row["dataset"], []).append(row)

    regrets = {c["name"]: [] for c in CANDIDATES}
    runtimes = {c["name"]: [] for c in CANDIDATES}
    details = {}

    for dataset, rows in by_dataset.items():
        best = min(float(r["best_val_nll"]) for r in rows)
        details[dataset] = {}
        for row in rows:
            regret = (float(row["best_val_nll"]) - best) / max(abs(best), 1e-12)
            regrets[row["candidate"]].append(regret)
            runtimes[row["candidate"]].append(float(row["elapsed_s"]))
            details[dataset][row["candidate"]] = {
                "best_val_nll": row["best_val_nll"],
                "relative_regret": regret,
                "relative_val_improvement": row["relative_val_improvement"],
            }

    aggregate = []
    for candidate in CANDIDATES:
        name = candidate["name"]
        aggregate.append({
            **candidate,
            "mean_relative_regret": mean(regrets[name]),
            "max_relative_regret": max(regrets[name]),
            "mean_runtime_s": mean(runtimes[name]),
        })
    aggregate.sort(key=lambda r: (
        r["mean_relative_regret"],
        r["max_relative_regret"],
        r["mean_runtime_s"],
    ))
    return {
        "selection_rule": (
            "minimize mean relative normal-validation NLL regret; then "
            "worst-dataset regret; then runtime"
        ),
        "selected": aggregate[0],
        "aggregate": aggregate,
        "dataset_details": details,
    }


def main():
    p=argparse.ArgumentParser()
    p.add_argument("dataset_root",type=Path)
    p.add_argument("output_dir",type=Path)
    p.add_argument("--batch-size",type=int,default=1024)
    p.add_argument("--loops",type=int,default=8)
    p.add_argument("--batches-per-loop",type=int,default=12)
    p.add_argument("--seed",type=int,default=123)
    a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=True)
    histories=a.output_dir/"histories"; histories.mkdir(exist_ok=True)

    results=[]
    total=len(DATASETS)*len(CANDIDATES); counter=0
    for dataset in DATASETS:
        for candidate in CANDIDATES:
            counter+=1
            print(f"[{counter}/{total}] {dataset}/{candidate['name']}",flush=True)
            result=run_candidate(
                a.dataset_root,dataset,candidate,
                batch_size=a.batch_size,loops=a.loops,
                batches_per_loop=a.batches_per_loop,seed=a.seed,
            )
            history=result.pop("history")
            (histories/f"{dataset}__{candidate['name']}.json").write_text(
                json.dumps(history,indent=2),encoding="utf-8"
            )
            results.append(result)
            print(
                f"  val={result['best_val_nll']:.6f} "
                f"imp={100*result['relative_val_improvement']:.2f}% "
                f"time={result['elapsed_s']:.1f}s",
                flush=True,
            )

    selection=select(results)
    (a.output_dir/"results.json").write_text(json.dumps(results,indent=2),encoding="utf-8")
    (a.output_dir/"selection.json").write_text(json.dumps(selection,indent=2),encoding="utf-8")
    print("SELECTED",json.dumps(selection["selected"],indent=2),flush=True)
    print("AGGREGATE",json.dumps(selection["aggregate"],indent=2),flush=True)

if __name__=="__main__":
    main()
