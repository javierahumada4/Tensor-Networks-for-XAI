"""Train one final Born-MPS model on a prepared ADBench dataset.

Hyperparameters are supplied explicitly so the exact common configuration
selected by the screening can be frozen in CI. Training and checkpoint
selection use normal train/validation data only. Test labels are never consulted
during optimization; test NLL scores are saved only for downstream evaluation.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch

from data_artifacts import load_encoded_bundle
from dmrg_trainer import DMRGConfig, dmrg_train
from mps import MPS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset_root", type=Path)
    parser.add_argument("dataset")
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-bond-dim", type=int, required=True)
    parser.add_argument("--epsilon-trunc", type=float, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--batch-size", type=int, default=1024)
    parser.add_argument("--max-loops", type=int, default=30)
    parser.add_argument("--lr-patience", type=int, default=5)
    parser.add_argument("--lr-shrink", type=float, default=0.5)
    parser.add_argument("--lr-min", type=float, default=1e-5)
    parser.add_argument("--early-stopping-patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=123)
    args = parser.parse_args()

    data = load_encoded_bundle(args.dataset_root / args.dataset)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    model = MPS.from_empirical_frequencies(
        data.train.x,
        physical_dims=data.physical_dims,
        dtype=torch.float64,
        pseudocount=1e-6,
    )

    initial = {
        "train_nll": float(model.nll(data.train.x, batch_size=args.batch_size)),
        "val_nll": float(model.nll(data.val.x, batch_size=args.batch_size)),
    }

    config = DMRGConfig(
        num_descent_steps=1,
        max_bond_dim=args.max_bond_dim,
        epsilon_trunc=args.epsilon_trunc,
        lr=args.lr,
        num_loops=args.max_loops,
        batch_size=args.batch_size,
        lr_shrink=args.lr_shrink,
        lr_min=args.lr_min,
        patience=args.lr_patience,
        improvement_threshold=1e-4,
        early_stopping_patience=args.early_stopping_patience,
        abort_after_dead_loops=3,
        batches_per_loop=0,
        metric_for_stopping="val_nll",
        seed=args.seed,
        log_path=str(args.output_dir / "train_log.jsonl"),
    )

    start = time.perf_counter()
    history = dmrg_train(model, data.train.x, data.val.x, config=config)
    elapsed = time.perf_counter() - start

    if not history:
        raise RuntimeError("training produced an empty history")

    valid_records = [
        record
        for record in history
        if math.isfinite(float(record.get("val_nll", float("inf"))))
    ]
    if not valid_records:
        raise RuntimeError("training produced no finite validation NLL")

    best_record = min(valid_records, key=lambda record: record["val_nll"])

    final_train_nll = float(model.nll(data.train.x, batch_size=args.batch_size))
    final_val_nll = float(model.nll(data.val.x, batch_size=args.batch_size))
    test_scores = model.anomaly_score(
        data.test.x,
        batch_size=args.batch_size,
    ).detach().cpu()

    model.save(str(args.output_dir / "model.pt"))
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2),
        encoding="utf-8",
    )

    torch.save(
        {
            "nll": test_scores,
            "y": data.test.y.cpu(),
            "row_indices": data.test.row_indices.cpu(),
        },
        args.output_dir / "test_scores.pt",
    )

    summary = {
        "dataset": args.dataset,
        "feature_names": data.feature_names,
        "physical_dims": data.physical_dims,
        "counts": data.manifest["counts"],
        "config": {
            "max_bond_dim": args.max_bond_dim,
            "epsilon_trunc": args.epsilon_trunc,
            "initial_lr": args.lr,
            "batch_size": args.batch_size,
            "max_loops": args.max_loops,
            "lr_patience": args.lr_patience,
            "lr_shrink": args.lr_shrink,
            "lr_min": args.lr_min,
            "early_stopping_patience": args.early_stopping_patience,
            "seed": args.seed,
        },
        "initial": initial,
        "best_record": best_record,
        "restored_model": {
            "train_nll": final_train_nll,
            "val_nll": final_val_nll,
            "bond_dims": list(model.bond_dims),
        },
        "elapsed_s": elapsed,
        "num_history_records": len(history),
        "total_skipped_nan": sum(
            int(record.get("num_skipped_nan", 0)) for record in history
        ),
        "max_observed_discarded_weight": max(
            float(record.get("max_discarded_weight", 0.0))
            for record in history
        ),
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
