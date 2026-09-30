"""NSL-KDD regression test for the refactored Born-MPS trainer.

This script consumes the *legacy TFG encoder artifacts* (train_X.pt,
train_meta.pt, test_X.pt, test_meta.pt, encoding_schema.json) and trains the
current paper MPS/trainer on exactly the same normal-only train/validation split
policy used in the TFG.

It is intentionally a regression harness, not part of the new generic data
pipeline.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    average_precision_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from dmrg_trainer import DMRGConfig, dmrg_train
from mps import MPS


VARIANTS = {
    "paper": {
        "max_bond_dim": 16,
        "epsilon_trunc": 0.0,
        "lr": 2e-3,
        "num_descent_steps": 1,
        "num_loops": 30,
        "lr_min": 1e-5,
        "improvement_threshold": 1e-4,
        "early_stopping_patience": 10,
    },
    "d64_tfgscale": {
        "max_bond_dim": 64,
        "epsilon_trunc": 0.0,
        "lr": 8e-4,
        "num_descent_steps": 2,
        "num_loops": 45,
        "lr_min": 5e-5,
        "improvement_threshold": 1e-3,
        "early_stopping_patience": 15,
    },
}


def split_normal_train(
    data_dir: Path,
    *,
    val_fraction: float = 0.15,
    seed: int = 123,
):
    x_all = torch.load(data_dir / "train_X.pt", map_location="cpu", weights_only=True)
    meta = torch.load(data_dir / "train_meta.pt", map_location="cpu", weights_only=True)
    normal = x_all[meta["is_attack"] == 0].long()

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(normal), generator=generator)
    n_val = max(1, int(round(val_fraction * len(normal))))
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val:]
    return normal[train_idx].contiguous(), normal[val_idx].contiguous()


def load_test(data_dir: Path):
    x = torch.load(data_dir / "test_X.pt", map_location="cpu", weights_only=True).long()
    meta = torch.load(data_dir / "test_meta.pt", map_location="cpu", weights_only=True)
    return x, meta


def physical_dims(data_dir: Path):
    schema = json.loads((data_dir / "encoding_schema.json").read_text())
    return list(schema["physical_dims"])


def evaluate(scores: np.ndarray, meta: dict, val_scores: np.ndarray) -> dict:
    y = meta["is_attack"].numpy().astype(np.int64)
    auc_roc = float(roc_auc_score(y, scores))
    auc_pr = float(average_precision_score(y, scores))

    percentiles = np.arange(90.0, 99.5001, 0.5)
    sweep = []
    for pct in percentiles:
        threshold = float(np.percentile(val_scores, pct))
        pred = (scores >= threshold).astype(np.int64)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y, pred, average="binary", zero_division=0
        )
        sweep.append({
            "percentile": float(pct),
            "threshold": threshold,
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
        })
    best_f1 = max(sweep, key=lambda row: row["f1"])

    families = {}
    family_names = list(meta["family_names"])
    family_code = meta["family_code"].numpy().astype(np.int64)
    normal_code = family_names.index("normal")
    normal_mask = family_code == normal_code

    for family in ("dos", "probe", "r2l", "u2r"):
        if family not in family_names:
            continue
        code = family_names.index(family)
        mask = normal_mask | (family_code == code)
        y_family = (family_code[mask] == code).astype(np.int64)
        family_scores = scores[mask]
        families[family] = {
            "n_attack": int(y_family.sum()),
            "auc_roc": float(roc_auc_score(y_family, family_scores)),
            "auc_pr": float(average_precision_score(y_family, family_scores)),
        }

    return {
        "auc_roc": auc_roc,
        "auc_pr": auc_pr,
        "best_f1": best_f1,
        "families": families,
    }


def independent_scores(train_x: torch.Tensor, x: torch.Tensor, dims: list[int]):
    score = torch.zeros(len(x), dtype=torch.float64)
    for site, dim in enumerate(dims):
        counts = torch.bincount(train_x[:, site], minlength=dim).double()
        probabilities = counts + 1e-6
        probabilities /= probabilities.sum()
        score -= torch.log(probabilities[x[:, site]])
    return score.numpy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    args = parser.parse_args()

    cfg = VARIANTS[args.variant]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_x, val_x = split_normal_train(args.data_dir)
    test_x, test_meta = load_test(args.data_dir)
    dims = physical_dims(args.data_dir)

    if train_x.shape[1] != 40:
        raise ValueError(f"expected 40 TFG features, got {train_x.shape[1]}")
    if len(test_x) != 22544:
        raise ValueError(f"expected 22544 KDDTest+ rows, got {len(test_x)}")

    torch.manual_seed(123)
    model = MPS.from_empirical_frequencies(
        train_x,
        physical_dims=dims,
        dtype=torch.float64,
        pseudocount=1e-6,
    )

    initial_train_nll = float(model.nll(train_x, batch_size=1024))
    initial_val_nll = float(model.nll(val_x, batch_size=1024))
    baseline_test = independent_scores(train_x, test_x, dims)
    baseline_val = independent_scores(train_x, val_x, dims)
    baseline_metrics = evaluate(baseline_test, test_meta, baseline_val)

    trainer_cfg = DMRGConfig(
        num_descent_steps=cfg["num_descent_steps"],
        max_bond_dim=cfg["max_bond_dim"],
        epsilon_trunc=cfg["epsilon_trunc"],
        lr=cfg["lr"],
        num_loops=cfg["num_loops"],
        batch_size=1024,
        lr_shrink=0.5,
        lr_min=cfg["lr_min"],
        patience=5,
        improvement_threshold=cfg["improvement_threshold"],
        early_stopping_patience=cfg["early_stopping_patience"],
        abort_after_dead_loops=3,
        batches_per_loop=0,
        metric_for_stopping="val_nll",
        seed=123,
        log_path=str(args.output_dir / "train_log.jsonl"),
    )

    start = time.perf_counter()
    history = dmrg_train(model, train_x, val_x, config=trainer_cfg)
    elapsed = time.perf_counter() - start

    test_scores = model.anomaly_score(test_x, batch_size=1024).cpu().numpy()
    val_scores = model.anomaly_score(val_x, batch_size=1024).cpu().numpy()
    trained_metrics = evaluate(test_scores, test_meta, val_scores)

    finite = [
        row for row in history
        if math.isfinite(float(row.get("val_nll", float("inf"))))
    ]
    best = min(finite, key=lambda row: row["val_nll"])

    summary = {
        "variant": args.variant,
        "config": cfg,
        "data": {
            "train_normal": len(train_x),
            "val_normal": len(val_x),
            "test_total": len(test_x),
            "test_attack": int(test_meta["is_attack"].sum().item()),
            "features": train_x.shape[1],
            "physical_dims": dims,
        },
        "initial": {
            "train_nll": initial_train_nll,
            "val_nll": initial_val_nll,
            "independent_detection": baseline_metrics,
        },
        "trained": {
            "best_loop": int(best["loop"]),
            "best_val_nll": float(best["val_nll"]),
            "restored_train_nll": float(model.nll(train_x, batch_size=1024)),
            "restored_val_nll": float(model.nll(val_x, batch_size=1024)),
            "bond_dims": list(model.bond_dims),
            "detection": trained_metrics,
        },
        "elapsed_s": elapsed,
        "history_records": len(history),
        "total_skipped_nan": sum(int(row.get("num_skipped_nan", 0)) for row in history),
    }

    model.save(str(args.output_dir / "model.pt"))
    torch.save(
        {
            "nll": torch.from_numpy(test_scores),
            "is_attack": test_meta["is_attack"],
            "family_code": test_meta["family_code"],
            "family_names": test_meta["family_names"],
        },
        args.output_dir / "test_scores.pt",
    )
    (args.output_dir / "history.json").write_text(
        json.dumps(history, indent=2), encoding="utf-8"
    )
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
