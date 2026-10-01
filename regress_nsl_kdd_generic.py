"""Regression of the generic paper encoder on the official NSL-KDD split.

This intentionally tests the current generic TabularEncoder, not the legacy TFG
encoder. It preserves the NSL-KDD train/test split, keeps only normal KDDTrain+
rows for model fitting, reserves 15% of those normals for validation, and leaves
KDDTest+ untouched.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from dmrg_trainer import DMRGConfig, dmrg_train
from encoder import TabularEncoder
from mps import MPS


COLUMNS = [
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised",
    "root_shell", "su_attempted", "num_root", "num_file_creations",
    "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count",
    "serror_rate", "srv_serror_rate", "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty",
]

CATEGORICAL = ["protocol_type", "service", "flag"]
DROP = ["num_outbound_cmds"]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["label"] = df["label"].astype(str).str.rstrip(".")
    return df


def split_normal(df: pd.DataFrame, seed: int = 123):
    normal = df[df["label"] == "normal"].reset_index(drop=True)
    generator = torch.Generator().manual_seed(seed)
    perm = torch.randperm(len(normal), generator=generator).numpy()
    n_val = max(1, int(round(0.15 * len(normal))))
    return normal.iloc[perm[n_val:]].reset_index(drop=True), normal.iloc[perm[:n_val]].reset_index(drop=True)


def metrics(scores: np.ndarray, y: np.ndarray) -> dict:
    return {
        "auc_roc": float(roc_auc_score(y, scores)),
        "auc_pr": float(average_precision_score(y, scores)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--train-model", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_raw = load(args.data_dir / "KDDTrain+.txt")
    test_raw = load(args.data_dir / "KDDTest+.txt")
    train_df, val_df = split_normal(train_raw)

    feature_cols = [c for c in COLUMNS if c not in {"label", "difficulty"}]
    encoder = TabularEncoder(
        n_bins=8,
        categorical_columns=CATEGORICAL,
        drop_columns=DROP,
        max_categories=128,
    )
    encoder.fit(train_df[feature_cols])

    train_x = encoder.transform(train_df[feature_cols])
    val_x = encoder.transform(val_df[feature_cols])
    test_x = encoder.transform(test_raw[feature_cols])
    y = (test_raw["label"].to_numpy() != "normal").astype(np.int64)

    spec_rows = []
    for spec in encoder.specs:
        train_unique = int(train_df[spec.name].nunique(dropna=True))
        encoded_unique = int(torch.unique(train_x[:, encoder.feature_names.index(spec.name)]).numel())
        spec_rows.append({
            "name": spec.name,
            "kind": spec.kind,
            "train_unique_raw": train_unique,
            "train_unique_encoded": encoded_unique,
            "regular_states": spec.regular_states,
            "physical_dim": spec.physical_dim,
            "collapsed": train_unique > 1 and encoded_unique == 1,
        })

    collapsed = [r for r in spec_rows if r["collapsed"]]

    model = MPS.from_empirical_frequencies(
        train_x,
        physical_dims=encoder.physical_dims,
        dtype=torch.float64,
        pseudocount=1e-6,
    )

    baseline_scores = model.anomaly_score(test_x, batch_size=1024).cpu().numpy()
    result = {
        "encoder": {
            "n_bins": 8,
            "features": len(encoder.feature_names),
            "physical_dims": encoder.physical_dims,
            "collapsed_features": collapsed,
            "num_collapsed_features": len(collapsed),
            "specs": spec_rows,
        },
        "data": {
            "train_normal": len(train_x),
            "val_normal": len(val_x),
            "test_total": len(test_x),
            "test_attack": int(y.sum()),
        },
        "independent_detection": metrics(baseline_scores, y),
    }

    if args.train_model:
        cfg = DMRGConfig(
            num_descent_steps=1,
            max_bond_dim=16,
            epsilon_trunc=0.0,
            lr=2e-3,
            num_loops=30,
            batch_size=1024,
            lr_shrink=0.5,
            lr_min=1e-5,
            patience=5,
            improvement_threshold=1e-4,
            early_stopping_patience=10,
            abort_after_dead_loops=3,
            batches_per_loop=0,
            metric_for_stopping="val_nll",
            seed=123,
            log_path=str(args.output_dir / "train_log.jsonl"),
        )
        start = time.perf_counter()
        history = dmrg_train(model, train_x, val_x, config=cfg)
        elapsed = time.perf_counter() - start
        finite = [r for r in history if math.isfinite(float(r.get("val_nll", float("inf"))))]
        best = min(finite, key=lambda r: r["val_nll"])
        trained_scores = model.anomaly_score(test_x, batch_size=1024).cpu().numpy()
        result["trained"] = {
            "detection": metrics(trained_scores, y),
            "best_loop": int(best["loop"]),
            "best_val_nll": float(best["val_nll"]),
            "bond_dims": list(model.bond_dims),
            "elapsed_s": elapsed,
            "total_skipped_nan": sum(int(r.get("num_skipped_nan", 0)) for r in history),
        }
        model.save(str(args.output_dir / "model.pt"))
        (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))

    encoder.save(args.output_dir / "encoder.json")
    (args.output_dir / "summary.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
