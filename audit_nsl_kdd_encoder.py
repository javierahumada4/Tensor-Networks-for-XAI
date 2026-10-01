"""Fast NSL-KDD encoder-only audit.

Compares two generic encoding policies on the exact official NSL-KDD split:
1) old generic behavior: every numeric feature forced through quantile binning;
2) corrected behavior: low-cardinality integer-like numerics preserved exactly.

No MPS training is performed. Detection uses only the product of empirical
training marginals, isolating the encoder from the tensor-network optimizer.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from encoder import TabularEncoder


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
NUMERIC = [
    c for c in COLUMNS
    if c not in set(CATEGORICAL) | set(DROP) | {"label", "difficulty"}
]


def load(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, names=COLUMNS)
    df["label"] = df["label"].astype(str).str.rstrip(".")
    return df


def split_normal(df: pd.DataFrame):
    normal = df[df["label"] == "normal"].reset_index(drop=True)
    g = torch.Generator().manual_seed(123)
    perm = torch.randperm(len(normal), generator=g).numpy()
    n_val = int(round(0.15 * len(normal)))
    return normal.iloc[perm[n_val:]].reset_index(drop=True)


def independent_scores(train_x, test_x, dims):
    score = torch.zeros(len(test_x), dtype=torch.float64)
    for site, dim in enumerate(dims):
        counts = torch.bincount(train_x[:, site], minlength=dim).double()
        probs = (counts + 1e-6) / (counts.sum() + 1e-6 * dim)
        score -= torch.log(probs[test_x[:, site]])
    return score.numpy()


def evaluate(policy, encoder, train_df, test_df, features):
    encoder.fit(train_df[features])
    train_x = encoder.transform(train_df[features])
    test_x = encoder.transform(test_df[features])
    y = (test_df["label"].to_numpy() != "normal").astype(np.int64)
    scores = independent_scores(train_x, test_x, encoder.physical_dims)

    specs = []
    for i, spec in enumerate(encoder.specs):
        raw_unique = int(train_df[spec.name].nunique(dropna=True))
        encoded_unique = int(torch.unique(train_x[:, i]).numel())
        specs.append({
            "name": spec.name,
            "kind": spec.kind,
            "raw_unique": raw_unique,
            "encoded_unique": encoded_unique,
            "regular_states": spec.regular_states,
            "physical_dim": spec.physical_dim,
            "collapsed": raw_unique > 1 and encoded_unique == 1,
        })

    return {
        "policy": policy,
        "auc_roc": float(roc_auc_score(y, scores)),
        "auc_pr": float(average_precision_score(y, scores)),
        "num_collapsed_nonconstant": sum(row["collapsed"] for row in specs),
        "collapsed_features": [row for row in specs if row["collapsed"]],
        "feature_kinds": {
            kind: sum(spec.kind == kind for spec in encoder.specs)
            for kind in sorted(set(spec.kind for spec in encoder.specs))
        },
        "physical_dims": encoder.physical_dims,
        "specs": specs,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    train_raw = load(args.data_dir / "KDDTrain+.txt")
    test_raw = load(args.data_dir / "KDDTest+.txt")
    train_df = split_normal(train_raw)
    features = [c for c in COLUMNS if c not in {"label", "difficulty"}]

    old = TabularEncoder(
        n_bins=8,
        categorical_columns=CATEGORICAL,
        continuous_columns=NUMERIC,
        drop_columns=DROP,
        max_categories=128,
    )
    fixed = TabularEncoder(
        n_bins=8,
        categorical_columns=CATEGORICAL,
        drop_columns=DROP,
        max_categories=128,
        max_discrete_numeric_states=8,
    )

    report = {
        "train_normal": len(train_df),
        "test_total": len(test_raw),
        "test_attack": int((test_raw["label"] != "normal").sum()),
        "old_generic": evaluate("all_numeric_quantiles", old, train_df, test_raw, features),
        "fixed_generic": evaluate("infer_low_card_numeric", fixed, train_df, test_raw, features),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
