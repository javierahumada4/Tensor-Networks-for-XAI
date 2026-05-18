"""
NSL-KDD EDA — step 1 of the anomaly detection pipeline with MPS.

Usage:
    python eda_nsl_kdd.py /path/to/nsl_kdd

Expects to find KDDTrain+.txt and KDDTest+.txt in the given directory.
If invoked without an argument, it uses './nsl_kdd' as default.

Prints a textual report and saves two artifacts in the same directory:
    - eda_summary.json   structured summary (we will use it in later steps)
    - eda_report.txt     same content in plain text (for a quick overview)
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd


# ----------------------------------------------------------------------
# NSL-KDD schema
# ----------------------------------------------------------------------

# 41 features + label + difficulty (the latter is added by NSL-KDD; KDD99 does not have it)
COLUMNS: List[str] = [
    # connection basics
    "duration", "protocol_type", "service", "flag",
    "src_bytes", "dst_bytes",
    # content / host flags
    "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in",
    "num_compromised", "root_shell", "su_attempted",
    "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login",
    # time-based traffic
    "count", "srv_count",
    "serror_rate", "srv_serror_rate",
    "rerror_rate", "srv_rerror_rate",
    "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    # host-based traffic
    "dst_host_count", "dst_host_srv_count",
    "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate",
    "dst_host_rerror_rate", "dst_host_srv_rerror_rate",
    # labels
    "label", "difficulty",
]

CATEGORICAL_COLS: List[str] = ["protocol_type", "service", "flag"]
BINARY_COLS: List[str] = [
    "land", "logged_in", "root_shell", "su_attempted",
    "is_host_login", "is_guest_login",
]

# Mapping from specific attack type -> family (5 classes). We will use it to
# evaluate the model: train only on 'normal', test against the rest.
ATTACK_FAMILY: Dict[str, str] = {
    "normal": "normal",
    # DoS
    "back": "dos", "land": "dos", "neptune": "dos", "pod": "dos",
    "smurf": "dos", "teardrop": "dos", "apache2": "dos", "udpstorm": "dos",
    "processtable": "dos", "worm": "dos", "mailbomb": "dos",
    # Probe
    "satan": "probe", "ipsweep": "probe", "nmap": "probe", "portsweep": "probe",
    "mscan": "probe", "saint": "probe",
    # R2L
    "guess_passwd": "r2l", "ftp_write": "r2l", "imap": "r2l", "phf": "r2l",
    "multihop": "r2l", "warezmaster": "r2l", "warezclient": "r2l", "spy": "r2l",
    "xlock": "r2l", "xsnoop": "r2l", "snmpguess": "r2l", "snmpgetattack": "r2l",
    "httptunnel": "r2l", "sendmail": "r2l", "named": "r2l",
    # U2R
    "buffer_overflow": "u2r", "loadmodule": "u2r", "rootkit": "u2r",
    "perl": "u2r", "sqlattack": "u2r", "xterm": "u2r", "ps": "u2r",
}


def load_split(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Cannot find {path}")
    df = pd.read_csv(path, names=COLUMNS, header=None)
    # Removes the '.' suffix that some versions of the dataset carry in the label
    df["label"] = df["label"].str.rstrip(".")
    df["family"] = df["label"].map(ATTACK_FAMILY)
    unmapped = df["family"].isna()
    if unmapped.any():
        unknown = sorted(df.loc[unmapped, "label"].unique())
        raise ValueError(
            f"Labels without a known family in {path.name}: {unknown}. "
            "Add them to the ATTACK_FAMILY dictionary."
        )
    df["is_attack"] = (df["family"] != "normal").astype(int)
    return df


def summarise_split(df: pd.DataFrame, name: str) -> Dict:
    n = len(df)
    out: Dict = {
        "name": name,
        "n_rows": int(n),
        "n_cols": int(df.shape[1]),
        # Binary balance (normal vs attack)
        "n_normal": int((df["is_attack"] == 0).sum()),
        "n_attack": int((df["is_attack"] == 1).sum()),
        "attack_pct": float(100 * df["is_attack"].mean()),
        # By family
        "family_counts": df["family"].value_counts().to_dict(),
        # By specific label (top 15)
        "top_labels": df["label"].value_counts().head(15).to_dict(),
        # Missing values
        "n_missing_total": int(df.isna().sum().sum()),
        # Cardinality of categorical columns
        "categorical_cardinality": {
            col: int(df[col].nunique()) for col in CATEGORICAL_COLS
        },
        # Features with zero variance in this split (the paper removes them)
        "zero_variance_features": [
            col for col in df.columns
            if col not in {"label", "family", "difficulty"}
            and df[col].nunique(dropna=False) <= 1
        ],
    }
    return out


def basic_numeric_stats(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include=[np.number]).drop(
        columns=["difficulty", "is_attack"], errors="ignore"
    )
    return numeric.describe().T[["mean", "std", "min", "50%", "max"]]


def fmt_dict(d: Dict, indent: int = 2) -> str:
    return "\n".join(f"{' ' * indent}{k}: {v}" for k, v in d.items())


def main(data_dir: Path) -> None:
    print(f"\n=== Loading NSL-KDD from {data_dir} ===\n")
    train = load_split(data_dir / "KDDTrain+.txt")
    test = load_split(data_dir / "KDDTest+.txt")

    train_summary = summarise_split(train, "train")
    test_summary = summarise_split(test, "test")

    # ------------------------------------------------------------------
    # Report to stdout
    # ------------------------------------------------------------------
    lines: List[str] = []
    add = lines.append

    for s in (train_summary, test_summary):
        add(f"--- Split: {s['name']} ---")
        add(f"  rows   : {s['n_rows']:,}")
        add(f"  cols   : {s['n_cols']} (41 features + label + difficulty + family + is_attack)")
        add(f"  normal : {s['n_normal']:,}")
        add(f"  attack : {s['n_attack']:,}  ({s['attack_pct']:.2f}%)")
        add(f"  total missing values: {s['n_missing_total']}")
        add("  count by family:")
        add(fmt_dict(s["family_counts"], indent=4))
        add("  top 15 specific labels:")
        add(fmt_dict(s["top_labels"], indent=4))
        add("  categorical cardinality:")
        add(fmt_dict(s["categorical_cardinality"], indent=4))
        if s["zero_variance_features"]:
            add(f"  zero-variance features: {s['zero_variance_features']}")
        else:
            add("  zero-variance features: none")
        add("")

    # Categoricals that appear in test but NOT in train -> important for encoding
    add("--- Categorical coverage (test vs train) ---")
    for col in CATEGORICAL_COLS:
        train_vals = set(train[col].unique())
        test_vals = set(test[col].unique())
        only_in_test = sorted(test_vals - train_vals)
        add(f"  {col}: {len(test_vals)} values in test, "
            f"{len(only_in_test)} NOT seen in train"
            + (f" -> {only_in_test[:8]}{'...' if len(only_in_test) > 8 else ''}"
               if only_in_test else ""))
    add("")

    # Quick numeric stats from train
    add("--- Numeric stats (train, first 10 columns) ---")
    stats = basic_numeric_stats(train).round(3)
    add(stats.head(10).to_string())
    add("")

    # Stats from train ONLY on 'normal' samples (what will train the MPS)
    add("--- Numeric stats (train ∩ normal, first 10 columns) ---")
    stats_n = basic_numeric_stats(train[train.is_attack == 0]).round(3)
    add(stats_n.head(10).to_string())

    report = "\n".join(lines)
    print(report)

    # ------------------------------------------------------------------
    # Artifacts
    # ------------------------------------------------------------------
    out_summary = {
        "train": train_summary,
        "test": test_summary,
        "schema": {
            "columns": COLUMNS,
            "categorical": CATEGORICAL_COLS,
            "binary": BINARY_COLS,
        },
    }
    (data_dir / "eda_summary.json").write_text(json.dumps(out_summary, indent=2))
    (data_dir / "eda_report.txt").write_text(report)
    print(f"\nSaved: {data_dir / 'eda_summary.json'}")
    print(f"Saved: {data_dir / 'eda_report.txt'}")


if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./nsl_kdd")
    main(data_dir)