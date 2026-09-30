"""CLI for creating persistent encoded one-class dataset artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from data_artifacts import OneClassSplitConfig, prepare_one_class_artifacts


def _parse_list(value: Optional[str]) -> List[str]:
    if value is None or value.strip() == "":
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def _parse_scalar(value: str):
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def _load_input(path: Path, label_column: Optional[str]):
    suffix = path.suffix.lower()
    if suffix == ".csv":
        frame = pd.read_csv(path)
        if label_column is None:
            raise ValueError("--label-column is required for CSV input")
        labels = frame[label_column].copy()
        features = frame.drop(columns=[label_column])
        return features, labels

    if suffix == ".npz":
        payload = np.load(path, allow_pickle=False)
        if "X" not in payload or "y" not in payload:
            raise ValueError("NPZ input must contain arrays named X and y")
        X = payload["X"]
        y = payload["y"]
        if X.ndim != 2:
            raise ValueError(f"X must be 2D, got shape {X.shape}")
        features = pd.DataFrame(
            X,
            columns=[f"x{i}" for i in range(X.shape[1])],
        )
        labels = pd.Series(y)
        return features, labels

    raise ValueError(
        f"unsupported input format {suffix!r}; use CSV or NPZ"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Create a persistent one-class train/val/test split and encode it "
            "for the Born-MPS."
        )
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--label-column", default=None)
    parser.add_argument("--normal-label", default="0")
    parser.add_argument("--categorical", default=None)
    parser.add_argument("--continuous", default=None)
    parser.add_argument("--drop", default=None)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--max-categories", type=int, default=64)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=123)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    features, labels = _load_input(args.input, args.label_column)

    bundle = prepare_one_class_artifacts(
        features,
        labels,
        args.output_dir,
        normal_label=_parse_scalar(args.normal_label),
        dataset_name=args.dataset_name or args.input.stem,
        split_config=OneClassSplitConfig(
            train_fraction=args.train_fraction,
            val_fraction=args.val_fraction,
            seed=args.seed,
        ),
        n_bins=args.bins,
        categorical_columns=_parse_list(args.categorical),
        continuous_columns=_parse_list(args.continuous),
        drop_columns=_parse_list(args.drop),
        max_categories=args.max_categories,
    )

    counts = bundle.manifest["counts"]
    print(
        f"saved {bundle.manifest['dataset_name']!r} -> {bundle.root} | "
        f"train={counts['train']} val={counts['val']} "
        f"test={counts['test']} (anomalies={counts['test_anomaly']})"
    )


if __name__ == "__main__":
    main()
