"""Persistent one-class dataset splits and encoded artifact bundles."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Sequence

import numpy as np
import pandas as pd
import torch

from encoder import TabularEncoder


ARTIFACT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class OneClassSplitConfig:
    train_fraction: float = 0.70
    val_fraction: float = 0.15
    seed: int = 123

    @property
    def test_normal_fraction(self) -> float:
        return 1.0 - self.train_fraction - self.val_fraction

    def validate(self) -> None:
        if not (0.0 < self.train_fraction < 1.0):
            raise ValueError("train_fraction must lie strictly between 0 and 1")
        if not (0.0 < self.val_fraction < 1.0):
            raise ValueError("val_fraction must lie strictly between 0 and 1")
        if self.train_fraction + self.val_fraction >= 1.0:
            raise ValueError(
                "train_fraction + val_fraction must be < 1 so normal test "
                "rows remain available"
            )


@dataclass
class EncodedSplit:
    x: torch.Tensor
    y: torch.Tensor
    row_indices: torch.Tensor


@dataclass
class EncodedDatasetBundle:
    root: Path
    manifest: Dict
    encoder: TabularEncoder
    train: EncodedSplit
    val: EncodedSplit
    test: EncodedSplit

    @property
    def physical_dims(self):
        return list(self.manifest["physical_dims"])

    @property
    def feature_names(self):
        return list(self.manifest["feature_names"])


def _normal_mask(labels: pd.Series, normal_label) -> np.ndarray:
    return labels.to_numpy() == normal_label


def _split_normal_indices(
    normal_indices: np.ndarray,
    config: OneClassSplitConfig,
    rng: np.random.Generator,
):
    shuffled = rng.permutation(normal_indices)
    n = len(shuffled)
    if n < 3:
        raise ValueError(
            "at least three normal rows are required to create train/val/test"
        )

    n_train = int(np.floor(config.train_fraction * n))
    n_val = int(np.floor(config.val_fraction * n))

    n_train = max(n_train, 1)
    n_val = max(n_val, 1)
    if n_train + n_val >= n:
        n_val = max(1, n - n_train - 1)
    if n_train + n_val >= n:
        n_train = n - n_val - 1

    train_idx = shuffled[:n_train]
    val_idx = shuffled[n_train:n_train + n_val]
    test_normal_idx = shuffled[n_train + n_val:]
    return train_idx, val_idx, test_normal_idx


def _save_split(path: Path, split: EncodedSplit) -> None:
    torch.save(
        {
            "x": split.x.cpu().long(),
            "y": split.y.cpu().long(),
            "row_indices": split.row_indices.cpu().long(),
        },
        path,
    )


def _load_split(path: Path) -> EncodedSplit:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    return EncodedSplit(
        x=payload["x"].long(),
        y=payload["y"].long(),
        row_indices=payload["row_indices"].long(),
    )


def prepare_one_class_artifacts(
    features: pd.DataFrame,
    labels: pd.Series,
    output_dir: str | Path,
    *,
    normal_label=0,
    dataset_name: str = "dataset",
    split_config: Optional[OneClassSplitConfig] = None,
    n_bins: int = 8,
    categorical_columns: Optional[Sequence[str]] = None,
    continuous_columns: Optional[Sequence[str]] = None,
    drop_columns: Optional[Sequence[str]] = None,
    max_categories: int = 64,
    max_discrete_numeric_states: int = 8,
) -> EncodedDatasetBundle:
    """Create one reproducible one-class split and persist all encoded artifacts.

    The encoder is fitted exclusively on normal training rows. Validation also
    contains only normal rows. Test contains the held-out normal rows plus every
    anomaly row. Binary targets are redefined as 0=normal and 1=anomaly.

    Original row positions are persisted in every split and in split_indices.pt,
    so downstream scripts never need to recreate the partition.
    """
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame")
    if not isinstance(labels, pd.Series):
        labels = pd.Series(labels)
    if len(features) != len(labels):
        raise ValueError(
            f"features and labels have different lengths: "
            f"{len(features)} != {len(labels)}"
        )
    if len(features) == 0:
        raise ValueError("cannot prepare an empty dataset")

    config = split_config or OneClassSplitConfig()
    config.validate()

    labels = labels.reset_index(drop=True)
    features = features.reset_index(drop=True)

    normal_mask = _normal_mask(labels, normal_label)
    normal_indices = np.flatnonzero(normal_mask)
    anomaly_indices = np.flatnonzero(~normal_mask)

    if len(normal_indices) == 0:
        raise ValueError(f"no rows match normal_label={normal_label!r}")
    if len(anomaly_indices) == 0:
        raise ValueError("dataset contains no anomaly rows")

    rng = np.random.default_rng(config.seed)
    train_idx, val_idx, test_normal_idx = _split_normal_indices(
        normal_indices,
        config,
        rng,
    )

    test_idx = np.concatenate([test_normal_idx, anomaly_indices])
    test_idx = rng.permutation(test_idx)

    encoder = TabularEncoder(
        n_bins=n_bins,
        categorical_columns=categorical_columns,
        continuous_columns=continuous_columns,
        drop_columns=drop_columns,
        max_categories=max_categories,
        max_discrete_numeric_states=max_discrete_numeric_states,
    )
    encoder.fit(features.iloc[train_idx])

    train_x = encoder.transform(features.iloc[train_idx])
    val_x = encoder.transform(features.iloc[val_idx])
    test_x = encoder.transform(features.iloc[test_idx])

    train_y = torch.zeros(len(train_idx), dtype=torch.long)
    val_y = torch.zeros(len(val_idx), dtype=torch.long)
    test_y = torch.from_numpy((~normal_mask[test_idx]).astype(np.int64))

    train = EncodedSplit(
        x=train_x,
        y=train_y,
        row_indices=torch.from_numpy(train_idx.astype(np.int64)),
    )
    val = EncodedSplit(
        x=val_x,
        y=val_y,
        row_indices=torch.from_numpy(val_idx.astype(np.int64)),
    )
    test = EncodedSplit(
        x=test_x,
        y=test_y,
        row_indices=torch.from_numpy(test_idx.astype(np.int64)),
    )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    encoder.save(root / "encoder.json")
    _save_split(root / "train.pt", train)
    _save_split(root / "val.pt", val)
    _save_split(root / "test.pt", test)
    torch.save(
        {
            "train": train.row_indices,
            "val": val.row_indices,
            "test": test.row_indices,
        },
        root / "split_indices.pt",
    )

    manifest = {
        "format_version": ARTIFACT_FORMAT_VERSION,
        "dataset_name": dataset_name,
        "normal_label": str(normal_label),
        "split": {
            **asdict(config),
            "test_normal_fraction": config.test_normal_fraction,
        },
        "encoder": {
            "n_bins": encoder.n_bins,
            "max_categories": encoder.max_categories,
            "max_discrete_numeric_states": encoder.max_discrete_numeric_states,
        },
        "feature_names": encoder.feature_names,
        "feature_types": encoder.feature_types,
        "physical_dims": encoder.physical_dims,
        "counts": {
            "total": len(features),
            "normal_total": int(normal_mask.sum()),
            "anomaly_total": int((~normal_mask).sum()),
            "train": len(train_idx),
            "val": len(val_idx),
            "test": len(test_idx),
            "test_normal": len(test_normal_idx),
            "test_anomaly": len(anomaly_indices),
        },
        "files": {
            "encoder": "encoder.json",
            "split_indices": "split_indices.pt",
            "train": "train.pt",
            "val": "val.pt",
            "test": "test.pt",
        },
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return EncodedDatasetBundle(
        root=root,
        manifest=manifest,
        encoder=encoder,
        train=train,
        val=val,
        test=test,
    )


def load_encoded_bundle(path: str | Path) -> EncodedDatasetBundle:
    """Load an already prepared dataset without refitting or re-splitting."""
    root = Path(path)
    manifest = json.loads(
        (root / "manifest.json").read_text(encoding="utf-8")
    )
    if manifest.get("format_version") != ARTIFACT_FORMAT_VERSION:
        raise ValueError(
            f"unsupported artifact format version "
            f"{manifest.get('format_version')!r}"
        )

    files = manifest["files"]
    encoder = TabularEncoder.load(root / files["encoder"])
    train = _load_split(root / files["train"])
    val = _load_split(root / files["val"])
    test = _load_split(root / files["test"])

    if encoder.feature_names != manifest["feature_names"]:
        raise ValueError("encoder and manifest feature_names disagree")
    if encoder.physical_dims != manifest["physical_dims"]:
        raise ValueError("encoder and manifest physical_dims disagree")

    return EncodedDatasetBundle(
        root=root,
        manifest=manifest,
        encoder=encoder,
        train=train,
        val=val,
        test=test,
    )
