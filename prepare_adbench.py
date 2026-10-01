"""Download and prepare the selected ADBench datasets for the paper.

The source revision is pinned so the exact raw datasets are reproducible.
Generated encoded bundles are intentionally not committed to git; CI publishes
them as a workflow artifact and local users can regenerate the same directory.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

from data_artifacts import OneClassSplitConfig, prepare_one_class_artifacts


ADBENCH_REPOSITORY = "Minqi824/ADBench"
ADBENCH_COMMIT = "3dac8221081e190f157d78e93bfa8867f90d0965"
ADBENCH_RAW_ROOT = (
    "https://raw.githubusercontent.com/"
    f"{ADBENCH_REPOSITORY}/{ADBENCH_COMMIT}/adbench/datasets/Classical"
)

DATASETS = {
    "annthyroid": "2_annthyroid.npz",
    "cardio": "6_cardio.npz",
    "cover": "10_cover.npz",
    "mammography": "23_mammography.npz",
    "shuttle": "32_shuttle.npz",
    "vowels": "40_vowels.npz",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return
    with urllib.request.urlopen(url) as response, destination.open("wb") as out:
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)


def _load_npz(path: Path):
    payload = np.load(path, allow_pickle=False)
    if "X" not in payload or "y" not in payload:
        raise ValueError(f"{path} does not contain X and y arrays")
    X = payload["X"]
    y = payload["y"]
    if X.ndim != 2:
        raise ValueError(f"{path}: X must be 2D, got {X.shape}")
    if y.ndim != 1:
        y = np.asarray(y).reshape(-1)
    if len(X) != len(y):
        raise ValueError(f"{path}: len(X) != len(y)")
    frame = pd.DataFrame(
        X,
        columns=[f"x{i}" for i in range(X.shape[1])],
    )
    return frame, pd.Series(y)


def prepare_suite(
    output_root: Path,
    *,
    cache_dir: Path,
    seed: int = 123,
    n_bins: int = 8,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
) -> dict:
    output_root.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    suite = {
        "format_version": 1,
        "source": {
            "repository": ADBENCH_REPOSITORY,
            "commit": ADBENCH_COMMIT,
        },
        "encoding": {
            "seed": seed,
            "n_bins": n_bins,
            "train_fraction": train_fraction,
            "val_fraction": val_fraction,
            "normal_label": 0,
        },
        "datasets": {},
    }

    split_config = OneClassSplitConfig(
        train_fraction=train_fraction,
        val_fraction=val_fraction,
        seed=seed,
    )

    for name, filename in DATASETS.items():
        url = f"{ADBENCH_RAW_ROOT}/{filename}"
        source_path = cache_dir / filename
        print(f"[{name}] downloading {url}", flush=True)
        _download(url, source_path)

        source_sha256 = _sha256(source_path)
        source_size = source_path.stat().st_size

        features, labels = _load_npz(source_path)
        unique_labels = sorted(np.unique(labels.to_numpy()).tolist())
        if not set(unique_labels).issubset({0, 1}):
            raise ValueError(
                f"{name}: expected binary ADBench labels 0/1, got "
                f"{unique_labels}"
            )

        print(
            f"[{name}] rows={len(features)} features={features.shape[1]} "
            f"anomalies={int((labels != 0).sum())}",
            flush=True,
        )

        bundle = prepare_one_class_artifacts(
            features,
            labels,
            output_root / name,
            normal_label=0,
            dataset_name=name,
            split_config=split_config,
            n_bins=n_bins,
        )

        manifest_path = bundle.root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["source"] = {
            "benchmark": "ADBench",
            "repository": ADBENCH_REPOSITORY,
            "commit": ADBENCH_COMMIT,
            "path": f"adbench/datasets/Classical/{filename}",
            "url": url,
            "sha256": source_sha256,
            "size_bytes": source_size,
        }
        manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

        suite["datasets"][name] = {
            "source_file": filename,
            "source_sha256": source_sha256,
            "source_size_bytes": source_size,
            "rows": len(features),
            "features": features.shape[1],
            "physical_dims": bundle.physical_dims,
            "counts": manifest["counts"],
        }

    suite_path = output_root / "adbench_suite_manifest.json"
    suite_path.write_text(
        json.dumps(suite, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare the six ADBench datasets used in the paper."
    )
    parser.add_argument(
        "output_root",
        type=Path,
        nargs="?",
        default=Path("artifacts/adbench"),
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(".cache/adbench"),
    )
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--bins", type=int, default=8)
    parser.add_argument("--train-fraction", type=float, default=0.70)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    suite = prepare_suite(
        args.output_root,
        cache_dir=args.cache_dir,
        seed=args.seed,
        n_bins=args.bins,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
    )
    print(json.dumps(suite["datasets"], indent=2), flush=True)


if __name__ == "__main__":
    main()
