"""
Step 3 of the pipeline: training the Born Machine (MPS) on NSL-KDD.

Trains ONLY on normal events: the MPS learns the distribution of
benign activity and, during evaluation, a high NLL flags an anomaly.

Usage:
    python train_mps_nsl_kdd.py /path/to/nsl_kdd

Expects to find the encoder artifacts in that directory:
    train_X.pt, train_meta.pt, encoding_schema.json
and optionally test_X.pt / test_meta.pt (only for an informational check).

Produces in the same directory:
    mps_trained.pt        the trained MPS (MPS.save format)
    train_history.json    history per loop (NLL, lr, bond_dims, ...)
    train_log.jsonl       JSONL log written by the trainer during the run
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

from mps import MPS
from dmrg_trainer import DMRGConfig, dmrg_train

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("train_mps")


# ----------------------------------------------------------------------
# Hyperparameters (see justification in the conversation)
# ----------------------------------------------------------------------

DTYPE = torch.float64
INIT_BOND_DIM = 2
VAL_FRACTION = 0.10
SEED = 0

def bond_schedule(loop: int) -> int:
    if loop < 5:
        return 16
    if loop < 15:
        return 32
    if loop < 30:
        return 64
    if loop < 50:
        return 96
    return 128

CONFIG = DMRGConfig(
    num_loops=80,
    num_descent_steps=6,

    max_bond_dim=128,
    max_bond_dim_schedule=bond_schedule,
    svd_cutoff=1e-10,

    lr=2e-2,
    lr_shrink=0.7,
    lr_min=1e-6,
    patience=8,
    improvement_threshold=1e-6,

    batch_size=1024,
    batches_per_loop=0,
    stochastic=False,

    adaptive_lr=False,

    metric_for_stopping="val_nll",
    eval_max_samples=0,

    seed=123,
    checkpoint_every=5,
    checkpoint_dir="./checkpoints",
    log_path="./logs/mps.jsonl",
)


# ----------------------------------------------------------------------
def load_normal_train(data_dir: Path) -> torch.Tensor:
    """Loads train_X.pt and keeps only the normal rows."""
    x_path = data_dir / "train_X.pt"
    meta_path = data_dir / "train_meta.pt"
    if not x_path.exists() or not meta_path.exists():
        raise FileNotFoundError(
            f"Missing {x_path.name} or {meta_path.name}. "
            "Run encoder_nsl_kdd.py first."
        )
    x_all = torch.load(x_path)
    meta = torch.load(meta_path)
    is_attack = meta["is_attack"]

    normal_mask = is_attack == 0
    x_normal = x_all[normal_mask].long()
    logger.info(
        "train_X: %d total rows, %d normal (%.1f%%) -> training with those",
        len(x_all), len(x_normal), 100.0 * len(x_normal) / len(x_all),
    )
    return x_normal


def split_train_val(
    x: torch.Tensor, val_fraction: float, seed: int
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Reproducible random normal partition -> (train, val)."""
    if not (0.0 <= val_fraction < 1.0):
        raise ValueError("val_fraction must be in [0, 1)")
    if val_fraction == 0.0:
        return x, None

    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(x), generator=generator)
    n_val = max(1, int(round(val_fraction * len(x))))
    val_idx = permutation[:n_val]
    train_idx = permutation[n_val:]
    return x[train_idx].contiguous(), x[val_idx].contiguous()


def load_physical_dims(data_dir: Path) -> list[int]:
    """Reads the physical dimensions per site from the encoder schema."""
    schema_path = data_dir / "encoding_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(
            f"Missing {schema_path.name}. Run encoder_nsl_kdd.py first."
        )
    schema = json.loads(schema_path.read_text())
    return list(schema["physical_dims"])


def check_columns_within_dims(x: torch.Tensor, physical_dims: list[int]) -> None:
    """Checks that each column falls in [0, d_k); fails early if not."""
    if x.dim() != 2:
            raise ValueError(f"X must be 2D, got shape {tuple(x.shape)}")
    if x.shape[1] != len(physical_dims):
        raise ValueError(
            f"X has {x.shape[1]} sites but the schema declares "
            f"{len(physical_dims)}."
        )
    if x.dtype != torch.long:
            raise ValueError(f"X dtype must be long, got {x.dtype}")
    col_min = x.min(dim=0).values
    col_max = x.max(dim=0).values
    for site, (lo, hi, d) in enumerate(
        zip(col_min.tolist(), col_max.tolist(), physical_dims)
    ):
        if lo < 0 or hi >= d:
            raise ValueError(
                f"site {site}: range [{lo}, {hi}] outside [0, {d})"
            )


# ----------------------------------------------------------------------
def main(data_dir: Path) -> None:
    torch.manual_seed(SEED)

    # --- data --------------------------------------------------------
    physical_dims = load_physical_dims(data_dir)
    x_normal = load_normal_train(data_dir)
    check_columns_within_dims(x_normal, physical_dims)

    train_data, val_data = split_train_val(x_normal, VAL_FRACTION, SEED)
    n_val = 0 if val_data is None else len(val_data)
    logger.info("partition: %d training, %d validation (both normal-only)",
                len(train_data), n_val)

    # --- model -------------------------------------------------------
    num_sites = len(physical_dims)
    mps = MPS(
        num_sites=num_sites,
        bond_dim=INIT_BOND_DIM,
        physical_dims=physical_dims,
        dtype=DTYPE,
    )
    logger.info(
        "MPS: %d sites, initial bond %d, max(d)=%d, initial parameters %d",
        num_sites, INIT_BOND_DIM, max(physical_dims), mps.num_parameters,
    )

    # --- training ----------------------------------------------------
    config = CONFIG
    config.checkpoint_dir = str(data_dir / "checkpoints")
    config.log_path = str(data_dir / "train_log.jsonl")

    logger.info("starting DMRG: %d loops, max_bond_dim=%d, lr=%.2e",
                config.num_loops, config.max_bond_dim, config.lr)
    history = dmrg_train(mps, train_data, val_data, config=config)

    # --- saving ------------------------------------------------------
    mps_path = data_dir / "mps_trained.pt"
    history_path = data_dir / "train_history.json"
    mps.save(str(mps_path))
    history_path.write_text(json.dumps(history, indent=2))

    if history:
        last = history[-1]
        logger.info(
            "done: loop %d, train_nll=%.4f%s, bond_dims=%s",
            last["loop"], last["train_nll"],
            f", val_nll={last['val_nll']:.4f}" if "val_nll" in last else "",
            last["bond_dims"],
        )
    logger.info("saved: %s", mps_path)
    logger.info("saved: %s", history_path)

if __name__ == "__main__":
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./nsl_kdd")
    main(data_dir)

