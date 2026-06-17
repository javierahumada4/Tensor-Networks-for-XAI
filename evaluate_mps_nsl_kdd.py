"""
Evaluating the trained Born Machine (MPS).

The MPS was trained ONLY on normal traffic, so its negative log-likelihood
(NLL) doubles as an anomaly score: a high NLL means "unlikely under the
learned model of normal behaviour" -> probable attack.

This evaluator answers the only question that matters for a detector:
does that score actually separate attacks from normal traffic?

Threshold policy
----------------
A detector needs a cut-off, and the cut-off must NEVER be chosen using
attack data (there are no attacks at training time). It is fixed as a
percentile of the NLL over held-out NORMAL traffic. We reuse the *exact*
val-normal split the trainer held out, by importing the trainer's own
split helpers -- so the threshold is calibrated on data the model never
saw, and the split cannot silently drift from the trainer's.

Usage:
    python evaluate_mps_nsl_kdd.py /path/to/nsl_kdd
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Figure style: readable typography after LaTeX scales figures to ~\textwidth
# ----------------------------------------------------------------------
# Rule of thumb: on-page font size = (font size set here) x (display width / figure width).
# These figures are included at ~\textwidth (~6.3 in), so we keep figure widths
# close to that and set generous base sizes. Tune SAVE_DPI / the sizes to taste.
SAVE_DPI = 200
plt.rcParams.update({
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11,
    "figure.titlesize": 14,
    "savefig.dpi":      SAVE_DPI,
})

from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from mps import MPS

from train_mps_nsl_kdd import (
    CONFIG,
    VAL_FRACTION,
    load_normal_train,
    split_train_val,
)


logger = logging.getLogger("evaluate_mps")

# ----------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------

THRESHOLD_PERCENTILES: Tuple[float, ...] = (94.5, 99.0)

SWEEP_PERCENTILES = np.round(np.arange(90.0, 99.91, 0.5), 2)

SCORE_BATCH_SIZE = 1024
LOG_NORM_TOLERANCE = 1e-2


class EvaluationError(RuntimeError):
    """The evaluator cannot produce a trustworthy report."""


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------

def load_model(data_dir: Path) -> MPS:
    """Load the trained MPS, or fail with an actionable message."""
    path = data_dir / "mps_trained.pt"
    if not path.exists():
        raise EvaluationError(
            f"Missing {path}. Run train_mps_nsl_kdd.py first."
        )
    mps = MPS.load(str(path))
    mps.eval()
    logger.info(
        "loaded MPS: %d sites, %d parameters, bond_dims=%s",
        mps.num_sites, mps.num_parameters, list(mps.bond_dims),
    )
    return mps


def load_test(data_dir: Path) -> Tuple[torch.Tensor, Dict]:
    """Load the encoded test split and its metadata."""
    x_path = data_dir / "test_X.pt"
    meta_path = data_dir / "test_meta.pt"
    if not x_path.exists() or not meta_path.exists():
        raise EvaluationError(
            f"Missing {x_path.name} or {meta_path.name}. "
            "Run encoder_nsl_kdd.py first."
        )
    x = torch.load(x_path, weights_only=True).long()
    meta = torch.load(meta_path, weights_only=True)
    return x, meta


def reserve_splits(data_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    """Reconstruct the trainer's (train-normal, val-normal) partition.
    """
    if VAL_FRACTION <= 0.0:
        raise EvaluationError(
            "VAL_FRACTION is 0 in the trainer config: no val-normal was "
            "held out, so no attack-free threshold can be calibrated. Set "
            "VAL_FRACTION > 0 and retrain before evaluating."
        )
    x_normal = load_normal_train(data_dir)
    seed = CONFIG.seed if CONFIG.seed is not None else 0
    train_normal, val_normal = split_train_val(x_normal, VAL_FRACTION, seed)
    if val_normal is None:
        raise EvaluationError("split_train_val returned no validation set.")
    logger.info(
        "reserved trainer split: %d train-normal, %d val-normal "
        "(seed=%d, val_fraction=%.2f)",
        len(train_normal), len(val_normal), seed, VAL_FRACTION,
    )
    return train_normal, val_normal


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------

def model_sanity_checks(mps: MPS) -> Dict:
    """Cheap integrity checks on the model itself, before trusting metrics.
    """
    issues: List[str] = []

    log_norm = float(mps.log_norm().item())
    if not np.isfinite(log_norm) or abs(log_norm) > LOG_NORM_TOLERANCE:
        issues.append(
            f"log_norm={log_norm:.3e} (expected ~0; state may be miscalibrated)"
        )

    tensors_finite = all(
        torch.isfinite(t).all().item() for t in mps.site_tensors
    )
    if not tensors_finite:
        issues.append("non-finite values found in the site tensors")

    for msg in issues:
        logger.warning("model sanity: %s", msg)
    return {
        "log_norm": log_norm,
        "tensors_finite": tensors_finite,
        "issues": issues,
    }


def compute_scores(mps: MPS, x: torch.Tensor, where: str) -> np.ndarray:
    """Anomaly score (NLL) per row, with non-finite values made explicit.

    Returns finite-cleaned scores: NaN is left as NaN (callers drop it via
    ``np.isfinite``), +inf is mapped to the largest finite score so it still
    ranks as maximally anomalous, -inf to the smallest. Counts are logged.
    """
    raw = mps.anomaly_score(x.long(), batch_size=SCORE_BATCH_SIZE)
    scores = raw.detach().cpu().numpy().astype(np.float64)

    nan = np.isnan(scores)
    pos_inf = np.isposinf(scores)
    neg_inf = np.isneginf(scores)
    n_bad = int(nan.sum() + pos_inf.sum() + neg_inf.sum())
    if n_bad:
        logger.warning(
            "%s: %d/%d non-finite scores (nan=%d, +inf=%d, -inf=%d)",
            where, n_bad, scores.size,
            int(nan.sum()), int(pos_inf.sum()), int(neg_inf.sum()),
        )

    finite = scores[np.isfinite(scores)]
    if finite.size == 0:
        raise EvaluationError(
            f"{where}: every score is non-finite; the model is broken."
        )
    scores[pos_inf] = finite.max()
    scores[neg_inf] = finite.min()
    return scores


# ----------------------------------------------------------------------
# Metrics
# ----------------------------------------------------------------------

def threshold_from_percentile(val_scores: np.ndarray, pct: float) -> float:
    """Threshold = ``pct`` percentile of val-normal NLL (attack-free)."""
    return float(np.percentile(val_scores, pct))


def auc_metrics(scores: np.ndarray, is_attack: np.ndarray) -> Dict:
    """Threshold-agnostic separation quality."""
    if np.unique(is_attack).size < 2:
        raise EvaluationError(
            "test set contains a single class; cannot compute AUC."
        )
    return {
        "auc_roc": float(roc_auc_score(is_attack, scores)),
        "auc_pr": float(average_precision_score(is_attack, scores)),
    }


def threshold_metrics(
    scores: np.ndarray, is_attack: np.ndarray, threshold: float
) -> Dict:
    """Confusion-matrix-derived metrics at a fixed threshold.

    A row is flagged as an attack iff its score is >= ``threshold``.
    """
    attack = is_attack.astype(bool)
    flagged = scores >= threshold

    tp = int(np.sum(flagged & attack))
    fp = int(np.sum(flagged & ~attack))
    fn = int(np.sum(~flagged & attack))
    tn = int(np.sum(~flagged & ~attack))

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (
        2.0 * precision * recall / (precision + recall)
        if (precision + recall) else 0.0
    )
    fpr = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold": float(threshold),
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
        "precision": precision, "recall": recall, "f1": f1, "fpr": fpr,
    }


def per_family_breakdown(
    scores: np.ndarray,
    is_attack: np.ndarray,
    family_code: np.ndarray,
    family_names: List[str],
    thresholds: Dict[str, float],
) -> Dict:
    """AUC and per-threshold detection recall per attack family.

    The global AUC can hide that whole families (classically r2l / u2r on
    NSL-KDD) are nearly indistinguishable from normal traffic. Each family
    is scored against the normal class only.
    """
    normal_mask = is_attack == 0
    out: Dict[str, Dict] = {}
    for fi, fname in enumerate(family_names):
        if fname == "normal":
            continue
        fam_mask = family_code == fi
        if fam_mask.sum() == 0:
            continue

        entry: Dict = {"n_samples": int(fam_mask.sum())}
        selection = normal_mask | fam_mask
        y = fam_mask[selection].astype(int)
        if np.unique(y).size > 1:
            entry["auc_roc"] = float(roc_auc_score(y, scores[selection]))
            entry["auc_pr"] = float(
                average_precision_score(y, scores[selection])
            )
            # Prevalence of the family within (family + normal): this is the
            # no-skill baseline for AUC-PR, i.e. the value a random classifier
            # would reach. Reported so that AUC-PR can be read relative to it,
            # since a small absolute AUC-PR (e.g. u2r) can still be many times
            # above chance when the family is heavily under-represented.
            entry["auc_pr_baseline"] = float(y.mean())
        fam_scores = scores[fam_mask]
        entry["recall_at_threshold"] = {
            name: float(np.mean(fam_scores >= thr))
            for name, thr in thresholds.items()
        }
        out[fname] = entry
    return out


def per_difficulty_breakdown(
    scores: np.ndarray,
    is_attack: np.ndarray,
    difficulty: np.ndarray,
    threshold: float,
    n_buckets: int = 3,
) -> Dict:
    """Detection recall on attacks bucketed by NSL-KDD difficulty terciles."""
    attack_mask = is_attack == 1
    diff_attack = difficulty[attack_mask]
    score_attack = scores[attack_mask]
    if diff_attack.size == 0:
        return {}

    edges = np.unique(
        np.quantile(diff_attack, np.linspace(0.0, 1.0, n_buckets + 1))
    )
    out: Dict[str, Dict] = {}
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        last = i == len(edges) - 2
        bucket = (
            (diff_attack >= lo) & (diff_attack <= hi) if last
            else (diff_attack >= lo) & (diff_attack < hi)
        )
        if bucket.sum() == 0:
            continue
        out[f"difficulty_{lo:.0f}_to_{hi:.0f}"] = {
            "n_samples": int(bucket.sum()),
            "recall_at_threshold": float(
                np.mean(score_attack[bucket] >= threshold)
            ),
        }
    return out


def shift_diagnostics(
    train_normal: np.ndarray,
    val_normal: np.ndarray,
    test_normal: np.ndarray,
    thresholds: Dict[str, float],
) -> Dict:
    """Distribution-shift check across the normal subsets.

    The thresholds were calibrated for a nominal FPR on val-normal. The
    realised FPR on test-normal is the honest number: if it is far above
    nominal, train and test normal traffic differ (NSL-KDD is known for
    this) and the precision figures should be read with caution.
    """
    def pct(a: np.ndarray) -> Dict[str, float]:
        """Summarise an array by its 50/90/95/99th percentiles."""
        return {f"p{p}": float(np.percentile(a, p)) for p in (50, 90, 95, 99)}

    realised = {
        name: float(np.mean(test_normal >= thr))
        for name, thr in thresholds.items()
    }
    return {
        "train_normal": {"n": int(train_normal.size), "percentiles": pct(train_normal)},
        "val_normal": {"n": int(val_normal.size), "percentiles": pct(val_normal)},
        "test_normal": {"n": int(test_normal.size), "percentiles": pct(test_normal)},
        "realised_fpr_on_test_normal": realised,
    }


# ----------------------------------------------------------------------
# Plots  (regenerated by reading the CSVs back, like explain_mps_nsl_kdd.py)
# ----------------------------------------------------------------------

_NORMAL_COLOR = "#2c7fb8"
_ATTACK_COLOR = "#d95f0e"


def _read(csv_dir: Path, name: str, **kwargs) -> "pd.DataFrame | None":
    """Read a CSV if it exists, else log and return None."""
    path = csv_dir / name
    if not path.exists():
        logger.warning("skip %s (not found)", name)
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:  # noqa: BLE001 - keep rendering the other figures
        logger.warning("skip %s (read error: %s)", name, exc)
        return None


def _save(fig: plt.Figure, out_path: Path) -> None:
    """Save a figure at the configured DPI and close it."""
    fig.savefig(out_path, dpi=SAVE_DPI)
    plt.close(fig)
    logger.info("wrote %s", out_path.name)


def plot_score_histograms(csv_dir: Path, out_dir: Path) -> None:
    """Overlaid NLL histograms for normal vs attack test traffic."""
    df = _read(csv_dir, "test_scores.csv")
    if df is None:
        return
    scores = df["score"].to_numpy(dtype=float)
    is_attack = df["is_attack"].to_numpy(dtype=int)

    normal = scores[is_attack == 0]
    attack = scores[is_attack == 1]
    upper = float(np.percentile(scores, 99.5))
    bins = np.linspace(float(scores.min()), upper, 80)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.hist(normal, bins=bins, density=True, alpha=0.6,
            color=_NORMAL_COLOR, label=f"normal (n={normal.size})")
    ax.hist(attack, bins=bins, density=True, alpha=0.6,
            color=_ATTACK_COLOR, label=f"ataque (n={attack.size})")
    ax.set_xlabel("puntuación de anomalía (NLL)")
    ax.set_ylabel("densidad")
    ax.set_title("Distribución de la NLL: normal frente a ataque  (eje X recortado en p99.5)")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "eval_nll_histograms.png")


def plot_roc_pr(csv_dir: Path, out_dir: Path) -> None:
    """ROC and precision-recall curves with their AUROC/AUPRC annotated."""
    df = _read(csv_dir, "test_scores.csv")
    gm = _read(csv_dir, "global_metrics.csv")
    if df is None or gm is None:
        return
    scores = df["score"].to_numpy(dtype=float)
    is_attack = df["is_attack"].to_numpy(dtype=int)
    auc = {
        "auc_roc": float(gm["auc_roc"].iloc[0]),
        "auc_pr": float(gm["auc_pr"].iloc[0]),
    }

    fpr, tpr, _ = roc_curve(is_attack, scores)
    precision, recall, _ = precision_recall_curve(is_attack, scores)
    baseline = float(np.mean(is_attack))

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 4.0))
    axes[0].plot(fpr, tpr, lw=2, color=_NORMAL_COLOR,
                 label=f"AUC-ROC = {auc['auc_roc']:.4f}")
    axes[0].plot([0, 1], [0, 1], ls="--", lw=1, color="grey")
    axes[0].set_xlabel("tasa de falsos positivos")
    axes[0].set_ylabel("tasa de verdaderos positivos")
    axes[0].set_title("ROC")
    axes[0].legend(loc="lower right")

    axes[1].plot(recall, precision, lw=2, color=_ATTACK_COLOR,
                 label=f"AUC-PR = {auc['auc_pr']:.4f}")
    axes[1].axhline(baseline, ls="--", lw=1, color="grey",
                    label=f"línea base = {baseline:.3f}")
    axes[1].set_xlabel("recall")
    axes[1].set_ylabel("precisión")
    axes[1].set_title("Precisión-Recall")
    axes[1].legend(loc="lower left")

    fig.tight_layout()
    _save(fig, out_dir / "eval_roc_pr.png")


def plot_threshold_sweep(csv_dir: Path, out_dir: Path) -> None:
    """Detection metrics as the threshold moves across normal-NLL percentiles."""
    df = _read(csv_dir, "threshold_sweep.csv")
    if df is None:
        return
    df = df.sort_values("percentile")
    percentiles = df["percentile"].to_numpy(dtype=float)
    f1s = df["f1"].to_numpy(dtype=float)
    precisions = df["precision"].to_numpy(dtype=float)
    recalls = df["recall"].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    ax.plot(percentiles, f1s, "-o", ms=3, color="#1b9e77", label="F1")
    ax.plot(percentiles, precisions, "-o", ms=3, color="#7570b3",
            label="precisión")
    ax.plot(percentiles, recalls, "-o", ms=3, color="#d95f02", label="recall")
    ax.set_xlabel("percentil del normal de validación usado como umbral")
    ax.set_ylabel("métrica sobre el test")
    ax.set_title("Barrido del punto de operación")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "eval_threshold_sweep.png")


def plot_confusion(csv_dir: Path, out_dir: Path) -> None:
    """Confusion matrix at the chosen operating threshold."""
    df = _read(csv_dir, "metrics_per_threshold.csv")
    if df is None:
        return
    items = [
        (str(r["threshold_name"]), {
            "tn": int(r["tn"]), "fp": int(r["fp"]),
            "fn": int(r["fn"]), "tp": int(r["tp"]),
            "precision": float(r["precision"]), "recall": float(r["recall"]),
            "f1": float(r["f1"]), "fpr": float(r["fpr"]),
        })
        for _, r in df.iterrows()
    ]

    fig, axes = plt.subplots(
        1, len(items), figsize=(4.6 * len(items), 4.2), squeeze=False
    )
    for ax, (name, m) in zip(axes[0], items):
        cm = np.array([[m["tn"], m["fp"]], [m["fn"], m["tp"]]])
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["pred. normal", "pred. ataque"])
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["real normal", "real ataque"])
        threshold_for_white = cm.max() / 2.0
        for i in range(2):
            for j in range(2):
                ax.text(
                    j, i, f"{cm[i, j]:,}", ha="center", va="center",
                    color="white" if cm[i, j] > threshold_for_white else "black",
                )
        ax.set_title(
            f"{name}\nP={m['precision']:.3f}  R={m['recall']:.3f}  "
            f"F1={m['f1']:.3f}  FPR={m['fpr']:.3f}"
        )
    fig.tight_layout()
    _save(fig, out_dir / "eval_confusion.png")


# ----------------------------------------------------------------------
# Figure orchestration  (read every CSV back, one bad CSV skips one figure)
# ----------------------------------------------------------------------
def render_figures(tables_dir: Path, graphs_dir: Path) -> None:
    """Render every figure from the CSVs in ``tables_dir`` into ``graphs_dir``."""
    graphs_dir.mkdir(parents=True, exist_ok=True)
    logger.info("reading CSVs from %s, writing figures to %s/", tables_dir, graphs_dir)

    plotters = [
        plot_score_histograms,
        plot_roc_pr,
        plot_threshold_sweep,
        plot_confusion,
    ]
    for plotter in plotters:
        try:
            plotter(tables_dir, graphs_dir)
        except Exception as exc:  # noqa: BLE001 - one bad CSV shouldn't stop the rest
            logger.warning("%s failed: %s", plotter.__name__, exc)


# ----------------------------------------------------------------------
# Tables (CSV)
# ----------------------------------------------------------------------

def _write_csv(path: Path, header: List[str], rows: List[List]) -> None:
    """Write a header + rows to ``path`` with the csv module (handles quoting)."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info("wrote %s", path.name)


def write_tables(
    tables_dir: Path,
    *,
    global_auc: Dict,
    per_threshold: Dict[str, Dict],
    thresholds: Dict[str, float],
    families: Dict[str, Dict],
    difficulties: Dict[str, Dict],
    shift: Dict,
    strict_name: str,
    scores: np.ndarray,
    is_attack: np.ndarray,
    family_code: np.ndarray,
    family_names: List[str],
    difficulty: np.ndarray,
    val_scores: np.ndarray,
    sweep_percentiles: np.ndarray,
) -> None:
    """Write every tabular artefact of the evaluation as a CSV.

    These mirror the machine-readable sections of eval_report.json, plus the
    raw per-row scores that underlie all the plots and the threshold sweep
    that feeds eval_threshold_sweep.png.
    """
    tables_dir.mkdir(parents=True, exist_ok=True)

    # 1. Global, threshold-agnostic separation quality.
    _write_csv(
        tables_dir / "global_metrics.csv",
        ["auc_roc", "auc_pr"],
        [[f"{global_auc['auc_roc']:.6f}", f"{global_auc['auc_pr']:.6f}"]],
    )

    # 2. Confusion-matrix metrics at each operating threshold.
    rows = []
    for name, m in per_threshold.items():
        rows.append([
            name, f"{m['threshold']:.6f}",
            m["tp"], m["fp"], m["fn"], m["tn"],
            f"{m['precision']:.6f}", f"{m['recall']:.6f}",
            f"{m['f1']:.6f}", f"{m['fpr']:.6f}",
        ])
    _write_csv(
        tables_dir / "metrics_per_threshold.csv",
        ["threshold_name", "threshold", "tp", "fp", "fn", "tn",
         "precision", "recall", "f1", "fpr"],
        rows,
    )

    # 3. Per-family AUC and per-threshold detection recall (one recall
    #    column per operating threshold, ordered from lenient to strict).
    recall_names = sorted(thresholds, key=lambda n: thresholds[n])
    rows = []
    for fname, entry in families.items():
        recalls = entry.get("recall_at_threshold", {})
        rows.append([
            fname, entry.get("n_samples", ""),
            f"{entry['auc_roc']:.6f}" if "auc_roc" in entry else "",
            f"{entry['auc_pr']:.6f}" if "auc_pr" in entry else "",
            f"{entry['auc_pr_baseline']:.6f}" if "auc_pr_baseline" in entry else "",
            *[f"{recalls[name]:.6f}" if name in recalls else ""
              for name in recall_names],
        ])
    _write_csv(
        tables_dir / "per_family.csv",
        ["family", "n_samples", "auc_roc", "auc_pr", "auc_pr_baseline",
         *[f"recall_{name}" for name in recall_names]],
        rows,
    )

    # 4. Per-difficulty detection recall (at the strict threshold).
    rows = []
    for bucket, entry in difficulties.items():
        rows.append([
            bucket, entry["n_samples"],
            f"{entry['recall_at_threshold']:.6f}", strict_name,
        ])
    _write_csv(
        tables_dir / "per_difficulty.csv",
        ["difficulty_bucket", "n_samples", "recall_at_threshold", "evaluated_at"],
        rows,
    )

    # 5. Distribution shift across the normal subsets.
    rows = []
    for subset in ("train_normal", "val_normal", "test_normal"):
        entry = shift[subset]
        pct = entry["percentiles"]
        rows.append([
            subset, entry["n"],
            f"{pct['p50']:.6f}", f"{pct['p90']:.6f}",
            f"{pct['p95']:.6f}", f"{pct['p99']:.6f}",
        ])
    _write_csv(
        tables_dir / "distribution_shift.csv",
        ["subset", "n", "p50", "p90", "p95", "p99"],
        rows,
    )

    # 6. Realised FPR on test-normal at each threshold.
    rows = [
        [name, f"{thresholds[name]:.6f}", f"{fpr:.6f}"]
        for name, fpr in shift["realised_fpr_on_test_normal"].items()
    ]
    _write_csv(
        tables_dir / "realised_fpr.csv",
        ["threshold_name", "threshold", "realised_fpr_on_test_normal"],
        rows,
    )

    # 7. Operating-point sweep (same data as eval_threshold_sweep.png).
    rows = []
    for pct in sweep_percentiles:
        thr = threshold_from_percentile(val_scores, float(pct))
        m = threshold_metrics(scores, is_attack, thr)
        rows.append([
            f"{float(pct):g}", f"{thr:.6f}",
            f"{m['precision']:.6f}", f"{m['recall']:.6f}", f"{m['f1']:.6f}",
        ])
    _write_csv(
        tables_dir / "threshold_sweep.csv",
        ["percentile", "threshold", "precision", "recall", "f1"],
        rows,
    )

    # 8. Raw per-row scores underlying every plot.
    fam_lookup = np.array(family_names, dtype=object)
    rows = [
        [int(i), f"{float(s):.6f}", int(a),
         fam_lookup[c] if 0 <= c < len(fam_lookup) else str(c), int(d)]
        for i, (s, a, c, d) in enumerate(
            zip(scores, is_attack, family_code, difficulty)
        )
    ]
    _write_csv(
        tables_dir / "test_scores.csv",
        ["row", "score", "is_attack", "family", "difficulty"],
        rows,
    )


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main(data_dir: Path) -> None:
    """Score the test set, write all metric tables, render the figures, report.

    Loads the trained MPS and the held-out normal split, fixes the threshold from
    the normal NLL percentile (never from attack data), scores test traffic and
    breaks the results down globally and by attack family and difficulty. Writes
    the CSV tables and PNG figures under ``evaluate_tables/`` and
    ``evaluate_graphs/``, plus a machine-readable ``eval_report.json``.
    """
    tables_dir = data_dir / "evaluate_tables"
    graphs_dir = data_dir / "evaluate_graphs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    graphs_dir.mkdir(parents=True, exist_ok=True)

    # --- model ------------------------------------------------------
    mps = load_model(data_dir)
    sanity = model_sanity_checks(mps)

    # --- data -------------------------------------------------------
    test_x, test_meta = load_test(data_dir)
    train_normal, val_normal = reserve_splits(data_dir)

    # --- scoring ----------------------------------------------------
    test_scores_raw = compute_scores(mps, test_x, where="test")
    val_scores = compute_scores(mps, val_normal, where="val-normal")
    train_scores = compute_scores(mps, train_normal, where="train-normal")

    is_attack = test_meta["is_attack"].cpu().numpy().astype(np.int64)
    family_code = test_meta["family_code"].cpu().numpy().astype(np.int64)
    family_names = list(test_meta["family_names"])
    difficulty = test_meta["difficulty"].cpu().numpy().astype(np.int64)

    if not (len(test_scores_raw) == len(is_attack) == len(family_code)):
        raise EvaluationError(
            "test_X and test_meta disagree on the number of rows."
        )

    # Drop NaN scores consistently across scores and every label array.
    valid = np.isfinite(test_scores_raw)
    n_dropped = int((~valid).sum())
    if n_dropped:
        logger.warning(
            "dropping %d test rows with NaN scores from all metrics", n_dropped
        )
    scores = test_scores_raw[valid]
    is_attack = is_attack[valid]
    family_code = family_code[valid]
    difficulty = difficulty[valid]
    val_scores = val_scores[np.isfinite(val_scores)]
    train_scores = train_scores[np.isfinite(train_scores)]

    # --- thresholds (calibrated on val-normal only) -----------------
    thresholds = {
        f"p{pct:g}": threshold_from_percentile(val_scores, pct)
        for pct in THRESHOLD_PERCENTILES
    }
    logger.info("thresholds from val-normal: %s",
                {k: round(v, 4) for k, v in thresholds.items()})

    # --- metrics ----------------------------------------------------
    global_auc = auc_metrics(scores, is_attack)
    logger.info("global AUC-ROC=%.4f  AUC-PR=%.4f",
                global_auc["auc_roc"], global_auc["auc_pr"])

    per_threshold = {
        name: threshold_metrics(scores, is_attack, thr)
        for name, thr in thresholds.items()
    }
    for name, m in per_threshold.items():
        logger.info(
            "@%s: precision=%.4f recall=%.4f f1=%.4f fpr=%.4f",
            name, m["precision"], m["recall"], m["f1"], m["fpr"],
        )

    # The difficulty breakdown uses the stricter (p99) threshold; the
    # per-family breakdown reports recall at *every* operating threshold.
    strict_name = max(thresholds, key=lambda n: thresholds[n])
    strict_threshold = thresholds[strict_name]
    families = per_family_breakdown(
        scores, is_attack, family_code, family_names, thresholds
    )
    for fname, entry in families.items():
        logger.info(
            "family %-7s: AUC-ROC=%s  recall@%s=%.4f  (n=%d)",
            fname,
            f"{entry['auc_roc']:.4f}" if "auc_roc" in entry else "n/a",
            strict_name, entry["recall_at_threshold"][strict_name],
            entry["n_samples"],
        )
    difficulties = per_difficulty_breakdown(
        scores, is_attack, difficulty, strict_threshold
    )

    test_normal_scores = scores[is_attack == 0]
    shift = shift_diagnostics(
        train_scores, val_scores, test_normal_scores, thresholds
    )
    for name, fpr in shift["realised_fpr_on_test_normal"].items():
        logger.info(
            "realised FPR on test-normal @%s = %.4f", name, fpr
        )

    # --- report -----------------------------------------------------
    report = {
        "model": {
            "num_sites": mps.num_sites,
            "num_parameters": mps.num_parameters,
            "bond_dims": list(mps.bond_dims),
            "sanity": sanity,
        },
        "dataset": {
            "n_test_total": int(test_scores_raw.size),
            "n_test_scored": int(scores.size),
            "n_dropped_nan": n_dropped,
            "n_normal": int(np.sum(is_attack == 0)),
            "n_attack": int(np.sum(is_attack == 1)),
            "attack_rate": float(np.mean(is_attack)),
            "per_family_counts": {
                fname: int(np.sum(family_code == fi))
                for fi, fname in enumerate(family_names)
            },
        },
        "thresholds": thresholds,
        "threshold_policy": (
            "percentile of val-normal NLL; val-normal is the trainer's "
            "held-out split, reconstructed via its own split helpers"
        ),
        "global_metrics": global_auc,
        "metrics_per_threshold": per_threshold,
        "per_family": {"evaluated_at": list(thresholds.keys()), "families": families},
        "per_difficulty": {
            "evaluated_at": strict_name, "buckets": difficulties,
        },
        "distribution_shift": shift,
        "plots": [
            "evaluate_graphs/eval_nll_histograms.png",
            "evaluate_graphs/eval_roc_pr.png",
            "evaluate_graphs/eval_threshold_sweep.png",
            "evaluate_graphs/eval_confusion.png",
        ],
        "tables": [
            "evaluate_tables/global_metrics.csv",
            "evaluate_tables/metrics_per_threshold.csv",
            "evaluate_tables/per_family.csv",
            "evaluate_tables/per_difficulty.csv",
            "evaluate_tables/distribution_shift.csv",
            "evaluate_tables/realised_fpr.csv",
            "evaluate_tables/threshold_sweep.csv",
            "evaluate_tables/test_scores.csv",
        ],
    }
    report_path = data_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    logger.info("saved: %s", report_path)

    # --- tables (CSV) ----------------------------------------------
    write_tables(
        tables_dir,
        global_auc=global_auc,
        per_threshold=per_threshold,
        thresholds=thresholds,
        families=families,
        difficulties=difficulties,
        shift=shift,
        strict_name=strict_name,
        scores=scores,
        is_attack=is_attack,
        family_code=family_code,
        family_names=family_names,
        difficulty=difficulty,
        val_scores=val_scores,
        sweep_percentiles=SWEEP_PERCENTILES,
    )
    logger.info("saved 8 tables to %s/", tables_dir)

    # --- plots (regenerated by reading the CSVs we just wrote) ------
    render_figures(tables_dir, graphs_dir)
    logger.info("saved 4 plots to %s/", graphs_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./nsl_kdd")
    main(data_dir)