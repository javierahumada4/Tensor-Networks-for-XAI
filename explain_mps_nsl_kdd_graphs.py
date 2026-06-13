"""Generate figures from the CSV artefacts written by explain_mps_nsl_kdd.py.

The explainability script separates *computation* (CSV output) from
*presentation*; this script is the presentation half.  Point it at the
directory that holds the CSVs and it writes one PNG per analysis into a
``figures/`` subdirectory.

    python plot_explain.py ./nsl_kdd/explain

Each plot is independent and wrapped so that a missing or malformed CSV
only skips that one figure instead of aborting the whole run.

Expected input files (as produced by the explainability script):
    probability_extraction.csv
    vn_entropy.csv
    mutual_information.csv
    feature_importance.csv
    anomaly_breakdown.csv
    bond_entropy.csv
    conditional_probabilities.csv
    joint_probabilities.csv
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

logger = logging.getLogger("plot_explain")

DPI = 140


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------
def _read(csv_dir: Path, name: str, **kwargs) -> Optional[pd.DataFrame]:
    """Read a CSV if it exists, else log and return None."""
    path = csv_dir / name
    if not path.exists():
        logger.warning("skip %s (not found)", name)
        return None
    try:
        return pd.read_csv(path, **kwargs)
    except Exception as exc:  # noqa: BLE001 - want to keep going
        logger.warning("skip %s (read error: %s)", name, exc)
        return None


def _save(fig: plt.Figure, out_path: Path) -> None:
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path.name)


# ----------------------------------------------------------------------
# Direct probability extraction  (empirical frequency vs MPS marginal)
# ----------------------------------------------------------------------
def plot_probability_extraction(
    csv_dir: Path, out_dir: Path, max_panels: int = 12
) -> None:
    df = _read(csv_dir, "probability_extraction.csv")
    if df is None:
        return

    # rank features by their (constant per feature) L1 disparity
    per_feature = (
        df[["site", "feature", "disparity"]]
        .drop_duplicates("site")
        .sort_values("disparity", ascending=False)
    )
    sites = per_feature["site"].tolist()[:max_panels]

    ncols = 3
    nrows = int(np.ceil(len(sites) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 2.8 * nrows))
    axes = np.atleast_1d(axes).ravel()

    for ax_idx, site in enumerate(sites):
        ax = axes[ax_idx]
        sub = df[df["site"] == site].sort_values("value_index")
        name = sub["feature"].iloc[0]
        idx = sub["value_index"].to_numpy()
        width = 0.4
        ax.bar(idx - width / 2, sub["freq_prob"], width, label="Freq.", color="0.2")
        ax.bar(idx + width / 2, sub["mps_prob"], width, label="MPS", color="crimson")
        ax.set_yscale("log")
        ax.set_title(f"[{site}] {name}", fontsize=9)
        ax.set_xlabel("value index", fontsize=8)
        ax.tick_params(labelsize=7)
    for j in range(len(sites), len(axes)):
        axes[j].axis("off")

    axes[0].set_ylabel("probability")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=9)
    fig.suptitle("Empirical frequency vs MPS-derived marginals", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_dir / "probability_extraction.png")


# ----------------------------------------------------------------------
# Single-site von Neumann entropy
# ----------------------------------------------------------------------
def plot_vn_entropy(csv_dir: Path, out_dir: Path) -> None:
    df = _read(csv_dir, "vn_entropy.csv")
    if df is None:
        return
    df = df.sort_values("site")
    names = df["feature"].tolist()
    entropies = df["entropy"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(8, 0.45 * len(names)), 4.0))
    denom = max(entropies.max(), 1e-12)
    ax.bar(np.arange(len(names)), entropies, color=plt.cm.viridis(entropies / denom))
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=7)
    ax.set_ylabel("von Neumann entropy  S(rho_k)")
    ax.set_title("Single-site von Neumann entropy "
                 "(higher = more entangled with the rest)")
    _save(fig, out_dir / "vn_entropy.png")


# ----------------------------------------------------------------------
# Mutual information heatmap
# ----------------------------------------------------------------------
def plot_mutual_information(csv_dir: Path, out_dir: Path) -> None:
    df = _read(csv_dir, "mutual_information.csv", index_col=0)
    if df is None:
        return
    names = list(df.columns)
    mi = df.to_numpy(dtype=float)

    mi_display = mi.copy()
    np.fill_diagonal(mi_display, np.nan)  # diagonal holds single-site entropy

    fig, ax = plt.subplots(figsize=(8.5, 7.5))
    im = ax.imshow(mi_display, cmap="hot", interpolation="nearest")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=6)
    ax.set_yticklabels(names, fontsize=6)
    ax.set_title("Mutual information between features  I(i;j)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="mutual information")
    _save(fig, out_dir / "mutual_information.png")


# ----------------------------------------------------------------------
# Feature importance  (benign vs attack marginal of observed value)
# ----------------------------------------------------------------------
def plot_feature_importance(csv_dir: Path, out_dir: Path) -> None:
    df = _read(csv_dir, "feature_importance.csv")
    if df is None:
        return
    df = df.sort_values("site")
    names = df["feature"].tolist()
    idx = np.arange(len(names))
    width = 0.4

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(8, 0.45 * len(names)), 8.0)
    )

    ax1.bar(idx - width / 2, df["mean_prob_benign"], width,
            label="benign", color="steelblue")
    ax1.bar(idx + width / 2, df["mean_prob_attack"], width,
            label="attack", color="indianred")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(names, rotation=90, fontsize=7)
    ax1.set_ylabel("mean P_i(observed value)")
    ax1.set_title("Feature importance: mean marginal probability (benign vs attack)")
    ax1.legend()

    # discriminative gap, sorted
    gap = df.sort_values("discriminative_gap", ascending=False)
    colors = ["seagreen" if g >= 0 else "firebrick" for g in gap["discriminative_gap"]]
    ax2.bar(np.arange(len(gap)), gap["discriminative_gap"], color=colors)
    ax2.set_xticks(np.arange(len(gap)))
    ax2.set_xticklabels(gap["feature"], rotation=90, fontsize=7)
    ax2.set_ylabel("benign - attack")
    ax2.set_title("Discriminative gap (sorted)")
    ax2.axhline(0.0, color="0.3", linewidth=0.8)

    fig.tight_layout()
    _save(fig, out_dir / "feature_importance.png")


# ----------------------------------------------------------------------
# Anomaly breakdown  (per-feature NLL of the top anomalies)
# ----------------------------------------------------------------------
def plot_anomaly_breakdown(csv_dir: Path, out_dir: Path) -> None:
    df = _read(csv_dir, "anomaly_breakdown.csv")
    if df is None:
        return

    nll_cols = [c for c in df.columns if c.startswith("nll[")]
    feat_names = [c[len("nll["):-1] for c in nll_cols]
    matrix = df[nll_cols].to_numpy(dtype=float)          # (top_k, n_features)
    ranks = df["rank"].to_numpy()
    is_attack = df["is_attack"].to_numpy()

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(9, 0.45 * len(feat_names)), 9.0),
        gridspec_kw={"height_ratios": [3, 1]},
    )

    # heatmap: rows = anomalies, cols = features, colour = per-feature NLL
    im = ax1.imshow(matrix, aspect="auto", cmap="magma", interpolation="nearest")
    ax1.set_xticks(np.arange(len(feat_names)))
    ax1.set_xticklabels(feat_names, rotation=90, fontsize=6)
    ax1.set_yticks(np.arange(len(ranks)))
    ax1.set_yticklabels([f"#{r} ({'atk' if a else 'ben'})"
                         for r, a in zip(ranks, is_attack)], fontsize=7)
    ax1.set_title("Per-feature NLL contribution of the top anomalies")
    fig.colorbar(im, ax=ax1, shrink=0.8, label="NLL contribution")

    # true score vs marginal attribution, with the correlation residual
    x = np.arange(len(ranks))
    width = 0.4
    ax2.bar(x - width / 2, df["true_nll"], width, label="true NLL", color="0.25")
    ax2.bar(x + width / 2, df["attribution_sum"], width,
            label="sum of attributions", color="darkorange")
    ax2.plot(x, df["correlation_residual"], "o-", color="teal",
             label="correlation residual", linewidth=1.2, markersize=4)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"#{r}" for r in ranks], fontsize=7)
    ax2.set_ylabel("NLL")
    ax2.set_title("True score vs marginal attribution (gap = correlation residual)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir / "anomaly_breakdown.png")


# ----------------------------------------------------------------------
# Bond entropy  (entanglement that crosses each cut, vs the ln(D) ceiling)
# ----------------------------------------------------------------------
def plot_bond_entropy(csv_dir: Path, out_dir: Path) -> None:
    df = _read(csv_dir, "bond_entropy.csv")
    if df is None:
        return
    df = df.sort_values("bond")
    bonds = df["bond"].to_numpy()
    entropy = df["entropy"].to_numpy()
    ceiling = df["max_entropy"].to_numpy()
    saturation = df["saturation"].to_numpy()
    labels = [f"{l}|{r}" for l, r in zip(df["left_feature"], df["right_feature"])]

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(9, 0.42 * len(bonds)), 7.5), sharex=True
    )

    ax1.bar(bonds, entropy, color="mediumpurple", label="S(k)")
    ax1.step(bonds, ceiling, where="mid", color="black",
             linewidth=1.2, label="ceiling  ln(D_k)")
    ax1.set_ylabel("bond entropy (nats)")
    ax1.set_title("Bipartite entanglement across each cut vs its capacity ln(D_k)")
    ax1.legend(fontsize=9)

    colors = plt.cm.RdYlGn_r(np.clip(saturation, 0, 1))
    ax2.bar(bonds, saturation, color=colors)
    ax2.axhline(1.0, color="0.3", linewidth=0.8, linestyle="--")
    ax2.set_ylim(0, max(1.05, float(saturation.max()) * 1.05))
    ax2.set_ylabel("saturation  S(k)/ln(D_k)")
    ax2.set_xticks(bonds)
    ax2.set_xticklabels(labels, rotation=90, fontsize=6)
    ax2.set_xlabel("bond (left feature | right feature)")

    fig.tight_layout()
    _save(fig, out_dir / "bond_entropy.png")


# ----------------------------------------------------------------------
# Conditional probabilities  (how knowing v_j reshapes belief about v_i)
# ----------------------------------------------------------------------
def plot_conditional_probabilities(csv_dir: Path, out_dir: Path) -> None:
    df = _read(csv_dir, "conditional_probabilities.csv")
    if df is None:
        return
    df = df.sort_values("value_index")
    labels = df["value_label"].astype(str).tolist()
    idx = np.arange(len(labels))
    width = 0.4

    fig, ax = plt.subplots(figsize=(max(7, 0.5 * len(labels)), 4.2))
    ax.bar(idx - width / 2, df["not_conditioned"], width,
           label="P(v_i)", color="slategray")
    ax.bar(idx + width / 2, df["conditioned"], width,
           label="P(v_i | v_j)", color="goldenrod")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7)
    ax.set_ylabel("probability")
    ax.set_title("Marginal vs conditional distribution of feature i")
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "conditional_probabilities.png")


# ----------------------------------------------------------------------
# Joint probabilities  (value co-occurrence lift vs independence)
# ----------------------------------------------------------------------
def plot_joint_probabilities(csv_dir: Path, out_dir: Path) -> None:
    df = _read(csv_dir, "joint_probabilities.csv")
    if df is None:
        return

    name_i = df["feature_i"].iloc[0]
    name_j = df["feature_j"].iloc[0]

    # pivot lift into a (value_i, value_j) grid; log2 so independence -> 0
    pivot = df.pivot_table(index="value_i_index", columns="value_j_index", values="lift")
    grid = pivot.to_numpy(dtype=float)
    log_lift = np.log2(np.clip(grid, 1e-12, None))

    labels_i = (df.drop_duplicates("value_i_index").sort_values("value_i_index")
                ["value_i_label"].astype(str).tolist())
    labels_j = (df.drop_duplicates("value_j_index").sort_values("value_j_index")
                ["value_j_label"].astype(str).tolist())

    vmax = float(np.nanmax(np.abs(log_lift))) or 1.0
    fig, ax = plt.subplots(figsize=(max(6, 0.6 * grid.shape[1] + 2),
                                    max(4, 0.6 * grid.shape[0] + 2)))
    im = ax.imshow(log_lift, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticklabels(labels_j, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(labels_i, fontsize=7)
    ax.set_xlabel(name_j)
    ax.set_ylabel(name_i)
    ax.set_title("Value co-occurrence: log2 lift  P(v_i,v_j) / (P(v_i)P(v_j))\n"
                 "red = more than independence, blue = less")
    fig.colorbar(im, ax=ax, shrink=0.8, label="log2 lift")
    fig.tight_layout()
    _save(fig, out_dir / "joint_probabilities.png")


# ----------------------------------------------------------------------
# Per-family attribution  (attack signatures: family x feature NLL)
# ----------------------------------------------------------------------
def plot_family_attribution(csv_dir: Path, out_dir: Path) -> None:
    df = _read(csv_dir, "family_attribution.csv")
    if df is None:
        return

    nll_cols = [c for c in df.columns if c.startswith("nll[")]
    feat_names = [c[len("nll["):-1] for c in nll_cols]
    fam = df["family"].tolist()
    matrix = df[nll_cols].to_numpy(dtype=float)        # (n_families, n_features)

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(9, 0.45 * len(feat_names)), 2.0 + 0.6 * len(fam) + 3.0),
        gridspec_kw={"height_ratios": [len(fam), 3]},
    )

    # signature heatmap: rows = families, cols = features, colour = mean NLL
    im = ax1.imshow(matrix, aspect="auto", cmap="magma", interpolation="nearest")
    ax1.set_xticks(np.arange(len(feat_names)))
    ax1.set_xticklabels(feat_names, rotation=90, fontsize=6)
    ax1.set_yticks(np.arange(len(fam)))
    ax1.set_yticklabels([f"{f} (n={n})" for f, n in zip(fam, df["n"])], fontsize=8)
    ax1.set_title("Per-family attack signature: mean per-feature NLL contribution")
    fig.colorbar(im, ax=ax1, shrink=0.8, label="mean NLL")

    # per-family true score vs marginal attribution, with residual
    x = np.arange(len(fam))
    width = 0.4
    ax2.bar(x - width / 2, df["mean_true_nll"], width, label="true NLL", color="0.25")
    ax2.bar(x + width / 2, df["mean_attribution_sum"], width,
            label="sum of attributions", color="darkorange")
    ax2.plot(x, df["mean_correlation_residual"], "o-", color="teal",
             label="correlation residual", linewidth=1.2, markersize=5)
    ax2.set_xticks(x)
    ax2.set_xticklabels(fam, fontsize=8)
    ax2.set_ylabel("mean NLL")
    ax2.set_title("Mean score vs marginal attribution per family "
                  "(gap = correlation residual)")
    ax2.legend(fontsize=8)

    fig.tight_layout()
    _save(fig, out_dir / "family_attribution.png")


# ----------------------------------------------------------------------
# Per-family feature importance  (discriminative gap vs normal)
# ----------------------------------------------------------------------
def plot_family_feature_importance(csv_dir: Path, out_dir: Path) -> None:
    df = _read(csv_dir, "family_feature_importance.csv")
    if df is None:
        return

    gap_cols = [c for c in df.columns if c.startswith("gap_")]
    fam_names = [c[len("gap_"):] for c in gap_cols]
    feats = df["feature"].tolist()
    matrix = df[gap_cols].to_numpy(dtype=float).T       # (n_families, n_features)

    vmax = float(np.nanmax(np.abs(matrix))) or 1.0
    fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(feats)),
                                    2.0 + 0.6 * len(fam_names)))
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(fam_names)))
    ax.set_yticklabels(fam_names, fontsize=9)
    ax.set_title("Discriminative gap per family   "
                 "(normal - family;  red = decisive for that family)")
    fig.colorbar(im, ax=ax, shrink=0.8, label="gap  P(obs|normal) - P(obs|family)")
    fig.tight_layout()
    _save(fig, out_dir / "family_feature_importance.png")


# ----------------------------------------------------------------------
# Per-family probability extraction  (how each family departs from normal)
# ----------------------------------------------------------------------
def plot_family_probability_extraction(csv_dir: Path, out_dir: Path) -> None:
    # parsed manually: value_label may contain a comma and break a CSV reader
    path = csv_dir / "family_probability_extraction.csv"
    if not path.exists():
        logger.warning("skip family_probability_extraction.csv (not found)")
        return

    disparity = {}          # (family, site) -> disparity
    feat_of = {}            # site -> feature
    with path.open() as fh:
        next(fh)            # header
        for line in fh:
            parts = line.rstrip("\n").split(",")
            family, site, feature = parts[0], int(parts[1]), parts[2]
            disparity[(family, site)] = float(parts[-1])
            feat_of[site] = feature

    families = list(dict.fromkeys(k[0] for k in disparity))   # preserve order
    sites = sorted(feat_of)
    feats = [feat_of[s] for s in sites]
    matrix = np.array([[disparity.get((f, s), np.nan) for s in sites]
                       for f in families])

    fig, ax = plt.subplots(figsize=(max(9, 0.45 * len(feats)),
                                    2.0 + 0.6 * len(families)))
    im = ax.imshow(matrix, aspect="auto", cmap="hot", interpolation="nearest")
    ax.set_xticks(np.arange(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=6)
    ax.set_yticks(np.arange(len(families)))
    ax.set_yticklabels(families, fontsize=9)
    ax.set_title("Per-family departure from normal: L1 disparity "
                 "of empirical vs model marginal")
    fig.colorbar(im, ax=ax, shrink=0.8, label="L1 disparity")
    fig.tight_layout()
    _save(fig, out_dir / "family_probability_extraction.png")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(csv_dir: Path) -> None:
    if not csv_dir.exists():
        raise FileNotFoundError(f"CSV directory not found: {csv_dir}")
    out_dir = csv_dir / "figures"
    out_dir.mkdir(exist_ok=True)
    logger.info("reading CSVs from %s, writing figures to %s/", csv_dir, out_dir)

    plotters = [
        plot_probability_extraction,
        plot_vn_entropy,
        plot_mutual_information,
        plot_feature_importance,
        plot_anomaly_breakdown,
        plot_bond_entropy,
        plot_conditional_probabilities,
        plot_joint_probabilities,
        plot_family_attribution,
        plot_family_feature_importance,
        plot_family_probability_extraction,
    ]
    for plotter in plotters:
        try:
            plotter(csv_dir, out_dir)
        except Exception as exc:  # noqa: BLE001 - one bad CSV shouldn't stop the rest
            logger.warning("%s failed: %s", plotter.__name__, exc)

    logger.info("done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    csv_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./nsl_kdd/explain")
    main(csv_dir)