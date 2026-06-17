"""End-to-end MPS explainability for NSL-KDD: compute + plot.

    python explain_mps_nsl_kdd.py ./nsl_kdd

Each plot is wrapped so that a missing or malformed CSV only skips that one
figure instead of aborting the whole run.
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch

from mps import MPS
from mps_explainability import MPSExplainer

# Matplotlib is only needed for the figures; import headless.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------
# Figure style: readable typography after LaTeX scales figures to ~\textwidth
# ----------------------------------------------------------------------
# on-page font size ~= (size set here) x (display width / figure width).
# Many plots below have one label per feature (~40), so they are wide; those
# are meant to be included LANDSCAPE in the document (sidewaysfigure). The two
# constants control the dense per-feature ticks and heatmap cell numbers.
plt.rcParams.update({
    "font.size":        12,
    "axes.titlesize":   13,
    "axes.labelsize":   13,
    "xtick.labelsize":  12,
    "ytick.labelsize":  12,
    "legend.fontsize":  11,
    "figure.titlesize": 14,
})
FS_DENSE = 9   # rotated per-feature tick labels when there are many features
FS_CELL  = 8   # numbers printed inside heatmap / matrix cells

logger = logging.getLogger("explain_mps")

DPI = 200


# ----------------------------------------------------------------------
# CSV writing (single, escaping-safe path for every table)
# ----------------------------------------------------------------------
def _write_csv(path: Path, header: List[str], rows: List[List]) -> None:
    """Write a CSV with proper quoting/escaping via the csv module."""
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)
    logger.info("wrote %s", path.name)


# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_schema(data_dir: Path) -> dict:
    """Read ``encoding_schema.json`` written by the encoder."""
    schema_path = data_dir / "encoding_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing {schema_path.name}; run encoder_nsl_kdd.py first.")
    return json.loads(schema_path.read_text())


def feature_names(schema: dict) -> List[str]:
    """Feature name for each site, in schema order."""
    return [f["name"] for f in schema["features"]]


def value_labels(schema: dict, site: int) -> List[str]:
    """Human-readable labels for the physical values of one site."""
    feat = schema["features"][site]
    d = feat["d"]
    if "vocab" in feat:
        return [str(v) for v in feat["vocab"]]
    if "edges" in feat:
        edges = feat["edges"]
        labels = []
        for k in range(d):
            lo = edges[k]
            hi = edges[k + 1]
            lo_s = "-inf" if lo is None else f"{lo:.3g}"
            hi_s = "+inf" if hi is None else f"{hi:.3g}"
            labels.append(f"[{lo_s};{hi_s})")
        return labels
    if "normal_value" in feat:
        normal_value = feat["normal_value"]
        return [f"{normal_value:g} (normal)", "distinto"]
    raise ValueError(
        f"site {site} ({feat.get('name', '?')}, kind={feat.get('kind', '?')}): "
        f"cannot build value labels; expected one of 'vocab', 'edges' or 'normal_value'"
    )


def load_split(data_dir: Path, split: str) -> Tuple[torch.Tensor, dict]:
    """Load the encoded tensor and metadata for ``"train"`` or ``"test"``."""
    x_path = data_dir / f"{split}_X.pt"
    meta_path = data_dir / f"{split}_meta.pt"
    if not x_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing {x_path.name}/{meta_path.name}.")
    x = torch.load(x_path, weights_only=True).long()
    meta = torch.load(meta_path, weights_only=True)
    return x, meta


# ----------------------------------------------------------------------
# Empirical helpers
# ----------------------------------------------------------------------
def empirical_marginals(x: torch.Tensor, physical_dims: List[int]) -> List[np.ndarray]:
    """Per-site empirical frequency distribution P_emp(v_k = s)."""
    out: List[np.ndarray] = []
    n = len(x)
    for site, d in enumerate(physical_dims):
        counts = torch.bincount(x[:, site], minlength=d).double()
        out.append((counts / max(n, 1)).numpy())
    return out


# ----------------------------------------------------------------------
# Direct probability extraction
# ----------------------------------------------------------------------
def probability_extraction(
    explainer: MPSExplainer,
    x_ref: torch.Tensor,
    schema: dict,
    out_csv: Path,
) -> None:
    """Empirical frequency vs MPS-derived marginal, per feature value."""
    names = feature_names(schema)
    physical_dims = schema["physical_dims"]
    mps_probs = [p.cpu().numpy() for p in explainer.all_feature_probabilities()]
    emp_probs = empirical_marginals(x_ref, physical_dims)

    disparity = np.array(
        [np.abs(mps_probs[k] - emp_probs[k]).sum() for k in range(len(names))]
    )

    header = ["site", "feature", "value_index", "value_label",
              "freq_prob", "mps_prob", "disparity"]
    rows: List[List] = []
    for site, name in enumerate(names):
        labels = value_labels(schema, site)
        for v in range(physical_dims[site]):
            lab = labels[v]
            rows.append([
                site, name, v, lab,
                f"{emp_probs[site][v]:.8f}", f"{mps_probs[site][v]:.8f}",
                f"{disparity[site]:.8f}",
            ])
    _write_csv(out_csv, header, rows)

# ----------------------------------------------------------------------
# Von Neumann entropy
# ----------------------------------------------------------------------
def vn_entropy(
    explainer: MPSExplainer, schema: dict, out_csv: Path
) -> None:
    """Single-site von Neumann entropy per feature"""
    names = feature_names(schema)
    entropies = explainer.site_entropies().cpu().numpy()

    header = ["site", "feature", "entropy"]
    rows = [[site, name, f"{entropies[site]:.8f}"]
            for site, name in enumerate(names)]
    _write_csv(out_csv, header, rows)


# ----------------------------------------------------------------------
# Mutual information heatmap
# ----------------------------------------------------------------------
def mutual_information(
    explainer: MPSExplainer, schema: dict, out_csv: Path
) -> None:
    """Full N x N mutual-information matrix."""
    names = feature_names(schema)
    mi = explainer.mutual_information_matrix().cpu().numpy()

    header = ["feature"] + list(names)
    rows: List[List] = []
    for i, name in enumerate(names):
        rows.append([name] + [f"{mi[i, j]:.8f}" for j in range(len(names))])
    _write_csv(out_csv, header, rows)

# ----------------------------------------------------------------------
# Feature importance
# ----------------------------------------------------------------------
def feature_importance(
    explainer: MPSExplainer,
    x: torch.Tensor,
    is_attack: torch.Tensor,
    schema: dict,
    out_csv: Path,
) -> None:
    """
    Mean per-feature marginal probability of the observed values,
    split into benign vs attack rows.
    """
    names = feature_names(schema)
    marginals = [p.cpu().numpy() for p in explainer.all_feature_probabilities()]
 
    benign_mask = (is_attack == 0).numpy()
    attack_mask = ~benign_mask
    x_np = x.numpy()
 
    dict_rows: List[Dict] = []
    for site, name in enumerate(names):
        p_site = marginals[site]
        per_row_prob = p_site[x_np[:, site]]
        mean_benign = float(per_row_prob[benign_mask].mean()) if benign_mask.any() else float("nan")
        mean_attack = float(per_row_prob[attack_mask].mean()) if attack_mask.any() else float("nan")
        dict_rows.append({
            "site": site,
            "feature": name,
            "mean_prob_benign": mean_benign,
            "mean_prob_attack": mean_attack,
            "discriminative_gap": mean_benign - mean_attack,
        })

    header = ["site", "feature", "mean_prob_benign",
              "mean_prob_attack", "discriminative_gap"]
    rows = [
        [r["site"], r["feature"], f"{r['mean_prob_benign']:.6f}",
         f"{r['mean_prob_attack']:.6f}", f"{r['discriminative_gap']:.6f}"]
        for r in dict_rows
    ]
    _write_csv(out_csv, header, rows)

# ----------------------------------------------------------------------
# Per-family feature importance (discriminative gap vs normal)
# ----------------------------------------------------------------------
def family_feature_importance(
    explainer: MPSExplainer,
    x: torch.Tensor,
    family_code: torch.Tensor,
    family_names: List[str],
    schema: dict,
    out_csv: Path,
) -> None:
    """Discriminative gap of each feature, normal vs EACH attack family.

    The binary benign-vs-all-attacks gap averages over every family and
    washes out family-specific signals.  Here the gap is computed against
    each family separately:  gap_f = mean P_i(observed | normal)
                                     - mean P_i(observed | family f).
    A large positive gap_f flags a feature decisive for that family.
    """
    names = feature_names(schema)
    marginals = [p.cpu().numpy() for p in explainer.all_feature_probabilities()]
    x_np = x.numpy()
    fc = family_code.numpy()

    normal_idx = family_names.index("normal")
    normal_mask = fc == normal_idx
    attack_families = [(c, f) for c, f in enumerate(family_names) if f != "normal"]

    header = (["site", "feature", "mean_prob_normal"]
              + [f"{tag}_{f}" for _, f in attack_families
                 for tag in ("mean_prob", "gap")])
    rows: List[List] = []
    for site, name in enumerate(names):
        per_row = marginals[site][x_np[:, site]]
        mean_normal = float(per_row[normal_mask].mean()) if normal_mask.any() else float("nan")
        cells: List = [site, name, f"{mean_normal:.6f}"]
        for code, _ in attack_families:
            m = fc == code
            mean_f = float(per_row[m].mean()) if m.any() else float("nan")
            cells += [f"{mean_f:.6f}", f"{mean_normal - mean_f:.6f}"]
        rows.append(cells)
    _write_csv(out_csv, header, rows)

# ----------------------------------------------------------------------
# Anomaly identification (per-feature NLL breakdown)
# ----------------------------------------------------------------------
def anomaly_breakdown(
    mps: MPS,
    explainer: MPSExplainer,
    x: torch.Tensor,
    is_attack: torch.Tensor,
    family_code: torch.Tensor,
    family_names: List[str],
    schema: dict,
    out_csv: Path,
    n_each: int = 3,
    nll_percentile: float = 90.0,
) -> None:
    """Decompose, per feature, the NLL of the anomalies with the lowest and
    highest correlation share.

    The selection is meant to showcase the correlation-share reliability
    metric: among clearly anomalous connections (NLL above ``nll_percentile``),
    it picks the ``n_each`` with the smallest share (value anomalies, whose
    per-feature breakdown is faithful) and the ``n_each`` with the largest
    share (correlation anomalies, whose breakdown is misleading because the
    surprise lives in the combination of values rather than in any single one).
    Restricting to high-NLL connections is essential: for low-NLL ones the
    share |residual| / NLL is dominated by a tiny denominator and meaningless.
    """
    names = feature_names(schema)
    marginals = [p.cpu().numpy() for p in explainer.all_feature_probabilities()]
    eps = 1e-30

    true_scores = mps.anomaly_score(x, batch_size=4096).cpu().numpy()
    x_np = x.numpy()
    n, num = x_np.shape
    fc = family_code.numpy()
    ia = is_attack.numpy()
    fam_lookup = np.array(family_names, dtype=object)

    attrib = np.zeros((n, num))
    for site in range(num):
        attrib[:, site] = -np.log(np.clip(marginals[site][x_np[:, site]], eps, None))
    attribution_sum = attrib.sum(axis=1)
    residual = true_scores - attribution_sum
    share = np.abs(residual) / np.clip(true_scores, eps, None)

    # restrict to clearly anomalous connections so the share is meaningful
    floor = np.percentile(true_scores, nll_percentile)
    pool = np.where(true_scores >= floor)[0]
    by_share = pool[np.argsort(share[pool])]

    # NSL-KDD contains exact duplicate connections, so pick *distinct* ones
    def take_distinct(order, k):
        """Take the first ``k`` rows in ``order`` that are distinct configurations."""
        seen, out = set(), []
        for r in order:
            key = tuple(int(v) for v in x_np[r])
            if key in seen:
                continue
            seen.add(key)
            out.append(int(r))
            if len(out) >= k:
                break
        return out

    low = take_distinct(by_share, n_each)        # value anomalies (reliable)
    high = take_distinct(by_share[::-1], n_each)  # correlation anomalies (misleading)
    selected = sorted(set(low) | set(high), key=lambda r: share[r])

    header = (["family", "row", "is_attack", "true_nll",
               "attribution_sum", "correlation_residual", "correlation_share"]
              + [f"nll[{nm}]" for nm in names])
    rows: List[List] = []
    for row in selected:
        fam = fam_lookup[fc[row]] if 0 <= fc[row] < len(fam_lookup) else str(fc[row])
        record = ([fam, int(row), int(ia[row]),
                   f"{true_scores[row]:.4f}", f"{attribution_sum[row]:.4f}",
                   f"{residual[row]:.4f}", f"{share[row]:.4f}"]
                  + [f"{v:.4f}" for v in attrib[row]])
        rows.append(record)
    _write_csv(out_csv, header, rows)

# ----------------------------------------------------------------------
# Bond entropy
# ----------------------------------------------------------------------
def bond_entropy(
    explainer: MPSExplainer, schema: dict, out_csv: Path
) -> None:
    """Bipartite von Neumann entropy at every bond of the chain.

    S(k) = -sum_i p_i ln p_i,  p_i = sigma_i^2 / sum_j sigma_j^2,

    where sigma_i are the singular values at bond k.  Whereas the
    single-site entropy (vn_entropy) measures how entangled ONE feature
    is with the rest, the bond entropy measures the entanglement across
    the CUT that splits the chain into features [0..k] and [k+1..N-1],
    i.e. how much correlation crosses that point of the ordering.

    We tabulate it next to the actual bond dimension D_k and the
    theoretical ceiling ln(D_k): the trained model can only carry
    S(k) <= ln(D_k) of entanglement across bond k, so the ratio
    S(k) / ln(D_k) shows how "used up" each bond's capacity is.
    """
    names = feature_names(schema)
    entropies = explainer.bond_entropies()
    bond_dims = explainer.mps.bond_dims

    header = ["bond", "left_feature", "right_feature", "bond_dim",
              "entropy", "max_entropy", "saturation"]
    rows: List[List] = []
    for k, s in enumerate(entropies):
        d_k = bond_dims[k]
        max_s = float(np.log(d_k)) if d_k > 1 else 0.0
        saturation = (s / max_s) if max_s > 0 else 0.0
        rows.append([
            k, names[k], names[k + 1], d_k,
            f"{s:.8f}", f"{max_s:.8f}", f"{saturation:.8f}",
        ])
    _write_csv(out_csv, header, rows)

# ----------------------------------------------------------------------
# Conditional probabilities
# ----------------------------------------------------------------------
def conditional_probabilities(
    explainer: MPSExplainer,
    schema: dict,
    out_csv: Path,
    site_i: Optional[int] = None,
    site_j: Optional[int] = None,
    value_j: int = 0,
) -> None:
    """Compare P(v_i) against P(v_i | v_j = value_j).

    If site_i / site_j are not given, pick the most strongly correlated
    pair from the MI matrix so the example is actually illustrative.
    """
    names = feature_names(schema)
    physical_dims = schema["physical_dims"]

    if site_i is None or site_j is None:
        mi = explainer.mutual_information_matrix().cpu().numpy()
        np.fill_diagonal(mi, -np.inf)
        flat = int(np.argmax(mi))
        site_i, site_j = divmod(flat, mi.shape[0])

    unconditional = explainer.feature_probabilities(site_i).cpu().numpy()
    conditioned = explainer.conditional_probabilities(site_i, site_j, value_j).cpu().numpy()

    labels_i = value_labels(schema, site_i)
    labels_j = value_labels(schema, site_j)
    fname_i, fname_j = names[site_i], names[site_j]
    cond_label = labels_j[value_j] if value_j < len(labels_j) else str(value_j)
    header = [
        "feature_i", "value_index", "value_i_label",
        "feature_j", "value_j_index", "value_j_label",
        "not_conditioned", "conditioned",
    ]
    rows: List[List] = []
    for k in range(physical_dims[site_i]):
        lab = labels_i[k] if k < len(labels_i) else str(k)
        rows.append([
            fname_i, k, lab,
            fname_j, value_j, cond_label,
            f"{unconditional[k]:.6f}", f"{conditioned[k]:.6f}",
        ])
    _write_csv(out_csv, header, rows)

# ----------------------------------------------------------------------
# Joint probabilities (two-feature co-occurrence)
# ----------------------------------------------------------------------
def joint_probabilities(
    explainer: MPSExplainer,
    schema: dict,
    out_csv: Path,
    site_i: Optional[int] = None,
    site_j: Optional[int] = None,
) -> None:
    """Joint distribution P(v_i, v_j) for a feature pair, compared against
    the product of marginals P(v_i)·P(v_j).

    If site_i / site_j are not given, pick the most strongly correlated
    pair from the MI matrix.  The 'lift' = P(v_i, v_j) / (P(v_i)·P(v_j))
    flags value combinations that co-occur far more (lift > 1) or far less
    (lift < 1) than independence would predict -- i.e. the joint value
    patterns the model actually learned, which MI summarises into a single
    scalar and the conditional fixes to one value.
    """
    names = feature_names(schema)
    physical_dims = schema["physical_dims"]
    eps = 1e-30

    if site_i is None or site_j is None:
        mi = explainer.mutual_information_matrix().cpu().numpy()
        np.fill_diagonal(mi, -np.inf)
        flat = int(np.argmax(mi))
        site_i, site_j = divmod(flat, mi.shape[0])

    joint = explainer.joint_probabilities(site_i, site_j).cpu().numpy()
    p_i = explainer.feature_probabilities(site_i).cpu().numpy()
    p_j = explainer.feature_probabilities(site_j).cpu().numpy()
    independent = np.outer(p_i, p_j)

    labels_i = value_labels(schema, site_i)
    labels_j = value_labels(schema, site_j)

    header = ["feature_i", "value_i_index", "value_i_label",
              "feature_j", "value_j_index", "value_j_label",
              "joint", "independent", "lift"]
    rows: List[List] = []
    for vi in range(physical_dims[site_i]):
        lab_i = labels_i[vi] if vi < len(labels_i) else str(vi)
        for vj in range(physical_dims[site_j]):
            lab_j = labels_j[vj] if vj < len(labels_j) else str(vj)
            j = joint[vi, vj]
            ind = independent[vi, vj]
            lift = j / (ind + eps)
            rows.append([
                names[site_i], vi, lab_i,
                names[site_j], vj, lab_j,
                f"{j:.8f}", f"{ind:.8f}", f"{lift:.6f}",
            ])
    _write_csv(out_csv, header, rows)



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
    """Save a figure (tight bounding box) at the configured DPI and close it."""
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    logger.info("wrote %s", out_path.name)


# ----------------------------------------------------------------------
# Direct probability extraction  (empirical frequency vs MPS marginal)
# ----------------------------------------------------------------------
def plot_probability_extraction(
    csv_dir: Path, out_dir: Path, max_panels: int = 3
) -> None:
    """Model vs empirical marginals for a few features, as paired bar panels."""
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
        ax.bar(idx - width / 2, sub["freq_prob"], width, label="Frec.", color="0.2")
        ax.bar(idx + width / 2, sub["mps_prob"], width, label="MPS", color="crimson")
        ax.set_yscale("log")
        ax.set_title(f"[{site}] {name}", fontsize=10)
        labels = sub["value_label"].astype(str).tolist()
        ax.set_xticks(idx)
        ax.set_xticklabels(labels, rotation=90, fontsize=FS_DENSE)
        ax.tick_params(axis="y", labelsize=10)
    for j in range(len(sites), len(axes)):
        axes[j].axis("off")

    axes[0].set_ylabel("probabilidad")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", fontsize=11)
    fig.suptitle("Frecuencia empírica frente a marginales del MPS", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    _save(fig, out_dir / "probability_extraction.png")


# ----------------------------------------------------------------------
# Single-site von Neumann entropy
# ----------------------------------------------------------------------
def plot_vn_entropy(csv_dir: Path, out_dir: Path) -> None:
    """Per-site von Neumann entropy as a bar chart over features."""
    df = _read(csv_dir, "vn_entropy.csv")
    if df is None:
        return
    df = df.sort_values("site")
    names = df["feature"].tolist()
    entropies = df["entropy"].to_numpy()

    fig, ax = plt.subplots(figsize=(max(8, 0.30 * len(names)), 4.0))
    denom = max(entropies.max(), 1e-12)
    ax.bar(np.arange(len(names)), entropies, color=plt.cm.viridis(entropies / denom))
    ax.set_xticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=FS_DENSE)
    ax.set_ylabel("entropía de Von Neumann  S(rho_k)")
    ax.set_title("Entropía de Von Neumann por sitio "
                 "(mayor = más entrelazada con el resto)")
    _save(fig, out_dir / "vn_entropy.png")


# ----------------------------------------------------------------------
# Mutual information heatmap
# ----------------------------------------------------------------------
def plot_mutual_information(csv_dir: Path, out_dir: Path) -> None:
    """Mutual-information matrix between features, as a heatmap."""
    df = _read(csv_dir, "mutual_information.csv", index_col=0)
    if df is None:
        return
    names = list(df.columns)
    mi = df.to_numpy(dtype=float)

    mi_display = mi.copy()
    np.fill_diagonal(mi_display, np.nan)  # diagonal holds single-site entropy

    fig, ax = plt.subplots(figsize=(8.5, 7.5), layout="constrained")
    im = ax.imshow(mi_display, cmap="hot", interpolation="nearest")
    ax.set_xticks(np.arange(len(names)))
    ax.set_yticks(np.arange(len(names)))
    ax.set_xticklabels(names, rotation=90, fontsize=FS_DENSE)
    ax.set_yticklabels(names, fontsize=FS_DENSE)
    ax.set_title("Información mutua entre características  I(i;j)")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("información mutua", fontsize=11)
    _save(fig, out_dir / "mutual_information.png")


# ----------------------------------------------------------------------
# Feature importance  (benign vs attack marginal of observed value)
# ----------------------------------------------------------------------
def plot_feature_importance(csv_dir: Path, out_dir: Path) -> None:
    """Features ranked by how much they shift the NLL when perturbed."""
    df = _read(csv_dir, "feature_importance.csv")
    if df is None:
        return
    df = df.sort_values("site")
    names = df["feature"].tolist()
    idx = np.arange(len(names))
    width = 0.4

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(max(11, 0.34 * len(names)), 9.0), layout="constrained"
    )

    ax1.bar(idx - width / 2, df["mean_prob_benign"], width,
            label="benigno", color="steelblue")
    ax1.bar(idx + width / 2, df["mean_prob_attack"], width,
            label="ataque", color="indianred")
    ax1.set_xticks(idx)
    ax1.set_xticklabels(names, rotation=90, fontsize=FS_DENSE)
    ax1.set_ylabel("P_i(valor observado) media")
    ax1.set_title("Importancia de características: probabilidad marginal media (benigno vs ataque)")
    # headroom so the legend sits above the bars instead of covering them
    ax1.set_ylim(top=ax1.get_ylim()[1] * 1.25)
    ax1.legend(ncol=2, loc="upper left", framealpha=0.9)

    # discriminative gap, sorted
    gap = df.sort_values("discriminative_gap", ascending=False)
    colors = ["seagreen" if g >= 0 else "firebrick" for g in gap["discriminative_gap"]]
    ax2.bar(np.arange(len(gap)), gap["discriminative_gap"], color=colors)
    ax2.set_xticks(np.arange(len(gap)))
    ax2.set_xticklabels(gap["feature"], rotation=90, fontsize=FS_DENSE)
    ax2.set_ylabel("benigno - ataque")
    ax2.set_title("Brecha discriminativa (ordenada)")
    ax2.axhline(0.0, color="0.3", linewidth=0.8)

    _save(fig, out_dir / "feature_importance.png")


# ----------------------------------------------------------------------
# Anomaly breakdown  (per-feature NLL of the top anomalies)
# ----------------------------------------------------------------------
def _share_color(c: float) -> str:
    """Reliability colour of a per-feature breakdown given its correlation
    share: green if the marginal attributions explain the score, amber if
    correlations carry a noticeable part, red if they dominate."""
    if c < 0.2:
        return "seagreen"
    if c < 0.5:
        return "goldenrod"
    return "indianred"


def _plot_correlation_share(ax, shares, ypos) -> None:
    """Horizontal bar of the correlation share (|residual| / true NLL) for
    each row, aligned with a heatmap that shares the same rows."""
    shares = np.asarray(shares, dtype=float)
    xmax = max(1.0, float(shares.max()) * 1.10) if shares.size else 1.0
    ax.barh(ypos, shares, color=[_share_color(c) for c in shares])
    ax.set_xlim(0, xmax)
    for xv in (0.2, 0.5):
        ax.axvline(xv, color="0.6", lw=0.8, ls=":")
    if xmax > 1.0:
        ax.axvline(1.0, color="0.6", lw=0.8, ls="--")
    ax.set_yticks([])
    ax.set_title("cuota de correlación", fontsize=10)
    for y, c in zip(ypos, shares):
        ax.text(c + 0.02 * xmax, y, f"{c * 100:.0f}%",
                va="center", ha="left", fontsize=FS_CELL)


def plot_anomaly_breakdown(csv_dir: Path, out_dir: Path) -> None:
    """Per-feature contribution to the anomaly score, attacks vs normal."""
    df = _read(csv_dir, "anomaly_breakdown.csv")
    if df is None:
        return

    nll_cols = [c for c in df.columns if c.startswith("nll[")]
    feat_names = [c[len("nll["):-1] for c in nll_cols]
    matrix = df[nll_cols].to_numpy(dtype=float)          # (n_rows, n_features)
    families = df["family"].astype(str).tolist()
    is_attack = df["is_attack"].to_numpy()
    residual = df["correlation_residual"].to_numpy(dtype=float)
    nrows = len(df)
    labels = [f"{fam} ({'atq' if a else 'ben'})"
              for fam, a in zip(families, is_attack)]

    # heatmap (left) + correlation-residual gap per anomaly (right), sharing rows
    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(max(9, 0.30 * len(feat_names)), 1.4 + 0.7 * nrows),
        gridspec_kw={"width_ratios": [8, 1.8]},
    )

    im = ax1.imshow(matrix, aspect="auto", cmap="magma", interpolation="nearest")
    ax1.set_xticks(np.arange(len(feat_names)))
    ax1.set_xticklabels(feat_names, rotation=90, fontsize=FS_DENSE)
    ax1.set_yticks(np.arange(nrows))
    ax1.set_yticklabels(labels, fontsize=10)
    ax1.set_title("Contribución NLL por característica")
    fig.colorbar(im, ax=ax1, fraction=0.025, pad=0.01, label="contribución NLL")

    # correlation share per anomaly: how much of the score lives in the
    # correlations -> tells the analyst whether to trust this per-feature
    # breakdown (low share) or inspect the correlation structure (high share)
    if "correlation_share" in df.columns:
        shares = df["correlation_share"].to_numpy(dtype=float)
    else:
        tn = df["true_nll"].to_numpy(dtype=float)
        shares = np.abs(residual) / np.clip(tn, 1e-30, None)
    ypos = np.arange(nrows)
    _plot_correlation_share(ax2, shares, ypos)
    ax2.set_ylim(ax1.get_ylim())          # align (inverted) with heatmap rows

    fig.tight_layout()
    _save(fig, out_dir / "anomaly_breakdown.png")


# ----------------------------------------------------------------------
# Bond entropy  (entanglement that crosses each cut, vs the ln(D) ceiling)
# ----------------------------------------------------------------------
def plot_bond_entropy(csv_dir: Path, out_dir: Path) -> None:
    """Bipartite entanglement entropy along the chain, bond by bond."""
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
        2, 1, figsize=(max(11, 0.34 * len(bonds)), 8.5), sharex=True
    )

    ax1.bar(bonds, entropy, color="mediumpurple", label="S(k)")
    ax1.step(bonds, ceiling, where="mid", color="black",
             linewidth=1.2, label="cota  ln(D_k)")
    ax1.set_ylabel("entropía de enlace (nats)")
    ax1.set_title("Entrelazamiento bipartito en cada corte frente a su capacidad ln(D_k)")
    # compact two-column legend in the empty top-left corner
    ax1.legend(fontsize=10, ncol=2, loc="upper left", framealpha=0.9)

    colors = plt.cm.RdYlGn_r(np.clip(saturation, 0, 1))
    ax2.bar(bonds, saturation, color=colors)
    ax2.axhline(1.0, color="0.3", linewidth=0.8, linestyle="--")
    ax2.set_ylim(0, max(1.05, float(saturation.max()) * 1.05))
    ax2.set_ylabel("saturación  S(k)/ln(D_k)")
    ax2.set_xticks(bonds)
    ax2.set_xticklabels(labels, rotation=90, fontsize=FS_DENSE)
    ax2.set_xlabel("enlace (característica izquierda | derecha)")

    fig.tight_layout()
    _save(fig, out_dir / "bond_entropy.png")


# ----------------------------------------------------------------------
# Conditional probabilities  (how knowing v_j reshapes belief about v_i)
# ----------------------------------------------------------------------
def plot_conditional_probabilities(csv_dir: Path, out_dir: Path) -> None:
    """Selected conditional distributions P(v_i | v_j) as bars."""
    df = _read(csv_dir, "conditional_probabilities.csv")
    if df is None:
        return
    df = df.sort_values("value_index")

    # value labels of the conditioned feature i (support old CSVs too)
    label_col = "value_i_label" if "value_i_label" in df.columns else "value_label"
    labels = df[label_col].astype(str).tolist()
    idx = np.arange(len(labels))
    width = 0.4

    if "feature_i" in df.columns:
        name_i = str(df["feature_i"].iloc[0])
        name_j = str(df["feature_j"].iloc[0])
        cond_label = str(df["value_j_label"].iloc[0])
        leg_marg = f"P({name_i})"
        leg_cond = f"P({name_i} | {name_j} = {cond_label})"
        title = (f"Distribución de {name_i}: marginal frente a "
                 f"condicionada a {name_j} = {cond_label}")
        xlabel = f"valor de {name_i}"
    else:
        leg_marg, leg_cond = "P(v_i)", "P(v_i | v_j)"
        title = "Distribución marginal frente a condicional de la característica i"
        xlabel = "valor"

    fig, ax = plt.subplots(figsize=(max(7, 0.36 * len(labels)), 4.2))
    ax.bar(idx - width / 2, df["not_conditioned"], width,
           label=leg_marg, color="slategray")
    ax.bar(idx + width / 2, df["conditioned"], width,
           label=leg_cond, color="goldenrod")
    ax.set_xticks(idx)
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=FS_DENSE)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("probabilidad")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    _save(fig, out_dir / "conditional_probabilities.png")


# ----------------------------------------------------------------------
# Joint probabilities  (value co-occurrence lift vs independence)
# ----------------------------------------------------------------------
def plot_joint_probabilities(csv_dir: Path, out_dir: Path) -> None:
    """Selected joint distributions P(v_i, v_j) as heatmaps."""
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
    fig, ax = plt.subplots(figsize=(max(6, 0.45 * grid.shape[1] + 2),
                                    max(4, 0.6 * grid.shape[0] + 2)),
                           layout="constrained")
    im = ax.imshow(log_lift, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   interpolation="nearest", aspect="auto")
    ax.set_xticks(np.arange(grid.shape[1]))
    ax.set_yticks(np.arange(grid.shape[0]))
    ax.set_xticklabels(labels_j, rotation=45, ha="right", fontsize=FS_DENSE)
    ax.set_yticklabels(labels_i, fontsize=FS_DENSE)
    ax.set_xlabel(name_j)
    ax.set_ylabel(name_i)
    ax.set_title("Coaparición de valores: \n"
                 "rojo = más que la independencia, azul = menos")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("log2 lift", fontsize=11)
    _save(fig, out_dir / "joint_probabilities.png")


# ----------------------------------------------------------------------
# Per-family feature importance  (discriminative gap vs normal)
# ----------------------------------------------------------------------
def plot_family_feature_importance(csv_dir: Path, out_dir: Path) -> None:
    """Feature importance broken down per attack family, as a heatmap."""
    df = _read(csv_dir, "family_feature_importance.csv")
    if df is None:
        return

    gap_cols = [c for c in df.columns if c.startswith("gap_")]
    fam_names = [c[len("gap_"):] for c in gap_cols]
    feats = df["feature"].tolist()
    matrix = df[gap_cols].to_numpy(dtype=float).T       # (n_families, n_features)

    vmax = float(np.nanmax(np.abs(matrix))) or 1.0
    fig, ax = plt.subplots(figsize=(max(11, 0.32 * len(feats)),
                                    2.6 + 0.7 * len(fam_names)),
                           layout="constrained")
    im = ax.imshow(matrix, aspect="auto", cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax, interpolation="nearest")
    ax.set_xticks(np.arange(len(feats)))
    ax.set_xticklabels(feats, rotation=90, fontsize=FS_DENSE)
    ax.set_yticks(np.arange(len(fam_names)))
    ax.set_yticklabels(fam_names, fontsize=11)
    ax.set_title("Brecha discriminativa por familia\n"
                 "(normal - familia;  rojo = decisiva para esa familia)")
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.012)
    cbar.set_label("brecha  P(obs|normal) - P(obs|familia)", fontsize=11)
    _save(fig, out_dir / "family_feature_importance.png")


# ----------------------------------------------------------------------
# Figure orchestration
# ----------------------------------------------------------------------
def render_figures(tables_dir: Path, graphs_dir: Path) -> None:
    """Render every figure from the CSVs in ``tables_dir`` into ``graphs_dir``."""
    graphs_dir.mkdir(parents=True, exist_ok=True)
    logger.info("reading CSVs from %s, writing figures to %s/", tables_dir, graphs_dir)

    plotters = [
        plot_probability_extraction,
        plot_vn_entropy,
        plot_mutual_information,
        plot_feature_importance,
        plot_anomaly_breakdown,
        plot_bond_entropy,
        plot_conditional_probabilities,
        plot_joint_probabilities,
        plot_family_feature_importance,
    ]
    for plotter in plotters:
        try:
            plotter(tables_dir, graphs_dir)
        except Exception as exc:  # noqa: BLE001 - one bad CSV shouldn't stop the rest
            logger.warning("%s failed: %s", plotter.__name__, exc)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------
def main(data_dir: Path) -> None:
    """Compute every explainability table, then render its figure.

    Loads the trained MPS and the encoding schema, builds an
    :class:`MPSExplainer`, and walks the analyses in turn — marginals, entropies,
    mutual information, feature importance (overall and per family), anomaly
    breakdown, conditional and joint probabilities. Each one writes a CSV under
    ``explain_tables/`` and a matching PNG under ``explain_graphs/``; a failure in
    one plot is logged and skipped so the rest still render.
    """
    tables_dir = data_dir / "explain_tables"
    graphs_dir = data_dir / "explain_graphs"
    tables_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tables_dir  # CSVs are written here

    schema = load_schema(data_dir)
    names = feature_names(schema)
    logger.info("schema: %d features", len(names))

    mps = MPS.load(str(data_dir / "mps_trained.pt"))
    logger.info("loaded MPS: %d sites, bond_dims=%s", mps.num_sites, mps.bond_dims)

    # Reference data for the empirical baseline and the splits
    train_x, _ = load_split(data_dir, "train")
    test_x, test_meta = load_split(data_dir, "test")
    is_attack = test_meta["is_attack"]
    family_code = test_meta["family_code"]
    family_names = test_meta["family_names"]

    explainer = MPSExplainer(mps)
    explainer.precompute_environments()

    # Direct probability extraction
    logger.info("Probability extraction -> probability_extraction.csv")
    probability_extraction(
        explainer, train_x, schema, out_dir / "probability_extraction.csv"
    )

    # Von Neumann entropy
    logger.info("Von Neumann entropy -> vn_entropy.csv")
    vn_entropy(explainer, schema, out_dir / "vn_entropy.csv")

    # Mutual information heatmap
    logger.info("Mutual information -> mutual_information.csv")
    mutual_information(explainer, schema, out_dir / "mutual_information.csv")

    # Feature importance
    logger.info("Feature importance -> feature_importance.csv")
    feature_importance(
        explainer, test_x, is_attack, schema,
        out_dir / "feature_importance.csv",
    )

    # Per-family feature importance (gap vs normal)
    logger.info("Family feature importance -> family_feature_importance.csv")
    family_feature_importance(
        explainer, test_x, family_code, family_names, schema,
        out_dir / "family_feature_importance.csv",
    )

    # Anomaly identification (per-feature NLL breakdown)
    logger.info("Anomaly breakdown -> anomaly_breakdown.csv")
    anomaly_breakdown(
        mps, explainer, test_x, is_attack, family_code, family_names, schema,
        out_dir / "anomaly_breakdown.csv"
    )

    # Bond entropy
    logger.info("Bond entropy -> bond_entropy.csv")
    bond_entropy(explainer, schema, out_dir / "bond_entropy.csv")

    # Conditional probabilities
    logger.info("Conditional probabilities -> conditional_probabilities.csv")
    conditional_probabilities(
        explainer, schema, out_dir / "conditional_probabilities.csv"
    )

    # Joint probabilities (two-feature co-occurrence)
    logger.info("Joint probabilities -> joint_probabilities.csv")
    joint_probabilities(
        explainer, schema, out_dir / "joint_probabilities.csv"
    )

    # Presentation half: turn every CSV into a figure
    render_figures(tables_dir, graphs_dir)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./nsl_kdd")
    main(data_dir)