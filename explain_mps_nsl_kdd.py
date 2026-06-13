"""End-to-end MPS explainability for NSL-KDD: compute + plot.

    python explain_mps_nsl_kdd.py ./nsl_kdd

Each plot is wrapped so that a missing or malformed CSV only skips that one
figure instead of aborting the whole run.
"""

from __future__ import annotations

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

logger = logging.getLogger("explain_mps")

DPI = 140
# ----------------------------------------------------------------------
# Loading
# ----------------------------------------------------------------------
def load_schema(data_dir: Path) -> dict:
    schema_path = data_dir / "encoding_schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"Missing {schema_path.name}; run encoder_nsl_kdd.py first.")
    return json.loads(schema_path.read_text())


def feature_names(schema: dict) -> List[str]:
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
        return [f"{normal_value:g} (normal)", "different"]
    raise ValueError(
        f"site {site} ({feat.get('name', '?')}, kind={feat.get('kind', '?')}): "
        f"cannot build value labels; expected one of 'vocab', 'edges' or 'normal_value'"
    )


def load_split(data_dir: Path, split: str) -> Tuple[torch.Tensor, dict]:
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

    header = "site,feature,value_index,value_label,freq_prob,mps_prob,disparity"
    lines = [header]
    for site, name in enumerate(names):
        labels = value_labels(schema, site)
        for v in range(physical_dims[site]):
            lab = labels[v]
            lines.append(
                f"{site},{name},{v},{lab},"
                f"{emp_probs[site][v]:.8f},{mps_probs[site][v]:.8f},"
                f"{disparity[site]:.8f}"
            )
    out_csv.write_text("\n".join(lines) + "\n")

# ----------------------------------------------------------------------
# Per-family probability extraction (how each family deviates from normal)
# ----------------------------------------------------------------------
def family_probability_extraction(
    explainer: MPSExplainer,
    x: torch.Tensor,
    family_code: torch.Tensor,
    family_names: List[str],
    schema: dict,
    out_csv: Path,
) -> None:
    """Empirical per-family marginal vs the model's (normal) marginal.

    The MPS marginal is the model of normal traffic and is the same for
    every family; comparing each family's empirical frequency against it
    shows, feature by feature, where that family departs from normal.
    """
    names = feature_names(schema)
    physical_dims = schema["physical_dims"]
    mps_probs = [p.cpu().numpy() for p in explainer.all_feature_probabilities()]
    fc = family_code.numpy()

    header = ("family,site,feature,value_index,value_label,"
              "family_freq_prob,mps_prob,disparity")
    lines = [header]
    for code, fname in enumerate(family_names):
        mask = fc == code
        if not mask.any():
            continue
        emp = empirical_marginals(x[torch.from_numpy(mask)], physical_dims)
        for site, name in enumerate(names):
            disparity = float(np.abs(mps_probs[site] - emp[site]).sum())
            labels = value_labels(schema, site)
            for v in range(physical_dims[site]):
                lab = labels[v] if v < len(labels) else str(v)
                lines.append(
                    f"{fname},{site},{name},{v},{lab},"
                    f"{emp[site][v]:.8f},{mps_probs[site][v]:.8f},{disparity:.8f}"
                )
    out_csv.write_text("\n".join(lines) + "\n")

# ----------------------------------------------------------------------
# Von Neumann entropy
# ----------------------------------------------------------------------
def vn_entropy(
    explainer: MPSExplainer, schema: dict, out_csv: Path
) -> None:
    """Single-site von Neumann entropy per feature"""
    names = feature_names(schema)
    entropies = explainer.site_entropies().cpu().numpy()

    lines = ["site,feature,entropy"]
    for site, name in enumerate(names):
        lines.append(f"{site},{name},{entropies[site]:.8f}")
    out_csv.write_text("\n".join(lines) + "\n")


# ----------------------------------------------------------------------
# Mutual information heatmap
# ----------------------------------------------------------------------
def mutual_information(
    explainer: MPSExplainer, schema: dict, out_csv: Path
) -> None:
    """Full N x N mutual-information matrix."""
    names = feature_names(schema)
    mi = explainer.mutual_information_matrix().cpu().numpy()

    header = "feature," + ",".join(names)
    lines = [header]
    for i, name in enumerate(names):
        row = ",".join(f"{mi[i, j]:.8f}" for j in range(len(names)))
        lines.append(f"{name},{row}")
    out_csv.write_text("\n".join(lines) + "\n")

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
 
    rows: List[Dict] = []
    for site, name in enumerate(names):
        p_site = marginals[site]
        per_row_prob = p_site[x_np[:, site]]
        mean_benign = float(per_row_prob[benign_mask].mean()) if benign_mask.any() else float("nan")
        mean_attack = float(per_row_prob[attack_mask].mean()) if attack_mask.any() else float("nan")
        rows.append({
            "site": site,
            "feature": name,
            "mean_prob_benign": mean_benign,
            "mean_prob_attack": mean_attack,
            "discriminative_gap": mean_benign - mean_attack,
        })
 
    header = "site,feature,mean_prob_benign,mean_prob_attack,discriminative_gap"
    lines = [header]
    for r in rows:
        lines.append(
            f"{r['site']},{r['feature']},{r['mean_prob_benign']:.6f},"
            f"{r['mean_prob_attack']:.6f},{r['discriminative_gap']:.6f}"
        )
    out_csv.write_text("\n".join(lines) + "\n")

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
    lines = [",".join(header)]
    for site, name in enumerate(names):
        per_row = marginals[site][x_np[:, site]]
        mean_normal = float(per_row[normal_mask].mean()) if normal_mask.any() else float("nan")
        cells = [str(site), name, f"{mean_normal:.6f}"]
        for code, _ in attack_families:
            m = fc == code
            mean_f = float(per_row[m].mean()) if m.any() else float("nan")
            cells += [f"{mean_f:.6f}", f"{mean_normal - mean_f:.6f}"]
        lines.append(",".join(cells))
    out_csv.write_text("\n".join(lines) + "\n")

# ----------------------------------------------------------------------
# Anomaly identification (per-feature NLL breakdown)
# ----------------------------------------------------------------------
def anomaly_breakdown(
    mps: MPS,
    explainer: MPSExplainer,
    x: torch.Tensor,
    is_attack: torch.Tensor,
    schema: dict,
    out_csv: Path,
    top_k: int = 10,
) -> None:
    """
    For the highest-scoring anomalies, decompose the alert per feature.
    """
    names = feature_names(schema)
    marginals = [p.cpu().numpy() for p in explainer.all_feature_probabilities()]
    eps = 1e-30
 
    true_scores = mps.anomaly_score(x, batch_size=4096).cpu().numpy()
    x_np = x.numpy()
 
    order = np.argsort(-true_scores)[:top_k]
 
    header = (["rank", "row", "is_attack", "true_nll",
               "attribution_sum", "correlation_residual"]
              + [f"nll[{n}]" for n in names])
    lines = [",".join(header)]
    detail = []
    for rank, row in enumerate(order):
        per_feat = []
        for site in range(len(names)):
            p = marginals[site][x_np[row, site]]
            per_feat.append(-np.log(max(p, eps)))
        attribution_sum = float(np.sum(per_feat))
        true_nll = float(true_scores[row])
        residual = true_nll - attribution_sum
        record = ([rank, int(row), int(is_attack[row].item()),
                   f"{true_nll:.4f}", f"{attribution_sum:.4f}",
                   f"{residual:.4f}"]
                  + [f"{v:.4f}" for v in per_feat])
        lines.append(",".join(str(c) for c in record))

        top_feats = np.argsort(-np.array(per_feat))[:3]
        detail.append({
            "row": int(row),
            "is_attack": int(is_attack[row].item()),
            "true_nll": true_nll,
            "top_features": [names[t] for t in top_feats],
        })
    out_csv.write_text("\n".join(lines) + "\n")

# ----------------------------------------------------------------------
# Per-family attribution (attack signatures)
# ----------------------------------------------------------------------
def family_attribution(
    mps: MPS,
    explainer: MPSExplainer,
    x: torch.Tensor,
    family_code: torch.Tensor,
    family_names: List[str],
    schema: dict,
    out_csv: Path,
) -> None:
    """Mean per-feature NLL attribution within each attack family.

    The per-feature attribution is the same single-site quantity used in
    anomaly_breakdown (-log P_i(observed value)); here it is averaged over
    ALL rows of each family instead of the top-k anomalies, yielding the
    'signature' of each family as seen by the model: which features drive
    the surprise for dos vs probe vs r2l vs u2r.  We also report the true
    full-MPS NLL and the correlation residual per family.
    """
    names = feature_names(schema)
    marginals = [p.cpu().numpy() for p in explainer.all_feature_probabilities()]
    eps = 1e-30

    x_np = x.numpy()
    n, num = x_np.shape
    attrib = np.zeros((n, num))
    for site in range(num):
        p = marginals[site][x_np[:, site]]
        attrib[:, site] = -np.log(np.clip(p, eps, None))
    attribution_sum = attrib.sum(axis=1)
    true_nll = mps.anomaly_score(x, batch_size=4096).cpu().numpy()
    residual = true_nll - attribution_sum

    fc = family_code.numpy()
    header = (["family", "n", "mean_true_nll", "mean_attribution_sum",
               "mean_correlation_residual"] + [f"nll[{nm}]" for nm in names])
    lines = [",".join(header)]
    for code, fname in enumerate(family_names):
        mask = fc == code
        if not mask.any():
            continue
        per_feat = attrib[mask].mean(axis=0)
        rec = ([fname, int(mask.sum()),
                f"{true_nll[mask].mean():.4f}",
                f"{attribution_sum[mask].mean():.4f}",
                f"{residual[mask].mean():.4f}"]
               + [f"{v:.4f}" for v in per_feat])
        lines.append(",".join(str(c) for c in rec))
    out_csv.write_text("\n".join(lines) + "\n")

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

    header = "bond,left_feature,right_feature,bond_dim,entropy,max_entropy,saturation"
    lines = [header]
    for k, s in enumerate(entropies):
        d_k = bond_dims[k]
        max_s = float(np.log(d_k)) if d_k > 1 else 0.0
        saturation = (s / max_s) if max_s > 0 else 0.0
        lines.append(
            f"{k},{names[k]},{names[k + 1]},{d_k},"
            f"{s:.8f},{max_s:.8f},{saturation:.8f}"
        )
    out_csv.write_text("\n".join(lines) + "\n")

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

    labels = value_labels(schema, site_i)
    header = "value_index,value_label,not_conditioned,conditioned"
    lines = [header]
    for k in range(physical_dims[site_i]):
        lab = labels[k] if k < len(labels) else str(k)
        lines.append(f"{k},{lab},{unconditional[k]:.6f},{conditioned[k]:.6f}")
    out_csv.write_text("\n".join(lines) + "\n")

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

    header = ("feature_i,value_i_index,value_i_label,"
              "feature_j,value_j_index,value_j_label,"
              "joint,independent,lift")
    lines = [header]
    for vi in range(physical_dims[site_i]):
        lab_i = labels_i[vi] if vi < len(labels_i) else str(vi)
        for vj in range(physical_dims[site_j]):
            lab_j = labels_j[vj] if vj < len(labels_j) else str(vj)
            j = joint[vi, vj]
            ind = independent[vi, vj]
            lift = j / (ind + eps)
            lines.append(
                f"{names[site_i]},{vi},{lab_i},"
                f"{names[site_j]},{vj},{lab_j},"
                f"{j:.8f},{ind:.8f},{lift:.6f}"
            )
    out_csv.write_text("\n".join(lines) + "\n")



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
        plot_family_attribution,
        plot_family_feature_importance,
        plot_family_probability_extraction,
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

    # Per-family probability extraction
    logger.info("Family probability extraction -> family_probability_extraction.csv")
    family_probability_extraction(
        explainer, test_x, family_code, family_names, schema,
        out_dir / "family_probability_extraction.csv",
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
        mps, explainer, test_x, is_attack, schema, out_dir / "anomaly_breakdown.csv"
    )

    # Per-family attribution (attack signatures)
    logger.info("Family attribution -> family_attribution.csv")
    family_attribution(
        mps, explainer, test_x, family_code, family_names, schema,
        out_dir / "family_attribution.csv",
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
