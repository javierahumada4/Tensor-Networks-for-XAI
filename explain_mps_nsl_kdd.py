"""
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from mps import MPS
from mps_explainability import MPSExplainer

logger = logging.getLogger("explain_mps")

# Matplotlib is only needed for the figures; import lazily / headless.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


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
# Main
# ----------------------------------------------------------------------
def main(data_dir: Path) -> None:
    out_dir = data_dir / "explain"
    out_dir.mkdir(exist_ok=True)

    schema = load_schema(data_dir)
    names = feature_names(schema)
    logger.info("schema: %d features", len(names))

    mps = MPS.load(str(data_dir / "mps_trained.pt"))
    logger.info("loaded MPS: %d sites, bond_dims=%s", mps.num_sites, mps.bond_dims)

    # Reference data for the empirical baseline and the splits
    train_x, _ = load_split(data_dir, "train")
    test_x, test_meta = load_split(data_dir, "test")
    is_attack = test_meta["is_attack"]
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


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)s  %(message)s",
        datefmt="%H:%M:%S",
    )
    data_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("./nsl_kdd")
    main(data_dir)
