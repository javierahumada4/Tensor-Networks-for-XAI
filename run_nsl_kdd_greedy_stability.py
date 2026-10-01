"""Controls for residual-aware greedy explanations on NSL-KDD.

Controls:
1. Compare greedy against top-|I| and random selection at equal budgets.
2. Local stability against the nearest non-identical attack from the same family.
3. Cross-fidelity: transfer x's selected subsets to its local neighbour.
4. Target-shuffle: ask x's interaction pool to reconstruct another anomaly's score.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from mps import MPS
from mps_interactions import raw_interactions


BUDGETS = [1, 2, 5, 10, 20]
THRESHOLDS = [0.10, 0.05, 0.01]


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def select_prior_100(anomaly_positions: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    n = min(100, len(anomaly_positions))
    return np.sort(rng.choice(anomaly_positions, size=n, replace=False))


def pop_nearest(values, subsets, target):
    if not values:
        return None
    idx = bisect.bisect_left(values, target)
    candidates = []
    if idx < len(values):
        candidates.append(idx)
    if idx > 0:
        candidates.append(idx - 1)
    best = min(candidates, key=lambda i: abs(target - values[i]))
    value, subset = values[best], subsets[best]
    del values[best]
    del subsets[best]
    return value, subset


def greedy_order(target: float, interactions: dict, max_terms: int = 50):
    positive = sorted((float(v), s) for s, v in interactions.items() if v > 0)
    negative = sorted((float(v), s) for s, v in interactions.items() if v < 0)
    pv, ps = [v for v, _ in positive], [s for _, s in positive]
    nv, ns = [v for v, _ in negative], [s for _, s in negative]

    residual = float(target)
    selected = []
    residuals = [abs(residual) / max(abs(target), 1e-12)]
    for _ in range(max_terms):
        picked = pop_nearest(pv, ps, residual) if residual > 0 else pop_nearest(nv, ns, residual)
        if picked is None:
            break
        value, subset = picked
        new_residual = residual - value
        if abs(new_residual) >= abs(residual) - 1e-15:
            break
        selected.append(subset)
        residual = new_residual
        residuals.append(abs(residual) / max(abs(target), 1e-12))
        if abs(residual) <= 1e-12:
            break
    return selected, residuals


def top_abs_order(interactions: dict):
    return [
        s for s, _ in sorted(
            interactions.items(),
            key=lambda kv: (-abs(float(kv[1])), kv[0]),
        )
    ]


def random_order(interactions: dict, seed: int):
    subsets = list(interactions)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(subsets))
    return [subsets[int(i)] for i in perm]


def fidelity(score: float, interactions: dict, selected) -> float:
    recon = math.fsum(float(interactions[s]) for s in selected)
    return abs(float(score) - recon) / max(abs(float(score)), 1e-12)


def jaccard(a, b) -> float:
    a, b = set(a), set(b)
    if not a and not b:
        return 1.0
    return len(a & b) / len(a | b)


def feature_set(subsets):
    return {i for subset in subsets for i in subset}


def sign_agreement(ix: dict, iy: dict, sx, sy):
    common = set(sx) & set(sy)
    if not common:
        return float("nan")
    return sum(np.sign(ix[s]) == np.sign(iy[s]) for s in common) / len(common)


def first_at(residuals, threshold):
    for terms, value in enumerate(residuals):
        if value <= threshold:
            return terms
    return None


def nearest_same_family(
    test_x: torch.Tensor,
    family_code: np.ndarray,
    position: int,
    attack_positions: np.ndarray,
):
    fam = family_code[position]
    candidates = attack_positions[
        (family_code[attack_positions] == fam) & (attack_positions != position)
    ]
    if len(candidates) == 0:
        raise RuntimeError(f"no same-family neighbour for test position {position}")
    x = test_x[position]
    d = (test_x[candidates] != x).sum(dim=1).numpy()
    positive = d > 0
    if positive.any():
        candidates, d = candidates[positive], d[positive]
    min_d = int(d.min())
    tied = candidates[d == min_d]
    return int(tied.min()), min_d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("data_dir", type=Path)
    p.add_argument("model_path", type=Path)
    p.add_argument("output_dir", type=Path)
    p.add_argument("--seed", type=int, default=123)
    p.add_argument("--shard-index", type=int, default=0)
    p.add_argument("--num-shards", type=int, default=10)
    args = p.parse_args()

    test_x = torch.load(args.data_dir / "test_X.pt", map_location="cpu", weights_only=True).long()
    meta = torch.load(args.data_dir / "test_meta.pt", map_location="cpu", weights_only=True)
    schema = json.loads((args.data_dir / "encoding_schema.json").read_text())
    model = MPS.load(str(args.model_path), map_location="cpu")
    model.eval()
    if model.num_sites != 40 or list(schema["physical_dims"]) != list(model.physical_dims):
        raise ValueError("unexpected NSL-KDD representation")

    family_names = list(meta["family_names"])
    family_code = meta["family_code"].numpy().astype(np.int64)
    attacks = torch.nonzero(meta["is_attack"] == 1, as_tuple=False).flatten().numpy()
    selected_all = select_prior_100(attacks, args.seed)
    selected = selected_all[args.shard_index :: args.num_shards]

    all_scores = model.anomaly_score(test_x[selected_all]).detach().cpu().numpy()
    shuffled_target_by_position = {
        int(pos): float(all_scores[(i + 37) % len(selected_all)])
        for i, pos in enumerate(selected_all)
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, shuffle_rows = [], []
    cache = {}

    def explain(position):
        if position not in cache:
            x = test_x[position]
            score = float(model.anomaly_score(x.unsqueeze(0))[0].item())
            cache[position] = (score, raw_interactions(model, x, max_order=3))
        return cache[position]

    for local_i, pos in enumerate(selected, 1):
        pos = int(pos)
        score_x, ix = explain(pos)
        nei, hamming = nearest_same_family(test_x, family_code, pos, attacks)
        score_y, iy = explain(nei)
        family = family_names[int(family_code[pos])]

        gx, gx_res = greedy_order(score_x, ix)
        gy, gy_res = greedy_order(score_y, iy)
        tx, ty = top_abs_order(ix), top_abs_order(iy)
        rx = random_order(ix, args.seed * 100000 + pos)
        ry = random_order(iy, args.seed * 100000 + nei + 17)

        methods = {
            "greedy": (gx, gy),
            "top_abs": (tx, ty),
            "random": (rx, ry),
        }
        sample_index = int(np.searchsorted(selected_all, pos) + 1)

        for method, (ox, oy) in methods.items():
            for budget in BUDGETS:
                sx, sy = ox[:budget], oy[:budget]
                rows.append({
                    "sample_index": sample_index,
                    "test_position": pos,
                    "neighbor_position": nei,
                    "family": family,
                    "hamming_distance": hamming,
                    "score_x": score_x,
                    "score_neighbor": score_y,
                    "score_relative_difference": abs(score_x-score_y) / max(abs(score_x), 1e-12),
                    "method": method,
                    "budget": budget,
                    "terms_x": len(sx),
                    "terms_neighbor": len(sy),
                    "own_fidelity_x": fidelity(score_x, ix, sx),
                    "own_fidelity_neighbor": fidelity(score_y, iy, sy),
                    "interaction_jaccard": jaccard(sx, sy),
                    "feature_jaccard": jaccard(feature_set(sx), feature_set(sy)),
                    "sign_agreement_shared": sign_agreement(ix, iy, sx, sy),
                    "cross_fidelity_x_to_neighbor": fidelity(score_y, iy, sx),
                    "cross_fidelity_neighbor_to_x": fidelity(score_x, ix, sy),
                })

        shuffled_target = shuffled_target_by_position[pos]
        _, shuffled_res = greedy_order(shuffled_target, ix)
        shuffle_row = {
            "sample_index": sample_index,
            "test_position": pos,
            "family": family,
            "true_score": score_x,
            "shuffled_target": shuffled_target,
            "target_relative_difference": abs(score_x-shuffled_target) / max(abs(score_x), 1e-12),
        }
        for threshold in THRESHOLDS:
            tag = str(int(threshold * 100))
            shuffle_row[f"true_terms_to_{tag}pct"] = first_at(gx_res, threshold)
            shuffle_row[f"shuffle_terms_to_{tag}pct"] = first_at(shuffled_res, threshold)
        shuffle_row["true_final_c"] = gx_res[-1]
        shuffle_row["shuffle_final_c"] = shuffled_res[-1]
        shuffle_row["true_terms_selected"] = len(gx)
        shuffle_row["shuffle_terms_selected"] = len(shuffled_res)-1
        shuffle_rows.append(shuffle_row)

        print(
            f"[stability {args.shard_index} {local_i}/{len(selected)}] "
            f"pos={pos} neighbor={nei} d={hamming} family={family} "
            f"greedy_terms={len(gx)} shuffle_terms={len(shuffled_res)-1}",
            flush=True,
        )

    write_csv(args.output_dir / "stability_controls.csv", rows)
    write_csv(args.output_dir / "target_shuffle.csv", shuffle_rows)
    (args.output_dir / "metadata.json").write_text(json.dumps({
        "dataset": "NSL-KDD",
        "seed": args.seed,
        "sample_count_total": 100,
        "interaction_pool": "all subsets through order 3 (10,700 per sample)",
        "budgets": BUDGETS,
        "local_neighbor": "nearest non-identical labeled attack in same attack family under encoded Hamming distance",
        "methods": ["greedy", "top_abs", "random"],
        "target_shuffle": "deterministic cyclic offset 37 over the same 100 anomaly scores",
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
    }, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
