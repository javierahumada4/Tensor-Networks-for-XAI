"""NSL-KDD follow-up: residual-aware greedy sparse interaction explanations."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
from pathlib import Path

import numpy as np
import torch

from mps import MPS
from mps_interactions import interaction_count, raw_interactions


CHECKPOINTS = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000]


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


def _pop_nearest(
    values: list[float],
    subsets: list[tuple[int, ...]],
    target: float,
):
    if not values:
        return None
    index = bisect.bisect_left(values, target)
    candidates = []
    if index < len(values):
        candidates.append(index)
    if index > 0:
        candidates.append(index - 1)
    best = min(candidates, key=lambda i: abs(target - values[i]))
    value = values[best]
    subset = subsets[best]
    del values[best]
    del subsets[best]
    return value, subset


def greedy_residual_curve(
    anomaly_score: float,
    interactions: dict[tuple[int, ...], float],
    *,
    max_terms: int = 2000,
    eps: float = 1e-12,
) -> tuple[list[dict], str]:
    """Greedily add the remaining interaction that most reduces |residual|.

    Because the target is scalar, an opposite-sign interaction can never reduce
    the current absolute residual. We therefore search the remaining terms with
    the same sign as the residual and choose the value nearest to that residual.
    The procedure stops when no single remaining term improves fidelity.
    """
    score = float(anomaly_score)
    denominator = max(abs(score), eps)

    positive = sorted(
        (float(v), subset) for subset, v in interactions.items() if v > 0.0
    )
    negative = sorted(
        (float(v), subset) for subset, v in interactions.items() if v < 0.0
    )
    pos_values = [v for v, _ in positive]
    pos_subsets = [s for _, s in positive]
    neg_values = [v for v, _ in negative]
    neg_subsets = [s for _, s in negative]

    reconstruction = 0.0
    residual = score
    rows = [{
        "num_interactions": 0,
        "selected_order": 0,
        "selected_subset": "",
        "selected_value": 0.0,
        "reconstruction": reconstruction,
        "signed_residual": residual,
        "c_k": abs(residual) / denominator,
    }]

    stop_reason = "max_terms"
    for step in range(1, max_terms + 1):
        if abs(residual) <= eps:
            stop_reason = "numerical_zero"
            break

        if residual > 0:
            picked = _pop_nearest(pos_values, pos_subsets, residual)
        else:
            picked = _pop_nearest(neg_values, neg_subsets, residual)

        if picked is None:
            stop_reason = "no_same_sign_terms"
            break

        value, subset = picked
        new_residual = residual - value

        if abs(new_residual) >= abs(residual) - 1e-15:
            stop_reason = "no_single_term_improves_residual"
            break

        reconstruction += value
        residual = new_residual
        rows.append({
            "num_interactions": step,
            "selected_order": len(subset),
            "selected_subset": ",".join(str(i) for i in subset),
            "selected_value": value,
            "reconstruction": reconstruction,
            "signed_residual": residual,
            "c_k": abs(residual) / denominator,
        })
    else:
        stop_reason = "max_terms"

    return rows, stop_reason


def first_terms_at(curve: list[dict], threshold: float):
    for row in curve:
        if float(row["c_k"]) <= threshold:
            return int(row["num_interactions"])
    return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--max-terms", type=int, default=2000)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=10)
    args = parser.parse_args()

    test_x = torch.load(
        args.data_dir / "test_X.pt", map_location="cpu", weights_only=True
    ).long()
    test_meta = torch.load(
        args.data_dir / "test_meta.pt", map_location="cpu", weights_only=True
    )
    schema = json.loads((args.data_dir / "encoding_schema.json").read_text())

    model = MPS.load(str(args.model_path), map_location="cpu")
    model.eval()
    if model.num_sites != 40 or test_x.shape[1] != 40:
        raise ValueError("expected the 40-site NSL-KDD representation")
    if list(schema["physical_dims"]) != list(model.physical_dims):
        raise ValueError("model and encoder physical dimensions disagree")

    anomaly_positions = torch.nonzero(
        test_meta["is_attack"] == 1, as_tuple=False
    ).flatten().numpy()
    selected_all = select_prior_100(anomaly_positions, args.seed)
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    selected = selected_all[args.shard_index :: args.num_shards]

    family_names = list(test_meta["family_names"])
    family_code = test_meta["family_code"].numpy().astype(np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    curve_rows: list[dict] = []
    sample_rows: list[dict] = []

    for local_index, test_position in enumerate(selected, start=1):
        x = test_x[int(test_position)]
        score = float(model.anomaly_score(x.unsqueeze(0))[0].item())
        family = family_names[int(family_code[int(test_position)])]
        interactions = raw_interactions(model, x, max_order=3)
        curve, stop_reason = greedy_residual_curve(
            score, interactions, max_terms=args.max_terms
        )

        sample_index = int(np.searchsorted(selected_all, test_position) + 1)
        common = {
            "sample_index": sample_index,
            "test_position": int(test_position),
            "family": family,
        }
        for row in curve:
            curve_rows.append({**common, **row})

        final = curve[-1]
        sample_rows.append({
            **common,
            "anomaly_score": score,
            "num_interactions_available": len(interactions),
            "terms_selected": int(final["num_interactions"]),
            "final_c_k": float(final["c_k"]),
            "terms_to_10pct": first_terms_at(curve, 0.10),
            "terms_to_5pct": first_terms_at(curve, 0.05),
            "terms_to_1pct": first_terms_at(curve, 0.01),
            "stop_reason": stop_reason,
        })

        print(
            f"[greedy shard {args.shard_index} {local_index}/{len(selected)}] "
            f"test_position={test_position} family={family} "
            f"terms={final['num_interactions']} final_c={final['c_k']:.6g} "
            f"stop={stop_reason}",
            flush=True,
        )

    write_csv(args.output_dir / "samples.csv", sample_rows)
    write_csv(args.output_dir / "greedy_per_sample.csv", curve_rows)
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "dataset": "NSL-KDD",
            "seed": args.seed,
            "sample_count_total": 100,
            "samples_in_shard": len(selected),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "interaction_order_cap": 3,
            "interactions_available_per_sample": interaction_count(40, 3),
            "max_greedy_terms": args.max_terms,
            "selection_rule": (
                "At each step choose the remaining same-sign interaction whose "
                "addition minimizes the absolute residual; stop if no single "
                "term strictly improves the residual."
            ),
            "checkpoints": CHECKPOINTS,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
