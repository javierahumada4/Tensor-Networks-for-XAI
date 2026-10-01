"""Run raw-interaction explanation experiments on the validated NSL-KDD MPS.

This is the NSL-KDD counterpart of run_interaction_experiments.py. It consumes
legacy TFG encoder artifacts so the explainability experiment uses exactly the
same 40-site representation as the successful D=16 regression model.

For 40 features a full 2^40-1 decomposition is intractable. The experiment
therefore computes every subset through a declared maximum order (order 3 in
CI: 10,700 interactions per sample), without sampling interaction subsets.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
import torch

from mps import MPS
from mps_interactions import interaction_count, order_fidelity_curve, raw_interactions


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def stable_threshold(curve: list[dict], key: str, threshold: float, x_key: str):
    values = [float(row[key]) for row in curve]
    for index in range(len(values)):
        if all(value <= threshold for value in values[index:]):
            return curve[index][x_key]
    return None


def fast_sparsity_curve(
    anomaly_score: float,
    interactions: dict[tuple[int, ...], float],
    *,
    eps: float = 1e-12,
) -> list[dict]:
    """O(n log n) top-k positive + top-k negative sparsity curve.

    The repository's original sparsity_curve recomputes prefix sums for every k,
    which is quadratic in the number of interactions. NSL-KDD has 10,700 terms
    through order 3, so we compute the same curve using cumulative sums.
    """
    score = float(anomaly_score)
    denominator = max(abs(score), eps)

    positive = np.asarray(
        sorted((float(v) for v in interactions.values() if v > 0.0), reverse=True),
        dtype=np.float64,
    )
    negative = np.asarray(
        sorted(float(v) for v in interactions.values() if v < 0.0),
        dtype=np.float64,
    )

    pos_prefix = np.concatenate(([0.0], np.cumsum(positive, dtype=np.float64)))
    neg_prefix = np.concatenate(([0.0], np.cumsum(negative, dtype=np.float64)))
    max_k = max(len(positive), len(negative))

    curve = []
    for k in range(max_k + 1):
        kp = min(k, len(positive))
        kn = min(k, len(negative))
        reconstruction = float(pos_prefix[kp] + neg_prefix[kn])
        residual = score - reconstruction
        curve.append(
            {
                "k_per_sign": k,
                "num_interactions": kp + kn,
                "positive_terms": kp,
                "negative_terms": kn,
                "reconstruction": reconstruction,
                "signed_residual": residual,
                "c_k": abs(residual) / denominator,
            }
        )
    return curve


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--max-samples", type=int, default=100)
    parser.add_argument("--max-order", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    args = parser.parse_args()

    if args.max_samples < 1:
        raise ValueError("max_samples must be >= 1")
    if args.num_shards < 1:
        raise ValueError("num_shards must be >= 1")
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("shard-index must lie in [0, num-shards)")

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
        raise ValueError(
            f"expected the 40-site NSL-KDD representation, got "
            f"model={model.num_sites}, test={test_x.shape[1]}"
        )
    if list(schema["physical_dims"]) != list(model.physical_dims):
        raise ValueError("model physical dimensions disagree with TFG encoding schema")
    if args.max_order < 1 or args.max_order > model.num_sites:
        raise ValueError("invalid max-order")

    anomaly_positions = torch.nonzero(
        test_meta["is_attack"] == 1, as_tuple=False
    ).flatten().numpy()
    rng = np.random.default_rng(args.seed)
    sample_count = min(args.max_samples, len(anomaly_positions))
    if sample_count == len(anomaly_positions):
        selected_all = np.sort(anomaly_positions)
    else:
        selected_all = np.sort(
            rng.choice(anomaly_positions, size=sample_count, replace=False)
        )

    selected = selected_all[args.shard_index :: args.num_shards]
    family_names = list(test_meta["family_names"])
    family_code = test_meta["family_code"].numpy().astype(np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fidelity_rows: list[dict] = []
    sparsity_rows: list[dict] = []
    sample_rows: list[dict] = []

    for local_index, test_position in enumerate(selected, start=1):
        x = test_x[int(test_position)]
        score = float(model.anomaly_score(x.unsqueeze(0))[0].item())
        family = family_names[int(family_code[int(test_position)])]

        interactions = raw_interactions(model, x, max_order=args.max_order)
        fidelity = order_fidelity_curve(
            score, interactions, max_order=args.max_order
        )
        sparsity = fast_sparsity_curve(score, interactions)

        global_sample_index = int(
            np.searchsorted(selected_all, test_position) + 1
        )

        common = {
            "sample_index": global_sample_index,
            "test_position": int(test_position),
            "family": family,
        }
        for row in fidelity:
            fidelity_rows.append({**common, **row})
        for row in sparsity:
            sparsity_rows.append({**common, **row})

        sample_rows.append(
            {
                **common,
                "anomaly_score": score,
                "max_order": args.max_order,
                "num_interactions_computed": len(interactions),
                "positive_interactions": sum(v > 0 for v in interactions.values()),
                "negative_interactions": sum(v < 0 for v in interactions.values()),
                "c_at_max_order": fidelity[-1]["c_m"],
                "stable_order_10pct": stable_threshold(
                    fidelity, "c_m", 0.10, "order"
                ),
                "stable_order_5pct": stable_threshold(
                    fidelity, "c_m", 0.05, "order"
                ),
                "stable_terms_10pct": stable_threshold(
                    sparsity, "c_k", 0.10, "num_interactions"
                ),
                "stable_terms_5pct": stable_threshold(
                    sparsity, "c_k", 0.05, "num_interactions"
                ),
            }
        )

        print(
            f"[shard {args.shard_index} {local_index}/{len(selected)}] "
            f"test_position={test_position} family={family} score={score:.6f} "
            f"interactions={len(interactions)} c_mmax={fidelity[-1]['c_m']:.6g}",
            flush=True,
        )

    write_csv(args.output_dir / "samples.csv", sample_rows)
    write_csv(args.output_dir / "fidelity_per_sample.csv", fidelity_rows)
    write_csv(args.output_dir / "sparsity_per_sample.csv", sparsity_rows)

    metadata = {
        "dataset": "NSL-KDD",
        "representation": "legacy TFG 40-site encoder",
        "seed": args.seed,
        "sampling": "uniform without replacement from labeled KDDTest+ attacks",
        "requested_samples": args.max_samples,
        "samples_in_shard": len(selected),
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "max_order": args.max_order,
        "num_features": model.num_sites,
        "interaction_subsets_per_sample": interaction_count(
            model.num_sites, args.max_order
        ),
        "full_decomposition": args.max_order == model.num_sites,
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
