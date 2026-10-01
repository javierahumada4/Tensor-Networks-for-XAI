"""NSL-KDD follow-up: explanation fidelity through interaction order 4."""

from __future__ import annotations

import argparse
import csv
import json
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


def select_followup_samples(
    anomaly_positions: np.ndarray,
    *,
    seed: int,
    base_count: int = 100,
    followup_count: int = 20,
) -> np.ndarray:
    """Choose 20 anomalies as an unbiased deterministic subset of the prior 100."""
    rng = np.random.default_rng(seed)
    base_count = min(base_count, len(anomaly_positions))
    base = np.sort(rng.choice(anomaly_positions, size=base_count, replace=False))
    followup_count = min(followup_count, len(base))
    sub_rng = np.random.default_rng(seed + 1)
    return np.sort(sub_rng.choice(base, size=followup_count, replace=False))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir", type=Path)
    parser.add_argument("model_path", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--seed", type=int, default=123)
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
    selected_all = select_followup_samples(
        anomaly_positions, seed=args.seed, base_count=100, followup_count=20
    )
    if not 0 <= args.shard_index < args.num_shards:
        raise ValueError("invalid shard index")
    selected = selected_all[args.shard_index :: args.num_shards]

    family_names = list(test_meta["family_names"])
    family_code = test_meta["family_code"].numpy().astype(np.int64)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    fidelity_rows: list[dict] = []
    sample_rows: list[dict] = []

    for local_index, test_position in enumerate(selected, start=1):
        x = test_x[int(test_position)]
        score = float(model.anomaly_score(x.unsqueeze(0))[0].item())
        family = family_names[int(family_code[int(test_position)])]

        interactions = raw_interactions(model, x, max_order=4)
        fidelity = order_fidelity_curve(score, interactions, max_order=4)
        sample_index = int(np.searchsorted(selected_all, test_position) + 1)

        common = {
            "sample_index": sample_index,
            "test_position": int(test_position),
            "family": family,
        }
        for row in fidelity:
            fidelity_rows.append({**common, **row})

        sample_rows.append({
            **common,
            "anomaly_score": score,
            "num_interactions_computed": len(interactions),
            "c1": fidelity[0]["c_m"],
            "c2": fidelity[1]["c_m"],
            "c3": fidelity[2]["c_m"],
            "c4": fidelity[3]["c_m"],
        })

        print(
            f"[order4 shard {args.shard_index} {local_index}/{len(selected)}] "
            f"test_position={test_position} family={family} "
            f"c1={fidelity[0]['c_m']:.6g} c2={fidelity[1]['c_m']:.6g} "
            f"c3={fidelity[2]['c_m']:.6g} c4={fidelity[3]['c_m']:.6g}",
            flush=True,
        )

    write_csv(args.output_dir / "samples.csv", sample_rows)
    write_csv(args.output_dir / "fidelity_per_sample.csv", fidelity_rows)
    (args.output_dir / "metadata.json").write_text(
        json.dumps({
            "dataset": "NSL-KDD",
            "seed": args.seed,
            "selection": (
                "20 anomalies sampled with seed+1 from the exact 100-anomaly "
                "seed-123 population used by the order<=3 experiment"
            ),
            "sample_count_total": 20,
            "samples_in_shard": len(selected),
            "shard_index": args.shard_index,
            "num_shards": args.num_shards,
            "num_features": 40,
            "max_order": 4,
            "interaction_subsets_per_sample": interaction_count(40, 4),
            "full_decomposition": False,
        }, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
