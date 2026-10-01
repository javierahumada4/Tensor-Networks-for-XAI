"""Plot the two NSL-KDD interaction explanation experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    fidelity = pd.read_csv(args.results_dir / "fidelity_summary.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        fidelity["order"],
        fidelity["median_c_m"],
        marker="o",
        linewidth=2,
    )
    ax.fill_between(
        fidelity["order"],
        fidelity["q25_c_m"],
        fidelity["q75_c_m"],
        alpha=0.18,
    )
    ax.axhline(0.10, linestyle=":", linewidth=1)
    ax.axhline(0.05, linestyle=":", linewidth=1)
    ax.set_xticks(fidelity["order"])
    ax.set_xlabel("Maximum interaction order m")
    ax.set_ylabel("Relative NLL reconstruction residual c_m")
    ax.set_title("NSL-KDD: explanation fidelity vs interaction order")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(args.output_dir / "nsl_kdd_fidelity_vs_order.png", dpi=220)
    fig.savefig(args.output_dir / "nsl_kdd_fidelity_vs_order.pdf")
    plt.close(fig)

    sparsity = pd.read_csv(args.results_dir / "sparsity_summary.csv")
    sparsity = sparsity[sparsity["max_interaction_budget"] > 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        sparsity["median_num_interactions"],
        sparsity["median_c_k"],
        marker="o",
        linewidth=2,
    )
    ax.fill_between(
        sparsity["median_num_interactions"],
        sparsity["q25_c_k"],
        sparsity["q75_c_k"],
        alpha=0.18,
    )
    ax.axhline(0.10, linestyle=":", linewidth=1)
    ax.axhline(0.05, linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Median number of retained interactions")
    ax.set_ylabel("Relative NLL reconstruction residual c_k")
    ax.set_title("NSL-KDD: explanation sparsity vs number of interactions")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(args.output_dir / "nsl_kdd_sparsity_vs_interactions.png", dpi=220)
    fig.savefig(args.output_dir / "nsl_kdd_sparsity_vs_interactions.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
