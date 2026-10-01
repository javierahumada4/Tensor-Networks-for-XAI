"""Publication-style plots for the NSL-KDD follow-up experiments."""

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

    order = pd.read_csv(args.results_dir / "order4_fidelity_summary.csv")
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(order["order"], order["median_c_m"], marker="o", linewidth=2)
    ax.fill_between(
        order["order"], order["q25_c_m"], order["q75_c_m"], alpha=0.18
    )
    ax.axhline(0.10, linestyle=":", linewidth=1)
    ax.axhline(0.05, linestyle=":", linewidth=1)
    ax.set_xticks(order["order"])
    ax.set_xlabel("Maximum interaction order m")
    ax.set_ylabel("Relative NLL reconstruction residual c_m")
    ax.set_title("NSL-KDD: fidelity through interaction order 4")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(args.output_dir / "nsl_kdd_fidelity_order4.png", dpi=220)
    fig.savefig(args.output_dir / "nsl_kdd_fidelity_order4.pdf")
    plt.close(fig)

    greedy = pd.read_csv(args.results_dir / "greedy_sparsity_summary.csv")
    greedy = greedy[greedy["interaction_budget"] > 0]
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.plot(
        greedy["interaction_budget"],
        greedy["median_c_k"],
        marker="o",
        linewidth=2,
    )
    ax.fill_between(
        greedy["interaction_budget"],
        greedy["q25_c_k"],
        greedy["q75_c_k"],
        alpha=0.18,
    )
    ax.axhline(0.10, linestyle=":", linewidth=1)
    ax.axhline(0.05, linestyle=":", linewidth=1)
    ax.axhline(0.01, linestyle=":", linewidth=1)
    ax.set_xscale("log")
    ax.set_xlabel("Residual-aware greedy interaction budget")
    ax.set_ylabel("Relative NLL reconstruction residual c_k")
    ax.set_title("NSL-KDD: greedy explanation sparsity")
    ax.set_ylim(bottom=0)
    fig.tight_layout()
    fig.savefig(args.output_dir / "nsl_kdd_greedy_sparsity.png", dpi=220)
    fig.savefig(args.output_dir / "nsl_kdd_greedy_sparsity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
