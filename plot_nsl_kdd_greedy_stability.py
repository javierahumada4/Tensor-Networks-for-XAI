"""Plots for NSL-KDD greedy stability controls."""

from __future__ import annotations

import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


def main():
    p=argparse.ArgumentParser()
    p.add_argument("results_dir", type=Path)
    p.add_argument("output_dir", type=Path)
    args=p.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    s=pd.read_csv(args.results_dir/"stability_summary.csv")

    fig,ax=plt.subplots(figsize=(7.2,4.8))
    for method,g in s.groupby("method"):
        ax.plot(g["budget"],g["interaction_jaccard_median"],marker="o",label=method)
    ax.set_xscale("log")
    ax.set_xlabel("Explanation budget")
    ax.set_ylabel("Median interaction Jaccard with nearest same-family attack")
    ax.set_title("NSL-KDD: local explanation stability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir/"greedy_stability_jaccard.png",dpi=220)
    fig.savefig(args.output_dir/"greedy_stability_jaccard.pdf")
    plt.close(fig)

    fig,ax=plt.subplots(figsize=(7.2,4.8))
    for method,g in s.groupby("method"):
        ax.plot(g["budget"],g["cross_fidelity_x_to_neighbor_median"],marker="o",label=method)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Explanation budget")
    ax.set_ylabel("Median cross-fidelity residual")
    ax.set_title("NSL-KDD: explanation transfer to local neighbour")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output_dir/"greedy_stability_cross_fidelity.png",dpi=220)
    fig.savefig(args.output_dir/"greedy_stability_cross_fidelity.pdf")
    plt.close(fig)


if __name__ == "__main__":
    main()
