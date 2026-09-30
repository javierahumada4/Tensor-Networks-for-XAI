"""Generate publication figures for the raw-interaction experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DATASETS = [
    "annthyroid",
    "cardio",
    "cover",
    "mammography",
    "shuttle",
    "vowels",
]


def label(dataset: str) -> str:
    return "cardio (orders <= 3)" if dataset == "cardio" else dataset


def line_kwargs(dataset: str) -> dict:
    return {"linestyle": "--"} if dataset == "cardio" else {}


def plot_fidelity(results_root: Path, output_dir: Path) -> None:
    # Presentation view: the scientifically relevant residual range.
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset in DATASETS:
        frame = pd.read_csv(results_root / dataset / "fidelity_summary.csv")
        kwargs = line_kwargs(dataset)
        ax.plot(
            frame["order"],
            frame["median_c_m"],
            label=label(dataset),
            **kwargs,
        )
        ax.fill_between(
            frame["order"],
            frame["q25_c_m"],
            frame["q75_c_m"],
            alpha=0.12,
        )
    ax.set_xlabel("Maximum interaction order m")
    ax.set_ylabel("Median relative NLL reconstruction residual c_m")
    ax.set_title("Explanation fidelity vs interaction order")
    ax.set_ylim(bottom=0.0, top=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fidelity_vs_order.png", dpi=220)
    fig.savefig(output_dir / "fidelity_vs_order.pdf")
    plt.close(fig)

    # Numerical-completeness view: shows convergence to machine precision.
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset in DATASETS:
        frame = pd.read_csv(results_root / dataset / "fidelity_summary.csv")
        residual = frame["median_c_m"].clip(lower=1e-12)
        kwargs = line_kwargs(dataset)
        ax.plot(frame["order"], residual, label=label(dataset), **kwargs)
    ax.set_yscale("log")
    ax.set_xlabel("Maximum interaction order m")
    ax.set_ylabel("Median relative NLL reconstruction residual c_m")
    ax.set_title("Explanation fidelity vs interaction order (log residual)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "fidelity_vs_order_log.png", dpi=220)
    fig.savefig(output_dir / "fidelity_vs_order_log.pdf")
    plt.close(fig)


def plot_sparsity(results_root: Path, output_dir: Path) -> None:
    # Presentation view: omit k=0 so that x can be logarithmic.
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset in DATASETS:
        frame = pd.read_csv(results_root / dataset / "sparsity_summary.csv")
        frame = frame[frame["max_interaction_budget"] > 0]
        kwargs = line_kwargs(dataset)
        ax.plot(
            frame["max_interaction_budget"],
            frame["median_c_k"],
            label=label(dataset),
            **kwargs,
        )
        ax.fill_between(
            frame["max_interaction_budget"],
            frame["q25_c_k"],
            frame["q75_c_k"],
            alpha=0.12,
        )
    ax.set_xscale("log")
    ax.set_xlabel("Maximum retained-interaction budget (2k)")
    ax.set_ylabel("Median relative NLL reconstruction residual c_k")
    ax.set_title("Explanation sparsity vs number of interactions")
    ax.set_ylim(bottom=0.0, top=1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sparsity_vs_interactions.png", dpi=220)
    fig.savefig(output_dir / "sparsity_vs_interactions.pdf")
    plt.close(fig)

    # Full numerical view.
    fig, ax = plt.subplots(figsize=(8, 5))
    for dataset in DATASETS:
        frame = pd.read_csv(results_root / dataset / "sparsity_summary.csv")
        residual = frame["median_c_k"].clip(lower=1e-12)
        kwargs = line_kwargs(dataset)
        ax.plot(
            frame["max_interaction_budget"],
            residual,
            label=label(dataset),
            **kwargs,
        )
    ax.set_yscale("log")
    ax.set_xlabel("Maximum retained-interaction budget (2k)")
    ax.set_ylabel("Median relative NLL reconstruction residual c_k")
    ax.set_title("Explanation sparsity vs number of interactions (log residual)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "sparsity_vs_interactions_log.png", dpi=220)
    fig.savefig(output_dir / "sparsity_vs_interactions_log.pdf")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("results_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_fidelity(args.results_root, args.output_dir)
    plot_sparsity(args.results_root, args.output_dir)


if __name__ == "__main__":
    main()
