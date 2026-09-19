from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from aggregate_multiseed_results import (
    POLICY_LABELS,
    ROOT,
    aggregate_privacy_trajectory,
    discover_sources,
    validate_sources,
    write_csv,
)

FORMAL_PRIVACY_POLICIES = ("dynamic_mode_fixed_privacy", "full_dynfl")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the frozen Q96 Dynamic Privacy vs Fixed Privacy study."
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT / "out" / "paper_v31_final" / "fig3_privacy"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 42, 44])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--edges", type=int, default=10)
    parser.add_argument("--selection-period", type=int, default=1)
    parser.add_argument("--privacy-budget", type=float, default=8.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "out" / "paper_v31_final" / "fig3_privacy_aggregate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    policies = list(FORMAL_PRIVACY_POLICIES)
    sources = discover_sources(
        root=args.root.resolve(),
        seeds=seeds,
        policies=policies,
        dataset="cifar10",
        model="resnet18_pretrained_head",
        rounds=args.rounds,
        clients=args.clients,
        edges=args.edges,
        selection_period=args.selection_period,
        privacy_budget=args.privacy_budget,
    )
    validate_sources(sources, seeds, policies)
    rows = aggregate_privacy_trajectory(sources, seeds, policies, rounds=args.rounds)

    output_dir = args.output_dir.resolve()
    write_csv(output_dir / "privacy_trajectory.csv", rows)
    plot_privacy_trajectory(rows, output_dir / "fig3_dynamic_vs_fixed_privacy.png")
    print(output_dir / "privacy_trajectory.csv")


def plot_privacy_trajectory(rows: list[dict[str, object]], output: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 4.2), sharex=True)
    for policy in FORMAL_PRIVACY_POLICIES:
        selected = [row for row in rows if row["policy"] == policy]
        xs = [float(row["max_update_epsilon_mean"]) for row in selected]
        accuracy = [100.0 * float(row["test_accuracy_mean"]) for row in selected]
        latency = [float(row["accounted_system_time_sec_mean"]) for row in selected]
        label = POLICY_LABELS[policy]
        axes[0].plot(xs, accuracy, linewidth=1.35, label=label)
        axes[1].plot(xs, latency, linewidth=1.35, label=label)
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_ylabel("System latency (s)")
    axes[1].set_xlabel("Realized update-level privacy consumption (epsilon)")
    for axis in axes:
        axis.grid(True, alpha=0.25, linewidth=0.55)
        axis.legend(frameon=False, fontsize=7)
        axis.tick_params(labelsize=7.4, pad=2)
    fig.tight_layout(pad=0.55)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
