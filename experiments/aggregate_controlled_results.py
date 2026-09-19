from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from aggregate_multiseed_results import (
    POLICY_COLORS,
    POLICY_LABELS,
    ROOT,
    aggregate_summaries,
    aggregate_time,
    discover_sources,
    plot_accuracy_over_time,
    validate_sources,
    write_csv,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Aggregate the versioned controlled paper experiments."
    )
    parser.add_argument(
        "--root", type=Path, default=ROOT / "out" / "paper_v31_final" / "fig5_sp"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 42, 44])
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--edges", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out" / "paper_v31_final" / "sensitivity_aggregate")
    parser.add_argument("--paper-figure-dir", type=Path, default=ROOT / "tex" / "paper" / "figures")
    return parser.parse_args()


def collect_case(
    *,
    root: Path,
    seeds: list[int],
    policies: list[str],
    rounds: int,
    clients: int,
    edges: int,
    selection_period: int,
    privacy_budget: float,
) -> tuple[dict[tuple[int, str], Any], list[dict[str, Any]]]:
    sources = discover_sources(
        root=root,
        seeds=seeds,
        policies=policies,
        dataset="cifar10",
        model="resnet18_pretrained_head",
        rounds=rounds,
        clients=clients,
        edges=edges,
        selection_period=selection_period,
        privacy_budget=privacy_budget,
    )
    validate_sources(sources, seeds, policies)
    return sources, aggregate_summaries(sources, seeds, policies)


def main() -> None:
    """Aggregate only the frozen Q94 S_P sensitivity study."""
    args = parse_args()
    root = args.root.resolve()
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    output_dir = args.output_dir.resolve()
    figure_dir = args.paper_figure_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    period_rows: list[dict[str, Any]] = []
    for period in (1, 5, 10):
        _sources, rows = collect_case(
            root=root / f"sp_{period}",
            seeds=seeds,
            policies=["full_dynfl"],
            rounds=args.rounds,
            clients=100,
            edges=args.edges,
            selection_period=period,
            privacy_budget=8.0,
        )
        period_rows.append({"period": period, **rows[0]})
    write_csv(output_dir / "strategy_period_statistics.csv", period_rows)
    plot_period(period_rows, figure_dir / "fig5_strategy_period_sensitivity.png")
    print(output_dir)


def plot_period(rows: list[dict[str, Any]], output: Path) -> None:
    xs = [int(row["period"]) for row in rows]
    fig, axes = plt.subplots(3, 1, figsize=(3.45, 5.0), sharex=True)
    _error_series(
        axes[0], xs, rows, "final_test_accuracy", "Final", percent=True
    )
    _error_series(
        axes[0], xs, rows, "avg_last_10_accuracy", "Last 10", percent=True
    )
    _error_series(
        axes[1], xs, rows, "total_selection_wall_time_sec", "Decision"
    )
    _error_series(
        axes[2], xs, rows, "accounted_system_time_sec", "System"
    )
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_ylabel("Decision time (s)")
    axes[2].set_ylabel("System time (s)")
    axes[2].set_xlabel("Strategy update period")
    axes[0].legend(frameon=False, fontsize=7)
    _finish_small_figure(fig, axes, output)


def plot_privacy(rows: list[dict[str, Any]], output: Path) -> None:
    xs = [float(row["privacy_budget"]) for row in rows]
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.8), sharex=True)
    _error_series(
        axes[0], xs, rows, "final_test_accuracy", "Final", percent=True
    )
    _error_series(
        axes[0], xs, rows, "avg_last_10_accuracy", "Last 10", percent=True
    )
    _error_series(axes[1], xs, rows, "max_update_epsilon", "Update DP")
    axes[0].set_ylabel("Accuracy (%)")
    axes[1].set_ylabel("Realized privacy loss")
    axes[1].set_xlabel("Total privacy target")
    for axis in axes:
        axis.legend(frameon=False, fontsize=7)
    _finish_small_figure(fig, axes, output)


def plot_scale(rows: list[dict[str, Any]], output: Path) -> None:
    policies = ("full_dynfl",)
    fig, axes = plt.subplots(2, 1, figsize=(3.45, 3.8), sharex=True)
    for policy in policies:
        policy_rows = sorted(
            (row for row in rows if row["policy"] == policy),
            key=lambda row: int(row["clients"]),
        )
        xs = [int(row["clients"]) for row in policy_rows]
        _error_series(
            axes[0],
            xs,
            policy_rows,
            "total_selection_wall_time_sec",
            POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )
        _error_series(
            axes[1],
            xs,
            policy_rows,
            "accounted_system_time_sec",
            POLICY_LABELS[policy],
            color=POLICY_COLORS[policy],
        )
    axes[0].set_ylabel("Decision time (s)")
    axes[1].set_ylabel("System time (s)")
    axes[1].set_xlabel("Number of clients")
    for axis in axes:
        axis.legend(frameon=False, fontsize=7)
    _finish_small_figure(fig, axes, output)


def _error_series(
    axis: Any,
    xs: list[float] | list[int],
    rows: list[dict[str, Any]],
    metric: str,
    label: str,
    *,
    percent: bool = False,
    color: str | None = None,
) -> None:
    scale = 100.0 if percent else 1.0
    means = [float(row[f"{metric}_mean"]) * scale for row in rows]
    lower = [
        (float(row[f"{metric}_mean"]) - float(row[f"{metric}_ci95_low"])) * scale
        for row in rows
    ]
    axis.errorbar(
        xs,
        means,
        yerr=lower,
        marker="o",
        markersize=3.5,
        linewidth=1.35,
        capsize=2.2,
        label=label,
        color=color,
    )


def _finish_small_figure(fig: Any, axes: Any, output: Path) -> None:
    for axis in axes:
        axis.grid(True, alpha=0.25, linewidth=0.55)
        axis.tick_params(labelsize=7.4, pad=2)
        axis.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=5))
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5))
    fig.tight_layout(pad=0.55)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    main()
