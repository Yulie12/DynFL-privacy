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
        "--root", type=Path, default=ROOT / "out" / "paper_v28_controlled"
    )
    parser.add_argument("--seeds", type=int, nargs="+", default=[40, 42, 44])
    parser.add_argument(
        "--ablation-seeds",
        type=int,
        nargs="+",
        default=[40, 41, 42, 43, 44],
    )
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--edges", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "out" / "paper_v28_controlled_aggregate")
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
        model="resnet18_pretrained",
        rounds=rounds,
        clients=clients,
        edges=edges,
        selection_period=selection_period,
        privacy_budget=privacy_budget,
    )
    validate_sources(sources, seeds, policies)
    return sources, aggregate_summaries(sources, seeds, policies)


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    ablation_seeds = list(
        dict.fromkeys(int(seed) for seed in args.ablation_seeds)
    )
    output_dir = args.output_dir.resolve()
    figure_dir = args.paper_figure_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    period_rows: list[dict[str, Any]] = []
    for period in (1, 5, 10, 20):
        _sources, rows = collect_case(
            root=root / "period" / f"sp_{period}",
            seeds=seeds,
            policies=["ours"],
            rounds=args.rounds,
            clients=100,
            edges=args.edges,
            selection_period=period,
            privacy_budget=8.0,
        )
        period_rows.append({"period": period, **rows[0]})
    write_csv(output_dir / "strategy_period_statistics.csv", period_rows)
    plot_period(period_rows, figure_dir / "cifar10_strategy_period_v26.png")

    privacy_rows: list[dict[str, Any]] = []
    for budget in (1.0, 2.0, 4.0, 8.0):
        _sources, rows = collect_case(
            root=root / "privacy" / f"eps_{budget:g}",
            seeds=seeds,
            policies=["ours"],
            rounds=args.rounds,
            clients=100,
            edges=args.edges,
            selection_period=1,
            privacy_budget=budget,
        )
        privacy_rows.append({"privacy_budget": budget, **rows[0]})
    write_csv(output_dir / "privacy_budget_statistics.csv", privacy_rows)
    plot_privacy(privacy_rows, figure_dir / "cifar10_privacy_budget_v26.png")

    scale_rows: list[dict[str, Any]] = []
    for clients in (20, 50, 100):
        _sources, rows = collect_case(
            root=root / "scale" / f"clients_{clients}",
            seeds=seeds,
            policies=["ours", "individual_optimal"],
            rounds=args.rounds,
            clients=clients,
            edges=args.edges,
            selection_period=1,
            privacy_budget=8.0,
        )
        scale_rows.extend({"clients": clients, **row} for row in rows)
    write_csv(output_dir / "decision_scalability_statistics.csv", scale_rows)
    plot_scale(scale_rows, figure_dir / "cifar10_decision_scalability_v26.png")

    ablation_policies = [
        "ours",
        "individual_optimal",
        "ours_no_omega",
        "ours_fixed_liieiiic",
    ]
    ablation_sources, ablation_rows = collect_case(
        root=root / "ablation",
        seeds=ablation_seeds,
        policies=ablation_policies,
        rounds=args.rounds,
        clients=100,
        edges=args.edges,
        selection_period=1,
        privacy_budget=8.0,
    )
    write_csv(output_dir / "ablation_statistics.csv", ablation_rows)
    time_rows, plotted, _horizon = aggregate_time(
        ablation_sources,
        ablation_seeds,
        ablation_policies,
        rounds=args.rounds,
    )
    write_csv(output_dir / "ablation_wall_time_statistics.csv", time_rows)
    plot_accuracy_over_time(
        plotted,
        ablation_policies,
        figure_dir / "cifar10_ablation_wall_time_accuracy_v26.png",
        tail_fraction=0.25,
        labels={
            "ours": "Ours",
            "individual_optimal": "No Global Coordination",
            "ours_no_omega": "No Error Cost Estimate",
            "ours_fixed_liieiiic": "Fixed LIIEIIIC",
        },
    )
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
    policies = ("ours", "individual_optimal")
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
