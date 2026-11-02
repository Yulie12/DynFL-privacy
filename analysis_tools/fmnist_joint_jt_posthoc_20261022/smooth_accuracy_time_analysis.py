"""Add five round centered moving average views to an existing analysis."""
from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import plot_accuracy_time_analysis as base


WINDOW = 5


def centered_mean(values: list[float], window: int = WINDOW) -> list[float]:
    radius = window // 2
    return [
        sum(values[max(0, index - radius) : min(len(values), index + radius + 1)])
        / len(values[max(0, index - radius) : min(len(values), index + radius + 1)])
        for index in range(len(values))
    ]


def ensure_new(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise FileExistsError("refusing to overwrite smoothed outputs: " + ", ".join(existing))


def plot_all_time(output: Path, metrics: dict[str, list[dict]], summaries: list[dict]) -> None:
    plt = base.setup_matplotlib()
    common_budget = min(row["final_T"] for row in summaries)
    fig = plt.figure(figsize=(13.2, 8.3))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.35], hspace=0.30)
    common = fig.add_subplot(grid[0])
    full = fig.add_subplot(grid[1])
    for strategy in base.STRATEGIES:
        rows = metrics[strategy]
        times = [float(row["system_time"]) for row in rows]
        raw = [100.0 * float(row["test_accuracy"]) for row in rows]
        smooth = centered_mean(raw)
        plot_style = base.style(strategy)
        common.plot(times, raw, color=base.COLORS[strategy], linewidth=0.55, alpha=0.13, zorder=0)
        common.plot(times, smooth, label=base.DISPLAY[strategy], **plot_style)
        full.plot(times, raw, color=base.COLORS[strategy], linewidth=0.45, alpha=0.10, zorder=0)
        full.plot(times, smooth, label=base.DISPLAY[strategy], **plot_style)
        full.scatter(times[-1], raw[-1], s=42, color=base.COLORS[strategy], edgecolors="black", linewidths=0.45, zorder=7)
    common.set_xlim(0.0, common_budget)
    common.axvline(common_budget, color="black", linewidth=1.0, linestyle=":")
    common.text(
        common_budget,
        common.get_ylim()[1],
        f" Common budget {common_budget:.3f}s ",
        ha="right",
        va="top",
        fontsize=8,
        backgroundcolor="white",
    )
    common.set_title("Common latency budget: five round centered moving average")
    common.set_ylabel("Test accuracy (%)")
    common.grid(alpha=0.20)
    common.legend(ncol=4, loc="lower right")
    full.set_title("Full range with true raw final points")
    full.set_xlabel("Cumulative paper system latency T (simulated seconds)")
    full.set_ylabel("Test accuracy (%)")
    full.grid(alpha=0.20)
    full.set_xlim(0.0, max(row["final_T"] for row in summaries) + 42.0)
    endpoint_rows = sorted(summaries, key=lambda row: row["final_accuracy_percent"])
    endpoint_slots = [12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 49.0, 57.0]
    label_x = max(row["final_T"] for row in summaries) + 1.5
    for row, slot in zip(endpoint_rows, endpoint_slots):
        full.annotate(
            f"{base.DISPLAY[row['strategy']]}  {row['final_accuracy_percent']:.2f}% @ {row['final_T']:.2f}s",
            (row["final_T"], row["final_accuracy_percent"]),
            xytext=(label_x, slot),
            textcoords="data",
            fontsize=6.5,
            color=base.COLORS[row["strategy"]],
            arrowprops={"arrowstyle": "-", "color": base.COLORS[row["strategy"]], "linewidth": 0.55},
            va="center",
        )
    fig.suptitle("FMNIST seed 42: smoothed accuracy over cumulative paper T", fontsize=15)
    fig.subplots_adjust(top=0.93, left=0.07, right=0.98, bottom=0.08)
    fig.savefig(output / "accuracy_vs_cumulative_T_all_strategies_smoothed.png", dpi=210)
    plt.close(fig)


def plot_all_round(output: Path, metrics: dict[str, list[dict]]) -> None:
    plt = base.setup_matplotlib()
    fig, axis = plt.subplots(figsize=(12.4, 7.0))
    for strategy in base.STRATEGIES:
        rows = metrics[strategy]
        rounds = [int(row["round"]) for row in rows]
        raw = [100.0 * float(row["test_accuracy"]) for row in rows]
        smooth = centered_mean(raw)
        axis.plot(rounds, raw, color=base.COLORS[strategy], linewidth=0.55, alpha=0.12, zorder=0)
        axis.plot(rounds, smooth, label=base.DISPLAY[strategy], **base.style(strategy))
        axis.scatter(rounds[-1], raw[-1], s=34, color=base.COLORS[strategy], edgecolors="white", linewidths=0.5, zorder=7)
    axis.set(
        xlabel="Round",
        ylabel="Test accuracy (%)",
        title="FMNIST seed 42: five round centered moving average",
    )
    axis.grid(alpha=0.20)
    axis.legend(ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "accuracy_vs_round_all_strategies_smoothed.png", dpi=210)
    plt.close(fig)


def plot_p1_p2(output: Path, metrics: dict[str, list[dict]], summaries: list[dict]) -> None:
    plt = base.setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    for strategy in ("fixed_P1", "fixed_P2"):
        rows = metrics[strategy]
        rounds = [int(row["round"]) for row in rows]
        times = [float(row["system_time"]) for row in rows]
        raw = [100.0 * float(row["test_accuracy"]) for row in rows]
        smooth = centered_mean(raw)
        summary = next(row for row in summaries if row["strategy"] == strategy)
        for axis, xvalues in ((axes[0], times), (axes[1], rounds)):
            axis.plot(xvalues, raw, color=base.COLORS[strategy], linewidth=0.8, alpha=0.18)
            axis.plot(xvalues, smooth, label=base.DISPLAY[strategy], **base.style(strategy))
            axis.scatter(xvalues[-1], raw[-1], marker="X", s=85, color=base.COLORS[strategy], edgecolors="black", linewidths=0.5, zorder=8)
        axes[1].scatter(
            summary["best_accuracy_round"],
            summary["best_accuracy_percent"],
            marker="*",
            s=150,
            color=base.COLORS[strategy],
            edgecolors="black",
            linewidths=0.5,
            zorder=8,
        )
        note = (
            f"{base.DISPLAY[strategy]} raw metrics\n"
            f"Final {summary['final_accuracy_percent']:.2f}%\n"
            f"Best {summary['best_accuracy_percent']:.2f}% at r{summary['best_accuracy_round']}\n"
            f"Last 10 mean {summary['last10_accuracy_percent']:.2f}%\n"
            f"Final T {summary['final_T']:.3f}s"
        )
        axes[0].text(
            0.02 if strategy == "fixed_P1" else 0.69,
            0.98,
            note,
            transform=axes[0].transAxes,
            va="top",
            fontsize=8,
            color=base.COLORS[strategy],
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": base.COLORS[strategy], "alpha": 0.88},
        )
    axes[0].set(xlabel="Cumulative paper system latency T (simulated seconds)", ylabel="Test accuracy (%)", title="Smoothed accuracy versus cumulative simulation time")
    axes[1].set(xlabel="Round", ylabel="Test accuracy (%)", title="Smoothed accuracy versus round")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(loc="lower right")
    fig.suptitle("Fixed P1 and Fixed P2: five round centered moving average", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "fixed_P1_vs_fixed_P2_accuracy_time_smoothed.png", dpi=210)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_dir():
        raise FileNotFoundError(output)
    targets = [
        output / "accuracy_vs_cumulative_T_all_strategies_smoothed.png",
        output / "fixed_P1_vs_fixed_P2_accuracy_time_smoothed.png",
        output / "accuracy_vs_round_all_strategies_smoothed.png",
        output / "smooth_accuracy_time_analysis.py",
    ]
    ensure_new(targets)
    _, metrics, summaries, _ = base.load_and_validate(args.root.resolve())
    plot_all_time(output, metrics, summaries)
    plot_p1_p2(output, metrics, summaries)
    plot_all_round(output, metrics)
    shutil.copyfile(Path(__file__), output / "smooth_accuracy_time_analysis.py")


if __name__ == "__main__":
    main()
