"""Add relative 90 percent final accuracy threshold analysis to FMNIST plots."""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil

import plot_accuracy_time_analysis as base
from smooth_accuracy_time_analysis import centered_mean


def read_summary(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def compute_t90(metrics: dict[str, list[dict]]) -> dict[str, dict]:
    result: dict[str, dict] = {}
    for strategy in base.STRATEGIES:
        rows = metrics[strategy]
        accuracy = [float(row["test_accuracy"]) for row in rows]
        times = [float(row["system_time"]) for row in rows]
        final_accuracy = accuracy[-1]
        threshold = 0.9 * final_accuracy
        first_index = next(index for index, value in enumerate(accuracy) if value >= threshold)
        below_after = [
            index for index in range(first_index + 1, len(accuracy)) if accuracy[index] < threshold
        ]
        stable_index = next(
            index
            for index in range(first_index, len(accuracy))
            if all(value >= threshold for value in accuracy[index:])
        )
        result[strategy] = {
            "final_accuracy": final_accuracy,
            "A90": threshold,
            "T90_round": first_index + 1,
            "T90": times[first_index],
            "accuracy_at_T90": accuracy[first_index],
            "sustained_after_first_T90": not below_after,
            "drops_below_A90_after_first": len(below_after),
            "first_drop_below_A90_round": below_after[0] + 1 if below_after else "",
            "last_drop_below_A90_round": below_after[-1] + 1 if below_after else "",
            "stable_A90_round": stable_index + 1,
            "stable_A90_T": times[stable_index],
        }
    return result


def update_summary(path: Path, values: dict[str, dict]) -> None:
    rows = read_summary(path)
    for row in rows:
        item = values[row["strategy"]]
        row.update(
            {
                "A_final_percent": 100.0 * item["final_accuracy"],
                "A90_percent": 100.0 * item["A90"],
                "T90_first_round": item["T90_round"],
                "T90_first_time": item["T90"],
                "accuracy_at_T90_percent": 100.0 * item["accuracy_at_T90"],
                "sustained_after_first_T90": item["sustained_after_first_T90"],
                "drops_below_A90_after_first": item["drops_below_A90_after_first"],
                "first_drop_below_A90_round": item["first_drop_below_A90_round"],
                "last_drop_below_A90_round": item["last_drop_below_A90_round"],
                "stable_A90_round": item["stable_A90_round"],
                "stable_A90_T": item["stable_A90_T"],
                "T90_definition": "first raw test accuracy >= 0.9 * round 100 test accuracy",
            }
        )
    base.write_csv(path, rows)


def plot_threshold_panels(output: Path, metrics: dict[str, list[dict]], values: dict[str, dict]) -> None:
    plt = base.setup_matplotlib()
    fig, axes = plt.subplots(2, 4, figsize=(17.0, 8.8), sharex=False, sharey=False)
    for axis, strategy in zip(axes.flat, base.STRATEGIES):
        rows = metrics[strategy]
        times = [float(row["system_time"]) for row in rows]
        raw = [100.0 * float(row["test_accuracy"]) for row in rows]
        smooth = centered_mean(raw)
        item = values[strategy]
        threshold = 100.0 * item["A90"]
        index = item["T90_round"] - 1
        axis.plot(times, raw, color=base.COLORS[strategy], linewidth=0.7, alpha=0.22, label="Raw accuracy")
        axis.plot(times, smooth, color=base.COLORS[strategy], linewidth=2.0, label="5 round centered mean")
        axis.axhline(threshold, color="black", linestyle="--", linewidth=1.0, alpha=0.75)
        axis.scatter(times[index], raw[index], marker="*", s=115, color="#ffd166", edgecolors="black", linewidths=0.6, zorder=8)
        axis.annotate(
            f"First hit r{item['T90_round']}\nT90={item['T90']:.2f}s",
            (times[index], raw[index]),
            xytext=(6, 8),
            textcoords="offset points",
            fontsize=7,
            arrowprops={"arrowstyle": "-", "linewidth": 0.5, "color": "black"},
        )
        axis.text(
            0.98,
            0.04,
            f"90% Final Accuracy Threshold = {threshold:.2f}%\n"
            f"Final = {100.0 * item['final_accuracy']:.2f}%\n"
            f"Sustained after first hit: {'Yes' if item['sustained_after_first_T90'] else 'No'}",
            transform=axis.transAxes,
            ha="right",
            va="bottom",
            fontsize=7,
            bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "#777777", "alpha": 0.88},
        )
        axis.set_title(base.DISPLAY[strategy])
        axis.set_xlabel("Cumulative T (simulated seconds)")
        axis.set_ylabel("Test accuracy (%)")
        axis.grid(alpha=0.20)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    handles.append(plt.Line2D([0], [0], color="black", linestyle="--", linewidth=1.0))
    labels.append("90% Final Accuracy Threshold")
    handles.append(plt.Line2D([0], [0], marker="*", color="none", markerfacecolor="#ffd166", markeredgecolor="black", markersize=11))
    labels.append("First threshold hit")
    fig.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.965))
    fig.suptitle("FMNIST seed 42: strategy specific 90% final accuracy thresholds", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(output / "accuracy_vs_cumulative_T_T90_thresholds.png", dpi=210)
    plt.close(fig)


def plot_t90_comparison(output: Path, values: dict[str, dict]) -> None:
    plt = base.setup_matplotlib()
    ordered = sorted(base.STRATEGIES, key=lambda strategy: values[strategy]["T90"])
    times = [values[strategy]["T90"] for strategy in ordered]
    labels = [base.DISPLAY[strategy] for strategy in ordered]
    colors = [base.COLORS[strategy] for strategy in ordered]
    fig, axis = plt.subplots(figsize=(10.8, 6.2))
    bars = axis.barh(labels, times, color=colors, edgecolor="black", linewidth=0.45)
    for bar, strategy, value in zip(bars, ordered, times):
        if not values[strategy]["sustained_after_first_T90"]:
            bar.set_hatch("//")
        axis.text(
            value + max(times) * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:.2f}s  r{values[strategy]['T90_round']}",
            va="center",
            fontsize=8,
        )
    axis.invert_yaxis()
    axis.set_xlabel("T90: cumulative paper system latency (simulated seconds)")
    axis.set_title("FMNIST seed 42: time to first reach 90% of each strategy final accuracy")
    axis.grid(axis="x", alpha=0.22)
    axis.text(
        0.99,
        0.02,
        "Hatched bars fell below their own threshold after the first hit",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
        fontsize=8,
    )
    axis.set_xlim(0, max(times) * 1.18)
    fig.tight_layout()
    fig.savefig(output / "T90_strategy_comparison.png", dpi=210)
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
        output / "accuracy_vs_cumulative_T_T90_thresholds.png",
        output / "T90_strategy_comparison.png",
        output / "add_t90_accuracy_time_analysis.py",
    ]
    if any(path.exists() for path in targets):
        raise FileExistsError("refusing to overwrite an existing T90 output")
    _, metrics, _, _ = base.load_and_validate(args.root.resolve())
    values = compute_t90(metrics)
    update_summary(output / "accuracy_time_summary.csv", values)
    plot_threshold_panels(output, metrics, values)
    plot_t90_comparison(output, values)
    shutil.copyfile(Path(__file__), output / "add_t90_accuracy_time_analysis.py")


if __name__ == "__main__":
    main()
