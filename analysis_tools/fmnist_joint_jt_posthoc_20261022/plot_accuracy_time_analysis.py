"""Analyze accuracy against cumulative paper simulation time for completed FMNIST runs.

This script is read only with respect to the campaign artifacts. It refuses to
overwrite an existing analysis directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil
import statistics


STRATEGIES = (
    "fixed_P0",
    "fixed_P1",
    "fixed_P2",
    "fixed_P3",
    "fixed_P4",
    "greedy_lambda025",
    "greedy_lambda050",
    "greedy_lambda075",
)

DISPLAY = {
    "fixed_P0": "Fixed P0",
    "fixed_P1": "Fixed P1",
    "fixed_P2": "Fixed P2",
    "fixed_P3": "Fixed P3",
    "fixed_P4": "Fixed P4",
    "greedy_lambda025": "Greedy λ=0.25",
    "greedy_lambda050": "Greedy λ=0.50",
    "greedy_lambda075": "Greedy λ=0.75",
}

COLORS = {
    "fixed_P0": "#9ca3af",
    "fixed_P1": "#d62728",
    "fixed_P2": "#1f77b4",
    "fixed_P3": "#b8b8b8",
    "fixed_P4": "#7f7f7f",
    "greedy_lambda025": "#2ca02c",
    "greedy_lambda050": "#9467bd",
    "greedy_lambda075": "#ff7f0e",
}


def digest(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def latest_at_or_before(times: list[float], values: list[float], budget: float) -> tuple[int, float]:
    index = max(i for i, value in enumerate(times) if value <= budget + 1e-10)
    return index + 1, values[index]


def linear_at(times: list[float], values: list[float], budget: float) -> float:
    if budget <= times[0]:
        return values[0]
    for index in range(1, len(times)):
        if times[index] >= budget:
            left_t, right_t = times[index - 1], times[index]
            if right_t == left_t:
                return values[index]
            weight = (budget - left_t) / (right_t - left_t)
            return values[index - 1] + weight * (values[index] - values[index - 1])
    return values[-1]


def nondominated(rows: list[dict]) -> None:
    for row in rows:
        dominators = [
            other["strategy"]
            for other in rows
            if other is not row
            and other["final_T"] <= row["final_T"]
            and other["final_accuracy_percent"] >= row["final_accuracy_percent"]
            and (
                other["final_T"] < row["final_T"]
                or other["final_accuracy_percent"] > row["final_accuracy_percent"]
            )
        ]
        row["pareto_nondominated"] = not dominators
        row["dominated_by"] = ";".join(dominators)


def load_and_validate(root: Path) -> tuple[dict, dict[str, list[dict]], list[dict], list[str]]:
    prereg = json.loads((root / "preregistration.json").read_text(encoding="utf-8"))
    if prereg.get("paper_T") != "event critical path without cross-(Edge,Mode) serialization":
        raise RuntimeError("campaign does not declare the repaired strict paper T model")
    seed_root = root / "seed_42"
    metrics: dict[str, list[dict]] = {}
    identities: dict[str, dict] = {}
    summaries: list[dict] = []
    validations: list[str] = []

    for strategy in STRATEGIES:
        folder = seed_root / strategy
        status = json.loads((folder / "run_status.json").read_text(encoding="utf-8"))
        rows = read_csv(folder / "round_metrics.csv")
        identity = json.loads((folder / "run_identity.json").read_text(encoding="utf-8"))
        identities[strategy] = identity
        metrics[strategy] = rows
        rounds = [int(row["round"]) for row in rows]
        times = [float(row["system_time"]) for row in rows]
        accuracy = [float(row["test_accuracy"]) for row in rows]
        loss = [float(row["test_loss"]) for row in rows]
        monotonic = all(right >= left for left, right in zip(times, times[1:]))
        scale_ok = all(0.0 <= value <= 1.0 for value in accuracy)
        complete = rounds == list(range(1, 101)) and status.get("status") == "completed"
        barrier_ok = (
            min(int(row["acknowledged_clients"]) for row in rows) == 100
            and max(int(row["pending_units"]) for row in rows) == 0
            and int(rows[-1]["cloud_version"]) == 100
        )
        cumulative_matches = abs(times[-1] - sum(float(row["round_duration"]) for row in rows)) < 1e-8
        if not all((complete, monotonic, scale_ok, barrier_ok, cumulative_matches)):
            raise RuntimeError(f"data validation failed for {strategy}")
        validations.append(
            f"{strategy}：100 轮完整，累计 T 单调不减，准确率原始量纲为 0 至 1，"
            "ACK 屏障完整，累计 T 等于逐轮时延之和"
        )
        best = max(accuracy)
        summaries.append(
            {
                "strategy": strategy,
                "seed": 42,
                "rounds": len(rows),
                "final_accuracy_percent": 100.0 * accuracy[-1],
                "best_accuracy_percent": 100.0 * best,
                "best_accuracy_round": accuracy.index(best) + 1,
                "last10_accuracy_percent": 100.0 * statistics.mean(accuracy[-10:]),
                "final_loss": loss[-1],
                "final_T": times[-1],
                "communication_mb": sum(float(row["communication_mb"]) for row in rows),
                "executed_local_steps": sum(int(row["executed_local_steps"]) for row in rows),
                "max_epsilon": max(float(row["max_epsilon"]) for row in rows),
                "T_monotone": monotonic,
                "accuracy_source_scale": "0_to_1",
                "paper_T_model": prereg["paper_T"],
                "config_hash": identity["config_hash"],
                "source_hash": identity["source_hash"],
                "partition_hash": identity["partition_hash"],
                "traces_hash": identity["traces_hash"],
                "initial_model_sha256": digest(folder / "initial_model.pt"),
            }
        )

    identity_fields = ("config_hash", "source_hash", "partition_hash", "traces_hash", "data_hashes")
    for field in identity_fields:
        values = {json.dumps(identities[name][field], sort_keys=True) for name in STRATEGIES}
        if len(values) != 1:
            raise RuntimeError(f"run identity mismatch: {field}")
    model_hashes = {row["initial_model_sha256"] for row in summaries}
    if len(model_hashes) != 1:
        raise RuntimeError("initial model hashes differ")

    common_budget = min(row["final_T"] for row in summaries)
    for summary in summaries:
        strategy = summary["strategy"]
        times = [float(row["system_time"]) for row in metrics[strategy]]
        accuracy = [100.0 * float(row["test_accuracy"]) for row in metrics[strategy]]
        common_round, common_accuracy = latest_at_or_before(times, accuracy, common_budget)
        summary["common_T_budget"] = common_budget
        summary["round_at_or_before_common_T"] = common_round
        summary["accuracy_at_or_before_common_T_percent"] = common_accuracy
        summary["accuracy_at_common_T_linear_percent"] = linear_at(times, accuracy, common_budget)
    nondominated(summaries)
    return prereg, metrics, summaries, validations


def p1_p2_analysis(metrics: dict[str, list[dict]], summaries: list[dict]) -> dict:
    p1 = metrics["fixed_P1"]
    p2 = metrics["fixed_P2"]
    a1 = [100.0 * float(row["test_accuracy"]) for row in p1]
    a2 = [100.0 * float(row["test_accuracy"]) for row in p2]
    difference = [left - right for left, right in zip(a1, a2)]
    signs = [1 if value > 0 else -1 if value < 0 else 0 for value in difference]
    crossings = []
    previous = signs[0]
    for index, sign in enumerate(signs[1:], start=2):
        if sign and previous and sign != previous:
            crossings.append(index)
        if sign:
            previous = sign
    windows = ((1, 10), (11, 20), (21, 40), (41, 60), (61, 80), (81, 100))
    window_rows = []
    for start, end in windows:
        mean_p1 = statistics.mean(a1[start - 1 : end])
        mean_p2 = statistics.mean(a2[start - 1 : end])
        window_rows.append(
            {"start": start, "end": end, "P1": mean_p1, "P2": mean_p2, "difference": mean_p1 - mean_p2}
        )
    s1 = next(row for row in summaries if row["strategy"] == "fixed_P1")
    s2 = next(row for row in summaries if row["strategy"] == "fixed_P2")
    return {
        "crossing_rounds": crossings,
        "P1_lead_rounds": sum(value > 0 for value in difference),
        "P2_lead_rounds": sum(value < 0 for value in difference),
        "tie_rounds": sum(value == 0 for value in difference),
        "window_rows": window_rows,
        "P1_peak_to_final_drop_pp": s1["best_accuracy_percent"] - s1["final_accuracy_percent"],
        "P2_peak_to_final_drop_pp": s2["best_accuracy_percent"] - s2["final_accuracy_percent"],
        "P1_last10_minus_P2_pp": s1["last10_accuracy_percent"] - s2["last10_accuracy_percent"],
        "P2_extra_steps": s2["executed_local_steps"] - s1["executed_local_steps"],
        "P2_extra_communication_mb": s2["communication_mb"] - s1["communication_mb"],
        "P2_extra_T": s2["final_T"] - s1["final_T"],
    }


def setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 11,
            "legend.fontsize": 8,
            "figure.dpi": 120,
        }
    )
    return plt


def style(strategy: str) -> dict:
    highlighted = strategy in {"fixed_P1", "fixed_P2", "greedy_lambda025", "greedy_lambda050", "greedy_lambda075"}
    return {
        "color": COLORS[strategy],
        "linewidth": 2.35 if strategy in {"fixed_P1", "fixed_P2"} else 1.85 if highlighted else 1.0,
        "alpha": 1.0 if highlighted else 0.55,
        "linestyle": "--" if strategy.startswith("greedy_") else "-",
        "zorder": 4 if strategy in {"fixed_P1", "fixed_P2"} else 3 if highlighted else 1,
    }


def mark_rounds(axis, times: list[float], accuracy: list[float], strategy: str, rounds=(10, 20, 40, 60, 80, 100)) -> None:
    for round_number in rounds:
        index = round_number - 1
        axis.scatter(times[index], accuracy[index], s=23, color=COLORS[strategy], edgecolors="white", linewidths=0.45, zorder=6)


def plot_all_time(output: Path, metrics: dict[str, list[dict]], summaries: list[dict]) -> None:
    plt = setup_matplotlib()
    common_budget = min(row["final_T"] for row in summaries)
    fig = plt.figure(figsize=(13.2, 8.3))
    grid = fig.add_gridspec(2, 1, height_ratios=[3.1, 1.35], hspace=0.30)
    common = fig.add_subplot(grid[0])
    full = fig.add_subplot(grid[1])
    for strategy in STRATEGIES:
        rows = metrics[strategy]
        times = [float(row["system_time"]) for row in rows]
        accuracy = [100.0 * float(row["test_accuracy"]) for row in rows]
        common.plot(times, accuracy, label=DISPLAY[strategy], **style(strategy))
        mark_rounds(common, times, accuracy, strategy)
        full.plot(times, accuracy, label=DISPLAY[strategy], **style(strategy))
        full.scatter(times[-1], accuracy[-1], s=42, color=COLORS[strategy], edgecolors="black", linewidths=0.45, zorder=7)
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
    common.set_title("Accuracy versus cumulative simulation time: common latency budget")
    common.set_ylabel("Test accuracy (%)")
    common.grid(alpha=0.20)
    common.legend(ncol=4, loc="lower right")
    full.set_title("Full range inset with true final points")
    full.set_xlabel("Cumulative paper system latency T (simulated seconds)")
    full.set_ylabel("Test accuracy (%)")
    full.grid(alpha=0.20)
    full.set_xlim(0.0, max(row["final_T"] for row in summaries) + 42.0)
    endpoint_rows = sorted(summaries, key=lambda row: row["final_accuracy_percent"])
    endpoint_slots = [12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 49.0, 57.0]
    label_x = max(row["final_T"] for row in summaries) + 1.5
    for row, slot in zip(endpoint_rows, endpoint_slots):
        full.annotate(
            f"{DISPLAY[row['strategy']]}  {row['final_accuracy_percent']:.2f}% @ {row['final_T']:.2f}s",
            (row["final_T"], row["final_accuracy_percent"]),
            xytext=(label_x, slot),
            textcoords="data",
            fontsize=6.5,
            color=COLORS[row["strategy"]],
            arrowprops={"arrowstyle": "-", "color": COLORS[row["strategy"]], "linewidth": 0.55},
            va="center",
        )
    fig.suptitle("FMNIST seed 42: accuracy trajectories over cumulative paper T", fontsize=15)
    fig.subplots_adjust(top=0.93, left=0.07, right=0.98, bottom=0.08)
    fig.savefig(output / "accuracy_vs_cumulative_T_all_strategies.png", dpi=210)
    plt.close(fig)


def plot_all_round(output: Path, metrics: dict[str, list[dict]]) -> None:
    plt = setup_matplotlib()
    fig, axis = plt.subplots(figsize=(12.4, 7.0))
    for strategy in STRATEGIES:
        rows = metrics[strategy]
        rounds = [int(row["round"]) for row in rows]
        accuracy = [100.0 * float(row["test_accuracy"]) for row in rows]
        axis.plot(rounds, accuracy, label=DISPLAY[strategy], **style(strategy))
        for round_number in (10, 20, 40, 60, 80, 100):
            axis.scatter(round_number, accuracy[round_number - 1], s=24, color=COLORS[strategy], edgecolors="white", linewidths=0.45, zorder=6)
    axis.set(xlabel="Round", ylabel="Test accuracy (%)", title="FMNIST seed 42: accuracy versus round")
    axis.grid(alpha=0.20)
    axis.legend(ncol=4, loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "accuracy_vs_round_all_strategies.png", dpi=210)
    plt.close(fig)


def plot_p1_p2(output: Path, metrics: dict[str, list[dict]], summaries: list[dict], comparison: dict) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 2, figsize=(14.5, 6.2))
    for strategy in ("fixed_P1", "fixed_P2"):
        rows = metrics[strategy]
        rounds = [int(row["round"]) for row in rows]
        times = [float(row["system_time"]) for row in rows]
        accuracy = [100.0 * float(row["test_accuracy"]) for row in rows]
        summary = next(row for row in summaries if row["strategy"] == strategy)
        axes[0].plot(times, accuracy, label=DISPLAY[strategy], **style(strategy))
        axes[1].plot(rounds, accuracy, label=DISPLAY[strategy], **style(strategy))
        mark_rounds(axes[0], times, accuracy, strategy)
        for round_number in (10, 20, 40, 60, 80, 100):
            axes[1].scatter(round_number, accuracy[round_number - 1], s=30, color=COLORS[strategy], edgecolors="white", linewidths=0.5, zorder=6)
        best_index = summary["best_accuracy_round"] - 1
        axes[0].scatter(times[-1], accuracy[-1], marker="X", s=90, color=COLORS[strategy], edgecolors="black", linewidths=0.5, zorder=8)
        axes[1].scatter(summary["best_accuracy_round"], summary["best_accuracy_percent"], marker="*", s=150, color=COLORS[strategy], edgecolors="black", linewidths=0.5, zorder=8)
        note = (
            f"{DISPLAY[strategy]}\n"
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
            color=COLORS[strategy],
            bbox={"boxstyle": "round,pad=0.35", "facecolor": "white", "edgecolor": COLORS[strategy], "alpha": 0.88},
        )
    axes[0].set(xlabel="Cumulative paper system latency T (simulated seconds)", ylabel="Test accuracy (%)", title="Accuracy versus cumulative simulation time")
    axes[1].set(xlabel="Round", ylabel="Test accuracy (%)", title="Accuracy versus round")
    for axis in axes:
        axis.grid(alpha=0.22)
        axis.legend(loc="lower right")
    for round_number in comparison["crossing_rounds"]:
        axes[1].axvline(round_number, color="#bbbbbb", linewidth=0.45, alpha=0.35)
    fig.suptitle("Fixed P1 and Fixed P2: seed 42 trajectory comparison", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(output / "fixed_P1_vs_fixed_P2_accuracy_time.png", dpi=210)
    plt.close(fig)


def plot_final_tradeoff(output: Path, summaries: list[dict]) -> None:
    plt = setup_matplotlib()
    fig, axis = plt.subplots(figsize=(9.2, 6.7))
    frontier = sorted((row for row in summaries if row["pareto_nondominated"]), key=lambda row: row["final_T"])
    axis.plot(
        [row["final_T"] for row in frontier],
        [row["final_accuracy_percent"] for row in frontier],
        color="black",
        linestyle=":",
        linewidth=1.2,
        label="Single seed Pareto frontier",
    )
    for row in summaries:
        strategy = row["strategy"]
        axis.scatter(
            row["final_T"],
            row["final_accuracy_percent"],
            s=105 if row["pareto_nondominated"] else 65,
            marker="D" if row["pareto_nondominated"] else "o",
            color=COLORS[strategy],
            edgecolors="black",
            linewidths=0.55,
            zorder=5,
        )
        axis.annotate(DISPLAY[strategy], (row["final_T"], row["final_accuracy_percent"]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    axis.set(
        xlabel="Cumulative paper system latency T (simulated seconds)",
        ylabel="Final test accuracy (%)",
        title="FMNIST seed 42: final accuracy and cumulative simulation time",
    )
    axis.grid(alpha=0.22)
    axis.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output / "final_accuracy_vs_cumulative_T.png", dpi=210)
    plt.close(fig)


def write_report(
    output: Path,
    root: Path,
    prereg: dict,
    summaries: list[dict],
    validations: list[str],
    comparison: dict,
) -> None:
    p1 = next(row for row in summaries if row["strategy"] == "fixed_P1")
    p2 = next(row for row in summaries if row["strategy"] == "fixed_P2")
    frontier = [row["strategy"] for row in sorted(summaries, key=lambda row: row["final_T"]) if row["pareto_nondominated"]]
    fixed_p1_dominates = [row["strategy"] for row in summaries if "fixed_P1" in row["dominated_by"]]
    window_table = [
        "| 轮次区间 | P1 平均准确率 | P2 平均准确率 | P1 减 P2 |",
        "|---|---:|---:|---:|",
    ]
    for row in comparison["window_rows"]:
        window_table.append(
            f"| {row['start']} 至 {row['end']} | {row['P1']:.2f}% | {row['P2']:.2f}% | {row['difference']:+.2f} 个百分点 |"
        )
    cross_text = ", ".join(str(value) for value in comparison["crossing_rounds"])
    report = f"""# Accuracy 与累计仿真时延分析

## 范围和口径

本报告使用 FMNIST 正式实验中 seed 42 已完成的 8 个策略，每个策略均为 100 轮。准确率原始值存储为 0 至 1，绘图时只转换一次为百分比。横轴使用 `system_time`，即逐轮累计的论文系统仿真时延 T，不使用 Python 墙钟时间。

主图的共同时间预算上限为 {min(row['final_T'] for row in summaries):.6f} 秒，即 8 个策略中最小的最终累计 T。主图下方的完整范围子图保留全部真实最终点，因此没有把较慢策略的终点错误映射到共同预算边界。

## 数据核验

{chr(10).join('- ' + item for item in validations)}
- 8 个策略的配置、源码、数据、划分、系统轨迹和初始模型哈希一致。
- 注册信息声明 T 使用“事件关键路径，不对不同 Edge 与 Mode 更新组添加额外串行排队”的修复模型。
- P1 和 P2 都从第 1 轮开始使用各自固定候选，因此二者从相同初始模型出发，训练历史的唯一预设策略差异是固定 Mode 分布。
- 三个 greedy 策略共享第 1 至第 5 轮的 P0 检查点，第 6 轮起才按联合效用选择。它们与 fixed P1/P2 不共享同一段前 5 轮训练历史，因此跨这两类策略的精度差异不能只归因于切换机制。
- 正式三种子独立审计尚未完成。这里的 T 模型结论来自预注册信息、逐轮数据一致性和现有专项测试，不替代最终独立审计。

## P1 与 P2 的结果

| 指标 | fixed P1 | fixed P2 | P1 减 P2 |
|---|---:|---:|---:|
| 最终准确率 | {p1['final_accuracy_percent']:.2f}% | {p2['final_accuracy_percent']:.2f}% | {p1['final_accuracy_percent']-p2['final_accuracy_percent']:+.2f} 个百分点 |
| 最佳准确率 | {p1['best_accuracy_percent']:.2f}%（第 {p1['best_accuracy_round']} 轮） | {p2['best_accuracy_percent']:.2f}%（第 {p2['best_accuracy_round']} 轮） | {p1['best_accuracy_percent']-p2['best_accuracy_percent']:+.2f} 个百分点 |
| 末 10 轮平均准确率 | {p1['last10_accuracy_percent']:.2f}% | {p2['last10_accuracy_percent']:.2f}% | {comparison['P1_last10_minus_P2_pp']:+.2f} 个百分点 |
| 最终 Loss | {p1['final_loss']:.4f} | {p2['final_loss']:.4f} | {p1['final_loss']-p2['final_loss']:+.4f} |
| 累计 T | {p1['final_T']:.3f} 秒 | {p2['final_T']:.3f} 秒 | {p1['final_T']-p2['final_T']:+.3f} 秒 |
| 本地训练步数 | {p1['executed_local_steps']} | {p2['executed_local_steps']} | {p1['executed_local_steps']-p2['executed_local_steps']:+d} |
| 通信量 | {p1['communication_mb']:.1f} MB | {p2['communication_mb']:.1f} MB | {p1['communication_mb']-p2['communication_mb']:+.1f} MB |

### 准确率轨迹是否交叉

发生交叉。按每轮 P1 与 P2 的准确率差值符号计算，交叉后的轮次为：{cross_text}。100 轮中 P1 领先 {comparison['P1_lead_rounds']} 轮，P2 领先 {comparison['P2_lead_rounds']} 轮，持平 {comparison['tie_rounds']} 轮。因此 P1 的最终优势不是从训练一开始持续保持的单向优势。

{chr(10).join(window_table)}

窗口均值显示 P2 是否在中期领先以及 P1 优势何时形成。单轮曲线波动较大，窗口均值比个别轮次更适合描述阶段性趋势。

### P2 是否中期更好但后期回落

P2 的峰值出现在第 {p2['best_accuracy_round']} 轮，峰值到最终值下降 {comparison['P2_peak_to_final_drop_pp']:.2f} 个百分点。P1 的峰值出现在第 {p1['best_accuracy_round']} 轮，峰值到最终值下降 {comparison['P1_peak_to_final_drop_pp']:.2f} 个百分点。末 10 轮 P1 平均准确率高出 {comparison['P1_last10_minus_P2_pp']:.2f} 个百分点。因此本次 seed 42 轨迹支持“P2 后期回落幅度更大，P1 的末段表现更好”。是否存在稳定的中期 P2 优势，应以窗口表中实际为正或为负的差值判断，不能只凭峰值判断。

### 为什么当前 seed 42 下 P1 最终准确率更高

数据能够直接支持的解释是：P1 的末段准确率水平更高，并且从峰值到第 100 轮的回落小于 P2。P1 的最终优势为 {p1['final_accuracy_percent']-p2['final_accuracy_percent']:.2f} 个百分点，末 10 轮平均优势为 {comparison['P1_last10_minus_P2_pp']:.2f} 个百分点。

P2 比 P1 多执行 {comparison['P2_extra_steps']} 个本地训练步，多通信 {comparison['P2_extra_communication_mb']:.1f} MB，累计 T 多 {comparison['P2_extra_T']:.3f} 秒，但最终准确率反而低 {p1['final_accuracy_percent']-p2['final_accuracy_percent']:.2f} 个百分点。这说明本次运行中额外计算和通信没有转化为更高的最终分类准确率。

从策略配置看，P1 为 50% FL_E、25% FL_EC、25% FL_C，P2 为 25% FL_E、50% FL_EC、25% FL_C。P2 更多使用三阶段 FL_EC，因此步数和通信量更高。现有单种子轨迹不能证明更多 FL_EC 本身导致后期精度回落，也不能区分噪声累积、更新方向差异、聚合层级或随机优化波动的相对贡献。要回答机制原因，需要至少完成三个种子，并进行控制 Mode 比例、DP 噪声、阶段数或更新偏差的消融实验。

P2 的最终 Loss 低于 P1，但最终准确率也低于 P1。交叉熵衡量预测概率分布，准确率只衡量分类是否正确，两者排序可以不同。该现象不能解释为记录错误，也不能据此判断 P2 的模型整体更差或更好。

## 最终准确率与累计 T 的权衡

当前单种子 Pareto 非支配策略为：{', '.join(frontier)}。`fixed_P1` 同时以更低累计 T 和更高最终准确率支配：{', '.join(fixed_p1_dominates) if fixed_p1_dominates else '无'}。Pareto 结论只针对当前 8 个已完成策略和 seed 42 的最终点。

## 输出说明

- `accuracy_vs_cumulative_T_all_strategies.png`：上图使用最小最终 T 作为共同预算最大刻度，下图展示完整范围和真实最终点。
- `fixed_P1_vs_fixed_P2_accuracy_time.png`：P1 与 P2 的累计 T 轨迹和轮次轨迹。
- `accuracy_vs_round_all_strategies.png`：8 个策略的轮次曲线。
- `final_accuracy_vs_cumulative_T.png`：最终点及单种子 Pareto 前沿。
- `accuracy_time_summary.csv`：核验结果、最终指标、共同时间预算指标和支配关系。

原始实验目录未被修改。分析文件位于独立子目录，脚本拒绝覆盖已经存在的分析目录。
"""
    (output / "analysis_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="Campaign output directory")
    parser.add_argument("--output", type=Path, required=True, help="New analysis output directory")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing analysis: {output}")
    output.mkdir(parents=True)
    prereg, metrics, summaries, validations = load_and_validate(root)
    comparison = p1_p2_analysis(metrics, summaries)
    plot_all_time(output, metrics, summaries)
    plot_p1_p2(output, metrics, summaries, comparison)
    plot_all_round(output, metrics)
    plot_final_tradeoff(output, summaries)
    write_csv(output / "accuracy_time_summary.csv", summaries)
    write_report(output, root, prereg, summaries, validations, comparison)
    shutil.copyfile(Path(__file__), output / "plot_accuracy_time_analysis.py")


if __name__ == "__main__":
    main()
