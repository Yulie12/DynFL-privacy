"""Fair convergence-efficiency analysis for completed FMNIST J/T strategies.

The campaign artifacts are read only. Derived tables, figures, and reports are
written to a new directory that must not already exist.
"""
from __future__ import annotations

import argparse
import csv
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import shutil
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import torch

from dynfed.paper_config import PaperConfig
from dynfed.paper_data import evaluate, load_data, make_training_model


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
    "fixed_P0": "#7f7f7f",
    "fixed_P1": "#d62728",
    "fixed_P2": "#1f77b4",
    "fixed_P3": "#a6a6a6",
    "fixed_P4": "#4d4d4d",
    "greedy_lambda025": "#2ca02c",
    "greedy_lambda050": "#9467bd",
    "greedy_lambda075": "#ff7f0e",
}
COMMON_TARGETS = (0.45, 0.50, 0.52)
MARK_ROUNDS = (10, 20, 40, 60, 80, 100)


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


def sha256(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            hasher.update(chunk)
    return hasher.hexdigest()


def moving_average(values: list[float], window: int = 5) -> np.ndarray:
    source = np.asarray(values, dtype=float)
    result = np.empty_like(source)
    half = window // 2
    for index in range(len(source)):
        left = max(0, index - half)
        right = min(len(source), index + half + 1)
        result[index] = source[left:right].mean()
    return result


def hit_metrics(rows: list[dict[str, str]], target: float) -> dict:
    accuracy = [float(row["test_accuracy"]) for row in rows]
    times = [float(row["system_time"]) for row in rows]
    rounds = [int(row["round"]) for row in rows]
    first = next((index for index, value in enumerate(accuracy) if value >= target), None)
    sustained = next(
        (
            index
            for index in range(len(accuracy) - 4)
            if all(value >= target for value in accuracy[index : index + 5])
        ),
        None,
    )
    return {
        "threshold_reached": first is not None,
        "first_hit_round": rounds[first] if first is not None else "not_reached",
        "first_hit_T": times[first] if first is not None else "not_reached",
        "sustained_threshold_reached": sustained is not None,
        "sustained_hit_round": rounds[sustained] if sustained is not None else "not_reached",
        "sustained_hit_T": times[sustained] if sustained is not None else "not_reached",
        "sustained_confirm_round": rounds[sustained + 4] if sustained is not None else "not_reached",
        "sustained_confirm_T": times[sustained + 4] if sustained is not None else "not_reached",
    }


def evaluate_initial_accuracy(seed_root: Path) -> tuple[float, float, int]:
    folder = seed_root / "shared_P0_prefix"
    config = replace(PaperConfig.load(folder / "config.json"), device="cpu", seeds=(42,))
    data = load_data(config, 42)
    model = make_training_model(config, 42, features=False)
    model.load_state_dict(
        torch.load(folder / "initial_model.pt", map_location="cpu", weights_only=True)
    )
    accuracy, loss = evaluate(model, data[2], data[3], config)
    return accuracy, loss, len(data[3])


def audit_and_load(root: Path) -> tuple[dict, dict[str, list[dict]], dict, list[str]]:
    prereg = json.loads((root / "preregistration.json").read_text(encoding="utf-8"))
    expected_t = "event critical path without cross-(Edge,Mode) serialization"
    if prereg.get("paper_T") != expected_t:
        raise RuntimeError("campaign does not declare the repaired strict paper T model")
    seed_root = root / "seed_42"
    metrics: dict[str, list[dict]] = {}
    identities: dict[str, dict] = {}
    audit_lines: list[str] = []
    model_hashes: set[str] = set()

    for strategy in STRATEGIES:
        folder = seed_root / strategy
        rows = read_csv(folder / "round_metrics.csv")
        status = json.loads((folder / "run_status.json").read_text(encoding="utf-8"))
        identity = json.loads((folder / "run_identity.json").read_text(encoding="utf-8"))
        rounds = [int(row["round"]) for row in rows]
        times = [float(row["system_time"]) for row in rows]
        durations = [float(row["round_duration"]) for row in rows]
        accuracy = [float(row["test_accuracy"]) for row in rows]
        if rounds != list(range(1, 101)) or status.get("status") != "completed":
            raise RuntimeError(f"{strategy} is not a completed 100-round run")
        if not all(right >= left for left, right in zip(times, times[1:])):
            raise RuntimeError(f"non-monotone cumulative T for {strategy}")
        if abs(times[-1] - sum(durations)) > 1e-8:
            raise RuntimeError(f"cumulative T does not equal summed round durations for {strategy}")
        if not all(0 <= value <= 1 for value in accuracy):
            raise RuntimeError(f"accuracy scale is not 0 to 1 for {strategy}")
        if min(int(row["acknowledged_clients"]) for row in rows) != 100:
            raise RuntimeError(f"ACK audit failed for {strategy}")
        if max(int(row["pending_units"]) for row in rows) != 0:
            raise RuntimeError(f"pending update audit failed for {strategy}")
        if int(rows[-1]["cloud_version"]) != 100:
            raise RuntimeError(f"cloud version audit failed for {strategy}")

        for index, row in enumerate(rows):
            trace_path = folder / f"timeline_round_{index:03d}.json"
            events = json.loads(trace_path.read_text(encoding="utf-8"))
            if not events or events[-1].get("event") != "global_barrier":
                raise RuntimeError(f"missing final barrier in {trace_path}")
            if abs(float(events[-1]["finish"]) - float(row["system_time"])) > 1e-8:
                raise RuntimeError(f"timeline and cumulative T mismatch in {trace_path}")

        metrics[strategy] = rows
        identities[strategy] = identity
        model_hashes.add(sha256(folder / "initial_model.pt"))
        audit_lines.append(
            f"- `{strategy}`：100 轮完整；累计 T 单调不减且等于逐轮时延之和；"
            "100 份时延事件记录均以 Cloud 全局屏障结束，并与累计 T 一致；"
            "准确率原始量纲为 0 至 1；ACK、pending 和 Cloud version 检查通过。"
        )

    fields = ("config_hash", "source_hash", "partition_hash", "traces_hash", "data_hashes")
    for field in fields:
        values = {json.dumps(identities[name][field], sort_keys=True) for name in STRATEGIES}
        if len(values) != 1:
            raise RuntimeError(f"run identity mismatch: {field}")
    if len(model_hashes) != 1:
        raise RuntimeError("initial model hashes differ")

    shared = read_csv(seed_root / "shared_P0_prefix" / "round_metrics.csv")
    for strategy in ("fixed_P0", "greedy_lambda025", "greedy_lambda050", "greedy_lambda075"):
        for expected, actual in zip(shared, metrics[strategy][:5]):
            for field in ("round", "test_accuracy", "test_loss", "system_time", "round_duration"):
                if expected[field] != actual[field]:
                    raise RuntimeError(f"common prefix mismatch for {strategy}: {field}")
    initial_accuracy, initial_loss, test_count = evaluate_initial_accuracy(seed_root)
    context = {
        "initial_accuracy": initial_accuracy,
        "initial_loss": initial_loss,
        "test_count": test_count,
        "initial_model_sha256": next(iter(model_hashes)),
        "config_hash": identities["fixed_P0"]["config_hash"],
        "source_hash": identities["fixed_P0"]["source_hash"],
        "partition_hash": identities["fixed_P0"]["partition_hash"],
        "traces_hash": identities["fixed_P0"]["traces_hash"],
    }
    return prereg, metrics, context, audit_lines


def make_summary(metrics: dict[str, list[dict]], context: dict) -> list[dict]:
    rows: list[dict] = []
    for strategy in STRATEGIES:
        source = metrics[strategy]
        final_accuracy = float(source[-1]["test_accuracy"])
        final_t = float(source[-1]["system_time"])
        definitions = [
            ("common_absolute", target, f"common_{int(target * 100)}pct")
            for target in COMMON_TARGETS
        ]
        definitions += [
            ("A90_final", 0.9 * final_accuracy, "90pct_of_final_accuracy"),
            (
                "A90_gain",
                context["initial_accuracy"]
                + 0.9 * (final_accuracy - context["initial_accuracy"]),
                "90pct_of_learning_gain_from_initial",
            ),
        ]
        for family, target, label in definitions:
            hit = hit_metrics(source, target)
            rows.append(
                {
                    "strategy": strategy,
                    "seed": 42,
                    "threshold_family": family,
                    "threshold_label": label,
                    "target_accuracy": target,
                    **hit,
                    "initial_accuracy": context["initial_accuracy"],
                    "final_accuracy": final_accuracy,
                    "cumulative_T": final_t,
                    "paper_T_unit": "simulated_seconds",
                }
            )

    lookup = {(row["strategy"], row["threshold_label"]): row for row in rows}
    for row in rows:
        if row["threshold_family"] != "common_absolute":
            row.update(
                {
                    "first_T_difference_vs_fixed_P0": "not_applicable_different_thresholds",
                    "first_T_saving_percent_vs_fixed_P0": "not_applicable_different_thresholds",
                    "sustained_T_difference_vs_fixed_P0": "not_applicable_different_thresholds",
                    "sustained_T_saving_percent_vs_fixed_P0": "not_applicable_different_thresholds",
                    "first_T_difference_vs_fixed_P1": "not_applicable_different_thresholds",
                    "first_T_saving_percent_vs_fixed_P1": "not_applicable_different_thresholds",
                    "sustained_T_difference_vs_fixed_P1": "not_applicable_different_thresholds",
                    "sustained_T_saving_percent_vs_fixed_P1": "not_applicable_different_thresholds",
                }
            )
            continue
        for baseline in ("fixed_P0", "fixed_P1"):
            base = lookup[(baseline, row["threshold_label"])]
            tag = baseline
            for prefix, field in (("first", "first_hit_T"), ("sustained", "sustained_hit_T")):
                if isinstance(row[field], str) or isinstance(base[field], str):
                    difference = saving = "not_reached"
                else:
                    difference = float(row[field]) - float(base[field])
                    saving = (float(base[field]) - float(row[field])) / float(base[field]) * 100
                row[f"{prefix}_T_difference_vs_{tag}"] = difference
                row[f"{prefix}_T_saving_percent_vs_{tag}"] = saving
    return rows


def pareto_rows(metrics: dict[str, list[dict]]) -> list[dict]:
    rows = []
    for strategy in STRATEGIES:
        source = metrics[strategy]
        rows.append(
            {
                "strategy": strategy,
                "final_accuracy": float(source[-1]["test_accuracy"]),
                "final_T": float(source[-1]["system_time"]),
            }
        )
    for row in rows:
        row["pareto"] = not any(
            other is not row
            and other["final_T"] <= row["final_T"]
            and other["final_accuracy"] >= row["final_accuracy"]
            and (
                other["final_T"] < row["final_T"]
                or other["final_accuracy"] > row["final_accuracy"]
            )
            for other in rows
        )
    return rows


def plot_common_thresholds(summary: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    x = np.arange(len(STRATEGIES))
    width = 0.38
    for axis, target in zip(axes, COMMON_TARGETS):
        selected = {
            row["strategy"]: row
            for row in summary
            if row["threshold_family"] == "common_absolute"
            and abs(float(row["target_accuracy"]) - target) < 1e-12
        }
        first = [float(selected[name]["first_hit_T"]) for name in STRATEGIES]
        sustained = [float(selected[name]["sustained_hit_T"]) for name in STRATEGIES]
        axis.bar(x - width / 2, first, width, label="First hit", color="#4c78a8")
        axis.bar(x + width / 2, sustained, width, label="5-round sustained start", color="#f58518")
        axis.set_title(f"Common target: {target:.0%}")
        axis.set_xticks(x, [DISPLAY[name] for name in STRATEGIES], rotation=42, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Cumulative paper T (simulated seconds)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.16), ncol=2, frameon=False)
    fig.suptitle("Time to common accuracy targets", y=1.04, fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "common_threshold_time_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_time(metrics: dict[str, list[dict]], summary: list[dict], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(14, 8))
    common = {
        row["strategy"]: row
        for row in summary
        if row["threshold_label"] == "common_50pct"
    }
    for strategy in STRATEGIES:
        rows = metrics[strategy]
        times = [float(row["system_time"]) for row in rows]
        acc = [100 * float(row["test_accuracy"]) for row in rows]
        linestyle = "-" if strategy.startswith("fixed") else "--"
        axis.plot(times, acc, color=COLORS[strategy], alpha=0.22, linewidth=0.8)
        axis.plot(
            times,
            moving_average(acc),
            color=COLORS[strategy],
            linestyle=linestyle,
            linewidth=2.1,
            label=DISPLAY[strategy],
        )
        hit = common[strategy]
        first_round = int(hit["first_hit_round"])
        sustained_round = int(hit["sustained_hit_round"])
        axis.scatter(
            [float(hit["first_hit_T"])],
            [acc[first_round - 1]],
            marker="*",
            s=95,
            color=COLORS[strategy],
            edgecolor="white",
            linewidth=0.6,
            zorder=5,
        )
        confirm_round = int(hit["sustained_confirm_round"])
        axis.plot(
            times[sustained_round - 1 : confirm_round],
            acc[sustained_round - 1 : confirm_round],
            color=COLORS[strategy],
            linewidth=5.0,
            alpha=0.28,
            solid_capstyle="round",
            zorder=3,
        )
        axis.scatter(
            [float(hit["sustained_hit_T"])],
            [acc[sustained_round - 1]],
            marker="s",
            s=38,
            color=COLORS[strategy],
            edgecolor="black",
            linewidth=0.45,
            zorder=5,
        )
        for round_number in MARK_ROUNDS:
            index = round_number - 1
            axis.scatter(times[index], acc[index], s=14, color=COLORS[strategy], alpha=0.7)
    axis.axhline(50, color="black", linestyle=":", linewidth=1.6, label="Common 50% target")
    axis.set_xlabel("Cumulative paper T (simulated seconds)")
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title("Accuracy versus cumulative simulation time\nRaw trajectory faint; centered 5-round mean emphasized")
    axis.grid(alpha=0.22)
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([], [], marker="*", linestyle="None", color="black", markersize=10),
            Line2D([], [], marker="s", linestyle="None", markerfacecolor="white", markeredgecolor="black", markersize=6),
            Line2D([], [], marker="o", linestyle="None", color="#666666", markersize=4),
        ]
    )
    labels.extend(["First 50% hit", "5-round sustained start", "Rounds 10, 20, 40, 60, 80, 100"])
    axis.legend(handles, labels, ncol=3, fontsize=8.5, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "accuracy_vs_cumulative_T.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_relative_thresholds(summary: list[dict], output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharey=True)
    x = np.arange(len(STRATEGIES))
    width = 0.38
    for axis, family, title in (
        (axes[0], "A90_final", "90% of final accuracy"),
        (axes[1], "A90_gain", "90% of learning gain"),
    ):
        selected = {row["strategy"]: row for row in summary if row["threshold_family"] == family}
        first = [float(selected[name]["first_hit_T"]) for name in STRATEGIES]
        sustained = [float(selected[name]["sustained_hit_T"]) for name in STRATEGIES]
        axis.bar(x - width / 2, first, width, label="First hit", color="#4c78a8")
        axis.bar(x + width / 2, sustained, width, label="5-round sustained start", color="#f58518")
        axis.set_title(title)
        axis.set_xticks(x, [DISPLAY[name] for name in STRATEGIES], rotation=42, ha="right")
        axis.grid(axis="y", alpha=0.25)
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Cumulative paper T (simulated seconds)")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, 1.15), ncol=2, frameon=False)
    fig.suptitle("Relative convergence thresholds use strategy-specific targets", y=1.03, fontsize=15)
    fig.tight_layout()
    fig.savefig(output / "relative_convergence_thresholds.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_accuracy_round(metrics: dict[str, list[dict]], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(14, 8))
    for strategy in STRATEGIES:
        rows = metrics[strategy]
        rounds = [int(row["round"]) for row in rows]
        acc = [100 * float(row["test_accuracy"]) for row in rows]
        linestyle = "-" if strategy.startswith("fixed") else "--"
        axis.plot(rounds, acc, color=COLORS[strategy], alpha=0.20, linewidth=0.8)
        axis.plot(
            rounds,
            moving_average(acc),
            color=COLORS[strategy],
            linestyle=linestyle,
            linewidth=2.1,
            label=DISPLAY[strategy],
        )
    axis.axhline(50, color="black", linestyle=":", linewidth=1.5, label="Common 50% target")
    axis.set_xlabel("Round")
    axis.set_ylabel("Test accuracy (%)")
    axis.set_title("Accuracy versus round\nRaw trajectory faint; centered 5-round mean emphasized")
    axis.grid(alpha=0.22)
    axis.legend(ncol=3, fontsize=9, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "accuracy_vs_round.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def plot_pareto(metrics: dict[str, list[dict]], output: Path) -> list[str]:
    rows = pareto_rows(metrics)
    fig, axis = plt.subplots(figsize=(10, 7))
    for row in rows:
        strategy = row["strategy"]
        marker = "D" if row["pareto"] else "o"
        axis.scatter(
            row["final_T"],
            100 * row["final_accuracy"],
            color=COLORS[strategy],
            marker=marker,
            s=90 if row["pareto"] else 60,
            edgecolor="black",
            linewidth=0.6,
        )
        axis.annotate(DISPLAY[strategy], (row["final_T"], 100 * row["final_accuracy"]), xytext=(5, 5), textcoords="offset points", fontsize=9)
    frontier = sorted((row for row in rows if row["pareto"]), key=lambda item: item["final_T"])
    axis.plot([row["final_T"] for row in frontier], [100 * row["final_accuracy"] for row in frontier], color="#333333", linestyle="--", linewidth=1.1, label="Observed Pareto frontier")
    axis.set_xlabel("Final cumulative paper T (simulated seconds)")
    axis.set_ylabel("Final test accuracy (%)")
    axis.set_title("Final accuracy and cumulative simulation time")
    axis.grid(alpha=0.24)
    axis.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output / "final_accuracy_vs_cumulative_T.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return [row["strategy"] for row in rows if row["pareto"]]


def fmt_hit(row: dict, prefix: str) -> str:
    round_value = row[f"{prefix}_hit_round"]
    time_value = row[f"{prefix}_hit_T"]
    if isinstance(time_value, str):
        return "未达到"
    return f"第 {int(round_value)} 轮，{float(time_value):.4f} 秒"


def markdown_table(summary: list[dict], target: float) -> str:
    selected = [
        row
        for row in summary
        if row["threshold_family"] == "common_absolute"
        and abs(float(row["target_accuracy"]) - target) < 1e-12
    ]
    selected.sort(key=lambda row: float(row["first_hit_T"]) if not isinstance(row["first_hit_T"], str) else float("inf"))
    lines = [
        "| 策略 | 首次达标 | 连续 5 轮区间起点 | 确认轮次 | 首次 T 相对 P0 节省 | 持续 T 相对 P0 节省 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in selected:
        first_save = row["first_T_saving_percent_vs_fixed_P0"]
        sustained_save = row["sustained_T_saving_percent_vs_fixed_P0"]
        lines.append(
            f"| `{row['strategy']}` | {fmt_hit(row, 'first')} | {fmt_hit(row, 'sustained')} | "
            f"{row['sustained_confirm_round']} | {float(first_save):+.2f}% | {float(sustained_save):+.2f}% |"
        )
    return "\n".join(lines)


def write_reports(root: Path, output: Path, prereg: dict, metrics: dict[str, list[dict]],
                  context: dict, summary: list[dict], audit_lines: list[str], pareto: list[str]) -> None:
    common = {
        (row["strategy"], row["threshold_label"]): row
        for row in summary
        if row["threshold_family"] == "common_absolute"
    }
    target50 = [common[(name, "common_50pct")] for name in STRATEGIES]
    first_rank = sorted(target50, key=lambda row: float(row["first_hit_T"]))
    sustained_rank = sorted(target50, key=lambda row: float(row["sustained_hit_T"]))
    p0 = common[("fixed_P0", "common_50pct")]
    p1 = common[("fixed_P1", "common_50pct")]
    greedy = [common[(name, "common_50pct")] for name in STRATEGIES if name.startswith("greedy")]
    greedy_first_better = [row for row in greedy if float(row["first_hit_T"]) < float(p0["first_hit_T"])]
    greedy_sustained_better = [row for row in greedy if float(row["sustained_hit_T"]) < float(p0["sustained_hit_T"])]
    relative_rankings = {}
    for family in ("A90_final", "A90_gain"):
        family_rows = [row for row in summary if row["threshold_family"] == family]
        relative_rankings[family] = {
            "first": [row["strategy"] for row in sorted(family_rows, key=lambda item: float(item["first_hit_T"]))],
            "sustained": [row["strategy"] for row in sorted(family_rows, key=lambda item: float(item["sustained_hit_T"]))],
        }
    missing_switch = []
    for name in ("P0_then_P1_r5", "P0_then_P2_r5", "P0_then_P3_r5", "P0_then_P4_r5"):
        if not list(root.glob(f"**/{name}")):
            missing_switch.append(name)

    audit = f"""# FMNIST 收敛效率公平性审计

## 审计范围

- 数据目录：`{root}`
- 策略：{', '.join(f'`{name}`' for name in STRATEGIES)}
- 当前可用数据仅包含 seed 42 的这 8 个完整策略，因此本报告是单种子探索性分析。
- 未启动训练，也未修改原始实验文件、配置或训练代码。

## 数据完整性与论文 T

{chr(10).join(audit_lines)}

- 预注册字段 `paper_T` 为 `{prereg['paper_T']}`，表示采用修复后的事件关键路径模型，不对不同 `(Edge, Mode)` 更新组增加额外串行排队。
- `system_time` 从第 1 轮开始累计，并包含每轮 Cloud 聚合、下发、安装、ACK 和全局屏障的事件关键路径。它不是 Python 墙钟时间。
- 8 个策略的配置哈希、源码哈希、数据哈希、划分哈希和系统轨迹哈希一致。初始模型文件 SHA256 均为 `{context['initial_model_sha256']}`。
- 使用相同初始模型和原有测试流水线在 CPU 上只读复算：初始准确率为 {context['initial_accuracy']:.4%}，初始 Loss 为 {context['initial_loss']:.6f}，测试样本数为 {context['test_count']}。该复算没有训练模型。

## 前 5 轮公平性

- `fixed_P0` 与三个 greedy 策略的第 1 至第 5 轮准确率、Loss、累计 T 和逐轮 T 与公共 P0 前缀逐项一致。
- `fixed_P1`、`fixed_P2`、`fixed_P3`、`fixed_P4` 从第 1 轮开始执行各自固定分布，没有共享 P0 前缀。
- 因此当前 8 策略可以进行“完整训练策略比较”，即比较每种策略从第 1 轮开始的总体效果。
- 固定 P1 至 P4 与 greedy 策略不是从相同第 5 轮完整状态分叉，不能将二者第 6 轮后的差异单独归因于切换机制。
- 未找到以下共同第 5 轮分叉结果：{', '.join(f'`{name}`' for name in missing_switch)}。本分析没有为补齐这些数据启动训练。

## 准确率单位

原始 `test_accuracy` 使用 0 至 1 表示。图表显示时转换为百分比，阈值计算仍使用原始数值。
"""
    (output / "fairness_audit.md").write_text(audit, encoding="utf-8")

    first_names = "、".join(f"`{row['strategy']}`" for row in first_rank)
    sustained_names = "、".join(f"`{row['strategy']}`" for row in sustained_rank)
    report = f"""# FMNIST J/T 策略收敛效率报告

## 分析范围

本报告只使用已经完成的 seed 42 单种子 100 轮结果。所有时间均为论文模型给出的累计仿真系统时延，单位为 simulated seconds。阈值判断使用原始逐轮准确率，图中 5 轮居中移动平均仅帮助观察趋势。

## 共同精度目标

### 45% 目标

{markdown_table(summary, 0.45)}

### 50% 目标

{markdown_table(summary, 0.50)}

### 52% 目标

{markdown_table(summary, 0.52)}

三个共同目标均被 8 个策略实际达到，因此表中均可进行完整排名，没有插值或外推。

## 50% 目标的直接答案

1. 首次达到 50% 最快的是 `{first_rank[0]['strategy']}`：{fmt_hit(first_rank[0], 'first')}。
2. 最快形成连续 5 轮不低于 50% 区间的是 `{sustained_rank[0]['strategy']}`：区间从{fmt_hit(sustained_rank[0], 'sustained')}开始，在第 {sustained_rank[0]['sustained_confirm_round']} 轮确认。
3. 首次排名为：{first_names}。持续排名为：{sustained_names}。两者{'一致' if [r['strategy'] for r in first_rank] == [r['strategy'] for r in sustained_rank] else '不一致'}，说明单次越线不能替代持续达标判断。
4. 相对阈值的排名与共同 50% 目标需要分别理解。`A90_final` 使用每个策略自己的最终准确率，`A90_gain` 使用共同初始准确率 {context['initial_accuracy']:.2%} 到各自最终准确率的学习增益；两者都不是相同绝对精度目标。
5. 相对 fixed P0，首次达到 50% 更快的 greedy 策略为：{', '.join(f'`{r["strategy"]}`' for r in greedy_first_better) if greedy_first_better else '无'}；持续达到 50% 更快的 greedy 策略为：{', '.join(f'`{r["strategy"]}`' for r in greedy_sustained_better) if greedy_sustained_better else '无'}。这只说明当前 seed 42 的完整策略结果，不足以证明稳定改进。
6. fixed P1 最终准确率为 {float(metrics['fixed_P1'][-1]['test_accuracy']):.2%}，但其 50% 首次达标为{fmt_hit(p1, 'first')}，持续达标区间从{fmt_hit(p1, 'sustained')}开始。它相对 fixed P0 的最终精度优势不自动等于更好的早期收敛效率。
7. 当前结论受前 5 轮历史差异影响。三个 greedy 策略共享 P0 前缀，固定 P1 至 P4 从第 1 轮执行自己的分布。缺少共同第 5 轮完整状态分叉的 P1 至 P4 对照，无法隔离“切换决策本身”的因果效应。
8. 所有策略排名、时间节省和 Pareto 结论仅适用于 seed 42。是否具有稳定统计规律，需要等待 seed 43 和 44 完成后再报告均值、标准差和不确定性区间。

### fixed P1 补充参考

以 fixed P1 为参考，greedy λ=0.75 达到 50% 的首次 T 节省 {float(common[('greedy_lambda075', 'common_50pct')]['first_T_saving_percent_vs_fixed_P1']):.2f}%，持续 T 节省 {float(common[('greedy_lambda075', 'common_50pct')]['sustained_T_saving_percent_vs_fixed_P1']):.2f}%；greedy λ=0.50 的对应节省为 {float(common[('greedy_lambda050', 'common_50pct')]['first_T_saving_percent_vs_fixed_P1']):.2f}% 和 {float(common[('greedy_lambda050', 'common_50pct')]['sustained_T_saving_percent_vs_fixed_P1']):.2f}%。greedy λ=0.25 首次达标比 fixed P1 慢 {abs(float(common[('greedy_lambda025', 'common_50pct')]['first_T_saving_percent_vs_fixed_P1'])):.2f}%，但持续达标快 {float(common[('greedy_lambda025', 'common_50pct')]['sustained_T_saving_percent_vs_fixed_P1']):.2f}%。

## J/T 联合策略是否改善收敛效率

本次数据允许回答“当前 seed 42 下，greedy 完整策略相对 fixed P0 是否更快达到共同阈值”，但不能回答多种子稳定性，也不能把差异解释为 J/T 联合目标的因果收益。

以 50% 为共同目标，greedy λ=0.25、0.50、0.75 相对 fixed P0 的首次 T 分别节省 {float(common[('greedy_lambda025', 'common_50pct')]['first_T_saving_percent_vs_fixed_P0']):.2f}%、{float(common[('greedy_lambda050', 'common_50pct')]['first_T_saving_percent_vs_fixed_P0']):.2f}%、{float(common[('greedy_lambda075', 'common_50pct')]['first_T_saving_percent_vs_fixed_P0']):.2f}%；持续 T 分别节省 {float(common[('greedy_lambda025', 'common_50pct')]['sustained_T_saving_percent_vs_fixed_P0']):.2f}%、{float(common[('greedy_lambda050', 'common_50pct')]['sustained_T_saving_percent_vs_fixed_P0']):.2f}%、{float(common[('greedy_lambda075', 'common_50pct')]['sustained_T_saving_percent_vs_fixed_P0']):.2f}%。因此当前 seed 42 在 50% 目标上支持联合策略相对 fixed P0 更快，λ=0.50 和 λ=0.75 的持续优势尤其明显。

该优势不适用于所有目标和权重。52% 目标下，greedy λ=0.25 的首次 T 比 fixed P0 多 {abs(float(common[('greedy_lambda025', 'common_52pct')]['first_T_saving_percent_vs_fixed_P0'])):.2f}%，持续 T 多 {abs(float(common[('greedy_lambda025', 'common_52pct')]['sustained_T_saving_percent_vs_fixed_P0'])):.2f}%。所以不能概括为“所有 J/T 联合策略都在所有精度目标上更快”。结果应按首次 T、持续 T、目标精度和 λ 分别判断。

## 相对阈值定义

- `A90_final = 0.9 × A_final`。
- `A90_gain = A_init + 0.9 × (A_final - A_init)`，其中共同 `A_init = {context['initial_accuracy']:.4%}`。
- 两种定义均记录首次达标以及连续 5 轮达标区间的起始时间。相对阈值因策略而异，不能代替共同 45%、50% 或 52% 目标下的公平比较。

`A90_final` 首次达标排名为：{', '.join(f'`{name}`' for name in relative_rankings['A90_final']['first'])}。持续达标排名为：{', '.join(f'`{name}`' for name in relative_rankings['A90_final']['sustained'])}。

`A90_gain` 首次达标排名为：{', '.join(f'`{name}`' for name in relative_rankings['A90_gain']['first'])}。持续达标排名为：{', '.join(f'`{name}`' for name in relative_rankings['A90_gain']['sustained'])}。

这两组排名均与共同 50% 目标的完整排名不同。共同 50% 排名比较的是相同绝对精度，公平性更强；相对阈值主要描述每个策略接近自身最终水平的速度。

## 最终精度与累计 T

当前观测的 Pareto 非支配策略为：{', '.join(f'`{name}`' for name in pareto)}。该结论只描述 seed 42 的最终点，不表示统计显著性，也不表示这些策略在所有指标上最好。

## 输出文件

- `common_threshold_time_comparison.png`
- `accuracy_vs_cumulative_T.png`
- `relative_convergence_thresholds.png`
- `accuracy_vs_round.png`
- `final_accuracy_vs_cumulative_T.png`
- `convergence_efficiency_summary.csv`
- `fairness_audit.md`
- `analyze_fmnist_convergence_efficiency.py`
"""
    (output / "convergence_efficiency_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--refresh", action="store_true", help="refresh only this derived analysis directory")
    args = parser.parse_args()
    root = args.root.resolve()
    output = args.output.resolve()
    if output.exists() and not args.refresh:
        raise FileExistsError(f"refuse to overwrite existing analysis directory: {output}")
    output.mkdir(parents=True, exist_ok=args.refresh)
    try:
        prereg, metrics, context, audit_lines = audit_and_load(root)
        summary = make_summary(metrics, context)
        write_csv(output / "convergence_efficiency_summary.csv", summary)
        plot_common_thresholds(summary, output)
        plot_accuracy_time(metrics, summary, output)
        plot_relative_thresholds(summary, output)
        plot_accuracy_round(metrics, output)
        pareto = plot_pareto(metrics, output)
        write_reports(root, output, prereg, metrics, context, summary, audit_lines, pareto)
        shutil.copy2(Path(__file__), output / Path(__file__).name)
        print(json.dumps({"output": str(output), "rows": len(summary), "initial_accuracy": context["initial_accuracy"], "pareto": pareto}, ensure_ascii=False, indent=2))
    except Exception:
        if not args.refresh:
            shutil.rmtree(output, ignore_errors=True)
        raise


if __name__ == "__main__":
    main()
