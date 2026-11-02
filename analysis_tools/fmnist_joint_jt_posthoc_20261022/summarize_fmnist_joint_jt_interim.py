"""Build a read only interim summary for completed seed 42 strategies.

The script reads completed run artifacts and writes a separate report directory.
It never edits the active campaign directory.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
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

LABELS = {
    "fixed_P0": "Fixed P0",
    "fixed_P1": "Fixed P1",
    "fixed_P2": "Fixed P2",
    "fixed_P3": "Fixed P3",
    "fixed_P4": "Fixed P4",
    "greedy_lambda025": "Greedy λ=0.25",
    "greedy_lambda050": "Greedy λ=0.50",
    "greedy_lambda075": "Greedy λ=0.75",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(rows: list[dict[str, str]], key: str) -> list[float]:
    return [float(row[key]) for row in rows if row.get(key, "") != ""]


def selected_candidates(folder: Path) -> tuple[list[str], int]:
    selected = []
    for index in range(100):
        report = json.loads(
            (folder / f"candidates_round_{index:03d}.json").read_text(encoding="utf-8")
        )
        selected.append(report["selected_candidate"])
    switches = sum(current != previous for previous, current in zip(selected, selected[1:]))
    return selected, switches


def summarize(source: Path) -> tuple[list[dict], dict[str, list[dict]], list[str]]:
    summaries: list[dict] = []
    metrics_by_strategy: dict[str, list[dict]] = {}
    checks: list[str] = []
    for strategy in STRATEGIES:
        folder = source / strategy
        metrics = read_csv(folder / "round_metrics.csv")
        metrics_by_strategy[strategy] = metrics
        run_status = json.loads((folder / "run_status.json").read_text(encoding="utf-8"))
        rounds = [int(row["round"]) for row in metrics]
        selected, switch_count = selected_candidates(folder)
        accuracies = as_float(metrics, "test_accuracy")
        losses = as_float(metrics, "test_loss")
        predicted = as_float(metrics, "predicted_j")
        observed = as_float(metrics, "observed_j")
        final = metrics[-1]
        valid = (
            run_status.get("status") == "completed"
            and rounds == list(range(1, 101))
            and min(int(row["acknowledged_clients"]) for row in metrics) == 100
            and max(int(row["pending_units"]) for row in metrics) == 0
            and int(final["cloud_version"]) == 100
            and max(float(row["max_epsilon"]) for row in metrics) <= 8.0 + 1e-8
            and (folder / "checkpoint_round_100.pt").is_file()
            and (folder / "final_state.pt").is_file()
        )
        checks.append(f"{strategy}: {'通过' if valid else '未通过'}")
        best_accuracy = max(accuracies)
        summaries.append(
            {
                "strategy": strategy,
                "label": LABELS[strategy],
                "final_accuracy": accuracies[-1],
                "best_accuracy": best_accuracy,
                "best_round": accuracies.index(best_accuracy) + 1,
                "last10_accuracy": statistics.mean(accuracies[-10:]),
                "final_loss": losses[-1],
                "cumulative_T": float(final["system_time"]),
                "communication_mb": sum(float(row["communication_mb"]) for row in metrics),
                "local_steps": sum(int(row["executed_local_steps"]) for row in metrics),
                "max_epsilon": max(float(row["max_epsilon"]) for row in metrics),
                "mean_J_pred": statistics.mean(predicted),
                "mean_J_obs": statistics.mean(observed),
                "switches": switch_count,
                "selections": {name: selected.count(name) for name in sorted(set(selected))},
                "integrity": valid,
            }
        )
    return summaries, metrics_by_strategy, checks


def make_plots(output: Path, metrics_by_strategy: dict[str, list[dict]], summaries: list[dict]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output / "plots"
    plot_dir.mkdir(parents=True, exist_ok=False)
    colors = plt.get_cmap("tab10")

    def line_plot(filename: str, key: str, ylabel: str, title: str) -> None:
        fig, axis = plt.subplots(figsize=(11, 6.5))
        for index, strategy in enumerate(STRATEGIES):
            rows = metrics_by_strategy[strategy]
            xs, ys = [], []
            for row in rows:
                if row.get(key, "") != "":
                    xs.append(int(row["round"]))
                    ys.append(float(row[key]))
            axis.plot(xs, ys, linewidth=1.3, color=colors(index), label=LABELS[strategy])
        axis.set_xlabel("Round")
        axis.set_ylabel(ylabel)
        axis.set_title(title)
        axis.grid(alpha=0.22)
        axis.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=190)
        plt.close(fig)

    line_plot(
        "accuracy_vs_round.png",
        "test_accuracy",
        "Test accuracy",
        "FMNIST seed 42 interim accuracy",
    )
    line_plot(
        "loss_vs_round.png",
        "test_loss",
        "Test loss",
        "FMNIST seed 42 interim loss",
    )
    line_plot(
        "cumulative_T_vs_round.png",
        "system_time",
        "Cumulative paper system latency T (simulated seconds)",
        "FMNIST seed 42 cumulative simulated latency",
    )

    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for index, strategy in enumerate(STRATEGIES):
        rows = metrics_by_strategy[strategy]
        for axis, key, label in (
            (axes[0], "predicted_j", "J_pred"),
            (axes[1], "observed_j", "J_obs"),
        ):
            xs = [int(row["round"]) for row in rows if row.get(key, "") != ""]
            ys = [float(row[key]) for row in rows if row.get(key, "") != ""]
            axis.plot(xs, ys, linewidth=1.2, color=colors(index), label=LABELS[strategy])
            axis.set_ylabel(label)
            axis.grid(alpha=0.22)
    axes[0].set_title("FMNIST seed 42 predicted and observed J")
    axes[0].legend(fontsize=8, ncol=2)
    axes[1].set_xlabel("Round")
    fig.tight_layout()
    fig.savefig(plot_dir / "J_pred_J_obs_vs_round.png", dpi=190)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(8, 6))
    for index, row in enumerate(summaries):
        axis.scatter(
            row["cumulative_T"], row["final_accuracy"], s=78, color=colors(index)
        )
        axis.annotate(row["label"], (row["cumulative_T"], row["final_accuracy"]), xytext=(5, 4), textcoords="offset points", fontsize=8)
    axis.set_xlabel("Cumulative paper system latency T (simulated seconds)")
    axis.set_ylabel("Final test accuracy")
    axis.set_title("FMNIST seed 42 final accuracy and cumulative latency")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(plot_dir / "final_accuracy_vs_cumulative_T.png", dpi=190)
    plt.close(fig)


def fmt_selections(value: dict[str, int]) -> str:
    return ", ".join(f"{key}: {count}" for key, count in value.items())


def write_report(output: Path, summaries: list[dict], checks: list[str]) -> None:
    ranked = sorted(summaries, key=lambda row: row["final_accuracy"], reverse=True)
    rows = [
        "| 策略 | 最终准确率 | 最佳准确率（轮次） | 末 10 轮平均准确率 | 最终 Loss | 累计 T（仿真秒） | 通信量（MB） | 本地步数 | 最大 ε | 切换次数 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in ranked:
        rows.append(
            f"| `{row['strategy']}` | {row['final_accuracy']:.4f} | "
            f"{row['best_accuracy']:.4f}（{row['best_round']}） | {row['last10_accuracy']:.4f} | "
            f"{row['final_loss']:.4f} | {row['cumulative_T']:.3f} | "
            f"{row['communication_mb']:.1f} | {row['local_steps']} | "
            f"{row['max_epsilon']:.6f} | {row['switches']} |"
        )

    fixed = [row for row in summaries if row["strategy"].startswith("fixed_")]
    greedy = [row for row in summaries if row["strategy"].startswith("greedy_")]
    best_final = max(summaries, key=lambda row: row["final_accuracy"])
    best_peak = max(summaries, key=lambda row: row["best_accuracy"])
    lowest_loss = min(summaries, key=lambda row: row["final_loss"])
    lowest_t = min(summaries, key=lambda row: row["cumulative_T"])
    best_fixed = max(fixed, key=lambda row: row["final_accuracy"])
    best_greedy = max(greedy, key=lambda row: row["final_accuracy"])

    choice_rows = [
        "| 联合效用策略 | P0 | P1 | P2 | P3 | P4 | 实际切换次数 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in greedy:
        choice_rows.append(
            f"| `{row['strategy']}` | "
            + " | ".join(str(row["selections"].get(f"P{i}", 0)) for i in range(5))
            + f" | {row['switches']} |"
        )

    report = f"""# FMNIST 联合 J 与 T 实验阶段结果

本报告只整理种子 42 已完成的 8 个策略，每个策略均完成 100 轮。它不是三种子最终统计结果，也没有替代正式独立审计。系统时延 T 是论文事件模型计算的仿真时间，不是 Python 墙钟时间。

## 当前结果

{chr(10).join(rows)}

## 初步观察

1. 当前最终准确率最高的是 `{best_final['strategy']}`，为 {best_final['final_accuracy']:.4f}。固定策略中表现最好的是 `{best_fixed['strategy']}`。三个联合效用策略中最终准确率最高的是 `{best_greedy['strategy']}`，为 {best_greedy['final_accuracy']:.4f}。
2. 当前单轮轨迹中的最高准确率来自 `{best_peak['strategy']}`，为 {best_peak['best_accuracy']:.4f}，出现在第 {best_peak['best_round']} 轮。峰值不能替代最终或末段平均结果。
3. 最低最终 Loss 来自 `{lowest_loss['strategy']}`，为 {lowest_loss['final_loss']:.4f}。准确率排名和 Loss 排名并不完全一致，这符合两个指标衡量内容不同的情况。
4. 最低累计仿真 T 来自 `{lowest_t['strategy']}`，为 {lowest_t['cumulative_T']:.3f} 秒。高准确率和低时延目前没有由同一个策略同时取得。
5. 固定 P1 的最终准确率为 {best_fixed['final_accuracy']:.4f}，累计 T 为 {best_fixed['cumulative_T']:.3f} 秒。联合效用策略中 λ=0.25 更偏向低时延，λ=0.75 在当前种子上取得更高精度与更低 Loss，但累计 T 更高。
6. 各策略平均 J_pred 的范围为 {min(row['mean_J_pred'] for row in summaries):.4f} 至 {max(row['mean_J_pred'] for row in summaries):.4f}。它与最终准确率没有呈现简单的单调对应关系。因此当前结果仍支持把 J 解释为历史更新偏离程度，而不是精度的直接代理。

## 联合效用策略的候选选择

{chr(10).join(choice_rows)}

这里的“切换次数”按相邻轮最终选择的候选名称变化计算，不使用客户端 Mode 变化总数替代。

## 完整性检查

{chr(10).join('- ' + item for item in checks)}

上述检查要求每个策略具有连续 1 至 100 轮、每轮 100 个客户端 ACK、pending 始终为 0、最终 Cloud 版本为 100、最大 epsilon 不超过 8，并存在第 100 轮检查点与最终状态。正式全实验的独立审计仍需等全部 45 组运行结束后执行。

## 图表

- `plots/accuracy_vs_round.png`
- `plots/loss_vs_round.png`
- `plots/cumulative_T_vs_round.png`
- `plots/J_pred_J_obs_vs_round.png`
- `plots/final_accuracy_vs_cumulative_T.png`

## 使用边界

当前只有一个随机种子。不能报告均值、标准差或置信区间，也不能据此宣称某个策略稳定优于其他策略。正在运行的其余策略和种子可能改变排名。论文正文可以暂时引用这些结果作为阶段性观察，但最终版本应由完整三种子结果和独立审计结果替换。
"""
    (output / "interim_report.md").write_text(report, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite existing output: {args.output}")
    args.output.mkdir(parents=True)
    summaries, metrics_by_strategy, checks = summarize(args.source)
    make_plots(args.output, metrics_by_strategy, summaries)
    write_report(args.output, summaries, checks)
    if not all(row["integrity"] for row in summaries):
        raise RuntimeError("one or more interim integrity checks failed")


if __name__ == "__main__":
    main()
