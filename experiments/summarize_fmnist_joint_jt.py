"""Summarize, plot, and independently audit the FMNIST J/T campaign."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import statistics
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynfed.paper_config import MODES, PaperConfig
from dynfed.paper_storage import atomic_json, file_digest
from experiments.run_fmnist_joint_jt_100r import CANDIDATES, SPECS


def _read_csv(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path, rows):
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def _mean_sd_ci(values):
    mean = statistics.mean(values)
    sd = statistics.stdev(values) if len(values) > 1 else 0.0
    return mean, sd, 1.96*sd/math.sqrt(len(values))


def _pareto(points):
    # Both J and T_norm are maximized; raw T is minimized.
    return [point for point in points if not any(
        other["J"] >= point["J"] and other["T"] <= point["T"] and
        (other["J"] > point["J"] or other["T"] < point["T"])
        for other in points)]


def summarize(root: Path, config: PaperConfig):
    root = Path(root)
    rounds, candidates, assignments, checkpoints, timelines, run_summaries = [], [], [], [], [], []
    for seed in config.seeds:
        for strategy, spec in SPECS.items():
            folder = root / f"seed_{seed}" / strategy
            metrics = _read_csv(folder / "round_metrics.csv")
            decisions = _read_csv(folder / "client_decisions.csv")
            reports = [json.loads((folder / f"candidates_round_{index:03d}.json").read_text(encoding="utf-8"))
                       for index in range(config.rounds)]
            previous_name, switch_count, cumulative_steps = None, 0, 0
            for index, (metric, report) in enumerate(zip(metrics, reports)):
                selected = report["selected_candidate"]
                switched = previous_name is not None and selected != previous_name
                switch_count += int(switched); cumulative_steps += int(metric["executed_local_steps"])
                rounds.append({"seed": seed, "strategy": strategy, "round": index+1,
                               "selected_candidate": selected, "candidate_switched": switched,
                               "selection_reason": report.get("selection_reason"),
                               "lambda": report.get("lambda"), "cumulative_local_steps": cumulative_steps,
                               **metric})
                for item in report["candidates"]:
                    candidates.append({"seed": seed, "strategy": strategy, "round": index+1,
                                       "selected_candidate": selected, "candidate": item["name"],
                                       "J": item.get("J"), "T": item.get("T"),
                                       "J_norm": item.get("J_norm"), "T_norm": item.get("T_norm"),
                                       "utility": item.get("utility"), "feasible": item.get("feasible"),
                                       "rank_J": item.get("rank_J"), "rank_T": item.get("rank_T"),
                                       "rank_utility": item.get("rank_utility"), "selected": item.get("selected"),
                                       "lambda": report.get("lambda"), "selection_reason": report.get("selection_reason")})
                previous_name = selected
            for row in decisions:
                assignments.append({"seed": seed, "strategy": strategy, **row})
            for path in sorted(folder.glob("checkpoint_round_*.pt")):
                checkpoints.append({"seed": seed, "strategy": strategy,
                                    "round": int(path.stem.rsplit("_", 1)[1]),
                                    "path": str(path.relative_to(root)), "sha256": file_digest(path),
                                    "size_bytes": path.stat().st_size})
            for path in sorted(folder.glob("timeline_round_*.json")):
                timelines.append({"seed": seed, "strategy": strategy,
                                  "round": int(path.stem.rsplit("_", 1)[1])+1,
                                  "path": str(path.relative_to(root)), "sha256": file_digest(path),
                                  "size_bytes": path.stat().st_size})
            last = metrics[-1]
            run_summaries.append({"seed": seed, "strategy": strategy,
                                  "final_accuracy": float(last["test_accuracy"]),
                                  "best_accuracy": max(float(row["test_accuracy"]) for row in metrics),
                                  "final_loss": float(last["test_loss"]),
                                  "cumulative_system_T": float(last["system_time"]),
                                  "communication_mb": sum(float(row["communication_mb"]) for row in metrics),
                                  "executed_local_steps": sum(int(row["executed_local_steps"]) for row in metrics),
                                  "max_epsilon": max(float(row["max_epsilon"]) for row in metrics),
                                  "switch_count": switch_count,
                                  "wall_sec": json.loads((folder / "summary.json").read_text(encoding="utf-8"))["wall_sec"]})
    aggregate = []
    for strategy in SPECS:
        group = [row for row in run_summaries if row["strategy"] == strategy]
        entry = {"strategy": strategy, "seeds": len(group)}
        for key in ("final_accuracy", "best_accuracy", "final_loss", "cumulative_system_T",
                    "communication_mb", "executed_local_steps", "max_epsilon", "switch_count"):
            mean, sd, ci = _mean_sd_ci([float(row[key]) for row in group])
            entry.update({f"{key}_mean": mean, f"{key}_std": sd, f"{key}_ci95": ci})
        aggregate.append(entry)
    _write_csv(root / "round_metrics.csv", rounds)
    _write_csv(root / "candidate_evaluations.csv", candidates)
    _write_csv(root / "client_mode_assignments.csv", assignments)
    _write_csv(root / "checkpoint_integrity.csv", checkpoints)
    _write_csv(root / "latency_trace_manifest.csv", timelines)
    _write_csv(root / "per_seed_summary.csv", run_summaries)
    _write_csv(root / "strategy_summary.csv", aggregate)
    _plots(root, rounds, candidates, run_summaries, aggregate, config)
    from experiments.audit_fmnist_joint_jt import audit
    audit_result = audit(root, config)
    _report(root, aggregate, audit_result)
    atomic_json(root / "run_status.json", {"status": "completed" if audit_result["status"] == "passed" else "audit_failed",
                                           "audit_status": audit_result["status"], "runs": len(run_summaries)})
    if audit_result["status"] != "passed":
        raise RuntimeError("independent audit failed")


def _plots(root, rounds, candidates, run_summaries, aggregate, config):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = root / "plots"; plot_dir.mkdir(exist_ok=True)
    for filename, key, ylabel in (("accuracy_vs_round.png", "test_accuracy", "Test accuracy"),
                                  ("loss_vs_round.png", "test_loss", "Test loss"),
                                  ("cumulative_T_vs_round.png", "system_time", "Cumulative paper T (s)")):
        fig, axis = plt.subplots(figsize=(11, 6.5))
        for strategy in SPECS:
            means = []
            for round_index in range(1, config.rounds+1):
                values = [float(row[key]) for row in rounds if row["strategy"] == strategy and int(row["round"]) == round_index]
                means.append(statistics.mean(values))
            axis.plot(range(1, config.rounds+1), means, linewidth=1.15, label=strategy)
        axis.set(xlabel="Round", ylabel=ylabel, title=f"FMNIST 3-seed mean: {ylabel}")
        axis.grid(alpha=.2); axis.legend(fontsize=7, ncol=3); fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180); plt.close(fig)
    fig, axes = plt.subplots(2, 1, figsize=(11, 9), sharex=True)
    for strategy in SPECS:
        for axis, key, title in ((axes[0], "predicted_j", "J_pred"), (axes[1], "observed_j", "J_obs")):
            values = []
            for round_index in range(1, config.rounds+1):
                current = [float(row[key]) for row in rounds if row["strategy"] == strategy and
                           int(row["round"]) == round_index and row[key] != ""]
                values.append(statistics.mean(current) if current else math.nan)
            axis.plot(range(1, config.rounds+1), values, linewidth=1.1, label=strategy)
            axis.set_ylabel(title); axis.grid(alpha=.2)
    axes[1].set_xlabel("Round"); axes[0].legend(fontsize=6, ncol=3); fig.tight_layout()
    fig.savefig(plot_dir / "J_pred_J_obs_vs_round.png", dpi=180); plt.close(fig)
    dynamic = [name for name, spec in SPECS.items() if spec["kind"] in {"greedy","hysteresis","periodic","max_j","min_t"}]
    codes = {name: index for index, name in enumerate(CANDIDATES)}
    fig, axis = plt.subplots(figsize=(12, 5.5))
    matrix = []
    for strategy in dynamic:
        matrix.append([statistics.mean([codes[row["selected_candidate"]] for row in rounds
                                        if row["strategy"] == strategy and int(row["round"]) == rnd])
                       for rnd in range(1, config.rounds+1)])
    image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="viridis", vmin=0, vmax=4)
    axis.set_yticks(range(len(dynamic)), dynamic); axis.set_xlabel("Round"); axis.set_title("Mean selected candidate code and switch pattern")
    bar = fig.colorbar(image, ax=axis, ticks=range(5)); bar.ax.set_yticklabels(CANDIDATES)
    fig.tight_layout(); fig.savefig(plot_dir / "mode_distribution_switch_events.png", dpi=180); plt.close(fig)
    points = [row for row in candidates if int(row["seed"]) == 42 and row["strategy"] == "greedy_lambda050"
              and int(row["round"]) == 6 and row["candidate"] in CANDIDATES]
    typed = [{**row, "J": float(row["J"]), "T": float(row["T"])} for row in points]
    front = sorted(_pareto(typed), key=lambda row: row["T"])
    fig, axis = plt.subplots(figsize=(7, 5))
    for row in typed:
        axis.scatter(row["T"], row["J"], s=70); axis.annotate(row["candidate"], (row["T"], row["J"]))
    axis.plot([row["T"] for row in front], [row["J"] for row in front], "k--", label="Pareto front")
    axis.set(xlabel="Paper T (s, lower is better)", ylabel="J_pred (higher deviation)",
             title="Seed 42, round 6 candidate J-T front"); axis.grid(alpha=.25); axis.legend(); fig.tight_layout()
    fig.savefig(plot_dir / "J_T_candidates_pareto.png", dpi=180); plt.close(fig)
    lambdas = ["greedy_lambda025", "greedy_lambda050", "greedy_lambda075"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    bottom = [0]*3
    for candidate in CANDIDATES:
        counts = [sum(row["selected_candidate"] == candidate for row in rounds if row["strategy"] == strategy)
                  for strategy in lambdas]
        axes[0].bar(lambdas, counts, bottom=bottom, label=candidate); bottom = [a+b for a,b in zip(bottom,counts)]
    axes[0].set_title("Selections across 3 seeds"); axes[0].tick_params(axis="x", rotation=20); axes[0].legend()
    final = [next(row["final_accuracy_mean"] for row in aggregate if row["strategy"] == strategy) for strategy in lambdas]
    axes[1].bar(lambdas, final); axes[1].set_title("Mean final accuracy"); axes[1].tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(plot_dir / "lambda_selection_performance.png", dpi=180); plt.close(fig)


def _report(root, aggregate, audit):
    ranked = sorted(aggregate, key=lambda row: -row["final_accuracy_mean"])
    table = ["| Strategy | Final accuracy mean±SD | Best accuracy | Final loss | Cumulative T | Communication MB | Epsilon | Switches |",
             "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for row in ranked:
        table.append(f"| {row['strategy']} | {row['final_accuracy_mean']:.4f}±{row['final_accuracy_std']:.4f} | "
                     f"{row['best_accuracy_mean']:.4f} | {row['final_loss_mean']:.4f} | "
                     f"{row['cumulative_system_T_mean']:.3f} | {row['communication_mb_mean']:.1f} | "
                     f"{row['max_epsilon_mean']:.4f} | {row['switch_count_mean']:.1f} |")
    text = f"""# FMNIST J/T joint-objective 100-round campaign

This report covers 15 strategies, seeds 42/43/44, and 100 rounds per run. T is simulated paper-system latency, never Python wall-clock time. J_pred is the online historical-deviation score; J_obs is reported only after execution.

## Strategy comparison

{chr(10).join(table)}

Uncertainty columns, including 95% normal-approximation intervals, are in `strategy_summary.csv`. With only three seeds, intervals are descriptive and should not be treated as strong evidence of superiority.

## Audit

- Status: **{audit['status']}**
- Checks: {len(audit['checks'])}
- Failures: {len(audit['failures'])}

The audit independently recomputed candidate J from frozen histories, candidate and executed T from the paper event model, joint utilities, selection rules, privacy bounds, ACK barriers, versions, allocations, and checkpoint checksums.
"""
    (root / "experiment_report.md").write_text(text, encoding="utf-8")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args(); summarize(args.root, PaperConfig.load(args.config))
