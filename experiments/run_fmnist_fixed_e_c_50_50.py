"""Isolated 20-client diagnostic: fixed 50% FL_E and 50% FL_C for 100 rounds."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import math
from pathlib import Path
import shutil
import sys
import time

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynfed.paper_config import MODES, PaperConfig
from dynfed.paper_history import History
from dynfed.paper_privacy import Ledger
from dynfed.paper_search import Evaluation
from dynfed.paper_storage import digest, file_digest, source_identity, tensor_digest
from dynfed.paper_timeline import mode_masses, plan
import dynfed.paper_runner as runner
from experiments.audit_paper import audit


DIAGNOSTIC_NAME = "fixed_fl_e_c_50_50"
SEARCH_PROTOCOL = "diagnostic_fixed_e_c_50_50_v1"
DEFAULT_CONFIG = ROOT / "configs" / "fmnist_fixed_e_c_50_50_100r_20261021.json"
DEFAULT_OUTPUT = ROOT / "out" / "fmnist_fixed_e_c_50_50_100r_20261021"


def fixed_profile(config: PaperConfig, pools: dict[int, tuple[str, ...]]) -> dict[int, str]:
    """Assign whole Edges 0..4 to FL_E and 5..9 to FL_C.

    With client->edge defined as client % edges, this keeps two FL_E clients on
    each FL_E Edge, so FL_E remains a genuine group update rather than a
    one-client degenerate Edge unit.
    """
    if config.clients != 20 or config.edges != 10:
        raise ValueError("diagnostic requires exactly 20 clients and 10 Edges")
    result = {}
    for client in sorted(pools):
        edge = client % config.edges
        mode = "FL_E" if edge < config.edges // 2 else "FL_C"
        if mode not in pools[client]:
            raise RuntimeError(f"fixed assignment infeasible for client {client}: {mode}")
        result[client] = mode
    if list(result.values()).count("FL_E") != 10 or list(result.values()).count("FL_C") != 10:
        raise RuntimeError("fixed assignment is not exactly 50/50")
    return result


def diagnostic_select(config, policy, pools, history, traces, samples, payload_mb, start,
                      round_index, previous, pending, force_reselect=False):
    if policy != "fixed_mixed":
        raise RuntimeError("isolated diagnostic entry accepts only fixed_mixed runtime identity")
    selected_at = time.perf_counter()
    profile = fixed_profile(config, pools)
    schedule = plan(config, profile, traces, samples, payload_mb, start, round_index, previous, pending)
    masses = mode_masses(schedule.accepted)
    _, j_value, status = history.predict(masses)
    missing = set(MODES) - set(history.deviations)
    coverage = len(set(masses) & missing)
    chosen = Evaluation(profile, schedule, schedule.finish - start, j_value, status, coverage)
    candidate = {
        "name": "Fixed-50E-50C", "profile": profile, "T": chosen.latency,
        "J": j_value, "status": status, "coverage": coverage, "masses": masses,
        "pareto": j_value is not None, "dominated_by": [], "chebyshev_score": None,
        "pareto_rank": 1 if j_value is not None else None,
    }
    return chosen, {
        "phase": "fixed", "evaluated_profiles": 1,
        "feasible_combinations": math.prod(len(pool) for pool in pools.values()),
        "unconstrained_combinations": 3 ** len(pools),
        "pareto_front_size": int(j_value is not None), "archive_size": int(j_value is not None),
        "selection_sec": time.perf_counter() - selected_at,
        "search_protocol": SEARCH_PROTOCOL,
        "assignment_rule": "whole_edges_0_4_FL_E_edges_5_9_FL_C_v1",
        "missing_history": sorted(missing), "coverage": coverage,
        "candidates": [candidate], "rejected_candidates": [],
        "archive_profiles": [profile] if j_value is not None else [],
        "selected_profile": profile, "selected_candidate": "Fixed-50E-50C",
        "feasible_pools": pools,
    }


def privacy_preview(config: PaperConfig) -> dict:
    pools = {client: MODES for client in range(config.clients)}
    profile = fixed_profile(config, pools)
    ledgers = {client: Ledger(config.epsilon, config.delta) for client in profile}
    round_rows = []
    sigmas = []
    for round_index in range(config.rounds):
        for client, mode in profile.items():
            if mode not in {"FL_E", "FL_C"}:
                raise RuntimeError("unexpected mode in privacy preview")
            events = config.local_steps
            sigma = ledgers[client].calibrate(events * (config.rounds - round_index))
            if sigma is None:
                raise RuntimeError(f"privacy calibration failed at round {round_index + 1}")
            ledgers[client].commit(events, sigma)
            sigmas.append(sigma)
        round_rows.append({"round": round_index + 1,
                           "max_epsilon": max(ledger.epsilon() for ledger in ledgers.values())})
    final_epsilon = max(ledger.epsilon() for ledger in ledgers.values())
    if final_epsilon > config.epsilon + 1e-8:
        raise RuntimeError("privacy preview exceeds epsilon budget")
    return {
        "accounting": "per-example replacement-adjacency Gaussian RDP",
        "events_per_client": config.rounds * config.local_steps,
        "total_client_events": config.clients * config.rounds * config.local_steps,
        "sigma_min": min(sigmas), "sigma_max": max(sigmas),
        "final_max_epsilon": final_epsilon, "epsilon_limit": config.epsilon,
        "delta": config.delta, "rounds": round_rows,
    }


def assignment_rows(config: PaperConfig) -> list[dict]:
    profile = fixed_profile(config, {client: MODES for client in range(config.clients)})
    return [{"client": client, "edge": client % config.edges, "fixed_mode": profile[client]}
            for client in sorted(profile)]


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def diagnostic_rows(folder: Path, config: PaperConfig) -> list[dict]:
    with (folder / "round_metrics.csv").open(encoding="utf-8-sig") as handle:
        metrics = list(csv.DictReader(handle))
    rows = []
    for index, metric in enumerate(metrics):
        state = torch.load(folder / f"decision_history_{index:03d}.pt", map_location="cpu", weights_only=False)
        snapshot = torch.load(folder / f"history_round_{index:03d}.pt", map_location="cpu", weights_only=False)
        reference = state["reference"]
        denominator = None if reference is None else float(reference.square().sum() + config.epsilon_j)
        mode_values = {}
        for mode in ("FL_E", "FL_C"):
            units = [unit for unit in snapshot["accepted"] if unit.mode == mode]
            mass = sum(unit.samples for unit in units)
            if reference is None or not units:
                raw_j = None
            else:
                mean = sum((unit.update.double() * unit.samples for unit in units),
                           torch.zeros_like(reference)) / mass
                raw_j = float((mean - reference).square().sum() / denominator)
            historical = state["deviations"].get(mode)
            mode_values[mode] = {
                "mass": mass,
                "history_norm_l2": None if historical is None else float(historical.norm()),
                "history_j": None if historical is None or reference is None else
                    float(historical.square().sum() / denominator),
                "raw_observed_j": raw_j,
            }
        rows.append({
            "round": index + 1,
            "J_pred": metric["predicted_j"] or None,
            "J_obs_cloud": metric["observed_j"] or None,
            "predicted_update_error_l2": metric["predicted_update_error_l2"] or None,
            "history_reference_sha256": None if reference is None else tensor_digest(reference),
            "history_reference_norm_l2": None if reference is None else float(reference.norm()),
            "FL_E_sample_mass": mode_values["FL_E"]["mass"],
            "FL_C_sample_mass": mode_values["FL_C"]["mass"],
            "FL_E_history_deviation_norm_l2": mode_values["FL_E"]["history_norm_l2"],
            "FL_C_history_deviation_norm_l2": mode_values["FL_C"]["history_norm_l2"],
            "FL_E_history_deviation_J": mode_values["FL_E"]["history_j"],
            "FL_C_history_deviation_J": mode_values["FL_C"]["history_j"],
            "FL_E_raw_observed_deviation_J": mode_values["FL_E"]["raw_observed_j"],
            "FL_C_raw_observed_deviation_J": mode_values["FL_C"]["raw_observed_j"],
            "definition": "J_pred uses pre-round EMA mode deviations; J_obs_cloud uses realized Cloud update; mode raw J uses realized per-mode sample-weighted mean update against the same pre-round reference",
        })
    return rows


def make_plots(output: Path, metrics: list[dict], j_rows: list[dict]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rounds = [int(row["round"]) for row in metrics]
    specifications = [
        ("accuracy_vs_round.png", [float(row["test_accuracy"]) for row in metrics],
         "Test accuracy", "Accuracy", "tab:blue"),
        ("loss_vs_round.png", [float(row["test_loss"]) for row in metrics],
         "Test loss", "Cross-entropy", "tab:orange"),
    ]
    for name, values, title, ylabel, color in specifications:
        fig, axis = plt.subplots(figsize=(8, 4.6))
        axis.plot(rounds, values, color=color, linewidth=1.8)
        axis.set(xlabel="Round", ylabel=ylabel, title=f"FMNIST fixed 50% FL_E + 50% FL_C: {title}")
        axis.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(output / name, dpi=180); plt.close(fig)

    j_rounds = [int(row["round"]) for row in j_rows]
    j_pred = [float(row["J_pred"]) if row["J_pred"] not in (None, "") else math.nan for row in j_rows]
    j_obs = [float(row["J_obs_cloud"]) if row["J_obs_cloud"] not in (None, "") else math.nan for row in j_rows]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(j_rounds, j_pred, label="J_pred (pre-round EMA prediction)", linewidth=1.8)
    axis.plot(j_rounds, j_obs, label="J_obs (realized Cloud deviation)", linewidth=1.5, linestyle="--")
    axis.set(xlabel="Round", ylabel="Normalized squared update deviation J",
             title="FMNIST fixed 50% FL_E + 50% FL_C: raw J trajectory")
    axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(output / "J_vs_round.png", dpi=180); plt.close(fig)

    valid = [(float(j["J_pred"]), float(m["test_loss"])) for j, m in zip(j_rows, metrics)
             if j["J_pred"] not in (None, "")]
    fig, axis = plt.subplots(figsize=(6.5, 5))
    axis.scatter([item[0] for item in valid], [item[1] for item in valid], s=18, alpha=.75)
    axis.set(xlabel="J_pred (pre-round EMA prediction)", ylabel="Test loss",
             title="J prediction versus test loss (no causal interpretation)")
    axis.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "J_vs_loss.png", dpi=180); plt.close(fig)


def final_report(output: Path, metrics: list[dict], j_rows: list[dict], preview: dict, audit_result):
    valid_j = [(int(row["round"]), float(row["J_pred"])) for row in j_rows if row["J_pred"] not in (None, "")]
    maximum = max(valid_j, key=lambda item: item[1]) if valid_j else (None, None)
    first, last = metrics[0], metrics[-1]
    text = f"""# FMNIST 固定 50% FL_E + 50% FL_C：100 轮诊断报告

## 配置与定义

- 20 客户端、10 Edge、seed 42；Edge 0–4 全部客户端固定 FL_E，Edge 5–9 全部客户端固定 FL_C。
- 每个 FL_E Edge 包含 2 个客户端；FL_C 为 10 个独立 Cloud 更新单元。
- J_pred 来自每轮训练前冻结的历史参考与模式 EMA 偏差；J_obs_cloud 来自当轮实际整体 Cloud 更新。
- 分模式 raw observed deviation J 是各模式当轮样本加权平均更新相对同一训练前参考的归一化平方偏差，不能与测试损失或准确率等同。

## 结果

- 准确率：{float(first['test_accuracy']):.4f} → {float(last['test_accuracy']):.4f}。
- 测试损失：{float(first['test_loss']):.6f} → {float(last['test_loss']):.6f}。
- J_pred 最大值：{maximum[1] if maximum[1] is not None else 'N/A'}（round {maximum[0] if maximum[0] is not None else 'N/A'}）。
- 最终最大 epsilon：{preview['final_max_epsilon']:.9f}，上限 {preview['epsilon_limit']}；delta={preview['delta']}。
- 完整审计：{'passed' if all(row['status']=='passed' for row in audit_result) else 'failed'}。

## 解释边界

本实验为单种子、固定分配的探索性诊断。J 仅表示更新偏差指标；其升降不能直接解释为模型性能改善或退化。没有同配置的 100 轮纯 FL_C 新运行，因此不得由本实验单独断言混合轨迹优于纯 FL_C；已有结果只能作为后续配对分析的数据来源。
"""
    (output / "experiment_report.md").write_text(text, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"refuse to overwrite existing output: {output}")
    config = PaperConfig.load(args.config)
    if (config.dataset, config.clients, config.edges, config.rounds, config.seeds, config.policies) != (
            "fmnist", 20, 10, 100, (42,), ("fixed_mixed",)):
        raise ValueError("configuration does not match the registered diagnostic scope")
    preview = privacy_preview(config)
    assignment = assignment_rows(config)
    output.mkdir(parents=True, exist_ok=False)
    runner.write_json(output / "config.json", config.to_dict())
    write_csv(output / "client_fixed_assignment.csv", assignment)
    source = source_identity()
    runner.write_json(output / "experiment_manifest.json", {
        "diagnostic_name": DIAGNOSTIC_NAME, "search_protocol": SEARCH_PROTOCOL,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": source["git_revision"], "source_hash": source["source_hash"],
        "config_hash": digest(config.to_dict()), "config_file_sha256": file_digest(args.config),
        "assignment": assignment, "privacy_preview": preview,
    })
    runner.select = diagnostic_select
    print(f"PRECHECK config_hash={digest(config.to_dict())} source_hash={source['source_hash']} ",
          f"sigma={preview['sigma_min']:.12f} final_epsilon={preview['final_max_epsilon']:.9f}", flush=True)
    runner.run(config, output, resume=True)
    audit_result = audit(output)
    runner.write_json(output / "audit.json", audit_result)
    folder = output / "seed_42" / "fixed_mixed"
    shutil.copyfile(folder / "round_metrics.csv", output / "round_metrics.csv")
    with (folder / "round_metrics.csv").open(encoding="utf-8-sig") as handle:
        metrics = list(csv.DictReader(handle))
    j_rows = diagnostic_rows(folder, config)
    write_csv(output / "J_trajectory.csv", j_rows)
    make_plots(output, metrics, j_rows)
    final_report(output, metrics, j_rows, preview, audit_result)
    print(f"COMPLETE output={output} audit=passed", flush=True)


if __name__ == "__main__":
    main()
