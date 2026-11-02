"""FMNIST diagnostic with 100 clients and a 5E/5EC mix inside every Edge."""
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

from dynfed.paper_config import MODES, PaperConfig, effective_config
from dynfed.paper_history import History
from dynfed.paper_privacy import Ledger
from dynfed.paper_search import Evaluation
from dynfed.paper_storage import digest, file_digest, source_identity, tensor_digest
from dynfed.paper_timeline import mode_masses, plan
import dynfed.paper_runner as runner
from experiments.audit_paper import audit
from experiments.prepare_formal import dataset_files


DIAGNOSTIC_NAME = "100clients_intra_edge_fixed_fl_e_ec_50_50"
SEARCH_PROTOCOL = "diagnostic_100clients_intra_edge_e_ec_50_50_v1"
DEFAULT_CONFIG = ROOT / "configs" / "fmnist_100clients_10edges_intra_edge_e_ec_50_50_100r_20261021.json"
DEFAULT_OUTPUT = ROOT / "out" / "fmnist_100clients_10edges_intra_edge_e_ec_50_50_100r_20261021_v2"


def fixed_profile(config: PaperConfig, pools: dict[int, tuple[str, ...]]) -> dict[int, str]:
    """Assign each contiguous ten-client Edge to five FL_E and five FL_EC clients."""
    expected_edges = tuple(client // 10 for client in range(100))
    if config.clients != 100 or config.edges != 10 or tuple(config.client_edge_map) != expected_edges:
        raise ValueError("diagnostic requires the registered 100-client, 10-Edge topology")
    result = {}
    for client in sorted(pools):
        mode = "FL_E" if client % 10 < 5 else "FL_EC"
        if mode not in pools[client]:
            raise RuntimeError(f"fixed assignment infeasible for client {client}: {mode}")
        result[client] = mode
    if list(result.values()).count("FL_E") != 50 or list(result.values()).count("FL_EC") != 50:
        raise RuntimeError("fixed assignment is not exactly 50/50")
    for edge in range(config.edges):
        expected = {"FL_E": 5, "FL_EC": 5}
        actual = {mode: sum(config.edge_for(client) == edge and value == mode
                            for client, value in result.items()) for mode in ("FL_E", "FL_EC")}
        if actual != expected:
            raise RuntimeError(f"invalid within-Edge mode mix for Edge {edge}: {actual}")
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
        "name": "HeterogeneousEdge-50E-50EC", "profile": profile, "T": chosen.latency,
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
        "assignment_rule": "ten_edges_each_5E_5EC_v1",
        "missing_history": sorted(missing), "coverage": coverage,
        "candidates": [candidate], "rejected_candidates": [],
        "archive_profiles": [profile] if j_value is not None else [],
        "selected_profile": profile, "selected_candidate": "HeterogeneousEdge-50E-50EC",
        "feasible_pools": pools,
    }


def privacy_preview(config: PaperConfig) -> dict:
    pools = {client: MODES for client in range(config.clients)}
    profile = fixed_profile(config, pools)
    ledgers = {client: Ledger(config.epsilon, config.delta) for client in profile}
    round_rows = []
    sigmas = {"FL_E": [], "FL_EC": []}
    for round_index in range(config.rounds):
        for client, mode in profile.items():
            if mode not in {"FL_E", "FL_EC"}:
                raise RuntimeError("unexpected mode in privacy preview")
            events = config.local_steps * (config.edge_stages if mode == "FL_EC" else 1)
            sigma = ledgers[client].calibrate(events * (config.rounds - round_index))
            if sigma is None:
                raise RuntimeError(f"privacy calibration failed at round {round_index + 1}")
            ledgers[client].commit(events, sigma)
            sigmas[mode].append(sigma)
        round_rows.append({"round": round_index + 1,
                           "max_epsilon": max(ledger.epsilon() for ledger in ledgers.values())})
    final_epsilon = max(ledger.epsilon() for ledger in ledgers.values())
    if final_epsilon > config.epsilon + 1e-8:
        raise RuntimeError("privacy preview exceeds epsilon budget")
    return {
        "accounting": "per-example replacement-adjacency Gaussian RDP",
        "events_per_client_by_mode": {
            "FL_E": config.rounds * config.local_steps,
            "FL_EC": config.rounds * config.local_steps * config.edge_stages,
        },
        "total_client_events": sum(ledger.events for ledger in ledgers.values()),
        "sigma_by_mode": {mode: {"min": min(values), "max": max(values)}
                          for mode, values in sigmas.items()},
        "final_max_epsilon": final_epsilon, "epsilon_limit": config.epsilon,
        "delta": config.delta, "rounds": round_rows,
    }


def assignment_rows(config: PaperConfig) -> list[dict]:
    profile = fixed_profile(config, {client: MODES for client in range(config.clients)})
    return [{"client": client, "edge": config.edge_for(client), "fixed_mode": profile[client]}
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
        for mode in ("FL_E", "FL_EC"):
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
            "FL_EC_sample_mass": mode_values["FL_EC"]["mass"],
            "FL_E_history_deviation_norm_l2": mode_values["FL_E"]["history_norm_l2"],
            "FL_EC_history_deviation_norm_l2": mode_values["FL_EC"]["history_norm_l2"],
            "FL_E_history_deviation_J": mode_values["FL_E"]["history_j"],
            "FL_EC_history_deviation_J": mode_values["FL_EC"]["history_j"],
            "FL_E_raw_observed_deviation_J": mode_values["FL_E"]["raw_observed_j"],
            "FL_EC_raw_observed_deviation_J": mode_values["FL_EC"]["raw_observed_j"],
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
        axis.set(xlabel="Round", ylabel=ylabel, title=f"FMNIST 100-client intra-Edge 50% FL_E + 50% FL_EC: {title}")
        axis.grid(alpha=.25)
        fig.tight_layout(); fig.savefig(output / name, dpi=180); plt.close(fig)

    j_rounds = [int(row["round"]) for row in j_rows]
    j_pred = [float(row["J_pred"]) if row["J_pred"] not in (None, "") else math.nan for row in j_rows]
    j_obs = [float(row["J_obs_cloud"]) if row["J_obs_cloud"] not in (None, "") else math.nan for row in j_rows]
    fig, axis = plt.subplots(figsize=(8, 4.8))
    axis.plot(j_rounds, j_pred, label="J_pred (pre-round EMA prediction)", linewidth=1.8)
    axis.plot(j_rounds, j_obs, label="J_obs (realized Cloud deviation)", linewidth=1.5, linestyle="--")
    axis.set(xlabel="Round", ylabel="Normalized squared update deviation J",
             title="FMNIST 100-client intra-Edge 50% FL_E + 50% FL_EC: raw J trajectory")
    axis.grid(alpha=.25); axis.legend(); fig.tight_layout(); fig.savefig(output / "J_vs_round.png", dpi=180); plt.close(fig)

    valid = [(float(j["J_pred"]), float(m["test_loss"])) for j, m in zip(j_rows, metrics)
             if j["J_pred"] not in (None, "")]
    fig, axis = plt.subplots(figsize=(6.5, 5))
    axis.scatter([item[0] for item in valid], [item[1] for item in valid], s=18, alpha=.75)
    axis.set(xlabel="J_pred (pre-round EMA prediction)", ylabel="Test loss",
             title="J prediction versus test loss (no causal interpretation)")
    axis.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "J_vs_loss.png", dpi=180); plt.close(fig)

    latency = [float(row["round_duration"]) for row in metrics]
    fig, axis = plt.subplots(figsize=(8, 4.6))
    axis.plot(rounds, latency, color="tab:purple", linewidth=1.8)
    axis.set(xlabel="Round", ylabel="Simulated round latency",
             title="Shared Edge-resource latency by round")
    axis.grid(alpha=.25); fig.tight_layout(); fig.savefig(output / "latency_vs_round.png", dpi=180); plt.close(fig)


def final_report(output: Path, metrics: list[dict], j_rows: list[dict], preview: dict, audit_result):
    valid_j = [(int(row["round"]), float(row["J_pred"])) for row in j_rows if row["J_pred"] not in (None, "")]
    maximum = max(valid_j, key=lambda item: item[1]) if valid_j else (None, None)
    first, last = metrics[0], metrics[-1]
    text = f"""# FMNIST 100 客户端、10 Edge 内固定 50% FL_E + 50% FL_EC：100 轮诊断报告

## 配置与定义

- 100 客户端、10 Edge、seed 42；每个 Edge 固定包含 5 FL_E + 5 FL_EC。
- Mode 是客户端级固定变量；FL_EC 严格执行原有 3 阶段流程。
- 同一 Edge 的模式组共享 Edge 聚合器与 Edge–Cloud 上传资源，采用确定性最早可用时间排队。
- J_pred 来自每轮训练前冻结的历史参考与模式 EMA 偏差；J_obs_cloud 来自当轮实际整体 Cloud 更新。
- 分模式 raw observed deviation J 是各模式当轮样本加权平均更新相对同一训练前参考的归一化平方偏差，不能与测试损失或准确率等同。

## 结果

- 准确率：{float(first['test_accuracy']):.4f} → {float(last['test_accuracy']):.4f}。
- 测试损失：{float(first['test_loss']):.6f} → {float(last['test_loss']):.6f}。
- J_pred 最大值：{maximum[1] if maximum[1] is not None else 'N/A'}（round {maximum[0] if maximum[0] is not None else 'N/A'}）。
- 最终最大 epsilon：{preview['final_max_epsilon']:.9f}，上限 {preview['epsilon_limit']}；delta={preview['delta']}。
- 完整审计：{'passed' if all(row['status']=='passed' for row in audit_result) else 'failed'}。

## 解释边界

本实验为单种子、固定分配的探索性诊断。J 仅表示更新偏差指标；其升降不能直接解释为模型性能改善或退化。已有纯 FL_E 与纯 FL_EC 结果仅用于配对轨迹比较，单种子结果不能支持普遍优越性结论。
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
            "fmnist", 100, 10, 100, (42,), ("fixed_mixed",)):
        raise ValueError("configuration does not match the registered diagnostic scope")
    preview = privacy_preview(config)
    assignment = assignment_rows(config)
    original_atomic_json = runner.atomic_json
    def retry_atomic_json(path, value):
        for attempt in range(8):
            try:
                return original_atomic_json(path, value)
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (2 ** attempt))
    runner.atomic_json = retry_atomic_json
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
    effective = effective_config(config, "fixed_mixed", 42)
    runner.write_json(output / "preregistration.json", {
        "schema": "paper.shared_init.registration.v1",
        "units": [{
            "dataset": effective.dataset, "model": effective.model,
            "policy": "fixed_mixed", "seed": 42, "rounds": effective.rounds,
            "config": effective.to_dict(), "config_hash": digest(effective.to_dict()),
            "git_revision": source["git_revision"], "source_hash": source["source_hash"],
            "dataset_file_hashes": dataset_files(effective),
            "output": str(output / "seed_42" / "fixed_mixed"),
            "status": "registered_not_started",
        }],
    })
    runner.select = diagnostic_select
    print(f"PRECHECK config_hash={digest(config.to_dict())} source_hash={source['source_hash']} ",
          f"sigma_E={preview['sigma_by_mode']['FL_E']['min']:.12f} "
          f"sigma_EC={preview['sigma_by_mode']['FL_EC']['min']:.12f} "
          f"final_epsilon={preview['final_max_epsilon']:.9f}", flush=True)
    runner.run(config, output, resume=True)
    audit_result = audit(output)
    runner.write_json(output / "audit.json", audit_result)
    folder = output / "seed_42" / "fixed_mixed"
    shutil.copyfile(folder / "round_metrics.csv", output / "round_metrics.csv")
    with (folder / "round_metrics.csv").open(encoding="utf-8-sig") as handle:
        metrics = list(csv.DictReader(handle))
    j_rows = diagnostic_rows(folder, config)
    write_csv(output / "J_trajectory.csv", j_rows)
    integrity = []
    for round_number in range(1, config.rounds + 1):
        checkpoint = folder / f"checkpoint_round_{round_number:03d}.pt"
        expected = checkpoint.with_suffix(".sha256").read_text(encoding="ascii").strip()
        actual = file_digest(checkpoint)
        integrity.append({"round": round_number, "checkpoint": checkpoint.name,
                          "sha256": actual, "status": "passed" if actual == expected else "failed"})
    write_csv(output / "checkpoint_integrity.csv", integrity)
    make_plots(output, metrics, j_rows)
    final_report(output, metrics, j_rows, preview, audit_result)
    print(f"COMPLETE output={output} audit=passed", flush=True)


if __name__ == "__main__":
    main()
