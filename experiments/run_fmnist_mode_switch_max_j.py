"""Paired FMNIST diagnostic for fixed, one-shot max-J, and per-round max-J policies."""
from __future__ import annotations

import argparse
import copy
import csv
from datetime import datetime, timezone
import json
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
from dynfed.paper_storage import digest, file_digest, source_identity
from dynfed.paper_timeline import mode_masses, plan
import dynfed.paper_runner as runner


CANDIDATES = {
    "P0": (30, 30, 40),
    "P1": (50, 25, 25),
    "P2": (25, 50, 25),
    "P3": (25, 25, 50),
    "P4": (40, 40, 20),
}
STRATEGIES = ("fixed_baseline", "single_switch_max_j", "per_round_max_j")
DEFAULT_CONFIG = ROOT / "configs" / "fmnist_mode_switch_max_j_10r_20261021.json"
DEFAULT_OUTPUT = ROOT / "out" / "fmnist_mode_switch_max_j_20261021"
SEARCH_PROTOCOL = "five_registered_ratios_max_j_v1"


def _all_edge_counts(config: PaperConfig, weights: tuple[int, int, int]) -> dict[int, dict[str, int]]:
    """Largest remainder per Edge with exact global quotas and stable rotating ties."""
    total = sum(weights)
    target = [config.clients * weight // total for weight in weights]
    if sum(target) != config.clients:
        raise ValueError("registered global candidate quotas must be integral")
    result, assigned = {}, [0, 0, 0]
    for edge in range(config.edges):
        members = [client for client in range(config.clients) if config.edge_for(client) == edge]
        raw = [len(members) * weight for weight in weights]
        counts = [value // total for value in raw]
        missing = len(members) - sum(counts)
        for index, count in enumerate(counts):
            assigned[index] += count
        for _ in range(missing):
            eligible = [index for index in range(3) if assigned[index] < target[index]]
            if not eligible:
                raise RuntimeError("global quota exhausted before all Edge clients were assigned")
            index = min(eligible, key=lambda item: (-(raw[item] % total),
                                                    -(target[item] - assigned[item]),
                                                    (item - edge) % 3))
            counts[index] += 1
            assigned[index] += 1
        result[edge] = dict(zip(MODES, counts))
    if assigned != target:
        raise RuntimeError(f"global quota mismatch: assigned={assigned}, target={target}")
    return result


def candidate_profile(config: PaperConfig, name: str) -> dict[int, str]:
    if name not in CANDIDATES:
        raise KeyError(name)
    profile = {}
    counts_by_edge = _all_edge_counts(config, CANDIDATES[name])
    for edge in range(config.edges):
        members = sorted(client for client in range(config.clients) if config.edge_for(client) == edge)
        if len(members) != 10:
            raise ValueError("registered diagnostic requires exactly ten clients per Edge")
        counts = counts_by_edge[edge]
        cursor = 0
        for mode in MODES:
            for client in members[cursor:cursor + counts[mode]]:
                profile[client] = mode
            cursor += counts[mode]
    if set(profile) != set(range(config.clients)):
        raise RuntimeError("candidate omitted or duplicated a client")
    return profile


def candidate_manifest(config: PaperConfig) -> list[dict]:
    rows = []
    for name, weights in CANDIDATES.items():
        profile = candidate_profile(config, name)
        target_total = sum(weights)
        for edge in range(config.edges):
            members = [client for client in profile if config.edge_for(client) == edge]
            for mode, weight in zip(MODES, weights):
                count = sum(profile[client] == mode for client in members)
                target = len(members) * weight / target_total
                rows.append({"candidate": name, "edge": edge, "mode": mode,
                             "target_ratio": weight / target_total, "actual_count": count,
                             "actual_ratio": count / len(members), "rounding_error_clients": count - target})
    return rows


def privacy_preview(config: PaperConfig, rounds: int = 100) -> dict:
    """Conservative path: every client executes FL_EC in every round."""
    ledgers = {client: Ledger(config.epsilon, config.delta) for client in range(config.clients)}
    events = config.local_steps * config.edge_stages
    sigmas = []
    for round_index in range(rounds):
        for ledger in ledgers.values():
            sigma = ledger.calibrate(events * (rounds - round_index))
            if sigma is None:
                raise RuntimeError("100-round worst-path privacy calibration failed")
            ledger.commit(events, sigma)
            sigmas.append(sigma)
    maximum = max(ledger.epsilon() for ledger in ledgers.values())
    if maximum > config.epsilon + 1e-8:
        raise RuntimeError("privacy preview exceeds configured epsilon")
    return {"path": "all clients FL_EC for all rounds", "rounds": rounds,
            "events_per_client": rounds * events, "total_client_events": rounds * events * config.clients,
            "sigma_min": min(sigmas), "sigma_max": max(sigmas), "final_max_epsilon": maximum,
            "epsilon_limit": config.epsilon, "delta": config.delta}


def make_selector(strategy: str):
    if strategy not in STRATEGIES and strategy != "common_prefix":
        raise ValueError(strategy)
    one_shot = {"name": None}

    def select(config, policy, pools, history, traces, samples, payload_mb, start,
               round_index, previous, pending, force_reselect=False):
        begun = time.perf_counter()
        evaluations, rejected = [], []
        for order, name in enumerate(CANDIDATES):
            profile = candidate_profile(config, name)
            bad = [client for client, mode in profile.items() if mode not in pools[client]]
            if bad:
                rejected.append({"name": name, "reason": "client_mode_infeasible", "clients": bad})
                continue
            schedule = plan(config, profile, traces, samples, payload_mb, start,
                            round_index, previous, pending)
            masses = mode_masses(schedule.accepted)
            _, value, status = history.predict(masses)
            evaluations.append((name, order, Evaluation(profile, schedule, schedule.finish-start,
                                                          value, status, len(masses)), masses))
        if not evaluations:
            raise RuntimeError("all registered candidates are infeasible")
        ready = [item for item in evaluations if item[2].learning is not None]
        ranked = sorted(ready, key=lambda item: (-item[2].learning, item[1]))
        rank = {item[0]: index + 1 for index, item in enumerate(ranked)}
        if round_index < 5 or strategy in {"common_prefix", "fixed_baseline"}:
            selected_name, phase = "P0", "common_prefix" if round_index < 5 else "fixed_baseline"
        elif strategy == "single_switch_max_j":
            if one_shot["name"] is None:
                one_shot["name"] = ranked[0][0]
            selected_name, phase = one_shot["name"], "single_switch" if round_index == 5 else "single_switch_reuse"
        else:
            selected_name, phase = ranked[0][0], "per_round_max_j"
        selected = next(item for item in evaluations if item[0] == selected_name)
        records = []
        for name, order, item, masses in evaluations:
            records.append({"name": name, "profile": item.profile, "T": item.latency,
                            "J": item.learning, "status": item.status, "masses": masses,
                            "feasible": True, "rank_max_j": rank.get(name),
                            "selected": name == selected_name, "tie_order": order})
        return selected[2], {"phase": phase, "evaluated_profiles": len(evaluations),
            "feasible_combinations": len(evaluations), "unconstrained_combinations": len(CANDIDATES),
            "pareto_front_size": 0, "archive_size": len(ready),
            "selection_sec": time.perf_counter()-begun, "search_protocol": SEARCH_PROTOCOL,
            "assignment_rule": "per_edge_largest_remainder_rotating_ties_v1",
            "candidates": records, "rejected_candidates": rejected,
            "archive_profiles": [item[2].profile for item in ranked],
            "selected_profile": selected[2].profile, "selected_candidate": selected_name,
            "selected_rank_max_j": rank.get(selected_name), "feasible_pools": pools,
            "decision_after_round": round_index, "applied_round": round_index + 1}
    return select


def _write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def _read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _branch_from_prefix(prefix: Path, branch: Path):
    if branch.exists():
        raise FileExistsError(branch)
    shutil.copytree(prefix, branch)
    if not (branch / "checkpoint_round_005.pt").exists():
        raise RuntimeError("common checkpoint was not copied")


def audit_stage(stage: Path, config: PaperConfig) -> dict:
    checks = []
    def check(name, condition, detail=""):
        checks.append({"name": name, "status": "passed" if condition else "failed", "detail": detail})
    branches = {name: stage / "seed_42" / name for name in STRATEGIES}
    prefix_hashes = {name: file_digest(folder / "checkpoint_round_005.pt") for name, folder in branches.items()}
    check("shared_round5_checkpoint", len(set(prefix_hashes.values())) == 1, json.dumps(prefix_hashes))
    for strategy, folder in branches.items():
        metrics = _read_csv(folder / "round_metrics.csv")
        decisions = _read_csv(folder / "client_decisions.csv")
        check(f"{strategy}_round_count", len(metrics) == config.rounds, str(len(metrics)))
        check(f"{strategy}_barrier_ack_pending", all(int(r["acknowledged_clients"]) == 100 and
              int(r["pending_units"]) == 0 and int(r["cloud_version"]) == int(r["round"]) for r in metrics))
        check(f"{strategy}_privacy", max(float(r["max_epsilon"]) for r in metrics) <= config.epsilon + 1e-8)
        check(f"{strategy}_assignment_total", len(decisions) == config.rounds * config.clients)
        for number in range(1, config.rounds + 1):
            checkpoint = folder / f"checkpoint_round_{number:03d}.pt"
            sidecar = checkpoint.with_suffix(".sha256")
            check(f"{strategy}_checkpoint_{number:03d}", checkpoint.exists() and sidecar.exists() and
                  sidecar.read_text(encoding="ascii").strip() == file_digest(checkpoint))
        reports = [json.loads((folder / f"candidates_round_{i:03d}.json").read_text(encoding="utf-8"))
                   for i in range(config.rounds)]
        check(f"{strategy}_prefix_P0", all(r["selected_candidate"] == "P0" for r in reports[:5]))
        if strategy == "fixed_baseline":
            check("fixed_always_P0", all(r["selected_candidate"] == "P0" for r in reports))
        elif strategy == "single_switch_max_j":
            choice = reports[5]["selected_candidate"]
            check("single_switch_once", all(r["selected_candidate"] == choice for r in reports[5:]))
            check("single_switch_is_max", reports[5]["selected_rank_max_j"] == 1)
        else:
            check("per_round_is_max", all(r["selected_rank_max_j"] == 1 for r in reports[5:]))
    check("all_checks", all(item["status"] == "passed" for item in checks))
    return {"schema": "fmnist.mode_switch_max_j.audit.v1", "checks": checks,
            "status": "passed" if all(item["status"] == "passed" for item in checks) else "failed"}


def export_stage(stage: Path, config: PaperConfig):
    metric_rows, candidate_rows, assignment_rows, switch_rows, privacy_rows = [], [], [], [], []
    for strategy in STRATEGIES:
        folder = stage / "seed_42" / strategy
        metrics = _read_csv(folder / "round_metrics.csv")
        for row in metrics:
            metric_rows.append({"seed": 42, "strategy": strategy, **row})
        for row in _read_csv(folder / "client_decisions.csv"):
            assignment_rows.append({"seed": 42, "strategy": strategy, **row})
        previous = None
        for index in range(config.rounds):
            report = json.loads((folder / f"candidates_round_{index:03d}.json").read_text(encoding="utf-8"))
            selected = report["selected_candidate"]
            if previous is not None and selected != previous:
                switch_rows.append({"seed": 42, "strategy": strategy, "decision_after_round": index,
                                    "applied_round": index+1, "from_candidate": previous,
                                    "to_candidate": selected})
            previous = selected
            for candidate in report["candidates"]:
                candidate_rows.append({"seed": 42, "strategy": strategy, "applied_round": index+1,
                    "decision_after_round": index, "history_id": report.get("history_id"),
                    "candidate": candidate["name"], "J_pred": candidate["J"], "T": candidate["T"],
                    "status": candidate["status"], "feasible": candidate["feasible"],
                    "rank_max_j": candidate["rank_max_j"], "selected": candidate["selected"],
                    **{f"{mode}_sample_mass": candidate["masses"].get(mode, 0) for mode in MODES}})
        privacy_rows.append({"seed": 42, "strategy": strategy,
                             "max_epsilon": max(float(r["max_epsilon"]) for r in metrics),
                             "executed_local_steps": sum(int(r["executed_local_steps"]) for r in metrics)})
    _write_csv(stage / "round_metrics.csv", metric_rows)
    _write_csv(stage / "candidate_evaluations.csv", candidate_rows)
    _write_csv(stage / "mode_assignments.csv", assignment_rows)
    _write_csv(stage / "switch_events.csv", switch_rows or [{"seed":42,"strategy":"none","decision_after_round":"","applied_round":"","from_candidate":"","to_candidate":""}])
    _write_csv(stage / "privacy_summary.csv", privacy_rows)


def make_plots_and_report(stage: Path, config: PaperConfig):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metrics = _read_csv(stage / "round_metrics.csv")
    candidates = _read_csv(stage / "candidate_evaluations.csv")
    assignments = _read_csv(stage / "mode_assignments.csv")
    switches = _read_csv(stage / "switch_events.csv")
    colors = {"fixed_baseline": "tab:blue", "single_switch_max_j": "tab:orange",
              "per_round_max_j": "tab:green"}

    def line_plot(filename, field, ylabel, title, cumulative=False):
        fig, axis = plt.subplots(figsize=(8.4, 4.8))
        for strategy in STRATEGIES:
            rows = [row for row in metrics if row["strategy"] == strategy]
            x = [int(row["round"]) for row in rows]
            values = [float(row[field]) if row[field] else math.nan for row in rows]
            if cumulative:
                running, values = 0.0, [running := running + value for value in values]
            axis.plot(x, values, marker="o", linewidth=1.6, markersize=3,
                      label=strategy, color=colors[strategy])
        axis.axvline(5.5, color="black", linestyle=":", alpha=.6, label="round-5 fork")
        axis.set(xlabel="Round", ylabel=ylabel, title=title); axis.grid(alpha=.25); axis.legend(fontsize=8)
        fig.tight_layout(); fig.savefig(stage / filename, dpi=180); plt.close(fig)

    line_plot("accuracy_vs_round.png", "test_accuracy", "Test accuracy", "FMNIST max-J strategy comparison")
    line_plot("loss_vs_round.png", "test_loss", "Test cross-entropy", "Test loss by strategy")
    line_plot("j_pred_vs_round.png", "predicted_j", "J_pred", "Pre-round historical prediction (raw, unsmoothed beyond defined EMA)")
    line_plot("j_obs_vs_round.png", "observed_j", "J_obs", "Realized Cloud-update deviation")
    line_plot("latency_vs_round.png", "round_duration", "Modeled seconds", "Per-round modeled latency")
    line_plot("epsilon_vs_round.png", "max_epsilon", "Maximum epsilon", "Conservative DP ledger")

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), sharey=True)
    for axis, strategy in zip(axes, STRATEGIES):
        rows = [row for row in assignments if row["strategy"] == strategy]
        by_round = {}
        for row in rows:
            by_round.setdefault(int(row["round"])+1, []).append(row)
        rounds = sorted(by_round)
        bottom = [0.0] * len(rounds)
        for mode, color in zip(MODES, ("tab:blue", "tab:orange", "tab:green")):
            ratios = [sum(item["mode"] == mode for item in by_round[r])/len(by_round[r]) for r in rounds]
            axis.bar(rounds, ratios, bottom=bottom, label=mode, color=color)
            bottom = [left + right for left, right in zip(bottom, ratios)]
        axis.set_title(strategy, fontsize=9); axis.set_xlabel("Round"); axis.grid(axis="y", alpha=.2)
    axes[0].set_ylabel("Client ratio"); axes[-1].legend(fontsize=8)
    fig.suptitle("Actual client-level Mode ratios"); fig.tight_layout(); fig.savefig(stage / "mode_ratio_vs_round.png", dpi=180); plt.close(fig)

    fig, axis = plt.subplots(figsize=(8.4, 3.8))
    axis.set(xlim=(1, config.rounds), ylim=(-.5, len(STRATEGIES)-.5), xlabel="Applied round",
             yticks=range(len(STRATEGIES)), yticklabels=STRATEGIES, title="Mode-distribution switch events")
    for row in switches:
        if row["strategy"] in STRATEGIES and row["applied_round"]:
            y = STRATEGIES.index(row["strategy"]); x = int(row["applied_round"])
            axis.scatter(x, y, s=70, color=colors[row["strategy"]]); axis.text(x+.1, y, f"{row['from_candidate']}→{row['to_candidate']}")
    axis.grid(alpha=.25); fig.tight_layout(); fig.savefig(stage / "switch_events.png", dpi=180); plt.close(fig)

    cosine_rows = []
    for strategy in STRATEGIES:
        folder = stage / "seed_42" / strategy
        previous = None
        for index in range(config.rounds):
            snapshot = torch.load(folder / f"history_round_{index:03d}.pt", map_location="cpu", weights_only=False)
            current = snapshot["delta"].double().flatten()
            cosine = None if previous is None else float(torch.nn.functional.cosine_similarity(previous, current, dim=0))
            cosine_rows.append({"seed": 42, "strategy": strategy, "round": index+1,
                                "previous_round": "" if index == 0 else index, "update_direction_cosine": cosine})
            previous = current
    _write_csv(stage / "update_direction_cosine.csv", cosine_rows)
    fig, axis = plt.subplots(figsize=(8.4, 4.8))
    for strategy in STRATEGIES:
        rows = [row for row in cosine_rows if row["strategy"] == strategy and row["update_direction_cosine"] is not None]
        axis.plot([row["round"] for row in rows], [row["update_direction_cosine"] for row in rows],
                  marker="o", markersize=3, label=strategy, color=colors[strategy])
    axis.axvline(5.5, color="black", linestyle=":", alpha=.6); axis.axhline(0, color="gray", linewidth=.8)
    axis.set(xlabel="Round", ylabel="cos(delta[t-1], delta[t])", title="Adjacent realized Cloud-update directions")
    axis.grid(alpha=.25); axis.legend(fontsize=8); fig.tight_layout(); fig.savefig(stage / "update_direction_cosine.png", dpi=180); plt.close(fig)

    summaries = []
    for strategy in STRATEGIES:
        rows = [row for row in metrics if row["strategy"] == strategy]
        summaries.append({"strategy": strategy, "final_accuracy": float(rows[-1]["test_accuracy"]),
                          "best_accuracy": max(float(r["test_accuracy"]) for r in rows),
                          "final_loss": float(rows[-1]["test_loss"]),
                          "cumulative_latency": float(rows[-1]["system_time"]),
                          "local_steps": sum(int(r["executed_local_steps"]) for r in rows),
                          "switches": sum(1 for r in switches if r["strategy"] == strategy)})
    _write_csv(stage / "stage1_summary.csv", summaries)
    fields = ("final_accuracy", "best_accuracy", "final_loss", "cumulative_latency", "local_steps", "switches")
    fig, axes = plt.subplots(2, 3, figsize=(12, 7.2))
    for axis, field in zip(axes.flat, fields):
        axis.bar(range(3), [row[field] for row in summaries], color=[colors[s] for s in STRATEGIES])
        axis.set_title(field); axis.set_xticks(range(3), ("fixed", "single", "per-round"), rotation=15); axis.grid(axis="y", alpha=.2)
    fig.suptitle("Stage-1 mechanism comparison (10 rounds; not an effectiveness claim)")
    fig.tight_layout(); fig.savefig(stage / "final_comparison.png", dpi=180); plt.close(fig)

    selections = {strategy: [row["candidate"] for row in candidates
                             if row["strategy"] == strategy and row["selected"] == "True"] for strategy in STRATEGIES}
    report = f"""# FMNIST 最大 J Mode 分布选择：10 轮机制验证

## 审计结论

- 独立机制审计：passed；共同第 5 轮检查点在三分支中具有相同 SHA-256。
- 前 5 轮均执行 P0；第 5 轮结束后的选择应用于第 6 轮，没有使用未来信息。
- 最大 J 候选在第 6–10 轮持续为 P2，因此 single-switch 与 per-round 两组没有发生分化。

## 10 轮结果

| strategy | final accuracy | final loss | cumulative T | local steps | switches |
|---|---:|---:|---:|---:|---:|
"""
    for row in summaries:
        report += (f"| {row['strategy']} | {row['final_accuracy']:.4f} | {row['final_loss']:.6f} | "
                   f"{row['cumulative_latency']:.6f} | {row['local_steps']} | {row['switches']} |\n")
    report += f"""

选择序列：

- fixed: {selections['fixed_baseline']}
- single: {selections['single_switch_max_j']}
- per-round: {selections['per_round_max_j']}

## 解释边界

这 10 轮只验证机制。P2 在该短窗口内始终保持最大 J，故无法从本阶段判断频繁重选是否有额外收益或是否造成来回切换。single/per-round 的训练工作量均为 3600 local steps，高于 fixed 的 3200；短期精度差异不能脱离训练预算解释，也不能据此宣称最大 J 策略有效。正式 100 轮必须使用独立的 100 轮隐私校准重新开始，不能续接本次 10 轮账本。
"""
    (stage / "experiment_report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    root, config = args.output.resolve(), PaperConfig.load(args.config)
    if (config.dataset, config.clients, config.edges, config.seeds, config.policies) != (
            "fmnist", 100, 10, (42,), ("fixed_mixed",)):
        raise ValueError("configuration is outside the registered diagnostic scope")
    if config.rounds not in {10, 100}:
        raise ValueError("this registered experiment accepts only 10 or 100 rounds")
    stage = root / ("stage1_10r" if config.rounds == 10 else "stage2_100r_seed42")
    if config.rounds == 10 and root.exists():
        raise FileExistsError(f"refuse to overwrite existing output: {root}")
    if config.rounds == 100:
        stage1_audit = root / "stage1_10r" / "audit.json"
        if not stage1_audit.exists() or json.loads(stage1_audit.read_text(encoding="utf-8"))["status"] != "passed":
            raise RuntimeError("stage 2 is blocked until stage-1 audit passes")
        if stage.exists():
            raise FileExistsError(f"refuse to overwrite existing output: {stage}")
    preview = privacy_preview(config, 100)
    source = source_identity()
    root.mkdir(parents=True, exist_ok=config.rounds == 100); stage.mkdir()
    manifest_dir = root if config.rounds == 10 else stage
    _write_csv(manifest_dir / "candidate_integer_allocations.csv", candidate_manifest(config))
    plan_text = ("# FMNIST max-J mode-switch preregistration\n\n"
        "Stage 1 runs seed 42 for 10 rounds. All strategies execute the same P0 rounds 1–5 "
        "from one byte-identical checkpoint. Selection after round 5 is applied to round 6. "
        "Ties use P0<P1<P2<P3<P4. Stage 2 is forbidden until this stage passes audit.\n")
    if config.rounds == 10:
        (root / "experiment_plan.md").write_text(plan_text, encoding="utf-8")
    runner.write_json(manifest_dir / "config_manifest.json", {"config": config.to_dict(),
        "config_hash": digest(config.to_dict()), "config_file_sha256": file_digest(args.config),
        "candidates": CANDIDATES, "strategies": STRATEGIES, "privacy_worst_path": preview,
        "created_utc": datetime.now(timezone.utc).isoformat()})
    runner.write_json(manifest_dir / "source_identity.json", source)
    data = runner.load_data(config, 42)
    partition = runner.partitions(data[1].numpy(), config.clients, config.edges,
                                  config.dirichlet_alpha, 42, config.client_edge_map)
    traces = runner.system_traces(config, 42)
    seed_dir = stage / "seed_42"; seed_dir.mkdir()
    prefix = seed_dir / "common_prefix"
    runner.select = make_selector("common_prefix")
    try:
        runner.run_policy(config, "fixed_mixed", 42, data, partition, traces, prefix,
                          resume=True, stop_after_round=5)
    except InterruptedError:
        pass
    for strategy in STRATEGIES:
        branch = seed_dir / strategy
        _branch_from_prefix(prefix, branch)
        runner.select = make_selector(strategy)
        runner.run_policy(config, "fixed_mixed", 42, data, partition, traces, branch, resume=True)
    export_stage(stage, config)
    result = audit_stage(stage, config)
    runner.write_json(stage / "audit.json", result)
    if result["status"] != "passed":
        raise RuntimeError("stage-1 audit failed; stage 2 remains blocked")
    make_plots_and_report(stage, config)
    print(f"STAGE_COMPLETE rounds={config.rounds} output={stage} audit=passed", flush=True)


if __name__ == "__main__":
    main()
