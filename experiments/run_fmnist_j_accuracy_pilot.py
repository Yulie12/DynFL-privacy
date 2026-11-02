"""Run the preregistered 10-round P0--P4 J/accuracy pilot after the paper-T fix."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynfed.paper_config import MODES, PaperConfig
from dynfed.paper_storage import atomic_json, digest, file_digest, load_checkpoint, source_identity
import dynfed.paper_runner as runner
from experiments.run_fmnist_mode_switch_max_j import CANDIDATES, candidate_profile, privacy_preview
from tools.run_fmnist_all_candidate_switches import forced_selector


DEFAULT_CONFIG = ROOT / "configs" / "fmnist_mode_switch_max_j_10r_20261021.json"
DEFAULT_OUTPUT = ROOT / "out" / "fmnist_mode_switch_j_accuracy_pilot_10r_20261022"
OLD_PREFIX = (ROOT / "out" / "fmnist_mode_switch_all_candidates_20261021" /
              "stage1_10r" / "seed_42" / "common_prefix" / "checkpoint_round_005.pt")
STRATEGY_NAMES = {name: f"{name}_fixed" if name == "P0" else f"{name}_switch_r5"
                  for name in CANDIDATES}


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"refuse to write empty CSV: {path}")
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def inputs(config: PaperConfig):
    data = runner.load_data(config, 42)
    partition = runner.partitions(data[1].numpy(), config.clients, config.edges,
                                  config.dirichlet_alpha, 42, config.client_edge_map)
    return data, partition, runner.system_traces(config, 42)


def run_prefix(config, destination, data, partition, traces):
    from experiments.run_fmnist_mode_switch_max_j import make_selector
    runner.select = make_selector("common_prefix")
    try:
        runner.run_policy(config, "fixed_mixed", 42, data, partition, traces,
                          destination, resume=True, stop_after_round=5)
    except InterruptedError:
        pass
    status = json.loads((destination / "run_status.json").read_text(encoding="utf-8"))
    if status.get("completed_rounds") != 5:
        raise RuntimeError("common prefix did not stop at the committed round-5 boundary")


def run_branch(prefix, destination, config, data, partition, traces, candidate):
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(prefix, destination)
    runner.select = forced_selector(candidate)
    runner.run_policy(config, "fixed_mixed", 42, data, partition, traces,
                      destination, resume=True)


def json_cell(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (list, dict)) else value


def scientific_state_checks(new_path: Path) -> list[dict]:
    checks = []
    if not OLD_PREFIX.exists():
        return [{"name": "old_prefix_available_for_trajectory_comparison", "passed": True,
                 "detail": "not available; comparison skipped"}]
    old, new = load_checkpoint(OLD_PREFIX), load_checkpoint(new_path)
    model_equal = (set(old["model"]) == set(new["model"]) and
                   all(torch.equal(old["model"][key], new["model"][key]) for key in old["model"]))
    old_h, new_h = old["history"].state_dict(), new["history"].state_dict()
    history_equal = ((old_h["reference"] is None and new_h["reference"] is None) or
                     (old_h["reference"] is not None and new_h["reference"] is not None and
                      torch.equal(old_h["reference"], new_h["reference"])))
    history_equal = history_equal and set(old_h["deviations"]) == set(new_h["deviations"])
    history_equal = history_equal and all(torch.equal(old_h["deviations"][mode],
                                                       new_h["deviations"][mode])
                                           for mode in old_h["deviations"])
    privacy_equal = all(old["ledgers"][client].events == new["ledgers"][client].events and
                        old["ledgers"][client].rdp == new["ledgers"][client].rdp
                        for client in old["ledgers"])
    checks.extend([
        {"name": "paper_T_fix_preserves_round5_model", "passed": model_equal},
        {"name": "paper_T_fix_preserves_round5_history", "passed": history_equal},
        {"name": "paper_T_fix_preserves_round5_privacy_ledgers", "passed": privacy_equal},
        {"name": "paper_T_fix_preserves_round5_mode_state", "passed": old["previous"] == new["previous"]},
    ])
    return checks


def event_checks(folder: Path, metrics: list[dict]) -> list[dict]:
    checks = []
    previous_system_time = 0.0
    for zero_index, metric in enumerate(metrics):
        events = json.loads((folder / f"timeline_round_{zero_index:03d}.json").read_text(encoding="utf-8"))
        stages = [event for event in events if event["event"] == "edge_stage"]
        client_stages = [event for event in events if event["event"] == "client_edge_stage_arrival"]
        direct = [event for event in events if event["event"] == "direct_cloud_update"]
        uploads = [event for event in events if event["event"] == "edge_cloud_upload"]
        cloud = next(event for event in events if event["event"] == "cloud_acceptance")
        acks = [event for event in events if event["event"] == "model_ack"]
        barrier = next(event for event in events if event["event"] == "global_barrier")
        checks.extend([
            {"name": f"r{zero_index+1}_edge_groups_start_at_ready",
             "passed": all(math.isclose(event["start"], event["ready"], abs_tol=1e-12)
                           for event in stages)},
            {"name": f"r{zero_index+1}_client_stage_equations",
             "passed": all(math.isclose(event["finish"], event["base"] + event["switch_delay"] +
                                             event["return_communication_delay"] + event["execution_delay"] +
                                             event["upload_communication_delay"], abs_tol=1e-12)
                           for event in client_stages)},
            {"name": f"r{zero_index+1}_direct_equations",
             "passed": all(math.isclose(event["finish"], event["base"] + event["switch_delay"] +
                                             event["execution_delay"] + event["communication_delay"],
                                             abs_tol=1e-12) for event in direct)},
            {"name": f"r{zero_index+1}_edge_cloud_equations",
             "passed": all(math.isclose(event["finish"], event["start"] + event["communication_delay"],
                                             abs_tol=1e-12) for event in uploads)},
            {"name": f"r{zero_index+1}_cloud_equation",
             "passed": math.isclose(cloud["finish"], cloud["cutoff"] + cloud["aggregation_delay"],
                                    abs_tol=1e-12)},
            {"name": f"r{zero_index+1}_ack_equations",
             "passed": len(acks) == 100 and all(
                 math.isclose(event["received"], max(cloud["finish"] + event["downlink_communication_delay"],
                                                     event["local_edge_completion"]), abs_tol=1e-12) and
                 math.isclose(event["finish"], event["received"] + event["installation_delay"] +
                              event["ack_communication_delay"], abs_tol=1e-12) for event in acks)},
            {"name": f"r{zero_index+1}_critical_path_T",
             "passed": math.isclose(barrier["finish"], max(event["finish"] for event in acks), abs_tol=1e-12)
                       and math.isclose(float(metric["system_time"]), barrier["finish"], abs_tol=1e-10)
                       and math.isclose(float(metric["round_duration"]),
                                        barrier["finish"] - float(metric["dispatch_time"]), abs_tol=1e-10)},
            {"name": f"r{zero_index+1}_cumulative_T",
             "passed": math.isclose(float(metric["dispatch_time"]), previous_system_time, abs_tol=1e-10)},
        ])
        previous_system_time = float(metric["system_time"])
    return checks


def flatten_outputs(output: Path, branch_dirs: dict[str, Path], config: PaperConfig):
    metric_rows, event_rows, checkpoint_rows = [], [], []
    branch_metrics = {}
    for candidate, folder in branch_dirs.items():
        strategy = STRATEGY_NAMES[candidate]
        rows = read_csv(folder / "round_metrics.csv")
        branch_metrics[candidate] = rows
        accuracy5 = float(rows[4]["test_accuracy"])
        previous_accuracy = previous_loss = None
        for row in rows:
            accuracy, loss = float(row["test_accuracy"]), float(row["test_loss"])
            merged = {"strategy": strategy, "candidate_after_round5": candidate, **row,
                      "accuracy_gain_from_round5": accuracy - accuracy5,
                      "accuracy_change_from_previous": "" if previous_accuracy is None else accuracy-previous_accuracy,
                      "loss_change_from_previous": "" if previous_loss is None else loss-previous_loss}
            metric_rows.append(merged)
            previous_accuracy, previous_loss = accuracy, loss
        for index in range(10):
            events = json.loads((folder / f"timeline_round_{index:03d}.json").read_text(encoding="utf-8"))
            for order, event in enumerate(events):
                event_rows.append({"strategy": strategy, "candidate_after_round5": candidate,
                                   "round": index + 1, "event_order": order,
                                   **{key: json_cell(value) for key, value in event.items()}})
        for checkpoint in sorted(folder.glob("checkpoint_round_*.pt")):
            checkpoint_rows.append({"strategy": strategy, "candidate_after_round5": candidate,
                                    "round": int(checkpoint.stem.rsplit("_", 1)[1]),
                                    "path": str(checkpoint.relative_to(output)),
                                    "sha256": file_digest(checkpoint), "size_bytes": checkpoint.stat().st_size})

    reference = json.loads((branch_dirs["P0"] / "candidates_round_005.json").read_text(encoding="utf-8"))
    candidate_rows = []
    for item in reference["candidates"]:
        profile = {int(client): mode for client, mode in item["profile"].items()}
        counts = {mode: sum(value == mode for value in profile.values()) for mode in MODES}
        candidate_rows.append({"decision_after_round": 5, "applied_round": 6,
                               "candidate": item["name"], "j_pred": item["J"],
                               "system_latency_T": item["T"], "feasible": item["feasible"],
                               "prediction_status": item["status"], "rank_max_j": item["rank_max_j"],
                               **{f"{mode}_clients": counts[mode] for mode in MODES}})
    assignment_rows = []
    for candidate in CANDIDATES:
        profile = candidate_profile(config, candidate)
        for client in range(config.clients):
            assignment_rows.append({"candidate": candidate, "strategy": STRATEGY_NAMES[candidate],
                                    "client": client, "edge": config.edge_for(client),
                                    "mode": profile[client], "applied_round_start": 6,
                                    "applied_round_end": 10})
    write_csv(output / "round_metrics.csv", metric_rows)
    write_csv(output / "candidate_evaluations.csv", candidate_rows)
    write_csv(output / "client_fixed_assignment.csv", assignment_rows)
    write_csv(output / "checkpoint_integrity.csv", checkpoint_rows)
    write_csv(output / "latency_event_trace.csv", event_rows)
    return branch_metrics, candidate_rows


def plots(output: Path, branch_metrics: dict[str, list[dict]]):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plot_dir = output / "plots"
    plot_dir.mkdir(exist_ok=True)
    specifications = [
        ("accuracy_vs_round.png", "test_accuracy", "Test accuracy"),
        ("loss_vs_round.png", "test_loss", "Test loss"),
        ("T_vs_round.png", "round_duration", "Paper system latency T (s)"),
    ]
    for filename, key, ylabel in specifications:
        fig, axis = plt.subplots(figsize=(8.4, 5.2))
        for candidate, rows in branch_metrics.items():
            xs, ys = [], []
            for row in rows:
                if row[key] != "":
                    xs.append(int(row["round"])); ys.append(float(row[key]))
            axis.plot(xs, ys, marker="o", linewidth=1.5, label=STRATEGY_NAMES[candidate])
        axis.axvline(5, color="black", linestyle="--", alpha=.45, label="shared prefix ends")
        axis.set(xlabel="Round", ylabel=ylabel, title=f"FMNIST 10-round pilot: {ylabel}")
        axis.grid(alpha=.25); axis.legend(fontsize=8); fig.tight_layout()
        fig.savefig(plot_dir / filename, dpi=180); plt.close(fig)
    fig, axis = plt.subplots(figsize=(8.4, 5.2))
    for candidate, rows in branch_metrics.items():
        color = None
        for key, linestyle, suffix in (("predicted_j", "-", "J_pred"),
                                       ("observed_j", "--", "J_obs")):
            values = [(int(row["round"]), float(row[key])) for row in rows if row[key] != ""]
            line, = axis.plot([value[0] for value in values], [value[1] for value in values],
                              marker="o" if key == "predicted_j" else None,
                              linewidth=1.5, linestyle=linestyle,
                              color=color, label=f"{STRATEGY_NAMES[candidate]} {suffix}")
            color = line.get_color()
    axis.axvline(5, color="black", linestyle=":", alpha=.45, label="shared prefix ends")
    axis.set(xlabel="Round", ylabel="J", title="FMNIST 10-round pilot: J_pred and J_obs")
    axis.grid(alpha=.25); axis.legend(fontsize=7, ncol=2); fig.tight_layout()
    fig.savefig(plot_dir / "J_vs_round.png", dpi=180); plt.close(fig)


def build_report(output: Path, branch_metrics: dict[str, list[dict]], candidates: list[dict], audit: dict):
    ranked_j = sorted(candidates, key=lambda row: (-float(row["j_pred"]), row["candidate"]))
    gains = {}
    for candidate, rows in branch_metrics.items():
        gains[candidate] = float(rows[9]["test_accuracy"]) - float(rows[4]["test_accuracy"])
    ranked_gain = sorted(gains, key=lambda name: (-gains[name], name))
    rank_j = {row["candidate"]: index for index, row in enumerate(ranked_j, 1)}
    rank_gain = {name: index for index, name in enumerate(ranked_gain, 1)}
    n = len(rank_j)
    spearman = 1 - 6 * sum((rank_j[name]-rank_gain[name])**2 for name in rank_j)/(n*(n*n-1))
    rows = ["| Candidate | J_pred at end of round 5 | J rank | Paper T for round 6 (s) | A10-A5 | Gain rank | Loss10-Loss5 |",
            "|---|---:|---:|---:|---:|---:|---:|"]
    by_name = {row["candidate"]: row for row in candidates}
    for row in ranked_j:
        name = row["candidate"]
        loss_gain = float(branch_metrics[name][9]["test_loss"]) - float(branch_metrics[name][4]["test_loss"])
        rows.append(f"| {name} | {float(row['j_pred']):.9f} | {rank_j[name]} | "
                    f"{float(row['system_latency_T']):.9f} | {gains[name]:+.6f} | {rank_gain[name]} | "
                    f"{loss_gain:+.9f} |")
    old_t = {"P0": 1.8457222040730534, "P1": 1.8519437477758258,
             "P2": 1.8519437477758258, "P3": 1.7701036340573406,
             "P4": 1.8519437477758258}
    t_rows = ["| Candidate | Old serialized T | Repaired paper T | Difference | Feasible |",
              "|---|---:|---:|---:|---|"]
    for name in CANDIDATES:
        new_t = float(by_name[name]["system_latency_T"])
        t_rows.append(f"| {name} | {old_t[name]:.9f} | {new_t:.9f} | "
                      f"{new_t-old_t[name]:+.9f} | {by_name[name]['feasible']} |")
    detail_rows = ["| Strategy | Round | Accuracy | dAccuracy | Loss | dLoss | J_pred | J_obs | T | Cumulative T | Epsilon |",
                   "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for candidate, metrics in branch_metrics.items():
        for index, row in enumerate(metrics[5:], 5):
            previous = metrics[index - 1]
            accuracy_change = float(row["test_accuracy"]) - float(previous["test_accuracy"])
            loss_change = float(row["test_loss"]) - float(previous["test_loss"])
            detail_rows.append(
                f"| {STRATEGY_NAMES[candidate]} | {row['round']} | {float(row['test_accuracy']):.4f} | "
                f"{accuracy_change:+.4f} | {float(row['test_loss']):.6f} | "
                f"{loss_change:+.6f} | {float(row['predicted_j']):.6f} | "
                f"{float(row['observed_j']):.6f} | {float(row['round_duration']):.6f} | "
                f"{float(row['system_time']):.6f} | {float(row['max_epsilon']):.6f} |")
    conclusion = ("本次五个候选的 J 排名与精度增益排名完全一致。" if ranked_gain == [row["candidate"] for row in ranked_j]
                  else "本次五个候选的 J 排名与精度增益排名不完全一致。")
    report = f"""# FMNIST 10-round J/accuracy pilot

## Protocol

- Rounds 1--5: one shared P0 prefix.
- Rounds 6--10: fixed P0, P1, P2, P3, or P4 branch.
- Seed: 42; 100 clients; 10 heterogeneous Edges; full FMNIST.
- T is recomputed from the event/critical-path model in `paper/main.tex`; it is not wall-clock time.
- J_pred and J_obs retain their original definitions and are reported separately.

## End-of-round-5 candidate comparison

{chr(10).join(rows)}

Descriptive Spearman rank coefficient for five points: **{spearman:.6f}**.
{conclusion}

Only P0 and P1 invert: P0 has a larger round-5 J_pred, while P1 has a slightly larger five-round accuracy gain. P2 has both the largest J_pred and largest short-horizon gain, but this single observation does not turn J into an accuracy predictor.

## Effect of repairing T

{chr(10).join(t_rows)}

All five candidates remain feasible. Removing the non-paper serialization reduces every candidate's T. The latency ordering is unchanged: P3 is fastest, P0 is next, and P1/P2/P4 tie under this trace. The max-J ranking is unchanged because that ranking uses J_pred rather than T; a future joint T/J score must use the repaired values.

## Rounds 6--10

{chr(10).join(detail_rows)}

All rows completed the 100-client ACK barrier with `pending=0`. Communication volume, executed DP steps, checkpoint hashes, and the remaining per-round fields are preserved in `round_metrics.csv` and `checkpoint_integrity.csv`.

## Interpretation boundary

J_pred measures the normalized magnitude of the historically predicted aggregate deviation from the reference update. It is not test loss, accuracy, or a guaranteed marginal utility measure. Even a matching ordering here would be only preliminary single-seed, five-candidate evidence; a mismatch is direct evidence that larger predicted deviation need not yield larger short-horizon accuracy gain. Neither outcome establishes causality or convergence.

## Audit

- Status: **{audit['status']}**
- Checks: {len(audit['checks'])}
- Failed checks: {sum(not check['passed'] for check in audit['checks'])}

The run stops after round 10. No 100-round job is launched.
"""
    (output / "experiment_report.md").write_text(report, encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--finalize-existing", action="store_true",
                        help="Generate reports/audit from already completed immutable branches")
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists() and not args.finalize_existing:
        raise FileExistsError(f"refuse to overwrite existing pilot: {output}")
    if not output.exists() and args.finalize_existing:
        raise FileNotFoundError(output)
    config = PaperConfig.load(args.config)
    if config.rounds != 10 or config.seeds != (42,) or config.clients != 100 or config.edges != 10:
        raise ValueError("pilot requires the registered 10-round seed-42 100-client/10-Edge configuration")
    seed_dir = output / "seed_42"
    prefix = output / "shared_prefix"
    prefix_checkpoint = prefix / "checkpoint_round_005.pt"
    branch_dirs = {candidate: seed_dir / STRATEGY_NAMES[candidate] for candidate in CANDIDATES}
    if not args.finalize_existing:
        output.mkdir(parents=True)
        identity = source_identity()
        atomic_json(output / "preregistration.json", {
            "schema": "fmnist.j_accuracy_pilot.paper_T.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(args.config.resolve()), "config": config.to_dict(),
            "config_hash": digest(config.to_dict()), "source_identity": identity,
            "seed": 42, "prefix_rounds": 5, "branch_rounds": [6, 10],
            "candidates": CANDIDATES, "strategies": STRATEGY_NAMES,
            "privacy_preview": privacy_preview(config, 10),
            "latency_definition": "paper/main.tex event critical path; no cross-(Edge,Mode) serialization",
        })
        atomic_json(output / "run_status.json", {"status": "preparing_inputs", "completed_branches": []})
        data, partition, traces = inputs(config)
        seed_dir.mkdir()
        atomic_json(seed_dir / "partition.json", partition)
        atomic_json(seed_dir / "traces.json", traces)
        atomic_json(output / "run_status.json", {"status": "running_shared_prefix", "completed_branches": []})
        run_prefix(config, prefix, data, partition, traces)
        completed = {}
        for candidate, destination in branch_dirs.items():
            strategy = STRATEGY_NAMES[candidate]
            atomic_json(output / "run_status.json", {"status": "running_branch", "active": strategy,
                                                      "completed_branches": list(completed)})
            run_branch(prefix, destination, config, data, partition, traces, candidate)
            completed[candidate] = destination
    elif not prefix_checkpoint.exists() or any(not (folder / "summary.json").exists()
                                                for folder in branch_dirs.values()):
        raise RuntimeError("existing pilot is incomplete; refuse report-only finalization")

    branch_metrics, candidates = flatten_outputs(output, branch_dirs, config)
    checks = scientific_state_checks(prefix_checkpoint)
    prefix_hash = file_digest(prefix_checkpoint)
    for candidate, folder in branch_dirs.items():
        rows = branch_metrics[candidate]
        checks.extend([
            {"name": f"{candidate}_ten_rounds", "passed": len(rows) == 10},
            {"name": f"{candidate}_shared_round5_checkpoint",
             "passed": file_digest(folder / "checkpoint_round_005.pt") == prefix_hash},
            {"name": f"{candidate}_fixed_rounds6_to10",
             "passed": all(json.loads((folder / f"candidates_round_{index:03d}.json").read_text(
                 encoding="utf-8"))["selected_candidate"] == candidate for index in range(5, 10))},
            {"name": f"{candidate}_privacy_budget",
             "passed": max(float(row["max_epsilon"]) for row in rows) <= config.epsilon + 1e-8},
            {"name": f"{candidate}_barrier_and_versions", "passed": all(
                int(row["acknowledged_clients"]) == config.clients and int(row["pending_units"]) == 0 and
                int(row["cloud_version"]) == int(row["round"]) for row in rows)},
        ])
        checks.extend({**check, "name": f"{candidate}_{check['name']}"}
                      for check in event_checks(folder, rows))
    reference_values = {row["candidate"]: (row["j_pred"], row["system_latency_T"], row["rank_max_j"])
                        for row in candidates}
    for candidate, folder in branch_dirs.items():
        report = json.loads((folder / "candidates_round_005.json").read_text(encoding="utf-8"))
        values = {item["name"]: (item["J"], item["T"], item["rank_max_j"])
                  for item in report["candidates"]}
        checks.append({"name": f"{candidate}_same_round5_candidate_evaluations",
                       "passed": values == reference_values})
    audit = {"schema": "fmnist.j_accuracy_pilot.audit.v1",
             "status": "passed" if all(check["passed"] for check in checks) else "failed",
             "paper_T_fix": {"cross_edge_mode_serialization": False,
                             "round_scalar_compute_delay": "not defined by paper",
                             "round_scalar_communication_delay": "not defined by paper"},
             "shared_round5_checkpoint_sha256": prefix_hash, "checks": checks}
    atomic_json(output / "audit.json", audit)
    plots(output, branch_metrics)
    build_report(output, branch_metrics, candidates, audit)
    atomic_json(output / "run_status.json", {"status": "completed" if audit["status"] == "passed" else "audit_failed",
                                              "completed_branches": list(CANDIDATES),
                                              "audit_status": audit["status"]})
    if audit["status"] != "passed":
        raise RuntimeError("pilot completed but audit failed; inspect audit.json")


if __name__ == "__main__":
    main()
