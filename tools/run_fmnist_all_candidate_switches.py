"""Queue fixed P0--P4 switch controls without changing registered FL source files."""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynfed.paper_config import MODES, PaperConfig
from dynfed.paper_search import Evaluation
from dynfed.paper_storage import atomic_json, file_digest, source_identity
from dynfed.paper_timeline import mode_masses, plan
import dynfed.paper_runner as runner
from experiments.run_fmnist_mode_switch_max_j import CANDIDATES, candidate_profile, make_selector


OUTPUT = ROOT / "out" / "fmnist_mode_switch_all_candidates_20261021"
PARENT = ROOT / "out" / "fmnist_mode_switch_max_j_20261021"
CONFIG10 = ROOT / "configs" / "fmnist_mode_switch_max_j_10r_20261021.json"
CONFIG100 = ROOT / "configs" / "fmnist_mode_switch_max_j_100r_seed42_20261021.json"
STRATEGIES = tuple(f"fixed_{name}" for name in CANDIDATES)


def forced_selector(candidate_name: str):
    if candidate_name not in CANDIDATES:
        raise KeyError(candidate_name)

    def select(config, policy, pools, history, traces, samples, payload_mb, start,
               round_index, previous, pending, force_reselect=False):
        begun = time.perf_counter()
        entries, rejected = [], []
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
            entries.append((name, order, Evaluation(profile, schedule, schedule.finish-start,
                                                       value, status, len(masses)), masses))
        ready = [item for item in entries if item[2].learning is not None]
        ranked = sorted(ready, key=lambda item: (-item[2].learning, item[1]))
        ranks = {item[0]: index+1 for index, item in enumerate(ranked)}
        selected_name = "P0" if round_index < 5 else candidate_name
        selected = next(item for item in entries if item[0] == selected_name)
        records = [{"name": name, "profile": item.profile, "T": item.latency,
                    "J": item.learning, "status": item.status, "masses": masses,
                    "feasible": True, "rank_max_j": ranks.get(name),
                    "selected": name == selected_name, "tie_order": order}
                   for name, order, item, masses in entries]
        return selected[2], {"phase": "common_prefix" if round_index < 5 else f"fixed_{candidate_name}",
            "evaluated_profiles": len(entries), "feasible_combinations": len(entries),
            "unconstrained_combinations": len(CANDIDATES), "pareto_front_size": 0,
            "archive_size": len(ready), "selection_sec": time.perf_counter()-begun,
            "search_protocol": "fixed_registered_candidate_after_round5_v1",
            "assignment_rule": "per_edge_largest_remainder_global_quota_v1",
            "candidates": records, "rejected_candidates": rejected,
            "archive_profiles": [item[2].profile for item in ranked],
            "selected_profile": selected[2].profile, "selected_candidate": selected_name,
            "selected_rank_max_j": ranks.get(selected_name), "feasible_pools": pools,
            "decision_after_round": round_index, "applied_round": round_index+1}
    return select


def write_status(**values):
    current = {}
    path = OUTPUT / "queue_status.json"
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
    current.update(values, pid=os.getpid(), updated_utc=datetime.now(timezone.utc).isoformat())
    atomic_json(path, current)


def inputs(config: PaperConfig):
    data = runner.load_data(config, 42)
    partition = runner.partitions(data[1].numpy(), config.clients, config.edges,
                                  config.dirichlet_alpha, 42, config.client_edge_map)
    return data, partition, runner.system_traces(config, 42)


def stop_at_five(config, destination, data, partition, traces):
    runner.select = make_selector("common_prefix")
    try:
        runner.run_policy(config, "fixed_mixed", 42, data, partition, traces,
                          destination, resume=True, stop_after_round=5)
    except InterruptedError:
        pass


def branch(prefix: Path, destination: Path, config, data, partition, traces, candidate):
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(prefix, destination)
    runner.select = forced_selector(candidate)
    runner.run_policy(config, "fixed_mixed", 42, data, partition, traces, destination, resume=True)


def audit_stage(stage: Path, rounds: int, names: tuple[str, ...], reference_hash=None):
    checks = []
    hashes = {}
    for name in names:
        folder = stage / "seed_42" / f"fixed_{name}"
        checkpoint = folder / "checkpoint_round_005.pt"
        hashes[name] = file_digest(checkpoint)
        metrics = list(csv.DictReader((folder / "round_metrics.csv").open(encoding="utf-8-sig")))
        reports = [json.loads((folder / f"candidates_round_{index:03d}.json").read_text(encoding="utf-8"))
                   for index in range(rounds)]
        checks.extend([
            {"name": f"{name}_rounds", "passed": len(metrics) == rounds},
            {"name": f"{name}_prefix", "passed": all(row["selected_candidate"] == "P0" for row in reports[:5])},
            {"name": f"{name}_fixed_after5", "passed": all(row["selected_candidate"] == name for row in reports[5:])},
            {"name": f"{name}_barrier", "passed": all(int(row["acknowledged_clients"]) == 100 and
                int(row["pending_units"]) == 0 and int(row["cloud_version"]) == int(row["round"]) for row in metrics)},
            {"name": f"{name}_privacy", "passed": max(float(row["max_epsilon"]) for row in metrics) <= 8.00000001},
        ])
    expected = reference_hash or next(iter(hashes.values()))
    checks.append({"name": "shared_round5_checkpoint", "passed": all(value == expected for value in hashes.values()),
                   "hashes": hashes, "expected": expected})
    result = {"status": "passed" if all(row["passed"] for row in checks) else "failed", "checks": checks}
    atomic_json(stage / "audit.json", result)
    if result["status"] != "passed":
        raise RuntimeError(f"audit failed: {stage}")


def wait_for_parent():
    audit = PARENT / "stage2_100r_seed42" / "audit.json"
    while True:
        if audit.exists():
            result = json.loads(audit.read_text(encoding="utf-8"))
            if result.get("status") == "passed":
                return
            raise RuntimeError("parent 100-round audit did not pass")
        write_status(status="waiting_for_parent_100r_audit")
        time.sleep(30)


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--resume-queue", action="store_true")
    parser.add_argument("--run-stage1-now", action="store_true",
                        help="Run the 10-round P0-P4 matrix now, then pause before 100 rounds")
    parser.add_argument("--continue-stage2", action="store_true",
                        help="Wait for the new 10-round audit, then run P0-P4 for 100 rounds")
    args = parser.parse_args()
    OUTPUT = args.output.resolve()
    if OUTPUT.exists() and not args.resume_queue:
        raise FileExistsError(f"refuse to overwrite queue output: {OUTPUT}")
    if args.resume_queue:
        status_path = OUTPUT / "queue_status.json"
        allowed = {"waiting_for_parent_100r_audit", "running_10r", "stage1_10r_passed_waiting_for_100r"}
        if not status_path.exists() or json.loads(status_path.read_text(encoding="utf-8")).get("status") not in allowed:
            raise RuntimeError("queue resume state is not safe for the requested continuation")
    else:
        OUTPUT.mkdir(parents=True)
        atomic_json(OUTPUT / "preregistration.json", {
            "schema": "fmnist.fixed_candidate_switches.v1", "seed": 42,
            "prefix_rounds": 5, "switch_applied_round": 6, "candidates": CANDIDATES,
            "stage_order": ["wait_parent", "10_round_P0_to_P4", "100_round_P1_P3_P4"],
            "existing_100r_references": {"P0": str(PARENT / "stage2_100r_seed42/seed_42/fixed_baseline"),
                                         "P2": str(PARENT / "stage2_100r_seed42/seed_42/single_switch_max_j")},
            "source_identity": source_identity(), "created_utc": datetime.now(timezone.utc).isoformat(),
        })
    if args.continue_stage2:
        audit10 = OUTPUT / "stage1_10r" / "audit.json"
        while not audit10.exists():
            time.sleep(15)
        if json.loads(audit10.read_text(encoding="utf-8")).get("status") != "passed":
            raise RuntimeError("10-round candidate audit did not pass")
    else:
        write_status(status="waiting_for_parent_100r_audit")
        if not args.run_stage1_now:
            wait_for_parent()

    if not args.continue_stage2:
        config10 = PaperConfig.load(CONFIG10)
        stage10 = OUTPUT / "stage1_10r"; seed10 = stage10 / "seed_42"; seed10.mkdir(parents=True)
        data10, partition10, traces10 = inputs(config10)
        prefix10 = seed10 / "common_prefix"
        write_status(status="running_10r", active="common_prefix")
        stop_at_five(config10, prefix10, data10, partition10, traces10)
        for name in CANDIDATES:
            write_status(status="running_10r", active=f"fixed_{name}")
            branch(prefix10, seed10 / f"fixed_{name}", config10, data10, partition10, traces10, name)
        audit_stage(stage10, 10, tuple(CANDIDATES))
        if args.run_stage1_now:
            write_status(status="stage1_10r_passed_waiting_for_100r")
            return
        write_status(status="stage1_10r_passed")

    config100 = PaperConfig.load(CONFIG100)
    stage100 = OUTPUT / "stage2_100r"; seed100 = stage100 / "seed_42"; seed100.mkdir(parents=True)
    data100, partition100, traces100 = inputs(config100)
    prefix100 = PARENT / "stage2_100r_seed42" / "seed_42" / "common_prefix"
    reference_hash = file_digest(prefix100 / "checkpoint_round_005.pt")
    for name in CANDIDATES:
        write_status(status="running_100r", active=f"fixed_{name}")
        branch(prefix100, seed100 / f"fixed_{name}", config100, data100, partition100, traces100, name)
    audit_stage(stage100, 100, tuple(CANDIDATES), reference_hash)
    atomic_json(stage100 / "reference_branches.json", {
        "P0": str(PARENT / "stage2_100r_seed42/seed_42/fixed_baseline"),
        "P2": str(PARENT / "stage2_100r_seed42/seed_42/single_switch_max_j"),
        "shared_checkpoint_sha256": reference_hash,
    })
    write_status(status="completed", active=None)


if __name__ == "__main__":
    main()
