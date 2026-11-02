"""Run the preregistered 3-seed, 15-strategy FMNIST J/T campaign."""
from __future__ import annotations

import argparse
import copy
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
from dynfed.paper_storage import atomic_json, digest, file_digest, source_identity
from dynfed.paper_timeline import mode_masses, plan
import dynfed.paper_runner as runner
from experiments.run_fmnist_mode_switch_max_j import CANDIDATES, candidate_manifest, candidate_profile, privacy_preview


DEFAULT_CONFIG = ROOT / "configs" / "fmnist_joint_jt_100r_3seed_20261022.json"
DEFAULT_OUTPUT = ROOT / "out" / "fmnist_joint_jt_100r_3seed_20261022"
TIE_ORDER = tuple(CANDIDATES)
SPECS = {
    **{f"fixed_{name}": {"kind": "fixed", "candidate": name} for name in CANDIDATES},
    "greedy_lambda025": {"kind": "greedy", "lambda": 0.25},
    "greedy_lambda050": {"kind": "greedy", "lambda": 0.50},
    "greedy_lambda075": {"kind": "greedy", "lambda": 0.75},
    "hysteresis_lambda050": {"kind": "hysteresis", "lambda": 0.50, "delta_u": 0.05},
    "periodic_lambda050": {"kind": "periodic", "lambda": 0.50, "period": 5},
    "max_J": {"kind": "max_j"},
    "min_T": {"kind": "min_t"},
    "pure_FL_E": {"kind": "pure", "mode": "FL_E"},
    "pure_FL_EC": {"kind": "pure", "mode": "FL_EC"},
    "pure_FL_C": {"kind": "pure", "mode": "FL_C"},
}
DYNAMIC = tuple(name for name, spec in SPECS.items() if spec["kind"] in {
    "greedy", "hysteresis", "periodic", "max_j", "min_t"})


def _normalize(values: dict[str, float], higher_is_better=True) -> dict[str, float]:
    low, high = min(values.values()), max(values.values())
    if math.isclose(low, high, rel_tol=0.0, abs_tol=1e-15):
        return {name: 0.5 for name in values}
    if higher_is_better:
        return {name: (value-low)/(high-low) for name, value in values.items()}
    return {name: (high-value)/(high-low) for name, value in values.items()}


def _candidate_name(config: PaperConfig, profile: dict[int, str]) -> str | None:
    for name in CANDIDATES:
        if candidate_profile(config, name) == profile:
            return name
    return None


def _evaluate(config, pools, history, traces, samples, payload_mb, start,
              round_index, previous, pending, include_pure=None):
    entries, rejected = [], []
    names = list(CANDIDATES)
    if include_pure:
        names.append(include_pure)
    for order, name in enumerate(names):
        profile = ({client: include_pure.removeprefix("pure_") for client in range(config.clients)}
                   if name == include_pure else candidate_profile(config, name))
        bad = [client for client, mode in profile.items() if mode not in pools[client]]
        if bad:
            rejected.append({"name": name, "reason": "client_mode_infeasible", "clients": bad})
            continue
        schedule = plan(config, profile, traces, samples, payload_mb, start,
                        round_index, previous, pending)
        masses = mode_masses(schedule.accepted)
        _, value, status = history.predict(masses)
        entries.append({"name": name, "order": order, "profile": profile, "schedule": schedule,
                        "T": schedule.finish-start, "J": value, "status": status, "masses": masses})
    return entries, rejected


def make_selector(strategy: str):
    if strategy not in SPECS and strategy != "shared_prefix":
        raise KeyError(strategy)
    spec = {"kind": "prefix"} if strategy == "shared_prefix" else SPECS[strategy]

    def select(config, policy, pools, history, traces, samples, payload_mb, start,
               round_index, previous, pending, force_reselect=False):
        begun = time.perf_counter()
        pure_name = f"pure_{spec['mode']}" if spec["kind"] == "pure" else None
        entries, rejected = _evaluate(config, pools, history, traces, samples, payload_mb,
                                      start, round_index, previous, pending, pure_name)
        by_name = {entry["name"]: entry for entry in entries}
        registered = [entry for entry in entries if entry["name"] in CANDIDATES]
        ready = [entry for entry in registered if entry["J"] is not None]
        j_norm = _normalize({entry["name"]: entry["J"] for entry in ready}) if ready else {}
        t_norm = _normalize({entry["name"]: entry["T"] for entry in registered}, False)
        lam = spec.get("lambda")
        utilities = ({name: lam*j_norm[name] + (1-lam)*t_norm[name] for name in j_norm}
                     if lam is not None else {})
        j_rank = {entry["name"]: rank for rank, entry in enumerate(
            sorted(ready, key=lambda item: (-item["J"], item["order"])), 1)}
        t_rank = {entry["name"]: rank for rank, entry in enumerate(
            sorted(registered, key=lambda item: (item["T"], item["order"])), 1)}
        utility_rank = {name: rank for rank, name in enumerate(
            sorted(utilities, key=lambda item: (-utilities[item], TIE_ORDER.index(item))), 1)}
        current = _candidate_name(config, previous) if previous else None
        reason = None
        if spec["kind"] in {"prefix"} or (strategy in DYNAMIC and round_index < 5):
            selected_name, reason = "P0", "shared_P0_prefix"
        elif spec["kind"] == "fixed":
            selected_name, reason = spec["candidate"], "fixed_candidate"
        elif spec["kind"] == "pure":
            selected_name, reason = pure_name, "fixed_pure_mode"
        elif not ready:
            raise RuntimeError("dynamic candidate utilities unavailable after the shared prefix")
        elif spec["kind"] == "max_j":
            selected_name = min(ready, key=lambda item: (-item["J"], item["order"]))["name"]
            reason = "maximum_J_pred"
        elif spec["kind"] == "min_t":
            selected_name = min(registered, key=lambda item: (item["T"], item["order"]))["name"]
            reason = "minimum_paper_T"
        else:
            best = min(utilities, key=lambda name: (-utilities[name], TIE_ORDER.index(name)))
            if spec["kind"] == "greedy":
                selected_name, reason = best, "maximum_joint_utility"
            elif spec["kind"] == "hysteresis":
                if current not in utilities:
                    selected_name, reason = best, "current_infeasible_or_unscored"
                elif best != current and utilities[best]-utilities[current] > spec["delta_u"]:
                    selected_name, reason = best, "utility_improvement_exceeds_hysteresis"
                else:
                    selected_name, reason = current, "hysteresis_hold"
            elif spec["kind"] == "periodic":
                allowed = (round_index-5) % spec["period"] == 0
                if allowed or current not in utilities:
                    selected_name, reason = best, "periodic_decision_round"
                else:
                    selected_name, reason = current, "periodic_hold"
            else:
                raise AssertionError(spec)
        selected = by_name[selected_name]
        switched = current is not None and selected_name != current
        records = []
        for entry in entries:
            name = entry["name"]
            records.append({"name": name, "profile": entry["profile"], "T": entry["T"],
                            "J": entry["J"], "J_norm": j_norm.get(name),
                            "T_norm": t_norm.get(name), "utility": utilities.get(name),
                            "status": entry["status"], "masses": entry["masses"], "feasible": True,
                            "rank_J": j_rank.get(name), "rank_T": t_rank.get(name),
                            "rank_utility": utility_rank.get(name), "selected": name == selected_name,
                            "tie_order": entry["order"]})
        evaluation = Evaluation(selected["profile"], selected["schedule"], selected["T"],
                                selected["J"], selected["status"], len(selected["masses"]))
        return evaluation, {
            "phase": "common_prefix" if reason == "shared_P0_prefix" else strategy,
            "strategy": strategy, "selection_reason": reason, "lambda": lam,
            "delta_u": spec.get("delta_u"), "period": spec.get("period"),
            "periodic_decision_allowed": (spec["kind"] == "periodic" and
                                           round_index >= 5 and (round_index-5) % spec["period"] == 0),
            "previous_candidate": current, "selected_candidate": selected_name,
            "candidate_switched": switched, "decision_after_round": round_index,
            "applied_round": round_index+1, "evaluated_profiles": len(entries),
            "feasible_combinations": len(entries), "unconstrained_combinations": len(entries)+len(rejected),
            "pareto_front_size": 0, "archive_size": len(ready),
            "selection_sec": time.perf_counter()-begun,
            "search_protocol": "five_candidates_joint_minmax_v1",
            "normalization": "per_round_feasible_minmax_equal_to_0.5",
            "tie_rule": "P0<P1<P2<P3<P4", "assignment_rule": "per_edge_largest_remainder_global_quota_v1",
            "candidates": records, "rejected_candidates": rejected,
            "archive_profiles": [entry["profile"] for entry in ready],
            "selected_profile": selected["profile"], "selected_rank_max_j": j_rank.get(selected_name),
            "selected_rank_utility": utility_rank.get(selected_name), "feasible_pools": pools,
        }
    return select


def inputs(config: PaperConfig, seed: int):
    data = runner.load_data(config, seed)
    partition = runner.partitions(data[1].numpy(), config.clients, config.edges,
                                  config.dirichlet_alpha, seed, config.client_edge_map)
    return data, partition, runner.system_traces(config, seed)


def run_one(config, seed, strategy, destination, data, partition, traces, resume):
    runner.select = make_selector(strategy)
    return runner.run_policy(config, "fixed_mixed", seed, data, partition, traces,
                             destination, resume=resume)


def stop_prefix(config, seed, destination, data, partition, traces):
    runner.select = make_selector("shared_prefix")
    try:
        runner.run_policy(config, "fixed_mixed", seed, data, partition, traces,
                          destination, resume=True, stop_after_round=5)
    except InterruptedError:
        pass
    checkpoint = destination / "checkpoint_round_005.pt"
    if not checkpoint.exists():
        raise RuntimeError("shared prefix did not commit round 5")
    return checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    output = args.output.resolve()
    config = PaperConfig.load(args.config)
    if (config.clients, config.edges, config.rounds, config.seeds) != (100, 10, 100, (42, 43, 44)):
        raise ValueError("campaign requires 100 clients, 10 Edges, 100 rounds, and seeds 42/43/44")
    registration = output / "preregistration.json"
    identity = source_identity()
    if output.exists() and not args.resume:
        raise FileExistsError(f"refuse to overwrite campaign: {output}")
    if not output.exists():
        output.mkdir(parents=True)
        atomic_json(registration, {
            "schema": "fmnist.joint_jt_100r_3seed.v1", "created_utc": datetime.now(timezone.utc).isoformat(),
            "config_path": str(args.config.resolve()), "config": config.to_dict(),
            "config_hash": digest(config.to_dict()), "source_identity": identity,
            "strategies": SPECS, "dynamic_strategies": DYNAMIC,
            "common_prefix": {"candidate": "P0", "rounds": [1, 5]},
            "candidate_manifest": candidate_manifest(config),
            "privacy_preview": privacy_preview(config, 100),
            "paper_T": "event critical path without cross-(Edge,Mode) serialization",
            "online_inputs": ["frozen_history", "system_trace", "feasible_pools", "previous_candidate"],
            "forbidden_online_inputs": ["future_accuracy", "future_loss", "future_updates"],
        })
    else:
        saved = json.loads(registration.read_text(encoding="utf-8"))
        if saved["config_hash"] != digest(config.to_dict()):
            raise RuntimeError("resume config hash mismatch")
        if saved["source_identity"]["source_hash"] != identity["source_hash"]:
            raise RuntimeError("resume source hash mismatch")

    completed = []
    for seed in config.seeds:
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=args.resume)
        data, partition, traces = inputs(config, seed)
        if not (seed_dir / "partition.json").exists():
            atomic_json(seed_dir / "partition.json", partition)
            atomic_json(seed_dir / "traces.json", traces)
        prefix = seed_dir / "shared_P0_prefix"
        prefix.mkdir(exist_ok=args.resume)
        atomic_json(output / "run_status.json", {"status": "running_prefix", "seed": seed,
                                                  "strategy": "shared_P0_prefix", "completed": completed,
                                                  "pid": os.getpid()})
        prefix_checkpoint = stop_prefix(config, seed, prefix, data, partition, traces)
        prefix_hash = file_digest(prefix_checkpoint)
        for strategy, spec in SPECS.items():
            destination = seed_dir / strategy
            atomic_json(output / "run_status.json", {"status": "running", "seed": seed,
                                                      "strategy": strategy, "completed": completed,
                                                      "shared_prefix_sha256": prefix_hash, "pid": os.getpid()})
            if strategy in DYNAMIC and not destination.exists():
                shutil.copytree(prefix, destination)
            resume_run = destination.exists()
            if (destination / "summary.json").exists():
                completed.append([seed, strategy]); continue
            run_one(config, seed, strategy, destination, data, partition, traces, resume_run)
            completed.append([seed, strategy])
    atomic_json(output / "run_status.json", {"status": "training_completed_pending_audit",
                                              "completed": completed, "pid": os.getpid()})
    from experiments.summarize_fmnist_joint_jt import summarize
    summarize(output, config)


if __name__ == "__main__":
    main()
