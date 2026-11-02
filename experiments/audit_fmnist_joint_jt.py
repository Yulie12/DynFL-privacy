"""Independent audit for the FMNIST J/T 100-round campaign."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import sys

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from dynfed.paper_config import MODES, PaperConfig
from dynfed.paper_runner import history_identity
from dynfed.paper_storage import atomic_json, digest, file_digest
from dynfed.paper_timeline import mode_masses, plan
from experiments.run_fmnist_joint_jt_100r import CANDIDATES, DYNAMIC, SPECS, TIE_ORDER, _normalize


def _csv(path):
    with Path(path).open(encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def _profile(report):
    return {int(client): mode for client, mode in report["selected_profile"].items()}


def _candidate_values(report):
    return {item["name"]: item for item in report["candidates"] if item["name"] in CANDIDATES}


def audit(root: Path, config: PaperConfig):
    root = Path(root)
    checks, failures = [], []

    def record(name, passed, detail=None):
        row = {"name": name, "passed": bool(passed)}
        if detail is not None:
            row["detail"] = detail
        checks.append(row)
        if not passed and len(failures) < 100:
            failures.append(row)

    for seed in config.seeds:
        seed_dir = root / f"seed_{seed}"
        partition = json.loads((seed_dir / "partition.json").read_text(encoding="utf-8"))
        traces = json.loads((seed_dir / "traces.json").read_text(encoding="utf-8"))
        samples = {client: len(indices) for client, indices in enumerate(partition)}
        prefix_hash = file_digest(seed_dir / "shared_P0_prefix" / "checkpoint_round_005.pt")
        for strategy, spec in SPECS.items():
            folder = seed_dir / strategy
            metrics = _csv(folder / "round_metrics.csv")
            decisions = _csv(folder / "client_decisions.csv")
            reports = [json.loads((folder / f"candidates_round_{index:03d}.json").read_text(encoding="utf-8"))
                       for index in range(config.rounds)]
            initial_state = torch.load(folder / "initial_model.pt", map_location="cpu", weights_only=True)
            payload_mb = sum(value.numel()*value.element_size() for value in initial_state.values())/1e6
            record(f"seed{seed}_{strategy}_round_count", len(metrics) == 100)
            record(f"seed{seed}_{strategy}_decision_count", len(decisions) == 100 * config.clients)
            record(f"seed{seed}_{strategy}_privacy", max(float(row["max_epsilon"]) for row in metrics)
                   <= config.epsilon + 1e-8)
            record(f"seed{seed}_{strategy}_barrier", all(
                int(row["acknowledged_clients"]) == config.clients and
                int(row["pending_units"]) == 0 and int(row["cloud_version"]) == int(row["round"])
                for row in metrics))
            if strategy in DYNAMIC:
                record(f"seed{seed}_{strategy}_shared_checkpoint",
                       file_digest(folder / "checkpoint_round_005.pt") == prefix_hash)
            previous, logical_time, switch_count = {}, 0.0, 0
            selected_names = []
            all_t_ok = all_j_ok = all_utility_ok = all_choice_ok = all_timeline_ok = True
            allocations_ok = True
            for index, (metric, report) in enumerate(zip(metrics, reports)):
                profile = _profile(report)
                selected_names.append(report["selected_candidate"])
                actual = {int(row["client"]): row["mode"] for row in decisions if int(row["round"]) == index}
                allocations_ok &= profile == actual
                schedule = plan(config, profile, traces[index], samples, payload_mb,
                                logical_time, index, previous, ())
                all_t_ok &= (math.isclose(schedule.finish-logical_time, float(metric["round_duration"]), abs_tol=1e-9)
                             and math.isclose(schedule.finish, float(metric["system_time"]), abs_tol=1e-9))
                events = json.loads((folder / f"timeline_round_{index:03d}.json").read_text(encoding="utf-8"))
                stages = [event for event in events if event["event"] == "edge_stage"]
                acks = [event for event in events if event["event"] == "model_ack"]
                all_timeline_ok &= (all(math.isclose(event["start"], event["ready"], abs_tol=1e-12)
                                        for event in stages) and len(acks) == config.clients and
                                    math.isclose(max(event["finish"] for event in acks), schedule.finish, abs_tol=1e-9))
                state = torch.load(folder / f"decision_history_{index:03d}.pt", map_location="cpu", weights_only=False)
                all_j_ok &= report.get("history_id") == history_identity(state)
                candidates = _candidate_values(report)
                for name, item in candidates.items():
                    candidate_profile = {int(client): mode for client, mode in item["profile"].items()}
                    candidate_schedule = plan(config, candidate_profile, traces[index], samples, payload_mb,
                                              logical_time, index, previous, ())
                    all_t_ok &= math.isclose(item["T"], candidate_schedule.finish-logical_time, abs_tol=1e-9)
                    masses = mode_masses(candidate_schedule.accepted)
                    active = {mode: mass for mode, mass in masses.items() if mass > 0}
                    if state["reference"] is None or not active or not set(active).issubset(state["deviations"]):
                        all_j_ok &= item["J"] is None
                    else:
                        total = sum(active.values())
                        deviation = sum((state["deviations"][mode] * (mass/total)
                                         for mode, mass in active.items()),
                                        torch.zeros_like(state["reference"]))
                        value = float(deviation.square().sum() /
                                      (state["reference"].square().sum()+config.epsilon_j))
                        all_j_ok &= math.isclose(value, item["J"], rel_tol=1e-9, abs_tol=1e-9)
                ready = {name: item for name, item in candidates.items() if item["J"] is not None}
                if ready:
                    jn = _normalize({name: item["J"] for name, item in ready.items()})
                    tn = _normalize({name: item["T"] for name, item in candidates.items()}, False)
                    lam = report.get("lambda")
                    for name, item in candidates.items():
                        all_utility_ok &= math.isclose(item["T_norm"], tn[name], abs_tol=1e-12)
                        if name in ready:
                            all_utility_ok &= math.isclose(item["J_norm"], jn[name], abs_tol=1e-12)
                            if lam is not None:
                                expected_u = lam*jn[name] + (1-lam)*tn[name]
                                all_utility_ok &= math.isclose(item["utility"], expected_u, abs_tol=1e-12)
                if strategy in DYNAMIC:
                    all_choice_ok &= (report["selected_candidate"] == "P0" if index < 5 else True)
                    if index >= 5 and spec["kind"] == "max_j":
                        expected = min(ready, key=lambda name: (-ready[name]["J"], TIE_ORDER.index(name)))
                        all_choice_ok &= report["selected_candidate"] == expected
                    elif index >= 5 and spec["kind"] == "min_t":
                        expected = min(candidates, key=lambda name: (candidates[name]["T"], TIE_ORDER.index(name)))
                        all_choice_ok &= report["selected_candidate"] == expected
                    elif index >= 5 and spec["kind"] == "periodic":
                        if (index-5) % spec["period"] != 0:
                            all_choice_ok &= report["selected_candidate"] == selected_names[index-1]
                        else:
                            expected = min(ready, key=lambda name: (-ready[name]["utility"], TIE_ORDER.index(name)))
                            all_choice_ok &= report["selected_candidate"] == expected
                    elif index >= 5 and spec["kind"] == "hysteresis":
                        current_name = selected_names[index-1]
                        best = min(ready, key=lambda name: (-ready[name]["utility"], TIE_ORDER.index(name)))
                        improvement = candidates[best]["utility"]-candidates[current_name]["utility"]
                        expected = best if best != current_name and improvement > spec["delta_u"] else current_name
                        all_choice_ok &= report["selected_candidate"] == expected
                    elif index >= 5 and spec["kind"] == "greedy":
                        expected = min(ready, key=lambda name: (-ready[name]["utility"], TIE_ORDER.index(name)))
                        all_choice_ok &= report["selected_candidate"] == expected
                changes = sum(client in previous and previous[client] != mode for client, mode in profile.items())
                if changes:
                    switch_count += 1
                previous, logical_time = profile, schedule.finish
            if spec["kind"] == "fixed":
                allocations_ok &= all(name == spec["candidate"] for name in selected_names)
                target = dict(zip(MODES, CANDIDATES[spec["candidate"]]))
                allocations_ok &= all({mode: sum(value == mode for value in _profile(report).values())
                                       for mode in MODES} == target for report in reports)
            if spec["kind"] == "pure":
                allocations_ok &= all(name == f"pure_{spec['mode']}" for name in selected_names)
                allocations_ok &= all(set(_profile(report).values()) == {spec["mode"]} for report in reports)
            record(f"seed{seed}_{strategy}_allocations", allocations_ok)
            record(f"seed{seed}_{strategy}_paper_T", all_t_ok)
            record(f"seed{seed}_{strategy}_timeline", all_timeline_ok)
            record(f"seed{seed}_{strategy}_J", all_j_ok)
            record(f"seed{seed}_{strategy}_normalization_utility", all_utility_ok)
            record(f"seed{seed}_{strategy}_selection_rule", all_choice_ok)
            sidecars_ok = all(path.with_suffix(".sha256").exists() and
                              path.with_suffix(".sha256").read_text(encoding="ascii").strip() == file_digest(path)
                              for path in folder.glob("checkpoint_round_*.pt"))
            record(f"seed{seed}_{strategy}_checkpoint_integrity", sidecars_ok)
    result = {"schema": "fmnist.joint_jt.independent_audit.v1",
              "status": "passed" if all(row["passed"] for row in checks) else "failed",
              "checks": checks, "failures": failures,
              "claims": {"paper_T": "recomputed from plan() for every executed and candidate profile",
                         "J": "recomputed from frozen decision history for every candidate",
                         "future_information": "selector inputs contain no accuracy/loss/future update"}}
    atomic_json(root / "audit.json", result)
    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    outcome = audit(args.root, PaperConfig.load(args.config))
    if outcome["status"] != "passed":
        raise SystemExit(1)
