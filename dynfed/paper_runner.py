"""Real DP-SGD training with the single paper protocol."""
from dataclasses import replace
import copy
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import time
import numpy as np
import torch
from .paper_config import MODES
from .paper_data import assign, evaluate, load_data, local_train, make_training_model, frozen_features, partitions, vector
from .paper_history import History
from .paper_privacy import Ledger
from .paper_barrier import VersionBarrier
from .paper_search import select
from .paper_timeline import mode_masses


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def system_traces(config, seed):
    rng = np.random.default_rng(seed)
    hetero = config.client_heterogeneity
    step_factors = rng.uniform(1 / hetero, hetero, config.clients)
    memories = config.client_memory_mb * rng.uniform(1 / hetero, hetero, config.clients)
    result = []
    for _ in range(config.rounds):
        rows = []
        for client in range(config.clients):
            jitter = lambda: rng.uniform(1-config.network_jitter, 1+config.network_jitter)
            rows.append({"client": client, "edge": client % config.edges,
                         "memory_mb": float(memories[client]),
                         "compute_capacity": float(config.client_compute_capacity / step_factors[client]),
                         "step_sec": float(config.local_step_sec * step_factors[client]),
                         "le_rate": float(config.end_edge_mb_s * jitter()),
                         "lc_rate": float(config.end_cloud_mb_s * jitter()),
                         "ec_rate": float(config.edge_cloud_mb_s * jitter())})
        result.append(rows)
    return result


def run_policy(config, policy, seed, data, partition, traces, destination):
    destination.mkdir(parents=True, exist_ok=False)
    model = make_training_model(config, seed, features=config.model == "resnet18_head")
    initial = vector(model)
    barrier = VersionBarrier(range(config.clients))
    payload_mb = initial.numel() * initial.element_size() / 1e6
    torch.save(model.state_dict(), destination / "initial_model.pt")
    history = History(config.gamma_d, config.gamma_c, config.epsilon_j)
    ledgers = {client: Ledger(config.epsilon, config.delta) for client in range(config.clients)}
    samples = {client: len(indices) for client, indices in enumerate(partition)}
    previous, pending = {}, []
    prior_phase = None
    logical_time = 0.0
    rows, decisions, receipts, cloud_units = [], [], [], []
    x, y, test_x, test_y = data
    config = replace(config, seeds=(seed,))
    started = time.perf_counter()
    for round_index in range(config.rounds):
        round_start = logical_time
        barrier.require_ready(logical_time)
        dispatch_time = logical_time
        dispatch_version = barrier.version
        trace = traces[round_index]
        pools, sigmas = {}, {}
        for client in range(config.clients):
            pool = []
            if (trace[client]["memory_mb"] >= config.required_memory_mb
                    and trace[client]["compute_capacity"] >= config.required_compute):
                for mode in MODES:
                    stages = config.edge_stages if mode == "FL_EC" else 1
                    events = stages * config.local_steps
                    sigma = ledgers[client].calibrate(events * (config.rounds-round_index))
                    if sigma is not None:
                        pool.append(mode)
                        sigmas[client, mode] = sigma
            pools[client] = tuple(pool) or ("SKIP",)
        # Shared Edge capacity is an admission constraint: excluded clients
        # keep FL_C when feasible, rather than silently changing fixed methods.
        for edge in range(config.edges):
            members = [client for client in pools if client % config.edges == edge]
            eligible = [client for client in members if any(mode in pools[client] for mode in ("FL_E", "FL_EC"))]
            for client in eligible[config.edge_group_capacity:]:
                pools[client] = tuple(mode for mode in pools[client] if mode == "FL_C") or ("SKIP",)
        chosen, search = select(config, policy, pools, history, trace, samples, payload_mb,
                                logical_time, round_index, previous, pending,
                                force_reselect=prior_phase == "history_initialization")
        prior_phase = search["phase"]
        profile, schedule = chosen.profile, chosen.schedule
        predicted, predicted_j, prediction_status = history.predict(mode_masses(schedule.accepted))
        history_before = copy.deepcopy(history.state_dict())
        before = vector(model)
        realized = {}
        save_snapshot = round_index < 2 or (round_index+1) % config.snapshot_interval == 0 or round_index+1 == config.rounds
        generation_records = []
        steps = 0
        for unit in schedule.units:
            edge_finish = None
            unit_base, base_round, base_source = before.clone(), dispatch_version, f"cloud:v{dispatch_version}"
            edge_model = unit_base.clone()
            stages = config.edge_stages if unit.mode == "FL_EC" else 1
            for stage in range(stages):
                local = []
                for client in unit.clients:
                    sigma = sigmas[client, unit.mode]
                    count_before = ledgers[client].events
                    worker_seed = int(seed * 1000003 + round_index * 10007 + client * 101 + stage)
                    try:
                        updated, _ = local_train(model, edge_model, x, y, partition[client], config,
                                                 sigma, ledgers[client], worker_seed)
                    finally:
                        write_json(destination / "privacy_state.json", {
                            str(key): {"rdp": value.rdp, "events": value.events,
                                       "epsilon": value.epsilon()} for key, value in ledgers.items()})
                    observed = ledgers[client].events-count_before
                    if observed != config.local_steps:
                        raise RuntimeError("actual DP-SGD receipt differs from planned local block")
                    steps += observed
                    receipts.append({"round": round_index, "client": client, "mode": unit.mode,
                                     "stage": stage+1, "sigma": sigma, "executed_events": observed,
                                     "epsilon_after": ledgers[client].epsilon()})
                    local.append((updated, samples[client]))
                mass = sum(count for _, count in local)
                edge_model = sum((values * (count/mass) for values, count in local),
                                 torch.zeros_like(before))
            realized[unit.identity] = replace(unit, update=edge_model.double()-unit_base.double(),
                                              base_source=base_source, base_round=base_round)
            if unit.mode == "FL_E":
                edge_finish = next(event["finish"] for event in schedule.events
                                   if event["event"] == "edge_stage" and event["mode"] == "FL_E" and event["edge"] == unit.edge)
            if save_snapshot:
                generation_records.append({"unit": realized[unit.identity], "base": unit_base,
                                           "end": edge_model, "edge_ready": edge_finish})
        accepted = [realized.get(unit.identity, unit) for unit in schedule.accepted]
        pending = [realized.get(unit.identity, unit) for unit in schedule.pending]
        observation = history.observe(accepted)
        if observation.get("predicted_j") is not None and abs(predicted_j-observation["predicted_j"]) > 1e-10:
            raise RuntimeError("pre-execution and recorded predictions differ")
        if accepted:
            assign(model, before.double() + observation["cloud_delta"])
            for unit, weight in zip(accepted, observation["weights"]):
                cloud_units.append({"accepted_round": round_index, "identity": unit.identity,
                                    "generated_round": unit.generated_round, "mode": unit.mode,
                                    "base_source": unit.base_source, "base_round": unit.base_round,
                                    "samples": unit.samples, "weight": weight, "arrival": unit.arrival,
                                    "cutoff": schedule.cutoff})
            if abs(sum(observation["weights"])-1) > 1e-12:
                raise RuntimeError("accepted unit weights do not sum to one")
        if pending or {unit.identity for unit in accepted} != {unit.identity for unit in schedule.units}:
            raise RuntimeError("100% quorum did not accept exactly the current-round units")
        barrier.begin(dispatch_version + bool(accepted))
        for event in schedule.events:
            if event["event"] == "model_ack":
                event["version"] = barrier.version
                barrier.acknowledge(event["client"], barrier.version, event["finish"])
            elif event["event"] == "global_barrier":
                event["version"] = barrier.version
        logical_time = schedule.finish
        barrier.require_ready(logical_time)
        accuracy, loss = evaluate(model, test_x, test_y, config)
        mode_changes = sum(client in previous and previous[client] != mode and mode != "SKIP"
                           for client, mode in profile.items())
        for client, mode in profile.items():
            decisions.append({"round": round_index, "client": client, "mode": mode,
                              "feasible_modes": ";".join(pools[client]),
                              "epsilon": ledgers[client].epsilon(),
                              "active": mode != "SKIP"})
        previous.update({client: mode for client, mode in profile.items() if mode != "SKIP"})
        # Save vectors so predictions can be independently recomputed from the
        # exact pre-execution reference, with no test data in selection.
        if save_snapshot:
            torch.save({"round_index": round_index, "history_before": history_before,
                    "dispatch_time": dispatch_time,
                    "dispatch_version": dispatch_version, "cloud_version": barrier.version,
                    "client_versions": dict(barrier.client_versions),
                    "generation_records": generation_records,
                    "reference_before": observation.get("reference_before"),
                    "predicted_before_execution": predicted,
                    "delta": observation.get("cloud_delta"),
                    "model_before": before,
                    "model_after": vector(model),
                    "history_after": history.state_dict(),
                    "accepted": accepted, "pending": pending},
                       destination / f"history_round_{round_index:03d}.pt")
        write_json(destination / f"timeline_round_{round_index:03d}.json", schedule.events)
        row = {"round": round_index+1, "test_accuracy": accuracy, "test_loss": loss,
               "system_time": logical_time, "round_duration": logical_time-round_start,
               "synchronous": schedule.synchronous, "cutoff": schedule.cutoff,
               "dispatch_time": dispatch_time, "dispatch_version": dispatch_version,
               "cloud_version": barrier.version, "acknowledged_clients": len(barrier.acks),
               "expected_cloud_units": len(schedule.units),
               "selection_sec": search["selection_sec"], "selection_phase": search["phase"],
               "evaluated_profiles": search["evaluated_profiles"],
               "feasible_combinations": search["feasible_combinations"],
               "pareto_front_size": search["pareto_front_size"],
               "predicted_j": predicted_j, "prediction_status": prediction_status,
               "observed_j": observation.get("observed_j"),
               "prediction_error": observation.get("prediction_absolute_error"),
               "predicted_update_error_l2": observation.get("predicted_update_error_l2"),
               "admitted_clients": sum(mode != "SKIP" for mode in profile.values()),
               "cloud_units": len(accepted), "pending_units": len(pending),
               "executed_local_steps": steps, "mode_changes": mode_changes,
               "communication_mb": (sum(payload_mb * (1 if unit.mode == "FL_C" else
                    (2*config.edge_stages-1)*len(unit.clients)+1 if unit.mode == "FL_EC" else
                    2*len(unit.clients)+1) for unit in schedule.units)
                    + sum((payload_mb+config.acknowledgement_mb)*(2 if profile.get(client) in {"FL_E", "FL_EC"} else 1)
                          for client in range(config.clients))),
               "max_epsilon": max(ledger.epsilon() for ledger in ledgers.values())}
        rows.append(row)
        write_csv(destination / "round_metrics.csv", rows)
        write_csv(destination / "client_decisions.csv", decisions)
        write_csv(destination / "privacy_receipts.csv", receipts)
        write_csv(destination / "cloud_units.csv", cloud_units)
        print(f"{policy} seed={seed} round={round_index+1}/{config.rounds} "
              f"acc={accuracy:.4f} phase={search['phase']} accepted={len(accepted)} pending={len(pending)}",
              flush=True)
    summary = {"policy": policy, "seed": seed, "rounds": len(rows),
               "final_test_accuracy": rows[-1]["test_accuracy"],
               "final_test_loss": rows[-1]["test_loss"], "system_time": logical_time,
               "selection_sec": sum(row["selection_sec"] for row in rows),
               "admission_fraction": sum(row["admitted_clients"] for row in rows)/(config.rounds*config.clients),
               "cloud_update_fraction": sum(row["cloud_units"] > 0 for row in rows)/config.rounds,
               "executed_local_steps": sum(row["executed_local_steps"] for row in rows),
               "communication_mb": sum(row["communication_mb"] for row in rows),
               "mode_changes": sum(row["mode_changes"] for row in rows),
               "history_observations": history.mode_observations,
               "all_modes_observed": set(history.deviations) == set(MODES),
               "max_epsilon": max(ledger.epsilon() for ledger in ledgers.values()),
               "pending_at_end": len(pending), "wall_sec": time.perf_counter()-started}
    write_json(destination / "summary.json", summary)
    torch.save({"model": model.state_dict(), "history": history.state_dict(), "pending": pending,
                "logical_time": logical_time, "cloud_version": barrier.version,
                "client_versions": barrier.client_versions}, destination / "final_state.pt")
    return summary


def run(config, output):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", config.to_dict())
    torch.set_num_threads(config.torch_threads)
    root = Path(__file__).resolve().parents[1]
    source_hashes = {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                     for folder in ("dynfed", "experiments") for path in (root/folder).glob("*.py")}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    write_json(output / "source_manifest.json", {"git_revision": revision, "sha256": source_hashes})
    write_json(output / "runtime.json", {"torch": torch.__version__, "device": config.device,
                                        "gpu": torch.cuda.get_device_name(0) if config.device == "cuda" else None,
                                        "frozen_feature_cache": config.model == "resnet18_head"})
    summaries = []
    for seed in config.seeds:
        data = load_data(config, seed)
        if config.model == "resnet18_head":
            data = frozen_features(config, seed, data)
        partition = partitions(data[1].numpy(), config.clients, config.edges, config.dirichlet_alpha, seed)
        traces = system_traces(config, seed)
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir()
        write_json(seed_dir / "partition.json", partition)
        write_json(seed_dir / "traces.json", traces)
        write_json(seed_dir / "input_hashes.json", {
            name: hashlib.sha256((seed_dir / name).read_bytes()).hexdigest()
            for name in ("partition.json", "traces.json")})
        for policy in config.policies:
            summaries.append(run_policy(config, policy, seed, data, partition, traces, seed_dir / policy))
            write_csv(output / "summary_table.csv", summaries)
    return summaries
