"""Real DP-SGD training with the single paper protocol."""
from dataclasses import replace
import copy
import csv
import hashlib
import json
import subprocess
from pathlib import Path
import time
import random
import traceback
import numpy as np
import torch
from .paper_config import MODES, CORE_POLICIES, effective_config
from .paper_data import assign, evaluate, load_data, local_train, make_training_model, frozen_features, partitions, vector
from .paper_history import History
from .paper_privacy import Ledger
from .paper_barrier import VersionBarrier
from .paper_search import select
from .paper_timeline import mode_masses
from .paper_acceptance import validate_units
from .paper_storage import atomic_json, atomic_torch, digest, tensor_digest, source_identity, save_checkpoint, load_checkpoint, file_digest


def write_json(path, data):
    atomic_json(path, data)


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


def _run_policy(config, policy, seed, data, partition, traces, destination,
                resume=False, stop_after_round=None):
    config = effective_config(config, policy, seed)
    if policy in CORE_POLICIES and (config.clients < 3 or config.rounds <= config.shared_init_rounds):
        raise ValueError("shared comparison requires at least three clients and post-initialization rounds")
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=resume)
    identity = {"config_hash": digest(config.to_dict()), "source_hash": source_identity()["source_hash"],
                "partition_hash": digest(partition), "traces_hash": digest(traces),
                "data_hashes": [tensor_digest(value) for value in data]}
    run_id = digest(identity)
    identity_path = destination/"run_identity.json"
    if identity_path.exists():
        if json.loads(identity_path.read_text(encoding="utf-8")) != identity:
            raise RuntimeError("resume configuration/source/data identity mismatch")
    else:
        write_json(identity_path, identity)
        write_json(destination/"config.json", config.to_dict())
    if (destination/"summary.json").exists():
        if not resume:
            raise RuntimeError("refuse to overwrite a completed run")
        return json.loads((destination/"summary.json").read_text(encoding="utf-8"))
    if (destination/"unsafe_privacy_state.json").exists():
        raise RuntimeError("uncommitted privacy state requires review; refusing ledger rollback")
    model = make_training_model(config, seed, features=config.model == "resnet18_head")
    initial = vector(model)
    barrier = VersionBarrier(range(config.clients))
    payload_mb = initial.numel() * initial.element_size() / 1e6
    if not (destination/"initial_model.pt").exists():
        atomic_torch(destination/"initial_model.pt", model.state_dict())
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
    first_round = 0
    checkpoints = sorted(destination.glob("checkpoint_round_*.pt"))
    if checkpoints:
        checkpoint = load_checkpoint(checkpoints[-1])
        if checkpoint["run_id"] != run_id:
            raise RuntimeError("checkpoint run identity mismatch")
        model.load_state_dict(checkpoint["model"])
        history, ledgers, barrier = checkpoint["history"], checkpoint["ledgers"], checkpoint["barrier"]
        previous, prior_phase = checkpoint["previous"], checkpoint["prior_phase"]
        logical_time = checkpoint["logical_time"]
        rows, decisions, receipts, cloud_units = (checkpoint[key] for key in ("rows", "decisions", "receipts", "cloud_units"))
        first_round = checkpoint["next_round"]
        torch.set_rng_state(checkpoint["torch_rng"])
        if config.device == "cuda":
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        np.random.set_state(checkpoint["numpy_rng"])
        random.setstate(checkpoint["python_rng"])
    for round_index in range(first_round, config.rounds):
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
        plan_path = destination/f"plan_round_{round_index:03d}.pt"
        if plan_path.exists():
            saved_plan = torch.load(plan_path, map_location="cpu", weights_only=False)
            if saved_plan["run_id"] != run_id:
                raise RuntimeError("cached selection belongs to a different run")
            chosen, search = saved_plan["chosen"], saved_plan["search"]
        else:
            chosen, search = select(config, policy, pools, history, trace, samples, payload_mb,
                                    logical_time, round_index, previous, pending,
                                    force_reselect=prior_phase in {"history_initialization", "shared_initialization"})
            atomic_torch(plan_path, {"run_id": run_id, "chosen": chosen, "search": search})
        prior_phase = search["phase"]
        profile, schedule = chosen.profile, chosen.schedule
        predicted, predicted_j, prediction_status = history.predict(mode_masses(schedule.accepted))
        history_before = copy.deepcopy(history.state_dict())
        write_json(destination/f"candidates_round_{round_index:03d}.json", search)
        atomic_torch(destination/f"decision_history_{round_index:03d}.pt", history_before)
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
                    block_path = destination/f"block_{unit.identity.replace(':','_')}_s{stage}_c{client}.pt"
                    block_id = {"run_id": run_id, "sigma": sigma, "seed": worker_seed,
                                "base": tensor_digest(edge_model), "events_before": count_before,
                                "rdp_before": list(ledgers[client].rdp)}
                    block = torch.load(block_path, map_location="cpu", weights_only=False) if block_path.exists() else None
                    if block is not None and block["identity"] != block_id:
                        raise RuntimeError("local transaction identity mismatch; refusing privacy reuse")
                    if block is not None:
                        ledgers[client] = copy.deepcopy(block["ledger"])
                    if block is not None and block["status"] == "complete":
                        updated = block["updated"]
                    else:
                        def persist_step(progress):
                            atomic_torch(block_path, {"identity": block_id, "status": "partial",
                                                       "ledger": copy.deepcopy(ledgers[client]), "progress": progress})
                        try:
                            updated, _ = local_train(model, edge_model, x, y, partition[client], config,
                                                     sigma, ledgers[client], worker_seed,
                                                     resume_state=None if block is None else block["progress"],
                                                     progress_callback=persist_step)
                            atomic_torch(block_path, {"identity": block_id, "status": "complete",
                                                      "ledger": copy.deepcopy(ledgers[client]), "updated": updated})
                        finally:
                            durable = torch.load(block_path, map_location="cpu", weights_only=False) if block_path.exists() else None
                            durable_events = count_before if durable is None else durable["ledger"].events
                            if durable_events != ledgers[client].events:
                                write_json(destination/"unsafe_privacy_state.json", {"client": client,
                                           "actual_events": ledgers[client].events, "durable_events": durable_events})
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
        validate_units(schedule.units, accepted, round_index, dispatch_version, schedule.cutoff)
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
               "initialization": policy in CORE_POLICIES and round_index < config.shared_init_rounds,
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
        if policy in CORE_POLICIES and round_index+1 == config.shared_init_rounds and set(history.deviations) != set(MODES):
            raise RuntimeError("shared initialization failed to produce all three histories")
        checkpoint_path = destination/f"checkpoint_round_{round_index+1:03d}.pt"
        save_checkpoint(checkpoint_path, {"run_id": run_id, "next_round": round_index+1,
                        "model": model.state_dict(), "history": history, "ledgers": ledgers,
                        "barrier": barrier, "previous": previous, "prior_phase": prior_phase,
                        "logical_time": logical_time, "rows": rows, "decisions": decisions,
                        "receipts": receipts, "cloud_units": cloud_units,
                        "optimizer": {"algorithm": "explicit SGD", "momentum": 0., "persistent_state": None},
                        "torch_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state_all() if config.device == "cuda" else [],
                        "numpy_rng": np.random.get_state(), "python_rng": random.getstate()})
        write_json(destination/"run_status.json", {"status": "running", "completed_rounds": round_index+1,
                   "total_rounds": config.rounds, "last_checkpoint": checkpoint_path.name,
                   "checkpoint_sha256": file_digest(checkpoint_path), "run_id": run_id})
        print(f"{policy} seed={seed} round={round_index+1}/{config.rounds} "
              f"acc={accuracy:.4f} phase={search['phase']} accepted={len(accepted)} pending={len(pending)}",
              flush=True)
        if stop_after_round is not None and round_index+1 == stop_after_round:
            raise InterruptedError("requested checkpoint-boundary interruption")
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
               "config_hash": digest(config.to_dict()), "run_id": run_id,
               "initialization_rounds": sum(row["initialization"] for row in rows),
               "initialization_system_time": sum(row["round_duration"] for row in rows if row["initialization"]),
               "initialization_steps": sum(row["executed_local_steps"] for row in rows if row["initialization"]),
               "initialization_max_epsilon": max((row["max_epsilon"] for row in rows if row["initialization"]), default=0.),
               "selection_phase_system_time": sum(row["round_duration"] for row in rows if not row["initialization"]),
               "selection_phase_steps": sum(row["executed_local_steps"] for row in rows if not row["initialization"]),
               "max_epsilon": max(ledger.epsilon() for ledger in ledgers.values()),
               "pending_at_end": len(pending), "wall_sec": time.perf_counter()-started}
    init_rows = [row for row in rows if row["initialization"]]
    selection_rows = [row for row in rows if not row["initialization"]]
    summary["initialization_test_accuracy"] = init_rows[-1]["test_accuracy"] if init_rows else None
    summary["post_initialization_accuracy_change"] = (rows[-1]["test_accuracy"]-init_rows[-1]["test_accuracy"]) if init_rows else None
    summary["mean_selection_round_duration"] = (sum(row["round_duration"] for row in selection_rows)/len(selection_rows)) if selection_rows else None
    selection_decisions = [row for row in decisions if not (policy in CORE_POLICIES and row["round"] < config.shared_init_rounds)]
    for mode in MODES:
        summary["selection_fraction_"+mode] = sum(row["mode"] == mode for row in selection_decisions)/max(1,len(selection_decisions))
    atomic_torch(destination / "final_state.pt", {"model": model.state_dict(), "history": history.state_dict(), "pending": pending,
                "logical_time": logical_time, "cloud_version": barrier.version,
                "client_versions": barrier.client_versions})
    write_json(destination / "summary.json", summary)
    return summary


def run_policy(config, policy, seed, data, partition, traces, destination, resume=False, stop_after_round=None):
    destination = Path(destination)
    # A complete run is immutable; return it only after validating identity in _run_policy.
    if destination.exists() and not resume:
        raise FileExistsError(destination)
    old_status = json.loads((destination/"run_status.json").read_text(encoding="utf-8")) if (destination/"run_status.json").exists() else {}
    attempts = list(old_status.get("attempts", []))
    attempt_started = time.perf_counter()
    try:
        result = _run_policy(config, policy, seed, data, partition, traces, destination, resume, stop_after_round)
    except BaseException as error:
        status = json.loads((destination/"run_status.json").read_text(encoding="utf-8")) if (destination/"run_status.json").exists() else {}
        attempts.append({"status": "interrupted" if isinstance(error, (KeyboardInterrupt, InterruptedError)) else "failed",
                         "wall_sec": time.perf_counter()-attempt_started, "error": traceback.format_exc()})
        status.update(status=attempts[-1]["status"], attempts=attempts, error=traceback.format_exc())
        if destination.exists() and old_status.get("status") != "completed":
            write_json(destination/"run_status.json", status)
        raise
    if old_status.get("status") != "completed":
        attempts.append({"status": "completed", "wall_sec": time.perf_counter()-attempt_started})
        result["attempt_wall_sec"] = sum(attempt["wall_sec"] for attempt in attempts)
        write_json(destination/"summary.json", result)
        status = json.loads((destination/"run_status.json").read_text(encoding="utf-8"))
        status.update(status="completed", attempts=attempts, audit_status="not_audited")
        write_json(destination/"run_status.json", status)
    return result


def run(config, output, resume=False):
    output = Path(output)
    if config.clients == 100 and config.rounds == 100 and not (output/"preregistration.json").exists():
        raise RuntimeError("formal matrix requires a preregistered output directory")
    output.mkdir(parents=True, exist_ok=resume)
    if (output/"config.json").exists():
        if digest(json.loads((output/"config.json").read_text(encoding="utf-8"))) != digest(config.to_dict()):
            raise RuntimeError("scenario configuration mismatch")
    else:
        write_json(output / "config.json", config.to_dict())
    torch.set_num_threads(config.torch_threads)
    root = Path(__file__).resolve().parents[1]
    identity = source_identity()
    if (output/"source_manifest.json").exists():
        registered = json.loads((output/"source_manifest.json").read_text(encoding="utf-8"))
        if registered["source_hash"] != identity["source_hash"]:
            raise RuntimeError("scenario source mismatch")
    else:
        write_json(output / "source_manifest.json", identity)
    write_json(output / "runtime.json", {"torch": torch.__version__, "device": config.device,
                                        "gpu": torch.cuda.get_device_name(0) if config.device == "cuda" else None,
                                        "frozen_feature_cache": config.model == "resnet18_head"})
    summaries = []
    for seed in config.seeds:
        preparation_started = time.perf_counter()
        data = load_data(config, seed)
        if config.model == "resnet18_head":
            data = frozen_features(config, seed, data)
        partition = partitions(data[1].numpy(), config.clients, config.edges, config.dirichlet_alpha, seed)
        traces = system_traces(config, seed)
        seed_dir = output / f"seed_{seed}"
        seed_dir.mkdir(exist_ok=resume)
        preparation_path = seed_dir/f"preparation_attempt_{len(list(seed_dir.glob('preparation_attempt_*.json')))+1:03d}.json"
        write_json(preparation_path, {"wall_sec": time.perf_counter()-preparation_started,
                                     "frozen_features": config.model == "resnet18_head"})
        for name, value in (("partition.json", partition), ("traces.json", traces)):
            path = seed_dir/name
            if path.exists():
                if digest(json.loads(path.read_text(encoding="utf-8"))) != digest(value):
                    raise RuntimeError("paired seed input mismatch")
            else:
                write_json(path, value)
        input_path = seed_dir/"input_hashes.json"
        if not input_path.exists():
            write_json(input_path, {name: file_digest(seed_dir/name) for name in ("partition.json", "traces.json")})
        for policy in config.policies:
            from experiments.audit_paper import audit
            if resume and (seed_dir/policy/"summary.json").exists():
                audit(output, required_runs={(seed,policy)})
            summaries.append(run_policy(config, policy, seed, data, partition, traces, seed_dir / policy, resume=resume))
            report = audit(output, required_runs={(seed,policy)})
            write_json(seed_dir/policy/"audit.json", report)
            status = json.loads((seed_dir/policy/"run_status.json").read_text(encoding="utf-8"))
            status["audit_status"] = "passed"
            write_json(seed_dir/policy/"run_status.json", status)
            write_csv(output / "summary_table.csv", summaries)
    return summaries
