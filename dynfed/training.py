from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import ExperimentConfig
from .nodes import ClientProfile, build_profiles, draw_bandwidth, draw_runtime_multiplier
from .privacy import OBJECT_SIZES, privacy_processing_time, protected_size, utility_penalty


DIRECT_CLOUD_MODES = {"LIC", "LIIC", "LIEIIC"}
EDGE_ONLY_MODES = {"LIE", "LIIE"}
MULTILEVEL_MODES = {"LIEIIIC", "LIIEIIIC"}


@dataclass(frozen=True)
class ModeSpec:
    name: str
    client_target: str
    client_objects: list[str]
    edge_to_cloud_objects: list[str]
    local_work: float
    edge_work: float
    cloud_work: float
    mode_penalty: float = 0.0
    E_edge_loops: int = 1
    alpha: float = 5.0  # privacy exposure level for mode selection cost
    edge_cpu: float = 0.0  # CPU demand on edge per unit sample_ratio
    cloud_cpu: float = 0.0  # CPU demand on cloud per unit sample_ratio
    local_memory: float = 1.0  # profiled end-device training-memory units


MODE_SPECS = {
    # Edge-only (data lost to global model → highest global penalty)
    # LIE (edge-only, emb+label+grad): SpeedTask-analogue, all stages on edge
    "LIE": ModeSpec("LIE", "edge", ["emb", "label", "grad"], [], 1.0, 0.35, 0.0, 0.17, alpha=6, edge_cpu=4, cloud_cpu=0, local_memory=0.72),
    # LIIE (edge-only, upd): SpeedTask in visualization (α=2)
    "LIIE": ModeSpec("LIIE", "edge", ["upd"], [], 1.15, 0.25, 0.0, 0.15, alpha=2, edge_cpu=4, cloud_cpu=0, local_memory=1.20),
    # LIC (cloud-direct, emb+label+grad): SuperTask in visualization (α=6)
    "LIC": ModeSpec("LIC", "cloud", ["emb", "label", "grad"], [], 1.0, 0.0, 0.75, 0.025, alpha=6, edge_cpu=0, cloud_cpu=4, local_memory=0.68),
    # LIIC (cloud-direct, upd): No-Split in visualization (α=4)
    "LIIC": ModeSpec("LIIC", "cloud", ["upd"], [], 1.15, 0.0, 0.7, 0.02, alpha=4, edge_cpu=1, cloud_cpu=1, local_memory=1.20),
    # LIEIIC (edge→cloud): Split in visualization (α=5)
    "LIEIIC": ModeSpec("LIEIIC", "edge", ["emb", "label", "grad"], ["upd"], 0.92, 0.7, 0.45, 0.003, alpha=5, edge_cpu=2, cloud_cpu=2, local_memory=0.74),
    # LIEIIIC (edge→cloud, 2 edge loops): Split with multi-epoch
    "LIEIIIC": ModeSpec("LIEIIIC", "edge", ["emb", "label", "grad"], ["upd"], 0.9, 1.1, 0.42, 0.001, E_edge_loops=3, alpha=5, edge_cpu=2, cloud_cpu=2, local_memory=0.78),
    # LIIEIIIC (edge→cloud upd, 2 edge loops): No-Split with multi-epoch
    "LIIEIIIC": ModeSpec("LIIEIIIC", "edge", ["upd"], ["upd"], 1.05, 0.9, 0.42, 0.004, E_edge_loops=3, alpha=3, edge_cpu=1, cloud_cpu=1, local_memory=1.24),
}


def run_experiment(config: ExperimentConfig) -> dict[str, Any]:
    mode_name = config.mode.name.upper()
    if mode_name not in MODE_SPECS:
        raise ValueError(f"Unsupported mode {config.mode.name}. Choose one of {sorted(MODE_SPECS)}.")

    spec = MODE_SPECS[mode_name]
    out_dir = config.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    clients, edges = build_profiles(
        num_clients=config.topology.num_clients,
        num_edges=config.topology.num_edges,
        client_heterogeneity=config.runtime.client_heterogeneity,
        edge_heterogeneity=config.runtime.edge_heterogeneity,
        seed=config.runtime.seed,
    )
    edge_by_id = {edge.edge_id: edge for edge in edges}
    clients_by_edge = {
        edge.edge_id: [client for client in clients if client.edge_id == edge.edge_id]
        for edge in edges
    }

    seed_offset = sum(ord(ch) for ch in f"{mode_name}:{config.privacy.mechanism.lower()}")
    rng = random.Random(config.runtime.seed + seed_offset)
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []
    logical_time = 0.0
    effective_updates = 0.0
    best_accuracy = 0.0

    for round_idx in range(config.runtime.rounds):
        round_result = _simulate_round(
            round_idx=round_idx,
            start_time=logical_time,
            spec=spec,
            clients=clients,
            clients_by_edge=clients_by_edge,
            edge_by_id=edge_by_id,
            config=config,
            rng=rng,
        )
        logical_time = round_result["round_end_time"]
        effective_updates += round_result["num_effective_clients"]
        accuracy = _synthetic_accuracy(
            effective_updates=effective_updates,
            total_clients=len(clients),
            round_idx=round_idx,
            mechanism=config.privacy.mechanism,
            epsilon=config.privacy.epsilon,
            mode_penalty=spec.mode_penalty,
            rng=rng,
        )
        best_accuracy = max(best_accuracy, accuracy)

        row = {
            "round": round_idx,
            "mode": mode_name,
            "privacy": config.privacy.mechanism.upper(),
            "epsilon": config.privacy.epsilon,
            "logical_time": logical_time,
            "round_duration": round_result["round_duration"],
            "accuracy": accuracy,
            "best_accuracy": best_accuracy,
            "num_effective_clients": round_result["num_effective_clients"],
            "num_effective_edges": round_result["num_effective_edges"],
            "client_compute_time": round_result["client_compute_time"],
            "edge_compute_time": round_result["edge_compute_time"],
            "communication_time": round_result["communication_time"],
            "aggregation_time": round_result["aggregation_time"],
            "waiting_time": round_result["waiting_time"],
            "communication_volume": round_result["communication_volume"],
            "slow_clients": round_result["slow_clients"],
        }
        rows.append(row)
        events.extend(round_result["events"])

    _write_json(out_dir / "config.json", config.to_dict())
    _write_csv(out_dir / "round_metrics.csv", rows)
    if config.output.write_events:
        _write_csv(out_dir / "event_log.csv", events)

    summary = _build_summary(rows, config, mode_name)
    _write_json(out_dir / "summary.json", summary)
    return summary


def _simulate_round(
    round_idx: int,
    start_time: float,
    spec: ModeSpec,
    clients: list[ClientProfile],
    clients_by_edge: dict[int, list[ClientProfile]],
    edge_by_id: dict[int, Any],
    config: ExperimentConfig,
    rng: random.Random,
) -> dict[str, Any]:
    mechanism = config.privacy.mechanism
    epsilon = config.privacy.epsilon
    agg_fraction = config.topology.aggregation_fraction

    client_compute_total = 0.0
    edge_compute_total = 0.0
    communication_total = 0.0
    aggregation_total = 0.0
    waiting_total = 0.0
    communication_volume = 0.0
    slow_clients = 0
    events: list[dict[str, Any]] = []

    client_arrivals: dict[int, float] = {}
    client_comm_times: dict[int, float] = {}
    selected_clients: list[int] = []

    for client in clients:
        multiplier, is_slow = draw_runtime_multiplier(
            rng,
            config.runtime.slow_client_rate,
            config.runtime.slow_factor_low,
            config.runtime.slow_factor_high,
        )
        slow_clients += int(is_slow)
        local_time = (
            config.runtime.local_update_base_time
            * spec.local_work
            * client.compute_factor
            * (client.samples / 150.0)
            * multiplier
        )
        local_time += privacy_processing_time(spec.client_objects, mechanism, epsilon)
        client_compute_total += local_time

        if spec.client_target == "edge":
            base_rate = 5.0
        else:
            base_rate = 2.2
        volume = protected_size(spec.client_objects, mechanism)
        comm_time = volume / draw_bandwidth(rng, base_rate, config.runtime.network_jitter)
        communication_volume += volume
        communication_total += comm_time
        client_comm_times[client.client_id] = comm_time
        client_arrivals[client.client_id] = start_time + local_time + comm_time

        # LIEIIC sends each client-derived edge update directly to the cloud;
        # unlike the multi-level modes, it has no edge pre-aggregation.
        if spec.name == "LIEIIC":
            edge_time = (
                config.runtime.edge_train_base_time
                * spec.edge_work
                * edge_by_id[client.edge_id].compute_factor
            )
            edge_cloud_volume = protected_size(spec.edge_to_cloud_objects, mechanism)
            edge_cloud_time = edge_cloud_volume / draw_bandwidth(
                rng, 8.0, config.runtime.network_jitter
            )
            edge_compute_total += edge_time
            communication_volume += edge_cloud_volume
            communication_total += edge_cloud_time
            client_arrivals[client.client_id] += edge_time + edge_cloud_time

        events.append(
            {
                "round": round_idx,
                "time": client_arrivals[client.client_id],
                "event_type": "client_ready",
                "client_id": client.client_id,
                "edge_id": client.edge_id,
                "is_slow": is_slow,
                "duration": client_arrivals[client.client_id] - start_time,
            }
        )

    edge_finish_times: dict[int, float] = {}
    if spec.name in DIRECT_CLOUD_MODES:
        k = max(1, math.ceil(len(clients) * agg_fraction))
        selected = sorted(clients, key=lambda item: client_arrivals[item.client_id])[:k]
        selected_clients = [client.client_id for client in selected]
        cloud_start = max(client_arrivals[cid] for cid in selected_clients)
        waiting_total += sum(cloud_start - client_arrivals[cid] for cid in selected_clients)
        cloud_payload = sum(
            _cloud_aggregation_payload(spec.name, mechanism) for _ in selected
        )
        cloud_agg = (
            config.runtime.cloud_aggregation_beta * cloud_payload
            + config.runtime.cloud_aggregation_fixed
        )
        aggregation_total += cloud_agg
        round_end_time = cloud_start + cloud_agg
        num_effective_edges = 1
        events.append(
            {
                "round": round_idx,
                "time": round_end_time,
                "event_type": "cloud_aggregate_direct",
                "num_clients": len(selected),
                "effective_payload": cloud_payload,
                "aggregation_beta": config.runtime.cloud_aggregation_beta,
                "aggregation_fixed": config.runtime.cloud_aggregation_fixed,
                "aggregation_time": cloud_agg,
            }
        )
    else:
        for edge_id, edge_clients in clients_by_edge.items():
            k = max(1, math.ceil(len(edge_clients) * agg_fraction))
            selected = sorted(edge_clients, key=lambda item: client_arrivals[item.client_id])[:k]
            local_selected = [client.client_id for client in selected]
            selected_clients.extend(local_selected)
            edge_start = max(client_arrivals[cid] for cid in local_selected)
            waiting_total += sum(edge_start - client_arrivals[cid] for cid in local_selected)

            edge_profile = edge_by_id[edge_id]
            loop_factor = spec.E_edge_loops if spec.name in MULTILEVEL_MODES else 1
            edge_train = (
                config.runtime.edge_train_base_time
                * spec.edge_work
                * loop_factor
                * max(1, config.mode.edge_local_cycles)
                * edge_profile.compute_factor
            )
            edge_payload = sum(
                _edge_aggregation_payload(spec.name, mechanism) for _ in local_selected
            )
            edge_agg = loop_factor * (
                config.runtime.edge_aggregation_beta * edge_payload
                + config.runtime.edge_aggregation_fixed
            )
            edge_compute_total += edge_train
            aggregation_total += edge_agg
            edge_ready = edge_start + edge_train + edge_agg

            volume = protected_size(spec.edge_to_cloud_objects, mechanism)
            comm_time = 0.0
            if volume > 0:
                comm_time = volume / draw_bandwidth(rng, 8.0, config.runtime.network_jitter)
                communication_volume += volume
                communication_total += comm_time
            edge_finish_times[edge_id] = edge_ready + comm_time
            events.append(
                {
                    "round": round_idx,
                    "time": edge_finish_times[edge_id],
                    "event_type": "edge_ready",
                    "edge_id": edge_id,
                    "num_clients": len(local_selected),
                    "duration": edge_finish_times[edge_id] - start_time,
                    "effective_payload": edge_payload,
                    "aggregation_beta": config.runtime.edge_aggregation_beta,
                    "aggregation_fixed": config.runtime.edge_aggregation_fixed,
                    "aggregation_time": edge_agg,
                    "loop_factor": loop_factor,
                }
            )

        if spec.name in EDGE_ONLY_MODES:
            round_end_time = max(edge_finish_times.values())
            num_effective_edges = len(edge_finish_times)
        else:
            edge_k = max(1, math.ceil(len(edge_finish_times) * agg_fraction))
            selected_edges = sorted(edge_finish_times, key=edge_finish_times.get)[:edge_k]
            cloud_start = max(edge_finish_times[edge_id] for edge_id in selected_edges)
            waiting_total += sum(
                cloud_start - edge_finish_times[edge_id] for edge_id in selected_edges
            )
            cloud_payload = sum(
                protected_size(spec.edge_to_cloud_objects, mechanism)
                for _ in selected_edges
            )
            cloud_agg = (
                config.runtime.cloud_aggregation_beta * cloud_payload
                + config.runtime.cloud_aggregation_fixed
            )
            aggregation_total += cloud_agg
            round_end_time = cloud_start + cloud_agg
            num_effective_edges = len(selected_edges)
            events.append(
                {
                    "round": round_idx,
                    "time": round_end_time,
                    "event_type": "cloud_aggregate_edge",
                    "num_edges": len(selected_edges),
                    "effective_payload": cloud_payload,
                    "aggregation_beta": config.runtime.cloud_aggregation_beta,
                    "aggregation_fixed": config.runtime.cloud_aggregation_fixed,
                    "aggregation_time": cloud_agg,
                }
            )

    return {
        "round_end_time": round_end_time,
        "round_duration": round_end_time - start_time,
        "num_effective_clients": len(selected_clients),
        "selected_client_ids": list(selected_clients),
        "num_effective_edges": num_effective_edges,
        "client_compute_time": client_compute_total,
        "edge_compute_time": edge_compute_total,
        "communication_time": communication_total,
        "aggregation_time": aggregation_total,
        "waiting_time": waiting_total,
        "communication_volume": communication_volume,
        "slow_clients": slow_clients,
        "events": events,
    }


def _edge_aggregation_payload(mode_name: str, mechanism: str) -> float:
    if mode_name in {"LIE", "LIEIIIC"}:
        return float(OBJECT_SIZES["upd"])
    if mode_name in {"LIIE", "LIIEIIIC"}:
        return protected_size(["upd"], mechanism)
    return 0.0


def _cloud_aggregation_payload(mode_name: str, mechanism: str) -> float:
    if mode_name in {"LIC"}:
        return float(OBJECT_SIZES["upd"])
    if mode_name in {"LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC"}:
        return protected_size(["upd"], mechanism)
    return 0.0


def _synthetic_accuracy(
    effective_updates: float,
    total_clients: int,
    round_idx: int,
    mechanism: str,
    epsilon: float,
    mode_penalty: float,
    rng: random.Random,
) -> float:
    progress = effective_updates / max(float(total_clients), 1.0)
    ceiling = 0.83 - utility_penalty(mechanism, epsilon) - mode_penalty
    warmup = 0.18 + (ceiling - 0.18) * (1.0 - math.exp(-progress / 8.0))
    noise = rng.uniform(-0.004, 0.004) / math.sqrt(round_idx + 1.0)
    return max(0.0, min(0.95, warmup + noise))


def _build_summary(rows: list[dict[str, Any]], config: ExperimentConfig, mode_name: str) -> dict[str, Any]:
    final = rows[-1]
    best = max(rows, key=lambda row: row["accuracy"])
    total_comm = sum(row["communication_volume"] for row in rows)
    total_wait = sum(row["waiting_time"] for row in rows)
    total_duration = final["logical_time"]
    return {
        "mode": mode_name,
        "privacy": config.privacy.mechanism.upper(),
        "epsilon": config.privacy.epsilon,
        "rounds": config.runtime.rounds,
        "final_accuracy": final["accuracy"],
        "best_accuracy": best["accuracy"],
        "round_to_best": best["round"],
        "total_logical_time": total_duration,
        "total_communication_volume": total_comm,
        "total_waiting_time": total_wait,
        "privacy_efficiency": best["accuracy"] / max(total_duration, 1e-9),
        "output_dir": str(config.output_path),
    }


def _write_json(path: Path, payload: Any) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
