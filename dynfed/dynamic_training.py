from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .nodes import build_profiles
from .privacy import mechanism_uses_dp, mechanism_uses_he
from .real_training import (
    RealTrainingConfig,
    _apply_privacy_to_update,
    _evaluate_softmax,
    _fedavg_update,
    _load_digits,
    _local_train_softmax,
    _partition_clients,
)
from .selection import SelectionConfig, choose_candidate, choose_global_pareto_profile, enumerate_candidates
from .training import MODE_SPECS


@dataclass(frozen=True)
class DynamicFedAvgConfig:
    selection: SelectionConfig
    training: RealTrainingConfig
    policies: tuple[str, ...] = (
        "ours",
        "ours_ideal",
        "ours_knee",
        "ours_time_first",
        "ours_acc_first",
        "fixed_dp",
        "fixed_he",
        "random",
        "performance_only",
        "privacy_only",
        "no_protection",
    )


def run_dynamic_fedavg_training(config: DynamicFedAvgConfig) -> dict[str, Any]:
    output_dir = Path(config.selection.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = _load_digits(
        config.selection.seed,
        config.training.test_size,
    )
    client_indices = _partition_clients(
        y_train=y_train,
        num_clients=config.selection.num_clients,
        iid=config.training.iid,
        seed=config.selection.seed,
    )
    clients, edges = build_profiles(
        num_clients=config.selection.num_clients,
        num_edges=config.selection.num_edges,
        client_heterogeneity=config.selection.client_heterogeneity,
        edge_heterogeneity=config.selection.edge_heterogeneity,
        seed=config.selection.seed,
    )
    edge_by_id = {edge.edge_id: edge for edge in edges}

    summaries = []
    for policy in config.policies:
        summaries.append(
            _run_policy(
                policy=policy,
                config=config,
                clients=clients,
                edge_by_id=edge_by_id,
                client_indices=client_indices,
                x_train=x_train,
                y_train=y_train,
                x_test=x_test,
                y_test=y_test,
                output_dir=output_dir / policy,
            )
        )

    _write_csv(output_dir / "summary_table.csv", summaries)
    _write_json(
        output_dir / "config.json",
        {
            "selection": config.selection.__dict__,
            "training": config.training.__dict__,
            "policies": list(config.policies),
        },
    )
    return {
        "output_dir": str(output_dir),
        "summary_table": str(output_dir / "summary_table.csv"),
        "summaries": summaries,
    }


def _run_policy(
    *,
    policy: str,
    config: DynamicFedAvgConfig,
    clients: list[Any],
    edge_by_id: dict[int, Any],
    client_indices: list[np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(config.selection.seed + _policy_offset(policy))
    np_rng = np.random.default_rng(config.selection.seed + _policy_offset(policy) + 1543)

    num_features = x_train.shape[1]
    num_classes = int(np.max(y_train)) + 1
    weights = np.zeros((num_features, num_classes), dtype=np.float64)
    bias = np.zeros(num_classes, dtype=np.float64)
    remaining = {client.client_id: config.selection.initial_epsilon for client in clients}

    round_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    best_accuracy = 0.0
    logical_time = 0.0
    client_by_id = {c.client_id: c for c in clients}
    previous_choices = {}

    for round_idx in range(config.selection.rounds):
        selected = []
        round_time = 0.0
        round_comm = 0.0
        round_risk = 0.0
        infeasible = 0

        selected_rows = []
        for client in clients:
            rem = remaining[client.client_id]
            candidates = enumerate_candidates(
                config=config.selection,
                client_id=client.client_id,
                edge_factor=edge_by_id[client.edge_id].compute_factor,
                compute_factor=client.compute_factor,
                memory_capacity_factor=client.memory_capacity_factor,
                samples=client.samples,
                remaining_epsilon=rem,
                round_idx=round_idx,
                rng=rng,
                policy=policy,
            )
            candidate = choose_candidate(
                candidates,
                policy=policy,
                rng=rng,
                require_feasible=config.selection.require_feasible,
                remaining_epsilon=rem,
            )
            selected_rows.append((client.client_id, candidate, candidates, rem))

        if policy == "ours":
            selected_rows, profile_evaluation = choose_global_pareto_profile(
                config=config.selection,
                selected=selected_rows,
                client_samples={client.client_id: float(client.samples) for client in clients},
                client_edges={client.client_id: int(client.edge_id) for client in clients},
                previous_choices=previous_choices,
            )
        else:
            profile_evaluation = None
        previous_choices = {client_id: candidate for client_id, candidate, _candidates, _rem in selected_rows}

        for client_id, candidate, _candidates, rem in selected_rows:
            client = client_by_id[client_id]
            remaining[client_id] = max(0.0, rem - candidate.epsilon_used)
            round_comm += candidate.communication_volume
            round_risk = max(round_risk, candidate.risk)
            infeasible += int(not candidate.feasible)
            selected.append((client_id, candidate))
            decision_rows.append(
                {
                    "policy": policy,
                    "round": round_idx,
                    "client_id": client_id,
                    "edge_id": client.edge_id,
                    "mode": candidate.mode,
                    "mechanisms": _mechanism_label(candidate.mechanisms),
                    "update_mechanism": _update_mechanism(candidate.mechanisms),
                    "time": candidate.time,
                    "risk": candidate.risk,
                    "epsilon_used": candidate.epsilon_used,
                    "remaining_epsilon": remaining[client_id],
                    "communication_volume": candidate.communication_volume,
                    "feasible": candidate.feasible,
                    "system_latency_objective": profile_evaluation.system_latency if profile_evaluation else "",
                    "system_omega_objective": profile_evaluation.system_omega if profile_evaluation else "",
                    "cloud_fusion_ratio": profile_evaluation.cloud_fusion_ratio if profile_evaluation else "",
                    "admitted_client_ids_objective": ";".join(str(cid) for cid in profile_evaluation.admitted_client_ids) if profile_evaluation else "",
                }
            )

        updates = []
        sample_counts = []
        skipped_clients = 0
        for client_id, candidate in selected:
            if candidate.mode == "SKIP":
                skipped_clients += 1
                continue
            idx = client_indices[client_id]
            if len(idx) == 0:
                continue

            t0 = time.perf_counter()
            local_w, local_b = _local_train_softmax(
                weights=weights,
                bias=bias,
                x=x_train[idx],
                y=y_train[idx],
                epochs=config.training.local_epochs,
                learning_rate=config.training.learning_rate,
                l2=config.training.l2,
            )
            measured_local = time.perf_counter() - t0

            delta_w = local_w - weights
            delta_b = local_b - bias
            delta_w, delta_b = _apply_privacy_to_update(
                delta_w=delta_w,
                delta_b=delta_b,
                mechanism=_update_mechanism(candidate.mechanisms),
                epsilon=max(config.selection.dp_upd_epsilon, 1e-6),
                clip_norm=config.training.dp_clip_norm,
                noise_multiplier=config.training.dp_noise_multiplier,
                rng=np_rng,
            )
            updates.append((delta_w, delta_b))
            sample_counts.append(len(idx))

            client_info = client_by_id[client_id]
            spec_mode = MODE_SPECS[candidate.mode]
            L = config.selection.L_block_cycles
            samples = client_info.samples
            compute_factor = client_info.compute_factor
            per_block_local = spec_mode.local_work * samples / 150.0 / max(L, 1) * compute_factor
            est_local = L * per_block_local
            client_time = candidate.time - est_local + measured_local
            round_time = max(round_time, client_time)

        if updates:
            weights, bias = _fedavg_update(weights, bias, updates, sample_counts)

        logical_time += round_time
        test_loss, test_accuracy = _evaluate_softmax(weights, bias, x_test, y_test, config.training.l2)
        train_loss, train_accuracy = _evaluate_softmax(weights, bias, x_train, y_train, config.training.l2)
        best_accuracy = max(best_accuracy, test_accuracy)
        round_rows.append(
            {
                "policy": policy,
                "round": round_idx,
                "logical_time": logical_time,
                "round_duration": round_time,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "train_loss": train_loss,
                "best_accuracy": best_accuracy,
                "communication_volume": round_comm,
                "max_risk": round_risk,
                "min_remaining_epsilon": min(remaining.values()),
                "infeasible_clients": infeasible,
                "skipped_clients": skipped_clients,
                "num_clients": len(clients),
                "system_latency_objective": profile_evaluation.system_latency if profile_evaluation else "",
                "system_omega_objective": profile_evaluation.system_omega if profile_evaluation else "",
                "cloud_fusion_ratio": profile_evaluation.cloud_fusion_ratio if profile_evaluation else "",
                "admitted_client_ids_objective": ";".join(str(cid) for cid in profile_evaluation.admitted_client_ids) if profile_evaluation else "",
            }
        )

    summary = _summarize_policy(
        policy,
        round_rows,
        decision_rows,
        output_dir,
        config.selection.time_limit,
    )
    _write_csv(output_dir / "round_metrics.csv", round_rows)
    _write_csv(output_dir / "client_decisions.csv", decision_rows)
    _write_json(output_dir / "summary.json", summary)
    return summary


def _update_mechanism(mechanisms: dict[str, str]) -> str:
    mechanism = str(mechanisms.get("upd", "none"))
    if mechanism_uses_dp(mechanism) or mechanism_uses_he(mechanism):
        return mechanism
    return "none"


def _summarize_policy(
    policy: str,
    round_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    output_dir: Path,
    time_limit: float,
) -> dict[str, Any]:
    final = round_rows[-1]
    best = max(round_rows, key=lambda row: row["test_accuracy"])
    return {
        "policy": policy,
        "dataset": "sklearn_digits",
        "rounds": len(round_rows),
        "final_test_accuracy": final["test_accuracy"],
        "best_test_accuracy": best["test_accuracy"],
        "round_to_best": best["round"],
        "final_train_accuracy": final["train_accuracy"],
        "total_logical_time": final["logical_time"],
        "total_communication_volume": sum(row["communication_volume"] for row in round_rows),
        "max_privacy_risk": max(row["max_risk"] for row in round_rows),
        "min_remaining_epsilon": final["min_remaining_epsilon"],
        "total_epsilon_used": sum(row["epsilon_used"] for row in decision_rows),
        "feasible_rate": _mean(float(row["feasible"]) for row in decision_rows),
        "time_satisfied_rate": _mean(float(float(row["time"]) <= time_limit) for row in decision_rows),
        "skip_rate": _mean(float(row["mode"] == "SKIP") for row in decision_rows),
        "mode_distribution": _distribution(row["mode"] for row in decision_rows),
        "mechanism_distribution": _distribution(row["update_mechanism"] for row in decision_rows),
        "output_dir": str(output_dir),
    }


def _policy_offset(policy: str) -> int:
    return sum(ord(char) for char in policy)


def _mechanism_label(mechanisms: dict[str, str]) -> str:
    return ";".join(f"{key}:{value}" for key, value in sorted(mechanisms.items()))


def _mean(values: Any) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def _distribution(values: Any) -> str:
    counts: dict[str, int] = {}
    total = 0
    for value in values:
        counts[value] = counts.get(value, 0) + 1
        total += 1
    return ";".join(
        f"{value}:{count}({count / max(total, 1):.3f})"
        for value, count in sorted(counts.items())
    )


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
