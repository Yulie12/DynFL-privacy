from __future__ import annotations

import csv
import gzip
import json
import random
import struct
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.preprocessing import StandardScaler

from .dynamic_training import _distribution, _mechanism_label, _policy_offset, _update_mechanism
from .nodes import build_profiles
from .real_training import (
    RealTrainingConfig,
    _apply_privacy_to_update,
    _evaluate_softmax,
    _fedavg_update,
    _local_train_softmax,
    _partition_clients,
)
from .selection import SelectionConfig, choose_candidate, enumerate_candidates
from .training import MODE_SPECS


DRIFTRACE_FMNIST_MEAN = 0.286
DRIFTRACE_FMNIST_STD = 0.3205


@dataclass(frozen=True)
class FmnistDynamicConfig:
    selection: SelectionConfig
    training: RealTrainingConfig
    data_root: str = "E:/YTT/GROUP/DriftRace/data/fmnist/FashionMNIST/raw"
    train_limit: int = 12000
    test_limit: int = 2000
    policies: tuple[str, ...] = (
        "ours",
        "ours_ideal",
        "ours_knee",
        "ours_time_first",
        "ours_acc_first",
        "fixed_dp",
        "fixed_he",
        "random",
        "privacy_only",
        "no_protection",
    )


def run_fmnist_dynamic_training(config: FmnistDynamicConfig) -> dict[str, Any]:
    output_dir = Path(config.selection.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    x_train, y_train, x_test, y_test = load_fmnist_arrays(
        Path(config.data_root),
        train_limit=config.train_limit,
        test_limit=config.test_limit,
        seed=config.selection.seed,
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
            "data_root": config.data_root,
            "train_limit": config.train_limit,
            "test_limit": config.test_limit,
            "policies": list(config.policies),
        },
    )
    return {
        "output_dir": str(output_dir),
        "summary_table": str(output_dir / "summary_table.csv"),
        "summaries": summaries,
    }


def load_fmnist_arrays(
    raw_dir: Path,
    train_limit: int,
    test_limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_images = _read_idx_images(_find_idx(raw_dir, "train-images-idx3-ubyte"))
    train_labels = _read_idx_labels(_find_idx(raw_dir, "train-labels-idx1-ubyte"))
    test_images = _read_idx_images(_find_idx(raw_dir, "t10k-images-idx3-ubyte"))
    test_labels = _read_idx_labels(_find_idx(raw_dir, "t10k-labels-idx1-ubyte"))

    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(train_labels))[: min(train_limit, len(train_labels))]
    test_idx = rng.permutation(len(test_labels))[: min(test_limit, len(test_labels))]

    x_train = train_images[train_idx].reshape(len(train_idx), -1).astype(np.float64) / 255.0
    y_train = train_labels[train_idx].astype(np.int64)
    x_test = test_images[test_idx].reshape(len(test_idx), -1).astype(np.float64) / 255.0
    y_test = test_labels[test_idx].astype(np.int64)

    x_train = ((x_train.astype(np.float32) - DRIFTRACE_FMNIST_MEAN) / DRIFTRACE_FMNIST_STD).astype(np.float32)
    x_test = ((x_test.astype(np.float32) - DRIFTRACE_FMNIST_MEAN) / DRIFTRACE_FMNIST_STD).astype(np.float32)
    return x_train, y_train, x_test, y_test


def _run_policy(
    *,
    policy: str,
    config: FmnistDynamicConfig,
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

    weights = np.zeros((x_train.shape[1], int(np.max(y_train)) + 1), dtype=np.float64)
    bias = np.zeros(weights.shape[1], dtype=np.float64)
    remaining = {client.client_id: config.selection.initial_epsilon for client in clients}

    round_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    best_accuracy = 0.0
    logical_time = 0.0
    client_by_id = {c.client_id: c for c in clients}

    for round_idx in range(config.selection.rounds):
        selected = []
        round_time = 0.0
        round_comm = 0.0
        round_risk = 0.0
        infeasible = 0

        for client in clients:
            candidates = enumerate_candidates(
                config=config.selection,
                client_id=client.client_id,
                edge_factor=edge_by_id[client.edge_id].compute_factor,
                compute_factor=client.compute_factor,
                samples=client.samples,
                remaining_epsilon=remaining[client.client_id],
                round_idx=round_idx,
                rng=rng,
                policy=policy,
            )
            candidate = choose_candidate(
                candidates,
                policy=policy,
                rng=rng,
                require_feasible=config.selection.require_feasible,
            )
            remaining[client.client_id] = max(0.0, remaining[client.client_id] - candidate.epsilon_used)
            round_comm += candidate.communication_volume
            round_risk = max(round_risk, candidate.risk)
            infeasible += int(not candidate.feasible)
            selected.append((client.client_id, candidate))
            decision_rows.append(
                {
                    "policy": policy,
                    "round": round_idx,
                    "client_id": client.client_id,
                    "edge_id": client.edge_id,
                    "mode": candidate.mode,
                    "mechanisms": _mechanism_label(candidate.mechanisms),
                    "update_mechanism": _update_mechanism(candidate.mechanisms),
                    "time": candidate.time,
                    "risk": candidate.risk,
                    "epsilon_used": candidate.epsilon_used,
                    "remaining_epsilon": remaining[client.client_id],
                    "communication_volume": candidate.communication_volume,
                    "feasible": candidate.feasible,
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

            # Real measured local training time
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
                epsilon=max(config.selection.initial_epsilon, 1e-6),
                clip_norm=config.training.dp_clip_norm,
                noise_multiplier=config.training.dp_noise_multiplier,
                rng=np_rng,
            )
            updates.append((delta_w, delta_b))
            sample_counts.append(len(idx))

            # Compute per-client time: real computation + estimated comm/privacy/agg
            client_info = client_by_id[client_id]
            spec_mode = MODE_SPECS[candidate.mode]
            L = config.selection.L_block_cycles
            samples = client_info.samples
            compute_factor = client_info.compute_factor
            # Estimated local portion (from _estimate_candidate per-block * L)
            per_block_local = spec_mode.local_work * samples / 150.0 / max(L, 1) * compute_factor
            est_local = L * per_block_local
            # Replace estimated local with real measurement
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
        "dataset": "fmnist",
        "model": "softmax",
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


def _find_idx(raw_dir: Path, name: str) -> Path:
    plain = raw_dir / name
    if plain.exists():
        return plain
    gz = raw_dir / f"{name}.gz"
    if gz.exists():
        return gz
    raise FileNotFoundError(f"Cannot find {name} or {name}.gz under {raw_dir}")


def _read_bytes(path: Path) -> bytes:
    if path.suffix == ".gz":
        with gzip.open(path, "rb") as file:
            return file.read()
    return path.read_bytes()


def _read_idx_images(path: Path) -> np.ndarray:
    data = _read_bytes(path)
    magic, count, rows, cols = struct.unpack(">IIII", data[:16])
    if magic != 2051:
        raise ValueError(f"Unexpected image IDX magic {magic} in {path}")
    return np.frombuffer(data, dtype=np.uint8, offset=16).reshape(count, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    data = _read_bytes(path)
    magic, count = struct.unpack(">II", data[:8])
    if magic != 2049:
        raise ValueError(f"Unexpected label IDX magic {magic} in {path}")
    return np.frombuffer(data, dtype=np.uint8, offset=8).reshape(count)


def _mean(values: Any) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


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
