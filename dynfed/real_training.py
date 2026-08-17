from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from .config import ExperimentConfig
from .nodes import build_profiles
from .privacy import normalize_mechanism
from .training import MODE_SPECS, _simulate_round


@dataclass(frozen=True)
class RealTrainingConfig:
    local_epochs: int = 2
    learning_rate: float = 0.15
    l2: float = 0.0001
    iid: bool = False
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 0.18
    test_size: float = 0.25


def run_real_federated_training(
    config: ExperimentConfig,
    train_config: RealTrainingConfig,
) -> dict[str, Any]:
    mode_name = config.mode.name.upper()
    if mode_name not in MODE_SPECS:
        raise ValueError(f"Unsupported mode {config.mode.name}. Choose one of {sorted(MODE_SPECS)}.")

    out_dir = config.output_path
    out_dir.mkdir(parents=True, exist_ok=True)

    x_train, x_test, y_train, y_test = _load_digits(config.runtime.seed, train_config.test_size)
    client_indices = _partition_clients(
        y_train=y_train,
        num_clients=config.topology.num_clients,
        iid=train_config.iid,
        seed=config.runtime.seed,
    )

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

    rng = random.Random(config.runtime.seed + 7919)
    np_rng = np.random.default_rng(config.runtime.seed + 1543)
    spec = MODE_SPECS[mode_name]

    num_features = x_train.shape[1]
    num_classes = int(np.max(y_train)) + 1
    weights = np.zeros((num_features, num_classes), dtype=np.float64)
    bias = np.zeros(num_classes, dtype=np.float64)

    logical_time = 0.0
    best_accuracy = 0.0
    rows: list[dict[str, Any]] = []
    events: list[dict[str, Any]] = []

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
        selected_client_ids = [
            cid for cid in round_result["selected_client_ids"] if len(client_indices[cid]) > 0
        ]

        updates = []
        sample_counts = []
        for client_id in selected_client_ids:
            idx = client_indices[client_id]
            local_w, local_b = _local_train_softmax(
                weights=weights,
                bias=bias,
                x=x_train[idx],
                y=y_train[idx],
                epochs=train_config.local_epochs,
                learning_rate=train_config.learning_rate,
                l2=train_config.l2,
            )
            delta_w = local_w - weights
            delta_b = local_b - bias
            delta_w, delta_b = _apply_privacy_to_update(
                delta_w=delta_w,
                delta_b=delta_b,
                mechanism=config.privacy.mechanism,
                epsilon=config.privacy.epsilon,
                clip_norm=train_config.dp_clip_norm,
                noise_multiplier=train_config.dp_noise_multiplier,
                rng=np_rng,
            )
            updates.append((delta_w, delta_b))
            sample_counts.append(len(idx))

        if updates:
            weights, bias = _fedavg_update(weights, bias, updates, sample_counts)

        test_loss, test_accuracy = _evaluate_softmax(weights, bias, x_test, y_test, train_config.l2)
        train_loss, train_accuracy = _evaluate_softmax(weights, bias, x_train, y_train, train_config.l2)
        best_accuracy = max(best_accuracy, test_accuracy)

        row = {
            "round": round_idx,
            "mode": mode_name,
            "privacy": config.privacy.mechanism.upper(),
            "epsilon": config.privacy.epsilon,
            "logical_time": logical_time,
            "round_duration": round_result["round_duration"],
            "test_accuracy": test_accuracy,
            "test_loss": test_loss,
            "train_accuracy": train_accuracy,
            "train_loss": train_loss,
            "best_accuracy": best_accuracy,
            "num_effective_clients": len(selected_client_ids),
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

    summary = _build_real_summary(rows, config, train_config, mode_name)
    _write_json(out_dir / "config.json", {"simulation": config.to_dict(), "real_training": train_config.__dict__})
    _write_csv(out_dir / "round_metrics.csv", rows)
    _write_csv(out_dir / "event_log.csv", events)
    _write_json(out_dir / "summary.json", summary)
    return summary


def _load_digits(seed: int, test_size: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_digits()
    x = data.data.astype(np.float64) / 16.0
    y = data.target.astype(np.int64)
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    return x_train, x_test, y_train, y_test


def _partition_clients(
    y_train: np.ndarray,
    num_clients: int,
    iid: bool,
    seed: int,
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_train))
    if iid:
        rng.shuffle(indices)
        return [part.astype(np.int64) for part in np.array_split(indices, num_clients)]

    client_parts = [[] for _ in range(num_clients)]
    for label in sorted(np.unique(y_train)):
        label_indices = indices[y_train == label]
        rng.shuffle(label_indices)
        preferred = [(int(label) * 3 + offset) % num_clients for offset in range(3)]
        chunks = np.array_split(label_indices, len(preferred))
        for client_id, chunk in zip(preferred, chunks):
            client_parts[client_id].extend(chunk.tolist())

    empty_clients = [i for i, part in enumerate(client_parts) if not part]
    if empty_clients:
        rng.shuffle(indices)
        fillers = np.array_split(indices[: len(empty_clients) * 4], len(empty_clients))
        for client_id, filler in zip(empty_clients, fillers):
            client_parts[client_id].extend(filler.tolist())

    return [np.array(sorted(set(part)), dtype=np.int64) for part in client_parts]


def _local_train_softmax(
    weights: np.ndarray,
    bias: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    learning_rate: float,
    l2: float,
) -> tuple[np.ndarray, np.ndarray]:
    local_w = weights.copy()
    local_b = bias.copy()
    y_onehot = np.eye(local_b.shape[0], dtype=np.float64)[y]
    n = max(1, len(y))

    for _ in range(epochs):
        probs = _softmax(x @ local_w + local_b)
        grad_logits = (probs - y_onehot) / n
        grad_w = x.T @ grad_logits + l2 * local_w
        grad_b = np.sum(grad_logits, axis=0)
        local_w -= learning_rate * grad_w
        local_b -= learning_rate * grad_b

    return local_w, local_b


def _apply_privacy_to_update(
    delta_w: np.ndarray,
    delta_b: np.ndarray,
    mechanism: str,
    epsilon: float,
    clip_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    mechanism = normalize_mechanism(mechanism)
    if mechanism == "none":
        return delta_w, delta_b

    flat_norm = math.sqrt(float(np.sum(delta_w * delta_w) + np.sum(delta_b * delta_b)))
    scale = min(1.0, clip_norm / max(flat_norm, 1e-12))
    delta_w = delta_w * scale
    delta_b = delta_b * scale

    if mechanism == "dp":
        sigma = noise_multiplier * clip_norm / max(float(epsilon), 1e-6)
        delta_w = delta_w + rng.normal(0.0, sigma, size=delta_w.shape)
        delta_b = delta_b + rng.normal(0.0, sigma, size=delta_b.shape)

    return delta_w, delta_b


def _fedavg_update(
    weights: np.ndarray,
    bias: np.ndarray,
    updates: list[tuple[np.ndarray, np.ndarray]],
    sample_counts: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    total = max(1, sum(sample_counts))
    avg_w = np.zeros_like(weights)
    avg_b = np.zeros_like(bias)
    for (delta_w, delta_b), count in zip(updates, sample_counts):
        factor = count / total
        avg_w += factor * delta_w
        avg_b += factor * delta_b
    return weights + avg_w, bias + avg_b


def _evaluate_softmax(
    weights: np.ndarray,
    bias: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    l2: float,
) -> tuple[float, float]:
    probs = _softmax(x @ weights + bias)
    loss = -np.mean(np.log(probs[np.arange(len(y)), y] + 1e-12)) + 0.5 * l2 * float(np.sum(weights * weights))
    pred = np.argmax(probs, axis=1)
    accuracy = float(np.mean(pred == y))
    return float(loss), accuracy


def _softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - np.max(logits, axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / np.sum(exp, axis=1, keepdims=True)


def _build_real_summary(
    rows: list[dict[str, Any]],
    config: ExperimentConfig,
    train_config: RealTrainingConfig,
    mode_name: str,
) -> dict[str, Any]:
    final = rows[-1]
    best = max(rows, key=lambda row: row["test_accuracy"])
    return {
        "mode": mode_name,
        "privacy": config.privacy.mechanism.upper(),
        "epsilon": config.privacy.epsilon,
        "dataset": "sklearn_digits",
        "iid": train_config.iid,
        "rounds": config.runtime.rounds,
        "final_test_accuracy": final["test_accuracy"],
        "best_test_accuracy": best["test_accuracy"],
        "round_to_best": best["round"],
        "final_train_accuracy": final["train_accuracy"],
        "total_logical_time": final["logical_time"],
        "total_communication_volume": sum(row["communication_volume"] for row in rows),
        "total_waiting_time": sum(row["waiting_time"] for row in rows),
        "accuracy_per_time": best["test_accuracy"] / max(final["logical_time"], 1e-9),
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
