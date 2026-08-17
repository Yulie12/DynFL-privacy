from __future__ import annotations

import csv
import json
import pickle
import random
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .dynamic_training import _distribution, _mechanism_label, _policy_offset, _update_mechanism
from .fmnist_dynamic_training import FmnistDynamicConfig as _FmnistDynamicConfig
from .fmnist_dynamic_training import load_fmnist_arrays
from .flow_executor import ClientFlowInput, execute_mixed_round_flow
from .he_backend import check_he_backend, has_he_mechanism
from .lenet5_training import (
    count_params,
)
from .split_learning import (
    apply_unified_dp,
    build_split_models,
    fedavg_split,
    normalize_model_name,
    split_evaluate,
    split_local_train_lenet5,
)
from .nodes import build_profiles
from .selection import Candidate, SelectionConfig, choose_candidate, enumerate_candidates
from .training import MODE_SPECS


ROOT = Path(__file__).resolve().parents[1]

DRIFTRACE_DATA_MEAN = {
    "fmnist": [0.286],
    "cifar10": [0.4914, 0.4822, 0.4465],
    "cifar100": [0.5071, 0.4865, 0.4409],
}

DRIFTRACE_DATA_STD = {
    "fmnist": [0.3205],
    "cifar10": [0.2023, 0.1994, 0.201],
    "cifar100": [0.2009, 0.1984, 0.2023],
}


@dataclass(frozen=True)
class Lenet5Config:
    dataset_name: str = "fmnist"
    model_name: str = "lenet5"
    model_revision: str = "groupnorm_v2"
    local_epochs: int = 2
    learning_rate: float = 0.15
    l2: float = 0.0001
    iid: bool = False
    partition_mode: str = "client_noniid"
    selection_period: int = 1
    dp_clip_norm: float = 20.0
    dp_noise_multiplier: float = 0.005
    dp_update_mode: str = "any_dp"
    test_size: float = 0.25
    device: str = "cpu"
    he_backend: str = "none"
    he_local_deps: str | None = ".he_deps"
    require_real_he: bool = False


def _should_apply_update_dp(mechanisms: dict[str, str], update_mode: str) -> bool:
    if update_mode == "off":
        return False
    if update_mode == "upd_only":
        return mechanisms.get("upd") == "dp"
    return any(mechanism == "dp" for mechanism in mechanisms.values())


def run_fmnist_lenet5_training(
    selection: SelectionConfig,
    train_config: Lenet5Config,
    data_root: str | None = None,
    train_limit: int = 12000,
    test_limit: int = 2000,
    resume_from_run: str | None = None,
    policies: tuple[str, ...] = (
        "ours",
        "fixed_dp",
        "fixed_he",
        "random",
        "privacy_only",
        "no_protection",
    ),
) -> dict[str, Any]:
    output_dir = Path(selection.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(train_config.device)
    dataset_name = _normalize_dataset_name(train_config.dataset_name)
    resolved_data_root = data_root or _default_data_root(dataset_name)

    x_train, y_train, x_test, y_test, input_shape, dataset_label, num_classes = load_image_dataset_arrays(
        dataset_name, Path(resolved_data_root), train_limit, test_limit, selection.seed
    )
    client_indices = _partition_clients_lenet5(
        y_train=y_train,
        num_clients=selection.num_clients,
        num_edges=selection.num_edges,
        iid=train_config.iid,
        partition_mode=train_config.partition_mode,
        seed=selection.seed,
    )
    client_train_indices, client_test_indices = _split_client_indices(
        client_indices, test_ratio=0.2, seed=selection.seed,
    )
    clients, edges = build_profiles(
        num_clients=selection.num_clients,
        num_edges=selection.num_edges,
        client_heterogeneity=selection.client_heterogeneity,
        edge_heterogeneity=selection.edge_heterogeneity,
        seed=selection.seed,
    )
    edge_by_id = {edge.edge_id: edge for edge in edges}
    validation_indices = np.concatenate(client_test_indices) if client_test_indices else np.array([], dtype=int)
    if len(validation_indices) > 0:
        x_val = x_train[validation_indices]
        y_val = y_train[validation_indices]
    else:
        x_val = x_test
        y_val = y_test

    summaries = []
    for policy_index, policy in enumerate(policies, start=1):
        _write_live_status(
            output_dir / "live_status.json",
            {
                "status": "running",
                "active_policy": policy,
                "policy_index": policy_index,
                "num_policies": len(policies),
                "round": 0,
                "rounds": selection.rounds,
                "progress": 0.0,
                "message": f"Starting policy {policy} ({policy_index}/{len(policies)})",
                "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            },
        )
        summaries.append(
            _run_lenet5_policy(
                policy=policy,
                selection=selection,
                train_config=train_config,
                clients=clients,
                edge_by_id=edge_by_id,
                train_client_indices=client_train_indices,
                client_test_indices=client_test_indices,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                x_test=x_test,
                y_test=y_test,
                device=device,
                input_shape=input_shape,
                num_classes=num_classes,
                dataset_label=dataset_label,
                output_dir=output_dir / policy,
                resume_from_policy_dir=(Path(resume_from_run) / policy) if resume_from_run else None,
            )
        )

    _write_csv(output_dir / "summary_table.csv", summaries)
    _write_json(
        output_dir / "config.json",
        {
            "selection": selection.__dict__,
            "training": train_config.__dict__,
            "data_root": resolved_data_root,
            "train_limit": train_limit,
            "test_limit": test_limit,
            "resume_from_run": resume_from_run,
            "policies": list(policies),
        },
    )
    final_live_status = _read_json_or_empty(output_dir / "live_status.json")
    final_live_status.update(
        {
            "status": "completed",
            "active_policy": None,
            "policy_index": len(policies),
            "num_policies": len(policies),
            "round": selection.rounds,
            "rounds": selection.rounds,
            "progress": 1.0,
            "summary_table": str(output_dir / "summary_table.csv"),
            "summaries": summaries,
            "message": "All policies completed",
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    _write_live_status(
        output_dir / "live_status.json",
        final_live_status,
    )
    print(f"[OK] wrote {len(summaries)} LeNet5 runs to {output_dir}")
    return {
        "output_dir": str(output_dir),
        "summary_table": str(output_dir / "summary_table.csv"),
        "summaries": summaries,
    }


def _normalize_dataset_name(dataset_name: str) -> str:
    value = str(dataset_name or "fmnist").strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "fmnist": "fmnist",
        "fashionmnist": "fmnist",
        "fashion": "fmnist",
        "cifar10": "cifar10",
        "cifar": "cifar10",
        "cifar100": "cifar100",
    }
    if value not in aliases:
        raise ValueError("Unsupported dataset. Choose 'fmnist', 'cifar10', or 'cifar100'.")
    return aliases[value]


def _default_data_root(dataset_name: str) -> str:
    if dataset_name == "cifar10":
        return str(ROOT / "experiments" / "data" / "cifar10")
    if dataset_name == "cifar100":
        return str(ROOT / "experiments" / "data" / "cifar100")
    return str(ROOT / "experiments" / "data" / "fmnist" / "FashionMNIST" / "raw")


def load_image_dataset_arrays(
    dataset_name: str,
    data_root: Path,
    train_limit: int,
    test_limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, tuple[int, int, int], str, int]:
    dataset_name = _normalize_dataset_name(dataset_name)
    if dataset_name == "cifar10":
        return (*_load_cifar10_arrays(data_root, train_limit, test_limit, seed), (3, 32, 32), "CIFAR-10", 10)
    if dataset_name == "cifar100":
        return (*_load_cifar100_arrays(data_root, train_limit, test_limit, seed), (3, 32, 32), "CIFAR-100", 100)
    x_train, y_train, x_test, y_test = load_fmnist_arrays(
        data_root, train_limit, test_limit, seed
    )
    return x_train, y_train, x_test, y_test, (1, 28, 28), "Fashion-MNIST", 10


def _load_cifar10_arrays(
    data_root: Path,
    train_limit: int,
    test_limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from torchvision.datasets import CIFAR10
        data_root.mkdir(parents=True, exist_ok=True)
        train_set = CIFAR10(root=str(data_root), train=True, download=True)
        test_set = CIFAR10(root=str(data_root), train=False, download=True)
        train_data = train_set.data
        train_targets = np.asarray(train_set.targets, dtype=np.int64)
        test_data = test_set.data
        test_targets = np.asarray(test_set.targets, dtype=np.int64)
    except Exception:
        train_data, train_targets, test_data, test_targets = _load_cifar10_pickles(data_root)
    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(train_targets))[: min(train_limit, len(train_targets))]
    test_idx = rng.permutation(len(test_targets))[: min(test_limit, len(test_targets))]

    x_train = _normalize_cifar_images(train_data[train_idx], "cifar10")
    y_train = train_targets[train_idx]
    x_test = _normalize_cifar_images(test_data[test_idx], "cifar10")
    y_test = test_targets[test_idx]
    return x_train, y_train, x_test, y_test


def _load_cifar10_pickles(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_dir = data_root / "cifar-10-batches-py"
    train_data = []
    train_labels = []
    for index in range(1, 6):
        batch = _read_cifar_pickle(batch_dir / f"data_batch_{index}")
        train_data.append(batch["data"])
        train_labels.extend(batch["labels"])
    test_batch = _read_cifar_pickle(batch_dir / "test_batch")
    train = np.concatenate(train_data, axis=0).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test = test_batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return (
        train,
        np.asarray(train_labels, dtype=np.int64),
        test,
        np.asarray(test_batch["labels"], dtype=np.int64),
    )


def _read_cifar_pickle(path: Path) -> dict:
    with path.open("rb") as file:
        return pickle.load(file, encoding="latin1")


def _load_cifar100_arrays(
    data_root: Path,
    train_limit: int,
    test_limit: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    try:
        from torchvision.datasets import CIFAR100
        data_root.mkdir(parents=True, exist_ok=True)
        train_set = CIFAR100(root=str(data_root), train=True, download=True)
        test_set = CIFAR100(root=str(data_root), train=False, download=True)
        train_data = train_set.data
        train_targets = np.asarray(train_set.targets, dtype=np.int64)
        test_data = test_set.data
        test_targets = np.asarray(test_set.targets, dtype=np.int64)
    except Exception:
        train_data, train_targets, test_data, test_targets = _load_cifar100_pickles(data_root)
    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(train_targets))[: min(train_limit, len(train_targets))]
    test_idx = rng.permutation(len(test_targets))[: min(test_limit, len(test_targets))]
    return (
        _normalize_cifar_images(train_data[train_idx], "cifar100"),
        train_targets[train_idx],
        _normalize_cifar_images(test_data[test_idx], "cifar100"),
        test_targets[test_idx],
    )


def _load_cifar100_pickles(data_root: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch_dir = data_root / "cifar-100-python"
    train_batch = _read_cifar_pickle(batch_dir / "train")
    test_batch = _read_cifar_pickle(batch_dir / "test")
    train = train_batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test = test_batch["data"].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return (
        train,
        np.asarray(train_batch["fine_labels"], dtype=np.int64),
        test,
        np.asarray(test_batch["fine_labels"], dtype=np.int64),
    )


def _normalize_cifar_images(images: np.ndarray, dataset_name: str) -> np.ndarray:
    x = images.astype(np.float32) / 255.0
    mean = np.asarray(DRIFTRACE_DATA_MEAN[dataset_name], dtype=np.float32)
    std = np.asarray(DRIFTRACE_DATA_STD[dataset_name], dtype=np.float32)
    x = (x - mean.reshape(1, 1, 1, 3)) / std.reshape(1, 1, 1, 3)
    x = np.transpose(x, (0, 3, 1, 2))
    return x.reshape(x.shape[0], -1).astype(np.float32)


def _run_lenet5_policy(
    *,
    policy: str,
    selection: SelectionConfig,
    train_config: Lenet5Config,
    clients: list[Any],
    edge_by_id: dict[int, Any],
    train_client_indices: list[np.ndarray],
    client_test_indices: list[np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    device: torch.device,
    input_shape: tuple[int, int, int],
    num_classes: int,
    dataset_label: str,
    output_dir: Path,
    resume_from_policy_dir: Path | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(selection.seed + _policy_offset(policy))
    np_rng = np.random.default_rng(selection.seed + _policy_offset(policy) + 1543)
    he_status = check_he_backend(train_config.he_backend, train_config.he_local_deps)
    if policy == "fixed_he" and not he_status.available:
        raise RuntimeError(f"Policy fixed_he requires real HE, but backend is unavailable: {he_status.detail}")
    effective_selection = replace(selection, allow_he=he_status.available)

    model_name = normalize_model_name(train_config.model_name)
    global_end, global_edge, _full_model = build_split_models(
        model_name,
        device,
        input_channels=input_shape[0],
        image_size=input_shape[1],
        num_classes=num_classes,
    )
    remaining_epsilon = {client.client_id: selection.initial_epsilon for client in clients}

    round_rows: list[dict[str, Any]] = []
    decision_rows: list[dict[str, Any]] = []
    flow_event_rows: list[dict[str, Any]] = []
    best_accuracy = 0.0
    logical_time = 0.0
    client_by_id = {c.client_id: c for c in clients}
    policy_started_at = time.perf_counter()
    parent_status_path = output_dir.parent / "live_status.json"
    selection_period = max(1, int(train_config.selection_period))
    previous_choices: dict[int, Candidate] = {}
    start_round = 0

    checkpoint_path = output_dir / "checkpoint.pt"
    if resume_from_policy_dir is not None:
        checkpoint = _load_policy_checkpoint(resume_from_policy_dir, device)
        if checkpoint is None:
            raise FileNotFoundError(
                f"Cannot resume policy {policy}: missing checkpoint.pt under {resume_from_policy_dir}. "
                "Existing CSV-only runs cannot be continued without a saved model checkpoint."
            )
        global_end.load_state_dict(checkpoint["global_end_state"])
        global_edge.load_state_dict(checkpoint["global_edge_state"])
        remaining_epsilon = {
            int(key): float(value) for key, value in checkpoint["remaining_epsilon"].items()
        }
        round_rows = list(checkpoint.get("round_rows", []))
        decision_rows = list(checkpoint.get("decision_rows", []))
        flow_event_rows = list(checkpoint.get("flow_event_rows", []))
        best_accuracy = float(checkpoint.get("best_accuracy", 0.0))
        logical_time = float(checkpoint.get("logical_time", 0.0))
        previous_choices = dict(checkpoint.get("previous_choices", {}))
        start_round = int(checkpoint.get("next_round", len(round_rows)))
        if "rng_state" in checkpoint:
            rng.setstate(checkpoint["rng_state"])
        if "np_rng_state" in checkpoint:
            np_rng.bit_generator.state = checkpoint["np_rng_state"]
        if start_round >= selection.rounds:
            print(
                f"  [{policy}] resume source already has {start_round} rounds; target={selection.rounds}",
                flush=True,
            )

    for round_idx in range(start_round, selection.rounds):
        selected = []
        round_comm = 0.0
        round_risk = 0.0
        infeasible = 0

        for client in clients:
            rem = remaining_epsilon[client.client_id]
            candidates = enumerate_candidates(
                config=effective_selection,
                client_id=client.client_id,
                edge_factor=edge_by_id[client.edge_id].compute_factor,
                compute_factor=client.compute_factor,
                samples=client.samples,
                remaining_epsilon=rem,
                round_idx=round_idx,
                rng=rng,
                policy=policy,
            )
            should_update = round_idx == 0 or round_idx % selection_period == 0
            candidate = None
            if not should_update:
                candidate = _reuse_previous_choice(
                    candidates,
                    previous_choices.get(client.client_id),
                    remaining_epsilon=rem,
                )
            if candidate is None:
                sensitivity = None
                if policy == "ours":
                    sensitivity = _client_coordination_sensitivity(
                        client.edge_id,
                        selection.num_edges,
                        train_config.partition_mode,
                    )
                candidate = choose_candidate(
                    candidates,
                    policy=policy,
                    rng=rng,
                    require_feasible=selection.require_feasible,
                    remaining_epsilon=rem,
                    end_sensitivity=sensitivity,
                )
            if train_config.require_real_he and has_he_mechanism(candidate.mechanisms) and not he_status.available:
                raise RuntimeError(
                    f"Policy {policy} selected HE mechanism {candidate.mechanisms}, "
                    f"but real HE backend is unavailable: {he_status.detail}"
                )
            selected.append((client.client_id, candidate, candidates, rem))

        should_update_policy = round_idx == 0 or round_idx % selection_period == 0
        if policy == "accuracy_oracle" and should_update_policy:
            selected = _coordinate_accuracy_oracle_round(
                selected=selected,
                client_by_id=client_by_id,
                train_client_indices=train_client_indices,
                x_train=x_train,
                y_train=y_train,
                x_val=x_val,
                y_val=y_val,
                global_end=global_end,
                global_edge=global_edge,
                train_config=train_config,
                model_name=model_name,
                input_shape=input_shape,
                num_classes=num_classes,
                device=device,
                np_rng=np_rng,
                top_k=3,
            )
        elif policy in {"best_accuracy", "performance_only"} and should_update_policy:
            selected = _coordinate_global_objective_round(
                selected,
                client_by_id,
                policy=policy,
            )
        elif policy == "ours":
            selected = _coordinate_ours_round(selected, client_by_id)

        previous_choices = {client_id: candidate for client_id, candidate, _candidates, _rem in selected}

        for client_id, candidate, _candidates, rem in selected:
            round_comm += candidate.communication_volume
            round_risk = max(round_risk, candidate.risk)
            infeasible += int(not candidate.feasible)
            remaining_epsilon[client_id] = max(0.0, rem - candidate.epsilon_used)
            rem = remaining_epsilon[client_id]
            decision_rows.append(
                {
                    "policy": policy,
                    "round": round_idx,
                    "client_id": client_id,
                    "edge_id": client_by_id[client_id].edge_id,
                    "mode": candidate.mode,
                    "mechanisms": _mechanism_label(candidate.mechanisms),
                    "update_mechanism": _update_mechanism(candidate.mechanisms),
                    "time": candidate.time,
                    "risk": candidate.risk,
                    "epsilon_used": candidate.epsilon_used,
                    "remaining_epsilon": rem,
                    "communication_volume": candidate.communication_volume,
                    "feasible": candidate.feasible,
                    "feasible_resource": candidate.feasible_resource,
                    "feasible_privacy": candidate.feasible_privacy,
                    "feasible_risk": candidate.feasible_risk,
                    "feasible_time": candidate.feasible_time,
                }
            )

        flow_inputs = []
        skipped_clients = 0
        dispatch_cursor = 0.0
        dispatch_sequence = 0

        for client_id, candidate, _candidates, _rem in selected:
            if candidate.mode == "SKIP":
                skipped_clients += 1
                continue
            idx = train_client_indices[client_id]
            if len(idx) == 0:
                continue

            t0 = time.perf_counter()
            dispatch_start = dispatch_cursor
            state_diff = split_local_train_lenet5(
                mode=candidate.mode,
                global_end_state=global_end.state_dict(),
                global_edge_state=global_edge.state_dict(),
                x=x_train[idx],
                y=y_train[idx],
                epochs=train_config.local_epochs,
                lr=train_config.learning_rate,
                device=device,
                model_name=model_name,
                input_shape=input_shape,
                num_classes=num_classes,
            )
            measured_local = time.perf_counter() - t0
            dispatch_cursor += measured_local

            has_dp = _should_apply_update_dp(candidate.mechanisms, train_config.dp_update_mode)
            state_diff = apply_unified_dp(
                state_diff,
                mechanism="dp" if has_dp else "none",
                clip_norm=train_config.dp_clip_norm,
                noise_multiplier=train_config.dp_noise_multiplier,
                rng=np_rng,
                device=device,
            )
            if not _state_diff_is_finite(state_diff):
                skipped_clients += 1
                continue
            # Replace estimated local with measured real time
            client_info = client_by_id[client_id]
            spec_mode = MODE_SPECS[candidate.mode]
            L = selection.L_block_cycles
            samples = client_info.samples
            compute_factor = client_info.compute_factor
            per_block_local = spec_mode.local_work * samples / 150.0 / max(L, 1) * compute_factor
            est_local = L * per_block_local
            flow_inputs.append(
                ClientFlowInput(
                    client_id=client_id,
                    edge_id=client_info.edge_id,
                    mode=candidate.mode,
                    candidate_time=candidate.time,
                    estimated_local_time=est_local,
                    measured_local_time=measured_local,
                    communication_volume=candidate.communication_volume,
                    state_diff=state_diff,
                    sample_count=len(idx),
                    dispatch_start_time=dispatch_start,
                    dispatch_sequence=dispatch_sequence,
                )
            )
            dispatch_sequence += 1

        flow_result = execute_mixed_round_flow(
            round_idx=round_idx,
            clients=flow_inputs,
            aggregation_fraction=selection.aggregation_fraction,
        )
        for event in flow_result.flow_events:
            flow_event_rows.append({"policy": policy, **event})

        selected_by_id = {client_id: candidate for client_id, candidate, _candidates, _rem in selected}
        global_updates = [
            (state_diff, sample_count, selected_by_id.get(client_id))
            for client_id, state_diff, sample_count in zip(
                flow_result.selected_client_ids,
                flow_result.state_diffs,
                flow_result.sample_counts,
            )
            if _mode_reaches_cloud(selected_by_id.get(client_id).mode if selected_by_id.get(client_id) else "")
        ]

        if global_updates:
            global_state_diffs = [item[0] for item in global_updates]
            global_sample_counts = [item[1] for item in global_updates]
            global_candidates = [item[2] for item in global_updates]
            use_real_he = (
                he_status.available
                and he_status.backend == "tenseal"
                and any(candidate is not None and has_he_mechanism(candidate.mechanisms) for candidate in global_candidates)
            )
            if use_real_he:
                global_end, global_edge = fedavg_split_tenseal(
                    global_state_diffs,
                    global_sample_counts,
                    global_end,
                    global_edge,
                    device,
                )
            else:
                global_end, global_edge = fedavg_split(
                    global_state_diffs,
                    global_sample_counts,
                    global_end,
                    global_edge,
                    device,
                )

        logical_time += flow_result.round_duration
        test_loss, test_accuracy = split_evaluate(
            global_end, global_edge, x_test, y_test, device, input_shape=input_shape
        )
        train_eval_idx = _train_eval_indices(len(y_train), selection.seed, round_idx)
        train_loss, train_accuracy = split_evaluate(
            global_end,
            global_edge,
            x_train[train_eval_idx],
            y_train[train_eval_idx],
            device,
            input_shape=input_shape,
        )
        best_accuracy = max(best_accuracy, test_accuracy)
        round_rows.append(
            {
                "policy": policy,
                "round": round_idx,
                "logical_time": logical_time,
                "round_duration": flow_result.round_duration,
                "test_accuracy": test_accuracy,
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "train_loss": train_loss,
                "best_accuracy": best_accuracy,
                "communication_volume": sum(item.communication_volume for item in flow_inputs if item.client_id in flow_result.selected_client_ids),
                "max_risk": round_risk,
                "min_remaining_epsilon": min(remaining_epsilon.values()),
                "infeasible_clients": infeasible,
                "skipped_clients": skipped_clients,
                "num_effective_clients": len(flow_result.selected_client_ids),
                "num_global_update_clients": len(global_updates),
                "num_effective_edges": flow_result.num_effective_edges,
                "waiting_time": flow_result.waiting_time,
                "edge_aggregation_time": flow_result.edge_aggregation_time,
                "cloud_aggregation_time": flow_result.cloud_aggregation_time,
                "return_time": flow_result.return_time,
                "num_clients": len(clients),
            }
        )
        current_round = round_rows[-1]
        total_epsilon_used = sum(float(row["epsilon_used"]) for row in decision_rows)
        cumulative_comm = sum(float(row["communication_volume"]) for row in round_rows)
        selected_details = [
            {
                "client_id": client_id,
                "edge_id": client_by_id[client_id].edge_id,
                "mode": candidate.mode,
                "mechanisms": _mechanism_label(candidate.mechanisms),
            }
            for client_id, candidate, _candidates, _rem in selected
            if client_id in flow_result.selected_client_ids
        ]
        live_payload = {
            "status": "running",
            "active_policy": policy,
            "policy": policy,
            "round": round_idx + 1,
            "rounds": selection.rounds,
            "progress": (round_idx + 1) / max(selection.rounds, 1),
            "test_accuracy": test_accuracy,
            "best_test_accuracy": best_accuracy,
            "train_accuracy": train_accuracy,
            "test_loss": test_loss,
            "train_loss": train_loss,
            "logical_time": logical_time,
            "round_duration": flow_result.round_duration,
            "communication_volume": current_round["communication_volume"],
            "cumulative_communication_volume": cumulative_comm,
            "epsilon_used": total_epsilon_used,
            "min_remaining_epsilon": current_round["min_remaining_epsilon"],
            "effective_clients": current_round["num_effective_clients"],
            "effective_edges": current_round["num_effective_edges"],
            "selected_client_ids": sorted(flow_result.selected_client_ids),
            "selected_clients": selected_details,
            "skipped_clients": skipped_clients,
            "infeasible_clients": infeasible,
            "mode_distribution": _distribution(row["mode"] for row in decision_rows),
            "wall_time_sec": time.perf_counter() - policy_started_at,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "recent_rounds": round_rows[-20:],
        }
        _write_live_status(output_dir / "live_status.json", live_payload)
        _write_live_status(parent_status_path, live_payload)
        _save_policy_checkpoint(
            checkpoint_path,
            policy=policy,
            next_round=round_idx + 1,
            global_end=global_end,
            global_edge=global_edge,
            remaining_epsilon=remaining_epsilon,
            previous_choices=previous_choices,
            round_rows=round_rows,
            decision_rows=decision_rows,
            flow_event_rows=flow_event_rows,
            best_accuracy=best_accuracy,
            logical_time=logical_time,
            rng=rng,
            np_rng=np_rng,
            train_config=train_config,
            selection=selection,
        )
        print(
            f"  [{policy}] round {round_idx + 1:03d}/{selection.rounds} "
            f"acc={test_accuracy:.4f} best={best_accuracy:.4f} "
            f"time={logical_time:.2f}s eps={total_epsilon_used:.3f} "
            f"clients={current_round['num_effective_clients']}",
            flush=True,
        )

    # Per-client test accuracy on each client's held-out data
    per_client_test = []
    for client_id in range(len(client_test_indices)):
        test_idx = client_test_indices[client_id]
        if len(test_idx) > 0:
            _loss, client_acc = split_evaluate(
                global_end, global_edge, x_train[test_idx], y_train[test_idx], device, input_shape=input_shape
            )
        else:
            client_acc = 0.0
        per_client_test.append({"client_id": client_id, "test_accuracy": float(client_acc)})

    client_accs = [item["test_accuracy"] for item in per_client_test]
    _write_csv(output_dir / "per_client_test.csv", per_client_test)
    print(f"  [{policy}] per-client test acc: mean={np.mean(client_accs):.4f}  "
          f"min={np.min(client_accs):.4f}  max={np.max(client_accs):.4f}")

    summary = _summarize_lenet5_policy(
        policy, round_rows, decision_rows, output_dir, selection.time_limit, model_name, dataset_label,
    )
    summary["per_client_test_accuracy_mean"] = float(np.mean(client_accs))
    summary["per_client_test_accuracy_min"] = float(np.min(client_accs))
    summary["per_client_test_accuracy_max"] = float(np.max(client_accs))
    summary["per_client_test_accuracy_std"] = float(np.std(client_accs))
    summary["per_client_test"] = per_client_test
    summary["he_backend"] = he_status.backend
    summary["he_available"] = he_status.available
    summary["he_status"] = he_status.detail
    _write_csv(output_dir / "round_metrics.csv", round_rows)
    _write_csv(output_dir / "client_decisions.csv", decision_rows)
    _write_csv(output_dir / "flow_events.csv", flow_event_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_live_status(
        output_dir / "live_status.json",
        {
            "status": "policy_completed",
            "active_policy": policy,
            "policy": policy,
            "round": selection.rounds,
            "rounds": selection.rounds,
            "progress": 1.0,
            "summary": summary,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "recent_rounds": round_rows[-20:],
        },
    )

    # Log model size
    n_params = count_params(global_end) + count_params(global_edge)
    final_test_accuracy = float(summary.get("final_test_accuracy", 0.0))
    print(f"  [{policy}] model_params={n_params}, test_acc={final_test_accuracy:.4f}, "
          f"total_time={logical_time:.2f}s, feasible={summary['feasible_rate']:.3f}")

    return summary


def _reuse_previous_choice(
    candidates: list[Candidate],
    previous: Candidate | None,
    *,
    remaining_epsilon: float,
) -> Candidate | None:
    if previous is None or previous.mode == "SKIP":
        return None
    for candidate in candidates:
        if candidate.mode != previous.mode:
            continue
        if candidate.mechanisms != previous.mechanisms:
            continue
        if not candidate.feasible_resource:
            continue
        if candidate.epsilon_used > remaining_epsilon + 1e-12:
            continue
        if candidate.feasible:
            return candidate
    return None


def _client_coordination_sensitivity(edge_id: int, num_edges: int, partition_mode: str) -> float:
    """Map edge position to the time/accuracy tradeoff used by DynFedPrivacy."""
    if partition_mode in {"edge_label_skew", "extreme_edge_label_skew"} and num_edges > 1:
        # Later edges carry labels that are less visible to other edges, so the
        # global model benefits from accuracy-oriented choices there.
        edge_rank = edge_id / max(num_edges - 1, 1)
        return max(0.18, min(0.62, 0.62 - 0.34 * edge_rank))
    return 0.42


def _coordinate_ours_round(
    selected: list[tuple[int, Candidate, list[Candidate], float]],
    client_by_id: dict[int, Any],
) -> list[tuple[int, Candidate, list[Candidate], float]]:
    """Apply a round-level edge-cloud coordination step for DynFedPrivacy.

    The per-client selector is deliberately local. This pass makes the policy
    global: in label-skewed settings, every edge should have a chance to send
    at least one feasible update that reaches the cloud when such a candidate
    exists within the current privacy budget.
    """
    if not selected:
        return selected

    by_edge: dict[int, list[int]] = {}
    for idx, (client_id, _candidate, _candidates, _rem) in enumerate(selected):
        by_edge.setdefault(client_by_id[client_id].edge_id, []).append(idx)

    coordinated = list(selected)
    for edge_id, indices in by_edge.items():
        has_cloud = any(_mode_reaches_cloud(coordinated[idx][1].mode) for idx in indices)
        if has_cloud:
            continue

        best_idx: int | None = None
        best_candidate: Candidate | None = None
        best_score = float("-inf")
        for idx in indices:
            client_id, _current, candidates, rem = coordinated[idx]
            pool = [
                candidate for candidate in candidates
                if _mode_reaches_cloud(candidate.mode)
                and candidate.feasible
                and candidate.epsilon_used <= rem + 1e-12
            ]
            if not pool:
                pool = [
                    candidate for candidate in candidates
                    if _mode_reaches_cloud(candidate.mode)
                    and candidate.feasible_resource
                    and candidate.epsilon_used <= rem + 1e-12
                ]
            for candidate in pool:
                score = _global_coordination_score(candidate, candidates)
                if score > best_score:
                    best_idx = idx
                    best_candidate = candidate
                    best_score = score

        if best_idx is not None and best_candidate is not None:
            client_id, _current, candidates, rem = coordinated[best_idx]
            coordinated[best_idx] = (client_id, best_candidate, candidates, rem)

    return coordinated


def _global_coordination_score(candidate: Candidate, candidates: list[Candidate]) -> float:
    viable = [item for item in candidates if item.feasible_resource] or candidates
    time_values = [item.time for item in viable]
    acc_values = [item.accuracy for item in viable]
    eps_values = [item.epsilon_used for item in viable]
    risk_values = [item.risk for item in viable]

    time_norm = _safe_norm(candidate.time, min(time_values), max(time_values))
    acc_norm = _safe_norm(candidate.accuracy, min(acc_values), max(acc_values))
    eps_norm = _safe_norm(candidate.epsilon_used, min(eps_values), max(eps_values))
    risk_norm = _safe_norm(candidate.risk, min(risk_values), max(risk_values))
    return 0.58 * acc_norm - 0.22 * time_norm - 0.12 * eps_norm - 0.08 * risk_norm


def _coordinate_accuracy_oracle_round(
    *,
    selected: list[tuple[int, Candidate, list[Candidate], float]],
    client_by_id: dict[int, Any],
    train_client_indices: list[np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    train_config: Lenet5Config,
    model_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    device: torch.device,
    np_rng: np.random.Generator,
    top_k: int = 3,
) -> list[tuple[int, Candidate, list[Candidate], float]]:
    """Greedy validation oracle for an empirical accuracy upper reference."""
    if not selected or len(x_val) == 0:
        return selected

    chosen_updates: list[tuple[dict[str, dict[str, torch.Tensor]], int]] = []
    oracle_rows: dict[int, tuple[Candidate, list[Candidate], float]] = {}
    base_rng_state = np_rng.bit_generator.state
    ordered = sorted(selected, key=lambda item: (int(getattr(client_by_id[item[0]], "edge_id", 0)), item[0]))

    for client_id, current, candidates, rem in ordered:
        idx = train_client_indices[client_id]
        if len(idx) == 0:
            oracle_rows[client_id] = (current, candidates, rem)
            continue

        trial_candidates = _accuracy_oracle_candidate_pool(candidates, current, rem, top_k=top_k)
        best_candidate = current
        best_update: tuple[dict[str, dict[str, torch.Tensor]], int] | None = None
        best_score = (-1.0, -float(current.time), -float(current.risk))

        for candidate in trial_candidates:
            eval_rng = np.random.default_rng()
            eval_rng.bit_generator.state = base_rng_state
            state_diff = split_local_train_lenet5(
                mode=candidate.mode,
                global_end_state=global_end.state_dict(),
                global_edge_state=global_edge.state_dict(),
                x=x_train[idx],
                y=y_train[idx],
                epochs=train_config.local_epochs,
                lr=train_config.learning_rate,
                device=device,
                model_name=model_name,
                input_shape=input_shape,
                num_classes=num_classes,
            )
            if _should_apply_update_dp(candidate.mechanisms, train_config.dp_update_mode):
                state_diff = apply_unified_dp(
                    state_diff,
                    mechanism="dp",
                    clip_norm=train_config.dp_clip_norm,
                    noise_multiplier=train_config.dp_noise_multiplier,
                    rng=eval_rng,
                    device=device,
                )
            if not _state_diff_is_finite(state_diff):
                continue

            trial_updates = list(chosen_updates)
            if _mode_reaches_cloud(candidate.mode):
                trial_updates.append((state_diff, len(idx)))
            temp_end, temp_edge = _clone_split_models(
                global_end, global_edge, device, model_name, input_shape, num_classes
            )
            if trial_updates:
                temp_end, temp_edge = fedavg_split(
                    [item[0] for item in trial_updates],
                    [item[1] for item in trial_updates],
                    temp_end,
                    temp_edge,
                    device,
                )
            _loss, val_accuracy = split_evaluate(temp_end, temp_edge, x_val, y_val, device, input_shape=input_shape)
            score = (float(val_accuracy), -float(candidate.time), -float(candidate.risk))
            if score > best_score:
                best_score = score
                best_candidate = candidate
                best_update = (state_diff, len(idx)) if _mode_reaches_cloud(candidate.mode) else None

        if best_update is not None:
            chosen_updates.append(best_update)
        oracle_rows[client_id] = (best_candidate, candidates, rem)

    return [
        (client_id, oracle_rows[client_id][0], oracle_rows[client_id][1], oracle_rows[client_id][2])
        for client_id, _current, _candidates, _rem in selected
    ]


def _accuracy_oracle_candidate_pool(
    candidates: list[Candidate],
    current: Candidate,
    remaining_epsilon: float,
    *,
    top_k: int,
) -> list[Candidate]:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    if not feasible:
        feasible = [candidate for candidate in candidates if candidate.feasible_resource]
    budget_ok = [candidate for candidate in feasible if candidate.epsilon_used <= remaining_epsilon + 1e-12]
    pool = budget_ok or feasible or [current]
    ranked = sorted(pool, key=lambda item: (item.accuracy, _mode_reaches_cloud(item.mode), -item.time), reverse=True)
    return _dedupe_candidates([current, *ranked[:max(1, top_k)]])


def _clone_split_models(
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    device: torch.device,
    model_name: str,
    input_shape: tuple[int, int, int] = (1, 28, 28),
    num_classes: int = 10,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    temp_end, temp_edge, _full_model = build_split_models(
        model_name,
        device,
        input_channels=input_shape[0],
        image_size=input_shape[1],
        num_classes=num_classes,
    )
    temp_end.load_state_dict({key: value.detach().clone() for key, value in global_end.state_dict().items()})
    temp_edge.load_state_dict({key: value.detach().clone() for key, value in global_edge.state_dict().items()})
    return temp_end, temp_edge


def _coordinate_global_objective_round(
    selected: list[tuple[int, Candidate, list[Candidate], float]],
    client_by_id: dict[int, Any],
    *,
    policy: str,
    max_passes: int = 4,
) -> list[tuple[int, Candidate, list[Candidate], float]]:
    """Approximate the paper's system-level objective for oracle baselines.

    best_accuracy minimizes the global steady-error proxy only. performance_only
    minimizes an ideal-point distance between system latency and global
    steady-error, matching the T_sys/Omega_sys structure in the tex model.
    """
    if not selected:
        return selected

    pools: dict[int, list[Candidate]] = {}
    for client_id, current, candidates, rem in selected:
        pool = [
            candidate for candidate in candidates
            if candidate.feasible and candidate.epsilon_used <= rem + 1e-12
        ]
        if not pool:
            pool = [
                candidate for candidate in candidates
                if candidate.feasible_resource and candidate.epsilon_used <= rem + 1e-12
            ]
        if not pool:
            pool = [current]
        pools[client_id] = _dedupe_candidates(pool)

    current_by_client = {
        client_id: _initial_global_objective_candidate(pools[client_id], policy)
        for client_id, _current, _candidates, _rem in selected
    }
    bounds = _global_objective_bounds(pools, current_by_client, client_by_id)

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        for client_id in list(current_by_client):
            best_candidate = current_by_client[client_id]
            best_score = _global_objective_score(
                current_by_client,
                client_by_id,
                policy=policy,
                bounds=bounds,
            )
            for candidate in pools[client_id]:
                if candidate == current_by_client[client_id]:
                    continue
                trial = dict(current_by_client)
                trial[client_id] = candidate
                score = _global_objective_score(
                    trial,
                    client_by_id,
                    policy=policy,
                    bounds=bounds,
                )
                if score + 1e-12 < best_score:
                    best_score = score
                    best_candidate = candidate
            if best_candidate != current_by_client[client_id]:
                current_by_client[client_id] = best_candidate
                improved = True

    return [
        (client_id, current_by_client[client_id], candidates, rem)
        for client_id, _current, candidates, rem in selected
    ]


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[tuple] = set()
    unique = []
    for candidate in candidates:
        key = (candidate.mode, tuple(sorted(candidate.mechanisms.items())))
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _initial_global_objective_candidate(candidates: list[Candidate], policy: str) -> Candidate:
    if policy == "best_accuracy":
        return min(candidates, key=lambda item: (_local_omega_proxy(item), item.time, item.risk))
    return min(candidates, key=lambda item: (item.time, _local_omega_proxy(item), item.risk))


def _global_objective_bounds(
    pools: dict[int, list[Candidate]],
    current_by_client: dict[int, Candidate],
    client_by_id: dict[int, Any],
) -> dict[str, float]:
    fastest = {
        client_id: min(candidates, key=lambda item: item.time)
        for client_id, candidates in pools.items()
    }
    lowest_omega = {
        client_id: min(candidates, key=_local_omega_proxy)
        for client_id, candidates in pools.items()
    }
    all_candidates = [candidate for candidates in pools.values() for candidate in candidates]
    max_time = max((candidate.time for candidate in all_candidates), default=1.0)
    omega_values = [
        _global_omega_proxy(selection, client_by_id)
        for selection in (fastest, lowest_omega, current_by_client)
    ]
    return {
        "t_min": max((candidate.time for candidate in fastest.values()), default=0.0),
        "t_max": max_time,
        "omega_min": min(omega_values),
        "omega_max": max(max(omega_values), min(omega_values) + 1e-6),
    }


def _global_objective_score(
    selection: dict[int, Candidate],
    client_by_id: dict[int, Any],
    *,
    policy: str,
    bounds: dict[str, float],
) -> float:
    omega = _global_omega_proxy(selection, client_by_id)
    if policy == "best_accuracy":
        return omega

    t_sys = max((candidate.time for candidate in selection.values()), default=0.0)
    t_norm = _safe_norm(t_sys, bounds["t_min"], bounds["t_max"])
    omega_norm = _safe_norm(omega, bounds["omega_min"], bounds["omega_max"])
    return (0.5 * t_norm * t_norm + 0.5 * omega_norm * omega_norm) ** 0.5


def _global_omega_proxy(selection: dict[int, Candidate], client_by_id: dict[int, Any]) -> float:
    total_samples = sum(float(client_by_id[client_id].samples) for client_id in selection)
    if total_samples <= 0:
        total_samples = float(max(len(selection), 1))
    aggregation_sizes = _aggregation_sizes(selection, client_by_id)
    weighted_local = sum(
        float(client_by_id[client_id].samples)
        / total_samples
        * _local_omega_proxy(candidate, aggregation_size=aggregation_sizes.get(client_id, 1))
        for client_id, candidate in selection.items()
    )
    cloud_samples = sum(
        float(client_by_id[client_id].samples)
        for client_id, candidate in selection.items()
        if _mode_reaches_cloud(candidate.mode)
    )
    r_cloud = cloud_samples / total_samples
    cloud_penalty_weight = 0.06
    smooth = 0.05
    return weighted_local + cloud_penalty_weight / (r_cloud + smooth)


def _aggregation_sizes(selection: dict[int, Candidate], client_by_id: dict[int, Any]) -> dict[int, int]:
    cloud_count = sum(1 for candidate in selection.values() if _mode_reaches_cloud(candidate.mode))
    edge_counts: dict[int, int] = {}
    for client_id, candidate in selection.items():
        if _mode_reaches_cloud(candidate.mode):
            continue
        edge_id = int(getattr(client_by_id[client_id], "edge_id", -1))
        edge_counts[edge_id] = edge_counts.get(edge_id, 0) + 1

    sizes: dict[int, int] = {}
    for client_id, candidate in selection.items():
        if _mode_reaches_cloud(candidate.mode):
            sizes[client_id] = max(cloud_count, 1)
        else:
            edge_id = int(getattr(client_by_id[client_id], "edge_id", -1))
            sizes[client_id] = max(edge_counts.get(edge_id, 1), 1)
    return sizes


def _local_omega_proxy(candidate: Candidate, aggregation_size: int = 1) -> float:
    spec = MODE_SPECS.get(candidate.mode)
    mode_residual = float(spec.mode_penalty) if spec is not None else 0.0
    protected_feature_objects = {"emb", "label", "grad", "weakemb", "strongemb", "pseudo_label"}
    feature_dp = any(
        obj in protected_feature_objects and mechanism == "dp"
        for obj, mechanism in candidate.mechanisms.items()
    )
    update_dp = any(obj == "upd" and mechanism == "dp" for obj, mechanism in candidate.mechanisms.items())

    base_variance = 0.035
    feature_dp_bias = 0.08 if feature_dp else 0.0
    feature_dp_loss = 0.04 if feature_dp else 0.0
    update_dp_noise = 0.12 / max(float(aggregation_size), 1.0) ** 2 if update_dp else 0.0
    return mode_residual + base_variance + feature_dp_bias + feature_dp_loss + update_dp_noise


def _safe_norm(value: float, low: float, high: float) -> float:
    span = high - low
    if abs(span) <= 1e-12:
        return 0.0
    return (value - low) / span


def _mode_reaches_cloud(mode: str) -> bool:
    spec = MODE_SPECS.get(mode)
    if spec is None:
        return False
    return spec.client_target == "cloud" or bool(spec.edge_to_cloud_objects) or spec.cloud_work > 0.0


def _save_policy_checkpoint(
    path: Path,
    *,
    policy: str,
    next_round: int,
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    remaining_epsilon: dict[int, float],
    previous_choices: dict[int, Candidate],
    round_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    flow_event_rows: list[dict[str, Any]],
    best_accuracy: float,
    logical_time: float,
    rng: random.Random,
    np_rng: np.random.Generator,
    train_config: Lenet5Config,
    selection: SelectionConfig,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy": policy,
            "next_round": next_round,
            "global_end_state": global_end.state_dict(),
            "global_edge_state": global_edge.state_dict(),
            "remaining_epsilon": remaining_epsilon,
            "previous_choices": previous_choices,
            "round_rows": round_rows,
            "decision_rows": decision_rows,
            "flow_event_rows": flow_event_rows,
            "best_accuracy": best_accuracy,
            "logical_time": logical_time,
            "rng_state": rng.getstate(),
            "np_rng_state": np_rng.bit_generator.state,
            "training": train_config.__dict__,
            "selection": selection.__dict__,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        path,
    )


def _load_policy_checkpoint(policy_dir: Path, device: torch.device) -> dict[str, Any] | None:
    path = policy_dir / "checkpoint.pt"
    if not path.exists():
        return None
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _partition_clients_lenet5(
    y_train: np.ndarray,
    num_clients: int,
    num_edges: int,
    iid: bool,
    partition_mode: str,
    seed: int,
) -> list[np.ndarray]:
    """Same as _partition_clients from real_training; copied to avoid cross-dep."""
    rng = np.random.default_rng(seed)
    indices = np.arange(len(y_train))
    if iid or partition_mode == "iid":
        rng.shuffle(indices)
        return [part.astype(np.int64) for part in np.array_split(indices, num_clients)]

    if partition_mode == "edge_label_skew":
        return _partition_edge_label_skew(y_train, num_clients, num_edges, rng, extreme=False)
    if partition_mode == "extreme_edge_label_skew":
        return _partition_edge_label_skew(y_train, num_clients, num_edges, rng, extreme=True)

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


def _split_client_indices(
    client_indices: list[np.ndarray],
    test_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Hold out a portion of each client's training data as a local test set."""
    rng = np.random.default_rng(seed + 9999)
    train_list: list[np.ndarray] = []
    test_list: list[np.ndarray] = []
    for indices in client_indices:
        arr = np.asarray(indices)
        if len(arr) < 2:
            train_list.append(arr)
            test_list.append(np.array([], dtype=np.int64))
            continue
        shuffled = arr.copy()
        rng.shuffle(shuffled)
        split = max(1, int(len(shuffled) * (1.0 - test_ratio)))
        train_list.append(np.sort(shuffled[:split]))
        test_list.append(np.sort(shuffled[split:]))
    return train_list, test_list


def _partition_edge_label_skew(
    y_train: np.ndarray,
    num_clients: int,
    num_edges: int,
    rng: np.random.Generator,
    extreme: bool = False,
) -> list[np.ndarray]:
    """Assign contiguous label groups to edges, then split each edge group over its clients."""
    indices = np.arange(len(y_train))
    labels = sorted(int(label) for label in np.unique(y_train))
    client_parts = [[] for _ in range(num_clients)]
    edge_clients = {
        edge_id: [client_id for client_id in range(num_clients) if client_id % num_edges == edge_id]
        for edge_id in range(num_edges)
    }
    edge_order = list(range(num_edges))
    if extreme:
        # With three edges this yields roughly 0-2, 3-5, 6-9.
        edge_order = [min(num_edges - 1, int(i * num_edges / max(len(labels), 1))) for i in range(len(labels))]

    for label_pos, label in enumerate(labels):
        label_indices = indices[y_train == label]
        rng.shuffle(label_indices)
        edge_id = edge_order[label_pos] if extreme else min(num_edges - 1, int((label_pos * num_edges) / max(len(labels), 1)))
        clients = edge_clients.get(edge_id) or list(range(num_clients))
        if extreme and len(clients) > 1:
            width = max(1, min(2, len(clients)))
            start = label_pos % len(clients)
            clients = [clients[(start + offset) % len(clients)] for offset in range(width)]
        chunks = np.array_split(label_indices, len(clients))
        for client_id, chunk in zip(clients, chunks):
            client_parts[client_id].extend(chunk.tolist())

    empty_clients = [client_id for client_id, part in enumerate(client_parts) if not part]
    if empty_clients:
        rng.shuffle(indices)
        fillers = np.array_split(indices[: len(empty_clients) * 4], len(empty_clients))
        for client_id, filler in zip(empty_clients, fillers):
            client_parts[client_id].extend(filler.tolist())

    return [np.array(sorted(set(part)), dtype=np.int64) for part in client_parts]


def _state_diff_is_finite(diff: dict[str, dict[str, torch.Tensor]]) -> bool:
    for part in diff.values():
        for value in part.values():
            if not torch.isfinite(value).all():
                return False
    return True


def _train_eval_indices(num_samples: int, seed: int, round_idx: int, limit: int = 2000) -> np.ndarray:
    if num_samples <= limit:
        return np.arange(num_samples, dtype=np.int64)
    rng = np.random.default_rng(seed + 10007 * (round_idx + 1))
    return rng.choice(num_samples, size=limit, replace=False).astype(np.int64)


def fedavg_split_tenseal(
    state_diffs: list[dict[str, dict[str, torch.Tensor]]],
    sample_counts: list[int],
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    device: torch.device,
    *,
    chunk_size: int = 4096,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    import tenseal as ts

    total = max(1, sum(sample_counts))
    flat_updates = [_flatten_state_diff(diff, global_end, global_edge, device) for diff in state_diffs]
    if not flat_updates:
        return global_end, global_edge

    size = int(flat_updates[0].numel())
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=8192,
        coeff_mod_bit_sizes=[60, 40, 40, 60],
    )
    context.global_scale = 2 ** 40
    encrypted_sum = None
    for flat_update, count in zip(flat_updates, sample_counts):
        factor = float(count) / float(total)
        weighted = (flat_update.detach().cpu().numpy().astype(np.float64) * factor).tolist()
        chunks = [
            ts.ckks_vector(context, weighted[start:start + chunk_size])
            for start in range(0, size, chunk_size)
        ]
        if encrypted_sum is None:
            encrypted_sum = chunks
        else:
            for idx, chunk in enumerate(chunks):
                encrypted_sum[idx] = encrypted_sum[idx] + chunk

    if encrypted_sum is None:
        return global_end, global_edge
    aggregated: list[float] = []
    for chunk in encrypted_sum:
        aggregated.extend(chunk.decrypt())
    aggregated_tensor = torch.tensor(aggregated[:size], dtype=torch.float32, device=device)
    _apply_flat_update(aggregated_tensor, global_end, global_edge)
    return global_end, global_edge


def _flatten_state_diff(
    diff: dict[str, dict[str, torch.Tensor]],
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    device: torch.device,
) -> torch.Tensor:
    pieces = []
    for part_name, model in (("end", global_end), ("edge", global_edge)):
        part = diff.get(part_name, {})
        for name, param in model.named_parameters():
            value = part.get(name)
            if value is None:
                value = torch.zeros_like(param.data, device=device)
            pieces.append(value.detach().to(device).reshape(-1))
    return torch.cat(pieces)


def _apply_flat_update(flat_update: torch.Tensor, global_end: torch.nn.Module, global_edge: torch.nn.Module) -> None:
    cursor = 0
    for model in (global_end, global_edge):
        state = model.state_dict()
        for name, param in model.named_parameters():
            count = int(param.numel())
            update = flat_update[cursor:cursor + count].view_as(param.data).to(param.data.device)
            state[name].data += update
            cursor += count


def _summarize_lenet5_policy(
    policy: str,
    round_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    output_dir: Path,
    time_limit: float,
    model_name: str,
    dataset_label: str = "Fashion-MNIST",
) -> dict[str, Any]:
    final = round_rows[-1]
    best = max(round_rows, key=lambda row: row["test_accuracy"])
    last_5 = round_rows[-5:]
    avg_last_5 = sum(r["test_accuracy"] for r in last_5) / max(len(last_5), 1)
    last_10 = round_rows[-10:]
    avg_last_10 = sum(r["test_accuracy"] for r in last_10) / max(len(last_10), 1)
    return {
        "policy": policy,
        "model": model_name,
        "dataset": dataset_label,
        "rounds": len(round_rows),
        "final_test_accuracy": final["test_accuracy"],
        "best_test_accuracy": best["test_accuracy"],
        "avg_last_5_accuracy": avg_last_5,
        "avg_last_10_accuracy": avg_last_10,
        "round_to_best": best["round"],
        "final_train_accuracy": final["train_accuracy"],
        "total_logical_time": final["logical_time"],
        "total_communication_volume": sum(row["communication_volume"] for row in round_rows),
        "total_waiting_time": sum(row["waiting_time"] for row in round_rows),
        "total_edge_aggregation_time": sum(row["edge_aggregation_time"] for row in round_rows),
        "total_cloud_aggregation_time": sum(row["cloud_aggregation_time"] for row in round_rows),
        "mean_effective_clients": _list_mean(row["num_effective_clients"] for row in round_rows),
        "mean_global_update_clients": _list_mean(row["num_global_update_clients"] for row in round_rows),
        "mean_effective_edges": _list_mean(row["num_effective_edges"] for row in round_rows),
        "max_privacy_risk": max(row["max_risk"] for row in round_rows),
        "min_remaining_epsilon": final["min_remaining_epsilon"],
        "total_epsilon_used": sum(row["epsilon_used"] for row in decision_rows),
        "feasible_rate": _list_mean(float(row["feasible_resource"]) for row in decision_rows),
        "all_constraint_feasible_rate": _list_mean(float(row["feasible"]) for row in decision_rows),
        "resource_feasible_rate": _list_mean(float(row["feasible_resource"]) for row in decision_rows),
        "privacy_feasible_rate": _list_mean(float(row["feasible_privacy"]) for row in decision_rows),
        "risk_feasible_rate": _list_mean(float(row["feasible_risk"]) for row in decision_rows),
        "time_satisfied_rate": _list_mean(float(float(row["time"]) <= time_limit) for row in decision_rows),
        "skip_rate": _list_mean(float(row["mode"] == "SKIP") for row in decision_rows),
        "mode_distribution": _distribution(row["mode"] for row in decision_rows),
        "update_mechanism_distribution": _distribution(row["update_mechanism"] for row in decision_rows),
        "object_mechanism_distribution": _object_mechanism_distribution(row["mechanisms"] for row in decision_rows),
        "output_dir": str(output_dir),
    }


def _list_mean(values: Any) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def _object_mechanism_distribution(values: Any) -> str:
    counts: dict[str, int] = {}
    total = 0
    for value in values:
        text = str(value)
        if not text:
            continue
        for item in text.split(";"):
            if ":" not in item:
                continue
            _, mechanism = item.split(":", 1)
            mechanism = mechanism.strip()
            counts[mechanism] = counts.get(mechanism, 0) + 1
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


def _read_json_or_empty(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        with path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except json.JSONDecodeError:
        return {}


def _write_live_status(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    for attempt in range(5):
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            with temp_path.open("w", encoding="utf-8") as file:
                json.dump(payload, file, indent=2, ensure_ascii=False)
            temp_path.replace(path)
            return
        except (FileNotFoundError, PermissionError):
            if attempt == 4:
                raise
            time.sleep(0.05 * (attempt + 1))
