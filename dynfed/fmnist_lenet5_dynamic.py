from __future__ import annotations

import csv
import copy
import gc
import hashlib
import importlib.metadata
import json
import math
import os
import pickle
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, replace
from functools import wraps
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .dynamic_training import _distribution, _mechanism_label, _policy_offset, _update_mechanism
from .fmnist_dynamic_training import FmnistDynamicConfig as _FmnistDynamicConfig
from .fmnist_dynamic_training import load_fmnist_arrays
from .flow_executor import EDGE_CLOUD_MODES, EDGE_ONLY_MODES, ClientFlowInput, execute_mixed_round_flow
from .he_backend import (
    CKKS_COEFF_MOD_BIT_SIZES,
    CKKS_POLY_MODULUS_DEGREE,
    CKKS_SCALE,
    HEOperationMetrics,
    check_he_backend,
    decode_seal_vector,
)
from .lenet5_training import (
    count_params,
)
from .split_learning import (
    apply_unified_dp,
    build_split_pair,
    clip_state_difference,
    fedavg_split,
    gaussian_state_difference,
    normalize_model_name,
    split_evaluate,
    split_local_train_lenet5,
)
from .nodes import build_profiles
from .privacy import ClientPrivacyLedger, mechanism_uses_dp, mechanism_uses_he, privacy_execution_audit, training_privacy_diagnostics
from .protection_rules import audit_update_release
from .selection import (
    candidate_meets_update_goal,
    validate_update_protection_goal,
    Candidate,
    ProfileEvaluation,
    SelectionConfig,
    candidate_arrival_with_switch,
    candidate_has_he,
    candidate_link_mechanism,
    candidate_mechanism_label,
    candidate_mechanisms_for_object,
    choose_candidate,
    choose_global_pareto_profile,
    evaluate_global_profile,
    enumerate_candidates,
    build_client_privacy_ledger,
    resolved_privacy_parameters,
    _local_omega_proxy as selection_local_omega_proxy,
)
from .training import MODE_SPECS
from .version import CURRENT_EXECUTION_REVISION, CURRENT_UPDATE_PARAMETER_SCOPE


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
    update_parameter_scope: str = CURRENT_UPDATE_PARAMETER_SCOPE
    execution_revision: str = CURRENT_EXECUTION_REVISION
    local_epochs: int = 1
    learning_rate: float = 0.15
    l2: float = 0.0001
    iid: bool = False
    partition_mode: str = "client_noniid"
    selection_period: int = 5
    dp_clip_norm: float = 1.0
    dp_noise_multiplier: float = 0.0002
    dp_update_mode: str = "upd_only"
    dp_release_calibration: str = "tex_packet"
    test_size: float = 0.25
    device: str = "cpu"
    he_backend: str = "none"
    he_execution: str = "real"
    he_local_deps: str | None = ".he_deps"
    require_real_he: bool = False
    he_aggregation_size: int = 0
    he_workers: int = 1
    executor: str = "serial"
    executor_workers: int | None = None


def _should_apply_update_dp(
    mechanisms: dict[str, str],
    update_mode: str,
    mode: str = "",
) -> bool:
    if update_mode == "off":
        return False
    if not mechanism_uses_dp(str(mechanisms.get("upd", "none"))):
        return False
    # LIEIIIC applies update DP after its explicit edge aggregation loop.
    # For LIEIIC, the worker result is already the edge-side update that is
    # uploaded directly to the cloud, so this is the correct E->C boundary.
    return mode != "LIEIIIC"


def _candidate_training_mechanisms(
    candidate: Candidate,
    aggregate_cloud_update_dp: bool = False,
) -> dict[str, str]:
    mechanisms = dict(candidate.mechanisms)
    feature_links = {
        "LIE": ("L_E_emb", "L_E_grad"),
        "LIC": ("L_C_emb", "L_C_grad"),
        "LIEIIC": ("L_E_emb", "L_E_grad"),
        "LIEIIIC": ("L_E_emb", "L_E_grad"),
    }
    links = feature_links.get(candidate.mode)
    if links is not None:
        mechanisms["emb"] = candidate_link_mechanism(
            candidate, links[0], fallback_object="emb"
        )
        mechanisms["grad"] = candidate_link_mechanism(
            candidate, links[1], fallback_object="grad"
        )
    update_link = {
        "LIIE": "L_E_upd",
        "LIIC": "L_C_upd",
        "LIEIIC": "E_C_upd",
        "LIEIIIC": "E_C_upd",
        "LIIEIIIC": "L_E_upd",
    }.get(candidate.mode)
    if update_link is not None:
        mechanisms["upd"] = candidate_link_mechanism(candidate, update_link)
    if (
        aggregate_cloud_update_dp
        and mechanism_uses_dp(_candidate_cloud_update_mechanism(candidate))
    ):
        mechanisms["upd"] = (
            "he3"
            if mechanism_uses_he(_candidate_cloud_update_mechanism(candidate))
            else "none"
        )
    return mechanisms


def _candidate_cloud_update_mechanism(candidate: Candidate | None) -> str:
    if candidate is None:
        return "none"
    link_id = {
        "LIIC": "L_C_upd",
        "LIEIIC": "E_C_upd",
        "LIEIIIC": "E_C_upd",
        "LIIEIIIC": "E_C_upd",
    }.get(candidate.mode)
    if link_id is None:
        return "none"
    return candidate_link_mechanism(candidate, link_id)


def _candidate_edge_update_mechanism(candidate: Candidate | None) -> str:
    if candidate is None or candidate.mode not in {"LIIE", "LIIEIIIC"}:
        return "none"
    return candidate_link_mechanism(candidate, "L_E_upd")


def _candidate_uses_cross_domain_update_dp(
    candidate: Candidate | None,
    trusted_edge_split_execution: bool = False,
) -> bool:
    mechanism = _candidate_cloud_update_mechanism(candidate)
    return trusted_edge_split_execution and mechanism_uses_dp(mechanism)


def _candidate_uses_local_packet_update_dp(
    candidate: Candidate | None,
    trusted_edge_split_execution: bool = False,
) -> bool:
    mechanism = _candidate_cloud_update_mechanism(candidate)
    return (
        trusted_edge_split_execution
        and mechanism_uses_dp(mechanism)
        and not mechanism_uses_he(mechanism)
    )


def _candidate_uses_secure_aggregate_update_dp(
    candidate: Candidate | None,
    trusted_edge_split_execution: bool = False,
) -> bool:
    mechanism = _candidate_cloud_update_mechanism(candidate)
    return (
        trusted_edge_split_execution
        and mechanism_uses_dp(mechanism)
        and mechanism_uses_he(mechanism)
    )


def _client_train_worker(
    payload: dict[str, Any],
    model_cache: dict[str, Any] | None = None,
) -> dict[str, Any]:
    start = time.perf_counter()
    rng = np.random.default_rng(int(payload["dp_seed"]))
    worker_device = torch.device(payload.get("device", "cpu"))
    _set_torch_seed(int(payload["training_seed"]), worker_device)
    state_diff = split_local_train_lenet5(
        mode=payload["mode"],
        global_end_state=payload["global_end_state"],
        global_edge_state=payload["global_edge_state"],
        x=payload["x"],
        y=payload["y"],
        epochs=payload["epochs"],
        lr=payload["lr"],
        device=worker_device,
        model_name=payload["model_name"],
        input_shape=payload["input_shape"],
        num_classes=payload["num_classes"],
        mechanisms=payload["mechanisms"],
        dp_clip_norm=payload["dp_clip_norm"],
        dp_noise_multiplier=payload["dp_feature_noise_multiplier"],
        dp_rng=rng,
        dp_epsilon=payload["dp_epsilon"],
        l2=payload["l2"],
        local_steps=payload["local_steps"],
        training_seed=payload["training_seed"],
        model_cache=model_cache,
    )
    has_dp = _should_apply_update_dp(
        payload["mechanisms"],
        payload["dp_update_mode"],
        payload["mode"],
    )
    state_diff = apply_unified_dp(
        state_diff,
        mechanism="dp" if has_dp else "none",
        clip_norm=payload["dp_clip_norm"],
        noise_multiplier=payload["dp_update_noise_multiplier"],
        rng=rng,
        device=worker_device,
    )
    finite = _state_diff_is_finite(state_diff)
    returned_diff = _state_dict_to_device_nested(state_diff, torch.device("cpu"))
    return {
        "client_id": payload["client_id"],
        "state_diff": returned_diff,
        "measured_local": time.perf_counter() - start,
        "finite": finite,
    }


def run_fmnist_lenet5_training(
    selection: SelectionConfig,
    train_config: Lenet5Config,
    data_root: str | None = None,
    train_limit: int = 12000,
    test_limit: int = 2000,
    resume_from_run: str | None = None,
    max_new_rounds: int | None = None,
    policies: tuple[str, ...] = (
        "ours",
        "fixed_dp",
        "fixed_he",
        "random",
        "privacy_only",
        "no_protection",
    ),
) -> dict[str, Any]:
    if max_new_rounds is not None and max_new_rounds < 1:
        raise ValueError("max_new_rounds must be positive")
    for policy in policies:
        validate_update_protection_goal(selection, policy)
    if train_config.dp_release_calibration not in {"tex_packet", "legacy_aggregate"}:
        raise ValueError("Unknown DP release calibration")
    if train_config.dp_release_calibration == "tex_packet" and train_config.execution_revision != CURRENT_EXECUTION_REVISION:
        raise ValueError("TeX packet calibration requires the current execution revision")
    output_dir = Path(selection.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(train_config.device)
    dataset_name = _normalize_dataset_name(train_config.dataset_name)
    resolved_data_root = data_root or _default_data_root(dataset_name)

    x_train, y_train, x_test, y_test, input_shape, dataset_label, num_classes = load_image_dataset_arrays(
        dataset_name, Path(resolved_data_root), train_limit, test_limit, selection.seed
    )
    profile_end, profile_edge = build_split_pair(
        normalize_model_name(train_config.model_name),
        torch.device("cpu"),
        input_channels=input_shape[0],
        image_size=input_shape[1],
        num_classes=num_classes,
    )
    end_parameter_count = count_params(profile_end)
    edge_parameter_count = count_params(profile_edge)
    update_parameter_count = end_parameter_count + edge_parameter_count
    trainable_parameter_names = {
        part: [name for name, param in model.named_parameters() if param.requires_grad]
        for part, model in (("end", profile_end), ("edge", profile_edge))
    }
    trainable_parameter_count = sum(
        param.numel()
        for model in (profile_end, profile_edge)
        for param in model.parameters() if param.requires_grad
    )
    update_payload_mb = update_parameter_count * 4.0 / 1_000_000.0
    selection = replace(
        selection,
        omega_update_dimension=float(trainable_parameter_count),
        update_payload_mb=float(update_payload_mb),
    )
    model_partition = {
        "model_name": normalize_model_name(train_config.model_name),
        "split_location": (
            "after torchvision layer2"
            if "resnet" in normalize_model_name(train_config.model_name)
            else "between the convolutional encoder and fully connected classifier"
        ),
        "end_parameter_names": [name for name, _value in profile_end.named_parameters()],
        "edge_parameter_names": [name for name, _value in profile_edge.named_parameters()],
        "end_parameter_count": end_parameter_count,
        "edge_parameter_count": edge_parameter_count,
        "trainable_parameter_names": trainable_parameter_names,
        "trainable_parameter_count": trainable_parameter_count,
        "frozen_parameter_count": update_parameter_count - trainable_parameter_count,
        "dp_parameter_scope": train_config.update_parameter_scope,
        "transport_parameter_count": update_parameter_count,
        "update_payload_mb": update_payload_mb,
    }
    del profile_end, profile_edge
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
    clients = _profiles_with_actual_samples(clients, client_train_indices)
    edge_by_id = {edge.edge_id: edge for edge in edges}
    validation_indices = np.concatenate(client_test_indices) if client_test_indices else np.array([], dtype=int)
    if len(validation_indices) > 0:
        x_val = x_train[validation_indices]
        y_val = y_train[validation_indices]
    else:
        x_val = x_test
        y_val = y_test

    runtime_environment = _runtime_environment(device)
    partition_manifest = {
        "dataset": dataset_name,
        "seed": selection.seed,
        "subset_rule": "numpy default_rng(seed) permutation followed by the configured limit",
        "partition_mode": train_config.partition_mode,
        "validation_fraction_per_client": 0.2,
        "num_clients": selection.num_clients,
        "num_edges": selection.num_edges,
        "clients": [
            {
                "client_id": client.client_id,
                "edge_id": client.edge_id,
                "train_indices_in_selected_subset": [
                    int(index) for index in client_train_indices[client.client_id]
                ],
                "validation_indices_in_selected_subset": [
                    int(index) for index in client_test_indices[client.client_id]
                ],
                "train_label_counts": {
                    str(label): int(count)
                    for label, count in enumerate(
                        np.bincount(
                            y_train[client_train_indices[client.client_id]],
                            minlength=num_classes,
                        )
                    )
                    if int(count) > 0
                },
            }
            for client in clients
        ],
    }
    _write_json(output_dir / "data_partition.json", partition_manifest)
    _write_json(output_dir / "runtime_environment.json", runtime_environment)
    _write_json(output_dir / "model_partition.json", model_partition)
    _write_csv(
        output_dir / "device_profiles.csv",
        [
            {
                "client_id": client.client_id,
                "edge_id": client.edge_id,
                "train_samples": len(client_train_indices[client.client_id]),
                "validation_samples": len(client_test_indices[client.client_id]),
                "compute_factor": client.compute_factor,
                "memory_capacity_factor": client.memory_capacity_factor,
            }
            for client in clients
        ],
    )
    _write_csv(
        output_dir / "edge_profiles.csv",
        [
            {
                "edge_id": edge.edge_id,
                "compute_factor": edge.compute_factor,
            }
            for edge in edges
        ],
    )

    _write_json(
        output_dir / "config.json",
        {
            "selection": selection.__dict__,
            "resolved_privacy": resolved_privacy_parameters(selection),
            "training": train_config.__dict__,
            "data_root": resolved_data_root,
            "train_limit": train_limit,
            "test_limit": test_limit,
            "resume_from_run": resume_from_run,
            "max_new_rounds": max_new_rounds,
            "policies": list(policies),
            "runtime_environment": runtime_environment,
            "model_partition": model_partition,
            "artifact_files": [
                "data_partition.json",
                "device_profiles.csv",
                "edge_profiles.csv",
                "model_partition.json",
                "runtime_environment.json",
            ],
        },
    )

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
                max_new_rounds=max_new_rounds,
            )
        )

    all_completed = all(summary.get("status") == "completed" for summary in summaries)
    summary_table_name = "summary_table.csv" if all_completed else "partial_summary_table.csv"
    summary_table_path = output_dir / summary_table_name
    _write_csv(summary_table_path, summaries)
    _write_json(
        output_dir / "config.json",
        {
            "selection": selection.__dict__,
            "resolved_privacy": resolved_privacy_parameters(selection),
            "training": train_config.__dict__,
            "data_root": resolved_data_root,
            "train_limit": train_limit,
            "test_limit": test_limit,
            "resume_from_run": resume_from_run,
            "max_new_rounds": max_new_rounds,
            "policies": list(policies),
            "runtime_environment": runtime_environment,
            "model_partition": model_partition,
            "artifact_files": [
                "data_partition.json",
                "device_profiles.csv",
                "edge_profiles.csv",
                "model_partition.json",
                "runtime_environment.json",
            ],
        },
    )
    final_live_status = _read_json_or_empty(output_dir / "live_status.json")
    final_live_status.update(
        {
            "status": "completed" if all_completed else "paused",
            "active_policy": None,
            "policy_index": len(policies),
            "num_policies": len(policies),
            "round": min(
                (int(summary.get("rounds", 0)) for summary in summaries),
                default=0,
            ),
            "rounds": selection.rounds,
            "progress": (
                1.0
                if all_completed
                else min(
                    (float(summary.get("rounds", 0)) / max(selection.rounds, 1) for summary in summaries),
                    default=0.0,
                )
            ),
            "summary_table": str(summary_table_path),
            "summaries": summaries,
            "message": (
                "All policies completed"
                if all_completed
                else "Invocation round limit reached; checkpoints are ready to resume"
            ),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    )
    _write_live_status(
        output_dir / "live_status.json",
        final_live_status,
    )
    outcome = "OK" if all_completed else "PAUSED"
    print(f"[{outcome}] wrote {len(summaries)} policy runs to {output_dir}")
    return {
        "status": "completed" if all_completed else "paused",
        "output_dir": str(output_dir),
        "summary_table": str(summary_table_path),
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
    max_new_rounds: int | None = None,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    parent_status_path = output_dir.parent / "live_status.json"
    policy_started_at = time.perf_counter()
    last_completed_status: dict[str, Any] = {}

    def emit_stage_status(
        message: str,
        *,
        stage: str,
        round_value: int = 0,
        progress: float | None = None,
    ) -> None:
        stage_offsets = {
            "policy_init": 0.02,
            "he_check": 0.04,
            "privacy_accounting": 0.06,
            "model_build": 0.08,
            "client_state_init": 0.10,
            "candidate_enumeration": 0.15,
            "feature_clip_profiles": 0.25,
            "pareto_selection": 0.35,
            "accuracy_oracle_selection": 0.35,
            "global_objective_selection": 0.35,
            "client_training": 0.50,
            "flow_execution": 0.68,
            "aggregation": 0.78,
            "evaluation": 0.90,
        }
        if progress is None:
            progress = (round_value + stage_offsets.get(stage, 0.0)) / max(selection.rounds, 1)
        payload = {
            **last_completed_status,
            "status": "running",
            "active_policy": policy,
            "policy": policy,
            "round": round_value,
            "rounds": selection.rounds,
            "progress": float(progress),
            "stage": stage,
            "message": message,
            "wall_time_sec": time.perf_counter() - policy_started_at,
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_live_status(output_dir / "live_status.json", payload)
        _write_live_status(parent_status_path, payload)

    emit_stage_status("Initializing policy runtime", stage="policy_init")
    _set_torch_seed(selection.seed, device)
    rng = random.Random(selection.seed + _policy_offset(policy))
    np_rng = np.random.default_rng(selection.seed + 1543)
    emit_stage_status("Checking HE backend", stage="he_check")
    he_status = check_he_backend(train_config.he_backend, train_config.he_local_deps)
    _reset_ckks_runtime()
    real_he_available = he_status.available and he_status.backend in {"seal", "tenseal"}
    if train_config.he_execution not in {"real", "profiled"}:
        raise ValueError("he_execution must be real or profiled")
    execute_real_he = train_config.he_execution == "real" and real_he_available
    if train_config.he_execution == "profiled":
        print(
            f"  [{policy}] HE execution is profiled; selected HE updates use "
            "plaintext aggregation with modeled CKKS cost.",
            flush=True,
        )
    if train_config.require_real_he and not execute_real_he:
        raise RuntimeError(
            "Real HE execution was required, but it is not enabled or unavailable: "
            f"{he_status.detail}"
        )
    if policy == "fixed_he" and not real_he_available:
        raise RuntimeError(
            "Policy fixed_he requires a supported CKKS backend for real encrypted aggregation, "
            f"but it is unavailable: {he_status.detail}"
        )
    effective_selection = replace(
        selection,
        allow_he=real_he_available,
        omega_feature_clip_norm=train_config.dp_clip_norm,
        omega_update_clip_norm=train_config.dp_clip_norm,
    )
    emit_stage_status("Preparing privacy accountant", stage="privacy_accounting")
    privacy_parameters = resolved_privacy_parameters(effective_selection)

    model_name = normalize_model_name(train_config.model_name)
    emit_stage_status("Building split model", stage="model_build")
    global_end, global_edge = build_split_pair(
        model_name,
        device,
        input_channels=input_shape[0],
        image_size=input_shape[1],
        num_classes=num_classes,
    )
    privacy_ledgers = {
        client.client_id: build_client_privacy_ledger(effective_selection)
        for client in clients
    }
    remaining_epsilon = {
        client_id: ledger.remaining_budget
        for client_id, ledger in privacy_ledgers.items()
    }

    round_rows: list[dict[str, Any]] = []
    completed_wall_time_offset_sec = 0.0
    decision_rows: list[dict[str, Any]] = []
    flow_event_rows: list[dict[str, Any]] = []
    link_state_rows: list[dict[str, Any]] = []
    best_accuracy = 0.0
    logical_time = 0.0
    client_by_id = {c.client_id: c for c in clients}
    client_edges = {c.client_id: int(c.edge_id) for c in clients}
    edge_total_samples: dict[int, float] = {}
    for client in clients:
        edge_id = int(client.edge_id)
        edge_total_samples[edge_id] = edge_total_samples.get(edge_id, 0.0) + float(
            client.samples
        )
    selection_period = max(1, int(train_config.selection_period))
    previous_choices: dict[int, Candidate] = {}
    start_round = 0
    real_he_rounds = 0
    real_he_aggregated_clients = 0
    global_pareto_selection_rounds = 0
    client_model_states: dict[int, dict[str, dict[str, torch.Tensor]]] = {}

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
        saved_training = checkpoint.get("training", {})
        if saved_training.get("update_parameter_scope") != train_config.update_parameter_scope:
            raise RuntimeError(
                "Cannot resume a checkpoint with a different DP update parameter scope. "
                "Start a new run so frozen parameter protection does not change mid-training."
            )
        saved_execution_revision = saved_training.get("execution_revision")
        if saved_training.get("dp_release_calibration", "legacy_aggregate") != train_config.dp_release_calibration:
            raise ValueError("Cannot resume a different DP release calibration")
        if saved_execution_revision != train_config.execution_revision:
            raise RuntimeError(
                "Cannot resume across different training execution revisions: "
                f"checkpoint={saved_execution_revision!r}, "
                f"requested={train_config.execution_revision!r}. Start a new run so "
                "aggregation semantics do not change mid-training."
            )
        saved_he_execution = saved_training.get("he_execution", "real")
        if saved_he_execution != train_config.he_execution:
            raise RuntimeError(
                "Cannot resume across different HE execution modes: "
                f"checkpoint={saved_he_execution!r}, "
                f"requested={train_config.he_execution!r}."
            )
        saved_privacy_ledgers = checkpoint.get("privacy_ledgers")
        if not saved_privacy_ledgers:
            raise RuntimeError(
                "The resume checkpoint predates RDP privacy accounting. Start a new run so "
                "the reported privacy guarantee is not mixed with the legacy scalar budget."
            )
        saved_selection = checkpoint.get("selection", {})
        saved_rounds = int(saved_selection.get("rounds", selection.rounds))
        if effective_selection.dp_accounting_mode == "rdp_auto" and saved_rounds != selection.rounds:
            raise RuntimeError(
                "Cannot change the target round horizon when resuming an rdp_auto run: "
                f"checkpoint T={saved_rounds}, requested T={selection.rounds}. Start a new run "
                "so the Gaussian noise is calibrated for the complete horizon."
            )
        privacy_ledgers = {
            int(client_id): ClientPrivacyLedger.from_state_dict(state)
            for client_id, state in saved_privacy_ledgers.items()
        }
        for ledger in privacy_ledgers.values():
            expected = privacy_parameters
            if (
                abs(ledger.feature.budget - float(expected["feature_budget"])) > 1e-12
                or abs(ledger.update.budget - float(expected["update_budget"])) > 1e-12
                or abs(ledger.feature.delta - float(expected["delta"])) > 1e-15
                or abs(
                    ledger.feature_noise_multiplier
                    - float(expected["feature_noise_multiplier"])
                ) > 1e-12
                or abs(
                    ledger.update_noise_multiplier
                    - float(expected["update_noise_multiplier"])
                ) > 1e-12
            ):
                raise RuntimeError(
                    "Cannot resume with different RDP targets, delta, or calibrated noise. "
                    "Start a new run for the changed privacy configuration."
                )
        remaining_epsilon = {
            client_id: ledger.remaining_budget
            for client_id, ledger in privacy_ledgers.items()
        }
        round_rows = list(checkpoint.get("round_rows", []))
        decision_rows = list(checkpoint.get("decision_rows", []))
        flow_event_rows = list(checkpoint.get("flow_event_rows", []))
        link_state_rows = list(checkpoint.get("link_state_rows", []))
        best_accuracy = float(checkpoint.get("best_accuracy", 0.0))
        logical_time = float(checkpoint.get("logical_time", 0.0))
        previous_choices = dict(checkpoint.get("previous_choices", {}))
        real_he_rounds = int(checkpoint.get("real_he_rounds", 0))
        real_he_aggregated_clients = int(checkpoint.get("real_he_aggregated_clients", 0))
        global_pareto_selection_rounds = int(checkpoint.get("global_pareto_selection_rounds", 0))
        if "client_model_states" not in checkpoint:
            raise RuntimeError(
                "The resume checkpoint predates per-client returned-model state tracking. "
                "Start a new run so edge-only and cloud-return modes do not mix execution semantics."
            )
        saved_client_states = checkpoint["client_model_states"]
        client_model_states = {
            int(client_id): {
                "end": _state_dict_to_device(parts["end"], torch.device("cpu")),
                "edge": _state_dict_to_device(parts["edge"], torch.device("cpu")),
            }
            for client_id, parts in saved_client_states.items()
        }
        start_round = int(checkpoint.get("next_round", len(round_rows)))
        if round_rows:
            completed_wall_time_offset_sec = float(
                round_rows[-1].get(
                    "cumulative_wall_time_sec",
                    sum(float(row.get("round_wall_time_sec", 0.0)) for row in round_rows),
                )
            )
            resumed_round = round_rows[-1]
            last_completed_status.update(
                {
                    "test_accuracy": resumed_round.get("test_accuracy"),
                    "best_test_accuracy": resumed_round.get("best_accuracy"),
                    "train_accuracy": resumed_round.get("train_accuracy"),
                    "test_loss": resumed_round.get("test_loss"),
                    "train_loss": resumed_round.get("train_loss"),
                    "logical_time": resumed_round.get("logical_time"),
                    "global_update_norm": resumed_round.get("global_update_norm"),
                    "effective_clients": resumed_round.get("num_effective_clients"),
                    "effective_edges": resumed_round.get("num_effective_edges"),
                    "actual_cloud_fusion_ratio": resumed_round.get("actual_cloud_fusion_ratio"),
                    "recent_rounds": round_rows[-20:],
                }
            )
        if "rng_state" in checkpoint:
            rng.setstate(checkpoint["rng_state"])
        if "np_rng_state" in checkpoint:
            np_rng.bit_generator.state = checkpoint["np_rng_state"]
        if start_round >= selection.rounds:
            print(
                f"  [{policy}] resume source already has {start_round} rounds; target={selection.rounds}",
                flush=True,
            )

    emit_stage_status(
        f"Prepared {len(client_model_states)} persistent client model states",
        stage="client_state_init",
    )

    stop_round = selection.rounds
    if max_new_rounds is not None:
        stop_round = min(selection.rounds, start_round + max_new_rounds)

    for round_idx in range(start_round, stop_round):
        round_wall_started_at = time.perf_counter()
        round_he_metrics = HEOperationMetrics(backend=he_status.backend)
        selection_wall_started_at = time.perf_counter()
        emit_stage_status(
            f"Round {round_idx + 1}/{selection.rounds}: enumerating client candidates",
            stage="candidate_enumeration",
            round_value=round_idx,
        )
        prior_choices = dict(previous_choices)
        selected = []
        round_comm = 0.0
        round_risk = 0.0
        round_decision_rows_by_client: dict[int, dict[str, Any]] = {}
        infeasible = 0
        forced_feasibility_repair = False
        selection_diagnostics: dict[str, Any] = {}

        for client in clients:
            rem = remaining_epsilon[client.client_id]
            candidates = enumerate_candidates(
                config=effective_selection,
                client_id=client.client_id,
                edge_factor=edge_by_id[client.edge_id].compute_factor,
                compute_factor=client.compute_factor,
                memory_capacity_factor=client.memory_capacity_factor,
                samples=client.samples,
                remaining_epsilon=rem,
                round_idx=round_idx,
                rng=rng,
                policy=policy,
                privacy_ledger=privacy_ledgers[client.client_id],
            )
            should_update = round_idx == 0 or round_idx % selection_period == 0
            candidate = None
            if not should_update:
                candidate = _reuse_previous_choice(
                    candidates,
                    prior_choices.get(client.client_id),
                    remaining_epsilon=rem,
                )
            if candidate is None:
                forced_feasibility_repair = forced_feasibility_repair or not should_update
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
            if train_config.require_real_he and candidate_has_he(candidate) and not real_he_available:
                raise RuntimeError(
                    f"Policy {policy} selected HE mechanism {candidate.mechanisms}, "
                    f"but real HE backend is unavailable: {he_status.detail}"
                )
            selected.append((client.client_id, candidate, candidates, rem))

        should_update_policy = (
            round_idx == 0
            or round_idx % selection_period == 0
            or forced_feasibility_repair
        )
        if (
            policy in {"ours", "ours_fixed_liieiiic"}
            and should_update_policy
            and bool(privacy_parameters["feature_dp_enabled"])
        ):
            emit_stage_status(
                f"Round {round_idx + 1}/{selection.rounds}: assigning public feature clip estimate",
                stage="feature_clip_profiles",
                round_value=round_idx,
            )
            # Selection must not inspect private embeddings. The configured
            # estimate is shared system metadata and is fixed before training.
            feature_clip_profiles = {
                client.client_id: float(selection.omega_feature_clip_excess_sq)
                for client in clients
            }
            selected = _attach_feature_clip_profiles(
                selected,
                feature_clip_profiles,
            )
        profile_evaluation: ProfileEvaluation | None = None
        if policy == "accuracy_oracle" and should_update_policy:
            emit_stage_status(
                f"Round {round_idx + 1}/{selection.rounds}: evaluating accuracy oracle candidates",
                stage="accuracy_oracle_selection",
                round_value=round_idx,
            )
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
                selection_config=selection,
                model_name=model_name,
                input_shape=input_shape,
                num_classes=num_classes,
                device=device,
                np_rng=np_rng,
                round_idx=round_idx,
                top_k=3,
            )
        elif policy in {"best_accuracy", "performance_only"} and should_update_policy:
            emit_stage_status(
                f"Round {round_idx + 1}/{selection.rounds}: coordinating global objective",
                stage="global_objective_selection",
                round_value=round_idx,
            )
            selected = _coordinate_global_objective_round(
                selected,
                client_by_id,
                policy=policy,
            )
        elif policy in {
            "ours",
            "ours_no_omega",
            "ours_fixed_liieiiic",
            "no_protection",
            "fixed_fedavg",
            "fixed_splitfed",
            "fixed_splitfed_no_protection",
            "fixed_splitfed_label_dp",
            "fixed_splitfed_trusted_edge",
            "fixed_splitfed_dp",
            "fixed_hfl",
            "nsga2",
        } and should_update_policy:
            objective = "latency" if policy == "ours_no_omega" else "pareto"
            search_method = "nsga2" if policy == "nsga2" else "bounded"
            emit_stage_status(
                (
                    f"Round {round_idx + 1}/{selection.rounds}: solving Pareto mode selection"
                    if objective == "pareto"
                    else f"Round {round_idx + 1}/{selection.rounds}: solving latency only mode selection"
                ),
                stage=("pareto_selection" if objective == "pareto" else "latency_selection"),
                round_value=round_idx,
            )
            client_samples = {client.client_id: float(client.samples) for client in clients}
            selected, profile_evaluation = choose_global_pareto_profile(
                config=effective_selection,
                selected=selected,
                client_samples=client_samples,
                client_edges=client_edges,
                previous_choices=prior_choices,
                objective=objective,
                search_method=search_method,
                diagnostics=selection_diagnostics,
            )
            if policy == "ours":
                global_pareto_selection_rounds += 1

        if policy in {
            "ours",
            "ours_no_omega",
            "ours_fixed_liieiiic",
            "no_protection",
            "fixed_fedavg",
            "fixed_splitfed",
            "fixed_splitfed_no_protection",
            "fixed_splitfed_label_dp",
            "fixed_splitfed_trusted_edge",
            "fixed_splitfed_dp",
            "fixed_hfl",
            "nsga2",
        } and profile_evaluation is None:
            profile_evaluation = evaluate_global_profile(
                config=effective_selection,
                selected=selected,
                client_samples={client.client_id: float(client.samples) for client in clients},
                client_edges={client.client_id: int(client.edge_id) for client in clients},
                previous_choices=prior_choices,
            )

        selection_wall_time_sec = time.perf_counter() - selection_wall_started_at
        previous_choices = {client_id: candidate for client_id, candidate, _candidates, _rem in selected}

        for client_id, candidate, _candidates, rem in selected:
            if not candidate_meets_update_goal(effective_selection, candidate):
                raise ValueError("Selected or reused candidate violates the result DP coverage gate")
            round_comm += candidate.communication_volume
            round_risk = max(round_risk, candidate.risk)
            infeasible += int(not candidate.feasible)
            ledger = privacy_ledgers[client_id]
            projection = ledger.project(
                candidate.feature_dp_events,
                candidate.update_dp_events,
            )
            if (effective_selection.update_protection_goal == "released_model_dp"
                    and not ledger.can_apply(projection)):
                raise ValueError("Selected DP release exceeds the recorded event budget")
            rem = ledger.remaining_budget
            row = {
                    "policy": policy,
                    "round": round_idx,
                    "client_id": client_id,
                    "update_protection_goal": effective_selection.update_protection_goal,
                    "edge_id": client_by_id[client_id].edge_id,
                    "mode": candidate.mode,
                    "mechanisms": candidate_mechanism_label(candidate),
                    "update_mechanism": "/".join(
                        sorted(set(candidate_mechanisms_for_object(candidate, "upd")))
                    ) or "none",
                    "time": candidate.time,
                    "pre_aggregation_time": candidate.pre_aggregation_time,
                    "risk": candidate.risk,
                    "epsilon_used": candidate.epsilon_used,
                    "remaining_epsilon": rem,
                    "feature_dp_events": candidate.feature_dp_events,
                    "update_dp_events": candidate.update_dp_events,
                    "feature_epsilon": ledger.feature.current_epsilon(),
                    "update_epsilon": ledger.update.current_epsilon(),
                    "projected_feature_epsilon": projection.feature_epsilon_after,
                    "projected_update_epsilon": projection.update_epsilon_after,
                    "communication_volume": candidate.communication_volume,
                    "feasible": candidate.feasible,
                    "feasible_resource": candidate.feasible_resource,
                    "feasible_memory": candidate.feasible_memory,
                    "memory_requirement": candidate.memory_requirement,
                    "memory_capacity": candidate.memory_capacity,
                    "feasible_privacy": candidate.feasible_privacy,
                    "feasible_risk": candidate.feasible_risk,
                    "feasible_time": candidate.feasible_time,
                    "omega_feature_clip_excess_sq": getattr(
                        candidate,
                        "omega_feature_clip_excess_sq",
                        None,
                    ),
                }
            decision_rows.append(row)
            round_decision_rows_by_client[client_id] = row
            for metric in candidate.link_metrics:
                link_state_rows.append(
                    {
                        "policy": policy,
                        "round": round_idx,
                        "client_id": client_id,
                        "edge_id": client_by_id[client_id].edge_id,
                        "mode": candidate.mode,
                        **metric,
                    }
                )

        flow_inputs = []
        skipped_clients = 0
        dispatch_sequence = 0
        train_tasks: list[tuple[int, Candidate, np.ndarray, int]] = []

        for client_id, candidate, _candidates, _rem in selected:
            if candidate.mode == "SKIP":
                skipped_clients += 1
                continue
            idx = train_client_indices[client_id]
            if len(idx) == 0:
                continue
            train_tasks.append((client_id, candidate, idx, dispatch_sequence))
            dispatch_sequence += 1

        training_wall_started_at = time.perf_counter()
        emit_stage_status(
            f"Round {round_idx + 1}/{selection.rounds}: training selected clients",
            stage="client_training",
            round_value=round_idx,
        )
        progress_stride = 1

        def training_progress_callback(done: int, total: int) -> None:
            if done < total and done % progress_stride != 0:
                return
            emit_stage_status(
                f"Round {round_idx + 1}/{selection.rounds}: training selected clients ({done}/{total})",
                stage="client_training",
                round_value=round_idx,
                progress=(round_idx + 0.50 + 0.15 * done / max(total, 1)) / max(selection.rounds, 1),
            )

        def training_event_callback(phase: str, done: int, total: int) -> None:
            emit_stage_status(
                f"Round {round_idx + 1}/{selection.rounds}: {phase} ({done}/{total})",
                stage="client_training",
                round_value=round_idx,
                progress=(round_idx + 0.50 + 0.15 * done / max(total, 1)) / max(selection.rounds, 1),
            )

        worker_results = _run_client_training_tasks(
            tasks=train_tasks,
            train_config=train_config,
            selection=selection,
            global_end=global_end,
            global_edge=global_edge,
            client_model_states=client_model_states,
            x_train=x_train,
            y_train=y_train,
            device=device,
            model_name=model_name,
            input_shape=input_shape,
            num_classes=num_classes,
            np_rng=np_rng,
            round_idx=round_idx,
            progress_callback=training_progress_callback,
            progress_event_callback=training_event_callback,
        )
        training_wall_time_sec = time.perf_counter() - training_wall_started_at

        for client_id, candidate, idx, sequence in train_tasks:
            result = worker_results.get(client_id)
            if result is None:
                skipped_clients += 1
                continue
            state_diff = result["state_diff"]
            measured_local = float(result["measured_local"])
            if not result["finite"]:
                skipped_clients += 1
                continue
            dispatch_start = 0.0
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
                    candidate_time=candidate_arrival_with_switch(
                        effective_selection,
                        client_id,
                        candidate,
                        prior_choices.get(client_id),
                    ),
                    estimated_local_time=est_local,
                    measured_local_time=measured_local,
                    communication_volume=candidate.communication_volume,
                    state_diff=state_diff,
                    sample_count=len(idx),
                    edge_loops=spec_mode.E_edge_loops,
                    edge_to_cloud_time=candidate.edge_to_cloud_time,
                    return_path_time=candidate.return_path_time,
                    edge_aggregation_payload=candidate.edge_aggregation_payload,
                    cloud_aggregation_payload=candidate.cloud_aggregation_payload,
                    aggregation_group=(
                        _candidate_cloud_update_mechanism(candidate)
                        if candidate.mode in EDGE_CLOUD_MODES
                        else ""
                    ),
                    dispatch_start_time=dispatch_start,
                    dispatch_sequence=sequence,
                )
            )

        flow_wall_started_at = time.perf_counter()
        emit_stage_status(
            f"Round {round_idx + 1}/{selection.rounds}: simulating link flow",
            stage="flow_execution",
            round_value=round_idx,
        )
        flow_result = execute_mixed_round_flow(
            round_idx=round_idx,
            clients=flow_inputs,
            aggregation_fraction=selection.aggregation_fraction,
            edge_aggregation_beta=selection.edge_aggregation_beta,
            edge_aggregation_fixed=selection.edge_aggregation_fixed,
            cloud_aggregation_beta=selection.cloud_aggregation_beta,
            cloud_aggregation_fixed=selection.cloud_aggregation_fixed,
        )
        flow_wall_time_sec = time.perf_counter() - flow_wall_started_at
        for event in flow_result.flow_events:
            flow_event_rows.append({"policy": policy, **event})

        aggregation_wall_started_at = time.perf_counter()
        emit_stage_status(
            f"Round {round_idx + 1}/{selection.rounds}: aggregating updates",
            stage="aggregation",
            round_value=round_idx,
        )
        selected_by_id = {client_id: candidate for client_id, candidate, _candidates, _rem in selected}
        admitted_client_ids = set(flow_result.selected_client_ids)
        for client_id, candidate in selected_by_id.items():
            ledger = privacy_ledgers[client_id]
            row = round_decision_rows_by_client[client_id]
            if client_id in admitted_client_ids:
                projection = ledger.add(
                    candidate.feature_dp_events,
                    candidate.update_dp_events,
                )
                row["feature_dp_events"] = projection.feature_events
                row["update_dp_events"] = projection.update_events
            else:
                row["feature_dp_events"] = 0
                row["update_dp_events"] = 0
            remaining_epsilon[client_id] = ledger.remaining_budget
            row["remaining_epsilon"] = ledger.remaining_budget
            row["feature_epsilon"] = ledger.feature.current_epsilon()
            row["update_epsilon"] = ledger.update.current_epsilon()
        round_real_he_used = False
        round_real_he_clients = 0
        multi_edge_loop_client_cycles = 0
        initial_admitted_updates = {
            client_id: (state_diff, sample_count)
            for client_id, state_diff, sample_count in zip(
                flow_result.selected_client_ids,
                flow_result.state_diffs,
                flow_result.sample_counts,
            )
        }
        overridden_admitted_diffs: dict[int, Any] = {}
        multi_edge_groups: dict[tuple[int, str], list[int]] = {}
        for client_id in flow_result.selected_client_ids:
            candidate = selected_by_id.get(client_id)
            if candidate is None or candidate.mode not in EDGE_CLOUD_MODES:
                continue
            if MODE_SPECS[candidate.mode].E_edge_loops <= 1:
                continue
            key = (int(client_by_id[client_id].edge_id), candidate.mode)
            multi_edge_groups.setdefault(key, []).append(client_id)

        for (_edge_id, mode), client_ids in multi_edge_groups.items():
            loop_count = max(1, int(MODE_SPECS[mode].E_edge_loops))
            returned_state: dict[str, dict[str, torch.Tensor]] | None = None
            for edge_loop_idx in range(loop_count):
                if edge_loop_idx == 0:
                    cycle_updates = [
                        (
                            client_id,
                            initial_admitted_updates[client_id][0],
                            initial_admitted_updates[client_id][1],
                            selected_by_id.get(client_id),
                        )
                        for client_id in client_ids
                    ]
                else:
                    cycle_tasks = [
                        (
                            client_id,
                            selected_by_id[client_id],
                            train_client_indices[client_id],
                            sequence,
                        )
                        for sequence, client_id in enumerate(client_ids)
                    ]
                    cycle_results = _run_client_training_tasks(
                        tasks=cycle_tasks,
                        train_config=train_config,
                        selection=selection,
                        global_end=global_end,
                        global_edge=global_edge,
                        client_model_states=client_model_states,
                        x_train=x_train,
                        y_train=y_train,
                        device=device,
                        model_name=model_name,
                        input_shape=input_shape,
                        num_classes=num_classes,
                        np_rng=np_rng,
                        round_idx=round_idx,
                        training_stage=edge_loop_idx,
                    )
                    cycle_updates = [
                        (
                            client_id,
                            cycle_results[client_id]["state_diff"],
                            len(train_client_indices[client_id]),
                            selected_by_id.get(client_id),
                        )
                        for client_id in client_ids
                        if client_id in cycle_results and cycle_results[client_id]["finite"]
                    ]
                if not cycle_updates:
                    continue
                multi_edge_loop_client_cycles += len(cycle_updates)
                returned_state, used_real_he = _aggregate_returned_client_models(
                    updates=cycle_updates,
                    client_model_states=client_model_states,
                    global_end=global_end,
                    global_edge=global_edge,
                    device=device,
                    model_name=model_name,
                    input_shape=input_shape,
                    num_classes=num_classes,
                    he_aggregation_size=train_config.he_aggregation_size,
                    he_backend=(
                        he_status.backend
                        if execute_real_he and mode == "LIIEIIIC"
                        else "none"
                    ),
                    he_metrics=round_he_metrics,
                )
                shared_returned_state = _state_dict_to_device_nested(
                    returned_state,
                    torch.device("cpu"),
                )
                for client_id in client_ids:
                    client_model_states[client_id] = shared_returned_state
                if used_real_he:
                    round_real_he_used = True
                    round_real_he_clients += sum(
                        candidate is not None
                        and mechanism_uses_he(_candidate_edge_update_mechanism(candidate))
                        for _client_id, _state_diff, _sample_count, candidate in cycle_updates
                    )

            if returned_state is not None:
                zero_diff = _zero_state_difference(global_end, global_edge, device)
                for client_id in client_ids:
                    overridden_admitted_diffs[client_id] = zero_diff

        admitted_updates = [
            (
                client_id,
                overridden_admitted_diffs.get(client_id, state_diff),
                sample_count,
                selected_by_id.get(client_id),
            )
            for client_id, state_diff, sample_count in zip(
                flow_result.selected_client_ids,
                flow_result.state_diffs,
                flow_result.sample_counts,
            )
        ]
        admitted_update_norms = [
            _state_difference_l2_norm(state_diff)
            for _client_id, state_diff, _sample_count, _candidate in admitted_updates
        ]
        admitted_candidates = [
            candidate
            for client_id, _state_diff, _sample_count, candidate in admitted_updates
            if candidate is not None and client_id in flow_result.selected_client_ids
        ]
        feature_objects = {"emb", "grad", "weakemb", "strongemb", "pseudo_label"}
        num_feature_dp_clients = sum(
            any(
                mechanism == "dp"
                for obj in feature_objects
                for mechanism in candidate_mechanisms_for_object(candidate, obj)
            )
            for candidate in admitted_candidates
        )
        num_update_dp_clients = sum(
            any(
                mechanism_uses_dp(mechanism)
                for mechanism in candidate_mechanisms_for_object(candidate, "upd")
            )
            for candidate in admitted_candidates
        )
        num_he_clients = sum(
            candidate_has_he(candidate)
            for candidate in admitted_candidates
        )
        update_protection_mechanisms = [
            _candidate_cloud_update_mechanism(candidate)
            for candidate in admitted_candidates
            if _mode_reaches_cloud(candidate.mode)
        ]
        num_update_dp_only_clients = sum(
            mechanism_uses_dp(mechanism) and not mechanism_uses_he(mechanism)
            for mechanism in update_protection_mechanisms
        )
        num_update_he_only_clients = sum(
            mechanism_uses_he(mechanism) and not mechanism_uses_dp(mechanism)
            for mechanism in update_protection_mechanisms
        )
        num_update_dp_he_clients = sum(
            mechanism_uses_dp(mechanism) and mechanism_uses_he(mechanism)
            for mechanism in update_protection_mechanisms
        )
        global_updates = [
            (client_id, state_diff, sample_count, candidate)
            for client_id, state_diff, sample_count, candidate in admitted_updates
            if _mode_reaches_cloud(candidate.mode if candidate else "")
        ]
        cross_domain_update_clients = len(global_updates)
        update_dp_coverage = (
            num_update_dp_clients / cross_domain_update_clients
            if cross_domain_update_clients
            else 0.0
        )
        update_he_coverage = (
            num_he_clients / cross_domain_update_clients
            if cross_domain_update_clients
            else 0.0
        )
        uniform_update_dp = (
            cross_domain_update_clients > 0
            and num_update_dp_clients == cross_domain_update_clients
        )
        uniform_local_update_dp = (
            cross_domain_update_clients > 0
            and num_update_dp_only_clients == cross_domain_update_clients
        )
        uniform_secure_aggregate_dp = (
            cross_domain_update_clients > 0
            and num_update_dp_he_clients == cross_domain_update_clients
        )
        all_cross_domain_updates_protected = (
            cross_domain_update_clients > 0
            and (
                num_update_dp_only_clients
                + num_update_he_only_clients
                + num_update_dp_he_clients
            )
            == cross_domain_update_clients
        )
        total_profile_samples = max(1, sum(int(client.samples) for client in clients))
        actual_cloud_samples = sum(item[2] for item in global_updates)
        actual_admitted_samples = max(1, sum(item[2] for item in admitted_updates))
        actual_cloud_fusion_ratio = actual_cloud_samples / total_profile_samples
        actual_cloud_share_of_admitted = actual_cloud_samples / actual_admitted_samples

        cloud_updates: list[tuple[Any, int, Candidate | None, list[int]]] = []
        cloud_signal_updates: list[Any] = []
        secure_aggregate_dp_client_fractions: list[float] = []
        update_dp_clip_scales: list[float] = []
        update_dp_preclip_norms: list[float] = []
        update_dp_clipped_clients = 0
        local_dp_packet_client_fractions: list[float] = []
        local_dp_packet_sensitivities: list[float] = []
        local_dp_pending_components: list[tuple[int, float, int]] = []
        edge_cloud_groups: dict[tuple[int, str, str], list[tuple[int, Any, int, Candidate | None]]] = {}
        for client_id, state_diff, sample_count, candidate in global_updates:
            if candidate is not None and candidate.mode in EDGE_CLOUD_MODES:
                key = (
                    int(client_by_id[client_id].edge_id),
                    candidate.mode,
                    _candidate_cloud_update_mechanism(candidate),
                )
                edge_cloud_groups.setdefault(key, []).append(
                    (client_id, state_diff, sample_count, candidate)
                )
            else:
                relative_update = _state_difference_from_client_update(
                    client_id=client_id,
                    state_diff=state_diff,
                    client_model_states=client_model_states,
                    global_end=global_end,
                    global_edge=global_edge,
                    device=torch.device("cpu"),
                )
                local_packet_dp = _candidate_uses_local_packet_update_dp(
                    candidate,
                    effective_selection.trusted_edge_split_execution,
                )
                secure_aggregate_dp = _candidate_uses_secure_aggregate_update_dp(
                    candidate,
                    effective_selection.trusted_edge_split_execution,
                )
                if train_config.dp_release_calibration == "tex_packet":
                    local_packet_dp = local_packet_dp or secure_aggregate_dp
                    secure_aggregate_dp = False
                if local_packet_dp or secure_aggregate_dp:
                    relative_update, _original_norm, clip_scale = clip_state_difference(
                        relative_update,
                        train_config.dp_clip_norm,
                        torch.device("cpu"),
                    )
                    update_dp_clip_scales.append(clip_scale)
                    update_dp_preclip_norms.append(float(_original_norm))
                    update_dp_clipped_clients += int(clip_scale < 1.0 - 1e-12)
                cloud_signal_updates.append(relative_update)
                cloud_index = len(cloud_updates)
                if local_packet_dp:
                    sensitivity, noise_std = _dp_update_release_parameters(
                        1.0,
                        clip_norm=train_config.dp_clip_norm,
                        noise_multiplier=float(
                            privacy_parameters["update_noise_multiplier"]
                        ),
                    )
                    local_dp_packet_client_fractions.append(1.0)
                    local_dp_packet_sensitivities.append(sensitivity)
                    local_dp_pending_components.append(
                        (
                            cloud_index,
                            noise_std,
                            _dp_noise_seed(
                                selection.seed,
                                round_idx,
                                client_id,
                                10_000,
                            ),
                        )
                    )
                cloud_updates.append(
                    (
                        relative_update,
                        sample_count,
                        candidate,
                        [client_id],
                    )
                )
                secure_aggregate_dp_client_fractions.append(
                    1.0 if secure_aggregate_dp else 0.0
                )

        for edge_id, updates in edge_cloud_groups.items():
            representative = updates[0][3]
            local_packet_dp = _candidate_uses_local_packet_update_dp(
                representative,
                effective_selection.trusted_edge_split_execution,
            )
            secure_aggregate_dp = _candidate_uses_secure_aggregate_update_dp(
                representative,
                effective_selection.trusted_edge_split_execution,
            )
            tex_packet = train_config.dp_release_calibration == "tex_packet"
            if tex_packet:
                local_packet_dp = local_packet_dp or secure_aggregate_dp
                secure_aggregate_dp = False
            if local_packet_dp or secure_aggregate_dp:
                clipped_updates = []
                clipped_counts = []
                for client_id, state_diff, sample_count, _candidate in updates:
                    relative_update = _state_difference_from_client_update(
                        client_id=client_id,
                        state_diff=state_diff,
                        client_model_states=client_model_states,
                        global_end=global_end,
                        global_edge=global_edge,
                        device=torch.device("cpu"),
                    )
                    if tex_packet:
                        clipped = relative_update
                    else:
                        clipped, _original_norm, clip_scale = clip_state_difference(
                            relative_update, train_config.dp_clip_norm, torch.device("cpu"),
                        )
                        update_dp_clip_scales.append(clip_scale)
                        update_dp_preclip_norms.append(float(_original_norm))
                        update_dp_clipped_clients += int(clip_scale < 1.0 - 1e-12)
                    clipped_updates.append(clipped)
                    clipped_counts.append(sample_count)
                edge_update = _weighted_average_state_differences(
                    clipped_updates,
                    clipped_counts,
                )
                group_total = max(float(sum(clipped_counts)), 1e-12)
                max_client_fraction = max(
                    (float(count) / group_total for count in clipped_counts),
                    default=0.0,
                )
                if tex_packet:
                    edge_update, _original_norm, clip_scale = clip_state_difference(
                        edge_update, train_config.dp_clip_norm, torch.device("cpu"),
                    )
                    update_dp_clip_scales.append(clip_scale)
                    update_dp_preclip_norms.append(float(_original_norm))
                    update_dp_clipped_clients += int(clip_scale < 1.0 - 1e-12)
                    max_client_fraction = 1.0
                cloud_signal_updates.append(edge_update)
                cloud_index = len(cloud_updates)
                if local_packet_dp:
                    sensitivity, noise_std = _dp_update_release_parameters(
                        max_client_fraction,
                        clip_norm=train_config.dp_clip_norm,
                        noise_multiplier=float(
                            privacy_parameters["update_noise_multiplier"]
                        ),
                    )
                    local_dp_packet_client_fractions.append(max_client_fraction)
                    local_dp_packet_sensitivities.append(sensitivity)
                    local_dp_pending_components.append(
                        (
                            cloud_index,
                            noise_std,
                            _dp_noise_seed(
                                selection.seed,
                                round_idx,
                                edge_id,
                                10_000,
                            ),
                        )
                    )
            else:
                edge_state, _edge_used_he = _aggregate_returned_client_models(
                    updates=updates,
                    client_model_states=client_model_states,
                    global_end=global_end,
                    global_edge=global_edge,
                    device=device,
                    model_name=model_name,
                    input_shape=input_shape,
                    num_classes=num_classes,
                    he_aggregation_size=train_config.he_aggregation_size,
                    he_backend="none",
                    he_metrics=round_he_metrics,
                )
                edge_update = _state_difference_from_model(
                    edge_state,
                    global_end,
                    global_edge,
                    torch.device("cpu"),
                )
                max_client_fraction = 0.0
                cloud_signal_updates.append(edge_update)
            cloud_updates.append(
                (
                    edge_update,
                    sum(item[2] for item in updates),
                    representative,
                    [item[0] for item in updates],
                )
            )
            secure_aggregate_dp_client_fractions.append(
                max_client_fraction if secure_aggregate_dp else 0.0
            )

        global_aggregation_weights = _edge_normalized_cloud_weights(
            cloud_updates,
            client_edges=client_edges,
            edge_total_samples=edge_total_samples,
        )
        pre_dp_global_update_norm = _weighted_state_difference_norm(
            cloud_signal_updates,
            global_aggregation_weights,
        )
        (
            normalized_global_weights,
            aggregate_dp_max_client_weight,
            aggregate_dp_sensitivity,
            aggregate_dp_noise_std,
            aggregate_dp_share_stds,
        ) = _distributed_aggregate_dp_parameters(
            global_aggregation_weights,
            secure_aggregate_dp_client_fractions,
            clip_norm=train_config.dp_clip_norm,
            noise_multiplier=float(privacy_parameters["update_noise_multiplier"]),
        )
        cloud_signal_updates.clear()
        local_dp_noise_accumulator: dict[str, dict[str, torch.Tensor]] = {}
        aggregate_dp_noise_accumulator: dict[str, dict[str, torch.Tensor]] = {}
        update_dp_component_noise_norms: list[float] = []
        for cloud_index, noise_std, noise_seed in local_dp_pending_components:
            state_diff, sample_count, candidate, client_ids = cloud_updates[cloud_index]
            noise = gaussian_state_difference(
                state_diff,
                noise_std,
                np.random.default_rng(noise_seed),
                torch.device("cpu"),
            )
            cloud_updates[cloud_index] = (
                _add_state_differences(state_diff, noise),
                sample_count,
                candidate,
                client_ids,
            )
            _accumulate_scaled_state_difference(
                local_dp_noise_accumulator,
                noise,
                normalized_global_weights[cloud_index],
            )
            update_dp_component_noise_norms.append(
                _state_difference_l2_norm(noise)
            )
            del noise

        aggregate_dp_noise_share_count = 0
        for cloud_index, share_std in enumerate(aggregate_dp_share_stds):
            if share_std <= 0.0:
                continue
            state_diff, sample_count, candidate, client_ids = cloud_updates[cloud_index]
            noise = gaussian_state_difference(
                state_diff,
                share_std,
                np.random.default_rng(
                    _dp_noise_seed(
                        selection.seed,
                        round_idx,
                        ("aggregate_dp_share", cloud_index, client_ids),
                        20_000,
                    )
                ),
                torch.device("cpu"),
            )
            cloud_updates[cloud_index] = (
                _add_state_differences(state_diff, noise),
                sample_count,
                candidate,
                client_ids,
            )
            _accumulate_scaled_state_difference(
                aggregate_dp_noise_accumulator,
                noise,
                normalized_global_weights[cloud_index],
            )
            update_dp_component_noise_norms.append(
                _state_difference_l2_norm(noise)
            )
            aggregate_dp_noise_share_count += 1
            del noise

        cloud_input_update_norms = [
            _state_difference_l2_norm(state_diff)
            for state_diff, _sample_count, _candidate, _client_ids in cloud_updates
        ]
        global_update_norm = _weighted_state_difference_norm(
            [item[0] for item in cloud_updates],
            global_aggregation_weights,
        )
        update_dp_release_count = len(local_dp_pending_components) + int(
            aggregate_dp_noise_share_count > 0
        )
        update_dp_release = update_dp_release_count > 0
        update_dp_max_client_fraction = max(
            local_dp_packet_client_fractions
            + secure_aggregate_dp_client_fractions,
            default=0.0,
        )
        update_dp_sensitivity = max(
            local_dp_packet_sensitivities + [aggregate_dp_sensitivity],
            default=0.0,
        )
        update_dp_noise_std = float(
            np.sqrt(
                sum(
                    (normalized_global_weights[index] * noise_std) ** 2
                    for index, noise_std, _seed in local_dp_pending_components
                )
                + aggregate_dp_noise_std ** 2
            )
        )
        local_dp_noise_norm = _state_difference_l2_norm(
            local_dp_noise_accumulator
        )
        aggregate_dp_noise_norm = _state_difference_l2_norm(
            aggregate_dp_noise_accumulator
        )
        if local_dp_noise_accumulator and aggregate_dp_noise_accumulator:
            update_dp_noise_norm = _state_difference_l2_norm(
                _add_state_differences(
                    local_dp_noise_accumulator,
                    aggregate_dp_noise_accumulator,
                )
            )
        else:
            update_dp_noise_norm = max(
                local_dp_noise_norm,
                aggregate_dp_noise_norm,
            )
        cloud_edge_ratios = _cloud_edge_sample_ratios(
            cloud_updates,
            client_edges=client_edges,
            edge_total_samples=edge_total_samples,
        )
        edge_only_groups: dict[tuple[int, str], list[tuple[int, Any, int, Candidate | None]]] = {}
        for client_id, state_diff, sample_count, candidate in admitted_updates:
            if candidate is None or candidate.mode not in EDGE_ONLY_MODES:
                continue
            key = (int(client_by_id[client_id].edge_id), candidate.mode)
            edge_only_groups.setdefault(key, []).append(
                (client_id, state_diff, sample_count, candidate)
            )

        for updates in edge_only_groups.values():
            returned_state, used_real_he = _aggregate_returned_client_models(
                updates=updates,
                client_model_states=client_model_states,
                global_end=global_end,
                global_edge=global_edge,
                device=device,
                model_name=model_name,
                input_shape=input_shape,
                num_classes=num_classes,
                he_aggregation_size=train_config.he_aggregation_size,
                he_backend=he_status.backend if execute_real_he else "none",
                he_metrics=round_he_metrics,
            )
            shared_returned_state = _state_dict_to_device_nested(
                returned_state,
                torch.device("cpu"),
            )
            for client_id, _state_diff, _sample_count, _candidate in updates:
                client_model_states[client_id] = shared_returned_state
            if used_real_he:
                round_real_he_used = True
                round_real_he_clients += sum(
                    candidate is not None
                    and mechanism_uses_he(_candidate_edge_update_mechanism(candidate))
                    for _client_id, _state_diff, _sample_count, candidate in updates
                )

        global_state_diffs: list[dict[str, dict[str, torch.Tensor]]] = []
        global_candidates: list[Candidate | None] = []
        global_he_mask: list[bool] = []
        if cloud_updates:
            global_state_diffs = [item[0] for item in cloud_updates]
            global_sample_counts = global_aggregation_weights
            global_candidates = [item[2] for item in cloud_updates]
            global_he_mask = [
                mechanism_uses_he(_candidate_cloud_update_mechanism(candidate))
                for candidate in global_candidates
            ]
            use_real_he = (
                execute_real_he
                and any(global_he_mask)
            )
            if use_real_he:
                round_real_he_used = True
                round_real_he_clients += sum(
                    len(item[3])
                    for item, encrypted in zip(cloud_updates, global_he_mask)
                    if encrypted
                )
                if he_status.backend == "seal":
                    global_end, global_edge = fedavg_split_seal(
                        global_state_diffs,
                        global_sample_counts,
                        global_end,
                        global_edge,
                        device,
                        encrypted_mask=global_he_mask,
                        he_aggregation_size=train_config.he_aggregation_size,
                        he_workers=train_config.he_workers,
                        he_metrics=round_he_metrics,
                    )
                else:
                    global_end, global_edge = fedavg_split_tenseal(
                        global_state_diffs,
                        global_sample_counts,
                        global_end,
                        global_edge,
                        device,
                        encrypted_mask=global_he_mask,
                        he_aggregation_size=train_config.he_aggregation_size,
                        he_metrics=round_he_metrics,
                    )
            else:
                global_end, global_edge = fedavg_split(
                    global_state_diffs,
                    global_sample_counts,
                    global_end,
                    global_edge,
                    device,
                )
        for client_id, candidate, _candidates, _remaining in selected:
            if _mode_reaches_cloud(candidate.mode):
                client_model_states.pop(client_id, None)
        aggregation_wall_time_sec = time.perf_counter() - aggregation_wall_started_at

        packet_dp_indices = {index for index, _std, _seed in local_dp_pending_components}
        protection_releases = []
        for packet_index, (_diff, _samples, candidate, client_ids) in enumerate(cloud_updates):
            mechanism = _candidate_cloud_update_mechanism(candidate)
            noise_location = (
                "packet" if packet_index in packet_dp_indices else
                "aggregate_share" if aggregate_dp_share_stds[packet_index] > 0.0 else "none"
            )
            packet_he = (
                "real" if execute_real_he else "profiled"
            ) if mechanism_uses_he(mechanism) else "not_selected"
            audit = audit_update_release(
                mechanism=mechanism, noise_location=noise_location,
                he_execution=packet_he,
                dp_budget_ok=all(
                    privacy_ledgers[cid].update.current_epsilon()
                    <= privacy_ledgers[cid].update.budget + 1e-12
                    for cid in client_ids
                ),
                key_isolation_enforced=False,
                aggregate_only_decryption_enforced=False,
            )
            protection_releases.append({
                "round": round_idx, "packet_index": packet_index,
                "source": "edge" if candidate and candidate.mode in EDGE_CLOUD_MODES else "end",
                "destination": "cloud", "object": "model_update",
                "client_ids": ";".join(str(cid) for cid in client_ids),
                "source_domain": _cloud_update_edge(client_ids, client_edges),
                "mode": candidate.mode if candidate else "none",
                **audit,
            })

        if round_real_he_used:
            real_he_rounds += 1
            real_he_aggregated_clients += round_real_he_clients

        logical_time += flow_result.round_duration
        evaluation_wall_started_at = time.perf_counter()
        emit_stage_status(
            f"Round {round_idx + 1}/{selection.rounds}: evaluating global model",
            stage="evaluation",
            round_value=round_idx,
        )
        test_loss, test_accuracy = split_evaluate(
            global_end, global_edge, x_test, y_test, device, input_shape=input_shape
        )
        train_eval_idx = _train_eval_indices(len(y_train), selection.seed, round_idx)
        train_loss, train_accuracy = _split_evaluate_indexed(
            global_end,
            global_edge,
            x_train,
            y_train,
            train_eval_idx,
            device,
            input_shape=input_shape,
        )
        evaluation_wall_time_sec = time.perf_counter() - evaluation_wall_started_at
        round_wall_time_sec = time.perf_counter() - round_wall_started_at
        cumulative_wall_time_sec = (
            completed_wall_time_offset_sec
            + time.perf_counter()
            - policy_started_at
        )
        cumulative_selection_wall_time_sec = sum(
            float(row.get("selection_wall_time_sec", 0.0)) for row in round_rows
        ) + selection_wall_time_sec
        accounted_system_time_sec = logical_time + cumulative_selection_wall_time_sec
        accounted_phase_wall_time_sec = (
            selection_wall_time_sec
            + training_wall_time_sec
            + flow_wall_time_sec
            + aggregation_wall_time_sec
            + evaluation_wall_time_sec
        )
        best_accuracy = max(best_accuracy, test_accuracy)
        round_rows.append(
            {
                "policy": policy,
                "round": round_idx,
                "logical_time": logical_time,
                "accounted_system_time_sec": accounted_system_time_sec,
                "round_duration": flow_result.round_duration,
                "round_wall_time_sec": round_wall_time_sec,
                "cumulative_wall_time_sec": cumulative_wall_time_sec,
                "selection_wall_time_sec": selection_wall_time_sec,
                "training_wall_time_sec": training_wall_time_sec,
                "flow_wall_time_sec": flow_wall_time_sec,
                "aggregation_wall_time_sec": aggregation_wall_time_sec,
                "evaluation_wall_time_sec": evaluation_wall_time_sec,
                "accounted_phase_wall_time_sec": accounted_phase_wall_time_sec,
                "unattributed_wall_time_sec": max(
                    0.0,
                    round_wall_time_sec - accounted_phase_wall_time_sec,
                ),
                "non_training_wall_time_sec": max(
                    0.0,
                    round_wall_time_sec - training_wall_time_sec,
                ),
                "test_accuracy": test_accuracy,
                "accuracy_change_from_previous": (
                    test_accuracy - float(round_rows[-1]["test_accuracy"])
                    if round_rows else 0.0
                ),
                "test_loss": test_loss,
                "train_accuracy": train_accuracy,
                "train_loss": train_loss,
                "numerically_valid": int(
                    math.isfinite(train_loss) and math.isfinite(test_loss)
                ),
                "dp_parameter_scope": train_config.update_parameter_scope,
                "dp_release_calibration": train_config.dp_release_calibration,
                "dp_clipping_unit": (
                    "released_packet" if train_config.dp_release_calibration == "tex_packet"
                    else "client_contribution"
                ),
                "dp_parameter_count": int(effective_selection.omega_update_dimension),
                "best_accuracy": best_accuracy,
                "communication_volume": sum(item.communication_volume for item in flow_inputs if item.client_id in flow_result.selected_client_ids),
                "max_risk": round_risk,
                "min_remaining_epsilon": min(remaining_epsilon.values()),
                "max_feature_epsilon": max(
                    ledger.feature.current_epsilon() for ledger in privacy_ledgers.values()
                ),
                "max_update_epsilon": max(
                    ledger.update.current_epsilon() for ledger in privacy_ledgers.values()
                ),
                "privacy_guarantee": (
                    "uniform_secure_aggregate_dp"
                    if uniform_secure_aggregate_dp
                    else "uniform_local_packet_dp"
                    if uniform_local_update_dp
                    else "hybrid_update_protection"
                    if all_cross_domain_updates_protected
                    else "incomplete_cross_domain_protection"
                ),
                "infeasible_clients": infeasible,
                "skipped_clients": skipped_clients,
                "num_effective_clients": len(flow_result.selected_client_ids),
                "num_global_update_clients": len(global_updates),
                "num_edge_only_update_clients": len(flow_result.selected_client_ids) - len(global_updates),
                "admitted_update_norm_mean": _list_mean(admitted_update_norms),
                "admitted_update_norm_max": max(admitted_update_norms, default=0.0),
                "cloud_input_update_norm_mean": _list_mean(cloud_input_update_norms),
                "cloud_input_update_norm_max": max(cloud_input_update_norms, default=0.0),
                "pre_dp_global_update_norm": pre_dp_global_update_norm,
                "global_update_norm": global_update_norm,
                "post_to_pre_update_norm_ratio": global_update_norm
                / max(pre_dp_global_update_norm, 1e-12),
                "update_dp_release": int(update_dp_release),
                "update_dp_release_count": update_dp_release_count,
                "update_dp_protected_clients": num_update_dp_clients,
                "update_dp_clipped_clients": update_dp_clipped_clients,
                "update_dp_clip_scale_mean": _list_mean(update_dp_clip_scales),
                "update_dp_clip_scale_min": min(update_dp_clip_scales, default=1.0),
                "update_dp_preclip_norm_count": len(update_dp_preclip_norms),
                "update_dp_preclip_norm_mean": _list_mean(update_dp_preclip_norms),
                "update_dp_preclip_norm_max": max(update_dp_preclip_norms, default=0.0),
                "update_dp_preclip_norm_p50": float(np.quantile(update_dp_preclip_norms, 0.5)) if update_dp_preclip_norms else 0.0,
                "update_dp_preclip_norm_p90": float(np.quantile(update_dp_preclip_norms, 0.9)) if update_dp_preclip_norms else 0.0,
                "dp_diagnostic_scope": "private_experiment_logs_not_public_dp_outputs",
                "dp_accountant_scope": "recorded_dp_events_only",
                "end_to_end_dp_status": "not_established",
                "update_dp_max_client_fraction": update_dp_max_client_fraction,
                "update_dp_sensitivity": update_dp_sensitivity,
                "update_dp_noise_std": update_dp_noise_std,
                "update_dp_noise_norm": update_dp_noise_norm,
                "update_dp_noise_norm_mean": _list_mean(
                    update_dp_component_noise_norms
                ),
                "update_dp_noise_norm_max": max(
                    update_dp_component_noise_norms,
                    default=0.0,
                ),
                "local_packet_dp_release_count": len(local_dp_pending_components),
                "local_packet_dp_released_noise_norm": local_dp_noise_norm,
                "aggregate_dp_release": int(aggregate_dp_noise_share_count > 0),
                "aggregate_dp_protocol": (
                    "distributed_noise_before_ckks_aggregation"
                    if aggregate_dp_noise_share_count > 0
                    else "not_selected"
                ),
                "aggregate_dp_transport": (
                    "real_ckks"
                    if aggregate_dp_noise_share_count > 0 and execute_real_he
                    else "profiled_ckks"
                    if aggregate_dp_noise_share_count > 0
                    else "not_selected"
                ),
                "aggregate_dp_protected_packets": sum(
                    fraction > 0.0
                    for fraction in secure_aggregate_dp_client_fractions
                ),
                "aggregate_dp_noise_share_count": aggregate_dp_noise_share_count,
                "aggregate_dp_max_client_weight": aggregate_dp_max_client_weight,
                "aggregate_dp_sensitivity": aggregate_dp_sensitivity,
                "aggregate_dp_noise_std": aggregate_dp_noise_std,
                "aggregate_dp_noise_norm": aggregate_dp_noise_norm,
                "num_feature_dp_clients": num_feature_dp_clients,
                "num_update_dp_clients": num_update_dp_clients,
                "num_he_clients": num_he_clients,
                "num_update_dp_only_clients": num_update_dp_only_clients,
                "num_update_he_only_clients": num_update_he_only_clients,
                "num_update_dp_he_clients": num_update_dp_he_clients,
                "he_selected_but_profiled_clients": (
                    num_he_clients
                    if train_config.he_execution == "profiled"
                    else 0
                ),
                "he_execution_status": (
                    "not_selected"
                    if num_he_clients == 0
                    else "real"
                    if execute_real_he
                    else "profiled"
                ),
                "protected_object": "cross_domain_model_update",
                "privacy_mechanism_scope": "local_packet_or_secure_aggregate",
                "protection_release_audit": json.dumps(protection_releases, sort_keys=True),
                "update_packets_without_dp_calibration": sum(
                    item["released_model_dp_status"] != "dp_coverage_pending_analysis"
                    for item in protection_releases
                ),
                "update_packets_with_real_he_pending_isolation": sum(
                    item["he_crypto_observed"] and not item["key_isolation_enforced"]
                    for item in protection_releases
                ),
                "protection_rule_scope": "observed_operations_not_a_security_proof",
                "released_model_dp_coverage": (
                    "no_new_cloud_update" if not protection_releases else
                    "all_contributions_pending_analysis" if all(
                        item["released_model_dp_status"] == "dp_coverage_pending_analysis"
                        for item in protection_releases
                    ) else "some_contributions_without_dp_calibration"
                ),
                "selection_pool_candidates_before_stability": sum(
                    selection_diagnostics.get(
                        "candidate_pool_sizes_before_stability", {}
                    ).values()
                ),
                "selection_pool_candidates_after_stability": sum(
                    selection_diagnostics.get("candidate_pool_sizes", {}).values()
                ),
                "selection_stability_removed_candidates": max(
                    0,
                    sum(
                        selection_diagnostics.get(
                            "candidate_pool_sizes_before_stability", {}
                        ).values()
                    )
                    - sum(
                        selection_diagnostics.get("candidate_pool_sizes", {}).values()
                    ),
                ),
                "selection_update_mechanisms_before_stability": json.dumps(
                    selection_diagnostics.get(
                        "update_mechanism_counts_before_stability", {}
                    ),
                    sort_keys=True,
                ),
                "selection_update_mechanisms_after_stability": json.dumps(
                    selection_diagnostics.get(
                        "update_mechanism_counts_after_stability", {}
                    ),
                    sort_keys=True,
                ),
                "update_dp_coverage": update_dp_coverage,
                "update_he_coverage": update_he_coverage,
                "uniform_update_dp": int(uniform_update_dp),
                "all_cross_domain_updates_protected": int(
                    all_cross_domain_updates_protected
                ),
                "num_budget_exhausted_clients": sum(
                    remaining <= 1e-12 for remaining in remaining_epsilon.values()
                ),
                "omega_feature_clip_excess_sq_mean": _list_mean(
                    float(getattr(candidate, "omega_feature_clip_excess_sq", 0.0) or 0.0)
                    for _client_id, candidate, _candidates, _rem in selected
                ),
                "omega_feature_clip_excess_sq_max": max(
                    (
                        float(getattr(candidate, "omega_feature_clip_excess_sq", 0.0) or 0.0)
                        for _client_id, candidate, _candidates, _rem in selected
                    ),
                    default=0.0,
                ),
                "multi_edge_loop_client_cycles": multi_edge_loop_client_cycles,
                "num_effective_edges": flow_result.num_effective_edges,
                "waiting_time": flow_result.waiting_time,
                "edge_aggregation_time": flow_result.edge_aggregation_time,
                "cloud_aggregation_time": flow_result.cloud_aggregation_time,
                "return_time": flow_result.return_time,
                "real_he_used": round_real_he_used,
                **round_he_metrics.as_dict(),
                "selection_trigger": (
                    "scheduled"
                    if round_idx == 0 or round_idx % selection_period == 0
                    else "feasibility_repair"
                    if forced_feasibility_repair
                    else "reuse"
                ),
                "system_latency_objective": profile_evaluation.system_latency if profile_evaluation else "",
                "system_omega_objective": profile_evaluation.system_omega if profile_evaluation else "",
                "cloud_fusion_ratio": profile_evaluation.cloud_fusion_ratio if profile_evaluation else "",
                "selector_cloud_fusion_ratio": profile_evaluation.cloud_fusion_ratio if profile_evaluation else "",
                "actual_cloud_fusion_ratio": actual_cloud_fusion_ratio,
                "actual_cloud_share_of_admitted": actual_cloud_share_of_admitted,
                "cloud_edge_sample_ratio_min": min(cloud_edge_ratios.values(), default=0.0),
                "cloud_edge_sample_ratio_max": max(cloud_edge_ratios.values(), default=0.0),
                "admitted_client_ids_objective": ";".join(str(cid) for cid in profile_evaluation.admitted_client_ids) if profile_evaluation else "",
                "admitted_client_ids_actual": ";".join(str(cid) for cid in flow_result.selected_client_ids),
                "num_clients": len(clients),
            }
        )
        current_round = round_rows[-1]
        current_round.update(training_privacy_diagnostics(current_round))
        max_feature_epsilon = current_round["max_feature_epsilon"]
        max_update_epsilon = current_round["max_update_epsilon"]
        larger_channel_epsilon = max(max_feature_epsilon, max_update_epsilon)
        cumulative_comm = sum(float(row["communication_volume"]) for row in round_rows)
        selected_details = [
            {
                "client_id": client_id,
                "edge_id": client_by_id[client_id].edge_id,
                "mode": candidate.mode,
                "mechanisms": candidate_mechanism_label(candidate),
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
            "larger_channel_epsilon": larger_channel_epsilon,
            "feature_epsilon": max_feature_epsilon,
            "update_epsilon": max_update_epsilon,
            "privacy_guarantee": current_round["privacy_guarantee"],
            "dp_delta": privacy_parameters["delta"],
            "dp_feature_noise_multiplier": privacy_parameters["feature_noise_multiplier"],
            "dp_update_noise_multiplier": privacy_parameters["update_noise_multiplier"],
            "min_remaining_epsilon": current_round["min_remaining_epsilon"],
            "effective_clients": current_round["num_effective_clients"],
            "effective_edges": current_round["num_effective_edges"],
            "global_update_norm": current_round["global_update_norm"],
            "admitted_update_norm_mean": current_round["admitted_update_norm_mean"],
            "admitted_update_norm_max": current_round["admitted_update_norm_max"],
            "cloud_input_update_norm_mean": current_round["cloud_input_update_norm_mean"],
            "cloud_input_update_norm_max": current_round["cloud_input_update_norm_max"],
            "num_feature_dp_clients": current_round["num_feature_dp_clients"],
            "num_update_dp_clients": current_round["num_update_dp_clients"],
            "num_he_clients": current_round["num_he_clients"],
            "num_update_dp_only_clients": current_round["num_update_dp_only_clients"],
            "num_update_he_only_clients": current_round["num_update_he_only_clients"],
            "num_update_dp_he_clients": current_round["num_update_dp_he_clients"],
            "he_selected_but_profiled_clients": current_round[
                "he_selected_but_profiled_clients"
            ],
            "he_execution_status": current_round["he_execution_status"],
            "protected_object": current_round["protected_object"],
            "privacy_mechanism_scope": current_round["privacy_mechanism_scope"],
            "update_dp_coverage": current_round["update_dp_coverage"],
            "update_he_coverage": current_round["update_he_coverage"],
            "uniform_update_dp": current_round["uniform_update_dp"],
            "all_cross_domain_updates_protected": current_round[
                "all_cross_domain_updates_protected"
            ],
            "num_budget_exhausted_clients": current_round["num_budget_exhausted_clients"],
            "omega_feature_clip_excess_sq_mean": current_round["omega_feature_clip_excess_sq_mean"],
            "omega_feature_clip_excess_sq_max": current_round["omega_feature_clip_excess_sq_max"],
            "real_he_used": round_real_he_used,
            "he_aggregation_calls": current_round["he_aggregation_calls"],
            "he_ciphertext_count": current_round["he_ciphertext_count"],
            "he_ciphertext_bytes": current_round["he_ciphertext_bytes"],
            "he_ciphertext_bytes_semantics": current_round[
                "he_ciphertext_bytes_semantics"
            ],
            "he_wall_time_sec": current_round["he_wall_time_sec"],
            "he_worker_cpu_time_sec": current_round["he_worker_cpu_time_sec"],
            "he_process_fallbacks": current_round["he_process_fallbacks"],
            "he_operation_time_sec": sum(
                float(current_round[field])
                for field in (
                    "he_key_setup_time_sec",
                    "he_encryption_time_sec",
                    "he_addition_time_sec",
                    "he_decryption_time_sec",
                )
            ),
            "he_max_abs_error": current_round["he_max_abs_error"],
            "system_latency_objective": current_round["system_latency_objective"],
            "system_omega_objective": current_round["system_omega_objective"],
            "cloud_fusion_ratio": current_round["cloud_fusion_ratio"],
            "selector_cloud_fusion_ratio": current_round["selector_cloud_fusion_ratio"],
            "actual_cloud_fusion_ratio": current_round["actual_cloud_fusion_ratio"],
            "actual_cloud_share_of_admitted": current_round["actual_cloud_share_of_admitted"],
            "cloud_edge_sample_ratio_min": current_round["cloud_edge_sample_ratio_min"],
            "cloud_edge_sample_ratio_max": current_round["cloud_edge_sample_ratio_max"],
            "admitted_client_ids_objective": current_round["admitted_client_ids_objective"],
            "admitted_client_ids_actual": current_round["admitted_client_ids_actual"],
            "selected_client_ids": sorted(flow_result.selected_client_ids),
            "selected_clients": selected_details,
            "skipped_clients": skipped_clients,
            "infeasible_clients": infeasible,
            "mode_distribution": _distribution(row["mode"] for row in decision_rows),
            "wall_time_sec": current_round["cumulative_wall_time_sec"],
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "recent_rounds": round_rows[-20:],
        }
        last_completed_status.clear()
        last_completed_status.update(live_payload)
        _write_live_status(output_dir / "live_status.json", live_payload)
        _write_live_status(parent_status_path, live_payload)
        _write_csv(output_dir / "round_metrics.csv", round_rows)
        _write_csv(output_dir / "protection_releases.csv", [
            release for row in round_rows
            for release in json.loads(row.get("protection_release_audit", "[]"))
        ])
        if current_round["training_health"] == "non_finite":
            failed_payload = dict(live_payload, status="failed", message=(
                "Non-finite training metrics or updates detected. The last valid "
                "checkpoint was preserved; inspect round_metrics.csv."
            ))
            _write_live_status(output_dir / "live_status.json", failed_payload)
            _write_live_status(parent_status_path, failed_payload)
            raise FloatingPointError(f"[{policy}] round {round_idx + 1}: non-finite training state")
        _save_policy_checkpoint(
            checkpoint_path,
            policy=policy,
            next_round=round_idx + 1,
            global_end=global_end,
            global_edge=global_edge,
            remaining_epsilon=remaining_epsilon,
            privacy_ledgers=privacy_ledgers,
            previous_choices=previous_choices,
            round_rows=round_rows,
            decision_rows=decision_rows,
            flow_event_rows=flow_event_rows,
            link_state_rows=link_state_rows,
            best_accuracy=best_accuracy,
            logical_time=logical_time,
            rng=rng,
            np_rng=np_rng,
            train_config=train_config,
            selection=selection,
            real_he_rounds=real_he_rounds,
            real_he_aggregated_clients=real_he_aggregated_clients,
            global_pareto_selection_rounds=global_pareto_selection_rounds,
            client_model_states=client_model_states,
        )
        privacy_progress = (
            f"feature_dp=off update_eps={max_update_epsilon:.3f}"
            if effective_selection.trusted_edge_split_execution
            else (
                f"feature_eps={max_feature_epsilon:.3f} "
                f"update_eps={max_update_epsilon:.3f}"
            )
        )
        print(
            f"  [{policy}] round {round_idx + 1:03d}/{selection.rounds} "
            f"acc={test_accuracy:.4f} "
            f"acc_delta={current_round['accuracy_change_from_previous']:+.4f} "
            f"best={best_accuracy:.4f} "
            f"logical_time={logical_time:.2f}s "
            f"wall_time={current_round['cumulative_wall_time_sec']:.2f}s {privacy_progress} "
            f"dp={num_update_dp_only_clients} he={num_update_he_only_clients} "
            f"dp_he={num_update_dp_he_clients} "
            f"he_exec={current_round['he_execution_status']} "
            f"clients={current_round['num_effective_clients']}",
            flush=True,
        )
        print(
            f"    health={current_round['training_health']} loss={test_loss:.6g} "
            f"signal_norm={pre_dp_global_update_norm:.6g} "
            f"noise_norm={update_dp_noise_norm:.6g} "
            f"noise/signal={current_round['update_dp_noise_to_signal_ratio']:.6g} "
            f"clip_fraction={current_round['update_dp_clip_fraction']:.3f} "
            f"dp_releases={update_dp_release_count} "
            f"candidate_filter="
            f"{current_round['selection_pool_candidates_before_stability']}->"
            f"{current_round['selection_pool_candidates_after_stability']} "
            f"he_wall={current_round.get('he_wall_time_sec', 0.0):.3f}s "
            f"packets_without_dp_calibration={current_round['update_packets_without_dp_calibration']} "
            f"end_to_end_dp=not_established",
            flush=True,
        )
        worker_results.clear()
        flow_inputs.clear()
        initial_admitted_updates.clear()
        overridden_admitted_diffs.clear()
        admitted_updates.clear()
        global_updates.clear()
        cloud_updates.clear()
        cloud_signal_updates.clear()
        local_dp_pending_components.clear()
        local_dp_noise_accumulator.clear()
        aggregate_dp_noise_accumulator.clear()
        edge_cloud_groups.clear()
        edge_only_groups.clear()
        global_state_diffs.clear()
        global_candidates.clear()
        global_he_mask.clear()
        flow_result.state_diffs.clear()
        flow_result = None
        result = None
        state_diff = None
        returned_state = None
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if stop_round < selection.rounds:
        partial_summary = _summarize_lenet5_policy(
            policy,
            round_rows,
            decision_rows,
            output_dir,
            selection.time_limit,
            model_name,
            dataset_label,
        )
        partial_summary.update(
            {
                "status": "paused",
                "target_rounds": selection.rounds,
                "next_round": stop_round,
                "checkpoint": str(checkpoint_path),
                "execution_revision": train_config.execution_revision,
                "dp_accounting_mode": privacy_parameters["accounting_mode"],
                "dp_feature_horizon_events": privacy_parameters["feature_horizon_events"],
                "dp_feature_enabled": privacy_parameters["feature_dp_enabled"],
                "dp_update_horizon_events": privacy_parameters["update_horizon_events"],
                "trusted_edge_split_execution": effective_selection.trusted_edge_split_execution,
                "dp_feature_noise_multiplier": privacy_parameters["feature_noise_multiplier"],
                "dp_update_noise_multiplier": privacy_parameters["update_noise_multiplier"],
                "he_backend": he_status.backend,
                "he_execution": train_config.he_execution,
                "he_wall_time_sec": sum(
                    float(row.get("he_wall_time_sec", 0.0)) for row in round_rows
                ),
                "he_worker_cpu_time_sec": sum(
                    float(row.get("he_worker_cpu_time_sec", 0.0)) for row in round_rows
                ),
                "he_max_abs_error": max(
                    (float(row.get("he_max_abs_error", 0.0)) for row in round_rows),
                    default=0.0,
                ),
                "he_process_fallbacks": sum(
                    int(row.get("he_process_fallbacks", 0)) for row in round_rows
                ),
            }
        )
        _write_csv(output_dir / "client_decisions.csv", decision_rows)
        _write_csv(output_dir / "flow_events.csv", flow_event_rows)
        _write_csv(output_dir / "link_state.csv", link_state_rows)
        _write_json(output_dir / "partial_summary.json", partial_summary)
        paused_payload = {
            **last_completed_status,
            "status": "paused",
            "active_policy": policy,
            "policy": policy,
            "round": stop_round,
            "rounds": selection.rounds,
            "progress": stop_round / max(selection.rounds, 1),
            "checkpoint": str(checkpoint_path),
            "message": (
                f"Stopped after {max_new_rounds} new round(s); "
                "resume with the same target horizon"
            ),
            "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        _write_live_status(output_dir / "live_status.json", paused_payload)
        _write_live_status(parent_status_path, paused_payload)
        return partial_summary

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
    summary["status"] = "completed"
    summary["target_rounds"] = selection.rounds
    summary["per_client_test_accuracy_mean"] = float(np.mean(client_accs))
    summary["per_client_test_accuracy_min"] = float(np.min(client_accs))
    summary["per_client_test_accuracy_max"] = float(np.max(client_accs))
    summary["per_client_test_accuracy_std"] = float(np.std(client_accs))
    summary["per_client_test"] = per_client_test
    summary["he_backend"] = he_status.backend
    summary["he_execution"] = train_config.he_execution
    summary["he_available"] = he_status.available
    summary["he_status"] = he_status.detail
    summary["real_he_available"] = real_he_available
    summary["real_he_rounds"] = real_he_rounds
    summary["real_he_aggregated_clients"] = real_he_aggregated_clients
    summary["he_complete_update_encryption"] = (
        train_config.he_execution == "real" and real_he_rounds > 0
    )
    summary["he_poly_modulus_degree"] = CKKS_POLY_MODULUS_DEGREE
    summary["he_coeff_mod_bit_sizes"] = list(CKKS_COEFF_MOD_BIT_SIZES)
    summary["he_scale_bits"] = int(np.log2(CKKS_SCALE))
    for field in (
        "he_aggregation_calls",
        "he_encrypted_updates",
        "he_encrypted_parameter_values",
        "he_ciphertext_count",
        "he_ciphertext_bytes",
        "he_process_tasks",
        "he_process_fallbacks",
        "he_failures",
    ):
        summary[field] = int(sum(int(row.get(field, 0)) for row in round_rows))
    for field in (
        "he_process_workers",
        "he_process_chunks_per_task",
        "he_shared_memory_bytes",
        "he_mapped_update_bytes",
    ):
        summary[field] = int(
            max((int(row.get(field, 0)) for row in round_rows), default=0)
        )
    for field in (
        "he_wall_time_sec",
        "he_key_setup_time_sec",
        "he_encryption_time_sec",
        "he_addition_time_sec",
        "he_decryption_time_sec",
    ):
        summary[field] = float(sum(float(row.get(field, 0.0)) for row in round_rows))
    summary["he_worker_cpu_time_sec"] = float(
        summary["he_encryption_time_sec"]
        + summary["he_addition_time_sec"]
        + summary["he_decryption_time_sec"]
    )
    summary["he_ciphertext_bytes_semantics"] = (
        "serialized_bytes"
        if he_status.backend == "tenseal"
        else "seal_save_size_upper_bound"
        if he_status.backend == "seal"
        else "not_applicable"
    )
    summary["he_max_abs_error"] = max(
        (float(row.get("he_max_abs_error", 0.0)) for row in round_rows),
        default=0.0,
    )
    summary["he_measured_expansion_ratio"] = (
        float(summary["he_ciphertext_bytes"])
        / max(1.0, 4.0 * float(summary["he_encrypted_parameter_values"]))
    )
    summary["global_pareto_selection_rounds"] = global_pareto_selection_rounds
    summary["execution_revision"] = train_config.execution_revision
    summary["dp_accounting_mode"] = privacy_parameters["accounting_mode"]
    summary["dp_delta"] = privacy_parameters["delta"]
    summary["dp_feature_epsilon_target"] = privacy_parameters["feature_budget"]
    summary["dp_update_epsilon_target"] = privacy_parameters["update_budget"]
    summary["dp_feature_noise_multiplier"] = privacy_parameters["feature_noise_multiplier"]
    summary["dp_update_noise_multiplier"] = privacy_parameters["update_noise_multiplier"]
    summary["dp_feature_horizon_events"] = privacy_parameters["feature_horizon_events"]
    summary["dp_update_horizon_events"] = privacy_parameters["update_horizon_events"]
    summary["dp_feature_enabled"] = privacy_parameters["feature_dp_enabled"]
    summary["trusted_edge_split_execution"] = effective_selection.trusted_edge_split_execution
    summary["tracks_returned_client_models"] = True
    summary["privacy_policy_scope"] = "per_link"
    summary["protected_object"] = "cross_domain_model_update"
    summary["privacy_mechanism_scope"] = "local_packet_or_secure_aggregate"
    summary["aggregate_dp_protocol"] = "distributed_noise_before_ckks_aggregation"
    summary["aggregate_dp_release_rounds"] = sum(
        int(row.get("aggregate_dp_release", 0)) for row in round_rows
    )
    summary["local_packet_dp_releases"] = sum(
        int(row.get("local_packet_dp_release_count", 0)) for row in round_rows
    )
    summary["max_aggregate_dp_noise_std"] = max(
        (float(row.get("aggregate_dp_noise_std", 0.0)) for row in round_rows),
        default=0.0,
    )
    summary["he_profiled_only"] = train_config.he_execution == "profiled"
    summary["mean_update_dp_only_clients"] = _list_mean(
        row.get("num_update_dp_only_clients", 0) for row in round_rows
    )
    summary["mean_update_he_only_clients"] = _list_mean(
        row.get("num_update_he_only_clients", 0) for row in round_rows
    )
    summary["mean_update_dp_he_clients"] = _list_mean(
        row.get("num_update_dp_he_clients", 0) for row in round_rows
    )
    summary["mean_update_dp_coverage"] = _list_mean(
        row.get("update_dp_coverage", 0.0) for row in round_rows
    )
    summary["mean_update_he_coverage"] = _list_mean(
        row.get("update_he_coverage", 0.0) for row in round_rows
    )
    summary["privacy_execution_audit"] = privacy_execution_audit(round_rows)
    summary["uniform_selected_update_dp"] = summary["privacy_execution_audit"]["uniform_selected_update_dp"]
    # Retain the legacy key without interpreting coverage as a privacy proof.
    summary["uniform_end_to_end_update_dp"] = None
    summary["all_cross_domain_updates_protected"] = all(
        bool(row.get("all_cross_domain_updates_protected", False))
        for row in round_rows
    )
    summary["torch_seed"] = selection.seed
    summary["deterministic_torch"] = True
    end_to_end_wall_time_sec = (
        completed_wall_time_offset_sec
        + time.perf_counter()
        - policy_started_at
    )
    summary["wall_time_sec"] = end_to_end_wall_time_sec
    summary["end_to_end_wall_time_sec"] = end_to_end_wall_time_sec
    summary["accounted_system_time_sec"] = (
        float(summary["total_logical_time"])
        + float(summary["total_selection_wall_time_sec"])
    )
    summary["timing_scope"] = {
        "end_to_end_wall_time_sec": "policy initialization through final summary preparation",
        "cumulative_wall_time_sec": "policy initialization through each completed round",
        "total_round_wall_time_sec": "sum of measured training rounds",
        "total_logical_time": "modeled cloud-edge-end execution path",
        "accounted_system_time_sec": "modeled execution path plus measured mode selection",
    }
    _write_csv(output_dir / "round_metrics.csv", round_rows)
    _write_csv(output_dir / "client_decisions.csv", decision_rows)
    _write_csv(output_dir / "flow_events.csv", flow_event_rows)
    _write_csv(output_dir / "link_state.csv", link_state_rows)
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
        if (candidate.link_mechanisms or candidate.mechanisms) != (
            previous.link_mechanisms or previous.mechanisms
        ):
            continue
        if not candidate.feasible_device:
            continue
        if candidate.epsilon_used > remaining_epsilon + 1e-12:
            continue
        if candidate.feasible:
            return replace(
                candidate,
                omega_feature_clip_excess_sq=getattr(
                    previous,
                    "omega_feature_clip_excess_sq",
                    None,
                ),
            )
    return None


def _run_client_training_tasks(
    *,
    tasks: list[tuple[int, Candidate, np.ndarray, int]],
    train_config: Lenet5Config,
    selection: SelectionConfig,
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    client_model_states: dict[int, dict[str, dict[str, torch.Tensor]]],
    x_train: np.ndarray,
    y_train: np.ndarray,
    device: torch.device,
    model_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    np_rng: np.random.Generator,
    round_idx: int,
    training_stage: int = 0,
    progress_callback: Callable[[int, int], None] | None = None,
    progress_event_callback: Callable[[str, int, int], None] | None = None,
) -> dict[int, dict[str, Any]]:
    if not tasks:
        return {}
    executor = train_config.executor.strip().lower()
    if executor not in {"serial", "process_pool"}:
        raise ValueError("Lenet5Config.executor must be 'serial' or 'process_pool'.")
    if executor == "process_pool" and device.type != "cpu":
        raise ValueError("process_pool executor is CPU-only; use --device cpu or --executor serial.")

    worker_device = device if executor == "serial" else torch.device("cpu")
    privacy_parameters = resolved_privacy_parameters(selection)

    def build_payload(client_id: int, candidate: Candidate, idx: np.ndarray) -> dict[str, Any]:
        if _mode_reaches_cloud(candidate.mode) and training_stage == 0:
            client_model_states.pop(client_id, None)
        returned_state = _training_base_state(
            client_id=client_id,
            candidate=candidate,
            client_model_states=client_model_states,
            global_end=global_end,
            global_edge=global_edge,
            device=torch.device("cpu"),
            training_stage=training_stage,
        )
        return {
            "client_id": client_id,
            "mode": candidate.mode,
            "global_end_state": _state_dict_to_device(returned_state["end"], worker_device),
            "global_edge_state": _state_dict_to_device(returned_state["edge"], worker_device),
            "x": x_train[idx],
            "y": y_train[idx],
            "epochs": train_config.local_epochs,
            "lr": train_config.learning_rate,
            "l2": train_config.l2,
            "local_steps": selection.L_block_cycles,
            "model_name": model_name,
            "input_shape": input_shape,
            "num_classes": num_classes,
            "mechanisms": _candidate_training_mechanisms(
                candidate,
                aggregate_cloud_update_dp=selection.trusted_edge_split_execution,
            ),
            "dp_clip_norm": train_config.dp_clip_norm,
            "dp_feature_noise_multiplier": privacy_parameters["feature_noise_multiplier"],
            "dp_update_noise_multiplier": privacy_parameters["update_noise_multiplier"],
            "dp_update_mode": train_config.dp_update_mode,
            "dp_epsilon": max(selection.dp_emb_epsilon, 1e-6),
            "dp_seed": _dp_noise_seed(
                selection.seed,
                round_idx,
                client_id,
                training_stage,
            ),
            "training_seed": _client_training_seed(
                selection.seed,
                round_idx,
                client_id,
                training_stage,
            ),
            "device": str(worker_device),
        }

    if executor == "serial":
        results: dict[int, dict[str, Any]] = {}
        model_cache: dict[str, Any] = {}
        total = len(tasks)
        for completed, (client_id, candidate, idx, _sequence) in enumerate(tasks, start=1):
            client_label = (
                f"client {client_id} mode {candidate.mode} "
                f"samples {len(idx)} epochs {train_config.local_epochs} "
                f"max steps {selection.L_block_cycles}"
            )
            if progress_event_callback is not None:
                progress_event_callback(f"preparing {client_label}", completed - 1, total)
            payload = build_payload(client_id, candidate, idx)
            if progress_event_callback is not None:
                progress_event_callback(f"training {client_label}", completed - 1, total)
            result = _client_train_worker(payload, model_cache=model_cache)
            results[int(result["client_id"])] = result
            if progress_callback is not None:
                progress_callback(completed, total)
        return results

    payloads = [
        build_payload(client_id, candidate, idx)
        for client_id, candidate, idx, _sequence in tasks
    ]
    max_workers = train_config.executor_workers or min(len(payloads), 4)
    max_workers = max(1, min(int(max_workers), len(payloads)))
    results: dict[int, dict[str, Any]] = {}
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = [pool.submit(_client_train_worker, payload) for payload in payloads]
        total = len(futures)
        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            results[int(result["client_id"])] = result
            if progress_callback is not None:
                progress_callback(completed, total)
    return results


def _training_base_state(
    *,
    client_id: int,
    candidate: Candidate,
    client_model_states: dict[int, dict[str, dict[str, torch.Tensor]]],
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    device: torch.device,
    training_stage: int = 0,
) -> dict[str, dict[str, torch.Tensor]]:
    stored_state = client_model_states.get(client_id)
    if stored_state is not None and (
        not _mode_reaches_cloud(candidate.mode) or training_stage > 0
    ):
        return stored_state
    return {
        "end": _state_dict_to_device(global_end.state_dict(), device),
        "edge": _state_dict_to_device(global_edge.state_dict(), device),
    }


def _profiles_with_actual_samples(
    clients: list[Any],
    client_train_indices: list[np.ndarray],
) -> list[Any]:
    return [
        replace(
            client,
            samples=max(1, int(len(client_train_indices[client.client_id]))),
        )
        for client in clients
    ]


def _feature_clip_excess_sq_by_client(
    end_model: torch.nn.Module,
    x_train: np.ndarray,
    client_train_indices: list[np.ndarray],
    *,
    clip_norm: float,
    device: torch.device,
    input_shape: tuple[int, int, int],
) -> dict[int, float]:
    if input_shape[0] > 1:
        return {
            client_id: 0.0
            for client_id in range(len(client_train_indices))
        }
    threshold = max(float(clip_norm), 1e-12)
    was_training = end_model.training
    end_model.eval()
    profiles: dict[int, float] = {}
    max_profile_samples = 32
    with torch.no_grad():
        for client_id, indices in enumerate(client_train_indices):
            if len(indices) == 0:
                profiles[client_id] = 0.0
                continue
            profile_indices = indices[: min(len(indices), max_profile_samples)]
            x = torch.from_numpy(x_train[profile_indices]).float().to(device).view(-1, *input_shape)
            embedding = end_model(x).reshape(len(profile_indices), -1)
            norms = torch.linalg.vector_norm(embedding, dim=1)
            excess_sq = torch.relu(norms - threshold) ** 2
            profiles[client_id] = float(excess_sq.mean().item())
    end_model.train(was_training)
    return profiles


def _set_torch_seed(seed: int, device: torch.device) -> None:
    torch.manual_seed(int(seed))
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def _client_training_seed(
    base_seed: int,
    round_idx: int,
    client_id: int,
    training_stage: int = 0,
) -> int:
    modulus = 2**31 - 1
    value = (
        (int(base_seed) + 1) * 1_000_003
        + (int(round_idx) + 1) * 10_007
        + (int(client_id) + 1) * 101
        + int(training_stage) * 1_009
    ) % modulus
    return max(1, value)


def _dp_noise_seed(
    base_seed: int,
    round_idx: int,
    entity_id: Any,
    stage: int = 0,
) -> int:
    modulus = 2**31 - 1
    if isinstance(entity_id, (int, np.integer)):
        entity_value = int(entity_id)
    else:
        encoded = json.dumps(entity_id, sort_keys=True, separators=(",", ":")).encode("utf-8")
        entity_value = int.from_bytes(hashlib.sha256(encoded).digest()[:8], "big")
    value = (
        (int(base_seed) + 1) * 1_000_033
        + (int(round_idx) + 1) * 10_009
        + (entity_value + 1) * 103
        + int(stage) * 1_021
    ) % modulus
    return max(1, value)


def _attach_feature_clip_profiles(
    selected: list[tuple[int, Candidate, list[Candidate], float]],
    profiles: dict[int, float],
) -> list[tuple[int, Candidate, list[Candidate], float]]:
    attached: list[tuple[int, Candidate, list[Candidate], float]] = []
    for client_id, current, candidates, remaining in selected:
        profile = max(float(profiles.get(client_id, 0.0)), 0.0)
        attached.append(
            (
                client_id,
                replace(current, omega_feature_clip_excess_sq=profile),
                [
                    replace(candidate, omega_feature_clip_excess_sq=profile)
                    for candidate in candidates
                ],
                remaining,
            )
        )
    return attached


def _state_dict_to_device(
    state: dict[str, torch.Tensor],
    device: torch.device,
) -> dict[str, torch.Tensor]:
    return {key: value.detach().to(device).clone() for key, value in state.items()}


def _state_dict_to_device_nested(
    state: dict[str, dict[str, torch.Tensor]],
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        part_key: _state_dict_to_device(part_state, device)
        for part_key, part_state in state.items()
    }


def _aggregate_returned_client_models(
    *,
    updates: list[tuple[int, Any, int, Candidate | None]],
    client_model_states: dict[int, dict[str, dict[str, torch.Tensor]]],
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    device: torch.device,
    model_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    he_aggregation_size: int = 0,
    he_backend: str = "none",
    he_metrics: HEOperationMetrics | None = None,
) -> tuple[dict[str, dict[str, torch.Tensor]], bool]:
    """FedAvg local models at an edge and return the aggregate model state."""
    if not updates:
        raise ValueError("Cannot aggregate an empty edge update set.")

    reference = client_model_states.get(updates[0][0])
    if reference is None:
        reference = {
            "end": _state_dict_to_device(global_end.state_dict(), torch.device("cpu")),
            "edge": _state_dict_to_device(global_edge.state_dict(), torch.device("cpu")),
        }
    temp_end, temp_edge = _clone_split_models(
        global_end,
        global_edge,
        device,
        model_name,
        input_shape,
        num_classes,
    )
    temp_end.load_state_dict(_state_dict_to_device(reference["end"], device))
    temp_edge.load_state_dict(_state_dict_to_device(reference["edge"], device))

    candidates: list[Candidate | None] = []
    sample_counts: list[int] = []
    for _client_id, _state_diff, sample_count, candidate in updates:
        candidates.append(candidate)
        sample_counts.append(sample_count)

    encrypted_mask = [
        mechanism_uses_he(_candidate_edge_update_mechanism(candidate))
        for candidate in candidates
    ]
    use_real_he = he_backend in {"seal", "tenseal"} and any(encrypted_mask)
    if use_real_he:
        temp_end, temp_edge = _fedavg_returned_models_bounded_he(
            updates=updates,
            client_model_states=client_model_states,
            reference=reference,
            temp_end=temp_end,
            temp_edge=temp_edge,
            sample_counts=sample_counts,
            encrypted_mask=encrypted_mask,
            he_backend=he_backend,
            he_aggregation_size=he_aggregation_size,
            he_metrics=he_metrics,
            device=device,
        )
    else:
        temp_end, temp_edge = _fedavg_returned_models_plain_streaming(
            updates=updates,
            client_model_states=client_model_states,
            reference=reference,
            temp_end=temp_end,
            temp_edge=temp_edge,
            sample_counts=sample_counts,
        )

    return (
        {
            "end": _state_dict_to_device(temp_end.state_dict(), torch.device("cpu")),
            "edge": _state_dict_to_device(temp_edge.state_dict(), torch.device("cpu")),
        },
        use_real_he,
    )


def _fedavg_returned_models_bounded_he(
    *,
    updates: list[tuple[int, Any, int, Candidate | None]],
    client_model_states: dict[int, dict[str, dict[str, torch.Tensor]]],
    reference: dict[str, dict[str, torch.Tensor]],
    temp_end: torch.nn.Module,
    temp_edge: torch.nn.Module,
    sample_counts: list[int],
    encrypted_mask: list[bool],
    he_backend: str,
    he_aggregation_size: int,
    he_metrics: HEOperationMetrics | None,
    device: torch.device,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    work_device = torch.device("cpu")
    total = max(1.0, float(sum(sample_counts)))
    total_size = sum(
        int(param.numel())
        for model in (temp_end, temp_edge)
        for param in model.parameters()
    )
    he_size = _resolved_he_aggregation_size(total_size, he_aggregation_size)
    aggregated: dict[str, dict[str, torch.Tensor]] = {"end": {}, "edge": {}}
    he_prefixes: list[torch.Tensor] = []

    for client_id, state_diff, sample_count, _candidate in updates:
        factor = float(sample_count) / total
        base = client_model_states.get(client_id, reference)
        prefix_parts: list[torch.Tensor] = []
        cursor = 0
        for part_name, model in (("end", temp_end), ("edge", temp_edge)):
            for name, param in model.named_parameters():
                base_value = base[part_name][name].to(work_device)
                reference_value = reference[part_name][name].to(work_device)
                diff_value = state_diff.get(part_name, {}).get(name)
                if diff_value is None:
                    diff_value = torch.zeros_like(base_value)
                relative = base_value + diff_value.to(work_device) - reference_value
                weighted = relative * factor
                if name in aggregated[part_name]:
                    aggregated[part_name][name] = aggregated[part_name][name] + weighted
                else:
                    aggregated[part_name][name] = weighted.clone()
                if cursor < he_size:
                    take = min(int(relative.numel()), he_size - cursor)
                    prefix_parts.append(relative.reshape(-1)[:take].detach().cpu())
                cursor += int(relative.numel())
        he_prefixes.append(torch.cat(prefix_parts) if prefix_parts else torch.zeros(0))

    flat_update = _flatten_state_diff(aggregated, temp_end, temp_edge, work_device)
    if he_prefixes and he_size > 0:
        if he_backend == "seal":
            he_prefix = _fedavg_prefix_seal(
                he_prefixes,
                sample_counts,
                encrypted_mask=encrypted_mask,
                he_metrics=he_metrics,
            )
        else:
            he_prefix = _fedavg_prefix_tenseal(
                he_prefixes,
                sample_counts,
                encrypted_mask=encrypted_mask,
                he_metrics=he_metrics,
            )
        flat_update[:he_size] = he_prefix.to(work_device)[:he_size]
    _apply_flat_update(flat_update, temp_end, temp_edge)
    return temp_end, temp_edge


def _fedavg_returned_models_plain_streaming(
    *,
    updates: list[tuple[int, Any, int, Candidate | None]],
    client_model_states: dict[int, dict[str, dict[str, torch.Tensor]]],
    reference: dict[str, dict[str, torch.Tensor]],
    temp_end: torch.nn.Module,
    temp_edge: torch.nn.Module,
    sample_counts: list[int],
) -> tuple[torch.nn.Module, torch.nn.Module]:
    work_device = torch.device("cpu")
    total = max(1.0, float(sum(sample_counts)))
    aggregated: dict[str, dict[str, torch.Tensor]] = {"end": {}, "edge": {}}
    for client_id, state_diff, sample_count, _candidate in updates:
        factor = float(sample_count) / total
        base = client_model_states.get(client_id, reference)
        for part_name, model in (("end", temp_end), ("edge", temp_edge)):
            for name, _param in model.named_parameters():
                base_value = base[part_name][name].to(work_device)
                reference_value = reference[part_name][name].to(work_device)
                diff_value = state_diff.get(part_name, {}).get(name)
                if diff_value is None:
                    diff_value = torch.zeros_like(base_value)
                relative = base_value + diff_value.to(work_device) - reference_value
                weighted = relative * factor
                if name in aggregated[part_name]:
                    aggregated[part_name][name] = aggregated[part_name][name] + weighted
                else:
                    aggregated[part_name][name] = weighted.clone()
    flat_update = _flatten_state_diff(aggregated, temp_end, temp_edge, work_device)
    _apply_flat_update(flat_update, temp_end, temp_edge)
    return temp_end, temp_edge


def _state_difference_from_client_update(
    *,
    client_id: int,
    state_diff: dict[str, dict[str, torch.Tensor]],
    client_model_states: dict[int, dict[str, dict[str, torch.Tensor]]],
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Express a client model update relative to the current global model."""
    base = client_model_states.get(client_id)
    if base is None:
        # The fixed training mask, not observed update values, defines DP support.
        state_diff = {
            part_name: {
                name: state_diff[part_name][name]
                for name, param in model.named_parameters()
                if param.requires_grad and name in state_diff.get(part_name, {})
            }
            for part_name, model in (("end", global_end), ("edge", global_edge))
        }
        if all(value.device == device for part in state_diff.values() for value in part.values()):
            return state_diff
        return _state_dict_to_device_nested(state_diff, device)
    difference: dict[str, dict[str, torch.Tensor]] = {"end": {}, "edge": {}}
    for part_name, model in (("end", global_end), ("edge", global_edge)):
        global_state = model.state_dict()
        for name, _param in model.named_parameters():
            if not _param.requires_grad:
                continue
            base_value = base[part_name][name].to(device)
            local_delta = state_diff.get(part_name, {}).get(
                name,
                torch.zeros_like(base_value),
            ).to(device)
            difference[part_name][name] = (
                base_value + local_delta - global_state[name].to(device)
            )
    return difference


def _state_difference_from_model(
    state: dict[str, dict[str, torch.Tensor]],
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    difference: dict[str, dict[str, torch.Tensor]] = {"end": {}, "edge": {}}
    for part_name, model in (("end", global_end), ("edge", global_edge)):
        reference = model.state_dict()
        for name, _param in model.named_parameters():
            if not _param.requires_grad:
                continue
            difference[part_name][name] = (
                state[part_name][name].to(device) - reference[name].to(device)
            )
    return difference


def _state_difference_l2_norm(
    state_diff: dict[str, dict[str, torch.Tensor]],
) -> float:
    return _weighted_state_difference_norm([state_diff], [1])


def _weighted_average_state_differences(
    state_diffs: list[dict[str, dict[str, torch.Tensor]]],
    sample_counts: list[float],
    device: torch.device = torch.device("cpu"),
) -> dict[str, dict[str, torch.Tensor]]:
    if not state_diffs:
        return {"end": {}, "edge": {}}
    total = max(1e-12, sum(max(float(count), 0.0) for count in sample_counts))
    averaged: dict[str, dict[str, torch.Tensor]] = {}
    for state_diff, count in zip(state_diffs, sample_counts):
        factor = max(float(count), 0.0) / total
        for part_name, values in state_diff.items():
            target = averaged.setdefault(part_name, {})
            for name, value in values.items():
                weighted = value.detach().to(device) * factor
                if name in target:
                    target[name].add_(weighted)
                else:
                    target[name] = weighted.clone()
    return averaged


def _add_state_differences(
    first: dict[str, dict[str, torch.Tensor]],
    second: dict[str, dict[str, torch.Tensor]],
    device: torch.device = torch.device("cpu"),
) -> dict[str, dict[str, torch.Tensor]]:
    result: dict[str, dict[str, torch.Tensor]] = {}
    for part_name in set(first) | set(second):
        target = result.setdefault(part_name, {})
        first_part = first.get(part_name, {})
        second_part = second.get(part_name, {})
        for name in set(first_part) | set(second_part):
            left = first_part.get(name)
            right = second_part.get(name)
            if left is None:
                target[name] = right.detach().to(device).clone()
            elif right is None:
                target[name] = left.detach().to(device).clone()
            else:
                target[name] = left.detach().to(device) + right.detach().to(device)
    return result


def _accumulate_scaled_state_difference(
    target: dict[str, dict[str, torch.Tensor]],
    source: dict[str, dict[str, torch.Tensor]],
    factor: float,
) -> None:
    for part_name, values in source.items():
        target_part = target.setdefault(part_name, {})
        for name, value in values.items():
            if not (torch.is_floating_point(value) or torch.is_complex(value)):
                continue
            scaled = value.detach().to(device="cpu") * float(factor)
            if name in target_part:
                target_part[name].add_(scaled)
            else:
                target_part[name] = scaled.clone()


def _weighted_state_difference_norm(
    state_diffs: list[dict[str, dict[str, torch.Tensor]]],
    sample_counts: list[float],
) -> float:
    if not state_diffs:
        return 0.0
    total = max(1e-12, sum(float(count) for count in sample_counts))
    return _linear_combination_state_difference_norm(
        state_diffs,
        [float(count) / total for count in sample_counts],
    )


def _linear_combination_state_difference_norm(
    state_diffs: list[dict[str, dict[str, torch.Tensor]]],
    factors: list[float],
) -> float:
    if not state_diffs:
        return 0.0
    aggregated: dict[tuple[str, str], torch.Tensor] = {}
    for state_diff, factor in zip(state_diffs, factors):
        for part_name, values in state_diff.items():
            for name, value in values.items():
                if not (torch.is_floating_point(value) or torch.is_complex(value)):
                    continue
                key = (part_name, name)
                weighted = value.detach().to(dtype=torch.float64) * factor
                if key in aggregated:
                    aggregated[key] = aggregated[key] + weighted
                else:
                    aggregated[key] = weighted.clone()
    norm_sq = sum(
        float(torch.sum(torch.abs(value) ** 2).item())
        for value in aggregated.values()
    )
    return float(np.sqrt(max(norm_sq, 0.0)))


def _cloud_update_edge(
    client_ids: list[int],
    client_edges: dict[int, int],
) -> int:
    edges = {int(client_edges[client_id]) for client_id in client_ids}
    if len(edges) != 1:
        raise ValueError("A cloud aggregation object must belong to exactly one edge domain.")
    return next(iter(edges))


def _dp_update_release_parameters(
    max_client_fraction: float,
    *,
    clip_norm: float,
    noise_multiplier: float,
) -> tuple[float, float]:
    """Return replacement sensitivity and Gaussian std for one update packet."""
    fraction = float(max_client_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("max_client_fraction must be finite and in [0, 1]")
    if not math.isfinite(clip_norm) or clip_norm <= 0.0:
        raise ValueError("clip_norm must be finite and positive")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be finite and positive")
    sensitivity = 2.0 * float(clip_norm) * fraction
    standard_deviation = float(noise_multiplier) * sensitivity
    if not math.isfinite(sensitivity) or not math.isfinite(standard_deviation):
        raise ValueError("Packet DP calibration overflowed")
    return sensitivity, standard_deviation


def _distributed_aggregate_dp_parameters(
    aggregation_weights: list[float],
    within_packet_client_fractions: list[float],
    *,
    clip_norm: float,
    noise_multiplier: float,
) -> tuple[list[float], float, float, float, list[float]]:
    """Calibrate independent packet noise shares for one DP aggregate release."""
    if len(aggregation_weights) != len(within_packet_client_fractions):
        raise ValueError("Aggregation weights and client fractions must have equal length")
    if not math.isfinite(clip_norm) or clip_norm <= 0.0:
        raise ValueError("clip_norm must be finite and positive")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be finite and positive")
    if any(not math.isfinite(weight) or weight < 0.0 for weight in aggregation_weights):
        raise ValueError("Aggregation weights must be finite and non-negative")
    if any(not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0 for fraction in within_packet_client_fractions):
        raise ValueError("Within-packet client fractions must be finite and in [0, 1]")
    total_weight = sum(max(float(weight), 0.0) for weight in aggregation_weights)
    if not math.isfinite(total_weight):
        raise ValueError("Total aggregation weight must be finite")
    if total_weight <= 0.0:
        return [0.0 for _ in aggregation_weights], 0.0, 0.0, 0.0, [
            0.0 for _ in aggregation_weights
        ]

    normalized_weights = [
        max(float(weight), 0.0) / total_weight for weight in aggregation_weights
    ]
    protected_indices = [
        index
        for index, (weight, fraction) in enumerate(
            zip(normalized_weights, within_packet_client_fractions)
        )
        if weight > 0.0 and float(fraction) > 0.0
    ]
    if not protected_indices:
        return normalized_weights, 0.0, 0.0, 0.0, [
            0.0 for _ in aggregation_weights
        ]

    max_client_weight = max(
        normalized_weights[index]
        * max(float(within_packet_client_fractions[index]), 0.0)
        for index in protected_indices
    )
    sensitivity = 2.0 * float(clip_norm) * max_client_weight
    aggregate_noise_std = float(noise_multiplier) * sensitivity
    weighted_share_std = aggregate_noise_std / math.sqrt(len(protected_indices))
    share_stds = [0.0 for _ in aggregation_weights]
    for index in protected_indices:
        share_stds[index] = weighted_share_std / normalized_weights[index]
    if not all(math.isfinite(value) for value in [sensitivity, aggregate_noise_std, *share_stds]):
        raise ValueError("Aggregate DP calibration overflowed")
    return (
        normalized_weights,
        max_client_weight,
        sensitivity,
        aggregate_noise_std,
        share_stds,
    )


def _edge_normalized_cloud_weights(
    cloud_updates: list[tuple[Any, int, Candidate | None, list[int]]],
    *,
    client_edges: dict[int, int],
    edge_total_samples: dict[int, float],
) -> list[float]:
    """Correct round-varying edge participation before global FedAvg."""
    if not cloud_updates:
        return []

    update_edges: list[int] = []
    admitted_by_edge: dict[int, float] = {}
    for _state_diff, sample_count, _candidate, client_ids in cloud_updates:
        edge_id = _cloud_update_edge(client_ids, client_edges)
        count = max(float(sample_count), 0.0)
        update_edges.append(edge_id)
        admitted_by_edge[edge_id] = admitted_by_edge.get(edge_id, 0.0) + count

    weights: list[float] = []
    for edge_id, (_state_diff, sample_count, _candidate, _client_ids) in zip(
        update_edges, cloud_updates
    ):
        admitted = admitted_by_edge[edge_id]
        if admitted <= 0.0:
            weights.append(0.0)
            continue
        edge_mass = max(float(edge_total_samples.get(edge_id, admitted)), 0.0)
        weights.append(edge_mass * max(float(sample_count), 0.0) / admitted)
    return weights


def _cloud_edge_sample_ratios(
    cloud_updates: list[tuple[Any, int, Candidate | None, list[int]]],
    *,
    client_edges: dict[int, int],
    edge_total_samples: dict[int, float],
) -> dict[int, float]:
    admitted_by_edge = {edge_id: 0.0 for edge_id in edge_total_samples}
    for _state_diff, sample_count, _candidate, client_ids in cloud_updates:
        edge_id = _cloud_update_edge(client_ids, client_edges)
        admitted_by_edge[edge_id] = admitted_by_edge.get(edge_id, 0.0) + max(
            float(sample_count), 0.0
        )
    return {
        edge_id: admitted_by_edge.get(edge_id, 0.0) / max(float(total), 1e-12)
        for edge_id, total in edge_total_samples.items()
    }


def _zero_state_difference(
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        "end": {
            name: torch.zeros_like(param, device=device)
            for name, param in global_end.named_parameters()
        },
        "edge": {
            name: torch.zeros_like(param, device=device)
            for name, param in global_edge.named_parameters()
        },
    }


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
                    and candidate.feasible_device
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
    viable = [item for item in candidates if item.feasible_device] or candidates
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
    selection_config: SelectionConfig,
    model_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    device: torch.device,
    np_rng: np.random.Generator,
    round_idx: int,
    top_k: int = 3,
) -> list[tuple[int, Candidate, list[Candidate], float]]:
    """Greedy validation oracle for an empirical accuracy upper reference."""
    privacy_parameters = resolved_privacy_parameters(selection_config)
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
                mechanisms=_candidate_training_mechanisms(
                    candidate,
                    aggregate_cloud_update_dp=selection_config.trusted_edge_split_execution,
                ),
                dp_clip_norm=train_config.dp_clip_norm,
                dp_noise_multiplier=float(privacy_parameters["feature_noise_multiplier"]),
                dp_rng=eval_rng,
                dp_epsilon=max(selection_config.dp_emb_epsilon, 1e-6),
                l2=train_config.l2,
                local_steps=selection_config.L_block_cycles,
                training_seed=_client_training_seed(
                    selection_config.seed,
                    round_idx,
                    client_id,
                ),
            )
            if _should_apply_update_dp(
                _candidate_training_mechanisms(
                    candidate,
                    aggregate_cloud_update_dp=selection_config.trusted_edge_split_execution,
                ),
                train_config.dp_update_mode,
                candidate.mode,
            ):
                state_diff = apply_unified_dp(
                    state_diff,
                    mechanism="dp",
                    clip_norm=train_config.dp_clip_norm,
                    noise_multiplier=float(privacy_parameters["update_noise_multiplier"]),
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
        feasible = [candidate for candidate in candidates if candidate.feasible_device]
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
    _ = model_name, input_shape, num_classes
    temp_end = copy.deepcopy(global_end).to(device)
    temp_edge = copy.deepcopy(global_edge).to(device)
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
                if candidate.feasible_device and candidate.epsilon_used <= rem + 1e-12
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
        key = (
            candidate.mode,
            tuple(sorted((candidate.link_mechanisms or candidate.mechanisms).items())),
        )
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
    return selection_local_omega_proxy(candidate, aggregation_size=aggregation_size)


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
    privacy_ledgers: dict[int, ClientPrivacyLedger],
    previous_choices: dict[int, Candidate],
    round_rows: list[dict[str, Any]],
    decision_rows: list[dict[str, Any]],
    flow_event_rows: list[dict[str, Any]],
    link_state_rows: list[dict[str, Any]],
    best_accuracy: float,
    logical_time: float,
    rng: random.Random,
    np_rng: np.random.Generator,
    train_config: Lenet5Config,
    selection: SelectionConfig,
    real_he_rounds: int,
    real_he_aggregated_clients: int,
    global_pareto_selection_rounds: int,
    client_model_states: dict[int, dict[str, dict[str, torch.Tensor]]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "policy": policy,
            "next_round": next_round,
            "global_end_state": global_end.state_dict(),
            "global_edge_state": global_edge.state_dict(),
            "remaining_epsilon": remaining_epsilon,
            "privacy_ledgers": {
                client_id: ledger.state_dict()
                for client_id, ledger in privacy_ledgers.items()
            },
            "previous_choices": previous_choices,
            "round_rows": round_rows,
            "decision_rows": decision_rows,
            "flow_event_rows": flow_event_rows,
            "link_state_rows": link_state_rows,
            "best_accuracy": best_accuracy,
            "logical_time": logical_time,
            "rng_state": rng.getstate(),
            "np_rng_state": np_rng.bit_generator.state,
            "training": train_config.__dict__,
            "selection": selection.__dict__,
            "real_he_rounds": real_he_rounds,
            "real_he_aggregated_clients": real_he_aggregated_clients,
            "global_pareto_selection_rounds": global_pareto_selection_rounds,
            "client_model_states": client_model_states,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        temporary_path,
    )
    temporary_path.replace(path)


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
        chunks = np.array_split(label_indices, len(clients))
        for client_id, chunk in zip(clients, chunks):
            client_parts[client_id].extend(chunk.tolist())

    empty_clients = [client_id for client_id, part in enumerate(client_parts) if not part]
    for client_id in empty_clients:
        donor_id = max(range(num_clients), key=lambda item: len(client_parts[item]))
        if len(client_parts[donor_id]) <= 1:
            break
        client_parts[client_id].append(client_parts[donor_id].pop())

    return [np.array(sorted(part), dtype=np.int64) for part in client_parts]


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


def _split_evaluate_indexed(
    end_model: torch.nn.Module,
    edge_model: torch.nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    *,
    input_shape: tuple[int, int, int],
    batch_size: int = 128,
) -> tuple[float, float]:
    if len(indices) == 0:
        return 0.0, 0.0
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    for start in range(0, len(indices), batch_size):
        batch_indices = indices[start:start + batch_size]
        loss, accuracy = split_evaluate(
            end_model,
            edge_model,
            x[batch_indices],
            y[batch_indices],
            device,
            input_shape=input_shape,
        )
        batch_count = int(len(batch_indices))
        total_loss += float(loss) * batch_count
        total_correct += int(round(float(accuracy) * batch_count))
        total_seen += batch_count
    return total_loss / max(total_seen, 1), total_correct / max(total_seen, 1)


_SEAL_CKKS_RUNTIME: dict[str, Any] | None = None
_TENSEAL_CKKS_RUNTIME: dict[str, Any] | None = None
_SEAL_PROCESS_RUNTIME: dict[str, Any] | None = None


def _measure_he_wall_time(function: Callable[..., Any]) -> Callable[..., Any]:
    """Record elapsed HE aggregation time separately from summed worker work."""

    @wraps(function)
    def measured(*args: Any, **kwargs: Any) -> Any:
        metrics = kwargs.get("he_metrics")
        started_at = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            if metrics is not None:
                metrics.wall_time_sec += time.perf_counter() - started_at

    return measured


def _reset_ckks_runtime() -> None:
    global _SEAL_CKKS_RUNTIME, _TENSEAL_CKKS_RUNTIME
    _SEAL_CKKS_RUNTIME = None
    _TENSEAL_CKKS_RUNTIME = None


def _seal_ckks_runtime() -> tuple[dict[str, Any], float]:
    global _SEAL_CKKS_RUNTIME
    if _SEAL_CKKS_RUNTIME is not None:
        return _SEAL_CKKS_RUNTIME, 0.0

    import seal

    started_at = time.perf_counter()
    parms = seal.EncryptionParameters(seal.scheme_type.ckks)
    parms.set_poly_modulus_degree(CKKS_POLY_MODULUS_DEGREE)
    parms.set_coeff_modulus(
        seal.CoeffModulus.Create(
            CKKS_POLY_MODULUS_DEGREE,
            list(CKKS_COEFF_MOD_BIT_SIZES),
        )
    )
    context = seal.SEALContext(parms)
    keygen = seal.KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()
    _SEAL_CKKS_RUNTIME = {
        "context": context,
        "public_key": public_key,
        "secret_key": secret_key,
        "encoder": seal.CKKSEncoder(context),
        "encryptor": seal.Encryptor(context, public_key),
        "decryptor": seal.Decryptor(context, secret_key),
        "evaluator": seal.Evaluator(context),
    }
    return _SEAL_CKKS_RUNTIME, time.perf_counter() - started_at


def _init_seal_process_runtime(
    public_key_path: str,
    secret_key_path: str,
    update_store_path: str,
    update_count: int,
    parameter_count: int,
    factors: tuple[float, ...],
    encrypted_mask: tuple[bool, ...],
) -> None:
    global _SEAL_PROCESS_RUNTIME

    import seal

    parms = seal.EncryptionParameters(seal.scheme_type.ckks)
    parms.set_poly_modulus_degree(CKKS_POLY_MODULUS_DEGREE)
    parms.set_coeff_modulus(
        seal.CoeffModulus.Create(
            CKKS_POLY_MODULUS_DEGREE,
            list(CKKS_COEFF_MOD_BIT_SIZES),
        )
    )
    context = seal.SEALContext(parms)
    public_key = seal.PublicKey()
    public_key.load(context, public_key_path)
    secret_key = seal.SecretKey()
    secret_key.load(context, secret_key_path)
    updates = np.memmap(
        update_store_path,
        mode="r",
        dtype=np.float32,
        shape=(update_count, parameter_count),
    )
    _SEAL_PROCESS_RUNTIME = {
        "updates": updates,
        "factors": factors,
        "encrypted_mask": encrypted_mask,
        "encoder": seal.CKKSEncoder(context),
        "encryptor": seal.Encryptor(context, public_key),
        "decryptor": seal.Decryptor(context, secret_key),
        "evaluator": seal.Evaluator(context),
    }


def _seal_process_aggregate_chunk(
    bounds: tuple[int, int],
) -> tuple[int, np.ndarray, int, int, float, float, float, float]:
    if _SEAL_PROCESS_RUNTIME is None:
        raise RuntimeError("SEAL process runtime was not initialized")

    start, stop = bounds
    runtime = _SEAL_PROCESS_RUNTIME
    encoder = runtime["encoder"]
    encryptor = runtime["encryptor"]
    decryptor = runtime["decryptor"]
    evaluator = runtime["evaluator"]
    encrypted_sum = None
    plaintext_sum = np.zeros(stop - start, dtype=np.float64)
    expected_chunk = np.zeros(stop - start, dtype=np.float64)
    ciphertext_count = 0
    ciphertext_bytes = 0
    encryption_time = 0.0
    addition_time = 0.0

    for update, factor, encrypted in zip(
        runtime["updates"],
        runtime["factors"],
        runtime["encrypted_mask"],
    ):
        values = np.ascontiguousarray(
            update[start:stop].astype(np.float64) * factor,
            dtype=np.float64,
        )
        expected_chunk += values
        if not encrypted:
            plaintext_sum += values
            continue
        encryption_started_at = time.perf_counter()
        ciphertext = encryptor.encrypt(encoder.encode(values, CKKS_SCALE))
        encryption_time += time.perf_counter() - encryption_started_at
        ciphertext_count += 1
        ciphertext_bytes += int(ciphertext.save_size())
        if encrypted_sum is None:
            encrypted_sum = ciphertext
        else:
            addition_started_at = time.perf_counter()
            encrypted_sum = evaluator.add(encrypted_sum, ciphertext)
            addition_time += time.perf_counter() - addition_started_at

    if encrypted_sum is None:
        raise ValueError("Encrypted aggregation requires at least one HE protected update")
    addition_started_at = time.perf_counter()
    encrypted_sum = evaluator.add_plain(
        encrypted_sum,
        encoder.encode(np.ascontiguousarray(plaintext_sum), CKKS_SCALE),
    )
    addition_time += time.perf_counter() - addition_started_at
    decryption_started_at = time.perf_counter()
    decoded = decode_seal_vector(encoder, decryptor.decrypt(encrypted_sum))[: stop - start]
    decryption_time = time.perf_counter() - decryption_started_at
    max_abs_error = float(np.max(np.abs(decoded - expected_chunk)))
    return (
        start,
        np.ascontiguousarray(decoded, dtype=np.float32),
        ciphertext_count,
        ciphertext_bytes,
        encryption_time,
        addition_time,
        decryption_time,
        max_abs_error,
    )


def _seal_process_aggregate_chunk_batch(
    bounds_batch: tuple[tuple[int, int], ...],
) -> list[tuple[int, np.ndarray, int, int, float, float, float, float]]:
    """Process adjacent parameter chunks in one executor task."""

    return [_seal_process_aggregate_chunk(bounds) for bounds in bounds_batch]


def _tenseal_ckks_runtime() -> tuple[dict[str, Any], float]:
    global _TENSEAL_CKKS_RUNTIME
    if _TENSEAL_CKKS_RUNTIME is not None:
        return _TENSEAL_CKKS_RUNTIME, 0.0

    import tenseal as ts

    started_at = time.perf_counter()
    context = ts.context(
        ts.SCHEME_TYPE.CKKS,
        poly_modulus_degree=CKKS_POLY_MODULUS_DEGREE,
        coeff_mod_bit_sizes=list(CKKS_COEFF_MOD_BIT_SIZES),
    )
    context.global_scale = CKKS_SCALE
    secret_key = context.secret_key()
    public_context = context.copy()
    public_context.make_context_public()
    _TENSEAL_CKKS_RUNTIME = {
        "context": context,
        "secret_key": secret_key,
        "public_context": public_context,
    }
    return _TENSEAL_CKKS_RUNTIME, time.perf_counter() - started_at


@_measure_he_wall_time
def fedavg_split_tenseal(
    state_diffs: list[dict[str, dict[str, torch.Tensor]]],
    sample_counts: list[float],
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    device: torch.device,
    *,
    chunk_size: int = 4096,
    encrypted_mask: list[bool] | None = None,
    he_aggregation_size: int | None = None,
    he_metrics: HEOperationMetrics | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    total = max(1, sum(sample_counts))
    total_size = sum(
        int(param.numel())
        for model in (global_end, global_edge)
        for param in model.parameters()
    )
    he_size = _resolved_he_aggregation_size(total_size, he_aggregation_size)
    if he_size < total_size:
        mask = _validated_encrypted_mask(encrypted_mask, len(state_diffs))
        return _fedavg_split_bounded_he_streaming(
            state_diffs,
            sample_counts,
            global_end,
            global_edge,
            device,
            encrypted_mask=mask,
            he_backend="tenseal",
            he_aggregation_size=he_size,
            he_metrics=he_metrics,
        )
    work_device = torch.device("cpu")
    flat_updates = [
        _flatten_state_diff(diff, global_end, global_edge, work_device)
        for diff in state_diffs
    ]
    if not flat_updates:
        return global_end, global_edge
    mask = _validated_encrypted_mask(encrypted_mask, len(flat_updates))

    size = int(flat_updates[0].numel())
    he_size = _resolved_he_aggregation_size(size, he_aggregation_size)
    runtime, key_setup_time_sec = _tenseal_ckks_runtime()
    context = runtime["context"]
    secret_key = runtime["secret_key"]
    public_context = runtime["public_context"]
    if he_metrics is not None:
        he_metrics.key_setup_time_sec += key_setup_time_sec
        he_metrics.aggregation_calls += 1
        he_metrics.encrypted_updates += sum(mask)
        he_metrics.encrypted_parameter_values += he_size * sum(mask)
    encrypted_sum = None
    plaintext_sum = np.zeros(size, dtype=np.float64)
    expected_sum = np.zeros(size, dtype=np.float64)
    for flat_update, count, encrypted in zip(flat_updates, sample_counts, mask):
        factor = float(count) / float(total)
        weighted_array = flat_update.detach().cpu().numpy().astype(np.float64) * factor
        expected_sum += weighted_array
        plaintext_sum[he_size:] += weighted_array[he_size:]
        if not encrypted:
            plaintext_sum[:he_size] += weighted_array[:he_size]
            continue
        weighted = weighted_array[:he_size].tolist()
        encryption_started_at = time.perf_counter()
        chunks = [
            ts.ckks_vector(public_context, weighted[start:start + chunk_size])
            for start in range(0, he_size, chunk_size)
        ]
        if he_metrics is not None:
            he_metrics.encryption_time_sec += time.perf_counter() - encryption_started_at
            he_metrics.ciphertext_count += len(chunks)
            he_metrics.ciphertext_bytes += sum(len(chunk.serialize()) for chunk in chunks)
        if encrypted_sum is None:
            encrypted_sum = chunks
        else:
            addition_started_at = time.perf_counter()
            for idx, chunk in enumerate(chunks):
                encrypted_sum[idx] = encrypted_sum[idx] + chunk
            if he_metrics is not None:
                he_metrics.addition_time_sec += time.perf_counter() - addition_started_at

    if encrypted_sum is None:
        raise ValueError("Encrypted aggregation requires at least one HE-protected update")
    for idx, start in enumerate(range(0, he_size, chunk_size)):
        plain_chunk = plaintext_sum[start:start + chunk_size].tolist()
        addition_started_at = time.perf_counter()
        encrypted_sum[idx] = encrypted_sum[idx] + plain_chunk
        if he_metrics is not None:
            he_metrics.addition_time_sec += time.perf_counter() - addition_started_at
    aggregated: list[float] = []
    decryption_started_at = time.perf_counter()
    for chunk in encrypted_sum:
        aggregated.extend(_tenseal_decrypt(chunk, context, secret_key))
    if he_metrics is not None:
        he_metrics.decryption_time_sec += time.perf_counter() - decryption_started_at
    aggregated_array = plaintext_sum.copy()
    aggregated_array[:he_size] = np.asarray(aggregated[:he_size], dtype=np.float64)
    if he_metrics is not None:
        he_metrics.max_abs_error = max(
            he_metrics.max_abs_error,
            float(np.max(np.abs(aggregated_array - expected_sum))),
        )
    aggregated_tensor = torch.tensor(aggregated_array[:size], dtype=torch.float32, device=device)
    _apply_flat_update(aggregated_tensor, global_end, global_edge)
    return global_end, global_edge


@_measure_he_wall_time
def fedavg_split_seal(
    state_diffs: list[dict[str, dict[str, torch.Tensor]]],
    sample_counts: list[float],
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    device: torch.device,
    *,
    chunk_size: int = 4096,
    encrypted_mask: list[bool] | None = None,
    he_aggregation_size: int | None = None,
    he_workers: int = 1,
    he_metrics: HEOperationMetrics | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    total = max(1, sum(sample_counts))
    total_size = sum(
        int(param.numel())
        for model in (global_end, global_edge)
        for param in model.parameters()
    )
    he_size = _resolved_he_aggregation_size(total_size, he_aggregation_size)
    if he_size < total_size:
        mask = _validated_encrypted_mask(encrypted_mask, len(state_diffs))
        return _fedavg_split_bounded_he_streaming(
            state_diffs,
            sample_counts,
            global_end,
            global_edge,
            device,
            encrypted_mask=mask,
            he_backend="seal",
            he_aggregation_size=he_size,
            he_metrics=he_metrics,
        )
    mask = _validated_encrypted_mask(encrypted_mask, len(state_diffs))
    if int(he_workers) > 1 and any(mask):
        metrics_snapshot = copy.deepcopy(he_metrics)
        try:
            return _fedavg_split_seal_processes(
                state_diffs,
                sample_counts,
                global_end,
                global_edge,
                device,
                chunk_size=chunk_size,
                encrypted_mask=mask,
                he_workers=he_workers,
                he_metrics=he_metrics,
            )
        except (BrokenProcessPool, MemoryError, OSError) as exc:
            if he_metrics is not None and metrics_snapshot is not None:
                he_metrics.__dict__.update(metrics_snapshot.__dict__)
                he_metrics.process_fallbacks += 1
            warnings.warn(
                "Parallel SEAL aggregation failed; retrying the same full update "
                f"in one process. Cause: {exc}",
                RuntimeWarning,
                stacklevel=2,
            )
    work_device = torch.device("cpu")
    flat_updates = [
        _flatten_state_diff(diff, global_end, global_edge, work_device)
        for diff in state_diffs
    ]
    if not flat_updates:
        return global_end, global_edge
    mask = _validated_encrypted_mask(mask, len(flat_updates))

    runtime, key_setup_time_sec = _seal_ckks_runtime()
    encoder = runtime["encoder"]
    encryptor = runtime["encryptor"]
    decryptor = runtime["decryptor"]
    evaluator = runtime["evaluator"]
    scale = CKKS_SCALE

    size = int(flat_updates[0].numel())
    he_size = _resolved_he_aggregation_size(size, he_aggregation_size)
    if he_metrics is not None:
        he_metrics.key_setup_time_sec += key_setup_time_sec
        he_metrics.aggregation_calls += 1
        he_metrics.encrypted_updates += sum(mask)
        he_metrics.encrypted_parameter_values += he_size * sum(mask)
    if not any(mask):
        raise ValueError("Encrypted aggregation requires at least one HE-protected update")
    factors = [float(count) / float(total) for count in sample_counts]
    aggregated_array = np.empty(size, dtype=np.float32)
    for start in range(0, he_size, chunk_size):
        stop = min(start + chunk_size, he_size)
        encrypted_sum = None
        plaintext_sum = np.zeros(stop - start, dtype=np.float64)
        expected_chunk = np.zeros(stop - start, dtype=np.float64)
        for flat_update, factor, encrypted in zip(flat_updates, factors, mask):
            values = np.ascontiguousarray(
                flat_update[start:stop].numpy().astype(np.float64) * factor,
                dtype=np.float64,
            )
            expected_chunk += values
            if not encrypted:
                plaintext_sum += values
                continue
            encryption_started_at = time.perf_counter()
            ciphertext = encryptor.encrypt(encoder.encode(values, scale))
            if he_metrics is not None:
                he_metrics.encryption_time_sec += time.perf_counter() - encryption_started_at
                he_metrics.ciphertext_count += 1
                he_metrics.ciphertext_bytes += int(ciphertext.save_size())
            if encrypted_sum is None:
                encrypted_sum = ciphertext
            else:
                addition_started_at = time.perf_counter()
                encrypted_sum = evaluator.add(encrypted_sum, ciphertext)
                if he_metrics is not None:
                    he_metrics.addition_time_sec += time.perf_counter() - addition_started_at
        if encrypted_sum is None:
            raise ValueError("Encrypted aggregation requires at least one HE-protected update")
        addition_started_at = time.perf_counter()
        encrypted_sum = evaluator.add_plain(
            encrypted_sum,
            encoder.encode(np.ascontiguousarray(plaintext_sum), scale),
        )
        if he_metrics is not None:
            he_metrics.addition_time_sec += time.perf_counter() - addition_started_at
        decryption_started_at = time.perf_counter()
        decoded = decode_seal_vector(
            encoder,
            decryptor.decrypt(encrypted_sum),
        )[: stop - start]
        if he_metrics is not None:
            he_metrics.decryption_time_sec += time.perf_counter() - decryption_started_at
            he_metrics.max_abs_error = max(
                he_metrics.max_abs_error,
                float(np.max(np.abs(decoded - expected_chunk))),
            )
        aggregated_array[start:stop] = decoded
    aggregated_tensor = torch.from_numpy(aggregated_array)
    _apply_flat_update(aggregated_tensor, global_end, global_edge)
    return global_end, global_edge


def _fedavg_split_seal_processes(
    state_diffs: list[dict[str, dict[str, torch.Tensor]]],
    sample_counts: list[float],
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    device: torch.device,
    *,
    chunk_size: int,
    encrypted_mask: list[bool],
    he_workers: int,
    he_metrics: HEOperationMetrics | None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    update_count = len(state_diffs)
    parameter_count = sum(
        int(param.numel())
        for model in (global_end, global_edge)
        for param in model.parameters()
    )
    if update_count == 0:
        return global_end, global_edge

    total = max(1.0, float(sum(sample_counts)))
    factors = tuple(float(count) / total for count in sample_counts)
    mask = tuple(_validated_encrypted_mask(encrypted_mask, update_count))
    worker_count = max(1, min(int(he_workers), os.cpu_count() or 1))
    mapped_update_bytes = (
        update_count * parameter_count * np.dtype(np.float32).itemsize
    )
    runtime, key_setup_time_sec = _seal_ckks_runtime()
    aggregated_array = np.empty(parameter_count, dtype=np.float32)
    configured_temp_root = os.environ.get("DYNFL_HE_TMPDIR")
    he_temp_root = (
        Path(configured_temp_root).resolve()
        if configured_temp_root
        else ROOT / "out" / ".he_tmp"
    )
    he_temp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="dynfl_seal_process_",
        dir=he_temp_root,
    ) as temp_dir:
        free_bytes = shutil.disk_usage(temp_dir).free
        required_free_bytes = mapped_update_bytes + 512 * 1024 ** 2
        if free_bytes < required_free_bytes:
            raise OSError(
                "SEAL process aggregation needs a file-backed update store of "
                f"{mapped_update_bytes / (1024 ** 3):.2f} GiB, but the temporary "
                f"drive has only {free_bytes / (1024 ** 3):.2f} GiB free."
            )

        update_store_path = str(Path(temp_dir) / "weighted_updates.float32")
        updates = np.memmap(
            update_store_path,
            mode="w+",
            dtype=np.float32,
            shape=(update_count, parameter_count),
        )
        for row_index, state_diff in enumerate(state_diffs):
            flattened = _flatten_state_diff(
                state_diff,
                global_end,
                global_edge,
                torch.device("cpu"),
            )
            updates[row_index] = flattened.numpy()
        updates.flush()

        chunk_bounds = [
            (start, min(start + chunk_size, parameter_count))
            for start in range(0, parameter_count, chunk_size)
        ]
        chunks_per_task = max(
            1,
            min(
                32,
                int(np.ceil(len(chunk_bounds) / max(1, worker_count * 8))),
            ),
        )
        chunk_batches = [
            tuple(chunk_bounds[index:index + chunks_per_task])
            for index in range(0, len(chunk_bounds), chunks_per_task)
        ]
        public_key_path = str(Path(temp_dir) / "public_key.bin")
        secret_key_path = str(Path(temp_dir) / "secret_key.bin")
        runtime["public_key"].save(public_key_path)
        runtime["secret_key"].save(secret_key_path)
        try:
            with ProcessPoolExecutor(
                max_workers=worker_count,
                initializer=_init_seal_process_runtime,
                initargs=(
                    public_key_path,
                    secret_key_path,
                    update_store_path,
                    update_count,
                    parameter_count,
                    factors,
                    mask,
                ),
            ) as executor:
                results = executor.map(
                    _seal_process_aggregate_chunk_batch,
                    chunk_batches,
                    chunksize=1,
                )
                for batch_results in results:
                    for (
                        start,
                        decoded,
                        ciphertext_count,
                        ciphertext_bytes,
                        encryption_time,
                        addition_time,
                        decryption_time,
                        max_abs_error,
                    ) in batch_results:
                        stop = start + int(decoded.size)
                        aggregated_array[start:stop] = decoded
                        if he_metrics is not None:
                            he_metrics.ciphertext_count += ciphertext_count
                            he_metrics.ciphertext_bytes += ciphertext_bytes
                            he_metrics.encryption_time_sec += encryption_time
                            he_metrics.addition_time_sec += addition_time
                            he_metrics.decryption_time_sec += decryption_time
                            he_metrics.max_abs_error = max(
                                he_metrics.max_abs_error,
                                max_abs_error,
                            )
        finally:
            del updates

    if he_metrics is not None:
        he_metrics.key_setup_time_sec += key_setup_time_sec
        he_metrics.aggregation_calls += 1
        he_metrics.encrypted_updates += sum(mask)
        he_metrics.encrypted_parameter_values += parameter_count * sum(mask)
        he_metrics.process_workers = max(he_metrics.process_workers, worker_count)
        he_metrics.process_tasks += len(chunk_batches)
        he_metrics.process_chunks_per_task = max(
            he_metrics.process_chunks_per_task,
            chunks_per_task,
        )
        he_metrics.mapped_update_bytes = max(
            he_metrics.mapped_update_bytes,
            mapped_update_bytes,
        )
    aggregated_tensor = torch.from_numpy(aggregated_array)
    _apply_flat_update(aggregated_tensor, global_end, global_edge)
    return global_end, global_edge


def _validated_encrypted_mask(
    encrypted_mask: list[bool] | None,
    update_count: int,
) -> list[bool]:
    mask = [True] * update_count if encrypted_mask is None else list(encrypted_mask)
    if len(mask) != update_count:
        raise ValueError(
            "encrypted_mask length must match the number of state differences"
        )
    return mask


def _resolved_he_aggregation_size(total_size: int, he_aggregation_size: int | None) -> int:
    if he_aggregation_size is None or int(he_aggregation_size) <= 0:
        return int(total_size)
    value = int(he_aggregation_size)
    if value < int(total_size):
        raise ValueError(
            "Partial CKKS aggregation is disabled because it leaves part of the update in plaintext. "
            "Set he_aggregation_size to 0 to encrypt the complete update."
        )
    return int(total_size)


def _fedavg_split_bounded_he_streaming(
    state_diffs: list[dict[str, dict[str, torch.Tensor]]],
    sample_counts: list[float],
    global_end: torch.nn.Module,
    global_edge: torch.nn.Module,
    device: torch.device,
    *,
    encrypted_mask: list[bool],
    he_backend: str,
    he_aggregation_size: int,
    he_metrics: HEOperationMetrics | None = None,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    work_device = torch.device("cpu")
    total = max(1.0, float(sum(sample_counts)))
    aggregated: dict[str, dict[str, torch.Tensor]] = {"end": {}, "edge": {}}
    he_prefixes: list[torch.Tensor] = []
    he_size = _resolved_he_aggregation_size(
        sum(int(param.numel()) for model in (global_end, global_edge) for param in model.parameters()),
        he_aggregation_size,
    )
    for state_diff, count in zip(state_diffs, sample_counts):
        factor = float(count) / total
        prefix_parts: list[torch.Tensor] = []
        cursor = 0
        for part_name, model in (("end", global_end), ("edge", global_edge)):
            values = state_diff.get(part_name, {})
            for name, param in model.named_parameters():
                value = values.get(name)
                if value is None:
                    value = torch.zeros_like(param.data, device=work_device)
                relative = value.to(work_device)
                weighted = relative * factor
                if name in aggregated[part_name]:
                    aggregated[part_name][name] = aggregated[part_name][name] + weighted
                else:
                    aggregated[part_name][name] = weighted.clone()
                if cursor < he_size:
                    take = min(int(relative.numel()), he_size - cursor)
                    prefix_parts.append(relative.reshape(-1)[:take].detach().cpu())
                cursor += int(relative.numel())
        he_prefixes.append(torch.cat(prefix_parts) if prefix_parts else torch.zeros(0))

    flat_update = _flatten_state_diff(aggregated, global_end, global_edge, work_device)
    if he_backend == "seal":
        he_prefix = _fedavg_prefix_seal(
            he_prefixes,
            [int(count) for count in sample_counts],
            encrypted_mask=encrypted_mask,
            he_metrics=he_metrics,
        )
    else:
        he_prefix = _fedavg_prefix_tenseal(
            he_prefixes,
            [int(count) for count in sample_counts],
            encrypted_mask=encrypted_mask,
            he_metrics=he_metrics,
        )
    flat_update[:he_size] = he_prefix.to(work_device)[:he_size]
    _apply_flat_update(flat_update, global_end, global_edge)
    return global_end, global_edge


@_measure_he_wall_time
def _fedavg_prefix_seal(
    prefix_updates: list[torch.Tensor],
    sample_counts: list[int],
    *,
    encrypted_mask: list[bool],
    chunk_size: int = 4096,
    he_metrics: HEOperationMetrics | None = None,
) -> torch.Tensor:
    if not prefix_updates:
        return torch.zeros(0)
    encrypted_mask = _validated_encrypted_mask(encrypted_mask, len(prefix_updates))
    size = int(prefix_updates[0].numel())
    total = max(1.0, float(sum(sample_counts)))
    runtime, key_setup_time_sec = _seal_ckks_runtime()
    encoder = runtime["encoder"]
    encryptor = runtime["encryptor"]
    decryptor = runtime["decryptor"]
    evaluator = runtime["evaluator"]
    scale = CKKS_SCALE
    if he_metrics is not None:
        he_metrics.key_setup_time_sec += key_setup_time_sec
        he_metrics.aggregation_calls += 1
        he_metrics.encrypted_updates += sum(encrypted_mask)
        he_metrics.encrypted_parameter_values += size * sum(encrypted_mask)
    encrypted_sum: list[Any] | None = None
    plaintext_sum = np.zeros(size, dtype=np.float64)
    expected_sum = np.zeros(size, dtype=np.float64)
    for update, count, encrypted in zip(prefix_updates, sample_counts, encrypted_mask):
        weighted = update.detach().cpu().numpy().astype(np.float64) * (float(count) / total)
        expected_sum += weighted
        if not encrypted:
            plaintext_sum += weighted
            continue
        encryption_started_at = time.perf_counter()
        encrypted_chunks = [
            encryptor.encrypt(
                encoder.encode(
                    np.ascontiguousarray(weighted[start:start + chunk_size], dtype=np.float64),
                    scale,
                )
            )
            for start in range(0, size, chunk_size)
        ]
        if he_metrics is not None:
            he_metrics.encryption_time_sec += time.perf_counter() - encryption_started_at
            he_metrics.ciphertext_count += len(encrypted_chunks)
            he_metrics.ciphertext_bytes += sum(
                int(chunk.save_size()) for chunk in encrypted_chunks
            )
        if encrypted_sum is None:
            encrypted_sum = encrypted_chunks
        else:
            addition_started_at = time.perf_counter()
            encrypted_sum = [
                evaluator.add(left, right)
                for left, right in zip(encrypted_sum, encrypted_chunks)
            ]
            if he_metrics is not None:
                he_metrics.addition_time_sec += time.perf_counter() - addition_started_at
    if encrypted_sum is None:
        return torch.tensor(plaintext_sum, dtype=torch.float32)
    for idx, start in enumerate(range(0, size, chunk_size)):
        addition_started_at = time.perf_counter()
        encrypted_sum[idx] = evaluator.add_plain(
            encrypted_sum[idx],
            encoder.encode(
                np.ascontiguousarray(
                    plaintext_sum[start:start + chunk_size],
                    dtype=np.float64,
                ),
                scale,
            ),
        )
        if he_metrics is not None:
            he_metrics.addition_time_sec += time.perf_counter() - addition_started_at
    decryption_started_at = time.perf_counter()
    decoded_parts = [
        decode_seal_vector(encoder, decryptor.decrypt(chunk))[: min(chunk_size, size - start)]
        for start, chunk in zip(range(0, size, chunk_size), encrypted_sum)
    ]
    decoded = np.concatenate(decoded_parts) if decoded_parts else np.zeros(0)
    if he_metrics is not None:
        he_metrics.decryption_time_sec += time.perf_counter() - decryption_started_at
        he_metrics.max_abs_error = max(
            he_metrics.max_abs_error,
            float(np.max(np.abs(decoded - expected_sum))),
        )
    return torch.tensor(decoded, dtype=torch.float32)


@_measure_he_wall_time
def _fedavg_prefix_tenseal(
    prefix_updates: list[torch.Tensor],
    sample_counts: list[int],
    *,
    encrypted_mask: list[bool],
    chunk_size: int = 4096,
    he_metrics: HEOperationMetrics | None = None,
) -> torch.Tensor:
    if not prefix_updates:
        return torch.zeros(0)
    encrypted_mask = _validated_encrypted_mask(encrypted_mask, len(prefix_updates))
    size = int(prefix_updates[0].numel())
    total = max(1.0, float(sum(sample_counts)))
    runtime, key_setup_time_sec = _tenseal_ckks_runtime()
    context = runtime["context"]
    secret_key = runtime["secret_key"]
    public_context = runtime["public_context"]
    if he_metrics is not None:
        he_metrics.key_setup_time_sec += key_setup_time_sec
        he_metrics.aggregation_calls += 1
        he_metrics.encrypted_updates += sum(encrypted_mask)
        he_metrics.encrypted_parameter_values += size * sum(encrypted_mask)
    encrypted_sum: list[Any] | None = None
    plaintext_sum = np.zeros(size, dtype=np.float64)
    expected_sum = np.zeros(size, dtype=np.float64)
    for update, count, encrypted in zip(prefix_updates, sample_counts, encrypted_mask):
        weighted = update.detach().cpu().numpy().astype(np.float64) * (float(count) / total)
        expected_sum += weighted
        if not encrypted:
            plaintext_sum += weighted
            continue
        encryption_started_at = time.perf_counter()
        encrypted_chunks = [
            ts.ckks_vector(public_context, weighted[start:start + chunk_size].tolist())
            for start in range(0, size, chunk_size)
        ]
        if he_metrics is not None:
            he_metrics.encryption_time_sec += time.perf_counter() - encryption_started_at
            he_metrics.ciphertext_count += len(encrypted_chunks)
            he_metrics.ciphertext_bytes += sum(
                len(chunk.serialize()) for chunk in encrypted_chunks
            )
        if encrypted_sum is None:
            encrypted_sum = encrypted_chunks
        else:
            addition_started_at = time.perf_counter()
            encrypted_sum = [
                left + right for left, right in zip(encrypted_sum, encrypted_chunks)
            ]
            if he_metrics is not None:
                he_metrics.addition_time_sec += time.perf_counter() - addition_started_at
    if encrypted_sum is None:
        return torch.tensor(plaintext_sum, dtype=torch.float32)
    for idx, start in enumerate(range(0, size, chunk_size)):
        addition_started_at = time.perf_counter()
        encrypted_sum[idx] = (
            encrypted_sum[idx] + plaintext_sum[start:start + chunk_size].tolist()
        )
        if he_metrics is not None:
            he_metrics.addition_time_sec += time.perf_counter() - addition_started_at
    decryption_started_at = time.perf_counter()
    decoded = np.concatenate(
        [
            np.asarray(_tenseal_decrypt(chunk, context, secret_key), dtype=np.float64)[
                : min(chunk_size, size - start)
            ]
            for start, chunk in zip(range(0, size, chunk_size), encrypted_sum)
        ]
    )
    if he_metrics is not None:
        he_metrics.decryption_time_sec += time.perf_counter() - decryption_started_at
        he_metrics.max_abs_error = max(
            he_metrics.max_abs_error,
            float(np.max(np.abs(decoded - expected_sum))),
        )
    return torch.tensor(decoded, dtype=torch.float32)


def _tenseal_decrypt(vector: Any, private_context: Any, secret_key: Any) -> list[float]:
    """Decrypt a TenSEAL vector across minor API differences."""
    try:
        return vector.decrypt(secret_key)
    except (TypeError, ValueError):
        pass
    if hasattr(vector, "link_context"):
        vector.link_context(private_context)
    return vector.decrypt()


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
            if param.requires_grad:
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
        "accounted_system_time_sec": final["logical_time"]
        + sum(row.get("selection_wall_time_sec", 0.0) for row in round_rows),
        "total_round_wall_time_sec": sum(row.get("round_wall_time_sec", 0.0) for row in round_rows),
        "total_selection_wall_time_sec": sum(row.get("selection_wall_time_sec", 0.0) for row in round_rows),
        "total_training_wall_time_sec": sum(row.get("training_wall_time_sec", 0.0) for row in round_rows),
        "total_flow_wall_time_sec": sum(row.get("flow_wall_time_sec", 0.0) for row in round_rows),
        "total_aggregation_wall_time_sec": sum(row.get("aggregation_wall_time_sec", 0.0) for row in round_rows),
        "total_evaluation_wall_time_sec": sum(row.get("evaluation_wall_time_sec", 0.0) for row in round_rows),
        "total_accounted_phase_wall_time_sec": sum(row.get("accounted_phase_wall_time_sec", 0.0) for row in round_rows),
        "total_unattributed_wall_time_sec": sum(row.get("unattributed_wall_time_sec", 0.0) for row in round_rows),
        "mean_selection_wall_time_sec": _list_mean(row.get("selection_wall_time_sec", 0.0) for row in round_rows),
        "mean_training_wall_time_sec": _list_mean(row.get("training_wall_time_sec", 0.0) for row in round_rows),
        "mean_round_wall_time_sec": _list_mean(row.get("round_wall_time_sec", 0.0) for row in round_rows),
        "total_communication_volume": sum(row["communication_volume"] for row in round_rows),
        "total_waiting_time": sum(row["waiting_time"] for row in round_rows),
        "total_edge_aggregation_time": sum(row["edge_aggregation_time"] for row in round_rows),
        "total_cloud_aggregation_time": sum(row["cloud_aggregation_time"] for row in round_rows),
        "mean_effective_clients": _list_mean(row["num_effective_clients"] for row in round_rows),
        "mean_global_update_clients": _list_mean(row["num_global_update_clients"] for row in round_rows),
        "mean_effective_edges": _list_mean(row["num_effective_edges"] for row in round_rows),
        "max_privacy_risk": max(row["max_risk"] for row in round_rows),
        "min_remaining_epsilon": final["min_remaining_epsilon"],
        "max_feature_epsilon": float(final.get("max_feature_epsilon", 0.0)),
        "max_update_epsilon": float(final.get("max_update_epsilon", 0.0)),
        "larger_channel_epsilon": max(
            float(final.get("max_feature_epsilon", 0.0)),
            float(final.get("max_update_epsilon", 0.0)),
        ),
        "privacy_execution_audit": privacy_execution_audit(round_rows),
        "privacy_guarantee": {
            "split_execution": "trusted end-to-edge execution domain",
            "protected_object": "cross-domain model update packet",
            "dp": "accounted DP events; complete transcript guarantee not established",
            "he": "check he_execution_status; profiled execution is plaintext",
            "uniform_end_to_end_update_dp": None,
            "combined_epsilon": None,
        },
        "feasible_rate": _list_mean(float(row["feasible_resource"]) for row in decision_rows),
        "all_constraint_feasible_rate": _list_mean(float(row["feasible"]) for row in decision_rows),
        "resource_feasible_rate": _list_mean(float(row["feasible_resource"]) for row in decision_rows),
        "memory_feasible_rate": _list_mean(float(row["feasible_memory"]) for row in decision_rows),
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


def _runtime_environment(device: torch.device) -> dict[str, Any]:
    package_names = (
        "numpy",
        "scipy",
        "torch",
        "torchvision",
        "matplotlib",
        "seal",
        "tenseal",
    )
    package_versions: dict[str, str | None] = {}
    for name in package_names:
        try:
            package_versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            package_versions[name] = None

    cuda_available = torch.cuda.is_available()
    cuda_device_name = None
    if cuda_available:
        cuda_device_name = torch.cuda.get_device_name(device if device.type == "cuda" else 0)
    git_commit = None
    git_dirty = None
    try:
        git_commit_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        git_status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
        git_commit = git_commit_result.stdout.strip() or None
        git_dirty = bool(git_status_result.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass
    return {
        "python_version": sys.version,
        "platform": platform.platform(),
        "processor": platform.processor() or os.environ.get("PROCESSOR_IDENTIFIER", ""),
        "torch_num_threads": torch.get_num_threads(),
        "requested_device": str(device),
        "cuda_available": cuda_available,
        "cuda_version": torch.version.cuda,
        "cuda_device_name": cuda_device_name,
        "git_commit": git_commit,
        "git_dirty": git_dirty,
        "package_versions": package_versions,
    }


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
