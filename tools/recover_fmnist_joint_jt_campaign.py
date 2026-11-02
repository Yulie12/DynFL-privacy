"""Fail-closed recovery orchestrator for the preregistered FMNIST J/T campaign.

The default mode is a read-only dry run.  Execution additionally requires the
exact plan token printed by that dry run, holds a campaign-wide kernel lock,
and revalidates process, disk, source, input, and destination preconditions at
every action boundary.  This file intentionally lives outside ``dynfed/*.py`` and
``experiments/*.py`` because those files form the preregistered scientific
source identity.

The orchestrator never calls the campaign's top-level ``main`` function.  In
particular, it cannot re-enter an already committed shared prefix and trigger
the round-5 stop-condition overrun.  Existing results are never deleted,
moved, or overwritten.  Source-identity corrections are written below a new
``_recovery/strict_source_<hash>`` subtree and recorded in a manifest.
"""
from __future__ import annotations

import argparse
import ctypes
from ctypes import wintypes
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys
import traceback
from typing import Any
import uuid

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dynfed.paper_config import PaperConfig, effective_config
from dynfed.paper_storage import (
    atomic_json,
    digest,
    exclusive_run,
    file_digest,
    load_checkpoint,
    source_identity,
    tensor_digest,
)
import dynfed.paper_runner as runner
import experiments.run_fmnist_joint_jt_100r as campaign


DEFAULT_CONFIG = ROOT / "configs" / "fmnist_joint_jt_100r_3seed_20261022.json"
DEFAULT_OUTPUT = ROOT / "out" / "fmnist_joint_jt_100r_3seed_20261022"
MIN_FREE_GIB = 80.0
AUTHOR_ACCEPTED_SOURCE_EXCEPTIONS = {
    (43, "fixed_P2"), (43, "fixed_P3"), (43, "fixed_P4"),
}
AUTHOR_ACCEPTED_ADVISORY_SIDECAR_EXCEPTIONS = {
    (42, "greedy_lambda025"), (42, "greedy_lambda050"),
    (42, "greedy_lambda075"), (42, "hysteresis_lambda050"),
    (42, "periodic_lambda050"), (42, "max_J"), (42, "min_T"),
}
EXECUTION_ENABLED = True


class RecoveryError(RuntimeError):
    """A fail-closed recovery precondition was not satisfied."""


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _checkpoint_round(path: Path) -> int:
    return int(path.stem.rsplit("_", 1)[1])


def _active_python_processes() -> list[dict[str, Any]]:
    """Return other console Python processes using the Toolhelp snapshot API.

    The formal trainer uses python.exe.  The dashboard uses pythonw.exe and is
    deliberately not treated as a training process.  Failing closed on any
    other python.exe avoids depending on command-line inspection privileges.
    Toolhelp is used because ``tasklist`` and WMI can be denied in restricted
    Windows sessions even though read-only process enumeration is available.
    """
    if os.name != "nt":
        return []

    class PROCESSENTRY32W(ctypes.Structure):
        _fields_ = [
            ("dwSize", wintypes.DWORD),
            ("cntUsage", wintypes.DWORD),
            ("th32ProcessID", wintypes.DWORD),
            ("th32DefaultHeapID", ctypes.c_size_t),
            ("th32ModuleID", wintypes.DWORD),
            ("cntThreads", wintypes.DWORD),
            ("th32ParentProcessID", wintypes.DWORD),
            ("pcPriClassBase", wintypes.LONG),
            ("dwFlags", wintypes.DWORD),
            ("szExeFile", wintypes.WCHAR * 260),
        ]

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_snapshot = kernel32.CreateToolhelp32Snapshot
    create_snapshot.argtypes = [wintypes.DWORD, wintypes.DWORD]
    create_snapshot.restype = wintypes.HANDLE
    process_first = kernel32.Process32FirstW
    process_first.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    process_first.restype = wintypes.BOOL
    process_next = kernel32.Process32NextW
    process_next.argtypes = [wintypes.HANDLE, ctypes.POINTER(PROCESSENTRY32W)]
    process_next.restype = wintypes.BOOL
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    close_handle.restype = wintypes.BOOL

    snapshot = create_snapshot(0x00000002, 0)  # TH32CS_SNAPPROCESS
    invalid_handle = ctypes.c_void_p(-1).value
    if snapshot == invalid_handle:
        raise RecoveryError(
            f"cannot enumerate Python processes: Windows error {ctypes.get_last_error()}"
        )
    processes: list[dict[str, Any]] = []
    entry = PROCESSENTRY32W()
    entry.dwSize = ctypes.sizeof(PROCESSENTRY32W)
    try:
        success = bool(process_first(snapshot, ctypes.byref(entry)))
        if not success:
            error = ctypes.get_last_error()
            if error != 18:  # ERROR_NO_MORE_FILES
                raise RecoveryError(f"cannot enumerate Python processes: Windows error {error}")
        while success:
            name = entry.szExeFile
            pid = int(entry.th32ProcessID)
            if name.casefold() == "python.exe" and pid != os.getpid():
                processes.append({"name": name, "pid": pid})
            success = bool(process_next(snapshot, ctypes.byref(entry)))
        terminal_error = ctypes.get_last_error()
        if terminal_error not in (0, 18):
            raise RecoveryError(
                f"process enumeration ended unexpectedly: Windows error {terminal_error}"
            )
    finally:
        close_handle(snapshot)
    return sorted(processes, key=lambda item: item["pid"])


def _expected_identity(
    config: PaperConfig,
    seed: int,
    data: tuple[torch.Tensor, ...],
    partition: list[list[int]],
    traces: list[list[dict[str, Any]]],
) -> dict[str, Any]:
    """Reproduce paper_runner._run_policy's identity without writing files."""
    effective = effective_config(config, "fixed_mixed", seed)
    runner.configure_runtime(effective)
    return {
        "config_hash": digest(effective.to_dict()),
        "source_hash": source_identity()["source_hash"],
        "partition_hash": digest(partition),
        "traces_hash": digest(traces),
        "data_hashes": [tensor_digest(value) for value in data],
        "runtime": {
            "python": sys.executable,
            "python_version": sys.version,
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version(),
            "threads": torch.get_num_threads(),
            "deterministic": torch.are_deterministic_algorithms_enabled(),
            "cublas_workspace": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
            "gpu": torch.cuda.get_device_name(0) if effective.device == "cuda" else None,
        },
    }


def _identity_without_source(identity: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in identity.items() if key != "source_hash"}


def _checkpoint_audit(
    destination: Path,
    checkpoint: Path,
    saved_identity: dict[str, Any],
    config: PaperConfig,
) -> dict[str, Any]:
    sidecar = checkpoint.with_suffix(".sha256")
    if not sidecar.is_file():
        raise RecoveryError(f"missing checkpoint sidecar: {sidecar}")
    actual_sha = file_digest(checkpoint)
    if sidecar.read_text(encoding="ascii").strip() != actual_sha:
        raise RecoveryError(f"checkpoint SHA mismatch: {checkpoint}")
    state = load_checkpoint(checkpoint)
    number = _checkpoint_round(checkpoint)
    if state.get("run_id") != digest(saved_identity):
        raise RecoveryError(f"checkpoint run_id mismatch: {checkpoint}")
    if state.get("next_round") != number:
        raise RecoveryError(f"checkpoint round mismatch: {checkpoint}")
    rows = state.get("rows", [])
    if len(rows) != number or (rows and int(rows[-1]["round"]) != number):
        raise RecoveryError(f"checkpoint metric history mismatch: {checkpoint}")
    if rows:
        last = rows[-1]
        if int(last.get("acknowledged_clients", -1)) != config.clients:
            raise RecoveryError(f"checkpoint ACK count mismatch: {checkpoint}")
        if int(last.get("pending_units", -1)) != 0:
            raise RecoveryError(f"checkpoint has pending units: {checkpoint}")
        if int(last.get("cloud_version", -1)) != number:
            raise RecoveryError(f"checkpoint Cloud version mismatch: {checkpoint}")
    barrier = state.get("barrier")
    expected_clients = set(range(config.clients))
    if barrier is None or int(getattr(barrier, "version", -1)) != number:
        raise RecoveryError(f"checkpoint barrier version mismatch: {checkpoint}")
    if set(getattr(barrier, "acks", set())) != expected_clients:
        raise RecoveryError(f"checkpoint barrier ACK state mismatch: {checkpoint}")
    client_versions = getattr(barrier, "client_versions", {})
    if set(client_versions) != expected_clients:
        raise RecoveryError(f"checkpoint client version membership mismatch: {checkpoint}")
    if any(int(value) != number for value in client_versions.values()):
        raise RecoveryError(f"checkpoint client version mismatch: {checkpoint}")
    if len(state.get("decisions", [])) != number * config.clients:
        raise RecoveryError(f"checkpoint decision history length mismatch: {checkpoint}")
    ledgers = state.get("ledgers", {})
    if set(ledgers) != expected_clients:
        raise RecoveryError(f"checkpoint privacy ledger membership mismatch: {checkpoint}")
    max_epsilon = max((ledger.epsilon() for ledger in ledgers.values()), default=0.0)
    if max_epsilon > config.epsilon + 1e-8:
        raise RecoveryError(f"privacy budget exceeded in {checkpoint}: {max_epsilon}")
    return {
        "round": number,
        "sha256": actual_sha,
        "rows": len(rows),
        "decisions": len(state.get("decisions", [])),
        "cloud_version": getattr(state.get("barrier"), "version", None),
        "max_epsilon": max_epsilon,
        "state": state,
    }


def _privacy_state_matches(destination: Path, checkpoint_state: dict[str, Any]) -> bool:
    path = destination / "privacy_state.json"
    if not path.is_file():
        return False
    saved = _read_json(path)
    ledgers = checkpoint_state.get("ledgers", {})
    if set(saved) != {str(client) for client in ledgers}:
        return False
    for client, ledger in ledgers.items():
        row = saved[str(client)]
        if int(row["events"]) != int(ledger.events):
            return False
        if abs(float(row["epsilon"]) - float(ledger.epsilon())) > 1e-10:
            return False
        if len(row.get("rdp", [])) != len(ledger.rdp):
            return False
        if any(abs(float(left) - float(right)) > 1e-12
               for left, right in zip(row["rdp"], ledger.rdp)):
            return False
    return True


def _has_uncommitted_round_files(destination: Path, next_round: int) -> list[str]:
    offenders = []
    for path in destination.glob("block_r*_*.pt"):
        try:
            index = int(path.name.split("_", 2)[1].removeprefix("r"))
        except (IndexError, ValueError):
            offenders.append(path.name)
        else:
            if index >= next_round:
                offenders.append(path.name)
    for path in destination.glob("plan_round_*.pt"):
        try:
            index = int(path.stem.rsplit("_", 1)[1])
        except ValueError:
            offenders.append(path.name)
        else:
            if index >= next_round:
                offenders.append(path.name)
    offenders.extend(path.name for path in destination.glob("*.tmp"))
    if (destination / "unsafe_privacy_state.json").exists():
        offenders.append("unsafe_privacy_state.json")
    return sorted(set(offenders))


def _inspect_run(
    destination: Path,
    expected_identity: dict[str, Any],
    config: PaperConfig,
    strategy: str | None = None,
    tolerate_partial_contamination: bool = False,
) -> dict[str, Any]:
    if not destination.exists():
        return {"state": "absent", "path": str(destination)}
    identity_path = destination / "run_identity.json"
    if not identity_path.is_file():
        raise RecoveryError(f"run_identity.json missing: {destination}")
    saved_identity = _read_json(identity_path)
    exact_identity = saved_identity == expected_identity
    non_source_match = _identity_without_source(saved_identity) == _identity_without_source(expected_identity)
    if not exact_identity and not non_source_match:
        raise RecoveryError(f"non-source identity mismatch: {destination}")
    checkpoints = sorted(destination.glob("checkpoint_round_*.pt"), key=_checkpoint_round)
    if not checkpoints:
        raise RecoveryError(f"no committed checkpoint: {destination}")
    checkpoint_numbers = [_checkpoint_round(path) for path in checkpoints]
    if checkpoint_numbers != list(range(1, checkpoint_numbers[-1] + 1)):
        raise RecoveryError(f"non-contiguous checkpoints: {destination}")
    missing_sidecars = []
    for checkpoint in checkpoints:
        sidecar = checkpoint.with_suffix(".sha256")
        actual = file_digest(checkpoint)
        if not sidecar.is_file():
            missing_sidecars.append({
                "checkpoint": str(checkpoint), "sha256": actual,
                "sidecar": str(sidecar),
            })
        elif sidecar.read_text(encoding="ascii").strip() != actual:
            raise RecoveryError(f"checkpoint SHA mismatch: {checkpoint}")
    latest = _checkpoint_audit(destination, checkpoints[-1], saved_identity, config)
    metrics_path = destination / "round_metrics.csv"
    if not metrics_path.is_file():
        raise RecoveryError(f"round_metrics.csv missing: {destination}")
    metrics = _csv_rows(metrics_path)
    expected_rounds = list(range(1, latest["round"] + 1))
    actual_rounds = [int(row["round"]) for row in metrics]
    if actual_rounds != expected_rounds:
        raise RecoveryError(f"non-contiguous round metrics: {destination}")
    if len(metrics) != latest["rows"]:
        raise RecoveryError(f"CSV/checkpoint row mismatch: {destination}")
    summary = destination / "summary.json"
    status_path = destination / "run_status.json"
    status = _read_json(status_path) if status_path.is_file() else {}
    offenders = _has_uncommitted_round_files(destination, latest["round"])
    privacy_matches = _privacy_state_matches(destination, latest["state"])
    if summary.is_file():
        if latest["round"] != config.rounds:
            raise RecoveryError(f"premature summary: {destination}")
        if status.get("status") != "completed":
            raise RecoveryError(f"completed run has non-completed status: {destination}")
        if offenders:
            raise RecoveryError(f"completed run has uncommitted files: {destination}: {offenders[:3]}")
        summary_value = _read_json(summary)
        if (int(summary_value.get("rounds", -1)) != config.rounds
                or summary_value.get("run_id") != digest(saved_identity)):
            raise RecoveryError(f"completed summary identity mismatch: {destination}")
        if not (destination / "final_state.pt").is_file():
            raise RecoveryError(f"completed run has no final_state.pt: {destination}")
        if len(list(destination.glob("candidates_round_*.json"))) != config.rounds:
            raise RecoveryError(f"completed candidate reports are incomplete: {destination}")
        if len(list(destination.glob("timeline_round_*.json"))) != config.rounds:
            raise RecoveryError(f"completed timeline reports are incomplete: {destination}")
        decisions_path = destination / "client_decisions.csv"
        if not decisions_path.is_file() or len(_csv_rows(decisions_path)) != config.rounds * config.clients:
            raise RecoveryError(f"completed client decisions are incomplete: {destination}")
        if strategy is not None:
            selected = [
                _read_json(destination / f"candidates_round_{index:03d}.json")["selected_candidate"]
                for index in range(config.rounds)
            ]
            spec = campaign.SPECS[strategy]
            if spec["kind"] == "fixed" and any(name != spec["candidate"] for name in selected):
                raise RecoveryError(f"fixed strategy selection mismatch: {destination}")
            if spec["kind"] == "pure" and any(name != f"pure_{spec['mode']}" for name in selected):
                raise RecoveryError(f"pure strategy selection mismatch: {destination}")
            if strategy in campaign.DYNAMIC and any(name != "P0" for name in selected[:5]):
                raise RecoveryError(f"dynamic strategy prefix mismatch: {destination}")
        if not exact_identity:
            state = "completed_source_mismatch"
        elif missing_sidecars:
            state = "completed_integrity_deficient"
        else:
            state = "completed_exact"
    else:
        if not exact_identity:
            raise RecoveryError(f"partial run source identity mismatch: {destination}")
        if missing_sidecars and not tolerate_partial_contamination:
            raise RecoveryError(f"partial run has missing checkpoint sidecars: {destination}")
        if offenders and not tolerate_partial_contamination:
            raise RecoveryError(f"partial run has uncommitted files: {destination}: {offenders[:3]}")
        if not privacy_matches and not tolerate_partial_contamination:
            raise RecoveryError(f"partial run privacy state differs from checkpoint: {destination}")
        state = (
            "partial_contaminated"
            if offenders or not privacy_matches or missing_sidecars
            else "partial"
        )
    latest.pop("state")
    artifact_hashes = {
        "run_identity_sha256": file_digest(identity_path),
        "round_metrics_sha256": file_digest(metrics_path),
        "checkpoint_manifest_hash": digest([
            {"round": _checkpoint_round(path), "sha256": file_digest(path)} for path in checkpoints
        ]),
        "summary_sha256": file_digest(summary) if summary.is_file() else None,
        "final_state_sha256": (
            file_digest(destination / "final_state.pt")
            if (destination / "final_state.pt").is_file() else None
        ),
    }
    return {
        "state": state,
        "path": str(destination),
        "saved_source_hash": saved_identity.get("source_hash"),
        "expected_source_hash": expected_identity.get("source_hash"),
        "run_id": digest(saved_identity),
        "latest_checkpoint": latest,
        "privacy_state_matches": privacy_matches,
        "uncommitted_files": offenders,
        "missing_sidecars": missing_sidecars,
        "artifact_hashes": artifact_hashes,
        "audit_status": status.get("audit_status"),
    }


def _inspect_prefix(
    destination: Path,
    expected_identity: dict[str, Any],
    config: PaperConfig,
) -> dict[str, Any]:
    if not destination.exists():
        return {"state": "absent", "path": str(destination)}
    run = _inspect_run(
        destination, expected_identity, config,
        tolerate_partial_contamination=True,
    )
    if run["state"] == "completed_exact":
        return {**run, "state": "contaminated", "reason": "prefix_has_summary"}
    if run["state"] not in {"partial", "partial_contaminated"}:
        return {**run, "state": "contaminated", "reason": run["state"]}
    latest_round = int(run["latest_checkpoint"]["round"])
    checkpoints = sorted(destination.glob("checkpoint_round_*.pt"), key=_checkpoint_round)
    checkpoint_five = destination / "checkpoint_round_005.pt"
    if (run["state"] == "partial" and latest_round == 5 and len(checkpoints) == 5
            and checkpoint_five.is_file()):
        return {**run, "state": "pristine_round5"}
    reason = "uncommitted_state" if run["state"] == "partial_contaminated" else f"latest_round_{latest_round}"
    return {**run, "state": "contaminated", "reason": reason}


def _validate_seed_inputs(
    output: Path,
    config: PaperConfig,
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], list[list[int]], list[list[dict[str, Any]]], dict[str, Any]]:
    data, partition, traces = campaign.inputs(config, seed)
    seed_dir = output / f"seed_{seed}"
    partition_path, traces_path = seed_dir / "partition.json", seed_dir / "traces.json"
    if partition_path.exists() and digest(_read_json(partition_path)) != digest(partition):
        raise RecoveryError(f"saved partition mismatch for seed {seed}")
    if traces_path.exists() and digest(_read_json(traces_path)) != digest(traces):
        raise RecoveryError(f"saved traces mismatch for seed {seed}")
    identity = _expected_identity(config, seed, data, partition, traces)
    return data, partition, traces, identity


def build_plan(config_path: Path, output: Path) -> dict[str, Any]:
    config_path, output = config_path.resolve(), output.resolve()
    registration_path = output / "preregistration.json"
    if not registration_path.is_file():
        raise RecoveryError(f"missing preregistration: {registration_path}")
    registration = _read_json(registration_path)
    config = PaperConfig.load(config_path)
    if (config.clients, config.edges, config.rounds, tuple(config.seeds)) != (100, 10, 100, (42, 43, 44)):
        raise RecoveryError("unexpected campaign dimensions")
    if digest(config.to_dict()) != registration.get("config_hash"):
        raise RecoveryError("configuration hash differs from preregistration")
    current_source = source_identity()
    registered_source = registration.get("source_identity", {})
    if current_source.get("source_hash") != registered_source.get("source_hash"):
        raise RecoveryError("current source hash differs from preregistration")
    if current_source.get("sha256") != registered_source.get("sha256"):
        raise RecoveryError("current source manifest differs from preregistration")
    active = _active_python_processes()
    if active:
        raise RecoveryError(f"other python.exe processes are active: {active}")
    free_gib = shutil.disk_usage(output).free / (1024 ** 3)
    if free_gib < MIN_FREE_GIB:
        raise RecoveryError(f"free disk space {free_gib:.2f} GiB is below {MIN_FREE_GIB:.0f} GiB")

    source_tag = current_source["source_hash"][:12]
    recovery_root = output / "_recovery" / f"strict_source_{source_tag}"
    inspections: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    certified, source_mismatch, integrity_deficient = [], [], []
    accepted_source_exceptions, accepted_sidecar_exceptions = [], []
    forensic_prefixes: list[dict[str, Any]] = []
    contexts: dict[int, dict[str, Any]] = {}

    for seed in config.seeds:
        data, partition, traces, expected_identity = _validate_seed_inputs(output, config, seed)
        contexts[seed] = {
            "data": data,
            "partition": partition,
            "traces": traces,
            "identity": expected_identity,
        }
        seed_dir = output / f"seed_{seed}"
        strategy_states: dict[str, dict[str, Any]] = {}
        for strategy in campaign.SPECS:
            run = _inspect_run(seed_dir / strategy, expected_identity, config, strategy=strategy)
            run.update({"seed": seed, "strategy": strategy, "kind": "strategy"})
            inspections.append(run)
            strategy_states[strategy] = run
            if run["state"] == "completed_exact":
                certified.append([seed, strategy])
            elif run["state"] == "completed_source_mismatch":
                source_mismatch.append([seed, strategy])
                if (seed, strategy) in AUTHOR_ACCEPTED_SOURCE_EXCEPTIONS:
                    accepted_source_exceptions.append([seed, strategy])
            elif run["state"] == "completed_integrity_deficient":
                integrity_deficient.append([seed, strategy])
                if (seed, strategy) in AUTHOR_ACCEPTED_ADVISORY_SIDECAR_EXCEPTIONS:
                    accepted_sidecar_exceptions.append([seed, strategy])

        missing_dynamic = [
            strategy for strategy in campaign.DYNAMIC
            if strategy_states[strategy]["state"] == "absent"
        ]
        prefix = _inspect_prefix(seed_dir / "shared_P0_prefix", expected_identity, config)
        prefix.update({"seed": seed, "strategy": "shared_P0_prefix", "kind": "prefix"})
        inspections.append(prefix)
        for extra in sorted(seed_dir.glob("shared_P0_prefix*")):
            if extra == seed_dir / "shared_P0_prefix" or not extra.is_dir():
                continue
            extra_state = _inspect_prefix(extra, expected_identity, config)
            forensic_prefixes.append({"seed": seed, "path": str(extra), **extra_state})
        if missing_dynamic:
            if prefix["state"] == "absent":
                actions.append({
                    "action": "create_prefix_round5", "seed": seed,
                    "strategy": "shared_P0_prefix", "destination": str(seed_dir / "shared_P0_prefix"),
                })
            elif prefix["state"] != "pristine_round5":
                raise RecoveryError(
                    f"seed {seed} needs dynamic branches but prefix is {prefix['state']}: {prefix['path']}"
                )
        checkpoint_five = seed_dir / "shared_P0_prefix" / "checkpoint_round_005.pt"
        if checkpoint_five.is_file():
            prefix_five_sha = file_digest(checkpoint_five)
            for strategy in campaign.DYNAMIC:
                branch_five = seed_dir / strategy / "checkpoint_round_005.pt"
                if branch_five.is_file() and file_digest(branch_five) != prefix_five_sha:
                    raise RecoveryError(
                        f"dynamic branch does not share seed {seed} round-5 prefix: {strategy}"
                    )

        for strategy in campaign.SPECS:
            run = strategy_states[strategy]
            if run["state"] == "completed_exact":
                continue
            if run["state"] == "completed_integrity_deficient":
                if (seed, strategy) not in AUTHOR_ACCEPTED_ADVISORY_SIDECAR_EXCEPTIONS:
                    raise RecoveryError(
                        f"unapproved completed sidecar deficiency: seed {seed} {strategy}"
                    )
                continue
            if run["state"] == "completed_source_mismatch":
                if (seed, strategy) not in AUTHOR_ACCEPTED_SOURCE_EXCEPTIONS:
                    raise RecoveryError(
                        f"unapproved completed source mismatch: seed {seed} {strategy}"
                    )
                continue
            if run["state"] == "partial":
                actions.append({
                    "action": "resume_strategy", "seed": seed, "strategy": strategy,
                    "destination": run["path"], "from_round": run["latest_checkpoint"]["round"],
                })
            elif run["state"] == "absent":
                action = "fork_prefix_and_start" if strategy in campaign.DYNAMIC else "start_strategy"
                actions.append({
                    "action": action, "seed": seed, "strategy": strategy,
                    "destination": str(seed_dir / strategy),
                    "prefix": str(seed_dir / "shared_P0_prefix") if strategy in campaign.DYNAMIC else None,
                })
            else:
                raise RecoveryError(f"unsupported state {run['state']}: seed {seed} {strategy}")

    public_plan = {
        "schema": "fmnist.joint_jt.safe_recovery.v1",
        "mode": "dry_run",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "output": str(output),
        "registered_source_hash": current_source["source_hash"],
        "free_disk_gib": free_gib,
        "active_training_processes": active,
        "structurally_complete_pending_audit": certified,
        "completed_source_mismatch": source_mismatch,
        "completed_integrity_deficient": integrity_deficient,
        "author_accepted_source_exceptions": accepted_source_exceptions,
        "author_accepted_advisory_sidecar_exceptions": accepted_sidecar_exceptions,
        "forensic_prefixes": forensic_prefixes,
        "recovery_root": str(recovery_root),
        "actions": [{"order": index + 1, **action} for index, action in enumerate(actions)],
        "inspections": inspections,
        "summary": {
            "structurally_complete_pending_audit": len(certified),
            "completed_source_mismatch": len(source_mismatch),
            "completed_integrity_deficient": len(integrity_deficient),
            "author_accepted_completed_exceptions": (
                len(accepted_source_exceptions) + len(accepted_sidecar_exceptions)
            ),
            "strategy_actions": sum(
                action["action"] not in {"create_prefix_round5", "reconstruct_advisory_sidecar"}
                for action in actions
            ),
            "prefix_actions": sum(action["action"] == "create_prefix_round5" for action in actions),
            "sidecar_reconstruction_actions": sum(
                action["action"] == "reconstruct_advisory_sidecar" for action in actions
            ),
        },
    }
    plan_basis = {
        "schema": public_plan["schema"],
        "config": public_plan["config"],
        "output": public_plan["output"],
        "registered_source_hash": public_plan["registered_source_hash"],
        "recovery_script_sha256": file_digest(Path(__file__)),
        "config_file_sha256": file_digest(config_path),
        "source_identity": current_source,
        "structurally_complete_pending_audit": public_plan["structurally_complete_pending_audit"],
        "completed_source_mismatch": public_plan["completed_source_mismatch"],
        "completed_integrity_deficient": public_plan["completed_integrity_deficient"],
        "author_accepted_source_exceptions": public_plan["author_accepted_source_exceptions"],
        "author_accepted_advisory_sidecar_exceptions": (
            public_plan["author_accepted_advisory_sidecar_exceptions"]
        ),
        "forensic_prefixes": public_plan["forensic_prefixes"],
        "recovery_root": public_plan["recovery_root"],
        "actions": public_plan["actions"],
        "inspections": public_plan["inspections"],
    }
    public_plan["recovery_script_sha256"] = plan_basis["recovery_script_sha256"]
    public_plan["config_file_sha256"] = plan_basis["config_file_sha256"]
    public_plan["source_identity"] = current_source
    public_plan["plan_sha256"] = digest(plan_basis)
    public_plan["confirmation_token"] = f"EXECUTE-{public_plan['plan_sha256']}"
    public_plan["_contexts"] = contexts
    public_plan["_config_object"] = config
    return public_plan


def _public_plan(plan: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in plan.items() if not key.startswith("_")}


def _display_plan(plan: dict[str, Any], full: bool) -> dict[str, Any]:
    public = _public_plan(plan)
    if full:
        return public
    return {
        key: value for key, value in public.items()
        if key not in {"inspections"}
    }


def _validate_pristine_prefix(path: Path, identity: dict[str, Any], config: PaperConfig) -> None:
    result = _inspect_prefix(path, identity, config)
    if result["state"] != "pristine_round5":
        raise RecoveryError(f"prefix is not an exact five-round boundary: {path}: {result['state']}")


def execute_plan(plan: dict[str, Any], confirmation: str) -> None:
    public = _public_plan(plan)
    if confirmation != public["confirmation_token"]:
        raise RecoveryError("execution confirmation token does not match the dry-run plan")
    if not EXECUTION_ENABLED:
        raise RecoveryError(
            "execution is deliberately locked; dry-run review and author choice are required first"
        )
    config: PaperConfig = plan["_config_object"]
    contexts: dict[int, dict[str, Any]] = plan["_contexts"]
    output = Path(public["output"])
    recovery_root = Path(public["recovery_root"])
    lock_target = output / "campaign_recovery"
    with exclusive_run(lock_target):
        if _active_python_processes():
            raise RecoveryError("another python.exe process appeared after the dry run")
        if shutil.disk_usage(output).free / (1024 ** 3) < MIN_FREE_GIB:
            raise RecoveryError("free disk space fell below the recovery threshold")
        if source_identity()["source_hash"] != public["registered_source_hash"]:
            raise RecoveryError("source identity changed after the dry run")
        if file_digest(Path(public["config"])) != public["config_file_sha256"]:
            raise RecoveryError("configuration file changed after the dry run")

        recovery_root.mkdir(parents=True, exist_ok=True)
        manifest_path = recovery_root / "continuation_manifest.json"
        if manifest_path.exists():
            manifest = _read_json(manifest_path)
            if manifest.get("registered_source_hash") != public["registered_source_hash"]:
                raise RecoveryError("existing continuation manifest has a different source identity")
        else:
            manifest = {
                "schema": "fmnist.joint_jt.safe_continuation.v1",
                "registered_source_hash": public["registered_source_hash"],
                "config_file_sha256": public["config_file_sha256"],
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "sessions": [],
                "events": [],
            }
        session = {
            "session_id": uuid.uuid4().hex,
            "plan_sha256": public["plan_sha256"],
            "started_utc": datetime.now(timezone.utc).isoformat(),
            "status": "running",
            "planned_actions": len(public["actions"]),
        }
        manifest["sessions"].append(session)
        manifest["status"] = "running"
        atomic_json(manifest_path, manifest)

        try:
            for action in public["actions"]:
                if _active_python_processes():
                    raise RecoveryError("another python.exe process appeared during recovery")
                if shutil.disk_usage(output).free / (1024 ** 3) < MIN_FREE_GIB:
                    raise RecoveryError("free disk space fell below the recovery threshold")
                if source_identity()["source_hash"] != public["registered_source_hash"]:
                    raise RecoveryError("source identity changed during recovery")

                seed, strategy = int(action["seed"]), action["strategy"]
                context = contexts[seed]
                destination = Path(action["destination"])
                seed_dir = output / f"seed_{seed}"
                seed_dir.mkdir(exist_ok=True)
                for filename, value in (("partition.json", context["partition"]),
                                        ("traces.json", context["traces"])):
                    path = seed_dir / filename
                    if path.exists():
                        if digest(_read_json(path)) != digest(value):
                            raise RecoveryError(f"seed input changed after dry run: {path}")
                    else:
                        atomic_json(path, value)

                event = {
                    "session_id": session["session_id"],
                    "order": action["order"],
                    "action": action["action"],
                    "seed": seed,
                    "strategy": strategy,
                    "destination": str(destination),
                    "started_utc": datetime.now(timezone.utc).isoformat(),
                    "status": "running",
                }
                manifest["events"].append(event)
                atomic_json(manifest_path, manifest)
                atomic_json(output / "run_status.json", {
                    "status": "recovery_running", "seed": seed, "strategy": strategy,
                    "recovery_action": action["action"], "action_order": action["order"],
                    "total_actions": len(public["actions"]), "pid": os.getpid(),
                    "recovery_manifest": str(manifest_path),
                })

                if action["action"] == "create_prefix_round5":
                    if destination.exists():
                        raise RecoveryError(f"prefix destination appeared after dry run: {destination}")
                    staging = destination.parent / (
                        f".{destination.name}.recovery_stage_{os.getpid()}_{uuid.uuid4().hex}"
                    )
                    event["staging"] = str(staging)
                    campaign.stop_prefix(
                        config, seed, staging, context["data"],
                        context["partition"], context["traces"],
                    )
                    _validate_pristine_prefix(staging, context["identity"], config)
                    if destination.exists():
                        raise RecoveryError(f"prefix destination appeared during staging: {destination}")
                    os.replace(staging, destination)
                else:
                    if action["action"] == "resume_strategy":
                        current = _inspect_run(
                            destination, context["identity"], config, strategy=strategy,
                        )
                        if (current["state"] != "partial"
                                or int(current["latest_checkpoint"]["round"]) != int(action["from_round"])):
                            raise RecoveryError(f"resume state changed after dry run: {destination}")
                        resume = True
                    elif action["action"] == "fork_prefix_and_start":
                        prefix = Path(action["prefix"])
                        _validate_pristine_prefix(prefix, context["identity"], config)
                        if destination.exists():
                            raise RecoveryError(f"strategy destination appeared after dry run: {destination}")
                        staging = destination.parent / (
                            f".{destination.name}.recovery_stage_{os.getpid()}_{uuid.uuid4().hex}"
                        )
                        event["staging"] = str(staging)
                        shutil.copytree(prefix, staging)
                        _validate_pristine_prefix(staging, context["identity"], config)
                        if destination.exists():
                            raise RecoveryError(f"strategy destination appeared during staging: {destination}")
                        os.replace(staging, destination)
                        resume = True
                    elif action["action"] == "start_strategy":
                        if destination.exists():
                            raise RecoveryError(f"strategy destination appeared after dry run: {destination}")
                        resume = False
                    else:
                        raise RecoveryError(f"unsupported recovery action: {action['action']}")
                    campaign.run_one(
                        config, seed, strategy, destination,
                        context["data"], context["partition"], context["traces"], resume,
                    )
                    completed = _inspect_run(
                        destination, context["identity"], config, strategy=strategy,
                    )
                    if completed["state"] != "completed_exact":
                        raise RecoveryError(f"strategy did not finish cleanly: {destination}")

                event["status"] = "completed"
                event["completed_utc"] = datetime.now(timezone.utc).isoformat()
                session["completed_actions"] = int(session.get("completed_actions", 0)) + 1
                atomic_json(manifest_path, manifest)

            session["status"] = "completed"
            session["completed_utc"] = datetime.now(timezone.utc).isoformat()
            manifest["status"] = "training_completed_pending_audit"
            atomic_json(manifest_path, manifest)
            atomic_json(output / "run_status.json", {
                "status": "training_completed_pending_audit",
                "completed_actions": session.get("completed_actions", 0),
                "recovery_manifest": str(manifest_path), "pid": os.getpid(),
            })
        except BaseException as error:
            session["status"] = "failed"
            session["failed_utc"] = datetime.now(timezone.utc).isoformat()
            session["error_type"] = type(error).__name__
            session["error"] = str(error)
            session["traceback"] = traceback.format_exc()
            manifest["status"] = "failed"
            atomic_json(manifest_path, manifest)
            atomic_json(output / "run_status.json", {
                "status": "recovery_failed", "error_type": type(error).__name__,
                "error": str(error), "recovery_manifest": str(manifest_path),
                "pid": os.getpid(),
            })
            raise


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--execute", action="store_true", help="execute only after a successful dry run")
    parser.add_argument("--confirmation", default="", help="exact token printed by the dry run")
    parser.add_argument("--full-plan", action="store_true", help="include every per-run inspection in stdout")
    args = parser.parse_args()
    os.chdir(ROOT)
    plan = build_plan(args.config, args.output)
    print(json.dumps(_display_plan(plan, args.full_plan), indent=2, ensure_ascii=False), flush=True)
    if args.execute:
        execute_plan(plan, args.confirmation)


if __name__ == "__main__":
    main()
