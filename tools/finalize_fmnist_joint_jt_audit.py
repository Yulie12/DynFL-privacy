"""Finalize the FMNIST campaign audit under the declared author exceptions.

This post-training tool never changes checkpoints, metrics, decisions, timelines,
or per-run status files.  It preserves the unqualified audit as
``audit_strict.json`` and writes an exception-aware ``audit.json`` only when the
strict failures and source-identity differences exactly match the predeclared
sets below.  A passing final status therefore means "passed under the declared
exception policy", not strict source/checksum identity.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from dynfed.paper_storage import digest, source_identity


REGISTERED_SOURCE_HASH = (
    "e447a45e156ce87c851892355ef535dc684f7b56b9f03412b832af4554b96988"
)
ACCEPTED_SOURCE_HASH = (
    "03440807648e43fcf95306edfdce5d55d748dda2a0b0b51b04f47102488dfc59"
)
CONFIG_FILE = PROJECT_ROOT / "configs" / "fmnist_joint_jt_100r_3seed_20261022.json"
CONFIG_FILE_SHA256 = (
    "bba1e50abbca3c4e7ad190e923bf85404991b5648b51d1dda8b4aaf1bb84650b"
)
STRATEGIES = (
    "fixed_P0", "fixed_P1", "fixed_P2", "fixed_P3", "fixed_P4",
    "greedy_lambda025", "greedy_lambda050", "greedy_lambda075",
    "hysteresis_lambda050", "periodic_lambda050", "max_J", "min_T",
    "pure_FL_E", "pure_FL_EC", "pure_FL_C",
)
SOURCE_EXCEPTIONS = {
    (43, "fixed_P2"), (43, "fixed_P3"), (43, "fixed_P4"),
}
SIDECAR_EXCEPTIONS = {
    (42, "greedy_lambda025"), (42, "greedy_lambda050"),
    (42, "greedy_lambda075"), (42, "hysteresis_lambda050"),
    (42, "periodic_lambda050"), (42, "max_J"), (42, "min_T"),
}
DYNAMIC_STRATEGIES = {
    "greedy_lambda025", "greedy_lambda050", "greedy_lambda075",
    "hysteresis_lambda050", "periodic_lambda050", "max_J", "min_T",
}


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            json.dump(value, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_bytes(path: Path, value: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def finalize(root: Path) -> dict[str, Any]:
    root = root.resolve()
    strict_path = root / "audit.json"
    preserved_path = root / "audit_strict.json"
    current_audit = _read_json(strict_path)
    if current_audit.get("schema") == "fmnist.joint_jt.exception_aware_audit.v1":
        if not preserved_path.exists():
            raise RuntimeError("exception-aware audit exists without audit_strict.json")
        strict = _read_json(preserved_path)
    else:
        strict = current_audit
    if strict.get("schema") != "fmnist.joint_jt.independent_audit.v1":
        raise RuntimeError("unexpected strict-audit schema")
    if strict.get("status") != "failed":
        raise RuntimeError("expected the unqualified audit to retain strict failures")

    expected_failure_names = {
        f"seed{seed}_{strategy}_checkpoint_integrity"
        for seed, strategy in SIDECAR_EXCEPTIONS
    }
    actual_failure_names = {item.get("name") for item in strict.get("failures", [])}
    if actual_failure_names != expected_failure_names:
        raise RuntimeError(
            "strict failures differ from the declared advisory-sidecar exceptions: "
            f"expected={sorted(expected_failure_names)!r}, actual={sorted(actual_failure_names)!r}"
        )
    check_names = [item.get("name") for item in strict.get("checks", [])]
    if len(check_names) != 516 or len(set(check_names)) != 516:
        raise RuntimeError("strict-audit checks are missing or duplicated")
    expected_check_names: set[str] = set()
    base_suffixes = {
        "round_count", "decision_count", "privacy", "barrier", "allocations",
        "paper_T", "timeline", "J", "normalization_utility", "selection_rule",
        "checkpoint_integrity",
    }
    for seed in (42, 43, 44):
        for strategy in STRATEGIES:
            expected_check_names.update(
                f"seed{seed}_{strategy}_{suffix}" for suffix in base_suffixes
            )
            if strategy in DYNAMIC_STRATEGIES:
                expected_check_names.add(f"seed{seed}_{strategy}_shared_checkpoint")
    if set(check_names) != expected_check_names:
        raise RuntimeError("strict-audit check-name set differs from the registered campaign")
    strict_check_by_name = {item["name"]: item for item in strict["checks"]}
    false_check_names = {
        item["name"] for item in strict["checks"] if not item.get("passed")
    }
    if false_check_names != actual_failure_names:
        raise RuntimeError("strict false-check set and failure list differ")
    if any(
        not item.get("passed") and item.get("name") not in expected_failure_names
        for item in strict["checks"]
    ):
        raise RuntimeError("an unapproved strict-audit failure is present")

    source_observations: list[dict[str, Any]] = []
    sidecar_observations: list[dict[str, Any]] = []
    config_hashes_by_seed: dict[int, set[str]] = {42: set(), 43: set(), 44: set()}
    partition_hashes_by_seed: dict[int, set[str]] = {42: set(), 43: set(), 44: set()}
    trace_hashes_by_seed: dict[int, set[str]] = {42: set(), 43: set(), 44: set()}
    actual_partition_hashes: dict[int, str] = {}
    actual_trace_hashes: dict[int, str] = {}
    additional_directories: dict[int, list[str]] = {}

    actual_source_identity = source_identity()
    if actual_source_identity.get("source_hash") != REGISTERED_SOURCE_HASH:
        raise RuntimeError("current scientific source identity differs from the registered hash")
    if _digest(CONFIG_FILE) != CONFIG_FILE_SHA256:
        raise RuntimeError("current campaign configuration file differs from the registered digest")

    for seed in (42, 43, 44):
        seed_dir = root / f"seed_{seed}"
        directory_names = {path.name for path in seed_dir.iterdir() if path.is_dir()}
        if not set(STRATEGIES).issubset(directory_names):
            raise RuntimeError(f"formal strategy set is incomplete for seed {seed}")
        additional_directories[seed] = sorted(directory_names - set(STRATEGIES))
        actual_partition_hashes[seed] = digest(_read_json(seed_dir / "partition.json"))
        actual_trace_hashes[seed] = digest(_read_json(seed_dir / "traces.json"))
        for strategy in STRATEGIES:
            folder = root / f"seed_{seed}" / strategy
            identity = _read_json(folder / "run_identity.json")
            observed_source = identity.get("source_hash")
            expected_source = (
                ACCEPTED_SOURCE_HASH if (seed, strategy) in SOURCE_EXCEPTIONS
                else REGISTERED_SOURCE_HASH
            )
            if observed_source != expected_source:
                raise RuntimeError(
                    f"unapproved source identity for seed {seed} {strategy}: {observed_source}"
                )
            source_observations.append({
                "seed": seed,
                "strategy": strategy,
                "source_hash": observed_source,
                "classification": (
                    "author_accepted_source_exception"
                    if (seed, strategy) in SOURCE_EXCEPTIONS else "registered_source"
                ),
            })
            config_hashes_by_seed[seed].add(identity["config_hash"])
            partition_hashes_by_seed[seed].add(identity["partition_hash"])
            trace_hashes_by_seed[seed].add(identity["traces_hash"])

            checkpoints = sorted(folder.glob("checkpoint_round_*.pt"))
            if len(checkpoints) != 100:
                raise RuntimeError(f"expected 100 checkpoints in {folder}")
            missing: list[str] = []
            invalid: list[str] = []
            # Recheck all 4,500 checkpoint digests after the strict audit.  This
            # closes the interval between strict-audit generation and policy
            # finalization instead of merely trusting the earlier result.
            for checkpoint in checkpoints:
                sidecar = checkpoint.with_suffix(".sha256")
                if not sidecar.exists():
                    missing.append(checkpoint.name)
                elif sidecar.read_text(encoding="ascii").strip() != _digest(checkpoint):
                    invalid.append(checkpoint.name)
            expected_missing = (
                ["checkpoint_round_001.pt"] if (seed, strategy) in SIDECAR_EXCEPTIONS else []
            )
            if missing != expected_missing or invalid:
                raise RuntimeError(
                    f"unexpected checkpoint integrity state for seed {seed} {strategy}: "
                    f"missing={missing!r}, invalid={invalid!r}"
                )
            sidecar_observations.append({
                "seed": seed,
                "strategy": strategy,
                "checkpoint_count": len(checkpoints),
                "verified_sidecar_count": len(checkpoints) - len(missing),
                "missing_sidecars": missing,
                "invalid_sidecars": invalid,
                "classification": (
                    "author_accepted_advisory_sidecar_exception"
                    if missing else "strictly_verified"
                ),
            })

    for seed in (42, 43, 44):
        if len(config_hashes_by_seed[seed]) != 1:
            raise RuntimeError(f"config identity differs within seed {seed}")
        if len(partition_hashes_by_seed[seed]) != 1:
            raise RuntimeError(f"partition identity differs within seed {seed}")
        if len(trace_hashes_by_seed[seed]) != 1:
            raise RuntimeError(f"trace identity differs within seed {seed}")
        if next(iter(partition_hashes_by_seed[seed])) != actual_partition_hashes[seed]:
            raise RuntimeError(f"partition file digest differs from run identity for seed {seed}")
        if next(iter(trace_hashes_by_seed[seed])) != actual_trace_hashes[seed]:
            raise RuntimeError(f"trace file digest differs from run identity for seed {seed}")

    prefix_observations: list[dict[str, Any]] = []
    seed42_prefix_round1 = (
        root / "seed_42" / "shared_P0_prefix" / "checkpoint_round_001.pt"
    )
    prefix_round1_digest = _digest(seed42_prefix_round1)
    round1_equivalence: list[dict[str, Any]] = []
    for seed, strategy in sorted(SIDECAR_EXCEPTIONS):
        checkpoint = root / f"seed_{seed}" / strategy / "checkpoint_round_001.pt"
        observed_digest = _digest(checkpoint)
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if not isinstance(loaded, dict):
            raise RuntimeError(f"round-1 checkpoint is not a mapping: {checkpoint}")
        if observed_digest != prefix_round1_digest:
            raise RuntimeError(f"round-1 checkpoint differs from the shared prefix: {checkpoint}")
        round1_equivalence.append({
            "seed": seed,
            "strategy": strategy,
            "sha256": observed_digest,
            "loadable": True,
            "matches_current_seed42_prefix_round1": True,
        })
    for seed in (42, 43, 44):
        prefix = root / f"seed_{seed}" / "shared_P0_prefix"
        rounds = sorted(
            int(path.stem.rsplit("_", 1)[1])
            for path in prefix.glob("checkpoint_round_*.pt")
        )
        prefix_observations.append({
            "seed": seed,
            "path": str(prefix.relative_to(root)),
            "checkpoint_rounds_present": rounds,
            "authoritative_rounds": [1, 2, 3, 4, 5],
            "excluded_overrun_rounds": [value for value in rounds if value > 5],
            "round1_sidecar_present": (
                prefix / "checkpoint_round_001.sha256"
            ).exists(),
        })

    # Preserve the strict result byte-for-byte before replacing audit.json.
    if preserved_path.exists():
        if _read_json(preserved_path) != strict:
            raise RuntimeError("existing audit_strict.json differs from the strict audit evidence")
    else:
        _atomic_bytes(preserved_path, strict_path.read_bytes())

    strict_audit_sha256 = _digest(preserved_path)
    auditor_path = PROJECT_ROOT / "experiments" / "audit_fmnist_joint_jt.py"
    finalizer_path = Path(__file__).resolve()
    recovery_manifest = (
        root / "_recovery" / f"strict_source_{REGISTERED_SOURCE_HASH[:12]}"
        / "continuation_manifest.json"
    )
    recovery = _read_json(recovery_manifest)
    completed_sessions = [
        session for session in recovery.get("sessions", [])
        if session.get("status") == "completed"
    ]
    if recovery.get("status") != "training_completed_pending_audit" or not completed_sessions:
        raise RuntimeError("recovery manifest does not establish completed training")

    result = {
        "schema": "fmnist.joint_jt.exception_aware_audit.v1",
        "status": "passed",
        "status_qualification": "passed_under_predeclared_author_exceptions",
        "unconditional_pass": False,
        "strict_compliance": False,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "strict_audit": {
            "path": preserved_path.name,
            "sha256": strict_audit_sha256,
            "status": strict["status"],
            "check_count": len(strict["checks"]),
            "passed_check_count": sum(bool(item["passed"]) for item in strict["checks"]),
            "failure_count": len(strict["failures"]),
            "failure_names": sorted(actual_failure_names),
        },
        "accepted_exceptions": {
            "source_identity": [
                {
                    "seed": seed,
                    "strategy": strategy,
                    "registered_source_hash": REGISTERED_SOURCE_HASH,
                    "observed_source_hash": ACCEPTED_SOURCE_HASH,
                    "scope": (
                        "author-waived source reproducibility mismatch; recorded numerical audit "
                        "checks pass, but training-code equivalence is not established"
                    ),
                }
                for seed, strategy in sorted(SOURCE_EXCEPTIONS)
            ],
            "advisory_checkpoint_sidecar": [
                {
                    "seed": seed,
                    "strategy": strategy,
                    "missing": "checkpoint_round_001.sha256",
                    "scope": "missing historical checksum sidecar only; no checksum was reconstructed",
                }
                for seed, strategy in sorted(SIDECAR_EXCEPTIONS)
            ],
        },
        "source_identity_observations": source_observations,
        "checkpoint_sidecar_observations": sidecar_observations,
        "round1_checkpoint_observations": {
            "shared_prefix_sha256": prefix_round1_digest,
            "shared_prefix_sidecar_present": False,
            "current_byte_equivalence_only": True,
            "historical_checksum_provenance_established": False,
            "runs": round1_equivalence,
        },
        "shared_prefix_observations": prefix_observations,
        "formal_result_scope": {
            "run_count": 45,
            "seeds": [42, 43, 44],
            "strategies": list(STRATEGIES),
            "additional_directories_excluded": {
                str(seed): names for seed, names in additional_directories.items()
            },
            "seed42_prefix_note": (
                "Only rounds 1--5 are authoritative. Rounds 6--8 in shared_P0_prefix and the "
                "separate resume-overrun directory are preserved incident evidence and excluded."
            ),
        },
        "execution_identity": {
            "current_source_identity": actual_source_identity,
            "registered_source_hash": REGISTERED_SOURCE_HASH,
            "config_file": str(CONFIG_FILE),
            "config_file_sha256": _digest(CONFIG_FILE),
            "registered_config_file_sha256": CONFIG_FILE_SHA256,
            "auditor_path": str(auditor_path),
            "auditor_sha256": _digest(auditor_path),
            "finalizer_path": str(finalizer_path),
            "finalizer_sha256": _digest(finalizer_path),
            "recovery_manifest": str(recovery_manifest),
            "recovery_manifest_sha256": _digest(recovery_manifest),
        },
        "within_seed_identity_checks": {
            str(seed): {
                "config_hash": next(iter(config_hashes_by_seed[seed])),
                "partition_hash": next(iter(partition_hashes_by_seed[seed])),
                "traces_hash": next(iter(trace_hashes_by_seed[seed])),
            }
            for seed in (42, 43, 44)
        },
        "claims": strict.get("claims", {}),
        "interpretation": (
            "All 509 unqualified checks passed. The seven strict failures are exactly the "
            "predeclared missing round-1 advisory checksum sidecars. Three completed seed-43 "
            "fixed runs retain a separately declared source-identity exception. This result "
            "must not be described as strict source/checksum identity."
        ),
    }
    _atomic_json(strict_path, result)
    current_root_status = _read_json(root / "run_status.json")
    updated_root_status = dict(current_root_status)
    updated_root_status.update({
        "status": "completed",
        "audit_status": "passed",
        "audit_qualification": result["status_qualification"],
        "unconditional_audit_pass": False,
        "strict_audit_status": strict["status"],
        "strict_audit_failures": len(strict["failures"]),
        "runs": 45,
        "completed_actions": completed_sessions[-1].get("completed_actions"),
        "recovery_status_before_audit": recovery["status"],
        "recovery_manifest": str(recovery_manifest),
    })
    _atomic_json(root / "run_status.json", updated_root_status)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    result = finalize(args.root)
    print(json.dumps({
        "status": result["status"],
        "qualification": result["status_qualification"],
        "strict_audit": result["strict_audit"],
        "accepted_source_exceptions": len(result["accepted_exceptions"]["source_identity"]),
        "accepted_sidecar_exceptions": len(
            result["accepted_exceptions"]["advisory_checkpoint_sidecar"]
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
