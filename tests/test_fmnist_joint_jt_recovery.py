from dataclasses import replace
import csv
import json

import pytest

from dynfed.paper_barrier import VersionBarrier
from dynfed.paper_config import PaperConfig
from dynfed.paper_privacy import Ledger
from dynfed.paper_storage import digest, save_checkpoint, source_identity
import tools.recover_fmnist_joint_jt_campaign as recovery
from tools.recover_fmnist_joint_jt_campaign import (
    RecoveryError,
    _display_plan,
    _has_uncommitted_round_files,
    _inspect_prefix,
    execute_plan,
)


def _write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _round5_prefix(tmp_path):
    destination = tmp_path / "prefix"
    destination.mkdir()
    identity = {
        "config_hash": "config", "source_hash": "source", "partition_hash": "partition",
        "traces_hash": "traces", "data_hashes": ["data"], "runtime": {"python": "test"},
    }
    (destination / "run_identity.json").write_text(json.dumps(identity), encoding="utf-8")
    ledgers = {client: Ledger(8.0, 1e-5) for client in range(2)}
    barrier = VersionBarrier(range(2))
    rows = []
    for round_number in range(1, 6):
        rows.append({
            "round": round_number, "acknowledged_clients": 2,
            "pending_units": 0, "cloud_version": round_number,
        })
        barrier.begin(round_number)
        for client in range(2):
            barrier.acknowledge(client, round_number, float(round_number))
        decisions = [
            {"round": round_index, "client_id": client}
            for round_index in range(1, round_number + 1)
            for client in range(2)
        ]
        save_checkpoint(destination / f"checkpoint_round_{round_number:03d}.pt", {
            "run_id": digest(identity), "next_round": round_number,
            "rows": list(rows), "decisions": decisions, "ledgers": ledgers, "barrier": barrier,
        })
    _write_csv(destination / "round_metrics.csv", rows)
    (destination / "privacy_state.json").write_text(json.dumps({
        str(client): {"rdp": ledger.rdp, "events": ledger.events, "epsilon": ledger.epsilon()}
        for client, ledger in ledgers.items()
    }), encoding="utf-8")
    return destination, identity


def test_recovery_tool_is_outside_preregistered_source_identity():
    manifest = source_identity()["sha256"]
    assert "tools/recover_fmnist_joint_jt_campaign.py" not in manifest
    assert all(not name.startswith("tools/") for name in manifest)


def test_exact_round5_prefix_is_accepted(tmp_path):
    destination, identity = _round5_prefix(tmp_path)
    config = replace(PaperConfig(), clients=2, edges=1, client_edge_map=(0, 0), rounds=100)
    result = _inspect_prefix(destination, identity, config)
    assert result["state"] == "pristine_round5"
    assert result["latest_checkpoint"]["round"] == 5
    assert result["privacy_state_matches"]


def test_round6_work_quarantines_prefix(tmp_path):
    destination, identity = _round5_prefix(tmp_path)
    (destination / "block_r5_FL_C_c0_s0_c0.pt").write_bytes(b"partial round six")
    config = replace(PaperConfig(), clients=2, edges=1, client_edge_map=(0, 0), rounds=100)
    result = _inspect_prefix(destination, identity, config)
    assert result["state"] == "contaminated"
    assert result["reason"] == "uncommitted_state"
    assert "block_r5_FL_C_c0_s0_c0.pt" in result["uncommitted_files"]


def test_uncommitted_detector_covers_plan_tmp_and_unsafe_state(tmp_path):
    for name in ("plan_round_005.pt", "block_r5_FL_C_c0_s0_c0.pt", "round.tmp",
                 "unsafe_privacy_state.json"):
        (tmp_path / name).write_bytes(b"x")
    offenders = _has_uncommitted_round_files(tmp_path, 5)
    assert set(offenders) == {
        "plan_round_005.pt", "block_r5_FL_C_c0_s0_c0.pt", "round.tmp",
        "unsafe_privacy_state.json",
    }


def test_execution_requires_exact_plan_token_before_any_write(tmp_path):
    plan = {
        "confirmation_token": "EXECUTE-correct", "output": str(tmp_path / "output"),
        "recovery_root": str(tmp_path / "recovery"), "actions": [],
        "registered_source_hash": "source", "_config_object": object(), "_contexts": {},
    }
    with pytest.raises(RecoveryError, match="confirmation token"):
        execute_plan(plan, "EXECUTE-wrong")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "recovery").exists()


def test_execution_lock_can_disable_all_writes_with_correct_token(tmp_path, monkeypatch):
    plan = {
        "confirmation_token": "EXECUTE-correct", "output": str(tmp_path / "output"),
        "recovery_root": str(tmp_path / "recovery"), "actions": [],
        "registered_source_hash": "source", "_config_object": object(), "_contexts": {},
    }
    monkeypatch.setattr(recovery, "EXECUTION_ENABLED", False)
    with pytest.raises(RecoveryError, match="deliberately locked"):
        execute_plan(plan, "EXECUTE-correct")
    assert not (tmp_path / "output").exists()
    assert not (tmp_path / "recovery").exists()


def test_unlocked_empty_plan_uses_manifest_and_finishes_atomically(tmp_path, monkeypatch):
    config_path = tmp_path / "config.json"
    config_path.write_text("{}", encoding="utf-8")
    output = tmp_path / "output"
    output.mkdir()
    recovery_root = output / "_recovery" / "strict_source_source"
    plan = {
        "confirmation_token": "EXECUTE-correct",
        "plan_sha256": "plan",
        "config": str(config_path),
        "config_file_sha256": recovery.file_digest(config_path),
        "output": str(output),
        "recovery_root": str(recovery_root),
        "actions": [],
        "registered_source_hash": "source",
        "_config_object": object(),
        "_contexts": {},
    }
    monkeypatch.setattr(recovery, "EXECUTION_ENABLED", True)
    monkeypatch.setattr(recovery, "_active_python_processes", lambda: [])
    monkeypatch.setattr(recovery, "source_identity", lambda: {"source_hash": "source"})
    recovery.execute_plan(plan, "EXECUTE-correct")
    manifest = json.loads(
        (recovery_root / "continuation_manifest.json").read_text(encoding="utf-8")
    )
    status = json.loads((output / "run_status.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "training_completed_pending_audit"
    assert manifest["sessions"][-1]["status"] == "completed"
    assert status["status"] == "training_completed_pending_audit"


def test_concise_display_omits_heavy_inspection_context():
    plan = {"summary": {"strategy_actions": 28}, "inspections": [1], "_contexts": {42: object()}}
    assert _display_plan(plan, full=False) == {"summary": {"strategy_actions": 28}}
    assert _display_plan(plan, full=True) == {
        "summary": {"strategy_actions": 28}, "inspections": [1]
    }
