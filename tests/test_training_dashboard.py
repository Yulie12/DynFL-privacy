from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.monitor_training import collect_snapshot, discover_latest_output


def write_json(path: Path, value: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")


def test_collect_snapshot_reads_live_run(tmp_path: Path):
    root = tmp_path / "campaign"
    write_json(root / "cifar" / "config.json", {
        "dataset": "cifar10", "rounds": 10, "seeds": [42],
        "policies": ["proposed", "latency_only"],
    })
    run = root / "cifar" / "seed_42" / "proposed"
    write_json(run / "config.json", {"dataset": "cifar10", "rounds": 10})
    write_json(run / "run_status.json", {
        "status": "running", "completed_rounds": 2, "total_rounds": 10,
        "attempt_started_utc": "2026-10-21T00:00:00+00:00",
        "updated_utc": "2026-10-21T00:01:00+00:00", "pid": 123,
    })
    with (run / "round_metrics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["round", "test_accuracy", "test_loss", "max_epsilon"])
        writer.writeheader()
        writer.writerow({"round": 1, "test_accuracy": .4, "test_loss": 1.2, "max_epsilon": 1.0})
        writer.writerow({"round": 2, "test_accuracy": .5, "test_loss": 1.0, "max_epsilon": 2.0})
    snapshot = collect_snapshot(root)
    assert snapshot["planned_runs"] == 2
    assert snapshot["discovered_runs"] == 1
    assert snapshot["runs"][0]["id"] == "cifar/seed_42/proposed"
    assert snapshot["runs"][0]["latest"]["test_accuracy"] == .5
    assert snapshot["runs"][0]["progress"] == .2


def test_snapshot_tolerates_partial_metrics_and_finds_latest(tmp_path: Path):
    older = tmp_path / "older" / "seed_1" / "policy"
    newer = tmp_path / "newer" / "seed_1" / "policy"
    write_json(older / "run_status.json", {"status": "completed", "completed_rounds": 1})
    write_json(newer / "run_status.json", {"status": "running", "completed_rounds": 0})
    (newer / "round_metrics.csv").write_text("round,test_accuracy\nnot-a-round,", encoding="utf-8")
    assert discover_latest_output(tmp_path) == tmp_path / "newer"
    assert collect_snapshot(tmp_path / "newer")["runs"][0]["metrics"] == []
