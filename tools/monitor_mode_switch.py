"""Enhanced read-only monitor kept outside registered training source identity."""
from pathlib import Path
import os
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import experiments.monitor_training as monitor


base_collect_snapshot = monitor.collect_snapshot


def collect_snapshot(root: Path) -> dict:
    snapshot = base_collect_snapshot(root)
    runs = [item for item in snapshot["runs"] if item["policy"] != "common_prefix"]
    for item in runs:
        if item["status"] == "running" and item.get("pid"):
            try:
                os.kill(int(item["pid"]), 0)
            except OSError:
                item["status"] = "paused"
        if item["completed_rounds"]:
            report = monitor._json(root / item["id"] /
                                   f"candidates_round_{item['completed_rounds']-1:03d}.json")
            item["latest"]["selected_candidate"] = report.get("selected_candidate")
    manifest = monitor._json(root / "config_manifest.json")
    config = manifest.get("config", {}) if isinstance(manifest.get("config"), dict) else {}
    declared = list(manifest.get("strategies", []))
    for strategy in declared:
        if any(item["policy"] == strategy for item in runs):
            continue
        runs.append({"id": f"seed_42/{strategy}", "dataset": config.get("dataset", "fmnist"),
                     "seed": "42", "policy": strategy, "status": "queued", "audit_status": None,
                     "completed_rounds": 0, "total_rounds": int(config.get("rounds") or 0),
                     "progress": 0.0, "updated_utc": None, "updated_age_sec": None,
                     "attempt_started_utc": None, "seconds_per_round": None, "eta_sec": None,
                     "pid": None, "latest": {}, "metrics": [], "error": None})
    extra_root = root.parents[1] / "fmnist_mode_switch_all_candidates_20261021"
    if extra_root.exists():
        extra = base_collect_snapshot(extra_root)
        for item in extra["runs"]:
            if item["policy"] == "common_prefix":
                continue
            item = dict(item)
            item["id"] = "all_candidates/" + item["id"]
            runs.append(item)
        queue = monitor._json(extra_root / "queue_status.json")
        if queue.get("status") not in {None, "completed"}:
            post_prefix = sum(max(0, item["completed_rounds"] - 5) for item in extra["runs"]
                              if item["policy"] != "common_prefix")
            prefix = max((item["completed_rounds"] for item in extra["runs"]
                          if item["policy"] == "common_prefix"), default=0)
            completed = prefix + post_prefix
            state = "running" if str(queue.get("status", "")).startswith("running") else "queued"
            runs.append({"id": "all_candidates/queue", "dataset": "fmnist", "seed": "42",
                         "policy": "P0–P4 fixed-switch 10r", "status": state,
                         "audit_status": None, "completed_rounds": completed, "total_rounds": 30,
                         "progress": completed/30, "updated_utc": queue.get("updated_utc"),
                         "updated_age_sec": None, "attempt_started_utc": None,
                         "seconds_per_round": None, "eta_sec": None, "pid": queue.get("pid"),
                         "latest": {"selected_candidate": queue.get("active") or queue.get("status")},
                         "metrics": [], "error": None})
    priority = {"running": 0, "failed": 1, "interrupted": 2, "queued": 3, "completed": 4}
    runs.sort(key=lambda item: priority.get(item["status"], 3))
    snapshot["runs"] = runs
    snapshot["planned_runs"] = max(len(declared), len(runs))
    snapshot["discovered_runs"] = len(runs)
    snapshot["completed_runs"] = sum(item["status"] == "completed" for item in runs)
    total = sum(item["total_rounds"] for item in runs)
    snapshot["round_progress"] = sum(item["completed_rounds"] for item in runs) / total if total else 0.0
    counts = {}
    for item in runs:
        counts[item["status"]] = counts.get(item["status"], 0) + 1
    snapshot["status_counts"] = counts
    return snapshot


monitor.collect_snapshot = collect_snapshot

if __name__ == "__main__":
    monitor.main()
