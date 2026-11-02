"""Operational dashboard launcher isolated from the experiment source hash.

This wrapper reuses the read-only dashboard implementation while excluding the
campaign-level orchestration status from per-policy runs.  It never imports or
starts the training runner and never writes to an experiment directory.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys
from threading import Lock
import time

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from experiments import monitor_training as base


_CACHE_LOCK = Lock()
_CACHE_ROOT: Path | None = None
_CACHE_AT = 0.0
_CACHE_PAYLOAD: dict | None = None


def _preregistered_plan(root: Path) -> tuple[set[tuple[str, str]], int | None]:
    preregistration = base._json(root / "preregistration.json")
    config = preregistration.get("config") or {}
    strategies = preregistration.get("strategies") or {}
    seeds = config.get("seeds") or []
    rounds = config.get("rounds")
    if not seeds or not strategies or not rounds:
        return set(), None
    return {(str(seed), str(policy)) for seed in seeds for policy in strategies}, int(rounds)


def _build_snapshot(root: Path) -> dict:
    root = root.resolve()
    now = time.time()
    runs = []
    for status_path in root.rglob("run_status.json"):
        folder = status_path.parent
        if folder == root:
            continue
        status = base._json(status_path)
        config = base._json(folder / "config.json")
        metrics = base._metrics(folder / "round_metrics.csv")
        dataset, seed, policy = base._run_identity(root, folder, config)
        completed = int(status.get("completed_rounds") or len(metrics))
        total = int(status.get("total_rounds") or config.get("rounds") or completed)
        updated = base._utc_timestamp(status.get("updated_utc")) or status_path.stat().st_mtime
        updated_age = max(0.0, now - updated)
        elapsed = max(0.0, now - base._utc_timestamp(status.get("attempt_started_utc")))
        seconds_per_round = elapsed / completed if completed else None
        eta = seconds_per_round * max(0, total - completed) if seconds_per_round else None
        run_status = str(status.get("status") or "unknown")
        if run_status == "running" and updated_age > 1800.0:
            run_status = "stale"
        runs.append({
            "id": folder.relative_to(root).as_posix(),
            "dataset": dataset,
            "seed": seed,
            "policy": policy,
            "status": run_status,
            "audit_status": status.get("audit_status"),
            "completed_rounds": completed,
            "total_rounds": total,
            "progress": completed / total if total else 0.0,
            "updated_utc": status.get("updated_utc"),
            "updated_age_sec": updated_age,
            "attempt_started_utc": status.get("attempt_started_utc"),
            "seconds_per_round": seconds_per_round,
            "eta_sec": eta,
            "pid": status.get("pid"),
            "latest": metrics[-1] if metrics else {},
            "metrics": metrics,
            "error": status.get("error"),
        })
    priority = {"running": 0, "failed": 1, "interrupted": 2, "stale": 2, "completed": 3}
    runs.sort(key=lambda item: (priority.get(item["status"], 2),
                                -base._utc_timestamp(item["updated_utc"])))
    plan, planned_rounds = _preregistered_plan(root)
    if plan and planned_rounds:
        planned_items = {
            (item["seed"], item["policy"]): item
            for item in runs
            if (item["seed"], item["policy"]) in plan
        }
        planned = len(plan)
        discovered = len(planned_items)
        total_rounds = planned * planned_rounds
        complete_rounds = sum(
            min(item["completed_rounds"], planned_rounds)
            for item in planned_items.values()
        )
        status_counts = {}
        for item in planned_items.values():
            status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
        queued = planned - discovered
        if queued:
            status_counts["queued"] = queued
        completed_runs = sum(item["status"] == "completed" for item in planned_items.values())
    else:
        planned = max(base._planned_runs(root), len(runs))
        discovered = len(runs)
        total_rounds = sum(item["total_rounds"] for item in runs)
        complete_rounds = sum(item["completed_rounds"] for item in runs)
        status_counts = {}
        for item in runs:
            status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
        completed_runs = status_counts.get("completed", 0)
    return {
        "root": str(root),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "planned_runs": planned,
        "discovered_runs": discovered,
        "completed_runs": completed_runs,
        "round_progress": complete_rounds / total_rounds if total_rounds else 0.0,
        "status_counts": status_counts,
        "runs": runs,
    }


def collect_snapshot(root: Path) -> dict:
    """Return a shared short-lived snapshot for all open dashboard tabs."""
    global _CACHE_ROOT, _CACHE_AT, _CACHE_PAYLOAD
    resolved = root.resolve()
    now = time.monotonic()
    if _CACHE_ROOT == resolved and _CACHE_PAYLOAD is not None and now - _CACHE_AT < 3.0:
        return _CACHE_PAYLOAD
    with _CACHE_LOCK:
        now = time.monotonic()
        if _CACHE_ROOT == resolved and _CACHE_PAYLOAD is not None and now - _CACHE_AT < 3.0:
            return _CACHE_PAYLOAD
        payload = _build_snapshot(resolved)
        _CACHE_ROOT = resolved
        _CACHE_AT = time.monotonic()
        _CACHE_PAYLOAD = payload
        return payload


def _quiet_log(self, fmt, *args):
    return None


base.collect_snapshot = collect_snapshot
base.DashboardHandler.log_message = _quiet_log


if __name__ == "__main__":
    base.main()
