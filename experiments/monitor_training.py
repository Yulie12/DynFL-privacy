"""Read-only browser dashboard for DynFL experiment output directories.

The monitor intentionally depends only on the Python standard library.  It
never imports the training runtime and never writes to the experiment tree.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
import json
import math
from pathlib import Path
import time
from urllib.parse import urlparse


ROOT = Path(__file__).resolve().parents[1]
PAGE = Path(__file__).with_name("training_dashboard.html")
SERIES_FIELDS = (
    "round", "test_accuracy", "test_loss", "system_time", "round_duration",
    "selection_sec", "predicted_j", "observed_j", "prediction_error",
    "admitted_clients", "cloud_units", "executed_local_steps", "mode_changes",
    "communication_mb", "max_epsilon",
)


def _json(path: Path) -> dict:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return {}


def _number(value):
    if value in (None, ""):
        return None
    try:
        result = float(value)
        return result if math.isfinite(result) else None
    except (TypeError, ValueError):
        return None


def _metrics(path: Path) -> list[dict]:
    try:
        with path.open(encoding="utf-8-sig", newline="") as handle:
            rows = []
            for source in csv.DictReader(handle):
                row = {key: _number(source.get(key)) for key in SERIES_FIELDS}
                if row["round"] is not None:
                    row["round"] = int(row["round"])
                    rows.append(row)
            return rows
    except (OSError, csv.Error, UnicodeDecodeError):
        # A writer may atomically replace or briefly expose a partial file.
        return []


def _utc_timestamp(value) -> float:
    if not value:
        return 0.0
    try:
        return datetime.fromisoformat(str(value).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def _run_identity(root: Path, folder: Path, config: dict) -> tuple[str, str, str]:
    relative = folder.relative_to(root)
    seed = next((part[5:] for part in relative.parts if part.startswith("seed_")), "—")
    return str(config.get("dataset") or relative.parts[0]), seed, folder.name


def _planned_runs(root: Path) -> int:
    plans = set()
    for path in root.rglob("config.json"):
        if any(part.startswith("seed_") for part in path.relative_to(root).parts):
            continue
        config = _json(path)
        dataset = str(config.get("dataset") or path.parent.name)
        for seed in config.get("seeds", []):
            for policy in config.get("policies", []):
                plans.add((dataset, str(seed), str(policy)))
    return len(plans)


def collect_snapshot(root: Path) -> dict:
    root = root.resolve()
    now = time.time()
    runs = []
    for status_path in root.rglob("run_status.json"):
        folder = status_path.parent
        status = _json(status_path)
        config = _json(folder / "config.json")
        metrics = _metrics(folder / "round_metrics.csv")
        dataset, seed, policy = _run_identity(root, folder, config)
        completed = int(status.get("completed_rounds") or len(metrics))
        total = int(status.get("total_rounds") or config.get("rounds") or completed)
        updated = _utc_timestamp(status.get("updated_utc")) or status_path.stat().st_mtime
        elapsed = max(0.0, now - _utc_timestamp(status.get("attempt_started_utc")))
        seconds_per_round = elapsed / completed if completed else None
        eta = seconds_per_round * max(0, total - completed) if seconds_per_round else None
        latest = metrics[-1] if metrics else {}
        run_status = str(status.get("status") or "unknown")
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
            "updated_age_sec": max(0.0, now - updated),
            "attempt_started_utc": status.get("attempt_started_utc"),
            "seconds_per_round": seconds_per_round,
            "eta_sec": eta,
            "pid": status.get("pid"),
            "latest": latest,
            "metrics": metrics,
            "error": status.get("error"),
        })
    priority = {"running": 0, "failed": 1, "interrupted": 2, "completed": 3}
    runs.sort(key=lambda item: (priority.get(item["status"], 2), -_utc_timestamp(item["updated_utc"])))
    planned = max(_planned_runs(root), len(runs))
    total_rounds = sum(item["total_rounds"] for item in runs)
    complete_rounds = sum(item["completed_rounds"] for item in runs)
    status_counts = {}
    for item in runs:
        status_counts[item["status"]] = status_counts.get(item["status"], 0) + 1
    return {
        "root": str(root),
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "planned_runs": planned,
        "discovered_runs": len(runs),
        "completed_runs": status_counts.get("completed", 0),
        "round_progress": complete_rounds / total_rounds if total_rounds else 0.0,
        "status_counts": status_counts,
        "runs": runs,
    }


def discover_latest_output(output_root: Path) -> Path:
    candidates = {}
    for status in output_root.rglob("run_status.json"):
        try:
            relative = status.relative_to(output_root)
            top = output_root / relative.parts[0]
            candidates[top] = max(candidates.get(top, 0.0), status.stat().st_mtime)
        except (OSError, ValueError):
            continue
    if not candidates:
        raise FileNotFoundError(f"no run_status.json found below {output_root}")
    return max(candidates, key=candidates.get)


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, experiment_root: Path, **kwargs):
        self.experiment_root = experiment_root
        super().__init__(*args, directory=str(PAGE.parent), **kwargs)

    def log_message(self, fmt, *args):
        if self.path != "/api/status":
            super().log_message(fmt, *args)

    def do_GET(self):
        route = urlparse(self.path).path
        if route == "/api/status":
            try:
                payload = json.dumps(collect_snapshot(self.experiment_root), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
            except Exception as error:  # Keep the page alive while files are changing.
                payload = json.dumps({"error": str(error)}, ensure_ascii=False).encode("utf-8")
                self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if route in ("/", "/index.html"):
            self.path = "/training_dashboard.html"
            return super().do_GET()
        self.send_error(404)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, help="Experiment directory; defaults to the latest directory below out/")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    experiment_root = (args.root or discover_latest_output(ROOT / "out")).resolve()
    if not experiment_root.is_dir():
        parser.error(f"not a directory: {experiment_root}")
    handler = partial(DashboardHandler, experiment_root=experiment_root)
    server = ThreadingHTTPServer((args.host, args.port), handler)
    print(f"DynFL dashboard: http://{args.host}:{args.port}", flush=True)
    print(f"Watching: {experiment_root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
