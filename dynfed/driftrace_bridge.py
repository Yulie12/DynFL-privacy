from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class DriftRaceRunSpec:
    run_name: str
    method: str
    dataset: str = "cifar10"
    model: str = "lenet5"
    global_epoch: int = 1
    local_epoch: int | None = None
    seed: int = 42
    use_cuda: bool = False
    extra_overrides: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class DriftRaceBridgeConfig:
    drift_race_root: str = "E:/YTT/GROUP/DriftRace"
    output_root: str = "out/driftrace_real"
    python_executable: str = sys.executable
    dry_run: bool = False
    isolated_cwd: bool = False


def run_driftrace_specs(
    specs: list[DriftRaceRunSpec],
    bridge_config: DriftRaceBridgeConfig,
) -> list[dict[str, Any]]:
    drift_root = Path(bridge_config.drift_race_root)
    if not drift_root.exists():
        raise FileNotFoundError(f"DriftRace root does not exist: {drift_root}")
    if not (drift_root / "main.py").exists():
        raise FileNotFoundError(f"DriftRace main.py not found under: {drift_root}")

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S_%f")
    scenario_dir = Path(bridge_config.output_root) / f"{timestamp}_driftrace_real"
    scenario_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for spec in specs:
        run_dir = scenario_dir / f"{spec.run_name}_seed{spec.seed}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = build_driftrace_command(
            spec=spec,
            run_dir=run_dir,
            python_executable=bridge_config.python_executable,
            drift_root=drift_root,
            isolated_cwd=bridge_config.isolated_cwd,
        )
        manifest = {
            "bridge_config": asdict(bridge_config),
            "spec": asdict(spec),
            "driftrace_root": str(drift_root),
            "run_dir": str(run_dir),
            "command": cmd,
        }
        _write_json(run_dir / "bridge_manifest.json", manifest)

        if bridge_config.dry_run:
            summary = _dry_run_summary(spec, run_dir, cmd)
        else:
            completed = subprocess.run(
                cmd,
                cwd=str(Path.cwd() if bridge_config.isolated_cwd else drift_root),
                check=False,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=_build_env(drift_root, bridge_config.isolated_cwd),
            )
            (run_dir / "bridge_stdout.log").write_text(completed.stdout or "", encoding="utf-8")
            (run_dir / "bridge_stderr.log").write_text(completed.stderr or "", encoding="utf-8")
            if completed.returncode != 0:
                summary = _failed_summary(spec, run_dir, cmd, completed.returncode)
            else:
                summary = collect_driftrace_summary(spec, run_dir)
        summaries.append(summary)

    _write_csv(scenario_dir / "summary_table.csv", summaries)
    _write_json(
        scenario_dir / "bridge_manifest.json",
        {
            "bridge_config": asdict(bridge_config),
            "num_runs": len(specs),
            "runs": summaries,
        },
    )
    return summaries


def build_driftrace_command(
    spec: DriftRaceRunSpec,
    run_dir: Path,
    python_executable: str,
    drift_root: Path,
    isolated_cwd: bool = False,
) -> list[str]:
    normalized_run_dir = str(run_dir.resolve()).replace("\\", "/")
    main_script = str((drift_root / "main.py").resolve()) if isolated_cwd else "main.py"
    cmd = [
        python_executable,
        main_script,
        f"method={spec.method}",
        f"dataset.name={spec.dataset}",
        f"model.name={spec.model}",
        f"common.global_epoch={spec.global_epoch}",
        f"common.seed={spec.seed}",
        f"common.use_cuda={str(spec.use_cuda).lower()}",
        f"+common.output_dir={normalized_run_dir}",
        f"hydra.run.dir={normalized_run_dir}",
    ]
    if spec.local_epoch is not None:
        cmd.append(f"common.local_epoch={spec.local_epoch}")
    cmd.extend(spec.extra_overrides)
    return cmd


def _build_env(drift_root: Path, isolated_cwd: bool) -> dict[str, str] | None:
    if not isolated_cwd:
        return None
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    paths = [str(drift_root)]
    if existing:
        paths.append(existing)
    env["PYTHONPATH"] = os.pathsep.join(paths)
    return env


def collect_driftrace_summary(spec: DriftRaceRunSpec, run_dir: Path) -> dict[str, Any]:
    summary_path = run_dir / "summary.json"
    summary: dict[str, Any] = {}
    if summary_path.exists():
        with summary_path.open("r", encoding="utf-8") as file:
            summary = json.load(file)

    row = {
        "run_name": spec.run_name,
        "method": spec.method,
        "dataset": spec.dataset,
        "model": spec.model,
        "seed": spec.seed,
        "global_epoch": spec.global_epoch,
        "output_dir": str(run_dir),
        "final_test_accuracy": summary.get("final_test_accuracy"),
        "best_test_accuracy": summary.get("best_test_accuracy"),
        "best_round": summary.get("best_round"),
        "total_virtual_time": summary.get("total_virtual_time"),
        "total_comm_cost": summary.get("total_comm_cost"),
        "status": "completed",
    }
    return row


def _dry_run_summary(spec: DriftRaceRunSpec, run_dir: Path, cmd: list[str]) -> dict[str, Any]:
    return {
        "run_name": spec.run_name,
        "method": spec.method,
        "dataset": spec.dataset,
        "model": spec.model,
        "seed": spec.seed,
        "global_epoch": spec.global_epoch,
        "output_dir": str(run_dir),
        "final_test_accuracy": None,
        "best_test_accuracy": None,
        "best_round": None,
        "total_virtual_time": None,
        "total_comm_cost": None,
        "status": "dry_run",
        "command": " ".join(cmd),
    }


def _failed_summary(
    spec: DriftRaceRunSpec,
    run_dir: Path,
    cmd: list[str],
    returncode: int,
) -> dict[str, Any]:
    return {
        "run_name": spec.run_name,
        "method": spec.method,
        "dataset": spec.dataset,
        "model": spec.model,
        "seed": spec.seed,
        "global_epoch": spec.global_epoch,
        "output_dir": str(run_dir),
        "final_test_accuracy": None,
        "best_test_accuracy": None,
        "best_round": None,
        "total_virtual_time": None,
        "total_comm_cost": None,
        "status": "failed",
        "returncode": returncode,
        "command": " ".join(cmd),
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
