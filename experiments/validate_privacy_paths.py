"""Run isolated short training checks with the original privacy horizon."""
from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_paper_config import DEFAULT_CONFIG, build_command, validate_config
from dynfed.utils import timestamped_dir


def collect_result(directory: Path, returncode: int) -> dict:
    paths = sorted(directory.rglob("round_metrics.csv"))
    rows = []
    for path in paths:
        with path.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream))
    result = {"process_exit_code": returncode, "recorded_rounds": len(rows),
              "run_directory": str(directory), "end_to_end_dp": "not_established"}
    if not rows:
        return dict(result, status="failed_no_metrics")
    def number(row, key):
        value = float(row.get(key) or 0)
        return value if math.isfinite(value) else None
    last = rows[-1]
    result.update(
        status="failed" if returncode else "short_run_finished",
        first_accuracy=number(rows[0], "test_accuracy"),
        final_accuracy=number(last, "test_accuracy"),
        final_loss=number(last, "test_loss"),
        final_noise_to_signal=number(last, "update_dp_noise_to_signal_ratio"),
        final_health=last.get("training_health", "unknown"),
        final_recorded_update_epsilon=number(last, "max_update_epsilon"),
        dp_release_count=sum(int(row.get("update_dp_release_count") or 0) for row in rows),
        real_he_rounds=sum(row.get("he_execution_status") == "real" for row in rows),
        profiled_he_rounds=sum(row.get("he_execution_status") == "profiled" for row in rows),
        wall_time_sec=number(last, "cumulative_wall_time_sec"),
        logical_time_sec=number(last, "logical_time"),
        he_wall_time_sec=sum(number(row, "he_wall_time_sec") or 0 for row in rows),
        he_max_abs_error=max(number(row, "he_max_abs_error") or 0 for row in rows),
    )
    if any(row.get("training_health") == "non_finite" for row in rows):
        result["status"] = "failed_non_finite"
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--summarize-only", type=Path,
                        help="Rebuild a diagnostic report from saved metrics without training")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-new-rounds", type=int, default=2)
    parser.add_argument("--he-execution", choices=["real", "profiled"], default="real")
    parser.add_argument("--he-workers", type=int, default=1)
    parser.add_argument("--clients", type=int)
    parser.add_argument("--edges", type=int)
    parser.add_argument("--train-limit", type=int)
    parser.add_argument("--test-limit", type=int)
    parser.add_argument("--output-root", default="out/privacy_path_validation")
    parser.add_argument("--policies", nargs="+", default=[
        "no_protection", "fixed_dp", "fixed_he", "fixed_dp_he", "ours"])
    args = parser.parse_args()
    if args.summarize_only is not None:
        report_path = args.summarize_only.resolve() / "validation_report.json"
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for policy, previous in list(report["results"].items()):
            result = collect_result(Path(previous["run_directory"]), previous["process_exit_code"])
            result["process_wall_time_sec"] = previous.get("process_wall_time_sec")
            if result["status"] == "short_run_finished" and result["recorded_rounds"] < min(
                    report["max_new_rounds"], report["privacy_horizon"]):
                result["status"] = "failed_incomplete"
            report["results"][policy] = result
        report_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Report {report_path}")
        return
    if args.max_new_rounds < 1 or args.he_workers < 1:
        parser.error("Round and worker limits must be positive")
    if len(set(args.policies)) != len(args.policies):
        parser.error("Policies must be unique")
    if any(not policy.replace("_", "").isalnum() for policy in args.policies):
        parser.error("Invalid policy name")
    config = json.loads(args.config.read_text(encoding="utf-8"))
    for section, names in (("training", ("train_limit", "test_limit")),
                           ("system", ("clients", "edges"))):
        for name in names:
            value = getattr(args, name)
            if value is not None:
                if value < 1:
                    parser.error(f"{name} must be positive")
                config[section][name] = value
    config["he"].update(execution=args.he_execution,
                        require_real_he=args.he_execution == "real", workers=args.he_workers)
    validate_config(config)
    output = timestamped_dir(args.output_root, "privacy_paths").resolve()
    output.mkdir(parents=True, exist_ok=False)
    config["output_root"] = str(output)
    (output / "effective_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8")
    report = {"scope": "diagnostic_not_paper_result", "config": str(args.config.resolve()),
              "privacy_horizon": config["training"]["rounds"], "seed": args.seed,
              "max_new_rounds": args.max_new_rounds, "results": {}}
    print(f"Diagnostic directory {output}", flush=True)
    print(f"Privacy horizon remains {report['privacy_horizon']} rounds", flush=True)
    for policy in args.policies:
        policy_dir = output / policy
        policy_dir.mkdir()
        config["output_root"] = str(policy_dir)
        command = build_command(config, seed=args.seed, policies=[policy],
                                rounds=None, max_new_rounds=args.max_new_rounds,
                                resume_from_run=None)
        (policy_dir / "command.json").write_text(json.dumps(command, indent=2), encoding="utf-8")
        started = time.perf_counter()
        with (policy_dir / "console.log").open("w", encoding="utf-8") as log:
            with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                                  stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                  errors="replace") as process:
                try:
                    for line in process.stdout:
                        print(line, end="", flush=True)
                        log.write(line)
                        log.flush()
                    returncode = process.wait()
                except KeyboardInterrupt:
                    process.terminate()
                    process.wait()
                    raise
        result = collect_result(policy_dir, returncode)
        result["process_wall_time_sec"] = time.perf_counter() - started
        if not returncode and result["recorded_rounds"] < min(
                args.max_new_rounds, int(config["training"]["rounds"])):
            result["status"] = "failed_incomplete"
        report["results"][policy] = result
        (output / "validation_report.json").write_text(
            json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
        print(f"Validation {policy} {result['status']}", flush=True)
    print(f"Report {output / 'validation_report.json'}", flush=True)
    if any(item["status"].startswith("failed") for item in report["results"].values()):
        sys.exit(1)


if __name__ == "__main__":
    main()
