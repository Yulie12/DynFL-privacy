from __future__ import annotations

"""Run a dedicated matched clean/private calibration pass.

This wrapper intentionally does *not* call the formal paper-config validator:
calibration is auxiliary offline work and uses exactly one bootstrap policy.
The resulting table can later be consumed by the formal joint-calibration
selector.  By default HE execution is profiled because cryptographic runtime is
irrelevant to the local nonlinear trajectory moments being estimated.
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_paper_config import build_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture DynFL matched clean/private Sample-DP trajectories into a calibration table."
    )
    parser.add_argument("--config", type=Path, required=True, help="Base paper JSON used for model/system/privacy settings.")
    parser.add_argument("--output", type=Path, required=True, help="Output joint calibration .pt table.")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--policy", default="full_dynfl")
    parser.add_argument("--rounds", type=int, default=None)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--period", type=int, default=10)
    parser.add_argument("--max-clients", type=int, default=8, help="0 means all clients.")
    parser.add_argument("--sample-limit", type=int, default=64, help="0 means all held-out samples per sampled client.")
    parser.add_argument("--scope", choices=["selected", "candidate_modes"], default="candidate_modes")
    parser.add_argument(
        "--real-he",
        action="store_true",
        help="Execute real HE during the calibration pass. Default is profiled HE because calibration targets learning trajectories, not HE timing.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_capture_command(args: argparse.Namespace) -> tuple[list[str], dict]:
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if str(config.get("privacy", {}).get("unit", "sample")).strip().lower() != "sample":
        raise ValueError("joint calibration capture is Sample-DP only")
    if int(args.trials) < 2:
        raise ValueError("--trials must be at least 2")
    if int(args.period) < 1:
        raise ValueError("--period must be positive")
    if int(args.max_clients) < 0 or int(args.sample_limit) < 0:
        raise ValueError("--max-clients and --sample-limit must be non-negative")

    capture_config = copy.deepcopy(config)
    capture_config["policies"] = [str(args.policy)]
    capture_config["seeds"] = [
        int(args.seed) if args.seed is not None else int(config.get("seeds", [0])[0])
    ]
    capture_config["output_root"] = str(
        ROOT / "out" / "joint_calibration_capture" / config_path.stem
    )
    capture_config.setdefault("learning", {})
    # Bootstrap selection must not depend on the table it is trying to create.
    capture_config["learning"] = {
        "objective": "legacy_fusion_dp",
        "state_key": "default",
        "e_alg_policy": "zero",
        "missing_policy": "legacy",
    }
    capture_config["calibration_capture"] = {
        "output_path": str(args.output.resolve()),
        "trials": int(args.trials),
        "period": int(args.period),
        "max_clients": int(args.max_clients),
        "sample_limit": int(args.sample_limit),
        "scope": str(args.scope),
    }
    if not args.real_he:
        capture_config.setdefault("he", {})
        capture_config["he"]["execution"] = "profiled"
        capture_config["he"]["require_real_he"] = False

    seed = int(capture_config["seeds"][0])
    command = build_command(
        capture_config,
        seed=seed,
        policies=[str(args.policy)],
        rounds=(int(args.rounds) if args.rounds is not None else None),
    )
    return command, capture_config


def main() -> None:
    args = parse_args()
    command, capture_config = build_capture_command(args)
    print(json.dumps({
        "capture_output": str(args.output.resolve()),
        "policy": capture_config["policies"][0],
        "seed": capture_config["seeds"][0],
        "scope": capture_config["calibration_capture"]["scope"],
        "trials": capture_config["calibration_capture"]["trials"],
        "period": capture_config["calibration_capture"]["period"],
        "he_execution": capture_config["he"].get("execution", "real"),
    }, indent=2), flush=True)
    print(subprocess.list2cmdline(command), flush=True)
    if not args.dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
