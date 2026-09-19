from __future__ import annotations

import argparse
import copy
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path

from run_paper_config import DEFAULT_CONFIG, ROOT, build_command
from dynfed.version import CURRENT_EXECUTION_REVISION, CURRENT_UPDATE_PARAMETER_SCOPE


@dataclass(frozen=True)
class SweepCase:
    study: str
    setting: str
    config: dict
    policies: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the versioned controlled experiments used by the paper."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--studies",
        nargs="+",
        choices=["period"],
        default=["period"],
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Override the default three formal seeds.",
    )
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--rerun",
        action="store_true",
        help="Run a case even when a compatible completed result already exists.",
    )
    return parser.parse_args()


def build_cases(base: dict, studies: list[str]) -> list[SweepCase]:
    """Build the final Q94 parameter-sensitivity study.

    Privacy-budget and client-scale sweeps are intentionally not part of the
    formal paper sensitivity suite.  The only parameter sensitivity retained
    by Q94 is the strategy update period S_P, using three representative
    values around the default.
    """
    cases: list[SweepCase] = []
    if "period" in studies:
        for period in (1, 5, 10):
            config = copy.deepcopy(base)
            config["training"]["selection_period"] = period
            config["output_root"] = f"out/paper_v31_final/sensitivity/sp_{period}"
            cases.append(SweepCase("period", str(period), config, ["full_dynfl"]))
    return cases


def completed_case_exists(
    output_root: Path,
    *,
    seed: int,
    policies: list[str],
    rounds: int,
) -> bool:
    if not output_root.exists():
        return False
    for config_path in output_root.rglob("config.json"):
        try:
            config = json.loads(config_path.read_text(encoding="utf-8"))
            selection = config["selection"]
            training = config["training"]
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if int(selection.get("seed", -1)) != seed:
            continue
        if int(selection.get("rounds", -1)) != rounds:
            continue
        if training.get("execution_revision") != CURRENT_EXECUTION_REVISION:
            continue
        if training.get("update_parameter_scope") != CURRENT_UPDATE_PARAMETER_SCOPE:
            continue
        complete = True
        for policy in policies:
            summary_path = config_path.parent / policy / "summary.json"
            if not summary_path.exists():
                complete = False
                break
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if (
                int(summary.get("rounds", -1)) != rounds
                or "accounted_system_time_sec" not in summary
            ):
                complete = False
                break
        if complete:
            return True
    return False


def main() -> None:
    args = parse_args()
    base = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    if base.get("execution_revision") != CURRENT_EXECUTION_REVISION:
        raise RuntimeError(
            f"Controlled experiments require {CURRENT_EXECUTION_REVISION}"
        )
    rounds = int(args.rounds or base["training"]["rounds"])
    manifest: list[dict[str, object]] = []

    for case in build_cases(base, list(dict.fromkeys(args.studies))):
        case_seeds = args.seeds or [40, 42, 44]
        for seed in list(dict.fromkeys(int(value) for value in case_seeds)):
            output_root = ROOT / case.config["output_root"]
            skipped = (
                not args.rerun
                and completed_case_exists(
                    output_root,
                    seed=seed,
                    policies=case.policies,
                    rounds=rounds,
                )
            )
            command = build_command(
                case.config,
                seed=seed,
                policies=case.policies,
                rounds=rounds,
            )
            print(
                f"[{case.study} setting={case.setting} seed={seed}] "
                + ("already complete" if skipped else subprocess.list2cmdline(command)),
                flush=True,
            )
            manifest.append(
                {
                    "study": case.study,
                    "setting": case.setting,
                    "seed": seed,
                    "rounds": rounds,
                    "policies": case.policies,
                    "output_root": case.config["output_root"],
                    "skipped_existing": skipped,
                }
            )
            if not args.dry_run and not skipped:
                subprocess.run(command, cwd=ROOT, check=True)

    manifest_path = ROOT / "out" / "paper_v31_final" / "sensitivity" / "sweep_manifest.json"
    if not args.dry_run:
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(manifest_path)


if __name__ == "__main__":
    main()
