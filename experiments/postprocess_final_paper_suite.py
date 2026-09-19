from __future__ import annotations

"""Post-process the frozen Q98 formal suite after all training runs finish.

This script only aggregates existing formal results, generates Tables I/II,
and runs the Step25 readiness gate.  It never launches training and never
fills missing results with placeholders.
"""

import argparse
import subprocess
import sys
from pathlib import Path

try:
    from paper_final_plan import DATA_DISTRIBUTIONS, FORMAL_SEEDS, RESOURCE_SCENARIOS
except ImportError:  # pragma: no cover
    from experiments.paper_final_plan import DATA_DISTRIBUTIONS, FORMAL_SEEDS, RESOURCE_SCENARIOS

ROOT = Path(__file__).resolve().parents[1]
FINAL_ROOT = ROOT / "out" / "paper_v31_final"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate and validate the frozen Q98 paper outputs.")
    parser.add_argument("--seeds", type=int, nargs="+", default=list(FORMAL_SEEDS))
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def build_commands(*, seeds: list[int], rounds: int) -> list[list[str]]:
    py = sys.executable
    seed_args = [str(seed) for seed in seeds]
    commands: list[list[str]] = []

    for setting, _partition, _alpha in DATA_DISTRIBUTIONS:
        case_root = FINAL_ROOT / "fig1_main" / setting
        commands.append([
            py, str(ROOT / "experiments" / "aggregate_multiseed_results.py"),
            "--root", str(case_root), "--seeds", *seed_args,
            "--rounds", str(rounds), "--clients", "100", "--edges", "10",
            "--selection-period", "1", "--privacy-budget", "8",
            "--output-dir", str(case_root / "aggregate"),
        ])

    for scenario in RESOURCE_SCENARIOS:
        case_root = FINAL_ROOT / "fig2_dynamic_resources" / scenario
        commands.append([
            py, str(ROOT / "experiments" / "aggregate_multiseed_results.py"),
            "--root", str(case_root), "--seeds", *seed_args,
            "--policies", "full_dynfl", "--rounds", str(rounds),
            "--clients", "100", "--edges", "10", "--selection-period", "1",
            "--privacy-budget", "8", "--output-dir", str(case_root / "aggregate"),
        ])

    commands.append([
        py, str(ROOT / "experiments" / "aggregate_final_privacy.py"),
        "--seeds", *seed_args, "--rounds", str(rounds),
    ])

    fast_root = FINAL_ROOT / "fig4_fast_response"
    commands.append([
        py, str(ROOT / "experiments" / "aggregate_multiseed_results.py"),
        "--root", str(fast_root), "--seeds", *seed_args,
        "--rounds", str(rounds), "--clients", "100", "--edges", "10",
        "--selection-period", "1", "--privacy-budget", "8",
        "--output-dir", str(fast_root / "aggregate"),
    ])

    commands.append([
        py, str(ROOT / "experiments" / "aggregate_controlled_results.py"),
        "--root", str(FINAL_ROOT / "fig5_sp"), "--seeds", *seed_args,
        "--rounds", str(rounds), "--output-dir", str(FINAL_ROOT / "sensitivity_aggregate"),
    ])
    commands.append([py, str(ROOT / "experiments" / "generate_table1_parameters.py")])
    commands.append([py, str(ROOT / "experiments" / "generate_table2_methods_results.py")])
    commands.append([
        py, str(ROOT / "experiments" / "validate_final_paper_outputs.py"), "--require-complete",
    ])
    return commands


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    commands = build_commands(seeds=seeds, rounds=int(args.rounds))
    for command in commands:
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
