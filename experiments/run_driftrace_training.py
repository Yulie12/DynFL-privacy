from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.driftrace_bridge import DriftRaceBridgeConfig, DriftRaceRunSpec, run_driftrace_specs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run DriftRace real-training experiments from this repository without modifying DriftRace."
    )
    parser.add_argument("--driftrace-root", default="E:/YTT/GROUP/DriftRace")
    parser.add_argument("--output-root", default="out/driftrace_real")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--scenario", choices=["smoke", "single_method", "baseline_compare", "eec_compare"], default="smoke")
    parser.add_argument("--method", default="fedavg")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="lenet5")
    parser.add_argument("--global-epoch", type=int, default=1)
    parser.add_argument("--local-epoch", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-cuda", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--isolated-cwd", action="store_true")
    parser.add_argument("--override", action="append", default=[])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    bridge_config = DriftRaceBridgeConfig(
        drift_race_root=args.driftrace_root,
        output_root=args.output_root,
        python_executable=args.python,
        dry_run=args.dry_run,
        isolated_cwd=args.isolated_cwd,
    )
    specs = build_specs(args)
    summaries = run_driftrace_specs(specs, bridge_config)
    for summary in summaries:
        print(
            "[OK]",
            summary["run_name"],
            summary["status"],
            "best_acc=",
            summary.get("best_test_accuracy"),
            "out=",
            summary["output_dir"],
        )


def build_specs(args: argparse.Namespace) -> list[DriftRaceRunSpec]:
    common = {
        "dataset": args.dataset,
        "model": args.model,
        "global_epoch": args.global_epoch,
        "local_epoch": args.local_epoch,
        "seed": args.seed,
        "use_cuda": args.use_cuda,
    }
    if args.scenario == "smoke":
        return [
            DriftRaceRunSpec(
                run_name="fedavg_smoke",
                method="fedavg",
                extra_overrides=[
                    "common.save_log=false",
                    "common.save_learning_curve_plot=false",
                    "common.save_model=false",
                    *args.override,
                ],
                **common,
            )
        ]
    if args.scenario == "single_method":
        return [
            DriftRaceRunSpec(
                run_name=args.method,
                method=args.method,
                extra_overrides=list(args.override),
                **common,
            )
        ]
    if args.scenario == "baseline_compare":
        return [
            DriftRaceRunSpec(run_name="fedavg", method="fedavg", extra_overrides=list(args.override), **common),
            DriftRaceRunSpec(run_name="fedprox", method="fedprox", extra_overrides=list(args.override), **common),
            DriftRaceRunSpec(run_name="fedbuff", method="fedbuff", extra_overrides=list(args.override), **common),
        ]
    return [
        DriftRaceRunSpec(
            run_name="asyncfedadcendedgecloud",
            method="asyncfedadcendedgecloud",
            extra_overrides=["common.buffers=local", *args.override],
            **common,
        ),
        DriftRaceRunSpec(
            run_name="fedproxfixedendedgecloud",
            method="fedproxfixedendedgecloud",
            extra_overrides=list(args.override),
            **common,
        ),
    ]


if __name__ == "__main__":
    main()
