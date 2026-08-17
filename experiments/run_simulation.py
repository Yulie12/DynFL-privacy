from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.config import ExperimentConfig, ModeConfig, OutputConfig, PrivacyConfig, RuntimeConfig, TopologyConfig
from dynfed.training import run_experiment
from dynfed.utils import expand_mode_sweep, expand_privacy_sweep, timestamped_dir
from dynfed.visualization import plot_round_metrics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CPU-only cloud-edge-end privacy/efficiency simulations."
    )
    parser.add_argument("--scenario", choices=["privacy_sweep", "mode_compare"], default="privacy_sweep")
    parser.add_argument("--output-root", default="out/dynfed_privacy_sim")
    parser.add_argument("--rounds", type=int, default=30)
    parser.add_argument("--clients", type=int, default=30)
    parser.add_argument("--edges", type=int, default=5)
    parser.add_argument("--mode", default="LIEIIC")
    parser.add_argument("--privacy", default="DP", choices=["none", "DP", "HE2", "HE3", "dp", "he2", "he3"])
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--no-plot", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = timestamped_dir(args.output_root, args.scenario)
    base = ExperimentConfig(
        topology=TopologyConfig(num_clients=args.clients, num_edges=args.edges),
        runtime=RuntimeConfig(rounds=args.rounds, seed=args.seed),
        privacy=PrivacyConfig(mechanism=args.privacy, epsilon=args.epsilon),
        mode=ModeConfig(name=args.mode),
        output=OutputConfig(output_dir=str(output_root), plot=not args.no_plot),
    )

    if args.scenario == "privacy_sweep":
        configs = expand_privacy_sweep(
            base,
            mechanisms=["DP", "HE2", "HE3"],
            epsilons=[0.5, 1.0, 2.0, 4.0, 8.0],
            output_root=output_root,
        )
    else:
        configs = expand_mode_sweep(
            replace(base, privacy=PrivacyConfig(mechanism=args.privacy, epsilon=args.epsilon)),
            modes=["LIE", "LIC", "LIIE", "LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC"],
            output_root=output_root,
        )

    summaries = []
    for config in configs:
        summaries.append(run_experiment(config))

    _write_summary_table(output_root / "summary_table.csv", summaries)
    if not args.no_plot:
        try:
            plot_round_metrics([Path(item["output_dir"]) for item in summaries], output_root / "summary.png")
        except Exception as exc:
            print(f"[WARN] plotting skipped: {exc}")

    print(f"[OK] wrote {len(summaries)} runs to {output_root}")
    print(f"[OK] summary table: {output_root / 'summary_table.csv'}")


def _write_summary_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
