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
from dynfed.real_training import RealTrainingConfig, run_real_federated_training
from dynfed.utils import timestamped_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run real CPU federated training on sklearn digits.")
    parser.add_argument("--scenario", choices=["single", "privacy_sweep", "mode_compare"], default="single")
    parser.add_argument("--output-root", default="out/real_training")
    parser.add_argument("--rounds", type=int, default=40)
    parser.add_argument("--clients", type=int, default=30)
    parser.add_argument("--edges", type=int, default=5)
    parser.add_argument("--mode", default="LIEIIC")
    parser.add_argument("--privacy", default="none", choices=["none", "DP", "HE2", "HE3", "dp", "he2", "he3"])
    parser.add_argument("--epsilon", type=float, default=4.0)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = timestamped_dir(args.output_root, args.scenario)
    train_config = RealTrainingConfig(
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        iid=args.iid,
    )
    base = ExperimentConfig(
        topology=TopologyConfig(num_clients=args.clients, num_edges=args.edges),
        runtime=RuntimeConfig(rounds=args.rounds, seed=args.seed),
        privacy=PrivacyConfig(mechanism=args.privacy, epsilon=args.epsilon),
        mode=ModeConfig(name=args.mode),
        output=OutputConfig(output_dir=str(output_root), plot=False),
    )

    configs = []
    if args.scenario == "single":
        configs = [replace(base, output=replace(base.output, output_dir=str(output_root / f"{args.mode}_{args.privacy}")))]
    elif args.scenario == "privacy_sweep":
        for mechanism, epsilon in [("none", 4.0), ("DP", 1.0), ("DP", 4.0), ("DP", 8.0), ("HE2", 4.0), ("HE3", 4.0)]:
            name = f"{args.mode}_{mechanism}_eps{epsilon}".replace(".", "p")
            configs.append(
                replace(
                    base,
                    privacy=PrivacyConfig(mechanism=mechanism, epsilon=epsilon),
                    output=replace(base.output, output_dir=str(output_root / name)),
                )
            )
    else:
        for mode in ["LIE", "LIC", "LIIE", "LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC"]:
            configs.append(
                replace(
                    base,
                    mode=ModeConfig(name=mode),
                    output=replace(base.output, output_dir=str(output_root / mode)),
                )
            )

    summaries = [run_real_federated_training(config, train_config) for config in configs]
    _write_summary_table(output_root / "summary_table.csv", summaries)
    print(f"[OK] wrote {len(summaries)} real-training runs to {output_root}")
    print(f"[OK] summary table: {output_root / 'summary_table.csv'}")


def _write_summary_table(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
