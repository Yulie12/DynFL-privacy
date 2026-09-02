from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.dynamic_training import DynamicFedAvgConfig, run_dynamic_fedavg_training
from dynfed.real_training import RealTrainingConfig
from dynfed.selection import SelectionConfig
from dynfed.utils import timestamped_dir


DEFAULT_POLICIES = (
    "ours",
    "ours_ideal",
    "ours_knee",
    "ours_time_first",
    "ours_acc_first",
    "fixed_dp",
    "fixed_he",
    "random",
    "performance_only",
    "privacy_only",
    "no_protection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run real FedAvg training with per-client dynamic hierarchy/privacy selection."
    )
    parser.add_argument("--output-root", default="out/dynamic_fedavg")
    parser.add_argument("--rounds", type=int, default=100)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--edges", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--initial-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-event-epsilon", type=float, default=0.05)
    parser.add_argument("--dp-emb-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-upd-epsilon", type=float, default=8.0)
    parser.add_argument("--resource-limit", type=float, default=1.35)
    parser.add_argument("--time-limit", type=float, default=8.0)
    parser.add_argument("--risk-limit", type=float, default=0.5)
    parser.add_argument("--require-feasible", action="store_true")
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--iid", action="store_true")
    parser.add_argument(
        "--policies",
        nargs="+",
        default=list(DEFAULT_POLICIES),
        choices=list(DEFAULT_POLICIES),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = timestamped_dir(args.output_root, "dynamic_fedavg")
    selection = SelectionConfig(
        rounds=args.rounds,
        num_clients=args.clients,
        num_edges=args.edges,
        seed=args.seed,
        initial_epsilon=args.initial_epsilon,
        dp_event_epsilon=args.dp_event_epsilon,
        dp_emb_epsilon=args.dp_emb_epsilon,
        dp_upd_epsilon=args.dp_upd_epsilon,
        omega_learning_rate=args.lr,
        resource_limit=args.resource_limit,
        time_limit=args.time_limit,
        risk_limit=args.risk_limit,
        require_feasible=args.require_feasible,
        output_dir=str(output_dir),
    )
    training = RealTrainingConfig(
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        iid=args.iid,
    )
    result = run_dynamic_fedavg_training(
        DynamicFedAvgConfig(selection=selection, training=training, policies=tuple(args.policies))
    )
    print(f"[OK] wrote dynamic FedAvg runs to {result['output_dir']}")
    print(f"[OK] summary table: {result['summary_table']}")


if __name__ == "__main__":
    main()
