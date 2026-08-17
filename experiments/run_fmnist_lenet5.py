from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.fmnist_lenet5_dynamic import Lenet5Config, run_fmnist_lenet5_training
from dynfed.selection import SelectionConfig
from dynfed.utils import timestamped_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FMNIST + LeNet5 dynamic FedAvg with per-client mode selection."
    )
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--edges", type=int, default=2)
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--dataset", default="fmnist", choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument(
        "--model",
        default="lenet5",
        choices=[
            "lenet5",
            "smallcnn",
            "avgcnn",
            "tinyresnet",
            "resnet18",
            "resnet50",
            "resnet18_pretrained",
            "resnet50_pretrained",
        ],
    )
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iid", action="store_true")
    parser.add_argument(
        "--partition-mode",
        default="client_noniid",
        choices=["iid", "client_noniid", "edge_label_skew", "extreme_edge_label_skew"],
    )
    parser.add_argument("--selection-period", type=int, default=1)
    parser.add_argument("--initial-epsilon", type=float, default=10.0)
    parser.add_argument("--dp-event-epsilon", type=float, default=0.01)
    parser.add_argument("--dp-clip-norm", type=float, default=20.0)
    parser.add_argument("--dp-noise-multiplier", type=float, default=0.005)
    parser.add_argument("--dp-update-mode", default="any_dp", choices=["any_dp", "upd_only", "off"])
    parser.add_argument("--he-backend", default="none", choices=["none", "seal", "tenseal"])
    parser.add_argument("--he-local-deps", default=".he_deps")
    parser.add_argument("--require-real-he", action="store_true")
    parser.add_argument("--resource-limit", type=float, default=1.35)
    parser.add_argument("--time-limit", type=float, default=8.0)
    parser.add_argument("--risk-limit", type=float, default=0.5)
    parser.add_argument("--aggregation-fraction", type=float, default=0.5)
    parser.add_argument("--client-heterogeneity", type=float, default=2.0)
    parser.add_argument("--edge-heterogeneity", type=float, default=1.5)
    parser.add_argument("--require-feasible", action="store_true")
    parser.add_argument("--require-cloud", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--output-root", default="out/fmnist_lenet5")
    parser.add_argument(
        "--resume-from-run",
        default=None,
        help="Existing lenet5_dynamic run directory with per-policy checkpoint.pt files.",
    )
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["ours", "fixed_dp", "fixed_he", "random", "privacy_only", "no_protection"],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = timestamped_dir(args.output_root, "lenet5_dynamic_newtex202608")

    selection = SelectionConfig(
        rounds=args.rounds,
        num_clients=args.clients,
        num_edges=args.edges,
        seed=args.seed,
        initial_epsilon=args.initial_epsilon,
        dp_event_epsilon=args.dp_event_epsilon,
        resource_limit=args.resource_limit,
        time_limit=args.time_limit,
        risk_limit=args.risk_limit,
        aggregation_fraction=args.aggregation_fraction,
        client_heterogeneity=args.client_heterogeneity,
        edge_heterogeneity=args.edge_heterogeneity,
        require_feasible=args.require_feasible,
        require_cloud_participation=args.require_cloud,
        output_dir=str(output_root),
    )
    train_config = Lenet5Config(
        dataset_name=args.dataset,
        model_name=args.model,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        iid=args.iid,
        partition_mode="iid" if args.iid else args.partition_mode,
        selection_period=args.selection_period,
        dp_clip_norm=args.dp_clip_norm,
        dp_noise_multiplier=args.dp_noise_multiplier,
        dp_update_mode=args.dp_update_mode,
        device=args.device,
        he_backend=args.he_backend,
        he_local_deps=args.he_local_deps,
        require_real_he=args.require_real_he,
    )

    result = run_fmnist_lenet5_training(
        selection=selection,
        train_config=train_config,
        data_root=args.data_root,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        resume_from_run=args.resume_from_run,
        policies=tuple(args.policies),
    )
    print(f"[OK] summary table: {result['summary_table']}")


if __name__ == "__main__":
    main()
