from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dynamic cloud-edge-end federated learning collaboration reconfiguration."
    )
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument(
        "--execution-revision",
        default="paper_flow_v22_wall_raw_nsga",
    )
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--edges", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--dataset", default="fmnist", choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--local-epochs", type=int, default=1)
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
    parser.add_argument("--selection-period", type=int, default=5)
    parser.add_argument("--initial-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-event-epsilon", type=float, default=0.05)
    parser.add_argument("--dp-emb-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-upd-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-clip-norm", type=float, default=1.0)
    parser.add_argument("--dp-noise-multiplier", type=float, default=0.0002)
    parser.add_argument(
        "--dp-accounting-mode",
        default="rdp_auto",
        choices=["rdp_auto", "rdp_manual"],
        help="Auto calibrates Gaussian noise from the fixed total target and round horizon.",
    )
    parser.add_argument("--dp-feature-epsilon-budget", type=float, default=None)
    parser.add_argument("--dp-update-epsilon-budget", type=float, default=None)
    parser.add_argument("--dp-feature-noise-multiplier", type=float, default=None)
    parser.add_argument("--dp-update-noise-multiplier", type=float, default=None)
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-update-mode", default="upd_only", choices=["upd_only", "off"])
    parser.add_argument("--he-backend", default="none", choices=["none", "seal", "tenseal"])
    parser.add_argument("--he-local-deps", default=".he_deps")
    parser.add_argument("--require-real-he", action="store_true")
    parser.add_argument(
        "--he-aggregation-size",
        type=int,
        default=0,
        help="Use 0 for complete update encryption. Partial CKKS aggregation is rejected.",
    )
    parser.add_argument(
        "--he-workers",
        type=int,
        default=1,
        help="Number of CPU processes used for full SEAL CKKS parameter chunks.",
    )
    parser.add_argument("--resource-limit", type=float, default=1.35)
    parser.add_argument("--memory-limit", type=float, default=1.35)
    parser.add_argument("--time-limit", type=float, default=8.0)
    parser.add_argument("--risk-limit", type=float, default=0.5)
    parser.add_argument("--aggregation-fraction", type=float, default=1.0)
    parser.add_argument("--pareto-archive-size", type=int, default=16)
    parser.add_argument("--pareto-beam-size", type=int, default=4)
    parser.add_argument("--pareto-max-iters", type=int, default=50)
    parser.add_argument("--pareto-neighbor-top-k", type=int, default=0)
    parser.add_argument("--pareto-conflict-only", action="store_true")
    parser.add_argument("--cloud-fusion-xi", type=float, default=0.2)
    parser.add_argument("--cloud-fusion-eps", type=float, default=0.05)
    parser.add_argument("--edge-aggregation-beta", type=float, default=0.01)
    parser.add_argument("--edge-aggregation-fixed", type=float, default=0.02)
    parser.add_argument("--cloud-aggregation-beta", type=float, default=0.015)
    parser.add_argument("--cloud-aggregation-fixed", type=float, default=0.04)
    parser.add_argument("--client-heterogeneity", type=float, default=2.0)
    parser.add_argument("--edge-heterogeneity", type=float, default=1.5)
    parser.add_argument("--network-jitter", type=float, default=0.25)
    parser.add_argument("--network-periodic-amplitude", type=float, default=0.2)
    parser.add_argument("--network-period-rounds", type=float, default=3.0)
    parser.add_argument("--end-edge-rate-mb-s", type=float, default=5.0)
    parser.add_argument("--end-cloud-rate-mb-s", type=float, default=2.2)
    parser.add_argument("--edge-cloud-rate-mb-s", type=float, default=8.0)
    parser.add_argument("--end-edge-base-latency-sec", type=float, default=0.015)
    parser.add_argument("--end-cloud-base-latency-sec", type=float, default=0.04)
    parser.add_argument("--edge-cloud-base-latency-sec", type=float, default=0.01)
    parser.add_argument("--require-feasible", action="store_true")
    parser.add_argument("--require-cloud", action="store_true")
    parser.add_argument("--require-edge-cloud-coverage", action="store_true")
    parser.add_argument("--min-edge-cloud-fusion-ratio", type=float, default=0.0)
    parser.add_argument("--enforce-cloud-dp-stability", action="store_true")
    parser.add_argument("--cloud-dp-stability-threshold", type=float, default=1.0)
    parser.add_argument("--assume-encoder-feasible", action="store_true")
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--executor", default="serial", choices=["serial", "process_pool"])
    parser.add_argument("--executor-workers", type=int, default=None)
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
    # Imported here (not at module top) so Windows spawn children -- which re-run
    # this file as __mp_main__ -- never load torch: they only need the torch-free
    # SEAL workers, and importing torch in each child exceeds the commit limit.
    from dynfed.fmnist_lenet5_dynamic import Lenet5Config, run_fmnist_lenet5_training
    from dynfed.selection import SelectionConfig
    from dynfed.utils import timestamped_dir

    args = parse_args()
    output_root = timestamped_dir(args.output_root, "lenet5_dynamic_newtex202608")
    output_root.mkdir(parents=True, exist_ok=True)
    status_payload = {
        "status": "preparing",
        "message": "Preparing data, model, privacy accountant, and HE backend.",
        "run_dir": str(output_root),
        "round": 0,
        "rounds": int(args.rounds),
        "dataset": args.dataset,
        "model": args.model,
        "policies": list(args.policies),
        "updated_at": time.time(),
    }
    for status_path in (output_root / "live_status.json", output_root.parent / "live_status.json"):
        status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    selection = SelectionConfig(
        rounds=args.rounds,
        num_clients=args.clients,
        num_edges=args.edges,
        seed=args.seed,
        initial_epsilon=args.initial_epsilon,
        dp_event_epsilon=args.dp_event_epsilon,
        dp_emb_epsilon=args.dp_emb_epsilon,
        dp_upd_epsilon=args.dp_upd_epsilon,
        dp_noise_multiplier=args.dp_noise_multiplier,
        dp_accounting_mode=args.dp_accounting_mode,
        dp_feature_epsilon_budget=args.dp_feature_epsilon_budget,
        dp_update_epsilon_budget=args.dp_update_epsilon_budget,
        dp_feature_noise_multiplier=args.dp_feature_noise_multiplier,
        dp_update_noise_multiplier=args.dp_update_noise_multiplier,
        dp_delta=args.dp_delta,
        omega_learning_rate=args.lr,
        omega_feature_clip_norm=args.dp_clip_norm,
        omega_update_clip_norm=args.dp_clip_norm,
        resource_limit=args.resource_limit,
        memory_limit=args.memory_limit,
        time_limit=args.time_limit,
        risk_limit=args.risk_limit,
        aggregation_fraction=args.aggregation_fraction,
        pareto_archive_size=args.pareto_archive_size,
        pareto_beam_size=args.pareto_beam_size,
        pareto_max_iters=args.pareto_max_iters,
        pareto_neighbor_top_k=args.pareto_neighbor_top_k,
        pareto_conflict_only=args.pareto_conflict_only,
        cloud_fusion_xi=args.cloud_fusion_xi,
        cloud_fusion_eps=args.cloud_fusion_eps,
        edge_aggregation_beta=args.edge_aggregation_beta,
        edge_aggregation_fixed=args.edge_aggregation_fixed,
        cloud_aggregation_beta=args.cloud_aggregation_beta,
        cloud_aggregation_fixed=args.cloud_aggregation_fixed,
        client_heterogeneity=args.client_heterogeneity,
        edge_heterogeneity=args.edge_heterogeneity,
        network_jitter=args.network_jitter,
        network_periodic_amplitude=args.network_periodic_amplitude,
        network_period_rounds=args.network_period_rounds,
        end_edge_rate_mb_s=args.end_edge_rate_mb_s,
        end_cloud_rate_mb_s=args.end_cloud_rate_mb_s,
        edge_cloud_rate_mb_s=args.edge_cloud_rate_mb_s,
        end_edge_base_latency_sec=args.end_edge_base_latency_sec,
        end_cloud_base_latency_sec=args.end_cloud_base_latency_sec,
        edge_cloud_base_latency_sec=args.edge_cloud_base_latency_sec,
        require_feasible=args.require_feasible,
        require_cloud_participation=args.require_cloud,
        require_edge_cloud_coverage=args.require_edge_cloud_coverage,
        min_edge_cloud_fusion_ratio=args.min_edge_cloud_fusion_ratio,
        enforce_cloud_dp_stability=args.enforce_cloud_dp_stability,
        cloud_dp_stability_threshold=args.cloud_dp_stability_threshold,
        assume_encoder_feasible=args.assume_encoder_feasible,
        output_dir=str(output_root),
    )
    train_config = Lenet5Config(
        execution_revision=args.execution_revision,
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
        he_aggregation_size=args.he_aggregation_size,
        he_workers=max(1, args.he_workers),
        executor=args.executor,
        executor_workers=args.executor_workers,
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
