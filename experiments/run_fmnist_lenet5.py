from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.fmnist_lenet5_dynamic import Lenet5Config, run_fmnist_lenet5_training
from dynfed.selection import SelectionConfig
from dynfed.version import CURRENT_EXECUTION_REVISION
from dynfed.utils import timestamped_dir



LEGACY_DEFAULT_POLICIES = (
    "ours", "fixed_dp", "fixed_he", "random", "privacy_only", "no_protection"
)
FUSION_DEFAULT_POLICIES = (
    "ours",
    "individual_optimal",
    "random",
    "fixed_fedavg",
    "fixed_splitfed",
    "fixed_hfl",
    "nsga2",
)


def _resolved_policies(args: argparse.Namespace) -> tuple[str, ...]:
    if args.policies:
        return tuple(args.policies)
    return FUSION_DEFAULT_POLICIES if args.mainline_fusion else LEGACY_DEFAULT_POLICIES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run dynamic cloud-edge-end federated learning collaboration reconfiguration."
    )
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument(
        "--max-new-rounds",
        type=int,
        default=None,
        help=(
            "Stop after this many newly executed rounds without changing the --rounds "
            "privacy-accounting horizon; writes a resumable paused checkpoint."
        ),
    )
    parser.add_argument(
        "--execution-revision",
        default=CURRENT_EXECUTION_REVISION,
    )
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--edges", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--dataset", default="fmnist", choices=["fmnist", "cifar10", "cifar100"])
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--local-epochs", type=int, default=1)
    parser.add_argument(
        "--equal-optimizer-work-control",
        action="store_true",
        help=(
            "Fairness-control only: in fused Method 2, keep total optimizer epochs per "
            "client equal to --local-epochs for every collaboration mode instead of "
            "multiplying by the mode's private-stage count."
        ),
    )
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
            "resnet18_pretrained_head",
            "resnet18_pretrained_head256",
            "resnet18_pretrained_adapter",
            "resnet50_pretrained",
        ],
    )
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--server-step", type=float, default=1.0,
                        help="Scale the final cloud model update after DP/HE/SecAgg aggregation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--iid", action="store_true")
    parser.add_argument(
        "--partition-mode",
        default="client_noniid",
        choices=["iid", "dirichlet", "client_noniid", "edge_label_skew", "extreme_edge_label_skew"],
    )
    parser.add_argument(
        "--dirichlet-alpha", type=float, default=0.5,
        help="Dirichlet concentration alpha when --partition-mode dirichlet (paper: 0.5 or 0.1).",
    )
    parser.add_argument("--selection-period", type=int, default=5)
    parser.add_argument(
        "--exclude-modes",
        nargs="+",
        default=[],
        help="Exclude collaboration modes from dynamic selection, e.g. --exclude-modes LIE",
    )
    parser.add_argument("--initial-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-event-epsilon", type=float, default=0.05)
    parser.add_argument("--dp-emb-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-upd-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--dp-feature-clip-norm", type=float, default=None,
        help="Feature/embedding clipping norm C_f. Defaults to --dp-clip-norm for backward compatibility.",
    )
    parser.add_argument(
        "--dp-update-clip-norm", type=float, default=None,
        help="Model-update clipping norm C_u. Defaults to --dp-clip-norm for backward compatibility.",
    )
    parser.add_argument(
        "--dp-sample-optimizer-clip-norm",
        type=float,
        default=1.0,
        help=(
            "Joint per-sample parameter-gradient clipping norm C_s used only "
            "by the Sample-level DP-SGD optimizer path."
        ),
    )
    parser.add_argument("--dp-noise-multiplier", type=float, default=0.0002)
    parser.add_argument(
        "--dp-accounting-mode",
        default="rdp_auto",
        choices=["rdp_auto", "rdp_manual"],
        help="Auto calibrates Gaussian noise from the fixed total target and round horizon.",
    )
    parser.add_argument(
        "--privacy-unit",
        default="sample",
        choices=["client", "sample"],
        help=(
            "Privacy adjacency unit. Formal paper runs use 'sample'. "
            "The legacy client-level path remains available only for backward-compatible ablations."
        ),
    )
    parser.add_argument("--dp-feature-epsilon-budget", type=float, default=None)
    parser.add_argument("--dp-update-epsilon-budget", type=float, default=None)
    parser.add_argument("--dp-feature-noise-multiplier", type=float, default=None)
    parser.add_argument("--dp-update-noise-multiplier", type=float, default=None)
    parser.add_argument("--dp-sample-epsilon-budget", type=float, default=None)
    parser.add_argument("--dp-sample-embedding-noise-multiplier", type=float, default=None)
    parser.add_argument("--dp-sample-label-grad-noise-multiplier", type=float, default=None)
    parser.add_argument("--dp-sample-optimizer-noise-multiplier", type=float, default=None)
    parser.add_argument(
        "--learning-objective",
        default="legacy_fusion_dp",
        choices=["legacy_fusion_dp", "joint_calibration"],
        help=(
            "Pareto learning objective. joint_calibration uses the latest joint update-space "
            "proxy from an offline paired clean/private calibration table."
        ),
    )
    parser.add_argument(
        "--joint-calibration-path",
        default=None,
        help="Torch calibration table built from offline paired clean/private trajectories.",
    )
    parser.add_argument(
        "--joint-calibration-state-key",
        default="default",
        help=(
            "Calibration state key. Use 'auto_round' to query round:<t>; the table may "
            "fall back to the nearest calibrated round or default."
        ),
    )
    parser.add_argument(
        "--joint-calibration-e-alg-policy",
        default="table",
        choices=["table", "zero"],
        help=(
            "table requires a calibrated ideal/reference update and keeps e_alg explicitly; "
            "zero is an empirical approximation that must be validated against paired-MC."
        ),
    )
    parser.add_argument(
        "--joint-calibration-missing-policy",
        default="error",
        choices=["error", "legacy"],
        help="Whether missing calibration data is fatal or falls back to the retired legacy objective.",
    )
    parser.add_argument(
        "--joint-calibration-capture-path",
        default=None,
        help=(
            "Opt-in offline/periodic matched clean/private capture output (.pt). "
            "This is calibration work, not an online selector input measurement."
        ),
    )
    parser.add_argument(
        "--joint-calibration-capture-trials",
        type=int,
        default=0,
        help="Private DP-RNG trials per captured (round, client, mode) cell; use >=2 when capture is enabled.",
    )
    parser.add_argument(
        "--joint-calibration-capture-period",
        type=int,
        default=10,
        help="Capture every N rounds in a dedicated calibration pass.",
    )
    parser.add_argument(
        "--joint-calibration-capture-max-clients",
        type=int,
        default=8,
        help="Maximum held-out clients sampled per capture round; 0 means all clients.",
    )
    parser.add_argument(
        "--joint-calibration-capture-sample-limit",
        type=int,
        default=64,
        help="Maximum held-out calibration samples per sampled client; 0 means all held-out samples.",
    )
    parser.add_argument(
        "--joint-calibration-capture-scope",
        default="candidate_modes",
        choices=["selected", "candidate_modes"],
        help=(
            "selected captures only the chosen mode; candidate_modes captures one feasible Sample-DP "
            "candidate for every available mode of each sampled client."
        ),
    )
    parser.add_argument("--dp-delta", type=float, default=1e-5)
    parser.add_argument("--dp-update-mode", default="upd_only", choices=["upd_only", "off"])
    parser.add_argument("--dp-release-calibration", default="tex_packet", choices=["tex_packet", "legacy_aggregate", "global_release"])
    parser.add_argument("--update-mechanisms", nargs="+", choices=["dp", "he3", "dp_he3"], default=["dp", "he3"])
    parser.add_argument(
        "--liie-edge-dp-plan", choices=["independent", "aggregate"],
        default="independent",
        help="Experimental opt-in Edge SecAgg DP ablation; not yet a Pareto action.",
    )
    parser.add_argument(
        "--cloud-dp-plan", choices=["legacy", "packet", "aggregate", "pareto"],
        default="legacy",
        help="Opt-in whole-profile Cloud DP release selection; legacy preserves old execution.",
    )
    parser.add_argument("--update-protection-goal", choices=["packet_protection", "released_model_dp"], default="packet_protection")
    parser.add_argument(
        "--mainline-fusion",
        action="store_true",
        help=(
            "LEGACY/REMOVED: the Method2 global-release overlay is no longer a valid DynFL mainline option."
        ),
    )
    parser.add_argument(
        "--trusted-edge-split-execution",
        action="store_true",
        help=(
            "Explicitly declare End and Edge as one trusted execution domain; "
            "required for --trusted-lie-joint-sample-dp."
        ),
    )
    parser.add_argument(
        "--trusted-lie-joint-sample-dp",
        action="store_true",
        help=(
            "Opt into joint per-sample gradient DP-SGD for trusted LIE only. "
            "Requires --trusted-edge-split-execution and --privacy-unit sample; "
            "incompatible with the old joint-calibration learning objective."
        ),
    )
    parser.add_argument("--he-backend", default="none", choices=["none", "seal", "tenseal"])
    parser.add_argument("--he-execution", default="real", choices=["real", "profiled"])
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
    parser.add_argument(
        "--split-interaction-mode", choices=["fixed", "workload"], default="fixed",
        help="Use fixed L_block_cycles or workload-derived split minibatch interactions.",
    )
    parser.add_argument(
        "--split-batch-size", type=int, default=128,
        help="Batch size used to derive split interaction count in workload mode.",
    )
    parser.add_argument(
        "--fl-first-split-on-demand", action="store_true",
        help=(
            "For dynamic mode selection, suppress split/I modes whenever at least one "
            "full-local/update mode is device-feasible; fixed baselines are unchanged."
        ),
    )
    parser.add_argument(
        "--edge-only-requires-fast-deadline",
        action="store_true",
        help=(
            "Reserve Edge-only modes for clients with an explicit fast-response "
            "deadline; ordinary clients must use Cloud-reaching modes."
        ),
    )
    parser.add_argument("--time-limit", type=float, default=8.0)
    parser.add_argument(
        "--fast-client-deadlines",
        nargs="*",
        default=[],
        metavar="CLIENT_ID:SECONDS",
        help="Per-client hard fast-response QoS deadlines, e.g. 0:2.5 3:4.0.",
    )
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
    parser.add_argument("--resource-scenario", choices=["none", "communication", "compute"], default="none")
    parser.add_argument("--constrained-start-fraction", type=float, default=1.0 / 3.0)
    parser.add_argument("--constrained-end-fraction", type=float, default=2.0 / 3.0)
    parser.add_argument("--communication-constrained-multiplier", type=float, default=0.35)
    parser.add_argument("--compute-constrained-multiplier", type=float, default=2.0)
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
        default=None,
        help="Policies to run. Mainline fusion defaults exclude privacy-bypass controls.",
    )
    return parser.parse_args()


def _validate_trusted_lie_cli(args: argparse.Namespace) -> None:
    """Fail closed before constructing the training run or writing output."""
    if args.trusted_edge_split_execution and not args.trusted_lie_joint_sample_dp:
        raise ValueError(
            "--trusted-edge-split-execution requires explicit "
            "--trusted-lie-joint-sample-dp opt-in in this experimental CLI"
        )
    if not args.trusted_lie_joint_sample_dp:
        return
    if not args.trusted_edge_split_execution:
        raise ValueError("trusted LIE requires --trusted-edge-split-execution")
    if args.privacy_unit != "sample":
        raise ValueError("trusted LIE requires --privacy-unit sample")
    if args.learning_objective == "joint_calibration":
        raise ValueError(
            "trusted LIE requires a new learning-proxy calibration; "
            "the old noisy-embedding joint calibration is incompatible"
        )


def main() -> None:
    args = parse_args()
    _validate_trusted_lie_cli(args)
    policies = _resolved_policies(args)
    feature_clip_norm = float(
        args.dp_feature_clip_norm if args.dp_feature_clip_norm is not None else args.dp_clip_norm
    )
    update_clip_norm = float(
        args.dp_update_clip_norm if args.dp_update_clip_norm is not None else args.dp_clip_norm
    )
    sample_optimizer_clip_norm = float(
        args.dp_sample_optimizer_clip_norm
    )
    if feature_clip_norm <= 0.0:
        raise ValueError("--dp-feature-clip-norm must be positive")
    if update_clip_norm <= 0.0:
        raise ValueError("--dp-update-clip-norm must be positive")
    if sample_optimizer_clip_norm <= 0.0:
        raise ValueError(
            "--dp-sample-optimizer-clip-norm must be positive"
        )
    if args.max_new_rounds is not None and args.max_new_rounds < 1:
        raise ValueError("--max-new-rounds must be positive")
    output_root = timestamped_dir(args.output_root, "lenet5_dynamic_newtex202608")
    output_root.mkdir(parents=True, exist_ok=True)
    status_payload = {
        "status": "preparing",
        "message": "Preparing data, model, privacy accountant, and HE backend.",
        "run_dir": str(output_root),
        "round": 0,
        "rounds": int(args.rounds),
        "max_new_rounds": args.max_new_rounds,
        "dataset": args.dataset,
        "model": args.model,
        "policies": list(policies),
        "updated_at": time.time(),
    }
    for status_path in (output_root / "live_status.json", output_root.parent / "live_status.json"):
        status_path.write_text(json.dumps(status_payload, indent=2), encoding="utf-8")

    fast_client_deadlines: list[tuple[int, float]] = []
    for item in args.fast_client_deadlines:
        try:
            client_text, deadline_text = item.split(":", 1)
            fast_client_deadlines.append((int(client_text), float(deadline_text)))
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"invalid --fast-client-deadlines entry {item!r}; expected CLIENT_ID:SECONDS"
            ) from exc

    normalized_model_for_selection = (
        str(args.model)
        .strip()
        .lower()
        .replace("-", "")
        .replace("_", "")
    )

    # Torchvision pretrained split End contains conv1/bn1/layer1/layer2.
    # The current training policy only enables later layers / heads, so the
    # split End has no trainable parameters for these model families.
    frozen_split_end_models = {
        "resnet18pretrainedadapter",
        "resnet18pretrained",
        "resnet18pretrainedhead",
        "resnet18pretrainedhead256",
        "resnet18pretrainedlayer4head",
        "resnet50pretrained",
    }

    split_end_optimizer_enabled = (
        normalized_model_for_selection
        not in frozen_split_end_models
    )

    selection = SelectionConfig(
        rounds=args.rounds,
        num_clients=args.clients,
        num_edges=args.edges,
        seed=args.seed,
        excluded_modes=tuple(args.exclude_modes),
        initial_epsilon=args.initial_epsilon,
        dp_event_epsilon=args.dp_event_epsilon,
        dp_emb_epsilon=args.dp_emb_epsilon,
        dp_upd_epsilon=args.dp_upd_epsilon,
        dp_noise_multiplier=args.dp_noise_multiplier,
        dp_accounting_mode=args.dp_accounting_mode,
        privacy_unit=args.privacy_unit,
        dp_feature_epsilon_budget=args.dp_feature_epsilon_budget,
        dp_update_epsilon_budget=args.dp_update_epsilon_budget,
        dp_feature_noise_multiplier=args.dp_feature_noise_multiplier,
        dp_update_noise_multiplier=args.dp_update_noise_multiplier,
        dp_sample_epsilon_budget=args.dp_sample_epsilon_budget,
        dp_sample_embedding_noise_multiplier=args.dp_sample_embedding_noise_multiplier,
        dp_sample_label_grad_noise_multiplier=args.dp_sample_label_grad_noise_multiplier,
        dp_sample_optimizer_noise_multiplier=args.dp_sample_optimizer_noise_multiplier,
        dp_delta=args.dp_delta,
        omega_learning_rate=args.lr,
        omega_feature_clip_norm=feature_clip_norm,
        omega_update_clip_norm=update_clip_norm,
        resource_limit=args.resource_limit,
        memory_limit=args.memory_limit,
        split_interaction_mode=args.split_interaction_mode,
        split_batch_size=args.split_batch_size,
        fl_first_split_on_demand=args.fl_first_split_on_demand,
        edge_only_requires_fast_deadline=args.edge_only_requires_fast_deadline,
        time_limit=args.time_limit,
        fast_client_deadlines=tuple(fast_client_deadlines),
        risk_limit=args.risk_limit,
        aggregation_fraction=args.aggregation_fraction,
        privacy_local_epochs=args.local_epochs,
        split_end_optimizer_enabled=split_end_optimizer_enabled,
        trusted_edge_split_execution=args.trusted_edge_split_execution,
        trusted_lie_joint_sample_dp=args.trusted_lie_joint_sample_dp,
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
        resource_scenario=args.resource_scenario,
        constrained_start_fraction=args.constrained_start_fraction,
        constrained_end_fraction=args.constrained_end_fraction,
        communication_constrained_multiplier=args.communication_constrained_multiplier,
        compute_constrained_multiplier=args.compute_constrained_multiplier,
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
        update_mechanism_options=tuple(args.update_mechanisms),
        liie_edge_dp_plan=args.liie_edge_dp_plan,
        cloud_dp_plan=args.cloud_dp_plan,
        update_protection_goal=(
            "released_model_dp" if args.mainline_fusion else args.update_protection_goal
        ),
        mainline_fusion=args.mainline_fusion,
        learning_objective=args.learning_objective,
        joint_calibration_path=args.joint_calibration_path,
        joint_calibration_state_key=args.joint_calibration_state_key,
        joint_calibration_e_alg_policy=args.joint_calibration_e_alg_policy,
        joint_calibration_missing_policy=args.joint_calibration_missing_policy,
    )
    train_config = Lenet5Config(
        execution_revision=args.execution_revision,
        dataset_name=args.dataset,
        model_name=args.model,
        local_epochs=args.local_epochs,
        equal_optimizer_work_control=args.equal_optimizer_work_control,
        learning_rate=args.lr,
        server_step=args.server_step,
        iid=args.iid,
        partition_mode="iid" if args.iid else args.partition_mode,
        dirichlet_alpha=args.dirichlet_alpha,
        selection_period=args.selection_period,
        dp_clip_norm=args.dp_clip_norm,
        dp_feature_clip_norm=feature_clip_norm,
        dp_update_clip_norm=update_clip_norm,
        dp_sample_optimizer_clip_norm=sample_optimizer_clip_norm,
        dp_noise_multiplier=args.dp_noise_multiplier,
        dp_update_mode=args.dp_update_mode,
        dp_release_calibration=args.dp_release_calibration,
        device=args.device,
        he_backend=args.he_backend,
        he_execution=args.he_execution,
        he_local_deps=args.he_local_deps,
        require_real_he=(args.require_real_he or args.mainline_fusion),
        he_aggregation_size=args.he_aggregation_size,
        he_workers=max(1, args.he_workers),
        executor=args.executor,
        executor_workers=args.executor_workers,
        joint_calibration_capture_path=args.joint_calibration_capture_path,
        joint_calibration_capture_trials=args.joint_calibration_capture_trials,
        joint_calibration_capture_period=args.joint_calibration_capture_period,
        joint_calibration_capture_max_clients=(
            None if args.joint_calibration_capture_max_clients == 0
            else args.joint_calibration_capture_max_clients
        ),
        joint_calibration_capture_sample_limit=(
            None if args.joint_calibration_capture_sample_limit == 0
            else args.joint_calibration_capture_sample_limit
        ),
        joint_calibration_capture_scope=args.joint_calibration_capture_scope,
    )

    result = run_fmnist_lenet5_training(
        selection=selection,
        train_config=train_config,
        data_root=args.data_root,
        train_limit=args.train_limit,
        test_limit=args.test_limit,
        resume_from_run=args.resume_from_run,
        max_new_rounds=args.max_new_rounds,
        policies=policies,
    )
    print(f"[{result['status'].upper()}] summary table: {result['summary_table']}")


if __name__ == "__main__":
    main()
