from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper_v30_cifar10_resnet18.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.he_backend import (
    CKKS_COEFF_MOD_BIT_SIZES,
    CKKS_POLY_MODULUS_DEGREE,
    CKKS_SCALE_BITS,
)
from dynfed.version import CURRENT_EXECUTION_REVISION

FORMAL_BASELINE_POLICIES = (
    "fixed_mode_fixed_privacy",
    "dynamic_mode_fixed_privacy",
    "fixed_mode_dynamic_privacy",
    "full_dynfl",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a versioned paper experiment configuration."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--server-step", type=float,
                        help="Diagnostic override of the cloud server step (default 1.0)")
    parser.add_argument("--he-execution", choices=["real", "profiled"])
    parser.add_argument("--liie-edge-dp-plan", choices=["independent", "aggregate"])
    parser.add_argument("--cloud-dp-plan", choices=["legacy", "packet", "aggregate", "pareto"])
    parser.add_argument("--output-root")
    parser.add_argument(
        "--equal-optimizer-work-control",
        action="store_true",
        help="Run the supplementary equal-optimizer-work fairness control; main formal runs leave this off.",
    )
    parser.add_argument(
        "--max-new-rounds",
        type=int,
        help=(
            "Run at most this many new rounds while preserving the configured --rounds "
            "as the RDP accounting horizon. The resulting checkpoint can be resumed."
        ),
    )
    parser.add_argument(
        "--resume-from-run",
        type=Path,
        help="Resume one seed from an existing run directory containing policy checkpoints.",
    )
    parser.add_argument(
        "--final-round-diagnostic-extra-epsilon",
        type=float,
        default=0.0,
        help="DIAGNOSTIC only; adds an epsilon allowance to final-round Sample-DP ledger (never a formal epsilon guarantee).",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--disable-update-dp",
        action="store_true",
        help="Diagnostic mode: disable update DP for ablation runs.",
    )
    parser.add_argument(
        "--exclude-modes",
        nargs="+",
        default=[],
        help="Exclude collaboration modes from dynamic selection, e.g. --exclude-modes LIE",
    )
    return parser.parse_args()


def _flag(name: str) -> str:
    return "--" + name.replace("_", "-")


def _append_value(command: list[str], name: str, value: object) -> None:
    command.extend([_flag(name), str(value)])


def build_command(
    config: dict,
    *,
    seed: int,
    policies: list[str],
    rounds: int | None,
    max_new_rounds: int | None = None,
    resume_from_run: Path | None = None,
    equal_optimizer_work_control: bool = False,
    server_step: float | None = None,
    disable_update_dp: bool = False,
    exclude_modes: list[str] | None = None,
    liie_edge_dp_plan: str | None = None,
    cloud_dp_plan: str | None = None,
    final_round_diagnostic_extra_epsilon: float = 0.0,
) -> list[str]:
    if config.get("execution_protocol") is not None:
        raise ValueError("Legacy fixed Method2 execution protocols are not part of the current DynFL mainline")
    forbidden_async_keys = {
        "async", "async_training", "staleness", "max_version_gap", "B_edge", "B_cloud"
    }
    present_async_keys = sorted(forbidden_async_keys.intersection(config))
    if present_async_keys:
        raise ValueError(
            "Async/staleness controls are not part of the formal DynFL paper mainline: "
            + ", ".join(present_async_keys)
        )
    training = config["training"]
    system = config["system"]
    privacy = config["privacy"]
    he = config["he"]
    optimization = config["optimization"]
    learning = config["learning"]
    network = config["network"]

    command = [sys.executable, str(ROOT / "experiments" / "run_fmnist_lenet5.py")]
    values = {
        "execution_revision": config["execution_revision"],
        "dataset": training["dataset"],
        "model": training["model"],
        "rounds": rounds if rounds is not None else training["rounds"],
        "train_limit": training["train_limit"],
        "test_limit": training["test_limit"],
        "partition_mode": training["partition_mode"],
        "dirichlet_alpha": training.get("dirichlet_alpha", 0.5),
        "local_epochs": training["local_epochs"],
        "train_batch_size": training.get("batch_size"),
        "lr": training["learning_rate"],
        "server_step": (
            server_step if server_step is not None else training.get("server_step", 1.0)
        ),
        "device": training["device"],
        "executor": training["executor"],
        "selection_period": training["selection_period"],
        "clients": system["clients"],
        "edges": system["edges"],
        "resource_limit": system["resource_limit"],
        "memory_limit": system["memory_limit"],
        "tier_capacity_policy": system.get("tier_capacity_policy", "legacy"),
        "edge_cpu_limit": system.get("edge_cpu_limit", 15.0),
        "cloud_cpu_limit": system.get("cloud_cpu_limit", 20.0),
        "edge_memory_limit_mb": system.get("edge_memory_limit_mb"),
        "cloud_memory_limit_mb": system.get("cloud_memory_limit_mb"),
        "split_interaction_mode": system.get("split_interaction_mode", "fixed"),
        "split_batch_size": system.get("split_batch_size", 128),
        "time_limit": system["time_limit"],
        "risk_limit": system["risk_limit"],
        "aggregation_fraction": system["aggregation_fraction"],
        "client_heterogeneity": system["client_heterogeneity"],
        "edge_heterogeneity": system["edge_heterogeneity"],
        "resource_scenario": system.get("resource_scenario", "none"),
        "constrained_start_fraction": system.get("constrained_start_fraction", 1.0 / 3.0),
        "constrained_end_fraction": system.get("constrained_end_fraction", 2.0 / 3.0),
        "communication_constrained_multiplier": system.get("communication_constrained_multiplier", 0.35),
        "compute_constrained_multiplier": system.get("compute_constrained_multiplier", 2.0),
        "network_jitter": network["jitter"],
        "network_periodic_amplitude": network["periodic_amplitude"],
        "network_period_rounds": network["period_rounds"],
        "end_edge_rate_mb_s": network["end_edge_rate_mb_s"],
        "end_cloud_rate_mb_s": network["end_cloud_rate_mb_s"],
        "edge_cloud_rate_mb_s": network["edge_cloud_rate_mb_s"],
        "end_edge_base_latency_sec": network["end_edge_base_latency_sec"],
        "end_cloud_base_latency_sec": network["end_cloud_base_latency_sec"],
        "edge_cloud_base_latency_sec": network["edge_cloud_base_latency_sec"],
        "min_edge_cloud_fusion_ratio": system["min_edge_cloud_fusion_ratio"],
        "dp_accounting_mode": privacy["accounting_mode"],
        "privacy_unit": privacy.get("unit", "sample"),
        "update_protection_goal": privacy.get("update_protection_goal", "packet_protection"),
        "liie_edge_dp_plan": liie_edge_dp_plan or privacy.get("liie_edge_dp_plan", "independent"),
        "cloud_dp_plan": cloud_dp_plan or privacy.get("cloud_dp_plan", "legacy"),
        "initial_epsilon": privacy["initial_epsilon"],
        "dp_upd_epsilon": privacy["update_epsilon_budget"],
        "dp_update_epsilon_budget": privacy["update_epsilon_budget"],
        "dp_sample_epsilon_budget": privacy.get("sample_epsilon_budget"),
        "dp_sample_embedding_noise_multiplier": privacy.get("sample_embedding_noise_multiplier"),
        "dp_sample_label_grad_noise_multiplier": privacy.get("sample_label_grad_noise_multiplier"),
        "dp_sample_optimizer_noise_multiplier": privacy.get("sample_optimizer_noise_multiplier"),
        "dp_delta": privacy["delta"],
        "dp_clip_norm": privacy["clip_norm"],
        "dp_sample_optimizer_clip_norm": privacy.get(
            "sample_optimizer_clip_norm",
            1.0,
        ),
        "dp_update_mode": privacy["update_mode"],
        "dp_release_calibration": privacy.get("release_calibration", "legacy_aggregate"),
        "cloud_dp_stability_threshold": privacy["cloud_dp_stability_threshold"],
        "he_backend": he["backend"],
        "he_execution": he.get("execution", "real"),
        "he_aggregation_size": he["aggregation_size"],
        "he_workers": he.get("workers", 1),
        "pareto_archive_size": optimization["pareto_archive_size"],
        "pareto_beam_size": optimization["pareto_beam_size"],
        "pareto_max_iters": optimization["pareto_max_iters"],
        "pareto_neighbor_top_k": optimization["pareto_neighbor_top_k"],
        "cloud_fusion_xi": optimization["cloud_fusion_xi"],
        "cloud_fusion_eps": optimization["cloud_fusion_eps"],
        "learning_objective": learning["objective"],
        "online_bound_manifest_path": learning.get("online_bound_manifest_path"),
        "seed": seed,
        "output_root": config["output_root"],
    }
    for name, value in values.items():
        if value is not None:
            _append_value(command, name, value)
    if final_round_diagnostic_extra_epsilon > 0:
        _append_value(command, "final_round_diagnostic_extra_epsilon", final_round_diagnostic_extra_epsilon)
    if training.get("uncapped_local_batches", False):
        command.append("--uncapped-local-batches")
    if system.get("corrected_admission_policy", False):
        command.append("--corrected-admission-policy")
    if disable_update_dp:
        command.extend(["--dp-update-mode", "off"])
    if exclude_modes:
        command.extend(["--exclude-modes", *exclude_modes])
    # V370_STATIC_PAIR_MANIFEST_PAPER: opt-in, fail closed on inconsistent flags.
    static_pair_manifest = system.get("static_pair_admission_manifest")
    if static_pair_manifest is not None:
        if not isinstance(static_pair_manifest, str) or not static_pair_manifest.strip():
            raise ValueError("system.static_pair_admission_manifest must be a nonempty path")
        if not system.get("fl_first_split_on_demand", False):
            raise ValueError("V370 requires system.fl_first_split_on_demand=true")
        if system.get("edge_only_requires_fast_deadline", False):
            raise ValueError("V370 requires system.edge_only_requires_fast_deadline=false; LIE is not fast-only")
        if system.get("fast_client_deadlines"):
            raise ValueError("V370 fast_client_deadlines must come from the manifest, not both sources")
        command.extend(["--static-pair-admission-manifest", static_pair_manifest])
    fast_deadlines = system.get("fast_client_deadlines", {})
    if fast_deadlines:
        if not isinstance(fast_deadlines, dict):
            raise ValueError("system.fast_client_deadlines must be an object mapping client id to seconds")
        command.extend(
            ["--fast-client-deadlines", *[f"{client_id}:{deadline}" for client_id, deadline in fast_deadlines.items()]]
        )
    command.extend(["--update-mechanisms", *privacy.get("candidate_mechanisms", ["dp", "he3", "dp_he3"])])
    if system.get("fl_first_split_on_demand", False):
        command.append("--fl-first-split-on-demand")
    if system.get("edge_only_requires_fast_deadline", False):
        command.append("--edge-only-requires-fast-deadline")
    if equal_optimizer_work_control:
        command.append("--equal-optimizer-work-control")
    if max_new_rounds is not None:
        _append_value(command, "max_new_rounds", max_new_rounds)
    if resume_from_run is not None:
        _append_value(command, "resume_from_run", resume_from_run.resolve())

    for name in (
        "require_feasible",
        "require_edge_cloud_coverage",
    ):
        if system.get(name):
            command.append(_flag(name))
    if privacy.get("enforce_cloud_dp_stability"):
        command.append("--enforce-cloud-dp-stability")
    if he.get("require_real_he"):
        command.append("--require-real-he")
    command.extend(["--policies", *policies])
    return command


def main() -> None:
    args = parse_args()
    config_path = args.config.resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if args.he_execution is not None:
        config["he"]["execution"] = args.he_execution
        config["he"]["require_real_he"] = args.he_execution == "real"
    if args.output_root is not None:
        config["output_root"] = args.output_root
    learning_config = config.get("learning", {})
    if isinstance(learning_config, dict) and learning_config.get("history_ema_j_enabled") is True:
        os.environ["DYNFL_HISTORY_EMA_J_ENABLED"] = "1"
        os.environ["DYNFL_HISTORY_EMA_J_DECAY"] = str(
            learning_config.get("history_ema_j_decay", 0.9)
        )
    validate_config(config)
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    policies = args.policies or list(config["policies"])
    if args.resume_from_run is not None and len(seeds) != 1:
        raise ValueError("--resume-from-run requires exactly one seed")
    if args.max_new_rounds is not None and args.max_new_rounds < 1:
        raise ValueError("--max-new-rounds must be positive")
    if not math.isfinite(args.final_round_diagnostic_extra_epsilon) or args.final_round_diagnostic_extra_epsilon < 0:
        raise ValueError("--final-round-diagnostic-extra-epsilon must be finite and nonnegative")
    effective_server_step = (
        args.server_step if args.server_step is not None
        else config["training"].get("server_step", 1.0)
    )
    if not (0.0 < effective_server_step <= 1.0):
        raise ValueError("--server-step must be in (0, 1]")

    for seed in seeds:
        command = build_command(
            config,
            seed=seed,
            policies=policies,
            rounds=args.rounds,
            max_new_rounds=args.max_new_rounds,
            resume_from_run=args.resume_from_run,
            equal_optimizer_work_control=args.equal_optimizer_work_control,
            server_step=args.server_step,
            liie_edge_dp_plan=args.liie_edge_dp_plan,
            cloud_dp_plan=args.cloud_dp_plan,
            final_round_diagnostic_extra_epsilon=args.final_round_diagnostic_extra_epsilon,
            disable_update_dp=args.disable_update_dp,
            exclude_modes=args.exclude_modes,
        )
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


def validate_config(config: dict) -> None:
    if config.get("execution_protocol") is not None:
        raise ValueError("Legacy fixed Method2 execution protocols are archived and cannot be used for formal DynFL runs")
    if config.get("execution_revision") != CURRENT_EXECUTION_REVISION:
        raise RuntimeError(
            f"Paper experiments require {CURRENT_EXECUTION_REVISION}"
        )
    he = config["he"]
    learning = config.get("learning", {})
    history_ema_j_enabled = (
        isinstance(learning, dict) and learning.get("history_ema_j_enabled") is True
    )
    if not history_ema_j_enabled:
        expected = {
            "poly_modulus_degree": int(CKKS_POLY_MODULUS_DEGREE),
            "coeff_modulus_bits": [int(value) for value in CKKS_COEFF_MOD_BIT_SIZES],
            "scale_bits": int(CKKS_SCALE_BITS),
        }
        actual = {
            "poly_modulus_degree": int(he["poly_modulus_degree"]),
            "coeff_modulus_bits": [int(value) for value in he["coeff_modulus_bits"]],
            "scale_bits": int(he["scale_bits"]),
        }
        if actual != expected:
            raise RuntimeError(
                f"Configured CKKS parameters {actual} do not match runtime {expected}"
            )
    if int(config["system"]["clients"]) < int(config["system"]["edges"]):
        raise ValueError("The number of clients must be at least the number of edges")
    if bool(config["system"].get("trusted_edge_split_execution")):
        raise ValueError(
            "trusted_edge_split_execution is legacy and is not valid in the current formal DynFL design; "
            "privacy must be derived from real exposures instead of trusted/untrusted node labels"
        )
    if int(he.get("workers", 1)) < 1:
        raise ValueError("The number of HE workers must be positive")
    if he.get("execution", "real") not in {"real", "profiled"}:
        raise ValueError("HE execution must be real or profiled")
    if he.get("require_real_he") and he.get("execution", "real") != "real":
        raise ValueError("require_real_he is only valid with real HE execution")
    policies = tuple(config.get("policies", ()))
    if policies != FORMAL_BASELINE_POLICIES:
        raise ValueError(
            "Formal paper configurations must use exactly the four frozen internal baselines: "
            + ", ".join(FORMAL_BASELINE_POLICIES)
        )
    privacy = config["privacy"]
    learning = config.get("learning")
    history_ema_j_enabled = (
        isinstance(learning, dict) and learning.get("history_ema_j_enabled") is True
    )
    if str(privacy.get("unit", "sample")).strip().lower() != "sample":
        raise ValueError(
            "Formal paper experiments are Sample-DP only. Legacy client-level DP "
            "may remain in the repository but cannot be selected by run_paper_config.py."
        )
    if config["system"].get("mainline_fusion"):
        raise ValueError(
            "mainline_fusion/Method2 global-release overlay has been removed "
            "from the current DynFL design"
        )
    if history_ema_j_enabled:
        if privacy.get("protection_scope") not in {
            "cross_domain_update", "cross_domain_update_and_secure_aggregate"
        }:
            raise ValueError("History-EMA J requires upload-boundary update protection")
        if privacy.get("candidate_mechanisms") != ["dp"]:
            raise ValueError("History-EMA J must use DP-only candidate mechanisms")
    else:
        if privacy.get("protection_scope") != "cross_domain_update_and_secure_aggregate":
            raise ValueError(
                "Formal configurations must protect cross-domain updates and secure aggregate releases"
            )
        if privacy.get("candidate_mechanisms") != ["dp", "he3", "dp_he3"]:
            raise ValueError(
                "Formal configurations must expose DP, HE, and their combined DP+HE update mechanism"
            )
    if privacy.get("release_calibration") != "tex_packet":
        raise ValueError("TeX configurations require whole packet DP calibration")
    if float(privacy["cloud_dp_stability_threshold"]) <= 0.0:
        raise ValueError("The cloud DP stability threshold must be positive")
    if not isinstance(learning, dict) or not learning.get("objective"):
        raise ValueError(
            "Formal configurations must declare learning.objective explicitly; "
            "no learning method may be selected implicitly"
        )
    objective = learning["objective"]
    if history_ema_j_enabled:
        if objective != "history_ema_j":
            raise ValueError("History-EMA J configuration must use learning.objective='history_ema_j'")
        if learning.get("online_bound_manifest_path"):
            raise ValueError("History-EMA J must not depend on an online-bound certificate")
        if he.get("backend") != "none" or he.get("execution") != "real":
            raise ValueError("History-EMA J requires HE disabled: backend='none', execution='real'")
    else:
        if objective != "online_upper_bound":
            raise ValueError(
                f"Unsupported learning.objective: {objective}; legacy configs use online_upper_bound."
            )
        manifest = learning.get("online_bound_manifest_path")
        if not isinstance(manifest, str) or not manifest.strip():
            raise ValueError("online_upper_bound requires learning.online_bound_manifest_path")


if __name__ == "__main__":
    main()
