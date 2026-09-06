from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper_v26_cifar10_resnet18.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.he_backend import (
    CKKS_COEFF_MOD_BIT_SIZES,
    CKKS_POLY_MODULUS_DEGREE,
    CKKS_SCALE_BITS,
)
from dynfed.version import CURRENT_EXECUTION_REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a versioned paper experiment configuration."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--he-execution", choices=["real", "profiled"])
    parser.add_argument("--output-root")
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
    parser.add_argument("--dry-run", action="store_true")
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
) -> list[str]:
    training = config["training"]
    system = config["system"]
    privacy = config["privacy"]
    he = config["he"]
    optimization = config["optimization"]
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
        "local_epochs": training["local_epochs"],
        "lr": training["learning_rate"],
        "device": training["device"],
        "executor": training["executor"],
        "selection_period": training["selection_period"],
        "clients": system["clients"],
        "edges": system["edges"],
        "resource_limit": system["resource_limit"],
        "memory_limit": system["memory_limit"],
        "time_limit": system["time_limit"],
        "risk_limit": system["risk_limit"],
        "aggregation_fraction": system["aggregation_fraction"],
        "client_heterogeneity": system["client_heterogeneity"],
        "edge_heterogeneity": system["edge_heterogeneity"],
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
        "initial_epsilon": privacy["initial_epsilon"],
        "dp_upd_epsilon": privacy["update_epsilon_budget"],
        "dp_update_epsilon_budget": privacy["update_epsilon_budget"],
        "dp_delta": privacy["delta"],
        "dp_clip_norm": privacy["clip_norm"],
        "dp_update_mode": privacy["update_mode"],
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
        "seed": seed,
        "output_root": config["output_root"],
    }
    if not system.get("trusted_edge_split_execution"):
        feature_budget = privacy["feature_epsilon_budget"]
        values["dp_emb_epsilon"] = feature_budget
        values["dp_feature_epsilon_budget"] = feature_budget
    for name, value in values.items():
        _append_value(command, name, value)
    if max_new_rounds is not None:
        _append_value(command, "max_new_rounds", max_new_rounds)
    if resume_from_run is not None:
        _append_value(command, "resume_from_run", resume_from_run.resolve())

    for name in (
        "require_feasible",
        "require_edge_cloud_coverage",
        "trusted_edge_split_execution",
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
    validate_config(config)
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    policies = args.policies or list(config["policies"])
    if args.resume_from_run is not None and len(seeds) != 1:
        raise ValueError("--resume-from-run requires exactly one seed")
    if args.max_new_rounds is not None and args.max_new_rounds < 1:
        raise ValueError("--max-new-rounds must be positive")

    for seed in seeds:
        command = build_command(
            config,
            seed=seed,
            policies=policies,
            rounds=args.rounds,
            max_new_rounds=args.max_new_rounds,
            resume_from_run=args.resume_from_run,
        )
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


def validate_config(config: dict) -> None:
    if config.get("execution_revision") != CURRENT_EXECUTION_REVISION:
        raise RuntimeError(
            f"Paper experiments require {CURRENT_EXECUTION_REVISION}"
        )
    he = config["he"]
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
    if not bool(config["system"].get("trusted_edge_split_execution")):
        raise ValueError(
            "Formal configurations must enable trusted edge split execution"
        )
    if int(he.get("workers", 1)) < 1:
        raise ValueError("The number of HE workers must be positive")
    if he.get("execution", "real") not in {"real", "profiled"}:
        raise ValueError("HE execution must be real or profiled")
    if he.get("require_real_he") and he.get("execution", "real") != "real":
        raise ValueError("require_real_he is only valid with real HE execution")
    privacy = config["privacy"]
    if not bool(privacy.get("enforce_cloud_dp_stability")):
        raise ValueError(
            "Formal paper configurations must enable the cloud DP stability safeguard"
        )
    if float(privacy["cloud_dp_stability_threshold"]) <= 0.0:
        raise ValueError("The cloud DP stability threshold must be positive")


if __name__ == "__main__":
    main()
