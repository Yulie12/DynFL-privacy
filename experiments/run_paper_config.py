from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper_v22_cifar10_resnet18.json"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.he_backend import (
    CKKS_COEFF_MOD_BIT_SIZES,
    CKKS_POLY_MODULUS_DEGREE,
    CKKS_SCALE_BITS,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a versioned paper experiment configuration."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--rounds", type=int)
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
        "dp_emb_epsilon": privacy["feature_epsilon_budget"],
        "dp_upd_epsilon": privacy["update_epsilon_budget"],
        "dp_feature_epsilon_budget": privacy["feature_epsilon_budget"],
        "dp_update_epsilon_budget": privacy["update_epsilon_budget"],
        "dp_delta": privacy["delta"],
        "dp_clip_norm": privacy["clip_norm"],
        "dp_update_mode": privacy["update_mode"],
        "cloud_dp_stability_threshold": privacy["cloud_dp_stability_threshold"],
        "he_backend": he["backend"],
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
    for name, value in values.items():
        _append_value(command, name, value)

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
    validate_config(config)
    seeds = args.seeds or [int(seed) for seed in config["seeds"]]
    policies = args.policies or list(config["policies"])

    for seed in seeds:
        command = build_command(
            config,
            seed=seed,
            policies=policies,
            rounds=args.rounds,
        )
        print(subprocess.list2cmdline(command), flush=True)
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)


def validate_config(config: dict) -> None:
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
    if int(he.get("workers", 1)) < 1:
        raise ValueError("The number of HE workers must be positive")
    privacy = config["privacy"]
    if not bool(privacy.get("enforce_cloud_dp_stability")):
        raise ValueError(
            "Formal paper configurations must enable the cloud DP stability safeguard"
        )
    if float(privacy["cloud_dp_stability_threshold"]) <= 0.0:
        raise ValueError("The cloud DP stability threshold must be positive")


if __name__ == "__main__":
    main()
