"""Validate the exact-noise *functionality* seam without overstating security."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from dynfed.privacy import calibrate_gaussian_noise
from experiments.client_secure_aggregate_common import (
    ClientMaskingContext,
    client_secure_aggregate_plan,
    clip_l2,
    make_masked_client_packet_with_external_noise,
    untrusted_edge_fixed_cohort_sum,
)
from experiments.joint_noise_mpc_common import (
    ideal_joint_gaussian_shares,
    joint_noise_contract_audit,
    validate_joint_noise_requirement,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--dimension", type=int, default=4096)
    p.add_argument("--clip-norm", type=float, default=0.25)
    p.add_argument("--epsilon", type=float, default=8.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--privacy-horizon", type=int, default=1)
    p.add_argument("--minimum-unknown-clients", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.clients < 2 or args.dimension <= 0:
        raise ValueError("Need at least two clients and positive dimension")
    torch.manual_seed(args.seed)
    weights = [1.0 / args.clients] * args.clients
    sigma = calibrate_gaussian_noise(
        args.epsilon, args.delta, args.privacy_horizon
    )
    plan = client_secure_aggregate_plan(
        weights, args.clip_norm, sigma, args.minimum_unknown_clients
    )
    requirement = validate_joint_noise_requirement(
        args.clients, args.minimum_unknown_clients, plan.target_noise_std
    )
    shares, target_noise, backend = ideal_joint_gaussian_shares(
        requirement=requirement, dimension=args.dimension
    )

    contexts = [ClientMaskingContext(i) for i in range(args.clients)]
    public_keys = {c.client_id: c.public_key for c in contexts}
    updates = {
        i: torch.randn(args.dimension, dtype=torch.float64) / math.sqrt(args.dimension)
        for i in range(args.clients)
    }
    packets = {}
    clipped_weighted = []
    for i, context in enumerate(contexts):
        clipped, _, _ = clip_l2(updates[i], plan.clip_norm)
        clipped_weighted.append(clipped * weights[i])
        packets[i], _ = make_masked_client_packet_with_external_noise(
            context=context,
            update=updates[i],
            external_noise_share=shares[i],
            public_keys=public_keys,
            plan=plan,
            round_idx=0,
        )
    clean = torch.stack(clipped_weighted).sum(0)
    released = untrusted_edge_fixed_cohort_sum(packets, cohort_size=args.clients)
    observed_noise = released - clean
    max_noise_error = float((observed_noise - target_noise).abs().max())

    report = joint_noise_contract_audit(requirement, backend_metadata=backend)
    report.update({
        "dimension": args.dimension,
        "epsilon_target": args.epsilon,
        "delta": args.delta,
        "privacy_horizon": args.privacy_horizon,
        "aggregate_dp_sensitivity": plan.aggregate_sensitivity,
        "measured_target_noise_norm": float(torch.linalg.vector_norm(target_noise)),
        "measured_released_noise_norm": float(torch.linalg.vector_norm(observed_noise)),
        "release_matches_ideal_target_max_abs_error": max_noise_error,
        "secagg_external_noise_plumbing_exact": max_noise_error <= 1e-9,
        "security_warning": (
            "The ideal joint-noise generator is centralized diagnostic code. "
            "Do not use this result as an MPC/threshold security claim."
        ),
    })
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
