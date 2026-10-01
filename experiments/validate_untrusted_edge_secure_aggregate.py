"""Protocol-level validation of fixed-cohort SecAgg + distributed aggregate DP."""
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

from experiments.client_secure_aggregate_common import (
    ClientMaskingContext,
    client_secure_aggregate_plan,
    make_masked_client_packet,
    protocol_audit,
    untrusted_edge_fixed_cohort_sum,
)
from dynfed.privacy import calibrate_gaussian_noise


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--dimension", type=int, default=4096)
    p.add_argument("--clip-norm", type=float, default=0.25)
    p.add_argument("--epsilon", type=float, default=8.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--privacy-horizon", type=int, default=1)
    p.add_argument("--minimum-unknown-clients", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    if args.clients < 2 or args.dimension < 1:
        raise ValueError("Need at least two clients and a positive dimension")

    torch.manual_seed(args.seed)
    weights = [1.0 / args.clients] * args.clients
    sigma = calibrate_gaussian_noise(args.epsilon, args.delta, args.privacy_horizon)
    plan = client_secure_aggregate_plan(
        weights, args.clip_norm, sigma, args.minimum_unknown_clients
    )
    contexts = [ClientMaskingContext(i) for i in range(args.clients)]
    public_keys = {c.client_id: c.public_key for c in contexts}
    updates = [torch.randn(args.dimension, dtype=torch.float64) * 0.02
               for _ in contexts]

    clipped = []
    packets = {}
    for context, update in zip(contexts, updates):
        norm = float(torch.linalg.vector_norm(update))
        clipped_update = update * min(1.0, args.clip_norm / max(norm, 1e-30))
        clipped.append(clipped_update)
        packets[context.client_id], _ = make_masked_client_packet(
            context=context,
            update=update,
            public_keys=public_keys,
            plan=plan,
            round_idx=0,
        )

    protected = untrusted_edge_fixed_cohort_sum(packets, cohort_size=args.clients)
    clean_clipped = sum(u * w for u, w in zip(clipped, weights))
    effective_noise = protected - clean_clipped
    measured_noise_norm = float(torch.linalg.vector_norm(effective_noise))
    expected_noise_norm = plan.all_clients_noise_std * math.sqrt(args.dimension)
    audit = protocol_audit(
        plan,
        received_client_ids=list(packets),
        assumed_unknown_client_ids=list(range(args.minimum_unknown_clients)),
    )
    report = {
        **audit,
        "clients": args.clients,
        "dimension": args.dimension,
        "epsilon_target": args.epsilon,
        "delta": args.delta,
        "privacy_horizon": args.privacy_horizon,
        "noise_multiplier": sigma,
        "measured_aggregate_noise_norm": measured_noise_norm,
        "expected_aggregate_noise_norm": expected_noise_norm,
        "noise_norm_ratio_measured_to_expected": measured_noise_norm / expected_noise_norm,
    }
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
