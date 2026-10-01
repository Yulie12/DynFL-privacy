"""Validate that zero-sum correlated masks do not repair iid DP-noise variance."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.joint_noise_backend_common import (
    add_pairwise_zero_sum_noise_masks,
    audit_masked_iid_exact_target_backend,
    backend_audit_dict,
    iid_exact_target_noise_shares,
)
from experiments.joint_noise_mpc_common import validate_joint_noise_requirement


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=10)
    p.add_argument("--dimension", type=int, default=4096)
    p.add_argument("--target-noise-std", type=float, default=0.034556001557948454)
    p.add_argument("--minimum-unknown-clients", type=int, default=2)
    p.add_argument("--mask-std", type=float, default=1.0)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    req = validate_joint_noise_requirement(
        args.clients, args.minimum_unknown_clients, args.target_noise_std
    )
    raw = iid_exact_target_noise_shares(requirement=req, dimension=args.dimension)
    masked = add_pairwise_zero_sum_noise_masks(raw, mask_std=args.mask_std)
    raw_sum = torch.stack([raw[i] for i in range(args.clients)]).sum(0)
    masked_sum = torch.stack([masked[i] for i in range(args.clients)]).sum(0)
    cancellation_error = float((raw_sum - masked_sum).abs().max())

    report = backend_audit_dict(audit_masked_iid_exact_target_backend(req))
    report.update({
        "cohort_size": args.clients,
        "dimension": args.dimension,
        "minimum_unknown_clients": args.minimum_unknown_clients,
        "target_noise_std": args.target_noise_std,
        "pairwise_mask_sum_cancellation_max_abs_error": cancellation_error,
        "aggregate_unchanged_by_pairwise_masks": cancellation_error <= 1e-9,
        "security_warning": (
            "Pairwise zero-sum masks can hide individual shares in transit but "
            "cancel from the aggregate. They do not raise the H/K conditional "
            "variance of exact-target iid Gaussian shares."
        ),
    })
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
