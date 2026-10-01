from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.joint_noise_information_boundary import (
    additive_gaussian_full_robustness_obstruction,
    audit_linear_gaussian_coalition,
    equicorrelated_exact_covariance,
    iid_exact_covariance,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clients", type=int, default=100)
    p.add_argument("--minimum-unknown-clients", type=int, default=2)
    p.add_argument("--target-noise-std", type=float, required=True)
    p.add_argument("--rho", type=float, default=0.25)
    args = p.parse_args()

    k = args.clients
    h = args.minimum_unknown_clients
    if not (1 <= h <= k):
        raise SystemExit("minimum-unknown-clients must be in [1, clients]")
    target_var = args.target_noise_std ** 2
    coalition = tuple(range(k - h))

    iid = iid_exact_covariance(k, target_var)
    iid_audit = audit_linear_gaussian_coalition(iid, coalition, target_var)

    corr = equicorrelated_exact_covariance(k, target_var, args.rho)
    corr_audit = audit_linear_gaussian_coalition(corr, coalition, target_var)
    obstruction = additive_gaussian_full_robustness_obstruction(corr)

    result = {
        "protocol_scope": "jointly_gaussian_locally_known_additive_client_shares",
        "cohort_size": k,
        "minimum_unknown_clients": h,
        "coalition_size": k - h,
        "target_noise_std": args.target_noise_std,
        "target_noise_variance": target_var,
        "iid_exact_conditional_variance_ratio": iid_audit.conditional_variance_ratio,
        "iid_expected_ratio_h_over_k": h / k,
        "equicorrelation_rho": args.rho,
        "equicorrelated_exact_aggregate_variance_ratio": corr_audit.aggregate_variance / target_var,
        "equicorrelated_conditional_variance_ratio": corr_audit.conditional_variance_ratio,
        "equicorrelated_covariance_with_coalition_norm": corr_audit.covariance_with_coalition_norm,
        "covariance_identity_error": obstruction["covariance_identity_error"],
        "some_locally_known_share_must_be_informative_about_exact_positive_aggregate": obstruction[
            "exact_positive_variance_requires_some_informative_share"
        ],
        "correlation_only_closes_exact_and_arbitrary_collusion_gap": False,
        "general_mpc_impossibility_claim": False,
        "required_next_abstraction": "secret_shared_or_mpc_generated_global_noise_not_locally_known_final_additive_gaussian_contributions",
        "protocol_status": "linear_gaussian_additive_share_class_ruled_out_for_exact_full_conditional_variance",
        "security_warning": (
            "This is a boundary result for locally known jointly Gaussian additive shares. "
            "It does not rule out MPC, threshold secret sharing, VSS, or cryptographic joint sampling."
        ),
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
