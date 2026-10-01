"""Joint-noise backend interface diagnostics for untrusted-edge SecAgg.

This module keeps three concepts separate:

1. *aggregate distribution*: what distribution the sum of client noise shares has;
2. *share hiding*: whether an edge/colluding clients can read an individual share;
3. *conditional aggregate variance*: how much Gaussian variance remains unknown
   after conditioning on colluding clients' private noise contributions.

A key non-result is encoded explicitly: adding pairwise zero-sum masks to iid
Gaussian shares can hide individual shares in transit, but cannot improve the
conditional variance of the *released aggregate noise*.  The zero-sum masks
cancel from the aggregate, so exact-target iid shares still retain only H/K of
the target variance when only H client Gaussian seeds remain unknown.

All generators in this module are diagnostic simulators.  They are not MPC or
threshold cryptographic realizations.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Mapping

import torch

from experiments.joint_noise_mpc_common import JointNoiseRequirement


@dataclass(frozen=True)
class JointNoiseBackendAudit:
    backend: str
    aggregate_variance_ratio: float
    conditional_variance_ratio: float
    exact_target_variance: bool
    pairwise_zero_sum_masks: bool
    individual_share_hidden_in_transit: bool
    cryptographic_realization: bool
    collusion_resistance_established: bool


def _private_generator() -> torch.Generator:
    seed = int.from_bytes(os.urandom(8), "little", signed=False) & ((1 << 63) - 1)
    g = torch.Generator(device="cpu")
    g.manual_seed(seed)
    return g


def iid_exact_target_noise_shares(
    *, requirement: JointNoiseRequirement, dimension: int
) -> dict[int, torch.Tensor]:
    """Diagnostic iid shares with exact *distributional* target variance.

    Each share is N(0, target^2/K).  Their sum is therefore exactly Gaussian
    with variance target^2 in distribution, but a coalition that learns K-H
    client seeds leaves only H/K of the target variance unknown.
    """
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    share_std = requirement.target_noise_std / math.sqrt(requirement.cohort_size)
    return {
        i: torch.randn(
            dimension, generator=_private_generator(), dtype=torch.float64
        ) * share_std
        for i in range(requirement.cohort_size)
    }


def add_pairwise_zero_sum_noise_masks(
    shares: Mapping[int, torch.Tensor], *, mask_std: float
) -> dict[int, torch.Tensor]:
    """Add diagnostic pairwise Gaussian masks whose full-cohort sum is zero.

    This models the algebra of correlated pairwise share-hiding masks.  One
    process creates all pair masks, so it is deliberately not a security
    implementation.  The purpose is to verify that such masks do not alter the
    final aggregate-noise distribution or its H/K conditional-variance bound.
    """
    if not math.isfinite(mask_std) or mask_std <= 0.0:
        raise ValueError("mask_std must be finite and positive")
    ids = sorted(shares)
    if ids != list(range(len(ids))) or len(ids) < 2:
        raise ValueError("shares must contain a contiguous cohort of at least 2 clients")
    vectors = {
        i: shares[i].detach().to(device="cpu", dtype=torch.float64).reshape(-1).clone()
        for i in ids
    }
    d = vectors[0].numel()
    if d <= 0 or any(v.shape != vectors[0].shape or not bool(torch.isfinite(v).all())
                     for v in vectors.values()):
        raise ValueError("finite matching share vectors required")
    for i in ids:
        for j in ids:
            if j <= i:
                continue
            mask = torch.randn(
                d, generator=_private_generator(), dtype=torch.float64
            ) * float(mask_std)
            vectors[i] = vectors[i] + mask
            vectors[j] = vectors[j] - mask
    return vectors


def exact_iid_conditional_variance_ratio(
    requirement: JointNoiseRequirement, *, unknown_clients: int | None = None
) -> float:
    """Unknown aggregate Gaussian variance / target variance for iid shares."""
    h = requirement.minimum_unknown_clients if unknown_clients is None else unknown_clients
    if type(h) is not int or not 1 <= h <= requirement.cohort_size:
        raise ValueError("unknown_clients must be in [1, cohort_size]")
    return h / requirement.cohort_size


def audit_masked_iid_exact_target_backend(
    requirement: JointNoiseRequirement,
    *, unknown_clients: int | None = None,
) -> JointNoiseBackendAudit:
    """Audit iid exact-target shares even when pairwise zero-sum masks are used."""
    ratio = exact_iid_conditional_variance_ratio(
        requirement, unknown_clients=unknown_clients
    )
    return JointNoiseBackendAudit(
        backend="iid_exact_target_plus_pairwise_zero_sum_masks_diagnostic",
        aggregate_variance_ratio=1.0,
        conditional_variance_ratio=ratio,
        exact_target_variance=True,
        pairwise_zero_sum_masks=True,
        individual_share_hidden_in_transit=True,
        cryptographic_realization=False,
        collusion_resistance_established=(ratio >= 1.0 - 1e-15),
    )


def backend_audit_dict(audit: JointNoiseBackendAudit) -> dict[str, object]:
    return {
        "backend": audit.backend,
        "aggregate_variance_ratio": audit.aggregate_variance_ratio,
        "conditional_variance_ratio": audit.conditional_variance_ratio,
        "exact_target_variance": audit.exact_target_variance,
        "pairwise_zero_sum_masks": audit.pairwise_zero_sum_masks,
        "individual_share_hidden_in_transit": audit.individual_share_hidden_in_transit,
        "cryptographic_realization": audit.cryptographic_realization,
        "collusion_resistance_established": audit.collusion_resistance_established,
        "protocol_status": (
            "diagnostic_only_pairwise_masks_do_not_fix_conditional_variance"
            if not audit.cryptographic_realization
            else "backend_requires_separate_security_review"
        ),
    }
