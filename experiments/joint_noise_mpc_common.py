"""Ideal joint-noise functionality and deployability contract.

This module deliberately separates *functionality* from *cryptographic
realization*.  ``ideal_joint_gaussian_shares`` is useful for plumbing and
utility diagnostics: it samples one target Gaussian vector and secret-share-like
additive pieces whose sum is exactly that vector.  It is NOT an MPC protocol;
the process creating all shares is a trusted simulator and therefore cannot be
used to claim protection against an untrusted edge or client-edge collusion.

A future reviewed MPC/threshold backend can implement the same external-share
interface without changing the secure-aggregation packet path.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import os
from typing import Mapping

import torch


@dataclass(frozen=True)
class JointNoiseRequirement:
    cohort_size: int
    minimum_unknown_clients: int
    target_noise_std: float


def validate_joint_noise_requirement(
    cohort_size: int, minimum_unknown_clients: int, target_noise_std: float
) -> JointNoiseRequirement:
    if type(cohort_size) is not int or cohort_size < 2:
        raise ValueError("cohort_size must be an integer >= 2")
    if (type(minimum_unknown_clients) is not int
            or not 2 <= minimum_unknown_clients <= cohort_size):
        raise ValueError("minimum_unknown_clients must be in [2, cohort_size]")
    if not math.isfinite(target_noise_std) or target_noise_std <= 0.0:
        raise ValueError("target_noise_std must be finite and positive")
    return JointNoiseRequirement(
        cohort_size=cohort_size,
        minimum_unknown_clients=minimum_unknown_clients,
        target_noise_std=float(target_noise_std),
    )


def independent_share_exact_target_conditional_variance_ratio(
    requirement: JointNoiseRequirement,
) -> float:
    """Residual variance ratio after conditioning to Hmin unknown iid shares.

    If iid client shares are calibrated so that *all K* shares sum to exactly
    the target variance, each has variance target^2/K.  Once all but Hmin
    shares are known to a colluding adversary, only Hmin/K of the target
    variance remains.  Thus iid shares cannot simultaneously provide exact
    full-cohort variance and the same target variance after such conditioning,
    unless Hmin == K.
    """
    return requirement.minimum_unknown_clients / requirement.cohort_size


def independent_share_robust_energy_inflation(
    requirement: JointNoiseRequirement,
) -> float:
    """Full-cohort energy inflation needed by iid shares for Hmin robustness."""
    return requirement.cohort_size / requirement.minimum_unknown_clients


def _private_generator() -> torch.Generator:
    seed = int.from_bytes(os.urandom(8), "little", signed=False) & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return generator


def ideal_joint_gaussian_shares(
    *,
    requirement: JointNoiseRequirement,
    dimension: int,
    share_mask_std: float | None = None,
) -> tuple[dict[int, torch.Tensor], torch.Tensor, dict[str, object]]:
    """Return additive shares that sum *exactly* to one target Gaussian draw.

    This is an IDEAL FUNCTIONALITY SIMULATOR.  One process samples the complete
    target noise and all shares.  It proves only that the downstream packet and
    aggregation plumbing can consume externally generated joint-noise shares.
    It provides no cryptographic secrecy against the process hosting it.
    """
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    if share_mask_std is None:
        share_mask_std = requirement.target_noise_std
    if not math.isfinite(share_mask_std) or share_mask_std <= 0.0:
        raise ValueError("share_mask_std must be finite and positive")

    target = torch.randn(
        dimension, generator=_private_generator(), dtype=torch.float64
    ) * requirement.target_noise_std

    shares: dict[int, torch.Tensor] = {}
    running = torch.zeros(dimension, dtype=torch.float64)
    # K-1 random additive masks; final share closes the sum exactly.  The
    # marginal distribution of individual shares is irrelevant here because
    # this simulator is not itself a privacy mechanism.
    for i in range(requirement.cohort_size - 1):
        share = torch.randn(
            dimension, generator=_private_generator(), dtype=torch.float64
        ) * float(share_mask_std)
        shares[i] = share
        running = running + share
    shares[requirement.cohort_size - 1] = target - running

    residual = torch.stack([shares[i] for i in range(requirement.cohort_size)]).sum(0) - target
    max_abs_error = float(residual.abs().max())
    return shares, target, {
        "backend": "ideal_joint_gaussian_functionality_simulator",
        "cryptographic_realization": False,
        "deployable_under_untrusted_edge": False,
        "collusion_resistance_established": False,
        "exact_target_sum": max_abs_error <= 1e-10,
        "max_abs_share_sum_error": max_abs_error,
        "required_real_backend": (
            "reviewed_MPC_or_threshold_secret_shared_noise_sampler"
        ),
    }


def joint_noise_contract_audit(
    requirement: JointNoiseRequirement,
    *,
    backend_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    metadata = dict(backend_metadata or {})
    cryptographic = metadata.get("cryptographic_realization") is True
    collusion = metadata.get("collusion_resistance_established") is True
    exact = metadata.get("exact_target_sum") is True
    closed = bool(cryptographic and collusion and exact)
    return {
        "cohort_size": requirement.cohort_size,
        "minimum_unknown_clients": requirement.minimum_unknown_clients,
        "target_noise_std": requirement.target_noise_std,
        "iid_exact_conditional_variance_ratio": (
            independent_share_exact_target_conditional_variance_ratio(requirement)
        ),
        "iid_robust_energy_inflation": (
            independent_share_robust_energy_inflation(requirement)
        ),
        "exact_target_aggregate_noise": exact,
        "cryptographic_realization": cryptographic,
        "collusion_resistance_established": collusion,
        "protocol_status": (
            "joint_noise_crypto_closed" if closed
            else "ideal_functionality_only_not_a_security_closure"
        ),
    }
