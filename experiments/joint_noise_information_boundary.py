"""Information-theoretic boundary diagnostics for exact distributed Gaussian noise.

Scope
-----
This module studies a restricted but important class of constructions:

    * clients locally know real-valued additive Gaussian shares X_i;
    * the released aggregate noise is Z = sum_i X_i;
    * a colluding client reveals its local share to the edge/adversary.

For jointly Gaussian X, exact target variance plus unchanged conditional target
variance against arbitrary colluding subsets cannot be achieved merely by
choosing a clever covariance matrix.  This is a boundary result for this
construction class, not an impossibility theorem for general MPC/threshold
protocols.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np


@dataclass(frozen=True)
class TotalVarianceAudit:
    marginal_variance: float
    expected_conditional_variance: float
    conditional_mean_variance: float
    identity_error: float


@dataclass(frozen=True)
class LinearGaussianAudit:
    target_variance: float
    aggregate_variance: float
    coalition: tuple[int, ...]
    conditional_variance: float
    conditional_variance_ratio: float
    covariance_with_coalition_norm: float


def total_variance_audit(
    marginal_variance: float,
    expected_conditional_variance: float,
    conditional_mean_variance: float,
) -> TotalVarianceAudit:
    """Audit Var(Z)=E[Var(Z|V)]+Var(E[Z|V])."""
    lhs = float(marginal_variance)
    rhs = float(expected_conditional_variance) + float(conditional_mean_variance)
    return TotalVarianceAudit(
        marginal_variance=lhs,
        expected_conditional_variance=float(expected_conditional_variance),
        conditional_mean_variance=float(conditional_mean_variance),
        identity_error=abs(lhs - rhs),
    )


def aggregate_variance(covariance: np.ndarray) -> float:
    cov = np.asarray(covariance, dtype=np.float64)
    ones = np.ones(cov.shape[0], dtype=np.float64)
    return float(ones @ cov @ ones)


def aggregate_share_covariances(covariance: np.ndarray) -> np.ndarray:
    """Return Cov(Z, X_i) for Z=sum_j X_j."""
    cov = np.asarray(covariance, dtype=np.float64)
    return np.sum(cov, axis=0)


def gaussian_conditional_aggregate_variance(
    covariance: np.ndarray,
    coalition: Sequence[int],
) -> float:
    """Var(sum_i X_i | X_coalition) for a jointly Gaussian share vector.

    Uses the Schur-complement formula.  A Moore-Penrose inverse is used so the
    diagnostic also handles positive-semidefinite singular covariance matrices.
    """
    cov = np.asarray(covariance, dtype=np.float64)
    coalition = tuple(int(i) for i in coalition)
    var_z = aggregate_variance(cov)
    if not coalition:
        return var_z
    c = np.asarray(coalition, dtype=np.int64)
    cov_z_x = aggregate_share_covariances(cov)[c]
    cov_x = cov[np.ix_(c, c)]
    reduction = float(cov_z_x @ np.linalg.pinv(cov_x) @ cov_z_x)
    value = var_z - reduction
    # Numerical roundoff may make a theoretically zero variance tiny negative.
    return float(max(0.0, value))


def audit_linear_gaussian_coalition(
    covariance: np.ndarray,
    coalition: Sequence[int],
    target_variance: float,
) -> LinearGaussianAudit:
    cov = np.asarray(covariance, dtype=np.float64)
    coalition = tuple(int(i) for i in coalition)
    cond = gaussian_conditional_aggregate_variance(cov, coalition)
    var_z = aggregate_variance(cov)
    cov_z_x = aggregate_share_covariances(cov)
    norm = float(np.linalg.norm(cov_z_x[list(coalition)])) if coalition else 0.0
    return LinearGaussianAudit(
        target_variance=float(target_variance),
        aggregate_variance=var_z,
        coalition=coalition,
        conditional_variance=cond,
        conditional_variance_ratio=(cond / float(target_variance)),
        covariance_with_coalition_norm=norm,
    )


def iid_exact_covariance(cohort_size: int, target_variance: float) -> np.ndarray:
    if cohort_size <= 0:
        raise ValueError("cohort_size must be positive")
    if target_variance <= 0:
        raise ValueError("target_variance must be positive")
    return np.eye(cohort_size, dtype=np.float64) * (float(target_variance) / cohort_size)


def equicorrelated_exact_covariance(
    cohort_size: int,
    target_variance: float,
    rho: float,
) -> np.ndarray:
    """Equicorrelated Gaussian shares scaled to exact aggregate variance.

    PSD requires -1/(K-1) <= rho <= 1 for K>1.  The matrix is rescaled so
    Var(sum_i X_i) equals ``target_variance`` exactly whenever the denominator
    is positive.
    """
    k = int(cohort_size)
    if k <= 0:
        raise ValueError("cohort_size must be positive")
    if target_variance <= 0:
        raise ValueError("target_variance must be positive")
    if k == 1:
        if abs(rho) > 1e-15:
            raise ValueError("rho must be zero for cohort_size=1")
        return np.array([[float(target_variance)]], dtype=np.float64)
    lower = -1.0 / (k - 1)
    if rho < lower - 1e-12 or rho > 1.0 + 1e-12:
        raise ValueError("rho is outside the PSD equicorrelation interval")
    base = np.full((k, k), float(rho), dtype=np.float64)
    np.fill_diagonal(base, 1.0)
    denom = float(k * (1.0 + (k - 1) * rho))
    if denom <= 1e-15:
        raise ValueError("rho makes aggregate variance zero; cannot rescale to positive target")
    return base * (float(target_variance) / denom)


def additive_gaussian_full_robustness_obstruction(covariance: np.ndarray) -> dict:
    """Return the algebraic obstruction for additive Gaussian share protocols.

    Since Z=sum_i X_i,

        sum_i Cov(Z, X_i) = Cov(Z, sum_i X_i) = Var(Z).

    Therefore, if Var(Z)>0, not all Cov(Z,X_i) can vanish.  For jointly
    Gaussian variables, any coalition observing an X_i with nonzero covariance
    with Z strictly lowers Var(Z | coalition view) (except degenerate cases
    where the observation itself has zero variance).  Thus a protocol that must
    tolerate arbitrary client collusion cannot keep the full exact target
    variance merely by correlating locally known additive Gaussian shares.
    """
    cov = np.asarray(covariance, dtype=np.float64)
    var_z = aggregate_variance(cov)
    covs = aggregate_share_covariances(cov)
    identity_error = abs(float(np.sum(covs)) - var_z)
    nonzero = np.flatnonzero(np.abs(covs) > 1e-12)
    return {
        "aggregate_variance": var_z,
        "sum_cov_z_xi": float(np.sum(covs)),
        "covariance_identity_error": identity_error,
        "clients_with_nonzero_cov_z_xi": [int(i) for i in nonzero],
        "all_client_covariances_zero": bool(len(nonzero) == 0),
        "exact_positive_variance_requires_some_informative_share": bool(var_z > 0 and len(nonzero) > 0),
        "scope": "jointly_gaussian_locally_known_additive_shares",
        "general_mpc_impossibility_claim": False,
    }
