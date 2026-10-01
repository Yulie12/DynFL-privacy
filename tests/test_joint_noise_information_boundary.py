import numpy as np

from experiments.joint_noise_information_boundary import (
    additive_gaussian_full_robustness_obstruction,
    aggregate_variance,
    audit_linear_gaussian_coalition,
    equicorrelated_exact_covariance,
    gaussian_conditional_aggregate_variance,
    iid_exact_covariance,
    total_variance_audit,
)


def test_total_variance_identity_audit():
    a = total_variance_audit(1.0, 0.7, 0.3)
    assert a.identity_error < 1e-12


def test_iid_exact_conditional_ratio_is_h_over_k():
    k, h = 100, 2
    cov = iid_exact_covariance(k, 3.0)
    coalition = tuple(range(k - h))
    audit = audit_linear_gaussian_coalition(cov, coalition, 3.0)
    assert abs(audit.aggregate_variance - 3.0) < 1e-12
    assert abs(audit.conditional_variance_ratio - h / k) < 1e-12


def test_equicorrelated_covariance_is_exact_target_but_conditioning_reduces_variance():
    k = 12
    cov = equicorrelated_exact_covariance(k, 2.5, rho=0.2)
    assert abs(aggregate_variance(cov) - 2.5) < 1e-10
    cond = gaussian_conditional_aggregate_variance(cov, tuple(range(6)))
    assert 0.0 <= cond < 2.5


def test_additive_gaussian_covariance_identity_forces_some_informative_share():
    cov = equicorrelated_exact_covariance(8, 1.7, rho=-0.05)
    result = additive_gaussian_full_robustness_obstruction(cov)
    assert result["covariance_identity_error"] < 1e-10
    assert result["exact_positive_variance_requires_some_informative_share"] is True
    assert result["all_client_covariances_zero"] is False
    assert result["general_mpc_impossibility_claim"] is False


def test_boundary_does_not_claim_general_mpc_impossibility():
    cov = iid_exact_covariance(4, 1.0)
    result = additive_gaussian_full_robustness_obstruction(cov)
    assert result["scope"] == "jointly_gaussian_locally_known_additive_shares"
    assert result["general_mpc_impossibility_claim"] is False
