import torch

from experiments.joint_noise_backend_common import (
    add_pairwise_zero_sum_noise_masks,
    audit_masked_iid_exact_target_backend,
    backend_audit_dict,
    exact_iid_conditional_variance_ratio,
    iid_exact_target_noise_shares,
)
from experiments.joint_noise_mpc_common import validate_joint_noise_requirement


def test_exact_iid_conditional_variance_is_h_over_k():
    req = validate_joint_noise_requirement(100, 2, 0.5)
    assert exact_iid_conditional_variance_ratio(req) == 0.02
    assert exact_iid_conditional_variance_ratio(req, unknown_clients=10) == 0.10


def test_pairwise_zero_sum_masks_leave_aggregate_unchanged():
    req = validate_joint_noise_requirement(8, 2, 0.2)
    raw = iid_exact_target_noise_shares(requirement=req, dimension=257)
    masked = add_pairwise_zero_sum_noise_masks(raw, mask_std=0.7)
    raw_sum = torch.stack([raw[i] for i in range(8)]).sum(0)
    masked_sum = torch.stack([masked[i] for i in range(8)]).sum(0)
    assert torch.allclose(raw_sum, masked_sum, rtol=0.0, atol=1e-10)


def test_pairwise_masks_do_not_upgrade_collusion_variance():
    req = validate_joint_noise_requirement(100, 2, 0.5)
    report = backend_audit_dict(audit_masked_iid_exact_target_backend(req))
    assert report["aggregate_variance_ratio"] == 1.0
    assert report["conditional_variance_ratio"] == 0.02
    assert report["pairwise_zero_sum_masks"] is True
    assert report["individual_share_hidden_in_transit"] is True
    assert report["cryptographic_realization"] is False
    assert report["collusion_resistance_established"] is False
    assert report["protocol_status"] == (
        "diagnostic_only_pairwise_masks_do_not_fix_conditional_variance"
    )


def test_h_equals_k_is_only_iid_exact_case_with_full_conditional_variance():
    req = validate_joint_noise_requirement(10, 10, 0.5)
    report = audit_masked_iid_exact_target_backend(req)
    assert report.conditional_variance_ratio == 1.0
    assert report.collusion_resistance_established


def test_invalid_mask_std_fails_closed():
    req = validate_joint_noise_requirement(2, 2, 0.5)
    shares = iid_exact_target_noise_shares(requirement=req, dimension=8)
    try:
        add_pairwise_zero_sum_noise_masks(shares, mask_std=0.0)
    except ValueError as exc:
        assert "mask_std" in str(exc)
    else:
        raise AssertionError("non-positive mask std must fail closed")
