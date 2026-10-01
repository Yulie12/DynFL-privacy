import torch

from experiments.client_secure_aggregate_common import (
    ClientMaskingContext,
    client_secure_aggregate_plan,
    make_masked_client_packet_with_external_noise,
    untrusted_edge_fixed_cohort_sum,
)
from experiments.joint_noise_mpc_common import (
    ideal_joint_gaussian_shares,
    joint_noise_contract_audit,
    validate_joint_noise_requirement,
)


def test_iid_exact_and_collusion_robust_tradeoff_is_explicit():
    req = validate_joint_noise_requirement(100, 2, 0.5)
    report = joint_noise_contract_audit(req)
    assert report["iid_exact_conditional_variance_ratio"] == 0.02
    assert report["iid_robust_energy_inflation"] == 50.0
    assert report["protocol_status"] == "ideal_functionality_only_not_a_security_closure"


def test_ideal_joint_shares_sum_exactly_but_are_not_crypto_backend():
    req = validate_joint_noise_requirement(10, 2, 0.2)
    shares, target, meta = ideal_joint_gaussian_shares(requirement=req, dimension=257)
    total = torch.stack([shares[i] for i in range(10)]).sum(0)
    assert torch.allclose(total, target, rtol=0.0, atol=1e-10)
    assert meta["exact_target_sum"]
    assert not meta["cryptographic_realization"]
    assert not meta["collusion_resistance_established"]


def test_secagg_accepts_external_joint_noise_share_and_preserves_exact_sum():
    k, d = 4, 64
    plan = client_secure_aggregate_plan([0.25] * k, 1.0, 0.5, 2)
    req = validate_joint_noise_requirement(k, 2, plan.target_noise_std)
    shares, target, _ = ideal_joint_gaussian_shares(requirement=req, dimension=d)
    contexts = [ClientMaskingContext(i) for i in range(k)]
    public_keys = {c.client_id: c.public_key for c in contexts}
    zero = torch.zeros(d, dtype=torch.float64)
    packets = {}
    for i, context in enumerate(contexts):
        packets[i], meta = make_masked_client_packet_with_external_noise(
            context=context,
            update=zero,
            external_noise_share=shares[i],
            public_keys=public_keys,
            plan=plan,
            round_idx=7,
        )
        assert meta["noise_source"] == "external_joint_noise_share"
    released = untrusted_edge_fixed_cohort_sum(packets, cohort_size=k)
    assert torch.allclose(released, target, rtol=0.0, atol=1e-9)


def test_external_noise_dimension_mismatch_fails_closed():
    plan = client_secure_aggregate_plan([0.5, 0.5], 1.0, 0.5, 2)
    contexts = [ClientMaskingContext(i) for i in range(2)]
    public_keys = {c.client_id: c.public_key for c in contexts}
    try:
        make_masked_client_packet_with_external_noise(
            context=contexts[0],
            update=torch.zeros(8),
            external_noise_share=torch.zeros(7),
            public_keys=public_keys,
            plan=plan,
            round_idx=0,
        )
    except ValueError as exc:
        assert "matching update dimension" in str(exc)
    else:
        raise AssertionError("dimension mismatch must fail closed")
