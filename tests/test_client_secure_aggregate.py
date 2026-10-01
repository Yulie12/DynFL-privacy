import math

import pytest
import torch

from experiments.client_secure_aggregate_common import (
    ClientMaskingContext,
    client_secure_aggregate_plan,
    make_masked_client_packet,
    protocol_audit,
    unknown_client_noise_sufficient,
    untrusted_edge_fixed_cohort_sum,
)


def _cohort(k):
    contexts = [ClientMaskingContext(i) for i in range(k)]
    return contexts, {c.client_id: c.public_key for c in contexts}


def test_uniform_replacement_sensitivity_scales_as_two_c_over_k():
    k, c = 10, 0.25
    plan = client_secure_aggregate_plan([1 / k] * k, c, 2.0, 2)
    assert plan.aggregate_sensitivity == pytest.approx(2 * c / k)


def test_nonuniform_sensitivity_uses_largest_public_weight():
    weights = [0.1, 0.2, 0.7]
    plan = client_secure_aggregate_plan(weights, 0.25, 2.0, 2)
    assert plan.aggregate_sensitivity == pytest.approx(2 * 0.25 * 0.7)


def test_noise_guarantee_is_conditional_on_minimum_unknown_clients():
    plan = client_secure_aggregate_plan([0.25] * 4, 0.25, 2.0, 3)
    assert unknown_client_noise_sufficient(plan, [0, 1, 2])
    assert not unknown_client_noise_sufficient(plan, [0, 1])
    assert plan.all_clients_noise_std / plan.target_noise_std == pytest.approx(math.sqrt(4 / 3))


def test_pairwise_masks_cancel_and_edge_gets_only_noisy_aggregate(monkeypatch):
    # Remove DP noise so this test isolates exact pairwise-mask cancellation.
    import experiments.client_secure_aggregate_common as common
    monkeypatch.setattr(common, "_private_noise", lambda dimension, std: torch.zeros(dimension, dtype=torch.float64))
    k, d = 4, 32
    plan = client_secure_aggregate_plan([1 / k] * k, 10.0, 1.0, 2)
    contexts, public_keys = _cohort(k)
    updates = [torch.arange(d, dtype=torch.float64) * (i + 1) / 100 for i in range(k)]
    packets = {}
    for context, update in zip(contexts, updates):
        packets[context.client_id], metrics = make_masked_client_packet(
            context=context, update=update, public_keys=public_keys, plan=plan, round_idx=7
        )
        assert metrics["individual_plaintext_sent_to_edge"] is False
    actual = untrusted_edge_fixed_cohort_sum(packets, cohort_size=k)
    expected = sum(update * (1 / k) for update in updates)
    assert torch.allclose(actual, expected, rtol=0, atol=1e-10)


def test_dropout_aborts_instead_of_releasing_uncancelled_masks(monkeypatch):
    import experiments.client_secure_aggregate_common as common
    monkeypatch.setattr(common, "_private_noise", lambda dimension, std: torch.zeros(dimension, dtype=torch.float64))
    k = 3
    plan = client_secure_aggregate_plan([1 / k] * k, 1.0, 1.0, 2)
    contexts, public_keys = _cohort(k)
    packets = {}
    for context in contexts[:-1]:
        packets[context.client_id], _ = make_masked_client_packet(
            context=context, update=torch.zeros(8), public_keys=public_keys,
            plan=plan, round_idx=0
        )
    with pytest.raises(ValueError, match="abort"):
        untrusted_edge_fixed_cohort_sum(packets, cohort_size=k)


def test_audit_only_closes_under_fixed_cohort_and_noise_assumption():
    plan = client_secure_aggregate_plan([0.25] * 4, 0.25, 2.0, 2)
    ok = protocol_audit(plan, received_client_ids=[0, 1, 2, 3], assumed_unknown_client_ids=[0, 1])
    assert ok["protocol_status"] == "untrusted_edge_secure_aggregate_dp_closed_conditional"
    assert ok["individual_plaintext_update_visible_to_edge"] is False
    assert ok["clean_aggregate_visible_to_edge"] is False
    bad = protocol_audit(plan, received_client_ids=[0, 1, 2], assumed_unknown_client_ids=[0, 1])
    assert bad["protocol_status"] == "not_established"


@pytest.mark.parametrize("weights,minimum", [([0.5, 0.4], 2), ([1.0], 1), ([0.5, 0.5], 1), ([0.5, 0.5], 3)])
def test_invalid_plans_rejected(weights, minimum):
    with pytest.raises(ValueError):
        client_secure_aggregate_plan(weights, 0.25, 1.0, minimum)
