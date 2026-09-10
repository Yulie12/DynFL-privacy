import pytest

from dynfed.privacy import PrivacyAccountant, calibrate_gaussian_noise
from dynfed.protection_rules import allowed_update_mechanisms, audit_update_release
from dynfed.protection_rules import aggregate_replacement_bound


def test_independent_packets_have_max_weight_sensitivity():
    assert aggregate_replacement_bound([0.25, 0.75], [{0}, {1}], clip_norm=1) == 1.5


def test_shared_state_can_remove_averaging_sensitivity_reduction():
    bound = aggregate_replacement_bound([0.5, 0.5], [{0, 1}, {0, 1}], clip_norm=1)
    # Both bounded scalar packets can change from +1 to -1 together.
    actual_change = abs((0.5 + 0.5) - (-0.5 - 0.5))
    assert bound == actual_change == 2.0


def test_edge_packet_dependency_uses_edge_weight_not_client_fraction():
    assert aggregate_replacement_bound(
        [0.5, 0.5], [{0, 1, 2}, {3, 4}], clip_norm=1,
    ) == 1.0


@pytest.mark.parametrize("weights,deps", [([], []), ([1.0], []),
    ([0.5], [{0}]), ([float("nan")], [{0}]), ([-1.0, 2.0], [{0}, {1}])])
def test_invalid_release_weights_are_rejected(weights, deps):
    with pytest.raises(ValueError):
        aggregate_replacement_bound(weights, deps, clip_norm=1)


def release(**changes):
    args = dict(mechanism="he3", noise_location="none", he_execution="real",
                dp_budget_ok=True, key_isolation_enforced=True,
                aggregate_only_decryption_enforced=True)
    args.update(changes)
    return audit_update_release(**args)


def test_budget_exhaustion_can_leave_he_only_for_packet_goal():
    ledger = PrivacyAccountant(8.0, 1e-5)
    sigma = calibrate_gaussian_noise(8.0, 1e-5, 1)
    ledger.add_events(sigma, 1)
    affordable = ledger.epsilon_after(sigma, 1) <= ledger.budget + 1e-12
    assert not affordable
    assert allowed_update_mechanisms(goal="packet_protection",
        next_dp_release_affordable=affordable, secure_aggregation_available=True) == ("he3",)
    assert allowed_update_mechanisms(goal="released_model_dp",
        next_dp_release_affordable=affordable, secure_aggregation_available=True) == ()


def test_he_does_not_consume_or_reset_dp_ledger():
    ledger = PrivacyAccountant(8.0, 1e-5)
    sigma = calibrate_gaussian_noise(8.0, 1e-5, 10)
    ledger.add_events(sigma, 2)
    previous = ledger.state_dict()
    assert release()["he_crypto_observed"]
    assert ledger.state_dict() == previous
    assert release()["released_model_dp_status"] == "no_valid_dp_calibration_for_contribution"


def test_output_goal_excludes_pure_he_even_with_available_budget():
    assert allowed_update_mechanisms(goal="released_model_dp",
        next_dp_release_affordable=True, secure_aggregation_available=True) == ("dp", "dp_he3")


def test_no_available_protection_does_not_fall_back_to_plaintext():
    assert allowed_update_mechanisms(goal="packet_protection",
        next_dp_release_affordable=False, secure_aggregation_available=False) == ()


def test_real_ckks_without_key_isolation_is_not_secure_aggregation():
    result = release(key_isolation_enforced=False)
    assert result["packet_protection_status"] == "real_he_pending_role_isolation"
    assert result["he_crypto_observed"]


def test_individual_ciphertext_decryption_is_not_aggregate_only():
    result = release(aggregate_only_decryption_enforced=False)
    assert result["packet_protection_status"] == "real_he_pending_role_isolation"


def test_profiled_he_does_not_cover_a_real_packet():
    result = release(he_execution="profiled")
    assert result["packet_protection_status"] == "profiled_he_only"
    assert not result["he_crypto_observed"]


def test_aggregate_noise_shares_alone_are_not_packet_dp():
    result = release(mechanism="dp_he3", noise_location="aggregate_share", he_execution="profiled")
    assert result["packet_protection_status"] == "aggregate_noise_without_secure_transport"
    assert result["formal_dp_status"] == "not_established"


def test_actual_packet_dp_and_budget_must_both_be_present():
    assert release(mechanism="dp", noise_location="none", he_execution="not_selected")[
        "packet_protection_status"] == "dp_requested_but_not_observed"
    result = release(mechanism="dp", noise_location="packet",
                     he_execution="not_selected", dp_budget_ok=False)
    assert result["packet_protection_status"] == "dp_budget_exceeded"
    assert result["released_model_dp_status"] == "no_valid_dp_calibration_for_contribution"


@pytest.mark.parametrize("goal", ["epsilon_zero", "encrypt_everything", ""])
def test_unknown_goal_is_rejected(goal):
    with pytest.raises(ValueError):
        allowed_update_mechanisms(goal=goal, next_dp_release_affordable=True,
                                  secure_aggregation_available=True)
