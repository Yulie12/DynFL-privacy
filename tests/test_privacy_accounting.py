from __future__ import annotations

import math

from dynfed.fmnist_lenet5_dynamic import _should_apply_update_dp
from dynfed.privacy import (
    ClientPrivacyLedger,
    PrivacyAccountant,
    calibrate_gaussian_noise,
    mechanism_uses_dp,
    mechanism_uses_he,
    privacy_execution_audit,
)
from dynfed.selection import SelectionConfig, resolved_privacy_parameters


def test_zero_events_have_zero_epsilon() -> None:
    accountant = PrivacyAccountant(budget=8.0, delta=1e-5)
    assert accountant.current_epsilon() == 0.0
    assert accountant.epsilon_after(noise_multiplier=2.0, event_count=0) == 0.0


def test_calibrated_noise_reaches_fixed_total_target() -> None:
    noise_multiplier = calibrate_gaussian_noise(8.0, 1e-5, 100)
    accountant = PrivacyAccountant(budget=8.0, delta=1e-5)
    accountant.add_events(noise_multiplier, 100)
    assert math.isclose(accountant.current_epsilon(), 8.0, rel_tol=1e-10, abs_tol=1e-10)


def test_feature_and_update_guarantees_are_not_added() -> None:
    ledger = ClientPrivacyLedger(
        feature_budget=8.0,
        update_budget=4.0,
        delta=1e-5,
        feature_noise_multiplier=calibrate_gaussian_noise(8.0, 1e-5, 10),
        update_noise_multiplier=calibrate_gaussian_noise(4.0, 1e-5, 2),
    )
    projection = ledger.add(feature_events=10, update_events=2)
    assert math.isclose(projection.feature_epsilon_after, 8.0, rel_tol=1e-10)
    assert math.isclose(projection.update_epsilon_after, 4.0, rel_tol=1e-10)
    assert ledger.remaining_budget <= 1e-10


def test_more_rounds_increase_noise_not_total_target() -> None:
    fifty = resolved_privacy_parameters(SelectionConfig(rounds=50, initial_epsilon=8.0))
    hundred = resolved_privacy_parameters(SelectionConfig(rounds=100, initial_epsilon=8.0))
    assert fifty["feature_budget"] == hundred["feature_budget"] == 8.0
    assert fifty["update_budget"] == hundred["update_budget"] == 8.0
    assert fifty["feature_horizon_events"] == hundred["feature_horizon_events"] == 0
    assert fifty["feature_noise_multiplier"] == hundred["feature_noise_multiplier"] == 1.0
    assert hundred["update_noise_multiplier"] > fifty["update_noise_multiplier"]


def test_update_dp_is_applied_on_the_tex_eligible_stage() -> None:
    mechanisms = {"upd": "dp"}
    assert _should_apply_update_dp(mechanisms, "upd_only", "LIIEIIIC")
    assert _should_apply_update_dp(mechanisms, "upd_only", "LIEIIC")
    assert not _should_apply_update_dp(mechanisms, "upd_only", "LIEIIIC")


def test_dp_and_he_are_distinct_update_mechanisms() -> None:
    assert mechanism_uses_dp("dp")
    assert not mechanism_uses_he("dp")
    assert mechanism_uses_he("he3")
    assert not mechanism_uses_dp("he3")


def test_combined_update_mechanism_uses_both_guarantees() -> None:
    assert mechanism_uses_dp("dp_he3")
    assert mechanism_uses_he("dp_he3")
    mechanisms = {"upd": "dp_he3"}
    assert _should_apply_update_dp(mechanisms, "upd_only", "LIEIIC")


def test_zero_dp_events_and_profiled_he_do_not_imply_transcript_dp() -> None:
    audit = privacy_execution_audit([{
        "num_global_update_clients": 10,
        "uniform_update_dp": False,
        "max_update_epsilon": 0.0,
        "he_execution_status": "profiled",
    }])
    assert audit["uniform_selected_update_dp"] is False
    assert audit["end_to_end_dp_status"] == "not_established"
    assert audit["end_to_end_epsilon"] is None
    assert audit["he_real_execution_rounds"] == 0
    assert audit["he_profiled_execution_rounds"] == 1


def test_full_dp_coverage_is_not_a_privacy_proof() -> None:
    audit = privacy_execution_audit([{
        "num_global_update_clients": 10,
        "uniform_update_dp": True,
        "he_execution_status": "real",
    }])
    assert audit["uniform_selected_update_dp"] is True
    assert audit["end_to_end_dp_status"] == "not_established"
    assert audit["he_real_execution_rounds"] == 1


def test_empty_and_legacy_audits_do_not_claim_full_coverage() -> None:
    assert privacy_execution_audit([])["uniform_selected_update_dp"] is None
    assert privacy_execution_audit([{"num_global_update_clients": 10}])["uniform_selected_update_dp"] is None


def test_later_dp_does_not_erase_an_earlier_non_dp_release() -> None:
    audit = privacy_execution_audit([
        {"num_global_update_clients": 10, "uniform_update_dp": False},
        {"num_global_update_clients": 10, "uniform_update_dp": True},
    ])
    assert audit["uniform_selected_update_dp"] is False
