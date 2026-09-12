import pytest

from dynfed.he_backend import check_he_backend


def test_current_arithmetic_harness_reports_its_actual_capabilities():
    availability = check_he_backend("seal")
    if not availability.available:
        pytest.skip(availability.detail)
    from experiments.audit_he_release_boundary import run_audit

    report = run_audit()
    assert report["runtime_holds_secret_key"]
    assert report["individual_ciphertext_decryption_succeeds"]
    assert report["fixed_supplied_plan_missing_packet_rejected"]
    assert report["caller_can_supply_new_single_edge_plan"]
    assert not report["authorized_cohort_binding_enforced"]
    assert not report["key_isolation_enforced"]
    assert not report["aggregate_only_decryption_enforced"]
    assert not report["protocol_ready"]
