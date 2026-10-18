"""The unvalidated mode-risk constants must not affect legal candidates.

This test exercises the real candidate enumeration path, not a mock
Candidate list. Trust, DP and resource checks remain independent.
"""
import random
from dataclasses import replace

from dynfed.selection import SelectionConfig, enumerate_candidates


def _candidates(config):
    return enumerate_candidates(
        config=config,
        client_id=0,
        connected_edge_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=120,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(2026),
        policy="full_dynfl",
    )


def test_liieiiic_is_no_longer_rejected_by_uncalibrated_risk():
    cfg = SelectionConfig(
        rounds=5,
        privacy_unit="sample",
        dp_accounting_mode="rdp_auto",
        fl_first_split_on_demand=True,
        corrected_admission_policy=True,
        strict_pair_admission=True,
        edge_only_requires_fast_deadline=True,
        num_clients=100,
        num_edges=10,
        privacy_local_epochs=3,
        risk_limit=0.5,
    )
    candidates = _candidates(cfg)
    by_mode = {c.mode: c for c in candidates}
    assert set(by_mode) == {"LIIC", "LIIEIIIC"}
    for mode in ("LIIC", "LIIEIIIC"):
        c = by_mode[mode]
        assert c.feasible
        assert c.feasible_risk
        assert not c.risk_hard_gate
        assert c.risk == 0.0  # neutral legacy field, NOT proven zero risk
        assert c.feasible_privacy
    assert by_mode["LIIEIIIC"].link_mechanisms == {
        "L_E_upd": "none", "E_C_upd": "he3"
    }


def test_risk_limit_changes_do_not_change_admission_or_ranking_inputs():
    config = SelectionConfig(
        rounds=5,
        privacy_unit="sample",
        dp_accounting_mode="rdp_auto",
        corrected_admission_policy=True,
        strict_pair_admission=True,
        fl_first_split_on_demand=True,
        edge_only_requires_fast_deadline=True,
        num_clients=100,
        num_edges=10,
        privacy_local_epochs=3,
    )
    a = _candidates(replace(config, risk_limit=0.01))
    b = _candidates(replace(config, risk_limit=0.99))
    assert a == b


def test_independent_privacy_resource_time_admission_remains():
    c = next(item for item in _candidates(SelectionConfig(
        rounds=5, privacy_unit="sample", dp_accounting_mode="rdp_auto",
        corrected_admission_policy=True, strict_pair_admission=True,
        fl_first_split_on_demand=True, edge_only_requires_fast_deadline=True,
        num_clients=100, num_edges=10, privacy_local_epochs=3,
    )) if item.mode == "LIIEIIIC")
    for field in ("feasible_resource", "feasible_memory", "feasible_privacy", "feasible_time", "feasible_edge", "feasible_cloud"):
        assert not replace(c, **{field: False}).feasible, field


def test_no_unjustified_risk_matrices_remain():
    from dynfed import selection
    assert not hasattr(selection, "OBJECT_RISK")
    assert not hasattr(selection, "MECHANISM_RISK")
