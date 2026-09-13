import pytest

from dynfed.resource_selection import DEFAULT_PROFILE, select_placement


def test_deterministic_public_plan_and_no_utility_score():
    args = ([100] * 6, 2, 3, 2048, 5130, 0, 42, DEFAULT_PROFILE)
    first = select_placement(*args)
    assert first == select_placement(*args)
    assert set(first[0]) == set(range(6))
    assert not first[1]["accuracy_objective_used"]
    assert first[1]["resource_feasible"]


def test_edge_work_limit_is_hard_constraint():
    profile = dict(DEFAULT_PROFILE, edge_work_limit_sec=1e-9)
    plan, metrics = select_placement([100] * 4, 2, 3, 1, 1000, 0, 42, profile)
    assert set(plan.values()) == {"LIIC"}
    assert metrics["max_edge_work_sec"] == 0


def test_fast_edge_can_be_selected_without_accuracy_bonus():
    profile = dict(DEFAULT_PROFILE, tail_sec_per_sample=1., edge_heterogeneity=100.,
                   communication_weight=0., edge_work_limit_sec=10000.)
    plan, metrics = select_placement([1] * 2, 2, 1, 1, 10, 0, 42, profile)
    assert "LIEIIC" in plan.values()


def test_bad_profile_rejected():
    with pytest.raises(ValueError):
        select_placement([10], 1, 1, 1, 1, 0, 42, dict(DEFAULT_PROFILE, rate_mb_s=0))
