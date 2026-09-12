import random
from dataclasses import replace

import pytest

from dynfed.selection import (
    SelectionConfig, _stable_cloud_candidate_pool, candidate_meets_update_goal,
    enumerate_candidates, validate_update_protection_goal,
)


def pool(goal="released_model_dp", policy="ours", **changes):
    config = replace(SelectionConfig(
        trusted_edge_split_execution=True, update_protection_goal=goal,
        require_cloud_participation=True,
    ), **changes)
    candidates = enumerate_candidates(
        config=config, client_id=0, edge_factor=1.0, compute_factor=1.0,
        samples=120, remaining_epsilon=8.0, round_idx=0,
        rng=random.Random(42), policy=policy,
    )
    return config, candidates


@pytest.mark.parametrize("policy", ["ours", "individual_optimal", "random", "nsga2",
                                    "fixed_splitfed", "fixed_dp", "fixed_dp_he"])
def test_actual_candidate_enumeration_requires_cloud_dp(policy):
    config, candidates = pool(policy=policy)
    assert candidates
    for candidate in candidates:
        links = [m for k, m in candidate.link_mechanisms.items() if k.endswith("_C_upd")]
        assert links and all(m in {"dp", "dp_he3"} for m in links)
        assert candidate_meets_update_goal(config, candidate)


def test_legacy_pool_retains_pure_he_but_strict_gate_rejects_reuse():
    config, candidates = pool("packet_protection")
    pure_he = [c for c in candidates if "he3" in c.link_mechanisms.values()
               and not any("dp" in m for m in c.link_mechanisms.values())]
    assert pure_he
    strict = replace(config, update_protection_goal="released_model_dp")
    assert all(not candidate_meets_update_goal(strict, c) for c in pure_he)
    assert not candidate_meets_update_goal(strict, replace(pure_he[0], link_mechanisms={}))


@pytest.mark.parametrize("policy", ["fixed_he", "no_protection", "fixed_splitfed_trusted_edge",
                                    "fixed_splitfed_no_protection"])
def test_incompatible_controls_fail_explicitly(policy):
    with pytest.raises(ValueError, match="control without result DP"):
        pool(policy=policy)


def test_stability_filter_cannot_restore_pure_he():
    config, candidates = pool(enforce_cloud_dp_stability=True)
    filtered = _stable_cloud_candidate_pool(config, candidates)
    assert filtered
    assert all(candidate_meets_update_goal(config, c) for c in filtered)


def test_unknown_goal_and_unsupported_trust_boundary_fail():
    with pytest.raises(ValueError, match="Unknown"):
        validate_update_protection_goal(SelectionConfig(update_protection_goal="typo"), "ours")
    with pytest.raises(ValueError, match="trusted edge"):
        pool(trusted_edge_split_execution=False)


def test_disabling_he_keeps_dp_only():
    _, candidates = pool(allow_he=False)
    assert candidates
    assert all(not any("he" in m for m in c.link_mechanisms.values()) for c in candidates)
