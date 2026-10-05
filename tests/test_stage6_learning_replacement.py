"""Stage 6: optimized one-client J_learn agrees with the complete objective."""

from dataclasses import replace
import random
import time

import pytest

from dynfed.selection import (
    Candidate, SelectionConfig, _LearningReplacementStats,
    _selection_learning_cost,
)


def _candidate(mode: str, link: str = "", mechanism: str = "none", plan: str = "independent", sigma: float = 0.5) -> Candidate:
    return Candidate(
        mode=mode,
        mechanisms={"upd": mechanism},
        link_mechanisms={link: mechanism} if link else {},
        time=1.0, accuracy=0.0, risk=0.0, epsilon_used=1.0,
        communication_volume=1.0, feasible_resource=True,
        feasible_privacy=True, feasible_risk=True, feasible_time=True,
        update_noise_multiplier=sigma, dp_execution_plan=plan,
    )


CANDIDATES = (
    _candidate("LIIE", "L_E_upd", "none"),
    _candidate("LIIE", "L_E_upd", "dp"),
    _candidate("LIIE", "L_E_upd", "he3"),
    _candidate("LIIC", "L_C_upd", "dp_he3"),
    _candidate("LIIEIIIC", "E_C_upd", "dp_he3", "cloud_packet"),
    _candidate("LIIEIIIC", "E_C_upd", "dp_he3", "cloud_aggregate"),
    _candidate("LIEIIC", "E_C_upd", "dp_he3", "cloud_packet"),
    _candidate("LIEIIIC", "E_C_upd", "dp_he3", "independent"),
    _candidate("LIC", "L_C_emb", "dp"),
    _candidate("LIE", "L_E_emb", "dp"),
)


def test_learning_replacement_matches_full_cost_for_heterogeneous_profiles():
    rng = random.Random(1234)
    config = SelectionConfig(omega_update_dimension=5130.0, omega_update_clip_norm=0.25)
    n = 50
    masses = {i: float(rng.randrange(1, 20)) for i in range(n)}
    edges = {i: i % 5 for i in range(n)}
    admission = tuple(range(n))
    for _ in range(6):
        profile = {i: rng.choice(CANDIDATES) for i in range(n)}
        # Exercise packet, secure aggregate, independent Cloud and local Edge costs.
        stats = _LearningReplacementStats(config, profile, masses, edges, admission)
        for cid in range(n):
            for candidate in CANDIDATES:
                if candidate == profile[cid]:
                    continue
                trial = dict(profile)
                trial[cid] = candidate
                actual = stats.replacement_cost(cid, candidate, admission)
                expected = _selection_learning_cost(config, trial, masses, edges, admission)
                assert actual == pytest.approx(expected, abs=1e-8, rel=1e-11)


def test_learning_replacement_handles_admission_change_with_full_fallback():
    config = SelectionConfig(omega_update_dimension=30.0, omega_update_clip_norm=0.25)
    profile = {0: CANDIDATES[0], 1: CANDIDATES[3]}
    masses = {0: 3.0, 1: 5.0}
    edges = {0: 0, 1: 0}
    stats = _LearningReplacementStats(config, profile, masses, edges, (0, 1))
    assert stats.replacement_cost(0, CANDIDATES[3], (1,)) is None


def test_learning_replacement_preserves_edge_aggregate_cohort_constraint():
    config = SelectionConfig(omega_update_dimension=30.0, omega_update_clip_norm=0.25)
    aggregated = _candidate("LIIE", "L_E_upd", "dp", "aggregate")
    profile = {0: aggregated, 1: aggregated, 2: CANDIDATES[3]}
    masses = {0: 2.0, 1: 4.0, 2: 3.0}
    edges = {0: 0, 1: 0, 2: 1}
    stats = _LearningReplacementStats(config, profile, masses, edges, (0, 1, 2))
    assert stats.replacement_cost(2, CANDIDATES[3], (0, 1, 2)) == pytest.approx(
        _selection_learning_cost(config, profile, masses, edges, (0, 1, 2))
    )
    with pytest.raises(ValueError, match="Inconsistent LIIE aggregate DP cohort"):
        stats.replacement_cost(0, CANDIDATES[1], (0, 1, 2))
