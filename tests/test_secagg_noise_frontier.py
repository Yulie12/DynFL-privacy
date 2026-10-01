import math

import pytest

from experiments.secagg_noise_frontier_common import (
    build_frontier,
    collusion_robust_independent_point,
    edge_only_exact_point,
    threshold_joint_noise_target,
)


def test_edge_only_exact_has_no_aggregate_noise_inflation():
    p = edge_only_exact_point(100, 2, 0.5)
    assert p.released_noise_std == pytest.approx(0.5)
    assert p.std_inflation == pytest.approx(1.0)
    assert p.energy_inflation == pytest.approx(1.0)
    assert p.conditional_unknown_variance_ratio == pytest.approx(2 / 100)
    assert not p.full_target_dp_after_minimum_collusion_conditioning


def test_collusion_robust_independent_matches_k_over_h_energy_law():
    p = collusion_robust_independent_point(100, 2, 0.5)
    assert p.std_inflation == pytest.approx(math.sqrt(50))
    assert p.energy_inflation == pytest.approx(50)
    assert p.conditional_unknown_variance_ratio == pytest.approx(1.0)
    assert p.full_target_dp_after_minimum_collusion_conditioning


def test_no_inflation_when_every_client_share_must_remain_unknown():
    p = collusion_robust_independent_point(10, 10, 0.5)
    assert p.std_inflation == pytest.approx(1.0)
    assert p.energy_inflation == pytest.approx(1.0)
    assert p.edge_only_exact_target


def test_threshold_joint_point_is_explicitly_not_implemented():
    p = threshold_joint_noise_target(100, 2, 0.5)
    assert p["std_inflation"] == 1.0
    assert p["full_target_dp_after_minimum_collusion_conditioning"] is True
    assert str(p["implementation_status"]).startswith("NOT_IMPLEMENTED")


def test_frontier_skips_h_larger_than_k_and_returns_three_contracts():
    rows = build_frontier([4], [2, 5], 0.5)
    assert len(rows) == 3
    assert {row["scheme"] for row in rows} == {
        "edge_only_exact",
        "collusion_robust_independent",
        "threshold_joint_exact_target",
    }


@pytest.mark.parametrize("k,h", [(1, 1), (10, 1), (10, 11)])
def test_invalid_frontier_inputs_rejected(k, h):
    with pytest.raises(ValueError):
        edge_only_exact_point(k, h, 0.5)
