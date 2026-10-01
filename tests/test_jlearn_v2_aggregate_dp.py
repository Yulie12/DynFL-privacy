from __future__ import annotations

import pytest

from dynfed.selection import (
    Candidate,
    SelectionConfig,
    _evaluate_profile,
    _global_dp_perturbation_cost,
    _global_update_clip_perturbation_cost,
    _local_omega_proxy,
)


def _candidate(mode: str, link_id: str, mechanism: str = "dp", sigma: float = 0.5) -> Candidate:
    return Candidate(
        mode=mode,
        mechanisms={"upd": mechanism},
        link_mechanisms={link_id: mechanism},
        time=1.0,
        accuracy=0.0,
        risk=0.0,
        epsilon_used=1.0,
        communication_volume=1.0,
        feasible_resource=True,
        feasible_privacy=True,
        feasible_risk=True,
        feasible_time=True,
        update_noise_multiplier=sigma,
    )


def test_liic_exact_target_cost_is_charged_once_at_aggregate_sensitivity() -> None:
    k = 10
    sigma = 0.5
    clip = 0.25
    dim = 1000.0
    config = SelectionConfig(
        omega_update_dimension=dim,
        omega_update_clip_norm=clip,
        omega_update_clip_excess_sq=0.0,
    )
    profile = {i: _candidate("LIIC", "L_C_upd", sigma=sigma) for i in range(k)}
    samples = {i: 1.0 for i in profile}
    edges = {i: i % 2 for i in profile}

    actual = _global_dp_perturbation_cost(config, profile, samples, edges, tuple(profile))
    expected = dim * (sigma * (2.0 * clip / k)) ** 2
    legacy_local_packet = dim * (sigma * 2.0 * clip) ** 2 / k

    assert actual == pytest.approx(expected)
    assert legacy_local_packet / actual == pytest.approx(float(k))


def test_liie_edge_local_cost_uses_each_edge_cohort_sensitivity() -> None:
    sigma = 0.5
    clip = 0.25
    dim = 1000.0
    config = SelectionConfig(
        omega_update_dimension=dim,
        omega_update_clip_norm=clip,
        omega_update_clip_excess_sq=0.0,
    )
    # Two equal edges, each with five clients.  Each edge release therefore has
    # max local weight 1/5 and both edges have the same expected noise energy.
    profile = {i: _candidate("LIIE", "L_E_upd", sigma=sigma) for i in range(10)}
    samples = {i: 1.0 for i in profile}
    edges = {i: i // 5 for i in profile}

    actual = _global_dp_perturbation_cost(config, profile, samples, edges, tuple(profile))
    per_edge = dim * (sigma * (2.0 * clip / 5.0)) ** 2

    assert actual == pytest.approx(per_edge)


def test_liieiiic_hierarchical_cost_matches_one_global_exact_target_release() -> None:
    k = 10
    sigma = 0.5
    clip = 0.25
    dim = 1000.0
    config = SelectionConfig(
        omega_update_dimension=dim,
        omega_update_clip_norm=clip,
        omega_update_clip_excess_sq=0.0,
    )
    profile = {i: _candidate("LIIEIIIC", "E_C_upd", sigma=sigma) for i in range(k)}
    samples = {i: 1.0 for i in profile}
    edges = {i: i // 5 for i in profile}

    actual = _global_dp_perturbation_cost(config, profile, samples, edges, tuple(profile))
    expected = dim * (sigma * (2.0 * clip / k)) ** 2

    assert actual == pytest.approx(expected)


def test_update_clip_is_diagnostic_only_for_formal_jlearn() -> None:
    k = 4
    clip_excess_sq = 0.03
    config = SelectionConfig(
        omega_update_dimension=100.0,
        omega_update_clip_norm=0.25,
        omega_update_clip_excess_sq=clip_excess_sq,
        cloud_fusion_xi=0.0,
    )
    profile = {i: _candidate("LIIC", "L_C_upd", sigma=0.5) for i in range(k)}
    samples = {i: 1.0 for i in profile}
    edges = {i: 0 for i in profile}

    clip_cost = _global_update_clip_perturbation_cost(config, profile, samples, tuple(profile))
    result = _evaluate_profile(
        config,
        profile,
        samples,
        edges,
        previous_choices={},
        flow_objectives=(tuple(profile), 1.0),
    )

    assert clip_cost == pytest.approx(clip_excess_sq)
    assert result.update_clip_perturbation == pytest.approx(clip_excess_sq)
    assert result.system_learning_error == pytest.approx(
        result.fusion_bound + result.dp_perturbation
    )


def test_dp_he3_full_local_cloud_path_keeps_aggregate_boundary_cost() -> None:
    k = 5
    sigma = 0.5
    clip = 0.25
    dim = 1000.0
    config = SelectionConfig(
        omega_update_dimension=dim,
        omega_update_clip_norm=clip,
        omega_update_clip_excess_sq=0.0,
    )
    profile = {
        i: _candidate("LIIEIIIC", "E_C_upd", mechanism="dp_he3", sigma=sigma)
        for i in range(k)
    }
    samples = {i: 1.0 for i in profile}
    edges = {i: i % 2 for i in profile}

    actual = _global_dp_perturbation_cost(config, profile, samples, edges, tuple(profile))
    expected = dim * (sigma * (2.0 * clip / k)) ** 2
    assert actual == pytest.approx(expected)


def test_liic_local_search_proxy_uses_aggregate_boundary_scaling() -> None:
    k = 10
    config = SelectionConfig(
        num_clients=k,
        num_edges=2,
        omega_local_variance=0.0,
        omega_update_dimension=1000.0,
        omega_update_clip_norm=0.25,
        omega_update_clip_excess_sq=0.0,
    )
    candidate = _candidate("LIIC", "L_C_upd", sigma=0.5)

    singleton = _local_omega_proxy(candidate, aggregation_size=1, config=config)
    search_proxy = _local_omega_proxy(candidate, config=config)

    assert singleton > 0.0
    assert search_proxy == pytest.approx(singleton / (k * k))


def test_liie_local_search_proxy_uses_expected_edge_cohort_scaling() -> None:
    config = SelectionConfig(
        num_clients=10,
        num_edges=2,
        omega_local_variance=0.0,
        omega_update_dimension=1000.0,
        omega_update_clip_norm=0.25,
        omega_update_clip_excess_sq=0.0,
    )
    candidate = _candidate("LIIE", "L_E_upd", sigma=0.5)

    singleton = _local_omega_proxy(candidate, aggregation_size=1, config=config)
    search_proxy = _local_omega_proxy(candidate, config=config)

    expected_cohort = 5
    assert singleton > 0.0
    assert search_proxy == pytest.approx(singleton / (expected_cohort * expected_cohort))


def test_split_local_packet_dp_keeps_legacy_local_search_proxy() -> None:
    config = SelectionConfig(
        num_clients=10,
        num_edges=2,
        omega_local_variance=0.0,
        omega_update_dimension=1000.0,
        omega_update_clip_norm=0.25,
        omega_update_clip_excess_sq=0.0,
    )
    candidate = _candidate("LIEIIC", "L_C_upd", mechanism="dp", sigma=0.5)

    singleton = _local_omega_proxy(candidate, aggregation_size=1, config=config)
    search_proxy = _local_omega_proxy(candidate, config=config)

    assert singleton > 0.0
    assert search_proxy == pytest.approx(singleton)
