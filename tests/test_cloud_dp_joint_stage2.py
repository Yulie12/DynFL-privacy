"""Cloud DP plan participates in whole-profile selection, not post-hoc rewriting."""
from dataclasses import replace
import pytest

from dynfed.selection import (
    Candidate, SelectionConfig, _candidate_key, _global_dp_perturbation_cost,
    _with_cloud_plan, _candidate_uses_secure_aggregate_update_dp_for_selection,
    choose_cloud_dp_pareto_profile,
)
from dynfed.fmnist_lenet5_dynamic import (
    _candidate_uses_local_packet_update_dp,
    _candidate_uses_secure_aggregate_update_dp,
    _candidate_worker_update_dp_applied,
)


def candidate(mode='LIIC', link='L_C_upd', mechanism='dp_he3'):
    links = {link: mechanism}
    if mode == 'LIIEIIIC':
        links['L_E_upd'] = 'none'  # Match the real two-link candidate, no worker DP.
    return Candidate(
        mode=mode, mechanisms={'upd': mechanism}, link_mechanisms=links,
        time=1., accuracy=.5, risk=0., epsilon_used=0., communication_volume=1.,
        feasible_resource=True, feasible_privacy=True, feasible_risk=True,
        feasible_time=True, update_noise_multiplier=3.1,
    )


def test_cloud_packet_vs_aggregate_have_distinct_release_routes_and_keys():
    legacy = candidate(mode='LIIEIIIC', link='E_C_upd')
    packet = _with_cloud_plan(legacy, 'packet')
    aggregate = _with_cloud_plan(legacy, 'aggregate')
    assert len(_candidate_key(legacy)) == 2
    assert len({_candidate_key(legacy), _candidate_key(packet), _candidate_key(aggregate)}) == 3
    assert _candidate_uses_secure_aggregate_update_dp_for_selection(aggregate)
    assert not _candidate_uses_secure_aggregate_update_dp_for_selection(packet)
    assert _candidate_uses_local_packet_update_dp(packet)
    assert not _candidate_uses_secure_aggregate_update_dp(packet)
    assert not _candidate_worker_update_dp_applied(packet, 'upd_only')
    assert _candidate_uses_secure_aggregate_update_dp(aggregate)
    assert not _candidate_uses_local_packet_update_dp(aggregate)


def test_edge_to_cloud_packet_sensitivity_uses_internal_client_weights():
    config = SelectionConfig(omega_update_dimension=4., omega_update_clip_norm=.25)
    base = candidate(mode='LIIEIIIC', link='E_C_upd')
    samples = {0: 1., 1: 1., 2: 1., 3: 1.}
    edges = {0: 0, 1: 0, 2: 1, 3: 1}
    packet = {i: _with_cloud_plan(base, 'packet') for i in samples}
    aggregate = {i: _with_cloud_plan(base, 'aggregate') for i in samples}
    # Two independent Edge packets with Cloud weight 1/2 and within-Edge
    # max-client weight 1/2: 2*(.5*3.1*.5*.5)^2 per coordinate.
    assert _global_dp_perturbation_cost(config, packet, samples, edges) == pytest.approx(
        4 * 2 * (.5 * 3.1 * .5 * .5)**2
    )
    # A single Cloud aggregate release sees per-client effective weight 1/4.
    assert _global_dp_perturbation_cost(config, aggregate, samples, edges) == pytest.approx(
        4 * (3.1 * .5 * .25)**2
    )


def test_direct_cloud_packet_has_higher_final_variance_than_aggregate():
    config = SelectionConfig(omega_update_dimension=4., omega_update_clip_norm=.25)
    base = candidate(mode='LIIEIIIC', link='E_C_upd')
    samples, edges = {0: 1., 1: 1.}, {0: 0, 1: 1}
    pkt = {i: _with_cloud_plan(base, 'packet') for i in samples}
    agg = {i: _with_cloud_plan(base, 'aggregate') for i in samples}
    assert _global_dp_perturbation_cost(config, pkt, samples, edges) == pytest.approx(
        4 * 2 * (.5 * 3.1 * .5)**2
    )
    assert _global_dp_perturbation_cost(config, agg, samples, edges) == pytest.approx(
        4 * (3.1 * .5 * .5)**2
    )


def test_pareto_compares_both_complete_release_profiles():
    config = SelectionConfig(
        cloud_dp_plan='pareto', num_clients=2, num_edges=2,
        omega_update_dimension=4., omega_update_clip_norm=.25,
        pareto_max_iters=1, pareto_archive_size=8,
    )
    base = candidate(mode='LIIEIIIC', link='E_C_upd')
    selected = [(i, base, [base], 8.) for i in (0, 1)]
    audit = {}
    result, evaluation = choose_cloud_dp_pareto_profile(
        config=config, selected=selected, client_samples={0: 1., 1: 1.},
        client_edges={0: 0, 1: 1}, diagnostics=audit,
    )
    assert {item[1].dp_execution_plan for item in result} == {'cloud_aggregate'}
    assert audit['selected_cloud_dp_plan'] == 'aggregate'
    assert set(audit['cloud_dp_branch_results']) == {'packet', 'aggregate'}
    assert evaluation.system_dp > 0


def test_direct_cloud_client_is_not_silently_given_unimplemented_packet_dp():
    direct = candidate()
    assert _with_cloud_plan(direct, 'packet') == direct
    assert _with_cloud_plan(direct, 'aggregate') == direct


def test_non_dp_cloud_candidates_not_forced_to_dp():
    plain = candidate(mode='LIIE', link='L_E_upd', mechanism='none')
    assert _with_cloud_plan(plain, 'packet') == plain
    assert _with_cloud_plan(plain, 'aggregate') == plain
