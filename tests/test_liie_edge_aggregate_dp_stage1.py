"""Opt-in LIIE fixed-cohort aggregate-DP ablation (not the final Pareto plan)."""
from dataclasses import replace
import math

import pytest
import torch

from dynfed.fmnist_lenet5_dynamic import (
    _candidate_training_mechanisms,
    _candidate_worker_update_dp_applied,
    _liie_planned_edge_aggregate_groups,
    _liie_streaming_secagg_eligible,
)
from dynfed.selection import (
    Candidate,
    SelectionConfig,
    _global_dp_perturbation_cost,
)
from dynfed.streaming_secagg import streaming_secure_aggregate_exact_target


def c(mode='LIIE', mechanism='dp', z=3.1):
    return Candidate(
        mode=mode,
        mechanisms={'upd': mechanism},
        link_mechanisms={'L_E_upd': mechanism},
        time=1.0,
        accuracy=0.5,
        risk=0.0,
        epsilon_used=0.0,
        communication_volume=1.0,
        feasible_resource=True,
        feasible_privacy=True,
        feasible_risk=True,
        feasible_time=True,
        update_noise_multiplier=z,
    )


def selected(*items):
    return [(i, item, [], 1.0) for i, item in enumerate(items)]


def test_protocol_opt_in_homogeneous_group_only():
    pair = selected(c(), c())
    assert _liie_planned_edge_aggregate_groups(pair, {0: 2, 1: 2}, plan='independent', execute_real_he=False) == {}
    assert _liie_planned_edge_aggregate_groups(pair, {0: 2, 1: 2}, plan='aggregate', execute_real_he=True) == {}
    assert _liie_planned_edge_aggregate_groups(pair, {0: 2, 1: 2}, plan='aggregate', execute_real_he=False) == {2: frozenset((0, 1))}
    assert _liie_planned_edge_aggregate_groups(selected(c(), c(mechanism='he3')), {0: 0, 1: 0}, plan='aggregate', execute_real_he=False) == {}
    assert _liie_planned_edge_aggregate_groups(selected(c(), c(mechanism='dp_he3')), {0: 0, 1: 0}, plan='aggregate', execute_real_he=False) == {}
    assert _liie_planned_edge_aggregate_groups(selected(c()), {0: 0}, plan='aggregate', execute_real_he=False) == {}
    with pytest.raises(ValueError, match='liie_edge_dp_plan'):
        _liie_planned_edge_aggregate_groups(pair, {0: 2, 1: 2}, plan='mystery', execute_real_he=False)


def test_worker_never_noises_preplanned_aggregate_packets():
    individual = c()
    aggregate = replace(individual, dp_execution_plan='aggregate')
    assert _candidate_worker_update_dp_applied(individual, 'upd_only')
    assert _candidate_training_mechanisms(individual, aggregate_cloud_update_dp=True)['upd'] == 'dp'
    assert not _candidate_worker_update_dp_applied(aggregate, 'upd_only')
    assert _candidate_training_mechanisms(aggregate, aggregate_cloud_update_dp=True)['upd'] == 'none'
    assert _liie_streaming_secagg_eligible([(0, None, 1, aggregate), (1, None, 1, aggregate)], execute_real_he=False)


def test_exact_target_aggregate_dp_has_lower_perturbation_than_independent():
    conf = SelectionConfig(omega_update_dimension=4.0, omega_update_clip_norm=.25)
    independent = {0: c(), 1: c()}
    aggregate = {i: replace(c(), dp_execution_plan='aggregate') for i in (0, 1)}
    samples, edges = {0: 1., 1: 1.}, {0: 0, 1: 0}
    ind = _global_dp_perturbation_cost(conf, independent, samples, edges)
    agg = _global_dp_perturbation_cost(conf, aggregate, samples, edges)
    assert ind == pytest.approx(4 * (3.1 * .5)**2 / 2)
    assert agg == pytest.approx(4 * (3.1 * .25)**2)
    assert agg < ind
    # End-to-end primitive is an exact-target *single aggregate release*.
    out, audit = streaming_secure_aggregate_exact_target(
        [{'end': {'w': torch.tensor([.3, .4])}, 'edge': {}},
         {'end': {'w': torch.tensor([.4, .3])}, 'edge': {}}],
        [1, 1], clip_norm=.25, noise_multiplier=3.1, round_seed=42,
    )
    assert audit.cohort_size == 2
    assert audit.aggregate_sensitivity == pytest.approx(.25)
    assert audit.target_noise_std == pytest.approx(.775)
    assert torch.isfinite(out['end']['w']).all()


def test_mixed_plan_cannot_take_aggregate_credit():
    conf = SelectionConfig(omega_update_dimension=4.0, omega_update_clip_norm=.25)
    profile = {0: replace(c(), dp_execution_plan='aggregate'), 1: c()}
    with pytest.raises(ValueError, match='Inconsistent LIIE aggregate'):
        _global_dp_perturbation_cost(conf, profile, {0: 1, 1: 1}, {0: 0, 1: 0})
