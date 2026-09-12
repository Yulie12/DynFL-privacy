import math

import pytest
import torch

from experiments.distributed_edge_dp_common import (
    distributed_noise_plan, seal_edge_sum, unknown_noise_sufficient,
)
from dynfed.privacy import PrivacyAccountant, calibrate_gaussian_noise


@pytest.mark.parametrize("counts,edges", [([60] * 100, 10), ([1, 2, 3, 4, 5], 2), ([2, 3], 1)])
def test_weighted_variance_and_sensitivity(counts, edges):
    plan = distributed_noise_plan(counts, edges, 0.1, 6.9, edges)
    assert plan["global_sensitivity"] == pytest.approx(0.2 * max(counts) / sum(counts))
    variance = sum((g["cloud_weight"] * g["noise_std_before_cloud_weight"]) ** 2
                   for g in plan["groups"])
    assert math.sqrt(variance) == pytest.approx(plan["target_noise_std"])
    assert unknown_noise_sufficient(plan, list(range(edges)))
    assert not unknown_noise_sufficient(plan, list(range(edges - 1)))


def test_robust_noise_calibration_is_conditional_and_costs_more_noise():
    plan = distributed_noise_plan([1] * 100, 10, 0.1, 6.9, 7)
    assert unknown_noise_sufficient(plan, list(range(7)))
    assert not unknown_noise_sufficient(plan, list(range(6)))
    assert plan["all_edges_noise_std"] / plan["target_noise_std"] == pytest.approx(math.sqrt(10 / 7))


@pytest.mark.parametrize("minimum,sigma", [(0, 1), (11, 1), (1.5, 1), (10, 0), (10, float('nan'))])
def test_invalid_assumptions_rejected(minimum, sigma):
    with pytest.raises(ValueError):
        distributed_noise_plan([1] * 100, 10, 0.1, sigma, minimum)


@pytest.mark.parametrize("unknown", [[0, 0], [-1], [10]])
def test_invalid_noise_observer_sets_rejected(unknown):
    plan = distributed_noise_plan([1] * 100, 10, 0.1, 1, 10)
    with pytest.raises(ValueError):
        unknown_noise_sufficient(plan, unknown)


def test_missing_edge_is_rejected_instead_of_silently_renormalizing():
    plan = distributed_noise_plan([1] * 10, 2, 0.1, 1, 2)
    with pytest.raises(ValueError, match="exactly all"):
        seal_edge_sum({0: torch.zeros(4)}, plan["groups"])


def test_unbalanced_weights_do_not_allocate_equal_preweight_noise():
    plan = distributed_noise_plan([1, 9], 2, 0.1, 1, 2)
    a, b = plan["groups"]
    assert a["noise_std_before_cloud_weight"] / b["noise_std_before_cloud_weight"] == pytest.approx(9)
    assert a["weighted_noise_std"] == b["weighted_noise_std"]


def test_global_accounting_uses_global_noise_not_individual_share_multiplier():
    sigma = calibrate_gaussian_noise(8, 1e-5, 100)
    plan = distributed_noise_plan([60] * 100, 10, 0.1, sigma, 10)
    effective_sigma = plan["all_edges_noise_std"] / plan["global_sensitivity"]
    assert effective_sigma == pytest.approx(sigma)
    ledger = PrivacyAccountant(8, 1e-5)
    ledger.add_events(effective_sigma, 100)
    assert ledger.current_epsilon() == pytest.approx(8)
    assert not ledger.can_add_event(effective_sigma)
    # Less noise per hidden edge packet is not an independent edge DP guarantee.
    group = plan["groups"][0]
    edge_sigma = group["noise_std_before_cloud_weight"] / group["sensitivity"]
    assert edge_sigma == pytest.approx(sigma / math.sqrt(10))


@pytest.mark.parametrize("weights", [[0.2, 0.2], [float('nan'), 0.5], [-0.1, 1.1], [0.0, 1.0]])
def test_invalid_cloud_weights_fail_before_he(weights):
    groups = [{"edge": i, "cloud_weight": w} for i, w in enumerate(weights)]
    with pytest.raises(ValueError, match="weights"):
        seal_edge_sum({0: torch.zeros(4), 1: torch.zeros(4)}, groups)


@pytest.mark.parametrize("second", [torch.zeros(3), torch.zeros(2, 2), torch.zeros(0),
                                     torch.tensor([float('nan'), 0, 0, 0]),
                                     torch.tensor([float('inf'), 0, 0, 0])])
def test_malformed_edge_packet_fails_before_he(second):
    plan = distributed_noise_plan([1, 1], 2, 0.1, 1, 2)
    with pytest.raises(ValueError, match="Finite matching"):
        seal_edge_sum({0: torch.zeros(4), 1: second}, plan["groups"])


def test_duplicate_planned_edge_and_extra_packet_rejected():
    plan = distributed_noise_plan([1, 1], 2, 0.1, 1, 2)
    with pytest.raises(ValueError, match="exactly all"):
        seal_edge_sum({0: torch.zeros(4)}, [plan["groups"][0]] * 2)
    with pytest.raises(ValueError, match="exactly all"):
        seal_edge_sum({i: torch.zeros(4) for i in range(3)}, plan["groups"])
