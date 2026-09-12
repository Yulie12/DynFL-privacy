import math

import pytest
import torch

from experiments.edge_dp_common import edge_noise_seed, edge_release_plan, release_edge
from dynfed.privacy import PrivacyAccountant, calibrate_gaussian_noise


def test_plan_has_disjoint_complete_cohorts_and_fixed_sample_weights():
    counts = [1, 2, 3, 4, 5]
    plan = edge_release_plan(counts, 2, 0.1)
    assert sorted(i for g in plan for i in g["clients"]) == list(range(5))
    assert sum(g["cloud_weight"] for g in plan) == pytest.approx(1)
    for g in plan:
        assert sum(g["weights"]) == pytest.approx(1)
        assert g["sensitivity"] == pytest.approx(0.2 * max(g["weights"]))
        for i, w in zip(g["clients"], g["weights"]):
            assert g["cloud_weight"] * w == pytest.approx(counts[i] / sum(counts))


@pytest.mark.parametrize("counts,edges,clip", [([], 1, 1), ([0, 1], 1, 1),
    ([1.5], 1, 1), ([1], 0, 1), ([1], 2, 1), ([1], 1, 0), ([1], 1, float("nan"))])
def test_invalid_plan_fails(counts, edges, clip):
    with pytest.raises(ValueError):
        edge_release_plan(counts, edges, clip)


def test_individual_clipping_precedes_aggregation_and_does_not_mutate_input():
    updates = torch.tensor([[3., 4.], [-6., 8.]])
    saved = updates.clone()
    packet, info = release_edge(updates, [0.25, 0.75], clip_norm=1,
                               noise_multiplier=0, seed=42)
    assert torch.allclose(packet, torch.tensor([-0.3, 0.8]))
    assert info["max_postclip_norm"] == pytest.approx(1)
    assert info["sensitivity"] == pytest.approx(1.5)
    assert torch.equal(updates, saved)


def test_replacement_bound_is_attainable_for_heaviest_client():
    first = torch.tensor([[0., 0.], [1., 0.]])
    second = torch.tensor([[0., 0.], [-1., 0.]])
    a, info = release_edge(first, [0.2, 0.8], clip_norm=1, noise_multiplier=0, seed=1)
    b, _ = release_edge(second, [0.2, 0.8], clip_norm=1, noise_multiplier=0, seed=1)
    assert float((a - b).norm()) == pytest.approx(info["sensitivity"])


def test_each_edge_noises_before_cloud_and_cloud_variance_matches_weights():
    plan = edge_release_plan([60] * 100, 10, 0.1)
    sigma = 2.0
    packets = []
    manual = []
    for g in plan:
        seed = edge_noise_seed(42, 0, g["edge"])
        packet, info = release_edge(torch.zeros(10, 4), g["weights"], clip_norm=0.1,
                                   noise_multiplier=sigma, seed=seed)
        expected = torch.randn(4, generator=torch.Generator().manual_seed(seed)) * info["noise_std"]
        assert torch.equal(packet, expected)
        packets.append(packet * g["cloud_weight"])
        manual.append(expected * g["cloud_weight"])
    assert torch.equal(torch.stack(packets).sum(0), torch.stack(manual).sum(0))
    cloud_std = math.sqrt(sum((g["cloud_weight"] * g["sensitivity"] * sigma) ** 2 for g in plan))
    assert cloud_std == pytest.approx(math.sqrt(10) * 2 * 0.1 / 100 * sigma)


def test_per_client_budget_counts_own_edge_once_not_all_edges():
    plan = edge_release_plan([1] * 100, 10, 0.1)
    sigma = calibrate_gaussian_noise(8, 1e-5, 100)
    ledgers = [PrivacyAccountant(8, 1e-5) for _ in range(100)]
    for _ in range(100):
        assert all(l.can_add_event(sigma) for l in ledgers)
        for group in plan:
            for i in group["clients"]:
                ledgers[i].add_event(sigma)
    assert all(l.current_epsilon() == pytest.approx(8) for l in ledgers)
    assert not any(l.can_add_event(sigma) for l in ledgers)


def test_research_streams_are_separate_across_edges_rounds_seeds():
    seeds = [edge_noise_seed(s, r, e) for s in [40, 42, 44] for r in range(100) for e in range(10)]
    assert len(set(seeds)) == len(seeds)


@pytest.mark.parametrize("updates,weights,multiplier,clip", [
    (torch.tensor([[float('nan')]]), [1.], 1., True),
    (torch.ones(2, 1), [1.], 1., True),
    (torch.ones(1, 1), [1.], 1., False),
    (torch.ones(1, 1), [1.], -1., True),
    (torch.ones(1, 1), [1.], float('nan'), True)])
def test_malformed_release_rejected(updates, weights, multiplier, clip):
    with pytest.raises(ValueError):
        release_edge(updates, weights, clip_norm=1, noise_multiplier=multiplier, seed=1, clip=clip)
