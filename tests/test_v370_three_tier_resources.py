"""Opt-in, independent L/E/C resource admission regression tests."""
from dataclasses import replace

from dynfed.selection import (
    Candidate, SelectionConfig, repair_selected_tier_capacity,
    _mode_device_feasibility_metrics,
)
from dynfed.tier_resources import evaluate_tier_capacities
from dynfed.training import MODE_SPECS


def candidate(mode):
    return Candidate(
        mode=mode, mechanisms={}, time=2., accuracy=.2, risk=0.,
        epsilon_used=0., communication_volume=1., feasible_resource=True,
        feasible_privacy=True, feasible_risk=True, feasible_time=True,
    )


def test_full_local_modes_share_end_memory_not_sum_of_three_tiers():
    cfg = SelectionConfig(memory_limit=1.35)
    requirements = set()
    for mode in ("LIIE", "LIIC", "LIIEIIIC"):
        s = MODE_SPECS[mode]
        metrics = _mode_device_feasibility_metrics(
            config=cfg, local_work=s.local_work, local_memory=s.local_memory,
            samples=150, memory_capacity_factor=1.0,
        )
        requirements.add(metrics[3])
    assert len(requirements) == 1
    assert MODE_SPECS["LIIEIIIC"].local_memory == MODE_SPECS["LIIC"].local_memory
    assert MODE_SPECS["LIIEIIIC"].local_work == MODE_SPECS["LIIC"].local_work


def test_edge_and_cloud_capacities_are_independent_and_grouped():
    cfg = SelectionConfig(
        edge_cpu_limit=8., cloud_cpu_limit=8.,
        edge_memory_limit_mb=15., cloud_memory_limit_mb=50.,
        update_payload_mb=4., embedding_payload_mb=1.,
    )
    profile = {0: candidate("LIIEIIIC"), 1: candidate("LIIEIIIC"), 2: candidate("LIIC")}
    edges = {0: 0, 1: 0, 2: 1}
    samples = {0: 150, 1: 150, 2: 150}
    a = evaluate_tier_capacities(cfg, profile, edges, samples)
    # Two full-local clients on the same Edge: one shared workspace, two packets.
    assert a.edge_memory_mb == {0: 16.0}
    # Cloud sees 1 Edge aggregate plus 1 direct client packet, not 3 uploads.
    assert a.cloud_memory_mb == 16.0
    assert a.violations == ("E0:memory",)
    assert evaluate_tier_capacities(replace(cfg, edge_memory_limit_mb=16.), profile, edges, samples).feasible


def test_tier_gate_only_drops_over_capacity_without_changing_modes():
    cfg = SelectionConfig(
        tier_capacity_policy="enforce", edge_cpu_limit=2., cloud_cpu_limit=5.,
        update_payload_mb=4.,
    )
    c = candidate("LIIEIIIC")
    selected = [(i, c, [c], 8.) for i in range(3)]
    rewritten, d = repair_selected_tier_capacity(
        cfg, selected,
        client_edges={0: 0, 1: 0, 2: 0},
        client_samples={0: 150., 1: 150., 2: 150.},
    )
    assert [r[1].mode for r in rewritten] == ["LIIEIIIC", "LIIEIIIC", "SKIP"]
    assert d["dropped_client_ids"] == (2,)
    assert d["dropped_reasons"]["2"] == ["E0:cpu"]


def test_legacy_policy_is_identical_and_has_no_repair():
    c = candidate("LIIEIIIC")
    selected = [(0, c, [c], 8.)]
    got, audit = repair_selected_tier_capacity(
        SelectionConfig(), selected, client_edges={0: 0}, client_samples={0: 150.},
    )
    assert got is selected
    assert audit == {}
