from dataclasses import replace

import pytest

from dynfed.paper_config import MODES, PaperConfig
from dynfed.paper_history import History
from dynfed.paper_runner import system_traces
from experiments.run_fmnist_mode_switch_max_j import (
    CANDIDATES, candidate_manifest, candidate_profile, make_selector, privacy_preview,
)


def config(rounds=10):
    edge_map = tuple(client // 10 for client in range(100))
    return replace(PaperConfig(), clients=100, edges=10, client_edge_map=edge_map,
                   rounds=rounds, seeds=(42,), policies=("fixed_mixed",),
                   experiment_version="staged_eight_v1", shared_init_rounds=10,
                   selection_period=1, evaluation_budget=8)


def test_registered_candidates_are_complete_deterministic_and_globally_exact():
    cfg = config()
    for name, weights in CANDIDATES.items():
        first, second = candidate_profile(cfg, name), candidate_profile(cfg, name)
        assert first == second and set(first) == set(range(100))
        assert len(first) == 100
        for mode, weight in zip(MODES, weights):
            assert sum(value == mode for value in first.values()) == weight
    p0 = candidate_profile(cfg, "P0")
    for edge in range(10):
        members = [client for client in p0 if cfg.edge_for(client) == edge]
        assert [sum(p0[c] == mode for c in members) for mode in MODES] == [3, 3, 4]
    assert any(row["rounding_error_clients"] for row in candidate_manifest(cfg)
               if row["candidate"] in {"P1", "P2", "P3"})


def test_max_j_selector_and_switch_frequency(monkeypatch):
    cfg = config()
    pools = {client: MODES for client in range(100)}
    traces = system_traces(cfg, 42)[5]
    samples = {client: 600 + client for client in range(100)}
    history = History(reference=pytest.importorskip("torch").ones(4),
                      deviations={"FL_E": pytest.importorskip("torch").ones(4),
                                  "FL_EC": pytest.importorskip("torch").ones(4)*2,
                                  "FL_C": pytest.importorskip("torch").ones(4)*3})
    for strategy in ("single_switch_max_j", "per_round_max_j"):
        selector = make_selector(strategy)
        chosen, report = selector(cfg, "fixed_mixed", pools, history, traces, samples,
                                  1.0, 0.0, 5, {}, [])
        assert report["selected_rank_max_j"] == 1
        assert report["applied_round"] == 6 and report["decision_after_round"] == 5
        assert chosen.profile == next(row["profile"] for row in report["candidates"] if row["selected"])


def test_worst_path_100_round_privacy_is_within_budget():
    preview = privacy_preview(config(), 100)
    assert preview["events_per_client"] == 600
    assert preview["total_client_events"] == 60000
    assert preview["final_max_epsilon"] <= 8 + 1e-8
