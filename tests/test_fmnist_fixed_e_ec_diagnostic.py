from dataclasses import replace

from dynfed.paper_config import PaperConfig
from dynfed.paper_history import History
from experiments.run_fmnist_fixed_e_ec_50_50 import (
    SEARCH_PROTOCOL, diagnostic_select, fixed_profile, privacy_preview,
)


def diagnostic_config():
    return replace(PaperConfig(), clients=20, edges=10, rounds=100, seeds=(42,),
                   policies=("fixed_mixed",), experiment_version="staged_eight_v1",
                   shared_init_rounds=10, selection_period=1, evaluation_budget=8)


def test_profile_is_exact_and_preserves_real_edge_groups():
    config = diagnostic_config()
    profile = fixed_profile(config, {client: ("FL_E", "FL_EC", "FL_C") for client in range(20)})
    assert sum(mode == "FL_E" for mode in profile.values()) == 10
    assert sum(mode == "FL_EC" for mode in profile.values()) == 10
    for edge in range(5):
        assert [profile[edge], profile[edge + 10]] == ["FL_E", "FL_E"]
    for edge in range(5, 10):
        assert [profile[edge], profile[edge + 10]] == ["FL_EC", "FL_EC"]


def test_selector_builds_five_edge_and_five_multistage_edge_units():
    config = diagnostic_config()
    pools = {client: ("FL_E", "FL_EC", "FL_C") for client in range(20)}
    traces = [{"client": client, "edge": client % 10, "memory_mb": 32.,
               "compute_capacity": 2., "step_sec": .1, "le_rate": 5.,
               "lc_rate": 2.2, "ec_rate": 8.} for client in range(20)]
    chosen, report = diagnostic_select(config, "fixed_mixed", pools, History(), traces,
                                       {client: 3000 for client in range(20)}, 1., 0., 0, {}, [])
    assert report["search_protocol"] == SEARCH_PROTOCOL
    assert len([unit for unit in chosen.schedule.units if unit.mode == "FL_E"]) == 5
    assert len([unit for unit in chosen.schedule.units if unit.mode == "FL_EC"]) == 5
    assert not chosen.schedule.pending


def test_100_round_privacy_preview_stays_within_budget():
    preview = privacy_preview(diagnostic_config())
    assert preview["events_per_client_by_mode"] == {"FL_E": 200, "FL_EC": 600}
    assert preview["total_client_events"] == 8000
    assert preview["final_max_epsilon"] <= 8 + 1e-8
