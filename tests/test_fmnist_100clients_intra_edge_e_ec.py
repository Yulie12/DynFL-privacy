from dataclasses import replace

from dynfed.paper_config import PaperConfig
from dynfed.paper_history import History
from dynfed.paper_runner import system_traces
from experiments.run_fmnist_100clients_intra_edge_e_ec import (
    SEARCH_PROTOCOL, diagnostic_select, fixed_profile, privacy_preview,
)


EDGE_MAP = tuple(client // 10 for client in range(100))


def diagnostic_config():
    return replace(PaperConfig(), clients=100, edges=10, client_edge_map=EDGE_MAP,
                   rounds=100, seeds=(42,), policies=("fixed_mixed",),
                   experiment_version="staged_eight_v1", shared_init_rounds=10,
                   selection_period=1, evaluation_budget=8)


def test_every_edge_has_five_clients_per_mode():
    config = diagnostic_config()
    pools = {client: ("FL_E", "FL_EC", "FL_C") for client in range(100)}
    profile = fixed_profile(config, pools)
    for edge in range(10):
        members = [client for client in range(100) if config.edge_for(client) == edge]
        assert len(members) == 10
        assert sum(profile[client] == "FL_E" for client in members) == 5
        assert sum(profile[client] == "FL_EC" for client in members) == 5


def test_selector_builds_twenty_distinct_group_units():
    config = diagnostic_config()
    pools = {client: ("FL_E", "FL_EC", "FL_C") for client in range(100)}
    chosen, report = diagnostic_select(config, "fixed_mixed", pools, History(),
                                       system_traces(config, 42)[0],
                                       {client: 600 for client in range(100)},
                                       1., 0., 0, {}, [])
    assert report["search_protocol"] == SEARCH_PROTOCOL
    assert len(chosen.schedule.units) == 20
    assert {(unit.edge, unit.mode) for unit in chosen.schedule.units} == {
        (edge, mode) for edge in range(10) for mode in ("FL_E", "FL_EC")}
    assert all(len(unit.clients) == 5 for unit in chosen.schedule.units)
    assert not chosen.schedule.pending


def test_edge_mode_groups_start_without_cross_group_serialization():
    config = diagnostic_config()
    pools = {client: ("FL_E", "FL_EC", "FL_C") for client in range(100)}
    chosen, _ = diagnostic_select(config, "fixed_mixed", pools, History(),
                                  system_traces(config, 42)[0],
                                  {client: 600 for client in range(100)},
                                  1., 0., 0, {}, [])
    stages = [event for event in chosen.schedule.events if event["event"] == "edge_stage"]
    assert all(event["start"] == event["ready"] for event in stages)
    for upload in (event for event in chosen.schedule.events
                   if event["event"] == "edge_cloud_upload"):
        final_stage = max((event for event in stages
                           if event["edge"] == upload["edge"] and event["mode"] == upload["mode"]),
                          key=lambda event: event["stage"])
        assert upload["start"] == final_stage["finish"]


def test_privacy_preview_counts_all_100_clients_and_ec_stages():
    preview = privacy_preview(diagnostic_config())
    assert preview["events_per_client_by_mode"] == {"FL_E": 200, "FL_EC": 600}
    assert preview["total_client_events"] == 40000
    assert preview["final_max_epsilon"] <= 8 + 1e-8
