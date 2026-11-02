from dataclasses import replace

import numpy as np

from dynfed.paper_config import PaperConfig
from dynfed.paper_data import partitions
from dynfed.paper_history import History
from dynfed.paper_runner import system_traces
from dynfed.paper_timeline import plan
from experiments.run_fmnist_heterogeneous_edge_e_ec import (
    SEARCH_PROTOCOL, diagnostic_select, fixed_profile, privacy_preview,
)


EDGE_MAP = (0, 0, 0, 0, 0, 0, 1, 1, 1, 1)


def diagnostic_config():
    return replace(PaperConfig(), clients=10, edges=2, client_edge_map=EDGE_MAP,
                   rounds=100, seeds=(42,), policies=("fixed_mixed",),
                   experiment_version="staged_eight_v1", shared_init_rounds=10,
                   selection_period=1, evaluation_budget=8)


def test_explicit_topology_reaches_traces_and_partitioning():
    config = diagnostic_config()
    assert [config.edge_for(client) for client in range(10)] == list(EDGE_MAP)
    assert [row["edge"] for row in system_traces(config, 42)[0]] == list(EDGE_MAP)
    split = partitions(np.tile(np.arange(10), 60), 10, 2, .5, 42, EDGE_MAP)
    assert len(split) == 10 and all(split)
    assert sum(map(len, split)) == 600


def test_client_level_profile_has_both_modes_inside_each_edge():
    config = diagnostic_config()
    pools = {client: ("FL_E", "FL_EC", "FL_C") for client in range(10)}
    profile = fixed_profile(config, pools)
    assert [profile[i] for i in range(10)] == [
        "FL_E", "FL_E", "FL_E", "FL_EC", "FL_EC", "FL_EC",
        "FL_E", "FL_E", "FL_EC", "FL_EC",
    ]
    for edge in range(2):
        assert {profile[c] for c in range(10) if config.edge_for(c) == edge} == {"FL_E", "FL_EC"}


def test_selector_builds_four_independent_edge_mode_units():
    config = diagnostic_config()
    pools = {client: ("FL_E", "FL_EC", "FL_C") for client in range(10)}
    traces = system_traces(config, 42)[0]
    chosen, report = diagnostic_select(config, "fixed_mixed", pools, History(), traces,
                                       {client: 6000 for client in range(10)}, 1., 0., 0, {}, [])
    assert report["search_protocol"] == SEARCH_PROTOCOL
    assert {(unit.edge, unit.mode) for unit in chosen.schedule.units} == {
        (0, "FL_E"), (0, "FL_EC"), (1, "FL_E"), (1, "FL_EC")}
    assert not chosen.schedule.pending
    stages = [event for event in chosen.schedule.events if event["event"] == "edge_stage"]
    assert all(event["start"] == event["ready"] for event in stages)
    for upload in (event for event in chosen.schedule.events
                   if event["event"] == "edge_cloud_upload"):
        final_stage = max((event for event in stages
                           if event["edge"] == upload["edge"] and event["mode"] == upload["mode"]),
                          key=lambda event: event["stage"])
        assert upload["start"] == final_stage["finish"]


def test_fl_ec_keeps_three_stages_and_privacy_counts_every_step():
    config = diagnostic_config()
    profile = fixed_profile(config, {client: ("FL_E", "FL_EC", "FL_C") for client in range(10)})
    schedule = plan(config, profile, system_traces(config, 42)[0],
                    {client: 6000 for client in range(10)}, 1., 0., 0)
    ec_events = [event for event in schedule.events if event["event"] == "edge_stage"
                 and event["mode"] == "FL_EC"]
    assert len(ec_events) == 2 * config.edge_stages
    preview = privacy_preview(config)
    assert preview["events_per_client_by_mode"] == {"FL_E": 200, "FL_EC": 600}
    assert preview["total_client_events"] == 4000
    assert preview["final_max_epsilon"] <= 8 + 1e-8
