from dataclasses import replace

import pytest
import torch

from dynfed.paper_config import MODES, PaperConfig
from dynfed.paper_history import History
from dynfed.paper_runner import system_traces
from experiments.run_fmnist_joint_jt_100r import (
    CANDIDATES, SPECS, _normalize, candidate_profile, make_selector,
)
from experiments.run_fmnist_mode_switch_max_j import privacy_preview


def config():
    return replace(PaperConfig(), clients=100, edges=10,
                   client_edge_map=tuple(client//10 for client in range(100)),
                   rounds=100, seeds=(42,43,44), policies=("fixed_mixed",),
                   experiment_version="staged_eight_v1", shared_init_rounds=10,
                   selection_period=1, evaluation_budget=8, snapshot_interval=10)


def initialized_history():
    return History(reference=torch.ones(4), deviations={
        "FL_E": torch.ones(4), "FL_EC": torch.ones(4)*2, "FL_C": torch.ones(4)*3})


def selector_call(strategy, round_index=5, previous=None):
    cfg = config()
    return make_selector(strategy)(
        cfg, "fixed_mixed", {client: MODES for client in range(100)}, initialized_history(),
        system_traces(cfg, 42)[round_index], {client: 600 for client in range(100)},
        .246824, 0., round_index, previous or {}, [])


def test_equal_minmax_values_map_to_half():
    assert _normalize({"P0": 2., "P1": 2.}) == {"P0": .5, "P1": .5}
    assert _normalize({"P0": 2., "P1": 2.}, False) == {"P0": .5, "P1": .5}


def test_all_registered_candidates_have_exact_global_quotas():
    cfg = config()
    for name, quota in CANDIDATES.items():
        profile = candidate_profile(cfg, name)
        assert {mode: sum(value == mode for value in profile.values()) for mode in MODES} == dict(zip(MODES, quota))
        assert all({profile[client] for client in range(100) if cfg.edge_for(client) == edge}
                   for edge in range(10))


@pytest.mark.parametrize("strategy", ["greedy_lambda025", "greedy_lambda050", "greedy_lambda075"])
def test_greedy_selects_maximum_stored_joint_utility(strategy):
    _, report = selector_call(strategy)
    scored = [item for item in report["candidates"] if item["utility"] is not None]
    expected = min(scored, key=lambda item: (-item["utility"], item["tie_order"]))
    assert report["selected_candidate"] == expected["name"]
    assert report["selection_reason"] == "maximum_joint_utility"


def test_dynamic_strategies_share_p0_prefix():
    for strategy, spec in SPECS.items():
        if spec["kind"] in {"greedy", "hysteresis", "periodic", "max_j", "min_t"}:
            _, report = selector_call(strategy, round_index=4)
            assert report["selected_candidate"] == "P0"
            assert report["selection_reason"] == "shared_P0_prefix"


def test_periodic_holds_between_predeclared_decision_rounds():
    cfg = config(); previous = candidate_profile(cfg, "P3")
    _, report = selector_call("periodic_lambda050", round_index=6, previous=previous)
    assert report["selected_candidate"] == "P3"
    assert report["selection_reason"] == "periodic_hold"
    assert not report["periodic_decision_allowed"]


def test_hysteresis_switch_obeys_strict_threshold():
    cfg = config(); previous = candidate_profile(cfg, "P3")
    _, report = selector_call("hysteresis_lambda050", previous=previous)
    items = {item["name"]: item for item in report["candidates"]}
    selected = report["selected_candidate"]
    if selected != "P3":
        assert items[selected]["utility"] - items["P3"]["utility"] > .05
    else:
        assert report["selection_reason"] == "hysteresis_hold"


def test_fixed_and_pure_baselines_start_from_round_one_without_prefix():
    _, fixed = selector_call("fixed_P4", round_index=0)
    _, pure = selector_call("pure_FL_EC", round_index=0)
    assert fixed["selected_candidate"] == "P4" and fixed["selection_reason"] == "fixed_candidate"
    assert pure["selected_candidate"] == "pure_FL_EC" and pure["selection_reason"] == "fixed_pure_mode"
    assert set(pure["selected_profile"].values()) == {"FL_EC"}


def test_worst_case_privacy_preview_passes_100_round_budget():
    preview = privacy_preview(config(), 100)
    assert preview["events_per_client"] == 600
    assert preview["final_max_epsilon"] <= 8 + 1e-8
