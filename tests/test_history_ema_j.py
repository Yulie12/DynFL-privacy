from __future__ import annotations

import json

import pytest
import torch

from dynfed.history_ema_j import HistoryEMAJ, normalized_squared_deviation, append_jsonl
from experiments.backtest_history_ema_j import backtest


def test_normalized_squared_deviation_matches_definition():
    candidate = torch.tensor([2.0, 1.0])
    reference = torch.tensor([1.0, 1.0])
    expected = 1.0 / (2.0 + 1e-12)
    assert normalized_squared_deviation(candidate, reference) == pytest.approx(expected)


def test_history_ema_j_prediction_and_observation_update():
    history = HistoryEMAJ(decay=0.5)
    assert history.predict("LIIC")["status"] == "no_global_history"
    first = history.observe(
        round_index=0, global_update=[1.0, 1.0],
        mode_updates={"LIIC": [1.0, 0.0], "LIIEIIIC": [0.0, 1.0]},
        global_history_eligible=True,
    )
    assert any(r["status"] == "observed_eligible_global_update" for r in first)
    assert history.predict("LIIC")["status"] == "ok"
    assert history.predict("LIIEIIIC")["status"] == "ok"
    liie = history.predict("LIIE")
    assert liie["status"] == "no_mode_history"
    history.observe(
        round_index=1, global_update=[1.0, 1.0],
        mode_updates={"LIIE": [1.0, 0.0]}, global_history_eligible=True,
    )
    assert history.predict("LIIE")["status"] == "ok"


def test_profile_j_uses_cloud_sample_mass_including_liie_edge_cloud_update():
    history = HistoryEMAJ(decay=0.5)
    history.observe(
        round_index=0, global_update=[1.0, 1.0],
        mode_updates={"LIIE": [0.0, 1.0], "LIIC": [1.0, 0.0], "LIIEIIIC": [0.0, 1.0]},
        global_history_eligible=True,
    )
    # LIIE now reaches Cloud, so its sample mass participates in the profile.
    score, source = history.predict_profile_j(
        {0: "LIIC", 1: "LIIC", 2: "LIIC", 3: "LIIEIIIC", 4: "LIIE"},
        {0: 100, 1: 100, 2: 100, 3: 100, 4: 100},
    )
    expected = normalized_squared_deviation([0.6, 0.4], history.global_ema)
    assert source == "history_ema_j"
    assert score == pytest.approx(expected)


def test_history_ema_j_rejects_dimension_changes():
    history = HistoryEMAJ(decay=0.9)
    history.observe(round_index=0, global_update=[1.0, 2.0], mode_updates={"LIIC": [1.0, 2.0]},
                    global_history_eligible=True)
    with pytest.raises(ValueError, match="dimension changed"):
        history.observe(round_index=1, global_update=[1.0], mode_updates={"LIIC": [1.0]},
                        global_history_eligible=True)


def test_backtest_pairs_prediction_with_realized_j(tmp_path):
    path = tmp_path / "history_ema_j.jsonl"
    rows = [
        {"event": "prediction", "round_index": 1, "mode": "LIIC", "predicted_j": 0.5, "status": "ok"},
        {"event": "realized_j", "round_index": 1, "mode": "LIIC", "observed_j": 0.25, "status": "observed"},
        {"event": "realized_j", "round_index": 2, "mode": "LIIC", "observed_j": None, "status": "excluded"},
    ]
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    report = backtest(path)
    assert report["paired_observations"] == 1
    assert report["overall"]["mae"] == pytest.approx(0.25)
    assert report["excluded_realized_rows"] == 1


def test_history_state_round_trip(tmp_path):
    history = HistoryEMAJ(decay=0.8)
    history.observe(round_index=0, global_update=[1.0, 0.0], mode_updates={"LIIC": [0.0, 1.0]},
                    global_history_eligible=True)
    path = tmp_path / "state.pt"
    history.save(path)
    loaded = HistoryEMAJ(decay=0.8)
    loaded.load(path)
    assert torch.equal(history.global_ema, loaded.global_ema)
    assert torch.equal(history.mode_ema["LIIC"], loaded.mode_ema["LIIC"])


def test_history_ema_j_protocol_disables_ec_protection_and_aggregate_dp(monkeypatch):
    import random

    from dynfed.selection import Candidate, SelectionConfig, _mode_link_transmissions, enumerate_candidates
    from dynfed.fmnist_lenet5_dynamic import (
        _candidate_cloud_update_mechanism,
        _candidate_uses_local_packet_update_dp,
        _candidate_uses_secure_aggregate_update_dp,
    )

    monkeypatch.setenv("DYNFL_HISTORY_EMA_J_ENABLED", "1")
    transmissions = _mode_link_transmissions("LIIEIIIC", local_block_cycles=1, edge_loops=3)
    ec = next(row for row in transmissions if row[0] == "E_C_upd")
    assert ec[3] is False

    candidate = Candidate(
        mode="LIIEIIIC", mechanisms={"upd": "dp"},
        time=1.0, accuracy=0.5, risk=0.1, epsilon_used=1.0,
        communication_volume=1.0, feasible_resource=True, feasible_privacy=True,
        feasible_risk=True, feasible_time=True,
        link_mechanisms={"L_E_upd": "dp", "E_C_upd": "dp"},
    )
    assert _candidate_cloud_update_mechanism(candidate) == "none"
    assert _candidate_uses_secure_aggregate_update_dp(candidate) is False

    liic = Candidate(
        mode="LIIC", mechanisms={"upd": "dp"},
        time=1.0, accuracy=0.5, risk=0.1, epsilon_used=1.0,
        communication_volume=1.0, feasible_resource=True, feasible_privacy=True,
        feasible_risk=True, feasible_time=True,
        link_mechanisms={"L_C_upd": "dp"},
    )
    assert _candidate_uses_local_packet_update_dp(liic) is True
    assert _candidate_uses_secure_aggregate_update_dp(liic) is False

    cfg = SelectionConfig(
        num_clients=10, num_edges=2, privacy_unit="sample", allow_he=False,
        update_mechanism_options=("dp",), update_protection_goal="packet_protection",
        require_cloud_participation=False, require_edge_cloud_coverage=False,
    )
    candidates = enumerate_candidates(
        config=cfg, client_id=0, edge_factor=1.0, compute_factor=1.0, samples=500,
        remaining_epsilon=8.0, round_idx=0, rng=random.Random(1),
        policy="full_dynfl", connected_edge_id=0,
    )
    assert {c.mode for c in candidates} <= {"LIIE", "LIIEIIIC", "LIIC"}



def test_history_ema_j_liie_is_cloud_reaching_not_edge_only(monkeypatch):
    from dynfed.selection import (
        Candidate, SelectionConfig, OBJECT_SIZES, _admitted_clients_for_profile,
        _final_return_volume, _mode_link_transmissions, candidate_link_mechanism,
    )

    monkeypatch.setenv("DYNFL_HISTORY_EMA_J_ENABLED", "1")
    transmissions = _mode_link_transmissions("LIIE", local_block_cycles=1, edge_loops=1)
    by_link = {row[0]: row for row in transmissions}
    assert by_link["L_E_upd"][3] is True
    assert by_link["E_C_upd"][3] is False
    assert "C_E_upd_final_return" in by_link
    assert "E_L_upd_final_return" in by_link

    candidate = Candidate(
        mode="LIIE", mechanisms={"upd": "dp"}, time=1.0, accuracy=0.5, risk=0.1,
        epsilon_used=1.0, communication_volume=1.0, feasible_resource=True,
        feasible_privacy=True, feasible_risk=True, feasible_time=True,
        link_mechanisms={"L_E_upd": "dp", "E_C_upd": "dp"},
    )
    # Even if a stale candidate object contains a Cloud DP label, E->C is
    # explicitly unprotected in the History-EMA J protocol.
    assert candidate_link_mechanism(candidate, "E_C_upd") == "none"
    assert _final_return_volume("LIIE") == pytest.approx(2.0 * OBJECT_SIZES["upd"])

    # LIE remains Edge-only, while LIIE has its own Edge->Cloud aggregation
    # group. With a 50% buffer fraction, neither mode should suppress the other.
    cfg = SelectionConfig(num_clients=2, num_edges=1, aggregation_fraction=0.5)
    lie = Candidate(
        mode="LIE", mechanisms={}, time=0.1, accuracy=0.5, risk=0.1, epsilon_used=0.0,
        communication_volume=1.0, feasible_resource=True, feasible_privacy=True,
        feasible_risk=True, feasible_time=True, pre_aggregation_time=0.1,
    )
    profile = {
        0: lie,
        1: Candidate(
            mode="LIIE", mechanisms={"upd": "dp"}, time=0.2, accuracy=0.5, risk=0.1,
            epsilon_used=1.0, communication_volume=1.0, feasible_resource=True,
            feasible_privacy=True, feasible_risk=True, feasible_time=True,
            pre_aggregation_time=0.2,
        ),
    }
    admitted = _admitted_clients_for_profile(cfg, profile, {0: 0, 1: 0}, {})
    assert admitted == [0, 1]

def test_history_ema_j_is_active_selection_objective_without_certificate():
    from dynfed.selection import (
        Candidate, SelectionConfig, choose_global_pareto_profile, _selection_learning_cost
    )

    history = HistoryEMAJ(decay=0.5)
    history.observe(
        round_index=0, global_update=[1.0, 1.0],
        mode_updates={"LIIE": [0.0, 1.0], "LIIC": [1.0, 0.0], "LIIEIIIC": [0.0, 1.0]},
        global_history_eligible=True,
    )
    cfg = SelectionConfig(
        num_clients=2, num_edges=1, privacy_unit="sample",
        learning_objective="history_ema_j", history_ema_j_history=history,
    )
    liic = Candidate(mode="LIIC", mechanisms={"upd": "dp"}, time=1.0, accuracy=0.5, risk=0.1,
                     epsilon_used=1.0, communication_volume=1.0, feasible_resource=True,
                     feasible_privacy=True, feasible_risk=True, feasible_time=True)
    lii = Candidate(mode="LIIE", mechanisms={"upd": "dp"}, time=1.0, accuracy=0.5, risk=0.1,
                    epsilon_used=1.0, communication_volume=1.0, feasible_resource=True,
                    feasible_privacy=True, feasible_risk=True, feasible_time=True)
    score = _selection_learning_cost(
        cfg, {0: liic, 1: lii}, {0: 100.0, 1: 100.0}, {0: 0, 1: 0}, (0, 1)
    )
    assert score >= 0.0
    assert cfg.online_bound_manifest_path is None
    # Exercise the actual global profile search: it must score candidates via J
    # and must not attempt to read an online-bound certificate.
    profile, evaluation = choose_global_pareto_profile(
        config=cfg,
        selected=[(0, liic, [liic, lii], 8.0), (1, liic, [liic, lii], 8.0)],
        client_samples={0: 100.0, 1: 300.0},
        client_edges={0: 0, 1: 0},
        objective="pareto",
    )
    assert len(profile) == 2
    assert evaluation.learning_proxy_source == "history_ema_j"
    assert evaluation.system_learning_error >= 0.0


def test_history_ema_j_config_has_no_certificate_dependency():
    import json
    from pathlib import Path
    from experiments.run_paper_config import validate_config

    path = Path(__file__).resolve().parents[1] / "configs" / "history_ema_j_fmnist_lenet5.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    assert config["learning"]["objective"] == "history_ema_j"
    assert "online_bound_manifest_path" not in config["learning"]
    validate_config(config)
