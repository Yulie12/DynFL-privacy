"""Stage 4: mode admissibility and Pareto search diagnostics only."""
import json
import random
from dataclasses import replace

from dynfed.selection import (
    Candidate, SelectionConfig, _mode_search_pool_audit,
    choose_cloud_dp_pareto_profile, enumerate_candidates,
)
from dynfed.fmnist_lenet5_dynamic import (
    _mode_selection_csv_rows, _summarize_mode_selection_audit,
)


def c(mode, mechanism="none"):
    links = {"E_C_upd": mechanism, "L_E_upd": "none"} if mode == "LIIEIIIC" else {"L_E_upd": mechanism}
    return Candidate(
        mode=mode, mechanisms={"upd": mechanism}, link_mechanisms=links,
        time=1., accuracy=.5, risk=0., epsilon_used=0., communication_volume=1.,
        feasible_resource=True, feasible_privacy=True, feasible_risk=True,
        feasible_time=True, update_noise_multiplier=3.1,
    )


def enumerate_with_audit(config, audit=None):
    return enumerate_candidates(
        config=config, client_id=0, edge_factor=1, compute_factor=1,
        samples=100, remaining_epsilon=8., round_idx=0,
        rng=random.Random(8), policy="full_dynfl", mode_audit=audit,
    )


def test_audit_snapshots_fl_first_removals_and_keeps_candidate_order():
    config = SelectionConfig(rounds=2, fl_first_split_on_demand=True)
    audit = {}
    audited = enumerate_with_audit(config, audit)
    original = enumerate_with_audit(config)
    assert audited == original
    assert audit["stages"]["generated"]["LIC"] > 0
    assert audit["stages"]["after_fl_first"]["LIC"] == 0
    assert audit["stages"]["returned"]["LIIEIIIC"] > 0
    assert audit["budget_feasible"]["LIC"] == 0


def test_audit_records_config_exclusions_and_no_fl_first_filter():
    config = SelectionConfig(rounds=2, excluded_modes=("LIIC",), fl_first_split_on_demand=False)
    audit = {}
    enumerate_with_audit(config, audit)
    assert audit["generation_exclusions"]["LIIC"] == "excluded_mode"
    assert audit["stages"]["generated"]["LIC"] > 0
    assert audit["stages"]["after_fl_first"]["LIC"] == audit["stages"]["generated"]["LIC"]
    assert set(audit["rejections"]) >= {"resource", "memory", "privacy", "epsilon_limit"}


def test_search_pool_reports_clients_separately_from_candidate_count():
    pools = {0: [c("LIIE"), c("LIIE"), c("LIIEIIIC")], 1: [c("LIIEIIIC")]}
    audit = _mode_search_pool_audit(pools)
    assert audit["LIIE"] == {"candidates": 2, "clients": 1}
    assert audit["LIIEIIIC"] == {"candidates": 2, "clients": 2}
    assert audit["LIIC"] == {"candidates": 0, "clients": 0}


def test_cloud_both_branches_preserve_audit_and_final_counts():
    config = SelectionConfig(
        cloud_dp_plan="pareto", num_clients=2, num_edges=2,
        omega_update_dimension=4., omega_update_clip_norm=.25,
        pareto_max_iters=1, pareto_archive_size=8,
    )
    base = c("LIIEIIIC", "dp_he3")
    selected = [(i, base, [base], 8.) for i in (0, 1)]
    diagnostics = {}
    result, winner = choose_cloud_dp_pareto_profile(
        config=config, selected=selected, client_samples={0: 1., 1: 1.},
        client_edges={0: 0, 1: 1}, diagnostics=diagnostics,
    )
    assert set(diagnostics["cloud_dp_mode_branches"]) == {"packet", "aggregate"}
    assert diagnostics["cloud_dp_mode_branches"]["packet"]["pool"]["LIIEIIIC"]["clients"] == 2
    assert diagnostics["mode_chosen"]["LIIEIIIC"] == 2
    assert diagnostics["mode_archive_presence"]["LIIEIIIC"] >= 1
    assert all(item[1] == winner.profile[item[0]] for item in result)


def test_summary_and_csv_keep_stage_counts_and_risk_is_diagnostic_only():
    audit = {}
    enumerate_with_audit(SelectionConfig(rounds=2, fl_first_split_on_demand=True), audit)
    selected = [(0, c("LIIE"), [c("LIIE")], 8.)]
    summary = _summarize_mode_selection_audit([audit], {}, selected)
    assert summary["LIC"]["generated"] > 0
    assert summary["LIC"]["returned"] == 0
    assert summary["LIIE"]["chosen_clients"] == 1
    assert summary["LIC"]["search_pool_candidates"] == ""
    records = _mode_selection_csv_rows([
        {"policy": "full_dynfl", "round": 0,
         "mode_selection_audit": json.dumps(summary)},
    ])
    assert len(records) == 7
    assert next(row for row in records if row["mode"] == "LIC")["generated"] > 0
    assert not _mode_selection_csv_rows([{"round": 1}])
