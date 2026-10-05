"""Stage 7: selection profiling is diagnostic-only and exposes the real hotspots."""

from dynfed.selection import Candidate, SelectionConfig, choose_cloud_dp_pareto_profile


def _candidate(mode: str, link: str, mechanism: str, *, time_value: float) -> Candidate:
    links = {link: mechanism}
    if mode == "LIIEIIIC":
        links["L_E_upd"] = "none"
    return Candidate(
        mode=mode,
        mechanisms={"upd": mechanism},
        link_mechanisms=links,
        time=time_value,
        accuracy=0.0,
        risk=0.0,
        epsilon_used=0.0,
        communication_volume=1.0,
        feasible_resource=True,
        feasible_privacy=True,
        feasible_risk=True,
        feasible_time=True,
        update_noise_multiplier=1.25,
    )


def test_stage7_profiler_reports_both_cloud_branches_without_changing_choice():
    config = SelectionConfig(
        cloud_dp_plan="pareto",
        num_clients=4,
        num_edges=2,
        omega_update_dimension=32.0,
        omega_update_clip_norm=0.25,
        pareto_max_iters=2,
        pareto_archive_size=8,
        pareto_beam_size=2,
        pareto_neighbor_top_k=3,
    )
    hierarchical = _candidate("LIIEIIIC", "E_C_upd", "dp_he3", time_value=2.0)
    local_edge = _candidate("LIIE", "L_E_upd", "none", time_value=1.0)
    selected = [
        (cid, hierarchical, [hierarchical, local_edge], 8.0)
        for cid in range(4)
    ]
    samples = {cid: 1.0 + cid for cid in range(4)}
    edges = {cid: cid % 2 for cid in range(4)}

    diagnostics = {}
    result, chosen = choose_cloud_dp_pareto_profile(
        config=config,
        selected=selected,
        client_samples=samples,
        client_edges=edges,
        diagnostics=diagnostics,
    )

    assert len(result) == 4
    assert chosen.profile
    assert set(diagnostics["cloud_dp_branch_performance"]) == {"packet", "aggregate"}
    for branch in ("packet", "aggregate"):
        profile = diagnostics["cloud_dp_branch_performance"][branch]
        assert profile["branch_wall_sec"] >= 0.0
        assert profile["evaluate_calls"] >= profile["evaluate_cache_misses"] >= 1
        assert profile["evaluate_profile_sec"] >= 0.0
        assert profile["pareto_archive_calls"] >= 1
        assert profile["dominance_compare_count"] >= 0
        assert profile["replacement_priority_calls"] >= 0
        assert profile["neighbor_candidates_generated"] >= profile["neighbor_candidates_retained"]


def test_stage7_profiler_is_diagnostic_only_for_fixed_inputs():
    config = SelectionConfig(
        cloud_dp_plan="aggregate",
        num_clients=3,
        num_edges=1,
        omega_update_dimension=16.0,
        omega_update_clip_norm=0.25,
        pareto_max_iters=1,
        pareto_archive_size=4,
        pareto_neighbor_top_k=2,
    )
    hierarchical = _candidate("LIIEIIIC", "E_C_upd", "dp_he3", time_value=2.0)
    local_edge = _candidate("LIIE", "L_E_upd", "none", time_value=1.0)
    selected = [(cid, hierarchical, [hierarchical, local_edge], 8.0) for cid in range(3)]
    samples = {0: 1.0, 1: 2.0, 2: 3.0}
    edges = {0: 0, 1: 0, 2: 0}

    plain_result, plain_eval = choose_cloud_dp_pareto_profile(
        config=config, selected=selected, client_samples=samples, client_edges=edges,
    )
    diagnostics = {}
    profiled_result, profiled_eval = choose_cloud_dp_pareto_profile(
        config=config, selected=selected, client_samples=samples, client_edges=edges,
        diagnostics=diagnostics,
    )

    assert [item[1] for item in plain_result] == [item[1] for item in profiled_result]
    assert plain_eval.system_latency == profiled_eval.system_latency
    assert plain_eval.system_omega == profiled_eval.system_omega
    assert diagnostics["performance_profile"]["search_total_sec"] >= 0.0
