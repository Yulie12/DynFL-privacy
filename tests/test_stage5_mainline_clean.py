"""The current paper selector must not depend on retired Omega heuristics."""
import pytest
import dynfed.selection as selection
from dynfed.selection import (
    Candidate, SelectionConfig, _evaluate_profile, _initial_profiles,
    _selection_learning_cost, choose_global_pareto_profile,
)


def _candidate(mode, time, mechanism="dp"):
    link = "L_E_upd" if mode == "LIIE" else "L_C_upd"
    return Candidate(
        mode=mode, mechanisms={"upd": mechanism}, link_mechanisms={link: mechanism},
        time=time, accuracy=0.1, risk=0.1, epsilon_used=1.0,
        communication_volume=1.0, feasible_resource=True,
        feasible_privacy=True, feasible_risk=True, feasible_time=True,
        update_noise_multiplier=0.5,
    )


def test_formal_learning_lookahead_matches_complete_profile_objective_with_fixed_admission():
    config = SelectionConfig(omega_update_clip_norm=0.25, omega_update_dimension=10.0)
    profile = {0: _candidate("LIIC", 3.0), 1: _candidate("LIIE", 1.0, "none")}
    masses = {0: 5.0, 1: 5.0}
    edges = {0: 0, 1: 0}
    admission = (0, 1)
    expected = _evaluate_profile(
        config, profile, masses, edges, {},
        flow_objectives=(admission, 4.0),
    )
    assert _selection_learning_cost(config, profile, masses, edges, admission) == pytest.approx(
        expected.system_learning_error
    )


def test_mainline_search_never_calls_retired_local_omega(monkeypatch):
    def retired(*args, **kwargs):
        raise AssertionError("retired Omega proxy entered current Pareto search")
    monkeypatch.setattr(selection, "_local_omega_proxy", retired)
    clients = range(6)
    edge = _candidate("LIIE", 1.0, "none")
    cloud = _candidate("LIIC", 2.0)
    pools = {client: [edge, cloud] for client in clients}
    config = SelectionConfig(
        num_clients=6, num_edges=2,
        omega_update_clip_norm=0.25, omega_update_dimension=10.0,
        pareto_max_iters=2, pareto_neighbor_top_k=2, pareto_archive_size=4,
    )
    assert _initial_profiles(config, pools, {}, {client: 1.0 for client in clients})
    chosen, result = choose_global_pareto_profile(
        config=config,
        selected=[(client, edge, pools[client], 8.0) for client in clients],
        client_samples={client: 1.0 for client in clients},
        client_edges={client: client % 2 for client in clients},
        previous_choices={},
    )
    assert len(chosen) == len(pools)
    assert result.system_learning_error == pytest.approx(result.fusion_bound + result.system_dp)
