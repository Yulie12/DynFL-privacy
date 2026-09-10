import dynfed.selection as selection_module
from dynfed.flow_executor import execute_mixed_round_flow
from dynfed.selection import Candidate, SelectionConfig, _initial_profiles
from dynfed.selection import (
    ProfileEvaluation,
    _apply_policy_candidate_filters,
    _pareto_search_client_ids,
    _repair_edge_cloud_coverage,
    _stable_cloud_candidate_pool,
    choose_global_pareto_profile,
    evaluate_global_profile,
)


def _candidate(mode: str, time: float) -> Candidate:
    return Candidate(
        mode=mode,
        mechanisms={"upd": "he3"},
        time=time,
        accuracy=0.5,
        risk=0.1,
        epsilon_used=0.0,
        communication_volume=1.0,
        feasible_resource=True,
        feasible_privacy=True,
        feasible_risk=True,
        feasible_time=True,
    )


def test_lowest_omega_seed_prefers_cloud_when_local_error_is_tied() -> None:
    pools = {
        client_id: [
            _candidate("LIIE", time=1.0),
            _candidate("LIIC", time=2.0),
        ]
        for client_id in range(8)
    }

    profiles = _initial_profiles(SelectionConfig(), pools, previous_choices={})

    assert all(candidate.mode == "LIIE" for candidate in profiles[0].values())
    assert all(candidate.mode == "LIIC" for candidate in profiles[-1].values())


def test_each_profile_derives_its_own_admitted_clients() -> None:
    config = SelectionConfig(
        num_clients=4,
        num_edges=1,
        aggregation_fraction=0.5,
    )
    samples = {client_id: 1.0 for client_id in range(4)}
    edges = {client_id: 0 for client_id in range(4)}

    first_profile = [
        (client_id, _candidate("LIIC", time), [], 8.0)
        for client_id, time in enumerate((1.0, 2.0, 3.0, 4.0))
    ]
    second_profile = [
        (client_id, _candidate("LIIC", time), [], 8.0)
        for client_id, time in enumerate((5.0, 2.0, 3.0, 4.0))
    ]

    first = evaluate_global_profile(
        config=config,
        selected=first_profile,
        client_samples=samples,
        client_edges=edges,
    )
    second = evaluate_global_profile(
        config=config,
        selected=second_profile,
        client_samples=samples,
        client_edges=edges,
    )

    assert first.admitted_client_ids == (0, 1)
    assert second.admitted_client_ids == (1, 2)
    assert first.system_latency != second.system_latency


def test_edge_cloud_coverage_repair_keeps_one_cloud_path_per_edge() -> None:
    edge = _candidate("LIIE", time=1.0)
    cloud = _candidate("LIIC", time=2.0)
    profile = {client_id: edge for client_id in range(4)}
    config = SelectionConfig(
        num_clients=4,
        num_edges=2,
        pareto_archive_size=4,
        require_edge_cloud_coverage=True,
        min_edge_cloud_fusion_ratio=0.5,
    )

    repaired = _repair_edge_cloud_coverage(
        config=config,
        chosen=ProfileEvaluation(profile, 1.0, 1.0, 0.0),
        pools={client_id: [edge, cloud] for client_id in profile},
        client_samples={client_id: 1.0 for client_id in profile},
        client_edges={0: 0, 1: 0, 2: 1, 3: 1},
        previous_choices={},
    )

    for edge_id in (0, 1):
        assert any(
            repaired.profile[client_id].mode == "LIIC"
            for client_id in profile
            if (0 if client_id < 2 else 1) == edge_id
        )


def test_pareto_archive_contains_only_coverage_feasible_profiles() -> None:
    edge = _candidate("LIIE", time=1.0)
    cloud = _candidate("LIIC", time=2.0)
    selected = [
        (client_id, edge, [edge, cloud], 8.0)
        for client_id in range(4)
    ]
    diagnostics: dict[str, object] = {}
    config = SelectionConfig(
        num_clients=4,
        num_edges=2,
        pareto_archive_size=8,
        pareto_max_iters=3,
        require_edge_cloud_coverage=True,
        min_edge_cloud_fusion_ratio=0.5,
    )

    choose_global_pareto_profile(
        config=config,
        selected=selected,
        client_samples={client_id: 1.0 for client_id in range(4)},
        client_edges={0: 0, 1: 0, 2: 1, 3: 1},
        diagnostics=diagnostics,
    )

    for evaluation in diagnostics["archive"]:
        for edge_clients in ((0, 1), (2, 3)):
            assert sum(
                evaluation.profile[client_id].mode == "LIIC"
                for client_id in edge_clients
            ) >= 1
    assert diagnostics["candidate_pool_sizes_before_stability"] == diagnostics["candidate_pool_sizes"]
    assert diagnostics["update_mechanism_counts_before_stability"] == {
        "dp_only": 0,
        "he_only": 8,
        "dp_he": 0,
        "neither": 0,
    }


def test_pareto_conflict_only_limits_search_clients() -> None:
    shared = _candidate("LIIC", time=1.0)
    edge = _candidate("LIIE", time=0.5)
    cloud = _candidate("LIIC", time=2.0)
    pools = {
        0: [edge, cloud],
        1: [shared],
        2: [edge, cloud],
    }
    config = SelectionConfig(pareto_conflict_only=True)
    seeds = _initial_profiles(config, pools, previous_choices={})

    assert _pareto_search_client_ids(config, pools, seeds) == (0, 2)


def test_pareto_neighbor_budget_search_runs() -> None:
    edge = _candidate("LIIE", time=0.5)
    cloud = _candidate("LIIC", time=2.0)
    selected = [
        (client_id, edge, [edge, cloud], 8.0)
        for client_id in range(6)
    ]
    config = SelectionConfig(
        pareto_archive_size=4,
        pareto_max_iters=2,
        pareto_neighbor_top_k=3,
        pareto_conflict_only=True,
    )

    rewritten, evaluation = choose_global_pareto_profile(
        config=config,
        selected=selected,
        client_samples={client_id: 1.0 for client_id in range(6)},
        client_edges={client_id: client_id % 2 for client_id in range(6)},
        previous_choices={},
    )

    assert len(rewritten) == len(selected)
    assert evaluation.profile


def test_nsga2_reference_search_is_deterministic_and_coverage_feasible() -> None:
    edge = _candidate("LIIE", time=0.5)
    cloud = _candidate("LIIC", time=1.5)
    selected = [
        (client_id, edge, [edge, cloud], 8.0)
        for client_id in range(6)
    ]
    config = SelectionConfig(
        seed=42,
        num_clients=6,
        num_edges=2,
        pareto_archive_size=6,
        pareto_max_iters=4,
        require_edge_cloud_coverage=True,
        min_edge_cloud_fusion_ratio=0.5,
    )
    samples = {client_id: 1.0 for client_id in range(6)}
    edges = {client_id: client_id % 2 for client_id in range(6)}
    diagnostics: dict[str, object] = {}

    first, first_evaluation = choose_global_pareto_profile(
        config=config,
        selected=selected,
        client_samples=samples,
        client_edges=edges,
        search_method="nsga2",
        diagnostics=diagnostics,
    )
    second, second_evaluation = choose_global_pareto_profile(
        config=config,
        selected=selected,
        client_samples=samples,
        client_edges=edges,
        search_method="nsga2",
    )

    assert diagnostics["search_method"] == "nsga2"
    assert [item[1] for item in first] == [item[1] for item in second]
    assert first_evaluation.profile_signature == second_evaluation.profile_signature
    for edge_id in (0, 1):
        edge_clients = [client_id for client_id in range(6) if edges[client_id] == edge_id]
        assert sum(
            first_evaluation.profile[client_id].mode == "LIIC"
            for client_id in edge_clients
        ) >= 2


def test_latency_ablation_ignores_convergence_objective() -> None:
    fast_dp = Candidate(
        **{
            **_candidate("LIIC", time=0.5).__dict__,
            "mechanisms": {"upd": "dp"},
        }
    )
    slow_he = _candidate("LIIC", time=2.0)
    selected = [(0, slow_he, [fast_dp, slow_he], 8.0)]

    rewritten, evaluation = choose_global_pareto_profile(
        config=SelectionConfig(
            num_clients=1,
            num_edges=1,
            pareto_archive_size=4,
            pareto_max_iters=2,
        ),
        selected=selected,
        client_samples={0: 1.0},
        client_edges={0: 0},
        previous_choices={},
        objective="latency",
    )

    assert rewritten[0][1] == fast_dp
    assert evaluation.profile[0] == fast_dp


def test_optimized_pareto_search_matches_full_event_evaluation(monkeypatch) -> None:
    edge = _candidate("LIIE", time=0.7)
    direct = Candidate(
        **{
            **_candidate("LIIC", time=1.1).__dict__,
            "cloud_aggregation_payload": 2.0,
            "return_path_time": 0.2,
        }
    )
    edge_cloud = Candidate(
        **{
            **_candidate("LIEIIIC", time=0.9).__dict__,
            "edge_to_cloud_time": 0.1,
            "edge_aggregation_payload": 1.5,
            "cloud_aggregation_payload": 1.0,
            "return_path_time": 0.1,
        }
    )
    selected = [
        (client_id, edge, [edge, direct, edge_cloud], 8.0)
        for client_id in range(10)
    ]
    config = SelectionConfig(
        num_clients=10,
        num_edges=2,
        aggregation_fraction=1.0,
        pareto_archive_size=6,
        pareto_max_iters=4,
        pareto_neighbor_top_k=0,
        pareto_conflict_only=False,
    )
    samples = {client_id: float(client_id + 1) for client_id in range(10)}
    edges = {client_id: client_id % 2 for client_id in range(10)}

    optimized_selected, optimized = choose_global_pareto_profile(
        config=config,
        selected=selected,
        client_samples=samples,
        client_edges=edges,
        previous_choices={},
    )

    monkeypatch.setattr(selection_module, "_full_buffer_flow_stats", lambda *args: None)
    monkeypatch.setattr(
        selection_module,
        "summarize_mixed_round_flow",
        execute_mixed_round_flow,
    )
    reference_selected, reference = choose_global_pareto_profile(
        config=config,
        selected=selected,
        client_samples=samples,
        client_edges=edges,
        previous_choices={},
    )

    assert [item[1] for item in optimized_selected] == [item[1] for item in reference_selected]
    assert optimized.profile_signature == reference.profile_signature
    assert optimized.admitted_client_ids == reference.admitted_client_ids
    assert abs(optimized.system_latency - reference.system_latency) < 1e-12
    assert abs(optimized.system_omega - reference.system_omega) < 1e-12
    assert abs(optimized.cloud_fusion_ratio - reference.cloud_fusion_ratio) < 1e-12


def test_unstable_cloud_dp_is_removed_when_he_is_available() -> None:
    config = SelectionConfig(
        rounds=100,
        dp_feature_epsilon_budget=8.0,
        dp_update_epsilon_budget=8.0,
        enforce_cloud_dp_stability=True,
    )
    cloud_feature_dp = Candidate(
        **{
            **_candidate("LIC", time=1.0).__dict__,
            "mechanisms": {"emb": "dp", "grad": "dp", "label": "none"},
        }
    )
    cloud_update_he = Candidate(
        **{
            **_candidate("LIIC", time=2.0).__dict__,
            "mechanisms": {"upd": "he3"},
        }
    )
    edge_update_dp = Candidate(
        **{
            **_candidate("LIIE", time=1.0).__dict__,
            "mechanisms": {"upd": "dp"},
        }
    )

    stable = _stable_cloud_candidate_pool(
        config,
        [cloud_feature_dp, cloud_update_he, edge_update_dp],
    )

    assert cloud_feature_dp not in stable
    assert cloud_update_he in stable
    assert edge_update_dp in stable


def test_tex_stability_uses_coordinate_noise_not_dimension(monkeypatch) -> None:
    monkeypatch.setattr(selection_module, "resolved_privacy_parameters", lambda config: {
        "feature_noise_multiplier": 0.4, "update_noise_multiplier": 0.4,
    })
    dp = Candidate(**{**_candidate("LIIC", time=1.0).__dict__,
                      "mechanisms": {"upd": "dp"}})
    he = _candidate("LIIC", time=2.0)
    config = SelectionConfig(trusted_edge_split_execution=True,
                             omega_update_dimension=10_490_890,
                             cloud_dp_stability_threshold=1.0)
    assert _stable_cloud_candidate_pool(config, [dp, he]) == [dp, he]
    monkeypatch.setattr(selection_module, "resolved_privacy_parameters", lambda config: {
        "feature_noise_multiplier": 0.6, "update_noise_multiplier": 0.6,
    })
    assert _stable_cloud_candidate_pool(config, [dp, he]) == [he]
    assert _stable_cloud_candidate_pool(config, [dp]) == [dp]


def test_adaptive_baselines_share_cloud_dp_stability_filter() -> None:
    config = SelectionConfig(
        rounds=100,
        dp_feature_epsilon_budget=8.0,
        dp_update_epsilon_budget=8.0,
        enforce_cloud_dp_stability=True,
    )
    cloud_update_dp = Candidate(
        **{
            **_candidate("LIIC", time=1.0).__dict__,
            "mechanisms": {"upd": "dp"},
        }
    )
    cloud_update_he = _candidate("LIIC", time=2.0)
    candidates = [cloud_update_dp, cloud_update_he]

    for policy in ("individual_optimal", "random"):
        filtered = _apply_policy_candidate_filters(config, policy, candidates)
        assert cloud_update_dp not in filtered
        assert cloud_update_he in filtered


def test_fixed_dp_is_not_rewritten_by_he_stability_filter() -> None:
    config = SelectionConfig(enforce_cloud_dp_stability=True)
    cloud_update_dp = Candidate(
        **{
            **_candidate("LIIC", time=1.0).__dict__,
            "mechanisms": {"upd": "dp"},
        }
    )
    cloud_update_he = _candidate("LIIC", time=2.0)

    filtered = _apply_policy_candidate_filters(
        config,
        "fixed_dp",
        [cloud_update_dp, cloud_update_he],
    )

    assert filtered == [cloud_update_dp, cloud_update_he]


def test_fixed_mode_ablation_keeps_global_privacy_selection() -> None:
    edge_mode = _candidate("LIIE", time=0.5)
    fixed_dp = Candidate(
        **{
            **_candidate("LIIEIIIC", time=1.0).__dict__,
            "mechanisms": {"upd": "dp"},
        }
    )
    fixed_he = _candidate("LIIEIIIC", time=1.5)
    config = SelectionConfig(
        num_clients=1,
        num_edges=1,
        rounds=200,
        enforce_cloud_dp_stability=True,
        pareto_max_iters=1,
    )
    candidates = _apply_policy_candidate_filters(
        config,
        "ours_fixed_liieiiic",
        [edge_mode, fixed_dp, fixed_he],
    )

    rewritten, _evaluation = choose_global_pareto_profile(
        config=config,
        selected=[(0, fixed_dp, candidates, 8.0)],
        client_samples={0: 1.0},
        client_edges={0: 0},
        previous_choices={},
    )

    assert candidates == [fixed_dp, fixed_he]
    assert rewritten[0][1] == fixed_he


def test_reference_policies_fix_only_the_collaboration_topology() -> None:
    candidates = [
        _candidate("LIIC", time=1.0),
        _candidate("LIEIIC", time=1.0),
        _candidate("LIIEIIIC", time=1.0),
        _candidate("LIIE", time=1.0),
    ]
    config = SelectionConfig()

    assert {
        item.mode
        for item in _apply_policy_candidate_filters(config, "fixed_fedavg", candidates)
    } == {"LIIC"}
    assert {
        item.mode
        for item in _apply_policy_candidate_filters(config, "fixed_splitfed", candidates)
    } == {"LIEIIC"}
    assert {
        item.mode
        for item in _apply_policy_candidate_filters(config, "fixed_hfl", candidates)
    } == {"LIIEIIIC"}


def test_cloud_update_dp_is_removed_when_only_update_link_can_use_he() -> None:
    config = SelectionConfig(
        rounds=100,
        dp_feature_epsilon_budget=8.0,
        dp_update_epsilon_budget=8.0,
        enforce_cloud_dp_stability=True,
    )
    all_dp = Candidate(
        **{
            **_candidate("LIEIIC", time=1.0).__dict__,
            "mechanisms": {"emb": "dp", "grad": "dp", "upd": "dp"},
            "link_mechanisms": {
                "L_E_emb": "dp",
                "E_L_grad": "dp",
                "E_C_upd": "dp",
            },
        }
    )
    he_update = Candidate(
        **{
            **_candidate("LIEIIC", time=2.0).__dict__,
            "mechanisms": {"emb": "dp", "grad": "dp", "upd": "he3"},
            "link_mechanisms": {
                "L_E_emb": "dp",
                "E_L_grad": "dp",
                "E_C_upd": "he3",
            },
        }
    )

    stable = _stable_cloud_candidate_pool(config, [all_dp, he_update])

    assert all_dp not in stable
    assert he_update in stable
