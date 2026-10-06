import random

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



def test_qos_admission_requires_cloud_for_ordinary_client() -> None:
    config = SelectionConfig(
        edge_only_requires_fast_deadline=True,
        fl_first_split_on_demand=True,
    )

    candidates = selection_module.enumerate_candidates(
        config=config,
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        memory_capacity_factor=1.35,
        samples=100,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(42),
        policy="full_dynfl",
        fast_response_deadline=None,
    )

    modes = {candidate.mode for candidate in candidates}

    assert not (modes & {"LIE", "LIIE"})
    assert modes
    assert all(
        selection_module._candidate_reaches_cloud(candidate)
        for candidate in candidates
    )


def test_qos_admission_allows_edge_only_for_fast_client() -> None:
    config = SelectionConfig(
        edge_only_requires_fast_deadline=True,
        fl_first_split_on_demand=True,
    )

    candidates = selection_module.enumerate_candidates(
        config=config,
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        memory_capacity_factor=1.35,
        samples=100,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(42),
        policy="full_dynfl",
        fast_response_deadline=1e9,
    )

    modes = {candidate.mode for candidate in candidates}

    assert "LIIE" in modes


def test_qos_admission_opens_split_cloud_when_full_local_memory_is_insufficient() -> None:
    config = SelectionConfig(
        edge_only_requires_fast_deadline=True,
        fl_first_split_on_demand=True,
    )

    candidates = selection_module.enumerate_candidates(
        config=config,
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        memory_capacity_factor=0.6,
        samples=100,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(42),
        policy="full_dynfl",
        fast_response_deadline=None,
    )

    modes = {candidate.mode for candidate in candidates}
    feasible = [candidate for candidate in candidates if candidate.feasible_device]
    split_modes = {"LIC", "LIEIIC", "LIEIIIC"}

    assert not (modes & {"LIE", "LIIE"})
    assert any(candidate.mode in split_modes for candidate in feasible)
    assert all(
        selection_module._candidate_reaches_cloud(candidate)
        for candidate in feasible
    )


def test_formal_learning_seed_explores_cloud_when_local_cost_is_tied() -> None:
    pools = {
        client_id: [
            _candidate("LIIE", time=1.0),
            _candidate("LIIC", time=2.0),
        ]
        for client_id in range(8)
    }

    profiles = _initial_profiles(SelectionConfig(), pools, previous_choices={})

    assert all(candidate.mode == "LIIE" for candidate in profiles[0].values())
    assert any(all(candidate.mode == "LIIC" for candidate in profile.values()) for profile in profiles)


def test_initial_profiles_include_sample_mass_cloud_coverage_anchors() -> None:
    pools = {
        client_id: [
            _candidate("LIIE", time=1.0),
            _candidate("LIIC", time=2.0 + 0.1 * client_id),
        ]
        for client_id in range(4)
    }
    samples = {0: 4.0, 1: 3.0, 2: 2.0, 3: 1.0}

    profiles = _initial_profiles(
        SelectionConfig(),
        pools,
        previous_choices={},
        client_samples=samples,
    )
    ratios = [
        sum(
            samples[client_id]
            for client_id, candidate in profile.items()
            if selection_module._candidate_reaches_cloud(candidate)
        )
        / sum(samples.values())
        for profile in profiles
    ]

    assert any(ratio >= 0.25 for ratio in ratios)
    assert any(ratio >= 0.50 for ratio in ratios)
    assert any(ratio >= 0.75 for ratio in ratios)
    assert any(ratio >= 1.00 for ratio in ratios)


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

    # Q70: aggregation_fraction controls Edge Buffer only; Cloud waits for all
    # legal cloud-bound contributions in the current round.
    assert first.admitted_client_ids == (0, 1, 2, 3)
    assert second.admitted_client_ids == (0, 1, 2, 3)
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


def test_coverage_infeasible_fallback_records_failure_edges() -> None:
    edge = _candidate("LIIE", time=1.0)
    selected = [
        (0, edge, [edge], 8.0),
        (1, edge, [edge], 8.0),
    ]
    diagnostics: dict[str, object] = {}
    config = SelectionConfig(
        num_clients=2,
        num_edges=1,
        require_edge_cloud_coverage=True,
        min_edge_cloud_fusion_ratio=0.5,
    )

    rewritten, evaluation = choose_global_pareto_profile(
        config=config,
        selected=selected,
        client_samples={0: 8.0, 1: 2.0},
        client_edges={0: 0, 1: 0},
        diagnostics=diagnostics,
    )

    assert all(candidate.mode == "SKIP" for _, candidate, _, _ in rewritten)
    assert evaluation.profile
    assert diagnostics["coverage_infeasible_fallback"] is True
    assert diagnostics["coverage_target_ratio"] == 0.5
    failures = diagnostics["coverage_failure_edges"]
    assert isinstance(failures, list)
    assert failures[0]["edge_id"] == 0
    assert failures[0]["maximum_coverage_ratio"] == 0.0


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


def test_infeasible_edge_cloud_coverage_uses_skip_fallback_instead_of_crashing() -> None:
    edge = _candidate("LIIE", time=1.0)
    cloud = _candidate("LIIC", time=2.0)
    selected = [
        (0, edge, [edge], 8.0),
        (1, edge, [edge], 8.0),
        (2, cloud, [cloud], 8.0),
        (3, cloud, [cloud], 8.0),
    ]
    diagnostics: dict[str, object] = {}
    config = SelectionConfig(
        num_clients=4,
        num_edges=2,
        require_edge_cloud_coverage=True,
        min_edge_cloud_fusion_ratio=0.5,
    )

    rewritten, evaluation = choose_global_pareto_profile(
        config=config,
        selected=selected,
        client_samples={client_id: 1.0 for client_id in range(4)},
        client_edges={0: 0, 1: 0, 2: 1, 3: 1},
        diagnostics=diagnostics,
    )

    assert all(candidate.mode == "SKIP" for _client_id, candidate, _pool, _remaining in rewritten)
    assert all(candidate.mode == "SKIP" for candidate in evaluation.profile.values())
    assert diagnostics["coverage_infeasible_fallback"] is True
    assert diagnostics["archive"] == ()


def test_sample_dp_optimizer_event_count_matches_runtime_step_limit():
    from dynfed.selection import _sample_dp_optimizer_event_count

    batch_size = 128
    epochs = 3
    local_steps = 5

    expected = {
        0: 0,
        1: 3,
        128: 3,
        129: 5,
        240: 5,
        640: 5,
        641: 5,
    }

    for samples, event_count in expected.items():
        assert (
            _sample_dp_optimizer_event_count(
                samples,
                batch_size,
                epochs,
                local_steps,
            )
            == event_count
        )


def test_sample_dp_optimizer_event_count_without_step_limit():
    from dynfed.selection import _sample_dp_optimizer_event_count

    assert _sample_dp_optimizer_event_count(
        samples=240,
        batch_size=128,
        epochs=3,
        local_steps=None,
    ) == 6


def test_sample_dp_optimizer_event_count_zero_step_limit():
    from dynfed.selection import _sample_dp_optimizer_event_count

    assert _sample_dp_optimizer_event_count(
        samples=240,
        batch_size=128,
        epochs=3,
        local_steps=0,
    ) == 0

def test_sample_mechanisms_split_force_embedding_and_gradient_dp():
    from dynfed.selection import _sample_mechanism_assignments
    from dynfed.training import MODE_SPECS

    assignments = _sample_mechanism_assignments(
        MODE_SPECS["LIEIIC"],
        allow_he=True,
    )

    assert len(assignments) == 1
    _summary, links = assignments[0]

    assert links["L_E_emb"] == "dp"
    assert links["L_E_grad"] == "dp"
    assert links["E_C_upd"] == "he3"


def test_sample_mechanisms_full_local_do_not_add_update_dp():
    from dynfed.selection import _sample_mechanism_assignments
    from dynfed.training import MODE_SPECS

    edge_assignments = _sample_mechanism_assignments(
        MODE_SPECS["LIIE"],
        allow_he=True,
    )
    cloud_assignments = _sample_mechanism_assignments(
        MODE_SPECS["LIIC"],
        allow_he=True,
    )

    assert edge_assignments[0][1]["L_E_upd"] == "none"
    assert cloud_assignments[0][1]["L_C_upd"] == "he3"


def test_sample_mechanisms_reject_cloud_update_when_he_is_unavailable():
    from dynfed.selection import _sample_mechanism_assignments
    from dynfed.training import MODE_SPECS

    assert _sample_mechanism_assignments(
        MODE_SPECS["LIIC"],
        allow_he=False,
    ) == []

    assert _sample_mechanism_assignments(
        MODE_SPECS["LIEIIC"],
        allow_he=False,
    ) == []

def test_sample_dp_event_counts_follow_actual_minibatch_steps():
    from dynfed.selection import SelectionConfig, _sample_dp_event_counts

    config = SelectionConfig(
        split_batch_size=128,
        privacy_local_epochs=3,
        L_block_cycles=5,
        mainline_fusion=False,
    )

    assert _sample_dp_event_counts(
        config,
        "LIIC",
        240,
    ) == (0, 0, 5)

    assert _sample_dp_event_counts(
        config,
        "LIEIIC",
        240,
    ) == (5, 5, 5)


def test_sample_dp_event_counts_small_split_client_follow_epochs():
    from dynfed.selection import SelectionConfig, _sample_dp_event_counts

    config = SelectionConfig(
        split_batch_size=128,
        privacy_local_epochs=3,
        L_block_cycles=5,
        mainline_fusion=False,
    )

    assert _sample_dp_event_counts(
        config,
        "LIEIIC",
        1,
    ) == (3, 3, 3)


def test_sample_dp_event_counts_full_local_have_no_split_releases():
    from dynfed.selection import SelectionConfig, _sample_dp_event_counts

    config = SelectionConfig(
        split_batch_size=128,
        privacy_local_epochs=3,
        L_block_cycles=5,
        mainline_fusion=False,
    )

    assert _sample_dp_event_counts(
        config,
        "LIIE",
        240,
    ) == (0, 0, 5)

def test_sample_selector_accounting_full_local_uses_optimizer_only():
    import random

    from dynfed.selection import (
        SelectionConfig,
        build_sample_privacy_ledger,
        enumerate_candidates,
    )
    from dynfed.training import MODE_SPECS

    config = SelectionConfig(
        privacy_unit="sample",
        excluded_modes=tuple(
            mode
            for mode in MODE_SPECS
            if mode != "LIIC"
        ),
        rounds=100,
        split_batch_size=128,
        privacy_local_epochs=3,
        L_block_cycles=5,
    )

    ledger = build_sample_privacy_ledger(config)

    candidates = enumerate_candidates(
        config=config,
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=240,
        remaining_epsilon=ledger.remaining_budget,
        round_idx=0,
        rng=random.Random(0),
        policy="ours",
        privacy_ledger=ledger,
    )

    assert len(candidates) == 1
    candidate = candidates[0]

    assert candidate.mode == "LIIC"
    assert candidate.link_mechanisms["L_C_upd"] == "he3"

    assert candidate.sample_embedding_events == 0
    assert candidate.sample_label_grad_events == 0
    assert candidate.sample_optimizer_events == 5

    assert candidate.sample_epsilon_after > 0.0

    assert candidate.feature_dp_events == 0
    assert candidate.update_dp_events == 0

    assert (
        candidate.sample_optimizer_noise_multiplier
        is not None
    )


def test_sample_selector_accounting_split_composes_three_event_classes():
    import random

    from dynfed.selection import (
        SelectionConfig,
        build_sample_privacy_ledger,
        enumerate_candidates,
    )
    from dynfed.training import MODE_SPECS

    config = SelectionConfig(
        privacy_unit="sample",
        excluded_modes=tuple(
            mode
            for mode in MODE_SPECS
            if mode != "LIEIIC"
        ),
        rounds=100,
        split_batch_size=128,
        privacy_local_epochs=3,
        L_block_cycles=5,
    )

    ledger = build_sample_privacy_ledger(config)

    candidates = enumerate_candidates(
        config=config,
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=240,
        remaining_epsilon=ledger.remaining_budget,
        round_idx=0,
        rng=random.Random(0),
        policy="ours",
        privacy_ledger=ledger,
    )

    assert len(candidates) == 1
    candidate = candidates[0]

    assert candidate.mode == "LIEIIC"

    assert candidate.link_mechanisms["L_E_emb"] == "dp"
    assert candidate.link_mechanisms["L_E_grad"] == "dp"
    assert candidate.link_mechanisms["E_C_upd"] == "he3"

    assert candidate.sample_embedding_events == 5
    assert candidate.sample_label_grad_events == 5
    assert candidate.sample_optimizer_events == 5

    assert candidate.sample_epsilon_after > 0.0

    assert candidate.feature_dp_events == 0
    assert candidate.update_dp_events == 0

    assert (
        candidate.sample_embedding_noise_multiplier
        is not None
    )
    assert (
        candidate.sample_label_grad_noise_multiplier
        is not None
    )
    assert (
        candidate.sample_optimizer_noise_multiplier
        is not None
    )
