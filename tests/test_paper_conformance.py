from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch

from dynfed.flow_executor import (
    ClientFlowInput,
    execute_mixed_round_flow,
    summarize_mixed_round_flow,
)
from dynfed.fmnist_lenet5_dynamic import (
    _aggregate_dp_release_parameters,
    _candidate_cloud_update_mechanism,
    _candidate_training_mechanisms,
    _candidate_uses_aggregate_update_dp,
    _should_apply_update_dp,
    _aggregate_returned_client_models,
    _state_difference_from_client_update,
    _training_base_state,
    _profiles_with_actual_samples,
    _feature_clip_excess_sq_by_client,
    _client_training_seed,
    _dp_noise_seed,
    _edge_normalized_cloud_weights,
    _weighted_state_difference_norm,
    _partition_clients_lenet5,
)
from dynfed.nodes import ClientProfile
from dynfed.selection import (
    Candidate,
    ProfileEvaluation,
    SelectionConfig,
    _admitted_clients_for_profile,
    _aggregation_sizes,
    _cloud_client_aggregation_weights,
    _candidate_dp_event_counts,
    _global_omega_proxy,
    _local_omega_proxy,
    _omega_from_profile_stats,
    _pareto_archive,
    _full_buffer_flow_stats,
    _profile_flow_inputs_by_candidate,
    _profile_omega_stats,
    _replace_full_buffer_flow_objectives,
    _replace_profile_omega_stats,
    _mode_link_events,
    _link_bandwidth,
    enumerate_candidates,
    resolved_privacy_parameters,
)
from dynfed.split_learning import (
    _protect_batched_average_gradient_dp,
    _protect_tensor_dp,
    _training_batches,
    apply_unified_dp,
    build_split_models,
    clip_state_difference,
    split_local_train_lenet5,
)
from dynfed.privacy import OBJECT_SIZES, PRIVACY_ALPHA
from experiments.run_paper_config import build_command


def test_paper_smoke_limit_does_not_change_rdp_round_horizon() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "paper_v25_cifar10_resnet18.json"
    )
    config = json.loads(config_path.read_text(encoding="utf-8"))

    command = build_command(
        config,
        seed=42,
        policies=["ours"],
        rounds=None,
        max_new_rounds=2,
    )

    assert command[command.index("--rounds") + 1] == "100"
    assert command[command.index("--max-new-rounds") + 1] == "2"
    assert "--trusted-edge-split-execution" in command


def _flow_client(client_id: int) -> ClientFlowInput:
    return ClientFlowInput(
        client_id=client_id,
        edge_id=0,
        mode="LIIEIIIC",
        candidate_time=1.0,
        estimated_local_time=1.0,
        measured_local_time=1.0,
        communication_volume=1.0,
        state_diff={},
        sample_count=1,
        edge_loops=3,
        edge_aggregation_payload=4.0,
        cloud_aggregation_payload=4.0,
    )


def _direct_cloud_client(client_id: int) -> ClientFlowInput:
    return ClientFlowInput(
        client_id=client_id,
        edge_id=0,
        mode="LIIC",
        candidate_time=1.0,
        estimated_local_time=1.0,
        measured_local_time=1.0,
        communication_volume=1.0,
        state_diff={},
        sample_count=1,
    )


def _lieiic_client(client_id: int) -> ClientFlowInput:
    return ClientFlowInput(
        client_id=client_id,
        edge_id=client_id,
        mode="LIEIIC",
        candidate_time=1.0,
        estimated_local_time=1.0,
        measured_local_time=1.0,
        communication_volume=1.0,
        state_diff={},
        sample_count=1,
    )


def _candidate(mode: str) -> Candidate:
    return Candidate(
        mode=mode,
        mechanisms={"upd": "dp"},
        time=1.0,
        accuracy=0.5,
        risk=0.1,
        epsilon_used=0.05,
        communication_volume=1.0,
        feasible_resource=True,
        feasible_privacy=True,
        feasible_risk=True,
        feasible_time=True,
    )


def test_multilevel_flow_uses_configured_edge_loop_count() -> None:
    result = execute_mixed_round_flow(
        round_idx=0,
        clients=[_flow_client(0), _flow_client(1)],
        aggregation_fraction=1.0,
        edge_aggregation_beta=0.01,
        edge_aggregation_fixed=0.02,
    )

    event = next(item for item in result.flow_events if item["event_type"] == "edge_aggregate")
    assert event["loop_factor"] == 3
    assert abs(result.edge_aggregation_time - 3 * (0.01 * 8.0 + 0.02)) < 1e-12
    assert event["effective_payload"] == 8.0


def test_direct_cloud_flow_counts_each_latency_stage_once() -> None:
    clients = [
        ClientFlowInput(
            client_id=client_id,
            edge_id=0,
            mode="LIIC",
            candidate_time=1.0,
            estimated_local_time=1.0,
            measured_local_time=0.0,
            communication_volume=4.0,
            state_diff={},
            sample_count=1,
            return_path_time=0.3,
            cloud_aggregation_payload=4.0,
        )
        for client_id in range(2)
    ]

    result = execute_mixed_round_flow(
        round_idx=0,
        clients=clients,
        aggregation_fraction=1.0,
        cloud_aggregation_beta=0.015,
        cloud_aggregation_fixed=0.04,
    )

    expected_cloud_agg = 0.015 * 8.0 + 0.04
    assert abs(result.cloud_aggregation_time - expected_cloud_agg) < 1e-12
    assert abs(result.return_time - 0.3) < 1e-12
    assert abs(result.round_duration - (1.0 + expected_cloud_agg + 0.3)) < 1e-12


def test_candidate_link_metric_matches_tex_link_formula() -> None:
    candidate = next(
        item
        for item in enumerate_candidates(
            config=SelectionConfig(),
            client_id=2,
            edge_factor=1.0,
            compute_factor=1.0,
            samples=100,
            remaining_epsilon=8.0,
            round_idx=3,
            rng=random.Random(19),
            policy="fixed_dp",
        )
        if item.mode == "LIEIIC"
    )

    for metric in candidate.link_metrics:
        expected = (
            metric["effective_size"] / metric["rate"]
            + metric["base_delay"]
            + metric["privacy_processing_time"]
        )
        assert abs(metric["per_execution_time"] - expected) < 1e-12
        assert abs(metric["total_link_time"] - metric["count"] * expected) < 1e-12
    expected_total = (
        candidate.first_aggregation_arrival_time
        + candidate.edge_to_cloud_time
        + SelectionConfig().cloud_aggregation_beta * candidate.cloud_aggregation_payload
        + SelectionConfig().cloud_aggregation_fixed
        + candidate.return_path_time
    )
    assert abs(candidate.time - expected_total) < 1e-12


def test_buffer_admits_all_clients_tied_at_trigger_time() -> None:
    result = execute_mixed_round_flow(
        round_idx=0,
        clients=[_direct_cloud_client(0), _direct_cloud_client(1), _direct_cloud_client(2)],
        aggregation_fraction=0.5,
    )

    assert result.selected_client_ids == [0, 1, 2]


def test_lieiic_uploads_directly_to_cloud_without_edge_preaggregation() -> None:
    result = execute_mixed_round_flow(
        round_idx=0,
        clients=[_lieiic_client(0), _lieiic_client(1)],
        aggregation_fraction=1.0,
    )

    event_types = [item["event_type"] for item in result.flow_events]
    assert "cloud_aggregate_direct" in event_types
    assert "edge_aggregate" not in event_types
    assert "cloud_aggregate_from_edges" not in event_types


def test_all_mode_topologies_use_the_tex_aggregation_endpoints() -> None:
    expected = {
        "LIE": {"edge_aggregate"},
        "LIIE": {"edge_aggregate"},
        "LIC": {"cloud_aggregate_direct"},
        "LIIC": {"cloud_aggregate_direct"},
        "LIEIIC": {"cloud_aggregate_direct"},
        "LIEIIIC": {"edge_aggregate", "cloud_aggregate_from_edges"},
        "LIIEIIIC": {"edge_aggregate", "cloud_aggregate_from_edges"},
    }
    aggregation_events = {
        "edge_aggregate",
        "cloud_aggregate_direct",
        "cloud_aggregate_from_edges",
    }

    for mode, expected_events in expected.items():
        client = ClientFlowInput(
            client_id=0,
            edge_id=0,
            mode=mode,
            candidate_time=1.0,
            estimated_local_time=1.0,
            measured_local_time=1.0,
            communication_volume=1.0,
            state_diff={},
            sample_count=1,
            edge_loops=3 if mode in {"LIEIIIC", "LIIEIIIC"} else 1,
        )
        result = execute_mixed_round_flow(
            round_idx=0,
            clients=[client],
            aggregation_fraction=1.0,
        )
        actual_events = {
            item["event_type"]
            for item in result.flow_events
            if item["event_type"] in aggregation_events
        }
        assert actual_events == expected_events


def test_summary_flow_matches_event_executor_when_all_buffers_fill() -> None:
    modes = ("LIE", "LIIE", "LIC", "LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC")
    clients = []
    for client_id in range(35):
        mode = modes[client_id % len(modes)]
        clients.append(
            ClientFlowInput(
                client_id=client_id,
                edge_id=client_id % 4,
                mode=mode,
                candidate_time=0.4 + 0.07 * (client_id % 9),
                estimated_local_time=0.3,
                measured_local_time=0.2,
                communication_volume=1.0 + client_id,
                state_diff={"client": client_id},
                sample_count=client_id + 1,
                edge_loops=3 if mode in {"LIEIIIC", "LIIEIIIC"} else 1,
                edge_to_cloud_time=0.05 * (client_id % 3),
                return_path_time=0.03 * (client_id % 5),
                edge_aggregation_payload=0.5 + 0.1 * (client_id % 4),
                cloud_aggregation_payload=0.8 + 0.1 * (client_id % 6),
                aggregation_group="he3" if client_id % 2 else "dp",
                dispatch_start_time=0.02 * (client_id % 3),
                dispatch_sequence=client_id,
            )
        )

    detailed = execute_mixed_round_flow(
        round_idx=4,
        clients=clients,
        aggregation_fraction=1.0,
        edge_aggregation_beta=0.013,
        edge_aggregation_fixed=0.027,
        cloud_aggregation_beta=0.019,
        cloud_aggregation_fixed=0.041,
    )
    summary = summarize_mixed_round_flow(
        round_idx=4,
        clients=clients,
        aggregation_fraction=1.0,
        edge_aggregation_beta=0.013,
        edge_aggregation_fixed=0.027,
        cloud_aggregation_beta=0.019,
        cloud_aggregation_fixed=0.041,
    )

    assert summary.selected_client_ids == detailed.selected_client_ids
    assert summary.state_diffs == detailed.state_diffs
    assert summary.sample_counts == detailed.sample_counts
    assert summary.num_effective_edges == detailed.num_effective_edges
    assert summary.flow_events == []
    np.testing.assert_allclose(
        [
            summary.round_duration,
            summary.waiting_time,
            summary.edge_aggregation_time,
            summary.cloud_aggregation_time,
            summary.return_time,
        ],
        [
            detailed.round_duration,
            detailed.waiting_time,
            detailed.edge_aggregation_time,
            detailed.cloud_aggregation_time,
            detailed.return_time,
        ],
        rtol=1e-12,
        atol=1e-12,
    )


def test_summary_flow_falls_back_for_partial_buffers() -> None:
    clients = [_direct_cloud_client(client_id) for client_id in range(5)]

    detailed = execute_mixed_round_flow(
        round_idx=2,
        clients=clients,
        aggregation_fraction=0.5,
    )
    summary = summarize_mixed_round_flow(
        round_idx=2,
        clients=clients,
        aggregation_fraction=0.5,
    )

    assert summary == detailed
    assert summary.flow_events


def test_incremental_full_buffer_latency_matches_event_executor() -> None:
    config = SelectionConfig(
        aggregation_fraction=1.0,
        edge_aggregation_beta=0.013,
        edge_aggregation_fixed=0.027,
        cloud_aggregation_beta=0.019,
        cloud_aggregation_fixed=0.041,
    )
    modes = ("LIE", "LIIE", "LIC", "LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC")
    pools = {}
    profile = {}
    for client_id in range(14):
        candidates = []
        for offset in range(3):
            mode = modes[(client_id + offset) % len(modes)]
            candidates.append(
                Candidate(
                    **{
                        **_candidate(mode).__dict__,
                        "time": 0.5 + 0.09 * ((client_id + offset) % 8),
                        "edge_to_cloud_time": 0.04 * ((client_id + offset) % 3),
                        "return_path_time": 0.02 * ((client_id + offset) % 5),
                        "edge_aggregation_payload": 0.6 + 0.1 * offset,
                        "cloud_aggregation_payload": 0.9 + 0.2 * offset,
                        "link_mechanisms": {"E_C_upd": "he3" if offset % 2 else "dp"},
                    }
                )
            )
        pools[client_id] = candidates
        profile[client_id] = candidates[0]

    samples = {client_id: float(client_id + 3) for client_id in profile}
    edges = {client_id: client_id % 4 for client_id in profile}
    client_order = tuple(sorted(profile))
    flow_inputs = _profile_flow_inputs_by_candidate(
        config,
        pools,
        samples,
        edges,
        previous_choices={},
        client_order=client_order,
    )
    stats = _full_buffer_flow_stats(config, profile, flow_inputs)
    assert stats is not None

    for client_id in client_order:
        for candidate in pools[client_id][1:]:
            incremental_ids, incremental_latency = _replace_full_buffer_flow_objectives(
                config,
                stats,
                client_id,
                flow_inputs[client_id][
                    (candidate.mode, tuple(sorted(candidate.link_mechanisms.items())))
                ],
            )
            replaced = dict(profile)
            replaced[client_id] = candidate
            clients = [
                flow_inputs[replaced_id][
                    (
                        replaced_candidate.mode,
                        tuple(sorted(replaced_candidate.link_mechanisms.items())),
                    )
                ]
                for replaced_id, replaced_candidate in sorted(replaced.items())
            ]
            detailed = execute_mixed_round_flow(
                round_idx=0,
                clients=clients,
                aggregation_fraction=config.aggregation_fraction,
                edge_aggregation_beta=config.edge_aggregation_beta,
                edge_aggregation_fixed=config.edge_aggregation_fixed,
                cloud_aggregation_beta=config.cloud_aggregation_beta,
                cloud_aggregation_fixed=config.cloud_aggregation_fixed,
            )

            assert incremental_ids == tuple(detailed.selected_client_ids)
            assert abs(incremental_latency - detailed.round_duration) < 1e-12


def test_proposed_policy_does_not_offer_unprotected_private_links() -> None:
    candidates = enumerate_candidates(
        config=SelectionConfig(),
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=100,
        remaining_epsilon=4.0,
        round_idx=0,
        rng=random.Random(7),
        policy="ours",
    )

    for candidate in candidates:
        for obj, mechanism in candidate.mechanisms.items():
            if obj != "label":
                assert mechanism != "none"


def test_random_policy_does_not_offer_unprotected_private_links() -> None:
    candidates = enumerate_candidates(
        config=SelectionConfig(),
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=100,
        remaining_epsilon=4.0,
        round_idx=0,
        rng=random.Random(7),
        policy="random",
    )

    for candidate in candidates:
        for obj, mechanism in candidate.mechanisms.items():
            if obj != "label":
                assert mechanism != "none"


def test_split_modes_keep_labels_local() -> None:
    candidates = enumerate_candidates(
        config=SelectionConfig(risk_limit=0.5),
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=100,
        remaining_epsilon=4.0,
        round_idx=0,
        rng=random.Random(7),
        policy="ours",
    )

    for mode in {"LIE", "LIC", "LIEIIC", "LIEIIIC"}:
        mode_candidates = [candidate for candidate in candidates if candidate.mode == mode]
        assert mode_candidates
        assert all("label" not in candidate.mechanisms for candidate in mode_candidates)
        assert all(candidate.feasible_risk for candidate in mode_candidates)
        objects = {obj for obj, _count, _eligible in _mode_link_events(mode, 5, 3)}
        assert "label" not in objects
        assert {"emb", "logits", "grad", "emb_grad"}.issubset(objects)


def test_output_gradient_dp_releases_clipped_batch_average() -> None:
    tensor = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    actual = _protect_batched_average_gradient_dp(
        tensor,
        mechanism="none",
        clip_norm=1.0,
        noise_multiplier=1.0,
        rng=np.random.default_rng(1),
        device=torch.device("cpu"),
    )
    expected = torch.tensor([[0.3, 0.4], [0.0, 0.5]])
    torch.testing.assert_close(actual, expected)


def test_record_privacy_horizon_uses_epochs_not_batch_link_count() -> None:
    resolved = resolved_privacy_parameters(
        SelectionConfig(rounds=200, L_block_cycles=5, privacy_local_epochs=3)
    )
    assert resolved["max_feature_events_per_round"] == 18
    assert resolved["feature_horizon_events"] == 3600


def test_trusted_edge_splitfed_uses_he_without_feature_dp() -> None:
    candidates = enumerate_candidates(
        config=SelectionConfig(risk_limit=0.5),
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=100,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(19),
        policy="fixed_splitfed_trusted_edge",
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.mode == "LIEIIC"
    assert candidate.link_mechanisms == {
        "L_E_emb": "trusted",
        "L_E_grad": "trusted",
        "E_C_upd": "he3",
    }
    assert candidate.feature_dp_events == 0
    assert candidate.update_dp_events == 0
    assert candidate.feasible_risk


def test_v25_trusted_edge_boundary_applies_to_all_policies() -> None:
    config = SelectionConfig(
        rounds=200,
        risk_limit=0.5,
        trusted_edge_split_execution=True,
    )
    candidates = enumerate_candidates(
        config=config,
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=100,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(29),
        policy="ours",
    )

    assert candidates
    assert all(candidate.mode != "LIC" for candidate in candidates)
    assert all(candidate.feature_dp_events == 0 for candidate in candidates)
    assert all(
        mechanism == "trusted"
        for candidate in candidates
        for link, mechanism in (candidate.link_mechanisms or {}).items()
        if link.startswith("L_E_")
    )
    edge_split = [
        candidate
        for candidate in candidates
        if candidate.mode in {"LIE", "LIEIIC", "LIEIIIC"}
    ]
    assert edge_split
    assert all(
        candidate.link_mechanisms.get("L_E_emb") == "trusted"
        and candidate.link_mechanisms.get("L_E_grad") == "trusted"
        for candidate in edge_split
    )

    resolved = resolved_privacy_parameters(config)
    assert resolved["feature_dp_enabled"] is False
    assert resolved["feature_horizon_events"] == 0
    assert resolved["max_update_events_per_round"] == 1
    assert resolved["update_horizon_events"] == 200


def test_v25_fixed_splitfed_uses_shared_trusted_edge_boundary() -> None:
    candidates = enumerate_candidates(
        config=SelectionConfig(
            rounds=200,
            risk_limit=0.5,
            trusted_edge_split_execution=True,
        ),
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=100,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(31),
        policy="fixed_splitfed",
    )

    assert candidates
    assert {candidate.mode for candidate in candidates} == {"LIEIIC"}
    assert all(candidate.link_mechanisms["L_E_emb"] == "trusted" for candidate in candidates)
    assert all(candidate.link_mechanisms["L_E_grad"] == "trusted" for candidate in candidates)
    assert {
        candidate.link_mechanisms["E_C_upd"] for candidate in candidates
    } == {"dp", "he3"}


def test_fixed_splitfed_dp_diagnostic_forces_aggregate_dp_boundary() -> None:
    candidates = enumerate_candidates(
        config=SelectionConfig(
            rounds=100,
            risk_limit=0.5,
            trusted_edge_split_execution=True,
        ),
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=100,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(37),
        policy="fixed_splitfed_dp",
    )

    assert len(candidates) == 1
    assert candidates[0].mode == "LIEIIC"
    assert candidates[0].link_mechanisms == {
        "L_E_emb": "trusted",
        "L_E_grad": "trusted",
        "E_C_upd": "dp",
    }
    assert candidates[0].update_dp_events == 1


def test_trusted_edge_aggregate_dp_is_not_applied_inside_client_training() -> None:
    candidate = Candidate(
        **{
            **_candidate("LIEIIC").__dict__,
            "mechanisms": {"upd": "he3_dp"},
            "link_mechanisms": {
                "L_E_emb": "trusted",
                "L_E_grad": "trusted",
                "E_C_upd": "he3_dp",
            },
        }
    )

    assert _candidate_uses_aggregate_update_dp(candidate)
    assert not _should_apply_update_dp(
        _candidate_training_mechanisms(candidate),
        "upd_only",
        candidate.mode,
    )

    dp_candidate = Candidate(
        **{
            **candidate.__dict__,
            "mechanisms": {"upd": "dp"},
            "link_mechanisms": {
                **candidate.link_mechanisms,
                "E_C_upd": "dp",
            },
        }
    )
    assert _candidate_uses_aggregate_update_dp(dp_candidate, True)
    training_mechanisms = _candidate_training_mechanisms(
        dp_candidate,
        aggregate_cloud_update_dp=True,
    )
    assert training_mechanisms["upd"] == "none"
    assert not _should_apply_update_dp(training_mechanisms, "upd_only", dp_candidate.mode)


def test_aggregate_dp_uses_max_normalized_client_weight() -> None:
    max_weight, sensitivity, noise_std = _aggregate_dp_release_parameters(
        [200.0, 600.0, 800.0],
        [1.0, 1.0, 0.0],
        clip_norm=2.0,
        noise_multiplier=3.0,
    )

    assert np.isclose(max_weight, 0.375)
    assert np.isclose(sensitivity, 1.5)
    assert np.isclose(noise_std, 4.5)


def test_aggregate_dp_clips_one_complete_client_contribution() -> None:
    diff = {
        "end": {"a": torch.tensor([3.0, 4.0])},
        "edge": {"b": torch.tensor([0.0, 12.0])},
    }

    clipped, original_norm, scale = clip_state_difference(
        diff,
        clip_norm=6.5,
        device=torch.device("cpu"),
    )

    assert np.isclose(original_norm, 13.0)
    assert np.isclose(scale, 0.5)
    flat = torch.cat([clipped["end"]["a"], clipped["edge"]["b"]])
    torch.testing.assert_close(torch.linalg.vector_norm(flat), torch.tensor(6.5))


def test_feature_dp_noise_uses_replacement_sensitivity() -> None:
    seed = 23
    expected_rng = np.random.default_rng(seed)
    expected = expected_rng.normal(0.0, 1.0, size=(1, 2)).astype(np.float32)
    actual = _protect_tensor_dp(
        torch.zeros((1, 2)),
        mechanism="dp",
        clip_norm=1.0,
        noise_multiplier=0.5,
        rng=np.random.default_rng(seed),
        device=torch.device("cpu"),
        epsilon=0.03,
    )

    torch.testing.assert_close(actual, torch.from_numpy(expected))


def test_feature_clip_profile_uses_actual_tex_excess() -> None:
    class Flatten(torch.nn.Module):
        def forward(self, value: torch.Tensor) -> torch.Tensor:
            return value.reshape(value.shape[0], -1)

    profiles = _feature_clip_excess_sq_by_client(
        Flatten(),
        np.asarray([[[[3.0, 4.0]]], [[[0.0, 1.0]]], [[[0.0, 3.0]]]], dtype=np.float32),
        [np.asarray([0, 1]), np.asarray([2])],
        clip_norm=1.0,
        device=torch.device("cpu"),
        input_shape=(1, 1, 2),
    )

    assert profiles[0] == 8.0
    assert profiles[1] == 4.0


def test_profiled_feature_clip_excess_changes_local_omega() -> None:
    config = SelectionConfig(omega_feature_clip_excess_sq=0.02)
    fallback = Candidate(
        **{
            **_candidate("LIC").__dict__,
            "mechanisms": {"emb": "dp", "grad": "dp", "label": "none"},
        }
    )
    profiled = Candidate(
        **{
            **fallback.__dict__,
            "omega_feature_clip_excess_sq": 4.0,
        }
    )

    assert _local_omega_proxy(profiled, config=config) > _local_omega_proxy(fallback, config=config) + 5.0


def test_mode_link_events_share_counts_between_budget_and_communication() -> None:
    config = SelectionConfig(dp_upd_epsilon=0.05, L_block_cycles=5)
    candidates = enumerate_candidates(
        config=config,
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=100,
        remaining_epsilon=4.0,
        round_idx=0,
        rng=random.Random(11),
        policy="fixed_dp",
    )
    candidate = next(item for item in candidates if item.mode == "LIIEIIIC")

    assert _mode_link_events("LIIEIIIC", 5, 3) == (
        ("upd", 3, True),
        ("upd", 3, False),
        ("upd", 1, True),
        ("upd", 1, False),
        ("upd", 1, False),
    )
    assert abs(candidate.epsilon_used - 4 * 0.05) < 1e-12
    expected_volume = (4 * PRIVACY_ALPHA["dp"] + 5) * OBJECT_SIZES["upd"]
    assert abs(candidate.communication_volume - expected_volume) < 1e-12


def test_dp_event_counts_match_every_tex_mode() -> None:
    expected = {
        "LIE": {"emb": 5, "grad": 5, "upd": 0},
        "LIC": {"emb": 5, "grad": 5, "upd": 0},
        "LIIE": {"emb": 0, "grad": 0, "upd": 1},
        "LIIC": {"emb": 0, "grad": 0, "upd": 1},
        "LIEIIC": {"emb": 5, "grad": 5, "upd": 1},
        "LIEIIIC": {"emb": 15, "grad": 15, "upd": 1},
        "LIIEIIIC": {"emb": 0, "grad": 0, "upd": 4},
    }

    for mode, counts in expected.items():
        events = _mode_link_events(mode, 5, 3)
        actual = {
            obj: sum(
                count
                for event_obj, count, privacy_eligible in events
                if event_obj == obj and privacy_eligible
            )
            for obj in counts
        }
        assert actual == counts


def test_liieiiic_update_upload_links_select_mechanisms_independently() -> None:
    candidates = enumerate_candidates(
        config=SelectionConfig(allow_he=True),
        client_id=3,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=100,
        remaining_epsilon=8.0,
        round_idx=2,
        rng=random.Random(17),
        policy="ours",
    )
    mode_candidates = [item for item in candidates if item.mode == "LIIEIIIC"]
    assignments = {
        (
            item.link_mechanisms["L_E_upd"],
            item.link_mechanisms["E_C_upd"],
        )
        for item in mode_candidates
    }

    assert assignments == {
        ("dp", "dp"),
        ("dp", "he3"),
        ("he3", "dp"),
        ("he3", "he3"),
    }
    mixed = next(
        item
        for item in mode_candidates
        if item.link_mechanisms["L_E_upd"] == "dp"
        and item.link_mechanisms["E_C_upd"] == "he3"
    )
    assert _candidate_training_mechanisms(mixed)["upd"] == "dp"
    assert _candidate_cloud_update_mechanism(mixed) == "he3"
    assert mixed.update_dp_events == 3


def test_link_bandwidth_is_shared_across_candidates_in_one_round() -> None:
    config = SelectionConfig(seed=23, network_jitter=0.25)

    first = _link_bandwidth(config, client_id=4, round_idx=7, link_id="L_E_emb")
    second = _link_bandwidth(config, client_id=4, round_idx=7, link_id="L_E_upd")

    assert first == second
    assert first != _link_bandwidth(config, client_id=5, round_idx=7, link_id="L_E_emb")


def test_profiled_memory_capacity_filters_full_local_modes() -> None:
    candidates = enumerate_candidates(
        config=SelectionConfig(memory_limit=0.7),
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        memory_capacity_factor=1.0,
        samples=100,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(5),
        policy="fixed_dp",
    )

    assert all(item.feasible_memory for item in candidates if item.mode == "LIE")
    assert all(not item.feasible_memory for item in candidates if item.mode == "LIIE")


def test_local_block_iterator_respects_epoch_and_step_limits() -> None:
    dataset = torch.utils.data.TensorDataset(torch.arange(3), torch.arange(3))
    loader = torch.utils.data.DataLoader(dataset, batch_size=2, shuffle=False)

    assert len(list(_training_batches(loader, epochs=1, local_steps=5))) == 2
    assert len(list(_training_batches(loader, epochs=3, local_steps=5))) == 5


def test_extreme_edge_partition_uses_every_sample_once_across_all_clients() -> None:
    labels = np.repeat(np.arange(10, dtype=np.int64), 100)

    parts = _partition_clients_lenet5(
        labels,
        num_clients=100,
        num_edges=10,
        iid=False,
        partition_mode="extreme_edge_label_skew",
        seed=42,
    )

    assigned = np.concatenate(parts)
    assert all(len(part) == 10 for part in parts)
    np.testing.assert_array_equal(np.sort(assigned), np.arange(len(labels)))
    for client_id, part in enumerate(parts):
        assert set(labels[part].tolist()) == {client_id % 10}


def test_client_training_seed_is_stable_and_stage_specific() -> None:
    first = _client_training_seed(42, 3, 7, 0)

    assert first == _client_training_seed(42, 3, 7, 0)
    assert first != _client_training_seed(42, 4, 7, 0)
    assert first != _client_training_seed(42, 3, 8, 0)
    assert first != _client_training_seed(42, 3, 7, 1)


def test_dp_noise_seed_supports_stable_composite_edge_group_ids() -> None:
    group = (3, "LIEIIC", "dp")
    first = _dp_noise_seed(42, 3, group, 10_000)
    assert first == _dp_noise_seed(42, 3, group, 10_000)
    assert first != _dp_noise_seed(42, 3, (3, "LIEIIC", "he"), 10_000)


def test_local_training_is_reproducible_with_explicit_seed() -> None:
    device = torch.device("cpu")
    torch.manual_seed(17)
    global_end, global_edge, _full = build_split_models("lenet5", device)
    x = np.random.default_rng(31).normal(size=(80, 28 * 28)).astype(np.float32)
    y = np.arange(80, dtype=np.int64) % 10
    kwargs = {
        "mode": "LIIC",
        "global_end_state": global_end.state_dict(),
        "global_edge_state": global_edge.state_dict(),
        "x": x,
        "y": y,
        "epochs": 1,
        "lr": 0.01,
        "device": device,
        "model_name": "lenet5",
        "mechanisms": {"upd": "none"},
        "local_steps": 2,
        "training_seed": 991,
    }

    model_cache = {}
    first = split_local_train_lenet5(**kwargs, model_cache=model_cache)
    second = split_local_train_lenet5(**kwargs, model_cache=model_cache)

    for part_name in first:
        for name in first[part_name]:
            torch.testing.assert_close(first[part_name][name], second[part_name][name])


def test_update_dp_is_applied_at_the_correct_multilevel_stage() -> None:
    mechanisms = {"upd": "dp"}

    assert _should_apply_update_dp(mechanisms, "upd_only", "LIEIIC")
    assert not _should_apply_update_dp(mechanisms, "upd_only", "LIEIIIC")
    assert _should_apply_update_dp(mechanisms, "upd_only", "LIIEIIIC")


def test_logical_admission_is_independent_of_serial_wall_time() -> None:
    fast_logical = ClientFlowInput(
        client_id=0,
        edge_id=0,
        mode="LIIC",
        candidate_time=1.0,
        estimated_local_time=1.0,
        measured_local_time=100.0,
        communication_volume=1.0,
        state_diff={},
        sample_count=1,
    )
    slow_logical = ClientFlowInput(
        client_id=1,
        edge_id=0,
        mode="LIIC",
        candidate_time=2.0,
        estimated_local_time=1.0,
        measured_local_time=0.001,
        communication_volume=1.0,
        state_diff={},
        sample_count=1,
    )

    result = execute_mixed_round_flow(
        round_idx=0,
        clients=[fast_logical, slow_logical],
        aggregation_fraction=0.5,
    )

    assert result.selected_client_ids == [0]


def test_update_dp_defaults_to_one_global_l2_clip() -> None:
    diff = {
        "end": {"a": torch.tensor([3.0, 4.0])},
        "edge": {"b": torch.tensor([0.0, 12.0])},
    }

    protected = apply_unified_dp(
        diff,
        mechanism="dp",
        clip_norm=6.5,
        noise_multiplier=0.0,
        rng=np.random.default_rng(7),
        device=torch.device("cpu"),
    )

    flat = torch.cat([protected["end"]["a"], protected["edge"]["b"]])
    torch.testing.assert_close(torch.linalg.vector_norm(flat), torch.tensor(6.5))
    torch.testing.assert_close(protected["end"]["a"], torch.tensor([1.5, 2.0]))


def test_update_dp_noise_uses_replacement_sensitivity() -> None:
    class RecordingRng:
        def __init__(self) -> None:
            self.scale = 0.0

        def normal(self, _loc, scale, size):
            self.scale = float(scale)
            return np.zeros(size, dtype=np.float32)

    rng = RecordingRng()
    apply_unified_dp(
        {"end": {"a": torch.tensor([1.0])}, "edge": {}},
        mechanism="dp",
        clip_norm=3.0,
        noise_multiplier=2.0,
        rng=rng,
        device=torch.device("cpu"),
    )

    assert rng.scale == 12.0


def test_multilevel_dp_events_distinguish_client_and_edge_releases() -> None:
    candidate = Candidate(
        **{
            **_candidate("LIIEIIIC").__dict__,
            "link_mechanisms": {"L_E_upd": "dp", "E_C_upd": "dp"},
        }
    )

    feature_events, client_events, edge_events = _candidate_dp_event_counts(
        candidate,
        SelectionConfig(L_block_cycles=5),
    )

    assert feature_events == 0
    assert client_events == 3
    assert edge_events == 1


def test_combined_he_dp_mechanism_consumes_update_budget() -> None:
    candidate = Candidate(
        **{
            **_candidate("LIEIIC").__dict__,
            "link_mechanisms": {"E_C_upd": "he3_dp"},
        }
    )

    feature_events, client_events, edge_events = _candidate_dp_event_counts(
        candidate,
        SelectionConfig(trusted_edge_split_execution=True),
    )

    assert feature_events == 0
    assert client_events == 1
    assert edge_events == 0


def test_update_dp_variance_uses_admitted_aggregation_size() -> None:
    profile = {
        0: _candidate("LIIC"),
        1: _candidate("LIIC"),
        2: _candidate("LIIE"),
        3: _candidate("LIIE"),
    }

    sizes = _aggregation_sizes(
        profile,
        client_edges={0: 0, 1: 0, 2: 1, 3: 1},
        admitted_client_ids=[0, 1, 2],
    )

    assert sizes == {0: 2, 1: 2, 2: 1, 3: 1}


def test_selector_cloud_weights_match_actual_edge_normalized_fedavg() -> None:
    cloud = _candidate("LIIC")
    edge_only = _candidate("LIIE")
    profile = {0: cloud, 1: edge_only, 2: cloud}
    samples = {0: 10.0, 1: 30.0, 2: 60.0}
    edges = {0: 0, 1: 0, 2: 1}

    selector_weights = _cloud_client_aggregation_weights(
        profile,
        samples,
        edges,
        admitted_client_ids=[0, 2],
    )
    training_weights = _edge_normalized_cloud_weights(
        [({}, 10, cloud, [0]), ({}, 60, cloud, [2])],
        client_edges=edges,
        edge_total_samples={0: 40.0, 1: 60.0},
    )
    training_total = sum(training_weights)

    np.testing.assert_allclose(
        [selector_weights[0], selector_weights[2]],
        [weight / training_total for weight in training_weights],
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(sum(selector_weights.values()), 1.0)


def test_global_omega_excludes_dropped_cloud_clients() -> None:
    profile = {client_id: _candidate("LIIC") for client_id in range(3)}
    samples = {0: 10.0, 1: 30.0, 2: 60.0}
    edges = {0: 0, 1: 0, 2: 1}
    config = SelectionConfig(aggregation_fraction=0.5)

    _omega, cloud_ratio = _global_omega_proxy(
        config,
        profile,
        samples,
        edges,
        admitted_client_ids=[0, 2],
    )

    np.testing.assert_allclose(cloud_ratio, 0.7, rtol=1e-12, atol=1e-12)


def test_edge_fedavg_returns_average_of_client_model_states() -> None:
    device = torch.device("cpu")
    global_end, global_edge, _full = build_split_models("lenet5", device)
    state0 = {
        "end": {key: value.detach().clone() for key, value in global_end.state_dict().items()},
        "edge": {key: value.detach().clone() for key, value in global_edge.state_dict().items()},
    }
    state1 = {
        "end": {key: value.detach().clone() for key, value in global_end.state_dict().items()},
        "edge": {key: value.detach().clone() for key, value in global_edge.state_dict().items()},
    }
    first_name = next(iter(dict(global_end.named_parameters())))
    state1["end"][first_name] += 2.0
    zero_diff = {
        "end": {name: torch.zeros_like(param) for name, param in global_end.named_parameters()},
        "edge": {name: torch.zeros_like(param) for name, param in global_edge.named_parameters()},
    }

    returned, used_he = _aggregate_returned_client_models(
        updates=[
            (0, zero_diff, 1, _candidate("LIIE")),
            (1, zero_diff, 1, _candidate("LIIE")),
        ],
        client_model_states={0: state0, 1: state1},
        global_end=global_end,
        global_edge=global_edge,
        device=device,
        model_name="lenet5",
        input_shape=(1, 28, 28),
        num_classes=10,
        he_backend="none",
    )

    assert not used_he
    torch.testing.assert_close(
        returned["end"][first_name],
        state0["end"][first_name] + 1.0,
    )


def test_direct_cloud_update_is_rebased_to_current_global_model() -> None:
    device = torch.device("cpu")
    global_end, global_edge, _full = build_split_models("lenet5", device)
    client_state = {
        "end": {key: value.detach().clone() for key, value in global_end.state_dict().items()},
        "edge": {key: value.detach().clone() for key, value in global_edge.state_dict().items()},
    }
    first_name = next(iter(dict(global_end.named_parameters())))
    client_state["end"][first_name] += 2.0
    local_diff = {
        "end": {name: torch.zeros_like(param) for name, param in global_end.named_parameters()},
        "edge": {name: torch.zeros_like(param) for name, param in global_edge.named_parameters()},
    }
    local_diff["end"][first_name] += 1.0

    rebased = _state_difference_from_client_update(
        client_id=7,
        state_diff=local_diff,
        client_model_states={7: client_state},
        global_end=global_end,
        global_edge=global_edge,
        device=device,
    )

    torch.testing.assert_close(
        rebased["end"][first_name],
        torch.full_like(rebased["end"][first_name], 3.0),
    )


def test_weighted_state_difference_norm_matches_fedavg_update() -> None:
    first = {
        "end": {"weight": torch.tensor([3.0, 4.0])},
        "edge": {},
    }
    second = {
        "end": {"weight": torch.tensor([0.0, 2.0])},
        "edge": {},
    }

    actual = _weighted_state_difference_norm([first, second], [1, 3])
    expected = float(np.sqrt(0.75 ** 2 + 2.5 ** 2))

    assert abs(actual - expected) < 1e-12


def test_edge_normalized_cloud_weights_remove_round_varying_edge_share() -> None:
    cloud_updates = [
        (None, 25, None, [0]),
        (None, 75, None, [1]),
        (None, 50, None, [2]),
    ]

    weights = _edge_normalized_cloud_weights(
        cloud_updates,
        client_edges={0: 0, 1: 0, 2: 1},
        edge_total_samples={0: 800.0, 1: 800.0},
    )

    assert np.allclose(weights, [200.0, 600.0, 800.0])
    assert np.isclose(sum(weights[:2]), weights[2])


def test_edge_normalized_cloud_weights_preserve_edge_dataset_mass() -> None:
    cloud_updates = [
        (None, 100, None, [0, 1]),
        (None, 100, None, [2]),
    ]

    weights = _edge_normalized_cloud_weights(
        cloud_updates,
        client_edges={0: 0, 1: 0, 2: 1},
        edge_total_samples={0: 1200.0, 1: 600.0},
    )

    assert np.allclose(weights, [1200.0, 600.0])


def test_cloud_mode_synchronizes_global_while_edge_mode_keeps_returned_state() -> None:
    device = torch.device("cpu")
    global_end, global_edge, _full = build_split_models("lenet5", device)
    returned_state = {
        "end": {key: value.detach().clone() for key, value in global_end.state_dict().items()},
        "edge": {key: value.detach().clone() for key, value in global_edge.state_dict().items()},
    }
    first_name = next(iter(dict(global_end.named_parameters())))
    returned_state["end"][first_name] += 2.0
    client_states = {3: returned_state}

    cloud_base = _training_base_state(
        client_id=3,
        candidate=_candidate("LIIC"),
        client_model_states=client_states,
        global_end=global_end,
        global_edge=global_edge,
        device=device,
    )
    edge_base = _training_base_state(
        client_id=3,
        candidate=_candidate("LIIE"),
        client_model_states=client_states,
        global_end=global_end,
        global_edge=global_edge,
        device=device,
    )
    continued_edge_base = _training_base_state(
        client_id=3,
        candidate=_candidate("LIIEIIIC"),
        client_model_states=client_states,
        global_end=global_end,
        global_edge=global_edge,
        device=device,
        training_stage=1,
    )

    torch.testing.assert_close(cloud_base["end"][first_name], global_end.state_dict()[first_name])
    torch.testing.assert_close(edge_base["end"][first_name], returned_state["end"][first_name])
    torch.testing.assert_close(
        continued_edge_base["end"][first_name],
        returned_state["end"][first_name],
    )


def test_selector_profiles_use_actual_partition_sample_counts() -> None:
    profiles = [
        ClientProfile(client_id=0, edge_id=0, samples=200, compute_factor=1.0),
        ClientProfile(client_id=1, edge_id=1, samples=80, compute_factor=1.5),
    ]

    updated = _profiles_with_actual_samples(
        profiles,
        [np.arange(3, dtype=np.int64), np.arange(17, dtype=np.int64)],
    )

    assert [profile.samples for profile in updated] == [3, 17]
    assert [profile.compute_factor for profile in updated] == [1.0, 1.5]


def test_pareto_archive_scan_matches_exhaustive_dominance() -> None:
    rng = np.random.default_rng(19)
    evaluations = [
        ProfileEvaluation(
            profile={0: _candidate(f"profile_{idx}")},
            system_latency=float(rng.integers(0, 20)),
            system_omega=float(rng.integers(0, 20)),
            cloud_fusion_ratio=0.0,
        )
        for idx in range(250)
    ]

    expected = {
        next(iter(item.profile.values())).mode
        for item in evaluations
        if not any(
            other is not item
            and other.system_latency <= item.system_latency
            and other.system_omega <= item.system_omega
            and (
                other.system_latency < item.system_latency
                or other.system_omega < item.system_omega
            )
            for other in evaluations
        )
    }
    actual = {
        next(iter(item.profile.values())).mode
        for item in _pareto_archive(evaluations, limit=len(evaluations))
    }

    assert actual == expected


def test_incremental_profile_omega_matches_full_formula() -> None:
    config = SelectionConfig(aggregation_fraction=1.0)
    profile = {
        0: _candidate("LIIC"),
        1: _candidate("LIIE"),
        2: _candidate("LIIE"),
        3: _candidate("LIIC"),
    }
    samples = {0: 2.0, 1: 3.0, 2: 5.0, 3: 7.0}
    edges = {0: 0, 1: 0, 2: 1, 3: 1}
    stats = _profile_omega_stats(config, profile, samples, edges)

    replacement = Candidate(
        **{
            **_candidate("LIIE").__dict__,
            "mechanisms": {"upd": "he3"},
        }
    )
    replaced_stats = _replace_profile_omega_stats(
        config,
        stats,
        client_id=0,
        old_candidate=profile[0],
        new_candidate=replacement,
        client_samples=samples,
        client_edges=edges,
    )
    replaced_profile = dict(profile)
    replaced_profile[0] = replacement
    admitted = _admitted_clients_for_profile(
        config,
        replaced_profile,
        edges,
        previous_choices={},
    )

    incremental = _omega_from_profile_stats(
        config,
        replaced_stats,
        replaced_profile,
        edges,
        admitted,
    )
    complete = _global_omega_proxy(
        config,
        replaced_profile,
        samples,
        edges,
        admitted_client_ids=admitted,
    )

    np.testing.assert_allclose(incremental, complete, rtol=1e-12, atol=1e-12)


def test_paper_search_defaults_are_explicit() -> None:
    config = SelectionConfig()

    assert config.rounds == 100
    assert config.initial_epsilon == 8.0
    assert config.aggregation_fraction == 1.0
    assert config.dp_accounting_mode == "rdp_auto"
    assert config.dp_delta == 1e-5
    assert config.dp_noise_multiplier == 0.0002
    assert config.omega_feature_clip_norm == 1.0
    assert config.omega_update_clip_norm == 1.0
    assert config.pareto_archive_size == 16
    assert config.pareto_max_iters == 50
    assert config.pareto_neighbor_top_k == 0
    assert not config.pareto_conflict_only
    assert config.cloud_fusion_xi == 0.2
    assert config.cloud_fusion_eps == 0.05
    assert OBJECT_SIZES["emb"] == 1.6
    assert OBJECT_SIZES["upd"] == 4.0
    assert PRIVACY_ALPHA["he3"] == 8.04
