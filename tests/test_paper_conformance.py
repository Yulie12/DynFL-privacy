from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from dynfed.flow_executor import (
    ClientFlowInput,
    execute_mixed_round_flow,
    summarize_mixed_round_flow,
)
from dynfed.fmnist_lenet5_dynamic import (
    _candidate_cloud_update_mechanism,
    _apply_flat_update,
    _candidate_training_mechanisms,
    _candidate_uses_local_packet_update_dp,
    _candidate_uses_cross_domain_update_dp,
    _candidate_uses_secure_aggregate_update_dp,
    _distributed_aggregate_dp_parameters,
    _dp_update_release_parameters,
    _should_apply_update_dp,
    _aggregate_returned_client_models,
    _state_difference_from_client_update,
    _state_difference_from_model,
    _training_base_state,
    _profiles_with_actual_samples,
    _feature_clip_excess_sq_by_client,
    _client_training_seed,
    _dp_noise_seed,
    _edge_normalized_cloud_weights,
    _weighted_state_difference_norm,
    _partition_clients_lenet5,
    _deadline_satisfaction_ratio,
)
from dynfed.nodes import ClientProfile
from dynfed.selection import (
    Candidate,
    ExposurePrivacyRequirement,
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
    build_client_privacy_ledger,
    resolved_privacy_parameters,
)
from dynfed.split_learning import (
    _protect_batched_average_gradient_dp,
    _protect_tensor_dp,
    _training_batches,
    apply_unified_dp,
    build_split_models,
    clip_state_difference,
    fedavg_split,
    gaussian_state_difference,
    split_local_train_lenet5,
)
from dynfed.privacy import OBJECT_SIZES, PRIVACY_ALPHA
from experiments.run_paper_config import build_command


def test_paper_smoke_limit_does_not_change_rdp_round_horizon() -> None:
    config_path = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "paper_v30_cifar10_resnet18.json"
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
    assert "--trusted-edge-split-execution" not in command
    assert "--dp-emb-epsilon" not in command
    assert "--dp-feature-epsilon-budget" not in command


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
    assert "cloud_aggregate_round" in event_types
    assert "edge_aggregate" not in event_types
    assert "cloud_aggregate_direct" not in event_types
    assert "cloud_aggregate_from_edges" not in event_types


def test_cloud_round_fuses_direct_and_interface_iii_contributions_once() -> None:
    direct = _direct_cloud_client(0)
    edge_cloud = ClientFlowInput(
        client_id=1, edge_id=0, mode="LIIEIIIC", candidate_time=2.0,
        estimated_local_time=2.0, measured_local_time=2.0,
        communication_volume=1.0, state_diff={}, sample_count=3,
        edge_to_cloud_time=0.2, edge_aggregation_payload=1.0,
        cloud_aggregation_payload=2.0, aggregation_group="dp",
    )
    result = execute_mixed_round_flow(
        round_idx=0, clients=[direct, edge_cloud], aggregation_fraction=1.0
    )
    cloud_events = [
        event for event in result.flow_events
        if event["event_type"] == "cloud_aggregate_round"
    ]
    assert len(cloud_events) == 1
    assert cloud_events[0]["num_direct_clients"] == 1
    assert cloud_events[0]["num_edge_aggregates"] == 1
    assert result.selected_client_ids == [0, 1]


def test_all_mode_topologies_use_the_tex_aggregation_endpoints() -> None:
    expected = {
        "LIE": {"edge_aggregate"},
        "LIIE": {"edge_aggregate"},
        "LIC": {"cloud_aggregate_round"},
        "LIIC": {"cloud_aggregate_round"},
        "LIEIIC": {"cloud_aggregate_round"},
        "LIEIIIC": {"edge_aggregate", "cloud_aggregate_round"},
        "LIIEIIIC": {"edge_aggregate", "cloud_aggregate_round"},
    }
    aggregation_events = {
        "edge_aggregate",
        "cloud_aggregate_round",
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


def test_direct_cloud_is_round_synchronous_even_when_edge_buffer_is_partial() -> None:
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

    assert detailed.selected_client_ids == [0, 1, 2, 3, 4]
    assert summary.selected_client_ids == detailed.selected_client_ids
    assert summary.flow_events == []
    np.testing.assert_allclose(summary.round_duration, detailed.round_duration)


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


def test_exposure_requirement_combines_confidentiality_and_dp_orthogonally() -> None:
    requirement = ExposurePrivacyRequirement(
        plaintext_forbidden_links=frozenset({"E_C_upd"}),
        dp_required_links=frozenset({"E_C_upd"}),
    )
    candidates = enumerate_candidates(
        config=SelectionConfig(),
        client_id=0, edge_factor=1.0, compute_factor=1.0, samples=100,
        remaining_epsilon=8.0, round_idx=0, rng=random.Random(7),
        policy="ours", privacy_requirement=requirement,
    )
    exposed = [c for c in candidates if "E_C_upd" in (c.link_mechanisms or {})]
    assert exposed
    assert all(c.link_mechanisms["E_C_upd"] == "dp_he3" for c in exposed)
    assert all("trusted" not in (c.link_mechanisms or {}).values() for c in candidates)


def test_exposure_requirement_can_forbid_plaintext_without_requiring_dp() -> None:
    requirement = ExposurePrivacyRequirement(
        plaintext_forbidden_links=frozenset({"L_C_upd"}),
    )
    candidates = enumerate_candidates(
        config=SelectionConfig(),
        client_id=0, edge_factor=1.0, compute_factor=1.0, samples=100,
        remaining_epsilon=8.0, round_idx=0, rng=random.Random(11),
        policy="ours", privacy_requirement=requirement,
    )
    direct = [c for c in candidates if c.mode == "LIIC"]
    assert direct
    assert all(c.link_mechanisms["L_C_upd"] in {"he3", "dp_he3"} for c in direct)
    assert any(c.link_mechanisms["L_C_upd"] == "he3" for c in direct)


def test_exposure_requirement_rejects_mode_when_requested_confidentiality_is_unavailable() -> None:
    requirement = ExposurePrivacyRequirement(
        plaintext_forbidden_links=frozenset({"L_E_emb"}),
    )
    candidates = enumerate_candidates(
        config=SelectionConfig(),
        client_id=0, edge_factor=1.0, compute_factor=1.0, samples=100,
        remaining_epsilon=8.0, round_idx=0, rng=random.Random(13),
        policy="ours", privacy_requirement=requirement,
    )
    assert not [c for c in candidates if c.mode in {"LIE", "LIEIIC", "LIEIIIC"}]


def test_proposed_policy_does_not_offer_unprotected_private_links() -> None:
    candidates = enumerate_candidates(
        config=SelectionConfig(trusted_edge_split_execution=True),
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
        for link_id, mechanism in (candidate.link_mechanisms or {}).items():
            if link_id.endswith("_C_upd"):
                assert mechanism in {"dp", "he3", "dp_he3"}


def test_random_policy_does_not_offer_unprotected_private_links() -> None:
    candidates = enumerate_candidates(
        config=SelectionConfig(trusted_edge_split_execution=True),
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
        for link_id, mechanism in (candidate.link_mechanisms or {}).items():
            if link_id.endswith("_C_upd"):
                assert mechanism in {"dp", "he3", "dp_he3"}


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


def test_unprotected_output_gradient_uses_ordinary_batch_average() -> None:
    tensor = torch.tensor([[3.0, 4.0], [0.0, 2.0]])
    actual = _protect_batched_average_gradient_dp(
        tensor,
        mechanism="none",
        clip_norm=1.0,
        noise_multiplier=1.0,
        rng=np.random.default_rng(1),
        device=torch.device("cpu"),
    )
    expected = torch.tensor([[1.5, 2.0], [0.0, 1.0]])
    torch.testing.assert_close(actual, expected)


def test_formal_feature_dp_horizon_is_disabled() -> None:
    resolved = resolved_privacy_parameters(
        SelectionConfig(rounds=200, L_block_cycles=5, privacy_local_epochs=3)
    )
    assert resolved["max_feature_events_per_round"] == 0
    assert resolved["feature_horizon_events"] == 0


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


def test_trusted_edge_boundary_applies_to_all_policies() -> None:
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
    for mode in {"LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC"}:
        assert {
            candidate.link_mechanisms.get(
                "L_C_upd" if mode == "LIIC" else "E_C_upd"
            )
            for candidate in candidates
            if candidate.mode == mode
        } == {"dp", "he3", "dp_he3"}

    resolved = resolved_privacy_parameters(config)
    assert resolved["feature_dp_enabled"] is False
    assert resolved["feature_horizon_events"] == 0
    assert resolved["max_update_events_per_round"] == 1
    assert resolved["update_horizon_events"] == 200


def test_fixed_splitfed_uses_shared_trusted_edge_boundary() -> None:
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
    } == {"dp", "he3", "dp_he3"}


def test_fixed_splitfed_dp_diagnostic_forces_update_packet_dp() -> None:
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


def test_cross_domain_update_dp_is_applied_to_the_transmission_packet() -> None:
    candidate = Candidate(
        **{
            **_candidate("LIEIIC").__dict__,
            "mechanisms": {"upd": "dp"},
            "link_mechanisms": {
                "L_E_emb": "trusted",
                "L_E_grad": "trusted",
                "E_C_upd": "dp",
            },
        }
    )

    assert _candidate_uses_cross_domain_update_dp(candidate, True)
    assert _candidate_uses_local_packet_update_dp(candidate, True)
    assert not _candidate_uses_secure_aggregate_update_dp(candidate, True)
    training_mechanisms = _candidate_training_mechanisms(
        candidate,
        aggregate_cloud_update_dp=True,
    )
    assert training_mechanisms["upd"] == "none"
    assert not _should_apply_update_dp(training_mechanisms, "upd_only", candidate.mode)


def test_combined_update_packet_keeps_he_after_worker_dp_is_deferred() -> None:
    candidate = Candidate(
        **{
            **_candidate("LIEIIC").__dict__,
            "mechanisms": {"upd": "dp_he3"},
            "link_mechanisms": {
                "L_E_emb": "trusted",
                "L_E_grad": "trusted",
                "E_C_upd": "dp_he3",
            },
        }
    )

    assert _candidate_uses_cross_domain_update_dp(candidate, True)
    assert not _candidate_uses_local_packet_update_dp(candidate, True)
    assert _candidate_uses_secure_aggregate_update_dp(candidate, True)
    training_mechanisms = _candidate_training_mechanisms(
        candidate,
        aggregate_cloud_update_dp=True,
    )
    assert training_mechanisms["upd"] == "he3"
    assert not _should_apply_update_dp(training_mechanisms, "upd_only", candidate.mode)


def test_fixed_dp_he_forces_combined_cross_domain_update_protection() -> None:
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
        rng=random.Random(41),
        policy="fixed_dp_he",
    )

    cloud_candidates = [
        candidate
        for candidate in candidates
        if "E_C_upd" in (candidate.link_mechanisms or {})
        or "L_C_upd" in (candidate.link_mechanisms or {})
    ]
    assert cloud_candidates
    assert all(
        next(
            mechanism
            for link, mechanism in candidate.link_mechanisms.items()
            if link in {"E_C_upd", "L_C_upd"}
        )
        == "dp_he3"
        for candidate in cloud_candidates
    )
    assert all(candidate.update_dp_events == 1 for candidate in cloud_candidates)


def test_update_packet_dp_uses_max_within_packet_client_fraction() -> None:
    sensitivity, noise_std = _dp_update_release_parameters(
        0.375,
        clip_norm=2.0,
        noise_multiplier=3.0,
    )

    assert np.isclose(sensitivity, 1.5)
    assert np.isclose(noise_std, 4.5)


def test_distributed_aggregate_dp_shares_match_release_noise() -> None:
    normalized, max_weight, sensitivity, release_std, share_stds = (
        _distributed_aggregate_dp_parameters(
            [1.0, 1.0],
            [0.5, 0.5],
            clip_norm=2.0,
            noise_multiplier=3.0,
        )
    )

    np.testing.assert_allclose(normalized, [0.5, 0.5])
    assert np.isclose(max_weight, 0.25)
    assert np.isclose(sensitivity, 1.0)
    assert np.isclose(release_std, 3.0)
    assert np.isclose(
        np.sqrt(sum((weight * std) ** 2 for weight, std in zip(normalized, share_stds))),
        release_std,
    )


def test_distributed_aggregate_dp_is_disabled_without_combined_packets() -> None:
    normalized, max_weight, sensitivity, release_std, share_stds = (
        _distributed_aggregate_dp_parameters(
            [2.0, 1.0],
            [0.0, 0.0],
            clip_norm=1.0,
            noise_multiplier=4.0,
        )
    )

    np.testing.assert_allclose(normalized, [2.0 / 3.0, 1.0 / 3.0])
    assert max_weight == 0.0
    assert sensitivity == 0.0
    assert release_std == 0.0
    assert share_stds == [0.0, 0.0]


@pytest.mark.parametrize("weights,fractions,clip,sigma", [
    ([-1.0, 2.0], [0.5, 0.5], 1.0, 3.0),
    ([float("nan")], [1.0], 1.0, 3.0),
    ([float("inf")], [1.0], 1.0, 3.0),
    ([1.0], [1.1], 1.0, 3.0),
    ([1.0], [float("nan")], 1.0, 3.0),
    ([1.0], [1.0], 0.0, 3.0),
    ([1.0], [1.0], 1.0, 0.0),
    ([1.0], [1.0], 1.0, float("nan")),
])
def test_aggregate_dp_rejects_invalid_calibration(weights, fractions, clip, sigma) -> None:
    with pytest.raises(ValueError):
        _distributed_aggregate_dp_parameters(
            weights, fractions, clip_norm=clip, noise_multiplier=sigma,
        )


def test_aggregate_dp_unequal_weights_with_unprotected_and_zero_weight_packets() -> None:
    weights, maximum, sensitivity, std, shares = _distributed_aggregate_dp_parameters(
        [2.0, 3.0, 5.0, 0.0], [0.5, 0.2, 0.0, 1.0],
        clip_norm=2.0, noise_multiplier=3.0,
    )
    assert np.isclose(maximum, 0.1)
    assert np.isclose(sensitivity, 0.4)
    assert np.isclose(std, 1.2)
    assert np.isclose(sum((w * s) ** 2 for w, s in zip(weights, shares)), std ** 2)
    assert shares[2:] == [0.0, 0.0]


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


@pytest.mark.parametrize("mode", ["LIIC", "LIEIIC"])
@pytest.mark.parametrize("model_name", ["resnet18_pretrained", "resnet18_pretrained_head"])
def test_pretrained_dp_keeps_frozen_parameters_constant(monkeypatch, mode, model_name) -> None:
    from torchvision.models import resnet18

    monkeypatch.setattr(
        "dynfed.split_learning._make_torchvision_resnet",
        lambda *args, **kwargs: resnet18(weights=None),
    )
    device = torch.device("cpu")
    end, edge, _full = build_split_models(
        model_name, device, input_channels=3, image_size=32,
    )
    base = {
        part: {name: value.detach().clone() for name, value in model.state_dict().items()}
        for part, model in (("end", end), ("edge", edge))
    }
    expected_names = {
        part: {name for name, param in model.named_parameters() if param.requires_grad}
        for part, model in (("end", end), ("edge", edge))
    }
    assert not expected_names["end"]
    assert expected_names["edge"]
    if model_name.endswith("_head"):
        assert sum(p.numel() for p in edge.parameters() if p.requires_grad) == 5130
        assert all(name.startswith(("classifier.", "fc.")) for name in expected_names["edge"])
    diff = split_local_train_lenet5(
        mode, base["end"], base["edge"],
        np.random.default_rng(7).normal(size=(4, 3 * 32 * 32)).astype(np.float32),
        np.array([0, 1, 2, 3], dtype=np.int64),
        epochs=1, lr=0.01, device=device, model_name=model_name,
        input_shape=(3, 32, 32), local_steps=1, training_seed=7,
    )
    for part, names in expected_names.items():
        assert set(diff[part]) == names
    for saved_states in ({}, {0: base}):
        relative = _state_difference_from_client_update(
            client_id=0, state_diff=diff, client_model_states=saved_states,
            global_end=end, global_edge=edge, device=device,
        )
        for part, names in expected_names.items():
            assert set(relative[part]) == names
    relative = _state_difference_from_model(base, end, edge, device)
    noise = gaussian_state_difference(relative, 0.001, np.random.default_rng(7), device)
    for part, names in expected_names.items():
        assert set(noise[part]) == names
    protected = apply_unified_dp(
        diff, "dp", clip_norm=1.0, noise_multiplier=0.001,
        rng=np.random.default_rng(7), device=device,
    )
    fedavg_split([protected], [1], end, edge, device)
    # CKKS decodes a dense vector; even a nonzero frozen slice must be ignored.
    size = sum(param.numel() for model in (end, edge) for param in model.parameters())
    _apply_flat_update(torch.full((size,), 0.001), end, edge)
    for part, model in (("end", end), ("edge", edge)):
        for name, param in model.named_parameters():
            if not param.requires_grad:
                torch.testing.assert_close(param, base[part][name], rtol=0, atol=0)


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
        ("dp", "dp_he3"),
        ("he3", "dp"),
        ("he3", "he3"),
        ("he3", "dp_he3"),
        ("dp_he3", "dp"),
        ("dp_he3", "he3"),
        ("dp_he3", "dp_he3"),
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

    # Cloud has no threshold buffer (Q70), so both legal cloud-bound
    # contributions remain in the round regardless of Edge buffer rho.
    assert result.selected_client_ids == [0, 1]


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


def test_edge_cloud_dp_packet_consumes_update_budget() -> None:
    candidate = Candidate(
        **{
            **_candidate("LIEIIC").__dict__,
            "link_mechanisms": {"E_C_upd": "dp"},
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


def test_selector_cloud_weights_match_actual_sample_mass_fedavg() -> None:
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


def test_cloud_weights_are_exact_represented_sample_mass() -> None:
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

    assert np.allclose(weights, [25.0, 75.0, 50.0])


def test_edge_aggregate_carries_sum_of_admitted_client_mass_only() -> None:
    cloud_updates = [
        (None, 100, None, [0, 1]),
        (None, 40, None, [2]),
    ]

    weights = _edge_normalized_cloud_weights(
        cloud_updates,
        client_edges={0: 0, 1: 0, 2: 1},
        edge_total_samples={0: 1200.0, 1: 600.0},
    )

    assert np.allclose(weights, [100.0, 40.0])


def test_edge_aggregate_and_direct_updates_match_flat_sample_mass_fedavg() -> None:
    # Eupd represents clients 0+1 (25+75 samples); direct Lupd represents client 2 (50).
    weights = _edge_normalized_cloud_weights(
        [(None, 100, None, [0, 1]), (None, 50, None, [2])],
        client_edges={0: 0, 1: 0, 2: 1},
        edge_total_samples={0: 1000.0, 1: 50.0},
    )

    normalized = np.asarray(weights) / sum(weights)
    assert np.allclose(normalized, [2.0 / 3.0, 1.0 / 3.0])


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


def test_dp_tiers_start_at_client_specific_minimum_feasible_noise() -> None:
    config = SelectionConfig(rounds=20, dp_tier_gamma=1.5, dp_tier_count=3)
    ledger = build_client_privacy_ledger(config)
    requirement = ExposurePrivacyRequirement(dp_required_links=frozenset({"L_C_upd"}))
    candidates = enumerate_candidates(
        config=config, client_id=0, edge_factor=1.0, compute_factor=1.0,
        samples=100, remaining_epsilon=ledger.remaining_budget, round_idx=0,
        rng=random.Random(101), policy="ours", privacy_ledger=ledger,
        privacy_requirement=requirement,
    )
    dp = [c for c in candidates if c.mode == "LIIC" and c.update_dp_events > 0]
    sigmas = sorted({round(float(c.update_noise_multiplier), 10) for c in dp})
    assert len(sigmas) == 3
    assert sigmas[1] == pytest.approx(sigmas[0] * 1.5, rel=1e-8)
    assert sigmas[2] == pytest.approx(sigmas[0] * 1.5**2, rel=1e-8)
    assert all(c.feasible_privacy for c in dp)


def test_remaining_budget_raises_next_round_sigma_minimum() -> None:
    config = SelectionConfig(rounds=20, dp_tier_gamma=1.5, dp_tier_count=3)
    ledger = build_client_privacy_ledger(config)
    first = ledger.minimum_feasible_update_noise(1)
    ledger.add(0, 1, update_noise_multiplier=first * 1.5)
    second = ledger.minimum_feasible_update_noise(1)
    assert second > first



def test_formal_paper_runner_isolated_from_async_staleness_mainline() -> None:
    root = Path(__file__).resolve().parents[1]
    runner = (root / "experiments" / "run_paper_config.py").read_text(encoding="utf-8")
    async_runner = (root / "experiments" / "run_async_fmnist.py").read_text(encoding="utf-8")

    assert "dynfed.async_training" not in runner
    assert "--legacy-async-experiment" in async_runner
    assert "legacy diagnostic" in async_runner.lower()

    for name in ("paper_v30_cifar10_resnet18.json", "paper_v30_fmnist_lenet5.json"):
        config = json.loads((root / "configs" / name).read_text(encoding="utf-8"))
        serialized = json.dumps(config).lower()
        assert "staleness" not in serialized
        assert "max_version_gap" not in serialized
        assert "b_cloud" not in serialized
        command = build_command(config, seed=42, policies=["ours"], rounds=1)
        assert "run_async_fmnist.py" not in " ".join(command)


def test_formal_runner_rejects_top_level_async_staleness_controls() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads(
        (root / "configs" / "paper_v30_fmnist_lenet5.json").read_text(encoding="utf-8")
    )
    config["max_version_gap"] = 3
    with pytest.raises(ValueError, match="Async/staleness controls"):
        build_command(config, seed=42, policies=["ours"], rounds=1)


def test_fast_response_deadline_is_hard_only_for_marked_client() -> None:
    config = SelectionConfig(time_limit=1e-12)
    ordinary = enumerate_candidates(
        config=config,
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=100,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(7),
        policy="performance_only",
    )
    assert ordinary
    assert all(candidate.feasible_time for candidate in ordinary)

    fast = enumerate_candidates(
        config=config,
        client_id=1,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=100,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(7),
        policy="performance_only",
        fast_response_deadline=1e-12,
    )
    assert fast == []


def test_fast_response_deadline_keeps_only_candidates_within_tau_max() -> None:
    config = SelectionConfig()
    baseline = enumerate_candidates(
        config=config,
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=100,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(11),
        policy="performance_only",
    )
    assert baseline
    times = sorted({candidate.time for candidate in baseline})
    assert len(times) > 1
    deadline = times[len(times) // 2]

    constrained = enumerate_candidates(
        config=config,
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=100,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(11),
        policy="performance_only",
        fast_response_deadline=deadline,
    )
    assert constrained
    assert all(candidate.time <= deadline + 1e-12 for candidate in constrained)
    assert len(constrained) < len(baseline)


def test_formal_configs_use_only_frozen_four_internal_baselines() -> None:
    root = Path(__file__).resolve().parents[1]
    expected = [
        "fixed_mode_fixed_privacy",
        "dynamic_mode_fixed_privacy",
        "fixed_mode_dynamic_privacy",
        "full_dynfl",
    ]
    for name in ("paper_v30_cifar10_resnet18.json", "paper_v30_fmnist_lenet5.json"):
        config = json.loads((root / "configs" / name).read_text(encoding="utf-8"))
        assert config["policies"] == expected
        assert config["privacy"]["candidate_mechanisms"] == ["dp", "he3", "dp_he3"]


def test_fixed_mode_baselines_use_liieiiic_without_relaxing_feasibility() -> None:
    config = SelectionConfig(update_mechanism_options=("dp", "he3", "dp_he3"))
    ledger = build_client_privacy_ledger(config)
    requirement = ExposurePrivacyRequirement(
        plaintext_forbidden_links=frozenset({"E_C_upd"}),
        dp_required_links=frozenset({"E_C_upd"}),
    )
    for policy in ("fixed_mode_fixed_privacy", "fixed_mode_dynamic_privacy"):
        candidates = enumerate_candidates(
            config=config, client_id=0, edge_factor=1.0, compute_factor=1.0,
            samples=100, remaining_epsilon=ledger.remaining_budget, round_idx=0,
            rng=random.Random(202), policy=policy, privacy_ledger=ledger,
            privacy_requirement=requirement,
        )
        assert candidates
        assert {candidate.mode for candidate in candidates} == {"LIIEIIIC"}
        assert all(candidate.feasible_privacy for candidate in candidates)


def test_fixed_privacy_profile_locks_mechanism_and_dp_strength() -> None:
    config = SelectionConfig(
        rounds=20,
        update_mechanism_options=("dp", "he3", "dp_he3"),
        dp_tier_count=3,
    )
    ledger = build_client_privacy_ledger(config)
    requirement = ExposurePrivacyRequirement(
        plaintext_forbidden_links=frozenset({"L_C_upd", "E_C_upd"}),
        dp_required_links=frozenset({"L_C_upd", "E_C_upd"}),
    )
    initial = enumerate_candidates(
        config=config, client_id=0, edge_factor=1.0, compute_factor=1.0,
        samples=100, remaining_epsilon=ledger.remaining_budget, round_idx=0,
        rng=random.Random(303), policy="dynamic_mode_fixed_privacy", privacy_ledger=ledger,
        privacy_requirement=requirement,
    )
    chosen = next(c for c in initial if c.mode == "LIIC" and c.update_dp_events > 0)
    profile = (dict(chosen.mechanisms), chosen.update_noise_multiplier)
    locked = enumerate_candidates(
        config=config, client_id=0, edge_factor=1.0, compute_factor=1.0,
        samples=100, remaining_epsilon=ledger.remaining_budget, round_idx=1,
        rng=random.Random(304), policy="dynamic_mode_fixed_privacy", privacy_ledger=ledger,
        privacy_requirement=requirement, fixed_privacy_profile=profile,
    )
    assert locked
    assert all(c.mechanisms == profile[0] for c in locked)
    assert all(c.update_noise_multiplier == pytest.approx(profile[1]) for c in locked if c.update_dp_events > 0)


def test_dirichlet_partition_is_seeded_complete_and_disjoint() -> None:
    from dynfed.fmnist_lenet5_dynamic import _partition_clients_lenet5

    labels = np.repeat(np.arange(10), 60)
    first = _partition_clients_lenet5(
        labels, num_clients=20, num_edges=4, iid=False,
        partition_mode="dirichlet", seed=2026, dirichlet_alpha=0.5,
    )
    second = _partition_clients_lenet5(
        labels, num_clients=20, num_edges=4, iid=False,
        partition_mode="dirichlet", seed=2026, dirichlet_alpha=0.5,
    )
    assert all(np.array_equal(a, b) for a, b in zip(first, second))
    merged = np.concatenate(first)
    assert len(merged) == len(labels)
    assert len(np.unique(merged)) == len(labels)
    assert set(merged.tolist()) == set(range(len(labels)))


def test_dirichlet_alpha_controls_non_iid_strength() -> None:
    from dynfed.fmnist_lenet5_dynamic import _partition_clients_lenet5

    labels = np.repeat(np.arange(10), 200)
    moderate = _partition_clients_lenet5(
        labels, 20, 4, False, "dirichlet", 77, dirichlet_alpha=0.5,
    )
    strong = _partition_clients_lenet5(
        labels, 20, 4, False, "dirichlet", 77, dirichlet_alpha=0.1,
    )

    def mean_label_concentration(parts):
        scores = []
        for part in parts:
            if len(part):
                counts = np.bincount(labels[part], minlength=10)
                scores.append(float(counts.max()) / float(counts.sum()))
        return float(np.mean(scores))

    assert mean_label_concentration(strong) > mean_label_concentration(moderate)


def test_paper_runner_forwards_dirichlet_alpha() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "paper_v30_cifar10_resnet18.json").read_text(encoding="utf-8"))
    config["training"]["partition_mode"] = "dirichlet"
    config["training"]["dirichlet_alpha"] = 0.1
    command = build_command(config, seed=42, policies=["full_dynfl"], rounds=None)
    assert "--partition-mode" in command
    assert command[command.index("--partition-mode") + 1] == "dirichlet"
    assert command[command.index("--dirichlet-alpha") + 1] == "0.1"


def test_staged_resource_phase_is_normal_constrained_normal() -> None:
    from dynfed.selection import resource_phase

    config = SelectionConfig(rounds=90, resource_scenario="communication")
    assert resource_phase(config, 0) == "normal"
    assert resource_phase(config, 29) == "normal"
    assert resource_phase(config, 30) == "constrained"
    assert resource_phase(config, 59) == "constrained"
    assert resource_phase(config, 60) == "normal"
    assert resource_phase(config, 89) == "normal"


def test_communication_scenario_only_reduces_middle_phase_bandwidth() -> None:
    config = SelectionConfig(
        rounds=90,
        resource_scenario="communication",
        communication_constrained_multiplier=0.25,
        network_jitter=0.0,
        network_periodic_amplitude=0.0,
    )
    normal = _link_bandwidth(config, client_id=0, round_idx=0, link_id="L_E_upd")
    constrained = _link_bandwidth(config, client_id=0, round_idx=45, link_id="L_E_upd")
    restored = _link_bandwidth(config, client_id=0, round_idx=75, link_id="L_E_upd")
    assert constrained == pytest.approx(normal * 0.25)
    assert restored == pytest.approx(normal)


def test_compute_scenario_only_slows_middle_phase_compute() -> None:
    from dynfed.selection import staged_compute_factor

    config = SelectionConfig(
        rounds=90,
        resource_scenario="compute",
        compute_constrained_multiplier=2.5,
    )
    assert staged_compute_factor(config, 1.2, 0) == pytest.approx(1.2)
    assert staged_compute_factor(config, 1.2, 45) == pytest.approx(3.0)
    assert staged_compute_factor(config, 1.2, 75) == pytest.approx(1.2)


def test_paper_runner_forwards_staged_resource_scenario() -> None:
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root / "configs" / "paper_v30_cifar10_resnet18.json").read_text(encoding="utf-8"))
    config["system"].update({
        "resource_scenario": "communication",
        "constrained_start_fraction": 0.25,
        "constrained_end_fraction": 0.75,
        "communication_constrained_multiplier": 0.4,
        "compute_constrained_multiplier": 2.2,
    })
    command = build_command(config, seed=42, policies=["full_dynfl"], rounds=None)
    assert command[command.index("--resource-scenario") + 1] == "communication"
    assert command[command.index("--constrained-start-fraction") + 1] == "0.25"
    assert command[command.index("--constrained-end-fraction") + 1] == "0.75"
    assert command[command.index("--communication-constrained-multiplier") + 1] == "0.4"
    assert command[command.index("--compute-constrained-multiplier") + 1] == "2.2"


def test_deadline_satisfaction_ratio_counts_skip_as_miss_and_ignores_ordinary_clients() -> None:
    rows = [
        {"fast_response_client": True, "deadline_satisfied": True},
        {"fast_response_client": True, "deadline_satisfied": False},
        {"fast_response_client": False, "deadline_satisfied": None},
    ]
    assert _deadline_satisfaction_ratio(rows) == pytest.approx(0.5)


def test_deadline_satisfaction_ratio_is_none_without_fast_response_clients() -> None:
    rows = [
        {"fast_response_client": False, "deadline_satisfied": None},
        {"fast_response_client": False, "deadline_satisfied": None},
    ]
    assert _deadline_satisfaction_ratio(rows) is None


def test_formal_configs_use_three_frozen_random_seeds() -> None:
    root = Path(__file__).resolve().parents[1]
    for name in ("paper_v30_cifar10_resnet18.json", "paper_v30_fmnist_lenet5.json"):
        config = json.loads((root / "configs" / name).read_text(encoding="utf-8"))
        assert config["seeds"] == [40, 42, 44]


def test_exact_solver_validation_reports_gap_and_both_solve_times() -> None:
    from experiments.validate_pareto_search import run_instance

    row = run_instance(
        num_clients=2, seed=40, candidates_per_client=4, archive_size=8, max_iters=5
    )
    assert row["scalarized_objective_gap"] >= 0.0
    assert row["bounded_pareto_solve_time_sec"] >= 0.0
    assert row["exact_solver_solve_time_sec"] >= 0.0


def test_multiseed_aggregator_defaults_to_final_four_and_mean_std_metrics() -> None:
    root = Path(__file__).resolve().parents[1]
    text = (root / "experiments" / "aggregate_multiseed_results.py").read_text(encoding="utf-8")
    for policy in (
        "fixed_mode_fixed_privacy", "dynamic_mode_fixed_privacy",
        "fixed_mode_dynamic_privacy", "full_dynfl",
    ):
        assert policy in text
    assert 'row[f"{metric}_mean"]' in text
    assert 'row[f"{metric}_std"]' in text


def test_final_paper_plan_is_five_figures_two_tables_and_three_seeds():
    from experiments.paper_final_plan import (
        FINAL_FIGURES, FINAL_TABLES, FORMAL_SEEDS, validate_final_plan,
    )
    validate_final_plan()
    assert len(FINAL_FIGURES) == 5
    assert len(FINAL_TABLES) == 2
    assert FORMAL_SEEDS == (40, 42, 44)
    from experiments.paper_final_plan import DATA_DISTRIBUTIONS
    assert DATA_DISTRIBUTIONS == (
        ("iid", "iid", None),
        ("dirichlet_0p5", "dirichlet", 0.5),
        ("dirichlet_0p1", "dirichlet", 0.1),
    )


def test_final_parameter_sensitivity_is_strategy_period_only():
    import json
    import sys
    from pathlib import Path
    experiments_dir = Path(__file__).resolve().parents[1] / "experiments"
    sys.path.insert(0, str(experiments_dir))
    from run_controlled_sweeps import build_cases
    from run_paper_config import DEFAULT_CONFIG

    base = json.loads(DEFAULT_CONFIG.resolve().read_text(encoding="utf-8"))
    cases = build_cases(base, ["period"])
    assert [(case.study, case.setting) for case in cases] == [
        ("period", "1"), ("period", "5"), ("period", "10")
    ]
    assert all(case.policies == ["full_dynfl"] for case in cases)


def test_final_controlled_aggregator_excludes_legacy_privacy_scale_ablation_runs():
    from pathlib import Path
    source = (Path(__file__).resolve().parents[1] / "experiments" / "aggregate_controlled_results.py").read_text(encoding="utf-8")
    main_body = source.split("def main() -> None:", 1)[1].split("\ndef plot_period", 1)[0]
    assert 'for period in (1, 5, 10)' in main_body
    assert 'root / "privacy"' not in main_body
    assert 'root / "scale"' not in main_body
    assert 'root / "ablation"' not in main_body


def test_final_suite_builds_frozen_q98_matrix_without_inventing_fast_deadlines() -> None:
    from experiments.run_final_paper_suite import build_final_cases
    from experiments.run_paper_config import DEFAULT_CONFIG

    base = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cases = build_final_cases(
        base,
        studies=["main", "dynamic_resources", "privacy", "sp"],
        fast_config=None,
    )
    keys = [(case["study"], case["setting"]) for case in cases]
    assert [(case["study"], case["setting"]) for case in cases if case["study"] == "main"] == [
        ("main", "iid"),
        ("main", "dirichlet_0p5"),
        ("main", "dirichlet_0p1"),
    ]
    main_cases = [case for case in cases if case["study"] == "main"]
    assert [case["config"]["training"]["partition_mode"] for case in main_cases] == [
        "iid", "dirichlet", "dirichlet"
    ]
    assert "dirichlet_alpha" not in main_cases[0]["config"]["training"]
    assert main_cases[1]["config"]["training"]["dirichlet_alpha"] == pytest.approx(0.5)
    assert main_cases[2]["config"]["training"]["dirichlet_alpha"] == pytest.approx(0.1)
    assert [case["config"]["output_root"] for case in main_cases] == [
        "out/paper_v31_final/fig1_main/iid",
        "out/paper_v31_final/fig1_main/dirichlet_0p5",
        "out/paper_v31_final/fig1_main/dirichlet_0p1",
    ]
    assert ("dynamic_resources", "communication") in keys
    assert ("dynamic_resources", "compute") in keys
    assert ("privacy", "dynamic_vs_fixed") in keys
    assert [(case["study"], case["setting"]) for case in cases if case["study"] == "sp"] == [
        ("sp", "1"), ("sp", "5"), ("sp", "10")
    ]
    privacy_case = next(case for case in cases if case["study"] == "privacy")
    assert privacy_case["policies"] == ["dynamic_mode_fixed_privacy", "full_dynfl"]


def test_final_suite_requires_explicit_fast_response_contract() -> None:
    from experiments.run_final_paper_suite import build_final_cases
    from experiments.run_paper_config import DEFAULT_CONFIG

    base = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="fast_response requires --fast-config"):
        build_final_cases(base, studies=["fast_response"], fast_config=None)


def test_multiseed_aggregator_defaults_match_formal_cifar_model() -> None:
    from experiments import aggregate_multiseed_results

    source = Path(aggregate_multiseed_results.__file__).read_text(encoding="utf-8")
    assert 'parser.add_argument("--model", default="resnet18_pretrained_head")' in source
    assert 'reconfiguration_policy = "full_dynfl"' in source


def test_frozen_fast_response_config_is_twenty_percent_at_twenty_seconds() -> None:
    root = Path(__file__).resolve().parents[1]
    path = root / "configs" / "paper_v31_cifar10_resnet18_fast_response.json"
    config = json.loads(path.read_text(encoding="utf-8"))
    deadlines = config["system"]["fast_client_deadlines"]
    assert config["system"]["clients"] == 100
    assert len(deadlines) == 20
    assert set(deadlines) == {str(client_id) for client_id in range(20)}
    assert set(float(value) for value in deadlines.values()) == {20.0}
    assert config["seeds"] == [40, 42, 44]


def test_final_suite_defaults_to_frozen_fast_response_config() -> None:
    from experiments.run_final_paper_suite import DEFAULT_FAST_CONFIG, build_final_cases
    from experiments.run_paper_config import DEFAULT_CONFIG

    base = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    fast = json.loads(DEFAULT_FAST_CONFIG.read_text(encoding="utf-8"))
    cases = build_final_cases(base, studies=["fast_response"], fast_config=fast)
    assert len(cases) == 1
    case = cases[0]
    assert case["study"] == "fast_response"
    assert case["config"]["output_root"] == "out/paper_v31_final/fig4_fast_response"
    assert len(case["config"]["system"]["fast_client_deadlines"]) == 20
    assert case["policies"] == list(fast["policies"])


def test_final_main_heterogeneity_cases_forward_frozen_partition_cli() -> None:
    from experiments.run_final_paper_suite import build_final_cases
    from experiments.run_paper_config import DEFAULT_CONFIG, build_command

    base = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    cases = build_final_cases(base, studies=["main"], fast_config=None)
    commands = [build_command(case["config"], seed=42, policies=case["policies"], rounds=1) for case in cases]
    observed = []
    for command in commands:
        mode = command[command.index("--partition-mode") + 1]
        alpha = command[command.index("--dirichlet-alpha") + 1]
        observed.append((mode, alpha))
    # The runner always forwards an alpha value, but IID ignores it.  The two
    # non-IID formal cases must preserve the frozen Q83 alpha values exactly.
    assert observed == [("iid", "0.5"), ("dirichlet", "0.5"), ("dirichlet", "0.1")]


def test_stage_mode_selection_uses_frozen_normal_constrained_normal_boundaries() -> None:
    from experiments.aggregate_multiseed_results import resource_stage

    assert [resource_stage(index, 9, 1.0 / 3.0, 2.0 / 3.0) for index in range(9)] == [
        "normal_before", "normal_before", "normal_before",
        "constrained", "constrained", "constrained",
        "normal_after", "normal_after", "normal_after",
    ]


def test_dynamic_resource_aggregation_writes_stage_mode_selection_output() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "aggregate_multiseed_results.py").read_text(encoding="utf-8")
    assert 'write_csv(output_dir / "stage_mode_selection.csv", stage_rows)' in source
    assert 'source.config["selection"].get("resource_scenario", "none")' in source
    assert '"normal_before", "constrained", "normal_after"' in source
    for mode in ("LIE", "LIC", "LIIE", "LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC"):
        assert mode in source


def test_final_plan_freezes_fmnist_lightweight_cnn_as_auxiliary_validation() -> None:
    from experiments.paper_final_plan import AUXILIARY_DATASET, AUXILIARY_MODEL

    assert AUXILIARY_DATASET == "fmnist"
    assert AUXILIARY_MODEL == "lenet5"


def test_final_suite_builds_fmnist_auxiliary_case_without_expanding_main_figures() -> None:
    from experiments.run_final_paper_suite import DEFAULT_AUX_CONFIG, build_final_cases
    from experiments.run_paper_config import DEFAULT_CONFIG

    base = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    aux = json.loads(DEFAULT_AUX_CONFIG.read_text(encoding="utf-8"))
    cases = build_final_cases(base, studies=["auxiliary"], fast_config=None, aux_config=aux)
    assert len(cases) == 1
    case = cases[0]
    assert (case["study"], case["setting"]) == ("auxiliary", "fmnist_lightweight_cnn")
    assert case["config"]["training"]["dataset"] == "fmnist"
    assert case["config"]["training"]["model"] == "lenet5"
    assert case["config"]["output_root"] == "out/paper_v31_final/aux_fmnist"
    assert case["policies"] == list(aux["policies"])


def test_final_suite_rejects_non_fmnist_auxiliary_config() -> None:
    from experiments.run_final_paper_suite import build_final_cases
    from experiments.run_paper_config import DEFAULT_CONFIG

    base = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="dataset=fmnist and model=lenet5"):
        build_final_cases(base, studies=["auxiliary"], fast_config=None, aux_config=base)


def test_q96_privacy_aggregation_uses_realized_update_epsilon_without_budget_sweep() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "aggregate_final_privacy.py").read_text(encoding="utf-8")
    assert 'FORMAL_PRIVACY_POLICIES = ("dynamic_mode_fixed_privacy", "full_dynfl")' in source
    assert '"max_update_epsilon_mean"' in source
    assert '"test_accuracy_mean"' in source
    assert '"accounted_system_time_sec_mean"' in source
    assert "privacy_budget" not in source.split("def plot_privacy_trajectory", 1)[1]
    assert "for privacy" not in source


def test_q96_privacy_trajectory_aggregates_roundwise_realized_consumption(tmp_path: Path) -> None:
    from experiments.aggregate_multiseed_results import SourceRun, aggregate_privacy_trajectory

    sources = {}
    for seed, epsilons in ((40, (0.2, 0.4)), (42, (0.3, 0.5)), (44, (0.4, 0.6))):
        policy = "full_dynfl"
        policy_dir = tmp_path / str(seed) / policy
        policy_dir.mkdir(parents=True)
        (policy_dir / "round_metrics.csv").write_text(
            "test_accuracy,accounted_system_time_sec,max_update_epsilon\n"
            + "\n".join(
                f"{accuracy},{latency},{epsilon}"
                for accuracy, latency, epsilon in ((0.5, 10.0, epsilons[0]), (0.6, 20.0, epsilons[1]))
            )
            + "\n",
            encoding="utf-8",
        )
        sources[(seed, policy)] = SourceRun(seed, policy, policy_dir.parent, policy_dir, {}, {})
    rows = aggregate_privacy_trajectory(sources, [40, 42, 44], ["full_dynfl"], rounds=2)
    assert len(rows) == 2
    assert rows[0]["max_update_epsilon_mean"] == pytest.approx(0.3)
    assert rows[1]["max_update_epsilon_mean"] == pytest.approx(0.5)
    assert rows[1]["test_accuracy_mean"] == pytest.approx(0.6)
    assert rows[1]["accounted_system_time_sec_mean"] == pytest.approx(20.0)


def test_controlled_aggregator_uses_current_formal_cifar_model_name() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "aggregate_controlled_results.py").read_text(encoding="utf-8")
    assert 'model="resnet18_pretrained_head"' in source
    assert 'model="resnet18_pretrained",' not in source


def test_q97_optimizer_summary_is_compact_mean_std_by_client_count() -> None:
    from experiments.validate_pareto_search import summarize_optimizer_validation

    rows = []
    for num_clients in (2, 3):
        for seed, gap, bounded, exact in (
            (40, 0.01, 0.1, 0.3),
            (42, 0.02, 0.2, 0.4),
            (44, 0.03, 0.3, 0.5),
        ):
            rows.append({
                "num_clients": num_clients,
                "seed": seed,
                "scalarized_objective_gap": gap,
                "bounded_pareto_solve_time_sec": bounded,
                "exact_solver_solve_time_sec": exact,
            })
    summary = summarize_optimizer_validation(rows)
    assert [row["num_clients"] for row in summary] == [2, 3]
    assert all(row["n_seeds"] == 3 for row in summary)
    assert summary[0]["scalarized_objective_gap_mean"] == pytest.approx(0.02)
    assert summary[0]["scalarized_objective_gap_std"] == pytest.approx(0.01)
    assert summary[0]["bounded_pareto_solve_time_sec_mean"] == pytest.approx(0.2)
    assert summary[0]["exact_solver_solve_time_sec_mean"] == pytest.approx(0.4)


def test_q97_optimizer_output_contract_has_no_pareto_frontier_plot() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "validate_pareto_search.py").read_text(encoding="utf-8")
    assert 'optimizer_summary.csv' in source
    assert 'fig5_optimizer_validation.png' in source
    assert '"scalarized_objective_gap"' in source
    assert '"bounded_pareto_solve_time_sec"' in source
    assert '"exact_solver_solve_time_sec"' in source
    assert "pareto_frontier" not in source


def test_q98_table1_is_generated_from_formal_config_without_duplicate_constants() -> None:
    from experiments.generate_table1_parameters import build_table1_rows
    from experiments.run_paper_config import DEFAULT_CONFIG

    config = json.loads(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    rows = build_table1_rows(config)
    by_path = {row["config_path"]: row for row in rows}
    assert by_path["training.dataset"]["value"] == "cifar10"
    assert by_path["training.model"]["value"] == "resnet18_pretrained_head"
    assert by_path["system.clients"]["value"] == "100"
    assert by_path["system.edges"]["value"] == "10"
    assert by_path["privacy.update_epsilon_budget"]["value"] == "8"
    assert by_path["privacy.delta"]["value"] == "1e-05"
    assert by_path["privacy.clip_norm"]["value"] == "0.25"
    assert by_path["privacy.candidate_mechanisms"]["value"] == "dp, he3, dp_he3"


def test_q98_table1_output_contract_is_machine_readable_and_traceable() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "generate_table1_parameters.py").read_text(encoding="utf-8")
    assert 'table1_parameters.csv' in source
    assert '"config_path"' in source
    assert '"training.selection_period"' in source
    assert '"privacy.update_epsilon_budget"' in source
    assert '"optimization.pareto_max_iters"' in source


def test_q98_table2_freezes_four_internal_method_definitions() -> None:
    from experiments.generate_table2_methods_results import METHOD_DEFINITIONS
    from experiments.paper_final_plan import FINAL_POLICIES

    assert tuple(METHOD_DEFINITIONS) == FINAL_POLICIES
    assert METHOD_DEFINITIONS["fixed_mode_fixed_privacy"]["mode_definition"] == "LIIEIIIC"
    assert METHOD_DEFINITIONS["fixed_mode_dynamic_privacy"]["mode_definition"] == "LIIEIIIC"
    assert METHOD_DEFINITIONS["dynamic_mode_fixed_privacy"]["mode_policy"] == "Dynamic"
    assert METHOD_DEFINITIONS["full_dynfl"]["privacy_policy"] == "Dynamic"


def test_q98_table2_joins_definitions_with_compact_core_results() -> None:
    from experiments.generate_table2_methods_results import build_table2_rows
    from experiments.paper_final_plan import FINAL_POLICIES

    summary = {}
    for index, policy in enumerate(FINAL_POLICIES):
        summary[policy] = {
            "final_test_accuracy_mean": 0.70 + index * 0.01,
            "final_test_accuracy_std": 0.01,
            "avg_last_10_accuracy_mean": 0.69 + index * 0.01,
            "avg_last_10_accuracy_std": 0.02,
            "accounted_system_time_sec_mean": 100.0 + index,
            "accounted_system_time_sec_std": 2.0,
            "max_update_epsilon_mean": 4.0 + index * 0.1,
            "max_update_epsilon_std": 0.1,
            "feasible_participation_ratio_mean": 0.95,
            "feasible_participation_ratio_std": 0.01,
        }
    rows = build_table2_rows(summary)
    assert [row["policy"] for row in rows] == list(FINAL_POLICIES)
    assert rows[-1]["label"] == "Full DynFL"
    assert rows[-1]["final_test_accuracy_mean"] == pytest.approx(0.73)
    assert rows[0]["accounted_system_time_sec_mean"] == pytest.approx(100.0)
    assert rows[0]["max_update_epsilon_mean"] == pytest.approx(4.0)


def test_q98_table2_output_is_machine_readable_and_uses_aggregated_results() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "generate_table2_methods_results.py").read_text(encoding="utf-8")
    assert "table2_methods_results.csv" in source
    assert "summary_statistics.csv" in source
    assert "FINAL_POLICIES" in source
    assert '"final_test_accuracy_mean"' in source
    assert '"accounted_system_time_sec_mean"' in source
    assert '"max_update_epsilon_mean"' in source
    assert "Run the formal multi-seed aggregation first" in source


def test_q98_final_output_readiness_map_is_exactly_five_figures_two_tables() -> None:
    from experiments.paper_final_plan import FINAL_FIGURES, FINAL_TABLES
    from experiments.validate_final_paper_outputs import OUTPUT_EVIDENCE

    assert set(OUTPUT_EVIDENCE) == set(FINAL_FIGURES) | set(FINAL_TABLES)
    assert len(FINAL_FIGURES) == 5
    assert len(FINAL_TABLES) == 2
    assert "fig1_main/iid/aggregate/summary_statistics.csv" in OUTPUT_EVIDENCE["fig1_main_four_methods"]
    assert "fig2_dynamic_resources/communication/aggregate/stage_mode_selection.csv" in OUTPUT_EVIDENCE["fig2_dynamic_resources"]
    assert OUTPUT_EVIDENCE["fig3_dynamic_vs_fixed_privacy"] == ("fig3_privacy_aggregate/privacy_trajectory.csv",)
    assert "fig5_optimizer/optimizer_summary.csv" in OUTPUT_EVIDENCE["fig5_optimizer_and_sp"]
    assert "sensitivity_aggregate/strategy_period_statistics.csv" in OUTPUT_EVIDENCE["fig5_optimizer_and_sp"]


def test_q98_readiness_never_marks_missing_results_complete(tmp_path: Path) -> None:
    from experiments.validate_final_paper_outputs import OUTPUT_EVIDENCE, build_readiness

    # Create only Table I evidence.  The checker must report partial readiness,
    # not infer or fabricate any of the still-missing experimental outputs.
    (tmp_path / OUTPUT_EVIDENCE["table1_parameters"][0]).write_text("section,parameter\n", encoding="utf-8")
    report = build_readiness(tmp_path)
    assert report["ready"] is False
    assert report["figures_ready"] == 0
    assert report["tables_ready"] == 1
    by_name = {item["name"]: item for item in report["items"]}
    assert by_name["table1_parameters"]["ready"] is True
    assert by_name["table2_methods_results"]["ready"] is False
    assert by_name["fig3_dynamic_vs_fixed_privacy"]["missing"] == [
        "fig3_privacy_aggregate/privacy_trajectory.csv"
    ]


def test_step26_postprocess_covers_all_q98_training_outputs() -> None:
    from experiments.postprocess_final_paper_suite import build_commands

    commands = build_commands(seeds=[40, 42, 44], rounds=100)
    rendered = [" ".join(command) for command in commands]
    assert len(commands) == 11
    for setting in ("iid", "dirichlet_0p5", "dirichlet_0p1"):
        assert any(f"fig1_main/{setting}" in command.replace("\\", "/") for command in rendered)
    for scenario in ("communication", "compute"):
        assert any(f"fig2_dynamic_resources/{scenario}" in command.replace("\\", "/") for command in rendered)
    assert any("aggregate_final_privacy.py" in command for command in rendered)
    assert any("fig4_fast_response" in command for command in rendered)
    assert any("aggregate_controlled_results.py" in command and "fig5_sp" in command for command in rendered)
    assert any("generate_table1_parameters.py" in command for command in rendered)
    assert any("generate_table2_methods_results.py" in command for command in rendered)
    assert rendered[-1].endswith("validate_final_paper_outputs.py --require-complete")


def test_step26_sp_aggregator_default_matches_final_suite_output_root() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "aggregate_controlled_results.py").read_text(encoding="utf-8")
    assert 'default=ROOT / "out" / "paper_v31_final" / "fig5_sp"' in source
    assert 'default=ROOT / "out" / "paper_v31_final" / "sensitivity"' not in source


def test_step26_postprocess_is_aggregation_only_and_requires_complete_evidence() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "postprocess_final_paper_suite.py").read_text(encoding="utf-8")
    assert "run_paper_config.py" not in source
    assert "run_final_paper_suite.py" not in source
    assert '"--require-complete"' in source
    assert "subprocess.run(command, cwd=ROOT, check=True)" in source


def test_step27_preflight_reports_exact_frozen_suite_workload(monkeypatch, tmp_path: Path) -> None:
    import experiments.preflight_final_paper_suite as preflight
    from experiments.run_final_paper_suite import _load

    root = Path(__file__).resolve().parents[1]
    base = _load(root / "configs" / "paper_v30_cifar10_resnet18.json")
    fast = _load(root / "configs" / "paper_v31_cifar10_resnet18_fast_response.json")
    aux = _load(root / "configs" / "paper_v30_fmnist_lenet5.json")
    monkeypatch.setattr(preflight, "_module_check", lambda name: {"ready": True, "version": "test"})
    monkeypatch.setattr(preflight, "check_he_backend", lambda backend: type("Status", (), {"available": True, "detail": "test"})())
    report = preflight.build_report(base, fast, aux, studies=list(preflight.FINAL_STUDIES), seeds=[40, 42, 44], output_root=tmp_path)
    workload = report["workload"]
    assert workload["formal_cases"] == 11
    assert workload["training_invocations"] == 33
    assert workload["policy_runs"] == 81
    assert workload["configured_rounds_per_policy"] == 100
    assert workload["policy_rounds"] == 8100
    assert workload["optimizer_validation_invocations"] == 1


def test_step27_preflight_checks_cuda_real_he_and_never_trains() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "experiments" / "preflight_final_paper_suite.py").read_text(encoding="utf-8")
    assert "torch.cuda.is_available()" in source
    assert "check_he_backend" in source
    assert '"require_real_he"' in source
    assert "subprocess.run" not in source
    assert "run_fmnist_lenet5.py" not in source
    assert "Dataset loaders may download CIFAR-10/Fashion-MNIST on first use." in source
    assert "pretrained ResNet-18" in source
