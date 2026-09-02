from __future__ import annotations

import random

from dynfed.config import ExperimentConfig, ModeConfig, RuntimeConfig, TopologyConfig
from dynfed.nodes import ClientProfile, EdgeProfile
from dynfed.training import MODE_SPECS, _simulate_round


def _profiles() -> tuple[list[ClientProfile], dict[int, list[ClientProfile]], dict[int, EdgeProfile]]:
    clients = [
        ClientProfile(client_id=0, edge_id=0, samples=150, compute_factor=1.0),
        ClientProfile(client_id=1, edge_id=1, samples=150, compute_factor=1.0),
    ]
    clients_by_edge = {0: [clients[0]], 1: [clients[1]]}
    edges = {0: EdgeProfile(0, 1.0), 1: EdgeProfile(1, 1.0)}
    return clients, clients_by_edge, edges


def _config(mode: str) -> ExperimentConfig:
    return ExperimentConfig(
        topology=TopologyConfig(num_clients=2, num_edges=2, aggregation_fraction=1.0),
        runtime=RuntimeConfig(
            rounds=1,
            seed=3,
            slow_client_rate=0.0,
            network_jitter=0.0,
            edge_aggregation_beta=0.1,
            edge_aggregation_fixed=0.5,
            cloud_aggregation_beta=0.2,
            cloud_aggregation_fixed=0.7,
        ),
        mode=ModeConfig(name=mode),
    )


def test_legacy_edge_only_round_stops_after_payload_based_edge_aggregation() -> None:
    clients, clients_by_edge, edges = _profiles()
    result = _simulate_round(
        round_idx=0,
        start_time=0.0,
        spec=MODE_SPECS["LIE"],
        clients=clients,
        clients_by_edge=clients_by_edge,
        edge_by_id=edges,
        config=_config("LIE"),
        rng=random.Random(11),
    )

    edge_events = [event for event in result["events"] if event["event_type"] == "edge_ready"]
    assert len(edge_events) == 2
    assert not any(event["event_type"].startswith("cloud_aggregate") for event in result["events"])
    assert result["aggregation_time"] == 2 * (0.1 * 4.0 + 0.5)
    assert all(event["effective_payload"] == 4.0 for event in edge_events)


def test_legacy_lieiic_uses_direct_cloud_buffer_and_admitted_payload() -> None:
    clients, clients_by_edge, edges = _profiles()
    result = _simulate_round(
        round_idx=0,
        start_time=0.0,
        spec=MODE_SPECS["LIEIIC"],
        clients=clients,
        clients_by_edge=clients_by_edge,
        edge_by_id=edges,
        config=_config("LIEIIC"),
        rng=random.Random(13),
    )

    cloud_event = next(
        event for event in result["events"] if event["event_type"] == "cloud_aggregate_direct"
    )
    assert not any(event["event_type"] == "edge_ready" for event in result["events"])
    assert cloud_event["effective_payload"] == 8.0
    assert cloud_event["aggregation_time"] == 0.2 * 8.0 + 0.7
    assert result["aggregation_time"] == cloud_event["aggregation_time"]
