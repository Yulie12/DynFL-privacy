from __future__ import annotations

import random
from dataclasses import dataclass


@dataclass(frozen=True)
class ClientProfile:
    client_id: int
    edge_id: int
    samples: int
    compute_factor: float
    memory_capacity_factor: float = 1.0


@dataclass(frozen=True)
class EdgeProfile:
    edge_id: int
    compute_factor: float


def build_profiles(
    num_clients: int,
    num_edges: int,
    client_heterogeneity: float,
    edge_heterogeneity: float,
    seed: int,
) -> tuple[list[ClientProfile], list[EdgeProfile]]:
    rng = random.Random(seed)

    clients = []
    for client_id in range(num_clients):
        edge_id = client_id % num_edges
        compute_factor = rng.uniform(1.0 / client_heterogeneity, client_heterogeneity)
        memory_capacity_factor = rng.uniform(
            1.0 / client_heterogeneity,
            client_heterogeneity,
        )
        samples = rng.randint(80, 220)
        clients.append(
            ClientProfile(
                client_id=client_id,
                edge_id=edge_id,
                samples=samples,
                compute_factor=compute_factor,
                memory_capacity_factor=memory_capacity_factor,
            )
        )

    edges = []
    for edge_id in range(num_edges):
        compute_factor = rng.uniform(1.0 / edge_heterogeneity, edge_heterogeneity)
        edges.append(EdgeProfile(edge_id=edge_id, compute_factor=compute_factor))

    return clients, edges


def draw_runtime_multiplier(
    rng: random.Random,
    slow_rate: float,
    slow_low: float,
    slow_high: float,
) -> tuple[float, bool]:
    if rng.random() < slow_rate:
        return rng.uniform(slow_low, slow_high), True
    return rng.uniform(0.9, 1.1), False


def draw_bandwidth(rng: random.Random, base_rate: float, jitter: float) -> float:
    low = max(0.05, 1.0 - jitter)
    high = 1.0 + jitter
    return max(0.05, base_rate * rng.uniform(low, high))
