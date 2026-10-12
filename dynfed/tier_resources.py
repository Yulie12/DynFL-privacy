"""Whole-profile three-tier resource accounting; not a physical VRAM measurement.

L computation/memory is checked by the existing per-client admission. This
module checks independently shared Edge and Cloud CPU, and packet/model-buffer
memory in MB. Edge and Cloud capacities must NEVER be pooled with L capacity.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .training import MODE_SPECS

_EDGE_MODES = frozenset({"LIE", "LIIE", "LIEIIC", "LIEIIIC", "LIIEIIIC"})
_CLOUD_MODES = frozenset({"LIC", "LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC"})
_EDGE_CLOUD_MODES = frozenset({"LIEIIC", "LIEIIIC", "LIIEIIIC"})
_SPLIT_MODES = frozenset({"LIE", "LIC", "LIEIIC", "LIEIIIC"})


@dataclass(frozen=True)
class TierCapacityAudit:
    feasible: bool
    edge_cpu: dict[int, float]
    edge_memory_mb: dict[int, float]
    cloud_cpu: float
    cloud_memory_mb: float
    violations: tuple[str, ...]


def evaluate_tier_capacities(
    config: Any,
    profile: Mapping[int, Any],
    client_edges: Mapping[int, int],
    client_samples: Mapping[int, float],
) -> TierCapacityAudit:
    """Peak-buffer surrogate at an Edge group and one round-synchronous Cloud.

    CPU coefficients are normalized mode-profile units (samples / 150); they
    are not CPU-seconds. Memory uses configurable update/embedding packet MB,
    with one shared aggregation workspace per active server. No model partition
    is charged to the client for E/C aggregation; no L capacity is borrowed.

    Edge memory: 2 update buffers per active Edge + one incoming packet per
    selected client served by that Edge. Cloud memory: 2 update buffers + one
    direct packet per direct-to-Cloud client + one aggregate per (Edge, mode)
    group reaching Cloud. This is a conservative staged-buffer *proxy*, not a
    measured peak-RAM curve, and does not model concurrency-driven reuse.
    """
    for name in ("edge_cpu_limit", "cloud_cpu_limit"):
        if float(getattr(config, name)) < 0:
            raise ValueError(f"{name} must be nonnegative")
    for name in ("edge_memory_limit_mb", "cloud_memory_limit_mb"):
        value = getattr(config, name, None)
        if value is not None and float(value) < 0:
            raise ValueError(f"{name} must be nonnegative or None")
    upd_mb = max(float(config.update_payload_mb), 0.0)
    emb_mb = max(float(config.embedding_payload_mb), 0.0)
    edge_cpu: dict[int, float] = {}
    edge_buffers: dict[int, float] = {}
    cloud_cpu = 0.0
    cloud_buffers = 0.0
    cloud_edge_groups: set[tuple[int, str]] = set()
    any_cloud = False
    for cid, candidate in profile.items():
        mode = candidate.mode
        if mode == "SKIP":
            continue
        if mode not in MODE_SPECS:
            raise ValueError(f"Unknown coordination mode {mode!r}")
        spec = MODE_SPECS[mode]
        edge = int(client_edges[cid])
        cpu_scale = max(float(client_samples[cid]), 0.0) / 150.0
        if mode in _EDGE_MODES:
            edge_cpu[edge] = edge_cpu.get(edge, 0.0) + spec.edge_cpu * cpu_scale
            edge_buffers[edge] = edge_buffers.get(edge, 0.0) + (
                emb_mb if mode in _SPLIT_MODES else upd_mb
            )
        if mode in _CLOUD_MODES:
            any_cloud = True
            cloud_cpu += spec.cloud_cpu * cpu_scale
            if mode in _EDGE_CLOUD_MODES:
                cloud_edge_groups.add((edge, mode))
            else:
                cloud_buffers += emb_mb if mode in _SPLIT_MODES else upd_mb
    edge_memory_mb = {edge: 2.0 * upd_mb + buffers for edge, buffers in edge_buffers.items()}
    cloud_memory_mb = (
        2.0 * upd_mb + cloud_buffers + len(cloud_edge_groups) * upd_mb
        if any_cloud else 0.0
    )
    violations = []
    for edge, load in sorted(edge_cpu.items()):
        if load > float(config.edge_cpu_limit) + 1e-12:
            violations.append(f"E{edge}:cpu")
    for edge, memory in sorted(edge_memory_mb.items()):
        limit = getattr(config, "edge_memory_limit_mb", None)
        if limit is not None and memory > float(limit) + 1e-12:
            violations.append(f"E{edge}:memory")
    if cloud_cpu > float(config.cloud_cpu_limit) + 1e-12:
        violations.append("C:cpu")
    limit = getattr(config, "cloud_memory_limit_mb", None)
    if limit is not None and cloud_memory_mb > float(limit) + 1e-12:
        violations.append("C:memory")
    return TierCapacityAudit(
        feasible=not violations,
        edge_cpu=edge_cpu,
        edge_memory_mb=edge_memory_mb,
        cloud_cpu=cloud_cpu,
        cloud_memory_mb=cloud_memory_mb,
        violations=tuple(violations),
    )
