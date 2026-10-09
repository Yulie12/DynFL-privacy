"""Keep private Sample-DP model updates out of the Flow timing simulator.

Flow makes a scheduling/admission decision based on metadata. It must not
consume raw worker model deltas. The caller performs private-release checks
and resolves admitted worker-owned updates only after that decision.

This is an in-process boundary; it does NOT certify transmission security or
end-to-end sample differential privacy.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any


def resolve_sample_admitted_updates(
    *,
    selected_client_ids: Sequence[int],
    flow_state_diffs: Sequence[Any],
    flow_sample_counts: Sequence[int],
    worker_results: Mapping[int, Mapping[str, Any]],
    train_sample_counts: Mapping[int, int],
) -> list[Any]:
    """Resolve exact, finite worker updates after metadata-only Flow admission."""
    n = len(selected_client_ids)
    if len(flow_state_diffs) != n or len(flow_sample_counts) != n:
        raise RuntimeError("Sample-DP Flow admission has inconsistent result lengths; fail closed")
    if any(value is not None for value in flow_state_diffs):
        raise RuntimeError("Sample-DP Flow received private state differences; fail closed")
    if len(set(selected_client_ids)) != n:
        raise RuntimeError("Sample-DP Flow returned duplicate client IDs; fail closed")
    resolved = []
    for client_id, observed_count in zip(selected_client_ids, flow_sample_counts):
        if client_id not in train_sample_counts or client_id not in worker_results:
            raise RuntimeError(
                f"Sample-DP Flow admitted unknown/missing worker client {client_id}; fail closed"
            )
        expected = train_sample_counts[client_id]
        if (isinstance(observed_count, bool) or isinstance(expected, bool)
                or not isinstance(observed_count, int) or not isinstance(expected, int)
                or observed_count <= 0 or observed_count != expected):
            raise RuntimeError(
                f"Sample-DP Flow sample-count mismatch for client {client_id}; fail closed"
            )
        worker = worker_results[client_id]
        if not bool(worker.get("finite", False)):
            raise RuntimeError(
                f"Sample-DP Flow admitted non-finite worker client {client_id}; fail closed"
            )
        diff = worker.get("state_diff")
        if not isinstance(diff, Mapping) or not diff:
            raise RuntimeError(
                f"Sample-DP Flow admitted missing/invalid private update for client {client_id}; fail closed"
            )
        resolved.append(diff)
    return resolved
