"""Conservative in-memory Sample-DP charge for each dispatched worker.

This does not authorize multi-stage execution: the existing hierarchical
preflight stays in force. The caller must durably checkpoint the charged
ledgers before executing any worker (and prohibit calibration-side releases).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .privacy import SamplePrivacyLedger, SamplePrivacyProjection


@dataclass(frozen=True)
class SampleDispatchCharge:
    projection: SamplePrivacyProjection
    embedding_sigma: float
    label_grad_sigma: float
    optimizer_sigma: float | None


def _charge_kwargs(candidate: Any) -> tuple[dict[str, Any], tuple[float, float, float | None]]:
    optimizer_sigma = candidate.sample_optimizer_noise_multiplier
    if candidate.sample_optimizer_events > 0 and optimizer_sigma is None:
        raise ValueError("Sample-DP optimizer events require a noise multiplier")
    embedding_sigma = (
        candidate.sample_embedding_noise_multiplier
        if candidate.sample_embedding_noise_multiplier is not None
        else optimizer_sigma if optimizer_sigma is not None else 1.0
    )
    label_grad_sigma = (
        candidate.sample_label_grad_noise_multiplier
        if candidate.sample_label_grad_noise_multiplier is not None
        else optimizer_sigma if optimizer_sigma is not None else 1.0
    )
    return {
        "embedding_events": candidate.sample_embedding_events,
        "label_grad_events": candidate.sample_label_grad_events,
        "optimizer_events": candidate.sample_optimizer_events,
        "embedding_noise_multiplier": embedding_sigma,
        "label_grad_noise_multiplier": label_grad_sigma,
        "optimizer_noise_multiplier": optimizer_sigma if optimizer_sigma is not None else 1.0,
    }, (float(embedding_sigma), float(label_grad_sigma), optimizer_sigma)


def charge_sample_dispatch_before_worker(
    *,
    train_tasks: list[tuple[int, Any, Any, int]],
    privacy_ledgers: Mapping[int, SamplePrivacyLedger],
) -> dict[int, SampleDispatchCharge]:
    """Preflight ALL budgets, then charge ALL dispatched clients before execution.

    Once a charge is made, it MUST NOT be rolled back merely because its
    update is dropped by the flow simulator. Caller writes an atomic checkpoint
    before launching any private worker and audits all returned observations.
    """
    prepared = {}
    for client_id, candidate, _indices, _sequence in train_tasks:
        cid = int(client_id)
        if cid in prepared:
            raise RuntimeError(f"Sample-DP duplicate dispatched client {cid}")
        ledger = privacy_ledgers[cid]
        if not isinstance(ledger, SamplePrivacyLedger):
            raise TypeError(f"Sample-DP client {cid} has the wrong privacy ledger")
        kwargs, sigmas = _charge_kwargs(candidate)
        projection = ledger.project(**kwargs)
        if not ledger.can_apply(projection):
            raise ValueError(f"Sample-DP client {cid} exceeds the privacy budget before worker")
        prepared[cid] = (kwargs, sigmas)

    charged = {}
    for cid, (kwargs, sigmas) in prepared.items():
        projection = privacy_ledgers[cid].add(**kwargs)
        charged[cid] = SampleDispatchCharge(projection, *sigmas)
    return charged
