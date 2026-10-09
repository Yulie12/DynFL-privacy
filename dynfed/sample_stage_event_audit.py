"""Pure, fail-closed stage event reconciliation for planned Sample-DP training.

This module does NOT reserve/commit privacy budgets and does NOT authorize
hierarchical execution. The runtime must reserve a conservative budget before
any release, identify all released objects, and pass per-stage worker evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping


_COUNTERS = (
    ("sample_embedding_events", "feature_dp_release_batches"),
    ("sample_label_grad_events", "sample_label_grad_dp_release_batches"),
    ("sample_optimizer_events", "sample_dp_optimizer_steps"),
)


def _nonnegative_integer(value: Any, *, field: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field}: bool is not a valid event counter")
    try:
        converted = int(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field}: missing or invalid event counter") from exc
    if converted < 0 or converted != value:
        raise ValueError(f"{field}: expected nonnegative integer, got {value!r}")
    return converted


def _positive_finite(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field}: boolean is not a valid DP parameter")
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(f"{field}: missing or invalid DP parameter") from exc
    if not math.isfinite(number) or number <= 0.0:
        raise ValueError(f"{field}: DP parameter must be positive and finite")
    return number


@dataclass(frozen=True)
class SampleEventCounts:
    embedding: int
    label_grad: int
    optimizer: int

    def __post_init__(self) -> None:
        for field in ("embedding", "label_grad", "optimizer"):
            _nonnegative_integer(getattr(self, field), field=field)

    @classmethod
    def from_worker(cls, diagnostic: Mapping[str, Any]) -> "SampleEventCounts":
        values = []
        for _candidate_name, worker_name in _COUNTERS:
            if worker_name not in diagnostic:
                raise ValueError(f"worker event audit missing {worker_name}")
            values.append(_nonnegative_integer(diagnostic[worker_name], field=worker_name))
        return cls(*values)

    @classmethod
    def from_candidate(cls, candidate: Any) -> "SampleEventCounts":
        values = []
        for candidate_name, _worker_name in _COUNTERS:
            if not hasattr(candidate, candidate_name):
                raise ValueError(f"candidate event audit missing {candidate_name}")
            values.append(_nonnegative_integer(getattr(candidate, candidate_name), field=candidate_name))
        return cls(*values)

    def __add__(self, other: "SampleEventCounts") -> "SampleEventCounts":
        return SampleEventCounts(
            self.embedding + other.embedding,
            self.label_grad + other.label_grad,
            self.optimizer + other.optimizer,
        )


def uniform_sample_stage_plan(*, candidate: Any, stage_count: int) -> dict[int, SampleEventCounts]:
    """Partition the *charged upper bound* across real worker invocations.

    The selection candidate has cumulative events; a worker runs one training
    block each time. Reject non-divisible counters rather than rounding away
    privacy events. This function does NOT authorize a release or a retry.
    """
    count = _nonnegative_integer(stage_count, field="stage_count")
    if count < 1:
        raise ValueError("Sample-DP stage_count must be positive")
    total = SampleEventCounts.from_candidate(candidate)
    values = (total.embedding, total.label_grad, total.optimizer)
    if any(value % count for value in values):
        raise ValueError(
            f"Sample-DP candidate event totals {total} are not divisible "
            f"by {count} worker stages; fail closed"
        )
    per_stage = SampleEventCounts(*(value // count for value in values))
    return {stage: per_stage for stage in range(count)}


class SampleStageEventAudit:
    """Reconcile actual per-stage events to an explicitly scheduled event plan.

    A schedule must cover EVERY execution, including retries. Event count
    equality is a *runtime consistency check*, not a Sample-DP proof.
    """

    def __init__(
        self,
        *,
        client_id: int,
        candidate: Any,
        scheduled_stages: Mapping[int, SampleEventCounts],
        require_runtime_parameters: bool = False,
        feature_clip_norm: float | None = None,
        optimizer_clip_norm: float | None = None,
    ) -> None:
        if not scheduled_stages:
            raise ValueError("Sample-DP stage plan cannot be empty")
        if set(scheduled_stages) != set(range(len(scheduled_stages))):
            raise ValueError("Sample-DP stage indices must be contiguous from zero")
        if any(not isinstance(c, SampleEventCounts) for c in scheduled_stages.values()):
            raise TypeError("stage counts must be SampleEventCounts")
        self.client_id = int(client_id)
        self.expected = SampleEventCounts.from_candidate(candidate)
        self.stages = dict(scheduled_stages)
        total = SampleEventCounts(0, 0, 0)
        for counters in self.stages.values():
            total += counters
        if total != self.expected:
            raise ValueError(
                f"Sample-DP client {self.client_id} stage plan {total} "
                f"!= candidate charged events {self.expected}"
            )
        self._observed: dict[int, SampleEventCounts] = {}
        # These are actual Edge aggregation authorizations, not predicted
        # privacy events. A receipt may authorize at most one such operation.
        self._edge_aggregated: set[int] = set()
        # Legacy count-only unit tests remain supported; production enables
        # strict verification of parameters recorded at DP execution sites.
        self.require_runtime_parameters = bool(require_runtime_parameters)
        self._expected_parameters: dict[str, float] = {}
        if self.require_runtime_parameters:
            for count, candidate_field, receipt_field in (
                (self.expected.embedding, "sample_embedding_noise_multiplier", "executed_sample_embedding_sigma"),
                (self.expected.label_grad, "sample_label_grad_noise_multiplier", "executed_sample_label_grad_sigma"),
                (self.expected.optimizer, "sample_optimizer_noise_multiplier", "executed_sample_optimizer_sigma"),
            ):
                if count:
                    self._expected_parameters[receipt_field] = _positive_finite(
                        getattr(candidate, candidate_field, None), field=candidate_field
                    )
            if self.expected.embedding or self.expected.label_grad:
                self._expected_parameters["executed_sample_feature_clip_norm"] = _positive_finite(
                    feature_clip_norm, field="feature_clip_norm"
                )
            if self.expected.optimizer:
                self._expected_parameters["executed_sample_optimizer_clip_norm"] = _positive_finite(
                    optimizer_clip_norm, field="optimizer_clip_norm"
                )

    def observe(self, *, stage: int, worker: Mapping[str, Any]) -> None:
        stage = _nonnegative_integer(stage, field="stage")
        if stage not in self.stages:
            raise RuntimeError(f"Sample-DP client {self.client_id}: unknown/retry stage {stage}")
        if stage in self._observed:
            raise RuntimeError(f"Sample-DP client {self.client_id}: duplicate stage {stage}")
        if stage != len(self._observed):
            raise RuntimeError(
                f"Sample-DP client {self.client_id}: out-of-order stage {stage}; "
                f"expected {len(self._observed)}"
            )
        actual = SampleEventCounts.from_worker(worker)
        expected = self.stages[stage]
        if actual != expected:
            raise RuntimeError(
                f"Sample-DP client {self.client_id} stage {stage}: "
                f"observed {actual}, expected {expected}; fail closed"
            )
        if self.require_runtime_parameters:
            for field, expected_value in self._expected_parameters.items():
                try:
                    observed_value = _positive_finite(worker.get(field), field=field)
                except ValueError as exc:
                    raise RuntimeError(
                        f"Sample-DP client {self.client_id} stage {stage}: "
                        f"missing/invalid runtime {field}; fail closed"
                    ) from exc
                if not math.isclose(observed_value, expected_value, rel_tol=0, abs_tol=1e-12):
                    raise RuntimeError(
                        f"Sample-DP client {self.client_id} stage {stage}: "
                        f"{field} observed={observed_value}, charged={expected_value}; fail closed"
                    )
        self._observed[stage] = actual

    def authorize_edge_aggregate(self, *, stage: int) -> None:
        """Authorize one Edge aggregation only after its private worker receipt.

        This does not prove end-to-end DP; it prevents a stale, missing or
        duplicated receipt from being used for hierarchical model feedback.
        """
        stage = _nonnegative_integer(stage, field="stage")
        if stage not in self.stages or stage not in self._observed:
            raise RuntimeError(
                f"Sample-DP client {self.client_id}: Edge aggregation stage {stage} "
                "has no audited worker receipt; fail closed"
            )
        if stage in self._edge_aggregated:
            raise RuntimeError(
                f"Sample-DP client {self.client_id}: duplicate Edge aggregation "
                f"at stage {stage}; fail closed"
            )
        if stage != len(self._edge_aggregated):
            raise RuntimeError(
                f"Sample-DP client {self.client_id}: Edge aggregation out of order, "
                f"stage={stage}, expected={len(self._edge_aggregated)}; fail closed"
            )
        self._edge_aggregated.add(stage)

    def finalize_edge_aggregations(self) -> None:
        """Require one audited Edge feedback operation per scheduled stage."""
        required = set(self.stages)
        if self._edge_aggregated != required:
            raise RuntimeError(
                f"Sample-DP client {self.client_id}: Edge aggregations "
                f"{sorted(self._edge_aggregated)} do not match stages "
                f"{sorted(required)}; fail closed"
            )

    def finalize_executed_prefix(self, *, executed_stages: int) -> SampleEventCounts:
        """Audit an explicitly stopped prefix (e.g., dropped after Cloud Flow).

        Only the caller, *after* determining which clients were admitted, may
        use this for a client whose later stages were not dispatched. The
        privacy ledger retains its conservative FULL pre-dispatch charge.
        This never proves absence of hidden releases outside the worker.
        """
        count = _nonnegative_integer(executed_stages, field="executed_stages")
        if count < 1 or count > len(self.stages):
            raise ValueError("executed_stages outside the scheduled stage range")
        expected = set(range(count))
        actual = set(self._observed)
        if actual != expected:
            raise RuntimeError(
                f"Sample-DP client {self.client_id}: stage receipt prefix mismatch "
                f"expected {sorted(expected)}, observed {sorted(actual)}; fail closed"
            )
        total = SampleEventCounts(0, 0, 0)
        for index in range(count):
            total += self._observed[index]
        return total

    def finalize(self) -> SampleEventCounts:
        missing = set(self.stages) - set(self._observed)
        if missing:
            raise RuntimeError(
                f"Sample-DP client {self.client_id}: missing stage observations "
                f"{sorted(missing)}; fail closed"
            )
        total = SampleEventCounts(0, 0, 0)
        for counters in self._observed.values():
            total += counters
        if total != self.expected:
            raise RuntimeError("Sample-DP total event count mismatch; fail closed")
        return total
