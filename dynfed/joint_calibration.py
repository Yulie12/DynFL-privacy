from __future__ import annotations

"""Joint update-space calibration for the Sample-DP selector.

This module deliberately does *not* linearize the training map.  A paired
clean/private run is treated as one sample of the nonlinear trajectory error

    D_i = u_{i,s} - u_{i,0}.

The calibration table stores empirical estimates of E[D_i] and Tr Cov(D_i),
plus the mean clean update used by the selection-time joint objective.  The
online selector only reads this table; it never performs paired training runs.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import torch


CALIBRATION_FORMAT = "dynfl_joint_update_calibration_v1"


def flatten_state_difference(
    state_diff: Mapping[str, Mapping[str, torch.Tensor]],
) -> torch.Tensor:
    """Flatten one nested {part: {parameter: tensor}} update deterministically."""
    parts: list[torch.Tensor] = []
    ordered_parts = [name for name in ("end", "edge") if name in state_diff]
    ordered_parts.extend(sorted(name for name in state_diff if name not in {"end", "edge"}))
    for part_name in ordered_parts:
        for name in sorted(state_diff.get(part_name, {})):
            parts.append(
                state_diff[part_name][name]
                .detach()
                .to(dtype=torch.float64, device="cpu")
                .reshape(-1)
            )
    if not parts:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat(parts)


def as_update_vector(value: Any) -> torch.Tensor:
    """Convert a flat tensor/array/list or nested state difference to float64 CPU."""
    if isinstance(value, Mapping):
        return flatten_state_difference(value)
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float64, device="cpu").reshape(-1)
    return torch.as_tensor(value, dtype=torch.float64, device="cpu").reshape(-1)


@dataclass(frozen=True)
class JointCalibrationEntry:
    """Empirical conditional moments for one (state, client?, mode) cell."""

    clean_update_mean: torch.Tensor
    bias_mean: torch.Tensor
    variance_trace: float
    sample_count: int

    def __post_init__(self) -> None:
        clean = as_update_vector(self.clean_update_mean)
        bias = as_update_vector(self.bias_mean)
        if clean.numel() != bias.numel():
            raise ValueError("clean_update_mean and bias_mean must have the same dimension")
        if int(self.sample_count) < 1:
            raise ValueError("sample_count must be positive")
        if float(self.variance_trace) < -1e-12:
            raise ValueError("variance_trace must be non-negative")
        object.__setattr__(self, "clean_update_mean", clean)
        object.__setattr__(self, "bias_mean", bias)
        object.__setattr__(self, "variance_trace", max(float(self.variance_trace), 0.0))
        object.__setattr__(self, "sample_count", int(self.sample_count))


class PairedUpdateMomentAccumulator:
    """Streaming estimator for paired clean/private nonlinear trajectory errors.

    For each matched trial m, provide u_0^(m) and u_s^(m).  The accumulator
    never assumes a linear relation between primitive DP noise and the final
    update.  It directly observes D^(m) = u_s^(m) - u_0^(m).

    Only O(d) memory is used: the clean mean, D mean, and the scalar Welford
    second-moment accumulator whose unbiased normalization estimates
    Tr(Cov(D)).
    """

    def __init__(self) -> None:
        self.count = 0
        self._clean_mean: torch.Tensor | None = None
        self._diff_mean: torch.Tensor | None = None
        self._diff_m2_trace = 0.0

    def update(self, clean_update: Any, private_update: Any) -> None:
        clean = as_update_vector(clean_update)
        private = as_update_vector(private_update)
        if clean.numel() != private.numel():
            raise ValueError("paired clean/private updates must have the same dimension")
        diff = private - clean

        if self.count == 0:
            self.count = 1
            self._clean_mean = clean.clone()
            self._diff_mean = diff.clone()
            self._diff_m2_trace = 0.0
            return

        assert self._clean_mean is not None
        assert self._diff_mean is not None
        if clean.numel() != self._clean_mean.numel():
            raise ValueError("all paired updates in one calibration cell must share a dimension")

        next_count = self.count + 1
        self._clean_mean += (clean - self._clean_mean) / float(next_count)

        delta = diff - self._diff_mean
        self._diff_mean += delta / float(next_count)
        delta2 = diff - self._diff_mean
        self._diff_m2_trace += float(torch.dot(delta, delta2).item())
        self.count = next_count

    def finalize(self) -> JointCalibrationEntry:
        if self.count < 1 or self._clean_mean is None or self._diff_mean is None:
            raise ValueError("cannot finalize an empty paired calibration accumulator")
        variance_trace = (
            self._diff_m2_trace / float(self.count - 1)
            if self.count > 1
            else 0.0
        )
        return JointCalibrationEntry(
            clean_update_mean=self._clean_mean.clone(),
            bias_mean=self._diff_mean.clone(),
            variance_trace=variance_trace,
            sample_count=self.count,
        )


class JointCalibrationCaptureSession:
    """Accumulate matched trajectory pairs and materialize a selector table.

    The session is intended for an *offline or periodic calibration pass*.  It
    never participates in online mode selection.  Each observation is one
    paired nonlinear trajectory sample with shared initial state / minibatch
    schedule and independent DP randomness.

    When ``include_mode_fallback`` is true, client-specific moment estimates
    are combined into a mode-level fallback cell.  The fallback bias/clean mean
    are weighted client means, while its variance trace is the weighted mean of
    *within-client* DP-RNG variance traces (not a pooled variance that would
    incorrectly absorb between-client bias heterogeneity).
    """

    def __init__(
        self,
        *,
        metadata: Mapping[str, Any] | None = None,
        include_mode_fallback: bool = True,
    ) -> None:
        self.metadata: dict[str, Any] = dict(metadata or {})
        self.include_mode_fallback = bool(include_mode_fallback)
        self._accumulators: dict[
            tuple[str, str, int | None], PairedUpdateMomentAccumulator
        ] = {}
        self._ideal_updates: dict[str, torch.Tensor] = {}

    def add_pair(
        self,
        *,
        state_key: str,
        mode: str,
        clean_update: Any,
        private_update: Any,
        client_id: int | None,
    ) -> None:
        state = JointUpdateCalibrationTable._normalize_state_key(state_key)
        normalized_client = None if client_id is None else int(client_id)
        key = (state, str(mode), normalized_client)
        self._accumulators.setdefault(
            key, PairedUpdateMomentAccumulator()
        ).update(clean_update, private_update)

    def set_ideal_update(self, update: Any, *, state_key: str) -> None:
        state = JointUpdateCalibrationTable._normalize_state_key(state_key)
        self._ideal_updates[state] = as_update_vector(update).clone()

    @property
    def pair_count(self) -> int:
        return sum(
            accumulator.count
            for (state, mode, client_id), accumulator in self._accumulators.items()
            if client_id is not None
        )

    @property
    def cell_count(self) -> int:
        return len(self._accumulators)

    def build_table(self, *, min_trials: int = 1) -> "JointUpdateCalibrationTable":
        if int(min_trials) < 1:
            raise ValueError("min_trials must be positive")
        table = JointUpdateCalibrationTable(
            metadata={
                **self.metadata,
                "capture_pair_count": int(self.pair_count),
                "mode_fallback_is_population_proxy": bool(
                    self.include_mode_fallback
                ),
                "mode_fallback_variance_semantics": (
                    "weighted_mean_of_within_client_variance_traces"
                    if self.include_mode_fallback
                    else "disabled"
                ),
            }
        )

        finalized: dict[
            tuple[str, str, int | None], JointCalibrationEntry
        ] = {}
        for key, accumulator in self._accumulators.items():
            if accumulator.count >= int(min_trials):
                finalized[key] = accumulator.finalize()

        for (state, mode, client_id), entry in sorted(
            finalized.items(),
            key=lambda item: (
                item[0][0],
                item[0][1],
                -1 if item[0][2] is None else item[0][2],
            ),
        ):
            table.add_entry(
                state_key=state,
                mode=mode,
                client_id=client_id,
                entry=entry,
            )

        # Build mode-level fallback cells from already-estimated *within-client*
        # moments.  Do not pool raw D samples across clients: that would add
        # between-client bias heterogeneity to Tr(V_i), which is not the
        # conditional DP-RNG variance used by the selector formula.
        if self.include_mode_fallback:
            grouped: dict[tuple[str, str], list[JointCalibrationEntry]] = {}
            for (state, mode, client_id), entry in finalized.items():
                if client_id is None:
                    continue
                grouped.setdefault((state, mode), []).append(entry)
            for (state, mode), entries in grouped.items():
                if (state, mode, None) in finalized:
                    continue
                total_weight = float(sum(entry.sample_count for entry in entries))
                if total_weight <= 0.0:
                    continue
                dimension = entries[0].clean_update_mean.numel()
                if any(entry.clean_update_mean.numel() != dimension for entry in entries):
                    raise ValueError(
                        "cannot build a mode fallback from calibration entries with different dimensions"
                    )
                clean = sum(
                    entry.clean_update_mean * (float(entry.sample_count) / total_weight)
                    for entry in entries
                )
                bias = sum(
                    entry.bias_mean * (float(entry.sample_count) / total_weight)
                    for entry in entries
                )
                variance_trace = sum(
                    float(entry.variance_trace) * (float(entry.sample_count) / total_weight)
                    for entry in entries
                )
                table.add_entry(
                    state_key=state,
                    mode=mode,
                    client_id=None,
                    entry=JointCalibrationEntry(
                        clean_update_mean=clean,
                        bias_mean=bias,
                        variance_trace=variance_trace,
                        sample_count=int(total_weight),
                    ),
                )

        for state, update in self._ideal_updates.items():
            table.set_ideal_update(update, state_key=state)
        return table

    def save(
        self,
        path: str | Path,
        *,
        min_trials: int = 1,
    ) -> Path:
        return self.build_table(min_trials=min_trials).save(path)


class JointUpdateCalibrationTable:
    """State-conditioned empirical proxy table used by the online selector.

    Entries may be client-specific or mode-level.  Lookup first tries the exact
    client and then a mode-level fallback.  State lookup tries the requested
    key, then the nearest available ``round:<int>`` key, then ``default``.
    """

    def __init__(self, *, metadata: Mapping[str, Any] | None = None) -> None:
        self.metadata: dict[str, Any] = dict(metadata or {})
        self._entries: dict[tuple[str, str, int | None], JointCalibrationEntry] = {}
        self._ideal_updates: dict[str, torch.Tensor] = {}

    @staticmethod
    def _normalize_state_key(state_key: str | None) -> str:
        text = str(state_key or "default").strip()
        return text or "default"

    def add_entry(
        self,
        *,
        mode: str,
        entry: JointCalibrationEntry,
        state_key: str = "default",
        client_id: int | None = None,
    ) -> None:
        key = (self._normalize_state_key(state_key), str(mode), None if client_id is None else int(client_id))
        self._entries[key] = entry

    def set_ideal_update(self, update: Any, *, state_key: str = "default") -> None:
        self._ideal_updates[self._normalize_state_key(state_key)] = as_update_vector(update).clone()

    @staticmethod
    def _round_number(state_key: str) -> int | None:
        if not state_key.startswith("round:"):
            return None
        try:
            return int(state_key.split(":", 1)[1])
        except ValueError:
            return None

    def _state_candidates(self, state_key: str | None) -> tuple[str, ...]:
        requested = self._normalize_state_key(state_key)
        candidates: list[str] = [requested]
        requested_round = self._round_number(requested)
        if requested_round is not None:
            available = {
                key[0]
                for key in self._entries
                if self._round_number(key[0]) is not None
            } | {
                key for key in self._ideal_updates if self._round_number(key) is not None
            }
            if available:
                nearest = min(
                    available,
                    key=lambda key: (abs(int(self._round_number(key)) - requested_round), int(self._round_number(key))),
                )
                if nearest not in candidates:
                    candidates.append(nearest)
        if "default" not in candidates:
            candidates.append("default")
        return tuple(candidates)

    def lookup(
        self,
        *,
        mode: str,
        state_key: str = "default",
        client_id: int | None = None,
    ) -> JointCalibrationEntry:
        for state in self._state_candidates(state_key):
            if client_id is not None:
                exact = self._entries.get((state, str(mode), int(client_id)))
                if exact is not None:
                    return exact
            generic = self._entries.get((state, str(mode), None))
            if generic is not None:
                return generic
        raise KeyError(
            f"no joint calibration entry for mode={mode!r}, client_id={client_id!r}, state={state_key!r}"
        )

    def ideal_update(self, *, state_key: str = "default") -> torch.Tensor:
        for state in self._state_candidates(state_key):
            value = self._ideal_updates.get(state)
            if value is not None:
                return value.clone()
        raise KeyError(f"no ideal/reference update for state={state_key!r}")

    @property
    def entry_count(self) -> int:
        return len(self._entries)

    def to_payload(self) -> dict[str, Any]:
        entries = []
        for (state, mode, client_id), entry in sorted(
            self._entries.items(),
            key=lambda item: (item[0][0], item[0][1], -1 if item[0][2] is None else item[0][2]),
        ):
            entries.append(
                {
                    "state_key": state,
                    "mode": mode,
                    "client_id": client_id,
                    "clean_update_mean": entry.clean_update_mean,
                    "bias_mean": entry.bias_mean,
                    "variance_trace": float(entry.variance_trace),
                    "sample_count": int(entry.sample_count),
                }
            )
        return {
            "format": CALIBRATION_FORMAT,
            "metadata": dict(self.metadata),
            "entries": entries,
            "ideal_updates": {key: value for key, value in self._ideal_updates.items()},
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "JointUpdateCalibrationTable":
        if payload.get("format") != CALIBRATION_FORMAT:
            raise ValueError(
                f"unsupported calibration format {payload.get('format')!r}; expected {CALIBRATION_FORMAT!r}"
            )
        table = cls(metadata=payload.get("metadata") or {})
        for raw in payload.get("entries") or []:
            table.add_entry(
                mode=str(raw["mode"]),
                state_key=str(raw.get("state_key", "default")),
                client_id=raw.get("client_id"),
                entry=JointCalibrationEntry(
                    clean_update_mean=raw["clean_update_mean"],
                    bias_mean=raw["bias_mean"],
                    variance_trace=float(raw["variance_trace"]),
                    sample_count=int(raw["sample_count"]),
                ),
            )
        for state_key, update in (payload.get("ideal_updates") or {}).items():
            table.set_ideal_update(update, state_key=str(state_key))
        return table

    def save(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_payload(), output)
        return output

    @classmethod
    def load(cls, path: str | Path) -> "JointUpdateCalibrationTable":
        source = Path(path)
        if not source.exists():
            raise FileNotFoundError(source)
        try:
            payload = torch.load(source, map_location="cpu", weights_only=True)
        except TypeError:  # older torch versions
            payload = torch.load(source, map_location="cpu")
        if not isinstance(payload, Mapping):
            raise ValueError("joint calibration file must contain a mapping payload")
        return cls.from_payload(payload)
