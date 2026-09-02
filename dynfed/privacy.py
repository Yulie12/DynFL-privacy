from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Iterable


OBJECT_SIZES = {
    "emb": 1.6,
    "label": 0.2,
    "grad": 1.6,
    "upd": 4.0,
    "weakemb": 3.0,
    "strongemb": 5.5,
    "pseudo_label": 0.4,
}

PRIVACY_ALPHA = {
    "none": 1.0,
    "dp": 1.0,
    "he2": 3.2,
    "he3": 24.01,
}

PRIVACY_BASE_TIME = {
    "none": 0.0,
    "dp": 0.04,
    "he2": 0.35,
    "he3": 1.53,
}

_RDP_ALPHAS: list[float] = []
for i in range(2, 20):
    _RDP_ALPHAS.append(1 + i * 0.5)  # 2.0, 2.5, ..., 10.5
_RDP_ALPHAS.extend(range(11, 65))
_RDP_ALPHAS.extend([128, 256, 512, 1024, 2048, 4096, 8192])


def _validate_dp_parameters(noise_multiplier: float, delta: float) -> None:
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be finite and positive")
    if not math.isfinite(delta) or not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")


def gaussian_rdp(
    noise_multiplier: float,
    event_count: int = 1,
    *,
    orders: Iterable[float] = _RDP_ALPHAS,
) -> dict[float, float]:
    """Exact RDP of repeated, non-subsampled Gaussian mechanisms.

    ``noise_multiplier`` is the Gaussian standard deviation divided by the
    mechanism sensitivity. RDP composes additively across adaptive releases.
    """
    if event_count < 0:
        raise ValueError("event_count must be non-negative")
    if event_count == 0:
        return {float(order): 0.0 for order in orders}
    _validate_dp_parameters(noise_multiplier, 1e-5)
    scale = float(event_count) / (2.0 * noise_multiplier * noise_multiplier)
    return {float(order): float(order) * scale for order in orders}


def epsilon_from_rdp(rdp_state: dict[float, float], delta: float) -> float:
    """Convert one RDP curve to an ``(epsilon, delta)`` guarantee."""
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")
    if not rdp_state or all(value <= 0.0 for value in rdp_state.values()):
        return 0.0
    return min(
        value + math.log(1.0 / delta) / (order - 1.0)
        for order, value in rdp_state.items()
        if order > 1.0
    )


def compute_event_epsilon(noise_multiplier: float, delta: float) -> float:
    """Return the RDP-accounted epsilon of one Gaussian release."""
    _validate_dp_parameters(noise_multiplier, delta)
    return epsilon_from_rdp(gaussian_rdp(noise_multiplier), delta)


def calibrate_gaussian_noise(
    target_epsilon: float,
    delta: float,
    event_count: int,
) -> float:
    """Calibrate Gaussian noise for a fixed total privacy target."""
    if not math.isfinite(target_epsilon) or target_epsilon <= 0.0:
        raise ValueError("target_epsilon must be finite and positive")
    if event_count <= 0:
        raise ValueError("event_count must be positive")
    if not 0.0 < delta < 1.0:
        raise ValueError("delta must be in (0, 1)")

    def composed_epsilon(noise_multiplier: float) -> float:
        return epsilon_from_rdp(
            gaussian_rdp(noise_multiplier, event_count),
            delta,
        )

    low = 1e-6
    high = 1.0
    while composed_epsilon(high) > target_epsilon:
        high *= 2.0
        if high > 1e9:
            raise ValueError(
                "target_epsilon is below the resolution supported by the RDP order grid"
            )
    for _ in range(100):
        midpoint = (low + high) / 2.0
        if composed_epsilon(midpoint) > target_epsilon:
            low = midpoint
        else:
            high = midpoint
    return high


class PrivacyAccountant:
    """RDP accountant for repeated non-subsampled Gaussian releases."""

    def __init__(self, budget: float, delta: float = 1e-5):
        if not math.isfinite(budget) or budget <= 0.0:
            raise ValueError("budget must be finite and positive")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be in (0, 1)")
        self.budget = float(budget)
        self.delta = float(delta)
        self._rdp_state: dict[float, float] = {order: 0.0 for order in _RDP_ALPHAS}

    def add_event(self, noise_multiplier: float) -> None:
        self.add_events(noise_multiplier, 1)

    def add_events(self, noise_multiplier: float, event_count: int) -> None:
        increment = gaussian_rdp(noise_multiplier, event_count)
        for order, value in increment.items():
            self._rdp_state[order] += value

    def current_epsilon(self) -> float:
        return epsilon_from_rdp(self._rdp_state, self.delta)

    def epsilon_after(self, noise_multiplier: float, event_count: int = 1) -> float:
        increment = gaussian_rdp(noise_multiplier, event_count)
        projected = {
            order: self._rdp_state[order] + increment[order]
            for order in self._rdp_state
        }
        return epsilon_from_rdp(projected, self.delta)

    def can_add_event(self, noise_multiplier: float) -> bool:
        return self.can_add_events(noise_multiplier, 1)

    def can_add_events(self, noise_multiplier: float, event_count: int) -> bool:
        return self.epsilon_after(noise_multiplier, event_count) <= self.budget + 1e-12

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.budget - self.current_epsilon())

    def state_dict(self) -> dict[str, Any]:
        return {
            "budget": self.budget,
            "delta": self.delta,
            "rdp_state": {str(order): value for order, value in self._rdp_state.items()},
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "PrivacyAccountant":
        accountant = cls(budget=float(state["budget"]), delta=float(state["delta"]))
        saved = state.get("rdp_state", {})
        accountant._rdp_state = {
            order: float(saved.get(str(order), saved.get(order, 0.0)))
            for order in _RDP_ALPHAS
        }
        return accountant


@dataclass(frozen=True)
class PrivacyProjection:
    feature_epsilon_before: float
    feature_epsilon_after: float
    update_epsilon_before: float
    update_epsilon_after: float
    feature_events: int
    update_events: int

    @property
    def feature_epsilon_increment(self) -> float:
        return max(0.0, self.feature_epsilon_after - self.feature_epsilon_before)

    @property
    def update_epsilon_increment(self) -> float:
        return max(0.0, self.update_epsilon_after - self.update_epsilon_before)


class ClientPrivacyLedger:
    """Separate record-level feature and client-level update accountants."""

    def __init__(
        self,
        *,
        feature_budget: float,
        update_budget: float,
        delta: float,
        feature_noise_multiplier: float,
        update_noise_multiplier: float,
    ):
        self.feature = PrivacyAccountant(feature_budget, delta)
        self.update = PrivacyAccountant(update_budget, delta)
        self.feature_noise_multiplier = float(feature_noise_multiplier)
        self.update_noise_multiplier = float(update_noise_multiplier)
        _validate_dp_parameters(self.feature_noise_multiplier, delta)
        _validate_dp_parameters(self.update_noise_multiplier, delta)

    def project(self, feature_events: int, update_events: int) -> PrivacyProjection:
        feature_before = self.feature.current_epsilon()
        update_before = self.update.current_epsilon()
        return PrivacyProjection(
            feature_epsilon_before=feature_before,
            feature_epsilon_after=self.feature.epsilon_after(
                self.feature_noise_multiplier,
                feature_events,
            ),
            update_epsilon_before=update_before,
            update_epsilon_after=self.update.epsilon_after(
                self.update_noise_multiplier,
                update_events,
            ),
            feature_events=feature_events,
            update_events=update_events,
        )

    def can_apply(self, projection: PrivacyProjection) -> bool:
        return (
            projection.feature_epsilon_after <= self.feature.budget + 1e-12
            and projection.update_epsilon_after <= self.update.budget + 1e-12
        )

    def add(self, feature_events: int, update_events: int) -> PrivacyProjection:
        projection = self.project(feature_events, update_events)
        if not self.can_apply(projection):
            raise ValueError("DP event would exceed the configured privacy target")
        self.feature.add_events(self.feature_noise_multiplier, feature_events)
        self.update.add_events(self.update_noise_multiplier, update_events)
        return projection

    @property
    def remaining_budget(self) -> float:
        return min(self.feature.remaining_budget, self.update.remaining_budget)

    def state_dict(self) -> dict[str, Any]:
        return {
            "feature": self.feature.state_dict(),
            "update": self.update.state_dict(),
            "feature_noise_multiplier": self.feature_noise_multiplier,
            "update_noise_multiplier": self.update_noise_multiplier,
        }

    @classmethod
    def from_state_dict(cls, state: dict[str, Any]) -> "ClientPrivacyLedger":
        feature = PrivacyAccountant.from_state_dict(state["feature"])
        update = PrivacyAccountant.from_state_dict(state["update"])
        ledger = cls(
            feature_budget=feature.budget,
            update_budget=update.budget,
            delta=feature.delta,
            feature_noise_multiplier=float(state["feature_noise_multiplier"]),
            update_noise_multiplier=float(state["update_noise_multiplier"]),
        )
        ledger.feature = feature
        ledger.update = update
        return ledger


def normalize_mechanism(mechanism: str) -> str:
    value = mechanism.strip().lower()
    if value not in PRIVACY_ALPHA:
        raise ValueError(f"Unsupported privacy mechanism: {mechanism}")
    return value


def protected_size(objects: list[str], mechanism: str) -> float:
    mechanism = normalize_mechanism(mechanism)
    raw_size = sum(OBJECT_SIZES[item] for item in objects)
    return raw_size * PRIVACY_ALPHA[mechanism]


def privacy_processing_time(objects: list[str], mechanism: str, epsilon: float) -> float:
    mechanism = normalize_mechanism(mechanism)
    if not objects:
        return 0.0
    eps_factor = 1.0
    if mechanism == "dp":
        eps_factor = 1.0 + 1.0 / max(float(epsilon), 1e-6)
    return len(objects) * PRIVACY_BASE_TIME[mechanism] * eps_factor


def utility_penalty(mechanism: str, epsilon: float) -> float:
    mechanism = normalize_mechanism(mechanism)
    if mechanism == "dp":
        return min(0.12, 0.01 + 0.08 / max(float(epsilon), 0.25))
    if mechanism == "he2":
        return 0.018
    if mechanism == "he3":
        return 0.0
    return 0.0


def privacy_score(mechanism: str, epsilon: float) -> float:
    mechanism = normalize_mechanism(mechanism)
    if mechanism == "dp":
        return 1.0 / (1.0 + math.log1p(max(float(epsilon), 0.0)))
    if mechanism == "he2":
        return 0.82
    if mechanism == "he3":
        return 0.88
    return 0.0
