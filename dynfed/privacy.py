from __future__ import annotations

import math


OBJECT_SIZES = {
    "emb": 4.0,
    "label": 0.2,
    "grad": 9.0,
    "upd": 10.0,
    "weakemb": 3.0,
    "strongemb": 5.5,
    "pseudo_label": 0.4,
}

PRIVACY_ALPHA = {
    "none": 1.0,
    "dp": 1.02,
    "he2": 3.2,
    "he3": 3.35,
}

PRIVACY_BASE_TIME = {
    "none": 0.0,
    "dp": 0.04,
    "he2": 0.35,
    "he3": 0.48,
}

_RDP_ALPHAS: list[float] = []
for i in range(2, 20):
    _RDP_ALPHAS.append(1 + i * 0.5)  # 2.0, 2.5, ..., 10.5
_RDP_ALPHAS.extend(range(11, 65))  # 11, 12, ..., 64
_RDP_ALPHAS.extend([128, 256, 512, 1024, 2048, 4096, 8192])


def compute_event_epsilon(noise_multiplier: float, delta: float) -> float:
    """RDP-based per-event epsilon for the Gaussian mechanism (first-event).

    Conservative estimate used for candidate feasibility screening.
    σ = noise_multiplier (Δf = 1 after clipping).
    """
    best = float("inf")
    for a in _RDP_ALPHAS:
        if a <= 1:
            continue
        eps = a / (2.0 * noise_multiplier * noise_multiplier) + math.log(1.0 / delta) / (a - 1)
        best = min(best, eps)
    return best


class PrivacyAccountant:
    """RDP-based privacy accountant for the Gaussian mechanism.

    Tracks Rényi divergence across a grid of α orders.
    Composition: ε_RDP(α) accumulates linearly.
    Conversion to (ε,δ): ε = min_α [ε_RDP(α) + log(1/δ)/(α-1)].
    """

    def __init__(self, budget: float, delta: float = 1e-5):
        self.budget = budget
        self.delta = delta
        self._rdp_state: dict[float, float] = {a: 0.0 for a in _RDP_ALPHAS}

    def add_event(self, noise_multiplier: float) -> None:
        """Record one Gaussian mechanism event."""
        s = noise_multiplier
        for a in _RDP_ALPHAS:
            self._rdp_state[a] += a / (2.0 * s * s)

    def current_epsilon(self) -> float:
        """Current (ε, δ)-DP guarantee."""
        best = float("inf")
        for a in _RDP_ALPHAS:
            if a <= 1:
                continue
            eps = self._rdp_state[a] + math.log(1.0 / self.delta) / (a - 1)
            best = min(best, eps)
        return best

    def can_add_event(self, noise_multiplier: float) -> bool:
        """Check if one more event stays within budget."""
        s = noise_multiplier
        best = float("inf")
        for a in _RDP_ALPHAS:
            rdp_after = self._rdp_state[a] + a / (2.0 * s * s)
            eps = rdp_after + math.log(1.0 / self.delta) / (a - 1)
            best = min(best, eps)
        return best <= self.budget + 1e-12

    @property
    def remaining_budget(self) -> float:
        return max(0.0, self.budget - self.current_epsilon())


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
        return 0.012
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
