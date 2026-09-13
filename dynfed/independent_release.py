"""Fixed-roster client-replacement accounting for independent contributions.

This contract does not prove independence or authorize data-dependent selection.
Transport, clipping and adding noise before decryption remain caller duties.
"""
from __future__ import annotations

import math

from .privacy import PrivacyAccountant, calibrate_gaussian_noise


class IndependentReleaseAccount:
    def __init__(self, counts, clip_norm, epsilon, delta, horizon):
        self.counts = tuple(counts)
        if (not self.counts or any(type(n) is not int or n <= 0 for n in self.counts)
                or not math.isfinite(clip_norm) or clip_norm <= 0
                or type(horizon) is not int or horizon <= 0):
            raise ValueError("Positive fixed counts, clipping and release horizon required")
        self.weights = tuple(n / sum(self.counts) for n in self.counts)
        self.clip_norm = float(clip_norm)
        self.sensitivity = 2 * self.clip_norm * max(self.weights)
        self.multiplier = calibrate_gaussian_noise(epsilon, delta, horizon)
        self.noise_std = self.multiplier * self.sensitivity
        self.horizon = horizon
        self.ledger = PrivacyAccountant(epsilon, delta)
        self.next_round = 0


    def state_dict(self):
        return {
            "counts": list(self.counts),
            "clip_norm": self.clip_norm,
            "horizon": self.horizon,
            "multiplier": self.multiplier,
            "next_round": self.next_round,
            "ledger": self.ledger.state_dict(),
        }

    def load_state_dict(self, state):
        if tuple(int(value) for value in state.get("counts", ())) != self.counts:
            raise ValueError("Release-account cohort mismatch")
        if int(state.get("horizon", -1)) != self.horizon:
            raise ValueError("Release-account horizon mismatch")
        if not math.isclose(
            float(state.get("clip_norm", float("nan"))),
            self.clip_norm,
            rel_tol=1e-12,
            abs_tol=1e-12,
        ):
            raise ValueError("Release-account clip norm mismatch")
        if not math.isclose(float(state.get("multiplier", float("nan"))), self.multiplier, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Release-account calibration mismatch")
        self.ledger = PrivacyAccountant.from_state_dict(state["ledger"])
        self.next_round = int(state["next_round"])
        if not 0 <= self.next_round <= self.horizon:
            raise ValueError("Invalid saved release round")

    def check(self, round_index, client_ids):
        ids = tuple(client_ids)
        if (type(round_index) is not int or round_index != self.next_round
                or round_index >= self.horizon):
            raise ValueError("Release round is stale, skipped or beyond the horizon")
        if len(ids) != len(self.counts) or set(ids) != set(range(len(self.counts))):
            raise ValueError("Changed participant cohort needs a separately calibrated protocol")
        if not self.ledger.can_add_event(self.multiplier):
            raise ValueError("Privacy budget exhausted")

    def reserve(self, round_index, client_ids):
        """Spend before exposing a release; failures conservatively retain the spend."""
        self.check(round_index, client_ids)
        self.ledger.add_event(self.multiplier)
        self.next_round += 1
        return self.ledger.current_epsilon()
