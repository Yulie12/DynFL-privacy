"""Crash-safe attempt identities for Sample-DP worker and aggregate randomness.

A successful pre-dispatch write-ahead checkpoint MUST persist the incremented
sequence before private execution. A resumed round creates another attempt,
never reusing the seed stream of a previous dispatch, even if the round number
and client assignments are unchanged.

This is a seed-uniqueness component, not a DP-accounting or release proof.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import secrets
from typing import Any, Mapping


def _valid_nonce(value: Any) -> str:
    if not isinstance(value, str) or len(value) != 32:
        raise ValueError("Sample-DP dispatch nonce must have 32 hexadecimal characters")
    try:
        bytes.fromhex(value)
    except ValueError as exc:
        raise ValueError("Sample-DP dispatch nonce must be hexadecimal") from exc
    return value.lower()


def _nonnegative_sequence(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("Sample-DP dispatch sequence must be a nonnegative integer")
    return value


@dataclass(frozen=True)
class SampleDispatchAttempt:
    run_nonce: str
    sequence: int

    def __post_init__(self) -> None:
        _valid_nonce(self.run_nonce)
        if _nonnegative_sequence(self.sequence) < 1:
            raise ValueError("Sample-DP attempt sequence must be positive")

    def seed(self, purpose: str, round_idx: int, entity: Any, stage: int = 0) -> int:
        """Stable within an attempt; distinct attempts use separate hash inputs.

        This 63-bit seed can initialize NumPy's generator and Torch RNG. Hash
        collisions are theoretically possible but negligible at experiment scale.
        """
        if not isinstance(purpose, str) or not purpose:
            raise ValueError("Sample-DP seed purpose must be a nonempty string")
        if isinstance(round_idx, bool) or int(round_idx) < 0:
            raise ValueError("Sample-DP round index must be nonnegative")
        if isinstance(stage, bool) or int(stage) < 0:
            raise ValueError("Sample-DP stage index must be nonnegative")
        payload = json.dumps(
            ["sample-dispatch-seed-v1", self.run_nonce, self.sequence,
             purpose, int(round_idx), entity, int(stage)],
            sort_keys=True, separators=(",", ":"),
        ).encode("utf-8")
        return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") & ((1 << 63) - 1)


@dataclass
class SampleDispatchIdentity:
    run_nonce: str = field(default_factory=lambda: secrets.token_hex(16))
    sequence: int = 0

    def __post_init__(self) -> None:
        self.run_nonce = _valid_nonce(self.run_nonce)
        _nonnegative_sequence(self.sequence)

    def next_attempt(self) -> SampleDispatchAttempt:
        self.sequence += 1
        return SampleDispatchAttempt(self.run_nonce, self.sequence)

    def state_dict(self) -> dict[str, Any]:
        return {"version": 1, "run_nonce": self.run_nonce, "sequence": self.sequence}

    @classmethod
    def from_state_dict(cls, state: Mapping[str, Any]) -> "SampleDispatchIdentity":
        if not isinstance(state, Mapping) or state.get("version") != 1:
            raise RuntimeError("Unsupported/missing Sample-DP dispatch identity checkpoint")
        return cls(run_nonce=_valid_nonce(state.get("run_nonce")),
                   sequence=_nonnegative_sequence(state.get("sequence")))
