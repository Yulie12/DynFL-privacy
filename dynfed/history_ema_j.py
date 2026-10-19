"""History-EMA J shadow evaluation.

J is a normalized squared distance between a candidate Mode's historical EMA
prediction of its Cloud contribution and the historical EMA of eligible global
Cloud updates. This is diagnostic-only: it never changes candidate selection.

Only already-produced update vectors are consumed. The caller is responsible
for attesting that the upload path was DP-protected before Cloud aggregation.
Aggregate-DP releases and E_C_upd protection are deliberately not modeled here.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch

TARGET_MODES = ("LIIE", "LIIEIIIC", "LIIC")


def flatten_update(update: Any) -> torch.Tensor:
    """Flatten a tensor or nested {'end'/'edge': {parameter: tensor}} update."""
    if isinstance(update, torch.Tensor):
        vector = update.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    elif isinstance(update, Mapping):
        parts: list[torch.Tensor] = []
        for part in sorted(update):
            value = update[part]
            if isinstance(value, Mapping):
                for name in sorted(value):
                    item = value[name]
                    if isinstance(item, torch.Tensor):
                        parts.append(item.detach().to(device="cpu", dtype=torch.float64).reshape(-1))
                    else:
                        parts.append(torch.as_tensor(item, dtype=torch.float64).reshape(-1))
            elif isinstance(value, torch.Tensor):
                parts.append(value.detach().to(device="cpu", dtype=torch.float64).reshape(-1))
            else:
                parts.append(torch.as_tensor(value, dtype=torch.float64).reshape(-1))
        vector = torch.cat(parts) if parts else torch.empty(0, dtype=torch.float64)
    else:
        vector = torch.as_tensor(update, dtype=torch.float64).reshape(-1)
    if not torch.isfinite(vector).all():
        raise ValueError("history EMA-J update contains non-finite values")
    return vector.contiguous()


def normalized_squared_deviation(candidate: torch.Tensor, reference: torch.Tensor,
                                  eps: float = 1e-12) -> float:
    """||candidate-reference||^2 / (||reference||^2 + eps)."""
    candidate = flatten_update(candidate)
    reference = flatten_update(reference)
    if candidate.numel() != reference.numel():
        raise ValueError("candidate/reference update dimensions do not match")
    if not math.isfinite(eps) or eps <= 0:
        raise ValueError("eps must be finite and positive")
    numerator = torch.sum((candidate - reference) ** 2).item()
    denominator = torch.sum(reference ** 2).item() + eps
    return float(numerator / denominator)


@dataclass
class HistoryEMAJ:
    decay: float = 0.9
    eps: float = 1e-12

    def __post_init__(self) -> None:
        if not (0.0 <= self.decay < 1.0):
            raise ValueError("EMA decay must be in [0, 1)")
        if not math.isfinite(self.eps) or self.eps <= 0:
            raise ValueError("eps must be finite and positive")
        self.global_ema: torch.Tensor | None = None
        self.mode_ema: dict[str, torch.Tensor] = {}
        self.global_updates: int = 0
        self.mode_updates: dict[str, int] = {}

    def _ema(self, previous: torch.Tensor | None, observed: torch.Tensor) -> torch.Tensor:
        observed = flatten_update(observed)
        if previous is None:
            return observed.clone()
        previous = flatten_update(previous)
        if previous.numel() != observed.numel():
            raise ValueError("EMA update dimension changed; refusing to mix model scopes")
        return self.decay * previous + (1.0 - self.decay) * observed

    def predict(self, mode: str) -> dict[str, Any]:
        mode = str(mode)
        if mode not in TARGET_MODES:
            return {"mode": mode, "status": "unsupported_mode", "predicted_j": None}
        if self.global_ema is None:
            return {"mode": mode, "status": "no_global_history", "predicted_j": None}
        # LIIE is Edge-only in this repository: no Cloud submission means a
        # zero Cloud contribution, not a fabricated Cloud packet.
        if mode == "LIIE":
            candidate = torch.zeros_like(self.global_ema)
            return {
                "mode": mode, "status": "no_cloud_submission_zero_contribution",
                "predicted_j": normalized_squared_deviation(candidate, self.global_ema, self.eps),
                "predicted_update_norm": 0.0,
                "global_ema_norm": float(torch.linalg.vector_norm(self.global_ema).item()),
                "mode_history_updates": 0,
            }
        candidate = self.mode_ema.get(mode)
        if candidate is None:
            return {"mode": mode, "status": "no_mode_history", "predicted_j": None}
        return {
            "mode": mode, "status": "ok",
            "predicted_j": normalized_squared_deviation(candidate, self.global_ema, self.eps),
            "predicted_update_norm": float(torch.linalg.vector_norm(candidate).item()),
            "global_ema_norm": float(torch.linalg.vector_norm(self.global_ema).item()),
            "mode_history_updates": int(self.mode_updates.get(mode, 0)),
        }

    def predict_all(self, round_index: int) -> list[dict[str, Any]]:
        rows = []
        for mode in TARGET_MODES:
            row = self.predict(mode)
            row.update({"event": "prediction", "round_index": int(round_index),
                        "available_before_round": int(round_index)})
            rows.append(row)
        return rows

    def observe(self, *, round_index: int, global_update: Any | None,
                mode_updates: Mapping[str, Any], global_history_eligible: bool,
                eligibility_reason: str = "") -> list[dict[str, Any]]:
        """Update histories only from eligible, already-observed DP-protected uploads.

        `mode_updates` must contain per-Mode sample-weighted Cloud contributions
        formed by the caller from the actual Cloud submission units. LIIE has no
        Cloud contribution and therefore does not appear in this mapping.
        """
        rows: list[dict[str, Any]] = []
        for mode, update in mode_updates.items():
            if mode not in {"LIIEIIIC", "LIIC"}:
                continue
            vec = flatten_update(update)
            self.mode_ema[mode] = self._ema(self.mode_ema.get(mode), vec)
            self.mode_updates[mode] = self.mode_updates.get(mode, 0) + 1
            rows.append({
                "event": "observation", "round_index": int(round_index), "mode": mode,
                "status": "observed_dp_protected_upload_contribution",
                "observed_update_norm": float(torch.linalg.vector_norm(vec).item()),
                "mode_history_updates": int(self.mode_updates[mode]),
            })
        if global_history_eligible and global_update is not None:
            vec = flatten_update(global_update)
            self.global_ema = self._ema(self.global_ema, vec)
            self.global_updates += 1
            rows.append({
                "event": "global_ema_update", "round_index": int(round_index),
                "status": "observed_eligible_global_update",
                "observed_update_norm": float(torch.linalg.vector_norm(vec).item()),
                "global_history_updates": int(self.global_updates),
            })
        else:
            rows.append({
                "event": "global_ema_update", "round_index": int(round_index),
                "status": "skipped_ineligible_global_update",
                "reason": str(eligibility_reason or "global update was not fully attributable to eligible upload-protected Modes"),
                "global_history_updates": int(self.global_updates),
            })
        return rows

    def state_dict(self) -> dict[str, Any]:
        return {
            "schema": "dynfl.history_ema_j.v1",
            "decay": self.decay, "eps": self.eps,
            "global_ema": None if self.global_ema is None else self.global_ema.clone(),
            "mode_ema": {k: v.clone() for k, v in self.mode_ema.items()},
            "global_updates": self.global_updates,
            "mode_updates": dict(self.mode_updates),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        if state.get("schema") != "dynfl.history_ema_j.v1":
            raise ValueError("unsupported history EMA-J state schema")
        if float(state.get("decay")) != self.decay or float(state.get("eps")) != self.eps:
            raise ValueError("history EMA-J settings changed across resume")
        raw_global = state.get("global_ema")
        self.global_ema = None if raw_global is None else flatten_update(raw_global)
        self.mode_ema = {str(k): flatten_update(v) for k, v in dict(state.get("mode_ema", {})).items()}
        self.global_updates = int(state.get("global_updates", 0))
        self.mode_updates = {str(k): int(v) for k, v in dict(state.get("mode_updates", {})).items()}

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temp = path.with_suffix(path.suffix + ".tmp")
        torch.save(self.state_dict(), temp)
        temp.replace(path)

    def load(self, path: str | Path) -> None:
        state = torch.load(Path(path), map_location="cpu", weights_only=False)
        self.load_state_dict(state)


def append_jsonl(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, sort_keys=True) + "\n")
