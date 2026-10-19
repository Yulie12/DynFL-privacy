#!/usr/bin/env python3
"""Backtest History-EMA J predictions and realized observations from history_ema_j.jsonl."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    ranks[order] = np.arange(len(values), dtype=float)
    # Average tied ranks.
    unique, inverse, counts = np.unique(values, return_inverse=True, return_counts=True)
    for idx, count in enumerate(counts):
        if count > 1:
            positions = np.flatnonzero(inverse == idx)
            ranks[positions] = ranks[positions].mean()
    return ranks


def _corr(x: np.ndarray, y: np.ndarray) -> float | None:
    if len(x) < 2 or np.std(x) <= 1e-15 or np.std(y) <= 1e-15:
        return None
    return float(np.corrcoef(x, y)[0, 1])


def _metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"n": 0, "reason": "no valid prediction/observation pairs"}
    x = np.asarray([float(r["predicted_j"]) for r in rows], dtype=float)
    y = np.asarray([float(r["observed_j"]) for r in rows], dtype=float)
    error = x - y
    return {
        "n": int(len(rows)),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error ** 2))),
        "prediction_mean": float(np.mean(x)),
        "observation_mean": float(np.mean(y)),
        "pearson": _corr(x, y),
        "spearman": _corr(_rank(x), _rank(y)),
    }


def backtest(path: str | Path, *, include_mixed: bool = False) -> dict[str, Any]:
    path = Path(path)
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    predictions = {
        (int(r["round_index"]), str(r["mode"])): r
        for r in records if r.get("event") == "prediction"
    }
    pairs: list[dict[str, Any]] = []
    for row in records:
        if row.get("event") != "realized_j" or row.get("observed_j") is None:
            continue
        if row.get("selected_mode_mix") is True and not include_mixed:
            continue
        key = (int(row["round_index"]), str(row["mode"]))
        pred = predictions.get(key)
        if pred is None or pred.get("predicted_j") is None:
            continue
        pairs.append({
            "round_index": key[0], "mode": key[1],
            "predicted_j": float(pred["predicted_j"]),
            "observed_j": float(row["observed_j"]),
            "prediction_status": pred.get("status"),
            "observation_status": row.get("status"),
        })
    by_mode = {
        mode: _metrics([r for r in pairs if r["mode"] == mode])
        for mode in ("LIIE", "LIIEIIIC", "LIIC")
    }
    return {
        "schema": "dynfl.history_ema_j.backtest.v1",
        "source": str(path),
        "paired_observations": len(pairs),
        "overall": _metrics(pairs) if pairs else {"n": 0, "reason": "no valid prediction/observation pairs"},
        "by_mode": by_mode,
        "pairs": pairs,
        "excluded_realized_rows": sum(
            1 for r in records if r.get("event") == "realized_j" and r.get("observed_j") is None
        ),
        "excluded_mixed_mode_rows": sum(
            1 for r in records if r.get("event") == "realized_j"
            and r.get("observed_j") is not None and r.get("selected_mode_mix") is True
        ) if not include_mixed else 0,
        "include_mixed": bool(include_mixed),
        "note": "Historical J backtest; metrics evaluate prediction calibration only and do not establish test-accuracy monotonicity.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl", type=Path, help="history_ema_j.jsonl")
    parser.add_argument("--output", type=Path, default=None, help="optional JSON report path")
    parser.add_argument("--include-mixed", action="store_true",
                        help="include rounds where several Modes were selected; default excludes them")
    args = parser.parse_args()
    report = backtest(args.jsonl, include_mixed=args.include_mixed)
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
