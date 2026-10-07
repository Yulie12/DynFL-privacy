from __future__ import annotations

"""Build a joint update-space calibration table from paired trajectory outputs.

This is intentionally separated from online mode selection.  The manifest must
contain paired clean/private updates generated on an independent public or
calibration workload.  Each pair is one observation of the nonlinear trajectory
error D = u_private - u_clean; no Jacobian or linear model is fitted.

Manifest example::

    {
      "metadata": {"cross_client_rng": "conditionally_independent"},
      "pairs": [
        {
          "state_key": "round:0",
          "mode": "LIC",
          "client_id": 0,
          "clean": "pairs/lic_c0_t0_clean.pt",
          "private": "pairs/lic_c0_t0_private.pt"
        }
      ],
      "ideal_updates": [
        {"state_key": "round:0", "path": "pairs/round0_ideal.pt"}
      ]
    }

Each update file may contain a flat tensor/list, a nested
{"end": {...}, "edge": {...}} state difference, or a mapping with a
``state_diff`` field containing that nested update.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.joint_calibration import (
    JointUpdateCalibrationTable,
    PairedUpdateMomentAccumulator,
    as_update_vector,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build DynFL joint update-space calibration from offline paired trajectories."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path)
    parser.add_argument(
        "--min-trials",
        type=int,
        default=2,
        help="Minimum paired trials required per (state, mode, client/global) calibration cell.",
    )
    return parser.parse_args()


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _load_update(path: Path) -> torch.Tensor:
    if path.suffix.lower() == ".npy":
        return as_update_vector(np.load(path))
    payload = _torch_load(path)
    if isinstance(payload, Mapping) and "state_diff" in payload:
        payload = payload["state_diff"]
    return as_update_vector(payload)


def _resolve(base: Path, value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else (base / path).resolve()


def main() -> None:
    args = parse_args()
    if args.min_trials < 1:
        raise ValueError("--min-trials must be positive")

    manifest_path = args.manifest.resolve()
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("manifest root must be a JSON object")
    base = manifest_path.parent

    accumulators: dict[tuple[str, str, int | None], PairedUpdateMomentAccumulator] = {}
    for index, pair in enumerate(payload.get("pairs", [])):
        if not isinstance(pair, dict):
            raise ValueError(f"pairs[{index}] must be an object")
        state_key = str(pair.get("state_key", "default"))
        mode = str(pair["mode"])
        client_id = pair.get("client_id")
        client_id = None if client_id is None else int(client_id)
        clean_path = _resolve(base, pair["clean"])
        private_path = _resolve(base, pair["private"])
        clean = _load_update(clean_path)
        private = _load_update(private_path)
        key = (state_key, mode, client_id)
        accumulators.setdefault(key, PairedUpdateMomentAccumulator()).update(clean, private)

    table = JointUpdateCalibrationTable(
        metadata={
            "cross_client_rng": "conditionally_independent",
            "source_manifest": str(manifest_path),
            **dict(payload.get("metadata") or {}),
        }
    )

    cell_summary: list[dict[str, Any]] = []
    for (state_key, mode, client_id), accumulator in sorted(
        accumulators.items(),
        key=lambda item: (item[0][0], item[0][1], -1 if item[0][2] is None else item[0][2]),
    ):
        if accumulator.count < args.min_trials:
            raise ValueError(
                f"calibration cell {(state_key, mode, client_id)} has {accumulator.count} trials; "
                f"need at least {args.min_trials}"
            )
        entry = accumulator.finalize()
        table.add_entry(
            state_key=state_key,
            mode=mode,
            client_id=client_id,
            entry=entry,
        )
        cell_summary.append(
            {
                "state_key": state_key,
                "mode": mode,
                "client_id": client_id,
                "trials": entry.sample_count,
                "dimension": int(entry.clean_update_mean.numel()),
                "bias_norm": float(torch.linalg.vector_norm(entry.bias_mean).item()),
                "variance_trace": float(entry.variance_trace),
            }
        )

    for index, item in enumerate(payload.get("ideal_updates", [])):
        if not isinstance(item, dict):
            raise ValueError(f"ideal_updates[{index}] must be an object")
        state_key = str(item.get("state_key", "default"))
        table.set_ideal_update(
            _load_update(_resolve(base, item["path"])),
            state_key=state_key,
        )

    output = table.save(args.output.resolve())
    summary = {
        "calibration_table": str(output),
        "entry_count": table.entry_count,
        "cells": cell_summary,
        "has_ideal_updates": bool(payload.get("ideal_updates")),
        "note": (
            "Online selection reads this table only; paired clean/private runs are offline calibration/validation work."
        ),
    }
    summary_path = args.summary.resolve() if args.summary else output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
