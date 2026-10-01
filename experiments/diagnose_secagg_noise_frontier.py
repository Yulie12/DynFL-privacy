"""Diagnose aggregate-DP noise inflation under an untrusted-edge threat model."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.privacy import calibrate_gaussian_noise
from dynfed.protection_rules import aggregate_replacement_bound
from experiments.secagg_noise_frontier_common import build_frontier


def _parse_ints(values: list[str]) -> list[int]:
    parsed = sorted(set(int(v) for v in values))
    if not parsed:
        raise ValueError("At least one integer value is required")
    return parsed


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--clients", nargs="+", default=["10", "20", "50", "100"])
    p.add_argument("--minimum-unknown-clients", nargs="+", default=["2", "5", "10"])
    p.add_argument("--clip-norm", type=float, default=0.25)
    p.add_argument("--epsilon", type=float, default=8.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--privacy-horizon", type=int, default=1)
    p.add_argument("--output", default="out/secagg_noise_frontier.csv")
    args = p.parse_args()

    clients_values = _parse_ints(args.clients)
    minimum_values = _parse_ints(args.minimum_unknown_clients)
    sigma = calibrate_gaussian_noise(args.epsilon, args.delta, args.privacy_horizon)

    rows: list[dict[str, object]] = []
    for k in clients_values:
        weights = [1.0 / k] * k
        sensitivity = aggregate_replacement_bound(
            weights, [{i} for i in range(k)], clip_norm=args.clip_norm
        )
        target_std = sensitivity * sigma
        for row in build_frontier([k], minimum_values, target_std):
            row.update({
                "clip_norm": args.clip_norm,
                "epsilon_target": args.epsilon,
                "delta": args.delta,
                "privacy_horizon": args.privacy_horizon,
                "noise_multiplier": sigma,
                "aggregate_dp_sensitivity": sensitivity,
            })
            rows.append(row)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with out.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print("SecAgg distributed-noise frontier")
    for row in rows:
        if row["scheme"] == "threshold_joint_exact_target":
            continue
        print(
            f"K={int(row['clients']):3d} Hmin={int(row['minimum_unknown_clients']):3d} "
            f"scheme={str(row['scheme']):31s} "
            f"stdInfl={float(row['std_inflation']):7.3f} "
            f"energyInfl={float(row['energy_inflation']):8.3f} "
            f"condVarRatio={float(row['conditional_unknown_variance_ratio']):7.3f}"
        )
    print(json.dumps({
        "output": str(out.resolve()),
        "interpretation": {
            "edge_only_exact": (
                "exact target aggregate variance; valid only when client noise shares remain hidden from edge"
            ),
            "collusion_robust_independent": (
                "retains target variance after conditioning down to Hmin unknown shares; over-noises by K/Hmin in energy"
            ),
            "threshold_joint_exact_target": (
                "desired exact-and-collusion-robust contract; not implemented"
            ),
        },
    }, indent=2))


if __name__ == "__main__":
    main()
