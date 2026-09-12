"""Public synthetic numerical check, not a secure protocol or DP certificate."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dynfed.fmnist_lenet5_dynamic import HEOperationMetrics, fedavg_split_seal
from experiments.validate_trusted_aggregate_dp import release_scales
from dynfed.privacy import calibrate_gaussian_noise
from dynfed.utils import timestamped_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="out/paired_he_release")
    args = parser.parse_args()
    clients, dimension, clip = 100, 5130, 0.1
    rng = np.random.default_rng(42)
    raw = rng.normal(0, 0.01, (clients, dimension))
    clipped = raw * np.minimum(1, clip / np.linalg.norm(raw, axis=1))[:, None]
    clipped = clipped.astype(np.float32)
    expected = clipped.astype(np.float64).mean(axis=0)
    end = torch.nn.Linear(dimension, 1, bias=False)
    edge = torch.nn.Identity()
    with torch.no_grad():
        end.weight.zero_()
    diffs = [{"end": {"weight": torch.from_numpy(row.copy()).reshape(1, -1)},
              "edge": {}} for row in clipped]
    metrics = HEOperationMetrics(backend="seal")
    started = time.perf_counter()
    fedavg_split_seal(diffs, [1.0] * clients, end, edge, torch.device("cpu"),
                     encrypted_mask=[True] * clients, he_aggregation_size=0,
                     he_workers=1, he_metrics=metrics)
    wall = time.perf_counter() - started
    actual = end.weight.detach().numpy().reshape(-1).astype(np.float64)
    sigma = calibrate_gaussian_noise(8.0, 1e-5, 100)
    scales = release_scales(clients, clip, sigma)
    # Common noise isolates encryption error; these are public synthetic inputs.
    noise = rng.normal(0, scales["aggregate_noise_std"], dimension)
    error = float(np.max(np.abs(actual - expected)))
    release_error = float(np.max(np.abs((actual + noise) - (expected + noise))))
    tolerance = 1e-5
    passed = bool(np.isfinite(actual).all() and error <= tolerance
                  and release_error <= tolerance
                  and metrics.encrypted_parameter_values == clients * dimension
                  and metrics.encrypted_updates == clients)
    report = {
        "status": "passed" if passed else "failed",
        "scope": "public_synthetic_numerical_equivalence_only",
        "clients": clients, "dimension": dimension, "clip_norm": clip,
        "privacy_horizon": 100, "target_epsilon": 8, "delta": 1e-5,
        "noise_multiplier": sigma, "scales": scales,
        "max_abs_aggregate_error": error, "max_abs_release_error": release_error,
        "absolute_tolerance": tolerance, "he_wall_sec": wall,
        "he_metrics": asdict(metrics),
        "key_isolation_enforced": False,
        "aggregate_only_decryption_enforced": False,
        "formal_protocol_dp_established": False,
        "limitations": ["No training or accuracy claim", "Single process holds secret key",
                        "Public seeded noise is for numerical testing only",
                        "Independent updates and fixed equal weights only"],
    }
    output = timestamped_dir(args.output_root, "paired_he_release").resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report saved to {output / 'report.json'}")
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
