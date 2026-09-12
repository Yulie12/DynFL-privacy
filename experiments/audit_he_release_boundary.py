"""Public synthetic audit of the current HE harness, not a network attack."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from dynfed.fmnist_lenet5_dynamic import _seal_ckks_runtime
from dynfed.he_backend import CKKS_SCALE, decode_seal_vector
from dynfed.utils import timestamped_dir
from experiments.distributed_edge_dp_common import distributed_noise_plan, seal_edge_sum


def run_audit():
    runtime, _ = _seal_ckks_runtime()
    # Public test vector. Never use raw training updates for this boundary test.
    vector = np.array([0.1, -0.2, 0.3, 0.05], dtype=np.float64)
    ciphertext = runtime["encryptor"].encrypt(runtime["encoder"].encode(vector, CKKS_SCALE))
    decoded = decode_seal_vector(runtime["encoder"], runtime["decryptor"].decrypt(ciphertext))[:len(vector)]
    individual_error = float(np.max(np.abs(decoded - vector)))
    plan = distributed_noise_plan([1, 1], 2, 0.1, 1, 2)
    packets = {0: torch.tensor([0.01, -0.02]), 1: torch.tensor([0.03, 0.04])}
    aggregate, metrics = seal_edge_sum(packets, plan["groups"])
    missing_rejected = False
    try:
        seal_edge_sum({0: packets[0]}, plan["groups"])
    except ValueError:
        missing_rejected = True
    # Input validation is relative to a caller-provided plan, not authorization.
    alternate_plan = [dict(plan["groups"][0], cloud_weight=1.0)]
    single, _ = seal_edge_sum({0: packets[0]}, alternate_plan)
    alternate_error = float((single - packets[0]).abs().max())
    report = dict(
        scope="local_public_synthetic_capability_audit_not_remote_exploit",
        runtime_holds_secret_key="secret_key" in runtime,
        individual_ciphertext_decryption_succeeds=individual_error < 1e-5,
        individual_ciphertext_error=individual_error,
        fixed_supplied_plan_missing_packet_rejected=missing_rejected,
        caller_can_supply_new_single_edge_plan=alternate_error < 1e-5,
        authorized_cohort_binding_enforced=False,
        round_binding_enforced=False, replay_rejection_enforced=False,
        key_isolation_enforced=metrics["key_isolation_enforced"],
        aggregate_only_decryption_enforced=metrics["aggregate_only_decryption_enforced"],
        threshold_decryption_implemented=False, protocol_ready=False,
        valid_aggregate=aggregate.tolist(),
        interpretation="Current interface is an arithmetic harness, not a cloud decryption service",
    )
    report["status"] = "audit_completed_protocol_not_ready"
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", default="out/he_release_boundary_audit")
    args = parser.parse_args()
    report = run_audit()
    output = timestamped_dir(args.output_root, "he_boundary").resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "report.json"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Report {path}")


if __name__ == "__main__":
    main()
