"""Offline Stage13 shadow analysis. This does not supply a J certificate.

Usage: python examples/report_real_liic_envelope.py --run out/.../full_dynfl
Reads predecision optimizer metadata already collected, no training rerun.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.online_j_liic_envelope import (
    liic_update_envelope, sample_weighted_liic_envelope,
    fmnist_lenet5_gradient_norm_cap,
    fmnist_lenet5_gradient_norm_cap_by_layers,
)


def report_run(directory: Path, *, local_epochs: int) -> list[dict]:
    if (directory / "full_dynfl").is_dir():
        directory = directory / "full_dynfl"
    shadows = sorted(directory.glob("online_j_liic_shadow_round_*.json"))
    if not shadows:
        raise ValueError("no real Stage13 LIIC shadow files found")
    summary_path = directory / "summary.json"
    if not summary_path.is_file():
        raise ValueError("summary.json required to confirm Fashion-MNIST/LeNet5")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("dataset") != "Fashion-MNIST" or summary.get("model") != "lenet5":
        raise ValueError("architecture gradient bound is only derived for Fashion-MNIST/LeNet5")
    decisions_path = directory / "client_decisions.csv"
    if not decisions_path.is_file():
        raise ValueError("client_decisions.csv required to validate actual selected LIIC participants")
    with decisions_path.open(newline="", encoding="utf-8-sig") as handle:
        decisions = list(csv.DictReader(handle))
    result = []
    for shadow_path in shadows:
        shadow = json.loads(shadow_path.read_text(encoding="utf-8"))
        rnd = shadow.get("round_index")
        if (shadow.get("schema") != "dynfl.liic.predecision_runtime_audit.v1"
                or shadow.get("formal_online_j_certified") is not False
                or shadow.get("training_data_accessed") is not False
                or not isinstance(rnd, int) or rnd < 0
                or shadow.get("available_before_round") != rnd):
            raise ValueError(f"invalid / postdecision LIIC shadow metadata: {shadow_path}")
        actual = {int(r["client_id"]): r for r in decisions if int(r["round"]) == rnd
                  and r.get("mode") == "LIIC"}
        if not actual:
            raise ValueError(f"round {rnd}: actual LIIC cohort unavailable")
        records = {}
        for record in shadow.get("liic_candidates", []):
            if record.get("status") != "runtime_optimizer_metadata_only_NOT_A_J_CERTIFICATE":
                raise ValueError(f"round {rnd}: unsupported LIIC optimizer metadata")
            cid = int(record["client_id"])
            if cid in records:
                raise ValueError(f"round {rnd}: duplicated LIIC candidate for client {cid}")
            records[cid] = liic_update_envelope(record["optimizer"], local_epochs=local_epochs)
        if set(actual) != set(records):
            raise ValueError(f"round {rnd}: selected-vs-shadow mismatch; cannot infer Cloud cohort")
        cohort = sorted(records.items())
        B = sample_weighted_liic_envelope(cohort)
        # All Client models in a round start from the same full state. The
        # recorded norm is floating-point; use a conservative inflation.
        # This is an architecture-level CE gradient cap under the documented
        # fixed FashionMNIST input range; not a certified numerical analysis.
        norm_upper = max(float(rec["optimizer"]["initial_parameter_norm"])
                         for rec in shadow.get("liic_candidates", []))
        norm_upper = norm_upper * (1 + 1e-5) + 1e-9
        G_arch = fmnist_lenet5_gradient_norm_cap(
            model_parameter_norm_upper=norm_upper,
        )
        total = sum(entry.client_samples for _, entry in cohort)
        noise = sum(entry.client_samples / total * entry.local_dp_noise_rms for _, entry in cohort)
        drift = sum(entry.client_samples / total * entry.clipped_gradient_part for _, entry in cohort)
        # Older Stage13 shadows have no layerwise norms. Never backfill from
        # a post-training checkpoint; that would leak future model state.
        G_layer=(fmnist_lenet5_gradient_norm_cap_by_layers(
            layer_norm_uppers=shadow['layer_norm_uppers'])
            if shadow.get('layer_norm_uppers') is not None else None)
        row = {
            "round": rnd, "liic_clients": len(cohort),
            "selected_LIIC_sample_weighted_local_update_rms_envelope": B,
            "weighted_dp_noise_rms_minkowski": noise,
            "fashion_mnist_lenet5_gradient_norm_architecture_cap": G_arch,
            "predecision_layerwise_gradient_norm_cap": G_layer,
            "weighted_clipped_gradient_component": drift,
            "J_formal_certified": False,
            "why_not_J": "missing proved gradient norm of full objective, Cloud execution equivalence, and post-optimizer noise/numeric bounds",
        }
        result.append(row)
    return result


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run", type=Path, required=True, help="Stage13 run or full_dynfl directory")
    p.add_argument("--local-epochs", type=int, required=True,
                   help="Exact training epoch count, e.g. 3; Stage13 JSON did not store it")
    args = p.parse_args()
    rows = report_run(args.run, local_epochs=args.local_epochs)
    print("round  LIIC   RMS update cap  CE gradient cap    CERTIFIED J")
    for row in rows:
        print(f'{row["round"]:>5} {row["liic_clients"]:>5} '
              f'{row["selected_LIIC_sample_weighted_local_update_rms_envelope"]:>16.6f} '
              f'{row["fashion_mnist_lenet5_gradient_norm_architecture_cap"]:>16.4e} '
              f'{str(row["J_formal_certified"]):>12}')
    for row in rows:
        if row['predecision_layerwise_gradient_norm_cap'] is not None:
            print(f'  round {row["round"]}: predecision G_layer={row["predecision_layerwise_gradient_norm_cap"]:.6g}')
    print("Selected-LIIC cohort proxy only; Flow/Cloud admission unverified.")
    print("The CE gradient cap assumes fixed FashionMNIST normalization and LeNet5 architecture.")
    print("Missing Cloud admission/equivalence, post-update noise and numerical RMS proofs.")
    print("NOT certified full J or DP. No new training run.")


if __name__ == "__main__":
    main()
