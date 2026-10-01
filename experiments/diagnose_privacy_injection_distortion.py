from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

# Reuse only state-extraction helpers from the already validated Stage-0 diagnostic.
from diagnose_hierarchical_state_fidelity import (  # noqa: E402
    MODES,
    _client_edge_map,
    _edge_states,
    _global_nested,
    _last_round_rows,
    _load_checkpoint,
    _nested_vector,
    _sq_norm,
    _tensor_vector,
)

RUNNER = ROOT / "experiments" / "run_forced_homogeneous_profile.py"
EDGE_PRIMARY_MODES = {"LIE", "LIIE"}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Stage-1 calibration of privacy-injection distortion. For every mode, "
            "compare a forced fixed-DP run against that mode's Stage-0 clean run in "
            "the same model-update space."
        )
    )
    p.add_argument(
        "--clean-root",
        default="out/hierarchical_state_fidelity_v65",
        help="Stage-0 output root containing hierarchical_state_fidelity.csv and clean checkpoints.",
    )
    p.add_argument("--output-root", default="out/privacy_injection_distortion_v65")
    p.add_argument(
        "--reuse-existing",
        action="store_true",
        help="Reuse newest protected checkpoint already present under each per-mode directory.",
    )
    p.add_argument(
        "runner_args",
        nargs=argparse.REMAINDER,
        help="Arguments forwarded to run_fmnist_lenet5.py after '--'.",
    )
    return p.parse_args()


def _strip_separator(values: list[str]) -> list[str]:
    return values[1:] if values and values[0] == "--" else values


def _latest_checkpoint(root: Path) -> Path | None:
    paths = list(root.rglob("checkpoint.pt")) if root.exists() else []
    if not paths:
        return None
    return max(paths, key=lambda p: p.stat().st_mtime)


def _clean_checkpoint_paths(clean_root: Path) -> dict[str, Path]:
    csv_path = clean_root / "hierarchical_state_fidelity.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"missing Stage-0 CSV: {csv_path}. Run diagnose_hierarchical_state_fidelity.py first."
        )
    out: dict[str, Path] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            mode = str(row.get("mode", ""))
            checkpoint = Path(str(row.get("checkpoint", "")))
            if mode in MODES and checkpoint.exists():
                out[mode] = checkpoint
    missing = [mode for mode in MODES if mode not in out]
    if missing:
        raise FileNotFoundError(f"Stage-0 CSV is missing usable clean checkpoints for: {missing}")
    return out


def _run_protected_mode(
    mode: str,
    output_root: Path,
    forwarded: list[str],
    reuse: bool,
) -> Path:
    mode_root = output_root / mode
    if reuse:
        existing = _latest_checkpoint(mode_root)
        if existing is not None:
            print(f"[{mode}] reuse protected {existing}")
            return existing

    excluded = [item for item in MODES if item != mode]
    cmd = [
        sys.executable,
        str(RUNNER),
        *forwarded,
        "--rounds", "1",
        "--policies", "fixed_dp",
        "--exclude-modes", *excluded,
        "--resource-limit", "100",
        "--memory-limit", "100",
        "--risk-limit", "1.0",
        "--aggregation-fraction", "1.0",
        "--min-edge-cloud-fusion-ratio", "0.0",
        "--he-backend", "none",
        "--output-root", str(mode_root),
    ]
    print(f"[{mode}] running protected homogeneous profile (fixed_dp)")
    subprocess.run(cmd, cwd=ROOT, check=True)
    checkpoint = _latest_checkpoint(mode_root)
    if checkpoint is None:
        raise FileNotFoundError(f"[{mode}] no protected checkpoint.pt produced under {mode_root}")
    return checkpoint


def _state_privacy_metric(
    protected: torch.Tensor,
    clean: torch.Tensor,
    initial: torch.Tensor,
) -> dict[str, float | int | None]:
    if protected.shape != clean.shape or clean.shape != initial.shape:
        raise ValueError(
            f"state-vector shape mismatch: protected={tuple(protected.shape)}, "
            f"clean={tuple(clean.shape)}, initial={tuple(initial.shape)}"
        )
    clean_update = clean - initial
    protected_update = protected - initial
    privacy_delta = protected_update - clean_update
    clean_energy = _sq_norm(clean_update)
    privacy_energy = _sq_norm(privacy_delta)
    protected_energy = _sq_norm(protected_update)
    active = int(clean_energy > 1e-24)

    cosine = None
    if clean_energy > 1e-24 and protected_energy > 1e-24:
        denom = math.sqrt(clean_energy * protected_energy)
        cosine = float(torch.dot(clean_update, protected_update) / denom)
        cosine = max(-1.0, min(1.0, cosine))

    relative = privacy_energy / clean_energy if active else None
    return {
        "active": active,
        "clean_update_norm": math.sqrt(max(clean_energy, 0.0)),
        "protected_update_norm": math.sqrt(max(protected_energy, 0.0)),
        "privacy_delta_norm": math.sqrt(max(privacy_energy, 0.0)),
        "privacy_delta_sq": privacy_energy,
        "relative_distortion": relative,
        "relative_noise_norm": math.sqrt(relative) if relative is not None else None,
        "protected_clean_cosine": cosine,
    }


def _round_accuracy(checkpoint: dict[str, Any]) -> float | None:
    rows = list(checkpoint.get("round_rows", []) or [])
    if not rows:
        return None
    value = rows[-1].get("test_accuracy")
    return float(value) if value is not None else None


def _protection_summary(checkpoint: dict[str, Any], mode: str) -> dict[str, Any]:
    rows = [row for row in _last_round_rows(checkpoint) if str(row.get("mode", "")) != "SKIP"]
    bad_modes = sorted({str(row.get("mode", "")) for row in rows if str(row.get("mode", "")) != mode})
    if bad_modes:
        raise RuntimeError(f"[{mode}] protected run executed unexpected modes: {bad_modes}")
    if not rows:
        raise RuntimeError(f"[{mode}] protected diagnostic executed zero clients")

    mechanisms = sorted({str(row.get("mechanisms", "")) for row in rows})
    dp_events = [
        int(row.get("feature_dp_events", 0) or 0) + int(row.get("update_dp_events", 0) or 0)
        for row in rows
    ]
    feature_events = [int(row.get("feature_dp_events", 0) or 0) for row in rows]
    update_events = [int(row.get("update_dp_events", 0) or 0) for row in rows]
    feature_sigmas = [
        float(row["feature_noise_multiplier"])
        for row in rows
        if row.get("feature_noise_multiplier") not in (None, "")
    ]
    update_sigmas = [
        float(row["update_noise_multiplier"])
        for row in rows
        if row.get("update_noise_multiplier") not in (None, "")
    ]
    eps = [float(row.get("epsilon_used", 0.0) or 0.0) for row in rows]
    return {
        "executed_clients": len(rows),
        "mechanisms": " | ".join(mechanisms),
        "mean_dp_events": sum(dp_events) / len(dp_events),
        "mean_feature_dp_events": sum(feature_events) / len(feature_events),
        "mean_update_dp_events": sum(update_events) / len(update_events),
        "feature_noise_multiplier": (sum(feature_sigmas) / len(feature_sigmas)) if feature_sigmas else None,
        "update_noise_multiplier": (sum(update_sigmas) / len(update_sigmas)) if update_sigmas else None,
        "mean_epsilon_used": sum(eps) / len(eps),
    }


def main() -> None:
    args = parse_args()
    forwarded = _strip_separator(list(args.runner_args))
    clean_root = Path(args.clean_root).resolve()
    output_root = Path(args.output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    clean_paths = _clean_checkpoint_paths(clean_root)
    clean_checkpoints = {mode: _load_checkpoint(path) for mode, path in clean_paths.items()}

    # The LIIE Stage-0 run is Edge-only, so its Cloud state remains at the common
    # initialization. It therefore supplies the common initial model state without
    # requiring an additional zero-round run.
    initial_state = _global_nested(clean_checkpoints["LIIE"])
    initial_vec = _nested_vector(initial_state)

    protected_paths: dict[str, Path] = {}
    protected_checkpoints: dict[str, dict[str, Any]] = {}
    for mode in MODES:
        path = _run_protected_mode(mode, output_root, forwarded, args.reuse_existing)
        protected_paths[mode] = path
        protected_checkpoints[mode] = _load_checkpoint(path)

    rows: list[dict[str, Any]] = []
    for mode in MODES:
        clean = clean_checkpoints[mode]
        protected = protected_checkpoints[mode]
        protection = _protection_summary(protected, mode)

        clean_cloud = _nested_vector(_global_nested(clean))
        protected_cloud = _nested_vector(_global_nested(protected))
        cloud = _state_privacy_metric(protected_cloud, clean_cloud, initial_vec)

        # Compare Edge states using the clean run's own assignment. With the same
        # seed/partition this is also the protected assignment, but using the clean
        # mapping makes the reference explicit and prevents cross-mode mixing.
        mapping = _client_edge_map(clean)
        canonical_edges = set(mapping.values())
        clean_edges = _edge_states(clean, initial_state, canonical_edges)
        protected_edges = _edge_states(protected, initial_state, canonical_edges)
        edge_counts = {
            edge_id: sum(1 for e in mapping.values() if e == edge_id)
            for edge_id in canonical_edges
        }
        total_clients = max(sum(edge_counts.values()), 1)

        edge_clean_energy = 0.0
        edge_privacy_energy = 0.0
        edge_protected_energy = 0.0
        edge_cosine_weighted = 0.0
        edge_cosine_weight = 0.0
        for edge_id in sorted(canonical_edges):
            weight = edge_counts.get(edge_id, 0) / total_clients
            clean_vec = _nested_vector(clean_edges[edge_id])
            protected_vec = _nested_vector(protected_edges[edge_id])
            clean_update = clean_vec - initial_vec
            protected_update = protected_vec - initial_vec
            delta = protected_update - clean_update
            ce = _sq_norm(clean_update)
            pe = _sq_norm(protected_update)
            de = _sq_norm(delta)
            edge_clean_energy += weight * ce
            edge_protected_energy += weight * pe
            edge_privacy_energy += weight * de
            if ce > 1e-24 and pe > 1e-24:
                cos = float(torch.dot(clean_update, protected_update) / math.sqrt(ce * pe))
                edge_cosine_weighted += weight * max(-1.0, min(1.0, cos))
                edge_cosine_weight += weight

        edge_active = int(edge_clean_energy > 1e-24)
        edge_relative = edge_privacy_energy / edge_clean_energy if edge_active else None
        edge = {
            "active": edge_active,
            "clean_update_norm": math.sqrt(max(edge_clean_energy, 0.0)),
            "protected_update_norm": math.sqrt(max(edge_protected_energy, 0.0)),
            "privacy_delta_norm": math.sqrt(max(edge_privacy_energy, 0.0)),
            "privacy_delta_sq": edge_privacy_energy,
            "relative_distortion": edge_relative,
            "relative_noise_norm": math.sqrt(edge_relative) if edge_relative is not None else None,
            "protected_clean_cosine": (
                edge_cosine_weighted / edge_cosine_weight if edge_cosine_weight > 0 else None
            ),
        }

        primary_name = "edge" if mode in EDGE_PRIMARY_MODES else "cloud"
        primary = edge if primary_name == "edge" else cloud
        clean_acc = _round_accuracy(clean)
        protected_acc = _round_accuracy(protected)

        row: dict[str, Any] = {
            "mode": mode,
            "primary_state": primary_name,
            **protection,
            "clean_checkpoint": str(clean_paths[mode]),
            "protected_checkpoint": str(protected_paths[mode]),
            "clean_test_accuracy": clean_acc,
            "protected_test_accuracy": protected_acc,
            "accuracy_delta": (
                protected_acc - clean_acc
                if clean_acc is not None and protected_acc is not None
                else None
            ),
            "primary_clean_update_norm": primary["clean_update_norm"],
            "primary_protected_update_norm": primary["protected_update_norm"],
            "primary_privacy_delta_norm": primary["privacy_delta_norm"],
            "primary_relative_distortion": primary["relative_distortion"],
            "primary_relative_noise_norm": primary["relative_noise_norm"],
            "primary_protected_clean_cosine": primary["protected_clean_cosine"],
            "cloud_active": cloud["active"],
            "cloud_relative_distortion": cloud["relative_distortion"],
            "cloud_relative_noise_norm": cloud["relative_noise_norm"],
            "cloud_protected_clean_cosine": cloud["protected_clean_cosine"],
            "edge_active": edge["active"],
            "edge_relative_distortion": edge["relative_distortion"],
            "edge_relative_noise_norm": edge["relative_noise_norm"],
            "edge_protected_clean_cosine": edge["protected_clean_cosine"],
        }
        rows.append(row)

    csv_path = output_root / "privacy_injection_distortion.csv"
    json_path = output_root / "privacy_injection_distortion.json"
    with csv_path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "definition": {
            "metric": "||Delta_protected-Delta_clean||_2^2 / ||Delta_clean||_2^2",
            "clean_source": str(clean_root),
            "protected_policy": "fixed_dp",
            "primary_state": {
                "LIE": "edge",
                "LIIE": "edge",
                "all_cloud_reaching_modes": "cloud",
            },
            "interpretation": (
                "A value of 1 means privacy-induced update-error energy equals clean update energy; "
                "sqrt(value) is the privacy-error norm relative to clean update norm."
            ),
        },
        "rows": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\nStage-1 privacy injection distortion")
    for row in rows:
        rel = row["primary_relative_distortion"]
        rel_text = "NA" if rel is None else f"{float(rel):.6g}"
        noise = row["primary_relative_noise_norm"]
        noise_text = "NA" if noise is None else f"{float(noise):.6g}"
        cos = row["primary_protected_clean_cosine"]
        cos_text = "NA" if cos is None else f"{float(cos):.4f}"
        print(
            f"{row['mode']:<10} target={row['primary_state']:<5} "
            f"Dpriv={rel_text:<12} relNorm={noise_text:<10} cos={cos_text:<7} "
            f"dAcc={row['accuracy_delta'] if row['accuracy_delta'] is not None else 'NA'} "
            f"featEv={row['mean_feature_dp_events']:.2f} updEv={row['mean_update_dp_events']:.2f} "
            f"sigmaF={row['feature_noise_multiplier']} sigmaU={row['update_noise_multiplier']}"
        )
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
