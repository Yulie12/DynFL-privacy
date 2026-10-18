"""Predecision runtime reconciliation for the Stage-12 LIIC online-J bound.

Only observable optimizer metadata are filled here.  A separately proved
ideal-gradient reference, cross-mode aggregation model and all v2 assumptions
remain mandatory.  In particular, this module never manufactures reference
radii, mode-equivalence attestations or DP guarantees.
"""
from __future__ import annotations

import copy
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping, Sequence

from .liic_online_certificate import planned_batch_sizes
from .online_j_bound import certificate_candidate_key


def _finite_nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite nonnegative number")
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite nonnegative number") from exc
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{label} must be a finite nonnegative number")
    return value


def _training_metadata(*, model_parts: Sequence[Any], client_samples: int,
                       local_epochs: int, batch_size: int, learning_rate: float,
                       weight_decay: float, sample_clip_norm: float,
                       privacy_unit: str, candidate: Any, step_cap: int | None,
                       model_name: str, ablation_no_dp: bool = False,
                       ablation_clip_only: bool = False) -> dict[str, Any]:
    """Metadata before training. No gradients or client sample values accessed."""
    if str(candidate.mode) != "LIIC" or model_name != "lenet5":
        raise ValueError("predecision LIIC hydration requires LeNet-5 LIIC")
    if not isinstance(client_samples, int) or client_samples <= 0:
        raise ValueError("LIIC requires exact positive local sample count")
    params = [p for model in model_parts for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("LIIC runtime model has no trainable parameters")
    dim = sum(int(p.numel()) for p in params)
    initial_norm_sq = math.fsum(float(p.detach().double().square().sum().item()) for p in params)
    batches = planned_batch_sizes(
        sample_count=client_samples, epochs=int(local_epochs),
        batch_size=int(batch_size), local_step_cap=step_cap,
    )
    is_sample_path = privacy_unit == "sample" and not ablation_no_dp
    if is_sample_path:
        raw_sigma = 0.0 if ablation_clip_only else candidate.sample_optimizer_noise_multiplier
        if raw_sigma is None:
            raise ValueError("LIIC Sample-DP candidate missing optimizer sigma")
        sigma = _finite_nonnegative(raw_sigma, "LIIC optimizer sigma")
        source, clip = "sample_dp_sgd", _finite_nonnegative(sample_clip_norm, "sample clip")
    else:
        source, sigma, clip = "clipped_sgd", 0.0, 5.0
    if clip <= 0:
        raise ValueError("LIIC optimizer clip must be positive")
    if dim != 61706:
        # This is a parameter-scope guard, NOT a claim that arbitrary LeNet-5
        # implementations have this dimension. This repository's GN-v2 does.
        raise ValueError(f"LIIC LeNet-5 parameter-scope mismatch: {dim} != 61706")
    return {
        "schema": "dynfl.liic.sgd.v1", "mode": "LIIC", "model": "lenet5",
        "optimizer": "torch.optim.SGD", "momentum": 0,
        "source": source, "initial_parameter_norm": math.sqrt(initial_norm_sq),
        "learning_rate": _finite_nonnegative(learning_rate, "learning rate"),
        "weight_decay": _finite_nonnegative(weight_decay, "weight decay"),
        "gradient_clip_norm": clip, "sample_noise_multiplier": sigma,
        "batch_sizes": list(batches), "planned_optimizer_steps": len(batches),
        "vector_dimension": dim,
    }


# The helper may fill ONLY observable optimizer inputs. The independently
# established ideal-step and mathematical attestation are not runtime facts.
_RUNTIME_FIELDS = frozenset({
    "schema", "mode", "model", "optimizer", "momentum", "source",
    "initial_parameter_norm", "learning_rate", "weight_decay",
    "gradient_clip_norm", "sample_noise_multiplier", "batch_sizes",
    "planned_optimizer_steps",
})


def reconcile_liic_proof(template: Mapping[str, Any], metadata: Mapping[str, Any]) -> dict[str, Any]:
    """Hydrate opted-in LIIC proof; reject conflicting prefilled runtime values.

    The outer certificate's runtime_mode_equivalence_verified and the proof's
    evidence/ideal_reference_bound_proved remain the supplier's obligations.
    """
    if template.get("runtime_fields_from_trainer") is not True:
        raise ValueError("LIIC proof must explicitly opt in to runtime_fields_from_trainer")
    result = copy.deepcopy(dict(template))
    result.pop("runtime_fields_from_trainer", None)
    for key in _RUNTIME_FIELDS:
        if key in result and result[key] != metadata[key]:
            # Floating-point values exported through JSON are not always bit
            # identical to runtime calculations; avoid stale proof reuse.
            raise ValueError(f"LIIC supplied {key} disagrees with current predecision runtime")
        result[key] = copy.deepcopy(metadata[key])
    # Do NOT insert assertions about unobserved reference-gradient conditions.
    return result


def hydrate_online_manifest(*, source: str | Path, round_index: int,
                            selected: Sequence[tuple[int, Any, Sequence[Any], float]],
                            client_sample_counts: Mapping[int, int], model_parts: Sequence[Any],
                            model_name: str, local_epochs: int, batch_size: int,
                            learning_rate: float, weight_decay: float,
                            sample_clip_norm: float, privacy_unit: str,
                            step_cap: int | None, ablation_no_dp: bool = False,
                            ablation_clip_only: bool = False) -> dict[str, Any]:
    """Generate ONE current-round v2 manifest from independently proved input.

    We deliberately do not manufacture a 'real' cert file when source is absent.
    This runs before mode selection or any current-round local training.
    """
    path = Path(source)
    supplied = json.loads(path.read_text(encoding="utf-8"))
    if supplied.get("schema") != "dynfl.online_j.v2" or supplied.get("target") != "ideal_full_gradient_one_step":
        raise ValueError("runtime hydration needs independently proved online-J v2 source")
    state = supplied.get("rounds", {}).get(f"round:{round_index}")
    if not isinstance(state, dict):
        raise ValueError(f"online-J source missing independently proved round:{round_index}")
    if state.get("round_index") != round_index or state.get("available_before_round") != round_index:
        raise ValueError("online-J predecision round metadata mismatch")
    hydrated = copy.deepcopy(state)
    entries = hydrated.get("clients")
    if not isinstance(entries, dict):
        raise ValueError("online-J v2 missing Client certificate records")
    for cid, _chosen, candidates, _remaining in selected:
        for candidate in candidates:
            if candidate.mode != "LIIC":
                continue
            certs = entries.get(str(cid), {}).get("candidates", {})
            key = certificate_candidate_key(candidate)
            cert = certs.get(key)
            # Certified modes outside LIIC are handled by existing v2 evaluator.
            if not isinstance(cert, dict) or "liic_sgd_proof" not in cert:
                continue
            evidence = cert["liic_sgd_proof"]
            if not isinstance(evidence, dict) or evidence.get("runtime_fields_from_trainer") is not True:
                continue
            metadata = _training_metadata(
                model_parts=model_parts, client_samples=client_sample_counts[cid],
                local_epochs=local_epochs, batch_size=batch_size,
                learning_rate=learning_rate, weight_decay=weight_decay,
                sample_clip_norm=sample_clip_norm, privacy_unit=privacy_unit,
                candidate=candidate, step_cap=step_cap, model_name=model_name,
                ablation_no_dp=ablation_no_dp, ablation_clip_only=ablation_clip_only,
            )
            cert["liic_sgd_proof"] = reconcile_liic_proof(evidence, metadata)
    return {"schema": "dynfl.online_j.v2", "target": "ideal_full_gradient_one_step",
            "rounds": {f"round:{round_index}": hydrated}}


def write_predecision_json(path: str | Path, value: Mapping[str, Any]) -> Path:
    """Atomic handoff, no partial certificates exposed to the selector."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    try:
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    return path


def liic_shadow_metadata(*, round_index: int,
                         selected: Sequence[tuple[int, Any, Sequence[Any], float]],
                         client_sample_counts: Mapping[int, int], model_parts: Sequence[Any],
                         model_name: str, local_epochs: int, batch_size: int,
                         learning_rate: float, weight_decay: float,
                         sample_clip_norm: float, privacy_unit: str,
                         step_cap: int | None, ablation_no_dp: bool = False,
                         ablation_clip_only: bool = False) -> dict[str, Any]:
    """Explicit NON-CERTIFIED audit; safe to create without external proof.

    Neither private gradients nor client updates are read or saved.
    """
    rows: list[dict[str, Any]] = []
    for cid, _chosen, candidates, _remaining in selected:
        for cand in candidates:
            if cand.mode != "LIIC":
                continue
            try:
                metadata = _training_metadata(
                    model_parts=model_parts, client_samples=client_sample_counts[cid],
                    local_epochs=local_epochs, batch_size=batch_size,
                    learning_rate=learning_rate, weight_decay=weight_decay,
                    sample_clip_norm=sample_clip_norm, privacy_unit=privacy_unit,
                    candidate=cand, step_cap=step_cap, model_name=model_name,
                    ablation_no_dp=ablation_no_dp, ablation_clip_only=ablation_clip_only,
                )
                rows.append({"client_id": cid, "candidate_key": certificate_candidate_key(cand),
                             "status": "runtime_optimizer_metadata_only_NOT_A_J_CERTIFICATE",
                             "optimizer": metadata})
            except (ValueError, KeyError) as exc:
                rows.append({"client_id": cid, "candidate_key": certificate_candidate_key(cand),
                             "status": "unsupported_fail_closed", "reason": str(exc)})
    from .online_j_liic_envelope import lenet5_layer_norm_upper_from_states
    layer_norms=(lenet5_layer_norm_upper_from_states(
        model_parts[0].state_dict(),model_parts[1].state_dict())
        if model_name=='lenet5' else None)
    return {"schema": "dynfl.liic.predecision_runtime_audit.v1",
            "round_index": int(round_index), "available_before_round": int(round_index),
            "formal_online_j_certified": False, "training_data_accessed": False,
            "layer_norm_uppers": layer_norms,
            "liic_candidates": rows,
            "missing_independent_proofs": [
                "client ideal-gradient reference and conditional RMS bound",
                "all other reachable real-mode update and aggregation equivalence",
                "conditional numerical and release-noise second moments",
            ]}
