"""Predecision LIIC/LeNet-5 no-momentum SGD relative-ideal RMS certificate.

This is a mathematical calculator, NOT a proof that caller-supplied state and
ideal-gradient bounds match a real deployment.  It matches two concrete paths
in split_learning.split_local_train_lenet5 (the full-model LIIC branch):

* sample_dp_sgd: sample_dp.clip_and_aggregate_per_sample_grads applies
  per-sample norm clipping C, adds N(0, (2*C*sigma)^2 I) to the SUM, then
  divides by the ACTUAL batch size before torch.optim.SGD(m=0, wd=lambda).
* clipped_sgd: torch.nn.utils.clip_grad_norm_(max_norm=5) then the same SGD
  with no per-minibatch Gaussian gradient noise.

Both paths must have exactly K steps with a fresh optimizer and no other
unmodeled transformation of the update.  If a later release clips the full
update, supply the real update clipping threshold; if the release has an
unmodeled nonlinear transformation, do NOT use the certificate.

For theta_0, q_k with ||q_k||<=C and gradient-level independent Gaussian z_k:
 theta_{k+1} = (1-eta*lambda) theta_k - eta(q_k+z_k).
Let alpha=1-eta*lambda, b_k=eta*alpha**(K-1-k), S=sum b_k.
For a predecision ideal-update reference vhat=-eta_star*g_ref and an
independently certified ||v_i-vhat||_L2 <= s_i,

 ||u_i-vhat||_L2 <= |S/eta_star-1|*||vhat|| + lambda*S*||theta_0||
                     + sum b_k E_k
                     + sqrt(d*sum b_k^2*(2*C*sigma/B_k)^2),

where E_k is a valid *pathwise* bound on ||q_k-g_ref||.  Without a
proven sharper bound, E_k=C+||g_ref|| holds automatically from clipping.
Then ||u_i-v_i||_L2 <= the displayed bound + s_i.  This is deliberately
conservative; it does NOT assume clipped stochastic gradients are unbiased,
or wrongly add the local DP-SGD noise to aggregate_share_std again.

The Gaussian term is the exact L2 norm of a LINEAR Gaussian sum. q_k may
adapt to earlier z's, so the full expression still uses Minkowski: it never
cancels cross terms between q_k and that Gaussian sum.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping


def _nonnegative(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label}: expected finite nonnegative number")
    try:
        number = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: expected finite nonnegative number") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{label}: expected finite nonnegative number")
    return number


def _positive(value: Any, label: str) -> float:
    result = _nonnegative(value, label)
    if result == 0:
        raise ValueError(f"{label}: expected positive number")
    return result


@dataclass(frozen=True)
class LIICRadius:
    relative_ideal_rms: float
    optimizer_step_scale: float
    center_mismatch: float
    clipped_gradient_drift: float
    local_dp_sgd_noise_rms: float
    ideal_reference_rms: float
    release_projection_extra: float
    steps: int
    source: str = "liic_lenet5_sgd_conditional_predecision"


def planned_batch_sizes(*, sample_count: int, epochs: int, batch_size: int,
                        local_step_cap: int | None = None) -> tuple[int, ...]:
    """Match DataLoader(drop_last=False) and _training_batches in LIIC.

    `_client_step_limit` can cap the step count; if true the runtime may also
    subsample data before its DataLoader, so a bound from FULL sample count
    may be wrong. Reject capped runs rather than guessing their batch sizes.
    """
    if local_step_cap is not None:
        raise ValueError("LIIC capped/subsampled batches need separate runtime proof")
    if any(isinstance(x, bool) or not isinstance(x, int) for x in (sample_count, epochs, batch_size)):
        raise ValueError("sample_count, epochs and batch_size must be integers")
    if sample_count <= 0 or epochs <= 0 or batch_size <= 0:
        raise ValueError("sample_count, epochs and batch_size must be positive")
    full, tail = divmod(sample_count, batch_size)
    one_epoch = (batch_size,) * full + ((tail,) if tail else ())
    return one_epoch * epochs


def calculate_liic_radius(
    *, proof: Mapping[str, Any], ideal_reference_norm: float,
    ideal_reference_rms: float, vector_dimension: int,
) -> LIICRadius:
    """Compute a conditional RMS certificate from verifiable optimizer algebra.

    `proof` must describe a predecision state, not any completed updates.
    Numeric inputs including theta_0 norm, sigma, actual batch schedule and
    gradient reference certificate STILL require trustworthy provenance.
    A manifest boolean or this function's success is not a privacy proof.
    """
    if not isinstance(proof, Mapping) or proof.get("schema") != "dynfl.liic.sgd.v1":
        raise ValueError("LIIC proof schema must be dynfl.liic.sgd.v1")
    if proof.get("mode") != "LIIC" or proof.get("model") != "lenet5":
        raise ValueError("LIIC SGD bound only supports LeNet-5 LIIC")
    if proof.get("optimizer") != "torch.optim.SGD" or proof.get("momentum") != 0:
        raise ValueError("LIIC SGD bound requires a fresh, zero-momentum torch SGD optimizer")
    if proof.get("source") not in {"sample_dp_sgd", "clipped_sgd"}:
        raise ValueError("unsupported LIIC clipped-gradient execution path")
    if proof.get("evidence") is None or not isinstance(proof["evidence"], str) or not proof["evidence"].strip():
        raise ValueError("LIIC proof requires concrete external evidence/provenance")
    if proof.get("training_state_matches_runtime") is not True:
        raise ValueError("LIIC predecision state must be verified against runtime")
    if proof.get("ideal_reference_bound_proved") is not True:
        raise ValueError("LIIC ideal reference radius must have independent proof")
    if proof.get("noise_distribution_matches_runtime") is not True:
        raise ValueError("LIIC Gaussian mechanism must match runtime")
    if proof.get("extra_transformations_absent_or_modeled") is not True:
        raise ValueError("LIIC cannot silently ignore further update transformations")
    if isinstance(vector_dimension, bool) or not isinstance(vector_dimension, int) or vector_dimension <= 0:
        raise ValueError("vector_dimension must be a positive integer")

    reference_norm = _nonnegative(ideal_reference_norm, "ideal_reference_norm")
    reference_rms = _nonnegative(ideal_reference_rms, "ideal_reference_rms")
    initial_norm = _nonnegative(proof.get("initial_parameter_norm"), "initial_parameter_norm")
    lr = _positive(proof.get("learning_rate"), "learning_rate")
    eta_star = _positive(proof.get("ideal_step"), "ideal_step")
    weight_decay = _nonnegative(proof.get("weight_decay"), "weight_decay")
    clip_norm = _positive(proof.get("gradient_clip_norm"), "gradient_clip_norm")
    noise_multiplier = _nonnegative(proof.get("sample_noise_multiplier"), "sample_noise_multiplier")
    batch_sizes = proof.get("batch_sizes")
    if not isinstance(batch_sizes, (list, tuple)) or not batch_sizes:
        raise ValueError("LIIC requires predecision actual batch_sizes")
    if len(batch_sizes) > 100000:
        raise ValueError("LIIC certificate exceeds maximum supported optimizer steps")
    for i, batch in enumerate(batch_sizes):
        if isinstance(batch, bool) or not isinstance(batch, int) or batch <= 0:
            raise ValueError(f"batch_sizes[{i}] must be positive integer")
    if proof.get("planned_optimizer_steps") != len(batch_sizes):
        raise ValueError("LIIC planned_optimizer_steps/batch_sizes mismatch")
    if proof.get("source") == "clipped_sgd" and noise_multiplier != 0:
        raise ValueError("clipped_sgd is noise-free; do not invent minibatch Gaussian noise")
    if proof.get("source") == "clipped_sgd" and clip_norm != 5.0:
        raise ValueError("LeNet-5 clean LIIC runtime uses clip_grad_norm_(max_norm=5.0)")

    alpha = 1.0 - lr * weight_decay
    if not (0 <= alpha <= 1):
        raise ValueError("LIIC proof needs stable SGD decay 0<=1-lr*weight_decay<=1")
    k = len(batch_sizes)
    b = [lr * alpha ** (k - 1 - j) for j in range(k)]
    scale = math.fsum(b)
    baseline_center = (
        abs(scale / eta_star - 1.0) * reference_norm
        + weight_decay * scale * initial_norm
    )
    # Bounded by triangle inequality for any mini-batch selection and earlier
    # DP randomness: the average of clipped per-sample gradients has norm <=C.
    # Non-sample branch likewise clips the WHOLE gradient to norm <=5.
    caps = proof.get("gradient_reference_residual_caps")
    if caps is None:
        eps = [clip_norm + reference_norm / eta_star] * k
    else:
        if proof.get("gradient_residual_caps_proved") is not True:
            raise ValueError("sharper LIIC per-step residual caps need independently proved evidence")
        if not isinstance(caps, list) or len(caps) != k:
            raise ValueError("gradient_reference_residual_caps must contain one cap per step")
        eps = [_nonnegative(x, f"gradient_reference_residual_caps[{i}]") for i, x in enumerate(caps)]
    drift = math.fsum(bj * ej for bj, ej in zip(b, eps))
    if proof.get("source") == "sample_dp_sgd":
        # Exactly sample_dp.py: Gaussian on SUM with std=2*C*sigma,
        # then divide by actual B (including the last partial minibatch).
        per_step_var = math.fsum(
            (bj * (2.0 * clip_norm * noise_multiplier / float(batch))) ** 2
            for bj, batch in zip(b, batch_sizes)
        )
        noise_rms = math.sqrt(vector_dimension * per_step_var)
    else:
        noise_rms = 0.0

    projection_extra = 0.0
    if proof.get("release_update_clip_norm") is not None:
        release_clip = _positive(proof["release_update_clip_norm"], "release_update_clip_norm")
        # The Euclidean projection is 1-Lipschitz:
        # ||Proj(u)-vhat||_L2 <= ||u-vhat||_L2 + dist(vhat,B_C).
        projection_extra = max(0.0, reference_norm - release_clip)
    result = baseline_center + drift + noise_rms + projection_extra + reference_rms
    if not math.isfinite(result):
        raise ValueError("LIIC calculated radius overflowed or is nonfinite")
    return LIICRadius(
        relative_ideal_rms=result, optimizer_step_scale=scale,
        center_mismatch=baseline_center, clipped_gradient_drift=drift,
        local_dp_sgd_noise_rms=noise_rms, ideal_reference_rms=reference_rms,
        release_projection_extra=projection_extra, steps=k,
    )
