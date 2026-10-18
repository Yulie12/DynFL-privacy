"""Opt-in, fail-closed, PRE-DECISION conditional online-J certificate evaluator.

This module only composes PROVIDED valid conditional certificates. It does not
certify that a training implementation, DP route, or noise generator meets the
assumptions, and cannot infer guarantees from profiling/heartbeat estimates.

Model:
  Delta* = -eta_star * grad F(w_t)
  Delta_clean = gamma * sum_i a_i u_i(h)
  c_i = -step_scale_i * g_ref, ||u_i-c_i||_L2 <= mode_rms_i
  ||grad F(w_t)-g_ref||_L2 <= gradient_rms_error
  xi = gamma * sum_i a_i z_i, z_i centered, conditionally uncorrelated
  E||z_i||^2 <= d * sigma_i^2, numerical RMS <= r_num

The manifest explicitly attests to update decomposition/weights, conditional
independence and noise timing. Assertions are NOT runtime-verifiable proofs.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .liic_online_certificate import calculate_liic_radius


def _finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: boolean is not a numeric certificate")
    try:
        parsed = float(value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name}: expected finite nonnegative number") from exc
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{name}: expected finite nonnegative number")
    return parsed


def _finite_positive(value: Any, name: str) -> float:
    number = _finite_nonnegative(value, name)
    if number <= 0:
        raise ValueError(f"{name}: expected positive number")
    return number


def certificate_candidate_key(candidate: Any) -> str:
    """Exact selection identity: mode + link mechanism map + DP execution plan."""
    mechanisms = candidate.link_mechanisms or candidate.mechanisms
    return json.dumps(
        [str(candidate.mode), sorted([str(k), str(v)] for k, v in mechanisms.items()),
         str(candidate.dp_execution_plan),
         [getattr(candidate, key, None) for key in (
             "update_noise_multiplier", "sample_embedding_noise_multiplier",
             "sample_label_grad_noise_multiplier", "sample_optimizer_noise_multiplier",
             "sample_embedding_events", "sample_label_grad_events", "sample_optimizer_events",
         )]],
        ensure_ascii=True, separators=(",", ":"),
    )


@dataclass(frozen=True)
class OnlineBoundResult:
    upper: float
    central_norm: float
    radius: float
    ideal_radius: float
    q_aggregate: float
    numeric_radius: float
    cloud_client_ids: tuple[int, ...]
    source: str = "online_upper_bound_provided_conditional_certificates"
    # Conservative lower bound on the SAME conditional expected learning error.
    lower: float = 0.0
    reference_radius: float = 0.0
    mode_radius: float = 0.0
    mode_variance_second_moment: float = 0.0
    # Guaranteed noise energy used for the lower bound, zero unless separately proved.
    q_aggregate_proved_floor: float = 0.0


@lru_cache(maxsize=12)
def _manifest_cached(path: str, mtime_ns: int) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") not in {"dynfl.online_j.v1", "dynfl.online_j.v2"}:
        raise ValueError("online-J manifest must declare schema dynfl.online_j.v1 or dynfl.online_j.v2")
    if data.get("target") != "ideal_full_gradient_one_step":
        raise ValueError("online-J manifest target must be ideal_full_gradient_one_step")
    if not isinstance(data.get("rounds"), dict):
        raise ValueError("online-J manifest requires per-round predecision rounds")
    return data



def _certified_blocks(state, dimension):
    blocks=state.get('parameter_blocks')
    if blocks is None: return ()
    if not isinstance(blocks,list) or not 1<=len(blocks)<=128:
        raise ValueError('parameter_blocks require 1..128 disjoint blocks')
    parsed=[]
    last=0
    for item in blocks:
        if not isinstance(item,dict) or set(item)!={'name','start','end'}:
            raise ValueError('parameter_blocks need name/start/end')
        lo,hi=item['start'],item['end']
        if (not isinstance(item['name'],str) or not item['name'].strip()
            or type(lo) is not int or type(hi) is not int or lo!=last
            or hi<=lo or hi>dimension):
            raise ValueError('parameter_blocks must be contiguous and disjoint')
        parsed.append((lo,hi));last=hi
    if last!=dimension: raise ValueError('parameter_blocks must cover all parameters')
    return tuple(parsed)


def _block_radii(entry,key,count):
    data=entry.get(key)
    if not isinstance(data,list) or len(data)!=count:
        raise ValueError(f'{key} requires independently proved {count} block RMS radii')
    return [_finite_nonnegative(x,f'{key}[{i}]') for i,x in enumerate(data)]


def _evaluate_coupled_v2(
    *, state: dict[str, Any], round_index: int,
    profile: dict[int, Any], client_samples: dict[int, float],
    admitted_client_ids: tuple[int, ...], reaches_cloud: Any,
) -> OnlineBoundResult:
    """Same-world structural cancellation for Delta*=-eta*grad F(w_t).

    Assume F=sum_i p_i F_i, v_i=-eta*grad F_i(w_t), Delta*=sum_i p_i v_i.
    For mode h, Delta_clean=gamma*sum_i a_i u_i(h), with a_i=0 for
    nonparticipating clients; u_i-v_i has a certified conditional error.

    The identity is exact BEFORE norm inequalities:
      Delta_clean-Delta* = sum_i (gamma*a_i-p_i)*vhat_i
        + sum_i (gamma*a_i-p_i)*(v_i-vhat_i)
        + gamma*sum_i a_i*(u_i-v_i).

    The predecision certificates MUST be independently proved for each
    execution path. In particular, not every Split / multistage executor
    permits the per-client linear decomposition assumed here.
    """
    if state.get("round_index") != round_index or state.get("available_before_round") != round_index:
        raise ValueError("online-J v2 certificate has invalid predecision round metadata")
    if not isinstance(state.get("evidence"), str) or not state["evidence"].strip():
        raise ValueError("online-J v2 requires nonempty evidence/provenance")
    assumptions = state.get("assumptions")
    required = (
        "global_objective_is_population_sample_weighted",
        "linear_client_contributions_match_runtime",
        "ideal_reference_rms_bounds_proved",
        "mode_relative_ideal_bounds_proved",
        "noise_shares_drawn_before_secagg_and_weight_fixed",
        "noise_conditionally_zero_mean_and_uncorrelated",
        "aggregate_share_second_moment_bounds_proved",
        "numeric_rms_bound_proved",
    )
    if not isinstance(assumptions, dict) or any(assumptions.get(k) is not True for k in required):
        raise ValueError("online-J v2 required conditional assumptions not certified")
    if state.get("aggregation_rule") != "cloud_participant_sample_weighted":
        raise ValueError("online-J v2 unsupported Cloud aggregation rule")
    gamma = _finite_positive(state.get("cloud_step"), "cloud_step")
    numeric_rms = _finite_nonnegative(state.get("numeric_rms"), "numeric_rms")
    population = state.get("population")
    clients = state.get("clients")
    if not isinstance(population, dict) or not population or not isinstance(clients, dict):
        raise ValueError("online-J v2 requires population and client-mode certificates")
    try:
        population_ids = {int(key) for key in population}
    except (ValueError, TypeError) as exc:
        raise ValueError("online-J v2 invalid population client ID") from exc
    if len(population_ids) != len(population) or population_ids != set(client_samples):
        raise ValueError("online-J v2 population IDs must equal runtime client IDs")
    if set(profile).difference(population_ids) or set(admitted_client_ids).difference(population_ids):
        raise ValueError("online-J v2 profile/admission contains unknown population clients")
    mass_by_id: dict[int, float] = {}
    references: dict[int, list[float]] = {}
    reference_scales: dict[int, float] = {}
    ideal_radii: dict[int, float] = {}
    # Shared-direction encoding avoids O(N*d) reference storage. It is a
    # genuine center certificate only if each client ideal_rms covers its
    # heterogeneity relative to scale_i*reference_direction.
    direction = state.get("reference_direction")
    dimension = None
    if direction is not None:
        if not isinstance(direction, list) or not direction or any(
            isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
            for x in direction
        ):
            raise ValueError("reference_direction must be a finite nonempty vector")
        dimension = len(direction)
        direction = [float(x) for x in direction]
    for cid in sorted(population_ids):
        cert = population.get(str(cid))
        if not isinstance(cert, dict):
            raise ValueError(f"client {cid}: missing ideal gradient reference certificate")
        expected_mass = _finite_nonnegative(cert.get("samples"), f"population[{cid}].samples")
        runtime_mass = _finite_nonnegative(client_samples[cid], f"runtime.samples[{cid}]")
        if not math.isclose(runtime_mass, expected_mass, abs_tol=1e-9, rel_tol=1e-9):
            raise ValueError(f"client {cid}: population/reference sample mass differs from runtime")
        has_vector = "ideal_reference" in cert
        has_scale = "ideal_reference_scale" in cert
        if has_vector == has_scale:
            raise ValueError(f"client {cid}: exactly one ideal_reference or ideal_reference_scale required")
        if has_scale:
            if direction is None:
                raise ValueError(f"client {cid}: ideal_reference_scale requires reference_direction")
            scale = cert["ideal_reference_scale"]
            if (isinstance(scale, bool) or not isinstance(scale, (int, float))
                    or not math.isfinite(scale)):
                raise ValueError(f"client {cid}: ideal_reference_scale must be a finite scalar")
            reference_scales[cid] = float(scale)
        else:
            ref = cert["ideal_reference"]
            if not isinstance(ref, list) or not ref:
                raise ValueError(f"client {cid}: ideal_reference must be a nonempty vector")
            if dimension is None:
                dimension = len(ref)
            if len(ref) != dimension or any(
                isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
                for x in ref
            ):
                raise ValueError(f"client {cid}: ideal_reference dimension/nonfinite error")
            references[cid] = [float(x) for x in ref]
        ideal_radii[cid] = _finite_nonnegative(cert.get("ideal_rms"), f"client {cid}.ideal_rms")
        mass_by_id[cid] = runtime_mass
    total_mass = math.fsum(mass_by_id.values())
    if total_mass <= 0:
        raise ValueError("online-J v2 total population sample mass must be positive")
    cloud = tuple(sorted(cid for cid in admitted_client_ids
                         if cid in profile and reaches_cloud(profile[cid])))
    cloud_mass = math.fsum(mass_by_id[cid] for cid in cloud)
    if cloud and cloud_mass <= 0:
        raise ValueError("online-J v2 admitted Cloud sample mass is zero")
    if not cloud and state.get("no_cloud_update_is_zero_verified") is not True:
        raise ValueError("online-J v2 no-Cloud update is not verified zero")

    uncertainty = state.get("mode_uncertainty", "rms")
    if uncertainty not in {"rms", "bias_variance"}:
        raise ValueError("online-J v2 mode_uncertainty must be rms or bias_variance")
    if uncertainty == "bias_variance" and (
        assumptions.get("mode_fluctuations_centered_given_ideal") is not True or
        assumptions.get("mode_fluctuations_cross_client_orthogonal_given_ideal") is not True
    ):
        raise ValueError("online-J v2 bias_variance needs conditional centering and cross-client orthogonality")

    blocks=_certified_blocks(state,dimension)
    if blocks and uncertainty!='rms':
        raise ValueError('parameter_blocks currently require RMS mode')
    ref_block=[0.]*len(blocks)
    mode_block=[0.]*len(blocks)
    weights = {cid: mass_by_id[cid] / cloud_mass for cid in cloud} if cloud else {}
    center_coordinates = [0.0] * dimension
    shared_direction_coefficient = 0.0
    reference_radius = 0.0
    ideal_radius = 0.0
    for cid in sorted(population_ids):
        p_i = mass_by_id[cid] / total_mass
        a_i = weights.get(cid, 0.0)
        coefficient = gamma * a_i - p_i
        if cid in reference_scales:
            shared_direction_coefficient += coefficient * reference_scales[cid]
        else:
            vhat = references[cid]
            for k in range(dimension):
                center_coordinates[k] += coefficient * vhat[k]
        reference_radius += abs(coefficient) * ideal_radii[cid]
        ideal_radius += p_i * ideal_radii[cid]
        if blocks:
            for j,value in enumerate(_block_radii(population[str(cid)],'ideal_rms_by_block',len(blocks))):
                ref_block[j]+=abs(coefficient)*value
    if direction is not None:
        for k in range(dimension):
            center_coordinates[k] += shared_direction_coefficient * direction[k]

    # An upper bound on DP variance is NOT a lower bound. Enable a positive
    # noise floor only with explicit exact-variance AND independence evidence.
    exact_aggregate_noise = state.get("aggregate_noise_variance_exactly_proved", False)
    if type(exact_aggregate_noise) is not bool:
        raise ValueError("aggregate_noise_variance_exactly_proved must be a boolean")
    if exact_aggregate_noise and assumptions.get(
        "aggregate_noise_orthogonal_to_clean_update_given_predecision"
    ) is not True:
        raise ValueError("exact aggregate noise floor needs clean-update orthogonality proof")
    mode_radius = 0.0
    variance = 0.0
    dp_noise = 0.0
    for cid in cloud:
        entries = clients.get(str(cid))
        if not isinstance(entries, dict) or not isinstance(entries.get("candidates"), dict):
            raise ValueError(f"client {cid}: missing mode certificates")
        key = certificate_candidate_key(profile[cid])
        cert = entries["candidates"].get(key)
        if not isinstance(cert, dict):
            raise ValueError(f"client {cid}: missing certificate for {key}")
        if cert.get("runtime_mode_equivalence_verified") is not True:
            raise ValueError(f"client {cid}: runtime mode contribution is not verified")
        if cert.get("local_dp_included_in_mode_rms") is not True:
            raise ValueError(f"client {cid}: local DP/SGD uncertainty not accounted in mode certificate")
        if "liic_sgd_proof" in cert and uncertainty != "rms":
            raise ValueError("LIIC SGD step proof only supports RMS mode, not unproved bias_variance")
        a_i = weights[cid]
        if uncertainty == "rms":
            # Optional formula-derived LIIC path: the v2 evaluator computes
            # its radius from the *actual* clipped-gradient SGD mechanism.
            # No oracle/current-round update is ever read. For other modes,
            # preserve existing independently certified relative_ideal_rms.
            if "liic_sgd_proof" in cert:
                if profile[cid].mode != "LIIC":
                    raise ValueError("LIIC SGD certificate cannot be used for a different mode")
                if "relative_ideal_rms" in cert:
                    raise ValueError("Specify LIIC proof OR fixed relative_ideal_rms, not both")
                if cid in reference_scales:
                    ref_norm = abs(reference_scales[cid]) * math.sqrt(
                        math.fsum(x * x for x in direction))
                else:
                    ref_norm = math.sqrt(math.fsum(x * x for x in references[cid]))
                derived = calculate_liic_radius(
                    proof=cert["liic_sgd_proof"],
                    ideal_reference_norm=ref_norm,
                    ideal_reference_rms=ideal_radii[cid],
                    vector_dimension=dimension,
                )
                r_mode = derived.relative_ideal_rms
            else:
                r_mode = _finite_nonnegative(
                    cert.get("relative_ideal_rms"), f"client {cid}.relative_ideal_rms")
            mode_radius += gamma * a_i * r_mode
            if blocks:
                for j,value in enumerate(_block_radii(cert,'relative_ideal_rms_by_block',len(blocks))):
                    mode_block[j]+=gamma*a_i*value
        else:
            mode_radius += gamma * a_i * _finite_nonnegative(
                cert.get("relative_ideal_bias"), f"client {cid}.relative_ideal_bias")
            variance += (gamma * a_i)**2 * _finite_nonnegative(
                cert.get("relative_ideal_variance"), f"client {cid}.relative_ideal_variance")
        sigma = _finite_nonnegative(cert.get("aggregate_share_std"), f"client {cid}.aggregate_share_std")
        if exact_aggregate_noise and cert.get("aggregate_share_std_exactly_proved") is not True:
            raise ValueError(
                f"client {cid}: exact aggregate noise floor requires separately proved share std"
            )
        # sigma is the PRE-WEIGHT Gaussian share-coordinate SD. Never count
        # local DP-SGD again here; it belongs in relative_ideal_* certificates.
        dp_noise += (gamma * a_i * sigma)**2

    central_norm = math.sqrt(math.fsum(x*x for x in center_coordinates))
    radius = reference_radius + mode_radius
    q_aggregate = dimension * dp_noise
    if uncertainty == "bias_variance":
        # Centered mode innovations are conditionally orthogonal GIVEN all
        # ideal gradients, in addition to aggregation DP conditional centering.
        base_sq = (central_norm + radius)**2 + variance + q_aggregate
    else:
        # No correlation or zero-mean assumption on model-training errors.
        base_sq = (central_norm + radius)**2 + q_aggregate
    upper = (math.sqrt(base_sq) + numeric_rms)**2
    # The noise energy here is an UPPER bound by default, so it cannot
    # contribute to a learning-error LOWER bound unless exactly proved.
    noise_floor = q_aggregate if exact_aggregate_noise else 0.0
    scalar_clean_floor = max(0.0, central_norm - radius)
    lower = max(
        0.0, math.hypot(scalar_clean_floor, math.sqrt(noise_floor)) - numeric_rms
    )**2
    source=f'online_coupled_v2_{uncertainty}_conditional_certificates'
    if blocks:
        # Distinct coordinate blocks are orthogonal as vectors; the random
        # training errors need NOT be independent across those blocks.
        norms=[math.sqrt(math.fsum(x*x for x in center_coordinates[lo:hi]))
               for lo,hi in blocks]
        radii=[x+y for x,y in zip(ref_block,mode_block)]
        grouped=(math.sqrt(math.fsum((a+r)**2 for a,r in zip(norms,radii))
                           +q_aggregate)+numeric_rms)**2
        # Reverse triangle independently on orthogonal parameter blocks;
        # then add only separately proved independent exact noise energy.
        block_clean_floor=math.sqrt(math.fsum(
            max(0.0,a-r)**2 for a,r in zip(norms,radii)
        ))
        grouped_lower=max(
            0.0, math.hypot(block_clean_floor, math.sqrt(noise_floor))-numeric_rms
        )**2
        if grouped<upper:
            upper=grouped
            source='online_coupled_v2_rms_blockwise_conditional_certificates'
        lower=max(lower,grouped_lower)
    if not math.isfinite(upper) or not math.isfinite(lower):
        raise ValueError("online-J v2 calculated bound is nonfinite")
    return OnlineBoundResult(
        upper=upper, lower=lower, central_norm=central_norm, radius=radius,
        reference_radius=reference_radius, mode_radius=mode_radius,
        mode_variance_second_moment=variance,
        ideal_radius=ideal_radius, q_aggregate=q_aggregate,
        numeric_radius=numeric_rms, cloud_client_ids=cloud,
        q_aggregate_proved_floor=noise_floor,
        source=source,
    )

def certified_j_interval_order(left: OnlineBoundResult, right: OnlineBoundResult) -> str:
    """Certify ordering of *actual* J, never rank by two loose upper bounds.

    Both inputs must refer to the same conditional ideal update and the
    same round. This is a pure interval comparison; callers enforce identity.
    Returns 'left', 'right', or 'undetermined'. Equality is not strict.
    """
    for name, result in (("left", left), ("right", right)):
        if (not isinstance(result, OnlineBoundResult)
                or any(not math.isfinite(value) or value < 0
                       for value in (result.lower, result.upper))
                or result.lower > result.upper + 1e-12 * max(1, result.upper)):
            raise ValueError(f"{name}: invalid online J interval")
    if left.upper < right.lower:
        return "left"
    if right.upper < left.lower:
        return "right"
    return "undetermined"


def evaluate_online_bound(
    *, manifest_path: str, round_index: int, profile: dict[int, Any],
    client_samples: dict[int, float], admitted_client_ids: tuple[int, ...],
    reaches_cloud: Any,
) -> OnlineBoundResult:
    """Compute J <= U conditionally on all declared certificate assumptions.

    Raises ValueError/KeyError on any missing evidence, round or mechanism;
    NEVER substitutes the old Fusion objective or offline calibration table.
    """
    file = Path(manifest_path).expanduser().resolve(strict=True)
    payload = _manifest_cached(str(file), file.stat().st_mtime_ns)
    if payload["schema"] == "dynfl.online_j.v2":
        state = payload["rounds"].get(f"round:{round_index}")
        if not isinstance(state, dict):
            raise ValueError(f"missing ONLINE predecision certificate round:{round_index}")
        return _evaluate_coupled_v2(
            state=state, round_index=round_index, profile=profile,
            client_samples=client_samples, admitted_client_ids=admitted_client_ids,
            reaches_cloud=reaches_cloud,
        )
    state = payload["rounds"].get(f"round:{round_index}")
    if not isinstance(state, dict):
        raise ValueError(f"missing ONLINE predecision certificate round:{round_index}")
    if state.get("round_index") != round_index or state.get("available_before_round") != round_index:
        raise ValueError("online-J certificate has missing/mismatched predecision round metadata")
    assumptions = state.get("assumptions", {})
    required = (
        "weighted_client_contributions_match_runtime",
        "gradient_reference_rms_bound_proved",
        "mode_rms_bounds_proved",
        "noise_shares_drawn_before_secagg_and_weight_fixed",
        "noise_conditionally_zero_mean_and_uncorrelated",
        "numeric_rms_bound_proved",
    )
    if not isinstance(assumptions, dict) or any(assumptions.get(x) is not True for x in required):
        raise ValueError("online-J required conditional assumptions not certified")
    evidence = state.get("evidence", "")
    if not isinstance(evidence, str) or not evidence.strip():
        raise ValueError("online-J certificate requires a meaningful evidence/provenance label")
    gref = state.get("reference_gradient")
    if not isinstance(gref, list) or not gref:
        raise ValueError("online-J reference_gradient must be a nonempty vector")
    if any(not isinstance(v, (int, float)) or isinstance(v, bool) or not math.isfinite(v) for v in gref):
        raise ValueError("online-J reference_gradient contains invalid coordinates")
    grad_norm = math.sqrt(math.fsum(float(v) * float(v) for v in gref))
    dimension = len(gref)
    eta_star = _finite_positive(state.get("ideal_step"), "ideal_step")
    gamma = _finite_positive(state.get("cloud_step"), "cloud_step")
    gradient_rms = _finite_nonnegative(state.get("gradient_rms_error"), "gradient_rms_error")
    numeric_rms = _finite_nonnegative(state.get("numeric_rms"), "numeric_rms")
    clients = state.get("clients")
    if not isinstance(clients, dict):
        raise ValueError("online-J requires client-mode certificates")
    if state.get("aggregation_rule") != "cloud_participant_sample_weighted":
        raise ValueError("unsupported/uncertified Cloud aggregation rule")
    cloud = tuple(sorted(int(cid) for cid in admitted_client_ids
                         if cid in profile and reaches_cloud(profile[cid])))
    mass = math.fsum(_finite_nonnegative(client_samples.get(cid, 0.0), f"samples[{cid}]") for cid in cloud)
    if cloud and mass <= 0:
        raise ValueError("Cloud aggregation has zero admitted sample mass")
    if not cloud and state.get("no_cloud_update_is_zero_verified") is not True:
        raise ValueError("cannot bound Edge-only/no-Cloud profile without a verified zero Cloud update")

    weighted_scale = 0.0
    radius = 0.0
    q_unscaled = 0.0
    for cid in cloud:
        entry = clients.get(str(cid))
        if not isinstance(entry, dict) or not isinstance(entry.get("candidates"), dict):
            raise ValueError(f"no candidate certificate for cloud client {cid}")
        key = certificate_candidate_key(profile[cid])
        cert = entry["candidates"].get(key)
        if not isinstance(cert, dict):
            raise ValueError(f"missing certificate for client {cid}, identity {key}")
        if cert.get("runtime_mode_equivalence_verified") is not True:
            raise ValueError(f"candidate {cid} has no verified update equivalence")
        if cert.get("local_dp_included_in_mode_rms") is not True:
            raise ValueError(f"candidate {cid}: local DP contribution is not tracked in mode RMS")
        center_scale = _finite_nonnegative(cert.get("step_scale"), f"client {cid} step_scale")
        r_mode = _finite_nonnegative(cert.get("mode_rms"), f"client {cid} mode_rms")
        sigma = _finite_nonnegative(cert.get("aggregate_share_std"), f"client {cid} share_std")
        a = float(client_samples[cid]) / mass
        weighted_scale += a * center_scale
        radius += a * r_mode
        q_unscaled += a * a * sigma * sigma

    central_norm = abs(eta_star - gamma * weighted_scale) * grad_norm
    ideal_radius = eta_star * gradient_rms
    mode_radius = gamma * radius
    radius = ideal_radius + mode_radius
    q_aggregate = (gamma * gamma) * dimension * q_unscaled
    upper = (math.sqrt((central_norm + radius) ** 2 + q_aggregate) + numeric_rms) ** 2
    if not math.isfinite(upper):
        raise ValueError("computed online-J upper bound is nonfinite")
    lower = max(0.0, central_norm-radius-numeric_rms)**2
    return OnlineBoundResult(
        upper=upper, lower=lower, central_norm=central_norm, radius=radius,
        reference_radius=ideal_radius, mode_radius=mode_radius,
        ideal_radius=ideal_radius, q_aggregate=q_aggregate,
        numeric_radius=numeric_rms, cloud_client_ids=cloud,
    )
