"""Simulator-only predecision *new-J* envelope, not a deployable certificate.

This optional path uses the raw client training examples *before the round*,
which is illegal for an untrusted FL server.  It exists to isolate the
mathematical objective from the old Fusion proxy in a controlled simulation.

The four-mode simulator implements LIIC, LIIEIIIC, edge-only LIIE (zero
Cloud coefficient), and *trusted/no-split-packet-noise/client-unit* LIEIIC.
Untrusted Sample-DP split, extra update DP and real CKKS numerical error
need separate proofs; unsupported candidate paths fail closed.

At the pre-round state let v_i=-eta*grad F_i(w_t), p_i=n_i/N.
For every profile, let c_i=gamma*n_i/sum_{j reaches Cloud} n_j if
client i contributes to Cloud, else zero (LIIE/LIE). Two-stage Edge-to-Cloud
weighted averaging telescopes to the same c_i. The simulator records the
predecision client ideal-update Gram matrix, so A=||sum_i(c_i-p_i)v_i||
is computed without reading future training updates. A conservative bound on
the RMS deviation of each actual upload u_i from v_i is B_{i,h}+||v_i||.
The selected new-J envelope is

  U_t=(A+sum_{i reaches Cloud} c_i*(B_{i,h}+||v_i||))**2.

The simulation assumes no *additional* Cloud/packet Gaussian noise: Q=0.
The local Sample-DP noise is included inside B once. LIIEIIIC includes
all 3 real Edge SGD stages; LIEIIC client-unit trusted clean Split applies
a joint 10 gradient-norm cap (two separately clipped optimizers at 5).
For other mechanisms fail closed, never silently set unmodeled Q=0.

This bound uses no unbiased-gradient assumption or fabricated mode bias.
It is generally VERY LOOSE.  The simulator's ideal gradient oracle and
unverified finite-precision/HE differences prevent formal deployment claims.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .online_j_bound import certificate_candidate_key
from .liic_online_certificate import planned_batch_sizes

SUPPORTED_MODES = frozenset({"LIIC", "LIIEIIIC", "LIEIIC", "LIIE", "LIE"})
CLOUD_MODES = frozenset({"LIIC", "LIIEIIIC", "LIEIIC"})


def finite_signed_finite(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: invalid boolean")
    try:
        x = float(value)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{name}: invalid number") from ex
    if not math.isfinite(x):
        raise ValueError(f"{name}: expected finite value")
    return x


def finite_nonnegative(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name}: invalid boolean")
    try:
        x = float(value)
    except (TypeError, ValueError) as ex:
        raise ValueError(f"{name}: invalid number") from ex
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{name}: expected nonnegative finite value")
    return x


def full_sgd_update_rms_bound(*, parameter_norm: float, dimension: int,
                              batch_sizes: Sequence[int], learning_rate: float,
                              momentum: float, weight_decay: float,
                              gradient_clip: float, noise_multiplier: float,
                              sample_dp: bool) -> float:
    """RMS bound for the *actual* fresh PyTorch SGD momentum recurrence.

    m_{k+1}=mu*m_k+q_k+z_k+lambda*theta_k,
    theta_{k+1}=theta_k-eta*m_{k+1}.
    Minkowski, without any independence/unbiasedness shortcut, gives a valid
    recursion for RMS norms even when q_k depends on past Gaussian noise.
    """
    theta = finite_nonnegative(parameter_norm, "parameter_norm")
    if type(dimension) is not int or dimension <= 0:
        raise ValueError("dimension must be a positive integer")
    eta = finite_nonnegative(learning_rate, "learning_rate")
    mu = finite_nonnegative(momentum, "momentum")
    lam = finite_nonnegative(weight_decay, "weight_decay")
    clip = finite_nonnegative(gradient_clip, "gradient_clip")
    sigma = finite_nonnegative(noise_multiplier, "noise_multiplier")
    if mu >= 1 or not batch_sizes or clip == 0 or (not sample_dp and sigma):
        raise ValueError("unsupported SGD hyperparameters/noise setup")
    velocity = 0.0
    displacement = 0.0
    for b in batch_sizes:
        if type(b) is not int or b <= 0:
            raise ValueError("invalid minibatch size")
        noise_rms = (2 * clip * sigma * math.sqrt(dimension) / b) if sample_dp else 0.0
        velocity = mu * velocity + clip + noise_rms + lam * theta
        displacement += eta * velocity
        theta += eta * velocity
        if not math.isfinite(displacement):
            raise ValueError("SGD RMS envelope overflow")
    return displacement


@dataclass(frozen=True)
class OracleDiagnosticResult:
    upper: float
    lower: float = 0.0
    central_norm: float = 0.0
    radius: float = 0.0
    reference_radius: float = 0.0
    mode_radius: float = 0.0
    ideal_radius: float = 0.0
    q_aggregate: float = 0.0
    numeric_radius: float = 0.0
    q_aggregate_proved_floor: float = 0.0
    mode_variance_second_moment: float = 0.0
    source: str = "SIMULATOR_ORACLE_NEW_J_NOT_DEPLOYMENT_CERTIFIED"


@lru_cache(maxsize=20)
def _load_snapshot(path: str, stamp: int) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") not in {"dynfl.online_j.oracle_diagnostic.v1", "dynfl.online_j.oracle_diagnostic.v2"}:
        raise ValueError("not an oracle diagnostic new-J snapshot")
    if data.get("formal_online_j_certified") is not False or data.get("used_raw_client_training_data") is not True:
        raise ValueError("oracle diagnostic must disclose non-certification and its data access")
    return data


def evaluate_oracle_diagnostic(*, snapshot_path: str, round_index: int,
                               profile: Mapping[int, Any],
                               client_samples: Mapping[int, float],
                               admitted_client_ids: Sequence[int],
                               reaches_cloud: Callable[[Any], bool]) -> OracleDiagnosticResult:
    path = Path(snapshot_path).resolve(strict=True)
    state = _load_snapshot(str(path), path.stat().st_mtime_ns)
    if state.get("round_index") != round_index or state.get("available_before_round") != round_index:
        raise ValueError("stale/late oracle diagnostic: invalid round")
    clients = state["clients"]
    if set(int(k) for k in clients) != set(client_samples):
        raise ValueError("oracle diagnostic population does not match runtime")
    for cid, n in client_samples.items():
        if not math.isclose(float(n), float(clients[str(cid)]["samples"]), rel_tol=1e-10, abs_tol=1e-9):
            raise ValueError(f"client {cid} population sample count differs")
    reference_norm = finite_nonnegative(state["ideal_population_update_norm"], "ideal population norm")
    gamma = finite_nonnegative(state["cloud_step"], "Cloud step")
    cloud = tuple(cid for cid in sorted(admitted_client_ids) if cid in profile and reaches_cloud(profile[cid]))
    mass = math.fsum(float(client_samples[cid]) for cid in cloud)
    if cloud and mass <= 0:
        raise ValueError("Cloud contribution has zero mass")
    mass_total = math.fsum(float(client_samples[cid]) for cid in client_samples)
    if mass_total <= 0:
        raise ValueError("population has zero sample mass")
    triangle_global = math.fsum(
        float(client_samples[cid]) / mass_total
        * finite_nonnegative(clients[str(cid)].get("ideal_client_update_norm"), "client ideal norm")
        for cid in client_samples
    )
    if reference_norm > triangle_global + 1e-8 * max(1.0, triangle_global):
        raise ValueError("oracle population norm violates triangle bound")
    direct_cloud_radius = 0.0
    reference_radius = 0.0
    mode_radius = 0.0
    # Actual two-stage Cloud weights: E sends a sample-weighted average of
    # its admitted clients, Cloud weights that E average by *admitted sample
    # mass*. The coefficients therefore telescope to n_i/sum_cloud(n_j).
    # Edge-only LIIE has coefficient zero, but its -p_i*v_i term remains in A.
    reference_coefficients: list[float] = []
    for cid in sorted(client_samples):
        pop = clients[str(cid)]
        p_i = float(client_samples[cid]) / mass_total
        a_i = (float(client_samples[cid]) / mass) if cid in cloud else 0.0
        c_i = gamma * a_i
        reference_coefficients.append(c_i - p_i)
        ideal_i = finite_nonnegative(pop.get("ideal_client_update_norm"), "client ideal norm")
        if cid not in cloud:
            continue
        cand = profile[cid]
        if cand.mode not in CLOUD_MODES:
            raise ValueError(f"unsupported Cloud mode {cand.mode} at client {cid}")
        key = certificate_candidate_key(cand)
        entry = pop["candidates"].get(key)
        if not isinstance(entry, dict):
            raise ValueError(f"client {cid}: no oracle new-J entry for actual candidate")
        if entry.get("mode") != cand.mode or entry.get("he_only_no_extra_release_dp") is not True:
            raise ValueError(f"client {cid}: unsupported aggregation/noise path")
        b = finite_nonnegative(entry.get("update_rms_bound"), "mode RMS bound")
        direct_cloud_radius += c_i * b
        mode_radius += c_i * (b + ideal_i)
    if state.get("schema") == "dynfl.online_j.oracle_diagnostic.v2":
        # Center A = ||sum_i (c_i-p_i)*v_i||; unlike the v1 common-center
        # triangle surrogate, the v2 Gram matrix preserves the actual
        # *direction* of the predecision ideal client updates. No raw gradient
        # vectors need be written to disk (Gram is still sensitive metadata).
        ids = sorted(client_samples)
        if state.get("ideal_gram_client_ids") != ids:
            raise ValueError("oracle ideal Gram client IDs do not match population")
        gram = state.get("ideal_client_update_gram")
        if not isinstance(gram, list) or len(gram) != len(ids):
            raise ValueError("oracle ideal Gram shape mismatch")
        entries = []
        for i, row in enumerate(gram):
            if not isinstance(row, list) or len(row) != len(ids):
                raise ValueError("oracle ideal Gram row shape mismatch")
            vals = [finite_signed_finite(v, "ideal Gram entry") for v in row]
            ideal_i = finite_nonnegative(clients[str(ids[i])]["ideal_client_update_norm"], "ideal norm")
            if not math.isclose(vals[i], ideal_i * ideal_i, rel_tol=1e-5, abs_tol=1e-8):
                raise ValueError("oracle ideal Gram diagonal/norm mismatch")
            entries.append(vals)
        for i, row in enumerate(entries):
            for j, value in enumerate(row):
                if not math.isclose(value, entries[j][i], rel_tol=1e-7, abs_tol=1e-8):
                    raise ValueError("oracle ideal Gram is not symmetric")
                cap = math.sqrt(max(entries[i][i], 0.0) * max(entries[j][j], 0.0))
                if abs(value) > cap + 1e-6 * max(1.0, cap):
                    raise ValueError("oracle ideal Gram violates Cauchy-Schwarz")
        # Pairwise Cauchy-Schwarz alone does not guarantee that an alleged
        # Gram is globally positive semidefinite. Check the full spectrum.
        import numpy as np
        eigenvalues = np.linalg.eigvalsh(np.asarray(entries, dtype=np.float64))
        if float(eigenvalues[0]) < -1e-6 * max(1.0, max(entries[i][i] for i in range(len(ids)))):
            raise ValueError("oracle ideal Gram is not positive semidefinite")
        weights = [float(client_samples[cid]) / mass_total for cid in ids]
        recovered_global_sq = sum(weights[i] * weights[j] * entries[i][j]
                                  for i in range(len(ids)) for j in range(len(ids)))
        if not math.isclose(max(recovered_global_sq, 0.0), reference_norm ** 2,
                            rel_tol=1e-4, abs_tol=1e-8):
            raise ValueError("oracle population ideal norm/Gram mismatch")
        center_sq = sum(reference_coefficients[i] * reference_coefficients[j] * entries[i][j]
                        for i in range(len(ids)) for j in range(len(ids)))
        if center_sq < -1e-7 * max(1.0, reference_norm ** 2):
            raise ValueError("oracle ideal Gram is not positive semidefinite")
        central_norm = math.sqrt(max(center_sq, 0.0))
        # Each v_i is calculated over that client's entire empirical dataset,
        # so s_i=0 relative to this *simulator* empirical ideal. Precision
        # and real HE residuals are NOT certified by this path.
        upper_coupled = (central_norm + mode_radius) ** 2
    else:
        # Backward-compatible v1 triangle envelope, not a deployment proof.
        for cid in sorted(client_samples):
            pop = clients[str(cid)]
            s_i = finite_nonnegative(pop.get("reference_deviation_upper"), "reference deviation")
            ideal_i = finite_nonnegative(pop.get("ideal_client_update_norm"), "ideal norm")
            if s_i + 1e-8 * max(1.0, ideal_i, reference_norm) < ideal_i + reference_norm:
                raise ValueError("oracle reference cap not derived conservatively")
            reference_radius += abs(gamma * ((float(client_samples[cid]) / mass) if cid in cloud else 0.0)
                                    - float(client_samples[cid]) / mass_total) * s_i
        central_norm = abs(gamma - 1.0) * reference_norm
        upper_coupled = (central_norm + reference_radius + mode_radius) ** 2
    # Minkowski direct upper bound remains valid even if the Gram-based center
    # is too conservative. Both include missing Cloud contribution from LIIE.
    direct = (direct_cloud_radius + reference_norm) ** 2
    # v2 uses the user-specified A+R coupled envelope as the learning
    # objective so absent Cloud ideal contributions are represented directly.
    # The tighter *direct* Minkowski inequality is kept only in v1 for
    # backward compatibility; optimizing that direct envelope would obscure
    # which client ideal contributions were omitted from Cloud.
    upper = upper_coupled if state.get("schema") == "dynfl.online_j.oracle_diagnostic.v2" else min(upper_coupled, direct)
    if not math.isfinite(upper):
        raise ValueError("oracle diagnostic bound is nonfinite")
    return OracleDiagnosticResult(
        upper=upper, central_norm=central_norm,
        radius=reference_radius + mode_radius,
        reference_radius=reference_radius,
        ideal_radius=reference_norm,
        mode_radius=mode_radius,
    )


def make_oracle_snapshot(*, round_index: int, selected: Sequence[tuple[int, Any, Sequence[Any], float]],
                         client_train_indices: Mapping[int, Any], x_train: Any, y_train: Any,
                         global_end: Any, global_edge: Any, model_name: str,
                         input_shape: tuple[int, int, int], device: Any,
                         learning_rate: float, cloud_step: float, weight_decay: float,
                         local_epochs: int, batch_size: int, sample_clip_norm: float,
                         privacy_unit: str, step_cap: int | None,
                         no_dp_ablation: bool = False, clipping_only_ablation: bool = False) -> dict[str, Any]:
    """Build one **simulator-only** predecision bound. Never uses future updates.

    The reference is computed on each client's real local data. This is an
    explicit CENTRAL ORACLE, not an FL-compatible or private measurement.
    """
    from .fmnist_lenet5_dynamic import _calibration_reference_update
    from .training import MODE_SPECS
    import torch
    if step_cap is not None:
        raise ValueError("oracle J supports only uncapped local minibatch schedules")
    if len(selected) != len(client_train_indices):
        raise ValueError("oracle J needs the entire simulated population")
    params = [p for part in (global_end, global_edge) for p in part.parameters() if p.requires_grad]
    d = sum(p.numel() for p in params)
    if d == 0:
        raise ValueError("no trainable parameters")
    theta_sq = math.fsum(float(p.detach().double().square().sum().item()) for p in params)
    theta_norm = math.sqrt(theta_sq)
    momentum = 0.9 if "resnet" in model_name.lower() else 0.0
    mass_total = math.fsum(len(client_train_indices[cid]) for cid in client_train_indices)
    if mass_total <= 0:
        raise ValueError("empty simulated population")
    global_reference: torch.Tensor | None = None
    clients: dict[str, Any] = {}
    ideal_norm_by_client: dict[int, float] = {}
    # Keep reference vectors only during predecision construction. The saved
    # snapshot has a small N-by-N Gram matrix, never N full model gradients.
    ideal_vectors: dict[int, Any] = {}
    for cid, _chosen, candidates, _remaining in selected:
        indices = client_train_indices[cid]
        n = int(len(indices))
        if n == 0:
            raise ValueError("oracle J requires nonempty client datasets")
        # Compute exact full-*empirical* reference gradient for this simulated
        # client, not a public-gradient certificate and not an FL deployment.
        relative = _calibration_reference_update(
            global_end=global_end, global_edge=global_edge,
            x=x_train[indices], y=y_train[indices], device=device,
            input_shape=input_shape, learning_rate=learning_rate,
            batch_size=batch_size,
        )
        parts = []
        for part_name, part_model in (("end", global_end), ("edge", global_edge)):
            for name, p in part_model.named_parameters():
                if p.requires_grad:
                    parts.append(relative[part_name][name].detach().double().flatten().cpu())
        v_i = torch.cat(parts)
        ideal_norm_by_client[cid] = float(torch.linalg.vector_norm(v_i).item())
        ideal_vectors[cid] = v_i
        global_reference = (v_i * (n / mass_total) if global_reference is None
                            else global_reference + v_i * (n / mass_total))
        batch_sizes = planned_batch_sizes(sample_count=n, epochs=local_epochs,
                                          batch_size=batch_size, local_step_cap=None)
        mode_entries: dict[str, Any] = {}
        for candidate in candidates:
            if not candidate.feasible or candidate.mode not in SUPPORTED_MODES:
                continue
            # LIIE is an edge-only return: its update is intentionally absent
            # from the current Cloud aggregate. It needs no Cloud noise term.
            if candidate.mode in {"LIIE", "LIE"}:
                continue
            mechs = candidate.link_mechanisms or candidate.mechanisms
            if any(value not in {"he3", "none", "trusted"} for value in mechs.values()):
                raise ValueError(f"oracle J supports no extra packet DP, not {mechs}")
            cloud_link = "L_C_upd" if candidate.mode == "LIIC" else "E_C_upd"
            if mechs.get(cloud_link) != "he3":
                raise ValueError(f"oracle J requires HE3 for mode {candidate.mode}")
            if candidate.mode == "LIEIIC":
                # split_learning.py clips End and Edge gradients separately
                # at norm 5 for non-Sample-DP split. Their concatenated norm
                # is <=10, a conservative joint clip for the RMS recursion.
                # Sample-DP split instead performs two *different* noisy
                # embedding/gradient mechanisms: cannot reuse full SGD cap.
                if privacy_unit != "client" or no_dp_ablation:
                    raise ValueError("trusted LIEIIC oracle bound requires client-unit clean Split SGD")
                if mechs.get("L_E_emb") != "trusted" or mechs.get("L_E_grad") != "trusted":
                    raise ValueError("LIEIIC requires trusted Client-Edge embedding/gradient path")
                grad_clip = 10.0
                sample_dp = False
                sigma = 0.0
                bound_source = "trusted_split_two_clipped_optimizers_5_plus_5"
            else:
                sample_dp = privacy_unit == "sample" and not no_dp_ablation
                sigma = (0.0 if clipping_only_ablation else candidate.sample_optimizer_noise_multiplier) if sample_dp else 0.0
                if sigma is None:
                    raise ValueError("oracle J requires resolved local optimizer sigma")
                grad_clip = sample_clip_norm if sample_dp else 5.0
                bound_source = "fresh_SGD_clipped_gradient_minkowski_recurrence"
            # The actual executor repeats full local SGD at each Edge
            # multistage loop (LIIEIIIC has three). Using only one pass would
            # undercount its training-noise/optimization displacement.
            stages = max(1, int(MODE_SPECS[candidate.mode].E_edge_loops))
            bound = full_sgd_update_rms_bound(
                parameter_norm=theta_norm, dimension=d, batch_sizes=batch_sizes * stages,
                learning_rate=learning_rate, momentum=momentum,
                weight_decay=weight_decay,
                gradient_clip=grad_clip, noise_multiplier=float(sigma),
                sample_dp=sample_dp,
            )
            mode_entries[certificate_candidate_key(candidate)] = {
                "mode": candidate.mode,
                "update_rms_bound": bound,
                "he_only_no_extra_release_dp": True,
                "bound_source": bound_source,
                "edge_training_stages": stages,
                # Stage20: decision-time shadow data only; this metadata does
                # NOT change the Stage19 learning objective or Pareto order.
                "sample_optimizer_noise_multiplier": float(sigma),
                "sample_optimizer_clip_norm": float(grad_clip),
                "planned_batches_per_stage": list(batch_sizes),
                "sample_dp_optimizer": bool(sample_dp),
            }
        clients[str(cid)] = {"samples": n, "candidates": mode_entries}
    if global_reference is None:
        raise ValueError("no full-population ideal reference")
    reference_norm = float(torch.linalg.vector_norm(global_reference).item())
    # Stage20 deterministic transfer term: ||a * theta_t - Delta_star||.
    # Using the SAME trainable-parameter enumeration as ideal_vectors above.
    # Both the norm and dot product are CENTRAL SIMULATOR ORACLE diagnostics
    # and must never be published as a DP-protected server certificate.
    reference_offset = 0
    theta_dot_reference_terms = []
    for parameter in params:
        flat = parameter.detach().double().flatten().cpu()
        stop = reference_offset + flat.numel()
        theta_dot_reference_terms.append(float(torch.dot(
            flat, global_reference[reference_offset:stop],
        ).item()))
        reference_offset = stop
    if reference_offset != d:
        raise ValueError("Stage20 shadow parameter/reference dimension mismatch")
    theta_dot_reference = math.fsum(theta_dot_reference_terms)
    if not math.isfinite(theta_dot_reference):
        raise ValueError("Stage20 shadow parameter/reference dot is not finite")
    if not math.isfinite(reference_norm):
        raise ValueError("simulator population ideal gradient is nonfinite")
    # Triangle bound for each client's deviation from the common center.
    # This requires only norms, not another 90*d array of private gradients.
    for cid, client in clients.items():
        ideal_i = ideal_norm_by_client[int(cid)]
        client["ideal_client_update_norm"] = ideal_i
        client["reference_deviation_upper"] = ideal_i + reference_norm
    ids = sorted(ideal_vectors)
    gram = [
        [float(torch.dot(ideal_vectors[i], ideal_vectors[j]).item()) for j in ids]
        for i in ids
    ]
    return {
        "schema": "dynfl.online_j.oracle_diagnostic.v2",
        "ideal_gram_client_ids": ids,
        "ideal_client_update_gram": gram,
        "round_index": round_index,
        "available_before_round": round_index,
        "formal_online_j_certified": False,
        "used_raw_client_training_data": True,
        "warning": "CENTRAL SIMULATOR ORACLE ONLY; NOT A PRIVATE FL CERTIFICATE. Numeric/HE implementation error not rigorously covered.",
        "ideal_population_update_norm": reference_norm,
        # Optional Stage20 shadow-only inputs. Existing Stage19 objective is
        # unchanged; old snapshots lacking these fields cannot be replayed.
        "stage20_shadow_inputs": {
            "trainable_dimension": int(d),
            "model_parameter_norm": float(theta_norm),
            "model_parameter_dot_ideal_update": float(theta_dot_reference),
            "local_epochs": int(local_epochs),
            "batch_size": int(batch_size),
            "learning_rate": float(learning_rate),
            "weight_decay": float(weight_decay),
            "momentum": float(momentum),
            "sample_optimizer_clip_norm": float(sample_clip_norm),
            "privacy_unit": str(privacy_unit),
            "assumptions_not_certified": True,
        },
        "objective_formula": "coupled_A_plus_R_squared_gram_center_Q0_diagnostic",
        "cloud_coefficients": "gamma*n_i/sum_admitted_cloud_n; edge_only=0",
        "ideal_step": learning_rate,
        "cloud_step": cloud_step,
        "clients": clients,
    }
