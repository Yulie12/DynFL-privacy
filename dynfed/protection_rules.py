"""Observer-specific protection rules and execution audits for update releases."""
from __future__ import annotations

from typing import Any
import math

from .privacy import mechanism_uses_dp, mechanism_uses_he


def aggregate_replacement_bound(
    weights: list[float], dependencies: list[set[int]], *, clip_norm: float,
) -> float:
    """Audit a fixed weighted release of whole packets clipped to clip_norm.

    dependencies[j] must conservatively contain every client that can change
    packet j, conditional on the prior public transcript. This function does
    not infer dependencies or certify the transcript, weights, or protocol.
    """
    if len(weights) != len(dependencies):
        raise ValueError("One dependency set is required per packet")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    if not weights or any(not math.isfinite(w) or w < 0 for w in weights):
        raise ValueError("Weights must be nonempty, finite and nonnegative")
    if not math.isclose(math.fsum(weights), 1.0, rel_tol=0, abs_tol=1e-12):
        raise ValueError("Weights must be normalized before sensitivity analysis")
    clients = set().union(*dependencies)
    affected_weight = max((math.fsum(
        w for w, sources in zip(weights, dependencies) if client in sources
    ) for client in clients), default=0.0)
    return 2.0 * clip_norm * affected_weight


def allowed_update_mechanisms(
    *, goal: str, next_dp_release_affordable: bool,
    secure_aggregation_available: bool,
) -> tuple[str, ...]:
    """Return design options, conditional on valid DP and aggregation protocols.

    packet_protection permits either a DP packet or hiding individual inputs
    beyond the authorized aggregate. released_model_dp additionally requires
    DP calibration for the released result. Availability includes key and
    decryption restrictions, not just an installed HE library.
    """
    if goal not in {"packet_protection", "released_model_dp"}:
        raise ValueError(f"Unknown protection goal: {goal}")
    options = []
    if next_dp_release_affordable:
        options.append("dp")
    if secure_aggregation_available and goal == "packet_protection":
        options.append("he3")
    if next_dp_release_affordable and secure_aggregation_available:
        options.append("dp_he3")
    return tuple(options)


def audit_update_release(
    *, mechanism: str, noise_location: str, he_execution: str,
    dp_budget_ok: bool, key_isolation_enforced: bool,
    aggregate_only_decryption_enforced: bool,
) -> dict[str, Any]:
    """Audit observed operations without treating coverage as a DP proof."""
    if noise_location not in {"none", "packet", "aggregate_share"}:
        raise ValueError(f"Unknown noise location: {noise_location}")
    if he_execution not in {"not_selected", "profiled", "real"}:
        raise ValueError(f"Unknown HE execution: {he_execution}")
    dp_requested = mechanism_uses_dp(mechanism)
    he_requested = mechanism_uses_he(mechanism)
    dp_observed = dp_requested and noise_location != "none"
    he_observed = he_requested and he_execution == "real"
    protocol_enforced = (
        he_observed and key_isolation_enforced and aggregate_only_decryption_enforced
    )
    if dp_observed and not dp_budget_ok:
        packet_status = "dp_budget_exceeded"
    elif dp_observed and noise_location == "packet":
        packet_status = "packet_dp_pending_analysis"
    elif protocol_enforced:
        packet_status = "secure_aggregation_configured"
    elif he_observed:
        packet_status = "real_he_pending_role_isolation"
    elif noise_location == "aggregate_share":
        packet_status = "aggregate_noise_without_secure_transport"
    elif he_requested and he_execution == "profiled":
        packet_status = "profiled_he_only"
    elif dp_requested:
        packet_status = "dp_requested_but_not_observed"
    else:
        packet_status = "unprotected_update"
    return {
        "mechanism": mechanism,
        "noise_location": noise_location,
        "he_execution": he_execution,
        "dp_event_observed": dp_observed,
        "dp_budget_ok": dp_budget_ok,
        "he_crypto_observed": he_observed,
        "key_isolation_enforced": key_isolation_enforced,
        "aggregate_only_decryption_enforced": aggregate_only_decryption_enforced,
        "packet_protection_status": packet_status,
        "released_model_dp_status": (
            "dp_coverage_pending_analysis" if dp_observed and dp_budget_ok
            else "no_valid_dp_calibration_for_contribution"
        ),
        "formal_dp_status": "not_established",
    }
