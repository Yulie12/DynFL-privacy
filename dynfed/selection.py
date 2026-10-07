from __future__ import annotations

import csv
import json
import math
import random
import time

import torch
from dataclasses import dataclass, replace
from functools import lru_cache
from itertools import product
from pathlib import Path
from typing import Any, Callable

from .nodes import build_profiles
from .flow_executor import (
    CLOUD_DIRECT_MODES,
    EDGE_CLOUD_MODES,
    EDGE_ONLY_MODES,
    ClientFlowInput,
    summarize_mixed_round_flow,
)
from .privacy import (
    ClientPrivacyLedger,
    SamplePrivacyLedger,
    OBJECT_SIZES,
    PRIVACY_ALPHA,
    PRIVACY_BASE_TIME,
    calibrate_gaussian_noise,
    mechanism_uses_dp,
    mechanism_uses_he,
    utility_penalty,
)
from .training import MODE_SPECS, ModeSpec


MECHANISMS_BY_OBJECT = {
    "emb": ("none", "dp"),
    "logits": ("none",),
    "grad": ("none",),
    "emb_grad": ("none",),
    "upd": ("none", "dp", "he3", "dp_he3"),
    "weakemb": ("none",),
    "strongemb": ("none",),
    "pseudo_label": ("none",),
}

OBJECT_RISK = {
    "grad": 0.82,
    "emb": 0.72,
    "strongemb": 0.68,
    "upd": 0.58,
    "weakemb": 0.5,
    "pseudo_label": 0.46,
}

MECHANISM_RISK = {
    "none": 1.0,
    "trusted": 0.35,
    "dp": 0.42,
    "he2": 0.18,
    "he3": 0.12,
    "dp_he3": 0.08,
}

# Only objects with an available privacy mechanism belong to the privacy-risk
# model. Logits and returned embedding gradients are post-processing payloads.
PRIVACY_RISK_OBJECTS = frozenset(
    obj for obj, mechanisms in MECHANISMS_BY_OBJECT.items()
    if mechanisms != ("none",)
)


def _perf_add(profiler: dict[str, Any] | None, key: str, value: float = 1.0) -> None:
    if profiler is not None:
        profiler[key] = profiler.get(key, 0) + value


@dataclass(frozen=True)
class ExposurePrivacyRequirement:
    """Per-client requirements over the transmissions actually exposed by a mode.

    Confidentiality and DP are independent requirements.  A link in
    ``plaintext_forbidden_links`` must use HE; a link in ``dp_required_links``
    must use DP.  If both apply, the selected mechanism must provide both.
    Links not named here are allowed to remain plaintext and do not require DP.
    """

    plaintext_forbidden_links: frozenset[str] = frozenset()
    dp_required_links: frozenset[str] = frozenset()


def paper_client_privacy_requirement() -> ExposurePrivacyRequirement:
    """Current paper profile: protect split embeddings and Cloud-bound updates.

    This is an explicit client requirement profile, not a trust label on Edge or
    Cloud. Split embeddings leaving the client require feature DP; model updates
    released toward Cloud require the configured update protection.
    """

    cloud_update_links = frozenset({"L_C_upd", "E_C_upd"})
    split_embedding_links = frozenset({"L_E_emb", "L_C_emb"})
    return ExposurePrivacyRequirement(
        plaintext_forbidden_links=cloud_update_links,
        dp_required_links=cloud_update_links | split_embedding_links,
    )


@dataclass(frozen=True)
class SelectionConfig:
    rounds: int = 100
    num_clients: int = 100
    num_edges: int = 10
    seed: int = 42
    excluded_modes: tuple[str, ...] = ()
    initial_epsilon: float = 8.0
    dp_event_epsilon: float = 0.05
    dp_emb_epsilon: float = 8.0
    dp_upd_epsilon: float = 8.0
    dp_noise_multiplier: float = 0.0002
    dp_delta: float = 1e-5
    dp_accounting_mode: str = "rdp_auto"
    privacy_unit: str = "client"
    dp_feature_epsilon_budget: float | None = None
    dp_update_epsilon_budget: float | None = None
    dp_feature_noise_multiplier: float | None = None
    dp_update_noise_multiplier: float | None = None
    dp_sample_epsilon_budget: float | None = None
    dp_sample_embedding_noise_multiplier: float | None = None
    dp_sample_label_grad_noise_multiplier: float | None = None
    dp_sample_optimizer_noise_multiplier: float | None = None
    client_heterogeneity: float = 2.0
    edge_heterogeneity: float = 1.5
    # Q86/Q87: optional staged Normal -> Constrained -> Normal resource scenario.
    resource_scenario: str = "none"
    constrained_start_fraction: float = 1.0 / 3.0
    constrained_end_fraction: float = 2.0 / 3.0
    communication_constrained_multiplier: float = 0.35
    compute_constrained_multiplier: float = 2.0
    network_jitter: float = 0.25
    network_periodic_amplitude: float = 0.2
    network_period_rounds: float = 3.0
    end_edge_rate_mb_s: float = 5.0
    end_cloud_rate_mb_s: float = 2.2
    edge_cloud_rate_mb_s: float = 8.0
    end_edge_base_latency_sec: float = 0.015
    end_cloud_base_latency_sec: float = 0.04
    edge_cloud_base_latency_sec: float = 0.01
    resource_limit: float = 1.35
    memory_limit: float = 1.35
    time_limit: float = 8.0  # reporting/reference only; not a universal hard deadline
    fast_client_deadlines: tuple[tuple[int, float], ...] = ()
    # When enabled, only clients carrying an explicit fast-response deadline
    # may terminate at Edge. Ordinary clients must use a Cloud-reaching mode.
    edge_only_requires_fast_deadline: bool = False
    risk_limit: float = 0.5
    aggregation_fraction: float = 1.0
    output_dir: str = "out/selection"
    require_feasible: bool = False
    L_block_cycles: int = 5
    split_interaction_mode: str = "fixed"
    split_batch_size: int = 128
    fl_first_split_on_demand: bool = False
    privacy_local_epochs: int = 1
    # Whether split End-side execution actually has trainable parameters.
    # This affects Sample-DP optimizer events only; full-local updates
    # remain independently trainable/accounted.
    split_end_optimizer_enabled: bool = True
    trusted_edge_split_execution: bool = False
    allow_he: bool = True
    update_mechanism_options: tuple[str, ...] = ("dp", "he3", "dp_he3")
    update_protection_goal: str = "packet_protection"
    # Opt-in LIIE protocol ablation. This is NOT yet a joint Pareto action.
    liie_edge_dp_plan: str = "independent"
    # Uniform Cloud release protocol chosen jointly with the entire mode profile.
    # Legacy preserves the prior runtime; pareto evaluates both complete profiles.
    cloud_dp_plan: str = "legacy"
    assume_encoder_feasible: bool = False
    minibatch_reference_samples: float = 600.0
    edge_cpu_limit: float = 15.0  # per-edge CPU capacity
    cloud_cpu_limit: float = 20.0  # global cloud CPU capacity
    require_cloud_participation: bool = False
    require_edge_cloud_coverage: bool = False
    min_edge_cloud_fusion_ratio: float = 0.0
    enforce_cloud_dp_stability: bool = False
    cloud_dp_stability_threshold: float = 1.0
    pareto_archive_size: int = 16
    pareto_max_iters: int = 50
    pareto_neighbor_top_k: int = 0
    pareto_conflict_only: bool = False
    pareto_beam_size: int = 4
    pareto_norm_eps: float = 1e-9
    cloud_fusion_xi: float = 0.2
    cloud_fusion_eps: float = 0.05
    switch_mode_cost: float = 0.02
    switch_placement_cost: float = 0.08
    edge_aggregation_beta: float = 0.01
    edge_aggregation_fixed: float = 0.02
    cloud_aggregation_beta: float = 0.015
    cloud_aggregation_fixed: float = 0.04
    omega_mu: float = 1.0
    omega_smoothness: float = 1.0
    omega_learning_rate: float = 0.15
    omega_local_variance: float = 0.035
    omega_feature_clip_norm: float = 1.0
    omega_feature_clip_excess_sq: float = 0.02
    omega_feature_jacobian_norm: float = 1.0
    omega_feature_lipschitz: float = 1.0
    omega_feature_backward_bias_sq: float = 0.01
    omega_feature_clf_pairwise_spread: float = 0.04
    # Number of scalar coordinates in one transmitted split representation.
    # The formal training entry overwrites this with the measured split tensor
    # size for the selected model/input shape.
    omega_feature_dimension: float = 1.0
    omega_update_clip_norm: float = 1.0
    omega_update_clip_excess_sq: float = 0.02
    omega_update_dimension: float = 61706.0
    embedding_payload_mb: float = 1.6
    update_payload_mb: float = 4.0
    mainline_fusion: bool = False

    def __post_init__(self) -> None:
        seen_fast_clients: set[int] = set()
        for client_id, deadline in self.fast_client_deadlines:
            if int(client_id) < 0:
                raise ValueError("fast-client ids must be non-negative")
            if int(client_id) in seen_fast_clients:
                raise ValueError("fast-client deadlines must contain unique client ids")
            if float(deadline) <= 0.0:
                raise ValueError("fast-client deadlines must be positive")
            seen_fast_clients.add(int(client_id))
        if self.mainline_fusion:
            raise ValueError("mainline_fusion/Method2 global-release overlay has been removed from the current DynFL design")


@dataclass(frozen=True)
class Candidate:
    mode: str
    mechanisms: dict[str, str]
    time: float
    accuracy: float
    risk: float
    epsilon_used: float
    communication_volume: float
    feasible_resource: bool
    feasible_privacy: bool
    feasible_risk: bool
    feasible_time: bool
    feasible_edge: bool = True
    feasible_cloud: bool = True
    pre_aggregation_time: float | None = None
    omega_feature_clip_excess_sq: float | None = None
    feature_dp_events: int = 0
    update_dp_events: int = 0
    feature_epsilon_after: float = 0.0
    update_epsilon_after: float = 0.0
    link_mechanisms: dict[str, str] | None = None
    memory_requirement: float = 0.0
    memory_capacity: float = float("inf")
    feasible_memory: bool = True
    first_aggregation_arrival_time: float = 0.0
    edge_to_cloud_time: float = 0.0
    return_path_time: float = 0.0
    edge_aggregation_payload: float = 0.0
    cloud_aggregation_payload: float = 0.0
    link_metrics: tuple[dict[str, Any], ...] = ()
    global_release_required: bool = False
    feature_noise_multiplier: float | None = None
    update_noise_multiplier: float | None = None
    dp_execution_plan: str = "independent"

    # Sample-level DP metadata. Keep these separate from the existing
    # client-level feature/update accounting fields above.
    sample_embedding_events: int = 0
    sample_label_grad_events: int = 0
    sample_optimizer_events: int = 0
    sample_epsilon_after: float = 0.0
    sample_embedding_noise_multiplier: float | None = None
    sample_label_grad_noise_multiplier: float | None = None
    sample_optimizer_noise_multiplier: float | None = None

    @property
    def feasible(self) -> bool:
        return (
            self.feasible_resource
            and self.feasible_memory
            and self.feasible_privacy
        )

    @property
    def feasible_device(self) -> bool:
        return self.feasible_resource and self.feasible_memory


def candidate_link_mechanisms(candidate: Candidate) -> dict[str, str]:
    return dict(candidate.link_mechanisms or {})


def candidate_mechanisms_for_object(candidate: Candidate, obj: str) -> list[str]:
    if candidate.link_mechanisms:
        spec = MODE_SPECS.get(candidate.mode)
        if spec is None:
            return []
        transmissions = _mode_link_transmissions(
            candidate.mode,
            local_block_cycles=1,
            edge_loops=spec.E_edge_loops,
        )
        if candidate.global_release_required:
            transmissions = _fusion_link_transmissions(candidate.mode, 1, spec.E_edge_loops)
        object_by_link = {link_id: event_obj for link_id, event_obj, _count, _eligible in transmissions}
        return [
            mechanism
            for link_id, mechanism in candidate.link_mechanisms.items()
            if object_by_link.get(link_id) == obj
        ]
    mechanism = candidate.mechanisms.get(obj)
    return [] if mechanism is None else [mechanism]


def candidate_link_mechanism(
    candidate: Candidate,
    link_id: str,
    *,
    fallback_object: str = "upd",
) -> str:
    if candidate.link_mechanisms and link_id in candidate.link_mechanisms:
        return candidate.link_mechanisms[link_id]
    return candidate.mechanisms.get(fallback_object, "none")


def candidate_has_he(candidate: Candidate) -> bool:
    mechanisms = (
        candidate.link_mechanisms.values()
        if candidate.link_mechanisms
        else candidate.mechanisms.values()
    )
    return any(mechanism_uses_he(str(mechanism)) for mechanism in mechanisms)


def candidate_mechanism_label(candidate: Candidate) -> str:
    mechanisms = candidate.link_mechanisms or candidate.mechanisms
    return ";".join(f"{key}:{value}" for key, value in sorted(mechanisms.items()))


@lru_cache(maxsize=128)
def resolved_privacy_parameters(config: SelectionConfig) -> dict[str, float | int | str]:
    """Resolve total targets and noise multipliers used by selection and training."""
    feature_budget = float(
        config.initial_epsilon
        if config.dp_feature_epsilon_budget is None
        else config.dp_feature_epsilon_budget
    )
    update_budget = float(
        config.initial_epsilon
        if config.dp_update_epsilon_budget is None
        else config.dp_update_epsilon_budget
    )
    if feature_budget <= 0.0 or update_budget <= 0.0:
        raise ValueError("DP epsilon targets must be positive")

    per_mode_counts = []
    for mode, spec in MODE_SPECS.items():
        if mode in config.excluded_modes:
            continue
        if (config.trusted_edge_split_execution or config.mainline_fusion) and mode == "LIC":
            continue
        events = _mode_link_transmissions(
            mode,
            config.L_block_cycles,
            spec.E_edge_loops,
        )
        feature_events = sum(
            _record_dp_event_count(config, mode, count)
            for link_id, obj, count, privacy_eligible in events
            if privacy_eligible
            and obj == "emb"
            and not (
                config.trusted_edge_split_execution
                and link_id.startswith("L_E_")
            )
        )
        update_events = sum(
            count
            for link_id, obj, count, privacy_eligible in events
            if privacy_eligible
            and obj == "upd"
            and not (
                config.trusted_edge_split_execution
                and link_id.startswith("L_E_")
            )
        )
        per_mode_counts.append((feature_events, update_events))

    max_feature_events_per_round = max((item[0] for item in per_mode_counts), default=0)
    max_update_events_per_round = max((item[1] for item in per_mode_counts), default=0)
    feature_horizon_events = config.rounds * max_feature_events_per_round
    update_horizon_events = max(1, config.rounds * max_update_events_per_round)

    if config.dp_accounting_mode == "rdp_auto":
        feature_noise_multiplier = (
            calibrate_gaussian_noise(
                feature_budget,
                config.dp_delta,
                feature_horizon_events,
            )
            if feature_horizon_events > 0
            else 1.0
        )
        update_noise_multiplier = calibrate_gaussian_noise(
            update_budget,
            config.dp_delta,
            update_horizon_events,
        )
    elif config.dp_accounting_mode == "rdp_manual":
        feature_noise_multiplier = float(
            config.dp_noise_multiplier
            if config.dp_feature_noise_multiplier is None
            else config.dp_feature_noise_multiplier
        )
        update_noise_multiplier = float(
            config.dp_noise_multiplier
            if config.dp_update_noise_multiplier is None
            else config.dp_update_noise_multiplier
        )
    else:
        raise ValueError(
            "dp_accounting_mode must be 'rdp_auto' or 'rdp_manual'"
        )

    return {
        "accounting_mode": config.dp_accounting_mode,
        "feature_budget": feature_budget,
        "update_budget": update_budget,
        "delta": float(config.dp_delta),
        "feature_noise_multiplier": feature_noise_multiplier,
        "update_noise_multiplier": update_noise_multiplier,
        "max_feature_events_per_round": max_feature_events_per_round,
        "max_update_events_per_round": max_update_events_per_round,
        "feature_horizon_events": feature_horizon_events,
        "update_horizon_events": update_horizon_events,
        "feature_dp_enabled": feature_horizon_events > 0,
    }


@lru_cache(maxsize=512)
def resolved_sample_privacy_parameters(
    config: SelectionConfig,
    horizon_events: int,
) -> dict[str, float | int | str]:
    """Resolve the unified sample-level DP target and Gaussian multipliers.

    Sample-level privacy uses one epsilon budget and one RDP composition state.
    Under rdp_auto all sample-DP Gaussian mechanisms share one multiplier,
    calibrated against the supplied total horizon event count. Under
    rdp_manual the three mechanism classes may use distinct multipliers.
    """
    sample_budget = float(
        config.initial_epsilon
        if config.dp_sample_epsilon_budget is None
        else config.dp_sample_epsilon_budget
    )
    if sample_budget <= 0.0:
        raise ValueError("sample-level DP epsilon target must be positive")

    total_horizon_events = int(horizon_events)
    if total_horizon_events != horizon_events or total_horizon_events < 0:
        raise ValueError("horizon_events must be a non-negative integer")

    if config.dp_accounting_mode == "rdp_auto":
        shared_sigma = (
            calibrate_gaussian_noise(
                sample_budget,
                config.dp_delta,
                total_horizon_events,
            )
            if total_horizon_events > 0
            else 1.0
        )
        embedding_sigma = shared_sigma
        label_grad_sigma = shared_sigma
        optimizer_sigma = shared_sigma

    elif config.dp_accounting_mode == "rdp_manual":
        embedding_sigma = float(
            config.dp_noise_multiplier
            if config.dp_sample_embedding_noise_multiplier is None
            else config.dp_sample_embedding_noise_multiplier
        )
        label_grad_sigma = float(
            config.dp_noise_multiplier
            if config.dp_sample_label_grad_noise_multiplier is None
            else config.dp_sample_label_grad_noise_multiplier
        )
        optimizer_sigma = float(
            config.dp_noise_multiplier
            if config.dp_sample_optimizer_noise_multiplier is None
            else config.dp_sample_optimizer_noise_multiplier
        )

        if (
            embedding_sigma <= 0.0
            or label_grad_sigma <= 0.0
            or optimizer_sigma <= 0.0
        ):
            raise ValueError(
                "sample-level DP noise multipliers must be positive"
            )

    else:
        raise ValueError(
            "dp_accounting_mode must be 'rdp_auto' or 'rdp_manual'"
        )

    return {
        "accounting_mode": config.dp_accounting_mode,
        "sample_budget": sample_budget,
        "delta": float(config.dp_delta),
        "horizon_events": total_horizon_events,
        "embedding_noise_multiplier": embedding_sigma,
        "label_grad_noise_multiplier": label_grad_sigma,
        "optimizer_noise_multiplier": optimizer_sigma,
    }

def mode_aware_feature_noise_multiplier(
    config: SelectionConfig,
    feature_events: int,
    *,
    resolved: dict[str, Any] | None = None,
) -> float | None:
    """Return the feature-DP sigma for one candidate round.

    Under automatic RDP calibration, the shared sigma is calibrated against the
    maximum number of feature releases in any admissible mode.  Scaling sigma
    by sqrt(m / m_max) for a candidate with m releases preserves the same
    per-round Gaussian RDP charge because m / sigma_m^2 is constant.  This
    therefore removes needless worst-case noise from lighter split modes while
    remaining safe under arbitrary dynamic mode switching.  Manual accounting
    keeps the explicitly configured sigma unchanged.
    """
    m = max(int(feature_events), 0)
    if m <= 0:
        return None
    privacy = resolved_privacy_parameters(config) if resolved is None else resolved
    shared_sigma = float(privacy["feature_noise_multiplier"])
    if config.dp_accounting_mode != "rdp_auto":
        return shared_sigma
    m_max = max(int(privacy["max_feature_events_per_round"]), 0)
    if m_max <= 0:
        return shared_sigma
    return shared_sigma * math.sqrt(float(m) / float(m_max))


def build_client_privacy_ledger(config: SelectionConfig) -> ClientPrivacyLedger:
    resolved = resolved_privacy_parameters(config)
    return ClientPrivacyLedger(
        feature_budget=float(resolved["feature_budget"]),
        update_budget=float(resolved["update_budget"]),
        delta=float(resolved["delta"]),
        feature_noise_multiplier=float(resolved["feature_noise_multiplier"]),
        update_noise_multiplier=float(resolved["update_noise_multiplier"]),
    )


def build_sample_privacy_ledger(config: SelectionConfig) -> SamplePrivacyLedger:
    """Build the unified sample-level RDP ledger for one client."""
    sample_budget = float(
        config.initial_epsilon
        if config.dp_sample_epsilon_budget is None
        else config.dp_sample_epsilon_budget
    )
    return SamplePrivacyLedger(
        budget=sample_budget,
        delta=float(config.dp_delta),
    )


def build_privacy_ledger(
    config: SelectionConfig,
) -> ClientPrivacyLedger | SamplePrivacyLedger:
    """Dispatch to the ledger matching the configured privacy unit."""
    if config.privacy_unit == "client":
        return build_client_privacy_ledger(config)
    if config.privacy_unit == "sample":
        return build_sample_privacy_ledger(config)
    raise ValueError("privacy_unit must be 'client' or 'sample'")


@dataclass(frozen=True)
class ProfileEvaluation:
    profile: dict[int, Candidate]
    system_latency: float
    # Kept as a compatibility field name for existing result readers/tests;
    # its value is now the paper's DP perturbation objective J_DP.
    system_omega: float
    cloud_fusion_ratio: float
    admitted_client_ids: tuple[int, ...] = ()
    profile_signature: tuple[int, ...] = ()
    fusion_distortion: float = 0.0
    fusion_cosine_distortion: float = 0.0
    fusion_objective_enabled: bool = False
    # Raw DP second moment J_DP.  ``system_omega`` is the selector's formal
    # learning objective J_learn = J_fusion^ub + J_DP.  Keep this separate so
    # diagnostics can still report the DP term on its own.
    dp_perturbation: float | None = None
    feature_perturbation: float = 0.0
    update_clip_perturbation: float = 0.0
    fusion_bound: float = 0.0

    @property
    def system_dp(self) -> float:
        # Backward compatibility for old tests/readers that construct a
        # ProfileEvaluation directly.
        return self.system_omega if self.dp_perturbation is None else self.dp_perturbation

    @property
    def system_learning_error(self) -> float:
        return self.system_omega


@dataclass(frozen=True)
class _ProfileOmegaStats:
    total_samples: float
    client_samples: dict[int, float]
    edge_total_samples: dict[int, float]
    cloud_samples_by_edge: dict[int, float]
    client_bias_by_edge: dict[int, float]
    client_variance_by_edge: dict[int, float]
    edge_group_samples: dict[tuple[int, str, str], float]
    edge_group_components: dict[tuple[int, str, str], tuple[float, float]]


@dataclass(frozen=True)
class _OmegaComponents:
    client_bias: float
    client_variance: float
    edge_bias: float = 0.0
    edge_variance: float = 0.0


@dataclass(frozen=True)
class _FullBufferGroupSummary:
    kind: str
    edge_aggregation_time: float
    terminal_time: float
    cloud_arrival_time: float
    cloud_aggregation_payload: float
    return_path_time: float
    member_count: int
    edge_payload_sum: float
    cloud_payload_sum: float
    arrival_top: tuple[float, int, int, int]
    arrival_second: tuple[float, int, int, int] | None
    return_top: tuple[float, int, int, int]
    return_second: tuple[float, int, int, int] | None
    edge_upload_top: tuple[float, int, int, int]
    edge_upload_second: tuple[float, int, int, int] | None
    cloud_payload_top: tuple[float, int, int, int]
    cloud_payload_second: tuple[float, int, int, int] | None


@dataclass(slots=True)
class _FullBufferLatencySummary:
    """Transient latency-only summary for one Pareto neighbor."""

    kind: str
    terminal_time: float
    cloud_arrival_time: float
    cloud_aggregation_payload: float
    return_path_time: float


@dataclass(frozen=True)
class _FullBufferFlowStats:
    client_inputs: dict[int, ClientFlowInput]
    groups: dict[tuple[str, int, str], tuple[ClientFlowInput, ...]]
    summaries: dict[tuple[str, int, str], _FullBufferGroupSummary]
    admitted_client_ids: tuple[int, ...]


def run_selection_experiment(
    config: SelectionConfig,
    policies: list[str],
    *,
    simulation_rounds: int | None = None,
) -> dict[str, Any]:
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clients, edges = build_profiles(
        num_clients=config.num_clients,
        num_edges=config.num_edges,
        client_heterogeneity=config.client_heterogeneity,
        edge_heterogeneity=config.edge_heterogeneity,
        seed=config.seed,
    )
    edge_by_id = {edge.edge_id: edge for edge in edges}
    rng = random.Random(config.seed)

    all_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    archive_rows: list[dict[str, Any]] = []

    # Assign per-end sensitivity based on edge
    # Edge 0: high sensitivity (fast response, edge modes)
    # Edge 1: medium sensitivity (balanced, edge→cloud)
    # Edge 2: very low sensitivity (high accuracy, cloud-direct)
    client_sensitivity: dict[int, float] = {}
    for client in clients:
        edge_idx = client.edge_id
        base_sens = {0: 0.85, 1: 0.35, 2: 0.05}.get(edge_idx, 0.50)
        cf_offset = (1.0 - min(client.compute_factor, 2.0) / 2.0) * 0.15
        sens = max(0.02, min(0.98, base_sens + cf_offset))
        client_sensitivity[client.client_id] = sens

    for policy in policies:
        privacy_ledgers = {
            client.client_id: build_client_privacy_ledger(config)
            for client in clients
        }
        remaining_epsilon = {
            client_id: ledger.remaining_budget
            for client_id, ledger in privacy_ledgers.items()
        }
        policy_rows: list[dict[str, Any]] = []
        previous_choices: dict[int, Candidate] = {}
        fixed_mode_assignments: dict[int, str] = {}
        fixed_privacy_profiles: dict[int, tuple[dict[str, str], float | None]] = {}

        evaluated_rounds = (
            config.rounds
            if simulation_rounds is None
            else max(1, min(int(simulation_rounds), config.rounds))
        )
        for round_idx in range(evaluated_rounds):
            round_selected: list[tuple[int, Candidate, list[Candidate], float]] = []
            for client in clients:
                rem = remaining_epsilon[client.client_id]
                candidates = enumerate_candidates(
                    config=config,
                    client_id=client.client_id,
                    edge_factor=edge_by_id[client.edge_id].compute_factor,
                    compute_factor=client.compute_factor,
                    memory_capacity_factor=client.memory_capacity_factor,
                    samples=client.samples,
                    remaining_epsilon=rem,
                    round_idx=round_idx,
                    rng=rng,
                    policy=policy,
                    privacy_ledger=privacy_ledgers[client.client_id],
                    fixed_privacy_profile=(
                        fixed_privacy_profiles.get(client.client_id)
                        if policy in {"fixed_mode_fixed_privacy", "dynamic_mode_fixed_privacy"}
                        else None
                    ),
                    frozen_mode=(
                        fixed_mode_assignments.get(client.client_id)
                        if policy in {"fixed_mode_fixed_privacy", "fixed_mode_dynamic_privacy"}
                        else None
                    ),
                )
                selected = choose_candidate(
                    candidates,
                    policy=policy,
                    rng=rng,
                    require_feasible=config.require_feasible,
                    end_sensitivity=client_sensitivity[client.client_id],
                    time_limit=config.time_limit,
                    remaining_epsilon=rem,
                )
                round_selected.append((client.client_id, selected, candidates, rem))

            if policy in {
                "ours",
                "fixed_mode_fixed_privacy",
                "dynamic_mode_fixed_privacy",
                "fixed_mode_dynamic_privacy",
                "full_dynfl",
            }:
                client_samples = {client.client_id: float(client.samples) for client in clients}
                client_edges = {client.client_id: int(client.edge_id) for client in clients}
                selection_diagnostics: dict[str, Any] = {}
                profile_selection_config = (
                    replace(config, require_edge_cloud_coverage=False)
                    if policy in {"fixed_mode_fixed_privacy", "fixed_mode_dynamic_privacy"}
                    and bool(fixed_mode_assignments)
                    else config
                )
                round_selected, profile_evaluation = choose_global_pareto_profile(
                    config=profile_selection_config,
                    selected=round_selected,
                    client_samples=client_samples,
                    client_edges=client_edges,
                    previous_choices=previous_choices,
                    diagnostics=selection_diagnostics,
                )
                for archive_index, evaluation in enumerate(
                    selection_diagnostics.get("archive", ())
                ):
                    profile_candidates = tuple(evaluation.profile.values())
                    archive_rows.append(
                        {
                            "policy": policy,
                            "round": round_idx,
                            "archive_index": archive_index,
                            "selected": evaluation.profile_signature
                            == profile_evaluation.profile_signature,
                            "system_latency_objective": evaluation.system_latency,
                            "dp_perturbation_objective": evaluation.system_dp,
                            "feature_perturbation_objective": evaluation.feature_perturbation,
                            "fusion_bound_objective": evaluation.fusion_bound,
                            "learning_error_objective": evaluation.system_learning_error,
                            "cloud_fusion_ratio": evaluation.cloud_fusion_ratio,
                            "he_clients": sum(
                                candidate_has_he(candidate)
                                for candidate in profile_candidates
                            ),
                            "dp_clients": sum(
                                any(
                                    mechanism_uses_dp(mechanism)
                                    for mechanism in (
                                        candidate.link_mechanisms
                                        or candidate.mechanisms
                                    ).values()
                                )
                                for candidate in profile_candidates
                            ),
                            "mode_counts": json.dumps(
                                {
                                    mode: sum(
                                        candidate.mode == mode
                                        for candidate in profile_candidates
                                    )
                                    for mode in MODE_SPECS
                                },
                                sort_keys=True,
                            ),
                            "evaluated_profile_count": selection_diagnostics.get(
                                "evaluated_profile_count", 0
                            ),
                        }
                    )
            else:
                profile_evaluation = None

            if (
                policy in {"fixed_mode_fixed_privacy", "fixed_mode_dynamic_privacy"}
                and not fixed_mode_assignments
            ):
                fixed_mode_assignments = {
                    client_id: selected.mode
                    for client_id, selected, _candidates, _rem in round_selected
                }
            if policy in {"fixed_mode_fixed_privacy", "dynamic_mode_fixed_privacy"}:
                for client_id, selected, _candidates, _rem in round_selected:
                    if client_id not in fixed_privacy_profiles and selected.mode != "SKIP":
                        fixed_privacy_profiles[client_id] = (
                            dict(selected.mechanisms),
                            selected.update_noise_multiplier,
                        )

            previous_choices = {
                client_id: selected
                for client_id, selected, _candidates, _rem in round_selected
            }

            for client_id, selected, candidates, rem in round_selected:
                client = clients[client_id]
                ledger = privacy_ledgers[client_id]
                projection = ledger.add(
                    selected.feature_dp_events,
                    selected.update_dp_events,
                    feature_noise_multiplier=selected.feature_noise_multiplier,
                    update_noise_multiplier=selected.update_noise_multiplier,
                )
                remaining_after = ledger.remaining_budget
                remaining_epsilon[client_id] = remaining_after
                row = {
                    "policy": policy,
                    "round": round_idx,
                    "client_id": client_id,
                    "edge_id": client.edge_id,
                    "mode": selected.mode,
                    "mechanisms": candidate_mechanism_label(selected),
                    "time": selected.time,
                    "pre_aggregation_time": selected.pre_aggregation_time,
                    "accuracy_estimate": selected.accuracy,
                    "risk": selected.risk,
                    "epsilon_used": selected.epsilon_used,
                    "remaining_epsilon": remaining_after,
                    "feature_dp_events": selected.feature_dp_events,
                    "update_dp_events": selected.update_dp_events,
                    "feature_noise_multiplier": selected.feature_noise_multiplier if selected.feature_noise_multiplier is not None else "",
                    "update_noise_multiplier": selected.update_noise_multiplier if selected.update_noise_multiplier is not None else "",
                    "feature_epsilon": projection.feature_epsilon_after,
                    "update_epsilon": projection.update_epsilon_after,
                    "communication_volume": selected.communication_volume,
                    "feasible": selected.feasible,
                    "feasible_resource": selected.feasible_resource,
                    "feasible_memory": selected.feasible_memory,
                    "memory_requirement": selected.memory_requirement,
                    "memory_capacity": selected.memory_capacity,
                    "feasible_privacy": selected.feasible_privacy,
                    "feasible_risk": selected.feasible_risk,
                    "feasible_time": selected.feasible_time,
                    "feasible_edge": selected.feasible_edge,
                    "feasible_cloud": selected.feasible_cloud,
                    "feasible_candidates": sum(item.feasible for item in candidates),
                    "total_candidates": len(candidates),
                    "budget_ok_candidates": sum(
                        item.epsilon_used <= rem + 1e-12 for item in candidates
                    ),
                    "fixed_mode_candidates": sum(
                        item.mode == "LIIEIIIC" for item in candidates
                    ),
                    "fixed_mode_feasible_candidates": sum(
                        item.mode == "LIIEIIIC" and item.feasible for item in candidates
                    ),
                    "fixed_mode_budget_ok_candidates": sum(
                        item.mode == "LIIEIIIC"
                        and item.feasible
                        and item.epsilon_used <= rem + 1e-12
                        for item in candidates
                    ),
                    "fixed_mode_epsilon_min": min(
                        (item.epsilon_used for item in candidates if item.mode == "LIIEIIIC"),
                        default="",
                    ),
                    "fixed_mode_epsilon_max": max(
                        (item.epsilon_used for item in candidates if item.mode == "LIIEIIIC"),
                        default="",
                    ),
                    "sensitivity": client_sensitivity[client_id],
                    "system_latency_objective": profile_evaluation.system_latency if profile_evaluation else "",
                    "dp_perturbation_objective": profile_evaluation.system_dp if profile_evaluation else "",
                    "feature_perturbation_objective": profile_evaluation.feature_perturbation if profile_evaluation else "",
                    "fusion_bound_objective": profile_evaluation.fusion_bound if profile_evaluation else "",
                    "learning_error_objective": profile_evaluation.system_learning_error if profile_evaluation else "",
                    "cloud_fusion_ratio": profile_evaluation.cloud_fusion_ratio if profile_evaluation else "",
                    "admitted_client_ids_objective": ";".join(str(cid) for cid in profile_evaluation.admitted_client_ids) if profile_evaluation else "",
                }
                policy_rows.append(row)
                all_rows.append(row)

        summary_rows.append(_summarize_policy(policy, policy_rows, config.time_limit))

    _write_csv(output_dir / "round_selection.csv", all_rows)
    _write_csv(output_dir / "summary_table.csv", summary_rows)
    _write_csv(output_dir / "pareto_archive.csv", archive_rows)
    _write_json(
        output_dir / "config.json",
        config.__dict__
        | {
            "policies": policies,
            "simulation_rounds": (
                config.rounds if simulation_rounds is None else int(simulation_rounds)
            ),
        },
    )
    return {
        "output_dir": str(output_dir),
        "summary_table": str(output_dir / "summary_table.csv"),
        "round_selection": str(output_dir / "round_selection.csv"),
        "pareto_archive": str(output_dir / "pareto_archive.csv"),
        "summaries": summary_rows,
    }


def enumerate_candidates(
    *,
    config: SelectionConfig,
    client_id: int,
    edge_factor: float,
    compute_factor: float,
    samples: int,
    remaining_epsilon: float,
    round_idx: int,
    rng: random.Random,
    policy: str,
    current_edge_load: float = 0.0,
    current_cloud_load: float = 0.0,
    memory_capacity_factor: float = 1.0,
    allow_none: bool = False,
    privacy_ledger: ClientPrivacyLedger | SamplePrivacyLedger | None = None,
    privacy_requirement: ExposurePrivacyRequirement | None = None,
    fast_response_deadline: float | None = None,
    fixed_privacy_profile: tuple[dict[str, str], float | None] | None = None,
    frozen_mode: str | None = None,
    mode_audit: dict[str, Any] | None = None,
) -> list[Candidate]:
    validate_update_protection_goal(config, policy)
    if privacy_ledger is not None:
        if config.privacy_unit == "client" and not isinstance(
            privacy_ledger,
            ClientPrivacyLedger,
        ):
            raise TypeError(
                "client privacy_unit requires ClientPrivacyLedger"
            )
        if config.privacy_unit == "sample" and not isinstance(
            privacy_ledger,
            SamplePrivacyLedger,
        ):
            raise TypeError(
                "sample privacy_unit requires SamplePrivacyLedger"
            )
    candidates: list[Candidate] = []
    # Counts describe candidates, not clients. Stages are snapshots; their
    # differences must not be conflated with Pareto rejection.
    audit_stages: dict[str, dict[str, int]] = {}
    audit_exclusions: dict[str, str] = {}
    audit_counts: dict[str, dict[str, int]] = {
        name: {mode: 0 for mode in MODE_SPECS}
        for name in ("mechanism_assignments", "fixed_profile_mismatches",
                     "missing_fixed_sigma", "budget_calibration_failures")
    }
    def snapshot(stage: str) -> None:
        audit_stages[stage] = {
            mode: sum(item.mode == mode for item in candidates)
            for mode in MODE_SPECS
        }

    # ------------------------------------------------------------
    # Admission-first mode pruning.
    #
    # Cheap client resource/QoS state decides which modes are worth
    # expanding.  DP/HE mechanism enumeration and candidate estimation
    # happen only after this mode-level pruning.
    # ------------------------------------------------------------
    admitted_mode_specs = []

    for mode, spec in MODE_SPECS.items():
        if mode in config.excluded_modes:
            audit_exclusions[mode] = "excluded_mode"
            continue

        if (config.trusted_edge_split_execution or config.mainline_fusion) and mode == "LIC":
            audit_exclusions[mode] = "trusted_edge_or_mainline_excludes_LIC"
            continue

        if config.require_cloud_participation and not _mode_reaches_cloud(spec):
            audit_exclusions[mode] = "cloud_participation_required"
            continue

        # Ordinary clients are not allowed to terminate at Edge.
        # The existence of a fast-response deadline grants Edge-only
        # modes admission; exact deadline satisfaction is checked later.
        if (
            config.edge_only_requires_fast_deadline
            and fast_response_deadline is None
            and not _mode_reaches_cloud(spec)
        ):
            audit_exclusions[mode] = "ordinary_client_requires_cloud"
            continue

        admitted_mode_specs.append((mode, spec))

    # For the new QoS-conditioned admission scheme, perform the FL-first
    # resource decision before mechanism expansion as well.
    #
    # If at least one full-local mode fits the device, split modes are
    # unnecessary.  Otherwise only device-feasible split modes are admitted.
    if (
        config.edge_only_requires_fast_deadline
        and config.fl_first_split_on_demand
        and _policy_uses_fl_first_mode_admissibility(policy)
    ):
        full_local_modes = {"LIIE", "LIIC", "LIIEIIIC"}
        device_feasible_specs = []

        for mode, spec in admitted_mode_specs:
            (
                _sample_scale,
                _local_load,
                feasible_resource,
                _memory_requirement,
                _memory_capacity,
                feasible_memory,
            ) = _mode_device_feasibility_metrics(
                config=config,
                local_work=spec.local_work,
                local_memory=spec.local_memory,
                samples=samples,
                memory_capacity_factor=memory_capacity_factor,
            )

            if not feasible_resource:
                audit_exclusions.setdefault(
                    mode,
                    "resource_infeasible_before_generation",
                )
                continue

            if not feasible_memory:
                audit_exclusions.setdefault(
                    mode,
                    "memory_infeasible_before_generation",
                )
                continue

            device_feasible_specs.append((mode, spec))

        full_local_device_feasible = any(
            mode in full_local_modes
            for mode, _spec in device_feasible_specs
        )

        if full_local_device_feasible:
            for mode, _spec in device_feasible_specs:
                if mode not in full_local_modes:
                    audit_exclusions.setdefault(
                        mode,
                        "full_local_feasible_excludes_split",
                    )

            admitted_mode_specs = [
                (mode, spec)
                for mode, spec in device_feasible_specs
                if mode in full_local_modes
            ]
        else:
            admitted_mode_specs = [
                (mode, spec)
                for mode, spec in device_feasible_specs
                if mode not in full_local_modes
            ]

    for mode, spec in admitted_mode_specs:
        policy_allow_none = allow_none or policy in {
            "performance_only",
            "best_accuracy",
            "accuracy_oracle",
        }
        assignments = (
            _sample_mechanism_assignments(
                spec,
                allow_he=config.allow_he,
                mainline_fusion=config.mainline_fusion,
            )
            if config.privacy_unit == "sample"
            else _mechanism_assignments(
                spec,
                policy,
                config.allow_he,
                policy_allow_none,
                trusted_edge_split_execution=config.trusted_edge_split_execution,
                update_mechanism_options=config.update_mechanism_options,
                mainline_fusion=config.mainline_fusion,
                privacy_requirement=privacy_requirement,
            )
        )
        audit_counts["mechanism_assignments"][mode] = len(assignments)
        if not assignments:
            audit_exclusions[mode] = "no_allowed_mechanism_assignment"
        for mechanisms, link_mechanisms in assignments:
            if fixed_privacy_profile is not None:
                fixed_mechanisms, _fixed_sigma = fixed_privacy_profile
                if mechanisms != fixed_mechanisms:
                    audit_counts["fixed_profile_mismatches"][mode] += 1
                    continue
            # sample selector accounting branch
            if config.privacy_unit == "sample":
                (
                    sample_embedding_events,
                    sample_label_grad_events,
                    sample_optimizer_events,
                ) = _sample_dp_event_counts(
                    config,
                    mode,
                    samples,
                )

                total_sample_events = (
                    sample_embedding_events
                    + sample_label_grad_events
                    + sample_optimizer_events
                )

                fixed_sigma = (
                    fixed_privacy_profile[1]
                    if fixed_privacy_profile is not None
                    else None
                )

                try:
                    if total_sample_events <= 0:
                        sample_parameters = (
                            resolved_sample_privacy_parameters(
                                config,
                                0,
                            )
                        )
                        embedding_sigma = float(
                            sample_parameters[
                                "embedding_noise_multiplier"
                            ]
                        )
                        label_grad_sigma = float(
                            sample_parameters[
                                "label_grad_noise_multiplier"
                            ]
                        )
                        optimizer_sigma = float(
                            sample_parameters[
                                "optimizer_noise_multiplier"
                            ]
                        )

                    elif fixed_privacy_profile is not None:
                        if fixed_sigma is None:
                            audit_counts[
                                "missing_fixed_sigma"
                            ][mode] += 1
                            continue

                        embedding_sigma = float(fixed_sigma)
                        label_grad_sigma = float(fixed_sigma)
                        optimizer_sigma = float(fixed_sigma)

                    elif config.dp_accounting_mode == "rdp_auto":
                        remaining_rounds = max(
                            1,
                            int(config.rounds)
                            - int(round_idx),
                        )

                        planned_sample_events = (
                            int(total_sample_events)
                            * remaining_rounds
                        )

                        if isinstance(
                            privacy_ledger,
                            SamplePrivacyLedger,
                        ):
                            shared_sigma = (
                                privacy_ledger
                                .minimum_feasible_shared_noise(
                                    planned_sample_events
                                )
                            )
                        else:
                            sample_parameters = (
                                resolved_sample_privacy_parameters(
                                    config,
                                    planned_sample_events,
                                )
                            )
                            shared_sigma = float(
                                sample_parameters[
                                    "optimizer_noise_multiplier"
                                ]
                            )

                        embedding_sigma = float(shared_sigma)
                        label_grad_sigma = float(shared_sigma)
                        optimizer_sigma = float(shared_sigma)

                    else:
                        sample_parameters = (
                            resolved_sample_privacy_parameters(
                                config,
                                0,
                            )
                        )
                        embedding_sigma = float(
                            sample_parameters[
                                "embedding_noise_multiplier"
                            ]
                        )
                        label_grad_sigma = float(
                            sample_parameters[
                                "label_grad_noise_multiplier"
                            ]
                        )
                        optimizer_sigma = float(
                            sample_parameters[
                                "optimizer_noise_multiplier"
                            ]
                        )

                except ValueError:
                    audit_counts[
                        "budget_calibration_failures"
                    ][mode] += 1
                    continue

                candidates.append(
                    _estimate_candidate(
                        config=config,
                        mode=mode,
                        spec=spec,
                        mechanisms=mechanisms,
                        link_mechanisms=link_mechanisms,
                        client_id=client_id,
                        edge_factor=edge_factor,
                        compute_factor=compute_factor,
                        samples=samples,
                        remaining_epsilon=remaining_epsilon,
                        round_idx=round_idx,
                        rng=rng,
                        current_edge_load=current_edge_load,
                        current_cloud_load=current_cloud_load,
                        memory_capacity_factor=memory_capacity_factor,
                        privacy_ledger=privacy_ledger,
                        update_noise_multiplier=None,
                        sample_embedding_noise_multiplier=(
                            embedding_sigma
                        ),
                        sample_label_grad_noise_multiplier=(
                            label_grad_sigma
                        ),
                        sample_optimizer_noise_multiplier=(
                            optimizer_sigma
                        ),
                        fast_response_deadline=(
                            fast_response_deadline
                        ),
                    )
                )
                continue

            update_events = sum(
                count
                for link_id, obj, count, privacy_eligible in _mode_link_transmissions(
                    mode, _split_interaction_count(config, samples), spec.E_edge_loops
                )
                if privacy_eligible
                and obj == "upd"
                and mechanism_uses_dp(link_mechanisms[link_id])
            )
            fixed_sigma = fixed_privacy_profile[1] if fixed_privacy_profile is not None else None
            if update_events > 0 and fixed_privacy_profile is not None:
                if fixed_sigma is None:
                    audit_counts["missing_fixed_sigma"][mode] += 1
                    continue
                noise_tiers = (float(fixed_sigma),)
            elif update_events > 0 and privacy_ledger is not None:
                try:
                    # Reserve enough privacy budget for the same candidate-level
                    # update-DP exposure through the remaining training horizon.
                    # Calibrating only the next release makes the minimum sigma
                    # consume the entire lifetime epsilon budget in one round.
                    remaining_rounds = max(1, int(config.rounds) - int(round_idx))
                    planned_update_events = int(update_events) * remaining_rounds
                    sigma_min = privacy_ledger.minimum_feasible_update_noise(
                        planned_update_events
                    )
                except ValueError:
                    audit_counts["budget_calibration_failures"][mode] += 1
                    continue
                # Step44: lifetime-aware calibration already reserves the
                # remaining candidate-level DP exposure horizon.  Enumerate
                # only that minimum feasible sigma; larger multiplicative
                # tiers add current distortion without expanding the current
                # paper model's future privacy-feasible mode coverage.
                noise_tiers = (float(sigma_min),)
            else:
                noise_tiers = (
                    float(resolved_privacy_parameters(config)["update_noise_multiplier"]),
                )
            for update_sigma in noise_tiers:
                candidates.append(
                    _estimate_candidate(
                        config=config,
                        mode=mode,
                        spec=spec,
                        mechanisms=mechanisms,
                        link_mechanisms=link_mechanisms,
                        client_id=client_id,
                        edge_factor=edge_factor,
                        compute_factor=compute_factor,
                        samples=samples,
                        remaining_epsilon=remaining_epsilon,
                        round_idx=round_idx,
                        rng=rng,
                        current_edge_load=current_edge_load,
                        current_cloud_load=current_cloud_load,
                        memory_capacity_factor=memory_capacity_factor,
                        privacy_ledger=privacy_ledger,
                        update_noise_multiplier=update_sigma,
                        fast_response_deadline=fast_response_deadline,
                    )
                )
    snapshot("generated")

    # These legacy audit stage names are retained for CSV compatibility.
    # Under admission-first, QoS/resource pruning has already happened before
    # candidate/mechanism generation.
    snapshot("after_qos_cloud_admissibility")

    # Preserve the historical post-generation FL-first behavior for configs
    # that have not opted into the new admission-first rule.
    if (
        not config.edge_only_requires_fast_deadline
        and config.fl_first_split_on_demand
        and _policy_uses_fl_first_mode_admissibility(policy)
    ):
        full_local_modes = {"LIIE", "LIIC", "LIIEIIIC"}
        full_local_device_feasible = any(
            candidate.mode in full_local_modes and candidate.feasible_device
            for candidate in candidates
        )
        if full_local_device_feasible:
            candidates = [
                candidate
                for candidate in candidates
                if candidate.mode in full_local_modes
            ]

    # Q72/Q73: fast-response QoS is a hard feasibility gate only for clients
    # that explicitly carry a per-client deadline. Ordinary clients keep time
    # in the latency objective and are not rejected by a universal time limit.
    snapshot("after_fl_first")
    if fast_response_deadline is not None:
        candidates = [candidate for candidate in candidates if candidate.feasible_time]
    snapshot("after_deadline")
    candidates = _apply_policy_candidate_filters(config, policy, candidates)
    snapshot("after_policy")
    if frozen_mode is not None:
        candidates = [
            candidate for candidate in candidates
            if candidate.mode == frozen_mode
        ]
    snapshot("returned")
    if mode_audit is not None:
        mode_audit.clear()
        mode_audit["stages"] = audit_stages
        mode_audit["generation_exclusions"] = audit_exclusions
        mode_audit["generation_counts"] = audit_counts
        mode_audit["feasible"] = {
            mode: sum(item.mode == mode and item.feasible for item in candidates)
            for mode in MODE_SPECS
        }
        mode_audit["budget_feasible"] = {
            mode: sum(
                item.mode == mode and item.feasible
                and item.epsilon_used <= remaining_epsilon + 1e-12
                for item in candidates
            ) for mode in MODE_SPECS
        }
        mode_audit["rejections"] = {
            reason: {
                mode: sum(item.mode == mode and not getattr(item, flag)
                          for item in candidates)
                for mode in MODE_SPECS
            }
            for reason, flag in (
                ("resource", "feasible_resource"),
                ("memory", "feasible_memory"),
                ("privacy", "feasible_privacy"),
                ("risk_diagnostic", "feasible_risk"),
                ("time_diagnostic", "feasible_time"),
                ("edge_diagnostic", "feasible_edge"),
                ("cloud_diagnostic", "feasible_cloud"),
            )
        }
        mode_audit["rejections"]["epsilon_limit"] = {
            mode: sum(item.mode == mode and item.epsilon_used > remaining_epsilon + 1e-12
                      for item in candidates)
            for mode in MODE_SPECS
        }
        # Resource/privacy/memory are hard gates; risk/time/edge/cloud flags
        # are separately reported and are NOT automatically hard rejections.
    return candidates


def _apply_policy_candidate_filters(
    config: SelectionConfig,
    policy: str,
    candidates: list[Candidate],
) -> list[Candidate]:
    """Apply shared admissibility rules before an adaptive policy selects."""
    validate_update_protection_goal(config, policy)
    candidates = [c for c in candidates if candidate_meets_update_goal(config, c)]
    fixed_mode = {
        "fixed_fedavg": "LIIC",
        "fixed_splitfed": "LIEIIC",
        "fixed_splitfed_no_protection": "LIEIIC",
        "fixed_splitfed_label_dp": "LIEIIC",
        "fixed_splitfed_trusted_edge": "LIEIIC",
        "fixed_splitfed_dp": "LIEIIC",
        "fixed_hfl": "LIIEIIIC",
        "fixed_liieiiic": "LIIEIIIC",
        "ours_fixed_liieiiic": "LIIEIIIC",
    }.get(policy)
    if fixed_mode is not None:
        candidates = [
            candidate for candidate in candidates
            if candidate.mode == fixed_mode
        ]
    if (
        config.enforce_cloud_dp_stability
        and policy in {"individual_optimal", "random"}
    ):
        return _stable_cloud_candidate_pool(config, candidates)
    return candidates


def _mode_reaches_cloud(spec: ModeSpec) -> bool:
    return spec.client_target == "cloud" or bool(spec.edge_to_cloud_objects)


def _candidate_reaches_cloud(candidate: Candidate) -> bool:
    if candidate.global_release_required:
        return True
    spec = MODE_SPECS.get(candidate.mode)
    return bool(spec and _mode_reaches_cloud(spec))


def validate_update_protection_goal(config: SelectionConfig, policy: str) -> None:
    if config.mainline_fusion:
        if not config.trusted_edge_split_execution:
            raise ValueError("Mainline fusion requires the trusted edge execution domain")
        if abs(float(config.aggregation_fraction) - 1.0) > 1e-12:
            raise ValueError("Mainline fusion requires aggregation_fraction=1.0")
        if config.update_protection_goal != "released_model_dp":
            raise ValueError("Mainline fusion requires update_protection_goal=released_model_dp")
    if config.update_protection_goal not in {"packet_protection", "released_model_dp"}:
        raise ValueError("Unknown update protection goal")
    if config.update_protection_goal == "released_model_dp":
        if not config.trusted_edge_split_execution:
            raise ValueError("Released model DP gate currently requires the trusted edge domain")
        if policy in {"no_protection", "fixed_he", "fixed_splitfed_no_protection",
                      "fixed_splitfed_trusted_edge"}:
            raise ValueError(f"{policy} is a control without result DP; use packet_protection separately")


def candidate_meets_update_goal(config: SelectionConfig, candidate: Candidate) -> bool:
    """Necessary cloud update coverage only, not a transcript privacy proof."""
    if config.mainline_fusion:
        return candidate.mode != "LIC" and candidate.global_release_required
    if config.update_protection_goal == "packet_protection":
        return True
    if config.update_protection_goal != "released_model_dp":
        raise ValueError("Unknown update protection goal")
    if not _candidate_reaches_cloud(candidate):
        return True
    cloud_links = {
        key: value for key, value in (candidate.link_mechanisms or {}).items()
        if key.endswith("_C_upd")
    }
    return bool(cloud_links) and all(mechanism_uses_dp(m) for m in cloud_links.values())


def choose_candidate(
    candidates: list[Candidate],
    policy: str,
    rng: random.Random,
    require_feasible: bool = False,
    mode_bonus: dict[str, float] | None = None,
    end_sensitivity: float | None = None,
    time_limit: float | None = None,
    remaining_epsilon: float | None = None,
    compute_factor: float | None = None,
) -> Candidate:
    resource_feasible = [item for item in candidates if item.feasible_device]
    if not resource_feasible:
        return skipped_candidate()

    privacy_feasible = [item for item in resource_feasible if item.feasible_privacy]
    if not privacy_feasible:
        return skipped_candidate()

    feasible = [item for item in privacy_feasible if item.feasible]
    if require_feasible and not feasible:
        return skipped_candidate()
    pool = feasible or privacy_feasible

    if policy == "ours_time_first":
        return min(pool, key=lambda item: (item.time, -item.accuracy, item.risk))
    if policy == "ours_acc_first":
        return max(pool, key=lambda item: (item.accuracy, -item.time, -item.risk))
    if policy == "ours_ideal":
        return choose_ideal_point(pool)
    if policy == "individual_optimal":
        return choose_ideal_point(pool)
    if policy == "ours_knee":
        return choose_knee_point(pool)
    if policy == "random":
        return rng.choice(pool)
    fixed_mode = {
        "fixed_fedavg": "LIIC",
        "fixed_splitfed": "LIEIIC",
        "fixed_splitfed_no_protection": "LIEIIC",
        "fixed_splitfed_label_dp": "LIEIIC",
        "fixed_splitfed_trusted_edge": "LIEIIC",
        "fixed_splitfed_dp": "LIEIIC",
        "fixed_hfl": "LIIEIIIC",
        "fixed_liieiiic": "LIIEIIIC",
        "ours_fixed_liieiiic": "LIIEIIIC",
    }.get(policy)
    if fixed_mode is not None:
        fixed_pool = [c for c in pool if c.mode == fixed_mode]
        if not fixed_pool:
            return skipped_candidate()
        return min(
            fixed_pool,
            key=lambda item: (
                _local_omega_proxy(item),
                item.time,
                item.risk,
                _candidate_key(item),
            ),
        )
    if policy == "performance_only":
        return max(resource_feasible or candidates, key=lambda item: item.accuracy)
    if policy == "privacy_only":
        return min(pool, key=lambda item: (item.risk, item.epsilon_used, item.time))
    if policy in {"no_protection", "fixed_splitfed_no_protection"}:
        accuracy_pool = [c for c in resource_feasible if _candidate_reaches_cloud(c)]
        if not accuracy_pool:
            accuracy_pool = resource_feasible
        return max(accuracy_pool, key=lambda c: (c.accuracy, -c.time))
    if policy == "accuracy_oracle":
        best_feasible = [c for c in pool if c.feasible]
        if not best_feasible:
            best_feasible = resource_feasible
        cloud_feasible = [c for c in best_feasible if _candidate_reaches_cloud(c)]
        oracle_pool = cloud_feasible or best_feasible
        if not oracle_pool:
            return skipped_candidate()
        return max(oracle_pool, key=lambda c: (c.accuracy, -c.time, -c.risk))
    if policy == "best_accuracy":
        best_feasible = [c for c in pool if c.feasible]
        if not best_feasible:
            best_feasible = resource_feasible
        if not best_feasible:
            return skipped_candidate()
        return max(best_feasible, key=lambda c: c.accuracy)
    if policy in {"fixed_dp", "fixed_he", "fixed_dp_he"}:
        return max(pool, key=lambda item: (item.accuracy, -item.time))

    # Legacy local selection path. The paper method uses the global Pareto
    # solver after resource, memory, and privacy feasibility filtering.
    if not feasible:
        feasible = resource_feasible
        if remaining_epsilon is not None:
            budget_ok = [c for c in feasible if c.epsilon_used <= remaining_epsilon + 1e-12]
            if budget_ok:
                feasible = budget_ok
    if not feasible:
        return skipped_candidate()

    pool2 = feasible

    # Privacy budget hard constraint.
    if remaining_epsilon is not None:
        budget_ok = [c for c in pool2 if c.epsilon_used <= remaining_epsilon + 1e-12]
        if budget_ok:
            pool2 = budget_ok
        else:
            wider = [c for c in (feasible or resource_feasible)
                     if c.epsilon_used <= remaining_epsilon + 1e-12]
            if wider:
                pool2 = wider
            else:
                zero_eps = [c for c in pool2 if c.epsilon_used <= 1e-12]
                if zero_eps:
                    pool2 = zero_eps
                else:
                    return skipped_candidate()

    if remaining_epsilon is not None:
        safe_pool = [c for c in pool2 if c.epsilon_used <= remaining_epsilon + 1e-12]
        if safe_pool:
            pool2 = safe_pool
        elif pool2:
            zero_eps = [c for c in pool2 if c.epsilon_used <= 1e-12]
            if zero_eps:
                pool2 = zero_eps
            else:
                return skipped_candidate()

    return _sensitivity_weighted_ideal(
        pool2, mode_bonus, end_sensitivity or 0.5,
    )


def pareto_frontier(candidates: list[Candidate]) -> list[Candidate]:
    frontier = []
    for candidate in candidates:
        dominated = False
        for other in candidates:
            no_worse = other.time <= candidate.time and other.accuracy >= candidate.accuracy
            strictly_better = other.time < candidate.time or other.accuracy > candidate.accuracy
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    return frontier or candidates


def choose_ideal_point(candidates: list[Candidate]) -> Candidate:
    frontier = pareto_frontier(candidates)
    t_values = [item.time for item in frontier]
    a_values = [item.accuracy for item in frontier]
    t_min, t_max = min(t_values), max(t_values)
    a_min, a_max = min(a_values), max(a_values)

    def distance(item: Candidate) -> tuple[float, float, float]:
        t_norm = _safe_norm(item.time, t_min, t_max)
        a_norm = _safe_norm(a_max - item.accuracy, 0.0, a_max - a_min)
        return (math.sqrt(t_norm * t_norm + a_norm * a_norm), item.risk, item.epsilon_used)

    return min(frontier, key=distance)


def choose_knee_point(candidates: list[Candidate]) -> Candidate:
    frontier = sorted(pareto_frontier(candidates), key=lambda item: item.time)
    if len(frontier) <= 2:
        return choose_ideal_point(frontier)

    t_values = [item.time for item in frontier]
    a_values = [item.accuracy for item in frontier]
    t_min, t_max = min(t_values), max(t_values)
    a_min, a_max = min(a_values), max(a_values)
    points = [(_safe_norm(item.time, t_min, t_max), _safe_norm(item.accuracy, a_min, a_max)) for item in frontier]
    start = points[0]
    end = points[-1]

    def line_distance(point: tuple[float, float]) -> float:
        x0, y0 = point
        x1, y1 = start
        x2, y2 = end
        numerator = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
        denominator = math.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)
        if denominator <= 1e-12:
            return 0.0
        return numerator / denominator

    return max(frontier, key=lambda item: (line_distance(points[frontier.index(item)]), item.accuracy, -item.time))


def _mode_search_pool_audit(pools: dict[int, list[Candidate]]) -> dict[str, dict[str, int]]:
    """Count the actual per-client candidates admitted to a search branch."""
    return {
        mode: {
            "candidates": sum(item.mode == mode for pool in pools.values() for item in pool),
            "clients": sum(any(item.mode == mode for item in pool) for pool in pools.values()),
        }
        for mode in MODE_SPECS
    }


def _mode_profile_distribution(profile: dict[int, Candidate]) -> dict[str, int]:
    return {mode: sum(item.mode == mode for item in profile.values()) for mode in MODE_SPECS}


def _mode_archive_audit(archive: list[ProfileEvaluation] | tuple[ProfileEvaluation, ...]) -> dict[str, int]:
    return {
        mode: sum(any(item.mode == mode for item in evaluation.profile.values())
                  for evaluation in archive)
        for mode in MODE_SPECS
    }


def choose_global_pareto_profile(
    *,
    config: SelectionConfig,
    selected: list[tuple[int, Candidate, list[Candidate], float]],
    client_samples: dict[int, float],
    client_edges: dict[int, int] | None = None,
    previous_choices: dict[int, Candidate] | None = None,
    objective: str = "pareto",
    search_method: str = "bounded",
    diagnostics: dict[str, Any] | None = None,
    previous_client_updates: dict[int, dict[str, dict[str, torch.Tensor]]] | None = None,
    fusion_objective_enabled: bool = False,
) -> tuple[list[tuple[int, Candidate, list[Candidate], float]], ProfileEvaluation]:
    """Approximate TeX Algorithm 1 over a global client profile.

    Each client contributes a feasible candidate set S_i. The search keeps a
    bounded non-dominated archive over (T_sys, J_learn), expands profiles by
    changing one client at a time, then selects the archive profile closest to
    the normalized ideal point by Tchebycheff distance.
    """
    if objective not in {"pareto", "latency"}:
        raise ValueError(f"Unsupported global profile objective: {objective}")
    if search_method not in {"bounded", "nsga2"}:
        raise ValueError(f"Unsupported global profile search method: {search_method}")
    if search_method == "nsga2" and objective != "pareto":
        raise ValueError("NSGA-II requires the Pareto objective")
    if not selected:
        empty = ProfileEvaluation({}, 0.0, 0.0, 0.0, ())
        return selected, empty

    perf: dict[str, Any] = {
        "search_iterations": 0,
        "evaluate_calls": 0,
        "evaluate_cache_hits": 0,
        "evaluate_cache_misses": 0,
        "evaluate_profile_sec": 0.0,
        "replacement_priority_calls": 0,
        "replacement_priority_sec": 0.0,
        "incremental_jlearn_calls": 0,
        "incremental_jlearn_sec": 0.0,
        "full_jlearn_fallback_calls": 0,
        "full_jlearn_fallback_sec": 0.0,
        "replacement_stats_build_sec": 0.0,
        "flow_objective_update_sec": 0.0,
        "neighbor_candidates_generated": 0,
        "neighbor_candidates_retained": 0,
        "neighbor_generation_sec": 0.0,
        "neighbor_evaluation_sec": 0.0,
        "pareto_archive_calls": 0,
        "pareto_archive_sec": 0.0,
        "dominance_compare_count": 0,
        "beam_sec": 0.0,
        "coverage_repair_calls": 0,
        "coverage_repair_total_sec": 0.0,
        "coverage_repair_iterations": 0,
        "coverage_repair_candidate_evaluations": 0,
        "coverage_repair_candidate_evaluation_sec": 0.0,
        "coverage_repair_final_evaluation_sec": 0.0,
    }
    search_started_at = time.perf_counter()
    previous_choices = previous_choices or {}
    client_edges = client_edges or {}
    pools: dict[int, list[Candidate]] = {}
    pools_before_stability: dict[int, list[Candidate]] = {}
    by_client: dict[int, tuple[Candidate, list[Candidate], float]] = {}
    for client_id, current, candidates, remaining in selected:
        by_client[client_id] = (current, candidates, remaining)
        pool = [
            item for item in candidates
            if item.feasible and item.epsilon_used <= remaining + 1e-12
        ]
        if not pool and not config.require_feasible:
            pool = [
                item for item in candidates
                if item.feasible_device and item.epsilon_used <= remaining + 1e-12
            ]
        if not pool:
            pool = [current if current.feasible_device else skipped_candidate()]
        pool = _dedupe_candidates(pool)
        pools_before_stability[client_id] = list(pool)
        if config.enforce_cloud_dp_stability:
            pool = _stable_cloud_candidate_pool(config, pool)
        pools[client_id] = pool

    if config.require_edge_cloud_coverage and not _edge_cloud_coverage_possible_from_pools(
        config, pools, client_samples, client_edges
    ):
        coverage_failures = _edge_cloud_coverage_failure_details(
            config, pools, client_samples, client_edges
        )
        fallback_profile = {client_id: skipped_candidate() for client_id in pools}
        fallback = _evaluate_profile(
            config,
            fallback_profile,
            client_samples,
            client_edges,
            previous_choices,
        )
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(
                {
                    "archive": (),
                    "chosen": fallback,
                    "coverage_infeasible_fallback": True,
                    "coverage_target_ratio": min(
                        max(float(config.min_edge_cloud_fusion_ratio), 0.0), 1.0
                    ),
                    "coverage_failure_edges": coverage_failures,
                    "candidate_pool_sizes": {
                        client_id: len(pool) for client_id, pool in pools.items()
                    },
                    "candidate_pool_sizes_before_stability": {
                        client_id: len(pool)
                        for client_id, pool in pools_before_stability.items()
                    },
                    "update_mechanism_counts_before_stability": (
                        _candidate_update_mechanism_population(pools_before_stability)
                    ),
                    "update_mechanism_counts_after_stability": (
                        _candidate_update_mechanism_population(pools)
                    ),
                    "search_method": search_method,
                    "mode_pool_before_stability": _mode_search_pool_audit(pools_before_stability),
                    "mode_pool_after_stability": _mode_search_pool_audit(pools),
                    "mode_archive_presence": _mode_archive_audit(()),
                    "mode_chosen": _mode_profile_distribution(fallback_profile),
                }
            )
        rewritten = [
            (client_id, fallback_profile[client_id], by_client[client_id][1], by_client[client_id][2])
            for client_id, _current, _candidates, _remaining in selected
        ]
        return rewritten, fallback

    seeds = (
        _initial_profiles(
            config,
            pools,
            previous_choices,
            client_samples=client_samples,
            client_edges=client_edges,
        )
        if objective == "pareto"
        else _initial_latency_profiles(pools, previous_choices)
    )
    search_client_ids = (
        _pareto_search_client_ids(config, pools, seeds)
        if objective == "pareto"
        else tuple(sorted(pools))
    )
    client_order = tuple(sorted(pools))
    client_positions = {
        client_id: position
        for position, client_id in enumerate(client_order)
    }
    flow_inputs_by_candidate = _profile_flow_inputs_by_candidate(
        config,
        pools,
        client_samples,
        client_edges,
        previous_choices,
        client_order,
    )
    candidate_tokens = {
        client_id: {
            _candidate_key(candidate): token
            for token, candidate in enumerate(pools[client_id])
        }
        for client_id in client_order
    }

    # Stage 12F1:
    # Materialize candidate key/token metadata once.  This preserves the
    # existing candidate_tokens[key] semantics while removing repeated
    # _candidate_key() construction from the million-call neighbor loop.
    candidate_neighbor_rows = {
        client_id: tuple(
            (
                candidate,
                candidate_key,
                candidate_tokens[client_id][candidate_key],
            )
            for candidate in pools[client_id]
            for candidate_key in (_candidate_key(candidate),)
        )
        for client_id in client_order
    }

    def signature(profile: dict[int, Candidate]) -> tuple[int, ...]:
        return tuple(
            candidate_tokens[client_id][_candidate_key(profile[client_id])]
            for client_id in client_order
        )

    evaluation_cache: dict[tuple[int, ...], ProfileEvaluation] = {}

    def evaluate(
        profile: dict[int, Candidate],
        profile_signature: tuple[int, ...] = (),
        flow_objectives: tuple[tuple[int, ...], float] | None = None,
    ) -> ProfileEvaluation:
        profile_key = profile_signature or signature(profile)
        _perf_add(perf, "evaluate_calls")
        cached = evaluation_cache.get(profile_key)
        if cached is not None:
            _perf_add(perf, "evaluate_cache_hits")
            return cached
        _perf_add(perf, "evaluate_cache_misses")
        started_at = time.perf_counter()
        evaluated = _evaluate_profile(
            config,
            profile,
            client_samples,
            client_edges,
            previous_choices,
            profile_signature=profile_key,
            flow_inputs_by_candidate=flow_inputs_by_candidate,
            flow_objectives=flow_objectives,
            previous_client_updates=previous_client_updates,
            fusion_objective_enabled=fusion_objective_enabled,
        )
        _perf_add(perf, "evaluate_profile_sec", time.perf_counter() - started_at)
        evaluation_cache[profile_key] = evaluated
        return evaluated

    def archive_selector(items: list[ProfileEvaluation], limit: int) -> list[ProfileEvaluation]:
        if objective == "pareto":
            return _pareto_archive(items, limit, profiler=perf)
        started_at = time.perf_counter()
        result = _latency_archive(items, limit)
        _perf_add(perf, "pareto_archive_calls")
        _perf_add(perf, "pareto_archive_sec", time.perf_counter() - started_at)
        return result
    seed_evaluations = [
            evaluate(
                item,
                profile_signature=signature(item),
            )
            for item in seeds
        ]
    if objective == "pareto":
        # A global-learning profile must contribute nonzero represented
        # sample mass to the Cloud. Individual Edge-terminating modes remain
        # valid inside a mixed profile as long as the overall r_C is positive.
        seed_evaluations = [
            item for item in seed_evaluations
            if item.cloud_fusion_ratio > 1e-12
        ]
        if not seed_evaluations:
            raise ValueError(
                "pareto selection requires at least one feasible profile "
                "with cloud_fusion_ratio > 0"
            )
    if config.require_edge_cloud_coverage:
        feasible_seeds: list[ProfileEvaluation] = []
        for evaluation in seed_evaluations:
            if _profile_satisfies_edge_cloud_coverage(
                config,
                evaluation.profile,
                client_samples,
                client_edges,
            ):
                feasible_seeds.append(evaluation)
                continue
            repair_objectives = (
                ("latency",)
                if objective == "latency"
                else ("latency", "learning", "pareto")
            )
            feasible_seeds.extend(
                _repair_edge_cloud_coverage(
                    config=config,
                    chosen=evaluation,
                    pools=pools,
                    client_samples=client_samples,
                    client_edges=client_edges,
                    previous_choices=previous_choices,
                    objective=repair_objective,
                    flow_inputs_by_candidate=flow_inputs_by_candidate,
                    profiler=perf,
                )
                for repair_objective in repair_objectives
            )
        seed_evaluations = _unique_evaluations(feasible_seeds)
    archive = archive_selector(
        seed_evaluations,
        config.pareto_archive_size,
    )
    if search_method == "nsga2":
        archive = _run_nsga2_search(
            config=config,
            pools=pools,
            initial=seed_evaluations,
            evaluate=evaluate,
            client_samples=client_samples,
            client_edges=client_edges,
        )
    beam: list[ProfileEvaluation] = []
    visited_profiles = set(evaluation_cache)
    visited_profiles.update(_evaluation_key(item) for item in seed_evaluations)

    bounded_iterations = config.pareto_max_iters if search_method == "bounded" else 0
    for _iter_idx in range(max(0, bounded_iterations)):
        _perf_add(perf, "search_iterations")
        neighbor_started_at = time.perf_counter()
        neighbors: list[
            tuple[
                dict[int, Candidate],
                tuple[int, ...],
                tuple[tuple[int, ...], float] | None,
            ]
        ] = []
        expansion_bases = _unique_evaluations(archive + beam)
        omega_endpoint_key = (
            _evaluation_key(
                min(archive, key=lambda item: (item.system_omega, item.system_latency))
            )
            if objective == "pareto"
            else None
        )
        for evaluated in expansion_bases:
            base_flow_stats = _full_buffer_flow_stats(
                config,
                evaluated.profile,
                flow_inputs_by_candidate,
            )
            stats_started_at = time.perf_counter()
            replacement_stats = (
                _LearningReplacementStats(
                    config, evaluated.profile, client_samples, client_edges,
                    evaluated.admitted_client_ids,
                ) if objective == "pareto" and config.pareto_neighbor_top_k > 0
                else None
            )
            _perf_add(perf, "replacement_stats_build_sec", time.perf_counter() - stats_started_at)
            profile_neighbors: list[
                tuple[
                    float,
                    tuple[int, ...],
                    tuple[tuple[int, ...], float] | None,
                    int,
                    Candidate,
                    _LearningReplacementStats | None,
                    float | None,
                ]
            ] = []
            for client_id in search_client_ids:
                current = evaluated.profile[client_id]

                old_cloud_entry = (
                    replacement_stats._cloud_entry(client_id, current)
                    if (
                        replacement_stats is not None
                        and objective != "latency"
                        and config.pareto_neighbor_top_k > 0
                    )
                    else None
                )

                for candidate, candidate_key, token in candidate_neighbor_rows[client_id]:
                    if candidate == current:
                        continue
                    key_started_at = time.perf_counter()

                    position = client_positions[client_id]
                    key = (
                        evaluated.profile_signature[:position]
                        + (token,)
                        + evaluated.profile_signature[position + 1:]
                    )

                    already_visited = key in visited_profiles
                    if not already_visited:
                        visited_profiles.add(key)

                    _perf_add(
                        perf,
                        "neighbor_key_visited_sec",
                        time.perf_counter() - key_started_at,
                    )

                    if already_visited:
                        continue
                    flow_objectives = None
                    if base_flow_stats is not None:
                        flow_started_at = time.perf_counter()
                        flow_objectives = _replace_full_buffer_flow_objectives(
                            config,
                            base_flow_stats,
                            client_id,
                            flow_inputs_by_candidate[client_id].get(
                                candidate_key
                            ),
                        )
                        _perf_add(perf, "flow_objective_update_sec", time.perf_counter() - flow_started_at)
                    if config.pareto_neighbor_top_k > 0:
                        (
                            priority,
                            priority_next_learning,
                        ) = _global_replacement_priority(
                            objective=objective,
                            config=config,
                            evaluated=evaluated,
                            client_id=client_id,
                            current=current,
                            candidate=candidate,
                            client_samples=client_samples,
                            client_edges=client_edges,
                            trial_profile=None,
                            flow_objectives=flow_objectives,
                            replacement_stats=replacement_stats,
                            old_cloud_entry=old_cloud_entry,
                            profiler=perf,
                        )
                    else:
                        priority = 0.0
                        priority_next_learning = None

                    append_started_at = time.perf_counter()

                    profile_neighbors.append(
                        (
                            priority,
                            key,
                            flow_objectives,
                            client_id,
                            candidate,
                            replacement_stats,
                            priority_next_learning,
                        )
                    )

                    _perf_add(
                        perf,
                        "neighbor_append_sec",
                        time.perf_counter() - append_started_at,
                    )
            _perf_add(perf, "neighbor_candidates_generated", len(profile_neighbors))
            if (
                config.pareto_neighbor_top_k > 0
                and (
                    objective == "latency"
                    or _evaluation_key(evaluated) != omega_endpoint_key
                )
            ):
                sort_started_at = time.perf_counter()

                profile_neighbors.sort(key=lambda item: item[0])
                profile_neighbors = profile_neighbors[
                    : config.pareto_neighbor_top_k
                ]

                _perf_add(
                    perf,
                    "neighbor_sort_topk_sec",
                    time.perf_counter() - sort_started_at,
                )
            _perf_add(perf, "neighbor_candidates_retained", len(profile_neighbors))
            materialize_started_at = time.perf_counter()

            for (
                _priority,
                key,
                flow_objectives,
                client_id,
                candidate,
                replacement_stats,
                priority_next_learning,
            ) in profile_neighbors:
                profile = dict(evaluated.profile)
                profile[client_id] = candidate

                neighbors.append(
                    (
                        profile,
                        key,
                        flow_objectives,
                        client_id,
                        candidate,
                        replacement_stats,
                        priority_next_learning,
                    )
                )

            _perf_add(
                perf,
                "neighbor_profile_materializations",
                len(profile_neighbors),
            )
            _perf_add(
                perf,
                "neighbor_profile_materialization_sec",
                time.perf_counter() - materialize_started_at,
            )
        _perf_add(perf, "neighbor_generation_sec", time.perf_counter() - neighbor_started_at)
        if not neighbors:
            break

        neighbor_eval_started_at = time.perf_counter()
        evaluated_neighbors: list[ProfileEvaluation] = []

        for (
            profile,
            profile_key,
            flow_objectives,
            changed_client_id,
            changed_candidate,
            replacement_stats,
            priority_next_learning,
        ) in neighbors:
            incremental_evaluation = None

            if (
                objective == "pareto"
                and flow_objectives is not None
                and replacement_stats is not None
            ):
                admitted, next_latency = flow_objectives

                next_learning = priority_next_learning

                if next_learning is not None:
                    _perf_add(
                        perf,
                        "neighbor_priority_learning_reuse_calls",
                    )
                else:
                    recompute_started_at = time.perf_counter()
                    next_learning = replacement_stats.replacement_cost(
                        changed_client_id,
                        changed_candidate,
                        admitted,
                    )
                    _perf_add(
                        perf,
                        "neighbor_retained_jlearn_recompute_calls",
                    )
                    _perf_add(
                        perf,
                        "neighbor_retained_jlearn_recompute_sec",
                        time.perf_counter() - recompute_started_at,
                    )

                if next_learning is not None:
                    cloud_fusion_ratio = _cloud_update_coverage_ratio(
                        profile,
                        client_samples,
                        admitted,
                    )

                    incremental_evaluation = ProfileEvaluation(
                        profile=profile,
                        system_latency=float(next_latency),
                        system_omega=float(next_learning),
                        cloud_fusion_ratio=float(cloud_fusion_ratio),
                        admitted_client_ids=tuple(sorted(admitted)),
                        profile_signature=profile_key,
                    )

                    _perf_add(
                        perf,
                        "neighbor_incremental_evaluation_calls",
                    )

            if incremental_evaluation is None:
                incremental_evaluation = evaluate(
                    profile,
                    profile_signature=profile_key,
                    flow_objectives=flow_objectives,
                )

                _perf_add(
                    perf,
                    "neighbor_full_evaluation_fallback_calls",
                )

            evaluated_neighbors.append(incremental_evaluation)

        _perf_add(
            perf,
            "neighbor_evaluation_sec",
            time.perf_counter() - neighbor_eval_started_at,
        )
        if objective == "pareto":
            evaluated_neighbors = [
                item for item in evaluated_neighbors
                if item.cloud_fusion_ratio > 1e-12
            ]
        expanded = archive + evaluated_neighbors
        if config.require_edge_cloud_coverage:
            expanded = [
                item
                for item in expanded
                if _profile_satisfies_edge_cloud_coverage(
                    config,
                    item.profile,
                    client_samples,
                    client_edges,
                )
            ]
        next_archive = archive_selector(expanded, config.pareto_archive_size)
        next_archive_keys = {_evaluation_key(item) for item in next_archive}
        beam_started_at = time.perf_counter()
        beam = _bounded_search_beam(
            [
                item
                for item in evaluated_neighbors
                if _evaluation_key(item) not in next_archive_keys
                and (
                    not config.require_edge_cloud_coverage
                    or _profile_satisfies_edge_cloud_coverage(
                        config,
                        item.profile,
                        client_samples,
                        client_edges,
                    )
                )
            ],
            reference=next_archive,
            limit=config.pareto_beam_size,
            norm_eps=config.pareto_norm_eps,
        )
        _perf_add(perf, "beam_sec", time.perf_counter() - beam_started_at)
        if next_archive_keys == {_evaluation_key(item) for item in archive} and not beam:
            archive = next_archive
            break
        archive = next_archive

    chosen = (
        _choose_tchebycheff(archive, config.pareto_norm_eps)
        if objective == "pareto"
        else min(
            archive,
            key=lambda item: (item.system_latency, _evaluation_key(item)),
        )
    )
    chosen = evaluate(
        chosen.profile,
        profile_signature=chosen.profile_signature,
    )
    if config.require_edge_cloud_coverage and not _profile_satisfies_edge_cloud_coverage(
        config,
        chosen.profile,
        client_samples,
        client_edges,
    ):
        chosen = _repair_edge_cloud_coverage(
            config=config,
            chosen=chosen,
            pools=pools,
            client_samples=client_samples,
            client_edges=client_edges,
            previous_choices=previous_choices,
            objective=objective,
            flow_inputs_by_candidate=flow_inputs_by_candidate,
            profiler=perf,
        )
    perf["search_total_sec"] = time.perf_counter() - search_started_at
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(
            {
                "archive": tuple(archive),
                "chosen": chosen,
                "evaluated_profile_count": len(evaluation_cache),
                "search_client_ids": search_client_ids,
                "candidate_pool_sizes": {
                    client_id: len(pool) for client_id, pool in pools.items()
                },
                "candidate_pool_sizes_before_stability": {
                    client_id: len(pool)
                    for client_id, pool in pools_before_stability.items()
                },
                "update_mechanism_counts_before_stability": (
                    _candidate_update_mechanism_population(pools_before_stability)
                ),
                "update_mechanism_counts_after_stability": (
                    _candidate_update_mechanism_population(pools)
                ),
                "cloud_reaching_candidate_count": sum(
                    _candidate_reaches_cloud(candidate)
                    for candidates in pools.values()
                    for candidate in candidates
                ),
                "clients_with_cloud_candidate": sum(
                    any(_candidate_reaches_cloud(candidate) for candidate in candidates)
                    for candidates in pools.values()
                ),
                "seed_evaluations": tuple(seed_evaluations),
                "search_method": search_method,
                "mode_pool_before_stability": _mode_search_pool_audit(pools_before_stability),
                "mode_pool_after_stability": _mode_search_pool_audit(pools),
                "mode_archive_presence": _mode_archive_audit(archive),
                "mode_chosen": _mode_profile_distribution(chosen.profile),
                "performance_profile": dict(perf),
            }
        )
    rewritten = [
        (client_id, chosen.profile[client_id], by_client[client_id][1], by_client[client_id][2])
        for client_id, _current, _candidates, _remaining in selected
    ]
    return rewritten, chosen



def _candidate_cloud_dp_eligible(candidate: Candidate) -> bool:
    """Stage-2 scope: hierarchical LIIEIIIC E->C already carrying Cloud DP.

    LIIC client-packet DP requires moving noise generation into the client
    worker before any raw upload, so is deliberately left on its existing
    aggregate-only route until the separate direct-Cloud protocol is ready.
    """
    return (
        candidate.mode == "LIIEIIIC"
        and _candidate_uses_secure_aggregate_update_dp_for_selection(candidate)
    )


def _with_cloud_plan(candidate: Candidate, plan: str) -> Candidate:
    if plan not in {"packet", "aggregate"}:
        raise ValueError("cloud plan must be packet or aggregate")
    if _candidate_cloud_dp_eligible(candidate):
        return replace(candidate, dp_execution_plan="cloud_" + plan)
    return candidate


def choose_cloud_dp_pareto_profile(
    *,
    config: SelectionConfig,
    selected: list[tuple[int, Candidate, list[Candidate], float]],
    client_samples: dict[int, float],
    client_edges: dict[int, int] | None = None,
    previous_choices: dict[int, Candidate] | None = None,
    objective: str = "pareto",
    search_method: str = "bounded",
    diagnostics: dict[str, Any] | None = None,
    previous_client_updates: dict[int, dict[str, dict[str, torch.Tensor]]] | None = None,
    fusion_objective_enabled: bool = False,
) -> tuple[list[tuple[int, Candidate, list[Candidate], float]], ProfileEvaluation]:
    """Search whole-mode profiles under both *uniform* Cloud release plans.

    This intentionally does NOT permit heterogeneous plans inside an E->C
    packet. It reoptimizes the entire mode profile in each feasible branch and
    selects from their combined non-dominated archive using the existing
    Tchebycheff decision rule. No per-client noise plan is changed afterward.
    """
    plan = config.cloud_dp_plan
    if plan not in {"legacy", "packet", "aggregate", "pareto"}:
        raise ValueError("cloud_dp_plan must be legacy, packet, aggregate, or pareto")
    if plan == "legacy":
        return choose_global_pareto_profile(
            config=config, selected=selected, client_samples=client_samples,
            client_edges=client_edges, previous_choices=previous_choices,
            objective=objective, search_method=search_method,
            diagnostics=diagnostics, previous_client_updates=previous_client_updates,
            fusion_objective_enabled=fusion_objective_enabled,
        )
    branches = ("packet", "aggregate") if plan == "pareto" else (plan,)
    outcomes = []
    branch_performance: dict[str, dict[str, Any]] = {}
    for branch in branches:
        branch_started_at = time.perf_counter()
        mapped = [
            (cid, _with_cloud_plan(current, branch),
             [_with_cloud_plan(c, branch) for c in pool], remaining)
            for cid, current, pool, remaining in selected
        ]
        branch_diagnostics: dict[str, Any] = {}
        result, chosen = choose_global_pareto_profile(
            config=config, selected=mapped, client_samples=client_samples,
            client_edges=client_edges, previous_choices=previous_choices,
            objective=objective, search_method=search_method,
            diagnostics=branch_diagnostics, previous_client_updates=previous_client_updates,
            fusion_objective_enabled=fusion_objective_enabled,
        )
        branch_elapsed = time.perf_counter() - branch_started_at
        branch_profile = dict(branch_diagnostics.get("performance_profile", {}))
        branch_profile["branch_wall_sec"] = branch_elapsed
        branch_performance[branch] = branch_profile
        outcomes.append((branch, result, chosen, branch_diagnostics))
    if len(outcomes) == 1:
        branch, result, chosen, branch_diagnostics = outcomes[0]
        if diagnostics is not None:
            diagnostics.clear()
            diagnostics.update(branch_diagnostics)
            diagnostics["selected_cloud_dp_plan"] = branch
            diagnostics["cloud_dp_branch_performance"] = branch_performance
        return result, chosen
    combined_evaluations = [
        replace(evaluation, profile_signature=(branch_index, *evaluation.profile_signature))
        for branch_index, (_branch, _result, _chosen, d) in enumerate(outcomes)
        for evaluation in d.get("archive", ())
    ]
    archive = (_pareto_archive(combined_evaluations, config.pareto_archive_size)
               if objective == "pareto" else
               _latency_archive(combined_evaluations, config.pareto_archive_size))
    if not archive:
        raise RuntimeError("Cloud plan comparison produced no feasible profile")
    winner = (_choose_tchebycheff(archive, config.pareto_norm_eps)
              if objective == "pareto" else min(archive, key=lambda x: x.system_latency))
    branch = outcomes[int(winner.profile_signature[0])][0]
    chosen_result = next(result for name, result, _chosen, _d in outcomes if name == branch)
    chosen_diagnostics = next(d for name, _result, _chosen, d in outcomes if name == branch)
    # The combined winner may be an archived profile rather than the branch's
    # own winner. Rebuild selected tuples from the winner's actual candidates.
    chosen_result = [
        (cid, winner.profile[cid], pool, remaining)
        for cid, _current, pool, remaining in chosen_result
    ]
    if diagnostics is not None:
        diagnostics.clear()
        diagnostics.update(chosen_diagnostics)
        diagnostics["archive"] = tuple(archive)
        diagnostics["chosen"] = winner
        diagnostics["selected_cloud_dp_plan"] = branch
        diagnostics["cloud_dp_branch_performance"] = branch_performance
        diagnostics["cloud_dp_mode_branches"] = {
            name: {
                "pool": d.get("mode_pool_after_stability", {}),
                "archive_presence": d.get("mode_archive_presence", {}),
                "branch_winner": d.get("mode_chosen", {}),
            }
            for name, _result, _evaluation, d in outcomes
        }
        # Combined winner may differ from the branch-local winner.
        diagnostics["mode_chosen"] = _mode_profile_distribution(winner.profile)
        diagnostics["mode_archive_presence"] = _mode_archive_audit(archive)
        diagnostics["cloud_dp_branch_results"] = {
            name: {"time": evaluation.system_latency,
                   "J_learn": evaluation.system_omega,
                   "J_DP": evaluation.system_dp}
            for name, _result, evaluation, _d in outcomes
        }
    return chosen_result, winner

def _run_nsga2_search(
    *,
    config: SelectionConfig,
    pools: dict[int, list[Candidate]],
    initial: list[ProfileEvaluation],
    evaluate: Callable[..., ProfileEvaluation],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
) -> list[ProfileEvaluation]:
    """Run a deterministic NSGA-II reference search on the same profile space."""
    population_size = max(4, int(config.pareto_archive_size))
    client_ids = tuple(sorted(pools))
    rng = random.Random(int(config.seed) * 1_000_003 + 91_733)
    population = _nsga2_environmental_selection(
        [item for item in initial if item.cloud_fusion_ratio > 1e-12],
        population_size,
    )

    attempts = 0
    while len(population) < population_size and attempts < population_size * 40:
        attempts += 1
        profile = {
            client_id: rng.choice(pools[client_id])
            for client_id in client_ids
        }
        if config.require_edge_cloud_coverage and not _profile_satisfies_edge_cloud_coverage(
            config, profile, client_samples, client_edges
        ):
            continue
        evaluated = evaluate(profile)
        if evaluated.cloud_fusion_ratio <= 1e-12:
            continue
        population = _nsga2_environmental_selection(
            population + [evaluated],
            population_size,
        )

    for _generation in range(max(1, int(config.pareto_max_iters))):
        if not population:
            break
        rank, crowding = _nsga2_rank_and_crowding(population)
        offspring: list[ProfileEvaluation] = []
        attempts = 0
        while len(offspring) < population_size and attempts < population_size * 40:
            attempts += 1
            first = _nsga2_tournament(population, rank, crowding, rng)
            second = _nsga2_tournament(population, rank, crowding, rng)
            profile = {
                client_id: (
                    first.profile[client_id]
                    if rng.random() < 0.5
                    else second.profile[client_id]
                )
                for client_id in client_ids
            }
            mutation_probability = 1.0 / max(1, len(client_ids))
            mutated = False
            for client_id in client_ids:
                if rng.random() < mutation_probability:
                    profile[client_id] = rng.choice(pools[client_id])
                    mutated = True
            if not mutated and client_ids:
                client_id = rng.choice(client_ids)
                profile[client_id] = rng.choice(pools[client_id])
            if config.require_edge_cloud_coverage and not _profile_satisfies_edge_cloud_coverage(
                config, profile, client_samples, client_edges
            ):
                continue
            evaluated = evaluate(profile)
            if evaluated.cloud_fusion_ratio <= 1e-12:
                continue
            offspring.append(evaluated)
        next_population = _nsga2_environmental_selection(
            population + offspring,
            population_size,
        )
        if {_evaluation_key(item) for item in next_population} == {
            _evaluation_key(item) for item in population
        }:
            population = next_population
            break
        population = next_population
    return _pareto_archive(population, config.pareto_archive_size)


def _nsga2_tournament(
    population: list[ProfileEvaluation],
    rank: dict[tuple, int],
    crowding: dict[tuple, float],
    rng: random.Random,
) -> ProfileEvaluation:
    first = rng.choice(population)
    second = rng.choice(population)

    def key(item: ProfileEvaluation) -> tuple[float, float, str]:
        item_key = _evaluation_key(item)
        return (rank[item_key], -crowding[item_key], repr(item_key))

    return min((first, second), key=key)


def _nsga2_environmental_selection(
    evaluations: list[ProfileEvaluation],
    limit: int,
) -> list[ProfileEvaluation]:
    unique = _unique_evaluations(evaluations)
    selected: list[ProfileEvaluation] = []
    for front in _nsga2_fronts(unique):
        if len(selected) + len(front) <= limit:
            selected.extend(front)
            continue
        crowding = _nsga2_crowding(front)
        selected.extend(
            sorted(
                front,
                key=lambda item: (
                    -crowding[_evaluation_key(item)],
                    repr(_evaluation_key(item)),
                ),
            )[: max(0, limit - len(selected))]
        )
        break
    return selected


def _nsga2_rank_and_crowding(
    evaluations: list[ProfileEvaluation],
) -> tuple[dict[tuple, int], dict[tuple, float]]:
    rank: dict[tuple, int] = {}
    crowding: dict[tuple, float] = {}
    for front_index, front in enumerate(_nsga2_fronts(evaluations)):
        rank.update({_evaluation_key(item): front_index for item in front})
        crowding.update(_nsga2_crowding(front))
    return rank, crowding


def _nsga2_fronts(
    evaluations: list[ProfileEvaluation],
) -> list[list[ProfileEvaluation]]:
    remaining = _unique_evaluations(evaluations)
    fronts: list[list[ProfileEvaluation]] = []
    while remaining:
        front = [
            candidate
            for candidate in remaining
            if not any(
                _profile_objectives_dominate(other, candidate)
                for other in remaining
                if other is not candidate
            )
        ]
        if not front:
            front = [min(remaining, key=lambda item: (item.system_latency, item.system_omega))]
        fronts.append(front)
        front_keys = {_evaluation_key(item) for item in front}
        remaining = [item for item in remaining if _evaluation_key(item) not in front_keys]
    return fronts


_PARETO_OBJECTIVE_DIGITS = 10


def _pareto_objective_value(value: float) -> float:
    """Canonical numeric representation for Pareto comparisons.

    Stage 10.1 measured incremental-vs-full implementation drift at roughly
    1e-13 for latency and 1e-16 for J_learn.  Quantizing to 10 decimal places
    prevents those machine-precision differences from changing the search
    trajectory while remaining far below meaningful objective differences.
    """
    return round(float(value), _PARETO_OBJECTIVE_DIGITS)


def _profile_objectives_dominate(
    left: ProfileEvaluation,
    right: ProfileEvaluation,
) -> bool:
    values_left = [
        _pareto_objective_value(left.system_latency),
        _pareto_objective_value(left.system_omega),
    ]
    values_right = [
        _pareto_objective_value(right.system_latency),
        _pareto_objective_value(right.system_omega),
    ]
    use_fusion = left.fusion_objective_enabled or right.fusion_objective_enabled
    if use_fusion:
        values_left.append(
            _pareto_objective_value(left.fusion_distortion)
        )
        values_right.append(
            _pareto_objective_value(right.fusion_distortion)
        )
    return (
        all(a <= b for a, b in zip(values_left, values_right))
        and any(a < b for a, b in zip(values_left, values_right))
    )


def _nsga2_crowding(front: list[ProfileEvaluation]) -> dict[tuple, float]:
    distances = {_evaluation_key(item): 0.0 for item in front}
    if len(front) <= 2:
        return {key: float("inf") for key in distances}
    for value in (
        lambda item: item.system_latency,
        lambda item: item.system_omega,
    ):
        ordered = sorted(
            front,
            key=lambda item: (value(item), repr(_evaluation_key(item))),
        )
        low = value(ordered[0])
        high = value(ordered[-1])
        distances[_evaluation_key(ordered[0])] = float("inf")
        distances[_evaluation_key(ordered[-1])] = float("inf")
        span = high - low
        if span <= 1e-12:
            continue
        for index in range(1, len(ordered) - 1):
            key = _evaluation_key(ordered[index])
            if math.isinf(distances[key]):
                continue
            distances[key] += (
                value(ordered[index + 1]) - value(ordered[index - 1])
            ) / span
    return distances


def _edge_cloud_coverage_failure_details(
    config: SelectionConfig,
    pools: dict[int, list[Candidate]],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
) -> list[dict[str, object]]:
    target_ratio = min(max(float(config.min_edge_cloud_fusion_ratio), 0.0), 1.0)
    edge_clients: dict[int, list[int]] = {}
    for client_id in pools:
        edge_clients.setdefault(int(client_edges[client_id]), []).append(client_id)
    failures: list[dict[str, object]] = []
    for edge_id, clients in sorted(edge_clients.items()):
        total = sum(float(client_samples[client_id]) for client_id in clients)
        cloud_capable = [
            client_id
            for client_id in clients
            if any(_candidate_reaches_cloud(candidate) for candidate in pools[client_id])
        ]
        maximum_covered = sum(float(client_samples[client_id]) for client_id in cloud_capable)
        ratio = maximum_covered / total if total > 0.0 else 0.0
        if ratio + 1e-12 < target_ratio:
            failures.append(
                {
                    "edge_id": int(edge_id),
                    "total_samples": float(total),
                    "maximum_covered_samples": float(maximum_covered),
                    "maximum_coverage_ratio": float(ratio),
                    "target_ratio": float(target_ratio),
                    "cloud_capable_client_ids": [int(client_id) for client_id in cloud_capable],
                }
            )
    return failures


def _edge_cloud_coverage_possible_from_pools(
    config: SelectionConfig,
    pools: dict[int, list[Candidate]],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
) -> bool:
    """Return whether each represented edge can meet the hard cloud-coverage constraint."""
    if not config.require_edge_cloud_coverage:
        return True
    target_ratio = min(max(float(config.min_edge_cloud_fusion_ratio), 0.0), 1.0)
    edge_clients: dict[int, list[int]] = {}
    for client_id in pools:
        edge_clients.setdefault(int(client_edges[client_id]), []).append(client_id)
    for clients in edge_clients.values():
        total = sum(float(client_samples[client_id]) for client_id in clients)
        cloud_capable = [
            client_id
            for client_id in clients
            if any(_candidate_reaches_cloud(candidate) for candidate in pools[client_id])
        ]
        if not cloud_capable:
            return False
        maximum_covered = sum(float(client_samples[client_id]) for client_id in cloud_capable)
        if maximum_covered + 1e-12 < target_ratio * total:
            return False
    return True


def _profile_satisfies_edge_cloud_coverage(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
) -> bool:
    if not config.require_edge_cloud_coverage:
        return True
    target_ratio = min(max(float(config.min_edge_cloud_fusion_ratio), 0.0), 1.0)
    edge_clients: dict[int, list[int]] = {}
    for client_id in profile:
        edge_clients.setdefault(int(client_edges[client_id]), []).append(client_id)
    for clients in edge_clients.values():
        total = sum(float(client_samples[client_id]) for client_id in clients)
        covered = sum(
            float(client_samples[client_id])
            for client_id in clients
            if _candidate_reaches_cloud(profile[client_id])
        )
        if covered + 1e-12 < target_ratio * total:
            return False
        if not any(_candidate_reaches_cloud(profile[client_id]) for client_id in clients):
            return False
    return True


def _candidate_update_mechanism_population(
    pools: dict[int, list[Candidate]],
) -> dict[str, int]:
    counts = {"dp_only": 0, "he_only": 0, "dp_he": 0, "neither": 0}
    for pool in pools.values():
        for candidate in pool:
            mechanisms = candidate_mechanisms_for_object(candidate, "upd")
            uses_dp = any(mechanism_uses_dp(item) for item in mechanisms)
            uses_he = any(mechanism_uses_he(item) for item in mechanisms)
            key = "dp_he" if uses_dp and uses_he else "dp_only" if uses_dp else "he_only" if uses_he else "neither"
            counts[key] += 1
    return counts


def _stable_cloud_candidate_pool(
    config: SelectionConfig,
    candidates: list[Candidate],
) -> list[Candidate]:
    """Remove unstable DP links when a feasible cloud-reaching HE alternative exists."""

    def has_feature_dp(candidate: Candidate) -> bool:
        return any(
            mechanism == "dp"
            for obj in {"emb", "grad", "weakemb", "strongemb", "pseudo_label"}
            for mechanism in candidate_mechanisms_for_object(candidate, obj)
        )

    def has_update_dp(candidate: Candidate) -> bool:
        return any(
            mechanism_uses_dp(mechanism)
            for mechanism in candidate_mechanisms_for_object(candidate, "upd")
        )

    def has_update_he(candidate: Candidate) -> bool:
        return any(
            mechanism_uses_he(mechanism)
            for mechanism in candidate_mechanisms_for_object(candidate, "upd")
        )

    feasible_cloud = [
        candidate
        for candidate in candidates
        if candidate.feasible and _candidate_reaches_cloud(candidate)
    ]
    feature_he_alternative = any(
        candidate_has_he(candidate) and not has_feature_dp(candidate)
        for candidate in feasible_cloud
    )
    update_he_alternative = any(
        has_update_he(candidate) and not has_update_dp(candidate)
        for candidate in feasible_cloud
    )
    if not feature_he_alternative and not update_he_alternative:
        return candidates

    privacy = resolved_privacy_parameters(config)
    threshold = max(float(config.cloud_dp_stability_threshold), 0.0)
    feature_ratio = 2.0 * float(privacy["feature_noise_multiplier"])
    update_ratio = 2.0 * float(privacy["update_noise_multiplier"])

    def unstable(candidate: Candidate) -> bool:
        if not _candidate_reaches_cloud(candidate):
            return False
        return (
            (
                feature_he_alternative
                and has_feature_dp(candidate)
                and feature_ratio > threshold
            )
            or (
                update_he_alternative
                and has_update_dp(candidate)
                and update_ratio > threshold
            )
        )

    return [candidate for candidate in candidates if not unstable(candidate)]


def _repair_edge_cloud_coverage(
    *,
    config: SelectionConfig,
    chosen: ProfileEvaluation,
    pools: dict[int, list[Candidate]],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    previous_choices: dict[int, Candidate],
    objective: str = "pareto",
    flow_inputs_by_candidate: dict[int, dict[tuple, ClientFlowInput]] | None = None,
    profiler: dict[str, Any] | None = None,
) -> ProfileEvaluation:
    """Repair edge-to-Cloud coverage with exact incremental candidate scoring.

    For full-buffer round-synchronous flows, a one-client replacement has exact
    O(affected-group) latency and J_learn updates. Only cases where those exact
    incremental formulas do not apply fall back to a full profile evaluation.
    """
    repair_started_at = time.perf_counter()
    _perf_add(profiler, "coverage_repair_calls")

    profile = dict(chosen.profile)

    # Reuse the candidate-level flow inputs already prepared by the global
    # selector. This avoids reconstructing the complete flow model for every
    # repair candidate.
    if flow_inputs_by_candidate is None:
        flow_inputs_by_candidate = _profile_flow_inputs_by_candidate(
            config,
            pools,
            client_samples,
            client_edges,
            previous_choices,
            tuple(sorted(profile)),
        )

    edge_ids = sorted({
        client_edges[client_id]
        for client_id in profile
    })

    for edge_id in edge_ids:
        edge_clients = [
            client_id
            for client_id in profile
            if client_edges[client_id] == edge_id
        ]

        edge_samples = sum(
            client_samples[client_id]
            for client_id in edge_clients
        )

        target_ratio = min(
            max(float(config.min_edge_cloud_fusion_ratio), 0.0),
            1.0,
        )

        target_samples = target_ratio * edge_samples

        def covered_samples() -> float:
            return sum(
                client_samples[client_id]
                for client_id in edge_clients
                if _candidate_reaches_cloud(profile[client_id])
            )

        while (
            covered_samples() + 1e-12 < target_samples
            or not any(
                _candidate_reaches_cloud(profile[client_id])
                for client_id in edge_clients
            )
        ):
            _perf_add(
                profiler,
                "coverage_repair_iterations",
            )

            previous_covered = covered_samples()

            # Exact incremental flow state for the current repair profile.
            # For aggregation_fraction=1.0 this is the full-buffer path used
            # by the current paper configuration.
            base_flow_stats = _full_buffer_flow_stats(
                config,
                profile,
                flow_inputs_by_candidate,
            )

            # Exact fixed-admission J_learn replacement statistics.
            replacement_stats = None

            if (
                base_flow_stats is not None
                and objective != "latency"
            ):
                replacement_stats = _LearningReplacementStats(
                    config,
                    profile,
                    client_samples,
                    client_edges,
                    base_flow_stats.admitted_client_ids,
                )

            repairs: list[ProfileEvaluation] = []

            for client_id in edge_clients:
                current = profile[client_id]

                if _candidate_reaches_cloud(current):
                    continue

                for candidate in pools[client_id]:
                    if not _candidate_reaches_cloud(candidate):
                        continue

                    repaired = dict(profile)
                    repaired[client_id] = candidate

                    candidate_started_at = time.perf_counter()

                    evaluated: ProfileEvaluation | None = None

                    # ----------------------------------------------------
                    # Exact incremental path
                    # ----------------------------------------------------
                    if base_flow_stats is not None:
                        new_input = flow_inputs_by_candidate[
                            client_id
                        ].get(
                            _candidate_key(candidate)
                        )

                        flow_objectives = (
                            _replace_full_buffer_flow_objectives(
                                config,
                                base_flow_stats,
                                client_id,
                                new_input,
                            )
                        )

                        admitted, next_latency = flow_objectives

                        if objective == "latency":
                            # Latency is already exact; J_learn is irrelevant
                            # for the latency-only repair objective.
                            evaluated = ProfileEvaluation(
                                profile=repaired,
                                system_latency=float(next_latency),
                                system_omega=0.0,
                                cloud_fusion_ratio=0.0,
                                admitted_client_ids=tuple(
                                    sorted(admitted)
                                ),
                            )

                        elif replacement_stats is not None:
                            next_learning = (
                                replacement_stats.replacement_cost(
                                    client_id,
                                    candidate,
                                    admitted,
                                )
                            )

                            if next_learning is not None:
                                evaluated = ProfileEvaluation(
                                    profile=repaired,
                                    system_latency=float(next_latency),
                                    system_omega=float(next_learning),
                                    cloud_fusion_ratio=0.0,
                                    admitted_client_ids=tuple(
                                        sorted(admitted)
                                    ),
                                )

                    # ----------------------------------------------------
                    # Exact full-evaluation fallback
                    # ----------------------------------------------------
                    if evaluated is None:
                        fallback_started_at = time.perf_counter()

                        evaluated = _evaluate_profile(
                            config,
                            repaired,
                            client_samples,
                            client_edges,
                            previous_choices,
                            flow_inputs_by_candidate=(
                                flow_inputs_by_candidate
                            ),
                        )

                        fallback_elapsed = (
                            time.perf_counter()
                            - fallback_started_at
                        )

                        _perf_add(
                            profiler,
                            "coverage_repair_full_fallback_evaluations",
                        )

                        _perf_add(
                            profiler,
                            "coverage_repair_full_fallback_evaluation_sec",
                            fallback_elapsed,
                        )

                        # Keep the Stage-8 field meaningful: it now measures
                        # actual full candidate evaluations only.
                        _perf_add(
                            profiler,
                            "coverage_repair_candidate_evaluation_sec",
                            fallback_elapsed,
                        )

                    else:
                        _perf_add(
                            profiler,
                            "coverage_repair_incremental_candidates",
                        )

                    _perf_add(
                        profiler,
                        "coverage_repair_candidate_evaluations",
                    )

                    _perf_add(
                        profiler,
                        "coverage_repair_candidate_scoring_sec",
                        (
                            time.perf_counter()
                            - candidate_started_at
                        ),
                    )

                    repairs.append(evaluated)

            if not repairs:
                raise RuntimeError(
                    "No feasible cloud-reaching candidate exists "
                    f"for edge {edge_id}"
                )

            # Same decision semantics as the original implementation.
            if objective == "latency":
                selected = min(
                    repairs,
                    key=lambda item: (
                        item.system_latency,
                        _evaluation_key(item),
                    ),
                )

            elif objective == "learning":
                selected = min(
                    repairs,
                    key=lambda item: (
                        item.system_omega,
                        item.system_latency,
                        _evaluation_key(item),
                    ),
                )

            else:
                selected = _choose_tchebycheff(
                    _pareto_archive(
                        repairs,
                        config.pareto_archive_size,
                    ),
                    config.pareto_norm_eps,
                )

            profile = selected.profile

            if (
                covered_samples()
                <= previous_covered + 1e-12
            ):
                raise RuntimeError(
                    "Cloud-coverage repair made no progress "
                    f"for edge {edge_id}"
                )

    # Only the final repaired profile needs the complete diagnostics.
    final_eval_started_at = time.perf_counter()

    result = _evaluate_profile(
        config,
        profile,
        client_samples,
        client_edges,
        previous_choices,
        flow_inputs_by_candidate=flow_inputs_by_candidate,
    )

    _perf_add(
        profiler,
        "coverage_repair_final_evaluation_sec",
        time.perf_counter() - final_eval_started_at,
    )

    _perf_add(
        profiler,
        "coverage_repair_total_sec",
        time.perf_counter() - repair_started_at,
    )

    return result

def evaluate_global_profile(
    *,
    config: SelectionConfig,
    selected: list[tuple[int, Candidate, list[Candidate], float]],
    client_samples: dict[int, float],
    client_edges: dict[int, int] | None = None,
    previous_choices: dict[int, Candidate] | None = None,
) -> ProfileEvaluation:
    """Evaluate a selected profile without running another Pareto search."""
    profile = {
        client_id: candidate
        for client_id, candidate, _candidates, _remaining in selected
    }
    return _evaluate_profile(
        config,
        profile,
        client_samples,
        client_edges or {},
        previous_choices or {},
    )


def _cloud_coverage_anchor_profiles(
    pools: dict[int, list[Candidate]],
    fastest: dict[int, Candidate],
    client_samples: dict[int, float] | None = None,
    targets: tuple[float, ...] = (0.25, 0.50, 0.75, 1.00),
) -> list[dict[int, Candidate]]:
    """Build deterministic search seeds spanning low-to-high Cloud coverage.

    These are search anchors only; they do not impose a minimum Cloud-coverage
    constraint and do not change the Pareto objectives.  Starting from the
    fastest profile, clients are switched to their fastest Cloud-reaching
    candidate in increasing latency-cost-per-sample order until each requested
    represented-sample target is reached or no further Cloud candidate exists.
    """
    if not pools:
        return []
    sample_mass = {
        client_id: max(float((client_samples or {}).get(client_id, 1.0)), 0.0)
        for client_id in pools
    }
    total_mass = sum(sample_mass.values())
    if total_mass <= 0.0:
        sample_mass = {client_id: 1.0 for client_id in pools}
        total_mass = float(len(pools))

    base = dict(fastest)
    covered_mass = sum(
        sample_mass[client_id]
        for client_id, candidate in base.items()
        if _candidate_reaches_cloud(candidate)
    )
    switches: list[tuple[float, float, int, Candidate]] = []
    for client_id, candidates in pools.items():
        if _candidate_reaches_cloud(base[client_id]):
            continue
        cloud_candidates = [
            candidate for candidate in candidates if _candidate_reaches_cloud(candidate)
        ]
        if not cloud_candidates:
            continue
        cloud_candidate = min(
            cloud_candidates,
            key=lambda item: (item.time, _candidate_key(item)),
        )
        mass = max(sample_mass[client_id], 1e-12)
        latency_delta = float(cloud_candidate.time - base[client_id].time)
        switches.append(
            (
                max(latency_delta, 0.0) / mass,
                latency_delta,
                client_id,
                cloud_candidate,
            )
        )
    switches.sort(key=lambda item: (item[0], item[1], item[2], _candidate_key(item[3])))

    anchors: list[dict[int, Candidate]] = []
    current = dict(base)
    switch_index = 0
    for target in targets:
        target_mass = min(max(float(target), 0.0), 1.0) * total_mass
        while covered_mass + 1e-12 < target_mass and switch_index < len(switches):
            _cost, _delta, client_id, cloud_candidate = switches[switch_index]
            switch_index += 1
            if _candidate_reaches_cloud(current[client_id]):
                continue
            current[client_id] = cloud_candidate
            covered_mass += sample_mass[client_id]
        anchors.append(dict(current))
    return _unique_profiles(anchors)


def _selection_learning_cost(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    admitted_client_ids: tuple[int, ...],
) -> float:
    """The *same* formal J_learn used by complete-profile Pareto evaluation.

    Fixed-admission lookahead is used for cheap seed/neighbor comparisons;
    selected neighbors are subsequently evaluated using their actual flow.
    """
    r_cloud = _cloud_update_coverage_ratio(profile, client_samples, admitted_client_ids)
    clip = max(float(config.omega_update_clip_norm), 0.0)
    return (
        4.0 * clip * clip * (1.0 - r_cloud) ** 2
        + _global_dp_perturbation_cost(
            config, profile, client_samples, client_edges,
            admitted_client_ids=admitted_client_ids,
        )
    )


def _initial_profiles(
    config: SelectionConfig,
    pools: dict[int, list[Candidate]],
    previous_choices: dict[int, Candidate],
    client_samples: dict[int, float] | None = None,
    client_edges: dict[int, int] | None = None,
) -> list[dict[int, Candidate]]:
    """Seed the bounded search from latency, Cloud coverage and formal J_learn.

    No retired per-client Omega or independently released packet cost is used
    when choosing a learning-oriented seed. All tentative comparisons share
    the same provisional admission set; exact flow is checked after seeding.
    """
    fastest = {
        cid: min(candidates, key=lambda item: (item.time, _candidate_key(item)))
        for cid, candidates in pools.items()
    }
    previous = {}
    for cid, candidates in pools.items():
        prior = previous_choices.get(cid)
        previous[cid] = next(
            (item for item in candidates
             if prior is not None and _candidate_key(item) == _candidate_key(prior)),
            fastest[cid],
        )
    coverage_anchors = _cloud_coverage_anchor_profiles(
        pools, fastest, client_samples=client_samples,
    )
    masses = client_samples or {cid: 1.0 for cid in pools}
    edges = client_edges or {cid: 0 for cid in pools}
    admitted = tuple(sorted(pools))

    # Start at maximum reachable Cloud coverage.  Only modify the seed when
    # the paper's formal learning cost improves, using the real cohort masses.
    #
    # Candidate trials replace exactly one client while the provisional
    # admission cohort remains fixed.  Reuse the exact incremental replacement
    # accounting already used by bounded neighbor search instead of rescanning
    # the complete profile for every trial.
    lowest_learning = dict(coverage_anchors[-1] if coverage_anchors else fastest)

    for cid in sorted(pools):
        current = lowest_learning[cid]
        best = current

        # Keep one exact full-profile evaluation for the current seed.
        best_cost = _selection_learning_cost(
            config, lowest_learning, masses, edges, admitted
        )

        replacement_stats = _LearningReplacementStats(
            config,
            lowest_learning,
            masses,
            edges,
            admitted,
        )

        for item in pools[cid]:
            if item == current:
                continue

            cost = replacement_stats.replacement_cost(
                cid,
                item,
                admitted,
            )

            # Defensive exact fallback.  This should normally not be needed
            # because the admission cohort is fixed during seed construction.
            if cost is None:
                lowest_learning[cid] = item
                cost = _selection_learning_cost(
                    config,
                    lowest_learning,
                    masses,
                    edges,
                    admitted,
                )
                lowest_learning[cid] = current

            if (cost, item.time, _candidate_key(item)) < (
                best_cost, best.time, _candidate_key(best)
            ):
                best_cost, best = cost, item

        lowest_learning[cid] = best

    # Evaluate the formal learning cost of single-client switches in the
    # learning-oriented seed rather than an obsolete local convergence proxy.
    reference_cost = _selection_learning_cost(
        config,
        lowest_learning,
        masses,
        edges,
        admitted,
    )

    hint_stats = _LearningReplacementStats(
        config,
        lowest_learning,
        masses,
        edges,
        admitted,
    )

    learning_hint: dict[int, dict[tuple, float]] = {}

    for cid, candidates in pools.items():
        learning_hint[cid] = {}
        current = lowest_learning[cid]

        for item in candidates:
            if item == current:
                cost = reference_cost
            else:
                cost = hint_stats.replacement_cost(
                    cid,
                    item,
                    admitted,
                )

                if cost is None:
                    lowest_learning[cid] = item
                    cost = _selection_learning_cost(
                        config,
                        lowest_learning,
                        masses,
                        edges,
                        admitted,
                    )
                    lowest_learning[cid] = current

            learning_hint[cid][_candidate_key(item)] = cost

    intermediate_profiles: list[dict[int, Candidate]] = []
    for latency_weight in (0.25, 0.5, 0.75):
        profile = {}
        for cid, candidates in pools.items():
            times = [item.time for item in candidates]
            costs = list(learning_hint[cid].values())
            t_min, t_max = min(times), max(times)
            l_min, l_max = min(costs), max(costs)
            profile[cid] = min(
                candidates,
                key=lambda item: (
                    max(
                        latency_weight * _safe_norm_eps(
                            item.time, t_min, t_max, config.pareto_norm_eps
                        ),
                        (1.0 - latency_weight) * _safe_norm_eps(
                            learning_hint[cid][_candidate_key(item)],
                            l_min, l_max, config.pareto_norm_eps,
                        ),
                    ),
                    item.time, _candidate_key(item),
                ),
            )
        intermediate_profiles.append(profile)
    return _unique_profiles(
        [previous, fastest, *intermediate_profiles, *coverage_anchors, lowest_learning]
    )


def _initial_latency_profiles(
    pools: dict[int, list[Candidate]],
    previous_choices: dict[int, Candidate],
) -> list[dict[int, Candidate]]:
    fastest = {
        client_id: min(
            candidates,
            key=lambda item: (item.time, _candidate_key(item)),
        )
        for client_id, candidates in pools.items()
    }
    previous = {}
    for client_id, candidates in pools.items():
        prior = previous_choices.get(client_id)
        previous[client_id] = next(
            (
                item
                for item in candidates
                if prior is not None and _candidate_key(item) == _candidate_key(prior)
            ),
            fastest[client_id],
        )
    return _unique_profiles([previous, fastest])


def _pareto_search_client_ids(
    config: SelectionConfig,
    pools: dict[int, list[Candidate]],
    seeds: list[dict[int, Candidate]],
) -> tuple[int, ...]:
    if not config.pareto_conflict_only or len(seeds) < 2:
        return tuple(sorted(pools))
    conflicting = []
    for client_id in sorted(pools):
        keys = {_candidate_key(seed[client_id]) for seed in seeds if client_id in seed}
        seed_candidates = [seed[client_id] for seed in seeds if client_id in seed]
        has_unsearched_cloud_tradeoff = (
            any(not _candidate_reaches_cloud(candidate) for candidate in seed_candidates)
            and any(_candidate_reaches_cloud(candidate) for candidate in pools[client_id])
        )
        if len(keys) > 1 or has_unsearched_cloud_tradeoff:
            conflicting.append(client_id)
    return tuple(conflicting or sorted(pools))



class _LearningReplacementStats:
    """Fixed-admission O(affected-group) single-client J_learn lookahead.

    The exact full-profile objective remains the reference; this snapshot
    replaces the repeated scan of every client during neighbor ranking.
    Cloud noise is stored in unnormalized sample-mass units, making changes
    to the Cloud denominator an exact one-scalar update. Edge-local costs
    need only recompute the client's own Edge group.
    """

    def __init__(
        self,
        config: SelectionConfig,
        profile: dict[int, Candidate],
        client_samples: dict[int, float],
        client_edges: dict[int, int],
        admitted_client_ids: tuple[int, ...] | list[int],
    ) -> None:
        self.config = config
        self.profile = profile
        self.samples = client_samples
        self.edges = client_edges
        self.admitted_ids = tuple(admitted_client_ids)
        self.admitted = frozenset(self.admitted_ids)
        self.total_samples = sum(max(float(value), 0.0) for value in client_samples.values())
        self.admitted_mass = sum(
            max(float(client_samples.get(cid, 0.0)), 0.0)
            for cid in self.admitted if cid in profile
        )
        self.dim = max(float(config.omega_update_dimension), 0.0)
        self.clip = max(float(config.omega_update_clip_norm), 0.0)
        self.factor = self.dim * (2.0 * self.clip) ** 2
        self.default_sigma = max(
            float(resolved_privacy_parameters(config)["update_noise_multiplier"]), 0.0
        ) if self.factor > 0.0 and self.admitted else 0.0
        self.cloud_mass = 0.0
        self.secure: dict[int, tuple[float, float]] = {}
        self.packets: dict[tuple[int, str, str], dict[int, tuple[float, float]]] = {}
        self.independent: dict[int, tuple[float, float]] = {}
        self.edge_groups: dict[int, dict[int, Candidate]] = {}
        for cid in self.admitted:
            candidate = profile.get(cid)
            if candidate is None:
                continue
            if _candidate_reaches_cloud(candidate):
                self.cloud_mass += self._mass(cid)
            classification, key, pair = self._cloud_entry(cid, candidate)
            if classification == "secure":
                self.secure[cid] = pair
            elif classification == "packet":
                self.packets.setdefault(key, {})[cid] = pair
            elif classification == "independent":
                self.independent[cid] = pair
            if candidate.mode == "LIIE":
                self.edge_groups.setdefault(int(client_edges.get(cid, -1)), {})[cid] = candidate
        self.secure_score = self._max_product_sq(self.secure)

        # Stage 12E1:
        # _max_product_sq(secure) depends only on the independent maxima of
        # mass and sigma.  A single-client replacement can remove at most one
        # member, so top-2 is sufficient for an exact O(1) update.
        self.secure_mass_top2 = tuple(
            sorted(
                (
                    (float(pair[0]), int(cid))
                    for cid, pair in self.secure.items()
                ),
                reverse=True,
            )[:2]
        )
        self.secure_sigma_top2 = tuple(
            sorted(
                (
                    (float(pair[1]), int(cid))
                    for cid, pair in self.secure.items()
                ),
                reverse=True,
            )[:2]
        )

        self.packet_scores = {
            key: self._max_product_sq(members)
            for key, members in self.packets.items()
        }

        # Stage 12E2:
        # Exact O(1) single-client replacement inside each packet group.
        self.packet_mass_top2 = {
            key: tuple(
                sorted(
                    (
                        (float(pair[0]), int(cid))
                        for cid, pair in members.items()
                    ),
                    reverse=True,
                )[:2]
            )
            for key, members in self.packets.items()
        }

        self.packet_sigma_top2 = {
            key: tuple(
                sorted(
                    (
                        (float(pair[1]), int(cid))
                        for cid, pair in members.items()
                    ),
                    reverse=True,
                )[:2]
            )
            for key, members in self.packets.items()
        }

        self.packet_score_total = sum(self.packet_scores.values())
        self.independent_score = sum(
            (mass * sigma) ** 2
            for mass, sigma in self.independent.values()
        )
        self.edge_scores = {
            edge: self._edge_cost(members)
            for edge, members in self.edge_groups.items()
        } if self.factor > 0.0 else {}
        self.edge_total = sum(self.edge_scores.values())

        # Stage 12E3:
        # Exact fast path for LIIE -> LIIE replacement. Membership and
        # group sample mass stay fixed, so only the candidate-dependent
        # DP mechanism/sigma contribution needs updating.
        self.edge_group_masses: dict[int, float] = {}
        self.edge_member_counts: dict[int, int] = {}
        self.edge_aggregate_counts: dict[int, int] = {}
        self.edge_independent_group_scores: dict[int, float] = {}
        self.edge_mass_max: dict[int, float] = {}
        self.edge_sigma_top2: dict[int, tuple[tuple[float, int], ...]] = {}

        for edge, members in self.edge_groups.items():
            group_mass = sum(
                self._mass(member_cid)
                for member_cid in members
            )

            self.edge_group_masses[edge] = group_mass
            self.edge_member_counts[edge] = len(members)

            self.edge_aggregate_counts[edge] = sum(
                1
                for member_candidate in members.values()
                if _candidate_uses_edge_local_exact_update_dp(
                    member_candidate
                )
            )

            if group_mass > 0.0:
                self.edge_independent_group_scores[edge] = sum(
                    (
                        self._mass(member_cid)
                        * self._sigma(member_candidate)
                        / group_mass
                    ) ** 2
                    for member_cid, member_candidate in members.items()
                    if mechanism_uses_dp(
                        candidate_link_mechanism(
                            member_candidate,
                            "L_E_upd",
                        )
                    )
                )
            else:
                self.edge_independent_group_scores[edge] = 0.0

            self.edge_mass_max[edge] = max(
                (
                    self._mass(member_cid)
                    for member_cid in members
                ),
                default=0.0,
            )

            self.edge_sigma_top2[edge] = tuple(
                sorted(
                    (
                        (
                            float(self._sigma(member_candidate)),
                            int(member_cid),
                        )
                        for member_cid, member_candidate
                        in members.items()
                    ),
                    reverse=True,
                )[:2]
            )

    def _mass(self, cid: int) -> float:
        return max(float(self.samples.get(cid, 0.0)), 0.0)

    def _sigma(self, candidate: Candidate) -> float:
        return max(float(candidate.update_noise_multiplier or self.default_sigma), 0.0)

    @staticmethod
    def _max_product_sq(members: dict[int, tuple[float, float]]) -> float:
        if not members:
            return 0.0
        mass = max(pair[0] for pair in members.values())
        sigma = max(pair[1] for pair in members.values())
        return (mass * sigma) ** 2

    @staticmethod
    def _replacement_max_from_top2(
        top2: tuple[tuple[float, int], ...],
        cid: int,
        new_value: float | None,
    ) -> float:
        """Exact max after removing cid and optionally inserting new_value."""
        if not top2:
            remaining = 0.0
        elif top2[0][1] == cid:
            remaining = top2[1][0] if len(top2) > 1 else 0.0
        else:
            remaining = top2[0][0]

        if new_value is not None:
            remaining = max(remaining, float(new_value))

        return float(remaining)


    def _cloud_entry(
        self, cid: int, candidate: Candidate,
    ) -> tuple[str, tuple[int, str, str] | None, tuple[float, float]]:
        mass, sigma = self._mass(cid), self._sigma(candidate)
        pair = (mass, sigma)
        if not _candidate_reaches_cloud(candidate):
            return "none", None, pair
        if _candidate_uses_secure_aggregate_update_dp_for_selection(candidate):
            return "secure", None, pair
        if candidate.dp_execution_plan == "cloud_packet" and candidate.mode in EDGE_CLOUD_MODES:
            key = (
                int(self.edges.get(cid, -1)), candidate.mode,
                candidate_link_mechanism(candidate, "E_C_upd"),
            )
            return "packet", key, pair
        spec = MODE_SPECS.get(candidate.mode)
        edge_loops = max(int(spec.E_edge_loops if spec is not None else 1), 1)
        for link_id, obj, _count, privacy_eligible in _mode_link_transmissions(
            candidate.mode, 1, edge_loops,
        ):
            if (obj == "upd" and privacy_eligible and link_id.endswith("_C_upd")
                and mechanism_uses_dp(candidate_link_mechanism(candidate, link_id, fallback_object="upd"))):
                return "independent", None, pair
        return "none", None, pair

    def _edge_cost(self, members: dict[int, Candidate]) -> float:
        group_mass = sum(self._mass(cid) for cid in members)
        if group_mass <= 0.0 or self.admitted_mass <= 0.0:
            return 0.0
        aggregate = [cid for cid, candidate in members.items()
                     if _candidate_uses_edge_local_exact_update_dp(candidate)]
        if aggregate:
            if len(aggregate) != len(members) or len(members) < 2:
                raise ValueError("Inconsistent LIIE aggregate DP cohort in profile")
            group_sigma = max(self._sigma(members[cid]) for cid in members)
            max_mass = max(self._mass(cid) for cid in members)
            group_score = (group_sigma * max_mass / group_mass) ** 2
        else:
            group_score = sum(
                (self._mass(cid) * self._sigma(candidate) / group_mass) ** 2
                for cid, candidate in members.items()
                if mechanism_uses_dp(candidate_link_mechanism(candidate, "L_E_upd"))
            )
        return self.factor * (group_mass / self.admitted_mass) * group_score

    def replacement_cost(
        self,
        cid: int,
        candidate: Candidate,
        admitted_client_ids: tuple[int, ...] | list[int],
        old_cloud_entry: tuple[Any, Any, Any] | None = None,
    ) -> float | None:
        """Return exact fixed-admission lookahead, or None for a changed cohort."""
        if admitted_client_ids != self.admitted_ids:
            if frozenset(admitted_client_ids) != self.admitted:
                return None
        old = self.profile[cid]
        mass = self._mass(cid)
        cloud_mass = self.cloud_mass + mass * (
            int(_candidate_reaches_cloud(candidate)) - int(_candidate_reaches_cloud(old))
        )
        fusion_mass = cloud_mass / self.total_samples if self.total_samples > 0.0 else 0.0
        fusion_bound = 4.0 * self.clip * self.clip * (1.0 - fusion_mass) ** 2
        if self.factor <= 0.0 or not self.admitted:
            return fusion_bound

        if old_cloud_entry is None:
            old_type, old_key, old_pair = self._cloud_entry(cid, old)
        else:
            old_type, old_key, old_pair = old_cloud_entry

        new_type, new_key, new_pair = self._cloud_entry(cid, candidate)
        secure_score = self.secure_score
        if old_type == "secure" or new_type == "secure":
            new_mass = (
                float(new_pair[0])
                if new_type == "secure"
                else None
            )
            new_sigma = (
                float(new_pair[1])
                if new_type == "secure"
                else None
            )

            secure_mass_max = self._replacement_max_from_top2(
                self.secure_mass_top2,
                cid,
                new_mass,
            )
            secure_sigma_max = self._replacement_max_from_top2(
                self.secure_sigma_top2,
                cid,
                new_sigma,
            )

            secure_score = (
                secure_mass_max * secure_sigma_max
            ) ** 2

        packet_score = self.packet_score_total
        keys = {
            key
            for key in (
                old_key if old_type == "packet" else None,
                new_key if new_type == "packet" else None,
            )
            if key is not None
        }

        for key in keys:
            packet_score -= self.packet_scores.get(key, 0.0)

            new_mass = (
                float(new_pair[0])
                if new_type == "packet" and new_key == key
                else None
            )
            new_sigma = (
                float(new_pair[1])
                if new_type == "packet" and new_key == key
                else None
            )

            mass_max = self._replacement_max_from_top2(
                self.packet_mass_top2.get(key, ()),
                cid,
                new_mass,
            )
            sigma_max = self._replacement_max_from_top2(
                self.packet_sigma_top2.get(key, ()),
                cid,
                new_sigma,
            )

            packet_score += (
                mass_max * sigma_max
            ) ** 2

        independent_score = self.independent_score
        if old_type == "independent":
            independent_score -= (old_pair[0] * old_pair[1]) ** 2
        if new_type == "independent":
            independent_score += (new_pair[0] * new_pair[1]) ** 2
        cloud_dp = (
            self.factor * (secure_score + packet_score + independent_score) / (cloud_mass ** 2)
            if cloud_mass > 0.0 else 0.0
        )

        edge_dp = self.edge_total
        if old.mode == "LIIE" or candidate.mode == "LIIE":
            edge = int(self.edges.get(cid, -1))
            edge_dp -= self.edge_scores.get(edge, 0.0)

            if old.mode == "LIIE" and candidate.mode == "LIIE":
                # Membership is unchanged. Therefore member_count,
                # group_mass and mass maximum are fixed.
                group_mass = self.edge_group_masses.get(edge, 0.0)
                member_count = self.edge_member_counts.get(edge, 0)

                old_aggregate = (
                    _candidate_uses_edge_local_exact_update_dp(old)
                )
                new_aggregate = (
                    _candidate_uses_edge_local_exact_update_dp(candidate)
                )

                aggregate_count = (
                    self.edge_aggregate_counts.get(edge, 0)
                    - int(old_aggregate)
                    + int(new_aggregate)
                )

                if group_mass <= 0.0 or self.admitted_mass <= 0.0:
                    next_edge_cost = 0.0

                elif aggregate_count:
                    # Preserve the exact consistency check from _edge_cost().
                    if (
                        aggregate_count != member_count
                        or member_count < 2
                    ):
                        raise ValueError(
                            "Inconsistent LIIE aggregate DP cohort in profile"
                        )

                    group_sigma = self._replacement_max_from_top2(
                        self.edge_sigma_top2.get(edge, ()),
                        cid,
                        self._sigma(candidate),
                    )

                    max_mass = self.edge_mass_max.get(edge, 0.0)

                    group_score = (
                        group_sigma
                        * max_mass
                        / group_mass
                    ) ** 2

                    next_edge_cost = (
                        self.factor
                        * (group_mass / self.admitted_mass)
                        * group_score
                    )

                else:
                    # Independent-DP cohort. group_mass is unchanged, so
                    # update only this client's normalized squared term.
                    group_score = (
                        self.edge_independent_group_scores.get(
                            edge,
                            0.0,
                        )
                    )

                    old_uses_dp = mechanism_uses_dp(
                        candidate_link_mechanism(
                            old,
                            "L_E_upd",
                        )
                    )
                    new_uses_dp = mechanism_uses_dp(
                        candidate_link_mechanism(
                            candidate,
                            "L_E_upd",
                        )
                    )

                    if old_uses_dp:
                        group_score -= (
                            mass
                            * self._sigma(old)
                            / group_mass
                        ) ** 2

                    if new_uses_dp:
                        group_score += (
                            mass
                            * self._sigma(candidate)
                            / group_mass
                        ) ** 2

                    next_edge_cost = (
                        self.factor
                        * (group_mass / self.admitted_mass)
                        * group_score
                    )

                edge_dp += next_edge_cost

            else:
                # Stage 12F13:
                # Exact O(1) fast path for membership changes when the
                # resulting LIIE cohort uses independent DP.
                #
                # The existing normalized independent score is rescaled
                # from the old group mass to the new group mass, then only
                # this client's contribution is removed/added.
                base_group_mass = self.edge_group_masses.get(edge, 0.0)
                new_group_mass = base_group_mass
                new_aggregate_count = self.edge_aggregate_counts.get(edge, 0)

                if old.mode == "LIIE":
                    new_group_mass -= mass
                    new_aggregate_count -= int(
                        _candidate_uses_edge_local_exact_update_dp(old)
                    )
                else:
                    new_group_mass += mass
                    new_aggregate_count += int(
                        _candidate_uses_edge_local_exact_update_dp(candidate)
                    )

                if new_group_mass <= 0.0 or self.admitted_mass <= 0.0:
                    next_edge_cost = 0.0

                elif new_aggregate_count == 0:
                    group_score = self.edge_independent_group_scores.get(
                        edge,
                        0.0,
                    )

                    # Remove this client's old contribution while still
                    # expressed using the original denominator.
                    if (
                        old.mode == "LIIE"
                        and base_group_mass > 0.0
                        and mechanism_uses_dp(
                            candidate_link_mechanism(
                                old,
                                "L_E_upd",
                            )
                        )
                    ):
                        group_score -= (
                            mass
                            * self._sigma(old)
                            / base_group_mass
                        ) ** 2

                    # Rescale all unchanged members to the new denominator.
                    if base_group_mass > 0.0:
                        group_score *= (
                            base_group_mass / new_group_mass
                        ) ** 2
                    else:
                        group_score = 0.0

                    # Add this client's new contribution using the new
                    # denominator.
                    if (
                        candidate.mode == "LIIE"
                        and mechanism_uses_dp(
                            candidate_link_mechanism(
                                candidate,
                                "L_E_upd",
                            )
                        )
                    ):
                        group_score += (
                            mass
                            * self._sigma(candidate)
                            / new_group_mass
                        ) ** 2

                    next_edge_cost = (
                        self.factor
                        * (new_group_mass / self.admitted_mass)
                        * group_score
                    )

                    edge_dp += next_edge_cost

                else:
                    # Aggregate-DP membership changes retain the original
                    # reference path, including its consistency checks.
                    members = dict(
                        self.edge_groups.get(edge, {})
                    )
                    members.pop(cid, None)

                    if candidate.mode == "LIIE":
                        members[cid] = candidate

                    edge_dp += self._edge_cost(members)

        return float(fusion_bound + cloud_dp + edge_dp)


def _global_replacement_priority(
    *,
    objective: str,
    config: SelectionConfig,
    evaluated: ProfileEvaluation,
    client_id: int,
    current: Candidate,
    candidate: Candidate,
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    trial_profile: dict[int, Candidate] | None,
    flow_objectives: tuple[tuple[int, ...], float] | None,
    replacement_stats: _LearningReplacementStats | None = None,
    old_cloud_entry: tuple[Any, Any, Any] | None = None,
    profiler: dict[str, Any] | None = None,
) -> tuple[float, float | None]:
    """Rank neighbors by the two *formal* paper objectives.

    This fixed-admission score is only a beam prefilter, not a proof of
    dominance.  Exact flow and J_learn are evaluated for retained neighbors.
    """
    priority_started_at = time.perf_counter()
    _perf_add(profiler, "replacement_priority_calls")
    next_time = (
        float(flow_objectives[1]) if flow_objectives is not None
        else float(evaluated.system_latency + candidate.time - current.time)
    )
    if objective == "latency":
        return next_time, None
    admitted = (
        tuple(flow_objectives[0]) if flow_objectives is not None
        else evaluated.admitted_client_ids
    )
    next_learning = None
    if replacement_stats is not None:
        incremental_started_at = time.perf_counter()
        next_learning = replacement_stats.replacement_cost(
            client_id,
            candidate,
            admitted,
            old_cloud_entry=old_cloud_entry,
        )
        _perf_add(profiler, "incremental_jlearn_calls")
        _perf_add(profiler, "incremental_jlearn_sec", time.perf_counter() - incremental_started_at)
    if next_learning is None:
        fallback_started_at = time.perf_counter()

        # Stage 11B: materialize the complete trial profile only when the
        # incremental learning-cost path is unavailable.
        fallback_profile = trial_profile
        if fallback_profile is None:
            fallback_profile = dict(evaluated.profile)
            fallback_profile[client_id] = candidate
            _perf_add(
                profiler,
                "neighbor_priority_fallback_profile_materializations",
            )

        next_learning = _selection_learning_cost(
            config,
            fallback_profile,
            client_samples,
            client_edges,
            admitted,
        )
        _perf_add(profiler, "full_jlearn_fallback_calls")
        _perf_add(
            profiler,
            "full_jlearn_fallback_sec",
            time.perf_counter() - fallback_started_at,
        )
    arithmetic_started_at = time.perf_counter()

    time_scale = max(abs(evaluated.system_latency), abs(next_time), 1.0)
    learning_scale = max(
        abs(evaluated.system_learning_error), abs(next_learning), 1e-12,
    )
    priority = (
        0.5 * (next_time - evaluated.system_latency) / time_scale
        + 0.5 * (next_learning - evaluated.system_learning_error) / learning_scale
    )

    _perf_add(
        profiler,
        "replacement_priority_arithmetic_sec",
        time.perf_counter() - arithmetic_started_at,
    )
    _perf_add(
        profiler,
        "replacement_priority_sec",
        time.perf_counter() - priority_started_at,
    )

    return priority, float(next_learning)


def _state_diff_vector(state_diff: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    parts: list[torch.Tensor] = []
    for part_name in ("end", "edge"):
        for name in sorted(state_diff.get(part_name, {})):
            parts.append(state_diff[part_name][name].detach().to(torch.float64).reshape(-1).cpu())
    if not parts:
        return torch.zeros(0, dtype=torch.float64)
    return torch.cat(parts)


def _weighted_update(
    updates: list[tuple[torch.Tensor, float]],
) -> torch.Tensor:
    if not updates:
        return torch.zeros(0, dtype=torch.float64)
    total = sum(max(float(weight), 0.0) for _vec, weight in updates)
    if total <= 0.0:
        total = float(len(updates))
        return sum(vec for vec, _weight in updates) / total
    out = torch.zeros_like(updates[0][0])
    for vec, weight in updates:
        out = out + vec * (max(float(weight), 0.0) / total)
    return out


def _profile_fusion_distortion(
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    admitted_client_ids: tuple[int, ...],
    previous_client_updates: dict[int, dict[str, dict[str, torch.Tensor]]] | None,
) -> tuple[float, float]:
    """Estimate global-update distortion caused by incomplete Cloud fusion.

    The reference is sample-weighted full-client fusion of the previous-round
    effective client updates. The candidate profile keeps direct Cloud
    contributions individually and aggregates Edge->Cloud contributions by
    Edge before the final Cloud fusion.
    """
    if not previous_client_updates:
        return 0.0, 0.0

    update_vectors = {
        client_id: _state_diff_vector(state)
        for client_id, state in previous_client_updates.items()
        if state
    }
    if not update_vectors:
        return 0.0, 0.0

    common_ids = [
        client_id
        for client_id in sorted(client_samples)
        if client_id in update_vectors
    ]
    if not common_ids:
        return 0.0, 0.0

    reference = _weighted_update(
        [
            (
                update_vectors[client_id],
                float(client_samples.get(client_id, 0.0)),
            )
            for client_id in common_ids
        ]
    )
    if reference.numel() == 0:
        return 0.0, 0.0

    admitted = set(admitted_client_ids)
    direct_updates: list[tuple[torch.Tensor, float]] = []
    edge_groups: dict[int, list[tuple[torch.Tensor, float]]] = {}

    for client_id in common_ids:
        if client_id not in admitted:
            continue
        candidate = profile.get(client_id)
        if candidate is None or not _candidate_reaches_cloud(candidate):
            continue
        weight = float(client_samples.get(client_id, 0.0))
        vector = update_vectors[client_id]
        if candidate.mode in EDGE_CLOUD_MODES:
            edge_groups.setdefault(int(client_edges.get(client_id, -1)), []).append((vector, weight))
        else:
            direct_updates.append((vector, weight))

    cloud_contributions = list(direct_updates)
    for grouped in edge_groups.values():
        group_weight = sum(max(float(weight), 0.0) for _vec, weight in grouped)
        if group_weight <= 0.0:
            continue
        cloud_contributions.append((_weighted_update(grouped), group_weight))

    actual = _weighted_update(cloud_contributions)
    if actual.numel() == 0 or actual.shape != reference.shape:
        return 1.0, 1.0

    ref_norm = float(torch.linalg.vector_norm(reference))
    diff_norm = float(torch.linalg.vector_norm(reference - actual))
    relative = diff_norm / max(ref_norm, 1e-12)

    actual_norm = float(torch.linalg.vector_norm(actual))
    if ref_norm <= 1e-12 or actual_norm <= 1e-12:
        cosine_distortion = 1.0 if ref_norm > 1e-12 or actual_norm > 1e-12 else 0.0
    else:
        cosine = float(torch.dot(reference, actual) / (ref_norm * actual_norm))
        cosine_distortion = 1.0 - max(-1.0, min(1.0, cosine))
    return float(relative), float(cosine_distortion)


def _global_feature_perturbation_cost(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    admitted_client_ids: tuple[int, ...] | list[int] | set[int],
) -> float:
    """Update-space second-moment bound induced by split-feature DP.

    Only split representations that ultimately contribute to the current Cloud
    aggregate enter the global-learning objective. Feature-DP events on an
    Edge-only path still consume the feature privacy ledger, but their noisy
    representations do not perturb this round's Cloud update.

    Let K_z bound the sensitivity of one local SGD update to the transmitted
    representation. For m independent Gaussian feature releases, the stochastic
    contribution scales as m, while deterministic clipping bias can align across
    releases and therefore scales as m^2. Client contributions are then mapped
    into the Cloud aggregate with the same represented-sample weights used by
    the fusion term. The result has squared-model-update units, matching J_DP.
    """
    admitted = set(admitted_client_ids)
    cloud_ids = [
        client_id
        for client_id, candidate in profile.items()
        if client_id in admitted and _candidate_reaches_cloud(candidate)
    ]
    total_cloud_samples = sum(max(float(client_samples.get(i, 0.0)), 0.0) for i in cloud_ids)
    if total_cloud_samples <= 0.0:
        return 0.0

    privacy = resolved_privacy_parameters(config)
    clip_norm = max(float(config.omega_feature_clip_norm), 0.0)
    delta_z = 2.0 * clip_norm
    feature_dim = max(float(config.omega_feature_dimension), 1.0)
    # Existing feature Jacobian/Lipschitz constants jointly define the
    # representation-to-gradient sensitivity bound. Multiplication by eta maps
    # it into one-step parameter-update units.
    k_update = (
        max(float(config.omega_learning_rate), 0.0)
        * max(float(config.omega_feature_jacobian_norm), 0.0)
        * max(float(config.omega_feature_lipschitz), 0.0)
    )

    weighted_bias_norm = 0.0
    weighted_noise_second_moment = 0.0
    for client_id in cloud_ids:
        candidate = profile[client_id]
        m = max(int(candidate.feature_dp_events), 0)
        if m <= 0:
            continue
        sigma_f = max(
            float(
                candidate.feature_noise_multiplier
                if candidate.feature_noise_multiplier is not None
                else mode_aware_feature_noise_multiplier(config, m, resolved=privacy)
            ),
            0.0,
        )
        weight = max(float(client_samples.get(client_id, 0.0)), 0.0) / total_cloud_samples
        clip_excess_sq = max(_candidate_feature_clip_excess_sq(candidate, config), 0.0)
        # Worst-case coherent accumulation of deterministic clipping bias.
        client_bias_norm = k_update * float(m) * math.sqrt(clip_excess_sq)
        # Independent zero-mean Gaussian releases accumulate in second moment.
        client_noise_second_moment = (
            k_update * k_update
            * float(m)
            * feature_dim
            * (sigma_f * delta_z) ** 2
        )
        weighted_bias_norm += weight * client_bias_norm
        weighted_noise_second_moment += weight * weight * client_noise_second_moment

    return float(weighted_bias_norm ** 2 + weighted_noise_second_moment)


def _evaluate_profile(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    previous_choices: dict[int, Candidate],
    profile_signature: tuple[int, ...] = (),
    flow_inputs_by_candidate: dict[int, dict[tuple, ClientFlowInput]] | None = None,
    flow_objectives: tuple[tuple[int, ...], float] | None = None,
    previous_client_updates: dict[int, dict[str, dict[str, torch.Tensor]]] | None = None,
    fusion_objective_enabled: bool = False,
) -> ProfileEvaluation:
    if flow_objectives is None:
        flow_result = _profile_flow_result(
            config,
            profile,
            client_samples,
            client_edges,
            previous_choices,
            flow_inputs_by_candidate=flow_inputs_by_candidate,
        )
        admitted_client_ids = tuple(flow_result.selected_client_ids)
        system_latency = flow_result.round_duration
    else:
        admitted_client_ids, system_latency = flow_objectives
    system_dp = _global_dp_perturbation_cost(
        config,
        profile,
        client_samples,
        client_edges,
        admitted_client_ids=admitted_client_ids,
    )
    system_feature = _global_feature_perturbation_cost(
        config,
        profile,
        client_samples,
        admitted_client_ids,
    )
    system_update_clip = _global_update_clip_perturbation_cost(
        config,
        profile,
        client_samples,
        admitted_client_ids,
    )
    cloud_fusion_ratio = _cloud_update_coverage_ratio(
        profile,
        client_samples,
        admitted_client_ids,
    )
    # Formal selection-time fusion bound from the paper:
    #   J_fusion^ub = 4 C_u^2 (1-r_C)^2.
    # For Edge-only profiles r_C=0 and _global_dp_perturbation_cost returns
    # zero because there is no Cloud release, hence J_learn=4 C_u^2.
    clip_norm = max(float(config.omega_update_clip_norm), 0.0)
    fusion_bound = 4.0 * clip_norm * clip_norm * (1.0 - cloud_fusion_ratio) ** 2
    # Formal selection-time learning objective:
    #   J_learn = J_fusion^ub + J_DP.
    # Feature perturbation and update-clipping distortion are retained as
    # diagnostics, but they do not participate in Pareto selection. Split-feature
    # exposure is controlled by the resource-aware split activation rule together
    # with the mode-specific privacy/protection feasibility constraints.
    # J_DP is evaluated at the actual release boundary, including exact aggregate
    # releases for the corresponding full-local/hierarchical paths.
    system_learning_error = fusion_bound + system_dp
    fusion_distortion, fusion_cosine_distortion = _profile_fusion_distortion(
        profile,
        client_samples,
        client_edges,
        admitted_client_ids,
        previous_client_updates,
    )
    return ProfileEvaluation(
        profile=profile,
        system_latency=system_latency,
        system_omega=system_learning_error,
        cloud_fusion_ratio=cloud_fusion_ratio,
        admitted_client_ids=tuple(sorted(admitted_client_ids)),
        profile_signature=profile_signature,
        fusion_distortion=fusion_distortion,
        fusion_cosine_distortion=fusion_cosine_distortion,
        # The old three-objective diagnostic is intentionally retired from
        # selection.  Observed fusion distortion remains diagnostic-only.
        fusion_objective_enabled=False,
        dp_perturbation=system_dp,
        feature_perturbation=system_feature,
        update_clip_perturbation=system_update_clip,
        fusion_bound=fusion_bound,
    )


def _profile_flow_result(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    previous_choices: dict[int, Candidate],
    flow_inputs_by_candidate: dict[int, dict[tuple, ClientFlowInput]] | None = None,
):
    if flow_inputs_by_candidate is None:
        flow_inputs_by_candidate = _profile_flow_inputs_by_candidate(
            config,
            {client_id: [candidate] for client_id, candidate in profile.items()},
            client_samples,
            client_edges,
            previous_choices,
            tuple(sorted(profile)),
        )
    clients = [
        flow_inputs_by_candidate[client_id][_candidate_key(candidate)]
        for client_id, candidate in sorted(profile.items())
        if candidate.mode != "SKIP"
    ]
    return summarize_mixed_round_flow(
        round_idx=0,
        clients=clients,
        aggregation_fraction=config.aggregation_fraction,
        edge_aggregation_beta=config.edge_aggregation_beta,
        edge_aggregation_fixed=config.edge_aggregation_fixed,
        cloud_aggregation_beta=config.cloud_aggregation_beta,
        cloud_aggregation_fixed=config.cloud_aggregation_fixed,
    )


def _profile_flow_inputs_by_candidate(
    config: SelectionConfig,
    pools: dict[int, list[Candidate]],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    previous_choices: dict[int, Candidate],
    client_order: tuple[int, ...],
) -> dict[int, dict[tuple, ClientFlowInput]]:
    inputs: dict[int, dict[tuple, ClientFlowInput]] = {}
    for sequence, client_id in enumerate(client_order):
        by_candidate: dict[tuple, ClientFlowInput] = {}
        for candidate in pools[client_id]:
            if candidate.mode == "SKIP":
                continue
            by_candidate[_candidate_key(candidate)] = ClientFlowInput(
                client_id=client_id,
                edge_id=int(client_edges.get(client_id, -1)),
                mode=candidate.mode,
                candidate_time=candidate_arrival_with_switch(
                    config,
                    client_id,
                    candidate,
                    previous_choices.get(client_id),
                ),
                estimated_local_time=candidate.first_aggregation_arrival_time,
                measured_local_time=0.0,
                communication_volume=candidate.communication_volume,
                state_diff={},
                sample_count=max(1, int(round(client_samples.get(client_id, 1.0)))),
                edge_loops=MODE_SPECS[candidate.mode].E_edge_loops,
                fused_release=config.mainline_fusion,
                edge_to_cloud_time=candidate.edge_to_cloud_time,
                return_path_time=candidate.return_path_time,
                edge_aggregation_payload=candidate.edge_aggregation_payload,
                cloud_aggregation_payload=candidate.cloud_aggregation_payload,
                aggregation_group=(
                    candidate_link_mechanism(candidate, "E_C_upd")
                    if candidate.mode in EDGE_CLOUD_MODES
                    else ""
                ),
                dispatch_sequence=sequence,
            )
        inputs[client_id] = by_candidate
    return inputs


def _full_buffer_flow_stats(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    flow_inputs_by_candidate: dict[int, dict[tuple, ClientFlowInput]],
) -> _FullBufferFlowStats | None:
    if config.mainline_fusion:
        return None
    client_inputs = {
        client_id: flow_inputs_by_candidate[client_id][_candidate_key(candidate)]
        for client_id, candidate in profile.items()
        if candidate.mode != "SKIP"
    }
    groups: dict[tuple[str, int, str], list[ClientFlowInput]] = {}
    for client in client_inputs.values():
        groups.setdefault(_full_buffer_group_key(client), []).append(client)

    # Q70: only Edge has a threshold buffer. Direct-cloud contributions and
    # Edge aggregates are all retained for one round-synchronous Cloud event.
    if any(
        key[0] != "__direct_cloud__"
        and _buffer_size(len(group), config.aggregation_fraction) != len(group)
        for key, group in groups.items()
    ):
        return None

    frozen_groups = {
        key: tuple(group)
        for key, group in groups.items()
    }
    return _FullBufferFlowStats(
        client_inputs=client_inputs,
        groups=frozen_groups,
        summaries={
            key: _summarize_full_buffer_group(config, key, group)
            for key, group in frozen_groups.items()
        },
        admitted_client_ids=tuple(sorted(client_inputs)),
    )


def _replace_full_buffer_flow_objectives(
    config: SelectionConfig,
    stats: _FullBufferFlowStats,
    client_id: int,
    new_input: ClientFlowInput | None,
) -> tuple[tuple[int, ...], float]:
    old_input = stats.client_inputs.get(client_id)
    old_key = None if old_input is None else _full_buffer_group_key(old_input)
    new_key = None if new_input is None else _full_buffer_group_key(new_input)
    affected_keys = {key for key in (old_key, new_key) if key is not None}
    summaries = [
        summary
        for key, summary in stats.summaries.items()
        if key not in affected_keys
    ]
    for key in affected_keys:
        summary = _replace_full_buffer_group_latency_summary(
            config,
            key,
            stats.summaries.get(key),
            old_input if old_key == key else None,
            new_input if new_key == key else None,
        )
        if summary is not None:
            summaries.append(summary)

    if old_input is None and new_input is not None:
        admitted_client_ids = tuple(sorted((*stats.admitted_client_ids, client_id)))
    elif old_input is not None and new_input is None:
        admitted_client_ids = tuple(
            admitted_id
            for admitted_id in stats.admitted_client_ids
            if admitted_id != client_id
        )
    else:
        admitted_client_ids = stats.admitted_client_ids
    return admitted_client_ids, _full_buffer_system_latency(config, summaries)


def _full_buffer_group_key(client: ClientFlowInput) -> tuple[str, int, str]:
    if client.mode in CLOUD_DIRECT_MODES:
        return ("__direct_cloud__", -1, "")
    return (client.mode, client.edge_id, client.aggregation_group)


def _summarize_full_buffer_group(
    config: SelectionConfig,
    key: tuple[str, int, str],
    clients: list[ClientFlowInput] | tuple[ClientFlowInput, ...],
) -> _FullBufferGroupSummary:
    arrival_top, arrival_second = _top_two(
        (
            client.arrival_time,
            client.dispatch_sequence,
            client.client_id,
            max(1, int(client.edge_loops)),
        )
        for client in clients
    )
    return_top, return_second = _top_two(
        (
            client.return_path_time,
            client.dispatch_sequence,
            client.client_id,
            0,
        )
        for client in clients
    )
    edge_upload_top, edge_upload_second = _top_two(
        (
            client.edge_to_cloud_time,
            client.dispatch_sequence,
            client.client_id,
            0,
        )
        for client in clients
    )
    cloud_payload_top, cloud_payload_second = _top_two(
        (
            client.cloud_aggregation_payload,
            client.dispatch_sequence,
            client.client_id,
            0,
        )
        for client in clients
    )
    assert arrival_top is not None
    assert return_top is not None
    assert edge_upload_top is not None
    assert cloud_payload_top is not None
    return _build_full_buffer_group_summary(
        config,
        key,
        member_count=len(clients),
        edge_payload_sum=sum(client.edge_aggregation_payload for client in clients),
        cloud_payload_sum=sum(client.cloud_aggregation_payload for client in clients),
        arrival_top=arrival_top,
        arrival_second=arrival_second,
        return_top=return_top,
        return_second=return_second,
        edge_upload_top=edge_upload_top,
        edge_upload_second=edge_upload_second,
        cloud_payload_top=cloud_payload_top,
        cloud_payload_second=cloud_payload_second,
    )


def _replace_full_buffer_group_latency_summary(
    config: SelectionConfig,
    key: tuple[str, int, str],
    summary: _FullBufferGroupSummary | None,
    old_input: ClientFlowInput | None,
    new_input: ClientFlowInput | None,
) -> _FullBufferLatencySummary | None:
    old_count = 0 if old_input is None else 1
    new_count = 0 if new_input is None else 1
    member_count = (
        (0 if summary is None else summary.member_count)
        - old_count
        + new_count
    )
    if member_count <= 0:
        return None

    old_client_id = None if old_input is None else old_input.client_id

    if summary is None:
        arrival_top = None
        return_top = None
        edge_upload_top = None
        cloud_payload_top = None
    else:
        arrival_top = (
            summary.arrival_second
            if summary.arrival_top[-2] == old_client_id
            else summary.arrival_top
        )
        return_top = (
            summary.return_second
            if summary.return_top[-2] == old_client_id
            else summary.return_top
        )
        edge_upload_top = (
            summary.edge_upload_second
            if summary.edge_upload_top[-2] == old_client_id
            else summary.edge_upload_top
        )
        cloud_payload_top = (
            summary.cloud_payload_second
            if summary.cloud_payload_top[-2] == old_client_id
            else summary.cloud_payload_top
        )

    if new_input is not None:
        dispatch_sequence = new_input.dispatch_sequence
        new_client_id = new_input.client_id

        new_arrival_top = (
            new_input.arrival_time,
            dispatch_sequence,
            new_client_id,
            max(1, int(new_input.edge_loops)),
        )
        if arrival_top is None or new_arrival_top > arrival_top:
            arrival_top = new_arrival_top

        new_return_top = (
            new_input.return_path_time,
            dispatch_sequence,
            new_client_id,
            0,
        )
        if return_top is None or new_return_top > return_top:
            return_top = new_return_top

        new_edge_upload_top = (
            new_input.edge_to_cloud_time,
            dispatch_sequence,
            new_client_id,
            0,
        )
        if edge_upload_top is None or new_edge_upload_top > edge_upload_top:
            edge_upload_top = new_edge_upload_top

        new_cloud_payload_top = (
            new_input.cloud_aggregation_payload,
            dispatch_sequence,
            new_client_id,
            0,
        )
        if (
            cloud_payload_top is None
            or new_cloud_payload_top > cloud_payload_top
        ):
            cloud_payload_top = new_cloud_payload_top

    assert arrival_top is not None
    assert return_top is not None
    assert edge_upload_top is not None
    assert cloud_payload_top is not None

    edge_payload_sum = (
        (0.0 if summary is None else summary.edge_payload_sum)
        - (0.0 if old_input is None else old_input.edge_aggregation_payload)
        + (0.0 if new_input is None else new_input.edge_aggregation_payload)
    )
    cloud_payload_sum = (
        (0.0 if summary is None else summary.cloud_payload_sum)
        - (0.0 if old_input is None else old_input.cloud_aggregation_payload)
        + (0.0 if new_input is None else new_input.cloud_aggregation_payload)
    )

    start_time = arrival_top[0]
    return_path_time = return_top[0]

    kind = "edge_cloud"
    terminal_time = 0.0
    cloud_arrival_time = 0.0
    cloud_aggregation_payload = cloud_payload_top[0]

    if key[0] == "__direct_cloud__":
        kind = "direct_cloud"
        cloud_arrival_time = start_time
        cloud_aggregation_payload = cloud_payload_sum
    else:
        edge_aggregation_time = arrival_top[3] * (
            config.edge_aggregation_beta * edge_payload_sum
            + config.edge_aggregation_fixed
        )
        edge_finish_time = start_time + edge_aggregation_time

        if key[0] in EDGE_ONLY_MODES:
            kind = "edge_only"
            terminal_time = edge_finish_time + return_path_time
            cloud_aggregation_payload = 0.0
        else:
            cloud_arrival_time = (
                edge_finish_time
                + edge_upload_top[0]
            )

    return _FullBufferLatencySummary(
        kind=kind,
        terminal_time=terminal_time,
        cloud_arrival_time=cloud_arrival_time,
        cloud_aggregation_payload=cloud_aggregation_payload,
        return_path_time=return_path_time,
    )


def _replace_full_buffer_group_summary(
    config: SelectionConfig,
    key: tuple[str, int, str],
    summary: _FullBufferGroupSummary | None,
    old_input: ClientFlowInput | None,
    new_input: ClientFlowInput | None,
) -> _FullBufferGroupSummary | None:
    old_count = 0 if old_input is None else 1
    new_count = 0 if new_input is None else 1
    member_count = (0 if summary is None else summary.member_count) - old_count + new_count
    if member_count <= 0:
        return None

    old_client_id = None if old_input is None else old_input.client_id
    arrival_top = _updated_top(
        None if summary is None else summary.arrival_top,
        None if summary is None else summary.arrival_second,
        old_client_id,
        None
        if new_input is None
        else (
            new_input.arrival_time,
            new_input.dispatch_sequence,
            new_input.client_id,
            max(1, int(new_input.edge_loops)),
        ),
    )
    return_top = _updated_top(
        None if summary is None else summary.return_top,
        None if summary is None else summary.return_second,
        old_client_id,
        None
        if new_input is None
        else (
            new_input.return_path_time,
            new_input.dispatch_sequence,
            new_input.client_id,
            0,
        ),
    )
    edge_upload_top = _updated_top(
        None if summary is None else summary.edge_upload_top,
        None if summary is None else summary.edge_upload_second,
        old_client_id,
        None
        if new_input is None
        else (
            new_input.edge_to_cloud_time,
            new_input.dispatch_sequence,
            new_input.client_id,
            0,
        ),
    )
    cloud_payload_top = _updated_top(
        None if summary is None else summary.cloud_payload_top,
        None if summary is None else summary.cloud_payload_second,
        old_client_id,
        None
        if new_input is None
        else (
            new_input.cloud_aggregation_payload,
            new_input.dispatch_sequence,
            new_input.client_id,
            0,
        ),
    )
    assert arrival_top is not None
    assert return_top is not None
    assert edge_upload_top is not None
    assert cloud_payload_top is not None
    return _build_full_buffer_group_summary(
        config,
        key,
        member_count=member_count,
        edge_payload_sum=(0.0 if summary is None else summary.edge_payload_sum)
        - (0.0 if old_input is None else old_input.edge_aggregation_payload)
        + (0.0 if new_input is None else new_input.edge_aggregation_payload),
        cloud_payload_sum=(0.0 if summary is None else summary.cloud_payload_sum)
        - (0.0 if old_input is None else old_input.cloud_aggregation_payload)
        + (0.0 if new_input is None else new_input.cloud_aggregation_payload),
        arrival_top=arrival_top,
        arrival_second=None,
        return_top=return_top,
        return_second=None,
        edge_upload_top=edge_upload_top,
        edge_upload_second=None,
        cloud_payload_top=cloud_payload_top,
        cloud_payload_second=None,
    )


def _build_full_buffer_group_summary(
    config: SelectionConfig,
    key: tuple[str, int, str],
    *,
    member_count: int,
    edge_payload_sum: float,
    cloud_payload_sum: float,
    arrival_top: tuple[float, int, int, int],
    arrival_second: tuple[float, int, int, int] | None,
    return_top: tuple[float, int, int, int],
    return_second: tuple[float, int, int, int] | None,
    edge_upload_top: tuple[float, int, int, int],
    edge_upload_second: tuple[float, int, int, int] | None,
    cloud_payload_top: tuple[float, int, int, int],
    cloud_payload_second: tuple[float, int, int, int] | None,
) -> _FullBufferGroupSummary:
    start_time = arrival_top[0]
    return_path_time = return_top[0]
    kind = "edge_cloud"
    edge_aggregation_time = 0.0
    terminal_time = 0.0
    cloud_arrival_time = 0.0
    cloud_aggregation_payload = cloud_payload_top[0]
    if key[0] == "__direct_cloud__":
        kind = "direct_cloud"
        # Direct contributions are Cloud sources, not an independent Cloud
        # buffer/event. They join Interface-III edge aggregates in the single
        # round-synchronous Cloud aggregation (Q70).
        cloud_arrival_time = start_time
        cloud_aggregation_payload = cloud_payload_sum
    else:
        edge_aggregation_time = arrival_top[3] * (
            config.edge_aggregation_beta * edge_payload_sum
            + config.edge_aggregation_fixed
        )
        edge_finish_time = start_time + edge_aggregation_time
        if key[0] in EDGE_ONLY_MODES:
            kind = "edge_only"
            terminal_time = edge_finish_time + return_path_time
            cloud_aggregation_payload = 0.0
        else:
            cloud_arrival_time = edge_finish_time + edge_upload_top[0]

    return _FullBufferGroupSummary(
        kind=kind,
        edge_aggregation_time=edge_aggregation_time,
        terminal_time=terminal_time,
        cloud_arrival_time=cloud_arrival_time,
        cloud_aggregation_payload=cloud_aggregation_payload,
        return_path_time=return_path_time,
        member_count=member_count,
        edge_payload_sum=edge_payload_sum,
        cloud_payload_sum=cloud_payload_sum,
        arrival_top=arrival_top,
        arrival_second=arrival_second,
        return_top=return_top,
        return_second=return_second,
        edge_upload_top=edge_upload_top,
        edge_upload_second=edge_upload_second,
        cloud_payload_top=cloud_payload_top,
        cloud_payload_second=cloud_payload_second,
    )


def _top_two(values):
    first = None
    second = None
    for value in values:
        if first is None or value > first:
            second = first
            first = value
        elif second is None or value > second:
            second = value
    return first, second


def _updated_top(first, second, old_client_id: int | None, new_value):
    remaining = second if first is not None and first[-2] == old_client_id else first
    if new_value is not None and (remaining is None or new_value > remaining):
        return new_value
    return remaining


def _full_buffer_system_latency(
    config: SelectionConfig,
    summaries: list[
        _FullBufferGroupSummary | _FullBufferLatencySummary
    ],
) -> float:
    edge_terminal_max: float | None = None
    cloud_arrival_max: float | None = None
    cloud_payload_sum = 0.0
    cloud_return_max: float | None = None

    for summary in summaries:
        kind = summary.kind
        if kind == "edge_only":
            terminal_time = summary.terminal_time
            if (
                edge_terminal_max is None
                or terminal_time > edge_terminal_max
            ):
                edge_terminal_max = terminal_time
        elif kind == "direct_cloud" or kind == "edge_cloud":
            cloud_arrival_time = summary.cloud_arrival_time
            if (
                cloud_arrival_max is None
                or cloud_arrival_time > cloud_arrival_max
            ):
                cloud_arrival_max = cloud_arrival_time

            cloud_payload_sum += summary.cloud_aggregation_payload

            return_path_time = summary.return_path_time
            if (
                cloud_return_max is None
                or return_path_time > cloud_return_max
            ):
                cloud_return_max = return_path_time

    terminal_max = edge_terminal_max

    if cloud_arrival_max is not None:
        assert cloud_return_max is not None
        cloud_terminal_time = (
            cloud_arrival_max
            + config.cloud_aggregation_beta * cloud_payload_sum
            + config.cloud_aggregation_fixed
            + cloud_return_max
        )
        if (
            terminal_max is None
            or cloud_terminal_time > terminal_max
        ):
            terminal_max = cloud_terminal_time

    return 0.0 if terminal_max is None else terminal_max


def _admitted_clients_for_profile(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_edges: dict[int, int],
    previous_choices: dict[int, Candidate],
) -> list[int]:
    """Approximate A^(t): earliest buffered arrivals at each aggregation endpoint."""
    if config.aggregation_fraction >= 1.0:
        return sorted(
            client_id
            for client_id, candidate in profile.items()
            if candidate.mode != "SKIP"
        )

    groups: dict[tuple[str, int], list[tuple[float, int]]] = {}
    edge_cloud_groups: dict[tuple[str, int], tuple[float, list[int]]] = {}
    for client_id, candidate in profile.items():
        if candidate.mode == "SKIP":
            continue
        latency = candidate_arrival_with_switch(config, client_id, candidate, previous_choices.get(client_id))
        edge_id = int(client_edges.get(client_id, -1))
        if candidate.mode in {"LIE", "LIIE"}:
            groups.setdefault(("edge_only", edge_id), []).append((latency, client_id))
        elif candidate.mode in {"LIC", "LIIC"}:
            groups.setdefault(("direct_cloud", -1), []).append((latency, client_id))
        elif candidate.mode == "LIEIIC":
            groups.setdefault(("direct_cloud", -1), []).append((latency, client_id))
        elif candidate.mode in {"LIEIIIC", "LIIEIIIC"}:
            groups.setdefault((candidate.mode, edge_id), []).append((latency, client_id))

    admitted: set[int] = set()
    for key, arrivals in groups.items():
        chosen = _admit_fastest(arrivals, config.aggregation_fraction)
        if key[0] == "edge_only":
            admitted.update(client_id for _latency, client_id in chosen)
        elif key[0] == "direct_cloud":
            # Cloud is round-synchronous: direct-cloud clients are not filtered
            # by the Edge buffer fraction.
            admitted.update(client_id for _latency, client_id in arrivals)
        else:
            if chosen:
                admitted.update(client_id for _latency, client_id in chosen)
    return sorted(admitted)


def candidate_latency_with_switch(
    config: SelectionConfig,
    client_id: int,
    candidate: Candidate,
    previous: Candidate | None,
) -> float:
    if previous is None or previous.mode in {"", "SKIP"} or candidate.mode == "SKIP":
        return candidate.time
    if previous.mode == candidate.mode:
        return candidate.time
    return candidate.time + config.switch_mode_cost + config.switch_placement_cost * _placement_distance(
        previous.mode,
        candidate.mode,
    )


def candidate_arrival_with_switch(
    config: SelectionConfig,
    client_id: int,
    candidate: Candidate,
    previous: Candidate | None,
) -> float:
    _ = client_id
    base = candidate.time if candidate.pre_aggregation_time is None else candidate.pre_aggregation_time
    if previous is None or previous.mode in {"", "SKIP"} or candidate.mode == "SKIP":
        return base
    if previous.mode == candidate.mode:
        return base
    return base + config.switch_mode_cost + config.switch_placement_cost * _placement_distance(
        previous.mode,
        candidate.mode,
    )


def _placement_distance(previous_mode: str, mode: str) -> float:
    prev = _placement_vector(previous_mode)
    curr = _placement_vector(mode)
    return sum(abs(a - b) for a, b in zip(prev, curr)) / 2.0


@lru_cache(maxsize=None)
def _placement_vector(mode: str) -> tuple[float, float, float]:
    spec = MODE_SPECS.get(mode)
    if spec is None:
        return (1.0, 0.0, 0.0)
    local = max(spec.local_work, 0.0)
    edge = max(spec.edge_work + spec.edge_cpu, 0.0)
    cloud = max(spec.cloud_work + spec.cloud_cpu, 0.0)
    total = max(local + edge + cloud, 1e-12)
    return (local / total, edge / total, cloud / total)


def _candidate_dp_event_counts(
    candidate: Candidate,
    config: SelectionConfig,
) -> tuple[int, int, int]:
    if config.mainline_fusion:
        return 0, 0, 0  # DP belongs to the final aggregate, never these links.
    feature_events = 0
    client_update_events = 0
    edge_update_events = 0
    spec = MODE_SPECS.get(candidate.mode)
    edge_loops = max(int(spec.E_edge_loops if spec is not None else 1), 1)
    for link_id, obj, count, privacy_eligible in _mode_link_transmissions(
        candidate.mode,
        max(int(config.L_block_cycles), 1),
        edge_loops,
    ):
        if not privacy_eligible:
            continue
        mechanism = candidate_link_mechanism(
            candidate,
            link_id,
            fallback_object=obj,
        )
        if not mechanism_uses_dp(mechanism):
            continue
        if obj != "upd":
            feature_events += count
        elif candidate.mode in EDGE_CLOUD_MODES and link_id == "E_C_upd":
            edge_update_events += count
        else:
            client_update_events += count
    return feature_events, client_update_events, edge_update_events


def _omega_edge_group_key(
    candidate: Candidate,
    edge_id: int,
    components: _OmegaComponents,
) -> tuple[int, str, str] | None:
    if components.edge_bias <= 0.0 and components.edge_variance <= 0.0:
        return None
    return (
        int(edge_id),
        candidate.mode,
        candidate_link_mechanism(candidate, "E_C_upd"),
    )


def _cloud_update_coverage_ratio(
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    admitted_client_ids: tuple[int, ...] | list[int],
) -> float:
    """Fraction of represented samples whose selected update reaches the Cloud."""
    admitted = set(admitted_client_ids)
    total = sum(max(float(v), 0.0) for v in client_samples.values())
    if total <= 0.0:
        return 0.0
    cloud = 0.0
    for client_id, candidate in profile.items():
        if client_id in admitted and _candidate_reaches_cloud(candidate):
            cloud += max(float(client_samples.get(client_id, 0.0)), 0.0)
    return cloud / total


def _candidate_uses_secure_aggregate_update_dp_for_selection(candidate: Candidate) -> bool:
    """Mirror the runtime aggregate-boundary update-DP eligibility.

    LIIC and LIIEIIIC use aggregate-boundary DP whenever their Cloud update uses
    DP (with or without HE).  Other modes only use this path when DP and HE are
    jointly selected, preserving the legacy secure-aggregate packet contract.
    """
    if candidate.dp_execution_plan == "cloud_packet":
        return False
    mechanism = candidate_link_mechanism(candidate, "L_C_upd")
    if candidate.mode == "LIIEIIIC":
        mechanism = candidate_link_mechanism(candidate, "E_C_upd")
    if candidate.mode in {"LIIC", "LIIEIIIC"}:
        return mechanism_uses_dp(mechanism)
    return mechanism_uses_dp(mechanism) and mechanism_uses_he(mechanism)


def _candidate_uses_edge_local_exact_update_dp(candidate: Candidate) -> bool:
    """Explicitly planned LIIE DP-only secure aggregate, never worker-DP input."""
    return (
        candidate.mode == "LIIE"
        and candidate.dp_execution_plan == "aggregate"
        and candidate_link_mechanism(candidate, "L_E_upd") == "dp"
    )


def _candidate_has_update_dp(candidate: Candidate) -> bool:
    spec = MODE_SPECS.get(candidate.mode)
    edge_loops = max(int(spec.E_edge_loops if spec is not None else 1), 1)
    for link_id, obj, _count, privacy_eligible in _mode_link_transmissions(
        candidate.mode,
        1,
        edge_loops,
    ):
        if obj != "upd" or not privacy_eligible:
            continue
        if mechanism_uses_dp(candidate_link_mechanism(candidate, link_id, fallback_object="upd")):
            return True
    return False


def _global_update_clip_perturbation_cost(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    admitted_client_ids: tuple[int, ...] | list[int],
) -> float:
    """Worst-case coherent update-clipping distortion for admitted DP updates.

    ``omega_update_clip_excess_sq`` is already an update-space squared norm.
    Weighting the clipped client mass before squaring keeps this term in the
    same units as the fusion bound and the effective Gaussian second moment.
    """
    excess_sq = max(float(config.omega_update_clip_excess_sq), 0.0)
    if excess_sq <= 0.0:
        return 0.0
    admitted = set(admitted_client_ids)
    total_mass = sum(
        max(float(client_samples.get(client_id, 0.0)), 0.0)
        for client_id in admitted
        if client_id in profile
    )
    if total_mass <= 0.0:
        return 0.0
    clipped_mass = sum(
        max(float(client_samples.get(client_id, 0.0)), 0.0)
        for client_id in admitted
        if client_id in profile and _candidate_has_update_dp(profile[client_id])
    )
    clipped_fraction = clipped_mass / total_mass
    return excess_sq * clipped_fraction * clipped_fraction


def _global_dp_perturbation_cost(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    admitted_client_ids: tuple[int, ...] | list[int] | None = None,
) -> float:
    """Effective update-noise second moment for J_learn^v2.

    The execution layer now gives the full-local II family aggregate-boundary
    DP semantics.  Consequently the utility cost must be charged once at the
    released aggregate sensitivity, not once per high-dimensional client
    packet.  Split/local-packet DP candidates retain the previous independent
    packet accounting.
    """
    admitted = set(profile) if admitted_client_ids is None else set(admitted_client_ids)
    dim = max(float(config.omega_update_dimension), 0.0)
    clip = max(float(config.omega_update_clip_norm), 0.0)
    if dim <= 0.0 or clip <= 0.0 or not admitted:
        return 0.0

    privacy = resolved_privacy_parameters(config)
    default_sigma = max(float(privacy["update_noise_multiplier"]), 0.0)
    sensitivity_unweighted = 2.0 * clip

    # Cloud-reaching represented mass defines the final aggregate weights.
    cloud_clients = [
        client_id
        for client_id in admitted
        if client_id in profile and _candidate_reaches_cloud(profile[client_id])
    ]
    cloud_mass = sum(max(float(client_samples.get(client_id, 0.0)), 0.0) for client_id in cloud_clients)
    cloud_weights = {
        client_id: max(float(client_samples.get(client_id, 0.0)), 0.0) / cloud_mass
        for client_id in cloud_clients
    } if cloud_mass > 0.0 else {}

    cost = 0.0

    # Exact-target aggregate DP at the Cloud boundary.  Runtime calibrates one
    # release from the maximum effective client weight and maximum participating
    # sigma, then distributes noise shares across direct/edge packets.
    secure_cloud_clients = [
        client_id
        for client_id in cloud_clients
        if _candidate_uses_secure_aggregate_update_dp_for_selection(profile[client_id])
    ]
    if secure_cloud_clients:
        max_weight = max(cloud_weights[client_id] for client_id in secure_cloud_clients)
        sigma = max(
            max(float(profile[client_id].update_noise_multiplier or default_sigma), 0.0)
            for client_id in secure_cloud_clients
        )
        cost += dim * (sigma * sensitivity_unweighted * max_weight) ** 2

    # An explicitly selected independent E->C packet is formed *after* its
    # Edge group has averaged individually clipped client contributions.
    # Its sensitivity is 2 C max_i w_(i|e), not the direct-client 2 C.
    packet_group_clients: set[int] = set()
    edge_packet_groups: dict[tuple[int, str, str], list[int]] = {}
    for cid in cloud_clients:
        candidate = profile[cid]
        if candidate.dp_execution_plan != "cloud_packet" or candidate.mode not in EDGE_CLOUD_MODES:
            continue
        key = (int(client_edges.get(cid, -1)), candidate.mode,
               candidate_link_mechanism(candidate, "E_C_upd"))
        edge_packet_groups.setdefault(key, []).append(cid)
    for members in edge_packet_groups.values():
        total = sum(max(float(client_samples.get(cid, 0)), 0.0) for cid in members)
        if total <= 0:
            continue
        cloud_weight = sum(cloud_weights[cid] for cid in members)
        max_within_weight = max(max(float(client_samples.get(cid, 0)), 0.0) / total
                                for cid in members)
        group_sigma = max(float(profile[cid].update_noise_multiplier or default_sigma)
                          for cid in members)
        cost += dim * (cloud_weight * group_sigma * sensitivity_unweighted * max_within_weight) ** 2
        packet_group_clients.update(members)

    # Cloud-bound DP that is not on the aggregate-boundary path keeps legacy
    # independent packet noise.  This is primarily the split/offload family.
    for client_id in cloud_clients:
        candidate = profile[client_id]
        if client_id in packet_group_clients:
            continue
        if _candidate_uses_secure_aggregate_update_dp_for_selection(candidate):
            continue
        weight = cloud_weights[client_id]
        spec = MODE_SPECS.get(candidate.mode)
        edge_loops = max(int(spec.E_edge_loops if spec is not None else 1), 1)
        cloud_update_dp = False
        for link_id, obj, _count, privacy_eligible in _mode_link_transmissions(candidate.mode, 1, edge_loops):
            if obj != "upd" or not privacy_eligible:
                continue
            if link_id.endswith("_C_upd") and mechanism_uses_dp(
                candidate_link_mechanism(candidate, link_id, fallback_object="upd")
            ):
                cloud_update_dp = True
                break
        if cloud_update_dp:
            sigma = max(float(candidate.update_noise_multiplier or default_sigma), 0.0)
            cost += dim * (weight * sigma * sensitivity_unweighted) ** 2

    # LIIE publishes either individual DP packets or one explicitly planned
    # Edge SecAgg + aggregate-DP release. Edge-group eligibility is verified
    # before dispatch and again before publishing the release.
    # LIIE publishes locally DP-protected individual L->E packets.  Edge
    # averaging is only post-processing, so independently sampled packet noise
    # combines in variance (sum of squared *actual edge weights*).  Include
    # every LIIE client in the denominator, even HE-only peers.
    edge_groups: dict[int, list[int]] = {}
    for client_id in admitted:
        candidate = profile.get(client_id)
        if candidate is not None and candidate.mode == "LIIE":
            edge_groups.setdefault(int(client_edges.get(client_id, -1)), []).append(client_id)
    admitted_mass = sum(
        max(float(client_samples.get(client_id, 0.0)), 0.0)
        for client_id in admitted if client_id in profile
    )
    for members in edge_groups.values():
        group_mass = sum(max(float(client_samples.get(cid, 0.0)), 0.0) for cid in members)
        if group_mass <= 0.0 or admitted_mass <= 0.0:
            continue
        aggregate_members = [
            cid for cid in members
            if _candidate_uses_edge_local_exact_update_dp(profile[cid])
        ]
        if aggregate_members:
            if len(aggregate_members) != len(members) or len(members) < 2:
                raise ValueError("Inconsistent LIIE aggregate DP cohort in profile")
            max_weight = max(
                max(float(client_samples.get(cid, 0.0)), 0.0) / group_mass
                for cid in members
            )
            group_sigma = max(
                max(float(profile[cid].update_noise_multiplier or default_sigma), 0.0)
                for cid in members
            )
            group_cost = dim * (group_sigma * sensitivity_unweighted * max_weight) ** 2
        else:
            group_cost = 0.0
            for cid in members:
                candidate = profile[cid]
                if not mechanism_uses_dp(candidate_link_mechanism(candidate, "L_E_upd")):
                    continue
                weight = max(float(client_samples.get(cid, 0.0)), 0.0) / group_mass
                sigma = max(float(candidate.update_noise_multiplier or default_sigma), 0.0)
                group_cost += dim * (weight * sigma * sensitivity_unweighted) ** 2
        cost += (group_mass / admitted_mass) * group_cost

    return float(cost)


def _profile_omega_stats(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    candidate_omega_components: dict[int, _OmegaComponents] | None = None,
) -> _ProfileOmegaStats:
    total_samples = sum(float(client_samples.get(client_id, 1.0)) for client_id in profile)
    if total_samples <= 0:
        total_samples = float(max(len(profile), 1))

    edge_total_samples: dict[int, float] = {}
    cloud_samples_by_edge: dict[int, float] = {}
    client_bias_by_edge: dict[int, float] = {}
    client_variance_by_edge: dict[int, float] = {}
    edge_group_samples: dict[tuple[int, str, str], float] = {}
    edge_group_components: dict[tuple[int, str, str], tuple[float, float]] = {}
    for client_id, candidate in profile.items():
        samples = max(float(client_samples.get(client_id, 1.0)), 0.0)
        edge_id = int(client_edges.get(client_id, -1))
        edge_total_samples[edge_id] = edge_total_samples.get(edge_id, 0.0) + samples
        if not _candidate_reaches_cloud(candidate):
            continue
        components = (
            candidate_omega_components[id(candidate)]
            if candidate_omega_components is not None
            else _local_omega_components(candidate, config)
        )
        cloud_samples_by_edge[edge_id] = (
            cloud_samples_by_edge.get(edge_id, 0.0) + samples
        )
        client_bias_by_edge[edge_id] = (
            client_bias_by_edge.get(edge_id, 0.0)
            + samples * components.client_bias
        )
        client_variance_by_edge[edge_id] = (
            client_variance_by_edge.get(edge_id, 0.0)
            + samples * samples * components.client_variance
        )
        group = _omega_edge_group_key(candidate, edge_id, components)
        if group is not None:
            edge_group_samples[group] = edge_group_samples.get(group, 0.0) + samples
            edge_group_components[group] = (
                components.edge_bias,
                components.edge_variance,
            )

    return _ProfileOmegaStats(
        total_samples=total_samples,
        client_samples={
            client_id: max(float(client_samples.get(client_id, 1.0)), 0.0)
            for client_id in profile
        },
        edge_total_samples=edge_total_samples,
        cloud_samples_by_edge=cloud_samples_by_edge,
        client_bias_by_edge=client_bias_by_edge,
        client_variance_by_edge=client_variance_by_edge,
        edge_group_samples=edge_group_samples,
        edge_group_components=edge_group_components,
    )


def _replace_profile_omega_stats(
    config: SelectionConfig,
    stats: _ProfileOmegaStats,
    client_id: int,
    old_candidate: Candidate,
    new_candidate: Candidate,
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    candidate_omega_components: dict[int, _OmegaComponents] | None = None,
) -> _ProfileOmegaStats:
    samples = max(float(client_samples.get(client_id, 1.0)), 0.0)
    edge_id = int(client_edges.get(client_id, -1))
    if candidate_omega_components is None:
        old_components = _local_omega_components(old_candidate, config)
        new_components = _local_omega_components(new_candidate, config)
    else:
        old_components = candidate_omega_components[id(old_candidate)]
        new_components = candidate_omega_components[id(new_candidate)]

    cloud_samples_by_edge = dict(stats.cloud_samples_by_edge)
    client_bias_by_edge = dict(stats.client_bias_by_edge)
    client_variance_by_edge = dict(stats.client_variance_by_edge)
    edge_group_samples = dict(stats.edge_group_samples)
    edge_group_components = dict(stats.edge_group_components)

    def apply_candidate(candidate: Candidate, components: _OmegaComponents, sign: float) -> None:
        if not _candidate_reaches_cloud(candidate):
            return
        cloud_samples_by_edge[edge_id] = (
            cloud_samples_by_edge.get(edge_id, 0.0) + sign * samples
        )
        client_bias_by_edge[edge_id] = (
            client_bias_by_edge.get(edge_id, 0.0)
            + sign * samples * components.client_bias
        )
        client_variance_by_edge[edge_id] = (
            client_variance_by_edge.get(edge_id, 0.0)
            + sign * samples * samples * components.client_variance
        )
        group = _omega_edge_group_key(candidate, edge_id, components)
        if group is not None:
            edge_group_samples[group] = edge_group_samples.get(group, 0.0) + sign * samples
            if sign > 0.0:
                edge_group_components[group] = (
                    components.edge_bias,
                    components.edge_variance,
                )
            if edge_group_samples[group] <= 1e-12:
                edge_group_samples.pop(group, None)
                edge_group_components.pop(group, None)

    apply_candidate(old_candidate, old_components, -1.0)
    apply_candidate(new_candidate, new_components, 1.0)

    return _ProfileOmegaStats(
        total_samples=stats.total_samples,
        client_samples=stats.client_samples,
        edge_total_samples=stats.edge_total_samples,
        cloud_samples_by_edge=cloud_samples_by_edge,
        client_bias_by_edge=client_bias_by_edge,
        client_variance_by_edge=client_variance_by_edge,
        edge_group_samples=edge_group_samples,
        edge_group_components=edge_group_components,
    )


def _omega_from_profile_stats(
    config: SelectionConfig,
    stats: _ProfileOmegaStats,
    profile: dict[int, Candidate],
    client_edges: dict[int, int],
    admitted_client_ids: list[int] | tuple[int, ...],
) -> tuple[float, float]:
    if config.aggregation_fraction < 1.0:
        return _global_omega_proxy_from_admitted(
            config,
            profile,
            admitted_client_ids,
            stats,
            client_edges,
            client_samples=stats.client_samples,
        )

    active_edges = {
        edge_id
        for edge_id, samples in stats.cloud_samples_by_edge.items()
        if samples > 1e-12
    }
    active_edge_mass = sum(
        stats.edge_total_samples.get(edge_id, 0.0)
        for edge_id in active_edges
    )
    weighted_local = 0.0
    if active_edge_mass > 0.0:
        for edge_id in active_edges:
            admitted_samples = stats.cloud_samples_by_edge[edge_id]
            edge_weight = stats.edge_total_samples.get(edge_id, 0.0) / active_edge_mass
            weighted_local += (
                edge_weight
                * stats.client_bias_by_edge.get(edge_id, 0.0)
                / admitted_samples
            )
            weighted_local += (
                edge_weight * edge_weight
                * stats.client_variance_by_edge.get(edge_id, 0.0)
                / (admitted_samples * admitted_samples)
            )
        for group, group_samples in stats.edge_group_samples.items():
            edge_id = group[0]
            admitted_samples = stats.cloud_samples_by_edge.get(edge_id, 0.0)
            if admitted_samples <= 0.0:
                continue
            edge_weight = stats.edge_total_samples.get(edge_id, 0.0) / active_edge_mass
            group_weight = edge_weight * group_samples / admitted_samples
            group_bias, group_variance = stats.edge_group_components[group]
            weighted_local += (
                group_weight * group_bias
                + group_weight * group_weight * group_variance
            )
    cloud_samples = sum(stats.cloud_samples_by_edge.values())
    cloud_fusion_ratio = cloud_samples / max(stats.total_samples, 1e-12)
    weighted_local += _fusion_aggregate_noise_cost(
        config, profile, client_edges, admitted_client_ids, stats.client_samples
    )
    return (
        weighted_local
        + config.cloud_fusion_xi / (cloud_fusion_ratio + config.cloud_fusion_eps),
        cloud_fusion_ratio,
    )


def _global_omega_proxy(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    admitted_client_ids: list[int] | tuple[int, ...] | None = None,
) -> tuple[float, float]:
    stats = _profile_omega_stats(config, profile, client_samples, client_edges)
    admitted = tuple(profile) if admitted_client_ids is None else admitted_client_ids
    return _global_omega_proxy_from_admitted(
        config,
        profile,
        admitted,
        stats,
        client_edges,
        client_samples=client_samples,
    )


def _global_omega_proxy_from_admitted(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    admitted_client_ids: list[int] | tuple[int, ...],
    stats: _ProfileOmegaStats,
    client_edges: dict[int, int],
    *,
    client_samples: dict[int, float] | None = None,
) -> tuple[float, float]:
    if client_samples is None:
        client_samples = stats.client_samples
    weights = _cloud_client_aggregation_weights(
        profile,
        client_samples,
        client_edges,
        admitted_client_ids,
    )
    weighted_local = 0.0
    edge_groups: dict[tuple[int, str, str], tuple[float, float, float]] = {}
    for client_id, weight in weights.items():
        candidate = profile[client_id]
        components = _local_omega_components(candidate, config)
        weighted_local += (
            weight * components.client_bias
            + weight * weight * components.client_variance
        )
        edge_id = int(client_edges.get(client_id, -1))
        group = _omega_edge_group_key(candidate, edge_id, components)
        if group is not None:
            prior_weight, _bias, _variance = edge_groups.get(
                group,
                (0.0, components.edge_bias, components.edge_variance),
            )
            edge_groups[group] = (
                prior_weight + weight,
                components.edge_bias,
                components.edge_variance,
            )
    for group_weight, group_bias, group_variance in edge_groups.values():
        weighted_local += (
            group_weight * group_bias
            + group_weight * group_weight * group_variance
        )
    admitted_cloud_samples = sum(
        max(float(client_samples.get(client_id, 1.0)), 0.0)
        for client_id in weights
    )
    cloud_fusion_ratio = admitted_cloud_samples / max(stats.total_samples, 1e-12)
    weighted_local += _fusion_aggregate_noise_cost(
        config, profile, client_edges, admitted_client_ids, client_samples
    )
    return (
        weighted_local
        + config.cloud_fusion_xi / (cloud_fusion_ratio + config.cloud_fusion_eps),
        cloud_fusion_ratio,
    )


def _cloud_client_aggregation_weights(
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    admitted_client_ids: list[int] | tuple[int, ...],
) -> dict[int, float]:
    """Normalize Cloud-reaching admitted clients by represented sample mass."""
    del client_edges
    admitted_cloud = [
        client_id
        for client_id in admitted_client_ids
        if client_id in profile and _candidate_reaches_cloud(profile[client_id])
    ]
    represented_mass = {
        client_id: max(float(client_samples.get(client_id, 1.0)), 0.0)
        for client_id in admitted_cloud
    }
    total_mass = sum(represented_mass.values())
    if total_mass <= 0.0:
        return {}
    return {
        client_id: mass / total_mass
        for client_id, mass in represented_mass.items()
    }


def _fusion_aggregate_noise_cost(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_edges: dict[int, int],
    admitted_client_ids: list[int] | tuple[int, ...],
    client_samples: dict[int, float],
) -> float:
    if not config.mainline_fusion or not admitted_client_ids:
        return 0.0
    weights = _cloud_client_aggregation_weights(
        profile, client_samples, client_edges, admitted_client_ids
    )
    if not weights:
        return 0.0
    max_weight = max(weights.values())
    privacy_parameters = resolved_privacy_parameters(config)
    sigma = float(privacy_parameters["update_noise_multiplier"])
    sensitivity = 2.0 * max(float(config.omega_update_clip_norm), 1e-12) * max_weight
    eta = max(float(config.omega_learning_rate), 1e-12)
    local_cycles = max(float(config.L_block_cycles), 1.0)
    variance_scale = (
        max(float(config.omega_smoothness), 1e-12)
        * eta
        * local_cycles
        / max(float(config.omega_mu), 1e-12)
    )
    gradient_variance = (
        sigma * sigma * sensitivity * sensitivity * float(config.omega_update_dimension)
        / (eta * eta * local_cycles * local_cycles)
    )
    return variance_scale * gradient_variance


def _aggregation_sizes(
    profile: dict[int, Candidate],
    client_edges: dict[int, int],
    admitted_client_ids: list[int] | tuple[int, ...] | None = None,
) -> dict[int, int]:
    admitted = set(profile) if admitted_client_ids is None else set(admitted_client_ids)
    cloud_count = sum(
        1
        for client_id, item in profile.items()
        if client_id in admitted and _candidate_reaches_cloud(item)
    )
    edge_counts: dict[int, int] = {}
    for client_id, candidate in profile.items():
        if client_id not in admitted or _candidate_reaches_cloud(candidate):
            continue
        edge_id = client_edges.get(client_id, -1)
        edge_counts[edge_id] = edge_counts.get(edge_id, 0) + 1
    return {
        client_id: max(cloud_count if _candidate_reaches_cloud(candidate) else edge_counts.get(client_edges.get(client_id, -1), 1), 1)
        for client_id, candidate in profile.items()
    }


def _local_proxy_aggregation_size_hint(
    candidate: Candidate,
    config: SelectionConfig,
) -> int:
    """Cheap cohort-size hint for aggregate-DP-aware search ordering.

    The exact profile objective recomputes aggregate sensitivity from the
    selected client masses.  Before a profile exists, the local seed/neighbor
    heuristic only needs a scale-consistent estimate so exact-target II-family
    candidates are not ranked with legacy per-client packet DP cost.
    """
    if _candidate_uses_secure_aggregate_update_dp_for_selection(candidate):
        return max(int(config.num_clients), 1)
    if candidate.mode == "LIIE" and mechanism_uses_dp(
        candidate_link_mechanism(candidate, "L_E_upd")
    ):
        # Locally DP-protected packets average with 1/K (variance) rather
        # than 1/K^2 (one DP-protected aggregate) in the search proxy.
        edges = max(int(config.num_edges), 1)
        return max(int(math.ceil(float(config.num_clients) / float(edges))), 1)
    return 1


def _local_omega_proxy(
    candidate: Candidate,
    aggregation_size: int | None = None,
    config: SelectionConfig | None = None,
) -> float:
    config = config or SelectionConfig()
    components = _local_omega_components(candidate, config)
    size = max(
        float(
            _local_proxy_aggregation_size_hint(candidate, config)
            if aggregation_size is None
            else aggregation_size
        ),
        1.0,
    )

    # Legacy independent client-packet noise averages as 1/K in the old local
    # convergence proxy. Exact-target aggregate DP is different: execution
    # calibrates the released aggregate with sensitivity 2 C / K (equal-mass
    # search hint), so its second moment scales as 1/K^2.  Apply that extra
    # attenuation only to the exact aggregate paths; split/local-packet DP keeps
    # the previous heuristic. Exact profile evaluation still uses real masses.
    if (
        _candidate_uses_secure_aggregate_update_dp_for_selection(candidate)
        or _candidate_uses_edge_local_exact_update_dp(candidate)
    ):
        return (
            components.client_bias
            + components.client_variance / (size * size)
            + components.edge_bias
            + components.edge_variance / (size * size)
        )

    return (
        components.client_bias
        + components.client_variance / size
        + components.edge_bias
        + components.edge_variance
    )


@lru_cache(maxsize=4096)
def _cached_local_omega_components(
    feature_dp_events: int,
    client_update_dp_events: int,
    edge_update_dp_events: int,
    feature_clip_excess_sq: float,
    update_clip_excess_sq: float,
    update_noise_multiplier: float,
    config: SelectionConfig,
) -> _OmegaComponents:
    privacy_parameters = resolved_privacy_parameters(config)
    mu = max(float(config.omega_mu), 1e-12)
    smoothness = max(float(config.omega_smoothness), 1e-12)
    eta = max(float(config.omega_learning_rate), 1e-12)
    local_cycles = max(float(config.L_block_cycles), 1.0)

    feature_clip_bias = 0.0
    feature_noise_bias = 0.0
    feature_loss_inflation = 0.0
    if feature_dp_events > 0:
        feature_clip_bias = (
            config.omega_feature_jacobian_norm ** 2
            * config.omega_feature_lipschitz ** 2
            * max(float(feature_clip_excess_sq), 0.0)
        )
        delta_z = 2.0 * max(config.omega_feature_clip_norm, 1e-12)
        feature_sigma_sq = float(privacy_parameters["feature_noise_multiplier"]) ** 2
        feature_noise_bias = (
            config.omega_feature_jacobian_norm ** 2
            * config.omega_feature_backward_bias_sq
        )
        feature_loss_inflation = (
            feature_sigma_sq
            * delta_z ** 2
            * config.omega_feature_clf_pairwise_spread
            / 4.0
        )

    update_clip_bias = max(float(update_clip_excess_sq), 0.0) / (
        eta ** 2 * local_cycles ** 2
    )
    update_sigma_sq = float(update_noise_multiplier) ** 2
    update_variance = (
        update_sigma_sq
        * (2.0 * config.omega_update_clip_norm) ** 2
        * config.omega_update_dimension
        / (eta ** 2 * local_cycles ** 2)
    )

    feature_bias = float(feature_dp_events) * (
        feature_clip_bias
        + feature_noise_bias
        + smoothness * feature_loss_inflation
    )
    variance_scale = smoothness * eta * local_cycles / mu
    return _OmegaComponents(
        client_bias=(3.0 / (2.0 * mu)) * (
            feature_bias
            + float(client_update_dp_events) * update_clip_bias
        ),
        client_variance=variance_scale * (
            config.omega_local_variance
            + float(client_update_dp_events) * update_variance
        ),
        edge_bias=(3.0 / (2.0 * mu))
        * float(edge_update_dp_events)
        * update_clip_bias,
        edge_variance=variance_scale
        * float(edge_update_dp_events)
        * update_variance,
    )


def _local_omega_components(
    candidate: Candidate,
    config: SelectionConfig,
) -> _OmegaComponents:
    feature_events, client_update_events, edge_update_events = (
        _candidate_dp_event_counts(candidate, config)
    )
    components = _cached_local_omega_components(
        feature_events,
        client_update_events,
        edge_update_events,
        _candidate_feature_clip_excess_sq(candidate, config),
        config.omega_update_clip_excess_sq,
        float(
            candidate.update_noise_multiplier
            if candidate.update_noise_multiplier is not None
            else resolved_privacy_parameters(config)["update_noise_multiplier"]
        ),
        config,
    )
    if config.mainline_fusion and candidate.global_release_required:
        eta = max(float(config.omega_learning_rate), 1e-12)
        local_cycles = max(float(config.L_block_cycles), 1.0)
        clip_bias = max(float(config.omega_update_clip_excess_sq), 0.0) / (
            eta * eta * local_cycles * local_cycles
        )
        return _OmegaComponents(
            client_bias=(
                components.client_bias
                + (3.0 / (2.0 * max(float(config.omega_mu), 1e-12))) * clip_bias
            ),
            client_variance=components.client_variance,
            edge_bias=0.0,
            edge_variance=0.0,
        )
    return components


def _candidate_feature_clip_excess_sq(
    candidate: Candidate,
    config: SelectionConfig,
) -> float:
    profiled = getattr(candidate, "omega_feature_clip_excess_sq", None)
    if profiled is None:
        return float(config.omega_feature_clip_excess_sq)
    return max(float(profiled), 0.0)


def _bounded_search_beam(
    evaluations: list[ProfileEvaluation],
    *,
    reference: list[ProfileEvaluation],
    limit: int,
    norm_eps: float,
) -> list[ProfileEvaluation]:
    if limit <= 0 or not evaluations:
        return []
    candidates = _unique_evaluations(evaluations)
    bounds_source = reference + candidates
    t_values = [
        _pareto_objective_value(item.system_latency)
        for item in bounds_source
    ]
    o_values = [
        _pareto_objective_value(item.system_omega)
        for item in bounds_source
    ]
    t_min, t_max = min(t_values), max(t_values)
    o_min, o_max = min(o_values), max(o_values)
    use_fusion = any(item.fusion_objective_enabled for item in bounds_source)
    if use_fusion:
        f_values = [
            _pareto_objective_value(item.fusion_distortion)
            for item in bounds_source
        ]
        f_min, f_max = min(f_values), max(f_values)
    else:
        f_min = f_max = 0.0

    def beam_key(item: ProfileEvaluation) -> tuple:
        stable_latency = _pareto_objective_value(
            item.system_latency
        )
        stable_learning = _pareto_objective_value(
            item.system_omega
        )
        stable_fusion = _pareto_objective_value(
            item.fusion_distortion
        )

        values = [
            _safe_norm_eps(
                stable_latency,
                t_min,
                t_max,
                norm_eps,
            ),
            _safe_norm_eps(
                stable_learning,
                o_min,
                o_max,
                norm_eps,
            ),
        ]
        if use_fusion:
            values.append(
                _safe_norm_eps(
                    stable_fusion,
                    f_min,
                    f_max,
                    norm_eps,
                )
            )
        return (
            max(values),
            stable_latency,
            stable_learning,
            stable_fusion,
            _evaluation_key(item),
        )

    return sorted(candidates, key=beam_key)[:limit]


def _pareto_archive(
    evaluations: list[ProfileEvaluation],
    limit: int,
    profiler: dict[str, Any] | None = None,
) -> list[ProfileEvaluation]:
    started_at = time.perf_counter()
    _perf_add(profiler, "pareto_archive_calls")
    unique = _unique_evaluations(evaluations)
    if not unique:
        _perf_add(profiler, "pareto_archive_sec", time.perf_counter() - started_at)
        return []

    frontier: list[ProfileEvaluation] = []
    for candidate in unique:
        dominated = False
        for other in unique:
            if other is candidate:
                continue
            _perf_add(profiler, "dominance_compare_count")
            if _profile_objectives_dominate(other, candidate):
                dominated = True
                break
        if not dominated:
            frontier.append(candidate)
    frontier = frontier or unique
    if len(frontier) <= max(int(limit), 1):
        _perf_add(profiler, "pareto_archive_sec", time.perf_counter() - started_at)
        return frontier

    # Deterministic normalized Tchebycheff truncation of a potentially large frontier.
    t_values = [
        _pareto_objective_value(item.system_latency)
        for item in frontier
    ]
    o_values = [
        _pareto_objective_value(item.system_omega)
        for item in frontier
    ]
    t_min, t_max = min(t_values), max(t_values)
    o_min, o_max = min(o_values), max(o_values)
    use_fusion = any(item.fusion_objective_enabled for item in frontier)
    if use_fusion:
        f_values = [
            _pareto_objective_value(item.fusion_distortion)
            for item in frontier
        ]
        f_min, f_max = min(f_values), max(f_values)
    else:
        f_min = f_max = 0.0

    def key(item: ProfileEvaluation) -> tuple:
        stable_latency = _pareto_objective_value(
            item.system_latency
        )
        stable_learning = _pareto_objective_value(
            item.system_omega
        )
        stable_fusion = _pareto_objective_value(
            item.fusion_distortion
        )

        vals = [
            _safe_norm_eps(
                stable_latency,
                t_min,
                t_max,
                1e-9,
            ),
            _safe_norm_eps(
                stable_learning,
                o_min,
                o_max,
                1e-9,
            ),
        ]
        if use_fusion:
            vals.append(
                _safe_norm_eps(
                    stable_fusion,
                    f_min,
                    f_max,
                    1e-9,
                )
            )
        return (
            max(vals),
            stable_latency,
            stable_learning,
            stable_fusion,
            repr(_evaluation_key(item)),
        )

    result = sorted(frontier, key=key)[:max(int(limit), 1)]
    _perf_add(profiler, "pareto_archive_sec", time.perf_counter() - started_at)
    return result


def _latency_archive(
    evaluations: list[ProfileEvaluation],
    limit: int,
) -> list[ProfileEvaluation]:
    unique = _unique_evaluations(evaluations)
    return sorted(
        unique,
        key=lambda item: (item.system_latency, _evaluation_key(item)),
    )[:max(int(limit), 1)]


def _choose_tchebycheff(archive: list[ProfileEvaluation], norm_eps: float) -> ProfileEvaluation:
    if not archive:
        return ProfileEvaluation({}, 0.0, 0.0, 0.0, ())
    t_values = [
        _pareto_objective_value(item.system_latency)
        for item in archive
    ]
    o_values = [
        _pareto_objective_value(item.system_omega)
        for item in archive
    ]
    t_min, t_max = min(t_values), max(t_values)
    o_min, o_max = min(o_values), max(o_values)
    use_fusion = any(item.fusion_objective_enabled for item in archive)
    if use_fusion:
        f_values = [
            _pareto_objective_value(item.fusion_distortion)
            for item in archive
        ]
        f_min, f_max = min(f_values), max(f_values)
    else:
        f_min = f_max = 0.0

    def distance(item: ProfileEvaluation) -> tuple[float, float, float, float, str]:
        stable_latency = _pareto_objective_value(
            item.system_latency
        )
        stable_learning = _pareto_objective_value(
            item.system_omega
        )
        stable_fusion = _pareto_objective_value(
            item.fusion_distortion
        )

        values = [
            _safe_norm_eps(
                stable_latency,
                t_min,
                t_max,
                norm_eps,
            ),
            _safe_norm_eps(
                stable_learning,
                o_min,
                o_max,
                norm_eps,
            ),
        ]
        if use_fusion:
            values.append(
                _safe_norm_eps(
                    stable_fusion,
                    f_min,
                    f_max,
                    norm_eps,
                )
            )
        return (
            max(abs(value) for value in values),
            stable_latency,
            stable_learning,
            stable_fusion,
            repr(_evaluation_key(item)),
        )

    return min(archive, key=distance)


def _safe_norm_eps(value: float, low: float, high: float, eps: float) -> float:
    return (value - low) / (high - low + eps)


def _profile_key(profile: dict[int, Candidate]) -> tuple:
    return tuple(
        (client_id, *_candidate_key(candidate))
        for client_id, candidate in sorted(profile.items())
    )


def _evaluation_key(evaluation: ProfileEvaluation) -> tuple:
    return evaluation.profile_signature or _profile_key(evaluation.profile)


def _candidate_key(candidate: Candidate) -> tuple:
    base = (candidate.mode, tuple(sorted((candidate.link_mechanisms or candidate.mechanisms).items())))
    # Preserve the legacy two-field keys in tests/old checkpoints, but keep
    # opt-in publication choices distinct when present.
    return base if candidate.dp_execution_plan == "independent" else (*base, candidate.dp_execution_plan)


def _dedupe_candidates(candidates: list[Candidate]) -> list[Candidate]:
    seen: set[tuple] = set()
    unique: list[Candidate] = []
    for candidate in candidates:
        key = _candidate_key(candidate)
        if key in seen:
            continue
        seen.add(key)
        unique.append(candidate)
    return unique


def _unique_profiles(profiles: list[dict[int, Candidate]]) -> list[dict[int, Candidate]]:
    seen: set[tuple] = set()
    unique = []
    for profile in profiles:
        key = _profile_key(profile)
        if key in seen:
            continue
        seen.add(key)
        unique.append(profile)
    return unique


def _unique_evaluations(evaluations: list[ProfileEvaluation]) -> list[ProfileEvaluation]:
    seen: set[tuple] = set()
    unique = []
    for item in evaluations:
        key = _evaluation_key(item)
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def _sensitivity_weighted_ideal(
    candidates: list[Candidate],
    mode_bonus: dict[str, float] | None,
    sensitivity: float,
) -> Candidate:
    """Select from Pareto frontier using sensitivity-weighted ideal point.

    High sensitivity (->1): time weight dominates → selects faster modes.
    Low sensitivity (->0): accuracy weight dominates → selects more accurate modes.
    Mode bonus (UCB exploration) adjusts effective accuracy so that untried or
    historically promising modes are explored.
    """
    frontier = pareto_frontier(candidates)
    if not frontier:
        return skipped_candidate()
    if len(frontier) <= 1:
        return frontier[0]

    t_values = [c.time for c in frontier]
    a_eff = [c.accuracy + (mode_bonus.get(c.mode, 0.0) if mode_bonus else 0.0) for c in frontier]
    t_min, t_max = min(t_values), max(t_values)
    a_min, a_max = min(a_eff), max(a_eff)

    alpha = max(0.05, min(0.95, sensitivity))   # time weight
    beta_w = 1.0 - alpha                         # accuracy weight

    def weighted_distance(c: Candidate) -> float:
        t_norm = _safe_norm(c.time, t_min, t_max)
        eff_acc = c.accuracy + (mode_bonus.get(c.mode, 0.0) if mode_bonus else 0.0)
        a_norm = _safe_norm(a_max - eff_acc, 0.0, a_max - a_min)
        return math.sqrt(alpha * t_norm * t_norm + beta_w * a_norm * a_norm)

    return min(frontier, key=weighted_distance)


def skipped_candidate() -> Candidate:
    return Candidate(
        mode="SKIP",
        mechanisms={},
        time=0.0,
        accuracy=0.0,
        risk=0.0,
        epsilon_used=0.0,
        communication_volume=0.0,
        feasible_resource=True,
        feasible_privacy=True,
        feasible_risk=True,
        feasible_time=True,
        feasible_edge=True,
        feasible_cloud=True,
    )


def _safe_norm(value: float, low: float, high: float) -> float:
    span = high - low
    if abs(span) <= 1e-12:
        return 0.0
    return (value - low) / span


def _sample_mechanism_assignments(
    spec: ModeSpec,
    *,
    allow_he: bool = True,
    mainline_fusion: bool = False,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    """Return the fixed protection layout for the sample-level DP path.

    Sample privacy is provided by local DP-SGD plus explicit DP protection
    of split embeddings and label gradients. Model-update packet DP is
    intentionally disabled here to avoid applying a second client-level
    Gaussian mechanism to an already sample-DP-trained update.

    HE remains a confidentiality layer for updates crossing into Cloud.
    """
    all_transmissions = _mode_link_transmissions(
        spec.name,
        local_block_cycles=1,
        edge_loops=spec.E_edge_loops,
    )
    if mainline_fusion:
        all_transmissions = _fusion_link_transmissions(
            spec.name,
            1,
            spec.E_edge_loops,
        )

    transmissions = [
        event
        for event in all_transmissions
        if event[3]
    ]
    link_objects = {
        link_id: obj
        for link_id, obj, _count, _eligible in all_transmissions
    }

    assignment: dict[str, str] = {}

    for link_id, obj, _count, _privacy_eligible in transmissions:
        if obj in {"emb", "grad"}:
            mechanism = "dp"
        elif obj == "upd" and link_id.endswith("_C_upd"):
            if not allow_he:
                return []
            mechanism = "he3"
        else:
            mechanism = "none"

        assignment[link_id] = mechanism

    return [
        (
            _object_mechanism_summary(
                assignment,
                link_objects,
            ),
            assignment,
        )
    ]

def _mechanism_assignments(
    spec: ModeSpec,
    policy: str,
    allow_he: bool = True,
    allow_none: bool = False,
    trusted_edge_split_execution: bool = False,
    update_mechanism_options: tuple[str, ...] = ("dp", "he3", "dp_he3"),
    mainline_fusion: bool = False,
    privacy_requirement: ExposurePrivacyRequirement | None = None,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    all_transmissions = _mode_link_transmissions(
        spec.name,
        local_block_cycles=1,
        edge_loops=spec.E_edge_loops,
    )
    if mainline_fusion:
        all_transmissions = _fusion_link_transmissions(spec.name, 1, spec.E_edge_loops)
    transmissions = [event for event in all_transmissions if event[3]]
    links = list(dict.fromkeys(event[0] for event in transmissions))
    link_objects = {event[0]: event[1] for event in all_transmissions}

    if privacy_requirement is not None:
        def legal(link: str, mechanism: str) -> bool:
            return (
                (allow_he or not mechanism_uses_he(mechanism))
                and (
                    link not in privacy_requirement.plaintext_forbidden_links
                    or mechanism_uses_he(mechanism)
                )
                and (
                    link not in privacy_requirement.dp_required_links
                    or mechanism_uses_dp(mechanism)
                )
            )

        def dynamic_options(link: str) -> tuple[str, ...]:
            obj = link_objects[link]
            if obj == "upd":
                options = tuple(dict.fromkeys(("none",) + tuple(update_mechanism_options)))
            else:
                options = MECHANISMS_BY_OBJECT[obj]
            return tuple(mech for mech in options if legal(link, mech))

        fixed_mechanism = None
        if policy in {"no_protection", "fixed_splitfed_no_protection"}:
            fixed_mechanism = lambda link: "none"
        elif policy in {"fixed_dp", "fixed_splitfed_dp"}:
            fixed_mechanism = lambda link: (
                "dp" if "dp" in MECHANISMS_BY_OBJECT[link_objects[link]] else "none"
            )
        elif policy in {"fixed_he", "fixed_splitfed_trusted_edge"}:
            fixed_mechanism = lambda link: _prefer_he(link_objects[link])
        elif policy == "fixed_dp_he":
            fixed_mechanism = lambda link: (
                "dp_he3" if link_objects[link] == "upd" else "none"
            )

        if fixed_mechanism is not None:
            assignment = {link: fixed_mechanism(link) for link in links}
            if not all(legal(link, mech) for link, mech in assignment.items()):
                return []
            return [(_object_mechanism_summary(assignment, link_objects), assignment)]

        choices = [dynamic_options(link) for link in links]
        if any(not values for values in choices):
            return []
        link_assignments = [dict(zip(links, values)) for values in product(*choices)]
        return [
            (_object_mechanism_summary(item, link_objects), item)
            for item in link_assignments
        ]

    trusted_links = {
        link
        for link in links
        if trusted_edge_split_execution
        and link.startswith("L_E_")
    }
    if mainline_fusion:
        # The selector chooses execution topology only. DP is enforced by the
        # global release gate; HE is the fixed confidentiality layer whenever
        # an update crosses into the untrusted cloud.
        link_assignment = {}
        for link in links:
            obj = link_objects[link]
            if link.startswith("L_E_"):
                link_assignment[link] = "trusted"
            elif obj == "upd" and link.endswith("_C_upd"):
                link_assignment[link] = "he3"
            else:
                link_assignment[link] = "none"
        return [(_object_mechanism_summary(link_assignment, link_objects), link_assignment)]
    if policy in {"no_protection", "fixed_splitfed_no_protection"}:
        link_assignments = [{
            link: ("trusted" if link in trusted_links else "none")
            for link in links
        }]
        return [(_object_mechanism_summary(item, link_objects), item) for item in link_assignments]
    if policy == "fixed_splitfed_label_dp":
        raise ValueError(
            "Feature DP diagnostics are not part of the trusted end-edge threat model."
        )
    if policy == "fixed_splitfed_trusted_edge":
        link_assignments = [{
            link: (
                "trusted"
                if link_objects[link] in {"emb", "grad"}
                else "he3"
            )
            for link in links
        }]
        return [(_object_mechanism_summary(item, link_objects), item) for item in link_assignments]
    if policy == "fixed_splitfed_dp":
        link_assignments = [{
            link: (
                "trusted"
                if link in trusted_links
                else "dp"
                if link_objects[link] == "upd" and link.endswith("_C_upd")
                else "none"
            )
            for link in links
        }]
        return [(_object_mechanism_summary(item, link_objects), item) for item in link_assignments]
    if policy == "fixed_dp":
        link_assignments = [
            {
                link: (
                    "trusted"
                    if link in trusted_links
                    else "dp"
                    if (
                        trusted_edge_split_execution
                        and link_objects[link] == "upd"
                        and link.endswith("_C_upd")
                    )
                    else "dp"
                    if "dp" in MECHANISMS_BY_OBJECT[link_objects[link]]
                    else "none"
                )
                for link in links
            }
        ]
        return [(_object_mechanism_summary(item, link_objects), item) for item in link_assignments]
    if policy == "fixed_he":
        link_assignments = [{
            link: (
                "trusted" if link in trusted_links else _prefer_he(link_objects[link])
            )
            for link in links
        }]
        return [(_object_mechanism_summary(item, link_objects), item) for item in link_assignments]
    if policy == "fixed_dp_he":
        link_assignments = [{
            link: (
                "trusted"
                if link in trusted_links
                else "dp_he3"
                if link_objects[link] == "upd"
                else "none"
            )
            for link in links
        }]
        return [(_object_mechanism_summary(item, link_objects), item) for item in link_assignments]

    def choices_for_link(link: str) -> tuple[str, ...]:
        if link in trusted_links:
            return ("trusted",)
        if (
            trusted_edge_split_execution
            and link_objects[link] == "upd"
            and link.endswith("_C_upd")
        ):
            if not update_mechanism_options or any(
                value not in {"dp", "he3", "dp_he3"} for value in update_mechanism_options
            ):
                raise ValueError("Invalid update mechanism options")
            return tuple(value for value in update_mechanism_options if allow_he or not mechanism_uses_he(value))
        return tuple(
            mech
            for mech in MECHANISMS_BY_OBJECT[link_objects[link]]
            if (allow_he or not mechanism_uses_he(mech))
            and (
                allow_none
                or mech != "none"
                or MECHANISMS_BY_OBJECT[link_objects[link]] == ("none",)
            )
        )

    choices = [choices_for_link(link) for link in links]
    link_assignments = [dict(zip(links, values)) for values in product(*choices)]
    return [
        (_object_mechanism_summary(item, link_objects), item)
        for item in link_assignments
    ]


def _object_mechanism_summary(
    link_mechanisms: dict[str, str],
    link_objects: dict[str, str],
) -> dict[str, str]:
    by_object: dict[str, list[str]] = {}
    for link, mechanism in link_mechanisms.items():
        by_object.setdefault(link_objects[link], []).append(mechanism)
    summary = {
        obj: values[0] if len(set(values)) == 1 else "mixed"
        for obj, values in by_object.items()
    }
    return summary


def _profile_object_size(config: SelectionConfig, obj: str) -> float:
    if obj in {"emb", "emb_grad"}:
        return max(float(config.embedding_payload_mb), 0.0)
    if obj == "upd":
        return max(float(config.update_payload_mb), 0.0)
    return max(float(OBJECT_SIZES[obj]), 0.0)


def _mode_device_feasibility_metrics(
    *,
    config: SelectionConfig,
    local_work: float,
    local_memory: float,
    samples: int,
    memory_capacity_factor: float,
) -> tuple[float, float, bool, float, float, bool]:
    """Cheap mode-level resource/memory check used before candidate expansion."""
    sample_scale = samples / max(config.minibatch_reference_samples, 1e-9)
    local_load = local_work * sample_scale

    if config.assume_encoder_feasible:
        feasible_resource = True
    else:
        feasible_resource = local_load <= config.resource_limit

    memory_requirement = local_memory * (0.75 + 0.25 * sample_scale)
    memory_capacity = config.memory_limit * max(float(memory_capacity_factor), 1e-6)
    feasible_memory = memory_requirement <= memory_capacity + 1e-12

    return (
        sample_scale,
        local_load,
        feasible_resource,
        memory_requirement,
        memory_capacity,
        feasible_memory,
    )


def _estimate_candidate(
    *,
    config: SelectionConfig,
    mode: str,
    spec: ModeSpec,
    mechanisms: dict[str, str],
    link_mechanisms: dict[str, str] | None = None,
    client_id: int,
    edge_factor: float,
    compute_factor: float,
    samples: int,
    remaining_epsilon: float,
    round_idx: int,
    rng: random.Random,
    current_edge_load: float = 0.0,
    current_cloud_load: float = 0.0,
    memory_capacity_factor: float = 1.0,
    privacy_ledger: ClientPrivacyLedger | SamplePrivacyLedger | None = None,
    update_noise_multiplier: float | None = None,
    sample_embedding_noise_multiplier: float | None = None,
    sample_label_grad_noise_multiplier: float | None = None,
    sample_optimizer_noise_multiplier: float | None = None,
    fast_response_deadline: float | None = None,
) -> Candidate:
    L = _split_interaction_count(config, samples)
    E = spec.E_edge_loops

    (
        sample_scale,
        local_load,
        feasible_resource,
        memory_requirement,
        memory_capacity,
        feasible_memory,
    ) = _mode_device_feasibility_metrics(
        config=config,
        local_work=spec.local_work,
        local_memory=spec.local_memory,
        samples=samples,
        memory_capacity_factor=memory_capacity_factor,
    )

    # Per-block computation (total load spread across L cycles)
    local_time = (local_load / max(L, 1)) * compute_factor
    edge_time = spec.edge_work * edge_factor * 0.75
    # L block cycles: end and edge pipeline across L iterations
    # In split edge-target modes, end computes block i while edge processes block i-1
    if mode in ("LIE", "LIEIIC", "LIEIIIC"):
        pipe_cycle = max(local_time, edge_time)
        block_compute = L * pipe_cycle + min(local_time, edge_time)
    else:
        block_compute = L * (local_time + edge_time)

    cloud_time = spec.cloud_work * 0.55
    if mode == "LIC":
        # Direct client-cloud split executes the remote partition once per
        # split interaction, just as L-E split modes repeat edge work.
        cloud_time *= L
    link_events = (_fusion_link_transmissions(mode, L, E) if config.mainline_fusion
                   else _mode_link_transmissions(mode, L, E))
    actual_link_mechanisms = dict(link_mechanisms or {})
    compute_time = E * block_compute if E > 1 else block_compute
    link_metrics: list[dict[str, Any]] = []
    for link_id, obj, count, privacy_eligible in link_events:
        mechanism = actual_link_mechanisms[link_id] if privacy_eligible else "none"
        raw_size = _profile_object_size(config, obj)
        effective_size = raw_size * float(PRIVACY_ALPHA[mechanism])
        rate = _link_bandwidth(config, client_id, round_idx, link_id)
        base_delay = _link_base_latency(config, link_id)
        reference_size = max(float(OBJECT_SIZES[obj]), 1e-12)
        privacy_processing = (
            float(PRIVACY_BASE_TIME[mechanism]) * raw_size / reference_size
        )
        per_execution_time = effective_size / rate + base_delay + privacy_processing
        source, target = _link_route(link_id).split("_")
        link_metrics.append(
            {
                "link_id": link_id,
                "source": source,
                "target": target,
                "object": obj,
                "count": count,
                "privacy_eligible": privacy_eligible,
                "mechanism": mechanism,
                "raw_size": raw_size,
                "effective_size": effective_size,
                "rate": rate,
                "base_delay": base_delay,
                "privacy_processing_time": privacy_processing,
                "per_execution_time": per_execution_time,
                "total_link_time": count * per_execution_time,
                "total_effective_size": count * effective_size,
            }
        )

    communication_volume = sum(item["total_effective_size"] for item in link_metrics)

    # Privacy budget: mode-dependent cost (alpha × base_cost per DP event)
    # Separate from formal RDP accounting — used only for mode selection feasibility
    return_path_time = sum(
        item["total_link_time"]
        for item in link_metrics
        if item["link_id"].endswith("_final_return")
    )
    edge_to_cloud_time = (
        sum(
            item["total_link_time"]
            for item in link_metrics
            if item["link_id"] == "E_C_upd"
        )
        if mode in {"LIEIIIC", "LIIEIIIC"}
        else 0.0
    )
    pre_aggregation_link_time = sum(
        item["total_link_time"]
        for item in link_metrics
        if not item["link_id"].endswith("_final_return")
        and not (
            mode in {"LIEIIIC", "LIIEIIIC"}
            and item["link_id"] == "E_C_upd"
        )
    )
    if mode in {"LIEIIIC", "LIIEIIIC"}:
        first_aggregation_arrival_time = compute_time + pre_aggregation_link_time
        edge_to_cloud_time += cloud_time
    else:
        first_aggregation_arrival_time = compute_time + cloud_time + pre_aggregation_link_time

    def effective_payload(link_id: str, default: float = 0.0) -> float:
        return next(
            (
                float(item["effective_size"])
                for item in link_metrics
                if item["link_id"] == link_id
            ),
            default,
        )

    if mode in {"LIE", "LIEIIIC"}:
        edge_aggregation_payload = _profile_object_size(config, "upd")
    elif mode in {"LIIE", "LIIEIIIC"}:
        edge_aggregation_payload = effective_payload(
            "L_E_upd", _profile_object_size(config, "upd")
        )
    else:
        edge_aggregation_payload = 0.0

    if mode == "LIC":
        cloud_aggregation_payload = _profile_object_size(config, "upd")
    elif mode == "LIIC":
        cloud_aggregation_payload = effective_payload(
            "L_C_upd", _profile_object_size(config, "upd")
        )
    elif mode in {"LIEIIC", "LIEIIIC", "LIIEIIIC"}:
        cloud_aggregation_payload = effective_payload(
            "E_C_upd", _profile_object_size(config, "upd")
        )
    else:
        cloud_aggregation_payload = 0.0

    if config.mainline_fusion:
        edge_to_cloud_time = sum(item["total_link_time"] for item in link_metrics
                                 if item["link_id"] == "E_C_upd")
        first_aggregation_arrival_time = compute_time + sum(
            item["total_link_time"] for item in link_metrics
            if item["link_id"] != "E_C_upd"
            and not item["link_id"].endswith("_final_return"))
        edge_aggregation_payload = _profile_object_size(config, "upd")
        cloud_aggregation_payload = effective_payload("E_C_upd")

    edge_aggregation_events = E if mode in {"LIEIIIC", "LIIEIIIC"} else int(
        mode in {"LIE", "LIIE"}
    )
    if config.mainline_fusion:
        edge_aggregation_events = 1
    edge_aggregation_time = edge_aggregation_events * (
        config.edge_aggregation_beta * edge_aggregation_payload
        + config.edge_aggregation_fixed
    )
    cloud_aggregation_time = int(config.mainline_fusion or _mode_reaches_cloud(spec)) * (
        config.cloud_aggregation_beta * cloud_aggregation_payload
        + config.cloud_aggregation_fixed
    )
    time = (
        first_aggregation_arrival_time
        + edge_aggregation_time
        + edge_to_cloud_time
        + cloud_aggregation_time
        + return_path_time
    )
    pre_aggregation_time = first_aggregation_arrival_time

    sample_embedding_events = 0
    sample_label_grad_events = 0
    sample_optimizer_events = 0
    sample_epsilon_after = 0.0

    resolved_sample_embedding_sigma = None
    resolved_sample_label_grad_sigma = None
    resolved_sample_optimizer_sigma = None

    if config.privacy_unit == "sample":
        feature_dp_events = 0
        update_dp_events = 0
        feature_noise_multiplier = None
        feature_epsilon_after = 0.0
        update_epsilon_after = 0.0

        (
            sample_embedding_events,
            sample_label_grad_events,
            sample_optimizer_events,
        ) = _sample_dp_event_counts(
            config,
            mode,
            samples,
        )

        total_sample_events = (
            sample_embedding_events
            + sample_label_grad_events
            + sample_optimizer_events
        )

        sample_parameters = (
            resolved_sample_privacy_parameters(
                config,
                total_sample_events,
            )
        )

        resolved_sample_embedding_sigma = float(
            sample_embedding_noise_multiplier
            if sample_embedding_noise_multiplier is not None
            else sample_parameters[
                "embedding_noise_multiplier"
            ]
        )

        resolved_sample_label_grad_sigma = float(
            sample_label_grad_noise_multiplier
            if sample_label_grad_noise_multiplier is not None
            else sample_parameters[
                "label_grad_noise_multiplier"
            ]
        )

        resolved_sample_optimizer_sigma = float(
            sample_optimizer_noise_multiplier
            if sample_optimizer_noise_multiplier is not None
            else sample_parameters[
                "optimizer_noise_multiplier"
            ]
        )

        if privacy_ledger is not None:
            if not isinstance(
                privacy_ledger,
                SamplePrivacyLedger,
            ):
                raise TypeError(
                    "sample privacy_unit requires "
                    "SamplePrivacyLedger"
                )

            projection = privacy_ledger.project(
                embedding_events=(
                    sample_embedding_events
                ),
                label_grad_events=(
                    sample_label_grad_events
                ),
                optimizer_events=(
                    sample_optimizer_events
                ),
                embedding_noise_multiplier=(
                    resolved_sample_embedding_sigma
                ),
                label_grad_noise_multiplier=(
                    resolved_sample_label_grad_sigma
                ),
                optimizer_noise_multiplier=(
                    resolved_sample_optimizer_sigma
                ),
            )

            epsilon_used = projection.epsilon_increment
            sample_epsilon_after = (
                projection.epsilon_after
            )
            feasible_privacy = (
                privacy_ledger.can_apply(projection)
            )

        else:
            temporary_sample_ledger = (
                build_sample_privacy_ledger(config)
            )

            projection = (
                temporary_sample_ledger.project(
                    embedding_events=(
                        sample_embedding_events
                    ),
                    label_grad_events=(
                        sample_label_grad_events
                    ),
                    optimizer_events=(
                        sample_optimizer_events
                    ),
                    embedding_noise_multiplier=(
                        resolved_sample_embedding_sigma
                    ),
                    label_grad_noise_multiplier=(
                        resolved_sample_label_grad_sigma
                    ),
                    optimizer_noise_multiplier=(
                        resolved_sample_optimizer_sigma
                    ),
                )
            )

            epsilon_used = projection.epsilon_after
            sample_epsilon_after = (
                projection.epsilon_after
            )

            feasible_privacy = (
                sample_epsilon_after
                <= remaining_epsilon + 1e-12
            )

    else:
        # Split modes release an intermediate embedding outside the client. When
        # that embedding link is assigned DP, charge one record-level feature-DP
        # event per local epoch (and per explicit edge loop for repeated split
        # execution), consistently with _record_dp_event_count().
        feature_dp_events = sum(
            _record_dp_event_count(config, mode, count)
            for link_id, obj, count, privacy_eligible in link_events
            if privacy_eligible
            and obj == "emb"
            and mechanism_uses_dp(actual_link_mechanisms[link_id])
        )
        update_dp_events = sum(
            count
            for link_id, obj, count, privacy_eligible in link_events
            if privacy_eligible
            and obj == "upd"
            and mechanism_uses_dp(actual_link_mechanisms[link_id])
        )
        feature_noise_multiplier = mode_aware_feature_noise_multiplier(
            config,
            feature_dp_events,
        )
        feature_epsilon_after = 0.0
        update_epsilon_after = 0.0
        if privacy_ledger is not None:
            projection = privacy_ledger.project(
                feature_dp_events,
                update_dp_events,
                feature_noise_multiplier=feature_noise_multiplier,
                update_noise_multiplier=update_noise_multiplier,
            )
            # Legacy scalar used only for diagnostics/old local policies. The
            # formal feasibility test remains two-ledger (feature, update).
            epsilon_used = max(
                projection.feature_epsilon_increment,
                projection.update_epsilon_increment,
            )
            feature_epsilon_after = projection.feature_epsilon_after
            update_epsilon_after = projection.update_epsilon_after
            feasible_privacy = privacy_ledger.can_apply(projection)
        else:
            # Compatibility path for callers that have not yet supplied an RDP ledger.
            epsilon_used = sum(
                count * _dp_event_epsilon(config, obj)
                for link_id, obj, count, privacy_eligible in link_events
                if privacy_eligible
                and obj == "upd"
                and mechanism_uses_dp(actual_link_mechanisms[link_id])
            )
            feasible_privacy = epsilon_used <= remaining_epsilon + 1e-12

    risk = max(
        (
            OBJECT_RISK[obj] * MECHANISM_RISK[mech]
            for link_id, mech in actual_link_mechanisms.items()
            for obj in [_link_object(link_id, link_events)]
            if obj in PRIVACY_RISK_OBJECTS
            and mech != "trusted"
        ),
        default=0.0,
    )
    feasible_risk = risk <= config.risk_limit
    feasible_time = (
        True
        if fast_response_deadline is None
        else time <= float(fast_response_deadline) + 1e-12
    )

    # Edge/cloud CPU feasibility (scaled by sample ratio)
    cpu_scale = samples / 150.0
    mode_edge_demand = spec.edge_cpu * cpu_scale
    mode_cloud_demand = spec.cloud_cpu * cpu_scale
    feasible_edge = (current_edge_load + mode_edge_demand) <= config.edge_cpu_limit + 1e-12
    feasible_cloud = (current_cloud_load + mode_cloud_demand) <= config.cloud_cpu_limit + 1e-12

    # Global penalty: only cloud-reaching objects affect global model accuracy
    protected_links = [event for event in link_events if event[3]]
    mech_penalty = (
        0.0
        if config.mainline_fusion
        else (
            sum(
                utility_penalty(
                    actual_link_mechanisms[link_id],
                    max(
                        float(
                            resolved_sample_privacy_parameters(
                                config,
                                0,
                            )["sample_budget"]
                            if config.privacy_unit == "sample"
                            else resolved_privacy_parameters(config)[
                                "update_budget"
                                if obj == "upd"
                                else "feature_budget"
                            ]
                        ),
                        1e-6,
                    ),
                )
                for link_id, obj, _count, _privacy_eligible in protected_links
            )
            / max(len(protected_links), 1)
        )
    )
    penalty = spec.mode_penalty + mech_penalty
    progress = (round_idx + 1.0) / max(config.rounds, 1)
    accuracy = 0.2 + (0.83 - penalty - 0.2) * (1.0 - math.exp(-3.0 * progress))
    accuracy += _candidate_accuracy_jitter(
        config,
        client_id,
        round_idx,
        mode,
        {} if config.mainline_fusion else actual_link_mechanisms,
    )

    return Candidate(
        mode=mode,
        mechanisms=mechanisms,
        time=time,
        accuracy=max(0.0, min(0.95, accuracy)),
        risk=risk,
        epsilon_used=epsilon_used,
        communication_volume=communication_volume,
        feasible_resource=feasible_resource,
        feasible_privacy=feasible_privacy,
        feasible_risk=feasible_risk,
        feasible_time=feasible_time,
        feasible_edge=feasible_edge,
        feasible_cloud=feasible_cloud,
        pre_aggregation_time=pre_aggregation_time,
        feature_dp_events=feature_dp_events,
        update_dp_events=update_dp_events,
        feature_epsilon_after=feature_epsilon_after,
        update_epsilon_after=update_epsilon_after,
        sample_embedding_events=sample_embedding_events,
        sample_label_grad_events=sample_label_grad_events,
        sample_optimizer_events=sample_optimizer_events,
        sample_epsilon_after=sample_epsilon_after,
        sample_embedding_noise_multiplier=(
            resolved_sample_embedding_sigma
            if sample_embedding_events > 0
            else None
        ),
        sample_label_grad_noise_multiplier=(
            resolved_sample_label_grad_sigma
            if sample_label_grad_events > 0
            else None
        ),
        sample_optimizer_noise_multiplier=(
            resolved_sample_optimizer_sigma
            if sample_optimizer_events > 0
            else None
        ),
        link_mechanisms=actual_link_mechanisms,
        memory_requirement=memory_requirement,
        memory_capacity=memory_capacity,
        feasible_memory=feasible_memory,
        first_aggregation_arrival_time=first_aggregation_arrival_time,
        edge_to_cloud_time=edge_to_cloud_time,
        return_path_time=return_path_time,
        edge_aggregation_payload=edge_aggregation_payload,
        cloud_aggregation_payload=cloud_aggregation_payload,
        link_metrics=tuple(link_metrics),
        global_release_required=bool(config.mainline_fusion),
        feature_noise_multiplier=(
            float(feature_noise_multiplier)
            if feature_dp_events > 0 and feature_noise_multiplier is not None
            else None
        ),
        update_noise_multiplier=(
            float(update_noise_multiplier) if update_dp_events > 0 and update_noise_multiplier is not None else None
        ),
    )


def _sample_dp_optimizer_event_count(
    samples: int,
    batch_size: int,
    epochs: int,
    local_steps: int | None,
) -> int:
    """Return the actual number of sample-DP optimizer minibatch events.

    This mirrors split_local_train_lenet5() and _training_batches():

    - empty data or zero epochs -> zero events;
    - without a local step limit, every minibatch in every epoch is used;
    - with a local step limit, the dataset is first capped at
      ``local_steps * batch_size`` samples and the step limit applies across
      the whole training call, not separately to each epoch.
    """
    sample_count = max(0, int(samples))
    batch = max(1, int(batch_size))
    epoch_count = max(0, int(epochs))

    if sample_count == 0 or epoch_count == 0:
        return 0

    if local_steps is None:
        batches_per_epoch = math.ceil(sample_count / batch)
        return epoch_count * batches_per_epoch

    step_limit = max(0, int(local_steps))
    if step_limit == 0:
        return 0

    max_local_samples = max(1, step_limit) * batch
    effective_samples = min(sample_count, max_local_samples)
    batches_per_epoch = math.ceil(effective_samples / batch)

    return min(
        step_limit,
        epoch_count * batches_per_epoch,
    )

def _sample_dp_event_counts(
    config: SelectionConfig,
    mode: str,
    samples: int,
) -> tuple[int, int, int]:
    """Return (embedding, label-gradient, optimizer) sample-DP events.

    Sample-level accounting follows the actual local optimizer minibatch
    schedule rather than the legacy static link execution count.

    For split training, every executed minibatch releases one protected
    embedding and one protected label gradient. All modes execute one
    sample-DP optimizer event per actual local optimizer minibatch.
    """
    spec = MODE_SPECS[mode]

    local_steps = (
        None
        if config.mainline_fusion
        else int(config.L_block_cycles)
    )

    local_batch_events = _sample_dp_optimizer_event_count(
        samples=int(samples),
        batch_size=int(config.split_batch_size),
        epochs=int(config.privacy_local_epochs),
        local_steps=local_steps,
    )

    if local_batch_events <= 0:
        return 0, 0, 0

    link_events = (
        _fusion_link_transmissions(
            mode,
            1,
            spec.E_edge_loops,
        )
        if config.mainline_fusion
        else _mode_link_transmissions(
            mode,
            1,
            spec.E_edge_loops,
        )
    )

    split_objects = {
        obj
        for _link_id, obj, _count, _privacy_eligible in link_events
    }
    is_split_execution = bool(
        {"emb", "grad"} & split_objects
    )

    released_objects = {
        obj
        for _link_id, obj, _count, privacy_eligible in link_events
        if privacy_eligible
    }

    embedding_events = (
        local_batch_events
        if "emb" in released_objects
        else 0
    )
    label_grad_events = (
        local_batch_events
        if "grad" in released_objects
        else 0
    )

    optimizer_events = (
        local_batch_events
        if (
            not is_split_execution
            or bool(config.split_end_optimizer_enabled)
        )
        else 0
    )

    return (
        embedding_events,
        label_grad_events,
        optimizer_events,
    )

def _record_dp_event_count(
    config: SelectionConfig,
    mode: str,
    communication_count: int,
) -> int:
    """Upper-bound releases involving one record during a training round.

    Communication counts batches. Under shuffled passes without replacement, a
    record occurs at most once per local epoch and once per explicit edge loop.
    """
    if mode not in {"LIE", "LIC", "LIEIIC", "LIEIIIC"}:
        return int(communication_count)
    edge_loops = MODE_SPECS[mode].E_edge_loops
    return max(1, int(config.privacy_local_epochs)) * max(1, int(edge_loops))


def _dp_event_count(spec: ModeSpec, obj: str, local_block_cycles: int, edge_loops: int) -> int:
    return sum(
        count
        for event_obj, count, privacy_eligible in _mode_link_events(
            spec.name,
            local_block_cycles,
            edge_loops,
        )
        if privacy_eligible and event_obj == obj
    )


def _mode_link_events(
    mode: str,
    local_block_cycles: int,
    edge_loops: int,
) -> tuple[tuple[str, int, bool], ...]:
    """Return (object, execution count, privacy-eligible) for one full flow."""
    return tuple(
        (obj, count, privacy_eligible)
        for _link_id, obj, count, privacy_eligible in _mode_link_transmissions(
            mode,
            local_block_cycles,
            edge_loops,
        )
    )


def _fusion_link_transmissions(mode, local_block_cycles, edge_loops):
    if mode == "LIC":
        return ()
    if mode in {"LIE", "LIEIIC", "LIEIIIC"}:
        training = tuple(event for event in _mode_link_transmissions(
            "LIE", max(1, local_block_cycles) * max(1, edge_loops), 1)
            if not event[0].endswith("_final_return"))
    else:
        training = (("L_E_upd", "upd", 1, True),)
    return training + (("E_C_upd", "upd", 1, True),
                       ("C_E_upd_final_return", "upd", 1, False),
                       ("E_L_upd_final_return", "upd", 1, False))


_DYNAMIC_MODE_POLICIES = {
    "ours",
    "ours_no_omega",
    "full_dynfl",
    "full_dynfl_3obj_diag",
    "dynamic_mode_fixed_privacy",
    "nsga2",
    "individual_optimal",
    "random",
}

_FROZEN_MODE_POLICIES = {
    "fixed_mode_fixed_privacy",
    "fixed_mode_dynamic_privacy",
}


def _policy_uses_dynamic_mode_selection(policy: str) -> bool:
    return policy in _DYNAMIC_MODE_POLICIES


def _policy_uses_fl_first_mode_admissibility(policy: str) -> bool:
    # The two fixed-mode factorial baselines use the same round-0 admissible
    # mode set as DynFL, then freeze that per-client assignment. Therefore the
    # FL-first split-on-demand gate also applies while initializing and
    # revalidating their frozen per-client modes.
    return policy in _DYNAMIC_MODE_POLICIES or policy in _FROZEN_MODE_POLICIES


def _split_interaction_count(config: SelectionConfig, samples: int) -> int:
    """Return communication interactions for one local split-training stage.

    ``fixed`` preserves the historical L_block_cycles model. ``workload``
    binds split communication to the actual local training workload: local
    epochs times the number of minibatches. Privacy accounting remains
    separate because record-level exposure is not the same as batch-level
    communication multiplicity.
    """
    if config.split_interaction_mode == "fixed":
        return max(1, int(config.L_block_cycles))
    if config.split_interaction_mode != "workload":
        raise ValueError(
            "split_interaction_mode must be 'fixed' or 'workload'"
        )
    batch_size = max(1, int(config.split_batch_size))
    batches_per_epoch = max(1, math.ceil(max(1, int(samples)) / batch_size))
    return max(1, int(config.privacy_local_epochs)) * batches_per_epoch


def _mode_link_transmissions(
    mode: str,
    local_block_cycles: int,
    edge_loops: int,
) -> tuple[tuple[str, str, int, bool], ...]:
    """Return (link id, object, count, privacy eligibility) for a full flow."""
    L = max(1, int(local_block_cycles))
    E = max(1, int(edge_loops))
    if mode == "LIE":
        return (
            ("L_E_emb", "emb", L, True),
            ("E_L_logits", "logits", L, False),
            ("L_E_grad", "grad", L, True),
            ("E_L_emb_grad", "emb_grad", L, False),
            ("E_L_upd_final_return", "upd", 1, False),
        )
    if mode == "LIC":
        return (
            ("L_C_emb", "emb", L, True),
            ("C_L_logits", "logits", L, False),
            ("L_C_grad", "grad", L, True),
            ("C_L_emb_grad", "emb_grad", L, False),
            ("C_L_upd_final_return", "upd", 1, False),
        )
    if mode == "LIIE":
        return (
            ("L_E_upd", "upd", 1, True),
            ("E_L_upd_final_return", "upd", 1, False),
        )
    if mode == "LIIC":
        return (
            ("L_C_upd", "upd", 1, True),
            ("C_L_upd_final_return", "upd", 1, False),
        )
    if mode == "LIEIIC":
        return (
            ("L_E_emb", "emb", L, True),
            ("E_L_logits", "logits", L, False),
            ("L_E_grad", "grad", L, True),
            ("E_L_emb_grad", "emb_grad", L, False),
            ("E_C_upd", "upd", 1, True),
            ("C_E_upd_final_return", "upd", 1, False),
            ("E_L_upd_final_return", "upd", 1, False),
        )
    if mode == "LIEIIIC":
        return (
            ("L_E_emb", "emb", L * E, True),
            ("E_L_logits", "logits", L * E, False),
            ("L_E_grad", "grad", L * E, True),
            ("E_L_emb_grad", "emb_grad", L * E, False),
            ("E_L_upd_loop_return", "upd", E, False),
            ("E_C_upd", "upd", 1, True),
            ("C_E_upd_final_return", "upd", 1, False),
            ("E_L_upd_final_return", "upd", 1, False),
        )
    if mode == "LIIEIIIC":
        return (
            ("L_E_upd", "upd", E, True),
            ("E_L_upd_loop_return", "upd", E, False),
            ("E_C_upd", "upd", 1, True),
            ("C_E_upd_final_return", "upd", 1, False),
            ("E_L_upd_final_return", "upd", 1, False),
        )
    return ()


def _link_object(
    link_id: str,
    transmissions: tuple[tuple[str, str, int, bool], ...],
) -> str:
    return next(obj for event_link, obj, _count, _eligible in transmissions if event_link == link_id)


def _link_route(link_id: str) -> str:
    parts = link_id.split("_", 2)
    if len(parts) < 2:
        raise ValueError(f"Invalid link id: {link_id}")
    return f"{parts[0]}_{parts[1]}"



def resource_phase(config: SelectionConfig, round_idx: int) -> str:
    """Return the Q86 staged resource phase for a training round."""
    if config.resource_scenario == "none":
        return "normal"
    if config.resource_scenario not in {"communication", "compute"}:
        raise ValueError(f"unknown resource_scenario: {config.resource_scenario}")
    progress = float(round_idx) / max(int(config.rounds), 1)
    if float(config.constrained_start_fraction) <= progress < float(config.constrained_end_fraction):
        return "constrained"
    return "normal"


def staged_compute_factor(config: SelectionConfig, base_factor: float, round_idx: int) -> float:
    if config.resource_scenario == "compute" and resource_phase(config, round_idx) == "constrained":
        return float(base_factor) * float(config.compute_constrained_multiplier)
    return float(base_factor)


def _link_bandwidth(
    config: SelectionConfig,
    client_id: int,
    round_idx: int,
    link_id: str,
) -> float:
    base_rates = {
        "L_E": config.end_edge_rate_mb_s,
        "E_L": config.end_edge_rate_mb_s,
        "L_C": config.end_cloud_rate_mb_s,
        "C_L": config.end_cloud_rate_mb_s,
        "E_C": config.edge_cloud_rate_mb_s,
        "C_E": config.edge_cloud_rate_mb_s,
    }
    route = _link_route(link_id)
    base = base_rates[route]
    if config.resource_scenario == "communication" and resource_phase(config, round_idx) == "constrained":
        base *= float(config.communication_constrained_multiplier)
    period = max(float(config.network_period_rounds), 1e-9)
    periodic = 1.0 + float(config.network_periodic_amplitude) * math.sin(
        round_idx / period
    )
    route_code = sum((index + 1) * ord(char) for index, char in enumerate(route))
    local_rng = random.Random(
        int(config.seed) * 1_000_003
        + int(client_id) * 10_007
        + int(round_idx) * 101
        + route_code
    )
    jitter = local_rng.uniform(
        max(0.05, 1.0 - config.network_jitter),
        1.0 + config.network_jitter,
    )
    return max(0.05, base * periodic * jitter)


def _link_base_latency(config: SelectionConfig, link_id: str) -> float:
    return {
        "L_E": config.end_edge_base_latency_sec,
        "E_L": config.end_edge_base_latency_sec,
        "L_C": config.end_cloud_base_latency_sec,
        "C_L": config.end_cloud_base_latency_sec,
        "E_C": config.edge_cloud_base_latency_sec,
        "C_E": config.edge_cloud_base_latency_sec,
    }[_link_route(link_id)]


def _candidate_accuracy_jitter(
    config: SelectionConfig,
    client_id: int,
    round_idx: int,
    mode: str,
    link_mechanisms: dict[str, str],
) -> float:
    signature = mode + ";" + ";".join(
        f"{key}:{value}" for key, value in sorted(link_mechanisms.items())
    )
    code = sum((index + 1) * ord(char) for index, char in enumerate(signature))
    local_rng = random.Random(
        int(config.seed) * 1_000_033
        + int(client_id) * 10_009
        + int(round_idx) * 103
        + code
    )
    return local_rng.uniform(-0.002, 0.002)


def _final_return_volume(mode: str) -> float:
    return OBJECT_SIZES["upd"] * (
        2.0 if mode in {"LIEIIC", "LIEIIIC", "LIIEIIIC"} else 1.0
    )


def _dp_event_epsilon(config: SelectionConfig, obj: str) -> float:
    if obj == "upd":
        return config.dp_upd_epsilon
    if obj in {"emb", "grad", "weakemb", "strongemb", "pseudo_label"}:
        return config.dp_emb_epsilon
    return config.dp_event_epsilon


def _admit_fastest(arrivals: list[tuple[float, int]], aggregation_fraction: float) -> list[tuple[float, int]]:
    k = _buffer_size(len(arrivals), aggregation_fraction)
    ordered = sorted(arrivals, key=lambda item: item[0])
    if k <= 0 or not ordered:
        return []
    threshold = ordered[min(k, len(ordered)) - 1][0]
    return [item for item in ordered if item[0] <= threshold + 1e-12]


def _buffer_size(n: int, aggregation_fraction: float) -> int:
    if n <= 0:
        return 0
    return max(1, math.ceil(n * aggregation_fraction))


def _prefer_he(obj: str) -> str:
    if "he2" in MECHANISMS_BY_OBJECT[obj]:
        return "he2"
    if "he3" in MECHANISMS_BY_OBJECT[obj]:
        return "he3"
    return "none"


def _mechanism_label(mechanisms: dict[str, str]) -> str:
    return ";".join(f"{key}:{value}" for key, value in sorted(mechanisms.items()))


def _summarize_policy(policy: str, rows: list[dict[str, Any]], time_limit: float) -> dict[str, Any]:
    return {
        "policy": policy,
        "mean_accuracy_estimate": _mean(row["accuracy_estimate"] for row in rows),
        "mean_time": _mean(row["time"] for row in rows),
        "total_communication_volume": sum(row["communication_volume"] for row in rows),
        "max_risk": max(row["risk"] for row in rows),
        "min_remaining_epsilon": min(row["remaining_epsilon"] for row in rows),
        "max_feature_epsilon": max(float(row.get("feature_epsilon", 0.0)) for row in rows),
        "max_update_epsilon": max(float(row.get("update_epsilon", 0.0)) for row in rows),
        "larger_channel_epsilon": max(
            max(float(row.get("feature_epsilon", 0.0)) for row in rows),
            max(float(row.get("update_epsilon", 0.0)) for row in rows),
        ),
        "privacy_guarantee": "record_feature_and_client_update",
        "feasible_rate": _mean(float(row.get("feasible_resource", row["feasible"])) for row in rows),
        "all_constraint_feasible_rate": _mean(float(row["feasible"]) for row in rows),
        "resource_feasible_rate": _mean(float(row.get("feasible_resource", row["feasible"])) for row in rows),
        "memory_feasible_rate": _mean(float(row.get("feasible_memory", row["feasible"])) for row in rows),
        "privacy_feasible_rate": _mean(float(row.get("feasible_privacy", row["feasible"])) for row in rows),
        "risk_feasible_rate": _mean(float(row.get("feasible_risk", row["feasible"])) for row in rows),
        "edge_feasible_rate": _mean(float(row.get("feasible_edge", 1.0)) for row in rows),
        "cloud_feasible_rate": _mean(float(row.get("feasible_cloud", 1.0)) for row in rows),
        "time_satisfied_rate": _mean(float(float(row["time"]) <= time_limit) for row in rows),
        "most_common_mode": _most_common(row["mode"] for row in rows),
        "mode_distribution": _distribution(row["mode"] for row in rows),
    }


def _mean(values: Any) -> float:
    values = list(values)
    return sum(values) / max(len(values), 1)


def _most_common(values: Any) -> str:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return max(counts, key=counts.get)


def _distribution(values: Any) -> str:
    counts: dict[str, int] = {}
    total = 0
    for value in values:
        counts[value] = counts.get(value, 0) + 1
        total += 1
    parts = []
    for value, count in sorted(counts.items()):
        ratio = count / max(total, 1)
        parts.append(f"{value}:{count}({ratio:.3f})")
    return ";".join(parts)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)
