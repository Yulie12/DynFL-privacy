from __future__ import annotations

import csv
import json
import math
import random
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
    "emb": ("none",),
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


@dataclass(frozen=True)
class SelectionConfig:
    rounds: int = 100
    num_clients: int = 100
    num_edges: int = 10
    seed: int = 42
    initial_epsilon: float = 8.0
    dp_event_epsilon: float = 0.05
    dp_emb_epsilon: float = 8.0
    dp_upd_epsilon: float = 8.0
    dp_noise_multiplier: float = 0.0002
    dp_delta: float = 1e-5
    dp_accounting_mode: str = "rdp_auto"
    dp_feature_epsilon_budget: float | None = None
    dp_update_epsilon_budget: float | None = None
    dp_feature_noise_multiplier: float | None = None
    dp_update_noise_multiplier: float | None = None
    client_heterogeneity: float = 2.0
    edge_heterogeneity: float = 1.5
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
    time_limit: float = 8.0
    risk_limit: float = 0.5
    aggregation_fraction: float = 1.0
    output_dir: str = "out/selection"
    require_feasible: bool = False
    L_block_cycles: int = 5
    privacy_local_epochs: int = 1
    trusted_edge_split_execution: bool = False
    allow_he: bool = True
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
    omega_update_clip_norm: float = 1.0
    omega_update_clip_excess_sq: float = 0.02
    omega_update_dimension: float = 61706.0
    embedding_payload_mb: float = 1.6
    update_payload_mb: float = 4.0


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
        if config.trusted_edge_split_execution and mode == "LIC":
            continue
        events = _mode_link_transmissions(
            mode,
            config.L_block_cycles,
            spec.E_edge_loops,
        )
        feature_events = sum(
            _record_dp_event_count(config, mode, count)
            for _link_id, obj, count, privacy_eligible in events
            if privacy_eligible and obj != "upd"
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

    max_feature_events_per_round = 0
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
        "feature_dp_enabled": False,
    }


def build_client_privacy_ledger(config: SelectionConfig) -> ClientPrivacyLedger:
    resolved = resolved_privacy_parameters(config)
    return ClientPrivacyLedger(
        feature_budget=float(resolved["feature_budget"]),
        update_budget=float(resolved["update_budget"]),
        delta=float(resolved["delta"]),
        feature_noise_multiplier=float(resolved["feature_noise_multiplier"]),
        update_noise_multiplier=float(resolved["update_noise_multiplier"]),
    )


@dataclass(frozen=True)
class ProfileEvaluation:
    profile: dict[int, Candidate]
    system_latency: float
    system_omega: float
    cloud_fusion_ratio: float
    admitted_client_ids: tuple[int, ...] = ()
    profile_signature: tuple[int, ...] = ()


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

            if policy == "ours":
                client_samples = {client.client_id: float(client.samples) for client in clients}
                client_edges = {client.client_id: int(client.edge_id) for client in clients}
                selection_diagnostics: dict[str, Any] = {}
                round_selected, profile_evaluation = choose_global_pareto_profile(
                    config=config,
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
                            "system_omega_objective": evaluation.system_omega,
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
                    "sensitivity": client_sensitivity[client_id],
                    "system_latency_objective": profile_evaluation.system_latency if profile_evaluation else "",
                    "system_omega_objective": profile_evaluation.system_omega if profile_evaluation else "",
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
    privacy_ledger: ClientPrivacyLedger | None = None,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for mode, spec in MODE_SPECS.items():
        if config.trusted_edge_split_execution and mode == "LIC":
            continue
        if config.require_cloud_participation and not _mode_reaches_cloud(spec):
            continue
        policy_allow_none = allow_none or policy in {
            "performance_only",
            "best_accuracy",
            "accuracy_oracle",
        }
        assignments = _mechanism_assignments(
            spec,
            policy,
            config.allow_he,
            policy_allow_none,
            trusted_edge_split_execution=config.trusted_edge_split_execution,
        )
        for mechanisms, link_mechanisms in assignments:
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
                )
            )
    return _apply_policy_candidate_filters(config, policy, candidates)


def _apply_policy_candidate_filters(
    config: SelectionConfig,
    policy: str,
    candidates: list[Candidate],
) -> list[Candidate]:
    """Apply shared admissibility rules before an adaptive policy selects."""
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
    spec = MODE_SPECS.get(candidate.mode)
    return bool(spec and _mode_reaches_cloud(spec))


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
) -> tuple[list[tuple[int, Candidate, list[Candidate], float]], ProfileEvaluation]:
    """Approximate TeX Algorithm 1 over a global client profile.

    Each client contributes a feasible candidate set S_i. The search keeps a
    bounded non-dominated archive over (T_sys, Omega_sys), expands profiles by
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

    previous_choices = previous_choices or {}
    client_edges = client_edges or {}
    pools: dict[int, list[Candidate]] = {}
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
        if config.enforce_cloud_dp_stability:
            pool = _stable_cloud_candidate_pool(config, pool)
        pools[client_id] = pool

    seeds = (
        _initial_profiles(config, pools, previous_choices)
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
    sample_total = max(sum(float(value) for value in client_samples.values()), 1.0)
    candidate_tokens = {
        client_id: {
            _candidate_key(candidate): token
            for token, candidate in enumerate(pools[client_id])
        }
        for client_id in client_order
    }
    omega_components_by_candidate = {
        id(candidate): _local_omega_components(candidate, config)
        for candidates in pools.values()
        for candidate in candidates
    }

    def signature(profile: dict[int, Candidate]) -> tuple[int, ...]:
        return tuple(
            candidate_tokens[client_id][_candidate_key(profile[client_id])]
            for client_id in client_order
        )

    evaluation_cache: dict[tuple[int, ...], ProfileEvaluation] = {}

    def evaluate(
        profile: dict[int, Candidate],
        omega_stats: _ProfileOmegaStats | None = None,
        profile_signature: tuple[int, ...] = (),
        flow_objectives: tuple[tuple[int, ...], float] | None = None,
    ) -> ProfileEvaluation:
        profile_key = profile_signature or signature(profile)
        cached = evaluation_cache.get(profile_key)
        if cached is not None:
            return cached
        evaluated = _evaluate_profile(
            config,
            profile,
            client_samples,
            client_edges,
            previous_choices,
            omega_stats=omega_stats,
            profile_signature=profile_key,
            flow_inputs_by_candidate=flow_inputs_by_candidate,
            flow_objectives=flow_objectives,
        )
        evaluation_cache[profile_key] = evaluated
        return evaluated

    archive_selector = (
        _pareto_archive
        if objective == "pareto"
        else _latency_archive
    )
    seed_evaluations = [
            evaluate(
                item,
                omega_stats=_profile_omega_stats(
                    config,
                    item,
                    client_samples,
                    client_edges,
                    candidate_omega_components=omega_components_by_candidate,
                ),
                profile_signature=signature(item),
            )
            for item in seeds
        ]
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
                else ("latency", "omega", "pareto")
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
        neighbors: list[
            tuple[
                dict[int, Candidate],
                _ProfileOmegaStats,
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
            base_stats = _profile_omega_stats(
                config,
                evaluated.profile,
                client_samples,
                client_edges,
                candidate_omega_components=omega_components_by_candidate,
            )
            profile_neighbors: list[
                tuple[
                    float,
                    dict[int, Candidate],
                    _ProfileOmegaStats,
                    tuple[int, ...],
                    tuple[tuple[int, ...], float] | None,
                ]
            ] = []
            for client_id in search_client_ids:
                candidates = pools[client_id]
                current = evaluated.profile[client_id]
                for candidate in candidates:
                    if candidate == current:
                        continue
                    profile = dict(evaluated.profile)
                    profile[client_id] = candidate
                    token = candidate_tokens[client_id][_candidate_key(candidate)]
                    position = client_positions[client_id]
                    key = (
                        evaluated.profile_signature[:position]
                        + (token,)
                        + evaluated.profile_signature[position + 1:]
                    )
                    if key in visited_profiles:
                        continue
                    visited_profiles.add(key)
                    stats = _replace_profile_omega_stats(
                        config,
                        base_stats,
                        client_id,
                        current,
                        candidate,
                        client_samples,
                        client_edges,
                        candidate_omega_components=omega_components_by_candidate,
                    )
                    flow_objectives = None
                    if base_flow_stats is not None:
                        flow_objectives = _replace_full_buffer_flow_objectives(
                            config,
                            base_flow_stats,
                            client_id,
                            flow_inputs_by_candidate[client_id].get(
                                _candidate_key(candidate)
                            ),
                        )
                    profile_neighbors.append(
                        (
                            _global_replacement_priority(
                                objective=objective,
                                config=config,
                                evaluated=evaluated,
                                client_id=client_id,
                                current=current,
                                candidate=candidate,
                                client_samples=client_samples,
                                sample_total=sample_total,
                                omega_stats=stats,
                                flow_objectives=flow_objectives,
                            )
                            if config.pareto_neighbor_top_k > 0
                            else 0.0,
                            profile,
                            stats,
                            key,
                            flow_objectives,
                        )
                    )
            if (
                config.pareto_neighbor_top_k > 0
                and (
                    objective == "latency"
                    or _evaluation_key(evaluated) != omega_endpoint_key
                )
            ):
                profile_neighbors.sort(key=lambda item: item[0])
                profile_neighbors = profile_neighbors[: config.pareto_neighbor_top_k]
            neighbors.extend(
                (profile, stats, key, flow_objectives)
                for _priority, profile, stats, key, flow_objectives in profile_neighbors
            )
        if not neighbors:
            break

        evaluated_neighbors = [
            evaluate(
                profile,
                omega_stats=stats,
                profile_signature=profile_key,
                flow_objectives=flow_objectives,
            )
            for profile, stats, profile_key, flow_objectives in neighbors
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
        )
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
                "search_method": search_method,
            }
        )
    rewritten = [
        (client_id, chosen.profile[client_id], by_client[client_id][1], by_client[client_id][2])
        for client_id, _current, _candidates, _remaining in selected
    ]
    return rewritten, chosen


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
    population = _nsga2_environmental_selection(initial, population_size)

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
        population = _nsga2_environmental_selection(
            population + [evaluate(profile)],
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
            offspring.append(evaluate(profile))
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


def _profile_objectives_dominate(
    left: ProfileEvaluation,
    right: ProfileEvaluation,
) -> bool:
    return (
        left.system_latency <= right.system_latency
        and left.system_omega <= right.system_omega
        and (
            left.system_latency < right.system_latency
            or left.system_omega < right.system_omega
        )
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

    def candidate_update_ratio(candidate: Candidate) -> float:
        mechanisms = candidate_mechanisms_for_object(candidate, "upd")
        if config.trusted_edge_split_execution and any(
            mechanism_uses_dp(mechanism) for mechanism in mechanisms
        ):
            expected_admitted = max(
                1.0,
                float(config.num_clients) * float(config.aggregation_fraction),
            )
            dimension_scale = math.sqrt(
                max(float(config.omega_update_dimension), 1.0)
            )
            if any(
                mechanism_uses_dp(mechanism) and mechanism_uses_he(mechanism)
                for mechanism in mechanisms
            ):
                return update_ratio * dimension_scale / expected_admitted
            if candidate.mode in CLOUD_DIRECT_MODES:
                return update_ratio * dimension_scale / math.sqrt(expected_admitted)
            return (
                update_ratio
                * dimension_scale
                * math.sqrt(max(float(config.num_edges), 1.0))
                / expected_admitted
            )
        return update_ratio

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
                and candidate_update_ratio(candidate) > threshold
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
) -> ProfileEvaluation:
    """Repair a global profile so every represented edge reaches the cloud."""
    profile = dict(chosen.profile)
    edge_ids = sorted({client_edges[client_id] for client_id in profile})
    for edge_id in edge_ids:
        edge_clients = [
            client_id
            for client_id in profile
            if client_edges[client_id] == edge_id
        ]
        edge_samples = sum(client_samples[client_id] for client_id in edge_clients)
        target_ratio = min(max(float(config.min_edge_cloud_fusion_ratio), 0.0), 1.0)
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
            previous_covered = covered_samples()
            repairs: list[ProfileEvaluation] = []
            for client_id in edge_clients:
                if _candidate_reaches_cloud(profile[client_id]):
                    continue
                for candidate in pools[client_id]:
                    if not _candidate_reaches_cloud(candidate):
                        continue
                    repaired = dict(profile)
                    repaired[client_id] = candidate
                    repairs.append(
                        _evaluate_profile(
                            config,
                            repaired,
                            client_samples,
                            client_edges,
                            previous_choices,
                        )
                    )
            if not repairs:
                raise RuntimeError(
                    f"No feasible cloud-reaching candidate exists for edge {edge_id}"
                )
            profile = (
                min(
                    repairs,
                    key=lambda item: (item.system_latency, _evaluation_key(item)),
                )
                if objective == "latency"
                else min(
                    repairs,
                    key=lambda item: (item.system_omega, item.system_latency, _evaluation_key(item)),
                )
                if objective == "omega"
                else _choose_tchebycheff(
                    _pareto_archive(repairs, config.pareto_archive_size),
                    config.pareto_norm_eps,
                )
            ).profile
            if covered_samples() <= previous_covered + 1e-12:
                raise RuntimeError(
                    f"Cloud-coverage repair made no progress for edge {edge_id}"
                )

    return _evaluate_profile(
        config,
        profile,
        client_samples,
        client_edges,
        previous_choices,
    )


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


def _initial_profiles(
    config: SelectionConfig,
    pools: dict[int, list[Candidate]],
    previous_choices: dict[int, Candidate],
) -> list[dict[int, Candidate]]:
    fastest = {client_id: min(candidates, key=lambda item: (item.time, _local_omega_proxy(item, config=config))) for client_id, candidates in pools.items()}
    # Zero-DP HE candidates can tie on local Omega; retain the cloud-fusion endpoint in that tie.
    lowest_omega = {
        client_id: min(
            candidates,
            key=lambda item: (
                _local_omega_proxy(item, config=config),
                0 if _candidate_reaches_cloud(item) else 1,
                item.time,
            ),
        )
        for client_id, candidates in pools.items()
    }
    previous = {}
    for client_id, candidates in pools.items():
        prior = previous_choices.get(client_id)
        previous[client_id] = next(
            (
                item for item in candidates
                if prior is not None and _candidate_key(item) == _candidate_key(prior)
            ),
            fastest[client_id],
        )
    intermediate_profiles: list[dict[int, Candidate]] = []
    for latency_weight in (0.25, 0.5, 0.75):
        profile: dict[int, Candidate] = {}
        for client_id, candidates in pools.items():
            times = [candidate.time for candidate in candidates]
            omegas = [
                _local_omega_proxy(candidate, config=config)
                for candidate in candidates
            ]
            t_min, t_max = min(times), max(times)
            o_min, o_max = min(omegas), max(omegas)
            profile[client_id] = min(
                candidates,
                key=lambda item: (
                    max(
                        latency_weight
                        * _safe_norm_eps(item.time, t_min, t_max, config.pareto_norm_eps),
                        (1.0 - latency_weight)
                        * _safe_norm_eps(
                            _local_omega_proxy(item, config=config),
                            o_min,
                            o_max,
                            config.pareto_norm_eps,
                        ),
                    ),
                    item.time,
                    _candidate_key(item),
                ),
            )
        intermediate_profiles.append(profile)
    return _unique_profiles(
        [previous, fastest, *intermediate_profiles, lowest_omega]
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


def _replacement_priority(
    config: SelectionConfig,
    evaluated: ProfileEvaluation,
    client_id: int,
    current: Candidate,
    candidate: Candidate,
    client_samples: dict[int, float],
    sample_total: float,
    stats: _ProfileOmegaStats,
) -> float:
    """Rank one-client replacements before exact profile evaluation.

    This is only a neighbor-generation budget. Exact Pareto filtering and
    Tchebycheff selection still run on evaluated system objectives.
    """
    weight = float(client_samples.get(client_id, 1.0)) / sample_total
    current_local = _local_omega_proxy(current, config=config)
    next_local = _local_omega_proxy(candidate, config=config)
    delta_omega = weight * (next_local - current_local)
    delta_time = candidate.time - current.time
    cloud_gain = float(_candidate_reaches_cloud(candidate)) - float(
        _candidate_reaches_cloud(current)
    )
    current_time_scale = max(abs(current.time), abs(candidate.time), 1.0)
    current_omega_scale = max(
        abs(evaluated.system_omega),
        abs(current_local),
        abs(next_local),
        1.0,
    )
    return (
        0.5 * (delta_time / current_time_scale)
        + 0.5 * (delta_omega / current_omega_scale)
        - 0.35 * cloud_gain
    )


def _global_replacement_priority(
    *,
    objective: str,
    config: SelectionConfig,
    evaluated: ProfileEvaluation,
    client_id: int,
    current: Candidate,
    candidate: Candidate,
    client_samples: dict[int, float],
    sample_total: float,
    omega_stats: _ProfileOmegaStats,
    flow_objectives: tuple[tuple[int, ...], float] | None,
) -> float:
    if objective == "latency":
        if flow_objectives is not None:
            return float(flow_objectives[1])
        return float(evaluated.system_latency + candidate.time - current.time)
    return _replacement_priority(
        config,
        evaluated,
        client_id,
        current,
        candidate,
        client_samples,
        sample_total,
        omega_stats,
    )


def _evaluate_profile(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    previous_choices: dict[int, Candidate],
    omega_stats: _ProfileOmegaStats | None = None,
    profile_signature: tuple[int, ...] = (),
    flow_inputs_by_candidate: dict[int, dict[tuple, ClientFlowInput]] | None = None,
    flow_objectives: tuple[tuple[int, ...], float] | None = None,
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
    if omega_stats is None:
        system_omega, cloud_fusion_ratio = _global_omega_proxy(
            config,
            profile,
            client_samples,
            client_edges,
            admitted_client_ids=admitted_client_ids,
        )
    else:
        system_omega, cloud_fusion_ratio = _omega_from_profile_stats(
            config,
            omega_stats,
            profile,
            client_edges,
            admitted_client_ids,
        )
    return ProfileEvaluation(
        profile=profile,
        system_latency=system_latency,
        system_omega=system_omega,
        cloud_fusion_ratio=cloud_fusion_ratio,
        admitted_client_ids=tuple(sorted(admitted_client_ids)),
        profile_signature=profile_signature,
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
    client_inputs = {
        client_id: flow_inputs_by_candidate[client_id][_candidate_key(candidate)]
        for client_id, candidate in profile.items()
        if candidate.mode != "SKIP"
    }
    groups: dict[tuple[str, int, str], list[ClientFlowInput]] = {}
    for client in client_inputs.values():
        groups.setdefault(_full_buffer_group_key(client), []).append(client)

    edge_cloud_group_count = sum(
        1 for key in groups if key[0] in EDGE_CLOUD_MODES
    )
    if any(
        _buffer_size(len(group), config.aggregation_fraction) != len(group)
        for group in groups.values()
    ):
        return None
    if (
        edge_cloud_group_count > 0
        and _buffer_size(edge_cloud_group_count, config.aggregation_fraction)
        != edge_cloud_group_count
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
        summary = _replace_full_buffer_group_summary(
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
        cloud_aggregation_time = (
            config.cloud_aggregation_beta * cloud_payload_sum
            + config.cloud_aggregation_fixed
        )
        terminal_time = start_time + cloud_aggregation_time + return_path_time
        cloud_aggregation_payload = 0.0
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
    summaries: list[_FullBufferGroupSummary],
) -> float:
    terminal_times = [
        summary.terminal_time
        for summary in summaries
        if summary.kind != "edge_cloud"
    ]
    edge_cloud = [
        summary
        for summary in summaries
        if summary.kind == "edge_cloud"
    ]
    if edge_cloud:
        terminal_times.append(
            max(summary.cloud_arrival_time for summary in edge_cloud)
            + config.cloud_aggregation_beta
            * sum(summary.cloud_aggregation_payload for summary in edge_cloud)
            + config.cloud_aggregation_fixed
            + max(summary.return_path_time for summary in edge_cloud)
        )
    return max(terminal_times, default=0.0)


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
            admitted.update(client_id for _latency, client_id in chosen)
        else:
            if chosen:
                finish = max(latency for latency, _client_id in chosen)
                edge_cloud_groups[key] = (finish, [client_id for _latency, client_id in chosen])

    if edge_cloud_groups:
        edge_arrivals = [
            (finish, group_key, client_ids)
            for group_key, (finish, client_ids) in edge_cloud_groups.items()
        ]
        k = _buffer_size(len(edge_arrivals), config.aggregation_fraction)
        ordered = sorted(edge_arrivals, key=lambda item: item[0])
        threshold = ordered[min(k, len(ordered)) - 1][0]
        for _finish, _group_key, client_ids in (
            item for item in ordered if item[0] <= threshold + 1e-12
        ):
            admitted.update(client_ids)
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
    edge_total_samples: dict[int, float] = {}
    for client_id in profile:
        edge_id = int(client_edges.get(client_id, -1))
        edge_total_samples[edge_id] = (
            edge_total_samples.get(edge_id, 0.0)
            + max(float(client_samples.get(client_id, 1.0)), 0.0)
        )

    admitted_cloud = [
        client_id
        for client_id in admitted_client_ids
        if client_id in profile and _candidate_reaches_cloud(profile[client_id])
    ]
    admitted_by_edge: dict[int, float] = {}
    for client_id in admitted_cloud:
        edge_id = int(client_edges.get(client_id, -1))
        admitted_by_edge[edge_id] = (
            admitted_by_edge.get(edge_id, 0.0)
            + max(float(client_samples.get(client_id, 1.0)), 0.0)
        )
    active_edge_mass = sum(
        edge_total_samples.get(edge_id, 0.0)
        for edge_id, admitted in admitted_by_edge.items()
        if admitted > 0.0
    )
    if active_edge_mass <= 0.0:
        return {}
    return {
        client_id: (
            edge_total_samples[int(client_edges.get(client_id, -1))]
            / active_edge_mass
            * max(float(client_samples.get(client_id, 1.0)), 0.0)
            / admitted_by_edge[int(client_edges.get(client_id, -1))]
        )
        for client_id in admitted_cloud
        if admitted_by_edge.get(int(client_edges.get(client_id, -1)), 0.0) > 0.0
    }


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


def _local_omega_proxy(
    candidate: Candidate,
    aggregation_size: int = 1,
    config: SelectionConfig | None = None,
) -> float:
    config = config or SelectionConfig()
    components = _local_omega_components(candidate, config)
    size = max(float(aggregation_size), 1.0)
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
    update_sigma_sq = float(privacy_parameters["update_noise_multiplier"]) ** 2
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
    return _cached_local_omega_components(
        feature_events,
        client_update_events,
        edge_update_events,
        _candidate_feature_clip_excess_sq(candidate, config),
        config.omega_update_clip_excess_sq,
        config,
    )


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
    t_values = [item.system_latency for item in bounds_source]
    o_values = [item.system_omega for item in bounds_source]
    t_min, t_max = min(t_values), max(t_values)
    o_min, o_max = min(o_values), max(o_values)
    return sorted(
        candidates,
        key=lambda item: (
            max(
                _safe_norm_eps(item.system_latency, t_min, t_max, norm_eps),
                _safe_norm_eps(item.system_omega, o_min, o_max, norm_eps),
            ),
            item.system_latency,
            item.system_omega,
            _evaluation_key(item),
        ),
    )[:limit]


def _pareto_archive(evaluations: list[ProfileEvaluation], limit: int) -> list[ProfileEvaluation]:
    unique: dict[tuple, ProfileEvaluation] = {}
    for item in evaluations:
        key = _evaluation_key(item)
        previous = unique.get(key)
        if previous is None or (item.system_latency, item.system_omega) < (previous.system_latency, previous.system_omega):
            unique[key] = item

    items = sorted(
        unique.values(),
        key=lambda item: (item.system_latency, item.system_omega),
    )
    frontier: list[ProfileEvaluation] = []
    best_omega = float("inf")
    cursor = 0
    while cursor < len(items):
        latency = items[cursor].system_latency
        group_end = cursor + 1
        while group_end < len(items) and items[group_end].system_latency == latency:
            group_end += 1

        min_group_omega = items[cursor].system_omega
        if min_group_omega < best_omega:
            frontier.extend(
                item
                for item in items[cursor:group_end]
                if item.system_omega == min_group_omega
            )
            best_omega = min_group_omega
        cursor = group_end

    frontier = frontier or items
    if len(frontier) <= max(limit, 1):
        return frontier
    if limit <= 2:
        return frontier[:limit]

    by_omega = min(frontier, key=lambda item: (item.system_omega, item.system_latency))
    endpoints = [frontier[0], by_omega]
    middle = [item for item in frontier if item not in endpoints]
    slots = max(limit - len(endpoints), 0)
    if not middle or slots <= 0:
        return endpoints[:limit]
    if slots >= len(middle):
        return sorted(endpoints + middle, key=lambda item: (item.system_latency, item.system_omega))[:limit]
    step = (len(middle) - 1) / max(slots - 1, 1)
    sampled = [middle[round(idx * step)] for idx in range(slots)]
    return sorted(_unique_evaluations(endpoints + sampled), key=lambda item: (item.system_latency, item.system_omega))[:limit]


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
    t_values = [item.system_latency for item in archive]
    o_values = [item.system_omega for item in archive]
    t_min, t_max = min(t_values), max(t_values)
    o_min, o_max = min(o_values), max(o_values)

    def distance(item: ProfileEvaluation) -> tuple[float, float, float, str]:
        t_norm = _safe_norm_eps(item.system_latency, t_min, t_max, norm_eps)
        o_norm = _safe_norm_eps(item.system_omega, o_min, o_max, norm_eps)
        return (
            max(abs(t_norm), abs(o_norm)),
            item.system_latency,
            item.system_omega,
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
    return (
        candidate.mode,
        tuple(sorted((candidate.link_mechanisms or candidate.mechanisms).items())),
    )


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


def _mechanism_assignments(
    spec: ModeSpec,
    policy: str,
    allow_he: bool = True,
    allow_none: bool = False,
    trusted_edge_split_execution: bool = False,
) -> list[tuple[dict[str, str], dict[str, str]]]:
    all_transmissions = _mode_link_transmissions(
        spec.name,
        local_block_cycles=1,
        edge_loops=spec.E_edge_loops,
    )
    transmissions = [event for event in all_transmissions if event[3]]
    links = list(dict.fromkeys(event[0] for event in transmissions))
    link_objects = {event[0]: event[1] for event in all_transmissions}
    trusted_links = {
        link
        for link in links
        if trusted_edge_split_execution
        and link.startswith("L_E_")
    }
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
            return ("dp", "he3", "dp_he3") if allow_he else ("dp",)
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
    privacy_ledger: ClientPrivacyLedger | None = None,
) -> Candidate:
    L = config.L_block_cycles
    E = spec.E_edge_loops

    local_load = spec.local_work * samples / max(config.minibatch_reference_samples, 1e-9)
    if config.assume_encoder_feasible:
        feasible_resource = True
    else:
        feasible_resource = local_load <= config.resource_limit
    sample_scale = samples / max(config.minibatch_reference_samples, 1e-9)
    memory_requirement = spec.local_memory * (0.75 + 0.25 * sample_scale)
    memory_capacity = config.memory_limit * max(float(memory_capacity_factor), 1e-6)
    feasible_memory = memory_requirement <= memory_capacity + 1e-12

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
    link_events = _mode_link_transmissions(mode, L, E)
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

    edge_aggregation_events = E if mode in {"LIEIIIC", "LIIEIIIC"} else int(
        mode in {"LIE", "LIIE"}
    )
    edge_aggregation_time = edge_aggregation_events * (
        config.edge_aggregation_beta * edge_aggregation_payload
        + config.edge_aggregation_fixed
    )
    cloud_aggregation_time = int(_mode_reaches_cloud(spec)) * (
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

    feature_dp_events = sum(
        _record_dp_event_count(config, mode, count)
        for link_id, obj, count, privacy_eligible in link_events
        if privacy_eligible
        and obj != "upd"
        and mechanism_uses_dp(actual_link_mechanisms[link_id])
    )
    update_dp_events = sum(
        count
        for link_id, obj, count, privacy_eligible in link_events
        if privacy_eligible
        and obj == "upd"
        and mechanism_uses_dp(actual_link_mechanisms[link_id])
    )
    feature_epsilon_after = 0.0
    update_epsilon_after = 0.0
    if privacy_ledger is not None:
        projection = privacy_ledger.project(feature_dp_events, update_dp_events)
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
            if privacy_eligible and mechanism_uses_dp(actual_link_mechanisms[link_id])
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
    feasible_time = time <= config.time_limit

    # Edge/cloud CPU feasibility (scaled by sample ratio)
    cpu_scale = samples / 150.0
    mode_edge_demand = spec.edge_cpu * cpu_scale
    mode_cloud_demand = spec.cloud_cpu * cpu_scale
    feasible_edge = (current_edge_load + mode_edge_demand) <= config.edge_cpu_limit + 1e-12
    feasible_cloud = (current_cloud_load + mode_cloud_demand) <= config.cloud_cpu_limit + 1e-12

    # Global penalty: only cloud-reaching objects affect global model accuracy
    protected_links = [event for event in link_events if event[3]]
    mech_penalty = (
        sum(
            utility_penalty(
                actual_link_mechanisms[link_id],
                max(
                    float(
                        resolved_privacy_parameters(config)[
                            "update_budget" if obj == "upd" else "feature_budget"
                        ]
                    ),
                    1e-6,
                ),
            )
            for link_id, obj, _count, _privacy_eligible in protected_links
        )
        / max(len(protected_links), 1)
    )
    penalty = spec.mode_penalty + mech_penalty
    progress = (round_idx + 1.0) / max(config.rounds, 1)
    accuracy = 0.2 + (0.83 - penalty - 0.2) * (1.0 - math.exp(-3.0 * progress))
    accuracy += _candidate_accuracy_jitter(config, client_id, round_idx, mode, actual_link_mechanisms)

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
