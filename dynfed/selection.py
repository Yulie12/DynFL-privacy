from __future__ import annotations

import csv
import json
import math
import random
from dataclasses import dataclass, replace
from itertools import product
from pathlib import Path
from typing import Any

from .nodes import build_profiles
from .privacy import OBJECT_SIZES, PRIVACY_ALPHA, PRIVACY_BASE_TIME, utility_penalty
from .training import MODE_SPECS, ModeSpec


MECHANISMS_BY_OBJECT = {
    "emb": ("none", "dp"),
    "label": ("none", "dp"),
    "grad": ("none", "dp", "he2"),
    "upd": ("none", "dp", "he3"),
    "weakemb": ("none", "dp"),
    "strongemb": ("none", "dp"),
    "pseudo_label": ("none", "dp"),
}

OBJECT_RISK = {
    "label": 1.0,
    "grad": 0.82,
    "emb": 0.72,
    "strongemb": 0.68,
    "upd": 0.58,
    "weakemb": 0.5,
    "pseudo_label": 0.46,
}

MECHANISM_RISK = {
    "none": 1.0,
    "dp": 0.42,
    "he2": 0.18,
    "he3": 0.12,
}


@dataclass(frozen=True)
class SelectionConfig:
    rounds: int = 20
    num_clients: int = 10
    num_edges: int = 2
    seed: int = 42
    initial_epsilon: float = 1.0
    dp_event_epsilon: float = 0.002
    dp_noise_multiplier: float = 0.001
    dp_delta: float = 1e-5
    client_heterogeneity: float = 2.0
    edge_heterogeneity: float = 1.5
    network_jitter: float = 0.25
    resource_limit: float = 1.35
    time_limit: float = 8.0
    risk_limit: float = 0.5
    aggregation_fraction: float = 0.5
    output_dir: str = "out/selection"
    require_feasible: bool = False
    L_block_cycles: int = 3
    allow_he: bool = True
    assume_encoder_feasible: bool = True
    minibatch_reference_samples: float = 600.0
    edge_cpu_limit: float = 15.0  # per-edge CPU capacity
    cloud_cpu_limit: float = 20.0  # global cloud CPU capacity
    require_cloud_participation: bool = False
    pareto_archive_size: int = 16
    pareto_max_iters: int = 4
    pareto_norm_eps: float = 1e-9
    cloud_fusion_xi: float = 0.06
    cloud_fusion_eps: float = 0.05
    switch_mode_cost: float = 0.02
    switch_placement_cost: float = 0.08


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

    @property
    def feasible(self) -> bool:
        return (
            self.feasible_resource
            and self.feasible_privacy
            and self.feasible_risk
            and self.feasible_edge
            and self.feasible_cloud
        )


@dataclass(frozen=True)
class ProfileEvaluation:
    profile: dict[int, Candidate]
    system_latency: float
    system_omega: float
    cloud_fusion_ratio: float


def run_selection_experiment(config: SelectionConfig, policies: list[str]) -> dict[str, Any]:
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
        remaining_epsilon = {client.client_id: config.initial_epsilon for client in clients}
        policy_rows: list[dict[str, Any]] = []

        for round_idx in range(config.rounds):
            for client in clients:
                rem = remaining_epsilon[client.client_id]
                candidates = enumerate_candidates(
                    config=config,
                    client_id=client.client_id,
                    edge_factor=edge_by_id[client.edge_id].compute_factor,
                    compute_factor=client.compute_factor,
                    samples=client.samples,
                    remaining_epsilon=rem,
                    round_idx=round_idx,
                    rng=rng,
                    policy=policy,
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
                remaining_after = max(0.0, rem - selected.epsilon_used)
                remaining_epsilon[client.client_id] = remaining_after
                row = {
                    "policy": policy,
                    "round": round_idx,
                    "client_id": client.client_id,
                    "edge_id": client.edge_id,
                    "mode": selected.mode,
                    "mechanisms": _mechanism_label(selected.mechanisms),
                    "time": selected.time,
                    "accuracy_estimate": selected.accuracy,
                    "risk": selected.risk,
                    "epsilon_used": selected.epsilon_used,
                    "remaining_epsilon": remaining_after,
                    "communication_volume": selected.communication_volume,
                    "feasible": selected.feasible,
                    "feasible_resource": selected.feasible_resource,
                    "feasible_privacy": selected.feasible_privacy,
                    "feasible_risk": selected.feasible_risk,
                    "feasible_time": selected.feasible_time,
                    "feasible_edge": selected.feasible_edge,
                    "feasible_cloud": selected.feasible_cloud,
                    "feasible_candidates": sum(item.feasible for item in candidates),
                    "total_candidates": len(candidates),
                    "sensitivity": client_sensitivity[client.client_id],
                }
                policy_rows.append(row)
                all_rows.append(row)

        summary_rows.append(_summarize_policy(policy, policy_rows, config.time_limit))

    _write_csv(output_dir / "round_selection.csv", all_rows)
    _write_csv(output_dir / "summary_table.csv", summary_rows)
    _write_json(output_dir / "config.json", config.__dict__ | {"policies": policies})
    return {
        "output_dir": str(output_dir),
        "summary_table": str(output_dir / "summary_table.csv"),
        "round_selection": str(output_dir / "round_selection.csv"),
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
    allow_none: bool = False,
) -> list[Candidate]:
    candidates: list[Candidate] = []
    for mode, spec in MODE_SPECS.items():
        if config.require_cloud_participation and not _mode_reaches_cloud(spec):
            continue
        policy_allow_none = allow_none or policy in {
            "ours",
            "ours_time_first",
            "ours_acc_first",
            "ours_ideal",
            "ours_knee",
            "individual_optimal",
            "random",
            "performance_only",
            "best_accuracy",
            "accuracy_oracle",
        }
        assignments = _mechanism_assignments(spec, policy, config.allow_he, policy_allow_none)
        for mechanisms in assignments:
            candidates.append(
                _estimate_candidate(
                    config=config,
                    mode=mode,
                    spec=spec,
                    mechanisms=mechanisms,
                    client_id=client_id,
                    edge_factor=edge_factor,
                    compute_factor=compute_factor,
                    samples=samples,
                    remaining_epsilon=remaining_epsilon,
                    round_idx=round_idx,
                    rng=rng,
                    current_edge_load=current_edge_load,
                    current_cloud_load=current_cloud_load,
                )
            )
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
    resource_feasible = [item for item in candidates if item.feasible_resource]
    if not resource_feasible:
        return skipped_candidate()

    feasible = [item for item in candidates if item.feasible]
    if require_feasible and not feasible:
        return skipped_candidate()
    pool = feasible or resource_feasible

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
    if policy == "fixed_liieiiic":
        fixed_pool = [c for c in pool if c.mode == "LIIEIIIC"]
        if not fixed_pool:
            return skipped_candidate()
        return max(fixed_pool, key=lambda item: (item.feasible, item.accuracy, -item.time, -item.risk))
    if policy == "performance_only":
        return max(resource_feasible or candidates, key=lambda item: item.accuracy)
    if policy == "privacy_only":
        return min(pool, key=lambda item: (item.risk, item.epsilon_used, item.time))
    if policy == "no_protection":
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
    if policy in {"fixed_dp", "fixed_he"}:
        return max(pool, key=lambda item: (item.accuracy, -item.time))

    # "ours": Pareto frontier + sensitivity-weighted ideal point selection.
    # Hard constraints (resource, privacy, risk, edge/cloud CPU) filter the
    # candidate set. The Pareto frontier captures the accuracy-vs-time tradeoff
    # among feasible candidates. The sensitivity-weighted ideal point selects
    # the best compromise: high sensitivity → time-optimal, low sensitivity →
    # accuracy-optimal. Mode bonus (UCB exploration) adjusts effective accuracy
    # so that untried or historically promising modes are explored.
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
) -> tuple[list[tuple[int, Candidate, list[Candidate], float]], ProfileEvaluation]:
    """Approximate TeX Algorithm 1 over a global client profile.

    Each client contributes a feasible candidate set S_i. The search keeps a
    bounded non-dominated archive over (T_sys, Omega_sys), expands profiles by
    changing one client at a time, then selects the archive profile closest to
    the normalized ideal point by Tchebycheff distance.
    """
    if not selected:
        empty = ProfileEvaluation({}, 0.0, 0.0, 0.0)
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
                if item.feasible_resource and item.epsilon_used <= remaining + 1e-12
            ]
        if not pool:
            pool = [current if current.feasible_resource else skipped_candidate()]
        pools[client_id] = _dedupe_candidates(pool)

    seeds = _initial_profiles(pools, previous_choices)
    archive = _pareto_archive(
        [_evaluate_profile(config, item, client_samples, client_edges, previous_choices) for item in seeds],
        config.pareto_archive_size,
    )

    for _iter_idx in range(max(0, config.pareto_max_iters)):
        neighbors: list[dict[int, Candidate]] = []
        seen = {_profile_key(item.profile) for item in archive}
        for evaluated in archive:
            for client_id, candidates in pools.items():
                current = evaluated.profile[client_id]
                for candidate in candidates:
                    if candidate == current:
                        continue
                    profile = dict(evaluated.profile)
                    profile[client_id] = candidate
                    key = _profile_key(profile)
                    if key in seen:
                        continue
                    seen.add(key)
                    neighbors.append(profile)
        if not neighbors:
            break

        expanded = archive + [
            _evaluate_profile(config, item, client_samples, client_edges, previous_choices)
            for item in neighbors
        ]
        next_archive = _pareto_archive(expanded, config.pareto_archive_size)
        if {_profile_key(item.profile) for item in next_archive} == {_profile_key(item.profile) for item in archive}:
            archive = next_archive
            break
        archive = next_archive

    chosen = _choose_tchebycheff(archive, config.pareto_norm_eps)
    rewritten = [
        (client_id, chosen.profile[client_id], by_client[client_id][1], by_client[client_id][2])
        for client_id, _current, _candidates, _remaining in selected
    ]
    return rewritten, chosen


def _initial_profiles(
    pools: dict[int, list[Candidate]],
    previous_choices: dict[int, Candidate],
) -> list[dict[int, Candidate]]:
    fastest = {client_id: min(candidates, key=lambda item: (item.time, _local_omega_proxy(item))) for client_id, candidates in pools.items()}
    lowest_omega = {client_id: min(candidates, key=lambda item: (_local_omega_proxy(item), item.time)) for client_id, candidates in pools.items()}
    previous = {}
    for client_id, candidates in pools.items():
        prior = previous_choices.get(client_id)
        previous[client_id] = next(
            (
                item for item in candidates
                if prior is not None and item.mode == prior.mode and item.mechanisms == prior.mechanisms
            ),
            fastest[client_id],
        )
    return _unique_profiles([previous, fastest, lowest_omega])


def _evaluate_profile(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
    previous_choices: dict[int, Candidate],
) -> ProfileEvaluation:
    latencies = [
        _candidate_latency_with_switch(config, client_id, candidate, previous_choices.get(client_id))
        for client_id, candidate in profile.items()
        if candidate.mode != "SKIP"
    ]
    system_latency = max(latencies, default=0.0)
    system_omega, cloud_fusion_ratio = _global_omega_proxy(config, profile, client_samples, client_edges)
    return ProfileEvaluation(
        profile=profile,
        system_latency=system_latency,
        system_omega=system_omega,
        cloud_fusion_ratio=cloud_fusion_ratio,
    )


def _candidate_latency_with_switch(
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


def _placement_distance(previous_mode: str, mode: str) -> float:
    prev = _placement_vector(previous_mode)
    curr = _placement_vector(mode)
    return sum(abs(a - b) for a, b in zip(prev, curr)) / 2.0


def _placement_vector(mode: str) -> tuple[float, float, float]:
    spec = MODE_SPECS.get(mode)
    if spec is None:
        return (1.0, 0.0, 0.0)
    local = max(spec.local_work, 0.0)
    edge = max(spec.edge_work + spec.edge_cpu, 0.0)
    cloud = max(spec.cloud_work + spec.cloud_cpu, 0.0)
    total = max(local + edge + cloud, 1e-12)
    return (local / total, edge / total, cloud / total)


def _global_omega_proxy(
    config: SelectionConfig,
    profile: dict[int, Candidate],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
) -> tuple[float, float]:
    total_samples = sum(float(client_samples.get(client_id, 1.0)) for client_id in profile)
    if total_samples <= 0:
        total_samples = float(max(len(profile), 1))
    aggregation_sizes = _aggregation_sizes(profile, client_edges)
    weighted_local = sum(
        float(client_samples.get(client_id, 1.0))
        / total_samples
        * _local_omega_proxy(candidate, aggregation_size=aggregation_sizes.get(client_id, 1))
        for client_id, candidate in profile.items()
    )
    cloud_samples = sum(
        float(client_samples.get(client_id, 1.0))
        for client_id, candidate in profile.items()
        if _candidate_reaches_cloud(candidate)
    )
    cloud_fusion_ratio = cloud_samples / total_samples
    return (
        weighted_local + config.cloud_fusion_xi / (cloud_fusion_ratio + config.cloud_fusion_eps),
        cloud_fusion_ratio,
    )


def _aggregation_sizes(profile: dict[int, Candidate], client_edges: dict[int, int]) -> dict[int, int]:
    cloud_count = sum(1 for item in profile.values() if _candidate_reaches_cloud(item))
    edge_counts: dict[int, int] = {}
    for client_id, candidate in profile.items():
        if _candidate_reaches_cloud(candidate):
            continue
        edge_id = client_edges.get(client_id, -1)
        edge_counts[edge_id] = edge_counts.get(edge_id, 0) + 1
    return {
        client_id: max(cloud_count if _candidate_reaches_cloud(candidate) else edge_counts.get(client_edges.get(client_id, -1), 1), 1)
        for client_id, candidate in profile.items()
    }


def _local_omega_proxy(candidate: Candidate, aggregation_size: int = 1) -> float:
    spec = MODE_SPECS.get(candidate.mode)
    mode_residual = float(spec.mode_penalty) if spec is not None else 0.0
    feature_objects = {"emb", "label", "grad", "weakemb", "strongemb", "pseudo_label"}
    feature_dp = any(obj in feature_objects and mechanism == "dp" for obj, mechanism in candidate.mechanisms.items())
    update_dp = any(obj == "upd" and mechanism == "dp" for obj, mechanism in candidate.mechanisms.items())

    base_variance = 0.035
    feature_clipping_bias = 0.08 if feature_dp else 0.0
    feature_loss_inflation = 0.04 if feature_dp else 0.0
    update_noise = 0.12 / max(float(aggregation_size), 1.0) if update_dp else 0.0
    return mode_residual + base_variance + feature_clipping_bias + feature_loss_inflation + update_noise


def _pareto_archive(evaluations: list[ProfileEvaluation], limit: int) -> list[ProfileEvaluation]:
    unique: dict[tuple, ProfileEvaluation] = {}
    for item in evaluations:
        key = _profile_key(item.profile)
        previous = unique.get(key)
        if previous is None or (item.system_latency, item.system_omega) < (previous.system_latency, previous.system_omega):
            unique[key] = item

    items = list(unique.values())
    frontier = []
    for item in items:
        dominated = False
        for other in items:
            if other is item:
                continue
            no_worse = other.system_latency <= item.system_latency and other.system_omega <= item.system_omega
            strictly_better = other.system_latency < item.system_latency or other.system_omega < item.system_omega
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            frontier.append(item)

    frontier = sorted(frontier or items, key=lambda item: (item.system_latency, item.system_omega))
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


def _choose_tchebycheff(archive: list[ProfileEvaluation], norm_eps: float) -> ProfileEvaluation:
    if not archive:
        return ProfileEvaluation({}, 0.0, 0.0, 0.0)
    t_values = [item.system_latency for item in archive]
    o_values = [item.system_omega for item in archive]
    t_min, t_max = min(t_values), max(t_values)
    o_min, o_max = min(o_values), max(o_values)

    def distance(item: ProfileEvaluation) -> tuple[float, float, float]:
        t_norm = _safe_norm_eps(item.system_latency, t_min, t_max, norm_eps)
        o_norm = _safe_norm_eps(item.system_omega, o_min, o_max, norm_eps)
        return (max(abs(t_norm), abs(o_norm)), item.system_latency, item.system_omega)

    return min(archive, key=distance)


def _safe_norm_eps(value: float, low: float, high: float, eps: float) -> float:
    return (value - low) / (high - low + eps)


def _profile_key(profile: dict[int, Candidate]) -> tuple:
    return tuple(
        (client_id, candidate.mode, tuple(sorted(candidate.mechanisms.items())))
        for client_id, candidate in sorted(profile.items())
    )


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
        key = _profile_key(item.profile)
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


def _mechanism_assignments(spec: ModeSpec, policy: str, allow_he: bool = True, allow_none: bool = False) -> list[dict[str, str]]:
    objects = sorted(set(spec.client_objects + spec.edge_to_cloud_objects))
    if policy == "no_protection":
        return [{obj: "none" for obj in objects}]
    if policy == "fixed_dp":
        return [{obj: "dp" for obj in objects}]
    if policy == "fixed_he":
        return [{obj: _prefer_he(obj) for obj in objects}]

    choices = [
        tuple(
            mech for mech in MECHANISMS_BY_OBJECT[obj]
            if (allow_he or not mech.startswith("he"))
            and (allow_none or mech != "none")
        )
        for obj in objects
    ]
    return [dict(zip(objects, values)) for values in product(*choices)]


def _estimate_candidate(
    *,
    config: SelectionConfig,
    mode: str,
    spec: ModeSpec,
    mechanisms: dict[str, str],
    client_id: int,
    edge_factor: float,
    compute_factor: float,
    samples: int,
    remaining_epsilon: float,
    round_idx: int,
    rng: random.Random,
    current_edge_load: float = 0.0,
    current_cloud_load: float = 0.0,
) -> Candidate:
    L = config.L_block_cycles
    E = spec.E_edge_loops

    local_load = spec.local_work * samples / max(config.minibatch_reference_samples, 1e-9)
    if config.assume_encoder_feasible:
        feasible_resource = True
    else:
        feasible_resource = local_load <= config.resource_limit

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
    privacy_time = sum(PRIVACY_BASE_TIME[mechanisms[obj]] for obj in mechanisms)

    bandwidth = _round_bandwidth(spec.client_target, round_idx, rng, config.network_jitter)
    communication_volume = sum(
        OBJECT_SIZES[obj] * PRIVACY_ALPHA[mechanisms[obj]] for obj in mechanisms
    )
    communication_time = communication_volume / bandwidth

    # Edge aggregation overhead (per aggregation event)
    edge_agg = 0.08 * len(mechanisms)

    # Time: E × (L×block + comm + privacy + agg) + cloud (tex eq. 316-321)
    if spec.client_target == "edge":
        if E > 1:
            # Multi-level: secondary edge aggregation loop
            edge_loop = E * (block_compute + communication_time + privacy_time + edge_agg)
            cloud_bw = _round_bandwidth("cloud", round_idx, rng, config.network_jitter)
            cloud_comm = communication_volume / max(0.05, cloud_bw)
            time = edge_loop + cloud_comm + cloud_time
        else:
            # Single-level: one edge aggregation
            time = block_compute + communication_time + privacy_time + edge_agg + cloud_time
    else:
        # Cloud-targeted: L block cycles then cloud aggregation
        time = block_compute + communication_time + privacy_time + cloud_time + 0.12

    # Privacy budget: mode-dependent cost (alpha × base_cost per DP event)
    # Separate from formal RDP accounting — used only for mode selection feasibility
    has_dp = any(mech == "dp" for mech in mechanisms.values())
    epsilon_used = spec.alpha * config.dp_event_epsilon if has_dp else 0.0
    feasible_privacy = epsilon_used <= remaining_epsilon + 1e-12

    risk = max((OBJECT_RISK[obj] * MECHANISM_RISK[mech] for obj, mech in mechanisms.items()), default=0.0)
    feasible_risk = risk <= config.risk_limit
    feasible_time = time <= config.time_limit

    # Edge/cloud CPU feasibility (scaled by sample ratio)
    cpu_scale = samples / 150.0
    mode_edge_demand = spec.edge_cpu * cpu_scale
    mode_cloud_demand = spec.cloud_cpu * cpu_scale
    feasible_edge = (current_edge_load + mode_edge_demand) <= config.edge_cpu_limit + 1e-12
    feasible_cloud = (current_cloud_load + mode_cloud_demand) <= config.cloud_cpu_limit + 1e-12

    # Global penalty: only cloud-reaching objects affect global model accuracy
    if spec.edge_to_cloud_objects:
        cloud_objs = spec.edge_to_cloud_objects
    elif spec.client_target == "cloud":
        cloud_objs = spec.client_objects
    else:
        cloud_objs = []
    mech_penalty = (
        sum(utility_penalty(mechanisms.get(obj, "none"), max(config.dp_event_epsilon, 1e-6)) for obj in cloud_objs)
        / max(len(cloud_objs), 1)
    )
    penalty = spec.mode_penalty + mech_penalty
    progress = (round_idx + 1.0) / max(config.rounds, 1)
    accuracy = 0.2 + (0.83 - penalty - 0.2) * (1.0 - math.exp(-3.0 * progress))
    accuracy += rng.uniform(-0.002, 0.002)

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
    )


def _round_bandwidth(target: str, round_idx: int, rng: random.Random, jitter: float) -> float:
    base = 5.0 if target == "edge" else 2.2
    periodic = 1.0 + 0.2 * math.sin(round_idx / 3.0)
    random_part = rng.uniform(max(0.05, 1.0 - jitter), 1.0 + jitter)
    return max(0.05, base * periodic * random_part)


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
        "total_epsilon_used": sum(row["epsilon_used"] for row in rows),
        "feasible_rate": _mean(float(row.get("feasible_resource", row["feasible"])) for row in rows),
        "all_constraint_feasible_rate": _mean(float(row["feasible"]) for row in rows),
        "resource_feasible_rate": _mean(float(row.get("feasible_resource", row["feasible"])) for row in rows),
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
