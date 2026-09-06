from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.nodes import build_profiles
from dynfed.selection import (
    Candidate,
    ProfileEvaluation,
    SelectionConfig,
    _candidate_key,
    _choose_tchebycheff,
    _evaluation_key,
    _local_omega_proxy,
    _pareto_archive,
    _profile_satisfies_edge_cloud_coverage,
    _stable_cloud_candidate_pool,
    build_client_privacy_ledger,
    choose_global_pareto_profile,
    enumerate_candidates,
    evaluate_global_profile,
)
from dynfed.version import CURRENT_EXECUTION_REVISION


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare bounded Pareto search with exact enumeration on small instances."
    )
    parser.add_argument("--client-sizes", default="2,3,4,5")
    parser.add_argument("--seeds", default="40,42,44")
    parser.add_argument("--candidates-per-client", type=int, default=4)
    parser.add_argument("--archive-size", type=int, default=16)
    parser.add_argument("--max-iters", type=int, default=50)
    parser.add_argument("--output-dir", default="out/pareto_validation_v25")
    return parser.parse_args()


def _parse_ints(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _candidate_pool(candidates: list[Candidate], limit: int) -> list[Candidate]:
    feasible = [candidate for candidate in candidates if candidate.feasible]
    source = feasible or [candidate for candidate in candidates if candidate.feasible_device]
    if not source:
        source = list(candidates)
    cloud = [
        candidate
        for candidate in source
        if candidate.mode in {"LIC", "LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC"}
    ]
    ranked_lists = (
        sorted(source, key=lambda item: (item.time, _local_omega_proxy(item))),
        sorted(source, key=lambda item: (_local_omega_proxy(item), item.time)),
        sorted(cloud, key=lambda item: (item.time, _local_omega_proxy(item))),
        sorted(cloud, key=lambda item: (_local_omega_proxy(item), item.time)),
        sorted(source, key=lambda item: (item.communication_volume, item.time)),
        sorted(
            source,
            key=lambda item: (
                not any(
                    str(value).startswith("he")
                    for value in (item.link_mechanisms or item.mechanisms).values()
                ),
                item.time,
            ),
        ),
    )
    chosen: list[Candidate] = []
    seen: set[tuple[Any, ...]] = set()
    for rank in range(max(len(items) for items in ranked_lists)):
        for items in ranked_lists:
            if rank >= len(items):
                continue
            candidate = items[rank]
            key = _candidate_key(candidate)
            if key in seen:
                continue
            seen.add(key)
            chosen.append(candidate)
            if len(chosen) >= max(1, limit):
                return chosen
    return chosen


def _normalized_point(
    evaluation: ProfileEvaluation,
    exact_front: list[ProfileEvaluation],
) -> tuple[float, float]:
    t_values = [item.system_latency for item in exact_front]
    o_values = [item.system_omega for item in exact_front]
    t_low, t_high = min(t_values), max(t_values)
    o_low, o_high = min(o_values), max(o_values)
    t = (evaluation.system_latency - t_low) / max(t_high - t_low, 1e-12)
    o = (evaluation.system_omega - o_low) / max(o_high - o_low, 1e-12)
    return t, o


def _hypervolume(
    front: list[ProfileEvaluation],
    exact_front: list[ProfileEvaluation],
    reference: tuple[float, float] = (1.1, 1.1),
) -> float:
    points = sorted({_normalized_point(item, exact_front) for item in front})
    if not points:
        return 0.0
    area = 0.0
    best_y = reference[1]
    for index, (x, y) in enumerate(points):
        best_y = min(best_y, y)
        next_x = points[index + 1][0] if index + 1 < len(points) else reference[0]
        area += max(0.0, next_x - x) * max(0.0, reference[1] - best_y)
    return area


def _tchebycheff_distance(
    evaluation: ProfileEvaluation,
    exact_front: list[ProfileEvaluation],
) -> float:
    t, o = _normalized_point(evaluation, exact_front)
    return max(abs(t), abs(o))


def _objective_key(evaluation: ProfileEvaluation) -> tuple[float, float]:
    return round(evaluation.system_latency, 10), round(evaluation.system_omega, 10)


def _exact_evaluations(
    *,
    config: SelectionConfig,
    pools: dict[int, list[Candidate]],
    client_samples: dict[int, float],
    client_edges: dict[int, int],
) -> list[ProfileEvaluation]:
    evaluations: list[ProfileEvaluation] = []
    client_ids = sorted(pools)
    for values in itertools.product(*(pools[client_id] for client_id in client_ids)):
        profile = dict(zip(client_ids, values))
        if not _profile_satisfies_edge_cloud_coverage(
            config,
            profile,
            client_samples,
            client_edges,
        ):
            continue
        selected = [
            (client_id, profile[client_id], pools[client_id], config.initial_epsilon)
            for client_id in client_ids
        ]
        evaluations.append(
            evaluate_global_profile(
                config=config,
                selected=selected,
                client_samples=client_samples,
                client_edges=client_edges,
            )
        )
    if not evaluations:
        raise RuntimeError("The exact validation instance has no feasible global profile")
    return evaluations


def run_instance(
    *,
    num_clients: int,
    seed: int,
    candidates_per_client: int,
    archive_size: int,
    max_iters: int,
) -> dict[str, Any]:
    num_edges = min(2, num_clients)
    config = SelectionConfig(
        rounds=200,
        num_clients=num_clients,
        num_edges=num_edges,
        seed=seed,
        time_limit=300.0,
        require_feasible=True,
        allow_he=True,
        require_edge_cloud_coverage=True,
        min_edge_cloud_fusion_ratio=0.5,
        enforce_cloud_dp_stability=True,
        pareto_archive_size=archive_size,
        pareto_max_iters=max_iters,
    )
    clients, edges = build_profiles(
        num_clients=num_clients,
        num_edges=num_edges,
        client_heterogeneity=config.client_heterogeneity,
        edge_heterogeneity=config.edge_heterogeneity,
        seed=seed,
    )
    edge_by_id = {edge.edge_id: edge for edge in edges}
    pools: dict[int, list[Candidate]] = {}
    for client in clients:
        rng = random.Random(seed * 1009 + client.client_id)
        ledger = build_client_privacy_ledger(config)
        candidates = enumerate_candidates(
            config=config,
            client_id=client.client_id,
            edge_factor=edge_by_id[client.edge_id].compute_factor,
            compute_factor=client.compute_factor,
            memory_capacity_factor=client.memory_capacity_factor,
            samples=client.samples,
            remaining_epsilon=ledger.remaining_budget,
            round_idx=0,
            rng=rng,
            policy="ours",
            privacy_ledger=ledger,
        )
        pools[client.client_id] = _candidate_pool(
            _stable_cloud_candidate_pool(config, candidates),
            candidates_per_client,
        )

    selected = [
        (client_id, pool[0], pool, config.initial_epsilon)
        for client_id, pool in sorted(pools.items())
    ]
    client_samples = {client.client_id: float(client.samples) for client in clients}
    client_edges = {client.client_id: int(client.edge_id) for client in clients}
    diagnostics: dict[str, Any] = {}
    _, approximate_choice = choose_global_pareto_profile(
        config=config,
        selected=selected,
        client_samples=client_samples,
        client_edges=client_edges,
        diagnostics=diagnostics,
    )
    approximate_front = list(diagnostics["archive"])

    exact_evaluations = _exact_evaluations(
        config=config,
        pools=pools,
        client_samples=client_samples,
        client_edges=client_edges,
    )
    exact_front = _pareto_archive(exact_evaluations, len(exact_evaluations))
    exact_choice = _choose_tchebycheff(exact_front, config.pareto_norm_eps)
    exact_profile_keys = {_evaluation_key(item) for item in exact_front}
    approximate_profile_keys = {_evaluation_key(item) for item in approximate_front}
    exact_objective_keys = {_objective_key(item) for item in exact_front}
    approximate_objective_keys = {_objective_key(item) for item in approximate_front}
    exact_hv = _hypervolume(exact_front, exact_front)
    approximate_hv = _hypervolume(approximate_front, exact_front)
    exact_distance = _tchebycheff_distance(exact_choice, exact_front)
    approximate_distance = _tchebycheff_distance(approximate_choice, exact_front)
    total_profile_count = math.prod(len(pool) for pool in pools.values())

    return {
        "num_clients": num_clients,
        "num_edges": num_edges,
        "seed": seed,
        "candidates_per_client": candidates_per_client,
        "total_profile_count": total_profile_count,
        "exact_profile_count": len(exact_evaluations),
        "approximate_evaluated_profile_count": int(diagnostics["evaluated_profile_count"]),
        "exact_front_size": len(exact_front),
        "approximate_front_size": len(approximate_front),
        "pareto_profile_recall": len(exact_profile_keys & approximate_profile_keys)
        / max(len(exact_profile_keys), 1),
        "pareto_objective_recall": len(exact_objective_keys & approximate_objective_keys)
        / max(len(exact_objective_keys), 1),
        "hypervolume_gap": max(0.0, exact_hv - approximate_hv) / max(exact_hv, 1e-12),
        "tchebycheff_regret": max(0.0, approximate_distance - exact_distance),
        "decision_profile_match": int(
            _evaluation_key(approximate_choice) == _evaluation_key(exact_choice)
        ),
        "decision_objective_match": int(
            _objective_key(approximate_choice) == _objective_key(exact_choice)
        ),
        "evaluation_reduction_ratio": 1.0
        - int(diagnostics["evaluated_profile_count"]) / max(total_profile_count, 1),
        "exact_choice_latency": exact_choice.system_latency,
        "exact_choice_omega": exact_choice.system_omega,
        "approximate_choice_latency": approximate_choice.system_latency,
        "approximate_choice_omega": approximate_choice.system_omega,
    }


def _mean(rows: list[dict[str, Any]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / max(len(rows), 1)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = [
        run_instance(
            num_clients=num_clients,
            seed=seed,
            candidates_per_client=max(2, args.candidates_per_client),
            archive_size=max(2, args.archive_size),
            max_iters=max(0, args.max_iters),
        )
        for num_clients in _parse_ints(args.client_sizes)
        for seed in _parse_ints(args.seeds)
    ]
    with (output_dir / "pareto_validation.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "instances": len(rows),
        "mean_pareto_profile_recall": _mean(rows, "pareto_profile_recall"),
        "mean_pareto_objective_recall": _mean(rows, "pareto_objective_recall"),
        "mean_hypervolume_gap": _mean(rows, "hypervolume_gap"),
        "mean_tchebycheff_regret": _mean(rows, "tchebycheff_regret"),
        "decision_profile_match_rate": _mean(rows, "decision_profile_match"),
        "decision_objective_match_rate": _mean(rows, "decision_objective_match"),
        "mean_evaluation_reduction_ratio": _mean(rows, "evaluation_reduction_ratio"),
        "execution_revision": CURRENT_EXECUTION_REVISION,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
