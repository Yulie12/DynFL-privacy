"""Bounded single-client-neighbor search and explicit history initialization."""
from dataclasses import dataclass
from itertools import product
import math
import random
import time
from .paper_config import MODES, CORE_POLICIES, POLICIES
from .paper_timeline import mode_masses, plan


EIGHT_CANDIDATES = (
    ("Extreme-E", (10, 0, 0)),
    ("Extreme-EC", (0, 10, 0)),
    ("Extreme-C", (0, 0, 10)),
    ("P1", (8, 1, 1)),
    ("P2", (1, 8, 1)),
    ("P3", (1, 1, 8)),
    ("P4", (6, 3, 1)),
    ("P5", (1, 3, 6)),
)


@dataclass
class Evaluation:
    profile: dict
    schedule: object
    latency: float
    learning: float | None
    status: str
    coverage: int


def identity(item):
    return tuple(sorted(item.profile.items()))


def nondominated(items):
    valid = [item for item in items if item.learning is not None]
    return [item for item in valid if not any(
        other.latency <= item.latency and other.learning <= item.learning
        and (other.latency < item.latency or other.learning < item.learning)
        for other in valid)]


def chebyshev(items):
    t0, t1 = min(x.latency for x in items), max(x.latency for x in items)
    j0, j1 = min(x.learning for x in items), max(x.learning for x in items)
    return min(items, key=lambda x: (max((x.latency-t0)/(t1-t0+1e-9),
                                        (x.learning-j0)/(j1-j0+1e-9)),
                                     x.latency, x.learning, identity(x)))


def ratio_counts(clients, weights):
    """Largest-remainder apportionment with stable mode-order ties."""
    total = sum(weights)
    counts = [clients * weight // total for weight in weights]
    remainder = clients - sum(counts)
    order = sorted(range(len(weights)), key=lambda index: (-(clients * weights[index] % total), index))
    for index in order[:remainder]:
        counts[index] += 1
    return dict(zip(MODES, counts))


def _mode_proxy(config, trace, mode, payload_mb, switching):
    """Deterministic additive assignment proxy; final T still uses plan()."""
    switch = config.switch_sec if switching else 0.0
    block = config.local_steps * trace["step_sec"]
    if mode == "FL_C":
        return switch + block + payload_mb / trace["lc_rate"] + config.end_cloud_delay
    stages = config.edge_stages if mode == "FL_EC" else 1
    elapsed = 0.0
    for stage in range(stages):
        returned = 0.0 if stage == 0 else payload_mb / trace["le_rate"] + config.end_edge_delay
        elapsed += returned + block + payload_mb / trace["le_rate"] + config.end_edge_delay
        elapsed += config.edge_aggregation_sec
    return switch + elapsed + payload_mb / trace["ec_rate"] + config.edge_cloud_delay


def assign_ratio(config, pools, traces, payload_mb, previous, counts):
    """Minimum additive-proxy feasible assignment with exact integer quotas.

    The proxy uses the frozen client compute/link trace and switch state. Edge
    grouping, Cloud-unit weights, the global barrier and ACKs are evaluated
    exactly by plan() after this deterministic assignment is constructed.
    """
    clients = sorted(pools)
    targets = (counts["FL_E"], counts["FL_EC"], counts["FL_C"])
    # state=(assigned E, assigned EC); C is implied by the processed count.
    states = {(0, 0): (0.0, ())}
    for position, client in enumerate(clients):
        updated = {}
        for (assigned_e, assigned_ec), (cost, path) in states.items():
            assigned_c = position - assigned_e - assigned_ec
            assigned = (assigned_e, assigned_ec, assigned_c)
            for mode_index, mode in enumerate(MODES):
                if mode not in pools[client] or assigned[mode_index] >= targets[mode_index]:
                    continue
                next_e = assigned_e + (mode == "FL_E")
                next_ec = assigned_ec + (mode == "FL_EC")
                trace = traces[client]
                proxy = _mode_proxy(config, trace, mode, payload_mb,
                                    client in previous and previous[client] != mode)
                candidate = (cost + proxy, path + (mode,))
                key = (next_e, next_ec)
                if key not in updated or candidate < updated[key]:
                    updated[key] = candidate
        states = updated
        if not states:
            return None, None
    target_key = targets[:2]
    if target_key not in states:
        return None, None
    cost, path = states[target_key]
    return dict(zip(clients, path)), cost


def select(config, policy, pools, history, traces, samples, payload_mb, start,
           round_index, previous, pending, force_reselect=False):
    started = time.perf_counter()
    if policy not in POLICIES:
        raise ValueError("unsupported policy; standalone J is not a core method")
    clients = sorted(pools)
    cache = {}
    candidate_metadata = {}
    rejected_candidates = []
    missing = set(MODES) - set(history.deviations)
    feasible_missing = missing.intersection({mode for pool in pools.values() for mode in pool})
    shared_initializing = policy in CORE_POLICIES and round_index < config.shared_init_rounds
    if policy in CORE_POLICIES and not shared_initializing and (history.reference is None or missing):
        raise RuntimeError("shared initialization did not collect all three valid mode histories")
    initializing = policy == "exhaustive" and (
        history.reference is None or bool(feasible_missing))

    def evaluate(profile):
        key = tuple(sorted(profile.items()))
        if key in cache:
            return cache[key]
        schedule = plan(config, profile, traces, samples, payload_mb, start, round_index, previous, pending)
        _, j, status = history.predict(mode_masses(schedule.accepted))
        # Coverage is about actually accepted observations, not selected labels.
        coverage = len({unit.mode for unit in schedule.accepted} & missing)
        value = Evaluation(dict(profile), schedule, schedule.finish-start, j, status, coverage)
        cache[key] = value
        return value

    default = {client: min(pools[client], key=lambda mode: (mode != "FL_C", mode)) for client in clients}
    fixed = {"fixed_fl_e": "FL_E", "fixed_fl_ec": "FL_EC", "fixed_fl_c": "FL_C"}
    staged_eight = (config.experiment_version == "staged_eight_v1"
                    and policy in CORE_POLICIES and not shared_initializing)
    special = None
    if shared_initializing:
        planned = {client: MODES[position % len(MODES)] for position, client in enumerate(clients)}
        if any(mode not in pools[client] for client, mode in planned.items()):
            raise RuntimeError("predetermined shared initialization is infeasible; do not invent history")
        special = planned
    elif policy in fixed:
        special = {client: fixed[policy] if fixed[policy] in pools[client] else "SKIP" for client in clients}
    elif policy == "fixed_mixed":
        special = {client: MODES[client % 3] if MODES[client % 3] in pools[client] else "SKIP" for client in clients}
    elif (not initializing and not force_reselect and previous and round_index % config.selection_period != 0
          and all(previous.get(client) in pools[client] for client in clients)):
        special = dict(previous)
    elif policy == "random_feasible":
        rng = random.Random(config.seeds[0] * 1000003 + round_index)
        special = {client: rng.choice(pools[client]) for client in clients}
    if special is not None:
        chosen = evaluate(special)
        phase = "shared_initialization" if shared_initializing else "fixed" if policy.startswith("fixed_") else "reuse_or_random"
    elif staged_eight:
        for name, weights in EIGHT_CANDIDATES:
            counts = ratio_counts(len(clients), weights)
            profile, proxy_cost = assign_ratio(config, pools, traces, payload_mb, previous, counts)
            if profile is None:
                rejected_candidates.append({"name": name, "weights": dict(zip(MODES, weights)),
                                            "target_counts": counts, "reason": "quota_infeasible"})
                continue
            item = evaluate(profile)
            key = identity(item)
            if key in candidate_metadata:
                raise RuntimeError("distinct staged candidates produced the same concrete profile")
            candidate_metadata[key] = {
                "name": name,
                "ratios": {mode: weight / sum(weights) for mode, weight in zip(MODES, weights)},
                "target_counts": counts,
                "assignment_proxy_cost": proxy_cost,
            }
        if rejected_candidates or len(cache) != len(EIGHT_CANDIDATES):
            raise RuntimeError("staged eight-candidate search could not realize every registered ratio")
        values = list(cache.values())
        front = nondominated(values)
        if policy == "latency_only":
            chosen = min(values, key=lambda item: (item.latency, identity(item)))
            phase = "eight_latency_only"
        else:
            chosen = chebyshev(front)
            phase = "eight_pareto"
    else:
        seeds = [default]
        seeds.extend({client: mode if mode in pools[client] else default[client]
                      for client in clients} for mode in MODES)
        # Deterministic coverage seeds rotate assignment when there are fewer
        # active clients than modes; histories are retained across rounds.
        missing_order = [mode for mode in MODES if mode in missing] or list(MODES)
        for shift in range(len(missing_order)):
            profile = {}
            for position, client in enumerate(clients):
                target = missing_order[(position + shift + round_index) % len(missing_order)]
                profile[client] = target if target in pools[client] else default[client]
            seeds.append(profile)
        if previous:
            seeds.append({client: previous.get(client) if previous.get(client) in pools[client]
                          else default[client] for client in clients})
        budget = config.evaluation_budget
        if policy == "exhaustive":
            combinations = math.prod(len(pools[client]) for client in clients)
            if combinations > config.exhaustive_limit:
                raise ValueError("exhaustive search exceeds configured exhaustive_limit")
            for values in product(*(pools[client] for client in clients)):
                evaluate(dict(zip(clients, values)))
        else:
            for profile in seeds[:budget]:
                evaluate(profile)
            expanded = set()
            while len(cache) < budget:
                candidates = [x for key, x in cache.items() if key not in expanded]
                if not candidates:
                    break
                if initializing:
                    base = min(candidates, key=lambda x: (-x.coverage, x.latency, identity(x)))
                elif policy == "latency_only":
                    base = min(candidates, key=lambda x: (x.latency, identity(x)))
                else:
                    front = nondominated(candidates)
                    base = chebyshev(front) if front else min(candidates, key=lambda x: (x.latency, identity(x)))
                expanded.add(identity(base))
                for client in clients:
                    for mode in pools[client]:
                        if len(cache) >= budget:
                            break
                        trial = dict(base.profile)
                        trial[client] = mode
                        evaluate(trial)
                    if len(cache) >= budget:
                        break
        values = list(cache.values())
        if initializing:
            chosen = min(values, key=lambda x: (-x.coverage, x.latency, identity(x)))
            phase = "history_initialization"
        elif policy == "latency_only":
            chosen = min(values, key=lambda x: (x.latency, identity(x)))
            phase = "latency_only"
        else:
            valid = [x for x in values if x.learning is not None]
            if not valid:
                chosen = min(values, key=lambda x: (-x.coverage, x.latency, identity(x)))
                phase = "history_initialization"
            else:
                chosen = chebyshev(nondominated(valid))
                phase = "pareto"
    front = nondominated(list(cache.values()))
    archive = sorted(front, key=lambda item: (item.latency, item.learning, identity(item)))
    if not staged_eight and len(archive) > config.archive_size:
        if config.archive_size == 1:
            archive = [chebyshev(archive)]
        else:
            archive = [archive[round(index*(len(archive)-1)/(config.archive_size-1))]
                       for index in range(config.archive_size)]
    if special is None and not initializing and policy in {"proposed", "joint_no_ema", "exhaustive"} and archive:
        chosen = chebyshev(archive)
    front_keys = {identity(item) for item in front}
    score_by_key = {}
    rank_by_key = {}
    if archive:
        t0, t1 = min(item.latency for item in archive), max(item.latency for item in archive)
        j0, j1 = min(item.learning for item in archive), max(item.learning for item in archive)
        ranked = sorted(archive, key=lambda item: (
            max((item.latency-t0)/(t1-t0+1e-9), (item.learning-j0)/(j1-j0+1e-9)),
            item.latency, item.learning, identity(item)))
        for rank, item in enumerate(ranked, 1):
            key = identity(item)
            score_by_key[key] = max((item.latency-t0)/(t1-t0+1e-9),
                                    (item.learning-j0)/(j1-j0+1e-9))
            rank_by_key[key] = rank

    def candidate_record(item):
        key = identity(item)
        total = sum(unit.samples for unit in item.schedule.accepted)
        dominators = [candidate_metadata.get(identity(other), {}).get("name") or str(identity(other))
                      for other in cache.values()
                      if other.learning is not None and item.learning is not None
                      and other.latency <= item.latency and other.learning <= item.learning
                      and (other.latency < item.latency or other.learning < item.learning)]
        record = {**candidate_metadata.get(key, {}), "profile": item.profile,
                "T": item.latency, "J": item.learning, "status": item.status,
                "coverage": item.coverage, "masses": mode_masses(item.schedule.accepted),
                "pareto": key in front_keys, "dominated_by": dominators,
                "chebyshev_score": score_by_key.get(key), "pareto_rank": rank_by_key.get(key)}
        if staged_eight:
            record.update({
                "cutoff": item.schedule.cutoff, "finish": item.schedule.finish,
                "cloud_units": [{"identity": unit.identity, "mode": unit.mode, "edge": unit.edge,
                                 "clients": list(unit.clients), "samples": unit.samples,
                                 "weight": unit.samples/total, "arrival": unit.arrival}
                                for unit in item.schedule.accepted],
                "timeline": item.schedule.events,
            })
        return record

    selected_key = identity(chosen)
    return chosen, {"phase": phase, "evaluated_profiles": len(cache),
                    "feasible_combinations": math.prod(len(pool) for pool in pools.values()),
                    "unconstrained_combinations": 3 ** len(clients),
                    "pareto_front_size": len(front),
                    "archive_size": len(archive),
                    "selection_sec": time.perf_counter()-started,
                    "search_protocol": "fixed_eight_ratios_v1" if staged_eight else "bounded_neighbors_v1",
                    "assignment_rule": "deterministic_min_additive_trace_proxy_dp_v1" if staged_eight else None,
                    "missing_history": sorted(missing),
                    "coverage": chosen.coverage,
                    "candidates": [candidate_record(item) for item in cache.values()],
                    "rejected_candidates": rejected_candidates,
                    "archive_profiles": [item.profile for item in archive],
                    "selected_profile": chosen.profile,
                    "selected_candidate": candidate_metadata.get(selected_key, {}).get("name"),
                    "feasible_pools": pools}
