"""Bounded single-client-neighbor search and explicit history initialization."""
from dataclasses import dataclass
from itertools import product
import math
import random
import time
from .paper_config import MODES
from .paper_timeline import mode_masses, plan


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


def select(config, policy, pools, history, traces, samples, payload_mb, start,
           round_index, previous, pending, force_reselect=False):
    started = time.perf_counter()
    clients = sorted(pools)
    cache = {}
    missing = set(MODES) - set(history.deviations)
    feasible_missing = missing.intersection({mode for pool in pools.values() for mode in pool})
    initializing = policy in {"proposed", "deviation_only", "exhaustive"} and (
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
    special = None
    if policy in fixed:
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
        phase = "fixed" if policy.startswith("fixed_") else "reuse_or_random"
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
                elif policy == "deviation_only":
                    base = min(candidates, key=lambda x: (math.inf if x.learning is None else x.learning,
                                                         x.latency, identity(x)))
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
            elif policy == "deviation_only":
                chosen = min(valid, key=lambda x: (x.learning, x.latency, identity(x)))
                phase = "deviation_only"
            else:
                chosen = chebyshev(nondominated(valid))
                phase = "pareto"
    front = nondominated(list(cache.values()))
    archive = sorted(front, key=lambda item: (item.latency, item.learning, identity(item)))
    if len(archive) > config.archive_size:
        if config.archive_size == 1:
            archive = [chebyshev(archive)]
        else:
            archive = [archive[round(index*(len(archive)-1)/(config.archive_size-1))]
                       for index in range(config.archive_size)]
    if special is None and not initializing and policy in {"proposed", "exhaustive"} and archive:
        chosen = chebyshev(archive)
    return chosen, {"phase": phase, "evaluated_profiles": len(cache),
                    "feasible_combinations": math.prod(len(pool) for pool in pools.values()),
                    "unconstrained_combinations": 3 ** len(clients),
                    "pareto_front_size": len(front),
                    "archive_size": len(archive),
                    "selection_sec": time.perf_counter()-started,
                    "missing_history": sorted(missing),
                    "coverage": chosen.coverage}
