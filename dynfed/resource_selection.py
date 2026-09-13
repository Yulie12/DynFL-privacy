"""Stage-one public-profile placement search; no accuracy objective or DP choice."""
import math
import random

DEFAULT_PROFILE = dict(source="analytical_assumption_not_device_measurement",
                       end_sec_per_sample=0.0004, tail_sec_per_sample=0.0006,
                       rate_mb_s=5.0, base_latency_sec=0.015, jitter=0.25,
                       client_heterogeneity=2.0, edge_heterogeneity=1.5,
                       edge_work_limit_sec=30.0, communication_weight=0.1,
                       resource_weight=0.1, search_passes=3)


def select_placement(counts, edges, epochs, embedding_values, update_values, round_index, seed, profile):
    """All inputs are public quotas, model dimensions, or resource assumptions.

    Latency is a conservative upload/compute phase model. Common HE/release costs
    are excluded from ranking and measured separately by the execution engine.
    """
    p = dict(profile)
    if set(p) != set(DEFAULT_PROFILE):
        raise ValueError("Explicit complete public cost profile required")
    for key, value in p.items():
        if key != "source" and (not math.isfinite(value) or value < 0):
            raise ValueError("Invalid cost profile")
    if min(p["rate_mb_s"], p["edge_work_limit_sec"], p["client_heterogeneity"], p["edge_heterogeneity"]) <= 0:
        raise ValueError("Positive rates and capacities required")
    if type(p["search_passes"]) is not int or p["search_passes"] < 1 or not 0 <= p["jitter"] < 1:
        raise ValueError("Invalid search bound or jitter")
    if not counts or min(counts) <= 0 or not 1 <= edges <= len(counts):
        raise ValueError("Invalid public cohort")
    rng = random.Random(seed)
    clients = [rng.uniform(1 / p["client_heterogeneity"], p["client_heterogeneity"]) for _ in counts]
    speeds = [rng.uniform(1 / p["edge_heterogeneity"], p["edge_heterogeneity"]) for _ in range(edges)]
    network = random.Random(seed + 1000003 * (round_index + 1))
    options = []
    for i, n in enumerate(counts):
        rate = p["rate_mb_s"] * network.uniform(1 - p["jitter"], 1 + p["jitter"])
        end_work = n * epochs * p["end_sec_per_sample"] / clients[i]
        tail_local = n * epochs * p["tail_sec_per_sample"] / clients[i]
        tail_edge = n * epochs * p["tail_sec_per_sample"] / speeds[i % edges]
        update_mb = update_values * 4 / 1e6
        split_mb = n * epochs * (2 * embedding_values + 20) * 4 / 1e6
        options.append(((end_work + tail_local + update_mb / rate + p["base_latency_sec"],
                         0.0, update_mb, end_work + tail_local),
                        (end_work + split_mb / rate + 2 * epochs * p["base_latency_sec"],
                         tail_edge, split_mb, end_work + tail_edge)))

    def metrics(bits):
        arrivals, loads = [0.0] * edges, [0.0] * edges
        communication = resource = 0.0
        for i, bit in enumerate(bits):
            arrival, work, volume, compute = options[i][bit]
            edge = i % edges
            arrivals[edge] = max(arrivals[edge], arrival)
            loads[edge] += work
            communication += volume
            resource += compute
        return max(a + b for a, b in zip(arrivals, loads)), communication, resource, max(loads)

    reference = metrics([0] * len(counts))

    def score(bits):
        latency, communication, resource, load = metrics(bits)
        if load > p["edge_work_limit_sec"]:
            return float("inf")
        return (latency / max(reference[0], 1e-12)
                + p["communication_weight"] * communication / max(reference[1], 1e-12)
                + p["resource_weight"] * resource / max(reference[2], 1e-12))

    best = [0] * len(counts)
    evaluations = 0
    for initial in (0, 1):
        bits = [initial] * len(counts)
        for _ in range(p["search_passes"]):
            changed = False
            for i in range(len(bits)):
                before = score(bits)
                bits[i] = 1 - bits[i]
                after = score(bits)
                evaluations += 2
                if after + 1e-12 < before:
                    changed = True
                else:
                    bits[i] = 1 - bits[i]
            if not changed:
                break
        if score(bits) < score(best):
            best = bits
    latency, comm, work, load = metrics(best)
    return {i: "LIEIIC" if bit else "LIIC" for i, bit in enumerate(best)}, dict(
        selector="resource_placement_v1_not_original_pareto", score=score(best),
        modeled_variable_phase_sec=latency, modeled_client_edge_mb=comm,
        modeled_compute_sec=work, max_edge_work_sec=load,
        resource_feasible=load <= p["edge_work_limit_sec"], evaluations=evaluations,
        accuracy_objective_used=False, profile_source=p["source"],
        common_he_release_cost="not_ranked_measured_separately", participation="fixed_all_clients")
