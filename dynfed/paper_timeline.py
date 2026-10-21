"""Common absolute timeline, with persistent asynchronous FL_E Cloud units."""
from dataclasses import dataclass, field
import torch


@dataclass(frozen=True)
class Unit:
    identity: str
    mode: str
    edge: int
    clients: tuple[int, ...]
    samples: int
    arrival: float
    update: torch.Tensor | None = field(default=None, compare=False)
    generated_round: int = 0


@dataclass
class Schedule:
    units: list[Unit]
    accepted: list[Unit]
    pending: list[Unit]
    cutoff: float | None
    finish: float
    synchronous: bool
    client_finish: dict[int, float]
    events: list[dict]


def plan(config, profile, traces, samples, payload_mb, start, round_index,
         previous=None, pending=()):
    """Plan exact sequential Edge stages and a Cloud acceptance cutoff.

    FL_E never contributes a dependency to a synchronous Cloud cutoff or return
    maximum. In an FL_E-only iteration, the earliest asynchronous arrival is an
    explicit event-driven Cloud trigger. Late units remain in pending.
    """
    previous = previous or {}
    groups = {}
    units, events, completion = [], [], {}
    sync_arrivals, sync_returns = [], []
    for client, mode in sorted(profile.items()):
        if mode == "SKIP":
            continue
        trace = traces[client]
        switch = config.switch_sec if client in previous and previous[client] != mode else 0.0
        block = config.local_steps * trace["step_sec"]
        if mode == "FL_C":
            arrival = start + switch + block + payload_mb / trace["lc_rate"] + config.end_cloud_delay
            unit = Unit(f"r{round_index}:FL_C:c{client}", mode, trace["edge"], (client,),
                        samples[client], arrival, generated_round=round_index)
            units.append(unit)
            sync_arrivals.append(arrival)
            sync_returns.append(payload_mb / trace["lc_rate"] + config.end_cloud_delay)
        else:
            groups.setdefault((trace["edge"], mode), []).append(client)
    for (edge, mode), members in sorted(groups.items()):
        last_finish = start
        stages = config.edge_stages if mode == "FL_EC" else 1
        for stage in range(stages):
            arrivals = []
            for client in members:
                trace = traces[client]
                switch = config.switch_sec if stage == 0 and client in previous and previous[client] != mode else 0
                returned = 0 if stage == 0 else payload_mb / trace["le_rate"] + config.end_edge_delay
                arrivals.append(last_finish + switch + returned + config.local_steps * trace["step_sec"]
                                + payload_mb / trace["le_rate"] + config.end_edge_delay)
            agg_start = max(arrivals)
            last_finish = agg_start + config.edge_aggregation_sec
            events.append({"event": "edge_stage", "edge": edge, "mode": mode,
                           "stage": stage + 1, "start": agg_start, "finish": last_finish,
                           "clients": members})
        ec_rate = min(traces[client]["ec_rate"] for client in members)
        arrival = last_finish + payload_mb / ec_rate + config.edge_cloud_delay
        unit = Unit(f"r{round_index}:{mode}:e{edge}", mode, edge, tuple(members),
                    sum(samples[client] for client in members), arrival,
                    generated_round=round_index)
        units.append(unit)
        if mode == "FL_EC":
            sync_arrivals.append(arrival)
            sync_returns.extend(payload_mb / ec_rate + config.edge_cloud_delay
                                + payload_mb / traces[client]["le_rate"] + config.end_edge_delay
                                for client in members)
        else:
            for client in members:
                completion[client] = last_finish + payload_mb / traces[client]["le_rate"] + config.end_edge_delay
            events.append({"event": "async_upload", "identity": unit.identity, "arrival": arrival})
    candidates = list(pending) + units
    synchronous = bool(sync_arrivals)
    if synchronous:
        cutoff = max(sync_arrivals)
    elif candidates:
        cutoff = max(start, min(unit.arrival for unit in candidates))
    else:
        cutoff = None
    accepted = [] if cutoff is None else [unit for unit in candidates if unit.arrival <= cutoff + 1e-12]
    deferred = [unit for unit in candidates if unit not in accepted]
    finish = (start if cutoff is None else cutoff + config.cloud_aggregation_sec
              + max(sync_returns, default=0.0))
    for client, mode in profile.items():
        if mode in {"FL_C", "FL_EC"}:
            completion[client] = finish
    events.append({"event": "cloud_acceptance", "cutoff": cutoff, "finish": finish,
                   "trigger": "synchronous" if synchronous else "earliest_async_arrival",
                   "accepted": [unit.identity for unit in accepted],
                   "pending": [unit.identity for unit in deferred]})
    return Schedule(units, accepted, deferred, cutoff, finish, synchronous, completion, events)


def mode_masses(units):
    masses = {}
    for unit in units:
        masses[unit.mode] = masses.get(unit.mode, 0) + unit.samples
    return masses
