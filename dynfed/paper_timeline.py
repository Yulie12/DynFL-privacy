"""100% current-round Cloud quorum followed by an all-client version barrier."""
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
    base_source: str = "cloud"
    base_round: int = 0


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
    acknowledgement_finish: dict[int, float]


def plan(config, profile, traces, samples, payload_mb, start, round_index,
         previous=None, pending=()):
    """FL_E finishes its local chain independently, but cannot start a new round.

    Units are fixed before execution; all must arrive, then all configured
    clients (including SKIP clients) must receive the model and return an ACK.
    Network times are simulated, not physical transport acknowledgements.
    """
    if pending:
        raise ValueError("cross-round pending units are forbidden by global_barrier_100")
    previous = previous or {}
    groups = {}
    units, events, completion = [], [], {}
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
        if mode == "FL_E":
            for client in members:
                completion[client] = last_finish + payload_mb / traces[client]["le_rate"] + config.end_edge_delay
            events.append({"event": "async_upload", "identity": unit.identity, "arrival": arrival})
    cutoff = max((unit.arrival for unit in units), default=start)
    cloud_finish = cutoff + (config.cloud_aggregation_sec if units else 0.)
    events.append({"event": "cloud_acceptance", "cutoff": cutoff, "finish": cloud_finish,
                   "trigger": "all_expected_units", "quorum": config.cloud_quorum,
                   "expected": [unit.identity for unit in units],
                   "accepted": [unit.identity for unit in units], "pending": []})
    acknowledgements = {}
    for client in range(config.clients):
        trace = traces[client]
        mode = profile.get(client, "SKIP")
        if mode in {"FL_E", "FL_EC"}:
            down = (payload_mb/trace["ec_rate"] + config.edge_cloud_delay
                    + payload_mb/trace["le_rate"] + config.end_edge_delay)
            up = (config.acknowledgement_mb/trace["le_rate"] + config.end_edge_delay
                  + config.acknowledgement_mb/trace["ec_rate"] + config.edge_cloud_delay)
        else:
            down = payload_mb/trace["lc_rate"] + config.end_cloud_delay
            up = config.acknowledgement_mb/trace["lc_rate"] + config.end_cloud_delay
        received = max(cloud_finish+down, completion.get(client, start))
        acknowledged = received + config.model_install_sec + up
        acknowledgements[client] = acknowledged
        events.append({"event": "model_ack", "client": client, "round": round_index,
                       "broadcast_start": cloud_finish, "received": received,
                       "finish": acknowledged, "mode": mode})
    finish = max(acknowledgements.values(), default=cloud_finish)
    events.append({"event": "global_barrier", "round": round_index,
                   "clients": list(range(config.clients)), "finish": finish})
    return Schedule(units, list(units), [], cutoff, finish, True, completion, events, acknowledgements)


def mode_masses(units):
    masses = {}
    for unit in units:
        masses[unit.mode] = masses.get(unit.mode, 0) + unit.samples
    return masses
