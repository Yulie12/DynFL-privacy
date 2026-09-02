from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from typing import Any


EDGE_ONLY_MODES = {"LIE", "LIIE"}
CLOUD_DIRECT_MODES = {"LIC", "LIIC", "LIEIIC"}
EDGE_CLOUD_MODES = {"LIEIIIC", "LIIEIIIC"}
MULTI_EDGE_LOOP_MODES = {"LIEIIIC", "LIIEIIIC"}


@dataclass(frozen=True)
class ClientFlowInput:
    client_id: int
    edge_id: int
    mode: str
    candidate_time: float
    estimated_local_time: float
    measured_local_time: float
    communication_volume: float
    state_diff: Any
    sample_count: int
    edge_loops: int = 1
    edge_to_cloud_time: float = 0.0
    return_path_time: float = 0.0
    edge_aggregation_payload: float = 0.0
    cloud_aggregation_payload: float = 0.0
    aggregation_group: str = ""
    dispatch_start_time: float = 0.0
    dispatch_sequence: int = 0

    @property
    def arrival_time(self) -> float:
        # Logical simulation time is independent of host CPU/GPU scheduling.
        # measured_local_time remains an execution-performance log only.
        return max(0.0, self.dispatch_start_time + self.candidate_time)


@dataclass(frozen=True)
class FlowExecutionResult:
    selected_client_ids: list[int]
    state_diffs: list[Any]
    sample_counts: list[int]
    round_duration: float
    waiting_time: float
    edge_aggregation_time: float
    cloud_aggregation_time: float
    return_time: float
    num_effective_edges: int
    flow_events: list[dict[str, Any]]


@dataclass(frozen=True)
class ClientBufferSelection:
    chosen: list[ClientFlowInput]
    dropped: list[ClientFlowInput]
    start_time: float
    waiting_time: float
    k: int
    n: int
    wait_by_client: dict[int, float]


@dataclass(frozen=True)
class EdgeBufferSelection:
    chosen: list["EdgeReadyInput"]
    dropped: list["EdgeReadyInput"]
    start_time: float
    waiting_time: float
    k: int
    n: int
    wait_by_edge: dict[int, float]


@dataclass(frozen=True)
class EdgeReadyInput:
    edge_id: int
    arrival_time: float
    client_ids: list[int]
    mode: str
    cloud_aggregation_payload: float
    return_path_time: float
    aggregation_group: str


def execute_mixed_round_flow(
    *,
    round_idx: int,
    clients: list[ClientFlowInput],
    aggregation_fraction: float,
    edge_aggregation_beta: float = 0.01,
    edge_aggregation_fixed: float = 0.02,
    cloud_aggregation_beta: float = 0.015,
    cloud_aggregation_fixed: float = 0.04,
) -> FlowExecutionResult:
    """Execute one TeX-style flow round over mixed per-client modes.

    This function encodes the block/flow distinction from tex/20260608.tex:
    client updates arrive after their repeated block execution, then flow-level
    submit/wait/aggregate/return stages happen at edge and/or cloud according to
    the selected mode.
    """

    selected: dict[int, ClientFlowInput] = {}
    waiting_time = 0.0
    edge_agg_total = 0.0
    cloud_agg_total = 0.0
    return_time = 0.0
    flow_events: list[dict[str, Any]] = []
    round_end_candidates: list[float] = []
    event_seq = 0
    event_heap: list[tuple[float, int, str, Any]] = []
    edge_groups: dict[tuple[str, int, str], list[ClientFlowInput]] = {}
    direct_cloud_clients: list[ClientFlowInput] = []
    edge_cloud_group_count = 0

    for client in clients:
        flow_events.append(
            {
                "round": round_idx,
                "event_type": "client_train_start",
                "tex_stage": "block",
                "client_id": client.client_id,
                "edge_id": client.edge_id,
                "mode": client.mode,
                "time": client.dispatch_start_time,
                "dispatch_sequence": client.dispatch_sequence,
                "measured_local_time": client.measured_local_time,
            }
        )
        heapq.heappush(event_heap, (client.arrival_time, event_seq, "client_arrival", client))
        event_seq += 1
        if client.mode in EDGE_ONLY_MODES | EDGE_CLOUD_MODES:
            key = (client.mode, client.edge_id, client.aggregation_group)
            if key not in edge_groups:
                edge_groups[key] = []
        elif client.mode in CLOUD_DIRECT_MODES:
            direct_cloud_clients.append(client)

    edge_cloud_group_count = sum(
        1 for mode, _edge_id, _group in edge_groups if mode in EDGE_CLOUD_MODES
    )
    direct_cloud_k = _buffer_size(len(direct_cloud_clients), aggregation_fraction)
    edge_cloud_k = _buffer_size(edge_cloud_group_count, aggregation_fraction)
    edge_buffers: dict[tuple[str, int, str], list[ClientFlowInput]] = {key: [] for key in edge_groups}
    edge_triggered: set[tuple[str, int, str]] = set()
    direct_cloud_buffer: list[ClientFlowInput] = []
    direct_cloud_triggered = False
    edge_cloud_buffer: list[EdgeReadyInput] = []
    edge_cloud_triggered = False
    tie_admitted_client_ids: set[int] = set()

    while event_heap:
        event_time, _, event_type, payload = heapq.heappop(event_heap)
        if event_type == "client_arrival":
            client = payload
            flow_events.append(
                {
                    "round": round_idx,
                    "event_type": "block_complete",
                    "tex_stage": "block",
                    "client_id": client.client_id,
                    "edge_id": client.edge_id,
                    "mode": client.mode,
                    "time": event_time,
                    "x_i_j_t": event_time,
                    "dispatch_sequence": client.dispatch_sequence,
                    "dispatch_start_time": client.dispatch_start_time,
                    "duration": client.measured_local_time,
                }
            )
            if client.client_id in tie_admitted_client_ids:
                continue
            if client.mode in EDGE_ONLY_MODES | EDGE_CLOUD_MODES:
                key = (client.mode, client.edge_id, client.aggregation_group)
                if key in edge_triggered:
                    _log_late_client(flow_events, round_idx, client, event_time, key)
                    continue
                edge_buffers[key].append(client)
                n = sum(
                    1
                    for item in clients
                    if item.mode == client.mode
                    and item.edge_id == client.edge_id
                    and item.aggregation_group == client.aggregation_group
                )
                k = _buffer_size(n, aggregation_fraction)
                if len(edge_buffers[key]) >= k:
                    edge_triggered.add(key)
                    eligible = [
                        item
                        for item in clients
                        if item.mode == client.mode
                        and item.edge_id == client.edge_id
                        and item.aggregation_group == client.aggregation_group
                        and item.arrival_time <= event_time + 1e-12
                    ]
                    buffer = _select_client_buffer(eligible, aggregation_fraction, total_n=n)
                    tie_admitted_client_ids.update(item.client_id for item in buffer.chosen)
                    waiting_time += buffer.waiting_time
                    loop_factor = max(1, int(client.edge_loops))
                    edge_payload = sum(
                        item.edge_aggregation_payload for item in buffer.chosen
                    )
                    edge_agg = loop_factor * (
                        edge_aggregation_beta * edge_payload
                        + edge_aggregation_fixed
                    )
                    edge_agg_total += edge_agg
                    finish = event_time + edge_agg
                    chosen_ids = [item.client_id for item in buffer.chosen]
                    flow_events.append(
                        _edge_event(
                            round_idx=round_idx,
                            mode=client.mode,
                            edge_id=client.edge_id,
                            start_time=event_time,
                            finish_time=finish,
                            aggregation_fraction=aggregation_fraction,
                            buffer=buffer,
                            edge_agg=edge_agg,
                            effective_payload=edge_payload,
                            aggregation_beta=edge_aggregation_beta,
                            aggregation_fixed=edge_aggregation_fixed,
                            loop_factor=loop_factor,
                        )
                    )
                    if client.mode in EDGE_ONLY_MODES:
                        for item in buffer.chosen:
                            selected[item.client_id] = item
                        edge_return_time = max(
                            (item.return_path_time for item in buffer.chosen),
                            default=0.0,
                        )
                        return_time += edge_return_time
                        round_end_candidates.append(finish + edge_return_time)
                    else:
                        edge_upload_time = max(
                            (item.edge_to_cloud_time for item in buffer.chosen),
                            default=0.0,
                        )
                        cloud_arrival = finish + edge_upload_time
                        heapq.heappush(
                            event_heap,
                            (
                                cloud_arrival,
                                event_seq,
                                "edge_ready",
                                EdgeReadyInput(
                                    edge_id=client.edge_id,
                                    arrival_time=cloud_arrival,
                                    client_ids=chosen_ids,
                                    mode=client.mode,
                                    cloud_aggregation_payload=max(
                                        (
                                            item.cloud_aggregation_payload
                                            for item in buffer.chosen
                                        ),
                                        default=0.0,
                                    ),
                                    return_path_time=max(
                                        (item.return_path_time for item in buffer.chosen),
                                        default=0.0,
                                    ),
                                    aggregation_group=client.aggregation_group,
                                ),
                            ),
                        )
                        event_seq += 1
            elif client.mode in CLOUD_DIRECT_MODES and not direct_cloud_triggered:
                direct_cloud_buffer.append(client)
                if len(direct_cloud_buffer) >= direct_cloud_k:
                    direct_cloud_triggered = True
                    eligible = [
                        item
                        for item in direct_cloud_clients
                        if item.arrival_time <= event_time + 1e-12
                    ]
                    buffer = _select_client_buffer(
                        eligible,
                        aggregation_fraction,
                        total_n=len(direct_cloud_clients),
                    )
                    tie_admitted_client_ids.update(item.client_id for item in buffer.chosen)
                    waiting_time += buffer.waiting_time
                    cloud_payload = sum(
                        item.cloud_aggregation_payload for item in buffer.chosen
                    )
                    cloud_agg = (
                        cloud_aggregation_beta * cloud_payload
                        + cloud_aggregation_fixed
                    )
                    cloud_agg_total += cloud_agg
                    finish = event_time + cloud_agg
                    chosen_ids = [item.client_id for item in buffer.chosen]
                    flow_events.append(
                        _direct_cloud_event(
                            round_idx=round_idx,
                            start_time=event_time,
                            finish_time=finish,
                            aggregation_fraction=aggregation_fraction,
                            buffer=buffer,
                            cloud_agg=cloud_agg,
                            effective_payload=cloud_payload,
                            aggregation_beta=cloud_aggregation_beta,
                            aggregation_fixed=cloud_aggregation_fixed,
                        )
                    )
                    for item in buffer.chosen:
                        selected[item.client_id] = item
                    direct_return_time = max(
                        (item.return_path_time for item in buffer.chosen),
                        default=0.0,
                    )
                    return_time += direct_return_time
                    round_end_candidates.append(finish + direct_return_time)
        elif event_type == "edge_ready" and not edge_cloud_triggered:
            edge_cloud_buffer.append(payload)
            if len(edge_cloud_buffer) >= edge_cloud_k:
                edge_cloud_triggered = True
                edge_buffer = _select_edge_buffer(
                    edge_cloud_buffer,
                    aggregation_fraction,
                    total_n=edge_cloud_group_count,
                )
                waiting_time += edge_buffer.waiting_time
                cloud_payload = sum(
                    item.cloud_aggregation_payload for item in edge_buffer.chosen
                )
                cloud_agg = (
                    cloud_aggregation_beta * cloud_payload
                    + cloud_aggregation_fixed
                )
                cloud_agg_total += cloud_agg
                finish = event_time + cloud_agg
                chosen_client_ids: list[int] = []
                for item in edge_buffer.chosen:
                    chosen_client_ids.extend(item.client_ids)
                flow_events.append(
                    _edge_cloud_event(
                        round_idx=round_idx,
                        start_time=event_time,
                        finish_time=finish,
                        aggregation_fraction=aggregation_fraction,
                        edge_buffer=edge_buffer,
                        cloud_agg=cloud_agg,
                        effective_payload=cloud_payload,
                        aggregation_beta=cloud_aggregation_beta,
                        aggregation_fixed=cloud_aggregation_fixed,
                        chosen_client_ids=chosen_client_ids,
                    )
                )
                chosen_client_set = set(chosen_client_ids)
                for item in clients:
                    if item.client_id in chosen_client_set:
                        selected[item.client_id] = item
                edge_cloud_return_time = max(
                    (item.return_path_time for item in edge_buffer.chosen),
                    default=0.0,
                )
                return_time += edge_cloud_return_time
                round_end_candidates.append(finish + edge_cloud_return_time)

    selected_items = [selected[cid] for cid in sorted(selected)]
    return FlowExecutionResult(
        selected_client_ids=[client.client_id for client in selected_items],
        state_diffs=[client.state_diff for client in selected_items],
        sample_counts=[client.sample_count for client in selected_items],
        round_duration=max(round_end_candidates, default=0.0),
        waiting_time=waiting_time,
        edge_aggregation_time=edge_agg_total,
        cloud_aggregation_time=cloud_agg_total,
        return_time=return_time,
        num_effective_edges=len({client.edge_id for client in selected_items}),
        flow_events=flow_events,
    )


def summarize_mixed_round_flow(
    *,
    round_idx: int,
    clients: list[ClientFlowInput],
    aggregation_fraction: float,
    edge_aggregation_beta: float = 0.01,
    edge_aggregation_fixed: float = 0.02,
    cloud_aggregation_beta: float = 0.015,
    cloud_aggregation_fixed: float = 0.04,
) -> FlowExecutionResult:
    """Return exact flow metrics without materializing events when all buffers fill.

    Pareto search only consumes admission and timing summaries. When every
    active buffer waits for all of its members, those summaries can be computed
    directly. Other buffer configurations retain the event executor as the
    exact fallback.
    """
    edge_groups: dict[tuple[str, int, str], list[tuple[int, ClientFlowInput]]] = {}
    direct_cloud_clients: list[tuple[int, ClientFlowInput]] = []
    for position, client in enumerate(clients):
        if client.mode in EDGE_ONLY_MODES | EDGE_CLOUD_MODES:
            key = (client.mode, client.edge_id, client.aggregation_group)
            edge_groups.setdefault(key, []).append((position, client))
        elif client.mode in CLOUD_DIRECT_MODES:
            direct_cloud_clients.append((position, client))

    edge_cloud_group_count = sum(
        1 for mode, _edge_id, _group in edge_groups if mode in EDGE_CLOUD_MODES
    )
    client_buffers = list(edge_groups.values())
    if direct_cloud_clients:
        client_buffers.append(direct_cloud_clients)
    all_client_buffers_fill = all(
        _buffer_size(len(group), aggregation_fraction) == len(group)
        for group in client_buffers
    )
    edge_cloud_buffer_fills = (
        edge_cloud_group_count == 0
        or _buffer_size(edge_cloud_group_count, aggregation_fraction)
        == edge_cloud_group_count
    )
    if not all_client_buffers_fill or not edge_cloud_buffer_fills:
        return execute_mixed_round_flow(
            round_idx=round_idx,
            clients=clients,
            aggregation_fraction=aggregation_fraction,
            edge_aggregation_beta=edge_aggregation_beta,
            edge_aggregation_fixed=edge_aggregation_fixed,
            cloud_aggregation_beta=cloud_aggregation_beta,
            cloud_aggregation_fixed=cloud_aggregation_fixed,
        )

    selected: dict[int, ClientFlowInput] = {}
    waiting_time = 0.0
    edge_agg_total = 0.0
    cloud_agg_total = 0.0
    return_time = 0.0
    round_end_candidates: list[float] = []
    edge_cloud_ready: list[EdgeReadyInput] = []

    for (mode, edge_id, aggregation_group), indexed_group in edge_groups.items():
        group = [client for _position, client in indexed_group]
        start_time = max(client.arrival_time for client in group)
        waiting_time += sum(start_time - client.arrival_time for client in group)
        trigger_client = max(
            indexed_group,
            key=lambda item: (item[1].arrival_time, item[0]),
        )[1]
        loop_factor = max(1, int(trigger_client.edge_loops))
        edge_payload = sum(client.edge_aggregation_payload for client in group)
        edge_agg = loop_factor * (
            edge_aggregation_beta * edge_payload + edge_aggregation_fixed
        )
        edge_agg_total += edge_agg
        finish_time = start_time + edge_agg
        if mode in EDGE_ONLY_MODES:
            for client in group:
                selected[client.client_id] = client
            group_return_time = max(
                (client.return_path_time for client in group),
                default=0.0,
            )
            return_time += group_return_time
            round_end_candidates.append(finish_time + group_return_time)
            continue

        edge_upload_time = max(
            (client.edge_to_cloud_time for client in group),
            default=0.0,
        )
        edge_cloud_ready.append(
            EdgeReadyInput(
                edge_id=edge_id,
                arrival_time=finish_time + edge_upload_time,
                client_ids=[client.client_id for client in group],
                mode=mode,
                cloud_aggregation_payload=max(
                    (client.cloud_aggregation_payload for client in group),
                    default=0.0,
                ),
                return_path_time=max(
                    (client.return_path_time for client in group),
                    default=0.0,
                ),
                aggregation_group=aggregation_group,
            )
        )

    if direct_cloud_clients:
        direct_group = [client for _position, client in direct_cloud_clients]
        start_time = max(client.arrival_time for client in direct_group)
        waiting_time += sum(start_time - client.arrival_time for client in direct_group)
        cloud_payload = sum(client.cloud_aggregation_payload for client in direct_group)
        cloud_agg = cloud_aggregation_beta * cloud_payload + cloud_aggregation_fixed
        cloud_agg_total += cloud_agg
        for client in direct_group:
            selected[client.client_id] = client
        direct_return_time = max(
            (client.return_path_time for client in direct_group),
            default=0.0,
        )
        return_time += direct_return_time
        round_end_candidates.append(start_time + cloud_agg + direct_return_time)

    if edge_cloud_ready:
        start_time = max(edge.arrival_time for edge in edge_cloud_ready)
        wait_by_edge = {
            edge.edge_id: start_time - edge.arrival_time
            for edge in sorted(edge_cloud_ready, key=lambda item: item.arrival_time)
        }
        waiting_time += sum(wait_by_edge.values())
        cloud_payload = sum(edge.cloud_aggregation_payload for edge in edge_cloud_ready)
        cloud_agg = cloud_aggregation_beta * cloud_payload + cloud_aggregation_fixed
        cloud_agg_total += cloud_agg
        chosen_client_ids = {
            client_id
            for edge in edge_cloud_ready
            for client_id in edge.client_ids
        }
        for client in clients:
            if client.client_id in chosen_client_ids:
                selected[client.client_id] = client
        edge_cloud_return_time = max(
            (edge.return_path_time for edge in edge_cloud_ready),
            default=0.0,
        )
        return_time += edge_cloud_return_time
        round_end_candidates.append(start_time + cloud_agg + edge_cloud_return_time)

    selected_items = [selected[client_id] for client_id in sorted(selected)]
    return FlowExecutionResult(
        selected_client_ids=[client.client_id for client in selected_items],
        state_diffs=[client.state_diff for client in selected_items],
        sample_counts=[client.sample_count for client in selected_items],
        round_duration=max(round_end_candidates, default=0.0),
        waiting_time=waiting_time,
        edge_aggregation_time=edge_agg_total,
        cloud_aggregation_time=cloud_agg_total,
        return_time=return_time,
        num_effective_edges=len({client.edge_id for client in selected_items}),
        flow_events=[],
    )


def _select_client_buffer(
    clients: list[ClientFlowInput],
    aggregation_fraction: float,
    total_n: int | None = None,
) -> ClientBufferSelection:
    if not clients:
        return ClientBufferSelection([], [], 0.0, 0.0, 0, 0, {})
    n = total_n if total_n is not None else len(clients)
    k = _buffer_size(n, aggregation_fraction)
    ordered = sorted(clients, key=lambda item: item.arrival_time)
    threshold = ordered[min(k, len(ordered)) - 1].arrival_time
    chosen = [client for client in ordered if client.arrival_time <= threshold + 1e-12]
    dropped = [client for client in ordered if client.arrival_time > threshold + 1e-12]
    start_time = max(client.arrival_time for client in chosen)
    wait_by_client = {
        client.client_id: start_time - client.arrival_time
        for client in chosen
    }
    waiting_time = sum(wait_by_client.values())
    return ClientBufferSelection(chosen, dropped, start_time, waiting_time, k, n, wait_by_client)


def _select_edge_buffer(
    edges: list[EdgeReadyInput],
    aggregation_fraction: float,
    total_n: int | None = None,
) -> EdgeBufferSelection:
    if not edges:
        return EdgeBufferSelection([], [], 0.0, 0.0, 0, 0, {})
    n = total_n if total_n is not None else len(edges)
    k = _buffer_size(n, aggregation_fraction)
    ordered = sorted(edges, key=lambda item: item.arrival_time)
    threshold = ordered[min(k, len(ordered)) - 1].arrival_time
    chosen = [edge for edge in ordered if edge.arrival_time <= threshold + 1e-12]
    dropped = [edge for edge in ordered if edge.arrival_time > threshold + 1e-12]
    start_time = max(edge.arrival_time for edge in chosen)
    wait_by_edge = {
        edge.edge_id: start_time - edge.arrival_time
        for edge in chosen
    }
    waiting_time = sum(wait_by_edge.values())
    return EdgeBufferSelection(chosen, dropped, start_time, waiting_time, k, n, wait_by_edge)


def _format_waits(wait_by_id: dict[int, float]) -> str:
    return ";".join(f"{key}:{value:.6f}" for key, value in sorted(wait_by_id.items()))


def _buffer_size(n: int, aggregation_fraction: float) -> int:
    if n <= 0:
        return 0
    return max(1, math.ceil(n * aggregation_fraction))


def _edge_event(
    *,
    round_idx: int,
    mode: str,
    edge_id: int,
    start_time: float,
    finish_time: float,
    aggregation_fraction: float,
    buffer: ClientBufferSelection,
    edge_agg: float,
    effective_payload: float,
    aggregation_beta: float,
    aggregation_fixed: float,
    loop_factor: int,
) -> dict[str, Any]:
    chosen_ids = [client.client_id for client in buffer.chosen]
    dropped_ids = [client.client_id for client in buffer.dropped]
    return {
        "round": round_idx,
        "event_type": "edge_aggregate",
        "tex_stage": "flow_edge_aggregation",
        "mode": mode,
        "edge_id": edge_id,
        "time": start_time,
        "X_j_t": start_time,
        "finish_time": finish_time,
        "F_j_t": finish_time,
        "rho": aggregation_fraction,
        "N_j": buffer.n,
        "K_j": buffer.k,
        "num_clients": len(buffer.chosen),
        "client_ids": ";".join(str(cid) for cid in chosen_ids),
        "S_j_t": ";".join(str(cid) for cid in chosen_ids),
        "dropped_client_ids": ";".join(str(cid) for cid in dropped_ids),
        "W_i_j_t": _format_waits(buffer.wait_by_client),
        "waiting_time": buffer.waiting_time,
        "aggregation_time": edge_agg,
        "effective_payload": effective_payload,
        "aggregation_beta": aggregation_beta,
        "aggregation_fixed": aggregation_fixed,
        "loop_factor": loop_factor,
    }


def _direct_cloud_event(
    *,
    round_idx: int,
    start_time: float,
    finish_time: float,
    aggregation_fraction: float,
    buffer: ClientBufferSelection,
    cloud_agg: float,
    effective_payload: float,
    aggregation_beta: float,
    aggregation_fixed: float,
) -> dict[str, Any]:
    chosen_ids = [client.client_id for client in buffer.chosen]
    dropped_ids = [client.client_id for client in buffer.dropped]
    return {
        "round": round_idx,
        "event_type": "cloud_aggregate_direct",
        "tex_stage": "flow_cloud_aggregation",
        "time": start_time,
        "X_c_t": start_time,
        "finish_time": finish_time,
        "F_c_t": finish_time,
        "rho": aggregation_fraction,
        "N_c": buffer.n,
        "K_c": buffer.k,
        "num_clients": len(buffer.chosen),
        "client_ids": ";".join(str(cid) for cid in chosen_ids),
        "S_c_t": ";".join(str(cid) for cid in chosen_ids),
        "dropped_client_ids": ";".join(str(cid) for cid in dropped_ids),
        "W_i_c_t": _format_waits(buffer.wait_by_client),
        "waiting_time": buffer.waiting_time,
        "aggregation_time": cloud_agg,
        "effective_payload": effective_payload,
        "aggregation_beta": aggregation_beta,
        "aggregation_fixed": aggregation_fixed,
    }


def _edge_cloud_event(
    *,
    round_idx: int,
    start_time: float,
    finish_time: float,
    aggregation_fraction: float,
    edge_buffer: EdgeBufferSelection,
    cloud_agg: float,
    effective_payload: float,
    aggregation_beta: float,
    aggregation_fixed: float,
    chosen_client_ids: list[int],
) -> dict[str, Any]:
    chosen_edge_ids = [edge.edge_id for edge in edge_buffer.chosen]
    dropped_edge_ids = [edge.edge_id for edge in edge_buffer.dropped]
    return {
        "round": round_idx,
        "event_type": "cloud_aggregate_from_edges",
        "tex_stage": "flow_cloud_aggregation",
        "time": start_time,
        "X_c_t": start_time,
        "finish_time": finish_time,
        "F_c_t": finish_time,
        "rho": aggregation_fraction,
        "N_c": edge_buffer.n,
        "K_c": edge_buffer.k,
        "num_edges": len(edge_buffer.chosen),
        "num_clients": len(chosen_client_ids),
        "edge_ids": ";".join(str(eid) for eid in chosen_edge_ids),
        "S_c_t": ";".join(str(eid) for eid in chosen_edge_ids),
        "dropped_edge_ids": ";".join(str(eid) for eid in dropped_edge_ids),
        "client_ids": ";".join(str(cid) for cid in chosen_client_ids),
        "W_j_c_t": _format_waits(edge_buffer.wait_by_edge),
        "waiting_time": edge_buffer.waiting_time,
        "aggregation_time": cloud_agg,
        "effective_payload": effective_payload,
        "aggregation_beta": aggregation_beta,
        "aggregation_fixed": aggregation_fixed,
    }


def _log_late_client(
    flow_events: list[dict[str, Any]],
    round_idx: int,
    client: ClientFlowInput,
    event_time: float,
    key: tuple[str, int, str],
) -> None:
    flow_events.append(
        {
            "round": round_idx,
            "event_type": "buffer_late_drop",
            "tex_stage": "flow_buffer",
            "client_id": client.client_id,
            "edge_id": client.edge_id,
            "mode": client.mode,
            "time": event_time,
            "buffer_key": f"{key[0]}:{key[1]}:{key[2]}",
        }
    )
