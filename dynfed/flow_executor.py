from __future__ import annotations

import math
import heapq
from dataclasses import dataclass
from typing import Any


EDGE_ONLY_MODES = {"LIE", "LIIE"}
CLOUD_DIRECT_MODES = {"LIC", "LIIC"}
EDGE_CLOUD_MODES = {"LIEIIC", "LIEIIIC", "LIIEIIIC"}
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
    dispatch_start_time: float = 0.0
    dispatch_sequence: int = 0

    @property
    def arrival_time(self) -> float:
        pre_aggregation_delay = max(0.0, self.candidate_time - self.estimated_local_time)
        return max(0.0, self.dispatch_start_time + self.measured_local_time + pre_aggregation_delay)


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
    chosen: list[tuple[int, float, list[int], str]]
    dropped: list[tuple[int, float, list[int], str]]
    start_time: float
    waiting_time: float
    k: int
    n: int
    wait_by_edge: dict[int, float]


def execute_mixed_round_flow(
    *,
    round_idx: int,
    clients: list[ClientFlowInput],
    aggregation_fraction: float,
    edge_aggregation_base_time: float = 0.08,
    cloud_aggregation_base_time: float = 0.12,
    edge_to_end_return_time: float = 0.05,
    cloud_to_edge_return_time: float = 0.08,
    cloud_to_end_return_time: float = 0.10,
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
    edge_groups: dict[tuple[str, int], list[ClientFlowInput]] = {}
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
            key = (client.mode, client.edge_id)
            if key not in edge_groups:
                edge_groups[key] = []
        elif client.mode in CLOUD_DIRECT_MODES:
            direct_cloud_clients.append(client)

    edge_cloud_group_count = sum(
        1 for mode, _ in edge_groups if mode in EDGE_CLOUD_MODES
    )
    direct_cloud_k = _buffer_size(len(direct_cloud_clients), aggregation_fraction)
    edge_cloud_k = _buffer_size(edge_cloud_group_count, aggregation_fraction)
    edge_buffers: dict[tuple[str, int], list[ClientFlowInput]] = {key: [] for key in edge_groups}
    edge_triggered: set[tuple[str, int]] = set()
    direct_cloud_buffer: list[ClientFlowInput] = []
    direct_cloud_triggered = False
    edge_cloud_buffer: list[tuple[int, float, list[int], str]] = []
    edge_cloud_triggered = False

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
            if client.mode in EDGE_ONLY_MODES | EDGE_CLOUD_MODES:
                key = (client.mode, client.edge_id)
                if key in edge_triggered:
                    _log_late_client(flow_events, round_idx, client, event_time, key)
                    continue
                edge_buffers[key].append(client)
                n = sum(1 for item in clients if item.mode == client.mode and item.edge_id == client.edge_id)
                k = _buffer_size(n, aggregation_fraction)
                if len(edge_buffers[key]) >= k:
                    edge_triggered.add(key)
                    buffer = _select_client_buffer(edge_buffers[key], aggregation_fraction, total_n=n)
                    waiting_time += buffer.waiting_time
                    loop_factor = 2 if client.mode in MULTI_EDGE_LOOP_MODES else 1
                    edge_agg = loop_factor * edge_aggregation_base_time * len(buffer.chosen)
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
                            loop_factor=loop_factor,
                        )
                    )
                    if client.mode in EDGE_ONLY_MODES:
                        for item in buffer.chosen:
                            selected[item.client_id] = item
                        return_time += edge_to_end_return_time
                        round_end_candidates.append(finish + edge_to_end_return_time)
                    else:
                        heapq.heappush(
                            event_heap,
                            (
                                finish,
                                event_seq,
                                "edge_ready",
                                (client.edge_id, finish, chosen_ids, client.mode),
                            ),
                        )
                        event_seq += 1
            elif client.mode in CLOUD_DIRECT_MODES and not direct_cloud_triggered:
                direct_cloud_buffer.append(client)
                if len(direct_cloud_buffer) >= direct_cloud_k:
                    direct_cloud_triggered = True
                    buffer = _select_client_buffer(
                        direct_cloud_buffer,
                        aggregation_fraction,
                        total_n=len(direct_cloud_clients),
                    )
                    waiting_time += buffer.waiting_time
                    cloud_agg = cloud_aggregation_base_time * len(buffer.chosen)
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
                        )
                    )
                    for item in buffer.chosen:
                        selected[item.client_id] = item
                    return_time += cloud_to_end_return_time
                    round_end_candidates.append(finish + cloud_to_end_return_time)
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
                cloud_agg = cloud_aggregation_base_time * len(edge_buffer.chosen)
                cloud_agg_total += cloud_agg
                finish = event_time + cloud_agg
                chosen_client_ids: list[int] = []
                for _, _, client_ids, _ in edge_buffer.chosen:
                    chosen_client_ids.extend(client_ids)
                flow_events.append(
                    _edge_cloud_event(
                        round_idx=round_idx,
                        start_time=event_time,
                        finish_time=finish,
                        aggregation_fraction=aggregation_fraction,
                        edge_buffer=edge_buffer,
                        cloud_agg=cloud_agg,
                        chosen_client_ids=chosen_client_ids,
                    )
                )
                chosen_client_set = set(chosen_client_ids)
                for item in clients:
                    if item.client_id in chosen_client_set:
                        selected[item.client_id] = item
                return_time += cloud_to_edge_return_time + edge_to_end_return_time
                round_end_candidates.append(finish + cloud_to_edge_return_time + edge_to_end_return_time)

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


def _select_client_buffer(
    clients: list[ClientFlowInput],
    aggregation_fraction: float,
    total_n: int | None = None,
) -> ClientBufferSelection:
    if not clients:
        return ClientBufferSelection([], [], 0.0, 0.0, 0, 0, {})
    n = total_n if total_n is not None else len(clients)
    k = _buffer_size(n, aggregation_fraction)
    chosen = sorted(clients, key=lambda item: item.arrival_time)[:k]
    dropped = sorted(clients, key=lambda item: item.arrival_time)[k:]
    start_time = max(client.arrival_time for client in chosen)
    wait_by_client = {
        client.client_id: start_time - client.arrival_time
        for client in chosen
    }
    waiting_time = sum(wait_by_client.values())
    return ClientBufferSelection(chosen, dropped, start_time, waiting_time, k, n, wait_by_client)


def _select_edge_buffer(
    edges: list[tuple[int, float, list[int], str]],
    aggregation_fraction: float,
    total_n: int | None = None,
) -> EdgeBufferSelection:
    if not edges:
        return EdgeBufferSelection([], [], 0.0, 0.0, 0, 0, {})
    n = total_n if total_n is not None else len(edges)
    k = _buffer_size(n, aggregation_fraction)
    chosen = sorted(edges, key=lambda item: item[1])[:k]
    dropped = sorted(edges, key=lambda item: item[1])[k:]
    start_time = max(edge_time for _, edge_time, _, _ in chosen)
    wait_by_edge = {
        edge_id: start_time - edge_time
        for edge_id, edge_time, _, _ in chosen
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
    }


def _edge_cloud_event(
    *,
    round_idx: int,
    start_time: float,
    finish_time: float,
    aggregation_fraction: float,
    edge_buffer: EdgeBufferSelection,
    cloud_agg: float,
    chosen_client_ids: list[int],
) -> dict[str, Any]:
    chosen_edge_ids = [edge_id for edge_id, _, _, _ in edge_buffer.chosen]
    dropped_edge_ids = [edge_id for edge_id, _, _, _ in edge_buffer.dropped]
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
    }


def _log_late_client(
    flow_events: list[dict[str, Any]],
    round_idx: int,
    client: ClientFlowInput,
    event_time: float,
    key: tuple[str, int],
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
            "buffer_key": f"{key[0]}:{key[1]}",
        }
    )
