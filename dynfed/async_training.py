"""Asynchronous version-controlled training loop.

Replaces round-based sync training with an event-driven simulation where:
- Each end tracks which edge/cloud parameter version it trained on
- Submissions are tagged with their base version
- Edge/cloud aggregate when their buffer reaches threshold B
- Stale submissions (version gap too large) are decayed or dropped
- After submission, an end can immediately start the next training round
"""

from __future__ import annotations

import heapq
import math
import random
import time as time_module
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch

from .fmnist_dynamic_training import load_fmnist_arrays
from .lenet5_training import count_params
from .nodes import build_profiles
from .privacy import OBJECT_SIZES, PRIVACY_ALPHA, PRIVACY_BASE_TIME, PrivacyAccountant
from .selection import SelectionConfig, choose_candidate, enumerate_candidates
from .split_learning import (
    EndNet,
    EdgeNet,
    apply_unified_dp,
    fedavg_split,
    split_evaluate,
    split_local_train_lenet5,
)
from .training import MODE_SPECS


@dataclass
class VersionedSubmission:
    """A training submission tagged with its base parameter versions."""
    end_id: int
    edge_id: int
    mode: str
    state_diff: dict[str, dict[str, torch.Tensor]]  # {"end": ..., "edge": ...}
    samples: int
    base_edge_version: int
    base_cloud_version: int
    submission_time: float
    compute_time: float
    feasible: bool = True
    edge_cycle: int = 0  # which edge cycle this submission belongs to (for E>1 modes)

    @property
    def is_stale(self, current_edge_version: int, max_gap: int = 3) -> bool:
        return (current_edge_version - self.base_edge_version) > max_gap

    def staleness_weight(self, current_version: int, decay: float = 0.85) -> float:
        gap = current_version - self.base_edge_version
        if gap <= 0:
            return 1.0
        return decay ** gap


@dataclass
class EndTrainState:
    """Per-end tracking state in the async loop."""
    end_id: int
    edge_id: int
    edge_version: int = 0
    cloud_version: int = 0
    edge_cycle: int = 0  # how many edge-rounds completed (for E>1)
    training: bool = False
    edge_params: dict[str, torch.Tensor] | None = None  # cached edge state dict
    cloud_params: dict[str, torch.Tensor] | None = None  # cached cloud state dict
    samples: int = 150
    compute_factor: float = 1.0
    remaining_epsilon: float = 4.0
    last_reselect_cloud_ver: int = -1  # re-select on first training
    train_cycles_since_reselect: int = 0  # training cycles since last mode re-select
    mode: str | None = None
    mechanisms: dict[str, str] | None = None
    sensitivity: float = 0.5  # response-time sensitivity (0=throughput, 1=low-latency)
    total_submissions: int = 0
    total_compute_time: float = 0.0
    cpu_edge_demand: float = 0.0  # current CPU reservation on edge
    cpu_cloud_demand: float = 0.0  # current CPU reservation on cloud

    @property
    def effective_compute_factor(self) -> float:
        return self.compute_factor


@dataclass
class EdgeAggState:
    """Per-edge aggregation state."""
    edge_id: int
    version: int = 0
    buffer: list[VersionedSubmission] = field(default_factory=list)
    global_params: dict[str, torch.Tensor] | None = None
    compute_factor: float = 1.0
    aggregation_count: int = 0

    def buffer_full(self, B: int) -> bool:
        return len(self.buffer) >= B


@dataclass
class CloudAggState:
    """Cloud-level aggregation state."""
    version: int = 0
    buffer: list[VersionedSubmission] = field(default_factory=list)
    global_end_params: dict[str, torch.Tensor] | None = None
    global_edge_params: dict[str, torch.Tensor] | None = None
    aggregation_count: int = 0

    def buffer_full(self, B: int) -> bool:
        return len(self.buffer) >= B


@dataclass
class AsyncConfig:
    B_edge: int = 3
    B_cloud: int = 5
    decay_rate: float = 0.85
    max_version_gap: int = 3
    local_epochs: int = 2
    learning_rate: float = 0.15
    dp_clip_norm: float = 5.0
    dp_noise_multiplier: float = 0.001
    dp_delta: float = 1e-5
    device: str = "cpu"
    max_events: int = 500
    eval_interval_events: int = 10
    reselect_interval: int = 3  # re-select mode every K cloud aggregations
    per_end_epsilon: list[float] | None = None  # per-end initial budgets (matches visualization)
    warmup_events: int = 100  # first N events: DP noise reduced to warmup_factor × σ
    warmup_factor: float = 0.02  # noise multiplier scaling during warmup
    adaptive_clip: bool = False  # Eq. (18): one global L2 norm for the update vector


class AsyncEvent:
    """Event in the async simulation priority queue."""
    def __init__(self, time: float, priority: int, callback: Callable, label: str = ""):
        self.time = time
        self.priority = priority
        self.callback = callback
        self.label = label

    def __lt__(self, other: "AsyncEvent") -> bool:
        if self.time != other.time:
            return self.time < other.time
        return self.priority < other.priority

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, AsyncEvent):
            return NotImplemented
        return self.time == other.time and self.priority == other.priority

    def __hash__(self) -> int:
        return id(self)


class AsyncSimulation:
    """Event-driven async training simulation with version control."""

    def __init__(
        self,
        *,
        selection: SelectionConfig,
        config: AsyncConfig,
        clients: list[Any],
        edges: list[Any],
        edge_by_id: dict[int, Any],
        client_indices: list[np.ndarray],
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        output_dir: Path,
        policy: str,
        rng: random.Random,
        np_rng: np.random.Generator,
    ):
        self.selection = selection
        self.config = config
        self.clients = clients
        self.edge_by_id = edge_by_id
        self.edges = edges
        self.client_indices = client_indices
        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test
        self.output_dir = output_dir
        self.policy = policy
        self.rng = rng
        self.np_rng = np_rng

        self.device = torch.device(config.device)
        self.current_time = 0.0
        self.event_count = 0
        self.event_queue: list[AsyncEvent] = []
        self.ended = False

        # Global models (authoritative)
        self.global_end = EndNet().to(self.device)
        self.global_edge = EdgeNet().to(self.device)

        # Mode performance history: (end_id, mode) → list of accuracy deltas
        self.mode_history: dict[tuple[int, str], list[float]] = {}
        # Mode usage since last eval (mode → count)
        self.mode_usage_since_eval: dict[str, int] = {}
        self.last_eval_accuracy: float | None = None

        # Edge/cloud CPU load tracking
        self._edge_cpu_load: dict[int, float] = {e.edge_id: 0.0 for e in edges}
        self._cloud_cpu_load: float = 0.0

        # Per-end tracking
        self.end_states: dict[int, EndTrainState] = {}
        self.accountants: dict[int, PrivacyAccountant] = {}
        for i, client in enumerate(clients):
            # Per-end sensitivity from individual characteristics:
            # - compute_factor 0.5~2.0: faster ends (lower factor) → higher sensitivity (time-urgent)
            # - samples 150+: more data → slightly lower sensitivity (accuracy-seeking)
            cf_norm = max(0.0, min(1.0, (2.0 - client.compute_factor) / 1.5))
            sample_norm = min(client.samples / 300.0, 1.0)
            sens = 0.1 + 0.8 * cf_norm - 0.2 * sample_norm
            sens = max(0.05, min(0.95, sens))

            # Per-end heterogeneous budget (matches visualization privacy_budget)
            if config.per_end_epsilon is not None and i < len(config.per_end_epsilon):
                eps = config.per_end_epsilon[i]
            else:
                eps = selection.initial_epsilon

            self.end_states[client.client_id] = EndTrainState(
                end_id=client.client_id,
                edge_id=client.edge_id,
                samples=client.samples,
                compute_factor=client.compute_factor,
                remaining_epsilon=eps,
                sensitivity=sens,
            )
            self.accountants[client.client_id] = PrivacyAccountant(
                budget=1000.0,  # large budget — only used for reporting current_epsilon()
                delta=config.dp_delta,
            )

        # Edge aggregation states
        self.edge_states: dict[int, EdgeAggState] = {}
        for edge in edges:
            self.edge_states[edge.edge_id] = EdgeAggState(
                edge_id=edge.edge_id,
                compute_factor=edge.compute_factor,
                global_params={
                    k: v.clone() for k, v in self.global_edge.state_dict().items()
                },
            )

        # Cloud aggregation state
        self.cloud_state = CloudAggState(
            global_end_params={
                k: v.clone() for k, v in self.global_end.state_dict().items()
            },
            global_edge_params={
                k: v.clone() for k, v in self.global_edge.state_dict().items()
            },
        )

        # Metrics
        self.round_rows: list[dict[str, Any]] = []
        self.decision_rows: list[dict[str, Any]] = []
        self.best_accuracy = 0.0

    # ── Event scheduling ──────────────────────────────────────────────

    def _schedule(self, delay: float, callback: Callable, label: str = "",
                  priority: int = 0) -> None:
        heapq.heappush(self.event_queue, AsyncEvent(
            self.current_time + delay, priority, callback, label,
        ))

    # ── Main loop ─────────────────────────────────────────────────────

    def run(self) -> dict[str, Any]:
        # Initial: start training for all ends
        for client in self.clients:
            self._start_end_training(client.client_id)

        # Main event loop
        while self.event_queue and not self.ended:
            event = heapq.heappop(self.event_queue)
            self.current_time = event.time
            self.event_count += 1
            event.callback()

            # Periodic evaluation checkpoint
            if self.event_count % self.config.eval_interval_events == 0:
                self._eval_checkpoint("periodic")

            if self.event_count >= self.config.max_events:
                self.ended = True

        # Final evaluation
        self._eval_checkpoint("final")

        return self._build_summary()

    # ── End training cycle ────────────────────────────────────────────

    def _mode_bonus(self, end_id: int) -> dict[str, float]:
        """Compute per-mode bonus: UCB exploration + historical deltas.

        Early events: exploration bonus encourages trying all available modes.
        Late events: exploitation based on observed accuracy deltas.
        Once a mode has enough trials, its bonus converges to the actual
        observed performance relative to the analytical estimate.
        """
        bonus: dict[str, float] = {}
        mode_data: dict[str, dict] = {}

        for (eid, mode), deltas in self.mode_history.items():
            if eid == end_id:
                avg_delta = sum(deltas) / len(deltas)
                mode_data[mode] = {"tries": len(deltas), "avg_delta": avg_delta}

        total_tries = sum(d["tries"] for d in mode_data.values())
        C = 0.06  # exploration scale (accuracy units)

        for mode in MODE_SPECS:
            data = mode_data.get(mode)
            if data is None:
                # Never tried → maximum exploration boost
                bonus[mode] = C * 3.0
            else:
                n = data["tries"]
                # UCB: bonus = C * sqrt(log(total_tries + 1) / (n + 1))
                ucb = C * math.sqrt(math.log(total_tries + 1) / (n + 1))
                # Exploitation: observed accuracy delta (bias correction)
                exploit = data["avg_delta"] * min(1.0, n / 3.0)
                bonus[mode] = ucb + exploit

        return bonus

    def _select_mode_for_end(self, end_id: int, allow_none: bool = False) -> Any:
        client = next(c for c in self.clients if c.client_id == end_id)
        end_state = self.end_states[end_id]
        edge = self.edge_by_id[client.edge_id]

        # Current resource load for this edge and cloud
        current_edge_load = self._edge_cpu_load.get(client.edge_id, 0.0)
        current_cloud_load = self._cloud_cpu_load

        candidates = enumerate_candidates(
            config=self.selection,
            client_id=client.client_id,
            edge_factor=edge.compute_factor,
            compute_factor=client.compute_factor,
            memory_capacity_factor=client.memory_capacity_factor,
            samples=client.samples,
            remaining_epsilon=end_state.remaining_epsilon,
            round_idx=self.edge_states[client.edge_id].version,
            rng=self.rng,
            policy=self.policy,
            current_edge_load=current_edge_load,
            current_cloud_load=current_cloud_load,
            allow_none=allow_none,
        )
        candidate = choose_candidate(
            candidates,
            policy=self.policy,
            rng=self.rng,
            require_feasible=self.selection.require_feasible,
            mode_bonus=self._mode_bonus(end_id),
            end_sensitivity=end_state.sensitivity,
            time_limit=self.selection.time_limit,
            remaining_epsilon=end_state.remaining_epsilon,
            compute_factor=end_state.compute_factor,
        )
        # Budget tracked by PrivacyAccountant.add_event() in _on_training_complete
        return client, candidate

    def _start_end_training(self, end_id: int) -> None:
        end_state = self.end_states[end_id]
        if end_state.training:
            return  # already training

        # When budget is too low for any DP mode, allow "none" mechanism fallback
        # so ends can still participate with HE/none on cloud-contributing modes
        min_dp_cost = min(
            _dp_cost_for_mode(self.selection, spec, {obj: "dp"})
            for spec in MODE_SPECS.values()
            for obj in spec.client_objects + spec.edge_to_cloud_objects
            if obj in {"emb", "grad", "upd", "weakemb", "strongemb", "pseudo_label"}
        )
        allow_none = end_state.remaining_epsilon < min_dp_cost

        # Re-evaluate mode periodically based on training cycles.
        # Using train cycles (not cloud aggregations) ensures sensitivity-based
        # re-evaluation happens even when cloud aggregation is infrequent.
        needs_reselect = (
            end_state.last_reselect_cloud_ver < 0
            or end_state.mode is None
            or end_state.train_cycles_since_reselect >= self.config.reselect_interval
        )
        if needs_reselect:
            client, candidate = self._select_mode_for_end(end_id, allow_none=allow_none)
            end_state.last_reselect_cloud_ver = self.cloud_state.version
            end_state.train_cycles_since_reselect = 0
            end_state.mode = candidate.mode
            end_state.mechanisms = candidate.mechanisms
        else:
            # Keep current mode — just re-evaluate for time/risk estimates
            client = next(c for c in self.clients if c.client_id == end_id)
            current_edge_load = self._edge_cpu_load.get(client.edge_id, 0.0)
            current_cloud_load = self._cloud_cpu_load
            candidates = enumerate_candidates(
                config=self.selection,
                client_id=client.client_id,
                edge_factor=self.edge_by_id[client.edge_id].compute_factor,
                compute_factor=client.compute_factor,
                memory_capacity_factor=client.memory_capacity_factor,
                samples=client.samples,
                remaining_epsilon=end_state.remaining_epsilon,
                round_idx=self.edge_states[client.edge_id].version,
                rng=self.rng,
                policy=self.policy,
                current_edge_load=current_edge_load,
                current_cloud_load=current_cloud_load,
                allow_none=allow_none,
            )
            candidate = next(
                (c for c in candidates if c.mode == end_state.mode and c.feasible),
                None,
            )
            if candidate is None:
                client, candidate = self._select_mode_for_end(end_id, allow_none=allow_none)
                end_state.mode = candidate.mode
                end_state.mechanisms = candidate.mechanisms

        spec = MODE_SPECS[candidate.mode] if candidate.mode != "SKIP" else None
        cpu_scale = (end_state.samples / 150.0) if spec else 0.0

        # Log decision
        self.decision_rows.append({
            "policy": self.policy,
            "event": self.event_count,
            "time": self.current_time,
            "end_id": end_id,
            "edge_id": client.edge_id,
            "mode": candidate.mode,
            "mechanisms": ";".join(f"{k}:{v}" for k, v in sorted(candidate.mechanisms.items())),
            "time_estimate": candidate.time,
            "risk": candidate.risk,
            "epsilon_used": candidate.epsilon_used,
            "remaining_epsilon": end_state.remaining_epsilon,
            "feasible": candidate.feasible,
            "edge_version": end_state.edge_version,
            "cloud_version": end_state.cloud_version,
            "sensitivity": end_state.sensitivity,
            "edge_cpu_load": self._edge_cpu_load.get(client.edge_id, 0.0),
            "cloud_cpu_load": self._cloud_cpu_load,
            "cpu_edge_demand": spec.edge_cpu * cpu_scale if spec else 0.0,
            "cpu_cloud_demand": spec.cloud_cpu * cpu_scale if spec else 0.0,
        })

        if candidate.mode == "SKIP":
            self._schedule(0.5 + self.rng.random(), lambda eid=end_id: self._start_end_training(eid),
                           f"end_{end_id}_retry")
            return

        end_state.training = True

        # Reserve edge/cloud CPU for this mode
        edge_demand = spec.edge_cpu * cpu_scale
        cloud_demand = spec.cloud_cpu * cpu_scale
        self._edge_cpu_load[client.edge_id] = self._edge_cpu_load.get(client.edge_id, 0.0) + edge_demand
        self._cloud_cpu_load += cloud_demand
        end_state.cpu_edge_demand = edge_demand
        end_state.cpu_cloud_demand = cloud_demand

        # Estimate training time (consistent with _estimate_candidate)
        L = self.selection.L_block_cycles
        E = spec.E_edge_loops
        local_load = spec.local_work * end_state.samples / 150.0
        local_time = (local_load / max(L, 1)) * end_state.effective_compute_factor
        edge_time = spec.edge_work * self.edge_by_id[client.edge_id].compute_factor * 0.75
        if candidate.mode in ("LIE", "LIEIIC", "LIEIIIC"):
            pipe_cycle = max(local_time, edge_time)
            block_compute = L * pipe_cycle + min(local_time, edge_time)
        else:
            block_compute = L * (local_time + edge_time)

        # Privacy processing overhead
        privacy_time = sum(PRIVACY_BASE_TIME[candidate.mechanisms[obj]] for obj in candidate.mechanisms)

        # Communication volume (uses correct sizes + HE expansion)
        bandwidth = 5.0 if spec.client_target == "edge" else 2.2
        comm_volume = sum(
            OBJECT_SIZES[obj] * PRIVACY_ALPHA[candidate.mechanisms[obj]]
            for obj in candidate.mechanisms
        )
        est_comm = comm_volume / max(bandwidth, 0.05)

        # Total time before submission: the end's own compute + privacy + communication
        est_time = block_compute + privacy_time + est_comm

        # Schedule training completion (use max to ensure minimum time)
        train_delay = max(0.1, est_time + self.rng.uniform(-0.05, 0.05))

        self._schedule(
            train_delay,
            lambda eid=end_id, mode=candidate.mode, mechs=candidate.mechanisms: self._on_training_complete(
                eid, mode, mechs,
            ),
            f"end_{end_id}_complete",
        )

    def _on_training_complete(
        self,
        end_id: int,
        mode: str,
        mechanisms: dict[str, str],
    ) -> None:
        end_state = self.end_states[end_id]
        client = next(c for c in self.clients if c.client_id == end_id)
        idx = self.client_indices[end_id]

        # Release CPU reservation (always, regardless of outcome)
        self._edge_cpu_load[client.edge_id] = max(
            0.0, self._edge_cpu_load.get(client.edge_id, 0.0) - end_state.cpu_edge_demand
        )
        self._cloud_cpu_load = max(0.0, self._cloud_cpu_load - end_state.cpu_cloud_demand)
        end_state.cpu_edge_demand = 0.0
        end_state.cpu_cloud_demand = 0.0

        if len(idx) == 0 or mode == "SKIP":
            end_state.training = False
            self._start_end_training(end_id)
            return

        # Record mode usage for experience tracking
        self.mode_usage_since_eval[mode] = self.mode_usage_since_eval.get(mode, 0) + 1

        effective_noise = self.config.dp_noise_multiplier
        if self.event_count < self.config.warmup_events:
            effective_noise *= self.config.warmup_factor

        # Real training with object-level DP on embeddings and gradients.
        t0 = time_module.perf_counter()
        try:
            state_diff = split_local_train_lenet5(
                mode=mode,
                global_end_state=end_state.cloud_params or self.global_end.state_dict(),
                global_edge_state=end_state.edge_params or self.global_edge.state_dict(),
                x=self.x_train[idx],
                y=self.y_train[idx],
                epochs=self.config.local_epochs,
                lr=self.config.learning_rate,
                device=self.device,
                mechanisms=mechanisms,
                dp_clip_norm=self.config.dp_clip_norm,
                dp_noise_multiplier=effective_noise,
                dp_rng=self.np_rng,
                dp_epsilon=max(self.selection.dp_emb_epsilon, 1e-6),
                training_seed=(
                    (int(self.selection.seed) + 1) * 1_000_003
                    + (int(self.event_count) + 1) * 10_007
                    + (int(end_id) + 1) * 101
                ) % (2**31 - 1),
            )
        except Exception:
            end_state.training = False
            self._schedule(0.1, lambda eid=end_id: self._start_end_training(eid),
                           f"end_{end_id}_retry_err")
            return

        measured = time_module.perf_counter() - t0

        # Update-level DP is applied only when the update object selects DP.
        has_update_dp = mechanisms.get("upd") == "dp"
        state_diff = apply_unified_dp(
            state_diff,
            mechanism="dp" if has_update_dp else "none",
            clip_norm=self.config.dp_clip_norm,
            noise_multiplier=effective_noise,
            rng=self.np_rng,
            device=self.device,
            adaptive=self.config.adaptive_clip,
        )

        # RDP accountant: formal (ε,δ)-DP tracking for reporting only
        # Uses the full noise (not warmup-reduced) for conservative accounting
        if any(m == "dp" for m in mechanisms.values()):
            self.accountants[end_id].add_event(self.config.dp_noise_multiplier)

        # Mode selection budget follows the same event-wise DP composition as selection.py.
        spec = MODE_SPECS[mode]
        mode_cost = _dp_cost_for_mode(self.selection, spec, mechanisms)
        end_state.remaining_epsilon = max(0.0, end_state.remaining_epsilon - mode_cost)

        # Version check: if stale, apply decay
        edge_state = self.edge_states[client.edge_id]
        version_gap = edge_state.version - end_state.edge_version

        submission = VersionedSubmission(
            end_id=end_id,
            edge_id=client.edge_id,
            mode=mode,
            state_diff=state_diff,
            samples=len(idx),
            base_edge_version=end_state.edge_version,
            base_cloud_version=end_state.cloud_version,
            submission_time=self.current_time,
            compute_time=measured,
            edge_cycle=end_state.edge_cycle,
        )

        end_state.total_submissions += 1
        end_state.total_compute_time += measured
        end_state.training = False
        end_state.train_cycles_since_reselect += 1

        # Route submission
        if spec.client_target == "cloud":
            # Direct to cloud buffer
            self._submit_to_cloud(submission)
        else:
            # Submit to edge buffer
            self._submit_to_edge(client.edge_id, submission)
            # For modes that need cloud forwarding after E cycles
            needs_cloud = bool(spec.edge_to_cloud_objects)
            if needs_cloud and end_state.edge_cycle + 1 >= spec.E_edge_loops:
                # Tag this end's next edge aggregation to forward to cloud
                submission.edge_cycle = end_state.edge_cycle

        # Increment edge cycle or start next round
        if spec.client_target != "cloud" and spec.edge_to_cloud_objects:
            end_state.edge_cycle += 1
            if end_state.edge_cycle >= spec.E_edge_loops:
                end_state.edge_cycle = 0
                # Will be forwarded to cloud after edge aggregation

        # Start next training round (lottery model: immediate)
        self._schedule(0.01, lambda eid=end_id: self._start_end_training(eid),
                       f"end_{end_id}_next")

    # ── Edge aggregation ──────────────────────────────────────────────

    def _submit_to_edge(self, edge_id: int, submission: VersionedSubmission) -> None:
        edge_state = self.edge_states[edge_id]
        edge_state.buffer.append(submission)

        if edge_state.buffer_full(self.config.B_edge):
            self._schedule(
                0.05, lambda eid=edge_id: self._aggregate_edge(eid),
                f"edge_{edge_id}_agg", priority=1,
            )

    def _aggregate_edge(self, edge_id: int) -> None:
        edge_state = self.edge_states[edge_id]
        buffer = edge_state.buffer
        edge_state.buffer = []  # clear for next round

        if not buffer:
            return

        # Filter: only aggregate submissions at the latest common version
        max_version = max(s.base_edge_version for s in buffer)
        current_version = edge_state.version

        valid = []
        for s in buffer:
            gap = current_version - s.base_edge_version
            if gap > self.config.max_version_gap:
                continue  # drop too-stale submissions
            weight = s.staleness_weight(current_version, self.config.decay_rate)
            valid.append((s, weight))

        if not valid:
            return

        # Separate submissions by their end_id for E-flow tracking
        needs_cloud_forward = False
        for s, _ in valid:
            spec = MODE_SPECS[s.mode]
            if spec.edge_to_cloud_objects and s.edge_cycle >= spec.E_edge_loops - 1:
                needs_cloud_forward = True

        # Weighted FedAvg
        total_weight = sum(w * s.samples for s, w in valid)
        if total_weight <= 0:
            return

        # Initialize aggregates
        edge_agg: dict[str, torch.Tensor] = {}
        for name, param in self.global_edge.named_parameters():
            edge_agg[name] = torch.zeros_like(param.data, device=self.device)

        end_agg: dict[str, torch.Tensor] = {}
        for name, param in self.global_end.named_parameters():
            end_agg[name] = torch.zeros_like(param.data, device=self.device)

        for s, w in valid:
            factor = w * s.samples / total_weight
            for name in edge_agg:
                if name in s.state_diff.get("edge", {}):
                    edge_agg[name] += factor * s.state_diff["edge"][name]
            for name in end_agg:
                if name in s.state_diff.get("end", {}):
                    end_agg[name] += factor * s.state_diff["end"][name]

        # Apply updates
        for name in edge_agg:
            self.global_edge.state_dict()[name].data += edge_agg[name]
        for name in end_agg:
            self.global_end.state_dict()[name].data += end_agg[name]

        # Increment edge version
        edge_state.version += 1
        edge_state.aggregation_count += 1
        edge_state.global_params = {
            k: v.clone() for k, v in self.global_edge.state_dict().items()
        }

        # Broadcast updated params to all ends under this edge
        for end_id, es in self.end_states.items():
            if es.edge_id == edge_id:
                es.edge_version = edge_state.version
                es.edge_params = edge_state.global_params

        # Cloud forwarding
        if needs_cloud_forward:
            # Create a consolidated submission for cloud
            cloud_sub = VersionedSubmission(
                end_id=-1,  # virtual
                edge_id=edge_id,
                mode="EDGE_AGG",
                state_diff={"end": end_agg, "edge": edge_agg},
                samples=sum(s.samples for s, _ in valid),
                base_edge_version=edge_state.version,
                base_cloud_version=self.cloud_state.version,
                submission_time=self.current_time,
                compute_time=0.0,
            )
            self._submit_to_cloud(cloud_sub)

    # ── Cloud aggregation ─────────────────────────────────────────────

    def _submit_to_cloud(self, submission: VersionedSubmission) -> None:
        self.cloud_state.buffer.append(submission)
        if self.cloud_state.buffer_full(self.config.B_cloud):
            self._schedule(
                0.1, self._aggregate_cloud,
                "cloud_agg", priority=2,
            )

    def _aggregate_cloud(self) -> None:
        cloud = self.cloud_state
        buffer = cloud.buffer
        cloud.buffer = []

        if not buffer:
            return

        # Same staleness filtering
        current_version = cloud.version
        valid = []
        for s in buffer:
            gap = current_version - s.base_cloud_version
            if gap > self.config.max_version_gap:
                continue
            weight = s.staleness_weight(current_version, self.config.decay_rate)
            valid.append((s, weight))

        if not valid:
            return

        total_weight = sum(w * s.samples for s, w in valid)

        # Initialize
        edge_agg: dict[str, torch.Tensor] = {}
        for name, param in self.global_edge.named_parameters():
            edge_agg[name] = torch.zeros_like(param.data, device=self.device)
        end_agg: dict[str, torch.Tensor] = {}
        for name, param in self.global_end.named_parameters():
            end_agg[name] = torch.zeros_like(param.data, device=self.device)

        for s, w in valid:
            factor = w * s.samples / max(total_weight, 1e-12)
            for name in edge_agg:
                if name in s.state_diff.get("edge", {}):
                    edge_agg[name] += factor * s.state_diff["edge"][name]
            for name in end_agg:
                if name in s.state_diff.get("end", {}):
                    end_agg[name] += factor * s.state_diff["end"][name]

        for name in edge_agg:
            self.global_edge.state_dict()[name].data += edge_agg[name]
        for name in end_agg:
            self.global_end.state_dict()[name].data += end_agg[name]

        cloud.version += 1
        cloud.aggregation_count += 1
        cloud.global_end_params = {
            k: v.clone() for k, v in self.global_end.state_dict().items()
        }
        cloud.global_edge_params = {
            k: v.clone() for k, v in self.global_edge.state_dict().items()
        }

        # Broadcast to all ends
        for es in self.end_states.values():
            es.cloud_version = cloud.version
            es.cloud_params = cloud.global_end_params

        # Evaluation checkpoint
        self._eval_checkpoint("cloud_agg")

    # ── Evaluation and metrics ────────────────────────────────────────

    def _eval_checkpoint(self, tag: str) -> None:
        loss, acc = split_evaluate(
            self.global_end, self.global_edge,
            self.x_test, self.y_test, self.device,
        )
        train_loss, train_acc = split_evaluate(
            self.global_end, self.global_edge,
            self.x_train, self.y_train, self.device,
        )
        self.best_accuracy = max(self.best_accuracy, acc)

        # Mode experience tracking: accuracy delta → mode history
        if self.last_eval_accuracy is not None and self.mode_usage_since_eval:
            delta = acc - self.last_eval_accuracy
            # Spread the accuracy change across all modes used since last eval
            for mode, count in self.mode_usage_since_eval.items():
                per_use = delta / max(count, 1)
                for es in self.end_states.values():
                    if es.mode == mode:
                        key = (es.end_id, mode)
                        self.mode_history.setdefault(key, []).append(per_use)
        self.last_eval_accuracy = acc
        self.mode_usage_since_eval.clear()

        # Compute staleness stats
        total_gap = 0
        max_gap = 0
        for es in self.end_states.values():
            eg = self.edge_states[es.edge_id].version - es.edge_version
            cg = self.cloud_state.version - es.cloud_version
            gap = max(eg, cg)
            total_gap += gap
            max_gap = max(max_gap, gap)

        num_active = sum(1 for es in self.end_states.values() if es.training)
        num_submitted = sum(es.total_submissions for es in self.end_states.values())

        # RDP formal (ε,δ)-DP guarantee (reporting only, not used for mode selection)
        rdp_epsilons = [acct.current_epsilon() for acct in self.accountants.values()]
        max_rdp_eps = max(rdp_epsilons) if rdp_epsilons else 0.0

        self.round_rows.append({
            "policy": self.policy,
            "event": self.event_count,
            "time": self.current_time,
            "tag": tag,
            "test_accuracy": acc,
            "test_loss": loss,
            "train_accuracy": train_acc,
            "best_accuracy": self.best_accuracy,
            "cloud_version": self.cloud_state.version,
            "edge_versions": ";".join(
                f"{eid}:{es.version}" for eid, es in self.edge_states.items()
            ),
            "mean_version_gap": total_gap / max(len(self.end_states), 1),
            "max_version_gap": max_gap,
            "active_ends": num_active,
            "total_submissions": num_submitted,
            "num_ends": len(self.end_states),
            "rdp_epsilon": max_rdp_eps,
            "edge_cpu_loads": ";".join(
                f"{eid}:{load:.1f}" for eid, load in sorted(self._edge_cpu_load.items())
            ),
            "cloud_cpu_load": self._cloud_cpu_load,
        })

    def _build_summary(self) -> dict[str, Any]:
        rows = self.round_rows
        if not rows:
            return {"policy": self.policy, "error": "no_events"}

        final = rows[-1]
        best = max(rows, key=lambda r: r["test_accuracy"])
        last_5 = rows[-5:]
        avg_last_5 = sum(r["test_accuracy"] for r in last_5) / max(len(last_5), 1)
        last_10 = rows[-10:]
        avg_last_10 = sum(r["test_accuracy"] for r in last_10) / max(len(last_10), 1)

        total_train_time = sum(
            es.total_compute_time for es in self.end_states.values()
        )
        end_agg_counts = sum(
            es.aggregation_count for es in self.edge_states.values()
        )

        # RDP formal guarantee (max across ends)
        rdp_eps = [acct.current_epsilon() for acct in self.accountants.values()]
        max_rdp_eps = max(rdp_eps) if rdp_eps else 0.0

        return {
            "policy": self.policy,
            "model": "lenet5_async",
            "dataset": "Fashion-MNIST",
            "events": len(rows),
            "final_test_accuracy": final["test_accuracy"],
            "best_test_accuracy": best["test_accuracy"],
            "avg_last_5_accuracy": avg_last_5,
            "avg_last_10_accuracy": avg_last_10,
            "round_to_best": best["event"],
            "final_train_accuracy": final["train_accuracy"],
            "total_simulation_time": final["time"],
            "total_train_time": total_train_time,
            "cloud_aggregations": self.cloud_state.aggregation_count,
            "edge_aggregations": end_agg_counts,
            "max_version_gap": max(r["max_version_gap"] for r in rows),
            "mean_version_gap": sum(r["mean_version_gap"] for r in rows) / max(len(rows), 1),
            "total_submissions": sum(r["total_submissions"] for r in rows) if rows else 0,
            "rdp_epsilon": max_rdp_eps,
            "output_dir": str(self.output_dir),
        }


def run_async_training(
    *,
    selection: SelectionConfig,
    async_config: AsyncConfig,
    clients: list[Any],
    edges: list[Any],
    edge_by_id: dict[int, Any],
    client_indices: list[np.ndarray],
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    output_dir: Path,
    policy: str,
    rng: random.Random,
    np_rng: np.random.Generator,
) -> dict[str, Any]:
    sim = AsyncSimulation(
        selection=selection,
        config=async_config,
        clients=clients,
        edges=edges,
        edge_by_id=edge_by_id,
        client_indices=client_indices,
        x_train=x_train,
        y_train=y_train,
        x_test=x_test,
        y_test=y_test,
        output_dir=output_dir,
        policy=policy,
        rng=rng,
        np_rng=np_rng,
    )
    summary = sim.run()

    # Write outputs
    from .fmnist_lenet5_dynamic import _write_csv, _write_json
    _write_csv(output_dir / "round_metrics.csv", sim.round_rows)
    _write_csv(output_dir / "client_decisions.csv", sim.decision_rows)
    _write_json(output_dir / "summary.json", summary)

    n_params = count_params(sim.global_end) + count_params(sim.global_edge)
    print(f"  [{policy}] model_params={n_params}, test_acc={summary['best_test_accuracy']:.4f}, "
          f"sim_time={summary['total_simulation_time']:.2f}s, "
          f"cloud_agg={summary['cloud_aggregations']}, edge_agg={summary['edge_aggregations']}")

    return summary


def _dp_cost_for_mode(selection: SelectionConfig, spec: Any, mechanisms: dict[str, str]) -> float:
    total = 0.0
    for obj, mechanism in mechanisms.items():
        if mechanism != "dp":
            continue
        if obj in {"emb", "grad", "weakemb", "strongemb", "pseudo_label"}:
            total += max(1, selection.L_block_cycles) * selection.dp_emb_epsilon
        elif obj == "upd":
            loops = max(1, spec.E_edge_loops) if obj in spec.edge_to_cloud_objects else 1
            total += loops * selection.dp_upd_epsilon
        else:
            total += selection.dp_event_epsilon
    return total
