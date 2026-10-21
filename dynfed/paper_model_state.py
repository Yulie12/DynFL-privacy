"""Availability-timed Edge replicas for FL_E, independent of Cloud returns."""
from dataclasses import dataclass
import torch


@dataclass
class Feedback:
    edge: int
    ready: float
    model: torch.Tensor
    round_index: int
    identity: str


class EdgeModels:
    def __init__(self, initial, edges):
        self.states = {edge: (initial.clone(), -1, f"edge:{edge}:initial") for edge in range(edges)}
        self.pending = []

    def advance(self, now):
        available = sorted((item for item in self.pending if item.ready <= now+1e-12),
                           key=lambda item: (item.ready, item.identity))
        for item in available:
            self.states[item.edge] = (item.model.clone(), item.round_index,
                                      f"edge:{item.edge}:{item.identity}")
        self.pending = [item for item in self.pending if item.ready > now+1e-12]

    def dispatch(self, edge):
        values, round_index, source = self.states[edge]
        return values.clone(), round_index, source

    def defer(self, edge, ready, model, round_index, identity):
        self.pending.append(Feedback(edge, ready, model.clone(), round_index, identity))
