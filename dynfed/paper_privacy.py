"""Unamplified Gaussian RDP, matching the conservative accountant in main.tex."""
from dataclasses import dataclass, field
import math

ORDERS = (1.25, 1.5, 1.75, *range(2, 65), 96, 128, 192, 256)


@dataclass
class Ledger:
    budget: float
    delta: float
    rdp: list[float] = field(default_factory=lambda: [0.0] * len(ORDERS))
    events: int = 0

    def epsilon(self, events=0, sigma=None):
        if events < 0 or (events and (sigma is None or sigma <= 0 or not math.isfinite(sigma))):
            raise ValueError("invalid Gaussian event projection")
        if self.events == 0 and events == 0:
            return 0.0
        return min(cost + (events * order / (2 * sigma * sigma) if events else 0)
                   + math.log(1 / self.delta) / (order - 1)
                   for order, cost in zip(ORDERS, self.rdp))

    def calibrate(self, remaining_events):
        if remaining_events <= 0:
            return None
        if self.epsilon() >= self.budget:
            return None
        low, high = 0.0, 1.0
        while self.epsilon(remaining_events, high) > self.budget * (1 - 1e-9):
            high *= 2
            if high > 1e12:
                return None
        for _ in range(80):
            mid = (low + high) / 2
            if self.epsilon(remaining_events, mid) > self.budget * (1 - 1e-9):
                low = mid
            else:
                high = mid
        return high

    def commit(self, events, sigma):
        if self.epsilon(events, sigma) > self.budget + 1e-8:
            raise RuntimeError("executed Gaussian mechanisms exceed lifetime budget")
        self.rdp = [value + events * order / (2 * sigma * sigma)
                    for value, order in zip(self.rdp, ORDERS)]
        self.events += events
