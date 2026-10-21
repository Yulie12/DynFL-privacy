"""Equations global_ema, mode_deviation_ema and predicted_cloud_update."""
from dataclasses import dataclass, field
import math
import torch
from .paper_config import MODES


@dataclass
class History:
    gamma_d: float = 0.9
    gamma_c: float = 0.9
    epsilon_j: float = 1e-12
    reference: torch.Tensor | None = None
    deviations: dict[str, torch.Tensor] = field(default_factory=dict)

    def __post_init__(self):
        if not 0 <= self.gamma_d < 1 or not 0 <= self.gamma_c < 1:
            raise ValueError("EMA smoothing must be in [0, 1)")
        if not math.isfinite(self.epsilon_j) or self.epsilon_j <= 0:
            raise ValueError("epsilon_j must be finite and positive")

    def predict(self, masses):
        """masses contains ONLY accepted Cloud unit sample counts, grouped by mode."""
        if any(mode not in MODES or not math.isfinite(mass) or mass < 0
               for mode, mass in masses.items()):
            raise ValueError("invalid Cloud mode sample mass")
        active = {mode: mass for mode, mass in masses.items() if mass > 0}
        if not active:
            return None, None, "no_cloud_update"
        if self.reference is None:
            return None, None, "initializing_global_reference"
        missing = sorted(set(active) - set(self.deviations))
        if missing:
            return None, None, "initializing_modes:" + ",".join(missing)
        total = sum(active.values())
        deviation = sum((self.deviations[mode] * (mass / total)
                         for mode, mass in active.items()), torch.zeros_like(self.reference))
        predicted = self.reference + deviation
        j = float(deviation.square().sum() / (self.reference.square().sum() + self.epsilon_j))
        return predicted, j, "ready"

    def observe(self, units):
        """Score against the frozen prior reference, then update deviations, then reference.

        Each unit has mode, samples, update and accepted identity. Pending or
        rejected units must never be supplied. The first Cloud update seeds only
        the reference: d_k is undefined until that reference exists.
        """
        if not units:
            return {"observed_j": None, "cloud_update": False}
        vectors = [unit.update.detach().cpu().double().flatten() for unit in units]
        if any(unit.mode not in MODES or unit.samples <= 0 for unit in units):
            raise ValueError("invalid accepted Cloud unit")
        if any(not torch.isfinite(vec).all() or vec.shape != vectors[0].shape for vec in vectors):
            raise ValueError("nonfinite or inconsistent Cloud updates")
        total = sum(unit.samples for unit in units)
        weights = [unit.samples / total for unit in units]
        cloud = sum((vec * weight for vec, weight in zip(vectors, weights)),
                    torch.zeros_like(vectors[0]))
        masses = {mode: sum(unit.samples for unit in units if unit.mode == mode) for mode in MODES}
        predicted, predicted_j, status = self.predict(masses)
        prior = None if self.reference is None else self.reference.clone()
        observed_j = None
        if prior is not None:
            observed_j = float((cloud - prior).square().sum() /
                               (prior.square().sum() + self.epsilon_j))
            for mode, mass in masses.items():
                if mass == 0:
                    continue
                mean = sum((vec * unit.samples for unit, vec in zip(units, vectors)
                            if unit.mode == mode), torch.zeros_like(cloud)) / mass
                deviation = mean - prior
                previous = self.deviations.get(mode)
                self.deviations[mode] = (deviation.clone() if previous is None else
                                        self.gamma_d * previous + (1 - self.gamma_d) * deviation)
        self.reference = (cloud.clone() if prior is None else
                          self.gamma_c * prior + (1 - self.gamma_c) * cloud)
        return {"cloud_update": True, "predicted_j": predicted_j, "observed_j": observed_j,
                "prediction_status": status,
                "prediction_absolute_error": None if predicted_j is None else abs(predicted_j - observed_j),
                "predicted_update_error_l2": None if predicted is None else float((predicted - cloud).norm()),
                "weights": weights, "accepted_samples": total,
                "reference_before": prior, "cloud_delta": cloud,
                "predicted_update": predicted}

    def state_dict(self):
        return {"schema": "paper.deviation_history.v1", "gamma_d": self.gamma_d,
                "gamma_c": self.gamma_c, "epsilon_j": self.epsilon_j,
                "reference": self.reference, "deviations": self.deviations}
