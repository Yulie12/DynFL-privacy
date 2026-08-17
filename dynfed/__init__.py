"""CPU-friendly simulators for dynamic cloud-edge-end federated learning."""

from .config import ExperimentConfig
from .training import run_experiment

__all__ = ["ExperimentConfig", "run_experiment"]
