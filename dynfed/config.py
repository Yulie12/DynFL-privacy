from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class TopologyConfig:
    num_clients: int = 30
    num_edges: int = 5
    aggregation_fraction: float = 0.5


@dataclass(frozen=True)
class RuntimeConfig:
    rounds: int = 30
    seed: int = 42
    local_update_base_time: float = 1.0
    edge_train_base_time: float = 0.8
    edge_aggregation_beta: float = 0.01
    edge_aggregation_fixed: float = 0.02
    cloud_aggregation_beta: float = 0.015
    cloud_aggregation_fixed: float = 0.04
    client_heterogeneity: float = 2.0
    edge_heterogeneity: float = 1.5
    slow_client_rate: float = 0.15
    slow_factor_low: float = 2.0
    slow_factor_high: float = 5.0
    network_jitter: float = 0.25


@dataclass(frozen=True)
class PrivacyConfig:
    mechanism: str = "DP"
    epsilon: float = 4.0


@dataclass(frozen=True)
class ModeConfig:
    name: str = "LIEIIC"
    edge_local_cycles: int = 1


@dataclass(frozen=True)
class OutputConfig:
    output_dir: str = "out/dynfed_privacy_sim"
    write_events: bool = True
    plot: bool = True


@dataclass(frozen=True)
class ExperimentConfig:
    topology: TopologyConfig = field(default_factory=TopologyConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    privacy: PrivacyConfig = field(default_factory=PrivacyConfig)
    mode: ModeConfig = field(default_factory=ModeConfig)
    output: OutputConfig = field(default_factory=OutputConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def output_path(self) -> Path:
        return Path(self.output.output_dir)
