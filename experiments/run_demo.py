from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.config import ExperimentConfig, ModeConfig, OutputConfig, PrivacyConfig, RuntimeConfig, TopologyConfig
from dynfed.training import run_experiment


def main() -> None:
    config = ExperimentConfig(
        topology=TopologyConfig(num_clients=12, num_edges=3),
        runtime=RuntimeConfig(rounds=5, seed=7),
        privacy=PrivacyConfig(mechanism="DP", epsilon=2.0),
        mode=ModeConfig(name="LIEIIC"),
        output=OutputConfig(output_dir="out/demo_lieiic_dp", plot=False),
    )
    summary = run_experiment(config)
    print(summary)


if __name__ == "__main__":
    main()
