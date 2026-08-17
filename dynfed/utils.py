from __future__ import annotations

from dataclasses import replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

from .config import ExperimentConfig, ModeConfig, OutputConfig, PrivacyConfig


def timestamped_dir(root: str | Path, name: str) -> Path:
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    return Path(root) / f"{stamp}_{name}"


def with_output(config: ExperimentConfig, output_dir: str | Path) -> ExperimentConfig:
    return replace(config, output=replace(config.output, output_dir=str(output_dir)))


def expand_privacy_sweep(
    config: ExperimentConfig,
    mechanisms: Iterable[str],
    epsilons: Iterable[float],
    output_root: Path,
) -> list[ExperimentConfig]:
    configs = []
    for mechanism in mechanisms:
        values = list(epsilons) if mechanism.strip().lower() == "dp" else [config.privacy.epsilon]
        for epsilon in values:
            run_name = f"{config.mode.name}_{mechanism}_eps{epsilon}".replace(".", "p")
            configs.append(
                replace(
                    config,
                    privacy=PrivacyConfig(mechanism=mechanism, epsilon=float(epsilon)),
                    output=OutputConfig(
                        output_dir=str(output_root / run_name),
                        write_events=config.output.write_events,
                        plot=config.output.plot,
                    ),
                )
            )
    return configs


def expand_mode_sweep(
    config: ExperimentConfig,
    modes: Iterable[str],
    output_root: Path,
) -> list[ExperimentConfig]:
    configs = []
    for mode in modes:
        configs.append(
            replace(
                config,
                mode=ModeConfig(name=mode, edge_local_cycles=config.mode.edge_local_cycles),
                output=OutputConfig(
                    output_dir=str(output_root / mode),
                    write_events=config.output.write_events,
                    plot=config.output.plot,
                ),
            )
        )
    return configs
