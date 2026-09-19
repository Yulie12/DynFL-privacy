from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = ROOT / "configs" / "paper_v30_cifar10_resnet18.json"
DEFAULT_OUTPUT = ROOT / "out" / "paper_v31_final" / "table1_parameters.csv"

# Frozen Q98 Table I: report formal system/training/privacy parameters directly
# from the executable paper configuration instead of maintaining a second copy.
TABLE1_FIELDS = (
    ("Training", "Dataset", "training.dataset"),
    ("Training", "Model", "training.model"),
    ("Training", "Rounds", "training.rounds"),
    ("Training", "Local epochs", "training.local_epochs"),
    ("Training", "Learning rate", "training.learning_rate"),
    ("Training", "Strategy period S_P", "training.selection_period"),
    ("System", "Clients", "system.clients"),
    ("System", "Edges", "system.edges"),
    ("System", "Client heterogeneity", "system.client_heterogeneity"),
    ("System", "Edge heterogeneity", "system.edge_heterogeneity"),
    ("Network", "End-Edge rate (MB/s)", "network.end_edge_rate_mb_s"),
    ("Network", "End-Cloud rate (MB/s)", "network.end_cloud_rate_mb_s"),
    ("Network", "Edge-Cloud rate (MB/s)", "network.edge_cloud_rate_mb_s"),
    ("Privacy", "Lifetime update epsilon", "privacy.update_epsilon_budget"),
    ("Privacy", "Delta", "privacy.delta"),
    ("Privacy", "Clipping norm C", "privacy.clip_norm"),
    ("Privacy", "Accounting mode", "privacy.accounting_mode"),
    ("Privacy", "Candidate mechanisms", "privacy.candidate_mechanisms"),
    ("Optimization", "Pareto archive size", "optimization.pareto_archive_size"),
    ("Optimization", "Pareto beam size", "optimization.pareto_beam_size"),
    ("Optimization", "Pareto max iterations", "optimization.pareto_max_iters"),
)


def _get(config: dict[str, Any], path: str) -> Any:
    value: Any = config
    for key in path.split("."):
        value = value[key]
    return value


def build_table1_rows(config: dict[str, Any]) -> list[dict[str, str]]:
    rows = []
    for section, parameter, path in TABLE1_FIELDS:
        value = _get(config, path)
        if isinstance(value, list):
            rendered = ", ".join(str(item) for item in value)
        elif isinstance(value, bool):
            rendered = str(value).lower()
        else:
            rendered = str(value)
        rows.append({"section": section, "parameter": parameter, "value": rendered, "config_path": path})
    return rows


def write_table1(rows: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("section", "parameter", "value", "config_path"))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate frozen Q98 Table I from the formal executable config.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.resolve().read_text(encoding="utf-8"))
    rows = build_table1_rows(config)
    write_table1(rows, args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
