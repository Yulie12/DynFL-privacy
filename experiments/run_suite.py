"""Matched paper ablations using the same strict configuration schema."""
import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dynfed.paper_config import PaperConfig
from dynfed.paper_runner import run, write_json


def matrix(base):
    cases = [("main", base)]
    proposed = replace(base, policies=("proposed",))
    for period in (1, 5, 10):
        cases.append((f"period_{period}", replace(proposed, selection_period=period)))
    for gamma_d in (0.5, 0.9, 0.99):
        for gamma_c in (0.5, 0.9, 0.99):
            cases.append((f"history_{gamma_d}_{gamma_c}", replace(proposed, gamma_d=gamma_d, gamma_c=gamma_c)))
    for stages in (1, 2, 3):
        cases.append((f"edge_stages_{stages}", replace(proposed, edge_stages=stages)))
    for epsilon in (4.0, 8.0, 12.0):
        cases.append((f"epsilon_{epsilon}", replace(proposed, epsilon=epsilon)))
    cases.extend([
        ("resource", replace(proposed, client_heterogeneity=3.0)),
        ("network", replace(proposed, network_jitter=0.5)),
        ("memory", replace(proposed, client_memory_mb=20.0)),
        ("edge_capacity", replace(proposed, edge_group_capacity=5)),
        ("small_exact", replace(proposed, clients=6, edges=2, policies=("proposed", "exhaustive"),
                                evaluation_budget=729)),
    ])
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper_fmnist.json")
    parser.add_argument("--scenario", nargs="+")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cases = matrix(PaperConfig.load(args.config))
    if args.scenario:
        unknown = set(args.scenario) - {name for name, _ in cases}
        if unknown:
            raise ValueError(f"unknown scenarios: {sorted(unknown)}")
        cases = [(name, cfg) for name, cfg in cases if name in args.scenario]
    output = args.output or ROOT / "out/paper_suite" / datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.dry_run:
        for name, cfg in cases:
            print(name, cfg.clients, cfg.rounds, cfg.seeds, cfg.policies)
        return
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "suite_manifest.json", {name: cfg.to_dict() for name, cfg in cases})
    for name, cfg in cases:
        run(cfg, output / name)


if __name__ == "__main__":
    main()
