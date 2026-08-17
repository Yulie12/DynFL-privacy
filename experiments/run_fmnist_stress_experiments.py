from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.fmnist_dynamic_training import FmnistDynamicConfig, run_fmnist_dynamic_training
from dynfed.real_training import RealTrainingConfig
from dynfed.selection import SelectionConfig
from dynfed.utils import timestamped_dir


DEFAULT_POLICIES = (
    "ours",
    "ours_ideal",
    "ours_knee",
    "ours_time_first",
    "ours_acc_first",
    "fixed_dp",
    "fixed_he",
    "random",
    "privacy_only",
    "no_protection",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run FMNIST stress experiments for per-client dynamic FedAvg selection."
    )
    parser.add_argument("--data-root", default="E:/YTT/GROUP/DriftRace/data/fmnist/FashionMNIST/raw")
    parser.add_argument("--output-root", default="out/fmnist_stress")
    parser.add_argument("--stress", choices=["resource", "epsilon", "jitter", "time", "all"], default="all")
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--edges", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.12)
    parser.add_argument("--iid", action="store_true")
    parser.add_argument("--require-feasible", action="store_true")
    parser.add_argument("--policies", nargs="+", default=list(DEFAULT_POLICIES), choices=list(DEFAULT_POLICIES))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = timestamped_dir(args.output_root, args.stress)
    training = RealTrainingConfig(
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        iid=args.iid,
    )

    aggregate_rows = []
    for case_name, overrides in _build_cases(args.stress):
        case_dir = output_root / case_name
        selection = SelectionConfig(
            rounds=args.rounds,
            num_clients=args.clients,
            num_edges=args.edges,
            seed=args.seed,
            output_dir=str(case_dir),
            initial_epsilon=overrides.get("initial_epsilon", 4.0),
            dp_event_epsilon=overrides.get("dp_event_epsilon", 0.05),
            resource_limit=overrides.get("resource_limit", 1.35),
            time_limit=overrides.get("time_limit", 8.0),
            risk_limit=overrides.get("risk_limit", 0.5),
            client_heterogeneity=overrides.get("client_heterogeneity", 2.0),
            edge_heterogeneity=overrides.get("edge_heterogeneity", 1.5),
            network_jitter=overrides.get("network_jitter", 0.25),
            require_feasible=args.require_feasible,
        )
        result = run_fmnist_dynamic_training(
            FmnistDynamicConfig(
                selection=selection,
                training=training,
                data_root=args.data_root,
                train_limit=args.train_limit,
                test_limit=args.test_limit,
                policies=tuple(args.policies),
            )
        )
        for row in result["summaries"]:
            aggregate = {"stress_case": case_name}
            aggregate.update(overrides)
            aggregate.update(row)
            aggregate_rows.append(aggregate)
        print(f"[OK] completed {case_name}: {result['summary_table']}")

    _write_summary(output_root / "aggregate_summary.csv", aggregate_rows)
    print(f"[OK] aggregate summary: {output_root / 'aggregate_summary.csv'}")


def _build_cases(stress: str) -> list[tuple[str, dict[str, float]]]:
    cases: list[tuple[str, dict[str, float]]] = []
    if stress in {"resource", "all"}:
        for value in [1.8, 1.35, 1.0, 0.75]:
            cases.append((f"resource_limit_{str(value).replace('.', 'p')}", {"resource_limit": value}))
    if stress in {"epsilon", "all"}:
        for value in [8.0, 4.0, 2.0, 1.0, 0.5]:
            cases.append((f"epsilon_{str(value).replace('.', 'p')}", {"initial_epsilon": value}))
    if stress in {"jitter", "all"}:
        for value in [0.0, 0.25, 0.5, 0.75]:
            cases.append((f"network_jitter_{str(value).replace('.', 'p')}", {"network_jitter": value}))
    if stress in {"time", "all"}:
        for value in [5.0, 6.0, 8.0, 10.0, 12.0]:
            cases.append((f"time_limit_{str(value).replace('.', 'p')}", {"time_limit": value}))
    return cases


def _write_summary(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
