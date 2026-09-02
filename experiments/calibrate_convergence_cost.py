from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import stats

from aggregate_multiseed_results import discover_sources


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Measure the empirical association between the selected convergence "
            "error cost and realized post update model utility."
        )
    )
    parser.add_argument("--root", type=Path, default=ROOT / "out")
    parser.add_argument("--seeds", type=int, nargs="+", required=True)
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--model", default="resnet18_pretrained")
    parser.add_argument("--rounds", type=int, default=200)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--edges", type=int, default=10)
    parser.add_argument("--selection-period", type=int, default=1)
    parser.add_argument("--privacy-budget", type=float, default=8.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_rounds(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    rows.sort(key=lambda row: int(float(row["round"])))
    return rows


def _finite_float(row: dict[str, str], key: str) -> float | None:
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return None
    return value if math.isfinite(value) else None


def _zscore(values: list[float]) -> list[float]:
    array = np.asarray(values, dtype=np.float64)
    std = float(array.std(ddof=0))
    if std <= 1e-12:
        return [0.0] * len(values)
    return [float(value) for value in ((array - array.mean()) / std)]


def _correlation(x: list[float], y: list[float]) -> dict[str, float]:
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return {
            "spearman_r": float("nan"),
            "spearman_p": float("nan"),
            "kendall_tau": float("nan"),
            "kendall_p": float("nan"),
        }
    spearman = stats.spearmanr(x, y)
    kendall = stats.kendalltau(x, y)
    return {
        "spearman_r": float(spearman.statistic),
        "spearman_p": float(spearman.pvalue),
        "kendall_tau": float(kendall.statistic),
        "kendall_p": float(kendall.pvalue),
    }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise RuntimeError(f"No rows available for {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    seeds = list(dict.fromkeys(int(seed) for seed in args.seeds))
    sources = discover_sources(
        root=args.root,
        seeds=seeds,
        policies=["ours"],
        dataset=args.dataset,
        model=args.model,
        rounds=args.rounds,
        clients=args.clients,
        edges=args.edges,
        selection_period=args.selection_period,
        privacy_budget=args.privacy_budget,
    )
    missing = [seed for seed in seeds if (seed, "ours") not in sources]
    if missing:
        raise RuntimeError(f"Missing compatible Ours runs for seeds {missing}")

    observations: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    pooled_by_metric: dict[str, tuple[list[float], list[float]]] = {
        "post_update_test_loss": ([], []),
        "test_loss_change": ([], []),
        "test_accuracy_change": ([], []),
    }

    for seed in seeds:
        source = sources[(seed, "ours")]
        rounds = _read_rounds(source.policy_dir / "round_metrics.csv")
        run_observations: list[dict[str, float]] = []
        for index, row in enumerate(rounds):
            omega = _finite_float(row, "system_omega_objective")
            test_loss = _finite_float(row, "test_loss")
            test_accuracy = _finite_float(row, "test_accuracy")
            if omega is None or test_loss is None or test_accuracy is None:
                continue
            previous_loss = (
                _finite_float(rounds[index - 1], "test_loss") if index > 0 else None
            )
            previous_accuracy = (
                _finite_float(rounds[index - 1], "test_accuracy") if index > 0 else None
            )
            item = {
                "round": float(row["round"]),
                "omega": omega,
                "post_update_test_loss": test_loss,
                "test_loss_change": (
                    test_loss - previous_loss if previous_loss is not None else float("nan")
                ),
                "test_accuracy_change": (
                    test_accuracy - previous_accuracy
                    if previous_accuracy is not None
                    else float("nan")
                ),
            }
            run_observations.append(item)
            observations.append({"seed": seed, **item})

        omega_values = [item["omega"] for item in run_observations]
        omega_z = _zscore(omega_values)
        for metric in pooled_by_metric:
            pairs = [
                (item["omega"], item[metric])
                for item in run_observations
                if math.isfinite(item[metric])
            ]
            x = [pair[0] for pair in pairs]
            y = [pair[1] for pair in pairs]
            result = _correlation(x, y)
            summary_rows.append(
                {
                    "scope": f"seed_{seed}",
                    "outcome": metric,
                    "n": len(pairs),
                    **result,
                }
            )

            valid_indices = [
                index
                for index, item in enumerate(run_observations)
                if math.isfinite(item[metric])
            ]
            metric_values = [run_observations[index][metric] for index in valid_indices]
            metric_z = _zscore(metric_values)
            pooled_by_metric[metric][0].extend(omega_z[index] for index in valid_indices)
            pooled_by_metric[metric][1].extend(metric_z)

    for metric, (x, y) in pooled_by_metric.items():
        summary_rows.append(
            {
                "scope": "pooled_within_seed_standardized",
                "outcome": metric,
                "n": len(x),
                **_correlation(x, y),
            }
        )

    output_dir = args.output_dir.resolve()
    _write_csv(output_dir / "convergence_cost_observations.csv", observations)
    _write_csv(output_dir / "convergence_cost_calibration.csv", summary_rows)
    print(output_dir / "convergence_cost_calibration.csv")


if __name__ == "__main__":
    main()
