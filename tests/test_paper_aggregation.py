from __future__ import annotations

import csv
from pathlib import Path

from experiments.aggregate_multiseed_results import SourceRun, aggregate_time


def _write_metrics(path: Path, times: list[float], accuracy: list[float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["round", "accounted_system_time_sec", "test_accuracy"],
        )
        writer.writeheader()
        for index, (wall_time, value) in enumerate(zip(times, accuracy)):
            writer.writerow(
                {
                    "round": index,
                    "accounted_system_time_sec": wall_time,
                    "test_accuracy": value,
                }
            )


def test_time_aggregation_uses_accounted_system_time(tmp_path: Path) -> None:
    sources = {}
    for seed, times, values in (
        (40, [2.0, 4.0, 6.0], [0.1, 0.3, 0.5]),
        (42, [3.0, 5.0, 7.0], [0.2, 0.4, 0.6]),
    ):
        policy_dir = tmp_path / str(seed) / "ours"
        _write_metrics(policy_dir / "round_metrics.csv", times, values)
        sources[(seed, "ours")] = SourceRun(
            seed=seed,
            policy="ours",
            run_dir=policy_dir.parent,
            policy_dir=policy_dir,
            config={},
            summary={},
        )

    rows, plotted, horizon = aggregate_time(
        sources,
        [40, 42],
        ["ours"],
        rounds=3,
        grid_points=3,
    )

    assert horizon == 6.0
    assert plotted["ours"]["time"] == [3.0, 4.5, 6.0]
    assert rows[0]["test_accuracy_mean"] == plotted["ours"]["mean"][0]
    assert all("ema" not in key.lower() for key in rows[0])
