from __future__ import annotations

import csv
import json
from pathlib import Path

from experiments.aggregate_multiseed_results import SourceRun, aggregate_time, discover_sources
from dynfed.version import CURRENT_EXECUTION_REVISION, CURRENT_UPDATE_PARAMETER_SCOPE


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


def test_discovery_excludes_results_from_old_update_parameter_scope(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    policy_dir = run_dir / "ours"
    _write_metrics(policy_dir / "round_metrics.csv", [1.0, 2.0], [0.1, 0.2])
    (policy_dir / "summary.json").write_text(json.dumps({"rounds": 2}), encoding="utf-8")
    config = {
        "selection": {
            "seed": 42, "rounds": 2, "num_clients": 4, "num_edges": 2,
            "initial_epsilon": 8.0,
        },
        "training": {
            "dataset_name": "cifar10", "model_name": "resnet18_pretrained",
            "selection_period": 1, "execution_revision": CURRENT_EXECUTION_REVISION,
        },
    }
    kwargs = dict(
        root=tmp_path, seeds=[42], policies=["ours"], dataset="cifar10",
        model="resnet18_pretrained", rounds=2, clients=4, edges=2,
        selection_period=1, privacy_budget=8.0,
    )
    config_path = run_dir / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert discover_sources(**kwargs) == {}
    config["training"]["update_parameter_scope"] = CURRENT_UPDATE_PARAMETER_SCOPE
    config_path.write_text(json.dumps(config), encoding="utf-8")
    assert set(discover_sources(**kwargs)) == {(42, "ours")}
