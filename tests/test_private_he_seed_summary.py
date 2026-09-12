import json

import pytest

from experiments.summarize_private_he_seeds import summarize


def report(seed):
    return dict(status="completed", model="head", wall_time_sec=1,
                config=dict(seed=seed, rounds=1, edges=2, epsilon=8),
                results=[dict(round=1, dp_events_per_client=1, method="distributed_dp_he",
                              dimensions=3, accuracy=.4, loss=2, max_client_epsilon=1,
                              he_metrics=dict(backend="seal", cloud_secret_key_transmitted=False,
                                              cloud_pid=2, custodian_pid=1, paired_max_abs_error=1e-8,
                                              encrypted_parameter_values=6, wall_time_sec=.1))])


def test_summary_and_duplicate_rejection(tmp_path):
    paths = [tmp_path / "a.json", tmp_path / "b.json"]
    for seed, path in enumerate(paths):
        path.write_text(json.dumps(report(seed)))
    result = summarize(paths)
    assert result["metrics"]["final_accuracy"] == dict(mean=.4, sample_std=0)
    with pytest.raises(ValueError, match="Duplicate"):
        summarize([paths[0], paths[0]])


@pytest.mark.parametrize("fault", ["status", "count", "key", "error", "budget"])
def test_summary_rejects_invalid_runs(tmp_path, fault):
    data = report(42)
    if fault == "status":
        data["status"] = "running"
    elif fault == "count":
        data["results"] = []
    elif fault == "key":
        data["results"][0]["he_metrics"]["cloud_secret_key_transmitted"] = True
    elif fault == "error":
        data["results"][0]["he_metrics"]["paired_max_abs_error"] = .1
    else:
        data["results"][0]["max_client_epsilon"] = 9
    path = tmp_path / "report.json"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError):
        summarize([path])
