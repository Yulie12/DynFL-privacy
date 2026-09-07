import csv

from experiments.validate_privacy_paths import collect_result


def write_metrics(tmp_path, **changes):
    row = dict(test_accuracy="0.3", test_loss="1.0", training_health="finite",
               logical_time="500", cumulative_wall_time_sec="12",
               update_dp_noise_to_signal_ratio="0.1", update_dp_release_count="1",
               max_update_epsilon="0.7", he_execution_status="real",
               he_wall_time_sec="2.5", he_max_abs_error="0.0000001")
    row.update(changes)
    with (tmp_path / "round_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=row)
        writer.writeheader()
        writer.writerow(row)


def test_report_distinguishes_real_execution_and_time_definitions(tmp_path):
    write_metrics(tmp_path)
    result = collect_result(tmp_path, 0)
    assert result["real_he_rounds"] == 1
    assert result["profiled_he_rounds"] == 0
    assert result["logical_time_sec"] == 500
    assert result["wall_time_sec"] == 12
    assert result["he_max_abs_error"] == 1e-7
    assert result["dp_release_count"] == 1


def test_failed_metrics_not_reported_as_success(tmp_path):
    write_metrics(tmp_path, training_health="non_finite", test_loss="nan")
    result = collect_result(tmp_path, 1)
    assert result["status"] == "failed_non_finite"
    assert result["final_loss"] is None


def test_missing_metrics_are_a_failure_even_with_zero_exit(tmp_path):
    assert collect_result(tmp_path, 0)["status"] == "failed_no_metrics"
