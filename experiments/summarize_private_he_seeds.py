"""Summarize completed private HE diagnostics, never formal dynamic-policy results."""
import argparse
import json
import statistics
from pathlib import Path


def summarize(paths):
    rows, reference, seeds = [], None, set()
    for path in paths:
        report = json.loads(Path(path).read_text(encoding="utf-8"))
        config = report["config"]
        signature = {k: v for k, v in config.items() if k not in {"seed", "output_root", "model"}}
        signature["model"] = report["model"]
        if reference is not None and signature != reference:
            raise ValueError("Incompatible diagnostic configurations")
        reference = signature
        seed = config["seed"]
        if seed in seeds:
            raise ValueError("Duplicate seed")
        seeds.add(seed)
        values = report["results"]
        if report["status"] != "completed" or len(values) != config["rounds"]:
            raise ValueError("Incomplete or multiple-trajectory report")
        for index, value in enumerate(values, 1):
            he = value["he_metrics"]
            if (value["round"] != index or value["dp_events_per_client"] != index
                    or value["method"] != "distributed_dp_he"
                    or he["backend"] != "seal" or he["cloud_secret_key_transmitted"]
                    or he["cloud_pid"] == he["custodian_pid"]
                    or he["paired_max_abs_error"] > 1e-5
                    or he["encrypted_parameter_values"] != value["dimensions"] * config["edges"]
                    or value["max_client_epsilon"] > config["epsilon"] + 1e-8):
                raise ValueError("Diagnostic HE or accounting invariant failed")
        rows.append(dict(seed=seed, source=str(Path(path).resolve()),
                         final_accuracy=values[-1]["accuracy"],
                         best_accuracy=max(v["accuracy"] for v in values),
                         last10_accuracy=statistics.mean(v["accuracy"] for v in values[-10:]),
                         final_loss=values[-1]["loss"],
                         epsilon=values[-1]["max_client_epsilon"],
                         wall_sec=report["wall_time_sec"],
                         he_sec=sum(v["he_metrics"]["wall_time_sec"] for v in values),
                         max_he_error=max(v["he_metrics"]["paired_max_abs_error"] for v in values)))
    if not rows:
        raise ValueError("No reports")
    metrics = {}
    for key in ("final_accuracy", "best_accuracy", "last10_accuracy", "final_loss"):
        samples = [row[key] for row in rows]
        metrics[key] = dict(mean=statistics.mean(samples),
                            sample_std=statistics.stdev(samples) if len(samples) > 1 else None)
    return dict(scope="private_fixed_architecture_diagnostic_not_Ours", config=reference,
                seeds=sorted(rows, key=lambda row: row["seed"]), metrics=metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    result = summarize(args.reports)
    Path(args.output).write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(result["metrics"], indent=2))
