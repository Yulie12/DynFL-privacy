"""Paired-seed summaries of completed paper runs; unreachable targets stay null."""
import argparse
import csv
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynfed.paper_runner import write_csv, write_json


def summarize(root, target):
    # Never trust the existence of summary.json or an old audit.json alone.
    from experiments.audit_paper import audit
    audit(root)
    records = []
    for path in sorted(Path(root).rglob("summary.json")):
        if not (path.parent / "round_metrics.csv").is_file():
            continue
        row = json.loads(path.read_text(encoding="utf-8"))
        with (path.parent / "round_metrics.csv").open(encoding="utf-8-sig") as handle:
            rounds = list(csv.DictReader(handle))
        row["run"] = str(path.parent)
        row["target_reached"] = any(float(item["test_accuracy"]) >= target for item in rounds)
        row["time_to_target"] = next((float(item["system_time"]) for item in rounds
                                     if float(item["test_accuracy"]) >= target), None)
        errors = [float(item["prediction_error"]) for item in rounds if item["prediction_error"]]
        row["prediction_mae"] = statistics.mean(errors) if errors else None
        row["prediction_pairs"] = len(errors)
        records.append(row)
    aggregates = []
    for policy in sorted({row["policy"] for row in records}):
        rows = [row for row in records if row["policy"] == policy]
        if len({row["seed"] for row in rows}) != len(rows):
            raise ValueError("duplicate seeds for one policy: summarize a single scenario directory")
        aggregate = {"policy": policy, "seeds": len(rows), "reached_target_seeds": sum(row["target_reached"] for row in rows)}
        for key in ("final_test_accuracy", "final_test_loss", "system_time", "selection_sec",
                    "admission_fraction", "cloud_update_fraction", "communication_mb", "max_epsilon",
                    "mode_changes", "prediction_mae", "initialization_system_time", "initialization_steps",
                    "initialization_max_epsilon", "selection_phase_system_time", "selection_phase_steps",
                    "initialization_test_accuracy", "post_initialization_accuracy_change", "mean_selection_round_duration",
                    "selection_fraction_FL_E", "selection_fraction_FL_EC", "selection_fraction_FL_C"):
            values = [row[key] for row in rows if row.get(key) is not None]
            aggregate[key + "_mean"] = statistics.mean(values) if values else None
            aggregate[key + "_std"] = statistics.stdev(values) if len(values) > 1 else None
        aggregates.append(aggregate)
    # Paired differences against proposed, only for seeds present in both arms.
    proposed = {row["seed"]: row for row in records if row["policy"] == "proposed"}
    paired = []
    for row in records:
        if row["policy"] != "proposed" and row["seed"] in proposed:
            paired.append({"policy": row["policy"], "seed": row["seed"],
                           "accuracy_difference_from_proposed": row["final_test_accuracy"] - proposed[row["seed"]]["final_test_accuracy"],
                           "system_time_difference_from_proposed": row["system_time"] - proposed[row["seed"]]["system_time"]})
    return records, aggregates, paired


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--target", type=float, default=0.8)
    args = parser.parse_args()
    records, aggregates, paired = summarize(args.root, args.target)
    write_csv(args.root / "runs.csv", records)
    write_csv(args.root / "aggregate.csv", aggregates)
    write_csv(args.root / "paired.csv", paired)
    write_json(args.root / "analysis.json", {"target": args.target, "runs": records,
                                             "aggregate": aggregates, "paired": paired})


if __name__ == "__main__":
    main()
