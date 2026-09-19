from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

from aggregate_multiseed_results import POLICY_LABELS
from paper_final_plan import FINAL_POLICIES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUMMARY = ROOT / "out" / "paper_v31_final" / "fig1_main" / "iid" / "aggregate" / "summary_statistics.csv"
DEFAULT_OUTPUT = ROOT / "out" / "paper_v31_final" / "table2_methods_results.csv"

# Frozen Q81/Q90/Q91 method definitions.  Keep these descriptive axes separate
# from measured results so Table II cannot silently redefine an ablation.
METHOD_DEFINITIONS = {
    "fixed_mode_fixed_privacy": {
        "mode_policy": "Fixed",
        "privacy_policy": "Fixed",
        "mode_definition": "LIIEIIIC",
        "privacy_definition": "Initial legal privacy profile is locked; infeasibility uses fallback.",
    },
    "dynamic_mode_fixed_privacy": {
        "mode_policy": "Dynamic",
        "privacy_policy": "Fixed",
        "mode_definition": "DynFL mode selection",
        "privacy_definition": "Initial legal privacy profile is locked; infeasibility uses fallback.",
    },
    "fixed_mode_dynamic_privacy": {
        "mode_policy": "Fixed",
        "privacy_policy": "Dynamic",
        "mode_definition": "LIIEIIIC",
        "privacy_definition": "Privacy configuration/noise may adapt within the same hard constraints.",
    },
    "full_dynfl": {
        "mode_policy": "Dynamic",
        "privacy_policy": "Dynamic",
        "mode_definition": "DynFL mode selection",
        "privacy_definition": "Privacy configuration/noise may adapt within the same hard constraints.",
    },
}

RESULT_FIELDS = (
    "final_test_accuracy_mean",
    "final_test_accuracy_std",
    "avg_last_10_accuracy_mean",
    "avg_last_10_accuracy_std",
    "accounted_system_time_sec_mean",
    "accounted_system_time_sec_std",
    "max_update_epsilon_mean",
    "max_update_epsilon_std",
    "feasible_participation_ratio_mean",
    "feasible_participation_ratio_std",
)


def read_summary(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_policy = {row["policy"]: row for row in rows}
    missing = [policy for policy in FINAL_POLICIES if policy not in by_policy]
    if missing:
        raise RuntimeError("Table II summary is missing formal policies: " + ", ".join(missing))
    return by_policy


def build_table2_rows(summary_by_policy: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for policy in FINAL_POLICIES:
        source = summary_by_policy[policy]
        missing = [field for field in RESULT_FIELDS if field not in source]
        if missing:
            raise RuntimeError(f"Table II summary for {policy} is missing fields: {', '.join(missing)}")
        row: dict[str, Any] = {
            "policy": policy,
            "label": POLICY_LABELS[policy],
            **METHOD_DEFINITIONS[policy],
        }
        row.update({field: source[field] for field in RESULT_FIELDS})
        rows.append(row)
    return rows


def write_table2(rows: list[dict[str, Any]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "policy",
        "label",
        "mode_policy",
        "privacy_policy",
        "mode_definition",
        "privacy_definition",
        *RESULT_FIELDS,
    )
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate frozen Q98 Table II from the four formal methods and aggregated Fig.1 results."
    )
    parser.add_argument("--summary-statistics", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary_path = args.summary_statistics.resolve()
    if not summary_path.exists():
        raise FileNotFoundError(
            f"Missing aggregated Fig.1 summary: {summary_path}. Run the formal multi-seed aggregation first."
        )
    rows = build_table2_rows(read_summary(summary_path))
    write_table2(rows, args.output.resolve())
    print(args.output.resolve())


if __name__ == "__main__":
    main()
