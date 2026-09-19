from __future__ import annotations

"""Validate the frozen Q98 five-figure/two-table paper-output contract.

This is a post-processing/readiness check only.  It never fabricates missing
experimental results: every formal output must already exist on disk.
"""

import argparse
import json
from pathlib import Path
from typing import Any

try:
    from paper_final_plan import FINAL_FIGURES, FINAL_TABLES
except ImportError:  # pragma: no cover - package import in tests
    from experiments.paper_final_plan import FINAL_FIGURES, FINAL_TABLES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ROOT = ROOT / "out" / "paper_v31_final"

# Machine-readable evidence required before the corresponding paper item can
# be marked ready.  Figure image composition can happen later in LaTeX; this
# gate deliberately checks the underlying formal evidence rather than merely
# the presence of a PNG.
OUTPUT_EVIDENCE = {
    "fig1_main_four_methods": (
        "fig1_main/iid/aggregate/summary_statistics.csv",
        "fig1_main/dirichlet_0p5/aggregate/summary_statistics.csv",
        "fig1_main/dirichlet_0p1/aggregate/summary_statistics.csv",
    ),
    "fig2_dynamic_resources": (
        "fig2_dynamic_resources/communication/aggregate/stage_mode_selection.csv",
        "fig2_dynamic_resources/compute/aggregate/stage_mode_selection.csv",
    ),
    "fig3_dynamic_vs_fixed_privacy": (
        "fig3_privacy_aggregate/privacy_trajectory.csv",
    ),
    "fig4_fast_response": (
        "fig4_fast_response/aggregate/summary_statistics.csv",
    ),
    "fig5_optimizer_and_sp": (
        "fig5_optimizer/optimizer_summary.csv",
        "sensitivity_aggregate/strategy_period_statistics.csv",
    ),
    "table1_parameters": ("table1_parameters.csv",),
    "table2_methods_results": ("table2_methods_results.csv",),
}


def build_readiness(root: Path) -> dict[str, Any]:
    expected = set(FINAL_FIGURES) | set(FINAL_TABLES)
    if set(OUTPUT_EVIDENCE) != expected:
        raise RuntimeError("Q98 readiness map does not match the frozen five-figure/two-table plan")

    items: list[dict[str, Any]] = []
    for name in (*FINAL_FIGURES, *FINAL_TABLES):
        relative_paths = OUTPUT_EVIDENCE[name]
        missing = [path for path in relative_paths if not (root / path).is_file()]
        items.append({
            "name": name,
            "kind": "figure" if name in FINAL_FIGURES else "table",
            "ready": not missing,
            "evidence": list(relative_paths),
            "missing": missing,
        })
    return {
        "contract": "Q98: 5 core figures + 2 tables",
        "ready": all(item["ready"] for item in items),
        "figures_ready": sum(item["ready"] for item in items if item["kind"] == "figure"),
        "tables_ready": sum(item["ready"] for item in items if item["kind"] == "table"),
        "items": items,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate frozen Q98 final paper-output readiness.")
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-complete", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.root.resolve()
    report = build_readiness(root)
    output = (args.output or (root / "paper_output_readiness.json")).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(output)
    print(f"figures={report['figures_ready']}/5 tables={report['tables_ready']}/2 ready={report['ready']}")
    if args.require_complete and not report["ready"]:
        missing = [item["name"] for item in report["items"] if not item["ready"]]
        raise SystemExit("Missing formal paper outputs: " + ", ".join(missing))


if __name__ == "__main__":
    main()
