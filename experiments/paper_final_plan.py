"""Frozen Q94--Q98 paper-output contract.

This module is intentionally declarative: it prevents legacy diagnostics and
older sweeps from silently becoming formal paper experiments again.
"""
from __future__ import annotations

FINAL_POLICIES = (
    "fixed_mode_fixed_privacy",
    "dynamic_mode_fixed_privacy",
    "fixed_mode_dynamic_privacy",
    "full_dynfl",
)
FORMAL_SEEDS = (40, 42, 44)
STRATEGY_PERIOD_VALUES = (1, 5, 10)
RESOURCE_SCENARIOS = ("communication", "compute")
AUXILIARY_DATASET = "fmnist"
AUXILIARY_MODEL = "lenet5"  # lightweight CNN auxiliary validation (Q84/Q85)

DATA_DISTRIBUTIONS = (
    ("iid", "iid", None),
    ("dirichlet_0p5", "dirichlet", 0.5),
    ("dirichlet_0p1", "dirichlet", 0.1),
)

FINAL_FIGURES = {
    "fig1_main_four_methods": "Four internal methods: accuracy/convergence, system latency, and privacy budget.",
    "fig2_dynamic_resources": "Communication/compute constrained Normal-Constrained-Normal scenarios with stage-wise Mode Selection Ratio.",
    "fig3_dynamic_vs_fixed_privacy": "Dynamic Privacy vs Fixed Privacy: accuracy/convergence and system latency against privacy-budget consumption.",
    "fig4_fast_response": "Fast-response scenario: Deadline Satisfaction Ratio, Feasible Participation Ratio, system latency, and model performance.",
    "fig5_optimizer_and_sp": "Exact Solver vs Bounded Pareto objective gap/solve time plus S_P sensitivity.",
}
FINAL_TABLES = {
    "table1_parameters": "System, training, and privacy parameters.",
    "table2_methods_results": "Definitions of the four internal methods and compact core-result summary.",
}


def validate_final_plan() -> None:
    assert len(FINAL_FIGURES) == 5
    assert len(FINAL_TABLES) == 2
    assert len(FORMAL_SEEDS) == 3
    assert STRATEGY_PERIOD_VALUES == (1, 5, 10)
    assert RESOURCE_SCENARIOS == ("communication", "compute")
    assert AUXILIARY_DATASET == "fmnist"
    assert AUXILIARY_MODEL == "lenet5"
    assert DATA_DISTRIBUTIONS == (
        ("iid", "iid", None),
        ("dirichlet_0p5", "dirichlet", 0.5),
        ("dirichlet_0p1", "dirichlet", 0.1),
    )
