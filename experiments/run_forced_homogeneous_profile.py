from __future__ import annotations

"""Diagnostic-only runner that bypasses global Pareto coordination.

The ordinary runtime first enumerates/chooses one candidate per client and then
runs a global Pareto coordinator.  Homogeneous Edge-only profiles (LIE/LIIE)
are intentionally rejected by the production coordinator because production
Pareto profiles must have positive Cloud fusion mass.  Stage-0 hierarchical
state calibration needs to execute those profiles exactly as specified, without
changing the production selector.  This wrapper replaces only the coordinator
inside this process with a pass-through evaluator; all candidate enumeration,
training, aggregation, persistence, accounting, and checkpointing remain the
normal runtime implementation.
"""

import runpy
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import dynfed.fmnist_lenet5_dynamic as runtime
from dynfed.selection import evaluate_global_profile


def _diagnostic_passthrough_profile(
    *,
    config: Any,
    selected: list[Any],
    client_samples: dict[int, float],
    client_edges: dict[int, int] | None = None,
    previous_choices: dict[int, Any] | None = None,
    **_: Any,
):
    evaluation = evaluate_global_profile(
        config=config,
        selected=selected,
        client_samples=client_samples,
        client_edges=client_edges,
        previous_choices=previous_choices,
    )
    return selected, evaluation


# Patch only this diagnostic subprocess.  The source implementation and every
# normal experiment continue to use the production Pareto coordinator.
runtime.choose_global_pareto_profile = _diagnostic_passthrough_profile

# Stage-0 is a numerical fidelity calibration, not a privacy experiment.
# The production runtime intentionally applies paper_client_privacy_requirement()
# during candidate enumeration; under policy=no_protection that requirement rejects
# plaintext Edge-only candidates and leaves an all-SKIP profile. Disable only the
# paper exposure gate inside this diagnostic subprocess so the requested clean mode
# is actually executed. Normal experiments remain unchanged.
runtime.paper_client_privacy_requirement = lambda: None

runpy.run_path(str(ROOT / "experiments" / "run_fmnist_lenet5.py"), run_name="__main__")
