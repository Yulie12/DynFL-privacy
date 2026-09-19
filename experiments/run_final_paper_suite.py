from __future__ import annotations

"""Run the frozen Q98 formal experiment matrix without changing DynFL semantics.

This is orchestration only.  It reuses the validated paper runner and keeps
legacy diagnostics outside the formal suite.
"""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.paper_final_plan import DATA_DISTRIBUTIONS, FORMAL_SEEDS, RESOURCE_SCENARIOS, STRATEGY_PERIOD_VALUES
from experiments.run_paper_config import DEFAULT_CONFIG, build_command, validate_config

DEFAULT_FAST_CONFIG = ROOT / "configs" / "paper_v31_cifar10_resnet18_fast_response.json"

FINAL_STUDIES = ("main", "dynamic_resources", "privacy", "fast_response", "optimizer", "sp")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Q98 final DynFL paper suite.")
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--fast-config",
        type=Path,
        default=DEFAULT_FAST_CONFIG,
        help=("Formal fast-response config. Defaults to the frozen Fig.4 contract: "
              "clients 0-19 (20%% of 100 clients), each with a 20 s hard deadline."),
    )
    parser.add_argument("--studies", nargs="+", choices=FINAL_STUDIES, default=list(FINAL_STUDIES))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def _load(path: Path) -> dict[str, Any]:
    config = json.loads(path.resolve().read_text(encoding="utf-8"))
    validate_config(config)
    return config


def _case_config(base: dict[str, Any], output_root: str, **system_updates: Any) -> dict[str, Any]:
    config = copy.deepcopy(base)
    config["output_root"] = output_root
    config["system"].update(system_updates)
    return config


def build_final_cases(
    base: dict[str, Any],
    *,
    studies: list[str],
    fast_config: dict[str, Any] | None,
) -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []
    if "main" in studies:
        # Q82/Q83: the formal four-method comparison is evaluated under IID
        # and two frozen non-IID Dirichlet settings.  The legacy
        # edge_label_skew partition remains available for diagnostics only.
        for setting, partition_mode, alpha in DATA_DISTRIBUTIONS:
            config = copy.deepcopy(base)
            config["training"]["partition_mode"] = partition_mode
            if alpha is None:
                config["training"].pop("dirichlet_alpha", None)
            else:
                config["training"]["dirichlet_alpha"] = alpha
            config["output_root"] = f"out/paper_v31_final/fig1_main/{setting}"
            cases.append({
                "study": "main", "setting": setting, "config": config,
                "policies": list(base["policies"]),
            })
    if "dynamic_resources" in studies:
        for scenario in RESOURCE_SCENARIOS:
            cases.append({
                "study": "dynamic_resources", "setting": scenario,
                "config": _case_config(base, f"out/paper_v31_final/fig2_dynamic_resources/{scenario}", resource_scenario=scenario),
                "policies": ["full_dynfl"],
            })
    if "privacy" in studies:
        cases.append({
            "study": "privacy", "setting": "dynamic_vs_fixed",
            "config": _case_config(base, "out/paper_v31_final/fig3_privacy"),
            "policies": ["dynamic_mode_fixed_privacy", "full_dynfl"],
        })
    if "fast_response" in studies:
        if fast_config is None:
            raise ValueError("fast_response requires --fast-config with explicit system.fast_client_deadlines")
        deadlines = fast_config.get("system", {}).get("fast_client_deadlines", {})
        if not isinstance(deadlines, dict) or not deadlines:
            raise ValueError("--fast-config must define a non-empty system.fast_client_deadlines object")
        fast = copy.deepcopy(fast_config)
        fast["output_root"] = "out/paper_v31_final/fig4_fast_response"
        cases.append({
            "study": "fast_response", "setting": "configured_fast_clients", "config": fast,
            "policies": list(fast["policies"]),
        })
    if "sp" in studies:
        for period in STRATEGY_PERIOD_VALUES:
            config = copy.deepcopy(base)
            config["training"]["selection_period"] = period
            config["output_root"] = f"out/paper_v31_final/fig5_sp/sp_{period}"
            cases.append({
                "study": "sp", "setting": str(period), "config": config, "policies": ["full_dynfl"],
            })
    return cases


def _optimizer_command(seeds: list[int]) -> list[str]:
    return [
        sys.executable, str(ROOT / "experiments" / "validate_pareto_search.py"),
        "--seeds", ",".join(str(seed) for seed in seeds),
        "--output-dir", "out/paper_v31_final/fig5_optimizer",
    ]


def main() -> None:
    args = parse_args()
    studies = list(dict.fromkeys(args.studies))
    seeds = list(dict.fromkeys(args.seeds or FORMAL_SEEDS))
    base = _load(args.config)
    fast = _load(args.fast_config) if "fast_response" in studies else None
    cases = build_final_cases(base, studies=studies, fast_config=fast)
    rounds = int(args.rounds or base["training"]["rounds"])
    manifest: list[dict[str, Any]] = []

    for case in cases:
        for seed in seeds:
            command = build_command(
                case["config"], seed=seed, policies=list(case["policies"]), rounds=rounds,
            )
            print(f"[{case['study']} setting={case['setting']} seed={seed}] " + subprocess.list2cmdline(command), flush=True)
            manifest.append({
                "study": case["study"], "setting": case["setting"], "seed": seed,
                "rounds": rounds, "policies": list(case["policies"]),
                "output_root": case["config"]["output_root"],
            })
            if not args.dry_run:
                subprocess.run(command, cwd=ROOT, check=True)

    if "optimizer" in studies:
        command = _optimizer_command(seeds)
        print("[optimizer exact_vs_bounded] " + subprocess.list2cmdline(command), flush=True)
        manifest.append({
            "study": "optimizer", "setting": "exact_vs_bounded", "seeds": seeds,
            "output_root": "out/paper_v31_final/fig5_optimizer",
        })
        if not args.dry_run:
            subprocess.run(command, cwd=ROOT, check=True)

    if not args.dry_run:
        path = ROOT / "out" / "paper_v31_final" / "final_suite_manifest.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
