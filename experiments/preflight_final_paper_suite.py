from __future__ import annotations

"""Preflight the frozen formal suite before launching long-running experiments."""

import argparse
import importlib
import json
import os
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.he_backend import check_he_backend
from experiments.paper_final_plan import FORMAL_SEEDS
from experiments.run_final_paper_suite import DEFAULT_AUX_CONFIG, DEFAULT_FAST_CONFIG, FINAL_STUDIES, _load, build_final_cases

REQUIRED_MODULES = ("numpy", "sklearn", "matplotlib", "torch", "torchvision")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preflight the frozen Q98 final paper suite.")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "paper_v30_cifar10_resnet18.json")
    parser.add_argument("--fast-config", type=Path, default=DEFAULT_FAST_CONFIG)
    parser.add_argument("--aux-config", type=Path, default=DEFAULT_AUX_CONFIG)
    parser.add_argument("--studies", nargs="+", choices=FINAL_STUDIES, default=list(FINAL_STUDIES))
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--report", type=Path, default=ROOT / "out" / "paper_v31_final" / "preflight_report.json")
    parser.add_argument("--require-ready", action="store_true")
    return parser.parse_args()


def _module_check(name: str) -> dict[str, Any]:
    try:
        module = importlib.import_module(name)
    except Exception as exc:
        return {"ready": False, "detail": f"{type(exc).__name__}: {exc}"}
    return {"ready": True, "version": str(getattr(module, "__version__", "unknown"))}


def _writable_check(path: Path) -> dict[str, Any]:
    target = path.resolve()
    probe_dir = target if target.exists() and target.is_dir() else target.parent
    probe_dir.mkdir(parents=True, exist_ok=True)
    return {"ready": bool(os.access(probe_dir, os.W_OK)), "path": str(probe_dir)}


def build_report(base: dict[str, Any], fast: dict[str, Any] | None, aux: dict[str, Any] | None, *, studies: list[str], seeds: list[int], output_root: Path) -> dict[str, Any]:
    cases = build_final_cases(base, studies=studies, fast_config=fast, aux_config=aux)
    rounds = int(base["training"]["rounds"])
    modules = {name: _module_check(name) for name in REQUIRED_MODULES}
    cuda_ready = False
    cuda_detail = "torch unavailable"
    if modules["torch"]["ready"]:
        import torch
        cuda_ready = bool(torch.cuda.is_available())
        cuda_detail = torch.cuda.get_device_name(0) if cuda_ready else "torch.cuda.is_available() is False"
    he_backend = str(base["he"]["backend"])
    he_status = check_he_backend(he_backend)
    checks = {
        "python_modules": {"ready": all(item["ready"] for item in modules.values()), "modules": modules},
        "cuda": {"ready": cuda_ready if base["training"].get("device") == "cuda" else True, "required_device": base["training"].get("device"), "detail": cuda_detail},
        "real_he": {"ready": bool(he_status.available) if base["he"].get("require_real_he") else True, "backend": he_backend, "required": bool(base["he"].get("require_real_he")), "detail": he_status.detail},
        "output_writable": _writable_check(output_root),
    }
    return {
        "ready": all(item["ready"] for item in checks.values()),
        "checks": checks,
        "workload": {
            "studies": studies, "seeds": seeds, "formal_cases": len(cases),
            "training_invocations": len(cases) * len(seeds),
            "policy_runs": sum(len(case["policies"]) for case in cases) * len(seeds),
            "configured_rounds_per_policy": rounds,
            "policy_rounds": sum(len(case["policies"]) * rounds for case in cases) * len(seeds),
            "optimizer_validation_invocations": 1 if "optimizer" in studies else 0,
        },
        "cases": [{"study": case["study"], "setting": case["setting"], "policies": list(case["policies"]), "output_root": case["config"]["output_root"]} for case in cases],
        "notes": [
            "Dataset loaders may download CIFAR-10/Fashion-MNIST on first use.",
            "The pretrained ResNet-18 path may download torchvision weights on first use.",
            "Preflight validates availability; it does not train models or create experimental evidence.",
        ],
    }


def main() -> None:
    args = parse_args()
    studies = list(dict.fromkeys(args.studies))
    seeds = list(dict.fromkeys(args.seeds or FORMAL_SEEDS))
    base = _load(args.config)
    fast = _load(args.fast_config) if "fast_response" in studies else None
    aux = _load(args.aux_config) if "auxiliary" in studies else None
    report = build_report(base, fast, aux, studies=studies, seeds=seeds, output_root=ROOT / "out" / "paper_v31_final")
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    workload = report["workload"]
    print(f"ready={report['ready']} cases={workload['formal_cases']} training_invocations={workload['training_invocations']} policy_runs={workload['policy_runs']} policy_rounds={workload['policy_rounds']}")
    for name, check in report["checks"].items():
        print(f"{name}: {'PASS' if check['ready'] else 'FAIL'}")
    print(args.report)
    if args.require_ready and not report["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
