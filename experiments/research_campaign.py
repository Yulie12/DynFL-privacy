"""Persisted study/formal campaign; never relabel a pilot as a formal result."""
import argparse
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import sys
import os
import json
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dynfed.paper_config import PaperConfig
from dynfed.paper_runner import run, write_json, write_csv
from audit_paper import audit
from summarize_paper import summarize


def stages(phase):
    fmnist = PaperConfig.load(ROOT / "configs/paper_fmnist.json")
    cifar = PaperConfig.load(ROOT / "configs/paper_cifar10.json")
    trial = replace(fmnist, clients=6, edges=2, rounds=20, train_limit=1200,
                    test_limit=500, local_steps=1, batch_size=32, evaluation_budget=64,
                    snapshot_interval=1, policies=("proposed",))
    cases = []
    if phase in {"studies", "all"}:
        cases.append(("fmnist_trial", "multiseed_trial", replace(trial, rounds=5,
                      train_limit=600, test_limit=100, batch_size=8, policies=fmnist.policies)))
        for gamma in (0.0, 0.5, 0.9, 0.99):
            cases.append((f"ema_{gamma:g}", "pilot_ablation",
                          replace(trial, gamma_d=gamma, gamma_c=gamma)))
        for gamma_d, gamma_c in ((0.,.9), (.9,0.)):
            cases.append((f"ema_{gamma_d:g}_{gamma_c:g}", "pilot_ablation",
                          replace(trial, gamma_d=gamma_d, gamma_c=gamma_c)))
        cases.append(("cifar_trial", "multiseed_trial",
                      replace(cifar, clients=6, edges=2, rounds=5, train_limit=600, test_limit=100,
                              local_steps=1, batch_size=8, evaluation_budget=64, snapshot_interval=1)))
    if phase in {"formal", "all"}:
        cases.extend((("formal_cifar10", "formal_main", cifar),
                      ("formal_fmnist", "formal_main", fmnist)))
    return cases


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("studies", "formal", "all"), default="all")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help="Retain completed training and repeat audits before advancing")
    args = parser.parse_args()
    cases = stages(args.phase)
    if args.dry_run:
        for name, kind, config in cases:
            print(name, kind, config.rounds, config.clients, config.seeds)
        return
    args.output.mkdir(parents=True, exist_ok=args.resume)
    manifest = {name: {"kind": kind, "config": config.to_dict()} for name, kind, config in cases}
    manifest_path = args.output / "manifest.json"
    if args.resume:
        if json.loads(manifest_path.read_text(encoding="utf-8")) != json.loads(json.dumps(manifest)):
            raise ValueError("cannot resume a different experiment matrix")
    else:
        write_json(manifest_path, manifest)
    completed = []
    for name, kind, config in cases:
        status = {"status": "running", "phase": name, "kind": kind,
                  "pid": os.getpid(),
                  "completed": completed, "updated_utc": datetime.now(timezone.utc).isoformat()}
        write_json(args.output / "campaign_status.json", status)
        destination = args.output / name
        try:
            if args.resume and destination.exists():
                summaries = list(destination.glob("seed_*/*/summary.json"))
                if len(summaries) != len(config.seeds)*len(config.policies):
                    raise RuntimeError(f"partial training exists at {destination}; do not overwrite or reuse its privacy state")
            else:
                run(config, destination)
            write_json(destination / "audit.json", audit(destination))
            runs, aggregate, paired = summarize(destination, .8)
            write_csv(destination / "aggregate.csv", aggregate)
            write_csv(destination / "paired.csv", paired)
            write_json(destination / "analysis.json", {"kind": kind, "runs": runs, "aggregate": aggregate,
                                                        "paired": paired})
            completed.append(name)
            if name == "cifar_trial":
                from export_paper_results import export_studies
                export_studies(args.output)
        except Exception:
            status.update(status="failed", error=traceback.format_exc())
            write_json(args.output / "campaign_status.json", status)
            raise
    if args.phase in {"formal", "all"}:
        from export_paper_results import export_formal
        export_formal(args.output)
    write_json(args.output / "campaign_status.json", {"status": "completed", "completed": completed,
                                                       "updated_utc": datetime.now(timezone.utc).isoformat()})


if __name__ == "__main__":
    main()
