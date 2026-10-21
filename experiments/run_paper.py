"""Single supported entry point for paper/main.tex experiments."""
import argparse
from dataclasses import replace
from datetime import datetime
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from dynfed.paper_config import PaperConfig
from dynfed.paper_runner import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/paper_fmnist.json")
    parser.add_argument("--rounds", type=int)
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--policies", nargs="+")
    parser.add_argument("--pilot", action="store_true", help="6 clients, 2 Edges, 600/100 samples")
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    config = PaperConfig.load(args.config)
    overrides = {key: getattr(args, key) for key in ("rounds", "device") if getattr(args, key) is not None}
    if args.seeds is not None:
        overrides["seeds"] = tuple(args.seeds)
    if args.policies is not None:
        overrides["policies"] = tuple(args.policies)
    if args.pilot:
        overrides.update(clients=6, edges=2, train_limit=600, test_limit=100,
                         batch_size=8, local_steps=1, evaluation_budget=64)
    config = replace(config, **overrides)
    if args.dry_run:
        import json
        print(json.dumps(config.to_dict(), indent=2))
        return
    output = args.output or ROOT / config.output_root / datetime.now().strftime("%Y%m%d_%H%M%S")
    run(config, output)
    print(output)


if __name__ == "__main__":
    main()
