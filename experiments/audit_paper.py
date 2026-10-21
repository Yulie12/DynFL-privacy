"""Recompute paper invariants from saved real-training outputs."""
import argparse
import csv
import json
from pathlib import Path
import sys
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynfed.paper_history import History
from dynfed.paper_timeline import mode_masses
from dynfed.paper_runner import write_json


def audit(root):
    root = Path(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    report = []
    for seed_dir in sorted(root.glob("seed_*")):
        partition = json.loads((seed_dir / "partition.json").read_text(encoding="utf-8"))
        reference_model = None
        for path in sorted(seed_dir.glob("*/summary.json")):
            folder = path.parent
            summary = json.loads(path.read_text(encoding="utf-8"))
            model = torch.load(folder / "initial_model.pt", map_location="cpu", weights_only=True)
            if reference_model is None:
                reference_model = model
            assert set(model) == set(reference_model)
            assert all(torch.equal(model[key], reference_model[key]) for key in model), "unpaired initial models"
            history = History(config["gamma_d"], config["gamma_c"], config["epsilon_j"])
            seen = set()
            late = 0
            for round_index, snapshot_path in enumerate(sorted(folder.glob("history_round_*.pt"))):
                snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
                units = snapshot["accepted"]
                pending = snapshot["pending"]
                assert not ({u.identity for u in units} & {u.identity for u in pending})
                for unit in units:
                    assert unit.identity not in seen, "Cloud unit accepted twice"
                    seen.add(unit.identity)
                    assert unit.samples == sum(len(partition[client]) for client in unit.clients)
                    late += unit.generated_round < round_index
                prediction, _, _ = history.predict(mode_masses(units))
                saved_prediction = snapshot["predicted_before_execution"]
                assert (prediction is None) == (saved_prediction is None)
                if prediction is not None:
                    torch.testing.assert_close(prediction, saved_prediction)
                observation = history.observe(units)
                if units:
                    assert abs(sum(observation["weights"])-1) < 1e-12
                    torch.testing.assert_close(observation["cloud_delta"], snapshot["delta"])
                torch.testing.assert_close(history.reference, snapshot["history_after"]["reference"])
                assert set(history.deviations) == set(snapshot["history_after"]["deviations"])
                for mode in history.deviations:
                    torch.testing.assert_close(history.deviations[mode], snapshot["history_after"]["deviations"][mode])
            with (folder / "privacy_receipts.csv").open(encoding="utf-8-sig") as handle:
                receipts = list(csv.DictReader(handle))
            actual_steps = sum(int(row["executed_events"]) for row in receipts)
            assert actual_steps == summary["executed_local_steps"]
            assert all(int(row["executed_events"]) == config["local_steps"] for row in receipts)
            assert all(float(row["epsilon_after"]) <= config["epsilon"] + 1e-8 for row in receipts)
            assert summary["max_epsilon"] <= config["epsilon"] + 1e-8
            report.append({"policy": summary["policy"], "seed": summary["seed"],
                           "accepted_units": len(seen), "cross_round_async_units": late,
                           "actual_dp_sgd_steps": actual_steps, "status": "passed"})
    if not report:
        raise ValueError("no completed runs to audit")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    results = audit(args.root)
    write_json(args.root / "audit.json", results)
    print(json.dumps(results, indent=2))
