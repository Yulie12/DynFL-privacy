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


def independent_equations(before, units, gamma_d, gamma_c, epsilon_j):
    """Direct equation-by-equation recomputation, independent of History.observe."""
    reference = before["reference"]
    deviations = {mode: value.clone() for mode, value in before["deviations"].items()}
    if not units:
        return reference, deviations, None, None
    total = sum(unit.samples for unit in units)
    cloud = sum((unit.update.double()*unit.samples for unit in units),
                torch.zeros_like(units[0].update).double())/total
    prediction, j_obs = None, None
    if reference is not None:
        modes = {unit.mode for unit in units}
        if modes.issubset(deviations):
            prediction = reference+sum((deviations[unit.mode]*(unit.samples/total) for unit in units),
                                       torch.zeros_like(reference))
        j_obs = float((cloud-reference).square().sum()/(reference.square().sum()+epsilon_j))
        for mode in modes:
            group = [unit for unit in units if unit.mode == mode]
            mean = sum((unit.update.double()*unit.samples for unit in group),
                       torch.zeros_like(cloud))/sum(unit.samples for unit in group)
            observed_deviation = mean-reference
            deviations[mode] = (observed_deviation if mode not in deviations else
                                gamma_d*deviations[mode]+(1-gamma_d)*observed_deviation)
    after = cloud if reference is None else gamma_c*reference+(1-gamma_c)*cloud
    return after, deviations, prediction, j_obs


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
            prior_edge_generations = []
            initial_vector = None
            for round_index, snapshot_path in enumerate(sorted(folder.glob("history_round_*.pt"))):
                snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
                if initial_vector is None:
                    initial_vector = snapshot.get("model_before")
                round_index = snapshot.get("round_index", round_index)
                if "history_before" in snapshot:
                    state = snapshot["history_before"]
                    history = History(config["gamma_d"], config["gamma_c"], config["epsilon_j"],
                                      reference=state["reference"], deviations=state["deviations"],
                                      mode_observations=state.get("mode_observations", {}))
                    manual_ref, manual_dev, manual_pred, _ = independent_equations(
                        state, snapshot["accepted"], config["gamma_d"], config["gamma_c"], config["epsilon_j"])
                    torch.testing.assert_close(manual_ref, snapshot["history_after"]["reference"])
                    for mode in manual_dev:
                        torch.testing.assert_close(manual_dev[mode], snapshot["history_after"]["deviations"][mode])
                    torch.testing.assert_close(manual_pred, snapshot["predicted_before_execution"])
                units = snapshot["accepted"]
                pending = snapshot["pending"]
                for generation in snapshot.get("generation_records", []):
                    unit = generation["unit"]
                    torch.testing.assert_close(unit.update, generation["end"].double()-generation["base"].double())
                    assert unit.base_source.startswith("edge:" if unit.mode == "FL_E" else "cloud:")
                    if unit.mode == "FL_E" and config.get("snapshot_interval", 1) == 1:
                        candidates = [item for item in prior_edge_generations
                                      if item["unit"].edge == unit.edge and item["edge_ready"] <= snapshot["dispatch_time"]+1e-12]
                        parent = max(candidates, key=lambda item: (item["edge_ready"],item["unit"].identity)) if candidates else None
                        expected = initial_vector if parent is None else parent["end"]
                        torch.testing.assert_close(generation["base"], expected)
                    elif unit.mode != "FL_E":
                        torch.testing.assert_close(generation["base"], snapshot["model_before"])
                prior_edge_generations.extend(item for item in snapshot.get("generation_records", [])
                                              if item["unit"].mode == "FL_E")
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
                    if "model_before" in snapshot:
                        torch.testing.assert_close(snapshot["model_after"].double()-snapshot["model_before"].double(),
                                                   snapshot["delta"], rtol=1e-4, atol=2e-7)
                torch.testing.assert_close(history.reference, snapshot["history_after"]["reference"])
                assert set(history.deviations) == set(snapshot["history_after"]["deviations"])
                for mode in history.deviations:
                    torch.testing.assert_close(history.deviations[mode], snapshot["history_after"]["deviations"][mode])
            receipts = []
            if (folder / "privacy_receipts.csv").exists():
                with (folder / "privacy_receipts.csv").open(encoding="utf-8-sig") as handle:
                    receipts = list(csv.DictReader(handle))
            actual_steps = sum(int(row["executed_events"]) for row in receipts)
            # All accepted unit identities/weights are checked even when large
            # runs save detailed tensor snapshots only at selected rounds.
            all_units = []
            if (folder / "cloud_units.csv").exists():
                with (folder / "cloud_units.csv").open(encoding="utf-8-sig") as handle:
                    all_units = list(csv.DictReader(handle))
            identities = [row["identity"] for row in all_units]
            assert len(set(identities)) == len(identities)
            for index in {row["accepted_round"] for row in all_units}:
                group = [row for row in all_units if row["accepted_round"] == index]
                total = sum(int(row["samples"]) for row in group)
                assert all(abs(float(row["weight"])-int(row["samples"])/total) < 1e-12 for row in group)
                assert all(float(row["arrival"]) <= float(row["cutoff"])+1e-12 for row in group)
                if group and "base_source" in group[0]:
                    assert all(row["base_source"].startswith("edge:" if row["mode"] == "FL_E" else "cloud:") for row in group)
            assert actual_steps == summary["executed_local_steps"]
            assert all(int(row["executed_events"]) == config["local_steps"] for row in receipts)
            assert all(float(row["epsilon_after"]) <= config["epsilon"] + 1e-8 for row in receipts)
            assert summary["max_epsilon"] <= config["epsilon"] + 1e-8
            report.append({"policy": summary["policy"], "seed": summary["seed"],
                           "accepted_units": len(all_units),
                           "cross_round_async_units": sum(int(row["generated_round"]) < int(row["accepted_round"]) for row in all_units),
                           "tensor_snapshots": len(list(folder.glob("history_round_*.pt"))),
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
