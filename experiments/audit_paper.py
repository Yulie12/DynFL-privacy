"""Recompute paper invariants from saved real-training outputs."""
import argparse
import csv
import json
from pathlib import Path
import sys
import torch
from dataclasses import replace
import math

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynfed.paper_history import History
from dynfed.paper_timeline import mode_masses, plan
from dynfed.paper_config import PaperConfig, CORE_POLICIES, MODES, effective_config
from dynfed.paper_runner import write_json
from dynfed.paper_storage import digest, file_digest, load_checkpoint, tensor_digest
from dynfed.paper_privacy import ORDERS


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


def reconstruct_dispatches(folder, seed_dir, config, partition, payload_mb):
    """Rebuild starts from metadata; never trust a stored end-of-round timestamp."""
    with (folder/"client_decisions.csv").open(encoding="utf-8-sig") as handle:
        decisions = list(csv.DictReader(handle))
    traces = json.loads((seed_dir/"traces.json").read_text(encoding="utf-8"))
    cfg = PaperConfig(**config)
    samples = {client:len(indices) for client,indices in enumerate(partition)}
    now, previous, pending = 0., {}, []
    reconstructed = {}
    for index in range(config["rounds"]):
        profile = {int(row["client"]):row["mode"] for row in decisions if int(row["round"]) == index}
        schedule = plan(cfg, profile, traces[index], samples, payload_mb, now, index, previous, pending)
        reconstructed[index] = (now, schedule)
        previous.update({client:mode for client,mode in profile.items() if mode != "SKIP"})
        now, pending = schedule.finish, schedule.pending
    return reconstructed


def audit(root, required_runs=None):
    root = Path(root)
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    if config.get("protocol") != "global_barrier_100":
        raise ValueError("old-protocol outputs cannot validate the global 100% barrier")
    if config.get("experiment_version") != "shared_init_v1":
        raise ValueError("results predate the registered shared-initialization experiment")
    scenario_config = PaperConfig(**config)
    expected = {(seed, policy) for seed in config["seeds"] for policy in config["policies"]}
    required = expected if required_runs is None else set(required_runs)
    if not required.issubset(expected):
        raise ValueError("requested audit unit is not registered in scenario")
    present = {(int(p.parent.parent.name.split('_')[1]),p.parent.name) for p in root.glob('seed_*/*/summary.json')}
    if not required.issubset(present):
        raise ValueError("incomplete experiment: required strategy-seed summaries are missing")
    report = []
    for seed_dir in sorted(root.glob("seed_*")):
        partition = json.loads((seed_dir / "partition.json").read_text(encoding="utf-8"))
        reference_model = None
        for path in sorted(seed_dir.glob("*/summary.json")):
            run_key = (int(seed_dir.name.split('_')[1]), path.parent.name)
            if run_key not in required:
                continue
            folder = path.parent
            config = effective_config(scenario_config, run_key[1], run_key[0]).to_dict()
            saved_config = json.loads((folder/"config.json").read_text(encoding="utf-8"))
            assert digest(saved_config) == digest(config), "effective policy configuration mismatch"
            summary = json.loads(path.read_text(encoding="utf-8"))
            assert summary["rounds"] == config["rounds"]
            assert summary["config_hash"] == digest(config)
            status = json.loads((folder/"run_status.json").read_text(encoding="utf-8"))
            assert status["status"] == "completed" and status["completed_rounds"] == config["rounds"]
            identity = json.loads((folder/"run_identity.json").read_text(encoding="utf-8"))
            assert identity["config_hash"] == digest(config) and summary["run_id"] == digest(identity)
            assert identity["partition_hash"] == digest(partition)
            traces = json.loads((seed_dir/"traces.json").read_text(encoding="utf-8"))
            assert identity["traces_hash"] == digest(traces)
            assert identity["source_hash"] == json.loads((root/"source_manifest.json").read_text(encoding="utf-8"))["source_hash"]
            for name, checksum in json.loads((seed_dir/"input_hashes.json").read_text(encoding="utf-8")).items():
                assert file_digest(seed_dir/name) == checksum
            prereg_path = root/"preregistration.json"
            if prereg_path.exists():
                registration = json.loads(prereg_path.read_text(encoding="utf-8"))
                entry = next(item for item in registration["units"] if (item["seed"],item["policy"]) == run_key)
                assert entry["config_hash"] == summary["config_hash"]
                assert entry["source_hash"] == identity["source_hash"]
            model = torch.load(folder / "initial_model.pt", map_location="cpu", weights_only=True)
            if reference_model is None:
                reference_model = model
            assert set(model) == set(reference_model)
            assert all(torch.equal(model[key], reference_model[key]) for key in model), "unpaired initial models"
            history = History(config["gamma_d"], config["gamma_c"], config["epsilon_j"])
            seen = set()
            late = 0
            initial_vector = None
            reconstructed = None
            for round_index, snapshot_path in enumerate(sorted(folder.glob("history_round_*.pt"))):
                snapshot = torch.load(snapshot_path, map_location="cpu", weights_only=False)
                if initial_vector is None:
                    initial_vector = snapshot.get("model_before")
                    if initial_vector is not None:
                        reconstructed = reconstruct_dispatches(folder, seed_dir, config, partition,
                                                               initial_vector.numel()*initial_vector.element_size()/1e6)
                round_index = snapshot.get("round_index", round_index)
                dispatch_time = snapshot.get("dispatch_time")
                if reconstructed is not None:
                    dispatch_time, reconstructed_schedule = reconstructed[round_index]
                    assert {unit.identity for unit in snapshot["accepted"]} == {unit.identity for unit in reconstructed_schedule.accepted}
                    assert {unit.identity for unit in snapshot["pending"]} == {unit.identity for unit in reconstructed_schedule.pending}
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
                    assert unit.base_source == f"cloud:v{snapshot['dispatch_version']}"
                    assert unit.base_round == snapshot["dispatch_version"]
                    torch.testing.assert_close(generation["base"], snapshot["model_before"])
                assert not pending
                assert set(snapshot["client_versions"]) == set(range(config["clients"]))
                assert set(snapshot["client_versions"].values()) == {snapshot["cloud_version"]}
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
                    assert all(row["base_source"].startswith("cloud:") for row in group)
                assert all(row["generated_round"] == row["accepted_round"] for row in group)
            with (folder/"round_metrics.csv").open(encoding="utf-8-sig") as handle:
                metrics = list(csv.DictReader(handle))
            with (folder/"client_decisions.csv").open(encoding="utf-8-sig") as handle:
                decisions_for_audit = list(csv.DictReader(handle))
            previous_finish, previous_version = 0., 0
            for index, metric in enumerate(metrics):
                events = json.loads((folder/f"timeline_round_{index:03d}.json").read_text(encoding="utf-8"))
                acks = [event for event in events if event["event"] == "model_ack"]
                cloud = next(event for event in events if event["event"] == "cloud_acceptance")
                assert set(cloud["expected"]) == set(cloud["accepted"]) and not cloud["pending"]
                if reconstructed is not None:
                    expected_start, expected_schedule = reconstructed[index]
                    assert abs(float(metric["dispatch_time"])-expected_start) < 1e-10
                    assert abs(cloud["cutoff"]-expected_schedule.cutoff) < 1e-10
                    assert set(cloud["expected"]) == {unit.identity for unit in expected_schedule.units}
                    assert all(abs(event["finish"]-expected_schedule.acknowledgement_finish[event["client"]]) < 1e-10
                               for event in acks)
                assert len(cloud["accepted"]) == int(metric["expected_cloud_units"])
                assert len(acks) == config["clients"] and {event["client"] for event in acks} == set(range(config["clients"]))
                assert all(event["received"] >= cloud["finish"] and event["finish"] >= event["received"] for event in acks)
                assert all(event["version"] == int(metric["cloud_version"]) for event in acks)
                assert int(metric["dispatch_version"]) == previous_version
                assert int(metric["cloud_version"]) == previous_version+bool(cloud["accepted"])
                assert float(metric["dispatch_time"]) >= previous_finish-1e-12
                assert abs(float(metric["system_time"])-max(event["finish"] for event in acks)) < 1e-10
                previous_finish, previous_version = float(metric["system_time"]), int(metric["cloud_version"])
            assert actual_steps == summary["executed_local_steps"]
            assert len(metrics) == config["rounds"]
            previous_model = initial_vector
            receipt_map = {(int(row["round"]),int(row["client"]),int(row["stage"])):row for row in receipts}
            assert len(receipt_map) == len(receipts), "duplicate DP receipt"
            accounts = {client:{"events":0,"rdp":[0.]*len(ORDERS)} for client in range(config["clients"])}
            for index, metric in enumerate(metrics):
                candidates = json.loads((folder/f"candidates_round_{index:03d}.json").read_text(encoding="utf-8"))
                state = torch.load(folder/f"decision_history_{index:03d}.pt", map_location="cpu", weights_only=False)
                assert len(candidates["candidates"]) == int(metric["evaluated_profiles"])
                assert len(candidates["candidates"]) <= config["evaluation_budget"] or folder.name == "exhaustive"
                candidate_map = {}
                for candidate in candidates["candidates"]:
                    profile = {int(client):mode for client, mode in candidate["profile"].items()}
                    candidate_map[tuple(sorted(profile.items()))] = candidate
                    schedule = plan(PaperConfig(**config), profile, traces[index],
                                    {client:len(indices) for client,indices in enumerate(partition)},
                                    initial_vector.numel()*initial_vector.element_size()/1e6,
                                    float(metric["dispatch_time"]), index,
                                    {} if index == 0 else {int(row["client"]):row["mode"] for row in decisions_for_audit if int(row["round"]) == index-1})
                    assert abs(candidate["T"]-(schedule.finish-float(metric["dispatch_time"]))) < 1e-9
                    masses = mode_masses(schedule.accepted)
                    assert candidate["masses"] == masses
                    active = {mode:mass for mode,mass in masses.items() if mass > 0}
                    available = state["reference"] is not None and active and set(active).issubset(state["deviations"])
                    if not available:
                        assert candidate["J"] is None
                    else:
                        dev = sum((state["deviations"][mode]*(mass/sum(active.values())) for mode,mass in active.items()),
                                  torch.zeros_like(state["reference"]))
                        j = float(dev.square().sum()/(state["reference"].square().sum()+config["epsilon_j"]))
                        assert abs(j-candidate["J"]) <= 1e-9*max(1.,abs(j))
                key = tuple(sorted((int(c),m) for c,m in candidates["selected_profile"].items()))
                actual_profile = tuple(sorted((int(row["client"]),row["mode"]) for row in decisions_for_audit
                                              if int(row["round"]) == index))
                assert key == actual_profile, "selected candidate differs from actual executed distribution"
                assert key in candidate_map
                selected_j = candidate_map[key]["J"]
                if selected_j is None:
                    assert metric["predicted_j"] == ""
                else:
                    assert math.isclose(selected_j,float(metric["predicted_j"]),rel_tol=1e-9,abs_tol=1e-9)
                if candidates["phase"] == "shared_initialization":
                    assert key == tuple((client,MODES[client%3]) for client in range(config["clients"]))
                if candidates["phase"] == "latency_only":
                    assert key == min(candidate_map, key=lambda k:(candidate_map[k]["T"],k))
                if candidates["phase"] == "pareto":
                    archive = [tuple(sorted((int(c),m) for c,m in profile.items())) for profile in candidates["archive_profiles"]]
                    t0,t1 = min(candidate_map[k]["T"] for k in archive),max(candidate_map[k]["T"] for k in archive)
                    j0,j1 = min(candidate_map[k]["J"] for k in archive),max(candidate_map[k]["J"] for k in archive)
                    def score(k):
                        item = candidate_map[k]
                        return (max((item["T"]-t0)/(t1-t0+1e-9),(item["J"]-j0)/(j1-j0+1e-9)),item["T"],item["J"],k)
                    assert key == min(archive,key=score)
                checkpoint = load_checkpoint(folder/f"checkpoint_round_{index+1:03d}.pt")
                assert checkpoint["run_id"] == summary["run_id"] and checkpoint["next_round"] == index+1
                assert checkpoint["barrier"].acks == set(range(config["clients"]))
                assert checkpoint["rows"][-1] == metric_as_typed(checkpoint["rows"][-1],metric)
                realized = []
                for unit in reconstructed[index][1].units:
                    base = previous_model
                    stages = config["edge_stages"] if unit.mode == "FL_EC" else 1
                    for stage in range(stages):
                        local = []
                        for client in unit.clients:
                            block = torch.load(folder/f"block_{unit.identity.replace(':','_')}_s{stage}_c{client}.pt",
                                               map_location="cpu",weights_only=False)
                            assert block["status"] == "complete" and block["identity"]["run_id"] == summary["run_id"]
                            assert block["identity"]["base"] == tensor_digest(base)
                            account = accounts[client]
                            assert block["identity"]["events_before"] == account["events"]
                            assert block["identity"]["rdp_before"] == account["rdp"]
                            receipt = receipt_map[index,client,stage+1]
                            sigma = float(receipt["sigma"])
                            assert sigma == block["identity"]["sigma"]
                            assert block["identity"]["seed"] == run_key[0]*1000003+index*10007+client*101+stage
                            account["events"] += config["local_steps"]
                            # Match the actual sequence of individual Gaussian
                            # events, not a differently rounded batched sum.
                            for _ in range(config["local_steps"]):
                                account["rdp"] = [value+order/(2*sigma*sigma)
                                                  for value,order in zip(account["rdp"],ORDERS)]
                            ledger = block["ledger"]
                            assert ledger.events == account["events"]
                            assert all(math.isclose(a,b,rel_tol=1e-12,abs_tol=1e-12) for a,b in zip(ledger.rdp,account["rdp"]))
                            epsilon = min(cost+math.log(1/config["delta"])/(order-1) for cost,order in zip(account["rdp"],ORDERS))
                            assert abs(epsilon-float(receipt["epsilon_after"])) < 1e-8
                            assert epsilon <= config["epsilon"]+1e-8
                            local.append((block["updated"],len(partition[client])))
                        mass = sum(n for _,n in local)
                        base = sum((value*(n/mass) for value,n in local),torch.zeros_like(previous_model))
                    realized.append(replace(unit,update=base.double()-previous_model.double()))
                manual_ref,manual_dev,manual_pred,j_obs = independent_equations(state,realized,config["gamma_d"],config["gamma_c"],config["epsilon_j"])
                torch.testing.assert_close(manual_ref,checkpoint["history"].reference)
                for mode in manual_dev:
                    torch.testing.assert_close(manual_dev[mode],checkpoint["history"].deviations[mode])
                current_model = torch.cat([value.flatten() for value in checkpoint["model"].values()])
                if realized:
                    total = sum(unit.samples for unit in realized)
                    cloud_delta = sum((unit.update*(unit.samples/total) for unit in realized),torch.zeros_like(previous_model).double())
                    torch.testing.assert_close(current_model.double()-previous_model.double(),cloud_delta,rtol=1e-4,atol=2e-7)
                else:
                    torch.testing.assert_close(current_model,previous_model)
                if j_obs is None:
                    assert metric["observed_j"] == ""
                else:
                    assert math.isclose(float(metric["observed_j"]),j_obs,rel_tol=1e-9,abs_tol=1e-9)
                previous_model = current_model
            assert all(int(row["executed_events"]) == config["local_steps"] for row in receipts)
            assert all(float(row["epsilon_after"]) <= config["epsilon"] + 1e-8 for row in receipts)
            assert summary["max_epsilon"] <= config["epsilon"] + 1e-8
            report.append({"policy": summary["policy"], "seed": summary["seed"],
                           "config_hash":summary["config_hash"],"source_hash":identity["source_hash"],
                           "accepted_units": len(all_units),
                           "cross_round_async_units": sum(int(row["generated_round"]) < int(row["accepted_round"]) for row in all_units),
                           "tensor_snapshots": len(list(folder.glob("history_round_*.pt"))),
                           "actual_dp_sgd_steps": actual_steps, "status": "passed"})
    if not report:
        raise ValueError("no completed runs to audit")
    return report


def metric_as_typed(expected, saved):
    """Compare CSV scalars without treating empty values as zeros."""
    result = {}
    for key,value in expected.items():
        text = saved[key]
        if value is None:
            result[key] = None if text == "" else text
        elif isinstance(value,bool):
            result[key] = text == "True"
        elif isinstance(value,int):
            result[key] = int(text)
        elif isinstance(value,float):
            result[key] = float(text)
        else:
            result[key] = text
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    results = audit(args.root)
    write_json(args.root / "audit.json", results)
    print(json.dumps(results, indent=2))
