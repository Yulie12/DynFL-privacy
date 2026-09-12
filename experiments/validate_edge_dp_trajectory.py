"""Fixed trusted edges release client-level DP packets before cloud aggregation.

Private research diagnostic, not a production security certificate.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch

from experiments.distributed_edge_dp_common import distributed_noise_plan, seal_edge_sum
from experiments.edge_dp_common import edge_noise_seed, edge_release_plan, release_edge
from experiments.validate_trusted_aggregate_trajectory import apply_vector_update
from dynfed.fmnist_lenet5_dynamic import load_image_dataset_arrays
from dynfed.privacy import PrivacyAccountant, calibrate_gaussian_noise
from dynfed.split_learning import build_split_pair, split_evaluate, split_local_train_lenet5
from dynfed.utils import timestamped_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--privacy-horizon", type=int, default=100)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--edges", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=6000)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--clip-norm", type=float, default=0.1)
    parser.add_argument("--epsilon", type=float, default=8)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--model", choices=["resnet18_pretrained_head", "resnet18_pretrained_adapter"],
                        default="resnet18_pretrained_head",
                        help="Low dimensional independent-client diagnostic scopes only")
    parser.add_argument("--dp-release", choices=["edge_local", "distributed_he"], default="edge_local",
                        help="Distributed HE is conditional on all edges online and unknown noise shares")
    parser.add_argument("--he-custody", choices=["arithmetic_harness", "trusted_edge"],
                        default="arithmetic_harness", help="Trusted edge uses a ciphertext-only cloud process")
    parser.add_argument("--methods", nargs="+", choices=["no_protection", "clip_only", "edge_dp", "distributed_dp_he"],
                        help="Optional subset; omitted runs both controls and the chosen DP release")
    parser.add_argument("--output-root", default="out/trusted_edge_dp_validation")
    args = parser.parse_args()
    if args.he_custody == "trusted_edge" and args.dp_release != "distributed_he":
        parser.error("Trusted HE custody requires distributed_he release")
    protected_method = "distributed_dp_he" if args.dp_release == "distributed_he" else "edge_dp"
    methods = args.methods or ["no_protection", "clip_only", protected_method]
    if len(set(methods)) != len(methods) or any(m not in {"no_protection", "clip_only", protected_method} for m in methods):
        parser.error("Methods must be unique and consistent with dp-release")
    if not (1 <= args.rounds <= args.privacy_horizon and 1 <= args.edges <= args.clients
            <= args.train_limit and args.test_limit > 0 and args.local_epochs > 0):
        parser.error("Positive counts, nonempty cohorts and rounds within horizon required")
    if not math.isfinite(args.lr) or args.lr <= 0:
        parser.error("Positive finite learning rate required")
    edge_noise_seed(args.seed, 0, 0)
    edge_release_plan([1] * args.clients, args.edges, args.clip_norm)
    sigma = calibrate_gaussian_noise(args.epsilon, args.delta, args.privacy_horizon)
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = args.model
    x, y, tx, ty, shape, _, classes = load_image_dataset_arrays(
        "cifar10", ROOT / "experiments/data/cifar10", args.train_limit, args.test_limit, args.seed)
    indices = np.array_split(np.random.default_rng(args.seed).permutation(len(y)), args.clients)
    plan = edge_release_plan([int(len(idx)) for idx in indices], args.edges, args.clip_norm)
    distributed = distributed_noise_plan([int(len(idx)) for idx in indices], args.edges,
                                         args.clip_norm, sigma, args.edges)
    end, edge_model = build_split_pair(model, device, input_channels=shape[0], image_size=shape[1],
                                     num_classes=classes)
    initial = {"end": {k: v.detach().clone() for k, v in end.state_dict().items()},
               "edge": {k: v.detach().clone() for k, v in edge_model.state_dict().items()}}
    initial_loss, initial_acc = split_evaluate(end, edge_model, tx, ty, device, shape)
    output = timestamped_dir(args.output_root, "edge_dp_trajectory").resolve()
    output.mkdir(parents=True, exist_ok=False)
    path = output / "report.json"
    report = dict(config=vars(args), model=model, status="running", results=[], cohorts=plan,
                  initial_accuracy=initial_acc, initial_loss=initial_loss, noise_multiplier=sigma,
                  adjacency="whole_client_replacement_fixed_public_counts_and_roster",
                  trust="associated_edge_trusted_cloud_honest_but_curious",
                  noise_location="each_edge_before_cloud", he_execution="not_used",
                  scope="private_fixed_independent_client_low_dimensional_diagnostic",
                  full_protocol_dp="not_established", transport_security_implemented=False,
                  diagnostic_outputs="internal_only_not_protected_public_outputs",
                  randomness="seeded_research_noise_not_production_private_randomness",
                  accountant_scope="each_client_own_edge_release_per_round_per_trajectory",
                  excluded="shared_private_edge_state_dynamic_selection_and_formal_full_model")
    if args.dp_release == "distributed_he":
        report.update(noise_location="edge_noise_shares_before_encryption_and_decryption",
                      he_execution="real_for_distributed_dp_he_only",
                      distributed_plan=distributed,
                      assumption="all_edges_online_noise_shares_unknown_to_cloud_no_edge_cloud_collusion",
                      accountant_scope="one_global_release_per_client_per_round_per_trajectory",
                      protocol_limit="single_process_keys_not_isolated_no_threshold_decryption",
                      noise_share_warning="individual_edge_shares_do_not_independently_meet_target_DP")
    if args.he_custody == "trusted_edge":
        report.update(protocol_limit="trusted_full_key_holder_can_bypass_policy_no_OS_sandbox_or_network_auth",
                      he_custodian_edge=0,
                      cloud_input="ciphertexts_and_public_round_manifest_only",
                      cohort_authorization="immutable_per_custodian_session",
                      privacy_randomness_warning="seeded_private_diagnostic_not_production_DP",
                      trusted_process_scope="simulates_all_edges_training_and_noise_not_separate_edge_hosts")

    def save():
        path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    started = time.perf_counter()
    save()
    print(f"Report {path}", flush=True)
    custodian = None
    try:
        if args.he_custody == "trusted_edge":
            from experiments.trusted_edge_custodian import TrustedEdgeCustodian
            custodian = TrustedEdgeCustodian(plan, release_limit=args.privacy_horizon)
            report["custodian_task_id"] = custodian.task_id
        for method in methods:
            state, cache = initial, {}
            ledgers = [PrivacyAccountant(args.epsilon, args.delta) for _ in indices]
            for round_idx in range(args.rounds):
                round_started = time.perf_counter()
                dp = method == protected_method
                distributed_dp = method == "distributed_dp_he"
                if dp and not all(l.can_add_event(sigma) for l in ledgers):
                    raise ValueError("Edge release would exceed a client budget")
                updates, keys = [], None
                for client, idx in enumerate(indices):
                    diff = split_local_train_lenet5(
                        "LIIC", state["end"], state["edge"], x[idx], y[idx], args.local_epochs,
                        args.lr, device, model, shape, classes,
                        training_seed=args.seed + round_idx * args.clients + client, model_cache=cache)
                    layout = [(p, k) for p in sorted(diff) for k in sorted(diff[p])]
                    if keys is not None and keys != layout:
                        raise ValueError("Client layouts differ")
                    keys = layout
                    updates.append(torch.cat([diff[p][k].reshape(-1) for p, k in keys]))
                if device.type == "cuda":
                    torch.cuda.synchronize()
                training_sec = time.perf_counter() - round_started
                release_started = time.perf_counter()
                packets, edge_rows = {}, []
                for group in plan:
                    multiplier = sigma if dp else 0
                    if distributed_dp:
                        multiplier = (distributed["groups"][group["edge"]]["noise_std_before_cloud_weight"]
                                      / group["sensitivity"])
                    # Noise is added here before encryption; no cloud-side noise.
                    packet, internal = release_edge(
                        torch.stack([updates[i] for i in group["clients"]]), group["weights"],
                        clip_norm=args.clip_norm, noise_multiplier=multiplier,
                        seed=edge_noise_seed(args.seed, round_idx, group["edge"]),
                        clip=method != "no_protection")
                    packets[group["edge"]] = packet
                    edge_rows.append(dict(edge=group["edge"], **internal))
                he_metrics = None
                if distributed_dp:
                    if custodian is None:
                        released, he_metrics = seal_edge_sum(packets, plan)
                        released = released.to(device)
                    else:
                        arrays = {i: v.detach().cpu().numpy() for i, v in packets.items()}
                        decoded, he_metrics = custodian.aggregate(arrays, round_idx)
                        expected = sum(arrays[g["edge"]].astype(np.float64) * g["cloud_weight"] for g in plan)
                        error = float(np.max(np.abs(decoded - expected)))
                        if not math.isfinite(error) or error > 1e-5:
                            raise ValueError("Custodian HE arithmetic check failed")
                        he_metrics["paired_max_abs_error"] = error
                        released = torch.as_tensor(decoded, device=device, dtype=packets[0].dtype)
                else:
                    released = torch.stack([packets[g["edge"]] * g["cloud_weight"] for g in plan]).sum(dim=0)
                if dp:
                    for ledger in ledgers:
                        ledger.add_event(sigma)
                state = apply_vector_update(state, keys, released)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                release_sec = time.perf_counter() - release_started
                end.load_state_dict(state["end"])
                edge_model.load_state_dict(state["edge"])
                eval_started = time.perf_counter()
                loss, acc = split_evaluate(end, edge_model, tx, ty, device, shape)
                if device.type == "cuda":
                    torch.cuda.synchronize()
                eps = [l.current_epsilon() for l in ledgers] if dp else [None] * args.clients
                row = dict(method=method, round=round_idx + 1, accuracy=acc, loss=loss,
                           dimensions=released.numel(), edges=edge_rows, client_epsilon=eps,
                           max_client_epsilon=max(eps) if dp else None,
                           dp_events_per_client=round_idx + 1 if dp else 0,
                           he_metrics=he_metrics,
                           noise_multiplier_for_accounting=sigma if dp else None,
                           global_sensitivity=distributed["global_sensitivity"] if distributed_dp else None,
                           edge_packets_this_round=len(plan),
                           cloud_noise_std=math.sqrt(sum((g["cloud_weight"] * r["noise_std"]) ** 2
                                                        for g, r in zip(plan, edge_rows))),
                           released_update_norm=float(released.norm()), training_wall_sec=training_sec,
                           release_wall_sec=release_sec, eval_wall_sec=time.perf_counter() - eval_started,
                           round_wall_sec=time.perf_counter() - round_started)
                report["results"].append(row)
                report["wall_time_sec"] = time.perf_counter() - started
                save()
                print(f"[{method}] {round_idx + 1}/{args.rounds} acc={acc:.4f} loss={loss:.4f} "
                      f"epsilon={row['max_client_epsilon']} cloud_noise_std={row['cloud_noise_std']:.6f}",
                      flush=True)
        report["status"] = "completed"
    except Exception as exc:
        report["status"], report["error"] = "failed", repr(exc)
        raise
    finally:
        if custodian is not None:
            custodian.close()
        report["wall_time_sec"] = time.perf_counter() - started
        save()
    print(f"Completed {path}", flush=True)


if __name__ == "__main__":
    main()
