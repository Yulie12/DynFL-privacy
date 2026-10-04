"""Short independent-client trajectories under a trusted aggregate-release model."""
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

from experiments.validate_trusted_aggregate_dp import release_scales
from dynfed.fmnist_lenet5_dynamic import load_image_dataset_arrays
from dynfed.privacy import PrivacyAccountant, calibrate_gaussian_noise
from dynfed.split_learning import build_split_pair, clip_state_difference, split_evaluate, split_local_train_lenet5
from dynfed.utils import timestamped_dir


def trajectory_noise_seed(seed: int, round_idx: int, stream: str = "seed_sequence") -> int:
    """Separate experiment/round streams, paired across clipping controls."""
    if seed < 0 or round_idx < 0:
        raise ValueError("Seed and round must be nonnegative")
    if stream == "legacy_additive":
        return seed + 10000 + round_idx
    if stream != "seed_sequence":
        raise ValueError("Unknown noise stream")
    return int(np.random.SeedSequence([seed, round_idx, 10000]).generate_state(1, dtype=np.uint64)[0])


def validate_server_step(server_step: float) -> None:
    if not math.isfinite(server_step) or not 0 < server_step <= 1:
        raise ValueError("Diagnostic server step must be finite and in (0, 1]")


def apply_vector_update(base, keys, vector, server_step: float = 1.0):
    validate_server_step(server_step)
    expected = sum(base[p][k].numel() for p, k in keys)
    if vector.ndim != 1 or vector.numel() != expected or len(set(keys)) != len(keys):
        raise ValueError("Update layout mismatch")
    if not bool(torch.isfinite(vector).all()):
        raise ValueError("Non-finite released update")
    state = {p: {k: v.detach().clone() for k, v in values.items()} for p, values in base.items()}
    offset = 0
    for p, k in keys:
        size = state[p][k].numel()
        state[p][k].add_(vector[offset:offset + size].reshape_as(state[p][k]), alpha=server_step)
        offset += size
    return state


def hierarchical_mean_from_edge_sums(edge_sums, edge_counts):
    """Weighted cloud mean of per-edge sums of already clipped client updates.

    The fixed trusted-curator diagnostic keeps edge sums internal. Neither edge
    sums nor the direct-reference mean are released or separately DP-protected.
    """
    if (not edge_sums or len(edge_sums) != len(edge_counts)
            or any(n <= 0 for n in edge_counts)):
        raise ValueError("Each configured edge must contain at least one client")
    total_clients = sum(edge_counts)
    template = edge_sums[0]
    result = torch.zeros_like(template)
    for edge_sum, count in zip(edge_sums, edge_counts):
        if edge_sum.shape != template.shape or edge_sum.device != template.device:
            raise ValueError("Incompatible edge update layouts")
        if not bool(torch.isfinite(edge_sum).all()):
            raise ValueError("Non-finite edge aggregate")
        # Weight each edge MEAN by its actual number of clients, not 1/edges.
        result.add_(edge_sum / count, alpha=count / total_clients)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--privacy-horizon", type=int, default=100)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--edges", type=int, default=1,
                        help="Fixed disjoint client groups; default keeps direct baseline")
    parser.add_argument("--aggregation-topology", choices=["direct", "hierarchical"],
                        default="direct", help="Only changes the internal aggregation arithmetic")
    parser.add_argument("--check-direct-parity", action="store_true",
                        help="Assert the hierarchical clipped mean matches the direct mean each round")
    parser.add_argument("--train-limit", type=int, default=6000)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=8)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clip-norms", type=float, nargs="+", default=[0.1, 0.15])
    parser.add_argument("--server-steps", type=float, nargs="+", default=[1.0])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--noise-stream", choices=["seed_sequence", "legacy_additive"],
                        default="seed_sequence", help="Legacy stream only for reproducing old diagnostics")
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--model", choices=["resnet18_pretrained_adapter", "resnet18_pretrained_head", "resnet18_pretrained_layer4_head", "resnet18_pretrained"],
                        default="resnet18_pretrained_head",
                        help="Independent-client utility reference, not the dynamic execution protocol")
    parser.add_argument("--methods", nargs="+", choices=["no_protection", "clip_only", "trusted_aggregate_dp"],
                        help="Optional subset of diagnostic paths; default retains all controls")
    parser.add_argument("--output-root", default="out/trusted_aggregate_head_trajectory")
    args = parser.parse_args()
    if args.methods is not None and len(set(args.methods)) != len(args.methods):
        parser.error("Methods must be unique")
    trajectory_noise_seed(args.seed, 0, args.noise_stream)
    for step in args.server_steps:
        validate_server_step(step)
    if len(set(args.server_steps)) != len(args.server_steps):
        parser.error("Server steps must be unique")
    if not 1 <= args.edges <= args.clients:
        parser.error("Edges must be in [1, clients]")
    if min(args.rounds, args.clients, args.test_limit, args.local_epochs) < 1 or args.train_limit < args.clients:
        parser.error("Positive counts and nonempty client partitions required")
    if args.rounds > args.privacy_horizon or len(set(args.clip_norms)) != len(args.clip_norms):
        parser.error("Rounds must fit accounting horizon and thresholds must be unique")
    sigma = calibrate_gaussian_noise(args.epsilon, args.delta, args.privacy_horizon)
    scales = {c: release_scales(args.clients, c, sigma) for c in args.clip_norms}
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    model = args.model
    x, y, tx, ty, shape, _, classes = load_image_dataset_arrays(
        "cifar10", ROOT / "experiments/data/cifar10", args.train_limit, args.test_limit, args.seed)
    indices = np.array_split(np.random.default_rng(args.seed).permutation(len(y)), args.clients)
    end, edge = build_split_pair(model, device, input_channels=shape[0], image_size=shape[1], num_classes=classes)
    initial = {"end": {k: v.detach().clone() for k, v in end.state_dict().items()},
               "edge": {k: v.detach().clone() for k, v in edge.state_dict().items()}}
    initial_loss, initial_acc = split_evaluate(end, edge, tx, ty, device, shape)
    output = timestamped_dir(args.output_root, "trajectory").resolve()
    output.mkdir(parents=True, exist_ok=False)
    report = dict(config=vars(args), model=model, status="running", results=[],
                  scope="private_diagnostic_not_paper_result", adjacency="client_replacement_fixed_roster",
                  trust="curator_authorized_for_raw_aggregate; individual_transport_not_implemented",
                  he_execution="not_used", full_protocol_dp="not_established",
                  accountant_scope="one_aggregate_per_round_per_trajectory_not_joint_experiment",
                  diagnostic_outputs="internal_only_including_controls_and_raw_norms",
                  randomness="deterministic_research_seed_not_production_DP_randomness",
                  noise_stream=args.noise_stream,
                  aggregation_scope="in_process_fixed_group_arithmetic_only_not_a_network_or_secure_protocol",
                  edge_aggregates="internal_unreleased_and_not_separately_DP_protected",
                  released_dp_events="one_cloud_aggregate_per_round_per_DP_trajectory",
                  initial_accuracy=initial_acc, initial_loss=initial_loss, noise_multiplier=sigma)
    path = output / "report.json"
    cases = [("no_protection", None)] + [(m, c) for c in args.clip_norms
             for m in ("clip_only", "trusted_aggregate_dp")]
    if args.methods is not None:
        cases = [(m, c) for m, c in cases if m in args.methods]
    print(f"Report {path}", flush=True)
    started = time.perf_counter()
    for step, method, c in [(step, m, c) for step in args.server_steps for m, c in cases]:
        state = initial
        cache = {}
        ledger = PrivacyAccountant(args.epsilon, args.delta)
        for round_idx in range(args.rounds):
            dp = method == "trusted_aggregate_dp"
            if dp and not ledger.can_add_event(sigma):
                raise ValueError("Aggregate event would exceed budget")
            raw_mean = mean = None
            keys = None
            clipped_count = 0
            use_edges = args.aggregation_topology == "hierarchical" or args.check_direct_parity
            edge_sums = None
            edge_counts = [0] * args.edges if use_edges else None
            for client, idx in enumerate(indices):
                diff = split_local_train_lenet5(
                    "LIIC", state["end"], state["edge"], x[idx], y[idx], args.local_epochs,
                    args.lr, device, model, shape, classes,
                    training_seed=args.seed + round_idx * args.clients + client, model_cache=cache)
                _, norm, _ = clip_state_difference(diff, c if c is not None else 1.0, device)
                layout = [(p, k) for p in sorted(diff) for k in sorted(diff[p])]
                if keys is not None and keys != layout:
                    raise ValueError("Client update layouts differ")
                keys = layout
                raw = torch.cat([diff[p][k].reshape(-1) for p, k in keys])
                if mean is None:
                    mean, raw_mean = torch.zeros_like(raw), torch.zeros_like(raw)
                    if use_edges:
                        edge_sums = [torch.zeros_like(raw) for _ in range(args.edges)]
                scale = min(1.0, c / max(norm, 1e-12)) if c is not None else 1.0
                mean.add_(raw, alpha=scale / args.clients)
                if use_edges:
                    edge_id = client % args.edges  # fixed, disjoint, public grouping
                    edge_sums[edge_id].add_(raw, alpha=scale)
                    edge_counts[edge_id] += 1
                raw_mean.add_(raw / args.clients)
                clipped_count += int(scale < 1.0)
            release_mean = mean
            parity_max_abs_error = None
            if use_edges:
                cloud_mean = hierarchical_mean_from_edge_sums(edge_sums, edge_counts)
                parity_max_abs_error = float((cloud_mean - mean).abs().max())
                if args.check_direct_parity and not torch.allclose(
                        cloud_mean, mean, rtol=2e-5, atol=2e-7):
                    raise AssertionError(
                        f"Hierarchical/direct mismatch in round {round_idx + 1}: "
                        f"max absolute error {parity_max_abs_error:.9g}")
                if args.aggregation_topology == "hierarchical":
                    release_mean = cloud_mean
            std = scales[c]["aggregate_noise_std"] if dp else 0.0
            generator = torch.Generator(device=device).manual_seed(
                trajectory_noise_seed(args.seed, round_idx, args.noise_stream))
            noise = torch.randn(mean.shape, generator=generator, device=device) * std
            state = apply_vector_update(state, keys, release_mean + noise, server_step=step)
            if dp:
                ledger.add_event(sigma)
            end.load_state_dict(state["end"])
            edge.load_state_dict(state["edge"])
            loss, accuracy = split_evaluate(end, edge, tx, ty, device, shape)
            row = dict(method=method, clip_norm=c, server_step=step, round=round_idx + 1, accuracy=accuracy, loss=loss,
                       signal_norm=float(release_mean.norm()), noise_norm=float(noise.norm()),
                       applied_signal_norm=float(release_mean.norm()) * step,
                       applied_noise_norm=float(noise.norm()) * step, applied_noise_std=std * step,
                       noise_std=std, clipping_bias_norm=float((release_mean - raw_mean).norm()),
                       clipped_fraction=clipped_count / args.clients, dimensions=mean.numel(),
                       recorded_epsilon=ledger.current_epsilon() if dp else None,
                       release_count=round_idx + 1, dp_event_count=round_idx + 1 if dp else 0,
                       aggregation_topology=args.aggregation_topology, edges=args.edges,
                       hierarchy_direct_max_abs_error=parity_max_abs_error)
            report["results"].append(row)
            report["wall_time_sec"] = time.perf_counter() - started
            path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(row, allow_nan=False), flush=True)
    report["status"] = "completed"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Completed {path}", flush=True)


if __name__ == "__main__":
    main()
