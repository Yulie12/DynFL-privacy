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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--privacy-horizon", type=int, default=100)
    parser.add_argument("--clients", type=int, default=100)
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
    parser.add_argument("--model", choices=["resnet18_pretrained_head", "resnet18_pretrained"],
                        default="resnet18_pretrained_head",
                        help="Independent-client utility reference, not the dynamic execution protocol")
    parser.add_argument("--output-root", default="out/trusted_aggregate_head_trajectory")
    args = parser.parse_args()
    trajectory_noise_seed(args.seed, 0, args.noise_stream)
    for step in args.server_steps:
        validate_server_step(step)
    if len(set(args.server_steps)) != len(args.server_steps):
        parser.error("Server steps must be unique")
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
                  initial_accuracy=initial_acc, initial_loss=initial_loss, noise_multiplier=sigma)
    path = output / "report.json"
    cases = [("no_protection", None)] + [(m, c) for c in args.clip_norms
             for m in ("clip_only", "trusted_aggregate_dp")]
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
                scale = min(1.0, c / max(norm, 1e-12)) if c is not None else 1.0
                mean.add_(raw, alpha=scale / args.clients)
                raw_mean.add_(raw / args.clients)
                clipped_count += int(scale < 1.0)
            std = scales[c]["aggregate_noise_std"] if dp else 0.0
            generator = torch.Generator(device=device).manual_seed(
                trajectory_noise_seed(args.seed, round_idx, args.noise_stream))
            noise = torch.randn(mean.shape, generator=generator, device=device) * std
            state = apply_vector_update(state, keys, mean + noise, server_step=step)
            if dp:
                ledger.add_event(sigma)
            end.load_state_dict(state["end"])
            edge.load_state_dict(state["edge"])
            loss, accuracy = split_evaluate(end, edge, tx, ty, device, shape)
            row = dict(method=method, clip_norm=c, server_step=step, round=round_idx + 1, accuracy=accuracy, loss=loss,
                       signal_norm=float(mean.norm()), noise_norm=float(noise.norm()),
                       applied_signal_norm=float(mean.norm()) * step,
                       applied_noise_norm=float(noise.norm()) * step, applied_noise_std=std * step,
                       noise_std=std, clipping_bias_norm=float((mean - raw_mean).norm()),
                       clipped_fraction=clipped_count / args.clients, dimensions=mean.numel(),
                       recorded_epsilon=ledger.current_epsilon() if dp else None,
                       release_count=round_idx + 1, dp_event_count=round_idx + 1 if dp else 0)
            report["results"].append(row)
            report["wall_time_sec"] = time.perf_counter() - started
            path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
            print(json.dumps(row, allow_nan=False), flush=True)
    report["status"] = "completed"
    path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Completed {path}", flush=True)


if __name__ == "__main__":
    main()
