"""One-round trusted-curator diagnostic; not a full-protocol DP certificate."""
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

from dynfed.fmnist_lenet5_dynamic import load_image_dataset_arrays
from dynfed.privacy import calibrate_gaussian_noise, compute_event_epsilon
from dynfed.protection_rules import aggregate_replacement_bound
from dynfed.split_learning import build_split_pair, clip_state_difference, split_evaluate, split_local_train_lenet5
from dynfed.utils import timestamped_dir


def release_scales(clients: int, clip_norm: float, multiplier: float) -> dict[str, float]:
    if clients < 1 or not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError("Positive client count and multiplier required")
    sensitivity = aggregate_replacement_bound(
        [1.0 / clients] * clients, [{i} for i in range(clients)], clip_norm=clip_norm,
    )
    return {
        "packet_sensitivity": 2 * clip_norm,
        "packet_noise_std": 2 * clip_norm * multiplier,
        "packet_dp_average_noise_std": 2 * clip_norm * multiplier / math.sqrt(clients),
        "aggregate_sensitivity": sensitivity,
        "aggregate_noise_std": sensitivity * multiplier,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clients", type=int, default=20)
    parser.add_argument("--train-limit", type=int, default=1200)
    parser.add_argument("--test-limit", type=int, default=200)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--model", default="resnet18_pretrained",
                        choices=["resnet18_pretrained", "resnet18_pretrained_head",
                                 "resnet18_pretrained_layer4_head", "resnet18_pretrained_adapter"])
    parser.add_argument("--privacy-horizon", type=int, default=100)
    parser.add_argument("--epsilon", type=float, default=8)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clip-norm", type=float, default=1)
    parser.add_argument("--clip-norms", type=float, nargs="+",
                        help="Private paired sweep; overrides --clip-norm without retraining clients")
    parser.add_argument("--noise-trials", type=int, default=3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument("--output-root", default="out/trusted_aggregate_single_round")
    args = parser.parse_args()
    if min(args.clients, args.local_epochs, args.noise_trials, args.test_limit) < 1 or args.train_limit < args.clients:
        parser.error("Positive counts and at least one sample per client required")
    sigma = calibrate_gaussian_noise(args.epsilon, args.delta, args.privacy_horizon)
    clip_norms = args.clip_norms or [args.clip_norm]
    if len(set(clip_norms)) != len(clip_norms):
        parser.error("Clipping thresholds must be unique")
    scales_by_clip = {c: release_scales(args.clients, c, sigma) for c in clip_norms}
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    output = timestamped_dir(args.output_root, "trusted_aggregate").resolve()
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    x, y, tx, ty, shape, _, classes = load_image_dataset_arrays(
        "cifar10", ROOT / "experiments/data/cifar10", args.train_limit, args.test_limit, args.seed,
    )
    end, edge = build_split_pair(args.model, device, input_channels=shape[0], image_size=shape[1], num_classes=classes)
    base = {"end": {k: v.detach().clone() for k, v in end.state_dict().items()},
            "edge": {k: v.detach().clone() for k, v in edge.state_dict().items()}}
    initial_loss, initial_accuracy = split_evaluate(end, edge, tx, ty, device, shape)
    indices = np.array_split(np.random.default_rng(args.seed).permutation(len(y)), args.clients)
    raw_sum = None
    clipped_sums = {}
    preclip_norms = []
    keys = None
    cache = {}
    for client, idx in enumerate(indices):
        diff = split_local_train_lenet5(
            "LIIC", base["end"], base["edge"], x[idx], y[idx], args.local_epochs,
            args.lr, device, args.model, shape, classes, training_seed=args.seed + client,
            model_cache=cache,
        )
        # Validate the complete update and obtain the same global norm as training.
        _, norm, _ = clip_state_difference(diff, clip_norms[0], device)
        current_keys = [(part, key) for part in sorted(diff) for key in sorted(diff[part])]
        if keys is not None and keys != current_keys:
            raise ValueError("Client parameter layouts differ")
        keys = current_keys
        raw = torch.cat([diff[p][k].reshape(-1) for p, k in keys])
        if raw_sum is None:
            raw_sum = torch.zeros_like(raw)
            clipped_sums = {c: torch.zeros_like(raw) for c in clip_norms}
        raw_sum.add_(raw / args.clients)
        for c, clipped_sum in clipped_sums.items():
            clipped_sum.add_(raw, alpha=min(1.0, c / max(norm, 1e-12)) / args.clients)
        preclip_norms.append(norm)
        print(f"Trained independent client {client + 1}/{args.clients}", flush=True)

    def evaluate(vector):
        states = {p: {k: v.clone() for k, v in state.items()} for p, state in base.items()}
        offset = 0
        for p, k in keys:
            count = states[p][k].numel()
            states[p][k].add_(vector[offset:offset + count].reshape_as(states[p][k]))
            offset += count
        assert offset == vector.numel()
        end.load_state_dict(states["end"])
        edge.load_state_dict(states["edge"])
        return split_evaluate(end, edge, tx, ty, device, shape)

    rows = []
    cases = [("no_protection", None, raw_sum, 0.0)]
    for c, signal in clipped_sums.items():
        scales = scales_by_clip[c]
        cases.extend([("clip_only", c, signal, 0.0),
                      ("packet_dp_average", c, signal, scales["packet_dp_average_noise_std"]),
                      ("trusted_aggregate_dp", c, signal, scales["aggregate_noise_std"])])
    for name, clip, signal, std in cases:
        for trial in range(args.noise_trials if std else 1):
            # Coupled standard normals isolate scale effects. This samples the
            # exact Gaussian law of averaged independent packet noise, not HE.
            generator = torch.Generator(device=device).manual_seed(args.seed + 10000 + trial)
            noise = torch.randn(signal.shape, generator=generator, device=device) * std
            loss, accuracy = evaluate(signal + noise)
            row = dict(method=name, clip_norm=clip, trial=trial, loss=loss, accuracy=accuracy,
                       signal_norm=float(signal.norm()), noise_norm=float(noise.norm()), noise_std=std)
            row["clipping_bias_norm"] = float((signal - raw_sum).norm())
            row["noise_to_signal"] = row["noise_norm"] / max(row["signal_norm"], 1e-12)
            rows.append(row)
            print(json.dumps(row, allow_nan=False), flush=True)
    report = dict(
        scope="single_round_private_diagnostic_not_paper_result", config=vars(args),
        adjacency="client_replacement_fixed_roster", partition="seeded_random_client_partition",
        trust="curator_authorized_for_raw_aggregate; only_noisy_result_released",
        transport_scope="single_process_oracle_sees_individual_updates; confidential_transport_not_implemented",
        he_execution="not_used", released_models_in_deployment=1,
        diagnostic_outputs="all_trials_and_raw_metrics_private_only; not_jointly_DP_accounted",
        randomness="deterministic_research_seed_not_production_DP_randomness",
        full_protocol_dp="not_established", observed_rounds=1,
        noise_multiplier=sigma, one_release_epsilon=compute_event_epsilon(sigma, args.delta),
        trainable_dimensions=raw_sum.numel(), initial_loss=initial_loss, initial_accuracy=initial_accuracy,
        clip_fractions={str(c): sum(n > c for n in preclip_norms) / args.clients for c in clip_norms},
        preclip_norms=preclip_norms, client_samples=[len(idx) for idx in indices],
        scales_by_clip={str(c): scales for c, scales in scales_by_clip.items()},
        results=rows, wall_time_sec=time.perf_counter() - started,
    )
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    print(f"Report {output / 'report.json'}", flush=True)


if __name__ == "__main__":
    main()
