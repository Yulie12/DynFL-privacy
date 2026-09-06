from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.fmnist_lenet5_dynamic import (  # noqa: E402
    _default_data_root,
    _partition_clients_lenet5,
    _split_client_indices,
    load_image_dataset_arrays,
)
from dynfed.privacy import calibrate_gaussian_noise  # noqa: E402
from dynfed.split_learning import (  # noqa: E402
    _make_optimizer,
    _prepare_model_for_training,
    build_split_pair,
    normalize_model_name,
)
from dynfed.utils import timestamped_dir  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Diagnose one-time record-level DP feature release with a frozen "
            "public encoder and federated downstream training."
        )
    )
    parser.add_argument("--dataset", default="cifar10", choices=["cifar10", "cifar100"])
    parser.add_argument("--model", default="resnet18_pretrained")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--edges", type=int, default=10)
    parser.add_argument("--partition-mode", default="extreme_edge_label_skew")
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument("--local-epochs", type=int, default=3)
    parser.add_argument("--local-steps", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epsilon", type=float, default=8.0)
    parser.add_argument("--delta", type=float, default=1e-5)
    parser.add_argument("--clip-norm", type=float, default=1.0)
    parser.add_argument(
        "--spatial-pool",
        action="store_true",
        help="Average pool the frozen feature map before clipping and release.",
    )
    parser.add_argument(
        "--deep-features",
        action="store_true",
        help="Release the frozen pretrained penultimate representation.",
    )
    parser.add_argument("--no-noise", action="store_true")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-root", default="out/frozen_feature_dp_diagnostic")
    return parser.parse_args()


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class _FrozenDeepEncoder(torch.nn.Module):
    def __init__(self, end: torch.nn.Module, edge: torch.nn.Module) -> None:
        super().__init__()
        self.end = end
        self.layer3 = edge.layer3
        self.layer4 = edge.layer4
        self.avgpool = edge.avgpool

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value = self.end(x)
        value = self.layer3(value)
        value = self.layer4(value)
        value = self.avgpool(value)
        return torch.flatten(value, 1)


def _encode_once(
    encoder: torch.nn.Module,
    x: np.ndarray,
    *,
    input_shape: tuple[int, int, int],
    clip_norm: float,
    noise_multiplier: float,
    device: torch.device,
    seed: int,
    spatial_pool: bool = False,
    batch_size: int = 256,
) -> tuple[torch.Tensor, dict[str, float]]:
    encoder.eval()
    generator = torch.Generator(device=device.type)
    generator.manual_seed(seed)
    outputs: list[torch.Tensor] = []
    norm_before_sum = 0.0
    clipped_count = 0
    signal_sq_sum = 0.0
    noise_sq_sum = 0.0
    count = 0
    std = 2.0 * float(clip_norm) * float(noise_multiplier)

    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            stop = min(start + batch_size, len(x))
            bx = torch.from_numpy(x[start:stop]).float().to(device).view(-1, *input_shape)
            embedding = encoder(bx)
            if spatial_pool and embedding.ndim == 4:
                embedding = F.adaptive_avg_pool2d(embedding, output_size=1).flatten(1)
            flat = embedding.reshape(len(embedding), -1)
            norms = torch.linalg.vector_norm(flat, dim=1).clamp_min(1e-12)
            scales = (float(clip_norm) / norms).clamp(max=1.0)
            clipped = embedding * scales.reshape([len(embedding)] + [1] * (embedding.ndim - 1))
            if noise_multiplier > 0.0:
                noise = torch.randn(
                    clipped.shape,
                    generator=generator,
                    device=device,
                    dtype=clipped.dtype,
                ) * std
            else:
                noise = torch.zeros_like(clipped)
            outputs.append((clipped + noise).cpu())
            norm_before_sum += float(norms.sum().item())
            clipped_count += int((norms > clip_norm).sum().item())
            signal_sq_sum += float((clipped * clipped).sum().item())
            noise_sq_sum += float((noise * noise).sum().item())
            count += len(embedding)

    signal_norm = math.sqrt(max(signal_sq_sum / max(count, 1), 0.0))
    noise_norm = math.sqrt(max(noise_sq_sum / max(count, 1), 0.0))
    return torch.cat(outputs, dim=0), {
        "mean_unclipped_norm": norm_before_sum / max(count, 1),
        "clip_fraction": clipped_count / max(count, 1),
        "mean_signal_norm": signal_norm,
        "mean_noise_norm": noise_norm,
        "signal_to_noise_ratio": signal_norm / max(noise_norm, 1e-12),
        "per_coordinate_noise_std": std,
    }


def _train_client(
    model: torch.nn.Module,
    features: torch.Tensor,
    labels: torch.Tensor,
    *,
    epochs: int,
    local_steps: int,
    lr: float,
    model_name: str,
    device: torch.device,
    seed: int,
) -> int:
    model.train()
    if not isinstance(model, torch.nn.Linear):
        _prepare_model_for_training(model, model_name)
    optimizer = _make_optimizer(model, lr, model_name)
    if optimizer is None or len(features) == 0:
        return 0
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(features, labels),
        batch_size=128,
        shuffle=True,
        generator=generator,
    )
    completed = 0
    for _ in range(max(0, epochs)):
        for batch_features, batch_labels in loader:
            if completed >= local_steps:
                return completed
            optimizer.zero_grad()
            logits = model(batch_features.to(device))
            loss = F.cross_entropy(logits, batch_labels.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for parameter in model.parameters() if parameter.requires_grad],
                max_norm=5.0,
            )
            optimizer.step()
            completed += 1
    return completed


def _evaluate(
    model: torch.nn.Module,
    features: torch.Tensor,
    labels: np.ndarray,
    device: torch.device,
) -> float:
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(features), 256):
            stop = min(start + 256, len(features))
            logits = model(features[start:stop].to(device))
            targets = torch.from_numpy(labels[start:stop]).long().to(device)
            correct += int((logits.argmax(dim=1) == targets).sum().item())
    return correct / max(len(features), 1)


def main() -> None:
    args = parse_args()
    _seed_everything(args.seed)
    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model_name = normalize_model_name(args.model)
    output_dir = timestamped_dir(args.output_root, "frozen_feature_dp")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_root = Path(args.data_root or _default_data_root(args.dataset))
    x_train, y_train, x_test, y_test, input_shape, _, num_classes = load_image_dataset_arrays(
        args.dataset,
        data_root,
        args.train_limit,
        args.test_limit,
        args.seed,
    )
    partitions = _partition_clients_lenet5(
        y_train,
        args.clients,
        args.edges,
        False,
        args.partition_mode,
        args.seed,
    )
    train_partitions, _ = _split_client_indices(partitions, test_ratio=0.2, seed=args.seed)

    encoder, downstream = build_split_pair(
        model_name,
        device,
        input_channels=input_shape[0],
        image_size=input_shape[1],
        num_classes=num_classes,
    )
    if args.spatial_pool and args.deep_features:
        raise ValueError("Choose only one of --spatial-pool and --deep-features")
    if args.deep_features:
        encoder = _FrozenDeepEncoder(encoder, downstream).to(device)
    for parameter in encoder.parameters():
        parameter.requires_grad_(False)

    noise_multiplier = 0.0 if args.no_noise else calibrate_gaussian_noise(
        args.epsilon,
        args.delta,
        1,
    )
    print(
        f"Encoding each record once with noise_multiplier={noise_multiplier:.6f}",
        flush=True,
    )
    private_train, train_stats = _encode_once(
        encoder,
        x_train,
        input_shape=input_shape,
        clip_norm=args.clip_norm,
        noise_multiplier=noise_multiplier,
        device=device,
        seed=args.seed + 1000,
        spatial_pool=args.spatial_pool,
    )
    private_test, test_stats = _encode_once(
        encoder,
        x_test,
        input_shape=input_shape,
        clip_norm=args.clip_norm,
        noise_multiplier=noise_multiplier,
        device=device,
        seed=args.seed + 2000,
        spatial_pool=args.spatial_pool,
    )
    clean_test, _ = _encode_once(
        encoder,
        x_test,
        input_shape=input_shape,
        clip_norm=args.clip_norm,
        noise_multiplier=0.0,
        device=device,
        seed=args.seed + 3000,
        spatial_pool=args.spatial_pool,
    )
    del encoder
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.spatial_pool or args.deep_features:
        feature_dim = int(private_train[0].numel())
        downstream = torch.nn.Linear(feature_dim, num_classes).to(device)
        client_model = torch.nn.Linear(feature_dim, num_classes).to(device)
    else:
        _unused, client_model = build_split_pair(
            model_name,
            device,
            input_channels=input_shape[0],
            image_size=input_shape[1],
            num_classes=num_classes,
        )
        del _unused
    rows: list[dict[str, float | int]] = []
    started = time.perf_counter()

    for round_idx in range(args.rounds):
        global_state = {name: value.detach().clone() for name, value in downstream.state_dict().items()}
        aggregate = {
            name: torch.zeros_like(value, device=device)
            for name, value in global_state.items()
            if torch.is_floating_point(value)
        }
        total_weight = 0
        total_steps = 0
        for client_id, indices in enumerate(train_partitions):
            if len(indices) == 0:
                continue
            client_model.load_state_dict(global_state)
            local_features = private_train[indices]
            local_labels = torch.from_numpy(y_train[indices]).long()
            total_steps += _train_client(
                client_model,
                local_features,
                local_labels,
                epochs=args.local_epochs,
                local_steps=args.local_steps,
                lr=args.lr,
                model_name=model_name,
                device=device,
                seed=args.seed + round_idx * 1000 + client_id,
            )
            weight = len(indices)
            local_state = client_model.state_dict()
            for name in aggregate:
                aggregate[name].add_(local_state[name] - global_state[name], alpha=float(weight))
            total_weight += weight
        updated = dict(global_state)
        for name, value in aggregate.items():
            updated[name] = global_state[name] + value / max(total_weight, 1)
        downstream.load_state_dict(updated)
        clean_accuracy = _evaluate(downstream, clean_test, y_test, device)
        private_accuracy = _evaluate(downstream, private_test, y_test, device)
        row = {
            "round": round_idx + 1,
            "clean_test_accuracy": clean_accuracy,
            "private_test_accuracy": private_accuracy,
            "elapsed_wall_time_sec": time.perf_counter() - started,
            "local_steps": total_steps,
        }
        rows.append(row)
        print(
            f"round {round_idx + 1:03d}/{args.rounds} "
            f"clean_acc={clean_accuracy:.4f} private_acc={private_accuracy:.4f}",
            flush=True,
        )

    with (output_dir / "round_metrics.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "status": "completed",
        "privacy_scope": "one record-level embedding release; labels treated as public",
        "epsilon": None if args.no_noise else args.epsilon,
        "delta": None if args.no_noise else args.delta,
        "noise_multiplier": noise_multiplier,
        "clip_norm": args.clip_norm,
        "train_feature_statistics": train_stats,
        "test_feature_statistics": test_stats,
        "final_clean_test_accuracy": rows[-1]["clean_test_accuracy"],
        "final_private_test_accuracy": rows[-1]["private_test_accuracy"],
        "config": vars(args),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    torch.save(downstream.state_dict(), output_dir / "downstream_model.pt")
    print(f"[COMPLETED] {output_dir}", flush=True)


if __name__ == "__main__":
    main()
