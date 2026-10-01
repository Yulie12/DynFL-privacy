from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.fmnist_lenet5_dynamic import (  # noqa: E402
    _partition_clients_lenet5,
    _split_client_indices,
    load_image_dataset_arrays,
)
from dynfed.selection import SelectionConfig, resolved_privacy_parameters  # noqa: E402
from dynfed.split_learning import (  # noqa: E402
    _make_optimizer,
    _prepare_model_for_training,
    _protect_tensor_dp,
    build_split_pair,
    normalize_model_name,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=(
            "Diagnose a channel-PCA privacy bottleneck before feature DP. "
            "Compares base-clean, bottleneck-clean, bottleneck+clip, and bottleneck+DP "
            "updates under identical data and seeds."
        )
    )
    p.add_argument("--dataset", default="cifar10", choices=["fmnist", "cifar10", "cifar100"])
    p.add_argument("--model", default="resnet18_pretrained_head")
    p.add_argument("--data-root", default=None)
    p.add_argument("--train-limit", type=int, default=12000)
    p.add_argument("--test-limit", type=int, default=2000)
    p.add_argument("--partition-mode", default="iid", choices=["iid", "dirichlet", "client_noniid", "edge_label_skew", "extreme_edge_label_skew"])
    p.add_argument("--dirichlet-alpha", type=float, default=0.5)
    p.add_argument("--clients", type=int, default=100)
    p.add_argument("--edges", type=int, default=10)
    p.add_argument("--client-ids", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    p.add_argument("--bottleneck-channels", nargs="+", type=int, default=[8, 16, 32])
    p.add_argument("--pca-calibration-samples", type=int, default=1024)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="cuda")
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--initial-epsilon", type=float, default=8.0)
    p.add_argument("--feature-epsilon-budget", type=float, default=None)
    p.add_argument("--dp-delta", type=float, default=1e-5)
    p.add_argument("--feature-clip-norm", type=float, default=0.25)
    p.add_argument("--feature-noise-multiplier", type=float, default=None)
    p.add_argument("--output", default="out/feature_bottleneck_diagnostic.csv")
    return p.parse_args()


def _state_vector(update: dict[str, dict[str, torch.Tensor]]) -> torch.Tensor:
    pieces: list[torch.Tensor] = []
    for part in ("end", "edge"):
        for name in sorted(update.get(part, {})):
            value = update[part][name]
            if torch.is_floating_point(value) or torch.is_complex(value):
                pieces.append(value.detach().float().cpu().reshape(-1))
    return torch.cat(pieces) if pieces else torch.zeros(0, dtype=torch.float32)


def _sq_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    delta = a - b
    return float(torch.dot(delta, delta).item())


def _norm(a: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(a).item())


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    na, nb = _norm(a), _norm(b)
    if na <= 1e-12 or nb <= 1e-12:
        return 0.0
    return float(torch.dot(a, b).item() / (na * nb))


def _fit_channel_pca(
    *,
    end: torch.nn.Module,
    x: np.ndarray,
    input_shape: tuple[int, int, int],
    device: torch.device,
    max_samples: int,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[int, int, int]]:
    if len(x) == 0:
        raise ValueError("PCA calibration set is empty")
    rng = np.random.default_rng(seed)
    count = min(len(x), max(1, int(max_samples)))
    idx = rng.choice(len(x), size=count, replace=False)
    x_t = torch.from_numpy(x[idx]).float().to(device).view(-1, *input_shape)

    was_training = end.training
    end.eval()
    sum_c: torch.Tensor | None = None
    sum_xx: torch.Tensor | None = None
    spatial_rows = 0
    first_shape: tuple[int, int, int] | None = None
    with torch.no_grad():
        for start in range(0, len(x_t), 128):
            feat = end(x_t[start : start + 128])
            if feat.ndim != 4:
                raise ValueError(
                    "channel-PCA bottleneck currently requires a 4-D split tensor [B,C,H,W]; "
                    f"got shape {tuple(feat.shape)}"
                )
            _, channels, height, width = feat.shape
            if first_shape is None:
                first_shape = (int(channels), int(height), int(width))
            rows = feat.permute(0, 2, 3, 1).reshape(-1, channels).double()
            batch_sum = rows.sum(dim=0)
            batch_xx = rows.T @ rows
            sum_c = batch_sum if sum_c is None else sum_c + batch_sum
            sum_xx = batch_xx if sum_xx is None else sum_xx + batch_xx
            spatial_rows += int(rows.shape[0])
    if was_training:
        end.train()

    assert sum_c is not None and sum_xx is not None and first_shape is not None
    mean = sum_c / float(spatial_rows)
    cov = sum_xx / float(spatial_rows) - torch.outer(mean, mean)
    cov = 0.5 * (cov + cov.T)
    evals, evecs = torch.linalg.eigh(cov)
    order = torch.argsort(evals, descending=True)
    evals = evals[order].clamp_min(0.0).float()
    evecs = evecs[:, order].float()
    return mean.float(), evecs, evals, first_shape


def _encode(emb: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    centered = emb - mean.view(1, -1, 1, 1)
    return torch.einsum("bchw,ck->bkhw", centered, basis)


def _decode(latent: torch.Tensor, mean: torch.Tensor, basis: torch.Tensor) -> torch.Tensor:
    reconstructed = torch.einsum("bkhw,ck->bchw", latent, basis)
    return reconstructed + mean.view(1, -1, 1, 1)


def _train_path(
    *,
    end_state: dict[str, torch.Tensor],
    edge_state: dict[str, torch.Tensor],
    x: np.ndarray,
    y: np.ndarray,
    model_name: str,
    input_shape: tuple[int, int, int],
    num_classes: int,
    epochs: int,
    lr: float,
    device: torch.device,
    training_seed: int,
    mean: torch.Tensor | None,
    basis: torch.Tensor | None,
    clip_norm: float,
    noise_multiplier: float,
    use_clipping: bool,
    use_noise: bool,
) -> tuple[torch.Tensor, dict[str, float]]:
    end, edge = build_split_pair(
        model_name,
        device,
        input_channels=input_shape[0],
        image_size=input_shape[1],
        num_classes=num_classes,
    )
    end.load_state_dict(end_state)
    edge.load_state_dict(edge_state)
    end.train()
    edge.train()
    _prepare_model_for_training(end, model_name)
    _prepare_model_for_training(edge, model_name)
    end_opt = _make_optimizer(end, lr, model_name, weight_decay=None)
    edge_opt = _make_optimizer(edge, lr, model_name, weight_decay=None)

    x_t = torch.from_numpy(x).float().to(device).view(-1, *input_shape)
    y_t = torch.from_numpy(y).long().to(device)
    dataset = torch.utils.data.TensorDataset(x_t, y_t)
    gen = torch.Generator().manual_seed(int(training_seed))
    loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=True, generator=gen)
    rng = np.random.default_rng(training_seed + 100_003)
    diagnostics: dict[str, float] = {}
    for key, value in {
        "feature_dp_release_batches": 0,
        "feature_dp_sample_count": 0,
        "feature_raw_norm_sum": 0.0,
        "feature_clipped_norm_sum": 0.0,
        "feature_noise_norm_sum": 0.0,
        "feature_distortion_norm_sum": 0.0,
        "feature_noise_l2_sq_sum": 0.0,
        "feature_distortion_l2_sq_sum": 0.0,
        "feature_clipped_sample_count": 0,
    }.items():
        diagnostics[key] = value

    for _ in range(max(0, int(epochs))):
        for bx, by in loader:
            if end_opt is not None:
                end_opt.zero_grad()
            if edge_opt is not None:
                edge_opt.zero_grad()
            emb = end(bx)
            if basis is None or mean is None:
                edge_input_value = emb
            else:
                latent = _encode(emb, mean, basis)
                if use_clipping or use_noise:
                    latent = _protect_tensor_dp(
                        latent,
                        "dp",
                        clip_norm,
                        noise_multiplier if use_noise else 0.0,
                        rng,
                        device,
                        1.0,
                        diagnostics=diagnostics,
                    )
                edge_input_value = _decode(latent, mean, basis)

            edge_input = edge_input_value.detach().requires_grad_(True)
            logits = edge(edge_input)
            client_logits = logits.detach().requires_grad_(True)
            client_loss = F.cross_entropy(client_logits, by, reduction="sum")
            label_grad = torch.autograd.grad(client_loss, client_logits)[0]
            logits.backward(label_grad)
            grad_to_end = edge_input.grad.detach()

            edge_trainable = [p for p in edge.parameters() if p.requires_grad]
            if edge_opt is not None and edge_trainable:
                torch.nn.utils.clip_grad_norm_(edge_trainable, max_norm=5.0)
                edge_opt.step()

            end_trainable = [p for p in end.parameters() if p.requires_grad]
            if end_opt is not None and end_trainable and emb.requires_grad:
                edge_input_value.backward(grad_to_end)
                torch.nn.utils.clip_grad_norm_(end_trainable, max_norm=5.0)
                end_opt.step()

    end_diff = {
        name: param.data - end_state[name]
        for name, param in end.named_parameters()
        if param.requires_grad
    }
    edge_diff = {
        name: param.data - edge_state[name]
        for name, param in edge.named_parameters()
        if param.requires_grad
    }
    return _state_vector({"end": end_diff, "edge": edge_diff}), diagnostics


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    x_train, y_train, _x_test, _y_test, input_shape, _label, num_classes = load_image_dataset_arrays(
        args.dataset,
        Path(args.data_root) if args.data_root else Path("data"),
        args.train_limit,
        args.test_limit,
        args.seed,
    )
    client_indices = _partition_clients_lenet5(
        y_train=y_train,
        num_clients=args.clients,
        num_edges=args.edges,
        iid=args.partition_mode == "iid",
        partition_mode=args.partition_mode,
        seed=args.seed,
        dirichlet_alpha=args.dirichlet_alpha,
    )
    client_train_indices, _ = _split_client_indices(client_indices, test_ratio=0.2, seed=args.seed)

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    end0, edge0 = build_split_pair(
        normalize_model_name(args.model),
        device,
        input_channels=input_shape[0],
        image_size=input_shape[1],
        num_classes=num_classes,
    )
    end_state = {k: v.detach().clone() for k, v in end0.state_dict().items()}
    edge_state = {k: v.detach().clone() for k, v in edge0.state_dict().items()}

    mean, evecs, evals, feature_shape = _fit_channel_pca(
        end=end0,
        x=x_train,
        input_shape=input_shape,
        device=device,
        max_samples=args.pca_calibration_samples,
        seed=args.seed + 17,
    )
    channels, height, width = feature_shape
    if max(args.bottleneck_channels) > channels:
        raise ValueError(f"bottleneck channel count cannot exceed source channels={channels}")
    total_var = max(float(evals.sum().item()), 1e-12)

    if args.feature_noise_multiplier is None:
        privacy_cfg = SelectionConfig(
            rounds=args.rounds,
            num_clients=args.clients,
            num_edges=args.edges,
            seed=args.seed,
            initial_epsilon=args.initial_epsilon,
            dp_feature_epsilon_budget=args.feature_epsilon_budget,
            dp_delta=args.dp_delta,
            privacy_local_epochs=args.local_epochs,
            omega_learning_rate=args.lr,
        )
        privacy = resolved_privacy_parameters(privacy_cfg)
        sigma_f = float(privacy["feature_noise_multiplier"])
        horizon_events = int(privacy["feature_horizon_events"])
    else:
        sigma_f = float(args.feature_noise_multiplier)
        horizon_events = -1

    rows: list[dict[str, float | int]] = []
    for k in args.bottleneck_channels:
        basis = evecs[:, :k].to(device)
        mean_dev = mean.to(device)
        retained_var = float(evals[:k].sum().item()) / total_var
        transmitted_dim = int(k * height * width)
        for client_id in args.client_ids:
            idx = client_train_indices[client_id]
            x = x_train[idx]
            y = y_train[idx]
            training_seed = args.seed * 1_000_003 + client_id * 97 + 17

            base, _ = _train_path(
                end_state=end_state, edge_state=edge_state, x=x, y=y,
                model_name=args.model, input_shape=input_shape, num_classes=num_classes,
                epochs=args.local_epochs, lr=args.lr, device=device, training_seed=training_seed,
                mean=None, basis=None, clip_norm=args.feature_clip_norm, noise_multiplier=0.0,
                use_clipping=False, use_noise=False,
            )
            bclean, _ = _train_path(
                end_state=end_state, edge_state=edge_state, x=x, y=y,
                model_name=args.model, input_shape=input_shape, num_classes=num_classes,
                epochs=args.local_epochs, lr=args.lr, device=device, training_seed=training_seed,
                mean=mean_dev, basis=basis, clip_norm=args.feature_clip_norm, noise_multiplier=0.0,
                use_clipping=False, use_noise=False,
            )
            bclip, _ = _train_path(
                end_state=end_state, edge_state=edge_state, x=x, y=y,
                model_name=args.model, input_shape=input_shape, num_classes=num_classes,
                epochs=args.local_epochs, lr=args.lr, device=device, training_seed=training_seed,
                mean=mean_dev, basis=basis, clip_norm=args.feature_clip_norm, noise_multiplier=0.0,
                use_clipping=True, use_noise=False,
            )
            bdp, diag = _train_path(
                end_state=end_state, edge_state=edge_state, x=x, y=y,
                model_name=args.model, input_shape=input_shape, num_classes=num_classes,
                epochs=args.local_epochs, lr=args.lr, device=device, training_seed=training_seed,
                mean=mean_dev, basis=basis, clip_norm=args.feature_clip_norm, noise_multiplier=sigma_f,
                use_clipping=True, use_noise=True,
            )

            n_diag = max(float(diag.get("feature_dp_sample_count", 0.0)), 1.0)
            base_norm = _norm(base)
            row = {
                "client_id": client_id,
                "bottleneck_channels": k,
                "transmitted_dimension": transmitted_dim,
                "pca_retained_variance_ratio": retained_var,
                "feature_clip_norm": float(args.feature_clip_norm),
                "feature_noise_multiplier": sigma_f,
                "base_clean_update_norm": base_norm,
                "D_bottleneck": _sq_distance(bclean, base),
                "D_clip_given_bottleneck": _sq_distance(bclip, bclean),
                "D_noise_given_clip": _sq_distance(bdp, bclip),
                "D_total": _sq_distance(bdp, base),
                "total_to_clean_norm_sq_ratio": _sq_distance(bdp, base) / max(base_norm * base_norm, 1e-12),
                "base_vs_dp_cosine": _cosine(base, bdp),
                "latent_raw_norm_mean": float(diag.get("feature_raw_norm_sum", 0.0)) / n_diag,
                "latent_clipped_norm_mean": float(diag.get("feature_clipped_norm_sum", 0.0)) / n_diag,
                "latent_noise_norm_mean": float(diag.get("feature_noise_norm_sum", 0.0)) / n_diag,
                "latent_clipping_fraction": float(diag.get("feature_clipped_sample_count", 0.0)) / n_diag,
            }
            rows.append(row)
            print(
                f"k={k:3d} d={transmitted_dim:4d} client={client_id:3d} "
                f"retained={retained_var:.4f} D_bn={row['D_bottleneck']:.6g} "
                f"D_clip={row['D_clip_given_bottleneck']:.6g} "
                f"D_noise={row['D_noise_given_clip']:.6g} D_total={row['D_total']:.6g} "
                f"cos={row['base_vs_dp_cosine']:.4f}",
                flush=True,
            )

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    summary_rows: list[dict[str, float | int]] = []
    for k in args.bottleneck_channels:
        subset = [r for r in rows if int(r["bottleneck_channels"]) == int(k)]
        summary_rows.append({
            "bottleneck_channels": k,
            "transmitted_dimension": int(subset[0]["transmitted_dimension"]),
            "pca_retained_variance_ratio": float(subset[0]["pca_retained_variance_ratio"]),
            "D_bottleneck_mean": float(np.mean([float(r["D_bottleneck"]) for r in subset])),
            "D_clip_given_bottleneck_mean": float(np.mean([float(r["D_clip_given_bottleneck"]) for r in subset])),
            "D_noise_given_clip_mean": float(np.mean([float(r["D_noise_given_clip"]) for r in subset])),
            "D_total_mean": float(np.mean([float(r["D_total"]) for r in subset])),
            "total_to_clean_norm_sq_ratio_mean": float(np.mean([float(r["total_to_clean_norm_sq_ratio"]) for r in subset])),
            "base_vs_dp_cosine_mean": float(np.mean([float(r["base_vs_dp_cosine"]) for r in subset])),
            "latent_raw_norm_mean": float(np.mean([float(r["latent_raw_norm_mean"]) for r in subset])),
            "latent_noise_norm_mean": float(np.mean([float(r["latent_noise_norm_mean"]) for r in subset])),
            "latent_clipping_fraction_mean": float(np.mean([float(r["latent_clipping_fraction"]) for r in subset])),
        })

    summary = {
        "clients": len(args.client_ids),
        "source_feature_shape": list(feature_shape),
        "source_feature_dimension": int(channels * height * width),
        "feature_clip_norm": float(args.feature_clip_norm),
        "feature_noise_multiplier": sigma_f,
        "feature_horizon_events": horizon_events,
        "pca_calibration_samples": min(len(x_train), int(args.pca_calibration_samples)),
        "by_bottleneck": summary_rows,
    }
    summary_path = out.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote {out}")
    print(f"wrote {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
