"""Standalone fixed-LIIC CIFAR-10 ResNet18 Layer4+Head no-DP probe.

Real run uses the project's pretrained model and CIFAR preprocessing and fixed
edge-label-skew partitions. Frozen prefix features are cached *exactly once* in
float32, with no augmentation. BatchNorm is forced to eval() throughout. Only
Layer4 and classifier gradients are trained and averaged (FedAvg), keeping all
frozen pretrained weights and running buffers constant.

Diagnostic ONLY: no DP, no HE, no dynamic selection, and no privacy claims.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.fmnist_lenet5_dynamic import (  # noqa: E402
    _default_data_root, _partition_clients_lenet5, _split_client_indices,
    load_image_dataset_arrays,
)
from dynfed.split_learning import build_full_model, _make_optimizer  # noqa: E402

MODEL_NAME = 'resnet18_pretrained_layer4_head'


def frozen_prefix(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    # Must exactly reproduce TorchvisionResNetFull through layer3.
    x = model.conv1(x)
    x = model.bn1(x)
    x = model.relu(x)
    x = model.maxpool(x)
    x = model.layer1(x)
    x = model.layer2(x)
    return model.layer3(x)


def trainable_tail(model: nn.Module, features: torch.Tensor) -> torch.Tensor:
    x = model.layer4(features)
    x = model.avgpool(x)
    return model.classifier(torch.flatten(x, 1))


def freeze_bn(model: nn.Module) -> None:
    # Call after EVERY train() to prevent dataset-dependent BN buffer updates.
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for p in module.parameters(recurse=False):
                p.requires_grad_(False)


def prepare_model(device: torch.device, synthetic: bool) -> nn.Module:
    if synthetic:
        # Synthetic code/shape test only; do NOT use to claim pretrained results.
        from torchvision.models import resnet18
        with patch('dynfed.split_learning._make_torchvision_resnet',
                   side_effect=lambda *_args, **_kwargs: resnet18(weights=None)):
            model = build_full_model(MODEL_NAME, device, 3, 32, 10)
    else:
        try:
            model = build_full_model(MODEL_NAME, device, 3, 32, 10)
        except Exception as exc:
            raise RuntimeError(
                'Failed to load ImageNet-pretrained ResNet-18. Check torchvision '
                'weights cache/network; this experiment refuses silent random initialization.'
            ) from exc
    freeze_bn(model)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or any(not name.startswith(('layer4.', 'classifier.')) for name in trainable):
        raise AssertionError(f'Unexpected trainable parameter set: {trainable[:12]}')
    if not any(name.startswith('layer4.') for name in trainable):
        raise AssertionError('Layer4 not trainable')
    return model


@torch.no_grad()
def cache_prefix_features(model: nn.Module, images: np.ndarray, batch: int,
                          device: torch.device, label: str) -> torch.Tensor:
    model.eval()
    chunks: list[torch.Tensor] = []
    for pos in range(0, len(images), batch):
        if pos and (pos // batch) % 128 == 0:
            print(f'[CACHE] {label}: {pos}/{len(images)}', flush=True)
        x = torch.from_numpy(images[pos:pos + batch]).reshape(-1, 3, 32, 32).to(device)
        result = frozen_prefix(model, x)
        chunks.append(result.float().cpu().contiguous())
    result = torch.cat(chunks, dim=0)
    mb = result.numel() * result.element_size() / 1024 ** 2
    print(f'[CACHE] {label}: {tuple(result.shape)}  fp32={mb:.1f} MiB', flush=True)
    return result


def synthetic_inputs(seed: int, clients: int, train_limit: int, test_limit: int):
    rng = np.random.default_rng(seed)
    # Precomputed synthetic *features* are used directly (no pretrained claims).
    n_train, n_test = min(train_limit, clients * 8), min(test_limit, 12)
    ft = torch.from_numpy(rng.normal(0, .15, (n_train, 256, 2, 2)).astype('float32'))
    fe = torch.from_numpy(rng.normal(0, .15, (n_test, 256, 2, 2)).astype('float32'))
    yt = np.asarray([i % 10 for i in range(n_train)], dtype=np.int64)
    ye = np.asarray([i % 10 for i in range(n_test)], dtype=np.int64)
    indices = [np.asarray(v, dtype=np.int64) for v in np.array_split(np.arange(n_train), clients)]
    return ft, yt, fe, ye, indices


def check_forward_equivalence(model: nn.Module, x: np.ndarray,
                              device: torch.device) -> float:
    model.eval()
    with torch.inference_mode():
        sample = torch.from_numpy(x[:2]).reshape(-1, 3, 32, 32).to(device)
        reference = model(sample)
        actual = trainable_tail(model, frozen_prefix(model, sample))
    difference = float((reference - actual).abs().max().item())
    if difference > 1e-5:
        raise AssertionError(f'Frozen-prefix split does not match model: max error {difference}')
    return difference


@torch.no_grad()
def evaluate(model: nn.Module, features: torch.Tensor, labels: np.ndarray,
             batch: int, device: torch.device) -> tuple[float, float]:
    model.eval()
    freeze_bn(model)
    n_ok, n_total, total_loss = 0, 0, 0.0
    for i in range(0, len(labels), batch):
        x = features[i:i + batch].to(device)
        target = torch.as_tensor(labels[i:i + batch], dtype=torch.long, device=device)
        predictions = trainable_tail(model, x)
        total_loss += float(F.cross_entropy(predictions, target, reduction='sum'))
        n_ok += int((predictions.argmax(dim=1) == target).sum().item())
        n_total += int(target.numel())
    return n_ok / max(1, n_total), total_loss / max(1, n_total)


def run(args: argparse.Namespace) -> Path:
    if not 0 < args.rounds <= 101 or args.epochs < 1 or args.batch < 1:
        raise ValueError('Require 1<=rounds<=101, positive epochs and physical batch')
    if args.clients <= 0 or args.edges <= 0 or args.train_limit <= 0 or args.test_limit <= 0:
        raise ValueError('Invalid data/partition size')
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA not available; use --device cpu only for tiny smoke tests')
    device = torch.device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.benchmark = False
    model = prepare_model(device, args.synthetic_smoke)
    trainable = {k: p for k, p in model.named_parameters() if p.requires_grad}
    pcount = sum(p.numel() for p in trainable.values())
    print(f'[MODEL] {MODEL_NAME}; trainable={pcount:,}; total={sum(p.numel() for p in model.parameters()):,}')
    print('[MODE] Fixed LIIC, no DP, no HE, no 5-step cap; BN frozen; 3 epochs by default')
    if args.synthetic_smoke:
        train_features, y_train, test_features, y_test, partitions = synthetic_inputs(
            args.seed, args.clients, args.train_limit, args.test_limit)
        print('[WARNING] SYNTHETIC SMOKE; random weights/features; NOT CIFAR-10')
    else:
        x_train, y_train, x_test, y_test, shape, dataset_name, classes = load_image_dataset_arrays(
            'cifar10', Path(_default_data_root('cifar10')),
            args.train_limit, args.test_limit, args.seed)
        if shape != (3, 32, 32) or classes != 10:
            raise RuntimeError('Unexpected dataset shape')
        partitions = _partition_clients_lenet5(
            y_train, args.clients, args.edges, False, 'edge_label_skew', args.seed, .5)
        partitions, _held_out = _split_client_indices(partitions, test_ratio=.2, seed=args.seed)
        print(f'[DATA] {dataset_name}; train={len(y_train)}; held-out evaluation={len(y_test)}')
        err = check_forward_equivalence(model, x_train, device)
        print(f'[CHECK] Frozen prefix + Layer4 + Head equals full eval model (max abs {err:.2e})')
        train_features = cache_prefix_features(model, x_train, args.cache_batch, device, 'train')
        test_features = cache_prefix_features(model, x_test, args.cache_batch, device, 'test')
        del x_train, x_test
    active = [(cid, ix) for cid, ix in enumerate(partitions) if len(ix) > 0]
    if not active:
        raise RuntimeError('No active clients')
    total_examples = sum(len(ix) for _, ix in active)
    print(f'[TRAIN] clients={len(active)}/{args.clients}; physical batch={args.batch}; '
          f'epochs={args.epochs}; lr={args.lr:g}; rounds={args.rounds}; samples/round={total_examples}')
    path = args.output_root / ('synthetic_smoke' if args.synthetic_smoke else 'cifar10_layer4_head_nodp') / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    path.mkdir(parents=True, exist_ok=False)
    meta = {
        'model': MODEL_NAME, 'pretrained': not args.synthetic_smoke,
        'dataset': 'synthetic_smoke' if args.synthetic_smoke else 'CIFAR-10',
        'privacy': 'NONE (non-DP baseline)', 'he': 'none', 'selection': 'fixed_LIIC',
        'clients': args.clients, 'active_clients': len(active), 'edges': args.edges,
        'train_limit': args.train_limit, 'test_limit': args.test_limit,
        'local_test_holdout_ratio': .2, 'epochs': args.epochs, 'physical_batch': args.batch,
        'step_cap': None, 'lr': args.lr, 'weight_decay': args.weight_decay,
        'rounds_run': args.rounds, 'seed': args.seed, 'trainable_params': pcount,
        'BN': 'running stats/affine fixed; all BN modules eval even inside local train',
        'feature_cache': 'frozen stem->layer3 in float32 on CPU; no data augmentation',
        'computation': 'FedAvg of trainable layer4+classifier only; momentum reset per client',
        'caution': 'standalone fixed LIIC; not dynamic DynFL mainline; not a DP claim',
    }
    (path / 'config.json').write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding='utf-8')
    global_state = {k: p.detach().cpu().clone() for k, p in trainable.items()}
    metric_file = path / 'round_metrics.csv'
    columns = ['round', 'test_accuracy', 'test_loss', 'optimizer_steps', 'skipped_clients', 'seconds']
    with metric_file.open('w', newline='', encoding='utf-8') as fp:
        writer = csv.DictWriter(fp, fieldnames=columns)
        writer.writeheader()
        for rnd in range(1, args.rounds + 1):
            start = time.perf_counter()
            accum = {k: torch.zeros_like(v) for k, v in global_state.items()}
            steps, skips = 0, 0
            for client_id, idx in active:
                if len(idx) == 0:
                    skips += 1
                    continue
                with torch.no_grad():
                    for name, param in trainable.items():
                        param.copy_(global_state[name].to(device=device))
                # Only Layer4 and Head are trained. Frozen prefix and ALL BN stats stay fixed.
                model.train()
                freeze_bn(model)
                if any(m.training for m in model.modules() if isinstance(m, nn.modules.batchnorm._BatchNorm)):
                    raise AssertionError('BatchNorm accidentally switched to training mode')
                optimizer = _make_optimizer(model, args.lr, MODEL_NAME, weight_decay=args.weight_decay)
                if optimizer is None:
                    raise RuntimeError('No optimizer')
                indices = np.asarray(idx, dtype=np.int64)
                generator = torch.Generator().manual_seed(args.seed * 100003 + rnd * 1009 + client_id)
                for _ep in range(args.epochs):
                    perm = torch.randperm(len(indices), generator=generator).numpy()
                    for offset in range(0, len(indices), args.batch):
                        subset = indices[perm[offset:offset + args.batch]]
                        features = train_features[subset].to(device)
                        targets = torch.as_tensor(y_train[subset], dtype=torch.long, device=device)
                        optimizer.zero_grad(set_to_none=True)
                        predictions = trainable_tail(model, features)
                        loss = F.cross_entropy(predictions, targets)
                        if not torch.isfinite(loss):
                            raise FloatingPointError(f'Loss nonfinite at client={client_id}, round={rnd}')
                        loss.backward()
                        if args.clip_batch_grad > 0:
                            torch.nn.utils.clip_grad_norm_(trainable.values(), args.clip_batch_grad)
                        optimizer.step()
                        steps += 1
                scale = float(len(idx)) / total_examples
                with torch.no_grad():
                    for name, param in trainable.items():
                        accum[name].add_(param.detach().cpu() - global_state[name], alpha=scale)
            with torch.no_grad():
                for name in global_state:
                    global_state[name].add_(accum[name])
                for name, param in trainable.items():
                    param.copy_(global_state[name].to(device=device))
            accuracy, test_loss = evaluate(model, test_features, y_test, args.eval_batch, device)
            seconds = time.perf_counter() - start
            info = dict(round=rnd, test_accuracy=f'{accuracy:.6f}', test_loss=f'{test_loss:.6f}',
                        optimizer_steps=steps, skipped_clients=skips, seconds=f'{seconds:.2f}')
            writer.writerow(info)
            fp.flush()
            # Keeps the trained Layer4+classifier weights for later analysis.
            # The pretrained/frozen prefix is intentionally not duplicated.
            torch.save({'round': rnd, 'trainable_state': global_state, 'config': meta},
                       path / 'last_trainable_state.pt')
            print(f'{rnd:03d}/{args.rounds:03d} Acc={100*accuracy:6.2f}%  '
                  f'TestLoss={test_loss:8.5f}  Steps={steps:5d}  Skip={skips}  {seconds:.1f}s', flush=True)
    print(f'[DONE] {path.resolve()}')
    return path


def main():
    ap = argparse.ArgumentParser(description='Standalone pretrained CIFAR10 ResNet18 Layer4+Head no-DP LIIC probe')
    ap.add_argument('--rounds', type=int, default=2)
    ap.add_argument('--clients', type=int, default=100)
    ap.add_argument('--edges', type=int, default=10)
    ap.add_argument('--train-limit', type=int, default=24000)
    ap.add_argument('--test-limit', type=int, default=2000)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--batch', type=int, default=8, help='physical training batch; NOT DP batch')
    ap.add_argument('--cache-batch', type=int, default=64, help='frozen feature extraction batch')
    ap.add_argument('--eval-batch', type=int, default=128)
    ap.add_argument('--lr', type=float, default=.005)
    ap.add_argument('--weight-decay', type=float, default=5e-4)
    ap.add_argument('--clip-batch-grad', type=float, default=0.0,
                    help='Optional ordinary SGD gradient clipping; 0=no clipping')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    ap.add_argument('--output-root', type=Path, default=Path('out/resnet18_layer4_liic_probe'))
    ap.add_argument('--synthetic-smoke', action='store_true')
    run(ap.parse_args())


if __name__ == '__main__':
    main()
