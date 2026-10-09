"""Independent Split-DP-v2 *diagnostic*, not an integrated seven-mode trainer.

Run from anywhere with --project-root pointing at the original DynFL repo.
No files under --project-root are modified.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mechanism import (OneReleaseCache, clip_rows, multiplier_for_zcdp_epsilon,
                       public_fixed_projection, zcdp_epsilon)


def parse_args():
    p = argparse.ArgumentParser(description='Split-DP-v2 frozen embedding diagnostic (NOT full DP training)')
    p.add_argument('--project-root', type=Path, required=True)
    p.add_argument('--data-root', type=Path, default=None)
    p.add_argument('--train-limit', type=int, default=1200)
    p.add_argument('--test-limit', type=int, default=400)
    p.add_argument('--dims', type=int, nargs='+', default=[64], choices=[32, 64, 128])
    p.add_argument('--epsilon-feature-total', type=float, default=4.0)
    p.add_argument('--delta-feature', type=float, default=1e-6)
    p.add_argument('--clip-norm', type=float, default=0.25)
    p.add_argument('--epochs', type=int, default=12)
    p.add_argument('--lr', type=float, default=0.02)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--seed', type=int, default=40)
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    p.add_argument('--output', type=Path, default=Path('out/split_dp_v2_probe.csv'))
    return p.parse_args()


@torch.no_grad()
def extract_features(end, x: np.ndarray, *, input_shape: tuple[int, ...], batch_size: int, device: torch.device):
    """Frozen End layer2 -> spatial mean pooling -> public 128-d vector."""
    features = []
    for start in range(0, len(x), batch_size):
        batch = torch.from_numpy(x[start:start + batch_size]).float().reshape(-1, *input_shape).to(device)
        spatial = end(batch)
        if spatial.ndim != 4 or spatial.shape[1] != 128:
            raise RuntimeError(f'Expected pretrained ResNet18 End output [B,128,H,W], got {tuple(spatial.shape)}')
        pooled = F.adaptive_avg_pool2d(spatial, output_size=1).flatten(1)
        features.append(pooled.float().cpu())
    return torch.cat(features, dim=0)


def train_and_eval(train_x, train_y, test_x, test_y, *, epochs: int, lr: float, seed: int):
    """Local-only classifier diagnostic; labels/metrics are not DP-protected."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = nn.Linear(train_x.shape[1], 10).cpu()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    gen = torch.Generator().manual_seed(seed + 23)
    for _ in range(epochs):
        perm = torch.randperm(len(train_y), generator=gen)
        for idx in perm.split(128):
            optimizer.zero_grad()
            loss = F.cross_entropy(model(train_x[idx]), train_y[idx])
            loss.backward()
            optimizer.step()
    with torch.no_grad():
        logits = model(test_x)
        return float((logits.argmax(1) == test_y).float().mean().item()), float(F.cross_entropy(logits, test_y))


def main():
    args = parse_args()
    if args.train_limit <= 0 or args.test_limit <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError('train-limit/test-limit/epochs/batch-size must be positive')
    if len(set(args.dims)) != len(args.dims):
        raise ValueError('duplicate dimensions not allowed')
    root = args.project_root.expanduser().resolve()
    if not (root / 'dynfed' / 'split_learning.py').is_file():
        raise FileNotFoundError(f'Not a DynFL project root: {root}')
    sys.path.insert(0, str(root))
    from dynfed.fmnist_lenet5_dynamic import load_image_dataset_arrays
    from dynfed.split_learning import build_split_pair

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA selected but not available')

    # This is a diagnostic budget across ALL dimensions in one invocation.
    # It does not account for any previous studies of the same private records.
    multiplier = multiplier_for_zcdp_epsilon(
        args.epsilon_feature_total, args.delta_feature, releases_per_record=len(args.dims)
    )
    eps_bound = zcdp_epsilon(multiplier, args.delta_feature, len(args.dims))
    data_root = args.data_root if args.data_root else root / 'data'
    x_tr, y_tr, x_te, y_te, shape, label, n_cls = load_image_dataset_arrays(
        'cifar10', Path(data_root), args.train_limit, args.test_limit, args.seed
    )
    if n_cls != 10:
        raise RuntimeError('Expected CIFAR10')
    end, _edge = build_split_pair('resnet18_pretrained_head', device, input_channels=shape[0], image_size=shape[1], num_classes=n_cls)
    end.eval()
    for param in end.parameters():
        param.requires_grad_(False)

    print('Extracting frozen layer2 features; torchvision pretrained weights must be available.', flush=True)
    feat_tr = extract_features(end, x_tr, input_shape=shape, batch_size=args.batch_size, device=device)
    feat_te = extract_features(end, x_te, input_shape=shape, batch_size=args.batch_size, device=device)
    y_tr = torch.from_numpy(np.asarray(y_tr)).long()
    y_te = torch.from_numpy(np.asarray(y_te)).long()
    rows = []
    for dim in args.dims:
        q = public_fixed_projection(128, dim, seed=1729 + dim)
        z_tr, z_te = feat_tr @ q, feat_te @ q
        c_tr, c_te = clip_rows(z_tr, args.clip_norm), clip_rows(z_te, args.clip_norm)

        cache = OneReleaseCache(args.clip_norm, multiplier, seed=args.seed + dim)
        context = ('frozen_resnet18_layer2_spatialmean', dim, 1729 + dim, args.clip_norm, multiplier)
        protected_tr = cache.release(context, [('train', i) for i in range(len(z_tr))], z_tr)
        protected_te = cache.release(context, [('test', i) for i in range(len(z_te))], z_te)
        # Explicit replay proof: no new release, same bytes, arbitrary fresh raw inputs ignored.
        first_count = cache.stats.new_releases
        replay = cache.release(context, [('train', i) for i in range(min(16, len(z_tr)))], z_tr[:16])
        if not torch.equal(replay, protected_tr[:len(replay)]) or cache.stats.new_releases != first_count:
            raise RuntimeError('DP cache replay failed')

        results = {}
        for j, (name, tr, te) in enumerate((('clean', z_tr, z_te), ('clip_only', c_tr, c_te), ('clip_plus_dp', protected_tr, protected_te))):
            acc, loss = train_and_eval(tr, y_tr, te, y_te, epochs=args.epochs, lr=args.lr, seed=args.seed)
            results[name] = (acc, loss)
        row = {
            'dimension': dim,
            'clean_acc': results['clean'][0],
            'clip_only_acc': results['clip_only'][0],
            'clip_plus_dp_acc': results['clip_plus_dp'][0],
            'clean_loss': results['clean'][1],
            'clip_only_loss': results['clip_only'][1],
            'clip_plus_dp_loss': results['clip_plus_dp'][1],
            'raw_feature_norm_mean': float(torch.linalg.vector_norm(z_tr, dim=1).mean()),
            'clipped_feature_norm_mean': float(torch.linalg.vector_norm(c_tr, dim=1).mean()),
            'noise_norm_mean': float(torch.linalg.vector_norm(protected_tr - c_tr, dim=1).mean()),
            'clip_fraction': float((torch.linalg.vector_norm(z_tr, dim=1) > args.clip_norm).float().mean()),
            'noise_multiplier': multiplier,
            'per_coordinate_noise_std': 2 * args.clip_norm * multiplier,
            'feature_epsilon_zcdp_bound_across_dims': eps_bound,
            'feature_delta_across_dims': args.delta_feature,
            'cached_records': cache.stats.cached_records,
            'unique_embedding_releases': cache.stats.new_releases,
            'replay_hits': cache.stats.cache_hits,
            'status': 'DIAGNOSTIC_ONLY_NOT_END_TO_END_DP',
        }
        rows.append(row)
        print(f'D={dim:3d} clean={100*row["clean_acc"]:5.1f}% clip={100*row["clip_only_acc"]:5.1f}% '
              f'DP={100*row["clip_plus_dp_acc"]:5.1f}% noise/signal≈'
              f'{row["noise_norm_mean"]/max(1e-12,row["clipped_feature_norm_mean"]):.1f} '
              f'cache={row["cached_records"]} eps_feature_upper≈{eps_bound:.3f}', flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    meta = {
        'status': 'DIAGNOSTIC_ONLY_NOT_END_TO_END_DP',
        'notes': [
            'No modifications to original project.',
            'DP guarantee NOT established for end-to-end training: private labels, model updates and output metrics are not privatized.',
            'zCDP feature-only epsilon accounts for dimensions released in this single run, no subsampling amplification.',
            'Other independent runs using the same private records must be privacy-composed separately.',
            'PRNG is torch.Generator seeded for reproducibility, not a deployment-safe cryptographic RNG.',
            'Spatial pooling + public fixed projection REPLACES original ResNet edge layer3/layer4 with a probe linear head.',
        ],
        'params': {'dims': args.dims, 'epsilon_feature_total': args.epsilon_feature_total,
                   'delta_feature': args.delta_feature, 'clip_norm': args.clip_norm,
                   'multiplier': multiplier, 'seed': args.seed,
                   'dataset': label, 'train_count': len(y_tr), 'test_count': len(y_te)}
    }
    args.output.with_suffix('.json').write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding='utf-8')
    print(f'Results: {args.output}\nWARNING: diagnostic only; NO full-system DP claim.', flush=True)


if __name__ == '__main__':
    main()
