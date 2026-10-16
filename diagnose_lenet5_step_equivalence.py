#!/usr/bin/env python3
"""LeNet5 centralized matched-update and fixed-batch training sanity check.

No FL / no DP / no HE. Diagnostics, not comparable private baselines.
Uses actual Fashion-MNIST loader and project FullNet, with same SGD and batch clipping.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import torch
from torch.nn import functional as F
from dynfed.fmnist_lenet5_dynamic import load_image_dataset_arrays
from dynfed.split_learning import build_full_model


def evaluate(model, x, y, shape, device, batch=256):
    model.eval()
    loss_sum = 0.0
    correct = 0
    with torch.no_grad():
        for i in range(0, len(y), batch):
            xx = torch.from_numpy(x[i:i+batch]).view(-1, *shape).to(device)
            yy = torch.from_numpy(y[i:i+batch]).to(device)
            prediction = model(xx)
            loss_sum += F.cross_entropy(prediction, yy, reduction='sum').item()
            correct += int((prediction.argmax(dim=1) == yy).sum().item())
    return loss_sum / len(y), correct / len(y)


def train_probe(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    raw = ROOT / 'experiments' / 'data' / 'fmnist' / 'FashionMNIST' / 'raw'
    x, y, xt, yt, shape, dataset, nclasses = load_image_dataset_arrays(
        'fmnist', raw, args.train_limit, args.test_limit, args.seed)
    assert set(np.unique(y)).issubset(set(range(nclasses)))
    print(f'LeNet-5 step-matched central sanity | {dataset} | lr={args.lr:.3g} | batch=64')
    print(f'train={len(y)} test={len(yt)}  x_range=({x.min():.3g}, {x.max():.3g})')
    print(f'train_labels={np.bincount(y,minlength=nclasses).tolist()}')
    print('NO FL  |  NO DP  |  NO HE  |  SGD momentum=0, grad_clip=5')
    initial = build_full_model('lenet5', device, input_channels=shape[0], image_size=shape[1], num_classes=nclasses)
    base = {k: v.detach().clone() for k, v in initial.state_dict().items()}
    # Two separate probes start from EXACTLY identical initial weights.
    for probe in ['shuffled', 'fixed_batch']:
        model = build_full_model('lenet5', device, input_channels=shape[0], image_size=shape[1], num_classes=nclasses)
        model.load_state_dict(base)
        opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0., weight_decay=0.0001)
        rng = np.random.default_rng(args.seed)
        perm = rng.permutation(len(y))
        pointer = 0
        if probe == 'fixed_batch':
            fixed_ids = perm[:64]
        print(f'\n[PROBE] {probe}  (fixed_batch measures ability to memorize one minibatch)')
        print('Step    TrainLoss  TestLoss     TestAcc   BatchAcc   GradientNorm')
        checkpoints = {0, 5, 30, 60, args.steps}
        first_batch_acc = None
        def report(step, batch_acc=float('nan'), grad_norm=float('nan')):
            eloss, acc = evaluate(model, xt, yt, shape, device)
            train_x = torch.from_numpy(x[:min(256,len(x))]).view(-1, *shape).to(device)
            train_y = torch.from_numpy(y[:min(256,len(y))]).to(device)
            model.eval()
            with torch.no_grad():
                tr_loss = F.cross_entropy(model(train_x), train_y).item()
            print(f'{step:>4d}    {tr_loss:>9.4f}  {eloss:>8.4f}    {acc:>7.2%}   {batch_acc:>8.2%}   {grad_norm:>12.4f}',flush=True)
        report(0)
        for step in range(1, args.steps + 1):
            if probe == 'fixed_batch':
                ids = fixed_ids
            else:
                if pointer + 64 > len(perm):
                    perm = rng.permutation(len(y))
                    pointer = 0
                ids = perm[pointer:pointer+64]
                pointer += 64
            bx = torch.from_numpy(x[ids]).view(-1, *shape).to(device)
            by = torch.from_numpy(y[ids]).to(device)
            model.train()
            opt.zero_grad(set_to_none=True)
            logits = model(bx)
            loss = F.cross_entropy(logits, by)
            batch_acc = (logits.argmax(dim=1) == by).float().mean().item()
            loss.backward()
            grad_norm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.).item())
            if not np.isfinite(grad_norm):
                raise RuntimeError(f'Nonfinite gradient norm at step {step}')
            opt.step()
            if step in checkpoints:
                report(step, batch_acc, grad_norm)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--steps', type=int, default=100)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--train-limit', type=int, default=24000)
    p.add_argument('--test-limit', type=int, default=2000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=['cpu','cuda'], default='cuda' if torch.cuda.is_available() else 'cpu')
    args = p.parse_args()
    if args.steps < 1 or args.lr <= 0: p.error('steps and lr must be positive')
    train_probe(args)

if __name__ == '__main__':
    main()
