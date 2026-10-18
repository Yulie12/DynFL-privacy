#!/usr/bin/env python3
"""Read-only DynFL training-health summary across all policy output folders."""
import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean


def csv_rows(path):
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def num(value):
    try:
        return float(value)
    except (ValueError, TypeError):
        return None


def fmt(value, digits=4):
    return '-' if value is None else f'{value:.{digits}f}'


def latest_run(root):
    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f'Output directory does not exist: {root}')
    if (root / 'round_metrics.csv').is_file() and (root / 'client_decisions.csv').is_file():
        return root
    candidates = []
    for metrics in root.rglob('round_metrics.csv'):
        if (metrics.parent / 'client_decisions.csv').is_file():
            candidates.append(metrics.parent)
    if candidates:
        return max(candidates, key=lambda folder: (folder / 'round_metrics.csv').stat().st_mtime)
    raise FileNotFoundError(
        f'Cannot find a folder containing both round_metrics.csv and client_decisions.csv under {root}. '
        f'Check that training finished writing CSV outputs.'
    )


def stats(group, field):
    values = [num(row.get(field)) for row in group]
    values = [value for value in values if value is not None]
    return mean(values) if values else None


def main():
    parser = argparse.ArgumentParser(description='Read-only training health summary for DynFL')
    parser.add_argument('path', type=Path, help='Root, run, or policy folder')
    args = parser.parse_args()
    folder = latest_run(args.path)
    metrics = csv_rows(folder / 'round_metrics.csv')
    decisions = csv_rows(folder / 'client_decisions.csv')
    byround = defaultdict(list)
    for row in decisions:
        byround[str(row.get('round', ''))].append(row)
    print(f'Run: {folder.parent}')
    print(f'Policy folder: {folder.name}')
    print('Round  Acc      TrainLoss  TestLoss   Steps  Skip   Mode       ClipFrac  OptSigma  UpdateNorm')
    print('-' * 103)
    for met in metrics:
        rnd = str(met.get('round', ''))
        items = byround.get(rnd, [])
        counts = Counter(row.get('mode', '?') for row in items)
        modes = ','.join(f'{key}:{val}' for key, val in counts.most_common(2)) or '-'
        sigma = stats(items, 'selected_sample_optimizer_noise_multiplier')
        if sigma is None:
            sigma = stats(items, 'accounted_sample_optimizer_noise_multiplier')
        frac = stats(items, 'sample_dp_clipping_fraction')
        norm = num(met.get('applied_cloud_update_norm'))
        if norm is None:
            norm = num(met.get('global_update_norm'))
        try:
            round_display = str(int(rnd) + 1)
        except ValueError:
            round_display = rnd or '?'
        print(f'{round_display:>5}  {fmt(num(met.get("test_accuracy")),4):>7}  '
              f'{fmt(num(met.get("train_loss")),4):>9}  '
              f'{fmt(num(met.get("test_loss")),4):>8}  '
              f'{met.get("actual_optimizer_steps", "-"):>6}  '
              f'{met.get("skipped_clients", "-"):>4}  '
              f'{modes:<11} {fmt(frac,3):>8}  {fmt(sigma,3):>8}  {fmt(norm,5):>10}')
    print('\nNotes: ClipFrac = unweighted client mean; - means not recorded.')
    print('OptSigma = per-sample optimizer noise multiplier if recorded.')
    print('UpdateNorm is the reported applied cloud/global update norm; it does not establish learning correctness.')


if __name__ == '__main__':
    main()
