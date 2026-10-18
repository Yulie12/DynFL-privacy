"""Readable, dependency-free view of DynFL CSV audits.

Usage:
  python tools/summarize_run.py out/fmnist_lenet5_period5_speed_probe_seed42
  python tools/summarize_run.py path/to/timestamped-run
"""
from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path


def resolve_metric_file(root: Path) -> Path:
    if root.is_file() and root.name == 'round_metrics.csv':
        return root
    files = list(root.rglob('round_metrics.csv'))
    files = [file for file in files if file.parent.name in ('full_dynfl', 'ours')]
    if not files:
        raise FileNotFoundError(f'No full_dynfl/round_metrics.csv under: {root}')
    return max(files, key=lambda file: file.stat().st_mtime)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open('r', encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def safe_float(value: str | None) -> float | None:
    try:
        return float(value) if value not in ('', None) else None
    except (TypeError, ValueError):
        return None


def brief(value: str | None, digits: int = 4) -> str:
    x = safe_float(value)
    return f'{x:.{digits}f}' if x is not None else '-'


def build_report(metric_file: Path) -> str:
    rows = read_rows(metric_file)
    decision_file = metric_file.with_name('client_decisions.csv')
    by_round: dict[str, collections.Counter[str]] = collections.defaultdict(collections.Counter)
    if decision_file.is_file():
        for row in read_rows(decision_file):
            by_round[row.get('round', '')][row.get('mode', '?')] += 1
    result = [f'Run: {metric_file.parent.parent}',
              'round   acc       train_loss test_loss  steps  skip modes',
              '-' * 86]
    for row in rows:
        r = row.get('round', '?')
        c = by_round.get(r, {})
        modes = ','.join(f'{mode}:{count}' for mode, count in c.most_common()) or '-'
        result.append(
            f'{r:>5}   {brief(row.get("test_accuracy")):>7}   '
            f'{brief(row.get("train_loss")):>10} {brief(row.get("test_loss")):>9}  '
            f'{row.get("actual_optimizer_steps", "-"):>5}  {row.get("skipped_clients", "-"):>4} {modes}'
        )
    if rows:
        final = rows[-1]
        result.extend(['-' * 86,
            'Final: acc=' + brief(final.get('test_accuracy')) +
            ' | rounds=' + str(len(rows)) +
            ' | data: ' + str(metric_file)])
    else:
        result.append('No metric rows yet.')
    return '\n'.join(result)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path, help='Experiment output root or single run directory')
    parser.add_argument('--save', type=Path, help='Optionally write the displayed report to a text file')
    args = parser.parse_args()
    result = build_report(resolve_metric_file(args.root))
    print(result)
    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(result + '\n', encoding='utf-8')
        print(f'Saved: {args.save}')


if __name__ == '__main__':
    main()
