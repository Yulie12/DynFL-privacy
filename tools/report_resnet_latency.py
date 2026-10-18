"""Read measured wall time vs modeled DynFL logical time; never invent missing logical data."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path


def find_csv(location: Path) -> Path:
    if location.is_file():
        if location.name != 'round_metrics.csv':
            raise ValueError('Expected round_metrics.csv or a result directory')
        return location
    matches = list(location.rglob('round_metrics.csv'))
    if not matches:
        raise FileNotFoundError('No round_metrics.csv under ' + str(location))
    return max(matches, key=lambda p: p.stat().st_mtime)


def number(row, key):
    x = row.get(key, '').strip()
    try: return float(x) if x else None
    except (ValueError, TypeError): return None


def readable(x, precision=2):
    return '-' if x is None else f'{x:.{precision}f}'


def report(p: Path):
    file = find_csv(p)
    with file.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError('Empty metrics: ' + str(file))
    logical = 'logical_time' in rows[0] and 'round_duration' in rows[0]
    print('Results:', file)
    print('Type:', 'DynFL modeled flow / Ours-capable mainline' if logical else 'standalone ResNet LIIC wall-time ONLY')
    print('Round   Acc%   ModeledRound(s)   ModeledTotal(s)  WallRound(s)  SelectWall(s)   Steps')
    wall_total = 0.0
    for i, r in enumerate(rows):
        acc = number(r, 'test_accuracy')
        if acc is not None: acc *= 100
        wall = number(r, 'round_wall_time_sec')
        if wall is None: wall = number(r, 'seconds')
        if wall is not None: wall_total += wall
        rd = number(r, 'round_duration')
        cum = number(r, 'logical_time')
        sw = number(r, 'selection_wall_time_sec')
        steps = r.get('actual_optimizer_steps', r.get('optimizer_steps', '-'))
        number_r = r.get('round', str(i + 1))
        print(f'{number_r:>5} {readable(acc):>7} {readable(rd):>17} {readable(cum):>17} '
              f'{readable(wall):>13} {readable(sw):>14} {str(steps):>8}')
    print('Observed wall total, training-round columns only:', readable(wall_total), 's')
    if logical:
        last = rows[-1]
        print('Modeled logical total:', readable(number(last, 'logical_time')), 's')
        print('Accounted system total (model + measured selection):',
              readable(number(last, 'accounted_system_time_sec')), 's')
        print('CAUTION: modeled logical time is NOT GPU runtime; selection wall is a separate measured overhead.')
    else:
        print('LOGICAL TIME NOT RECORDED. Existing LIIC probe did not run a modeled cloud-edge-end flow.')
        print('NOTE: earlier feature-cache preparation is outside these per-round wall timings.')
    if file.parent.name in ('full_dynfl', 'ours'):
        print('Mainline Ours policy folder:', file.parent.name)
    conf = file.parent.parent / 'config.json'
    if conf.exists():
        try:
            data = json.loads(conf.read_text(encoding='utf-8'))
            print('Experiment config:', conf)
        except Exception:
            pass


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description='Compare measured per-round time and DynFL modeled logical time')
    ap.add_argument('path', type=Path)
    a = ap.parse_args()
    report(a.path)
