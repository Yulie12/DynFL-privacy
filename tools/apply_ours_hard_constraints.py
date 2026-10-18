"""Fail-closed Ours candidate-selection patch (audited against 2026-10-15 snapshot).

Run from project root. Exact-match guards refuse unknown source versions.
Creates timestamped backup before replacement; no training results touched.
"""
from __future__ import annotations
import argparse
import datetime
from pathlib import Path


def replace_exact(src: str, old: str, new: str, expected: int, label: str) -> str:
    n = src.count(old)
    if n == 0 and new in src:
        print(f'[ALREADY PATCHED] {label}')
        return src
    if n != expected:
        raise RuntimeError(f'Cannot safely patch {label}: expected {expected} matches, got {n}. No changes written.')
    return src.replace(old, new)


def patch_text(src: str) -> str:
    src = replace_exact(src,
        '            and (not self.risk_hard_gate or self.feasible_risk)\n',
        '            and (not self.risk_hard_gate or self.feasible_risk)\n'
        '            and self.feasible_time\n'
        '            and self.feasible_edge\n'
        '            and self.feasible_cloud\n',
        1, 'candidate hard constraints (time / edge / cloud)')
    src = replace_exact(src,
        '        risk_hard_gate=False,  # heuristic score is diagnostic only\n',
        '        risk_hard_gate=True,  # policy requirement: risk is HARD; infeasible -> SKIP\n',
        1, 'risk hard gate')
    src = replace_exact(src,
        '        if not pool and not config.require_feasible and not config.strict_pair_admission:\n',
        '        if not pool and not (\n'
        '            config.require_feasible or config.strict_pair_admission\n'
        '            or any(item.risk_hard_gate for item in candidates)\n'
        '        ):\n',
        2, 'global profile fallback')
    src = replace_exact(src,
        '            pool = [current if (current.feasible if config.strict_pair_admission else current.feasible_device) else skipped_candidate()]\n',
        '            pool = [current if current.feasible else skipped_candidate()]\n',
        1, 'global profile last resort')
    return src


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--project-root', type=Path, default=Path('.'))
    ap.add_argument('--dry-run', action='store_true')
    args = ap.parse_args()
    file = args.project_root / 'dynfed' / 'selection.py'
    old = file.read_text(encoding='utf-8')
    new = patch_text(old)
    if new == old:
        print('[OK] Already installed. No change.')
        return
    compile(new, str(file), 'exec')
    if args.dry_run:
        print('[OK] Patch supported by source; syntax check passed. No files modified.')
        return
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup = file.with_name(f'selection.py.before_hard_gate_{stamp}.bak')
    backup.write_text(old, encoding='utf-8')
    file.write_text(new, encoding='utf-8')
    print('[OK] Patch applied:', file)
    print('[BACKUP]', backup)
    print('[NOTE] A candidate violating any hard feasibility flag is inadmissible; no admissible candidate -> SKIP.')
    print('[NOTE] Keep previous results as historical; rerun Ours to collect new logical timing.')

if __name__ == '__main__':
    main()
