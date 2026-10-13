#!/usr/bin/env python3
"""Safely add only per-sample DP-SGD diagnostic logging to DynFL.

Usage:
  python add_clipping_diagnostics_safe.py             # dry run / preview
  python add_clipping_diagnostics_safe.py --apply     # write only if safe

Does not alter clipping, Gaussian noise, accounting, selection or training.
"""
from __future__ import annotations

import argparse
import difflib
from pathlib import Path
import shutil
import sys

RELATIVE = Path('dynfed/fmnist_lenet5_dynamic.py')
ANCHOR = '                result = worker_results[row["client_id"]]'
GUARD = '            if row.get("round") == round_idx and row.get("client_id") in worker_results:'
FOLLOW = '                for field in ('
MARKER = 'row["sample_dp_clipping_fraction"]'
INSERT = '''                # DP-SGD diagnostic logging only; do not modify private gradients.
                sample_count_raw = result.get("sample_dp_sample_count")
                clipped_count_raw = result.get("sample_dp_clipped_sample_count")
                sample_count = int(sample_count_raw) if sample_count_raw is not None else None
                clipped_count = int(clipped_count_raw) if clipped_count_raw is not None else None
                row["sample_dp_sample_count"] = sample_count
                row["sample_dp_clipped_sample_count"] = clipped_count
                row["sample_dp_clipping_fraction"] = (
                    clipped_count / sample_count
                    if sample_count is not None and clipped_count is not None and sample_count > 0
                    else None
                )
                row["sample_dp_max_raw_grad_norm"] = result.get("sample_dp_max_raw_grad_norm")
                row["sample_dp_max_clipped_grad_norm"] = result.get("sample_dp_max_clipped_grad_norm")
'''


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true', help='Modify the file after validation')
    args = parser.parse_args()
    target = Path.cwd() / RELATIVE
    if not target.is_file():
        print(f'ERROR: not found: {target}', file=sys.stderr)
        return 2
    original = target.read_bytes()
    try:
        text = original.decode('utf-8-sig')
    except UnicodeDecodeError:
        print('ERROR: source is not UTF-8; refusing to edit.', file=sys.stderr)
        return 2
    if MARKER in text:
        print('No change: DP-SGD clipping logging already exists.')
        return 0
    if text.count(ANCHOR) != 1:
        print(f'ERROR: expected exactly one worker-result anchor; found {text.count(ANCHOR)}. No edit.', file=sys.stderr)
        return 2
    lines = text.splitlines(keepends=True)
    hit = next(i for i, line in enumerate(lines) if line.strip() == ANCHOR.strip())
    if hit == 0 or lines[hit - 1].strip() != GUARD.strip():
        print('ERROR: anchor guard differs; no edit.', file=sys.stderr)
        return 2
    if hit + 1 >= len(lines) or lines[hit + 1].strip() != FOLLOW.strip():
        print('ERROR: expected for-field logging immediately after worker result; no edit.', file=sys.stderr)
        return 2
    # Check the actual worker-return/data flow so blanks cannot silently masquerade as successful logging.
    if '**training_diagnostics' not in text or 'training_diagnostics=training_diagnostics' not in text:
        print('ERROR: worker diagnostics forwarding was not found; no edit.', file=sys.stderr)
        return 2
    # Detect newline convention before inserting; retain original bytes and BOM.
    newline = '\r\n' if b'\r\n' in original else '\n'
    additions = [line + newline for line in INSERT.rstrip('\n').split('\n')]
    modified_lines = lines[:hit + 1] + additions + lines[hit + 1:]
    updated = ''.join(modified_lines)
    try:
        compile(updated, str(target), 'exec')
    except SyntaxError as exc:
        print(f'ERROR: modified source failed Python syntax check: {exc}', file=sys.stderr)
        return 2
    print(f'Target: {target}')
    print('Change: add five DP-SGD diagnostic columns (no training/DP computation changed).')
    print(''.join(difflib.unified_diff(text.splitlines(keepends=True), updated.splitlines(keepends=True),
                                       fromfile='before', tofile='after', n=3)))
    if not args.apply:
        print('\nDRY RUN ONLY. To apply, rerun with --apply.')
        return 0
    backup = target.with_name(target.name + '.before_clip_log.bak')
    if backup.exists():
        print(f'ERROR: backup already exists: {backup}; refusing to overwrite.', file=sys.stderr)
        return 2
    shutil.copy2(target, backup)
    encoded = updated.encode('utf-8-sig' if original.startswith(b'\xef\xbb\xbf') else 'utf-8')
    target.write_bytes(encoded)
    print(f'Applied successfully. Backup: {backup}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
