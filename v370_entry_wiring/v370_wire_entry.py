#!/usr/bin/env python3
"""Idempotent V370 manifest wiring for the two actual paper-training entrypoints.

Usage: python v370_wire_entry.py --check | --apply
Only touches experiments/run_paper_config.py and experiments/run_fmnist_lenet5.py.
"""
from __future__ import annotations
import argparse
import ast
from pathlib import Path
import sys

PAPER = Path('experiments/run_paper_config.py')
TRAIN = Path('experiments/run_fmnist_lenet5.py')
MARK_PAPER = '# V370_STATIC_PAIR_MANIFEST_PAPER'
MARK_PARSER = '# V370_STATIC_PAIR_MANIFEST_ARG'
MARK_RUNNER = '# V370_STATIC_PAIR_MANIFEST_RUNNER'


def insert_after_line(lines, index, text):
    lines[index + 1:index + 1] = [line + '\n' for line in text.rstrip('\n').split('\n')]


def patch_paper(source):
    if MARK_PAPER in source:
        return source, False
    if '--static-pair-admission-manifest' in source:
        raise RuntimeError('paper entry already references V370 option without our marker; inspect manually')
    needle = '    fast_deadlines = system.get("fast_client_deadlines", {})'
    if source.count(needle) != 1:
        raise RuntimeError('paper entry has changed: cannot locate unique fast_deadlines assignment')
    addition = '''    # V370_STATIC_PAIR_MANIFEST_PAPER: opt-in, fail closed on inconsistent flags.
    static_pair_manifest = system.get("static_pair_admission_manifest")
    if static_pair_manifest is not None:
        if not isinstance(static_pair_manifest, str) or not static_pair_manifest.strip():
            raise ValueError("system.static_pair_admission_manifest must be a nonempty path")
        if not system.get("fl_first_split_on_demand", False):
            raise ValueError("V370 requires system.fl_first_split_on_demand=true")
        if system.get("edge_only_requires_fast_deadline", False):
            raise ValueError("V370 requires system.edge_only_requires_fast_deadline=false; LIE is not fast-only")
        if system.get("fast_client_deadlines"):
            raise ValueError("V370 fast_client_deadlines must come from the manifest, not both sources")
        command.extend(["--static-pair-admission-manifest", static_pair_manifest])
'''
    return source.replace(needle, addition + needle, 1), True


def _simple_name(node):
    return isinstance(node, ast.Name) and node.id == 'SelectionConfig'


def patch_train(source):
    if MARK_PARSER in source and MARK_RUNNER in source:
        return source, False
    if MARK_PARSER in source or MARK_RUNNER in source:
        raise RuntimeError('training entry is partially wired; inspect manually')
    tree = ast.parse(source)
    functions = [n for n in tree.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
    parser_func = next((n for n in functions if n.name == 'parse_args'), None)
    main_func = next((n for n in functions if n.name == 'main'), None)
    if parser_func is None or main_func is None:
        raise RuntimeError('cannot find parse_args/main functions')
    arg_calls = []
    for n in ast.walk(parser_func):
        if (isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
                and isinstance(n.value.func, ast.Attribute)
                and n.value.func.attr == 'add_argument'
                and isinstance(n.value.func.value, ast.Name)):
            arg_calls.append(n)
    if not arg_calls:
        raise RuntimeError('cannot find CLI argument definitions')
    parser_names = {n.value.func.value.id for n in arg_calls}
    if len(parser_names) != 1:
        raise RuntimeError(f'ambiguous parser variables: {sorted(parser_names)}')
    parser_name = parser_names.pop()
    latest_arg = max(arg_calls, key=lambda n: n.end_lineno)
    selection_stmts = []
    for n in ast.walk(main_func):
        if (isinstance(n, ast.Assign) and len(n.targets) == 1
                and isinstance(n.targets[0], ast.Name)
                and n.targets[0].id == 'selection'
                and isinstance(n.value, ast.Call) and _simple_name(n.value.func)):
            selection_stmts.append(n)
    if len(selection_stmts) != 1:
        raise RuntimeError(f'expected one selection = SelectionConfig(...), found {len(selection_stmts)}')
    selection_stmt = selection_stmts[0]
    lines = source.splitlines(keepends=True)
    arg_indent = lines[latest_arg.lineno - 1][:len(lines[latest_arg.lineno - 1]) - len(lines[latest_arg.lineno - 1].lstrip())]
    selection_indent = lines[selection_stmt.lineno - 1][:len(lines[selection_stmt.lineno - 1]) - len(lines[selection_stmt.lineno - 1].lstrip())]
    if len(selection_indent) != 4:
        raise RuntimeError('unexpected selection indentation')
    new_arg = (f'{arg_indent}{MARK_PARSER}\n'
               f'{arg_indent}{parser_name}.add_argument("--static-pair-admission-manifest", '
               'type=str, default=None, help="V370 fixed Client-Edge trust JSON manifest")')
    new_runner = '''    # V370_STATIC_PAIR_MANIFEST_RUNNER: load manifest before training starts.
    if args.static_pair_admission_manifest is not None:
        if not args.fl_first_split_on_demand:
            raise ValueError("V370 requires --fl-first-split-on-demand")
        if args.edge_only_requires_fast_deadline:
            raise ValueError("V370 disallows --edge-only-requires-fast-deadline (LIE need not be fast)")
        if args.fast_client_deadlines:
            raise ValueError("V370 fast client deadlines must come solely from its manifest")
        from dynfed.selection import enable_static_pair_admission
        selection = enable_static_pair_admission(selection, args.static_pair_admission_manifest)
        if not selection.strict_pair_admission:
            raise RuntimeError("V370 static pair admission did not activate")
        print("V370 static pair admission: ENABLED; manifest=", args.static_pair_admission_manifest, flush=True)
'''
    insertions = [(latest_arg.end_lineno - 1, new_arg), (selection_stmt.end_lineno - 1, new_runner)]
    for index, content in sorted(insertions, reverse=True):
        insert_after_line(lines, index, content)
    patched = ''.join(lines)
    ast.parse(patched)
    return patched, True


def main():
    a = argparse.ArgumentParser()
    group = a.add_mutually_exclusive_group(required=True)
    group.add_argument('--check', action='store_true')
    group.add_argument('--apply', action='store_true')
    args = a.parse_args()
    for path in (PAPER, TRAIN):
        if not path.is_file():
            raise SystemExit(f'MISSING: {path}; run from DynFL-privacy root')
    patches = []
    for path, fn in ((PAPER, patch_paper), (TRAIN, patch_train)):
        original = path.read_text(encoding='utf-8')
        updated, changed = fn(original)
        compile(updated, str(path), 'exec')
        patches.append((path, original, updated, changed))
    if args.check:
        for path, _old, _new, changed in patches:
            print(f'CHECK {path}: {"READY" if changed else "ALREADY WIRED"}')
        return
    # All edits have passed syntax/anchor checks before any write.
    for path, original, updated, changed in patches:
        if not changed:
            print(f'UNCHANGED {path} (already wired)')
            continue
        backup = path.with_suffix(path.suffix + '.before_v370_wiring')
        if backup.exists():
            raise RuntimeError(f'backup already exists; protect user changes: {backup}')
        backup.write_text(original, encoding='utf-8')
        path.write_text(updated, encoding='utf-8')
        print(f'WIRED {path}; backup={backup}')
    print('WIRING COMPLETE. Compile and run --help before smoke test.')

if __name__ == '__main__':
    main()
