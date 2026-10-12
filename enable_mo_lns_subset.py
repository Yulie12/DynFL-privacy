from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

ROOT = Path.cwd()
SEL = ROOT / "dynfed" / "selection.py"
RUN = ROOT / "dynfed" / "fmnist_lenet5_dynamic.py"

if not SEL.is_file() or not RUN.is_file():
    sys.exit("PATCH_ABORTED: run this script from the DynFL-privacy repository root.")


def read_source(path: Path) -> tuple[bytes, str, str, bool]:
    raw = path.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    newline = "\r\n" if "\r\n" in text else "\n"
    return raw, text.replace("\r\n", "\n"), newline, bom


sel_raw, sel, sel_nl, sel_bom = read_source(SEL)
run_raw, run, run_nl, run_bom = read_source(RUN)

# Require the exact source shape that was reviewed in the user's terminal.
checks = {
    "selector method validation": 'if search_method not in {"bounded", "nsga2"}:' ,
    "bounded iteration assignment": 'bounded_iterations = config.pareto_max_iters if search_method == "bounded" else 0',
    "neighbor loop": '            ] = []\n            for client_id in search_client_ids:',
    "runner dispatch": '            search_method = "nsga2" if policy == "nsga2" else "bounded"',
}
for label, needle in checks.items():
    haystack = run if label == "runner dispatch" else sel
    if haystack.count(needle) != 1:
        sys.exit(f"PATCH_ABORTED: expected exactly one match for {label}, found {haystack.count(needle)}. No files changed.")

sel_replacements = [
    (
        'if search_method not in {"bounded", "nsga2"}:',
        'if search_method not in {"bounded", "nsga2", "mo_lns"}:',
    ),
    (
        'bounded_iterations = config.pareto_max_iters if search_method == "bounded" else 0',
        'bounded_iterations = config.pareto_max_iters if search_method in {"bounded", "mo_lns"} else 0',
    ),
    (
        '            ] = []\n            for client_id in search_client_ids:',
        '''            ] = []
            # MO-LNS first version: explore a rotating subset of client coordinates
            # per search iteration. Candidate generation, privacy/resource checks,
            # exact priority calculation, archive update, and Top-K rules remain
            # unchanged. This is an approximate restricted-neighborhood backend.
            if search_method == "mo_lns":
                _lns_budget = max(
                    1, int(__import__("os").environ.get("DYNFL_MO_LNS_CLIENT_BUDGET", "20"))
                )
                _ordered_clients = sorted(search_client_ids)
                if len(_ordered_clients) > _lns_budget:
                    _start = (_iter_idx * _lns_budget) % len(_ordered_clients)
                    _rotated = _ordered_clients[_start:] + _ordered_clients[:_start]
                    _expansion_client_ids = _rotated[:_lns_budget]
                else:
                    _expansion_client_ids = _ordered_clients
            else:
                _expansion_client_ids = search_client_ids

            for client_id in _expansion_client_ids:''',
    ),
]
for old, new in sel_replacements:
    if sel.count(old) != 1:
        sys.exit(f"PATCH_ABORTED: selector replacement mismatch for: {old[:70]!r}. No files changed.")
    sel = sel.replace(old, new, 1)

old_dispatch = '            search_method = "nsga2" if policy == "nsga2" else "bounded"'
new_dispatch = '''            _requested_search_method = str(
                __import__("os").environ.get("DYNFL_SEARCH_METHOD", "")
            ).strip().lower()
            if policy == "full_dynfl" and _requested_search_method == "mo_lns":
                search_method = "mo_lns"
            else:
                search_method = "nsga2" if policy == "nsga2" else "bounded"'''
run = run.replace(old_dispatch, new_dispatch, 1)

# Validate syntax before writing either file.
try:
    compile(sel, str(SEL), "exec")
    compile(run, str(RUN), "exec")
except SyntaxError as exc:
    sys.exit(f"PATCH_ABORTED: syntax validation failed; no files changed: {exc}")

stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
sel_backup = SEL.with_name(SEL.name + f".bak_mo_lns_{stamp}")
run_backup = RUN.with_name(RUN.name + f".bak_mo_lns_{stamp}")
sel_backup.write_bytes(sel_raw)
run_backup.write_bytes(run_raw)

for path, text, newline, bom in [
    (SEL, sel, sel_nl, sel_bom),
    (RUN, run, run_nl, run_bom),
]:
    data = text.replace("\n", newline).encode("utf-8")
    if bom:
        data = b"\xef\xbb\xbf" + data
    path.write_bytes(data)

print("PATCH_OK: MO-LNS restricted-neighborhood backend installed")
print(f"Backup selection.py: {sel_backup}")
print(f"Backup fmnist_lenet5_dynamic.py: {run_backup}")
print("Run with DYNFL_SEARCH_METHOD=mo_lns; optional DYNFL_MO_LNS_CLIENT_BUDGET=20")
