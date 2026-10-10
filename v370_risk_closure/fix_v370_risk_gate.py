#!/usr/bin/env python3
"""Fail-closed V370 risk gate without changing legacy admission semantics."""
from pathlib import Path
import argparse
import ast
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--check', action='store_true')
parser.add_argument('--apply', action='store_true')
args = parser.parse_args()
if args.check == args.apply:
    parser.error('choose exactly one of --check or --apply')
p = Path('dynfed/selection.py')
if not p.is_file():
    raise SystemExit('Run this script from DynFL-privacy project root')
s = p.read_text(encoding='utf-8')
if 'risk_hard_gate: bool = False' in s:
    if ('and (not self.risk_hard_gate or self.feasible_risk)' in s
        and 'risk_hard_gate=config.strict_pair_admission,' in s
        and 'or any(item.risk_hard_gate for item in candidates)' in s
        and 'and not config.strict_pair_admission:' in s):
        print('ALREADY APPLIED: V370 risk gate is in place')
        sys.exit(0)
    raise SystemExit('PARTIAL PATCH detected; refusing to modify')
old_field = '    sample_optimizer_noise_multiplier: float | None = None\n\n    @property\n    def feasible(self) -> bool:'
new_field = '    sample_optimizer_noise_multiplier: float | None = None\n    risk_hard_gate: bool = False  # V370 opt-in; do not retroactively change legacy runs\n\n    @property\n    def feasible(self) -> bool:'
old_prop = '            self.feasible_resource\n            and self.feasible_memory\n            and self.feasible_privacy\n        )'
new_prop = '            self.feasible_resource\n            and self.feasible_memory\n            and self.feasible_privacy\n            and (not self.risk_hard_gate or self.feasible_risk)\n        )'
old_ctor = '        feasible_privacy=feasible_privacy,\n        feasible_risk=feasible_risk,\n        feasible_time=feasible_time,'
new_ctor = '        feasible_privacy=feasible_privacy,\n        feasible_risk=feasible_risk,\n        risk_hard_gate=config.strict_pair_admission,\n        feasible_time=feasible_time,'
old_choose = '    if require_feasible and not feasible:\n        return skipped_candidate()'
new_choose = '    if (require_feasible or any(item.risk_hard_gate for item in candidates)) and not feasible:\n        return skipped_candidate()'
old_fallback = '        if not pool and not config.require_feasible:\n            pool = ['
new_fallback = '        if not pool and not config.require_feasible and not config.strict_pair_admission:\n            pool = ['
old_current = '            pool = [current if current.feasible_device else skipped_candidate()]'
new_current = ('            pool = [current if (current.feasible if config.strict_pair_admission '
               'else current.feasible_device) else skipped_candidate()]')
anchors=[('Candidate field', old_field), ('feasible property', old_prop),
         ('estimate constructor', old_ctor), ('choose fallback', old_choose),
         ('Pareto fallback', old_fallback), ('Pareto current fallback', old_current)]
for title, old in anchors:
    c = s.count(old)
    if c != 1:
        raise SystemExit(f'{title}: expected exactly 1 match, found {c}; NO CHANGE')
out=s
for before, after in [(old_field,new_field),(old_prop,new_prop),(old_ctor,new_ctor),
                      (old_choose,new_choose),(old_fallback,new_fallback),(old_current,new_current)]:
    out=out.replace(before,after,1)
ast.parse(out, filename=str(p))
if args.check:
    print('CHECK OK: 6 unique anchors, syntax valid; no file changed')
else:
    backup=p.with_name('selection.py.before_v370_risk_gate')
    if backup.exists():
        raise SystemExit(f'backup exists: {backup}; refusing to overwrite')
    backup.write_text(s,encoding='utf-8')
    p.write_text(out,encoding='utf-8')
    print(f'APPLIED: risk gate only for strict_pair_admission=True; backup={backup}')
