"""Transactional reconciliation for DynFL clipping-only experiment.
Run from repository root: D:\\soft\\Python310\\python.exe repair_clipping_only.py
Makes backups, and writes only after all checks and Python compilation succeed.
Does NOT certify the experiment scientifically; inspect runtime counters before using results.
"""
from pathlib import Path
import ast
import datetime
import re
import shutil
import sys

ROOT = Path.cwd()
P = ROOT / 'dynfed' / 'fmnist_lenet5_dynamic.py'
S = ROOT / 'dynfed' / 'split_learning.py'
assert P.is_file() and S.is_file(), 'Run from DynFL-privacy project root'
files = {P: P.read_text(encoding='utf-8-sig'), S: S.read_text(encoding='utf-8-sig')}
original = dict(files)


def exact(path, before, after, description):
    src = files[path]
    if after in src:
        print('ALREADY:', description)
        return
    n = src.count(before)
    if n != 1:
        raise RuntimeError(f'{description}: expected one anchor, found {n}. NO FILES MODIFIED.')
    files[path] = src.replace(before, after)
    print('PATCH:', description)


def regex_once(path, pat, replacement, description, done=None):
    src = files[path]
    if done is not None and done in src:
        print('ALREADY:', description)
        return
    result, n = re.subn(pat, replacement, src, count=0, flags=re.MULTILINE)
    if n != 1:
        raise RuntimeError(f'{description}: expected one match, found {n}. NO FILES MODIFIED.')
    files[path] = result
    print('PATCH:', description)

# Repair an interrupted PowerShell insertion, if it exists. Restrict to the unique
# multi-edge worker stage directly following multi_edge_retraining_tasks.
regex_once(
    P,
    r'(multi_edge_retraining_tasks \+= len\(cycle_tasks\)\r?\n)\s*if effective_selection\.clipping_only_ablation:\r?\n\s*for key in \(\r?\n(?:[^\n]*\r?\n){3,7}?\s*elif effective_selection\.privacy_unit == "sample":',
    r'\1                    if effective_selection.privacy_unit == "sample" and not effective_selection.clipping_only_ablation:',
    'repair truncated multi-stage insertion',
    done='multi_edge_retraining_tasks += len(cycle_tasks)\n                    if effective_selection.privacy_unit == "sample" and not effective_selection.clipping_only_ablation:',
) if 'multi_edge_retraining_tasks += len(cycle_tasks)\n                    if effective_selection.clipping_only_ablation:' in files[P] else None

# Function argument and DP-only validation: trusted joint validation remains strictly positive.
exact(S, '    trusted_split_joint_sample_dp: bool = False,',
      '    clipping_only_ablation: bool = False,\n    trusted_split_joint_sample_dp: bool = False,', 'split-training option')
# Correct any accidental weakening of the trusted-joint noise check.
files[S] = files[S].replace(
    'if sigma is None or not math.isfinite(float(sigma)) or float(sigma) < 0.0 or (float(sigma) == 0.0 and not clipping_only_ablation):',
    'if sigma is None or not math.isfinite(float(sigma)) or float(sigma) <= 0.0:')
# The user's previous patch changed the condition but kept the original error text.
old_guard = 'if not math.isfinite(float(sigma)) or float(sigma) <= 0.0:'
new_guard = 'if not math.isfinite(float(sigma)) or float(sigma) < 0.0 or (float(sigma) == 0.0 and not clipping_only_ablation):'
if new_guard not in files[S]:
    exact(S, old_guard, new_guard, 'permit zero split noise only in clipping-only')
else:
    print('ALREADY: split zero-noise check')

exact(P,
      '        training_diagnostics=training_diagnostics,\n        privacy_unit=payload.get(',
      '        training_diagnostics=training_diagnostics,\n        clipping_only_ablation=payload.get("clipping_only_ablation", False),\n        privacy_unit=payload.get(',
      'pass flag into worker')

# Preserve non-DP identity in privacy reporting, including already-patched versions.
exact(P,
      '    no_dp_ablation: bool = False,\n    mainline_dp_enabled:',
      '    no_dp_ablation: bool = False,\n    clipping_only_ablation: bool = False,\n    mainline_dp_enabled:',
      'report signature')
# Replace even a partially pasted report branch with an authoritative complete branch.
report_branch = (
    '    if clipping_only_ablation:\n'
    '        return {\n'
    '            "dp_accountant_scope": "not_applicable",\n'
    '            "privacy_policy_scope": "full_dynfl_clipping_only",\n'
    '            "protected_object": "none",\n'
    '            "privacy_mechanism_scope": "per_sample_clipping_without_noise",\n'
    '            "privacy_guarantee": "no_dp_guarantee",\n'
    '        }\n'
)
if '    if clipping_only_ablation:' in files[P]:
    regex_once(P, r'^    if clipping_only_ablation:\r?\n(?:        .*\r?\n)+?(?=    if no_dp_ablation:)',
               report_branch, 'normalize partial privacy report',
               done=report_branch + '    if no_dp_ablation:')
else:
    exact(P, '    if no_dp_ablation:\n        return {',
          report_branch + '    if no_dp_ablation:\n        return {', 'no-DP clipping-only report')
exact(P,
      '        no_dp_ablation=effective_selection.no_dp_ablation,\n        mainline_dp_enabled=',
      '        no_dp_ablation=effective_selection.no_dp_ablation,\n        clipping_only_ablation=effective_selection.clipping_only_ablation,\n        mainline_dp_enabled=',
      'pass report flag')

# Preserve per-sample clipping but disable all three Gaussian noise sources.
# Normalize user-provided partial here-doc to a complete zero-noise payload.
payload_branch = (
    '        if selection.clipping_only_ablation:\n'
    '            payload["clipping_only_ablation"] = True\n'
    '            payload["sample_embedding_noise_multiplier"] = 0.0\n'
    '            payload["sample_label_grad_noise_multiplier"] = 0.0\n'
    '            payload["sample_optimizer_noise_multiplier"] = 0.0\n'
    '            payload["dp_update_mode"] = "off"\n\n'
)
if '        if selection.clipping_only_ablation:' in files[P]:
    regex_once(P, r'^        if selection.clipping_only_ablation:\r?\n(?:(?:            [^\n]*\r?\n)|(?:\r?\n)){1,12}(?=        if selection.no_dp_ablation:)',
               payload_branch, 'normalize payload zero-noise branch',
               done=payload_branch + '        if selection.no_dp_ablation:')
else:
    exact(P, '        if selection.no_dp_ablation:\n            # Use the ordinary optimizer',
          payload_branch + '        if selection.no_dp_ablation:\n            # Use the ordinary optimizer',
          'worker zero Gaussian noise')

exact(P,
      '        _assert_trusted_lie_accounting_plan(\n            selection=selection, candidate=candidate,',
      '        if selection.clipping_only_ablation and selection.trusted_lie_joint_sample_dp and candidate.mode == "LIE":\n            raise ValueError("Clipping-only does not support trusted joint Sample-DP")\n        _assert_trusted_lie_accounting_plan(\n            selection=selection, candidate=candidate,',
      'reject incompatible trusted-joint path')

exact(P,
      '    if selection.privacy_unit == "sample" and sample_dispatch_attempt is None:',
      '    if selection.privacy_unit == "sample" and not selection.clipping_only_ablation and sample_dispatch_attempt is None:',
      'dispatch without DP-noise attempt')

# Budget projection is a counterfactual selector heuristic; it must not be
# presented as a real privacy spend, and cannot be an execution admission gate.
exact(P,
      '                if not ledger.can_apply(projection):\n                    raise ValueError(\n                        "Selected Sample-DP release exceeds "',
      '                if not effective_selection.clipping_only_ablation and not ledger.can_apply(projection):\n                    raise ValueError(\n                        "Selected Sample-DP release exceeds "',
      'avoid formal privacy budget gate for non-DP experiment')

exact(P,
      '        sample_dispatch_attempt: SampleDispatchAttempt | None = None\n        if effective_selection.privacy_unit == "sample":',
      '        sample_dispatch_attempt: SampleDispatchAttempt | None = None\n        if effective_selection.privacy_unit == "sample" and not effective_selection.clipping_only_ablation:',
      'no Sample-DP precharge')

# Audit sections identified by content rather than generic conditions.
for marker, label in [
    ('        sample_worker_client_audit_count = 0', 'initial audit'),
    ('                    multi_edge_retraining_tasks += len(cycle_tasks)', 'multi-stage worker audit'),
    ('                # Stage 0 is audited', 'edge aggregation audit'),
    ('        # The Flow may drop', 'final audit'),
]:
    if label == 'initial audit':
        pat = r'(sample_worker_client_audit_count = 0\r?\n\s*)if effective_selection\.privacy_unit == "sample":'
    elif label == 'final audit':
        pat = r'if effective_selection\.privacy_unit == "sample":(?=\r?\n\s*# The Flow may drop)'
    elif label == 'multi-stage worker audit':
        pat = r'(multi_edge_retraining_tasks \+= len\(cycle_tasks\)\r?\n\s*)if effective_selection\.privacy_unit == "sample":'
    else:
        pat = r'if effective_selection\.privacy_unit == "sample":(?=\r?\n\s*# Stage 0 is audited)'
    if label in ('initial audit', 'multi-stage worker audit'):
        rep = r'\1if effective_selection.privacy_unit == "sample" and not effective_selection.clipping_only_ablation:'
    else:
        rep = 'if effective_selection.privacy_unit == "sample" and not effective_selection.clipping_only_ablation:'
    if re.search(pat, files[P]):
        regex_once(P, pat, rep, label)

# Other audit condition adjacent to authorize_edge_aggregate.
regex_once(P,
    r'if effective_selection\.privacy_unit == "sample":(?=\r?\n\s*# Stage 0 is audited)',
    'if effective_selection.privacy_unit == "sample" and not effective_selection.clipping_only_ablation:',
    'edge audit guard',
) if re.search(r'if effective_selection\.privacy_unit == "sample":(?=\r?\n\s*# Stage 0 is audited)', files[P]) else None

# Avoid entering ClientPrivacyLedger branch with a SamplePrivacyLedger.
# Support both pristine and already-edited commit branches.
commit_old = '                if effective_selection.privacy_unit == "sample":\n                    if not isinstance(\n                        ledger,\n                        SamplePrivacyLedger,'
commit_changed = commit_old.replace('privacy_unit == "sample":', 'privacy_unit == "sample" and not effective_selection.clipping_only_ablation:')
commit_new = '                if effective_selection.clipping_only_ablation:\n                    for key in ("feature_dp_events", "update_dp_events", "sample_embedding_events", "sample_label_grad_events", "sample_optimizer_events"):\n                        row[key] = 0\n                elif effective_selection.privacy_unit == "sample":\n                    if not isinstance(\n                        ledger,\n                        SamplePrivacyLedger,'
if commit_new not in files[P]:
    if files[P].count(commit_changed) == 1:
        exact(P, commit_changed, commit_new, 'separate sample-vs-client ledger commit')
    else:
        exact(P, commit_old, commit_new, 'separate sample-vs-client ledger commit')

# Ensure downstream output never claims that epsilon is a DP guarantee.
# Two further sections share the same condition; replace remaining occurrences.
files[P] = files[P].replace(
    'if effective_selection.privacy_unit == "sample" and not effective_selection.no_dp_ablation:',
    'if effective_selection.privacy_unit == "sample" and not (effective_selection.no_dp_ablation or effective_selection.clipping_only_ablation):')
files[P] = files[P].replace(
    'if effective_selection.no_dp_ablation:\n            current_round.update(privacy_reporting_scope)',
    'if effective_selection.no_dp_ablation or effective_selection.clipping_only_ablation:\n            current_round.update(privacy_reporting_scope)')
files[P] = files[P].replace(
    'if effective_selection.no_dp_ablation:\n        summary.update(privacy_reporting_scope)',
    'if effective_selection.no_dp_ablation or effective_selection.clipping_only_ablation:\n        summary.update(privacy_reporting_scope)')

# Do not run new ablation with cloud-side DP noise still enabled.
# Preserve optional update clipping through existing diagnostic clip-only path.
exact(P,
      '    update_dp_execution_enabled = train_config.dp_update_mode != "off"',
      '    update_dp_execution_enabled = (train_config.dp_update_mode != "off" and not effective_selection.clipping_only_ablation)',
      'disable cloud/edge update DP execution')
files[P] = files[P].replace(
    '        and diagnostic_update_ablation == "clip_only"\n    )',
    '        and (diagnostic_update_ablation == "clip_only" or effective_selection.clipping_only_ablation)\n    )')

# Summaries: never overwrite the no-DP guarantee with RDP metadata.
files[P] = files[P].replace(
    '    if effective_selection.privacy_unit == "sample":\n        summary["dp_accountant_scope"] = (',
    '    if effective_selection.privacy_unit == "sample" and not effective_selection.clipping_only_ablation:\n        summary["dp_accountant_scope"] = (')

# Check entire modified Python modules before writing anything.
for path, source in files.items():
    ast.parse(source, filename=str(path))
    compile(source, str(path), 'exec')

stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
for path in files:
    if files[path] != original[path]:
        backup = path.with_name(path.name + '.before_clipping_reconcile_' + stamp)
        shutil.copy2(path, backup)
        print('BACKUP:', backup)
for path in files:
    if files[path] != original[path]:
        path.write_text(files[path], encoding='utf-8')
        print('UPDATED:', path)
print('Syntax validated. NOT yet certified for training: inspect cloud route and run one-round smoke test.')
