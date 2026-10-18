#!/usr/bin/env python3
"""Stage22 counterfactual fixed-cohort shadow comparison (NO training).

For each round reuse precisely the SELECTED 17 Cloud-client IDs and their edge
placement, but hypothetically switch their modes to all LIIC or all LIIEIIIC.
Only candidates actually present in the PRE-DECISION oracle snapshot are used.

CRITICAL: pareto_profile_audit.csv records ONLY mode counts, not its client-level
candidate assignments. The two pure counterfactuals here ARE NOT the corresponding
archived pure Pareto evaluations; do not pair with archived system_latency.
No RDP-accountant replay or runtime admission/Edge/Cloud receipt validation is
performed for switched scenarios. This is explicitly simulator-oracle diagnostic.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path

from audit_stage20_rdp_shadow import (audit_selected_round, exact_int, number,
                                      read_csv, write_csv)

TARGET_MODES = ('LIIC', 'LIIEIIIC')
SCENARIOS = ('SELECTED', 'ALL_LIIC_FIXED_COHORT', 'ALL_LIIEIIIC_FIXED_COHORT')


def eligible_signature(entry: dict, mode: str, clip: float) -> tuple | None:
    stages = 1 if mode == 'LIIC' else 3
    if (entry.get('mode') != mode or
        entry.get('he_only_no_extra_release_dp') is not True or
        entry.get('sample_dp_optimizer') is not True or
        entry.get('edge_training_stages') != stages):
        return None
    try:
        sigma = number(entry['sample_optimizer_noise_multiplier'], 'candidate sigma')
        c = number(entry['sample_optimizer_clip_norm'], 'candidate clip')
        batches = entry['planned_batches_per_stage']
        if sigma <= 0 or not math.isclose(c, clip, rel_tol=1e-12, abs_tol=1e-12):
            return None
        if not isinstance(batches, list) or not batches:
            return None
        bs = tuple(exact_int(x, 'minibatch') for x in batches)
        if any(x <= 0 for x in bs):
            return None
    except (KeyError, TypeError, ValueError):
        return None
    return (sigma, c, stages, bs)


def change_row(row: dict, mode: str, snap: dict) -> dict:
    """Return a shadow-only row; fail closed if candidate is absent/ambiguous."""
    cid = exact_int(row['client_id'], 'client id')
    clip = number(snap['stage20_shadow_inputs']['sample_optimizer_clip_norm'], 'clip')
    entries = snap['clients'][str(cid)].get('candidates', {})
    signatures = {sig for entry in entries.values()
                  if (sig := eligible_signature(entry, mode, clip)) is not None}
    if not signatures:
        raise ValueError(f'client {cid}: no predecision {mode} Sample-DP/HE-only certificate')
    if len(signatures) != 1:
        raise ValueError(f'client {cid}: ambiguous {mode} candidate noise/batches')
    sigma, _, nstages, batches = signatures.pop()
    out = dict(row)
    out['mode'] = mode
    out['selected_sample_optimizer_noise_multiplier'] = repr(sigma)
    out['sample_optimizer_events'] = str(len(batches)*nstages)
    # This marker only satisfies Stage20's conditional *model* of HE-only
    # transfer. It is NOT an asserted actual mechanism or a recorded decision.
    out['mechanisms'] = 'L_C_upd:he3' if mode == 'LIIC' else 'E_C_upd:he3;L_E_upd:none'
    return out


def compute_round(snap: dict, decisions: list[dict], selected_profile: dict,
                  selected_logged: dict | None) -> list[dict]:
    selected = audit_selected_round(snap, decisions, selected_profile)
    rr = selected['round']
    if selected_logged is not None:
        expected = number(selected_logged['stage20_shadow_U_transfer'], 'existing Stage20 result')
        if not math.isclose(expected, selected['stage20_shadow_U_transfer'], rel_tol=1e-8, abs_tol=1e-7):
            raise ValueError(f'round {rr}: selected audit failed equality check vs saved Stage20 CSV')

    rows = []
    for scenario in SCENARIOS:
        if scenario == 'SELECTED':
            result = selected
            switched = 0
        else:
            mode = 'LIIC' if scenario == 'ALL_LIIC_FIXED_COHORT' else 'LIIEIIIC'
            hypothetical = []
            changed = 0
            try:
                for row in decisions:
                    if row['mode'] not in TARGET_MODES:
                        hypothetical.append(dict(row))
                    elif row['mode'] == mode:
                        hypothetical.append(dict(row))
                    else:
                        hypothetical.append(change_row(row, mode, snap))
                        changed += 1
                comparison_profile = dict(selected_profile)
                comparison_profile['LIIC_count'] = str(sum(r['mode'] == 'LIIC' for r in hypothetical))
                comparison_profile['LIIEIIIC_count'] = str(sum(r['mode'] == 'LIIEIIIC' for r in hypothetical))
                # Note: learning_error_objective remains old SELECTED J merely
                # because that field is required by the selected-audit API.
                # Never export that J as the hypothetical old J.
                result = audit_selected_round(snap, hypothetical, comparison_profile)
                switched = changed
            except (KeyError, TypeError, ValueError) as exc:
                rows.append(dict(round=rr, scenario=scenario, status='UNAVAILABLE_FAIL_CLOSED',
                                 reason=str(exc), LIIC='', LIIEIIIC='', SKIP='',
                                 changed_client_modes='', Stage19_J_selected_only='',
                                 Stage20_U_shadow='', D='', G='', sqrtV='', events='',
                                 assumption='FIXED_SELECTED_CLOUD_COHORT__FEASIBILITY_AND_LATENCY_NOT_REVALIDATED'))
                continue
        rows.append(dict(
            round=rr, scenario=scenario, status='COMPUTED_SELECTED' if scenario == 'SELECTED' else 'COMPUTED_HYPOTHETICAL',
            reason='', LIIC=result['LIIC'], LIIEIIIC=result['LIIEIIIC'], SKIP=result['SKIP'],
            changed_client_modes=switched,
            Stage19_J_selected_only=selected['old_J_logged'] if scenario == 'SELECTED' else '',
            Stage20_U_shadow=result['stage20_shadow_U_transfer'],
            D=result['D_deterministic_radius'], G=result['G_clipped_gradient_radius'],
            sqrtV=result['sqrt_V_RDP_gaussian_radius'], events=result['noise_event_count'],
            assumption=('SELECTED_SHADOW_ONLY__RECEIPTS_UNVERIFIED' if scenario == 'SELECTED'
                        else 'FIXED_SELECTED_CLOUD_COHORT__RDP_FEASIBILITY_AND_LATENCY_NOT_REVALIDATED'),
        ))
    return rows


def analyze(root: Path) -> list[dict]:
    decisions = defaultdict(list)
    for row in read_csv(root / 'client_decisions.csv'):
        decisions[exact_int(row['round'], 'round')].append(row)
    profiles = defaultdict(list)
    for p in read_csv(root / 'pareto_profile_audit.csv'):
        if str(p.get('selected', '')).strip() == '1':
            profiles[exact_int(p['round'], 'profile round')].append(p)
    saved_path = root / 'stage20_shadow_transfer_selected_NONCERTIFIED.csv'
    saved = {exact_int(x['round'], 'saved round'): x for x in read_csv(saved_path)} if saved_path.exists() else {}
    out = []
    for idx in sorted(decisions):
        if len(profiles[idx]) != 1:
            raise ValueError(f'round {idx}: expected one selected Pareto profile; got {len(profiles[idx])}')
        snap = json.loads((root / f'online_j_ORACLE_NONCERTIFIED_round_{idx:03d}.json').read_text(encoding='utf-8'))
        if exact_int(snap['round_index'], 'snapshot round') != idx:
            raise ValueError(f'round {idx}: stale snapshot')
        out.extend(compute_round(snap, decisions[idx], profiles[idx][0], saved.get(idx+1)))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('full_dynfl_dir', type=Path)
    args = ap.parse_args()
    rows = analyze(args.full_dynfl_dir)
    output = args.full_dynfl_dir / 'stage22_fixed_cohort_counterfactual_NONCERTIFIED.csv'
    write_csv(output, rows)
    print('Round | Scenario                | LIIC | LIIEIIIC | U20        | Status')
    for r in rows:
        u = r['Stage20_U_shadow']
        pretty = f'{u:10.4f}' if isinstance(u, (int, float)) else '         -'
        print(f"{r['round']:>5} | {r['scenario']:<23} | {str(r['LIIC']):>4} | "
              f"{str(r['LIIEIIIC']):>8} | {pretty} | {r['status']}")
        if r['status'] == 'UNAVAILABLE_FAIL_CLOSED':
            print('        REASON:', r['reason'])
    print('Saved:', output)
    print('WARNING: pure-mode scenarios hold the SELECTED Cloud cohort fixed; they are NOT'
          ' archived Pareto profiles. Recomputed RDP budget, latency, and stage receipts are NOT checked.')


if __name__ == '__main__':
    main()
