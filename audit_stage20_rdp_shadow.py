#!/usr/bin/env python3
"""Stage20 simulator-only, selected-profile shadow upper bound.

Requires the optional Stage20 snapshot metadata patch. Uses exact candidate
sample-DP sigma, clip C, planned batch sizes and the reference transfer
calculator. NEVER changes Pareto selection, trains a model, or certifies DP.

It is still *conditional*: actual worker stages, Edge cohorts and Cloud
admission must be checked against runtime receipts before interpreting this
numeric quantity as an execution-valid bound. HE='profiled' supplies no crypto.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

from dynfl_transfer_j_reference import Client, Step, build_transfer_bound

CLOUD_MODES = {'LIIC', 'LIIEIIIC'}
ALLOWED = CLOUD_MODES | {'LIIE', 'SKIP'}


def read_csv(path: Path) -> list[dict]:
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def number(value: object, label: str) -> float:
    try:
        n = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f'{label}: expected finite number, got {value!r}') from exc
    if not math.isfinite(n):
        raise ValueError(f'{label}: non-finite value {value!r}')
    return n


def exact_int(value: object, label: str) -> int:
    n = number(value, label)
    if not n.is_integer():
        raise ValueError(f'{label}: must be an integer, got {value!r}')
    return int(n)


def candidate_entry(pop: dict, row: dict, clip: float, stages: int) -> dict:
    mode = row['mode']
    sig = number(row.get('selected_sample_optimizer_noise_multiplier'), f'{mode} sigma')
    if sig <= 0:
        raise ValueError(f"client {row['client_id']} mode {mode}: Sample-DP sigma must be >0")
    matches = []
    for entry in pop['candidates'].values():
        if entry.get('mode') != mode or entry.get('he_only_no_extra_release_dp') is not True:
            continue
        if entry.get('sample_dp_optimizer') is not True or entry.get('edge_training_stages') != stages:
            continue
        es = number(entry.get('sample_optimizer_noise_multiplier'), 'snapshot sigma')
        ec = number(entry.get('sample_optimizer_clip_norm'), 'snapshot clip')
        if math.isclose(sig, es, rel_tol=1e-12, abs_tol=1e-12) and math.isclose(clip, ec, rel_tol=1e-12, abs_tol=1e-12):
            matches.append(entry)
    if not matches:
        raise ValueError(f"client {row['client_id']} mode {mode}: no snapshot candidate matches mode/stages/C/sigma")
    batch_lists = [entry.get('planned_batches_per_stage') for entry in matches]
    if any(v != batch_lists[0] for v in batch_lists[1:]):
        raise ValueError(f"client {row['client_id']}: ambiguous batch schedules for selected mode")
    return matches[0]


def audit_selected_round(snapshot: dict, decisions: list[dict], profile: dict) -> dict:
    if snapshot.get('schema') != 'dynfl.online_j.oracle_diagnostic.v2':
        raise ValueError('expected Stage19 oracle diagnostic v2')
    if snapshot.get('used_raw_client_training_data') is not True or snapshot.get('formal_online_j_certified') is not False:
        raise ValueError('expected explicitly non-certified CENTRAL ORACLE')
    inputs = snapshot.get('stage20_shadow_inputs')
    if not isinstance(inputs, dict):
        raise ValueError('missing stage20_shadow_inputs: add metadata patch and regenerate the run')
    if inputs.get('privacy_unit') != 'sample' or inputs.get('assumptions_not_certified') is not True:
        raise ValueError('unsupported privacy/verification contract')
    if number(inputs.get('momentum'), 'momentum') != .9:
        raise ValueError('only ResNet18 full-model momentum=0.9 path supported')
    batch_size = exact_int(inputs.get('batch_size'), 'batch_size')
    epochs = exact_int(inputs.get('local_epochs'), 'local_epochs')
    if batch_size < 1 or epochs < 1:
        raise ValueError('nonpositive planned batch/epoch count')
    lr = number(inputs.get('learning_rate'), 'lr')
    wd = number(inputs.get('weight_decay'), 'weight_decay')
    mu = number(inputs.get('momentum'), 'momentum')
    clip = number(inputs.get('sample_optimizer_clip_norm'), 'clip')
    if lr <= 0 or wd < 0 or clip <= 0:
        raise ValueError('invalid optimizer / clip parameters')
    by_id = {}
    for row in decisions:
        cid = exact_int(row['client_id'], 'client_id')
        if cid in by_id:
            raise ValueError(f'duplicate client decision {cid}')
        by_id[cid] = row
    raw_clients = snapshot['clients']
    if set(by_id) != {int(cid) for cid in raw_clients}:
        raise ValueError('client population does not match Oracle snapshot')
    selected = []
    for cid in sorted(by_id):
        row = by_id[cid]
        pop = raw_clients[str(cid)]
        mode = row['mode']
        if mode not in ALLOWED:
            raise ValueError(f'client {cid}: unsupported Split/other mode {mode}')
        n = exact_int(pop['samples'], f'client {cid} samples')
        edge = exact_int(row['edge_id'], f'client {cid} edge')
        if n <= 0 or edge < 0:
            raise ValueError('invalid sample count/edge assignment')
        if mode not in CLOUD_MODES:
            selected.append(Client(cid, mode, n, edge, (), lr, mu, wd))
            continue
        # Extra DP release noise, ciphertext numeric error, or protocol
        # mechanisms beyond HE profiled are outside this conditional model.
        mechs = row.get('mechanisms', '')
        if not mechs or any('dp' in token.lower() for token in mechs.replace(';',' ').split()):
            raise ValueError(f'client {cid}: unsupported update/link DP mechanism {mechs!r}')
        # Exact HE3 cloud link is required, no unmodeled update release.
        cloud_link = 'L_C_upd:he3' if mode == 'LIIC' else 'E_C_upd:he3'
        if cloud_link not in mechs:
            raise ValueError(f'client {cid}: expected {cloud_link}, got {mechs!r}')
        stages = 1 if mode == 'LIIC' else 3
        entry = candidate_entry(pop, row, clip, stages)
        batches = entry['planned_batches_per_stage']
        planned = [min(batch_size, n-start) for start in range(0, n, batch_size)]*epochs
        if batches != planned:
            raise ValueError(f'client {cid}: snapshot batch sizes differ from uncapped epoch plan')
        if 'sample_optimizer_events' in row and row['sample_optimizer_events'] not in ('',None):
            charged = exact_int(row['sample_optimizer_events'], 'charged optimizer events')
            if charged != len(batches)*stages:
                raise ValueError(f'client {cid}: RDP planned events {charged} != SGD steps {len(batches)*stages}')
        sigma = number(row['selected_sample_optimizer_noise_multiplier'], 'selected sigma')
        schedule = tuple(tuple(Step(b, clip, sigma) for b in batches) for _ in range(stages))
        selected.append(Client(cid, mode, n, edge, schedule, lr, mu, wd))
    if not any(c.mode in CLOUD_MODES for c in selected):
        raise ValueError('no Cloud-active clients: shadow needs separate no-cloud handling')
    counts = {mode: sum(c.mode == mode for c in selected) for mode in ALLOWED}
    for mode in CLOUD_MODES:
        logged = exact_int(profile[f'{mode}_count'], f'{mode} count')
        if logged != counts[mode]:
            raise ValueError(f'candidate cohort mismatch: logged {mode}={logged} vs decisions {counts[mode]}')
    gamma = number(snapshot['cloud_step'], 'cloud step')
    bound = build_transfer_bound(
        clients=selected,
        dimension=exact_int(inputs['trainable_dimension'], 'dimension'),
        theta_norm=number(inputs['model_parameter_norm'], 'theta_norm'),
        ideal_delta_norm=number(snapshot['ideal_population_update_norm'], 'ideal norm'),
        theta_dot_ideal_delta=number(inputs['model_parameter_dot_ideal_update'], 'theta_dot_ideal'),
        cloud_step=gamma,
    )
    return {
        'round': exact_int(snapshot['round_index'], 'round index') + 1,
        'LIIC': counts['LIIC'], 'LIIEIIIC': counts['LIIEIIIC'],
        'LIIE': counts['LIIE'], 'SKIP': counts['SKIP'],
        'old_J_logged': number(profile['learning_error_objective'], 'old_J'),
        'stage20_shadow_U_transfer': bound.upper_j,
        'D_deterministic_radius': bound.deterministic_norm,
        'G_clipped_gradient_radius': bound.gradient_radius,
        'sqrt_V_RDP_gaussian_radius': bound.noise_rms_radius,
        'a_weight_decay_model_scale': bound.displacement_scale,
        'noise_event_count': len(bound.events),
        'result_scope': 'SIMULATOR_SHADOW_ONLY_NOT_DEPLOYMENT_CERTIFIED',
        'cohort_validation': 'SELECTED_DECISIONS_ONLY_RUNTIME_STAGE_RECEIPTS_NOT_RECHECKED',
        'numeric_error_scope': 'HE_PROFILED_FLOAT_AND_NUMERIC_RESIDUAL_NOT_CERTIFIED',
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('full_dynfl_dir', type=Path)
    args = ap.parse_args()
    root: Path = args.full_dynfl_dir
    decisions = defaultdict(list)
    for row in read_csv(root/'client_decisions.csv'):
        decisions[exact_int(row['round'], 'decision round')].append(row)
    profiles = defaultdict(list)
    for row in read_csv(root/'pareto_profile_audit.csv'):
        if str(row.get('selected', '')).strip() == '1':
            profiles[exact_int(row['round'], 'profile round')].append(row)
    results = []
    for idx in sorted(decisions):
        if len(profiles[idx]) != 1:
            raise ValueError(f'round {idx}: expected one selected Pareto profile')
        path = root/f'online_j_ORACLE_NONCERTIFIED_round_{idx:03d}.json'
        snapshot = json.loads(path.read_text(encoding='utf-8'))
        if exact_int(snapshot['round_index'], 'Oracle round') != idx:
            raise ValueError(f'round {idx}: stale Oracle snapshot')
        results.append(audit_selected_round(snapshot, decisions[idx], profiles[idx][0]))
    output = root/'stage20_shadow_transfer_selected_NONCERTIFIED.csv'
    write_csv(output, results)
    for r in results:
        print(f"round={r['round']} LIIC={r['LIIC']} LIIEIIIC={r['LIIEIIIC']} "
              f"Stage19_J={r['old_J_logged']:.7g} transfer_shadow={r['stage20_shadow_U_transfer']:.7g} "
              f"D={r['D_deterministic_radius']:.5g} G={r['G_clipped_gradient_radius']:.5g} "
              f"sqrtV={r['sqrt_V_RDP_gaussian_radius']:.5g} [NOT CERTIFIED]")
    print('Shadow-only output:', output)


if __name__ == '__main__':
    main()
