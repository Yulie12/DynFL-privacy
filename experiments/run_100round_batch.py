"""Two-model 100-round STAGING batch with checked, policy-specific 10-round resume.

Prechecks the first seed of each model for 10 effective rounds. It reuses
those precheck checkpoints for the remaining 90 rounds, and runs each policy
serially. Fails closed on zero-training rounds or a failed/private dispatch;
never reuses a pre-failure checkpoint automatically.

This launcher does NOT change training, RDP budgets, DP noise or J certificates.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / 'experiments' / 'run_paper_config.py'
CONFIGS = {
    'lenet5': ROOT / 'configs' / 'stage11_fmnist_lenet5_legacy_5r_STAGING.json',
    'resnet18': ROOT / 'configs' / 'stage11_cifar10_resnet18_layer4_legacy_5r_STAGING.json',
}
POLICIES = ('fixed_mode_fixed_privacy', 'dynamic_mode_fixed_privacy',
            'fixed_mode_dynamic_privacy', 'full_dynfl')
TARGET_ROUNDS = 100
CHUNK_ROUNDS = 10


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.json.tmp')
    with temp.open('w', encoding='utf-8') as output:
        json.dump(payload, output, ensure_ascii=False, indent=2, allow_nan=False)
        output.flush()
        os.fsync(output.fileno())
    temp.replace(path)


def validate_config(config: dict, model: str) -> None:
    training = config['training']
    actual_model = ('lenet5' if model == 'lenet5' else 'resnet18_pretrained_layer4_head')
    if model not in CONFIGS or training['model'] != actual_model:
        raise ValueError(f'{model}: unexpected model in locked configuration')
    if (training['batch_size'] != 256 or training['local_epochs'] != 3
            or training.get('uncapped_local_batches') is not True):
        raise ValueError(f'{model}: locked batch=256, epochs=3, uncapped required')
    if tuple(config['policies']) != POLICIES or config['learning']['objective'] != 'legacy_fusion_dp':
        raise ValueError(f'{model}: expected four legacy J baseline strategies')
    if config['he']['execution'] != 'profiled':
        raise ValueError(f'{model}: expected same STAGING HE execution')


def _read_round_metrics(csv_file: Path, target: int, prior: int) -> tuple[float, float]:
    if not csv_file.is_file():
        raise RuntimeError(f'Missing recorded rounds: {csv_file}')
    with csv_file.open('r', encoding='utf-8-sig', newline='') as reader:
        rows = list(csv.DictReader(reader))
    if len(rows) != target:
        raise RuntimeError(f'Expected {target} recorded rounds, found {len(rows)} at {csv_file}')
    for j in range(prior, target):
        row = rows[j]
        steps = int(float(row['actual_optimizer_steps']))
        if steps <= 0:
            raise RuntimeError(
                f'EMPTY TRAINING at round {j+1}/{TARGET_ROUNDS} in {csv_file}; '
                'stop batch, inspect budget/admission; do not silently relax safeguards'
            )
        loss = float(row['test_loss'])
        if not math.isfinite(loss):
            raise RuntimeError(f'non-finite loss at round {j+1}: {csv_file}')
    return float(rows[-1]['test_accuracy']), float(rows[-1]['test_loss'])


def _run_dir(chunk_root: Path, policy: str) -> Path:
    dirs = [p for p in chunk_root.iterdir() if p.is_dir() and (p / policy / 'checkpoint.pt').is_file()]
    if len(dirs) != 1:
        raise RuntimeError(f'Expected one policy checkpoint in {chunk_root}, found {len(dirs)}')
    return dirs[0]


def _run_chunk(*, model: str, seed: int, policy: str, state: dict,
               root: Path, config: Path, python: str) -> None:
    key = f'{model}/{seed}/{policy}'
    previous = state['jobs'].get(key, {})
    prior = int(previous.get('rounds_done', 0))
    if prior >= TARGET_ROUNDS:
        return
    next_round = min(TARGET_ROUNDS, prior + CHUNK_ROUNDS)
    chunk_root = root / 'runs' / model / f'seed_{seed}' / policy / f'rounds_{prior+1:03d}_{next_round:03d}'
    if chunk_root.exists():
        # If a previously failed round attempted any private release, the
        # *latest in-progress* write-ahead checkpoint must be reviewed. Never
        # auto-resume from an older ledger or repeat already charged work.
        raise RuntimeError(
            f'Existing uncommitted chunk {chunk_root}. Stopping to avoid '
            'replaying possibly charged DP releases after a crash.'
        )
    chunk_root.mkdir(parents=True, exist_ok=False)
    command = [python, str(RUNNER), '--config', str(config),
               '--seeds', str(seed), '--policies', policy,
               '--rounds', str(TARGET_ROUNDS), '--max-new-rounds', str(next_round-prior),
               '--output-root', str(chunk_root)]
    if prior:
        previous_run = Path(previous['run_dir'])
        if not (previous_run / policy / 'checkpoint.pt').is_file():
            raise RuntimeError(f'Missing source checkpoint for {key}: {previous_run}')
        command.extend(['--resume-from-run', str(previous_run)])
    print(f'\n[RUN] {key}: rounds {prior+1}-{next_round}/{TARGET_ROUNDS}', flush=True)
    with (chunk_root / 'launcher_command.json').open('w', encoding='utf-8') as out:
        json.dump(command, out, indent=2, ensure_ascii=False)
    # subprocess exit status is authoritative; the child writes its own live logs.
    subprocess.run(command, cwd=ROOT, check=True)
    run_dir = _run_dir(chunk_root, policy)
    accuracy, loss = _read_round_metrics(
        run_dir / policy / 'round_metrics.csv', target=next_round, prior=prior
    )
    if next_round == TARGET_ROUNDS:
        summary_file = run_dir / policy / 'summary.json'
        if not summary_file.is_file():
            raise RuntimeError(f'Final summary missing: {summary_file}')
        summary = json.loads(summary_file.read_text(encoding='utf-8'))
        if summary.get('status') != 'completed' or int(summary['rounds']) != TARGET_ROUNDS:
            raise RuntimeError(f'Final summary not completed for {key}')
    state['jobs'][key] = {'rounds_done': next_round, 'run_dir': str(run_dir),
                          'last_accuracy': accuracy, 'last_test_loss': loss}
    _atomic_json(root / 'batch_state.json', state)
    print(f'[OK] {key}: through R{next_round}, acc={accuracy:.3%}, test_loss={loss:.4f}', flush=True)


def _summary(root: Path, state: dict) -> None:
    rows = []
    for key, item in sorted(state['jobs'].items()):
        if item['rounds_done'] != TARGET_ROUNDS:
            continue
        model, seed, policy = key.split('/')
        summary = json.loads((Path(item['run_dir']) / policy / 'summary.json').read_text(encoding='utf-8'))
        rows.append({
            'model': model, 'seed': seed, 'policy': policy,
            'accuracy_percent': round(float(summary['final_test_accuracy'])*100, 4),
            'logical_time_sec': float(summary['total_logical_time']),
            'skip_rate': float(summary['skip_rate']),
            'mode_distribution': summary['mode_distribution'],
            'run_dir': item['run_dir'],
            'objective': 'legacy_fusion_dp', 'training_rounds': TARGET_ROUNDS,
        })
    if not rows:
        return
    path = root / 'batch_results.csv'
    with path.open('w', newline='', encoding='utf-8-sig') as fp:
        writer = csv.DictWriter(fp, rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(f'[SUMMARY] {len(rows)} completed policies: {path}', flush=True)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-root', type=Path, default=ROOT / 'out' / 'stage16_100round_batch')
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026, 2027, 2028])
    parser.add_argument('--models', nargs='+', choices=CONFIGS, default=list(CONFIGS))
    parser.add_argument('--precheck-only', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args(argv)
    if len(set(args.seeds)) != len(args.seeds) or len(set(args.models)) != len(args.models):
        raise ValueError('Duplicate seeds or models')
    configs = {}
    checksums = {}
    for model in args.models:
        config_path = CONFIGS[model]
        config = json.loads(config_path.read_text(encoding='utf-8'))
        validate_config(config, model)
        configs[model] = config_path
        checksums[model] = hashlib.sha256(config_path.read_bytes()).hexdigest()
    signature = {'seeds': args.seeds, 'models': args.models, 'config_sha256': checksums,
                 'rounds': TARGET_ROUNDS, 'chunk': CHUNK_ROUNDS, 'objective': 'legacy_fusion_dp'}
    if args.dry_run:
        for model in args.models:
            command = [sys.executable, str(RUNNER), '--config', str(configs[model]),
                       '--rounds', str(TARGET_ROUNDS), '--max-new-rounds', str(CHUNK_ROUNDS),
                       '--policies', 'full_dynfl', '--seeds', str(args.seeds[0]),
                       '--output-root', str(args.output_root / 'dry_run' / model), '--dry-run']
            subprocess.run(command, cwd=ROOT, check=True)
        print('[DRY RUN] 100-round horizon / 10-round chunk / locked optimizer verified')
        return
    root = args.output_root.resolve()
    path = root / 'batch_state.json'
    if path.exists():
        state = json.loads(path.read_text(encoding='utf-8'))
        if state['signature'] != signature:
            raise RuntimeError('Batch configuration changed since first chunk; use a new output root')
        if state.get('failed_chunk'):
            raise RuntimeError('Earlier chunk failed: manual privacy/checkpoint review required')
    else:
        if root.exists() and any(root.iterdir()):
            raise RuntimeError('Output root nonempty without state; refuse accidental overwrite')
        root.mkdir(parents=True, exist_ok=True)
        for model in args.models:
            shutil.copyfile(configs[model], root / f'config_{model}.json')
        state = {'signature': signature, 'jobs': {}, 'precheck_passed': False}
        _atomic_json(path, state)

    def run_chunk(model: str, seed: int, policy: str):
        try:
            _run_chunk(model=model, seed=seed, policy=policy, state=state,
                       root=root, config=configs[model], python=sys.executable)
        except BaseException as exc:
            state['failed_chunk'] = {'model': model, 'seed': seed,
                                     'policy': policy, 'reason': str(exc)}
            _atomic_json(path, state)
            raise

    if not state['precheck_passed']:
        for model in args.models:
            key = f'{model}/{args.seeds[0]}/full_dynfl'
            if state['jobs'].get(key, {}).get('rounds_done', 0) == 0:
                run_chunk(model, args.seeds[0], 'full_dynfl')
            if state['jobs'][key]['rounds_done'] != CHUNK_ROUNDS:
                raise RuntimeError(f'{key}: precheck incomplete')
        state['precheck_passed'] = True
        _atomic_json(path, state)
        print('[PRECHECK OK] Both models completed 10 active rounds at target horizon T=100', flush=True)
    if args.precheck_only:
        return

    for model in args.models:
        for seed in args.seeds:
            for policy in POLICIES:
                key = f'{model}/{seed}/{policy}'
                while state['jobs'].get(key, {}).get('rounds_done', 0) < TARGET_ROUNDS:
                    run_chunk(model, seed, policy)
                    _summary(root, state)
    _summary(root, state)
    print('[COMPLETED] All locked-baseline jobs finished 100 rounds', flush=True)


if __name__ == '__main__':
    main()
