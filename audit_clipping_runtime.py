"""Run original experiments/run_fmnist_lenet5.py with Clipping-only runtime assertions.

Run from project root using serial executor. Does not edit project source.
"""
import math
import runpy
import sys
from pathlib import Path

root = Path.cwd()
runner = root / 'experiments' / 'run_fmnist_lenet5.py'
if not runner.is_file():
    raise SystemExit('Run this script from the DynFL-privacy project root.')
sys.path.insert(0, str(root))

import dynfed.split_learning as split
import dynfed.fmnist_lenet5_dynamic as runner_module

stats = {'workers': 0, 'sample_clipping_calls': 0, 'emb_gradient_dp_calls': 0}
original_worker = runner_module._client_train_worker
original_clipper = split.clip_and_aggregate_per_sample_grads
original_emb = split._protect_tensor_dp
original_grad = split._protect_batched_average_gradient_dp


def require_zero(value, name):
    if value is None or not math.isfinite(float(value)) or float(value) != 0.0:
        raise AssertionError(f'{name} must be exactly 0; got {value!r}')


def check_worker(payload, model_cache=None):
    if payload.get('clipping_only_ablation'):
        if payload.get('privacy_unit') != 'sample':
            raise AssertionError('Clipping-only worker must use sample-level training')
        if payload.get('dp_update_mode') != 'off':
            raise AssertionError('Clipping-only worker must disable client-update DP')
        if not float(payload.get('sample_optimizer_clip_norm', 0)) > 0:
            raise AssertionError('Sample optimizer clip norm must be > 0')
        for key in ('sample_optimizer_noise_multiplier', 'sample_embedding_noise_multiplier', 'sample_label_grad_noise_multiplier'):
            require_zero(payload.get(key), key)
        stats['workers'] += 1
        print(f"[AUDIT worker] client={payload['client_id']} mode={payload['mode']} privacy_unit=sample sigma_optimizer=0 sigma_emb=0 sigma_grad=0 clip={payload['sample_optimizer_clip_norm']}", flush=True)
    return original_worker(payload, model_cache=model_cache)


def check_clipper(*args, **kwargs):
    clip = kwargs.get('clip_norm')
    sigma = kwargs.get('noise_multiplier')
    require_zero(sigma, 'executed per-sample optimizer sigma')
    if clip is None or not math.isfinite(float(clip)) or float(clip) <= 0:
        raise AssertionError(f'Invalid executed clipping norm: {clip!r}')
    stats['sample_clipping_calls'] += 1
    return original_clipper(*args, **kwargs)


def check_emb(*args, **kwargs):
    mechanism = kwargs.get('mechanism', args[1] if len(args) > 1 else None)
    sigma = kwargs.get('noise_multiplier', args[3] if len(args) > 3 else None)
    if mechanism == 'dp':
        require_zero(sigma, 'executed embedding sigma')
        stats['emb_gradient_dp_calls'] += 1
    return original_emb(*args, **kwargs)


def check_grad(*args, **kwargs):
    mechanism = kwargs.get('mechanism', args[1] if len(args) > 1 else None)
    sigma = kwargs.get('noise_multiplier', args[3] if len(args) > 3 else None)
    if mechanism == 'dp':
        require_zero(sigma, 'executed gradient sigma')
        stats['emb_gradient_dp_calls'] += 1
    return original_grad(*args, **kwargs)


runner_module._client_train_worker = check_worker
split.clip_and_aggregate_per_sample_grads = check_clipper
split._protect_tensor_dp = check_emb
split._protect_batched_average_gradient_dp = check_grad

if '--executor' not in sys.argv[1:]:
    raise SystemExit('Pass --executor serial to ensure instrumentation runs in process.')
idx = sys.argv.index('--executor')
if idx + 1 >= len(sys.argv) or sys.argv[idx + 1] != 'serial':
    raise SystemExit('This runtime audit supports only --executor serial.')
if '--policies' not in sys.argv or 'full_dynfl_clipping_only' not in sys.argv:
    raise SystemExit('Pass --policies full_dynfl_clipping_only.')

sys.argv = [str(runner)] + sys.argv[1:]
try:
    runpy.run_path(str(runner), run_name='__main__')
finally:
    print('[AUDIT summary]', stats, flush=True)
if stats['workers'] == 0:
    raise AssertionError('No clipping-only worker was executed')
if stats['sample_clipping_calls'] == 0:
    raise AssertionError('No actual per-sample clipping function was executed')
print('[AUDIT PASS] Actual sample-gradient clipping executed with zero optimizer Gaussian noise. Split emb/grad audited calls:', stats['emb_gradient_dp_calls'])
