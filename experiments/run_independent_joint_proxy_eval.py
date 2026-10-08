"""Run independent official-test paired ranking in a dedicated local-only pass.

This is a calibrated-client SUBCOHORT diagnostic, NOT a full 100-client
end-to-end privacy proof. Uses model/state selection at round 0, captures
private updates only in local RAM, and writes scalar scores to --output.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.run_paper_config import build_command


def build_independent_command(args):
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if cfg.get('privacy', {}).get('unit') != 'sample':
        raise ValueError('Sample-DP config required')
    learning = cfg.get('learning', {})
    if learning.get('objective') != 'joint_calibration' or not learning.get('calibration_path'):
        raise ValueError('calibrated joint learning objective required')
    calibration = Path(learning['calibration_path'])
    if not calibration.is_absolute():
        calibration = ROOT / calibration
    if not calibration.exists():
        raise FileNotFoundError(calibration)
    if args.trials < 2 or args.clients < 1 or args.sample_limit < 1:
        raise ValueError('trials>=2, clients>=1, sample-limit>=1 required')
    cfg = copy.deepcopy(cfg)
    cfg['policies'] = ['full_dynfl']
    cfg['seeds'] = [int(args.selection_seed)]
    cfg['output_root'] = str((ROOT / 'out' / 'independent_joint_proxy_capture').resolve())
    cfg.pop('calibration_capture', None)
    cfg['he']['execution'] = 'profiled'  # local learning diagnostic, not a security claim
    cfg['he']['require_real_he'] = False
    cfg['learning']['calibration_path'] = str(calibration.resolve())
    cfg['learning']['missing_policy'] = 'error'
    cfg['learning']['state_key'] = 'auto_round'
    # Keep the 100-round privacy accounting horizon; exit before first training
    # round through the dedicated environment-controlled capture bridge.
    command = build_command(cfg, seed=int(args.selection_seed), policies=['full_dynfl'],
                            rounds=None)
    env = os.environ.copy()
    env.update({
        'DYNFL_INDEPENDENT_EVAL_OUTPUT': str(args.output.resolve()),
        'DYNFL_INDEPENDENT_EVAL_TRIALS': str(args.trials),
        'DYNFL_INDEPENDENT_EVAL_SEED': str(args.evaluation_seed),
        'DYNFL_INDEPENDENT_EVAL_CLIENTS': str(args.clients),
        'DYNFL_INDEPENDENT_EVAL_SAMPLE_LIMIT': str(args.sample_limit),
    })
    return command, env


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, default=ROOT / 'configs/paper_v34_cifar10_resnet18_joint_proxy.json')
    p.add_argument('--output', type=Path, default=ROOT / 'out/independent_joint_proxy_pairs.json')
    p.add_argument('--selection-seed', type=int, default=40)
    p.add_argument('--evaluation-seed', type=int, default=90217)
    p.add_argument('--trials', type=int, default=2)
    p.add_argument('--clients', type=int, default=3)
    p.add_argument('--sample-limit', type=int, default=64)
    p.add_argument('--dry-run', action='store_true')
    args = p.parse_args()
    command, env = build_independent_command(args)
    print(json.dumps({'output': env['DYNFL_INDEPENDENT_EVAL_OUTPUT'],
                      'test_seed': env['DYNFL_INDEPENDENT_EVAL_SEED'],
                      'client_limit': env['DYNFL_INDEPENDENT_EVAL_CLIENTS'],
                      'trials': env['DYNFL_INDEPENDENT_EVAL_TRIALS'],
                      'mode': 'local_only_subcohort'}, indent=2), flush=True)
    if args.dry_run:
        print(subprocess.list2cmdline(command), flush=True)
        return
    subprocess.run(command, cwd=ROOT, env=env, check=True)
    if not args.output.exists():
        raise RuntimeError('evaluation subprocess exited without writing independent scores')
    print(f'independent ranking saved: {args.output}', flush=True)


if __name__ == '__main__':
    main()
