"""Final joint-proxy acceptance gate. Never reuse calibration moments as independent paired trials.

Usage:
  python experiments/finalize_joint_proxy.py --calibration calibration/paper_v34_cifar10_resnet18_joint.pt
  python experiments/finalize_joint_proxy.py --calibration calibration/paper_v34_cifar10_resnet18_joint.pt --evaluation independent_pairs.json

The evaluation JSON must meet the existing validate_joint_proxy_ranking.py contract.
The independence declaration is user supplied; this script cannot independently prove disjointness.
No calibration tensors or evaluation trajectories are included in the report.
"""
from __future__ import annotations
import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from experiments.validate_joint_proxy_ranking import evaluate


def summarize_calibration(path: Path) -> dict:
    # Safe unpickling only. Do not silently fall back to weights_only=False.
    table = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(table, dict) or table.get("format") != "dynfl_joint_update_calibration_v1":
        raise ValueError("unexpected joint calibration table format")
    entries = table.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError("missing calibration entries")
    modes = collections.Counter()
    clients = set()
    for entry in entries:
        if not isinstance(entry, dict) or not {'state_key','mode','client_id','clean_update_mean','bias_mean','variance_trace','sample_count'} <= entry.keys():
            raise ValueError("malformed calibration entry")
        modes[str(entry['mode'])] += 1
        if entry['client_id'] is not None:
            clients.add(int(entry['client_id']))
    metadata = table.get('metadata') or {}
    return {
        'sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
        'entry_count': len(entries),
        'paired_observations_reported': metadata.get('capture_pair_count'),
        'client_count': len(clients),
        'modes': dict(sorted(modes.items())),
        'source': metadata.get('source', 'unknown'),
        'calibration_workload': metadata.get('calibration_workload', 'unknown'),
        'privacy_note': 'Calibration release is not established as differentially private; held-out private data may be involved.',
        'has_independent_paired_trials': False,
    }


def finalize(calibration: Path, evaluation: Path | None) -> dict:
    summary = summarize_calibration(calibration)
    result = {
        'version': 'dynfl_final_joint_proxy_gate_v1',
        'calibration': summary,
        'end_to_end_sample_dp': 'not_established',
        'spearman_validation': {'status': 'pending_independent_heldout_paired_trajectories'},
        'release_note': 'Do not export client-private calibration/evaluation trajectories to an untrusted server without separate privacy justification.',
    }
    if evaluation is not None:
        data = json.loads(evaluation.read_text(encoding='utf-8'))
        score = evaluate(data)
        result['spearman_validation'] = score
        result['spearman_validation']['independence_verification'] = 'declared_by_input_not_cryptographically_verified'
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--evaluation', type=Path, help='Separate, independently generated paired-trajectory JSON')
    parser.add_argument('--output', type=Path, default=Path('out/final_joint_proxy_acceptance.json'))
    args = parser.parse_args()
    report = finalize(args.calibration, args.evaluation)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    status = report['spearman_validation']['status']
    print('Final acceptance report:', args.output)
    print('Calibration entries:', report['calibration']['entry_count'])
    print('Independent Spearman status:', status)
    if status == 'computed_not_privacy_proof':
        print('Spearman rho:', report['spearman_validation']['spearman_rho'])
    else:
        print('No independent evaluation file supplied; no experimental correlation claimed.')

if __name__ == '__main__':
    main()
