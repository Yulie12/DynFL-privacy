"""LIIC SGD mechanism-algebra PREDECISION example, synthetic inputs only.

This exercise does not access client gradients or train modes.  The model
parameter norm is obtained from an actual freshly initialized LeNet-5.
Predecision ideal gradient bounds are ASSUMED for this public synthetic
illustration; the example does not establish them for FashionMNIST clients.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from dynfed.liic_online_certificate import planned_batch_sizes, calculate_liic_radius
from dynfed.split_learning import build_full_model


def main() -> None:
    torch.manual_seed(2026)
    model = build_full_model('lenet5', torch.device('cpu'))
    params = [p for p in model.parameters() if p.requires_grad]
    parameter_norm = sum(float(p.detach().square().sum()) for p in params)**.5
    d = sum(p.numel() for p in params)
    batch_sizes = planned_batch_sizes(sample_count=120, epochs=3, batch_size=256)
    proof = {
        'schema': 'dynfl.liic.sgd.v1', 'mode': 'LIIC', 'model': 'lenet5',
        'optimizer': 'torch.optim.SGD', 'momentum': 0,
        'source': 'sample_dp_sgd',
        'evidence': 'PUBLIC SYNTHETIC: assumes external ideal-gradient certificate, not production',
        'training_state_matches_runtime': True,
        'ideal_reference_bound_proved': True,
        'noise_distribution_matches_runtime': True,
        'extra_transformations_absent_or_modeled': True,
        'initial_parameter_norm': parameter_norm,
        'learning_rate': .15, 'ideal_step': .15, 'weight_decay': .0001,
        'gradient_clip_norm': 3., 'sample_noise_multiplier': .03,
        'planned_optimizer_steps': len(batch_sizes), 'batch_sizes': list(batch_sizes),
    }
    print(f'PUBLIC LE NET-5 parameter dimension: {d}')
    print(f'SYNTHETIC plan batches: {batch_sizes}; local epochs=3, no cap')
    # Zero is a valid synthetic center only with a separately proved radius.
    # A REAL run MUST NOT copy 0.02 as a certified bound without evidence.
    assumed_ideal_reference_norm = .05
    assumed_ideal_rms = .02
    base = calculate_liic_radius(
        proof=proof, ideal_reference_norm=assumed_ideal_reference_norm,
        ideal_reference_rms=assumed_ideal_rms, vector_dimension=d,
    )
    print(f'SYNTHETIC LIIC safe clipping-only radius: {base.relative_ideal_rms:.6f}')
    print(f'  gradient drift: {base.clipped_gradient_drift:.6f}')
    print(f'  local DP-SGD accumulated noise RMS: {base.local_dp_sgd_noise_rms:.6f}')
    tighter = dict(proof, gradient_reference_residual_caps=[.2]*len(batch_sizes),
                   gradient_residual_caps_proved=True)
    improved = calculate_liic_radius(
        proof=tighter, ideal_reference_norm=assumed_ideal_reference_norm,
        ideal_reference_rms=assumed_ideal_rms, vector_dimension=d,
    )
    print(f'SYNTHETIC hypothetical proved-residual radius: {improved.relative_ideal_rms:.6f}')
    print('NOT a real-mode certificate. The tighter input caps require independent proof.')
    print('Neither noise contribution should be counted again in aggregate_share_std.')


if __name__ == '__main__':
    main()
