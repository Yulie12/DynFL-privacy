"""Exact optimizer-algebra checks; never claim empirical online-J certification."""
from __future__ import annotations
import csv
import json
import math
from pathlib import Path
import tempfile
import unittest

from dynfed.online_j_liic_envelope import (
    conditional_liic_j_upper, liic_update_envelope, sample_weighted_liic_envelope,
    fmnist_lenet5_gradient_norm_cap,
)
from examples.report_real_liic_envelope import report_run


def metadata(**overrides):
    m = {
        'schema': 'dynfl.liic.sgd.v1', 'mode': 'LIIC', 'model': 'lenet5',
        'optimizer': 'torch.optim.SGD', 'momentum': 0,
        'source': 'sample_dp_sgd', 'initial_parameter_norm': 4.,
        'learning_rate': .15, 'weight_decay': .0001,
        'gradient_clip_norm': 3., 'sample_noise_multiplier': .02,
        'batch_sizes': [256, 256, 1] * 3, 'planned_optimizer_steps': 9,
        'vector_dimension': 61706,
    }
    m.update(overrides)
    return m


class LIICEnvelopeTests(unittest.TestCase):
    def test_clipped_no_dp_deterministic_worst_case(self):
        m = metadata(source='clipped_sgd', gradient_clip_norm=5.,
                     sample_noise_multiplier=0., weight_decay=0.,
                     initial_parameter_norm=0., batch_sizes=[32]*3,
                     planned_optimizer_steps=3)
        b = liic_update_envelope(m, local_epochs=3)
        self.assertAlmostEqual(b.bound_rms, 3 * .15 * 5.)
        self.assertEqual(b.local_dp_noise_rms, 0)
        self.assertEqual(b.client_samples, 32)
        self.assertFalse(b.reference_gradient_certified)

    def test_sample_dp_last_partial_batch_exact(self):
        b = liic_update_envelope(metadata(weight_decay=0.), local_epochs=3)
        std_sum = 2 * 3 * .02
        expected_noise = math.sqrt(61706 * (3 * (.15 * std_sum)**2 *
                                             (1/256**2 + 1/256**2 + 1.)))
        self.assertAlmostEqual(b.local_dp_noise_rms, expected_noise)
        self.assertEqual(b.client_samples, 513)
        self.assertEqual(b.steps, 9)
        self.assertAlmostEqual(b.clipped_gradient_part, .15 * 9 * 3)

    def test_with_weight_decay(self):
        m = metadata(batch_sizes=[120]*3, planned_optimizer_steps=3,
                     sample_noise_multiplier=0., learning_rate=.25,
                     weight_decay=.4, initial_parameter_norm=2.)
        b = liic_update_envelope(m, local_epochs=3)
        alpha = 1-.25*.4
        self.assertAlmostEqual(b.decay_part, (1-alpha**3)*2)
        self.assertAlmostEqual(b.clipped_gradient_part,
                               3*.25*(1+alpha+alpha*alpha))

    def test_explicit_epochs_needed_repeated_schedule_is_ambiguous(self):
        with self.assertRaisesRegex(ValueError, 'explicit positive local_epochs'):
            liic_update_envelope(metadata(), local_epochs=2)
        with self.assertRaisesRegex(ValueError, 'repeat'):
            liic_update_envelope(metadata(batch_sizes=[7,8,7],
                                          planned_optimizer_steps=3), local_epochs=3)

    def test_reject_unmodelled_paths_and_bad_sigma(self):
        for change in (
            {'momentum': .9}, {'vector_dimension': 61707},
            {'source': 'clipped_sgd'}, {'sample_noise_multiplier': -1},
            {'batch_sizes': [0]*9}, {'weight_decay': 20.},
        ):
            with self.subTest(change=change), self.assertRaises(ValueError):
                liic_update_envelope(metadata(**change), local_epochs=3)

    def test_weighted_bound_uses_sample_counts(self):
        left = liic_update_envelope(metadata(batch_sizes=[100]*3,
                                             planned_optimizer_steps=3), local_epochs=3)
        right = liic_update_envelope(metadata(batch_sizes=[200]*3,
                                              planned_optimizer_steps=3), local_epochs=3)
        expected = (100*left.bound_rms + 200*right.bound_rms)/300
        self.assertAlmostEqual(sample_weighted_liic_envelope([(0,left),(1,right)]), expected)
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            sample_weighted_liic_envelope([(0,left),(0,right)])

    def test_minkowski_J_envelope_requires_external_proof(self):
        args = dict(update_envelope=2., ideal_step=.15, global_gradient_norm_cap=4.,
                    aggregate_noise_rms=.3, numeric_rms=.1,
                    gradient_cap_independently_proved=True,
                    linear_decomposition_independently_proved=True,
                    noise_and_numeric_bounds_independently_proved=True,
                    evidence='external analytic per-round proof')
        self.assertAlmostEqual(conditional_liic_j_upper(**args), (2+.6+.3+.1)**2)
        with self.assertRaisesRegex(ValueError, 'missing independently'):
            conditional_liic_j_upper(**dict(args, global_gradient_norm_cap=None))
        for key in ('gradient_cap_independently_proved',
                    'linear_decomposition_independently_proved',
                    'noise_and_numeric_bounds_independently_proved'):
            with self.subTest(key=key), self.assertRaisesRegex(ValueError,'evidence missing'):
                conditional_liic_j_upper(**dict(args, **{key:False}))
        # A global cap on ||grad F|| cannot be derived from *any* clipping
        # metadata: for linear loss g(theta)=M*theta, M can be arbitrarily
        # large while the clipped SGD update remains bounded by C.

    def test_architecture_gradient_bound_matches_real_autograd(self):
        # Run actual LeNet5 CE gradient, not an algebra-only synthetic network.
        import torch
        from dynfed.split_learning import build_full_model
        torch.manual_seed(18)
        model = build_full_model('lenet5', torch.device('cpu'))
        R = math.sqrt(sum(float(p.detach().double().square().sum())
                          for p in model.parameters()))
        G = fmnist_lenet5_gradient_norm_cap(
            model_parameter_norm_upper=R*(1+1e-6)+1e-8,
        )
        for label in (0,3,9):
            model.zero_grad(set_to_none=True)
            x = (torch.rand(1,1,28,28)*4.5-2.25)
            import torch.nn.functional as F
            loss = F.cross_entropy(model(x),torch.tensor([label]))
            loss.backward()
            actual = math.sqrt(sum(float(p.grad.detach().double().square().sum())
                                   for p in model.parameters()))
            self.assertLessEqual(actual, G)
        with self.assertRaisesRegex(ValueError,'positive'):
            fmnist_lenet5_gradient_norm_cap(model_parameter_norm_upper=R,
                                            normalized_pixel_abs_bound=0)

    def test_read_existing_shadow_without_new_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path/'summary.json').write_text(json.dumps({
                'dataset':'Fashion-MNIST', 'model':'lenet5',
            }))
            with (path/'client_decisions.csv').open('w',newline='',encoding='utf8') as fh:
                writer=csv.DictWriter(fh,fieldnames=['round','client_id','mode'])
                writer.writeheader()
                for r in range(2):
                    writer.writerow(dict(round=r,client_id=0,mode='LIIC'))
                    writer.writerow(dict(round=r,client_id=1,mode='SKIP'))
            for r in range(2):
                shadow={
                    'schema': 'dynfl.liic.predecision_runtime_audit.v1',
                    'round_index': r, 'available_before_round': r,
                    'formal_online_j_certified': False,
                    'training_data_accessed':False,
                    'liic_candidates': [{
                        'status': 'runtime_optimizer_metadata_only_NOT_A_J_CERTIFICATE',
                        'client_id':0,'optimizer':metadata(batch_sizes=[120]*3,
                                                          planned_optimizer_steps=3),
                    }],
                }
                (path/f'online_j_liic_shadow_round_{r:03d}.json').write_text(json.dumps(shadow))
            rows=report_run(path,local_epochs=3)
            self.assertEqual(len(rows),2)
            self.assertEqual(rows[0]['liic_clients'],1)
            self.assertFalse(rows[0]['J_formal_certified'])
            self.assertGreater(rows[0]['fashion_mnist_lenet5_gradient_norm_architecture_cap'], 0)
            self.assertGreater(rows[0]['selected_LIIC_sample_weighted_local_update_rms_envelope'],0)
            with self.assertRaisesRegex(ValueError,'local_epochs'):
                report_run(path,local_epochs=2)

if __name__ == '__main__': unittest.main()
