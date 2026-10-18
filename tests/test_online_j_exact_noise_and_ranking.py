"""Tighter but proof-gated conditional J lower bounds (never empirical noise floors)."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from dynfed.online_j_bound import (
    OnlineBoundResult, certificate_candidate_key, certified_j_interval_order,
    evaluate_online_bound,
)
from dynfed.selection import Candidate, SelectionConfig, _candidate_reaches_cloud, _evaluate_profile


class OnlineJNoiseFloorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / 'cert.json'
        self.candidate = Candidate(mode='LIIC', mechanisms={'upd': 'he3'}, time=1.0,
                                   accuracy=.5, risk=0, epsilon_used=0,
                                   communication_volume=1, feasible_resource=True,
                                   feasible_privacy=True, feasible_risk=True, feasible_time=True)
        self.cert = {'runtime_mode_equivalence_verified': True,
                     'local_dp_included_in_mode_rms': True,
                     'relative_ideal_rms': 0., 'aggregate_share_std': 2.}
        self.state = {
            'round_index': 0, 'available_before_round': 0,
            'evidence': 'Symbolic and fully conditional synthetic example',
            'cloud_step': 1., 'numeric_rms': .1,
            'aggregation_rule': 'cloud_participant_sample_weighted',
            'population': {'0': {'samples': 1, 'ideal_reference': [0., 0.], 'ideal_rms': 0.}},
            'clients': {'0': {'candidates': {certificate_candidate_key(self.candidate): self.cert}}},
            'assumptions': dict.fromkeys((
                'global_objective_is_population_sample_weighted',
                'linear_client_contributions_match_runtime',
                'ideal_reference_rms_bounds_proved',
                'mode_relative_ideal_bounds_proved',
                'noise_shares_drawn_before_secagg_and_weight_fixed',
                'noise_conditionally_zero_mean_and_uncorrelated',
                'aggregate_share_second_moment_bounds_proved',
                'numeric_rms_bound_proved',
            ), True),
        }

    def evaluate(self):
        self.path.write_text(json.dumps({
            'schema': 'dynfl.online_j.v2', 'target': 'ideal_full_gradient_one_step',
            'rounds': {'round:0': self.state},
        }), encoding='utf-8')
        return evaluate_online_bound(
            manifest_path=str(self.path), round_index=0,
            profile={0: self.candidate}, client_samples={0: 1},
            admitted_client_ids=(0,), reaches_cloud=_candidate_reaches_cloud,
        )

    def enable_exact(self):
        self.state['aggregate_noise_variance_exactly_proved'] = True
        self.state['assumptions']['aggregate_noise_orthogonal_to_clean_update_given_predecision'] = True
        self.cert['aggregate_share_std_exactly_proved'] = True

    def test_upper_variance_does_not_imply_positive_lower(self):
        r = self.evaluate()
        self.assertEqual(r.lower, 0.)
        self.assertEqual(r.q_aggregate_proved_floor, 0.)
        self.assertAlmostEqual(r.q_aggregate, 8.)

    def test_exact_gaussian_energy_gives_positive_lower(self):
        self.enable_exact()
        r = self.evaluate()
        self.assertAlmostEqual(r.q_aggregate_proved_floor, 8.)
        self.assertAlmostEqual(r.lower, (math.sqrt(8.) - .1)**2)
        self.assertGreaterEqual(r.upper, r.lower)

    def test_exact_variance_without_independence_fails_closed(self):
        self.enable_exact()
        del self.state['assumptions']['aggregate_noise_orthogonal_to_clean_update_given_predecision']
        with self.assertRaisesRegex(ValueError, 'orthogonality'):
            self.evaluate()

    def test_exact_variance_without_client_proof_fails_closed(self):
        self.enable_exact()
        del self.cert['aggregate_share_std_exactly_proved']
        with self.assertRaisesRegex(ValueError, 'separately proved'):
            self.evaluate()

    def test_invalid_opt_in_type_rejected(self):
        self.state['aggregate_noise_variance_exactly_proved'] = 'true'
        with self.assertRaisesRegex(ValueError, 'boolean'):
            self.evaluate()

    def test_blockwise_and_noise_lower_are_combined(self):
        self.enable_exact()
        self.state['population']['0']['ideal_reference'] = [2., 0.]
        self.state['population']['0']['ideal_rms_by_block'] = [0., 0.]
        self.state['parameter_blocks'] = [
            {'name': 'x', 'start': 0, 'end': 1},
            {'name': 'y', 'start': 1, 'end': 2},
        ]
        self.cert['relative_ideal_rms_by_block'] = [0., 0.]
        # All-client cloud_step==1 -> center cancels: only exact DP noise floor.
        r = self.evaluate()
        self.assertAlmostEqual(r.lower, (math.sqrt(8.)-.1)**2)

    def test_noise_floor_reaches_actual_selector_audit(self):
        self.enable_exact()
        expected = self.evaluate()
        config = SelectionConfig(
            num_clients=1, num_edges=1,
            learning_objective='online_upper_bound',
            online_bound_manifest_path=str(self.path),
        )
        ev = _evaluate_profile(
            config, {0: self.candidate}, {0: 1.}, {0: 0}, {},
            flow_objectives=((0,), 1.0),
        )
        self.assertAlmostEqual(ev.system_learning_error, expected.upper)
        self.assertAlmostEqual(ev.online_j_lower_bound, expected.lower)
        self.assertAlmostEqual(ev.online_j_proved_noise_floor, 8.)

    def test_pairwise_interval_ranking_only_if_disjoint(self):
        def r(lo, hi):
            return OnlineBoundResult(upper=hi, lower=lo, central_norm=0.,
                                     radius=0., ideal_radius=0., q_aggregate=0.,
                                     numeric_radius=0., cloud_client_ids=(0,))
        self.assertEqual(certified_j_interval_order(r(1, 2), r(3, 4)), 'left')
        self.assertEqual(certified_j_interval_order(r(3, 4), r(1, 2)), 'right')
        self.assertEqual(certified_j_interval_order(r(1, 3), r(2, 4)), 'undetermined')
        self.assertEqual(certified_j_interval_order(r(1, 2), r(2, 3)), 'undetermined')
        with self.assertRaisesRegex(ValueError, 'invalid'):
            certified_j_interval_order(r(3, 2), r(1, 2))


if __name__ == '__main__':
    unittest.main()
