"""Opt-in online J conditional certificate: algebra, fail-closed and selector route."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from dynfed.online_j_bound import evaluate_online_bound, certificate_candidate_key
from dynfed.selection import (
    Candidate, SelectionConfig, _candidate_reaches_cloud,
    _evaluate_profile, _selection_learning_cost,
)


def c(mode='LIIC', mechanism='he3'):
    return Candidate(
        mode=mode, mechanisms={'upd': mechanism}, time=1.0, accuracy=0.5,
        risk=0.1, epsilon_used=0.0, communication_volume=1.0,
        feasible_resource=True, feasible_privacy=True, feasible_risk=True,
        feasible_time=True,
    )


def manifest_for(clients):
    return {
        'schema': 'dynfl.online_j.v1',
        'target': 'ideal_full_gradient_one_step',
        'rounds': {'round:0': {
            'round_index': 0, 'available_before_round': 0,
            'evidence': 'Synthetic proof-controlled quadratic test, NOT real DP/training certification',
            'reference_gradient': [3.0, 4.0],
            'gradient_rms_error': 0.02,
            'ideal_step': 0.1, 'cloud_step': 1.0, 'numeric_rms': 0.005,
            'aggregation_rule': 'cloud_participant_sample_weighted',
            'assumptions': {
                'weighted_client_contributions_match_runtime': True,
                'gradient_reference_rms_bound_proved': True,
                'mode_rms_bounds_proved': True,
                'noise_shares_drawn_before_secagg_and_weight_fixed': True,
                'noise_conditionally_zero_mean_and_uncorrelated': True,
                'numeric_rms_bound_proved': True,
            },
            'clients': clients,
            'no_cloud_update_is_zero_verified': True,
        }}
    }


class OnlineBoundTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'online.json'
        self.a = c('LIIC')
        self.b = c('LIIEIIIC')
        self.profile = {0: self.a, 1: self.b}
        self.samples = {0: 6., 1: 4.}
        self.clients = {}
        for cid, candidate, step, radius, sigma in [(0, self.a, 0.1, 0.01, 0.03), (1, self.b, 0.2, 0.02, 0.01)]:
            self.clients[str(cid)] = {'candidates': {
                certificate_candidate_key(candidate): {
                    'step_scale': step,
                    'mode_rms': radius,
                    'aggregate_share_std': sigma,
                    'runtime_mode_equivalence_verified': True,
                    'local_dp_included_in_mode_rms': True,
                },
            }}
        self.write(manifest_for(self.clients))

    def write(self, data):
        self.path.write_text(json.dumps(data), encoding='utf-8')

    def evaluate(self, profile=None, round_index=0):
        return evaluate_online_bound(
            manifest_path=str(self.path), round_index=round_index,
            profile=self.profile if profile is None else profile,
            client_samples=self.samples, admitted_client_ids=(0, 1),
            reaches_cloud=_candidate_reaches_cloud,
        )

    def test_exact_structural_bound_and_preaggregation_q(self):
        v = self.evaluate()
        center = abs(.1 - (.6 * .1 + .4 * .2)) * 5
        radius = .1 * .02 + .6 * .01 + .4 * .02
        q = 2 * (.6 ** 2 * .03 ** 2 + .4 ** 2 * .01 ** 2)
        expected = (math.sqrt((center + radius)**2 + q) + .005)**2
        self.assertAlmostEqual(v.central_norm, center)
        self.assertAlmostEqual(v.radius, radius)
        self.assertAlmostEqual(v.q_aggregate, q)
        self.assertAlmostEqual(v.upper, expected)

    def test_missing_round_fails_closed(self):
        with self.assertRaisesRegex(ValueError, 'missing ONLINE'):
            self.evaluate(round_index=1)

    def test_missing_mode_certificate_fails_closed(self):
        profile = {0: self.a, 1: c('LIIC', 'dp')}
        with self.assertRaisesRegex(ValueError, 'missing certificate'):
            self.evaluate(profile=profile)

    def test_unverified_noise_fails_closed(self):
        payload = manifest_for(self.clients)
        payload['rounds']['round:0']['assumptions']['noise_conditionally_zero_mean_and_uncorrelated'] = False
        self.write(payload)
        with self.assertRaisesRegex(ValueError, 'not certified'):
            self.evaluate()

    def test_nonfinite_input_fails_closed(self):
        payload = manifest_for(self.clients)
        payload['rounds']['round:0']['clients']['0']['candidates'][certificate_candidate_key(self.a)]['mode_rms'] = float('nan')
        self.write(payload)
        with self.assertRaisesRegex(ValueError, 'finite nonnegative'):
            self.evaluate()

    def test_selector_uses_same_online_objective_not_old_proxy(self):
        config = SelectionConfig(
            num_clients=2, num_edges=1, learning_objective='online_upper_bound',
            online_bound_manifest_path=str(self.path),
        )
        direct = self.evaluate().upper
        cost = _selection_learning_cost(config, self.profile, self.samples, {0: 0, 1: 0}, (0, 1))
        self.assertAlmostEqual(cost, direct)
        ev = _evaluate_profile(config, self.profile, self.samples, {0: 0, 1: 0}, {},
                               flow_objectives=((0, 1), 2.0))
        self.assertAlmostEqual(ev.system_learning_error, direct)
        self.assertEqual(ev.learning_proxy_source, 'online_upper_bound_provided_conditional_certificates')

    def test_no_config_fallback_and_manifest_required(self):
        with self.assertRaisesRegex(ValueError, 'online_bound_manifest_path'):
            SelectionConfig(learning_objective='online_upper_bound')

    def test_no_cloud_requires_evidence(self):
        payload = manifest_for(self.clients)
        payload['rounds']['round:0']['no_cloud_update_is_zero_verified'] = False
        self.write(payload)
        with self.assertRaisesRegex(ValueError, 'no-Cloud'):
            self.evaluate(profile={0: c('LIIE'), 1: c('LIE')})

    def test_unadmitted_cloud_group_not_counted(self):
        # Only client 0 contributes when 1 is not admitted.
        bound = evaluate_online_bound(
            manifest_path=str(self.path), round_index=0,
            profile=self.profile, client_samples=self.samples,
            admitted_client_ids=(0,), reaches_cloud=_candidate_reaches_cloud,
        )
        self.assertEqual(bound.cloud_client_ids, (0,))
        self.assertAlmostEqual(bound.central_norm, 0.)


class CoupledV2BoundTests(unittest.TestCase):
    """No oracle values or round-t gradients enter the predecision estimator."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / "coupled.json"
        self.a = c("LIIC")
        self.b = c("LIIEIIIC")
        self.candidate_profile = {0: self.a, 1: self.b}
        self.samples = {0: 6., 1: 4.}
        self.payload = {
            "schema": "dynfl.online_j.v2",
            "target": "ideal_full_gradient_one_step",
            "rounds": {"round:0": {
                "round_index": 0, "available_before_round": 0,
                "evidence": "Proved symbolic two-client conditional-balls benchmark; synthetic only",
                "cloud_step": 1.0, "numeric_rms": 0.0,
                "aggregation_rule": "cloud_participant_sample_weighted",
                "mode_uncertainty": "rms",
                "assumptions": {
                    "global_objective_is_population_sample_weighted": True,
                    "linear_client_contributions_match_runtime": True,
                    "ideal_reference_rms_bounds_proved": True,
                    "mode_relative_ideal_bounds_proved": True,
                    "noise_shares_drawn_before_secagg_and_weight_fixed": True,
                    "noise_conditionally_zero_mean_and_uncorrelated": True,
                    "aggregate_share_second_moment_bounds_proved": True,
                    "numeric_rms_bound_proved": True,
                },
                "population": {
                    "0": {"samples": 6.0, "ideal_reference": [1.0, 0.0], "ideal_rms": 0.01},
                    "1": {"samples": 4.0, "ideal_reference": [1.0, 0.0], "ideal_rms": 0.02},
                },
                "clients": {
                    "0": {"candidates": {certificate_candidate_key(self.a): {
                        "relative_ideal_rms": .03, "aggregate_share_std": .0,
                        "runtime_mode_equivalence_verified": True,
                        "local_dp_included_in_mode_rms": True,
                    }}},
                    "1": {"candidates": {certificate_candidate_key(self.b): {
                        "relative_ideal_rms": .02, "aggregate_share_std": .0,
                        "runtime_mode_equivalence_verified": True,
                        "local_dp_included_in_mode_rms": True,
                    }}},
                },
                "no_cloud_update_is_zero_verified": True,
            }}
        }
        self.write()

    @property
    def state(self):
        return self.payload["rounds"]["round:0"]

    def write(self):
        self.path.write_text(json.dumps(self.payload), encoding="utf-8")

    def score(self, profile=None, admission=(0, 1)):
        return evaluate_online_bound(
            manifest_path=str(self.path), round_index=0,
            profile=self.candidate_profile if profile is None else profile,
            client_samples=self.samples, admitted_client_ids=admission,
            reaches_cloud=_candidate_reaches_cloud,
        )

    def test_all_participating_structural_center_exactly_cancels(self):
        # Both gradient references agree but are nonzero. The full-population
        # ideal update cancels, instead of paying twice for its large norm.
        result = self.score()
        self.assertAlmostEqual(result.central_norm, 0.0)
        self.assertAlmostEqual(result.reference_radius, 0.0)
        self.assertAlmostEqual(result.mode_radius, .6 * .03 + .4 * .02)
        self.assertAlmostEqual(result.upper, (.6 * .03 + .4 * .02)**2)
        self.assertEqual(result.source, "online_coupled_v2_rms_conditional_certificates")

    def test_no_cloud_conditional_lower_and_upper(self):
        # No Cloud update is explicitly verified to be zero. Ideal vector
        # remains [1,0], uncertainty s=.014, so learning J is near 1.
        result = self.score(profile={0: c("LIIE"), 1: c("LIE")})
        self.assertAlmostEqual(result.central_norm, 1.0)
        self.assertAlmostEqual(result.radius, .6 * .01 + .4 * .02)
        self.assertAlmostEqual(result.lower, (1 - .014)**2)
        self.assertAlmostEqual(result.upper, (1 + .014)**2)

    def test_coupled_weights_and_unselected_ideal_contribution(self):
        result = self.score(admission=(0,))
        # coeffs: (a0-p0) = .4; (a1-p1) = -.4; identical refs cancel.
        self.assertAlmostEqual(result.central_norm, 0.0)
        self.assertAlmostEqual(result.reference_radius, .4 * .01 + .4 * .02)
        self.assertAlmostEqual(result.mode_radius, .03)

    def test_different_client_directions_reveal_participation_penalty(self):
        self.state["population"]["1"]["ideal_reference"] = [0.0, 1.0]
        self.write()
        result = self.score(admission=(0,))
        self.assertAlmostEqual(result.central_norm, math.sqrt(.4**2 + .4**2))
        self.assertLess(result.lower, result.upper)
        self.assertGreater(result.lower, 0.0)

    def test_manifest_population_must_match_real_sample_weights(self):
        self.state["population"]["1"]["samples"] = 7.0
        self.write()
        with self.assertRaisesRegex(ValueError, "sample mass differs"):
            self.score()

    def test_non_covered_population_fails_closed(self):
        self.state["population"].pop("1")
        self.write()
        with self.assertRaisesRegex(ValueError, "population IDs"):
            self.score()

    def test_missing_mode_specific_certificate_fails_closed(self):
        del self.state["clients"]["1"]["candidates"][certificate_candidate_key(self.b)]
        self.write()
        with self.assertRaisesRegex(ValueError, "missing certificate"):
            self.score()

    def test_rejects_unproved_mode_decomposition(self):
        self.state["assumptions"]["linear_client_contributions_match_runtime"] = False
        self.write()
        with self.assertRaisesRegex(ValueError, "not certified"):
            self.score()

    def test_rejects_fake_posterior_round_certificate(self):
        self.state["available_before_round"] = 1
        self.write()
        with self.assertRaisesRegex(ValueError, "predecision round"):
            self.score()

    def test_rms_bound_allows_arbitrary_correlated_mode_errors(self):
        # Both actual mode errors may have the same sign; triangle bound must
        # pay (.6*e0+.4*e1)^2, not just sum of variances.
        self.state["population"]["0"]["ideal_rms"] = 0.
        self.state["population"]["1"]["ideal_rms"] = 0.
        self.write()
        result = self.score()
        actual_difference = .6 * .03 + .4 * .02
        self.assertGreaterEqual(result.upper, actual_difference**2)

    def test_bias_variance_squares_only_centered_innovations(self):
        self.state["mode_uncertainty"] = "bias_variance"
        self.state["assumptions"]["mode_fluctuations_centered_given_ideal"] = True
        self.state["assumptions"]["mode_fluctuations_cross_client_orthogonal_given_ideal"] = True
        for cid, cand, bias, var in (
            (0, self.a, .01, .0004), (1, self.b, .02, .0009)
        ):
            cert = self.state["clients"][str(cid)]["candidates"][certificate_candidate_key(cand)]
            cert.pop("relative_ideal_rms")
            cert["relative_ideal_bias"] = bias
            cert["relative_ideal_variance"] = var
        self.write()
        res = self.score()
        b = .6 * .01 + .4 * .02
        v = .6**2 * .0004 + .4**2 * .0009
        self.assertAlmostEqual(res.upper, b*b+v)
        self.assertAlmostEqual(res.mode_variance_second_moment, v)
        self.assertEqual(res.source, "online_coupled_v2_bias_variance_conditional_certificates")

    def test_bias_variance_requires_actual_orthogonality_certificate(self):
        self.state["mode_uncertainty"] = "bias_variance"
        self.write()
        with self.assertRaisesRegex(ValueError, "conditional centering"):
            self.score()

    def test_shared_direction_compression_matches_full_vectors(self):
        full = self.score(admission=(0,))
        self.state["reference_direction"] = [1.0, 0.0]
        for row in self.state["population"].values():
            row.pop("ideal_reference")
            row["ideal_reference_scale"] = 1.0
        self.write()
        compact = self.score(admission=(0,))
        self.assertAlmostEqual(compact.upper, full.upper)
        self.assertAlmostEqual(compact.lower, full.lower)
        self.assertAlmostEqual(compact.reference_radius, full.reference_radius)

    def test_shared_direction_different_signed_scales(self):
        self.state["reference_direction"] = [2.0, -1.0]
        scales = {"0": 0.5, "1": -1.0}
        for cid, row in self.state["population"].items():
            row.pop("ideal_reference")
            row["ideal_reference_scale"] = scales[cid]
        self.write()
        result = self.score(admission=(0,))
        # (a-p)=(+.4,-.4) ; effective shared scale = .4*.5 + (-.4)*(-1)=.6
        self.assertAlmostEqual(result.central_norm, .6 * math.sqrt(5.0))

    def test_shared_direction_without_direction_rejected(self):
        self.state["population"]["0"].pop("ideal_reference")
        self.state["population"]["0"]["ideal_reference_scale"] = 1.0
        self.write()
        with self.assertRaisesRegex(ValueError, "requires reference_direction"):
            self.score()

    def test_v2_selector_uses_coupled_j_and_logs_lower_bound(self):
        config = SelectionConfig(
            num_clients=2, num_edges=1, learning_objective="online_upper_bound",
            online_bound_manifest_path=str(self.path),
        )
        res = self.score()
        cost = _selection_learning_cost(config, self.candidate_profile, self.samples, {0: 0, 1: 0}, (0, 1))
        self.assertAlmostEqual(res.upper, cost)
        ev = _evaluate_profile(config, self.candidate_profile, self.samples, {0: 0, 1: 0}, {},
                               flow_objectives=((0, 1), 2.0))
        self.assertAlmostEqual(ev.system_learning_error, res.upper)
        self.assertAlmostEqual(ev.online_j_lower_bound, res.lower)
        self.assertEqual(ev.learning_proxy_source, res.source)

    def test_explicit_share_noise_moment_only_once(self):
        cert = self.state["clients"]["0"]["candidates"][certificate_candidate_key(self.a)]
        cert["aggregate_share_std"] = .2
        self.write()
        res = self.score()
        # Two coordinates, independent preweighted noise share per client;
        # mode-local DP noise is part of mode RMS, not Q.
        self.assertAlmostEqual(res.q_aggregate, 2 * (.6 * .2)**2)


if __name__ == "__main__":
    unittest.main()
