"""Predecision runtime inputs, fail-closed source and real v2 evaluator contract."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from dynfed.online_j_bound import certificate_candidate_key, evaluate_online_bound
from dynfed.online_j_runtime import (
    _training_metadata, hydrate_online_manifest, liic_shadow_metadata,
    reconcile_liic_proof, write_predecision_json,
)
from dynfed.selection import Candidate, _candidate_reaches_cloud
from dynfed.split_learning import build_split_pair


def _candidate(sigma=.1):
    return Candidate(
        mode="LIIC", mechanisms={"upd": "he3"}, time=1, accuracy=.5,
        risk=0, epsilon_used=0, communication_volume=1,
        feasible_resource=True, feasible_privacy=True, feasible_risk=True,
        feasible_time=True, sample_optimizer_noise_multiplier=sigma,
    )


class LIICRuntimeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.end, cls.edge = build_split_pair("lenet5", torch.device("cpu"))

    def _facts(self, cand=None, **kw):
        inputs = dict(model_parts=(self.end, self.edge), client_samples=513,
                      local_epochs=3, batch_size=256, learning_rate=.15,
                      weight_decay=.0001, sample_clip_norm=3.,
                      privacy_unit="sample", candidate=cand or _candidate(),
                      step_cap=None, model_name="lenet5")
        inputs.update(kw)
        return _training_metadata(**inputs)

    def test_real_parameter_scope_and_untruncated_partial_batches(self):
        data = self._facts()
        self.assertEqual(data["vector_dimension"], 61706)
        self.assertEqual(data["planned_optimizer_steps"], 9)
        self.assertEqual(data["batch_sizes"], [256, 256, 1]*3)
        self.assertAlmostEqual(data["gradient_clip_norm"], 3)
        self.assertAlmostEqual(data["sample_noise_multiplier"], .1)
        self.assertGreater(data["initial_parameter_norm"], 0)

    def test_nondp_clip_gradient_path_and_ablation(self):
        data = self._facts(privacy_unit="client", candidate=_candidate(None))
        self.assertEqual(data["source"], "clipped_sgd")
        self.assertEqual(data["gradient_clip_norm"], 5)
        self.assertEqual(data["sample_noise_multiplier"], 0)
        with self.assertRaisesRegex(ValueError, "sigma"):
            self._facts(candidate=_candidate(None))
        data = self._facts(ablation_clip_only=True, candidate=_candidate(None))
        self.assertEqual(data["source"], "sample_dp_sgd")
        self.assertEqual(data["sample_noise_multiplier"], 0)

    def test_reject_capped_optimizer(self):
        with self.assertRaisesRegex(ValueError, "capped"):
            self._facts(step_cap=3)

    def test_external_attestations_are_never_synthesized(self):
        metadata = self._facts()
        proof = reconcile_liic_proof(
            {"runtime_fields_from_trainer": True, "ideal_step": .15,
             "evidence": "external proof still required"}, metadata,
        )
        for field in ("ideal_reference_bound_proved", "training_state_matches_runtime",
                      "noise_distribution_matches_runtime", "extra_transformations_absent_or_modeled"):
            self.assertNotIn(field, proof)
        self.assertEqual(proof["batch_sizes"], [256,256,1]*3)
        self.assertEqual(proof["ideal_step"], .15)
        with self.assertRaisesRegex(ValueError, "disagrees"):
            reconcile_liic_proof({"runtime_fields_from_trainer":True, "learning_rate": 99}, metadata)
        with self.assertRaisesRegex(ValueError, "opt in"):
            reconcile_liic_proof({}, metadata)

    def _source(self, cand):
        key = certificate_candidate_key(cand)
        proof = {
            "runtime_fields_from_trainer": True,
            "ideal_step": .15,
            "evidence": "SYNTHETIC TEST: assumes independent ideal reference certificate",
            "training_state_matches_runtime": True,
            "ideal_reference_bound_proved": True,
            "noise_distribution_matches_runtime": True,
            "extra_transformations_absent_or_modeled": True,
        }
        cert = {
            "runtime_mode_equivalence_verified": True,
            "local_dp_included_in_mode_rms": True,
            "aggregate_share_std": 0,
            "liic_sgd_proof": proof,
        }
        state = {
            "round_index": 0, "available_before_round": 0,
            "evidence": "SYNTHETIC verification only; not real population proof",
            "cloud_step": 1., "numeric_rms": 0.,
            "aggregation_rule": "cloud_participant_sample_weighted",
            "mode_uncertainty": "rms",
            "reference_direction": [0.]*61706,
            "assumptions": {k: True for k in (
                "global_objective_is_population_sample_weighted",
                "linear_client_contributions_match_runtime",
                "ideal_reference_rms_bounds_proved",
                "mode_relative_ideal_bounds_proved",
                "noise_shares_drawn_before_secagg_and_weight_fixed",
                "noise_conditionally_zero_mean_and_uncorrelated",
                "aggregate_share_second_moment_bounds_proved",
                "numeric_rms_bound_proved",
            )},
            "population": {"0": {"samples": 513, "ideal_reference_scale": 1,
                                  "ideal_rms": 0.}},
            "clients": {"0": {"candidates": {key: cert}}},
            "no_cloud_update_is_zero_verified": True,
        }
        return {"schema": "dynfl.online_j.v2", "target": "ideal_full_gradient_one_step",
                "rounds": {"round:0": state}}

    def test_hydrated_manifest_reaches_real_v2_selector(self):
        cand = _candidate()
        with tempfile.TemporaryDirectory() as temp:
            src = write_predecision_json(Path(temp)/"source.json", self._source(cand))
            args = dict(source=src, round_index=0, selected=[(0,cand,[cand],8.)],
                        client_sample_counts={0: 513}, model_parts=(self.end,self.edge),
                        model_name="lenet5", local_epochs=3, batch_size=256,
                        learning_rate=.15, weight_decay=.0001, sample_clip_norm=3.,
                        privacy_unit="sample",step_cap=None)
            document = hydrate_online_manifest(**args)
            out = write_predecision_json(Path(temp)/"hydrated.json",document)
            result = evaluate_online_bound(manifest_path=str(out),round_index=0,
                profile={0:cand},client_samples={0:513},admitted_client_ids=(0,),
                reaches_cloud=_candidate_reaches_cloud)
            self.assertGreater(result.upper,0)
            proof = document['rounds']['round:0']['clients']['0']['candidates'][certificate_candidate_key(cand)]['liic_sgd_proof']
            self.assertEqual(proof['planned_optimizer_steps'],9)
            self.assertEqual(proof['sample_noise_multiplier'],.1)
            self.assertEqual(proof['batch_sizes'],[256,256,1]*3)
            # The unproved fields were NOT updated or fabricated by hydration.
            self.assertEqual(proof['evidence'], 'SYNTHETIC TEST: assumes independent ideal reference certificate')
            with self.assertRaisesRegex(ValueError,'missing independently proved'):
                hydrate_online_manifest(**dict(args,round_index=1))

    def test_fails_if_reference_or_mode_cert_missing(self):
        cand=_candidate()
        with tempfile.TemporaryDirectory() as temp:
            source=self._source(cand)
            source['rounds']['round:0']['clients']={}
            src=write_predecision_json(Path(temp)/'source.json',source)
            doc=hydrate_online_manifest(source=src,round_index=0,
                selected=[(0,cand,[cand],8.)],client_sample_counts={0:513},
                model_parts=(self.end,self.edge),model_name='lenet5',local_epochs=3,
                batch_size=256,learning_rate=.15,weight_decay=.0001,
                sample_clip_norm=3.,privacy_unit='sample',step_cap=None)
            out=write_predecision_json(Path(temp)/'hydrated.json',doc)
            with self.assertRaisesRegex(ValueError,'missing mode certificates|missing certificate'):
                evaluate_online_bound(manifest_path=str(out),round_index=0,
                    profile={0:cand},client_samples={0:513},admitted_client_ids=(0,),
                    reaches_cloud=_candidate_reaches_cloud)

    def test_shadow_audit_is_explicitly_not_a_j_certificate(self):
        cand=_candidate()
        shadow=liic_shadow_metadata(round_index=0,selected=[(0,cand,[cand],8.)],
            client_sample_counts={0:513},model_parts=(self.end,self.edge),
            model_name='lenet5',local_epochs=3,batch_size=256,learning_rate=.15,
            weight_decay=.0001,sample_clip_norm=3.,privacy_unit='sample',step_cap=None)
        self.assertFalse(shadow['formal_online_j_certified'])
        self.assertEqual(shadow['liic_candidates'][0]['optimizer']['planned_optimizer_steps'],9)
        self.assertNotIn('ideal_reference_bound_proved',json.dumps(shadow))

if __name__ == '__main__':
    unittest.main()
