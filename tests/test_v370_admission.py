"""Selector admission regression tests using source-referenced dependency stubs.

The complete DynFL project was not provided. These tests execute the real
selection.py admission/enumerate_candidates function with stubbed external
runtime dependencies; they do not assert end-to-end DP or full runner results.
"""
from __future__ import annotations
import importlib.util
import sys
import types
import unittest
import random
from pathlib import Path
from dataclasses import dataclass
from unittest.mock import patch

SRC = Path(__file__).resolve().parents[1] / 'dynfed/selection.py'
TRAINING = Path(__file__).resolve().parents[1] / 'dynfed/training.py'

def load_selection():
    package = types.ModuleType('dynfed')
    package.__path__ = [str(SRC.parent)]
    sys.modules['dynfed'] = package
    def stub(name, **fields):
        m = types.ModuleType('dynfed.' + name)
        m.__dict__.update(fields)
        sys.modules[m.__name__] = m
        return m
    nodes = stub('nodes', build_profiles=lambda **kwargs: ([], []))
    stub('joint_calibration', JointUpdateCalibrationTable=object)
    stub('flow_executor', CLOUD_DIRECT_MODES=set(), EDGE_CLOUD_MODES=set(), EDGE_ONLY_MODES=set(), ClientFlowInput=object, summarize_mixed_round_flow=lambda *a, **k: None)
    class ClientPrivacyLedger: pass
    class SamplePrivacyLedger: pass
    stub('privacy', ClientPrivacyLedger=ClientPrivacyLedger, SamplePrivacyLedger=SamplePrivacyLedger, OBJECT_SIZES={'emb':1,'grad':1,'upd':1,'emb_grad':1, 'logits':1, 'weakemb':1, 'strongemb':1, 'pseudo_label':1}, PRIVACY_ALPHA=1., PRIVACY_BASE_TIME=1., calibrate_gaussian_noise=lambda *a,**k:1, mechanism_uses_dp=lambda m:False, mechanism_uses_he=lambda m:False, utility_penalty=lambda *a,**k:0.)
    # Load actual mode definitions from the supplied training.py (no DynFL runtime).
    namespace = {'dataclass':dataclass}
    source=TRAINING.read_text(encoding='utf-8')
    start=source.index('@dataclass(frozen=True)\nclass ModeSpec:')
    end=source.index('\n\ndef run_experiment(',start)
    exec(source[start:end],namespace)
    stub('training', MODE_SPECS=namespace['MODE_SPECS'], ModeSpec=namespace['ModeSpec'])
    spec=importlib.util.spec_from_file_location('dynfed.selection',SRC)
    m=importlib.util.module_from_spec(spec)
    sys.modules[m.__name__]=m
    spec.loader.exec_module(m)
    return m

m=load_selection()

class TestAdmission(unittest.TestCase):
    def setUp(self):
        self.patchers = [
            patch.object(m, 'validate_update_protection_goal', lambda *args:None),
            patch.object(m, '_sample_mechanism_assignments', lambda spec, **kw: [({'upd':'he3'}, {'E_C_upd':'he3'})]),
            patch.object(m, '_sample_dp_event_counts', lambda *args: (1,0,1)),
            patch.object(m, 'resolved_sample_privacy_parameters', lambda *a,**k:{'embedding_noise_multiplier':1.,'label_grad_noise_multiplier':1.,'optimizer_noise_multiplier':1.}),
            patch.object(m, '_apply_policy_candidate_filters', lambda cfg,pol,candidates:candidates),
        ]
        for p in self.patchers: p.start()
        self.addCleanup(lambda:[p.stop() for p in self.patchers[::-1]])
        self.mode_times={name:2. for name in m.MODE_SPECS}
        self.mode_times.update(LIE=1., LIIE=1.5, LIIC=5., LIIEIIIC=6., LIC=2., LIEIIC=3., LIEIIIC=3.)
        def estimate(**kw):
            mode=kw['mode']
            deadline=kw['fast_response_deadline']
            secs=self.mode_times[mode]
            return m.Candidate(mode=mode,mechanisms=kw['mechanisms'],time=secs,accuracy=.5,risk=0.,epsilon_used=0.,communication_volume=0.,feasible_resource=True,feasible_privacy=True,feasible_risk=True,feasible_time=deadline is None or secs<=deadline,feasible_edge=True,feasible_cloud=True)
        patcher=patch.object(m,'_estimate_candidate',side_effect=estimate)
        patcher.start();self.addCleanup(patcher.stop)

    def config(self,**kwargs):
        params=dict(num_clients=20,num_edges=2,fl_first_split_on_demand=True,edge_only_requires_fast_deadline=True,strict_pair_admission=True,trusted_client_edge_pairs=((1,0),),privacy_unit='sample',dp_accounting_mode='manual')
        params.update(kwargs)
        return m.SelectionConfig(**params)

    def run_modes(self,config=None,edge=0,client=1,**kwargs):
        a={}
        candidates=m.enumerate_candidates(config=config or self.config(),client_id=client,connected_edge_id=edge,edge_factor=1.,compute_factor=1.,samples=600,remaining_epsilon=8.,round_idx=1,rng=random.Random(4),policy='full_dynfl',mode_audit=a,**kwargs)
        return {c.mode for c in candidates},a

    def test_local_feasible_excludes_split_and_nonfast_liie(self):
        modes,a=self.run_modes()
        self.assertEqual(modes,{'LIIC','LIIEIIIC'})
        self.assertEqual(a['generation_exclusions']['LIIE'],'fast_response_not_required')
        self.assertTrue(a['local_training_feasible'])

    def test_local_resource_shortfall_allows_trusted_split(self):
        cfg=self.config(resource_limit=1.0)
        modes,a=self.run_modes(cfg)
        self.assertEqual(modes, {'LIE','LIC','LIEIIC','LIEIIIC'})
        self.assertFalse(a['local_training_feasible'])

    def test_untrusted_shortfall_skips(self):
        modes,a=self.run_modes(self.config(resource_limit=1.0),client=2)
        self.assertEqual(modes,set())
        self.assertEqual(a['generation_exclusions']['LIE'],'client_edge_untrusted')

    def test_wrong_edge_cannot_claim_pair(self):
        modes,a=self.run_modes(self.config(resource_limit=1.0),edge=1)
        self.assertEqual(modes,set())
        self.assertFalse(a['client_edge_trusted'])

    def test_liie_only_when_predeclared_fast_and_met_deadline(self):
        cfg=self.config(fast_client_deadlines=((1,4.),))
        modes,a=self.run_modes(cfg)
        self.assertIn('LIIE',modes)
        self.assertTrue(a['fast_response_predeclared'])
        cfg2=self.config(fast_client_deadlines=((1,1.),))
        modes2,a2=self.run_modes(cfg2)
        self.assertNotIn('LIIE',modes2)
        self.assertFalse(a2['local_training_feasible'])
        self.assertIn('LIE',modes2)

    def test_deadline_can_make_local_infeasible(self):
        cfg=self.config(fast_client_deadlines=((1,1.2),))
        modes,a=self.run_modes(cfg)
        self.assertEqual(modes,{'LIE'})
        self.assertFalse(a['local_training_feasible'])

    def test_missing_topology_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'connected_edge_id'):
            m.enumerate_candidates(config=self.config(),client_id=1,edge_factor=1.,compute_factor=1.,samples=600,remaining_epsilon=8.,round_idx=1,rng=random.Random(4),policy='full_dynfl')

    def test_mismatch_deadline_fails_closed(self):
        with self.assertRaisesRegex(ValueError,'disagrees'):
            self.run_modes(self.config(fast_client_deadlines=((1,3.),)),fast_response_deadline=5.)

    def test_manifest_load_requires_explicit_trust(self):
        from tempfile import TemporaryDirectory
        from pathlib import Path
        import json
        with TemporaryDirectory() as tmp:
            path=Path(tmp)/'admission.json'
            path.write_text(json.dumps({
                'trusted_client_edge_pairs': [[1,0]],
                'fast_client_deadlines': [[1,2.5]],
            }))
            configured=m.enable_static_pair_admission(
                self.config(strict_pair_admission=False), path
            )
            self.assertTrue(configured.strict_pair_admission)
            self.assertEqual(configured.trusted_client_edge_pairs,((1,0),))
            self.assertEqual(configured.fast_client_deadlines,((1,2.5),))
            path.write_text('{}')
            with self.assertRaisesRegex(ValueError,'requires explicit'):
                m.enable_static_pair_admission(self.config(strict_pair_admission=False),path)

    def test_malformed_trust_rejected(self):
        for trust in (((1,0),(1,1)), ((20,0),), ((1,2),)):
            with self.assertRaises(ValueError):
                self.config(trusted_client_edge_pairs=trust)

    def test_reporting_time_limit_is_not_an_unannounced_deadline(self):
        modes,a=self.run_modes(self.config(time_limit=0.01))
        self.assertIn('LIIC',modes)
        self.assertTrue(a['local_training_feasible'])

    def test_untrusted_local_training_remains_available(self):
        modes,a=self.run_modes(self.config(),client=2)
        self.assertEqual(modes, {'LIIC','LIIEIIIC'})
        self.assertFalse(a['client_edge_trusted'])

    def test_trusted_lie_privacy_calibration_gate_is_preserved(self):
        with self.assertRaisesRegex(ValueError, 'new learning-proxy calibration'):
            self.config(trusted_edge_split_execution=True,
                trusted_lie_joint_sample_dp=True,
                learning_objective='joint_calibration',
                joint_calibration_path='calibration/unused.pt')

    def test_opt_out_preserves_legacy_admission(self):
        cfg=self.config(strict_pair_admission=False)
        modes,a=self.run_modes(cfg,client=2,edge=1)
        self.assertIn('LIIC',modes)
        self.assertNotIn('LIE',modes)

if __name__=='__main__': unittest.main(verbosity=2)
