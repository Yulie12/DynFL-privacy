"""Synthetic-only tests for Stage20 log parsing and failure behavior."""
from __future__ import annotations

import copy
import unittest

from audit_stage20_rdp_shadow import audit_selected_round


class ShadowTests(unittest.TestCase):
    def fixture(self):
        clip=3.0
        pop={}
        rows=[]
        for cid, mode, edge, sigma in [(0, 'LIIC', 0, 2.0),(1, 'LIIEIIIC', 1, 4.0),(2,'SKIP',0,None)]:
            stages=1 if mode=='LIIC' else 3
            candidate={}
            if mode!='SKIP':
                candidate['key_test']={
                    'mode':mode,'he_only_no_extra_release_dp':True,
                    'edge_training_stages':stages,
                    'sample_optimizer_noise_multiplier':sigma,
                    'sample_optimizer_clip_norm':clip,
                    'planned_batches_per_stage':[4,1],
                    'sample_dp_optimizer':True,
                }
            pop[str(cid)]={'samples':5, 'candidates':candidate}
            rows.append({
                'client_id':str(cid),'edge_id':str(edge),'mode':mode,
                'selected_sample_optimizer_noise_multiplier':str(sigma) if sigma is not None else '',
                'mechanisms':'L_C_upd:he3' if mode=='LIIC' else 'E_C_upd:he3;L_E_upd:none' if mode=='LIIEIIIC' else '',
                'sample_optimizer_events':str(stages*2) if mode!='SKIP' else '0',
            })
        snap={
            'schema':'dynfl.online_j.oracle_diagnostic.v2',
            'used_raw_client_training_data':True,'formal_online_j_certified':False,
            'round_index':0,'cloud_step':1.,'ideal_population_update_norm':.1,
            'stage20_shadow_inputs':{
                'trainable_dimension':16,'model_parameter_norm':2.,
                'model_parameter_dot_ideal_update':.02,
                'local_epochs':1,'batch_size':4,'learning_rate':.01,
                'weight_decay':.0001,'momentum':.9,
                'sample_optimizer_clip_norm':3.,'privacy_unit':'sample',
                'assumptions_not_certified':True,
            },
            'clients':pop,
        }
        prof={'LIIC_count':'1','LIIEIIIC_count':'1','learning_error_objective':'2.0'}
        return snap,rows,prof

    def test_valid_metadata_produces_conditional_components(self):
        snap,rows,prof=self.fixture()
        r=audit_selected_round(snap,rows,prof)
        self.assertGreater(r['stage20_shadow_U_transfer'],0)
        self.assertGreater(r['sqrt_V_RDP_gaussian_radius'],0)
        self.assertGreater(r['G_clipped_gradient_radius'],0)
        self.assertEqual(r['noise_event_count'],8)
        self.assertEqual(r['LIIC'],1)
        self.assertEqual(r['LIIEIIIC'],1)
        self.assertIn('NOT_DEPLOYMENT_CERTIFIED',r['result_scope'])

    def test_old_snapshot_rejected(self):
        snap,rows,prof=self.fixture()
        del snap['stage20_shadow_inputs']
        with self.assertRaisesRegex(ValueError, 'missing stage20_shadow_inputs'):
            audit_selected_round(snap,rows,prof)

    def test_sigma_mismatch_rejected(self):
        snap,rows,prof=self.fixture()
        rows[1]['selected_sample_optimizer_noise_multiplier']='6.0'
        with self.assertRaisesRegex(ValueError, 'no snapshot candidate matches'):
            audit_selected_round(snap,rows,prof)

    def test_extra_dp_link_rejected(self):
        snap,rows,prof=self.fixture()
        rows[0]['mechanisms']='L_C_upd:dp_he3'
        with self.assertRaisesRegex(ValueError, 'unsupported update/link DP'):
            audit_selected_round(snap,rows,prof)

    def test_event_count_mismatch_rejected(self):
        snap,rows,prof=self.fixture()
        rows[1]['sample_optimizer_events']='3'
        with self.assertRaisesRegex(ValueError, 'RDP planned events'):
            audit_selected_round(snap,rows,prof)

    def test_invalid_aggregation_cohort_rejected(self):
        snap,rows,prof=self.fixture()
        prof['LIIC_count']='2'
        with self.assertRaisesRegex(ValueError, 'candidate cohort mismatch'):
            audit_selected_round(snap,rows,prof)


if __name__ == '__main__':
    unittest.main()
