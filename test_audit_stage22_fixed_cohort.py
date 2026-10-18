"""Stage22 fail-closed mathematical shadow tests; no training needed."""
import copy
import math
import unittest
from audit_stage22_fixed_cohort import compute_round
from test_audit_stage20_rdp_shadow import ShadowTests


class CounterfactualTests(unittest.TestCase):
    def fixture(self):
        snap, rows, prof = ShadowTests().fixture()
        # Both cloud modes are offered in snapshot for both active clients.
        for cid in (0, 1):
            entries = snap['clients'][str(cid)]['candidates']
            for mode, sigma, stages in [('LIIC', 2.0, 1), ('LIIEIIIC', 4.0, 3)]:
                entries[mode] = dict(mode=mode, he_only_no_extra_release_dp=True,
                                     edge_training_stages=stages,
                                     sample_optimizer_noise_multiplier=sigma,
                                     sample_optimizer_clip_norm=3.,
                                     planned_batches_per_stage=[4,1],sample_dp_optimizer=True)
        return snap,rows,prof

    def test_all_three_scenarios_with_same_cloud_cohort(self):
        snap,rows,profile=self.fixture()
        orig = copy.deepcopy(rows)
        out=compute_round(snap, rows, profile, None)
        self.assertEqual([r['status'] for r in out], ['COMPUTED_SELECTED','COMPUTED_HYPOTHETICAL','COMPUTED_HYPOTHETICAL'])
        self.assertEqual([r['events'] for r in out], [8,4,12])
        self.assertEqual([r['LIIC'] for r in out],[1,2,0])
        self.assertEqual([r['LIIEIIIC'] for r in out],[1,0,2])
        self.assertEqual([r['changed_client_modes'] for r in out],[0,1,1])
        self.assertEqual(rows,orig)
        self.assertEqual(out[1]['Stage19_J_selected_only'],'')
        self.assertIn('NOT_REVALIDATED',out[2]['assumption'])

    def test_missing_candidate_refuses_pure_scenario(self):
        snap,rows,prof=self.fixture()
        snap['clients']['1']['candidates'].pop('LIIC')
        out=compute_round(snap,rows,prof,None)
        self.assertEqual(out[1]['status'],'UNAVAILABLE_FAIL_CLOSED')
        self.assertIn('no predecision LIIC',out[1]['reason'])
        self.assertEqual(out[0]['status'],'COMPUTED_SELECTED')

    def test_ambiguous_candidate_refuses(self):
        snap,rows,prof=self.fixture()
        snap['clients']['1']['candidates']['other_liic']={
            **snap['clients']['1']['candidates']['LIIC'],
            'sample_optimizer_noise_multiplier':3.0,
        }
        out=compute_round(snap,rows,prof,None)
        self.assertEqual(out[1]['status'],'UNAVAILABLE_FAIL_CLOSED')
        self.assertIn('ambiguous',out[1]['reason'])

    def test_baseline_disagreement_fails(self):
        snap,rows,prof=self.fixture()
        with self.assertRaisesRegex(ValueError,'failed equality check'):
            compute_round(snap,rows,prof,{'stage20_shadow_U_transfer':'0'})

    def test_selected_profile_inconsistency_fails(self):
        snap,rows,prof=self.fixture()
        prof['LIIC_count']='2'
        with self.assertRaisesRegex(ValueError, 'cohort mismatch'):
            compute_round(snap,rows,prof,None)

    def test_comparison_depends_on_noise_multiplier(self):
        snap,rows,prof=self.fixture()
        out=compute_round(snap,rows,prof,None)
        lower = out[1]['Stage20_U_shadow']
        snap2,rows2,prof2=self.fixture()
        snap2['clients']['1']['candidates']['LIIC']['sample_optimizer_noise_multiplier'] = 6.0
        higher=compute_round(snap2,rows2,prof2,None)[1]['Stage20_U_shadow']
        self.assertGreater(higher,lower)


if __name__ == '__main__':
    unittest.main()
