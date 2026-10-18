"""Blockwise same-world J certificate: orthogonal coordinates, not noise guesses."""
import json,tempfile,unittest
from pathlib import Path
from dynfed.online_j_bound import certificate_candidate_key,evaluate_online_bound
from dynfed.selection import Candidate,_candidate_reaches_cloud

def cand(mode):
    return Candidate(mode=mode,mechanisms={'upd':'he3'},time=1,accuracy=.5,risk=0,
                     epsilon_used=0,communication_volume=1,feasible_resource=True,
                     feasible_privacy=True,feasible_risk=True,feasible_time=True)

class BlockCertificateTests(unittest.TestCase):
    def setUp(self):
        d=tempfile.TemporaryDirectory();self.addCleanup(d.cleanup)
        self.path=Path(d.name)/'cert.json'
        self.a,self.b=cand('LIIC'),cand('LIIEIIIC')
        self.state={
            'round_index':0,'available_before_round':0,'evidence':'public exact symbolic test',
            'cloud_step':1.,'numeric_rms':0.,'aggregation_rule':'cloud_participant_sample_weighted',
            'mode_uncertainty':'rms',
            'assumptions':dict.fromkeys((
              'global_objective_is_population_sample_weighted',
              'linear_client_contributions_match_runtime','ideal_reference_rms_bounds_proved',
              'mode_relative_ideal_bounds_proved','noise_shares_drawn_before_secagg_and_weight_fixed',
              'noise_conditionally_zero_mean_and_uncorrelated',
              'aggregate_share_second_moment_bounds_proved','numeric_rms_bound_proved'),True),
            'population':{
                '0':{'samples':6.,'ideal_reference':[10.,0.], 'ideal_rms':0.,'ideal_rms_by_block':[0.,0.]},
                '1':{'samples':4.,'ideal_reference':[0.,0.], 'ideal_rms':0.,'ideal_rms_by_block':[0.,0.]},
            },
            'clients':{'0':{'candidates':{certificate_candidate_key(self.a):{
                'relative_ideal_rms':10.,'relative_ideal_rms_by_block':[0.,10.],
                'aggregate_share_std':0.,'runtime_mode_equivalence_verified':True,
                'local_dp_included_in_mode_rms':True,
            }}}},'no_cloud_update_is_zero_verified':True,
        }
    def score(self):
        self.path.write_text(json.dumps({'schema':'dynfl.online_j.v2',
            'target':'ideal_full_gradient_one_step','rounds':{'round:0':self.state}}))
        return evaluate_online_bound(manifest_path=str(self.path),round_index=0,
            profile={0:self.a,1:self.b},client_samples={0:6.,1:4.},
            admitted_client_ids=(0,),reaches_cloud=_candidate_reaches_cloud)
    def test_strict_euclidean_tightening(self):
        old=self.score();self.assertAlmostEqual(old.upper,196.)
        self.state['parameter_blocks']=[{'name':'x','start':0,'end':1},
                                         {'name':'y','start':1,'end':2}]
        new=self.score()
        self.assertAlmostEqual(new.upper,116.)
        self.assertAlmostEqual(new.lower,16.)
        self.assertIn('blockwise',new.source)
    def test_missing_and_overlapping_blocks_rejected(self):
        self.state['parameter_blocks']=[{'name':'x','start':0,'end':1}]
        with self.assertRaisesRegex(ValueError,'parameter_blocks'):self.score()
        self.state['parameter_blocks']=[{'name':'x','start':0,'end':2}]
        with self.assertRaisesRegex(ValueError,'ideal_rms_by_block'):self.score()
    def test_fallback_if_new_cap_looser(self):
        old=self.score()
        self.state['parameter_blocks']=[{'name':'whole','start':0,'end':2}]
        for p in self.state['population'].values():p['ideal_rms_by_block']=[0.]
        self.state['clients']['0']['candidates'][certificate_candidate_key(self.a)]['relative_ideal_rms_by_block']=[100.]
        self.assertAlmostEqual(self.score().upper,old.upper)

if __name__=='__main__':unittest.main()
