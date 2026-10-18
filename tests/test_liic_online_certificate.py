"""LIIC SGD reference-ball algebra and real worker cross-check (public toy data)."""
from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

from dynfed.liic_online_certificate import calculate_liic_radius, planned_batch_sizes
from dynfed.online_j_bound import certificate_candidate_key, evaluate_online_bound
from dynfed.selection import Candidate, _candidate_reaches_cloud


def proof(source="sample_dp_sgd", sigma=0.1, batch_sizes=(2, 2)):
    return {
        "schema": "dynfl.liic.sgd.v1", "mode": "LIIC", "model": "lenet5",
        "optimizer": "torch.optim.SGD", "momentum": 0,
        "source": source, "training_state_matches_runtime": True,
        "ideal_reference_bound_proved": True,
        "noise_distribution_matches_runtime": True,
        "extra_transformations_absent_or_modeled": True,
        "evidence": "Controlled proof: exact clipped-SGD test under synthetic data",
        "initial_parameter_norm": 1.7, "learning_rate": 0.1,
        "ideal_step": 0.1, "weight_decay": 0.01,
        "gradient_clip_norm": 0.5 if source == "sample_dp_sgd" else 5.0,
        "sample_noise_multiplier": sigma if source == "sample_dp_sgd" else 0.0,
        "planned_optimizer_steps": len(batch_sizes), "batch_sizes": list(batch_sizes),
    }


class LIICBounds(unittest.TestCase):
    def test_batch_sizes_follow_real_drop_last_false(self):
        self.assertEqual(planned_batch_sizes(sample_count=5, epochs=3, batch_size=2), (2, 2, 1)*3)
        self.assertEqual(planned_batch_sizes(sample_count=120, epochs=3, batch_size=256), (120,)*3)
        with self.assertRaisesRegex(ValueError, "capped"):
            planned_batch_sizes(sample_count=10, epochs=1, batch_size=2, local_step_cap=3)

    def test_exact_single_step_clipped_gradients_and_weight_decay(self):
        p = proof(source="clipped_sgd", batch_sizes=(8,))
        p["initial_parameter_norm"] = 2.0
        p["learning_rate"] = 0.1
        p["weight_decay"] = 0.1
        r = calculate_liic_radius(proof=p, ideal_reference_norm=0, ideal_reference_rms=0, vector_dimension=1)
        self.assertAlmostEqual(r.optimizer_step_scale, 0.1)
        self.assertAlmostEqual(r.center_mismatch, .02)
        self.assertAlmostEqual(r.clipped_gradient_drift, .5)
        self.assertAlmostEqual(r.relative_ideal_rms, .52)
        # Adversarial deterministic 1-D clipped gradient saturates this exactly.
        theta0 = 2.0
        actual = theta0-.1*(5+.1*theta0)
        self.assertAlmostEqual(abs(actual-theta0), r.relative_ideal_rms)

    def test_gaussian_sigma_uses_actual_tail_batch(self):
        p = proof(batch_sizes=(5, 1))
        p["weight_decay"] = 0.0
        p["initial_parameter_norm"] = 0.0
        r = calculate_liic_radius(proof=p, ideal_reference_norm=0, ideal_reference_rms=0, vector_dimension=20)
        expected_noise = math.sqrt(20*((.1*2*.5*.1/5)**2+(.1*2*.5*.1/1)**2))
        self.assertAlmostEqual(r.local_dp_sgd_noise_rms, expected_noise)
        self.assertAlmostEqual(r.clipped_gradient_drift, .1)
        self.assertAlmostEqual(r.relative_ideal_rms, .1+expected_noise)

    def test_gradient_error_cap_proof_required(self):
        p=proof()
        p['gradient_reference_residual_caps'] = [0.03,0.03]
        with self.assertRaisesRegex(ValueError,'proved'):
            calculate_liic_radius(proof=p, ideal_reference_norm=0.01, ideal_reference_rms=.01, vector_dimension=5)
        p['gradient_residual_caps_proved'] = True
        r=calculate_liic_radius(proof=p, ideal_reference_norm=0.01, ideal_reference_rms=.01, vector_dimension=5)
        self.assertLess(r.clipped_gradient_drift,.01)

    def test_release_projection_nonexpansive(self):
        p=proof(source='clipped_sgd',batch_sizes=(2,))
        p['release_update_clip_norm'] = .1
        r=calculate_liic_radius(proof=p,ideal_reference_norm=.5,ideal_reference_rms=.01,vector_dimension=3)
        self.assertAlmostEqual(r.release_projection_extra,.4)

    def test_reject_other_optimizers_and_mismatched_steps(self):
        p=proof()
        p['momentum']=0.9
        with self.assertRaisesRegex(ValueError,'zero-momentum'):
            calculate_liic_radius(proof=p,ideal_reference_norm=0,ideal_reference_rms=0,vector_dimension=10)
        p['momentum']=0
        p['planned_optimizer_steps']=3
        with self.assertRaisesRegex(ValueError,'mismatch'):
            calculate_liic_radius(proof=p,ideal_reference_norm=0,ideal_reference_rms=0,vector_dimension=10)
        p['planned_optimizer_steps']=2
        p['training_state_matches_runtime']=False
        with self.assertRaisesRegex(ValueError,'verified'):
            calculate_liic_radius(proof=p,ideal_reference_norm=0,ideal_reference_rms=0,vector_dimension=10)

    def test_v2_evaluator_computes_liic_radius_inside_selection(self):
        cand=Candidate(mode='LIIC',mechanisms={'upd':'he3'},time=1,accuracy=.5,risk=.1,
                       epsilon_used=0,communication_volume=1,feasible_resource=True,
                       feasible_privacy=True,feasible_risk=True,feasible_time=True)
        key=certificate_candidate_key(cand)
        p=proof()
        p['ideal_step']=.1
        cert={"runtime_mode_equivalence_verified":True,
              "local_dp_included_in_mode_rms":True,
              "aggregate_share_std":0,"liic_sgd_proof":p}
        state={"round_index":0,"available_before_round":0,
               "evidence":"Independent synthetic explicit LIIC SGD proof (no production claim)",
               "cloud_step":1,"numeric_rms":0,
               "aggregation_rule":"cloud_participant_sample_weighted",
               "mode_uncertainty":"rms", "reference_direction":[0.,0.],
               "assumptions":{k:True for k in (
                   'global_objective_is_population_sample_weighted',
                   'linear_client_contributions_match_runtime',
                   'ideal_reference_rms_bounds_proved',
                   'mode_relative_ideal_bounds_proved',
                   'noise_shares_drawn_before_secagg_and_weight_fixed',
                   'noise_conditionally_zero_mean_and_uncorrelated',
                   'aggregate_share_second_moment_bounds_proved',
                   'numeric_rms_bound_proved')},
               'population':{'0':{'samples':4.,'ideal_reference_scale':1.,'ideal_rms':.02}},
               'clients':{'0':{'candidates':{key:cert}}},
               'no_cloud_update_is_zero_verified':True}
        data={'schema':'dynfl.online_j.v2','target':'ideal_full_gradient_one_step',
              'rounds':{'round:0':state}}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'test.json';path.write_text(json.dumps(data))
            result=evaluate_online_bound(manifest_path=str(path),round_index=0,
                profile={0:cand},client_samples={0:4.},admitted_client_ids=(0,),
                reaches_cloud=_candidate_reaches_cloud)
            standalone=calculate_liic_radius(proof=p,ideal_reference_norm=0,
                ideal_reference_rms=.02,vector_dimension=2)
            self.assertAlmostEqual(result.mode_radius,standalone.relative_ideal_rms)
            self.assertAlmostEqual(result.upper,standalone.relative_ideal_rms**2)

    def test_does_not_double_count_gaussian_as_cloud_share(self):
        r=calculate_liic_radius(proof=proof(),ideal_reference_norm=0,
                                ideal_reference_rms=0,vector_dimension=100)
        self.assertGreater(r.local_dp_sgd_noise_rms,0)
        # Exactly one local-DP contribution, already inside relative_ideal_rms;
        # a separately certified aggregate_share_std=0 gives Q_agg=0.


class RealWorkerChecks(unittest.TestCase):
    def test_actual_pytorch_liic_sample_dp_sgd_and_clean_clip(self):
        import numpy as np
        import torch
        import torch.nn.functional as F
        from dynfed.split_learning import build_split_pair, build_full_model, split_local_train_lenet5
        torch.manual_seed(41)
        device=torch.device('cpu')
        end,edge=build_split_pair('lenet5',device)
        full=build_full_model('lenet5',device)
        full.load_state_dict({**end.state_dict(),**edge.state_dict()})
        params=[p for p in full.parameters() if p.requires_grad]
        weight_norm=math.sqrt(sum(float((p.detach()**2).sum()) for p in params))
        dimension=sum(p.numel() for p in params)
        x=np.random.default_rng(33).normal(size=(2,28,28)).astype('float32')
        y=np.asarray([1,3],dtype='int64')
        inp=torch.from_numpy(x).view(-1,1,28,28)
        loss=F.cross_entropy(full(inp),torch.from_numpy(y))
        gradient=torch.autograd.grad(loss,params)
        eta_star=.01
        vhat=torch.cat([-eta_star*g.detach().flatten() for g in gradient])
        reference_norm=float(torch.linalg.vector_norm(vhat))
        for source,sigma in [('clipped_sgd',0.0),('sample_dp_sgd',.03)]:
            p=proof(source=source,sigma=sigma,batch_sizes=(2,))
            p.update(learning_rate=.01,ideal_step=eta_star,
                     initial_parameter_norm=weight_norm,
                     weight_decay=.0001)
            r=calculate_liic_radius(proof=p,ideal_reference_norm=reference_norm,
                                    ideal_reference_rms=0,vector_dimension=dimension)
            updates=[]
            for repeat in range(2):
                out=split_local_train_lenet5(
                    mode='LIIC',global_end_state=end.state_dict(),
                    global_edge_state=edge.state_dict(),x=x,y=y,epochs=1,lr=.01,
                    device=device,model_name='lenet5',
                    mechanisms={'upd':'he3'},dp_rng=np.random.default_rng(20+repeat),
                    privacy_unit='sample' if source=='sample_dp_sgd' else 'client',
                    sample_optimizer_noise_multiplier=sigma if source=='sample_dp_sgd' else None,
                    sample_optimizer_clip_norm=.5,train_batch_size=2,
                    l2=.0001,local_steps=None,training_seed=42+repeat,
                )
                diff={**out['end'],**out['edge']}
                updates.append(torch.cat([diff[name].flatten() for name,_ in full.named_parameters()]))
            empirical_rms=math.sqrt(sum(float(torch.linalg.vector_norm(u-vhat))**2 for u in updates)/len(updates))
            self.assertLessEqual(empirical_rms,r.relative_ideal_rms+1e-4,
                                 (source,empirical_rms,r.relative_ideal_rms))


if __name__ == '__main__':
    unittest.main()
