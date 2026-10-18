"""Stand-alone online-J algebra + selector smoke. SYNTHETIC, not certified DynFL."""
from __future__ import annotations
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.online_j_bound import certificate_candidate_key, evaluate_online_bound
from dynfed.selection import Candidate, SelectionConfig, _candidate_reaches_cloud, choose_global_pareto_profile


def candidate(mode, time):
    return Candidate(mode=mode, mechanisms={'upd': 'he3'}, time=time, accuracy=0., risk=0.,
                     epsilon_used=0., communication_volume=1., feasible_resource=True,
                     feasible_privacy=True, feasible_risk=True, feasible_time=True)


def main():
    slow = candidate('LIIC', 2.0)
    fast = candidate('LIIEIIIC', 1.5)
    entries = {}
    for mode_candidate, sigma, radius in [(slow, .01, .02), (fast, .01, .07)]:
        entries[certificate_candidate_key(mode_candidate)] = {
            'step_scale': .1, 'mode_rms': radius, 'aggregate_share_std': sigma,
            'runtime_mode_equivalence_verified': True,
            'local_dp_included_in_mode_rms': True,
        }
    assumptions = {x: True for x in (
        'weighted_client_contributions_match_runtime', 'gradient_reference_rms_bound_proved',
        'mode_rms_bounds_proved', 'noise_shares_drawn_before_secagg_and_weight_fixed',
        'noise_conditionally_zero_mean_and_uncorrelated', 'numeric_rms_bound_proved',
    )}
    manifest = {
        'schema': 'dynfl.online_j.v1', 'target': 'ideal_full_gradient_one_step',
        'rounds': {'round:0': {
            'round_index': 0, 'available_before_round': 0,
            'evidence': 'SYNTHETIC CONSTRUCTED BOUNDS, not physical DynFL certification',
            'reference_gradient': [3.0, 4.0], 'gradient_rms_error': .01,
            'ideal_step': .1, 'cloud_step': 1.0, 'numeric_rms': 0.,
            'aggregation_rule': 'cloud_participant_sample_weighted',
            'no_cloud_update_is_zero_verified': True, 'assumptions': assumptions,
            'clients': {'0': {'candidates': entries}},
        }},
    }
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / 'SYNTHETIC_ONLY.json'
        path.write_text(json.dumps(manifest), encoding='utf-8')
        config = SelectionConfig(num_clients=1, num_edges=1, learning_objective='online_upper_bound',
                                 online_bound_manifest_path=str(path), pareto_max_iters=2,
                                 pareto_neighbor_top_k=0)
        for item in (slow, fast):
            bound = evaluate_online_bound(manifest_path=str(path), round_index=0,
                                          profile={0: item}, client_samples={0: 100.},
                                          admitted_client_ids=(0,), reaches_cloud=_candidate_reaches_cloud)
            print(f'SYNTHETIC {item.mode}: T~{item.time:.3f}, J<= {bound.upper:.9f}, '
                  f'center^2={bound.central_norm**2:.9f}, Q={bound.q_aggregate:.9f}')
        _, chosen = choose_global_pareto_profile(config=config,
                                                  selected=[(0, slow, [slow, fast], 8.)],
                                                  client_samples={0: 100.}, client_edges={0: 0})
        print('SYNTHETIC selected:', chosen.profile[0].mode,
              'online_U=', chosen.system_learning_error,
              'source=', chosen.learning_proxy_source)
        print('NOT REAL TRAINING; for production, all predecision runtime certificates are missing.')


if __name__ == '__main__':
    main()
