"""Synthetic exact structural-cancellation proof example, NOT a real DynFL certificate.

No oracle, training, DP configuration or runtime is read. All assumed ideal
updates and mode effects below are fully specified and exactly known.
"""
import json
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.online_j_bound import certificate_candidate_key, evaluate_online_bound
from dynfed.selection import Candidate, _candidate_reaches_cloud


def candidate(mode: str) -> Candidate:
    return Candidate(
        mode=mode, mechanisms={"upd": "he3"}, time=1.0,
        accuracy=.5, risk=.1, epsilon_used=0., communication_volume=1.0,
        feasible_resource=True, feasible_privacy=True, feasible_risk=True,
        feasible_time=True,
    )


def main() -> None:
    cloud = candidate("LIIC")
    edge = candidate("LIIE")
    key = certificate_candidate_key(cloud)
    # Exact, public 2D quadratic data model with deterministic, zero-residual
    # one-step mode updates. Evidence applies ONLY to this constructed example.
    state = {
        "round_index": 0, "available_before_round": 0,
        "evidence": "Synthetic exact per-client one-step updates in public 2D target",
        "aggregation_rule": "cloud_participant_sample_weighted",
        "cloud_step": 1., "numeric_rms": 0.,
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
            "0": {"samples": 5, "ideal_reference": [-.1, 0.], "ideal_rms": 0.},
            "1": {"samples": 3, "ideal_reference": [-.1, .02], "ideal_rms": 0.},
            "2": {"samples": 2, "ideal_reference": [0., .08], "ideal_rms": 0.},
        },
        "clients": {
            str(i): {"candidates": {key: {
                "relative_ideal_rms": 0., "aggregate_share_std": 0.,
                "runtime_mode_equivalence_verified": True,
                "local_dp_included_in_mode_rms": True,
            }}} for i in range(3)
        },
        "no_cloud_update_is_zero_verified": True,
    }
    manifest = {"schema": "dynfl.online_j.v2",
                "target": "ideal_full_gradient_one_step",
                "rounds": {"round:0": state}}
    weights = {0: 5., 1: 3., 2: 2.}
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "synthetic.json"
        path.write_text(json.dumps(manifest), encoding="utf-8")
        choices = {
            "all_cloud": {0: cloud, 1: cloud, 2: cloud},
            "client0_cloud": {0: cloud, 1: edge, 2: edge},
        }
        outputs = {}
        for name, profile in choices.items():
            result = evaluate_online_bound(
                manifest_path=str(path), round_index=0, profile=profile,
                client_samples=weights, admitted_client_ids=(0, 1, 2),
                reaches_cloud=_candidate_reaches_cloud,
            )
            outputs[name] = result
            print(f"{name:20s} L_J={result.lower:.9f} U_J={result.upper:.9f}")
        assert outputs["all_cloud"].upper < outputs["client0_cloud"].lower
        print("CERTIFIED RANKING: all_cloud has smaller J (SYNTHETIC proof-controlled case)")
        print("This does NOT certify real Momentum/Split/DP execution or a 5-round experiment.")


if __name__ == "__main__":
    main()
