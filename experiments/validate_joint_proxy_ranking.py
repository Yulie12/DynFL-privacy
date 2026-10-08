"""Independent, profile-level ranking validation for DynFL joint Jlearn (v3.3 Eq. 127/136).

Input JSON must be produced from held-out paired clean/private trajectories not
used to fit the selector calibration table. This tool does NOT generate them.
No raw tensors are exported: only profile-level numeric scores in output JSON.
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def _vector(v, label):
    if not isinstance(v, list) or not v:
        raise ValueError(f"{label}: expected nonempty numeric vector")
    try:
        w = [float(x) for x in v]
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{label}: invalid numeric vector") from exc
    if not all(math.isfinite(x) for x in w):
        raise ValueError(f"{label}: nonfinite vector")
    return w


def _mean(xs):
    return [sum(col) / len(xs) for col in zip(*xs)]


def _sq(v):
    return sum(x*x for x in v)


def _ranks(values):
    indexed = sorted(range(len(values)), key=values.__getitem__)
    ranks = [0.] * len(values)
    k = 0
    while k < len(indexed):
        stop = k + 1
        while stop < len(indexed) and values[indexed[stop]] == values[indexed[k]]:
            stop += 1
        rank = (k + stop - 1) / 2 + 1
        for pos in indexed[k:stop]:
            ranks[pos] = rank
        k = stop
    return ranks


def spearman(x, y):
    """Pearson on average ranks; tied ranks handled; constant ranks undefined."""
    if len(x) != len(y) or len(x) < 3:
        raise ValueError("Spearman requires >=3 matched candidates")
    a, b = _ranks(x), _ranks(y)
    ma, mb = sum(a)/len(a), sum(b)/len(b)
    aa, bb = [v-ma for v in a], [v-mb for v in b]
    denom = math.sqrt(_sq(aa)*_sq(bb))
    if denom == 0:
        raise ValueError("Spearman undefined: at least one ranking is constant")
    return max(-1., min(1., sum(v*w for v,w in zip(aa,bb))/denom))


def evaluate(payload):
    # Dedicated trusted-local summary path. Unlike legacy raw-vector JSON,
    # this never serializes per-client parameter trajectories.
    if payload.get('format') in ('dynfl_independent_joint_summary_v1',
                                 'dynfl_independent_joint_summary_v2'):
        corrected = payload['format'].endswith('_v2')
        if corrected and payload.get('mc_estimator') != 'unbiased_conditional_second_moment_independent_clients_v2':
            raise ValueError('v2 unbiased MC estimator declaration required')
        if payload.get('evaluation_data_independent_of_calibration') is not True:
            raise ValueError('independence declaration required')
        if payload.get('evaluation_data_provenance') != 'private_local_only':
            raise ValueError('compact summary is local-only')
        if payload.get('conditional_cross_client_rng_independence') is not True:
            raise ValueError('cross-client DP RNG independence required')
        if payload.get('evaluation_protocol') != 'independent_official_test_paired_rng_v1':
            raise ValueError('unrecognized independent capture protocol')
        if payload.get('calibration_source_verified_as_train_subset') is not True:
            raise ValueError('calibration/train split provenance not verified')
        digest = payload.get('calibration_sha256')
        if not isinstance(digest, str) or len(digest) != 64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('calibration SHA256 is required')
        if payload.get('scope') != 'calibration-covered_subcohort_not_full_cohort':
            raise ValueError('compact ranking must declare restricted subcohort scope')
        if payload.get('measurement_boundary') != 'global_relative_worker_updates_before_cloud_release_processing':
            raise ValueError('measurement boundary must explicitly exclude cloud release processing')
        rows = payload.get('profiles')
        if not isinstance(rows, list) or len(rows) < 3:
            raise ValueError('need >=3 independently measured profiles')
        found = set()
        scores, measured = [], []
        for row in rows:
            key = str(row['profile_id'])
            if key in found:
                raise ValueError('duplicate profile ID')
            found.add(key)
            if int(row['client_count']) < 1 or int(row['trials_per_cell']) < 2:
                raise ValueError('insufficient independent paired observations')
            vals = [float(row[k]) for k in (
                'selector_joint_score', 'mc_joint_score', 'mc_bias_squared', 'mc_variance_trace')]
            if not all(math.isfinite(v) and v >= 0 for v in vals):
                raise ValueError('nonfinite/negative independent evaluation score')
            correction = 0.0
            if corrected:
                if 'mc_finite_sample_correction' not in row:
                    raise ValueError('v2 missing finite-sample MC correction')
                correction = float(row['mc_finite_sample_correction'])
                if (not math.isfinite(correction) or correction < 0
                        or correction > vals[3] + 1e-10):
                    raise ValueError('invalid finite-sample MC correction')
            if not math.isclose(vals[1], vals[2] + vals[3] - correction,
                                rel_tol=1e-8, abs_tol=1e-10):
                raise ValueError('inconsistent Monte Carlo bias/variance decomposition')
            scores.append(vals[0])
            measured.append(vals[1])
        rho = spearman(scores, measured)
        return {
            'status': 'computed_not_privacy_proof',
            'formula': ('finite_sample_corrected_joint_second_moment_v2'
                        if corrected else 'v3.3_eq127_independent_client_rng_legacy_v1'),
            'candidate_count': len(rows), 'spearman_rho': rho,
            'evaluation_data_provenance': 'private_local_only',
            'scope': payload['scope'], 'measurement_boundary': payload['measurement_boundary'],
            'calibration_sha256': digest,
            'profiles': rows,
        }
    if payload.get("evaluation_data_independent_of_calibration") is not True:
        raise ValueError("independence must be explicitly declared true; no in-sample ranking validation")
    if payload.get("evaluation_data_provenance") not in ("public", "synthetic", "private_local_only"):
        raise ValueError("evaluation_data_provenance must be public, synthetic or private_local_only")
    if payload.get("conditional_cross_client_rng_independence") is not True:
        raise ValueError("conditional cross-client RNG independence required for sum a_i^2 v_i")
    if not isinstance(payload.get("profiles"), list) or len(payload["profiles"]) < 3:
        raise ValueError("need at least 3 candidate profiles")
    ids, rows = set(), []
    shared_ideal = None
    for profile in payload["profiles"]:
        name = str(profile["profile_id"])
        if name in ids:
            raise ValueError(f"duplicate profile_id: {name}")
        ids.add(name)
        pred = float(profile["selector_joint_score"])
        if not math.isfinite(pred) or pred < 0:
            raise ValueError("selector_joint_score must be finite and nonnegative")
        ideal = _vector(profile["ideal_update"], "ideal_update")
        if shared_ideal is None:
            shared_ideal = ideal
        elif ideal != shared_ideal:
            raise ValueError("all profiles at the same state must share one ideal_update")
        d = len(ideal)
        clients = profile.get("clients")
        if not isinstance(clients, list) or not clients:
            raise ValueError(f"{name}: clients required")
        total_w = sum(float(c["weight"]) for c in clients)
        if not math.isclose(total_w, 1., abs_tol=1e-8):
            raise ValueError(f"{name}: Cloud weights must sum to 1")
        clean_global = [0.] * d
        bias_global = [0.] * d
        variance_global = 0.
        for c in clients:
            w = float(c["weight"])
            if not math.isfinite(w) or w < 0:
                raise ValueError("weights must be nonnegative finite")
            trials = c.get("paired_trials")
            if not isinstance(trials, list) or len(trials) < 2:
                raise ValueError(f"{name}: each client must have >=2 held-out paired trials")
            cleans, diffs = [], []
            for item in trials:
                clean = _vector(item["clean"], "clean")
                private = _vector(item["private"], "private")
                if len(clean) != d or len(private) != d:
                    raise ValueError(f"{name}: vector dimensions differ")
                cleans.append(clean)
                diffs.append([p-q for p,q in zip(private,clean)])
            cm, bm = _mean(cleans), _mean(diffs)
            var_trace = sum(_sq([v-m for v,m in zip(diff,bm)]) for diff in diffs)/(len(diffs)-1)
            clean_global = [x+w*y for x,y in zip(clean_global,cm)]
            bias_global = [x+w*y for x,y in zip(bias_global,bm)]
            variance_global += w*w*var_trace
        mean_error = [a+b-c for a,b,c in zip(clean_global,bias_global,ideal)]
        mc = _sq(mean_error) + variance_global
        rows.append({"profile_id":name,"selector_joint_score":pred,"mc_joint_score":mc,
                     "mc_bias_squared":_sq(mean_error),"mc_variance_trace":variance_global})
    rho = spearman([r["selector_joint_score"] for r in rows], [r["mc_joint_score"] for r in rows])
    return {"status":"computed_not_privacy_proof", "formula":"v3.3_eq127_independent_client_rng_legacy_raw",
            "candidate_count":len(rows), "spearman_rho":rho,
            "evaluation_data_provenance":payload["evaluation_data_provenance"], "profiles":rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    result = evaluate(json.loads(args.input.read_text(encoding="utf-8")))
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False)+"\n",encoding="utf-8")
    print(f"Spearman rho={result['spearman_rho']:.6f}, candidates={result['candidate_count']}; result={args.output}")


if __name__ == "__main__":
    main()
