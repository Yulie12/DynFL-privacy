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
    return {"status":"computed_not_privacy_proof", "formula":"v3.3_eq127_independent_client_rng",
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
