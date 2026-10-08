"""Independent ranking evaluation contracts; synthetic values only."""
import importlib.util
from pathlib import Path
import pytest

path = Path(__file__).resolve().parents[1]/"experiments"/"validate_joint_proxy_ranking.py"
spec = importlib.util.spec_from_file_location("validate_joint_proxy_ranking",path)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)


def data(scores=(1,2,3), offsets=(1,2,3)):
    return {"evaluation_data_independent_of_calibration":True,
            "evaluation_data_provenance":"synthetic",
            "conditional_cross_client_rng_independence":True,
            "profiles":[{"profile_id":str(i),"selector_joint_score":score,"ideal_update":[0.],
                         "clients":[{"weight":1.,"paired_trials":[
                             {"clean":[float(offset)],"private":[float(offset)]},
                             {"clean":[float(offset)],"private":[float(offset)]}]}]}
                        for i,(score,offset) in enumerate(zip(scores,offsets))]}


def test_exact_ranking():
    out=mod.evaluate(data())
    assert out["spearman_rho"]==pytest.approx(1.)
    assert [p["mc_joint_score"] for p in out["profiles"]]==[1.,4.,9.]


def test_reverse_ranking():
    assert mod.evaluate(data((3,2,1)))["spearman_rho"]==pytest.approx(-1.)


def test_variance_trace_is_unbiased_and_weight_squared():
    d=data()
    for p in d["profiles"]:
        p["clients"]=[{"weight":0.5,"paired_trials":[{"clean":[0.],"private":[0.]},{"clean":[0.],"private":[2.]}]},
                      {"weight":0.5,"paired_trials":[{"clean":[0.],"private":[0.]},{"clean":[0.],"private":[2.]}]}]
    for i,p in enumerate(d["profiles"]):
        for c in p["clients"]:
            for trial in c["paired_trials"]:
                trial["clean"]=[float(i)]
                trial["private"]=[trial["private"][0]+float(i)]
    out=mod.evaluate(d)
    # each client bias=1, unbiased sample var=2, aggregate variance=.5*2=1, squared bias=1
    assert [p["mc_joint_score"] for p in out["profiles"]]==pytest.approx([2.,5.,10.])
    with pytest.raises(ValueError,match="constant"):
        mod.spearman([1,2,3],[2,2,2])


def test_reject_in_sample_and_unverified_dependency():
    d=data(); d["evaluation_data_independent_of_calibration"]=False
    with pytest.raises(ValueError,match="independence"):
        mod.evaluate(d)
    d=data(); d["conditional_cross_client_rng_independence"]=False
    with pytest.raises(ValueError,match="RNG"):
        mod.evaluate(d)


def test_reject_missing_trials_and_duplicate_profiles():
    d=data(); d["profiles"][0]["clients"][0]["paired_trials"].pop()
    with pytest.raises(ValueError,match=">=2"):
        mod.evaluate(d)
    d=data(); d["profiles"][1]["profile_id"]=d["profiles"][0]["profile_id"]
    with pytest.raises(ValueError,match="duplicate"):
        mod.evaluate(d)
