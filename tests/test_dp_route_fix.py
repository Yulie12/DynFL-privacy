from __future__ import annotations

import numpy as np
import pytest
import torch

from dynfed.fmnist_lenet5_dynamic import (
    _candidate_training_mechanisms,
    _should_apply_update_dp,
    _client_train_worker,
)
from dynfed.selection import Candidate, SelectionConfig, _global_dp_perturbation_cost
from dynfed.split_learning import apply_unified_dp


def c(mode: str, links: dict[str, str], z=3.1):
    return Candidate(mode=mode, mechanisms={"upd":next(iter(links.values()))},
        link_mechanisms=links, time=1, accuracy=1, risk=0, epsilon_used=0,
        communication_volume=1, feasible_resource=True, feasible_privacy=True,
        feasible_risk=True, feasible_time=True, update_noise_multiplier=z)


def test_per_link_worker_dp_responsibility():
    for mech in ("dp", "dp_he3"):
        liie=c("LIIE", {"L_E_upd":mech})
        worker=_candidate_training_mechanisms(liie, True)
        assert _should_apply_update_dp(worker, "upd_only", liie.mode)
        hierarchical=c("LIIEIIIC", {"L_E_upd":mech,"E_C_upd":"dp_he3"})
        worker=_candidate_training_mechanisms(hierarchical, True)
        assert worker["upd"] == mech   # E->C DP does not erase L->E protection
        assert _should_apply_update_dp(worker,"upd_only",hierarchical.mode)
    for mode, link in (("LIIC","L_C_upd"),("LIEIIC","E_C_upd"),("LIEIIIC","E_C_upd")):
        candidate=c(mode,{link:"dp_he3"})
        assert not _should_apply_update_dp(_candidate_training_mechanisms(candidate,True),"upd_only",mode)


def test_worker_audit_does_not_change_dp_randomness():
    a={"end":{"w":torch.tensor([.3,.4],dtype=torch.float32)},"edge":{}}
    b={"end":{"w":a["end"]["w"].clone()},"edge":{}}
    audit={}
    v1=apply_unified_dp(a,"dp",.25,3.1,np.random.default_rng(42),torch.device("cpu"),audit_out=audit)
    v2=apply_unified_dp(b,"dp",.25,3.1,np.random.default_rng(42),torch.device("cpu"))
    assert torch.equal(v1["end"]["w"],v2["end"]["w"])
    assert audit["original_norm"] == pytest.approx(.5)
    assert audit["clip_scale"] == pytest.approx(.5)
    assert audit["noise_std"] == pytest.approx(1.55)
    assert audit["noise_norm"] > 0


def test_worker_reports_one_dp_release(monkeypatch):
    from dynfed import fmnist_lenet5_dynamic as m
    def train_stub(**kwargs):
        return {"end":{"w":torch.tensor([.3,.4])},"edge":{}}
    monkeypatch.setattr(m,"split_local_train_lenet5",train_stub)
    payload={"dp_seed":11,"device":"cpu","training_seed":41,"mode":"LIIE",
        "global_end_state":{},"global_edge_state":{},"x":np.zeros((1,1)),"y":np.zeros(1),
        "epochs":1,"lr":.01,"model_name":"none","input_shape":(1,1,1),"num_classes":1,
        "mechanisms":{"upd":"dp"},"dp_clip_norm":.25,"dp_feature_noise_multiplier":1.0,
        "dp_update_noise_multiplier":3.1,"dp_update_mode":"upd_only","dp_epsilon":8.0,
        "l2":0.0,"local_steps":1,"client_id":3}
    res=_client_train_worker(payload)
    assert res["finite"]
    assert res["worker_update_dp_audit"]["noise_norm"] > 0
    assert res["worker_update_dp_audit"]["clip_scale"] == pytest.approx(.5)


def test_selector_liie_noise_matches_independent_packet_average():
    conf=SelectionConfig(omega_update_dimension=4,omega_update_clip_norm=.25)
    profile={0:c("LIIE",{"L_E_upd":"dp"}),1:c("LIIE",{"L_E_upd":"dp"})}
    cost=_global_dp_perturbation_cost(conf,profile,{0:1,1:1},{0:0,1:0})
    expected=4 * (3.1*.5)**2 /2
    assert cost == pytest.approx(expected)
