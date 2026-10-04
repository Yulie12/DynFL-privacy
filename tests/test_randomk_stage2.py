from __future__ import annotations
import numpy as np
import pytest
import torch
from dynfed.randomk_update import build_public_mask, release_projected_aggregate


def ref():
    return {"end":{"layer.weight":torch.arange(20,dtype=torch.float32),
                   "classifier.weight":torch.arange(10,dtype=torch.float32)},"edge":{}}


def test_layerwise_exact_count_and_reproducibility():
    a=build_public_mask(ref(),.2,9,"layerwise_randomk")
    b=build_public_mask(ref(),.2,9,"layerwise_randomk")
    assert int(a['end']['layer.weight'].sum())==4
    assert int(a['end']['classifier.weight'].sum())==2
    assert all(torch.equal(a['end'][k],b['end'][k]) for k in a['end'])


def test_classifier_projection_only():
    m=build_public_mask(ref(),.01,2,'classifier_only')
    assert not m['end']['layer.weight'].any()
    assert m['end']['classifier.weight'].all()


def test_classifier_missing_is_rejected():
    with pytest.raises(ValueError,match="zero coordinates"):
        build_public_mask({'end':{'layer.weight':torch.ones(4)}},.1,3,'classifier_only')


def test_invalid_strategy_rejected():
    with pytest.raises(ValueError,match="unknown"):
        build_public_mask(ref(),.1,3,'private_topk')


def test_streaming_matches_list():
    updates=[ref(),ref()]
    m=build_public_mask(ref(),.2,9,'randomk')
    left,ld=release_projected_aggregate(iter(updates),[1,1],m,clip_norm=1,noise_multiplier=2,seed=3)
    right,rd=release_projected_aggregate(updates,[1,1],m,clip_norm=1,noise_multiplier=2,seed=3)
    assert torch.equal(left['end']['layer.weight'],right['end']['layer.weight'])
    assert ld==rd


def test_diagnostic_zero_noise_retains_clipping():
    m=build_public_mask(ref(),1,0)
    result,diag=release_projected_aggregate([ref(),ref()],[1,1],m,clip_norm=.5,noise_multiplier=0,seed=3)
    assert diag['noise_norm']==0
    assert diag['clipped_clients']==2
    assert diag['signal_norm']==pytest.approx(.5,abs=1e-5)


def test_retention_and_coordinate_metrics():
    m=build_public_mask(ref(),.2,9,'randomk')
    _,d=release_projected_aggregate([ref(),ref()],[1,1],m,clip_norm=999,noise_multiplier=1,seed=3)
    assert d['total_trainable_coordinates']==30
    assert d['selected_coordinates']==6
    assert 0<=d['mean_client_signal_retention']<=1
    assert d['expected_noise_norm']==pytest.approx(d['noise_std']*np.sqrt(6))


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_training_baseline_stays_on_model_device_and_is_a_snapshot(device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable in this test environment")
    from experiments.run_randomk_fedavg import snapshot_global_state

    model = torch.nn.Linear(3, 2).to(device_name)
    snapshot = snapshot_global_state(model)
    with torch.no_grad():
        model.weight.add_(1)
    for name, value in model.state_dict().items():
        assert snapshot[name].device == value.device
        assert snapshot[name].data_ptr() != value.data_ptr()
    # Same operation that previously raised a CUDA/CPU device mismatch.
    diff = model.weight.detach() - snapshot["weight"]
    assert torch.allclose(diff, torch.ones_like(diff))
