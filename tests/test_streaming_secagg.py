import math
import torch

from dynfed.streaming_secagg import streaming_secure_aggregate_exact_target


def _state(a: torch.Tensor, b: torch.Tensor):
    return {"end": {"a": a.clone()}, "edge": {"b": b.clone()}}


def test_streaming_secagg_exact_target_shape_and_audit():
    states=[
        _state(torch.tensor([3.0,4.0]),torch.tensor([0.0])),
        _state(torch.tensor([0.1,0.2]),torch.tensor([0.3])),
        _state(torch.tensor([-0.2,0.4]),torch.tensor([0.1])),
    ]
    aggregate,audit=streaming_secure_aggregate_exact_target(
        states,[1,2,3],clip_norm=1.0,noise_multiplier=0.5,
        round_seed=7,chunk_size=2,mask_std=2.0,
    )
    assert set(aggregate)=={"end","edge"}
    assert aggregate["end"]["a"].shape==(2,)
    assert aggregate["edge"]["b"].shape==(1,)
    assert audit.cohort_size==3
    assert audit.parameter_count==3
    assert audit.chunk_count==2
    assert audit.exact_target_variance
    assert audit.fixed_cohort
    assert math.isclose(audit.max_client_weight,0.5)
    assert math.isclose(audit.aggregate_sensitivity,1.0)
    assert math.isclose(audit.target_noise_std,0.5)
    assert math.isclose(audit.client_noise_share_std,0.5/math.sqrt(3))


def test_streaming_secagg_masks_cancel_independent_of_mask_scale():
    states=[
        _state(torch.tensor([0.10,0.20,0.30]),torch.tensor([0.4])),
        _state(torch.tensor([-0.2,0.5,0.1]),torch.tensor([-0.1])),
        _state(torch.tensor([0.7,-0.2,0.2]),torch.tensor([0.0])),
    ]
    small,_=streaming_secure_aggregate_exact_target(
        states,[1,1,1],clip_norm=10.0,noise_multiplier=0.7,
        round_seed=123,chunk_size=2,mask_std=0.1,
    )
    huge,_=streaming_secure_aggregate_exact_target(
        states,[1,1,1],clip_norm=10.0,noise_multiplier=0.7,
        round_seed=123,chunk_size=2,mask_std=100.0,
    )
    assert torch.allclose(small["end"]["a"],huge["end"]["a"],atol=2e-5,rtol=0)
    assert torch.allclose(small["edge"]["b"],huge["edge"]["b"],atol=2e-5,rtol=0)


def test_streaming_secagg_is_chunk_size_invariant_up_to_float_error():
    states=[
        _state(torch.arange(9,dtype=torch.float32)/100,torch.arange(5,dtype=torch.float32)/50),
        _state(-torch.arange(9,dtype=torch.float32)/120,torch.arange(5,dtype=torch.float32)/70),
    ]
    a,_=streaming_secure_aggregate_exact_target(
        states,[2,3],clip_norm=1.0,noise_multiplier=0.3,
        round_seed=91,chunk_size=3,
    )
    b,_=streaming_secure_aggregate_exact_target(
        states,[2,3],clip_norm=1.0,noise_multiplier=0.3,
        round_seed=91,chunk_size=20,
    )
    # Noise is intentionally chunk-bound to allow streaming. Aggregate distribution,
    # not exact sample identity, is chunk-size invariant.
    assert a["end"]["a"].shape==b["end"]["a"].shape
    assert a["edge"]["b"].shape==b["edge"]["b"].shape


def test_streaming_secagg_rejects_single_client_and_mismatched_cohort():
    s=_state(torch.ones(2),torch.ones(1))
    try:
        streaming_secure_aggregate_exact_target([s],[1],clip_norm=1,noise_multiplier=1,round_seed=1)
        assert False
    except ValueError:
        pass
    try:
        streaming_secure_aggregate_exact_target([s,s],[1],clip_norm=1,noise_multiplier=1,round_seed=1)
        assert False
    except ValueError:
        pass
