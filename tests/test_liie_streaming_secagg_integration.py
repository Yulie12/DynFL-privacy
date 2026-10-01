from types import SimpleNamespace

import torch

import dynfed.fmnist_lenet5_dynamic as dynamic
from dynfed.streaming_secagg import streaming_secure_aggregate_exact_target


def _update(a, b):
    return {
        "end": {"weight": torch.tensor([[a]], dtype=torch.float32)},
        "edge": {"weight": torch.tensor([[b]], dtype=torch.float32)},
    }


def _models():
    end = torch.nn.Linear(1, 1, bias=False)
    edge = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        end.weight.zero_()
        edge.weight.zero_()
    return end, edge


def test_liie_streaming_secagg_requires_pure_dp_edge_cohort(monkeypatch):
    dp = SimpleNamespace(mode="LIIE", mechanism="dp")
    he = SimpleNamespace(mode="LIIE", mechanism="dp_he3")
    monkeypatch.setattr(dynamic, "_candidate_edge_update_mechanism", lambda c: c.mechanism)
    updates = [(1, _update(1, 2), 3, dp), (2, _update(3, 4), 5, dp)]
    assert dynamic._liie_streaming_secagg_eligible(updates, execute_real_he=False)
    assert not dynamic._liie_streaming_secagg_eligible(updates, execute_real_he=True)
    assert not dynamic._liie_streaming_secagg_eligible(
        [(1, _update(1, 2), 3, dp), (2, _update(3, 4), 5, he)],
        execute_real_he=False,
    )
    assert not dynamic._liie_streaming_secagg_eligible(updates[:1], execute_real_he=False)


def test_edge_relative_updates_preserve_edge_local_reference_semantics():
    end, edge = _models()
    reference = {
        "end": {"weight": torch.tensor([[10.0]])},
        "edge": {"weight": torch.tensor([[20.0]])},
    }
    second_base = {
        "end": {"weight": torch.tensor([[12.0]])},
        "edge": {"weight": torch.tensor([[18.0]])},
    }
    states = {1: reference, 2: second_base}
    candidate = SimpleNamespace(mode="LIIE")
    updates = [
        (1, _update(1.0, 2.0), 1, candidate),
        (2, _update(3.0, 4.0), 3, candidate),
    ]
    relative, selected_reference = dynamic._edge_relative_updates_for_secure_aggregate(
        updates,
        client_model_states=states,
        global_end=end,
        global_edge=edge,
    )
    assert selected_reference is reference
    assert torch.allclose(relative[0]["end"]["weight"], torch.tensor([[1.0]]))
    assert torch.allclose(relative[0]["edge"]["weight"], torch.tensor([[2.0]]))
    assert torch.allclose(relative[1]["end"]["weight"], torch.tensor([[5.0]]))
    assert torch.allclose(relative[1]["edge"]["weight"], torch.tensor([[2.0]]))


def test_liie_streaming_aggregate_returns_shared_edge_local_state():
    end, edge = _models()
    reference = {
        "end": {"weight": torch.tensor([[10.0]])},
        "edge": {"weight": torch.tensor([[20.0]])},
    }
    candidate = SimpleNamespace(mode="LIIE")
    updates = [
        (1, _update(1.0, 2.0), 1, candidate),
        (2, _update(3.0, 4.0), 3, candidate),
    ]
    relative, selected_reference = dynamic._edge_relative_updates_for_secure_aggregate(
        updates,
        client_model_states={1: reference, 2: reference},
        global_end=end,
        global_edge=edge,
    )
    aggregate, audit = streaming_secure_aggregate_exact_target(
        relative,
        [1, 3],
        clip_norm=10.0,
        noise_multiplier=1e-9,
        round_seed=23,
        chunk_size=1,
        mask_std=0.5,
    )
    returned = dynamic._state_with_applied_difference(selected_reference, aggregate)
    # Weighted mean deltas are end=2.5 and edge=3.5.
    assert torch.allclose(returned["end"]["weight"], torch.tensor([[12.5]]), atol=1e-6)
    assert torch.allclose(returned["edge"]["weight"], torch.tensor([[23.5]]), atol=1e-6)
    assert audit.max_client_weight == 0.75
    assert audit.exact_target_variance
