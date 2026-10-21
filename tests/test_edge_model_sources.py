import torch
from dynfed.paper_model_state import EdgeModels


def test_edge_replica_cannot_read_unavailable_feedback():
    replicas = EdgeModels(torch.tensor([1.]), 2)
    replicas.defer(0, 5., torch.tensor([7.]), 0, "r0:FL_E:e0")
    replicas.advance(4.9)
    assert replicas.dispatch(0)[0].item() == 1
    replicas.advance(5.)
    value, base_round, source = replicas.dispatch(0)
    assert value.item() == 7
    assert base_round == 0
    assert source.startswith("edge:0:")
    assert replicas.dispatch(1)[0].item() == 1


def test_edge_feedback_is_applied_in_availability_order_not_in_generation_order():
    replicas = EdgeModels(torch.tensor([0.]), 1)
    replicas.defer(0, 7., torch.tensor([2.]), 0, "earlier-slow")
    replicas.defer(0, 4., torch.tensor([3.]), 1, "later-fast")
    replicas.advance(5.)
    assert replicas.dispatch(0)[0].item() == 3
    replicas.advance(8.)
    assert replicas.dispatch(0)[0].item() == 2
    assert not replicas.pending
