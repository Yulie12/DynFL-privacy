"""head256 diagnostic: same DynFL split semantics; only the classifier dimension changes."""
import torch
from torchvision.models import resnet18

import dynfed.split_learning as sl


def test_fixed_projection_does_not_change_rng_or_trainable_scope(monkeypatch):
    monkeypatch.setattr(sl, "_make_torchvision_resnet", lambda *a, **kw: resnet18(weights=None))
    previous = torch.get_rng_state().clone()
    projection = sl.FixedEvenChannelProjection()
    assert torch.equal(previous, torch.get_rng_state())
    x = torch.arange(512, dtype=torch.float).reshape(1, 512)
    assert torch.equal(projection(x), x[:, ::2])
    assert sum(p.numel() for p in projection.parameters()) == 0
    end, edge = sl.build_split_pair("resnet18_pretrained_head256", torch.device("cpu"), 3, 32)
    full = sl.build_full_model("resnet18_pretrained_head256", torch.device("cpu"), 3, 32)
    assert not any(p.requires_grad for p in end.parameters())
    for model in (edge, full):
        trainable = {name: param for name, param in model.named_parameters() if param.requires_grad}
        assert sum(p.numel() for p in trainable.values()) == 2570
        assert list(trainable) == ["classifier.1.weight", "classifier.1.bias"]
    assert edge.classifier[1].weight.shape == (10, 256)
    assert full.classifier[1].weight.shape == (10, 256)


def test_split_full_state_and_predictions_match(monkeypatch):
    monkeypatch.setattr(sl, "_make_torchvision_resnet", lambda *a, **kw: resnet18(weights=None))
    prev_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        end, edge = sl.build_split_pair("resnet18_pretrained_head256", torch.device("cpu"), 3, 32)
        full = sl.build_full_model("resnet18_pretrained_head256", torch.device("cpu"), 3, 32)
        full.load_state_dict({**end.state_dict(), **edge.state_dict()}, strict=True)
        end.eval(); edge.eval(); full.eval()
        with torch.no_grad():
            inputs = torch.randn(2, 3, 32, 32)
            torch.testing.assert_close(full(inputs), edge(end(inputs)), atol=1e-6, rtol=1e-5)
    finally:
        torch.set_num_threads(prev_threads)


def test_old_head_unchanged(monkeypatch):
    monkeypatch.setattr(sl, "_make_torchvision_resnet", lambda *a, **kw: resnet18(weights=None))
    end, edge = sl.build_split_pair("resnet18_pretrained_head", torch.device("cpu"), 3, 32)
    assert sum(p.numel() for p in edge.parameters() if p.requires_grad) == 5130
