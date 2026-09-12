"""Check all three trainable scopes without downloading model weights."""
import math

import pytest
import torch
from torchvision.models import resnet18

from dynfed.split_learning import _prepare_model_for_training, clip_state_difference


@pytest.mark.parametrize(
    "name,expected,prefixes",
    [
        ("resnet18_pretrained_head", 5130, ("fc.",)),
        ("resnet18_pretrained_layer4_head", 8393738, ("layer4.", "fc.")),
        ("resnet18_pretrained", 10490890, ("layer3.", "layer4.", "fc.")),
    ],
)
def test_trainable_scope_and_batchnorm_are_explicit(name, expected, prefixes):
    with torch.random.fork_rng(devices=[]):
        model = resnet18(weights=None, num_classes=10)
    model.train()
    _prepare_model_for_training(model, name)
    parameters = {k: p for k, p in model.named_parameters() if p.requires_grad}
    assert sum(p.numel() for p in parameters.values()) == expected
    assert all(k.startswith(prefixes) for k in parameters)
    for module in model.modules():
        if isinstance(module, torch.nn.BatchNorm2d):
            assert not module.training
            assert not any(p.requires_grad for p in module.parameters())


def test_clipping_bounds_whole_contribution_not_each_layer_separately():
    diff = {"end": {"layer3": torch.tensor([3.])}, "edge": {"layer4": torch.tensor([4.])}}
    clipped, norm, scale = clip_state_difference(diff, 1, torch.device("cpu"))
    assert norm == pytest.approx(5)
    assert scale == pytest.approx(0.2)
    assert math.sqrt(sum(float(v.square().sum()) for p in clipped.values() for v in p.values())) == pytest.approx(1)
    assert diff["end"]["layer3"].item() == 3


def test_adapter_identity_rng_and_private_training_scope():
    from dynfed.split_learning import FixedProjectionAdapter
    with torch.random.fork_rng(devices=[]):
        model = resnet18(weights=None, num_classes=10)
        rng = torch.get_rng_state().clone()
        adapter = FixedProjectionAdapter()
        assert torch.equal(rng, torch.get_rng_state())
        x = torch.randn(3, 512)
        assert torch.equal(adapter(x), x)
        model.fc = torch.nn.Sequential(adapter, model.fc)
        _prepare_model_for_training(model, "resnet18_pretrained_adapter")
        params = {k: p for k, p in model.named_parameters() if p.requires_grad}
        assert sum(p.numel() for p in params.values()) == 9738
        assert all(k.startswith("fc.") for k in params)
        before = adapter.projection.clone()
        optimizer = torch.optim.SGD(params.values(), lr=0.01)
        model.fc(x).square().mean().backward()
        optimizer.step()
        assert torch.equal(adapter.projection, before)
        assert adapter.up.weight.abs().max() > 0


def test_adapter_split_and_full_execution_match(monkeypatch):
    import dynfed.split_learning as sl
    monkeypatch.setattr(sl, "_make_torchvision_resnet", lambda *a, **kw: resnet18(weights=None))
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            end, edge = sl.build_split_pair("resnet18_pretrained_adapter", torch.device("cpu"), 3, 32)
            full = sl.build_full_model("resnet18_pretrained_adapter", torch.device("cpu"), 3, 32)
            full.load_state_dict({**end.state_dict(), **edge.state_dict()}, strict=True)
            end.eval()
            edge.eval()
            full.eval()
            x = torch.randn(2, 3, 32, 32)
            with torch.no_grad():
                torch.testing.assert_close(full(x), edge(end(x)))
            assert not any(p.requires_grad for p in end.parameters())
            assert sum(p.numel() for p in full.parameters() if p.requires_grad) == 9738
    finally:
        torch.set_num_threads(threads)


def test_same_coordinate_noise_has_dimension_dependent_rms_norm():
    sigma = 0.01382240062317938
    head = sigma * math.sqrt(5130)
    formal_scope = sigma * math.sqrt(10490890)
    assert head == pytest.approx(0.9900158754316749)
    assert formal_scope == pytest.approx(44.770262720651466)
    assert formal_scope / head == pytest.approx(math.sqrt(10490890 / 5130))
    middle = sigma * math.sqrt(8393738)
    assert middle / formal_scope == pytest.approx(math.sqrt(8393738 / 10490890))
    assert 40 < middle < 40.1


@pytest.mark.parametrize("model_name,expected,prefixes", [
    ("resnet18_pretrained_layer4_head", 8393738, ("layer4.", "fc.")),
    ("resnet18_pretrained_adapter", 9738, ("fc.",)),
])
def test_local_training_does_not_update_frozen_parameters_or_buffers(model_name, expected, prefixes):
    import numpy as np
    from dynfed.split_learning import split_local_train_lenet5

    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(17)
            model = resnet18(weights=None, num_classes=10)
            if model_name == "resnet18_pretrained_adapter":
                from dynfed.split_learning import FixedProjectionAdapter
                model.fc = torch.nn.Sequential(FixedProjectionAdapter(), model.fc)
        initial = {name: value.detach().clone() for name, value in model.state_dict().items()}
        x = np.random.default_rng(17).normal(size=(2, 3, 32, 32)).astype(np.float32)
        diff = split_local_train_lenet5(
            "LIIC", initial, {}, x, np.array([0, 1]), 1, 0.01, torch.device("cpu"),
            model_name=model_name, input_shape=(3, 32, 32),
            training_seed=17, model_cache={"full": model})
        trainable = {k for k, p in model.named_parameters() if p.requires_grad}
        assert set(diff["end"]) == trainable
        assert not diff["edge"]
        assert sum(v.numel() for v in diff["end"].values()) == expected
        assert all(k.startswith(prefixes) for k in trainable)
        assert any(bool(v.abs().max() > 0) for v in diff["end"].values())
        for name, value in model.state_dict().items():
            if name not in trainable:
                assert torch.equal(value, initial[name]), name
    finally:
        torch.set_num_threads(threads)
