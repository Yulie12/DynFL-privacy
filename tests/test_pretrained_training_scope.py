"""Check the two existing trainable scopes without downloading model weights."""
import math

import pytest
import torch
from torchvision.models import resnet18

from dynfed.split_learning import _prepare_model_for_training, clip_state_difference


@pytest.mark.parametrize("name,expected", [("resnet18_pretrained_head", 5130),
                                           ("resnet18_pretrained", 10490890)])
def test_trainable_scope_and_batchnorm_are_explicit(name, expected):
    with torch.random.fork_rng(devices=[]):
        model = resnet18(weights=None, num_classes=10)
    model.train()
    _prepare_model_for_training(model, name)
    parameters = {k: p for k, p in model.named_parameters() if p.requires_grad}
    assert sum(p.numel() for p in parameters.values()) == expected
    prefixes = ("fc.",) if name.endswith("_head") else ("layer3.", "layer4.", "fc.")
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


def test_same_coordinate_noise_has_dimension_dependent_rms_norm():
    sigma = 0.01382240062317938
    head = sigma * math.sqrt(5130)
    formal_scope = sigma * math.sqrt(10490890)
    assert head == pytest.approx(0.9900158754316749)
    assert formal_scope == pytest.approx(44.770262720651466)
    assert formal_scope / head == pytest.approx(math.sqrt(10490890 / 5130))
