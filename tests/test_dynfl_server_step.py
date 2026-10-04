from __future__ import annotations

import pytest
import torch

from dynfed.fmnist_lenet5_dynamic import (
    _apply_cloud_server_step,
    _snapshot_trainable_parameters,
)


def _models():
    end = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.Linear(2, 1))
    edge = torch.nn.Linear(2, 2)
    end[0].weight.requires_grad_(False)
    end[0].bias.requires_grad_(False)
    return end, edge


def test_step_half_scales_complete_noisy_update_and_leaves_frozen_weights():
    end, edge = _models()
    frozen_before = end[0].weight.detach().clone()
    before = _snapshot_trainable_parameters(end, edge)
    with torch.no_grad():
        for model in (end, edge):
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.add_(2.0)
    _apply_cloud_server_step(end, edge, before, 0.5)
    for label, model in (("end", end), ("edge", edge)):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter, before[label][name] + 1.0)
    torch.testing.assert_close(end[0].weight, frozen_before)


def test_default_step_one_preserves_existing_aggregation():
    end, edge = _models()
    before = _snapshot_trainable_parameters(end, edge)
    with torch.no_grad():
        for model in (end, edge):
            for parameter in model.parameters():
                if parameter.requires_grad:
                    parameter.add_(3.0)
    _apply_cloud_server_step(end, edge, before, 1.0)
    for label, model in (("end", end), ("edge", edge)):
        for name, parameter in model.named_parameters():
            if parameter.requires_grad:
                torch.testing.assert_close(parameter, before[label][name] + 3.0)


def test_invalid_step_rejected():
    end, edge = _models()
    before = _snapshot_trainable_parameters(end, edge)
    for invalid in (0.0, -0.5, 1.5):
        with pytest.raises(ValueError, match="server_step"):
            _apply_cloud_server_step(end, edge, before, invalid)


def test_missing_trainable_parameter_rejected():
    end, edge = _models()
    before = _snapshot_trainable_parameters(end, edge)
    del before["edge"]["weight"]
    with pytest.raises(RuntimeError, match="Trainable parameter set"):
        _apply_cloud_server_step(end, edge, before, 0.5)
