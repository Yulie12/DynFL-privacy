"""Audit dimensions and noise scale without changing DP output semantics."""

import math

import numpy as np
import pytest
import torch

from dynfed.split_learning import _protect_tensor_dp


def test_feature_noise_diagnostics_match_injected_gaussian_scale():
    rng = np.random.default_rng(21)
    diagnostics = {}
    tensor = torch.ones((3, 8), dtype=torch.float32)
    output = _protect_tensor_dp(
        tensor, "dp", 0.25, 2.5, rng, torch.device("cpu"), 8.0,
        diagnostics=diagnostics,
    )
    assert output.shape == tensor.shape
    assert diagnostics["feature_dp_release_batches"] == 1
    assert diagnostics["feature_dp_sample_count"] == 3
    assert diagnostics["feature_dimension_sum"] == 24
    assert diagnostics["feature_noise_std_sum"] == pytest.approx(3 * 1.25)
    assert diagnostics["feature_expected_noise_norm_sum"] == pytest.approx(
        3 * 1.25 * math.sqrt(8)
    )
    assert diagnostics["feature_clipped_sample_count"] == 3
    assert diagnostics["feature_clipped_norm_sum"] == pytest.approx(3 * .25)


def test_no_feature_dp_has_no_audit_events():
    diagnostics = {}
    tensor = torch.ones((3, 8), dtype=torch.float32)
    output = _protect_tensor_dp(
        tensor, "none", .25, 2.5, np.random.default_rng(1),
        torch.device("cpu"), 8.0, diagnostics=diagnostics,
    )
    assert output is tensor
    assert diagnostics == {}
