"""Mechanism/accountant scale contract tests (not an end-to-end DP proof)."""

import numpy as np
import pytest
import torch

from dynfed.privacy import gaussian_rdp
from dynfed.split_learning import (
    _protect_batched_average_gradient_dp,
    _protect_tensor_dp,
)


class CapturingGaussian:
    """Observe the noise parameters without Monte Carlo uncertainty."""

    def __init__(self):
        self.calls = []

    def normal(self, loc, scale, size):
        self.calls.append((float(loc), float(scale), tuple(size)))
        return np.zeros(size, dtype=np.float32)


@pytest.mark.parametrize("batch_size", [1, 2, 7])
def test_sample_embedding_release_clips_rows_and_uses_replacement_sensitivity(batch_size):
    clip, multiplier = 0.25, 3.0
    x = torch.tensor([[3.0, 4.0]] * batch_size)
    rng = CapturingGaussian()
    diagnostics = {}
    released = _protect_tensor_dp(
        x, "dp", clip, multiplier, rng, torch.device("cpu"),
        epsilon=1.0, diagnostics=diagnostics,
    )
    expected = torch.tensor([[0.15, 0.20]] * batch_size)
    torch.testing.assert_close(released, expected, atol=1e-7, rtol=0)
    assert rng.calls == [(0.0, 2.0 * clip * multiplier, (batch_size, 2))]
    assert diagnostics["feature_dp_release_batches"] == 1
    assert diagnostics["feature_dp_sample_count"] == batch_size
    # Non-subsampled Gaussian RDP charged per release, not per tensor row.
    order = 4.0
    assert gaussian_rdp(multiplier, orders=(order,))[order] == pytest.approx(order / (2 * multiplier**2))


@pytest.mark.parametrize("batch_size", [1, 2, 7])
def test_sample_label_gradient_release_clips_rows_and_averages_scale(batch_size):
    clip, multiplier = 0.25, 2.0
    gradients = torch.tensor([[3.0, 4.0]] * batch_size)
    rng = CapturingGaussian()
    released = _protect_batched_average_gradient_dp(
        gradients, "dp", clip, multiplier, rng, torch.device("cpu"),
    )
    expected = torch.tensor([[0.15 / batch_size, 0.20 / batch_size]] * batch_size)
    torch.testing.assert_close(released, expected, atol=1e-7, rtol=0)
    assert rng.calls == [
        (0.0, 2.0 * clip * multiplier / batch_size, (batch_size, 2))
    ]
    order = 4.0
    assert gaussian_rdp(multiplier, orders=(order,))[order] == pytest.approx(order / (2 * multiplier**2))


def test_sample_label_gradient_perturbs_every_coordinate_not_only_batch_mean():
    """Document the released tensor's shape (B,d), not an aggregated d-vector."""
    batch = torch.ones((3, 5))
    rng = CapturingGaussian()
    output = _protect_batched_average_gradient_dp(
        batch, "dp", 1.0, 2.0, rng, torch.device("cpu")
    )
    assert output.shape == batch.shape
    assert rng.calls[0][2] == (3, 5)
