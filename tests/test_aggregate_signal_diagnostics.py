import pytest
import torch

from experiments.edge_dp_common import aggregate_signal_diagnostics, edge_release_plan


def test_global_vectors_not_average_of_norms():
    updates = [torch.tensor([3., 4.]), torch.tensor([-3., -4.])]
    plan = edge_release_plan([1, 1], 2, 1)
    packets = {0: torch.tensor([.6, .8]), 1: torch.tensor([-.6, -.8])}
    metrics = aggregate_signal_diagnostics(updates, plan, packets, 1)
    assert metrics["preclip_norm_mean"] == 5
    assert metrics["aggregate_signal_norm"] == 0
    assert metrics["noise_signal_ratio"] is None
    assert metrics["global_clipped_fraction"] == 1


def test_known_signal_bias_and_noise():
    updates = [torch.tensor([3., 4.])]
    plan = edge_release_plan([1], 1, 1)
    metrics = aggregate_signal_diagnostics(updates, plan, {0: torch.tensor([.6, 2.8])}, 1)
    assert metrics["aggregate_signal_norm"] == pytest.approx(1)
    assert metrics["aggregate_clipping_bias_norm"] == pytest.approx(4)
    assert metrics["aggregate_noise_norm"] == pytest.approx(2)
    assert metrics["noise_signal_ratio"] == pytest.approx(2)
