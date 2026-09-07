import math

import pytest
import torch

from dynfed.fmnist_lenet5_dynamic import _dp_update_release_parameters
from dynfed.privacy import training_privacy_diagnostics
from dynfed.split_learning import clip_state_difference


def metrics(**changes):
    row = dict(train_loss=1.0, test_loss=1.0, train_accuracy=0.3,
               test_accuracy=0.3, global_update_norm=0.2,
               pre_dp_global_update_norm=0.1, update_dp_noise_norm=0.0,
               update_dp_clipped_clients=1, update_dp_preclip_norm_count=4)
    row.update(changes)
    return row


def test_noise_diagnostic_distinguishes_finite_and_noise_dominated():
    assert training_privacy_diagnostics(metrics())["training_health"] == "finite"
    result = training_privacy_diagnostics(metrics(update_dp_noise_norm=0.5))
    assert result["training_health"] == "noise_dominates_update"
    assert result["update_dp_noise_to_signal_ratio"] == 5.0
    assert result["update_dp_clip_fraction"] == 0.25


@pytest.mark.parametrize("field", ["train_loss", "test_loss", "global_update_norm",
                                  "update_dp_noise_norm", "test_accuracy"])
def test_non_finite_metrics_are_not_reported_as_success(field):
    assert training_privacy_diagnostics(metrics(**{field: float("nan")}))["training_health"] == "non_finite"


def test_zero_signal_is_explicitly_noise_dominated():
    result = training_privacy_diagnostics(metrics(pre_dp_global_update_norm=0.0,
                                                 update_dp_noise_norm=1.0))
    assert result["training_health"] == "noise_dominates_update"
    assert math.isfinite(result["update_dp_noise_to_signal_ratio"])


def test_float32_norm_overflow_does_not_zero_a_finite_update():
    update = {"end": {"weight": torch.tensor([1e20, -1e20])}}
    clipped, norm, scale = clip_state_difference(update, 1.0, torch.device("cpu"))
    assert math.isfinite(norm) and 0.0 < scale < 1.0
    assert torch.linalg.vector_norm(clipped["end"]["weight"]).item() == pytest.approx(1.0)
    assert update["end"]["weight"][0].item() > 1e19


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_non_finite_update_is_rejected_before_clipping(value):
    with pytest.raises(ValueError, match="non-finite"):
        clip_state_difference({"end": {"weight": torch.tensor([value])}},
                              1.0, torch.device("cpu"))


@pytest.mark.parametrize("fraction,clip,sigma", [(-0.1, 1, 1), (1.1, 1, 1),
    (float("nan"), 1, 1), (1, 0, 1), (1, 1, 0), (1, 1, float("inf"))])
def test_packet_calibration_rejects_invalid_parameters(fraction, clip, sigma):
    with pytest.raises(ValueError):
        _dp_update_release_parameters(fraction, clip_norm=clip, noise_multiplier=sigma)
