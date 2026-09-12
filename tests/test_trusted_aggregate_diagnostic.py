import math

import pytest

from experiments.validate_trusted_aggregate_dp import release_scales


@pytest.mark.parametrize("clients", [1, 4, 20, 100])
def test_independent_equal_weight_replacement_scales(clients):
    scales = release_scales(clients, 1.0, 6.0)
    assert scales["aggregate_sensitivity"] == pytest.approx(2 / clients)
    assert scales["aggregate_noise_std"] == pytest.approx(12 / clients)
    assert scales["packet_dp_average_noise_std"] / scales["aggregate_noise_std"] == pytest.approx(math.sqrt(clients))


@pytest.mark.parametrize("clients,clip,multiplier", [(0, 1, 1), (2, 0, 1), (2, 1, 0), (2, 1, float("nan"))])
def test_invalid_scales_fail(clients, clip, multiplier):
    with pytest.raises(ValueError):
        release_scales(clients, clip, multiplier)


def test_sweep_scaling_matches_complete_packet_clipping():
    import torch
    from dynfed.split_learning import clip_state_difference

    diff = {"end": {"w": torch.tensor([3.0, 4.0])}, "edge": {"b": torch.tensor([12.0])}}
    raw = torch.tensor([3.0, 4.0, 12.0])
    for c in [100.0, 1.0, 0.1, 0.00001]:
        clipped, norm, _ = clip_state_difference(diff, c, torch.device("cpu"))
        flattened = torch.cat([clipped["end"]["w"], clipped["edge"]["b"]])
        torch.testing.assert_close(flattened, raw * min(1.0, c / norm))


def test_noise_to_clipped_signal_is_scale_invariant_when_all_clip():
    ratios = []
    for c in [0.1, 0.01, 0.001]:
        # Aligned independent unit updates, all clipped, have mean norm C.
        ratios.append(release_scales(20, c, 6.0)["aggregate_noise_std"] / c)
    assert ratios == pytest.approx([0.6] * 3)
