import pytest
import torch

from experiments.validate_trusted_aggregate_trajectory import apply_vector_update, trajectory_noise_seed
from dynfed.privacy import PrivacyAccountant, calibrate_gaussian_noise


def test_next_round_retains_released_noise_without_mutating_initial_or_frozen_state():
    initial = {"end": {"fixed": torch.tensor([7.0])}, "edge": {"head": torch.tensor([1.0])}}
    keys = [("edge", "head")]
    first = apply_vector_update(initial, keys, torch.tensor([0.5]))
    second = apply_vector_update(first, keys, torch.tensor([0.25]))
    assert initial["edge"]["head"].item() == 1.0
    assert first["edge"]["head"].item() == 1.5
    assert second["edge"]["head"].item() == 1.75
    assert second["end"]["fixed"].item() == 7.0


@pytest.mark.parametrize("vector", [torch.tensor([]), torch.tensor([float("nan")]), torch.tensor([[1.0]])])
def test_malformed_releases_fail(vector):
    with pytest.raises(ValueError):
        apply_vector_update({"edge": {"h": torch.ones(1)}}, [("edge", "h")], vector)


def test_short_trajectory_preserves_full_horizon_calibration():
    sigma = calibrate_gaussian_noise(8, 1e-5, 100)
    ledger = PrivacyAccountant(8, 1e-5)
    ledger.add_events(sigma, 5)
    assert 0 < ledger.current_epsilon() < 8
    ledger.add_events(sigma, 95)
    assert ledger.current_epsilon() == pytest.approx(8)


def test_noise_stream_separates_seed_and_round():
    assert trajectory_noise_seed(40, 2, "legacy_additive") == trajectory_noise_seed(42, 0, "legacy_additive")
    seeds = [trajectory_noise_seed(s, r) for s in [40, 42, 44] for r in range(100)]
    assert len(set(seeds)) == 300
    assert trajectory_noise_seed(42, 3) == trajectory_noise_seed(42, 3)
    generator = torch.Generator().manual_seed(trajectory_noise_seed(42, 3))
    assert bool(torch.isfinite(torch.randn(10, generator=generator)).all())


@pytest.mark.parametrize("seed,round_idx,stream", [(-1, 0, "seed_sequence"), (42, -1, "seed_sequence"), (42, 0, "typo")])
def test_invalid_noise_stream_fails(seed, round_idx, stream):
    with pytest.raises(ValueError):
        trajectory_noise_seed(seed, round_idx, stream)


@pytest.mark.parametrize("step", [1.0, 0.5, 0.2])
def test_server_step_scales_complete_noisy_release(step):
    initial = {"edge": {"h": torch.tensor([1.0])}}
    signal, noise = torch.tensor([2.0]), torch.tensor([3.0])
    updated = apply_vector_update(initial, [("edge", "h")], signal + noise, step)
    assert updated["edge"]["h"].item() == pytest.approx(1 + 5 * step)
    assert initial["edge"]["h"].item() == 1


@pytest.mark.parametrize("step", [0.0, -1.0, 1.1, float("nan"), float("inf")])
def test_invalid_server_steps_fail(step):
    with pytest.raises(ValueError, match="server step"):
        apply_vector_update({"edge": {"h": torch.zeros(1)}}, [("edge", "h")], torch.ones(1), step)
