from __future__ import annotations
import numpy as np
import pytest
import torch
from dynfed.randomk_update import (
    public_randomk_mask, project_update, diff_norm, release_projected_aggregate,
)


def ref():
    return {"end": {"a": torch.tensor([1., 2., 3., 4.]), "b": torch.tensor([5., 6.])}, "edge": {}}


def test_exact_random_selection():
    assert sum(int(v.sum()) for part in public_randomk_mask(ref(), .5, 4).values() for v in part.values()) == 3


def test_mask_reproducible():
    x, y = public_randomk_mask(ref(), .5, 4), public_randomk_mask(ref(), .5, 4)
    assert all(torch.equal(v, y[k][n]) for k, part in x.items() for n, v in part.items())


def test_invalid_fraction():
    with pytest.raises(ValueError): public_randomk_mask(ref(), 0, 1)
    with pytest.raises(ValueError): public_randomk_mask(ref(), 1.1, 1)


def test_project_discards_unselected_raw_values():
    mask = public_randomk_mask(ref(), .5, 4)
    projected = project_update(ref(), mask)
    for part, values in projected.items():
        for name, tensor in values.items():
            assert torch.count_nonzero(tensor[~mask[part][name]]) == 0


def test_clip_before_aggregate():
    m = public_randomk_mask(ref(), 1, 3)
    x, d = release_projected_aggregate([ref(), ref()], [1,1], m,
                                        clip_norm=.5, noise_multiplier=1, seed=4)
    assert d['signal_norm'] == pytest.approx(.5, abs=1e-5)
    assert d['sensitivity'] == pytest.approx(.5)
    assert d['clipped_clients'] == 2


def test_noise_never_appears_outside_mask():
    m = public_randomk_mask(ref(), 1/3, 3)
    out, d = release_projected_aggregate([ref(), ref()], [1,1], m,
                                          clip_norm=.5, noise_multiplier=1, seed=4)
    for part, values in out.items():
        for name, value in values.items():
            assert torch.count_nonzero(value[~m[part][name]]) == 0
    assert d['selected_coordinates'] == 2


def test_release_deterministic_with_seed():
    m = public_randomk_mask(ref(), 1/3, 3)
    a,_ = release_projected_aggregate([ref(), ref()], [1,1], m, clip_norm=.5,noise_multiplier=1,seed=4)
    b,_ = release_projected_aggregate([ref(), ref()], [1,1], m, clip_norm=.5,noise_multiplier=1,seed=4)
    assert all(torch.equal(t,b[k][n]) for k,part in a.items() for n,t in part.items())


def test_public_weight_controls_sensitivity():
    m = public_randomk_mask(ref(), 1, 3)
    _, d = release_projected_aggregate([ref(), ref()], [3,1],m,
                                        clip_norm=.5,noise_multiplier=1,seed=4)
    assert d['sensitivity'] == pytest.approx(.75)
