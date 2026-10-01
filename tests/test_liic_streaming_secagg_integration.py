from types import SimpleNamespace

import torch

from dynfed.fmnist_lenet5_dynamic import (
    _apply_aggregate_state_difference,
    _liic_streaming_secagg_eligible,
)
from dynfed.streaming_secagg import streaming_secure_aggregate_exact_target


def _update(a, b):
    return {
        "end": {"weight": torch.tensor([[a]], dtype=torch.float32)},
        "edge": {"weight": torch.tensor([[b]], dtype=torch.float32)},
    }


def test_liic_streaming_secagg_requires_pure_fixed_direct_cloud_cohort():
    liic = SimpleNamespace(mode="LIIC")
    updates = [(_update(1, 2), 3, liic, [1]), (_update(3, 4), 5, liic, [2])]
    assert _liic_streaming_secagg_eligible(
        updates, [1.0, 1.0], execute_real_he=False, mainline_fusion=False
    )
    assert not _liic_streaming_secagg_eligible(
        updates, [1.0, 1.0], execute_real_he=True, mainline_fusion=False
    )
    assert not _liic_streaming_secagg_eligible(
        updates, [1.0, 0.0], execute_real_he=False, mainline_fusion=False
    )


def test_streaming_secagg_aggregate_can_be_applied_once_to_models():
    end = torch.nn.Linear(1, 1, bias=False)
    edge = torch.nn.Linear(1, 1, bias=False)
    with torch.no_grad():
        end.weight.zero_()
        edge.weight.zero_()

    aggregate, audit = streaming_secure_aggregate_exact_target(
        [_update(0.1, 0.2), _update(0.3, 0.4)],
        [1, 1],
        clip_norm=1.0,
        noise_multiplier=1e-9,
        round_seed=17,
        chunk_size=1,
        mask_std=0.5,
    )
    _apply_aggregate_state_difference(aggregate, end, edge, torch.device("cpu"))

    # With negligible DP noise the weighted mean is [0.2, 0.3].
    assert torch.allclose(end.weight.detach(), torch.tensor([[0.2]]), atol=1e-6)
    assert torch.allclose(edge.weight.detach(), torch.tensor([[0.3]]), atol=1e-6)
    assert audit.exact_target_variance
