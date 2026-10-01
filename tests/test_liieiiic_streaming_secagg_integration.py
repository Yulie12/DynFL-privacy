from types import SimpleNamespace
import math

import torch

import dynfed.fmnist_lenet5_dynamic as dynamic
from dynfed.streaming_secagg import streaming_secure_aggregate_exact_target


def _update(value: float):
    return {
        "end": {"weight": torch.tensor([[value]], dtype=torch.float32)},
        "edge": {"weight": torch.tensor([[value * 2.0]], dtype=torch.float32)},
    }


def test_liieiiic_plain_dp_moves_from_local_packet_to_secure_aggregate(monkeypatch):
    candidate = SimpleNamespace(mode="LIIEIIIC", mechanism="dp")
    monkeypatch.setattr(dynamic, "_candidate_cloud_update_mechanism", lambda c: c.mechanism)
    monkeypatch.setattr(dynamic, "mechanism_uses_dp", lambda m: "dp" in m)
    monkeypatch.setattr(dynamic, "mechanism_uses_he", lambda m: "he" in m)

    assert not dynamic._candidate_uses_local_packet_update_dp(candidate)
    assert dynamic._candidate_uses_secure_aggregate_update_dp(candidate)


def test_liieiiic_streaming_secagg_requires_pure_plain_dp_edge_cohort(monkeypatch):
    dp = SimpleNamespace(mode="LIIEIIIC", mechanism="dp")
    he = SimpleNamespace(mode="LIIEIIIC", mechanism="dp_he3")
    monkeypatch.setattr(dynamic, "_candidate_cloud_update_mechanism", lambda c: c.mechanism)
    monkeypatch.setattr(dynamic, "mechanism_uses_dp", lambda m: "dp" in m)
    monkeypatch.setattr(dynamic, "mechanism_uses_he", lambda m: "he" in m)

    updates = [(1, _update(0.1), 1, dp), (2, _update(0.2), 3, dp)]
    assert dynamic._liieiiic_streaming_secagg_eligible(updates, execute_real_he=False)
    assert not dynamic._liieiiic_streaming_secagg_eligible(updates, execute_real_he=True)
    assert not dynamic._liieiiic_streaming_secagg_eligible(updates[:1], execute_real_he=False)
    assert not dynamic._liieiiic_streaming_secagg_eligible(
        [(1, _update(0.1), 1, dp), (2, _update(0.2), 3, he)],
        execute_real_he=False,
    )


def test_hierarchical_edge_noise_shares_recompose_exact_global_target_variance():
    clip_norm = 0.25
    global_sigma = 0.691120031158969

    # Edge 0 carries 2 admitted samples; Edge 1 carries 4.  Within-edge max
    # client fractions are 1/2 on both edges.  Therefore the largest final
    # client weight is max((2/6)*(1/2), (4/6)*(1/2)) = 1/3.
    edge_masses = [2.0, 4.0]
    cloud_weights = [mass / sum(edge_masses) for mass in edge_masses]
    local_max = [0.5, 0.5]
    global_max_client_weight = max(
        w * f for w, f in zip(cloud_weights, local_max)
    )
    global_target_std = global_sigma * 2.0 * clip_norm * global_max_client_weight

    # The runtime distributes equal *weighted* variance shares across the two
    # protected Edge packets, then divides by the Cloud weight to obtain each
    # Edge-local target std.
    weighted_share_std = global_target_std / math.sqrt(2.0)
    edge_target_stds = [weighted_share_std / w for w in cloud_weights]

    audits = []
    edge_updates = [
        [_update(0.1), _update(0.3)],
        [_update(0.2), _update(0.4)],
    ]
    edge_counts = [[1, 1], [2, 2]]
    for edge_index, (updates, counts, target_std, max_weight) in enumerate(
        zip(edge_updates, edge_counts, edge_target_stds, local_max)
    ):
        local_sensitivity = 2.0 * clip_norm * max_weight
        effective_sigma = target_std / local_sensitivity
        _aggregate, audit = streaming_secure_aggregate_exact_target(
            updates,
            counts,
            clip_norm=clip_norm,
            noise_multiplier=effective_sigma,
            round_seed=100 + edge_index,
            chunk_size=1,
            mask_std=0.5,
        )
        audits.append(audit)
        assert math.isclose(audit.target_noise_std, target_std, rel_tol=1e-12, abs_tol=1e-12)

    final_variance = sum(
        (weight * audit.target_noise_std) ** 2
        for weight, audit in zip(cloud_weights, audits)
    )
    assert math.isclose(
        final_variance,
        global_target_std ** 2,
        rel_tol=1e-12,
        abs_tol=1e-15,
    )
