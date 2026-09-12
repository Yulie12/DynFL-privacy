"""Fixed public cohorts and independent-client releases for private diagnostics."""
from __future__ import annotations

import math
import numpy as np
import torch

from dynfed.protection_rules import aggregate_replacement_bound


def edge_release_plan(counts: list[int], edges: int, clip_norm: float) -> list[dict]:
    if not counts or any(type(n) is not int or n <= 0 for n in counts):
        raise ValueError("Positive fixed public sample quotas required")
    if type(edges) is not int or not 1 <= edges <= len(counts):
        raise ValueError("Edges must be between one and client count")
    result = []
    for edge in range(edges):
        clients = list(range(edge, len(counts), edges))
        total = sum(counts[i] for i in clients)
        weights = [counts[i] / total for i in clients]
        sensitivity = aggregate_replacement_bound(weights, [{i} for i in clients], clip_norm=clip_norm)
        result.append(dict(edge=edge, clients=clients, weights=weights,
                           cloud_weight=total / sum(counts), sensitivity=sensitivity))
    return result


def edge_noise_seed(seed: int, round_idx: int, edge: int) -> int:
    if any(type(x) is not int or x < 0 for x in (seed, round_idx, edge)):
        raise ValueError("Nonnegative seed, round and edge required")
    return int(np.random.SeedSequence([seed, round_idx, edge, 20000]).generate_state(
        1, dtype=np.uint64)[0])


def release_edge(updates: torch.Tensor, weights: list[float], *, clip_norm: float,
                 noise_multiplier: float, seed: int, clip: bool = True):
    """Return a noisy edge packet plus INTERNAL, non-DP diagnostic metrics.

    Each row must depend only on its own client's data and the prior public
    transcript. The caller must authorize and account for this release first.
    """
    sensitivity = aggregate_replacement_bound(weights, [{i} for i in range(len(weights))],
                                              clip_norm=clip_norm)
    if (updates.ndim != 2 or updates.shape[0] != len(weights) or updates.shape[1] < 1
            or not updates.is_floating_point() or not bool(torch.isfinite(updates).all())):
        raise ValueError("Finite floating point update rows matching weights required")
    if not math.isfinite(noise_multiplier) or noise_multiplier < 0 or (noise_multiplier > 0 and not clip):
        raise ValueError("DP requires clipping and a finite nonnegative noise multiplier")
    norms = updates.norm(dim=1)
    factors = (clip_norm / norms.clamp_min(1e-12)).clamp(max=1) if clip else torch.ones_like(norms)
    w = updates.new_tensor(weights)
    raw_mean = (updates * w[:, None]).sum(dim=0)
    clipped_mean = (updates * (w * factors)[:, None]).sum(dim=0)
    std = sensitivity * noise_multiplier
    generator = torch.Generator(device=updates.device).manual_seed(seed)
    noise = torch.randn(clipped_mean.shape, generator=generator, device=updates.device,
                        dtype=updates.dtype) * std
    packet = clipped_mean + noise
    diagnostics = dict(max_client_weight=max(weights), sensitivity=sensitivity,
                       noise_std=std, noise_norm=float(noise.norm()),
                       signal_norm=float(clipped_mean.norm()),
                       clipping_bias_norm=float((clipped_mean - raw_mean).norm()),
                       clipped_fraction=float((factors < 1).float().mean()),
                       max_postclip_norm=float((norms * factors).max()))
    return packet, diagnostics
