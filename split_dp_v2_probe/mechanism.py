"""Isolated experimental Gaussian embedding mechanism; NOT a complete DP training system.

One record -> one cached output per experiment context. The public linear
projection and L2 clipping give replacement sensitivity at most 2C.
Privacy cost here covers only the embedding releases, not labels/updates/logs.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Hashable

import torch


def zcdp_epsilon(multiplier: float, delta: float, releases_per_record: int = 1) -> float:
    """Conservative Gaussian zCDP -> (eps,delta)-DP without amplification.

    Gaussian std = multiplier * 2C, replacement adjacency sensitivity = 2C.
    Compose only releases associated with the SAME record. For independent
    disjoint records, use parallel composition instead of summing them.
    """
    if not 0.0 < delta < 1.0:
        raise ValueError('delta must lie in (0,1)')
    if releases_per_record < 0:
        raise ValueError('releases_per_record must be >= 0')
    if releases_per_record == 0:
        return 0.0
    if not math.isfinite(multiplier) or multiplier <= 0:
        raise ValueError('multiplier must be positive and finite')
    rho = releases_per_record / (2.0 * multiplier**2)
    return rho + 2.0 * math.sqrt(rho * math.log(1.0 / delta))


def multiplier_for_zcdp_epsilon(epsilon: float, delta: float, releases_per_record: int = 1) -> float:
    """Solve the conservative bound for a target epsilon and fixed # releases."""
    if not math.isfinite(epsilon) or epsilon <= 0:
        raise ValueError('epsilon must be positive and finite')
    if not 0.0 < delta < 1.0:
        raise ValueError('delta must lie in (0,1)')
    if releases_per_record < 1:
        raise ValueError('releases_per_record must be >= 1')
    root_rho = epsilon / (math.sqrt(math.log(1.0 / delta) + epsilon) + math.sqrt(math.log(1.0 / delta)))
    rho = root_rho * root_rho
    return math.sqrt(releases_per_record / (2.0 * rho))


def public_fixed_projection(input_dim: int, output_dim: int, seed: int = 1729) -> torch.Tensor:
    """Data-independent orthonormal columns; projection can't increase L2 norm."""
    if not 1 <= output_dim <= input_dim:
        raise ValueError('expected 1 <= output_dim <= input_dim')
    generator = torch.Generator(device='cpu').manual_seed(seed)
    raw = torch.randn((input_dim, output_dim), generator=generator, dtype=torch.float64)
    q, r = torch.linalg.qr(raw, mode='reduced')
    q *= torch.where(torch.diag(r) >= 0, 1.0, -1.0)[None, :]
    return q.float()


def clip_rows(tensor: torch.Tensor, clip_norm: float) -> torch.Tensor:
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError('clip_norm must be positive and finite')
    if tensor.ndim != 2 or tensor.shape[0] == 0:
        raise ValueError('tensor must have shape [N,D] and N>0')
    norm = torch.linalg.vector_norm(tensor, dim=1, keepdim=True).clamp_min(1e-12)
    return tensor * (clip_norm / norm).clamp(max=1.0)


@dataclass(frozen=True)
class CacheStats:
    new_releases: int
    cache_hits: int
    cached_records: int


class OneReleaseCache:
    """Volatile test cache; each (context, record_id) receives one noisy vector.

    The context must identify one FROZEN encoder, data transform and projection.
    Never reuse IDs for different individuals or split contexts; context changes
    imply fresh releases and must be composed by an external ledger.
    This class deliberately cannot claim global privacy budget tracking.
    """

    def __init__(self, clip_norm: float, multiplier: float, *, seed: int = 40):
        if not math.isfinite(multiplier) or multiplier <= 0:
            raise ValueError('multiplier must be positive')
        if not math.isfinite(clip_norm) or clip_norm <= 0:
            raise ValueError('clip_norm must be positive')
        self.clip_norm = float(clip_norm)
        self.multiplier = float(multiplier)
        self._generator = torch.Generator(device='cpu').manual_seed(seed)
        self._values: dict[tuple[Hashable, Hashable], torch.Tensor] = {}
        self._dims: dict[Hashable, int] = {}
        self._new = 0
        self._hits = 0

    def release(self, context: Hashable, record_ids: list[Hashable], features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 2 or features.shape[0] != len(record_ids):
            raise ValueError('features must be [len(record_ids), D]')
        if len(set(record_ids)) != len(record_ids):
            raise ValueError('duplicate record ID within batch (ambiguous releases)')
        if not features.is_floating_point() or not bool(torch.isfinite(features).all()):
            raise ValueError('features must be finite floating-point values')
        if context in self._dims and self._dims[context] != features.shape[1]:
            raise ValueError('context feature dimension changed')
        self._dims[context] = int(features.shape[1])
        clipped = clip_rows(features.detach().float().cpu(), self.clip_norm)
        output = []
        for i, record_id in enumerate(record_ids):
            key = (context, record_id)
            if key not in self._values:
                noise = torch.randn((features.shape[1],), generator=self._generator)
                self._values[key] = clipped[i].clone() + noise * (2.0 * self.clip_norm * self.multiplier)
                self._new += 1
            else:
                self._hits += 1
            output.append(self._values[key])
        return torch.stack(output).to(device=features.device, dtype=features.dtype)

    @property
    def stats(self) -> CacheStats:
        return CacheStats(self._new, self._hits, len(self._values))
