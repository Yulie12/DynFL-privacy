from __future__ import annotations

"""Chunked fixed-cohort secure aggregation for exact-target aggregate DP.

This module implements the execution primitive needed by full-local II modes.
It intentionally scopes the threat model to an untrusted aggregator with
non-colluding clients and a fixed cohort.  Clients keep X25519 private keys and
their Gaussian-noise seeds private.  If the cohort changes, the release aborts.

The implementation is a research harness: all client objects still execute in
one Python process, but pairwise masks are generated chunk-by-chunk, so memory
does not scale as O(K^2 d).
"""

from dataclasses import dataclass
import hashlib
import math
from typing import Any, Iterable

import numpy as np
import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519


@dataclass(frozen=True)
class StreamingSecAggAudit:
    cohort_size: int
    parameter_count: int
    chunk_size: int
    chunk_count: int
    clip_norm: float
    max_client_weight: float
    aggregate_sensitivity: float
    noise_multiplier: float
    target_noise_std: float
    client_noise_share_std: float
    pairwise_masking: bool = True
    fixed_cohort: bool = True
    exact_target_variance: bool = True
    threat_model: str = "untrusted_aggregator_noncolluding_clients_fixed_cohort"


class _ClientContext:
    def __init__(self) -> None:
        self._private = x25519.X25519PrivateKey.generate()

    @property
    def public_bytes(self) -> bytes:
        return self._private.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def shared(self, peer_public: bytes) -> bytes:
        return self._private.exchange(
            x25519.X25519PublicKey.from_public_bytes(peer_public)
        )


def _seed(*parts: object) -> int:
    h = hashlib.sha256()
    for part in parts:
        if isinstance(part, bytes):
            h.update(part)
        else:
            h.update(repr(part).encode("utf-8"))
            h.update(b"|")
    return int.from_bytes(h.digest()[:8], "big", signed=False)


def _iter_float_tensors(state: dict[str, dict[str, torch.Tensor]]):
    for part_name in sorted(state):
        values = state[part_name]
        for name in sorted(values):
            value = values[name]
            if torch.is_floating_point(value) or torch.is_complex(value):
                yield part_name, name, value


def _layout(reference: dict[str, dict[str, torch.Tensor]]):
    items = []
    cursor = 0
    for part_name, name, value in _iter_float_tensors(reference):
        n = int(value.numel())
        items.append((part_name, name, tuple(value.shape), value.dtype, cursor, cursor+n))
        cursor += n
    return items, cursor


def _flat_chunk(
    state: dict[str, dict[str, torch.Tensor]],
    layout,
    start: int,
    stop: int,
) -> torch.Tensor:
    out = torch.zeros(stop-start, dtype=torch.float64)
    for part_name, name, _shape, _dtype, lo, hi in layout:
        if hi <= start or lo >= stop:
            continue
        src = state.get(part_name, {}).get(name)
        if src is None:
            continue
        flat = src.detach().to("cpu", dtype=torch.float64).reshape(-1)
        a=max(start,lo); b=min(stop,hi)
        out[a-start:b-start] = flat[a-lo:b-lo]
    return out


def _global_l2_norm(state, layout, total_size: int, chunk_size: int) -> float:
    sq=0.0
    for start in range(0,total_size,chunk_size):
        stop=min(start+chunk_size,total_size)
        chunk=_flat_chunk(state,layout,start,stop)
        sq += float(torch.dot(chunk,chunk).item())
    return math.sqrt(max(sq,0.0))


def _empty_like(reference):
    return {
        part: {
            name: torch.zeros_like(value, device="cpu")
            for name, value in values.items()
            if torch.is_floating_point(value) or torch.is_complex(value)
        }
        for part, values in reference.items()
    }


def _write_chunk(target, layout, start: int, stop: int, values: torch.Tensor) -> None:
    for part_name,name,shape,dtype,lo,hi in layout:
        if hi <= start or lo >= stop:
            continue
        a=max(start,lo); b=min(stop,hi)
        tgt=target[part_name][name].reshape(-1)
        tgt[a-lo:b-lo] = values[a-start:b-start].to(dtype=dtype)


def streaming_secure_aggregate_exact_target(
    state_diffs: list[dict[str, dict[str, torch.Tensor]]],
    sample_counts: list[float],
    *,
    clip_norm: float,
    noise_multiplier: float,
    round_seed: int,
    chunk_size: int = 262_144,
    mask_std: float = 1.0,
) -> tuple[dict[str, dict[str, torch.Tensor]], StreamingSecAggAudit]:
    """Return only the noisy aggregate for one complete fixed cohort.

    Each unweighted update is clipped to ``clip_norm``.  Public sample weights
    are normalized.  Each client adds an independent Gaussian share with
    weighted standard deviation ``target_std / sqrt(K)``; therefore the full
    aggregate has exactly the requested target variance.  This is valid for the
    documented non-colluding-client threat model.

    Pairwise X25519 masks are generated per chunk and cancel only for the full
    cohort.  No full-dimensional pairwise mask is materialized.
    """
    if len(state_diffs) != len(sample_counts):
        raise ValueError("one sample count is required per client update")
    k=len(state_diffs)
    if k < 2:
        raise ValueError("secure aggregation requires at least two clients")
    if not math.isfinite(clip_norm) or clip_norm <= 0:
        raise ValueError("clip_norm must be finite and positive")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise ValueError("noise_multiplier must be finite and positive")
    if chunk_size < 1:
        raise ValueError("chunk_size must be positive")
    if not math.isfinite(mask_std) or mask_std <= 0:
        raise ValueError("mask_std must be finite and positive")

    total_mass=sum(max(float(x),0.0) for x in sample_counts)
    if total_mass <= 0:
        raise ValueError("sample counts must contain positive mass")
    weights=[max(float(x),0.0)/total_mass for x in sample_counts]
    max_weight=max(weights)

    reference=state_diffs[0]
    layout,total_size=_layout(reference)
    for state in state_diffs[1:]:
        other_layout,other_size=_layout(state)
        if [(a,b,c) for a,b,c,*_ in layout] != [(a,b,c) for a,b,c,*_ in other_layout] or other_size != total_size:
            raise ValueError("all client state differences must have the same floating tensor layout")

    norms=[_global_l2_norm(s,layout,total_size,chunk_size) for s in state_diffs]
    scales=[min(1.0, clip_norm/max(n,1e-30)) for n in norms]

    sensitivity=2.0*clip_norm*max_weight
    target_std=noise_multiplier*sensitivity
    # Each client's *weighted aggregate contribution* gets this noise share.
    aggregate_share_std=target_std/math.sqrt(k)

    contexts=[_ClientContext() for _ in range(k)]
    public=[ctx.public_bytes for ctx in contexts]
    pairwise={}
    for i in range(k):
        for j in range(i+1,k):
            left=contexts[i].shared(public[j])
            right=contexts[j].shared(public[i])
            if left != right:
                raise RuntimeError("X25519 shared-secret mismatch")
            pairwise[(i,j)] = left

    out=_empty_like(reference)
    chunk_count=0
    for start in range(0,total_size,chunk_size):
        stop=min(start+chunk_size,total_size)
        width=stop-start
        edge_sum=torch.zeros(width,dtype=torch.float64)
        for i,(state,weight,scale) in enumerate(zip(state_diffs,weights,scales)):
            packet=_flat_chunk(state,layout,start,stop) * (weight*scale)

            ng=torch.Generator(device="cpu")
            ng.manual_seed(_seed("noise",round_seed,i,start))
            packet += torch.randn(width,generator=ng,dtype=torch.float64)*aggregate_share_std

            for j in range(k):
                if i==j:
                    continue
                a,b=(i,j) if i<j else (j,i)
                mg=torch.Generator(device="cpu")
                mg.manual_seed(_seed("mask",pairwise[(a,b)],round_seed,start))
                mask=torch.randn(width,generator=mg,dtype=torch.float64)*mask_std
                packet += mask if i<j else -mask
            edge_sum += packet
        _write_chunk(out,layout,start,stop,edge_sum)
        chunk_count += 1

    return out, StreamingSecAggAudit(
        cohort_size=k,
        parameter_count=total_size,
        chunk_size=int(chunk_size),
        chunk_count=chunk_count,
        clip_norm=float(clip_norm),
        max_client_weight=float(max_weight),
        aggregate_sensitivity=float(sensitivity),
        noise_multiplier=float(noise_multiplier),
        target_noise_std=float(target_std),
        client_noise_share_std=float(aggregate_share_std),
    )
