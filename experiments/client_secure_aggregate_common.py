"""Fixed-cohort client secure aggregation with distributed aggregate-DP noise.

This module is a protocol primitive for the untrusted-edge threat model.  Each
client keeps an X25519 private key, clips its update locally, adds one Gaussian
noise share, and masks the weighted packet with pairwise PRG masks.  The edge
receives only masked packets and public metadata.  Pairwise masks cancel only
when the complete fixed cohort is present; dropout therefore aborts the release.

The implementation is an experimental harness, not a production SecAgg stack:
it does not implement authenticated transport, dropout recovery, malicious-client
proofs, or hardened process isolation.  Its security claim is conditional on
client private keys and at least ``minimum_unknown_clients`` noise seeds being
hidden from the edge/colluding clients.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import os
from typing import Mapping

import torch
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import x25519

from dynfed.protection_rules import aggregate_replacement_bound


@dataclass(frozen=True)
class ClientSecureAggregatePlan:
    weights: tuple[float, ...]
    clip_norm: float
    noise_multiplier: float
    minimum_unknown_clients: int
    aggregate_sensitivity: float
    target_noise_std: float
    client_noise_share_std: float
    all_clients_noise_std: float


def client_secure_aggregate_plan(
    weights: list[float] | tuple[float, ...],
    clip_norm: float,
    noise_multiplier: float,
    minimum_unknown_clients: int,
) -> ClientSecureAggregatePlan:
    """Calibrate aggregate sensitivity and distributed client noise shares.

    Replacement adjacency is used.  Every client clips its *unweighted* update
    to ``clip_norm`` before multiplying by its public aggregation weight.  If at
    least ``minimum_unknown_clients`` independently generated noise shares remain
    unknown to the adversary, their conditional variance is at least the target
    Gaussian variance.
    """
    ws = tuple(float(w) for w in weights)
    if len(ws) < 2 or any(not math.isfinite(w) or w <= 0.0 for w in ws):
        raise ValueError("At least two positive finite client weights are required")
    if not math.isclose(math.fsum(ws), 1.0, rel_tol=0.0, abs_tol=1e-12):
        raise ValueError("Client weights must be normalized")
    if not math.isfinite(clip_norm) or clip_norm <= 0.0:
        raise ValueError("clip_norm must be finite and positive")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0.0:
        raise ValueError("noise_multiplier must be finite and positive")
    if (type(minimum_unknown_clients) is not int
            or not 2 <= minimum_unknown_clients <= len(ws)):
        raise ValueError(
            "minimum_unknown_clients must be an integer between 2 and cohort size"
        )
    sensitivity = aggregate_replacement_bound(
        list(ws), [{i} for i in range(len(ws))], clip_norm=clip_norm
    )
    target_std = sensitivity * noise_multiplier
    share_std = target_std / math.sqrt(minimum_unknown_clients)
    return ClientSecureAggregatePlan(
        weights=ws,
        clip_norm=float(clip_norm),
        noise_multiplier=float(noise_multiplier),
        minimum_unknown_clients=minimum_unknown_clients,
        aggregate_sensitivity=sensitivity,
        target_noise_std=target_std,
        client_noise_share_std=share_std,
        all_clients_noise_std=share_std * math.sqrt(len(ws)),
    )


def unknown_client_noise_sufficient(
    plan: ClientSecureAggregatePlan, unknown_client_ids: list[int] | tuple[int, ...]
) -> bool:
    ids = tuple(int(i) for i in unknown_client_ids)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate client IDs")
    if any(i < 0 or i >= len(plan.weights) for i in ids):
        raise ValueError("Unknown client ID")
    variance = len(ids) * plan.client_noise_share_std ** 2
    return variance >= plan.target_noise_std ** 2 * (1.0 - 1e-12)


class ClientMaskingContext:
    """Client-owned X25519 key material.  Never hand this object to the edge."""

    def __init__(self, client_id: int):
        self.client_id = int(client_id)
        self._private_key = x25519.X25519PrivateKey.generate()
        self.public_key = self._private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )

    def shared_secret(self, peer_public_key: bytes) -> bytes:
        peer = x25519.X25519PublicKey.from_public_bytes(peer_public_key)
        return self._private_key.exchange(peer)


def _seed_from_shared_secret(
    shared_secret: bytes, *, round_idx: int, client_a: int, client_b: int
) -> int:
    lo, hi = sorted((int(client_a), int(client_b)))
    h = hashlib.sha256()
    h.update(b"dynfl-fixed-cohort-secagg-v1")
    h.update(shared_secret)
    h.update(int(round_idx).to_bytes(8, "little", signed=False))
    h.update(lo.to_bytes(8, "little", signed=False))
    h.update(hi.to_bytes(8, "little", signed=False))
    return int.from_bytes(h.digest()[:8], "little", signed=False) & ((1 << 63) - 1)


def _pairwise_mask(
    shared_secret: bytes,
    *,
    round_idx: int,
    client_a: int,
    client_b: int,
    dimension: int,
    mask_std: float,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(
        _seed_from_shared_secret(
            shared_secret, round_idx=round_idx, client_a=client_a, client_b=client_b
        )
    )
    return torch.randn(dimension, generator=generator, dtype=torch.float64) * mask_std


def _private_noise(dimension: int, std: float) -> torch.Tensor:
    # Seed the experimental PRG from OS entropy so the edge cannot reconstruct a
    # client's noise share from the public experiment seed.  Production systems
    # should use a reviewed CSPRNG implementation and protected client entropy.
    seed = int.from_bytes(os.urandom(8), "little", signed=False) & ((1 << 63) - 1)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    return torch.randn(dimension, generator=generator, dtype=torch.float64) * std


def clip_l2(update: torch.Tensor, clip_norm: float) -> tuple[torch.Tensor, float, bool]:
    vector = update.detach().to(device="cpu", dtype=torch.float64).reshape(-1)
    if vector.numel() == 0 or not bool(torch.isfinite(vector).all()):
        raise ValueError("Finite nonempty update required")
    norm = float(torch.linalg.vector_norm(vector))
    scale = min(1.0, clip_norm / max(norm, 1e-30))
    return vector * scale, norm, scale < 1.0


def make_masked_client_packet(
    *,
    context: ClientMaskingContext,
    update: torch.Tensor,
    public_keys: Mapping[int, bytes],
    plan: ClientSecureAggregatePlan,
    round_idx: int,
    mask_std: float = 1.0,
) -> tuple[torch.Tensor, dict[str, float | int | bool]]:
    """Create the only update-like object that the untrusted edge may receive."""
    i = context.client_id
    if i < 0 or i >= len(plan.weights) or set(public_keys) != set(range(len(plan.weights))):
        raise ValueError("Public-key roster must exactly match the fixed cohort")
    if public_keys[i] != context.public_key:
        raise ValueError("Client public key does not match roster")
    if not math.isfinite(mask_std) or mask_std <= 0.0:
        raise ValueError("mask_std must be finite and positive")
    clipped, preclip_norm, was_clipped = clip_l2(update, plan.clip_norm)
    dimension = clipped.numel()
    packet = clipped * plan.weights[i]
    noise = _private_noise(dimension, plan.client_noise_share_std)
    packet = packet + noise
    for j in range(len(plan.weights)):
        if j == i:
            continue
        shared = context.shared_secret(public_keys[j])
        mask = _pairwise_mask(
            shared,
            round_idx=round_idx,
            client_a=i,
            client_b=j,
            dimension=dimension,
            mask_std=mask_std,
        )
        packet = packet + mask if i < j else packet - mask
    return packet, {
        "client_id": i,
        "preclip_norm": preclip_norm,
        "clipped": was_clipped,
        "weight": plan.weights[i],
        "noise_share_std": plan.client_noise_share_std,
        "pairwise_masks": len(plan.weights) - 1,
        "individual_plaintext_sent_to_edge": False,
    }


def untrusted_edge_fixed_cohort_sum(
    packets: Mapping[int, torch.Tensor], *, cohort_size: int
) -> torch.Tensor:
    """Untrusted-edge operation: verify fixed cohort and sum opaque packets only."""
    if set(packets) != set(range(cohort_size)):
        raise ValueError("Fixed cohort mismatch; abort without release")
    vectors = [packets[i].detach().to(device="cpu", dtype=torch.float64).reshape(-1)
               for i in range(cohort_size)]
    if (not vectors or any(v.numel() == 0 or v.shape != vectors[0].shape
                           or not bool(torch.isfinite(v).all()) for v in vectors)):
        raise ValueError("Finite matching masked packets required")
    return torch.stack(vectors, dim=0).sum(dim=0)


def protocol_audit(
    plan: ClientSecureAggregatePlan,
    *,
    received_client_ids: list[int] | tuple[int, ...],
    assumed_unknown_client_ids: list[int] | tuple[int, ...],
) -> dict[str, object]:
    fixed_cohort = set(received_client_ids) == set(range(len(plan.weights)))
    unknown_noise_ok = unknown_client_noise_sufficient(plan, assumed_unknown_client_ids)
    at_least_two_unknown = len(set(assumed_unknown_client_ids)) >= 2
    closed = fixed_cohort and unknown_noise_ok and at_least_two_unknown
    return {
        "threat_model": "untrusted_edge_fixed_cohort_conditional_noncollusion",
        "individual_plaintext_update_visible_to_edge": False,
        "clean_aggregate_visible_to_edge": False,
        "pairwise_x25519_masking": True,
        "fixed_cohort_binding": fixed_cohort,
        "dropout_release_allowed": False,
        "minimum_unknown_clients": plan.minimum_unknown_clients,
        "assumed_unknown_clients": len(set(assumed_unknown_client_ids)),
        "unknown_noise_variance_sufficient": unknown_noise_ok,
        "two_or_more_noncolluding_clients": at_least_two_unknown,
        "aggregate_dp_sensitivity": plan.aggregate_sensitivity,
        "target_aggregate_noise_std": plan.target_noise_std,
        "all_clients_noise_std": plan.all_clients_noise_std,
        "single_edge_has_private_mask_keys": False,
        "single_edge_can_remove_all_dp_noise": not unknown_noise_ok,
        "protocol_status": (
            "untrusted_edge_secure_aggregate_dp_closed_conditional"
            if closed else "not_established"
        ),
        "formal_scope": (
            "conditional_on_client_key_secrecy_fixed_cohort_and_minimum_unknown_clients;"
            "no_dropout_recovery_or_malicious_client_proofs"
        ),
    }
