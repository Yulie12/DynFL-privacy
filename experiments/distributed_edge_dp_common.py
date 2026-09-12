"""Conditional noise algebra and real HE arithmetic, not a security protocol."""
from __future__ import annotations

from dataclasses import asdict
import math

import torch

from experiments.edge_dp_common import edge_release_plan
from dynfed.protection_rules import aggregate_replacement_bound


def distributed_noise_plan(counts, edges, clip_norm, noise_multiplier, minimum_unknown_edges):
    """Allocate equal variance AFTER public cloud weighting.

    Unknown means noise independent of the adversary's view. This function
    cannot certify honest nodes, key isolation, or secrecy of noise shares.
    """
    groups = edge_release_plan(counts, edges, clip_norm)
    if type(minimum_unknown_edges) is not int or not 1 <= minimum_unknown_edges <= edges:
        raise ValueError("Minimum unknown edges must be between one and edge count")
    if not math.isfinite(noise_multiplier) or noise_multiplier <= 0:
        raise ValueError("Positive finite noise multiplier required")
    sensitivity = aggregate_replacement_bound(
        [n / sum(counts) for n in counts], [{i} for i in range(len(counts))], clip_norm=clip_norm)
    target_std = sensitivity * noise_multiplier
    weighted_share_std = target_std / math.sqrt(minimum_unknown_edges)
    for group in groups:
        group["noise_std_before_cloud_weight"] = weighted_share_std / group["cloud_weight"]
        group["weighted_noise_std"] = weighted_share_std
    return dict(groups=groups, global_sensitivity=sensitivity, target_noise_std=target_std,
                minimum_unknown_edges=minimum_unknown_edges,
                all_edges_noise_std=target_std * math.sqrt(edges / minimum_unknown_edges))


def unknown_noise_sufficient(plan, unknown_edges):
    """Offline conditional check, NOT an online collusion detector."""
    if len(set(unknown_edges)) != len(unknown_edges):
        raise ValueError("Duplicate edge IDs")
    groups = {g["edge"]: g for g in plan["groups"]}
    if any(e not in groups for e in unknown_edges):
        raise ValueError("Unknown edge ID")
    variance = math.fsum(groups[e]["weighted_noise_std"] ** 2 for e in unknown_edges)
    return variance >= plan["target_noise_std"] ** 2 * (1 - 1e-12)


def seal_edge_sum(packets, groups):
    """Encrypt all noisy edge inputs using existing SEAL path, no fallback.

    This single-process arithmetic harness possesses the private key. It is
    NOT threshold decryption and does not protect against its host process.
    """
    from dynfed.fmnist_lenet5_dynamic import HEOperationMetrics, fedavg_split_seal

    ids = [g["edge"] for g in groups]
    if set(packets) != set(ids) or len(set(ids)) != len(ids):
        raise ValueError("Fixed cohort requires exactly all planned edge packets")
    weights = [g["cloud_weight"] for g in groups]
    if (any(not math.isfinite(w) or w <= 0 for w in weights)
            or not math.isclose(sum(weights), 1, rel_tol=0, abs_tol=1e-12)):
        raise ValueError("Public cloud weights must be positive and normalized")
    vectors = [packets[i].detach().to(device="cpu", dtype=torch.float32) for i in ids]
    if not vectors or any(v.ndim != 1 or v.numel() == 0 or v.shape != vectors[0].shape
                          or not bool(torch.isfinite(v).all()) for v in vectors):
        raise ValueError("Finite matching edge vectors required")
    # Allocate zeros directly; do not consume the training random stream.
    model = torch.nn.Module()
    model.register_parameter("update", torch.nn.Parameter(torch.zeros(vectors[0].numel())))
    empty = torch.nn.Identity()
    diffs = [{"end": {"update": v}, "edge": {}} for v in vectors]
    metrics = HEOperationMetrics(backend="seal")
    fedavg_split_seal(diffs, weights, model, empty, torch.device("cpu"),
                     encrypted_mask=[True] * len(vectors), he_aggregation_size=0,
                     he_workers=1, he_metrics=metrics)
    actual = model.update.detach().clone()
    expected = sum(v.double() * w for v, w in zip(vectors, weights))
    error = float((actual.double() - expected).abs().max())
    if not math.isfinite(error) or error > 1e-5:
        raise ValueError(f"HE arithmetic check failed: {error}")
    if metrics.encrypted_parameter_values != len(vectors) * actual.numel():
        raise ValueError("Not all edge parameters were encrypted")
    return actual, dict(**asdict(metrics), paired_max_abs_error=error,
                        key_isolation_enforced=False, aggregate_only_decryption_enforced=False)
