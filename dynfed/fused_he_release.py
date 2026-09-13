"""Method 2 release adapter: noisy client contributions -> trusted edge packets.

The cloud worker receives only ciphertexts. Each call authorizes one release
in a fresh custodian session; the training release account owns the FL horizon.
"""
from __future__ import annotations

import numpy as np
import torch


def aggregate_fused_release(updates, weights, client_edges, end, edge, metrics):
    from experiments.trusted_edge_custodian import TrustedEdgeCustodian

    layout = [(part, name, p) for part, model in (("end", end), ("edge", edge))
              for name, p in model.named_parameters() if p.requires_grad]
    if not layout or len(updates) != len(weights) or not updates:
        raise ValueError("Nonempty fixed update layout and matching weights required")
    weights = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Positive finite public release weights required")
    weights = weights / weights.sum()
    packets, masses = {}, {}
    for (diff, _count, _candidate, ids), weight in zip(updates, weights):
        domains = {client_edges[cid] for cid in ids}
        if len(domains) != 1:
            raise ValueError("Each trusted packet must belong to one edge domain")
        domain = domains.pop()
        vector = np.concatenate([
            diff[part][name].detach().cpu().numpy().reshape(-1)
            for part, name, _param in layout
        ]).astype(np.float64)
        packets[domain] = packets.get(domain, np.zeros_like(vector)) + weight * vector
        masses[domain] = masses.get(domain, 0.0) + float(weight)
    expected = sum(packets.values())
    groups = [{"edge": domain, "cloud_weight": masses[domain]} for domain in sorted(masses)]
    packets = {domain: value / masses[domain] for domain, value in packets.items()}
    with TrustedEdgeCustodian(groups, custodian_edge=min(masses), release_limit=1) as custodian:
        decoded, audit = custodian.aggregate(packets, 0)
    audit["max_abs_error"] = float(np.max(np.abs(decoded - expected)))
    audit["custodian_session_scope"] = "one_global_release_fresh_key"
    for field in ("encrypted_updates", "encrypted_parameter_values", "ciphertext_count",
                  "ciphertext_bytes", "key_setup_time_sec", "encryption_time_sec",
                  "decryption_time_sec"):
        setattr(metrics, field, getattr(metrics, field) + audit[field])
    metrics.aggregation_calls += 1
    metrics.wall_time_sec += audit["wall_time_sec"] + audit["key_setup_time_sec"]
    metrics.addition_time_sec += audit["trusted_ciphertext_verification_sum_sec"]
    metrics.max_abs_error = max(metrics.max_abs_error, audit["max_abs_error"])
    cursor = 0
    with torch.no_grad():
        for _part, _name, param in layout:
            count = param.numel()
            param.add_(torch.as_tensor(decoded[cursor:cursor + count].copy(),
                                       device=param.device, dtype=param.dtype).view_as(param))
            cursor += count
    return audit
