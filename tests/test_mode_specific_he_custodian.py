import sys
import types

import numpy as np
import pytest
import torch

from dynfed.fused_he_release import aggregate_fused_release
from dynfed.he_backend import HEOperationMetrics


class _FakeCustodian:
    last_groups = None

    def __init__(self, groups, *, custodian_edge=0, release_limit=1):
        self.groups = tuple((item["edge"], float(item["cloud_weight"])) for item in groups)
        type(self).last_groups = self.groups

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def aggregate(self, packets, round_idx):
        result = sum(packets[edge] * weight for edge, weight in self.groups)
        dimensions = int(result.size)
        return result, {
            "backend": "seal",
            "encrypted_updates": len(self.groups),
            "encrypted_parameter_values": len(self.groups) * dimensions,
            "ciphertext_count": len(self.groups),
            "ciphertext_bytes": 123,
            "key_setup_time_sec": 0.01,
            "encryption_time_sec": 0.02,
            "trusted_ciphertext_verification_sum_sec": 0.03,
            "decryption_time_sec": 0.04,
            "wall_time_sec": 0.05,
            "cloud_secret_key_transmitted": False,
            "key_isolation_enforced": True,
            "aggregate_only_decryption_enforced": True,
        }


def _install_fake_custodian(monkeypatch):
    module = types.ModuleType("experiments.trusted_edge_custodian")
    module.TrustedEdgeCustodian = _FakeCustodian
    monkeypatch.setitem(sys.modules, "experiments.trusted_edge_custodian", module)


def _update(value, client_id):
    return ({"end": {}, "edge": {"weight": torch.tensor([[value]], dtype=torch.float32)}},
            1, None, [client_id])


def test_mode_specific_mixed_he_preserves_global_weighting(monkeypatch):
    _install_fake_custodian(monkeypatch)
    end = torch.nn.Linear(1, 1, bias=False)
    end.weight.requires_grad_(False)
    edge = torch.nn.Linear(1, 1, bias=False)
    initial = edge.weight.detach().clone()
    updates = [_update(0.1, 0), _update(0.2, 1), _update(-0.3, 2)]
    metrics = HEOperationMetrics("seal")

    audit = aggregate_fused_release(
        updates, [1, 2, 7], {0: 0, 1: 0, 2: 1}, end, edge, metrics,
        encrypted_mask=[True, False, True],
    )

    # Global weighted update: 0.1*0.1 + 0.2*0.2 + 0.7*(-0.3) = -0.16.
    torch.testing.assert_close(edge.weight, initial - 0.16, rtol=0, atol=1e-7)
    assert audit["key_isolation_enforced"]
    assert audit["aggregate_only_decryption_enforced"]
    assert audit["mixed_release"]
    assert audit["encrypted_release_weight"] == pytest.approx(0.8)
    assert audit["plaintext_release_weight"] == pytest.approx(0.2)
    assert audit["max_abs_error"] < 1e-12
    # Only the HE subset enters the custodian; its weights are renormalized there.
    assert _FakeCustodian.last_groups[0][0] == 0
    assert _FakeCustodian.last_groups[1][0] == 1
    assert _FakeCustodian.last_groups[0][1] == pytest.approx(0.125)
    assert _FakeCustodian.last_groups[1][1] == pytest.approx(0.875)


def test_mode_specific_release_rejects_empty_he_subset(monkeypatch):
    _install_fake_custodian(monkeypatch)
    end = torch.nn.Linear(1, 1, bias=False)
    end.weight.requires_grad_(False)
    edge = torch.nn.Linear(1, 1, bias=False)
    metrics = HEOperationMetrics("seal")
    updates = [_update(0.1, 0), _update(0.2, 1)]

    try:
        aggregate_fused_release(
            updates, [1, 1], {0: 0, 1: 1}, end, edge, metrics,
            encrypted_mask=[False, False],
        )
    except ValueError as exc:
        assert "HE-protected" in str(exc)
    else:
        raise AssertionError("empty HE subset must be rejected")
