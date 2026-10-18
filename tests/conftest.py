"""Shared test fixtures.

The synthetic bound is only for search-mechanics tests. It makes no
certificate claim and must never appear on the formal training path.
"""
import types

import pytest

import dynfed.selection as selection_module


@pytest.fixture
def synthetic_online_bound(monkeypatch):
    def _bound(config, profile, client_samples, admitted_client_ids):
        # Aggregate cloud release averages noise across the packet and is
        # cheaper than per-client packet DP; keep that ordering visible in
        # the synthetic search score.
        packet_penalty = 2.0 * sum(
            1
            for candidate in profile.values()
            if getattr(candidate, "dp_execution_plan", None) == "cloud_packet"
        )
        return types.SimpleNamespace(
            upper=(
                1.0
                + sum(float(candidate.time) for candidate in profile.values())
                + packet_penalty
            ),
            central_norm=0.0, radius=0.0, ideal_radius=0.0,
            q_aggregate=0.0, q_aggregate_proved_floor=0.0, numeric_radius=0.0,
            lower=0.0, reference_radius=0.0, mode_radius=0.0,
            mode_variance_second_moment=0.0,
            source="synthetic_search_validation",
        )

    monkeypatch.setattr(selection_module, "_online_bound_for_profile", _bound)
    return _bound