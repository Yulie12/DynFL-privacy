"""Audit whether larger update-DP sigma tiers preserve future privacy flexibility.

Diagnostic only: no selector, accounting, or training behavior is changed.
"""
from __future__ import annotations

from dynfed.selection import (
    SelectionConfig,
    _mode_link_transmissions,
    build_client_privacy_ledger,
    paper_client_privacy_requirement,
)
from dynfed.training import MODE_SPECS


def _paper_required_update_events(mode: str, config: SelectionConfig) -> int:
    requirement = paper_client_privacy_requirement()
    spec = MODE_SPECS[mode]
    return sum(
        count
        for link_id, obj, count, privacy_eligible in _mode_link_transmissions(
            mode, config.L_block_cycles, spec.E_edge_loops
        )
        if privacy_eligible
        and obj == "upd"
        and link_id in requirement.dp_required_links
    )


def main() -> None:
    config = SelectionConfig(rounds=20, dp_tier_gamma=1.5, dp_tier_count=3)
    per_mode_events = {
        mode: _paper_required_update_events(mode, config)
        for mode in MODE_SPECS
    }
    max_events = max(per_mode_events.values(), default=0)

    print("Step42 future privacy-flexibility audit")
    print(f"paper-required update-DP events per mode: {per_mode_events}")
    print(f"maximum protected update-DP events in one mode-round: {max_events}")

    if max_events != 1:
        raise SystemExit(
            "FAIL: current paper mode family no longer has at most one required update-DP event per round"
        )

    horizon = int(config.rounds)
    base_ledger = build_client_privacy_ledger(config)
    sigma_min = base_ledger.minimum_feasible_update_noise(max_events * horizon)
    tiers = [
        sigma_min * (float(config.dp_tier_gamma) ** tier)
        for tier in range(int(config.dp_tier_count))
    ]

    future_events = max_events * (horizon - 1)
    future_sigma_floors: list[float] = []
    all_preserve_full_horizon = True
    for sigma in tiers:
        ledger = build_client_privacy_ledger(config)
        projection = ledger.add(0, max_events, update_noise_multiplier=sigma)
        future_sigma = ledger.minimum_feasible_update_noise(future_events)
        future_sigma_floors.append(future_sigma)
        preserves = ledger.update.can_add_events(sigma_min, future_events)
        all_preserve_full_horizon = all_preserve_full_horizon and preserves
        print(
            "tier "
            f"sigma={sigma:.10f}: epsilon_after_current={projection.update_epsilon_after:.10f}, "
            f"future_sigma_floor={future_sigma:.10f}, "
            f"remaining_{horizon - 1}_rounds_feasible_at_original_sigma_min={preserves}"
        )

    if not all_preserve_full_horizon:
        raise SystemExit(
            "FAIL: sigma_min did not preserve the full remaining one-release-per-round horizon"
        )
    if not all(
        right < left
        for left, right in zip(future_sigma_floors, future_sigma_floors[1:])
    ):
        raise SystemExit(
            "FAIL: larger current sigma did not lower the future lifetime-aware sigma floor"
        )

    print("PASS: every current tier preserves all paper-required future mode-round DP exposures")
    print("PASS: sigma_min already preserves one protected update release in every remaining round")
    print("PASS: larger current sigma lowers the future lifetime-aware sigma floor")
    print(
        "CONCLUSION: tiers do not expand future mode-feasibility coverage/horizon under the "
        "current paper exposure model; they only create intertemporal noise-allocation slack"
    )


if __name__ == "__main__":
    main()
