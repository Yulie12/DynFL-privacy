from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.privacy import calibrate_gaussian_noise
from dynfed.selection import (
    SelectionConfig,
    _mode_link_transmissions,
    _record_dp_event_count,
    resolved_privacy_parameters,
)
from dynfed.training import MODE_SPECS


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Audit mode-specific record-level feature-DP release counts and compare them with the shared worst-case calibration horizon."
    )
    p.add_argument("--rounds", type=int, default=1)
    p.add_argument("--local-epochs", type=int, default=3)
    p.add_argument("--epsilon", type=float, default=8.0)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--L-block-cycles", type=int, default=5)
    p.add_argument("--exclude-mode", action="append", default=[])
    p.add_argument("--output", type=Path, default=Path("out/mode_feature_release_audit.json"))
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.rounds <= 0:
        raise ValueError("--rounds must be positive")
    if args.local_epochs <= 0:
        raise ValueError("--local-epochs must be positive")

    cfg = SelectionConfig(
        rounds=args.rounds,
        privacy_local_epochs=args.local_epochs,
        initial_epsilon=args.epsilon,
        dp_feature_epsilon_budget=args.epsilon,
        dp_update_epsilon_budget=args.epsilon,
        dp_delta=args.delta,
        dp_accounting_mode="rdp_auto",
        L_block_cycles=args.L_block_cycles,
        excluded_modes=tuple(args.exclude_mode),
    )
    resolved = resolved_privacy_parameters(cfg)

    rows: list[dict[str, object]] = []
    for mode, spec in MODE_SPECS.items():
        if mode in cfg.excluded_modes:
            continue
        events = _mode_link_transmissions(mode, cfg.L_block_cycles, spec.E_edge_loops)
        emb_links = []
        feature_events_per_round = 0
        for link_id, obj, communication_count, privacy_eligible in events:
            if obj != "emb" or not privacy_eligible:
                continue
            record_events = _record_dp_event_count(cfg, mode, communication_count)
            feature_events_per_round += record_events
            emb_links.append(
                {
                    "link_id": link_id,
                    "communication_count": communication_count,
                    "record_dp_events": record_events,
                }
            )

        mode_horizon = args.rounds * feature_events_per_round
        isolated_sigma = (
            calibrate_gaussian_noise(args.epsilon, args.delta, mode_horizon)
            if mode_horizon > 0
            else None
        )
        rows.append(
            {
                "mode": mode,
                "edge_loops": int(spec.E_edge_loops),
                "embedding_links": emb_links,
                "feature_events_per_round": feature_events_per_round,
                "mode_feature_horizon_events": mode_horizon,
                "isolated_mode_sigma_if_fixed_for_all_rounds": isolated_sigma,
            }
        )

    payload = {
        "rounds": args.rounds,
        "privacy_local_epochs": args.local_epochs,
        "epsilon": args.epsilon,
        "delta": args.delta,
        "shared_max_feature_events_per_round": resolved["max_feature_events_per_round"],
        "shared_feature_horizon_events": resolved["feature_horizon_events"],
        "shared_feature_noise_multiplier": resolved["feature_noise_multiplier"],
        "important_note": (
            "Per-mode isolated sigma values are diagnostics only. They are not automatically safe to mix under dynamic mode switching; "
            "the accountant must compose the actual per-release sigmas if mode-dependent feature noise is enabled."
        ),
        "modes": rows,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print(
        f"shared calibration: max_events/round={resolved['max_feature_events_per_round']} "
        f"horizon={resolved['feature_horizon_events']} sigma={float(resolved['feature_noise_multiplier']):.9g}"
    )
    print("mode       events/round   horizon   isolated_sigma")
    for row in rows:
        sigma = row["isolated_mode_sigma_if_fixed_for_all_rounds"]
        sigma_text = "-" if sigma is None else f"{float(sigma):.9g}"
        print(
            f"{str(row['mode']):10s} {int(row['feature_events_per_round']):12d} "
            f"{int(row['mode_feature_horizon_events']):9d}   {sigma_text}"
        )
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
