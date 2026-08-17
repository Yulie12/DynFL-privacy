from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.nodes import build_profiles


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyze mode distribution by client compute groups.")
    parser.add_argument("--decisions", required=True)
    parser.add_argument("--output", default=None)
    parser.add_argument("--clients", type=int, default=10)
    parser.add_argument("--edges", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--client-heterogeneity", type=float, default=2.0)
    parser.add_argument("--edge-heterogeneity", type=float, default=1.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    decisions_path = Path(args.decisions)
    output_path = Path(args.output) if args.output else decisions_path.with_name("compute_group_mode_distribution.csv")

    clients, _ = build_profiles(
        num_clients=args.clients,
        num_edges=args.edges,
        client_heterogeneity=args.client_heterogeneity,
        edge_heterogeneity=args.edge_heterogeneity,
        seed=args.seed,
    )
    groups = _assign_groups({client.client_id: client.compute_factor for client in clients})

    rows = list(csv.DictReader(decisions_path.open(newline="", encoding="utf-8")))
    counts: dict[tuple[str, str], int] = {}
    totals: dict[str, int] = {}
    for row in rows:
        client_id = int(row["client_id"])
        group = groups[client_id]
        mode = row["mode"]
        counts[(group, mode)] = counts.get((group, mode), 0) + 1
        totals[group] = totals.get(group, 0) + 1

    output_rows = []
    for group in ["high_compute", "mid_compute", "low_compute"]:
        modes = sorted(mode for (g, mode) in counts if g == group)
        for mode in modes:
            count = counts[(group, mode)]
            output_rows.append(
                {
                    "compute_group": group,
                    "mode": mode,
                    "count": count,
                    "ratio": count / max(totals.get(group, 0), 1),
                    "total_group_decisions": totals.get(group, 0),
                }
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["compute_group", "mode", "count", "ratio", "total_group_decisions"],
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"[OK] wrote {output_path}")


def _assign_groups(compute_factors: dict[int, float]) -> dict[int, str]:
    ordered = sorted(compute_factors.items(), key=lambda item: item[1])
    n = len(ordered)
    groups = {}
    for idx, (client_id, _) in enumerate(ordered):
        if idx < n / 3:
            groups[client_id] = "high_compute"
        elif idx < 2 * n / 3:
            groups[client_id] = "mid_compute"
        else:
            groups[client_id] = "low_compute"
    return groups


if __name__ == "__main__":
    main()
