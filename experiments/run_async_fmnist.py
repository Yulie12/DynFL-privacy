from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.async_training import AsyncConfig, run_async_training
from dynfed.fmnist_lenet5_dynamic import Lenet5Config, _partition_clients_lenet5
from dynfed.fmnist_dynamic_training import load_fmnist_arrays
from dynfed.nodes import build_profiles
from dynfed.selection import SelectionConfig
from dynfed.utils import timestamped_dir


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Async version-controlled FedAvg for FMNIST + LeNet5."
    )
    parser.add_argument("--clients", type=int, default=100)
    parser.add_argument("--edges", type=int, default=10)
    parser.add_argument("--train-limit", type=int, default=12000)
    parser.add_argument("--test-limit", type=int, default=2000)
    parser.add_argument("--local-epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--partition-mode",
        default="client_noniid",
        choices=["iid", "client_noniid", "edge_label_skew", "extreme_edge_label_skew"],
    )
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--data-root", default="E:/YTT/GROUP/DriftRace/data/fmnist/FashionMNIST/raw")
    parser.add_argument("--output-root", default="out/async_fmnist")
    parser.add_argument("--max-events", type=int, default=200)
    parser.add_argument("--B-edge", type=int, default=3)
    parser.add_argument("--B-cloud", type=int, default=5)
    parser.add_argument("--decay", type=float, default=0.85)
    parser.add_argument("--max-gap", type=int, default=3)
    parser.add_argument(
        "--policies",
        nargs="+",
        default=["ours", "best_accuracy", "fixed_dp", "random", "no_protection"],
    )
    parser.add_argument("--eval-interval", type=int, default=10)
    parser.add_argument("--reselect-interval", type=int, default=3)
    parser.add_argument("--dp-noise", type=float, default=0.001)
    parser.add_argument("--initial-epsilon", type=float, default=4.0)
    parser.add_argument("--dp-emb-epsilon", type=float, default=8.0)
    parser.add_argument("--dp-upd-epsilon", type=float, default=8.0)
    parser.add_argument(
        "--viz", action="store_true",
        help="Use visualization-compatible init: 6 ends × 3 edges, per-end budgets & sample ratios"
    )
    parser.add_argument(
        "--per-end-epsilon", type=float, nargs="+", default=None,
        help="Per-end heterogeneous privacy budgets (e.g. 0.8 0.5 0.9 0.4 0.6 0.7)"
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = timestamped_dir(args.output_root, "async_lenet5")

    # Visualization-compatible initialization: 6 ends × 3 edges
    # with per-end heterogeneous budgets and sample ratios
    if args.viz:
        args.clients = 6
        args.edges = 3
        if args.per_end_epsilon is None:
            args.per_end_epsilon = [0.8, 0.5, 0.9, 0.4, 0.6, 0.7]

    extra_kwargs = dict(resource_limit=3.0, time_limit=15.0, edge_cpu_limit=4.0) if args.viz else {}

    selection = SelectionConfig(
        rounds=1,  # not used in async mode, but needed for selection config
        num_clients=args.clients,
        num_edges=args.edges,
        seed=args.seed,
        output_dir=str(output_root),
        dp_noise_multiplier=args.dp_noise,
        initial_epsilon=args.initial_epsilon,
        dp_emb_epsilon=args.dp_emb_epsilon,
        dp_upd_epsilon=args.dp_upd_epsilon,
        omega_learning_rate=args.lr,
        cloud_cpu_limit=20.0,
        **extra_kwargs,
    )
    async_config = AsyncConfig(
        B_edge=args.B_edge,
        B_cloud=args.B_cloud,
        decay_rate=args.decay,
        max_version_gap=args.max_gap,
        local_epochs=args.local_epochs,
        learning_rate=args.lr,
        device=args.device,
        max_events=args.max_events,
        eval_interval_events=args.eval_interval,
        reselect_interval=args.reselect_interval,
        dp_noise_multiplier=args.dp_noise,
        per_end_epsilon=args.per_end_epsilon,
    )

    data_root = args.data_root
    x_train, y_train, x_test, y_test = load_fmnist_arrays(
        Path(data_root), args.train_limit, args.test_limit, args.seed,
    )

    client_indices = _partition_clients_lenet5(
        y_train=y_train,
        num_clients=args.clients,
        num_edges=args.edges,
        iid=False,
        partition_mode=args.partition_mode,
        seed=args.seed,
    )
    clients, edges = build_profiles(
        num_clients=args.clients,
        num_edges=args.edges,
        client_heterogeneity=selection.client_heterogeneity,
        edge_heterogeneity=selection.edge_heterogeneity,
        seed=args.seed,
    )

    if args.viz:
        # Override client edge assignments to match visualization topology:
        # Edge1 → End0, End1; Edge2 → End2, End3; Edge3 → End4, End5
        from dynfed.nodes import ClientProfile as _CP
        viz_edge_map = {0: 0, 1: 0, 2: 1, 3: 1, 4: 2, 5: 2}
        viz_sample_ratios = [1, 2, 1, 1, 1.5, 1.2]
        base_samples = 150
        new_clients = []
        for client in clients:
            client_id = client.client_id
            new_edge = viz_edge_map.get(client_id, client.edge_id)
            target_samples = int(base_samples * viz_sample_ratios[client_id])
            new_clients.append(_CP(
                client_id=client_id,
                edge_id=new_edge,
                samples=target_samples,
                compute_factor=client.compute_factor,
            ))
            # Resize data partition to match target sample count
            idx = client_indices[client_id]
            if len(idx) > target_samples:
                client_indices[client_id] = idx[:target_samples]
            elif len(idx) < target_samples:
                repeats = (target_samples // max(len(idx), 1)) + 1
                extended = np.tile(idx, repeats)[:target_samples]
                client_indices[client_id] = extended
        clients = new_clients

    edge_by_id = {edge.edge_id: edge for edge in edges}

    summaries = []
    for policy in args.policies:
        offset = sum(ord(c) for c in policy)
        rng = random.Random(args.seed + offset)
        np_rng = np.random.default_rng(args.seed + offset + 1543)

        output_dir = output_root / policy
        output_dir.mkdir(parents=True, exist_ok=True)

        summary = run_async_training(
            selection=selection,
            async_config=async_config,
            clients=clients,
            edges=edges,
            edge_by_id=edge_by_id,
            client_indices=client_indices,
            x_train=x_train,
            y_train=y_train,
            x_test=x_test,
            y_test=y_test,
            output_dir=output_dir,
            policy=policy,
            rng=rng,
            np_rng=np_rng,
        )
        summaries.append(summary)

    # Write summary table
    from dynfed.fmnist_lenet5_dynamic import _write_csv, _write_json
    _write_csv(output_root / "summary_table.csv", summaries)
    _write_json(output_root / "config.json", {
        "selection": selection.__dict__,
        "async_config": async_config.__dict__,
        "policies": args.policies,
    })

    print(f"\n[OK] wrote {len(summaries)} async runs to {output_root}")
    for s in summaries:
        print(f"  {s['policy']}: best_acc={s['best_test_accuracy']:.4f}, "
              f"cloud_agg={s.get('cloud_aggregations', 0)}, events={s.get('events', 0)}")


if __name__ == "__main__":
    main()
