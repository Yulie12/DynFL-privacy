from __future__ import annotations
import argparse, json, math, sys
from pathlib import Path
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.streaming_secagg import streaming_secure_aggregate_exact_target


def main():
    p=argparse.ArgumentParser()
    p.add_argument("--clients",type=int,default=100)
    p.add_argument("--dimension",type=int,default=11_181_642)
    p.add_argument("--chunk-size",type=int,default=262_144)
    p.add_argument("--clip-norm",type=float,default=0.25)
    p.add_argument("--noise-multiplier",type=float,default=0.691120031158969)
    p.add_argument("--seed",type=int,default=42)
    args=p.parse_args()
    # Keep the validator memory-bounded: each client state is represented by one
    # tensor only; use a smaller dimension manually for quick protocol checks.
    states=[]
    g=torch.Generator().manual_seed(args.seed)
    for _ in range(args.clients):
        states.append({"end":{"update":torch.randn(args.dimension,generator=g,dtype=torch.float32)}, "edge":{}})
    aggregate,audit=streaming_secure_aggregate_exact_target(
        states,[1.0]*args.clients,clip_norm=args.clip_norm,
        noise_multiplier=args.noise_multiplier,round_seed=args.seed,
        chunk_size=args.chunk_size,
    )
    result=audit.__dict__.copy()
    result.update({
        "aggregate_norm": float(torch.linalg.vector_norm(aggregate["end"]["update"]).item()),
        "memory_scaling_claim": "pairwise_masks_generated_per_chunk_not_materialized_full_dimension",
        "release_semantics": "edge_receives_only_fixed_cohort_noisy_aggregate",
        "client_edge_collusion_resistance": False,
        "dropout_recovery": False,
        "protocol_status": "streaming_secagg_exact_target_execution_primitive_noncolluding_fixed_cohort",
    })
    print(json.dumps(result,indent=2,sort_keys=True))


if __name__=="__main__":
    main()
