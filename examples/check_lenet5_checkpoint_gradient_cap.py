"""Compare analytic LeNet5 gradient caps at a TRUSTED final checkpoint.

Only bounds the saved POST-round model; never backfills earlier decisions.
Torch checkpoints may use pickle: only load your own trusted files.
"""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:sys.path.insert(0,str(ROOT))
from dynfed.online_j_liic_envelope import (
    lenet5_layer_norm_upper_from_states,
    fmnist_lenet5_gradient_norm_cap_by_layers,
    fmnist_lenet5_gradient_norm_cap,
)


def checkpoint_gradient_caps(ckpt):
    if not isinstance(ckpt,dict) or type(ckpt.get('next_round')) is not int or ckpt['next_round']<0:
        raise ValueError('invalid DynFL checkpoint/next_round')
    if 'global_end_state' not in ckpt or 'global_edge_state' not in ckpt:
        raise ValueError('missing LeNet5 global model checkpoint states')
    norms=lenet5_layer_norm_upper_from_states(ckpt['global_end_state'],ckpt['global_edge_state'])
    R=math.sqrt(math.fsum(v*v for entry in norms.values() for v in entry.values()))
    old=fmnist_lenet5_gradient_norm_cap(model_parameter_norm_upper=R)
    new=fmnist_lenet5_gradient_norm_cap_by_layers(layer_norm_uppers=norms)
    if new>old*(1+1e-10):
        raise ValueError('layerwise gradient bound exceeds whole-model bound')
    return {'usable_from_predecision_round':ckpt['next_round'],
            'whole_model_gradient_cap':old,'layerwise_gradient_cap':new,
            'improvement_factor':old/new if new>0 else math.inf,
            'J_certified':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint',required=True,type=Path)
    args=parser.parse_args()
    import torch
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    row=checkpoint_gradient_caps(ckpt)
    print(f'Global model snapshot after round {row["usable_from_predecision_round"]}; '
          f'usable only from the NEXT predecision round {row["usable_from_predecision_round"]} (0-indexed).')
    print(f'Old whole-model gradient cap: {row["whole_model_gradient_cap"]:.8g}')
    print(f'New layerwise gradient cap:   {row["layerwise_gradient_cap"]:.8g}')
    print(f'Cap tightening factor:        {row["improvement_factor"]:.3f}x')
    print('NOT certified J; does NOT apply retroactively to previous rounds.')

if __name__=='__main__':main()
