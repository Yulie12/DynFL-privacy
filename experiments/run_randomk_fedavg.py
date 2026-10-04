"""FedAvg single trusted-aggregate release: Random-k / layerwise / head-projection.

NOT full DynFL; NOT packet protection. Raw updates reach trusted curator.
Private CSV includes uncloaked data-dependent norms. DO NOT publish it.
--diagnostic-disable-dp suppresses Gaussian noise but KEEPS clipping; such
runs do not provide a DP guarantee and must NOT be compared at same epsilon.
"""
from __future__ import annotations
import argparse
import csv
import sys
import secrets
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import torch
from dynfed.fmnist_lenet5_dynamic import load_image_dataset_arrays
from dynfed.privacy import calibrate_gaussian_noise, epsilon_from_rdp, gaussian_rdp
from dynfed.randomk_update import build_public_mask, release_projected_aggregate
from dynfed.split_learning import build_full_model, split_local_train_lenet5


def synthetic_images(n: int, seed: int):
    rng = np.random.default_rng(seed)
    y = rng.integers(0,10,size=n).astype(np.int64)
    x = rng.normal(0,.15,size=(n,1,28,28)).astype(np.float32)
    for i, label in enumerate(y):
        r,c = 2+5*(int(label)//5), 2+5*(int(label)%5)
        x[i,0,r:r+5,c:c+5] += 1.5
    return x,y


def snapshot_global_state(model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """Clone the global model state without moving it away from its device.

    Local training computes parameter differences on the model's device, so a
    CPU snapshot cannot be subtracted from CUDA parameters.
    """
    return {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("synthetic","cifar10","fmnist"), default="synthetic")
    p.add_argument("--data-root", type=Path, default=None)
    p.add_argument("--model", default=None)
    p.add_argument("--train-limit", type=int, default=200)
    p.add_argument("--test-limit", type=int, default=50)
    p.add_argument("--clients", type=int, default=4)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--local-epochs", type=int, default=1)
    p.add_argument("--local-steps", type=int, default=1)
    p.add_argument("--lr", type=float, default=.01)
    p.add_argument("--fractions", nargs="+", type=float, default=[1,.1,.01])
    p.add_argument("--mask-strategy", choices=("randomk","layerwise_randomk","classifier_only"),default="randomk")
    p.add_argument("--epsilon", type=float, default=144)
    p.add_argument("--delta", type=float, default=1e-5)
    p.add_argument("--clip-norm", type=float, default=.25)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", choices=("cpu","cuda"),default="cpu")
    p.add_argument("--diagnostic-disable-dp",action="store_true",
                   help="NO DP guarantee! Keep clipping, omit Gaussian noise.")
    p.add_argument("--output",type=Path,default=Path("out/randomk_stage2/metrics_private.csv"))
    args = p.parse_args()
    if args.rounds < 1 or args.clients < 2 or args.train_limit < args.clients or args.test_limit < 1:
        p.error("rounds>=1 clients>=2 train-limit>=clients test-limit>=1 required")
    if len(set(args.fractions)) != len(args.fractions):
        p.error("fractions must be unique")
    if args.device == "cuda" and not torch.cuda.is_available():
        p.error("CUDA device requested, but CUDA is unavailable")
    device = torch.device(args.device)
    if args.dataset == "synthetic":
        xtrain, ytrain = synthetic_images(args.train_limit,args.seed)
        xtest, ytest = synthetic_images(args.test_limit,args.seed+999)
        shape, classes = (1,28,28),10
    else:
        root = args.data_root or ROOT/"experiments"/"data"/args.dataset
        xtrain,ytrain,xtest,ytest,shape,_,classes = load_image_dataset_arrays(
            args.dataset,root,args.train_limit,args.test_limit,args.seed)
    model_name = args.model or ("lenet5" if args.dataset!="cifar10" else "resnet18_pretrained")
    if args.mask_strategy == "classifier_only" and len(args.fractions) != 1:
        p.error("classifier_only uses one fraction (ignored for selection)")
    sigma = calibrate_gaussian_noise(args.epsilon,args.delta,args.rounds)
    recorded_eps = epsilon_from_rdp(gaussian_rdp(sigma,args.rounds),args.delta)
    fixed_public_roster = np.array_split(np.random.default_rng(args.seed).permutation(len(ytrain)),args.clients)
    public_counts = [len(group) for group in fixed_public_roster]
    args.output.parent.mkdir(parents=True,exist_ok=True)
    print("PRIVATE metrics: do not publicly publish norms, clipping statistics or raw CSV",flush=True)
    if args.diagnostic_disable_dp:
        print("DIAGNOSTIC NO-DP: Gaussian noise disabled, clipping remains. EPSILON NOT CLAIMED.",flush=True)
    with args.output.open("w",newline="",encoding="utf-8") as file:
        writer = None
        for fraction in args.fractions:
            torch.manual_seed(args.seed)
            model = build_full_model(model_name,device,shape[0],shape[1],classes)
            # Cache local model separately: never mutate global model while processing clients.
            local_cache = {"full": build_full_model(model_name,device,shape[0],shape[1],classes)}
            for rnd in range(args.rounds):
                baseline = snapshot_global_state(model)
                reference = {"end": {n:p.detach() for n,p in model.named_parameters() if p.requires_grad},"edge":{}}
                mask = build_public_mask(reference,fraction,args.seed+rnd*1009,args.mask_strategy)

                def per_client_updates():
                    for cid, indices in enumerate(fixed_public_roster):
                        yield split_local_train_lenet5(
                            mode="LIIC",global_end_state=baseline,global_edge_state={},
                            x=xtrain[indices],y=ytrain[indices],epochs=args.local_epochs,
                            lr=args.lr,device=device,model_name=model_name,input_shape=shape,
                            num_classes=classes,mechanisms={"upd":"none"},
                            local_steps=None if args.local_steps<0 else args.local_steps,
                            training_seed=args.seed+rnd*100000+cid,model_cache=local_cache)

                noisy,diag = release_projected_aggregate(
                    per_client_updates(),public_counts,mask,clip_norm=args.clip_norm,
                    noise_multiplier=0.0 if args.diagnostic_disable_dp else sigma,
                    seed=secrets.randbits(128))
                with torch.no_grad():
                    params=dict(model.named_parameters())
                    for part in noisy.values():
                        for name,value in part.items():
                            if name in params:
                                params[name].add_(value.to(params[name].device))
                model.eval()
                with torch.no_grad():
                    correct=seen=0
                    for start in range(0,len(ytest),128):
                        xb=torch.from_numpy(xtest[start:start+128]).float().to(device).view(-1,*shape)
                        yb=torch.from_numpy(ytest[start:start+128]).long().to(device)
                        correct+=int((model(xb).argmax(1)==yb).sum().item())
                        seen+=len(yb)
                row=dict(dataset=args.dataset,model=model_name,strategy=args.mask_strategy,
                         fraction_requested=fraction,round=rnd+1,test_accuracy=correct/max(1,seen),
                         epsilon_target="" if args.diagnostic_disable_dp else args.epsilon,
                         epsilon_recorded="" if args.diagnostic_disable_dp else recorded_eps,
                         delta=args.delta,noise_multiplier_actual=0.0 if args.diagnostic_disable_dp else sigma,
                         clip_norm=args.clip_norm,clients=args.clients,
                         privacy_status="NOT_DP_DIAGNOSTIC" if args.diagnostic_disable_dp else "trusted_curator_aggregate_DP_only",
                         end_to_end_dynfl_dp="not_established",metrics_visibility="PRIVATE",**diag)
                if writer is None:
                    writer=csv.DictWriter(file,fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
                file.flush()  # preserve completed observations if GPU run gets interrupted
                print(f"{args.mask_strategy} q={fraction:g} round={rnd+1}/{args.rounds} "
                      f"acc={row['test_accuracy']:.4f} retention={diag['mean_client_signal_retention']:.4g} "
                      f"signal={diag['signal_norm']:.4g} noise={diag['noise_norm']:.4g} "
                      f"ratio={diag['noise_to_signal']:.4g} clip={diag['clipping_fraction']:.3g} "
                      f"status={row['privacy_status']}",flush=True)
    print(f"Saved PRIVATE metrics: {args.output}",flush=True)

if __name__ == "__main__":
    main()
