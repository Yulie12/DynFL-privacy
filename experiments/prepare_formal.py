"""Register an immutable formal matrix without launching any training."""
import argparse
from pathlib import Path
import sys
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from dynfed.paper_config import PaperConfig, effective_config
from dynfed.paper_storage import atomic_json, digest, file_digest, source_identity


def dataset_files(config):
    root=ROOT/config.data_root
    if config.dataset=="fmnist":
        paths=[root/name for name in ("train-images-idx3-ubyte","train-labels-idx1-ubyte",
                                     "t10k-images-idx3-ubyte","t10k-labels-idx1-ubyte")]
        paths=[p if p.exists() else p.with_name(p.name+".gz") for p in paths]
    else:
        paths=[root/"cifar-10-batches-py"/name for name in
               ("data_batch_1","data_batch_2","data_batch_3","data_batch_4","data_batch_5","test_batch")]
        paths.append(Path(torch.hub.get_dir())/"checkpoints/resnet18-f37072fd.pth")
    return {str(path.resolve()):file_digest(path) for path in paths}


def register_matrix(output):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    source=source_identity()
    matrix={}
    units=[]
    for dataset in ("cifar10","fmnist"):
        config=PaperConfig.load(ROOT/f"configs/paper_{dataset}.json")
        if (config.clients,config.edges,config.rounds,config.seeds)!=(100,10,100,(42,43,44)):
            raise ValueError("formal settings differ from user-confirmed design")
        name="formal_"+dataset
        matrix[name]={"kind":"formal_main","config":config.to_dict()}
        case=output/name
        case.mkdir()
        inputs=dataset_files(config)
        entries=[]
        for seed in config.seeds:
            for policy in config.policies:
                effective=effective_config(config,policy,seed)
                entry={"dataset":dataset,"model":effective.model,"policy":policy,"seed":seed,
                       "rounds":effective.rounds,"config":effective.to_dict(),"config_hash":digest(effective.to_dict()),
                       "git_revision":source["git_revision"],"source_hash":source["source_hash"],
                       "dataset_file_hashes":inputs,"output":str(case/f"seed_{seed}"/policy),
                       "status":"registered_not_started"}
                entries.append(entry)
                units.append(entry)
        atomic_json(case/"config.json",config.to_dict())
        atomic_json(case/"source_manifest.json",source)
        atomic_json(case/"preregistration.json",{"schema":"paper.shared_init.registration.v1","units":entries})
    atomic_json(output/"manifest.json",matrix)
    atomic_json(output/"registered_units.json",{"units":units,"unit_count":len(units),"source":source})
    atomic_json(output/"registration_hashes.json",{str(path.relative_to(output)):file_digest(path)
                for path in sorted(output.rglob("*.json"))})
    atomic_json(output/"campaign_status.json",{"status":"registered_not_started","registered_units":len(units),
                                               "completed":[],"pid":None})
    return units


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    entries=register_matrix(args.output)
    print(f"Registered {len(entries)} units; no training started: {args.output}")
