# DynFL-privacy: Fashion-MNIST / LeNet-5 runtime probe

These are engineering six-round timing trials, NOT completed 101-round paper results.

## Setup
Place both `.ps1` files in the project root: `E:\YTT\GROUP\DynFL-privacy`.
The scripts use your existing `D:\soft\Python310\python.exe` and the V370 code path.

If Fashion-MNIST IDX files are missing, download via:

```powershell
& "D:\soft\Python310\python.exe" -c "from torchvision.datasets import FashionMNIST; FashionMNIST(root=r'experiments/data/fmnist', train=True, download=True); FashionMNIST(root=r'experiments/data/fmnist', train=False, download=True)"
```

## Run

```powershell
cd E:\YTT\GROUP\DynFL-privacy
Set-ExecutionPolicy -Scope Process Bypass -Force
.\run_fmnist_lenet5_6round_probe.ps1
.\run_cifar10_resnet18_6round_compare.ps1
```

The first script uses `--dataset fmnist --model lenet5`, the second `--dataset cifar10 --model resnet18_pretrained_head`. They keep the same clients, data subset counts, local epochs, network profiles, sample-DP parameters, and P=5. Both use `--learning-objective legacy_fusion_dp`: this is a *speed probe* because the available `joint_calibration` table is specific to CIFAR-10/ResNet-18 and cannot be carried over to LeNet-5.

`--rounds 101 --max-new-rounds 6` preserves the accounting horizon and pauses after six training rounds. The printed *wall time* is the metric to compare for code runtime; the modeled logical time is a separate paper metric. The scripts temporarily disable the optional frozen ResNet head fast path in both runs to keep code conditions aligned; it is not applicable to LeNet-5.

CPU/GPU speed can also depend on selected modes, trainable parameters, and cold dataset/model loading. One speed probe does not establish a steady state performance gain; repeat or compare per-round wall times as needed.
