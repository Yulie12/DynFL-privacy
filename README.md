# Dynamic Cloud-Edge-End Collaboration Reconfiguration

This repository contains the simulator, real model training path, privacy
mechanisms, CKKS aggregation, and paper artifacts for the collaboration
reconfiguration framework.

It keeps only the code and TeX sources needed for the current NewTeX logic:

- `dynfed/`: core training, mode selection, privacy, and flow execution code.
- `experiments/`: runnable experiment scripts and the local web monitor.
- `tex/paper/`: current paper source copied from `tex/newtex/8月交`.

Large outputs, datasets, caches, IDE files, previous paper packages, and old
experiment artifacts are intentionally excluded.

See [`PAPER_CONFORMANCE.md`](PAPER_CONFORMANCE.md) for the maintained mapping
between paper equations, executable logic, and explicitly declared approximations.

## Main Entry Points

Start the local web monitor:

```powershell
python experiments\monitor_lenet5_training.py --port 8765
```

Inspect the exact main experiment commands without starting training:

```powershell
python experiments\run_paper_config.py --dry-run
```

Run one two round smoke test before the full experiment:

```powershell
python experiments\run_paper_config.py --seeds 42 --policies ours --rounds 2
```

Run the complete main experiment:

```powershell
python experiments\run_paper_config.py
```

The versioned configuration is
`configs/paper_v25_cifar10_resnet18.json`. It fixes CIFAR 10, pretrained
ResNet 18, 100 clients, 10 edge nodes, 100 rounds, seeds 40 through 44,
a client level update privacy target of 8.0, trusted end to edge split
execution, full update CKKS cost profiling, and the eight compared policies. The 100 round utility runs use
the measured CKKS profile and numerically equivalent aggregation so that the
same cryptographic benchmark is not repeated in every round. The configured seeds are 40, 41,
42, 43, and 44. The runner accepts `--seeds`,
`--policies`, and `--rounds` overrides so long experiments can be split into
separate processes without changing the recorded base configuration.

Resume one interrupted seed into a new output directory while retaining its
model, privacy ledger, random state, and completed round history:

```powershell
python experiments\run_paper_config.py --seeds 42 --policies ours --resume-from-run out\paper_v25_cifar10_resnet18\EXISTING_RUN_DIRECTORY
```

`--resume-from-run` deliberately accepts exactly one seed at a time.

The default executor is `serial`, which is the paper reproduction path. To
approximate multiple CPU workers running client local training concurrently,
use the optional CPU-only worker pool:

```powershell
python experiments\run_fmnist_lenet5.py --rounds 100 --executor process_pool --executor-workers 4 --device cpu --policies ours individual_optimal no_protection random
```

Run with real CKKS encrypted aggregation for HE-selected updates:

```powershell
python experiments\run_paper_config.py --seeds 42 --policies ours --max-new-rounds 1 --he-execution real --output-root out\paper_v25_cifar10_resnet18_real_he_validation
```

This validation encrypts the complete selected updates. It is separate from
the profiled 100 round utility experiment. Revision v22 checkpoints cannot be
resumed by revision v25.

To use the Fed3Scale SEAL binding instead of TenSEAL, build and install the
local PySEAL package from a Visual Studio x64 developer prompt:

```powershell
cd E:\YTT\GROUP\Fed3Scale_luqiwang\Fed3Scale\SEAL-python
cmake -S SEAL -B SEAL\build -G Ninja -DSEAL_USE_MSGSL=OFF -DSEAL_USE_ZLIB=OFF -DSEAL_USE_ZSTD=OFF
cmake --build SEAL\build
python -m pip install wheel
python -m pip install . --no-build-isolation
python E:\YTT\GROUP\DynFL-privacy\experiments\run_fmnist_lenet5.py --rounds 100 --he-backend seal --require-real-he --he-aggregation-size 0 --policies ours
```

The `seal` backend follows the Fed3Scale style PySEAL and CKKS flow with
`EncryptionParameters`, `SEALContext`, `KeyGenerator`, `CKKSEncoder`,
`Encryptor`, `Evaluator`, and `Decryptor`. It encrypts the update payload at the
selected aggregation link (including edge-preaggregated payloads), aggregates ciphertexts, and decrypts only the aggregate
before applying FedAvg. Partial update encryption is rejected. A value of zero
for `--he-aggregation-size` means that every parameter value is encrypted. At
startup, the backend validates encrypted vector
addition against its plaintext result. The decoder also handles the zero-stride
NumPy metadata produced by the Fed3Scale wrapper when built with pybind11 3.x.
The `tenseal` backend uses the same CKKS encrypted
aggregation semantics and is easier to install in the current Python
environment. Without `--he-backend seal` or `--he-backend tenseal`, HE
candidates are disabled in the real-training selector.

Before a full ResNet 18 run, benchmark the complete encrypted aggregation path
on the target machine. Start with a bounded validation case:

```powershell
python experiments\benchmark_seal_parallelism.py --workers 4 --aggregate-parameter-count 1000000 --aggregate-updates 4 --output out\seal_validation_benchmark.json
```

If it succeeds, test one actual ResNet 18 aggregation size before expanding the
worker and participant sweep:

```powershell
python experiments\benchmark_seal_parallelism.py --workers 4 --aggregate-parameter-count 11181642 --aggregate-updates 10 --output out\seal_resnet18_benchmark.json
```

The full aggregation rows include process startup, file-backed update-store population,
encryption, ciphertext addition, decryption, and result application. If
`psutil` is installed, they also include peak RSS across the parent and worker
processes. `he_wall_time_sec` is elapsed time observed by the caller, whereas
`he_worker_cpu_time_sec` is the sum of encryption, addition, and decryption work
reported by all workers. For SEAL, `he_ciphertext_bytes` retains the
`save_size()` upper-bound semantics; the benchmark separately writes one
representative ciphertext and reports its actual serialized file size.
Parallel SEAL aggregation stores its temporary update matrix under
`out/.he_tmp` instead of Windows page-file-backed shared memory. Set
`DYNFL_HE_TMPDIR` to a directory on another drive when more temporary disk
space is available there.

The main code path is:

```text
experiments/monitor_lenet5_training.py
  -> experiments/run_fmnist_lenet5.py
  -> dynfed/fmnist_lenet5_dynamic.py
  -> dynfed/selection.py
  -> dynfed/flow_executor.py
```

## NewTeX Logic

The current implementation is aligned with the paper logic around:

- global collaboration reconfiguration,
- a unified seven mode collaboration space,
- self contained event flow evaluation for every candidate profile,
- system latency `T_sys`,
- estimated system convergence error cost `Omega_sys`,
- cloud fusion ratio penalty,
- strategy update period,
- switching cost estimate,
- mixed edge and cloud flow execution,
- trusted end to edge split execution and client replacement update DP,
- complete CKKS validation and profiled CKKS utility execution, and
- accounted system time with measured decision cost.

Only outputs with `training.execution_revision=paper_flow_v25_trusted_edge_domain` are
accepted by the current aggregation scripts. Historical checkpoints and
results are not compatible with the revised DP, HE, timing, and Pareto logic.

Validate the bounded Pareto search against exact enumeration on small cases:

```powershell
python experiments\validate_pareto_search.py --output-dir out\pareto_validation_v25
```

Aggregate completed main runs and produce confidence intervals and paired
comparisons:

```powershell
python experiments\aggregate_multiseed_results.py --root out\paper_v25_cifar10_resnet18 --seeds 40 41 42 43 44 --output-dir out\paper_v25_cifar10_resnet18_aggregate --paper-figure tex\paper\figures\cifar10_resnet18_100r_wall_time_accuracy_v25.png --paper-reconfiguration-figure tex\paper\figures\cifar10_reconfiguration_trace_v25.png
```

Calibrate the estimated convergence error cost against the realized post
update loss and utility changes:

```powershell
python experiments\calibrate_convergence_cost.py --root out\paper_v25_cifar10_resnet18 --seeds 40 41 42 43 44 --output-dir out\paper_v25_convergence_calibration
```

Run and aggregate the strategy period, privacy budget, client scale, and
ablation studies. The first three studies use seeds 40, 42, and 44. The key
ablation uses seeds 40 through 44:

```powershell
python experiments\run_controlled_sweeps.py
python experiments\aggregate_controlled_results.py
```

Run and aggregate the Fashion MNIST and LeNet 5 setting:

```powershell
python experiments\run_paper_config.py --config configs\paper_v25_fmnist_lenet5.json
python experiments\aggregate_multiseed_results.py --root out\paper_v25_fmnist_lenet5 --seeds 40 41 42 43 44 --dataset fmnist --model lenet5 --output-dir out\paper_v25_fmnist_lenet5_aggregate --paper-figure tex\paper\figures\fmnist_lenet5_100r_wall_time_accuracy_v25.png
```

## Data

Datasets are not committed. Put data under a local path and pass `--data-root`
when needed, or use the existing local defaults on the machine where the
experiments were developed.

Each v25 run writes the exact selected subset partition, generated client and
edge profiles, model parameter split, runtime package versions, raw decisions,
link events, and round metrics beside its `config.json` file.

## TeX

The current paper source is:

```text
tex/paper/main.tex
```

Compile it in TeXstudio from `tex/paper/`.
