# DynFedPrivacy NewTeX

This is the cleaned maintenance copy of the DynFedPrivacy project.

It keeps only the code and TeX sources needed for the current NewTeX logic:

- `dynfed/`: core training, mode selection, privacy, and flow execution code.
- `experiments/`: runnable experiment scripts and the local web monitor.
- `tex/paper/`: current paper source copied from `tex/newtex/8月交`.

Large outputs, datasets, caches, IDE files, previous paper packages, and old
experiment artifacts are intentionally excluded.

## Main Entry Points

Start the local web monitor:

```powershell
python experiments\monitor_lenet5_training.py --port 8765
```

Run the current training entry directly:

```powershell
python experiments\run_fmnist_lenet5.py --rounds 50 --policies ours fixed_dp no_protection random
```

The main code path is:

```text
experiments/monitor_lenet5_training.py
  -> experiments/run_fmnist_lenet5.py
  -> dynfed/fmnist_lenet5_dynamic.py
  -> dynfed/selection.py
  -> dynfed/flow_executor.py
```

## NewTeX Logic

The current implementation is aligned with the new TeX logic around:

- global Pareto profile selection,
- system latency `T_sys`,
- system convergence proxy `Omega_sys`,
- cloud fusion ratio penalty,
- strategy update period,
- switching cost proxy,
- mixed edge/cloud flow execution.

New runs created by `run_fmnist_lenet5.py` include `newtex202608` in the run
directory name so they can be distinguished from historical outputs.

## Data

Datasets are not committed. Put data under a local path and pass `--data-root`
when needed, or use the existing local defaults on the machine where the
experiments were developed.

## TeX

The current paper source is:

```text
tex/paper/main.tex
```

Compile it in TeXstudio from `tex/paper/`.

