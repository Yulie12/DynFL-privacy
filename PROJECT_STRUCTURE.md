# Project Structure

## Code

- `dynfed/selection.py`: candidate enumeration and NewTeX global Pareto profile selection.
- `dynfed/fmnist_lenet5_dynamic.py`: main real-training loop used by the web monitor.
- `dynfed/flow_executor.py`: mixed-mode round execution, buffering, waiting, edge/cloud aggregation, and return latency.
- `dynfed/training.py`: seven collaborative mode definitions.
- `dynfed/split_learning.py`: split model construction and local training helpers.
- `dynfed/he_backend.py`: optional HE backend detection.

## Experiment Entrypoints

- `experiments/monitor_lenet5_training.py`: local web monitor and experiment launcher.
- `experiments/run_fmnist_lenet5.py`: main CLI training entry.
- `experiments/rebuild_paper_figures.py`: figure rebuild utilities.
- `experiments/analyze_client_mode_groups.py`: mode grouping analysis.

## TeX

- `tex/paper/main.tex`: current main paper file.
- `tex/paper/main_with_changes.tex`: legacy reviewer-marked snapshot; not the normative implementation specification.
- `tex/paper/refs.bib`: bibliography.
- `tex/paper/figures/`: paper figures.
- `tex/paper/figures_en/`: English ablation figures.

## Excluded

- `out/`: generated experiment results.
- `experiments/data/`: local datasets.
- `__pycache__/`, `.idea/`, `.he_deps/`: local caches and environment files.
- historical paper packages and zip archives.
