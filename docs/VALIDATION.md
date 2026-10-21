# Clean paper implementation validation (2026-10-21)

The current implementation contains only the paper three-mode protocol.
All old tests were removed together with their superseded implementations;
the replacement tests assert the paper equations and execution semantics.

- `python -m pytest -q`: 15 tests passed, including an analytic distinction
  between EMA(deviation) and EMA(raw update), sample weights, missing estimates,
  early/late asynchronous admission, sequential Edge feedback, initialization,
  fixed-mode SKIP, conservative lifetime RDP, exact small search and a numerical
  per-example gradient-clipping/Gaussian-noise test.
- `out/paper_clean_eight_methods`: real Fashion-MNIST/LeNet-5 training,
  6 clients / 2 Edges / 3 rounds / seed 42 / all eight comparison policies.
  All completed. `audit.json` independently verifies saved training outputs.
  FL_E-only and latency-only runs each accepted two units generated in an
  earlier round. FL_EC performed 54 DP-SGD optimizer steps, matching
  6 clients * 3 stages * 3 rounds * 1 local step.
- `out/paper_clean_final_7r`: real proposed-method training with the same pilot
  topology. Rounds 1-2 initialized histories, round 3 enabled Pareto selection,
  round 6 searched again, and round 7 accepted a queued asynchronous unit.

These are execution and numerical-conformance checks, not published accuracy
results. Formal 100-round, three-seed comparisons and CIFAR-10 training have not
been completed. Frozen ResNet-18 requires locally cached pretrained weights.

Explicit convention requiring a corresponding note in the paper: if no
synchronous Cloud units exist, the first pending asynchronous arrival triggers
a Cloud aggregation event. Otherwise FL_E never delays the synchronous cutoff.

The paper's optional NSGA-II comparison was removed with the old implementation;
bounded search is currently checked against exhaustive enumeration on small
topologies. Candidate-checkpoint replay ranking and per-Edge history ablation
are not part of this cleaned implementation yet. The active code and README
do not claim that these experiments have been implemented.
