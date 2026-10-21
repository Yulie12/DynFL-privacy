# Clean paper implementation validation (2026-10-21)

Current protocol: `global_barrier_100`; see `GLOBAL_BARRIER_100_ZH.md` for
the replacement barrier tests and real-training smoke runs. The records below
describe historical protocols and must not validate the new state machine.

The current implementation contains only the paper three-mode protocol.
All old tests were removed together with their superseded implementations;
the replacement tests assert the paper equations and execution semantics.

- Historical checkpoint `cecf7819`: 15 tests passed, including an analytic distinction
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

The subsequent mathematical audit supersedes the checkpoint's first-arrival
default. The main configuration now uses an explicit exogenous periodic clock;
first arrival is a sensitivity alternative. The clock is an experimental
assumption, not a derived property. Pending Cloud events take priority over
client activation waits. Random selection obeys the matched selection period.
Current verification has 24 passing tests, including vectorized/reference
DP-SGD equivalence, independent smoothing coefficients, missing observations,
periodic empty windows and these scheduling/fairness cases.
FL_E now has independent Edge replicas, published only when Edge feedback is
available. Generation snapshots record the dispatch source and increment base;
full pilot snapshots independently verify the feedback chain. The v4 campaign
was stopped after discovering this source inconsistency and is superseded by v5.

Campaign sources and runtime configuration are recorded per run. The first
multiseed trial (`out/research_trial_fmnist_v2`) completed 24 runs and audits.
`out/research_studies_v3` completed the first EMA/clock sensitivity study and
CIFAR-10 trial. Final experiments use the subsequent frozen corrected source;
preliminary v2/v3 outputs must not be silently pooled with those final runs.

The paper's optional NSGA-II comparison was removed with the old implementation;
bounded search is currently checked against exhaustive enumeration on small
topologies. Candidate-checkpoint replay ranking and per-Edge history ablation
are not part of this cleaned implementation yet. The active code and README
do not claim that these experiments have been implemented.
