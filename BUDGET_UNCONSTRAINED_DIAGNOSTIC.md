# Legacy Scalar-Budget Stability Diagnostic

This document records historical runs from the former uncalibrated scalar
budget implementation. It explains the observed round-80 mechanism transition,
but none of the epsilon values below is a formal DP guarantee. Current runs use
separate feature/update RDP accountants and must not be compared to these
privacy numbers as if the accounting semantics were unchanged.

Date: 2026-08-20

## Controlled change

The run retained the preceding 100-round configuration and changed only the
initial privacy budget from `4.0` to `100.0`:

- 100 clients and 10 edges
- Fashion-MNIST, LeNet-5, 12,000 train and 2,000 test samples
- `extreme_edge_label_skew`
- `learning_rate=0.15`, `L_B=5`, `local_epochs=1`
- `S_P=10`, aggregation fraction `1.0`
- real SEAL backend enabled
- `epsilon_emb=0.03`, `epsilon_upd=0.05`

The maximum candidate cost is approximately `0.95` epsilon per round, so an
initial budget of `100.0` is non-binding over 100 rounds.

## Observed evidence

At displayed round 13, test accuracy reached `0.4305` and test loss was
`1.7506`. At displayed round 20:

- test accuracy: `0.1880`
- test loss: `12.0498`
- minimum remaining epsilon: `96.5`
- budget-exhausted clients: `0`
- feature-DP clients: `47`
- update-DP clients: `52`
- HE clients: `2`
- weighted global update norm: `0.185392`

The monitor stop action removed the incomplete run directory after round 20;
these values were captured from live status before stopping.

## Root cause found

The TeX convergence proxy includes
`B_fclip = kappa_z^2 L_z^2 E[(||z_i^t|| - C)_+^2]`. The selector previously
used the fixed value `omega_feature_clip_excess_sq=0.02`, independent of the
model and client data. Actual profiles were orders of magnitude larger: the
repaired diagnostic observed a mean excess of `22.061` at the first strategy
profile and `2174.405` at the profile corresponding to the old collapse point.

Because Feature-DP clipping distortion was severely underestimated, the old
selector switched 46 clients to `LIC` at displayed round 11. It then mixed
aggressively feature-clipped split-learning updates with update-protected
client updates. Test loss grew from `1.99` to `7.85` by displayed round 20 and
accuracy fell to `0.153`.

The repaired selector profiles the clipping excess on each client's current
embedding distribution and attaches it to every candidate. In the controlled
20-round rerun it kept all clients on Update-DP (`LIIC:89; LIIE:11` at the
second profile), reached `0.445` accuracy, and reduced loss to `1.5639` at
displayed round 20. This isolates the incorrect convergence-objective input as
the collapse trigger in that run.

## Historical deterministic 50-round validation

The repaired code was also run for the then-current 50-round paper setting with
`epsilon_0=4.0`, `S_P=5`, seed 42, extreme edge label skew, and required real
SEAL. The artifacts are in:

`out/root_cause_fix_paper50_seeded/2026-08-20_16-28-18_lenet5_dynamic_newtex202608`

- final/best accuracy: `0.5785` / `0.6150`
- final/maximum loss: `1.0507` / `2.2934`
- mean accuracy, rounds 31--40 / 41--50: `0.5148` / `0.5346`
- mean global-update norm in rounds 41--50: `0.0960`
- minimum remaining epsilon: `1.5`; exhausted clients: `0`
- Feature-DP was never selected
- real HE was used; for example, round 40 used 34 HE and 66 Update-DP clients

Accuracy still oscillates under the extreme non-IID partition, but the final
10-round mean improves and the loss does not diverge. This confirms that the
old sharp collapse was removed rather than merely delayed beyond round 20.

## 100-round horizon mismatch found

After the main horizon was changed from 50 to 100 while retaining
`epsilon_0=4.0`, the deterministic seed-42 run completed with best accuracy
`0.6455` at displayed round 79 but fell to `0.4005` at round 100. Its artifacts
are in:

`out/fmnist_lenet5_xels_100c_10e_rounds100r_s42_sp5_dpbal_ser_cuda_1m_48537f61_a7d8ac90ef/2026-08-20_17-00-46_lenet5_dynamic_newtex202608`

The transition is directly visible in the round logs:

- round 80: remaining epsilon `0.25`, 100 Update-DP clients, loss `0.9191`
- round 86: 28 exhausted clients, 36 HE clients, global-update norm `0.1816`
- round 91: 77 exhausted clients, 80 HE clients, cloud-fusion ratio `0.3571`,
  global-update norm `0.2485`, loss `2.0155`
- round 96: 95 exhausted clients, 95 HE clients, cloud-fusion ratio `0.3581`,
  loss `2.2067`
- round 100: final accuracy `0.4005`, final loss `2.0852`

The original 50-round design allocated `4/50=0.08` total epsilon per training
round. Extending only the horizon halved this normalized allowance. The
100-round main setting therefore uses `epsilon_0=8.0` to preserve the original
budget intensity, while `epsilon_0=4.0` remains a budget-constrained ablation.

## Horizon-scaled 100-round validation

The same seed-42 configuration was rerun from scratch with only
`epsilon_0=8.0`. The artifacts are in:

`out/fmnist_lenet5_xels_100c_10e_rounds100r_s42_sp5_dpbal_ser_cuda_1m_48537f61_a1761c2f49/2026-08-20_17-33-34_lenet5_dynamic_newtex202608`

- final/best accuracy: `0.6230` / `0.6710`
- final loss and rounds 91--100 mean loss: `1.0794` / `1.0172`
- mean accuracy in rounds 71--80 / 81--90 / 91--100:
  `0.6115` / `0.6021` / `0.6157`
- minimum remaining epsilon: `3.25`; maximum exhausted clients: `0`
- maximum global-update norm in rounds 81--100: `0.1480`
- real HE was used without divergence

At displayed round 91 the selector voluntarily used 45 HE and 55 Update-DP
clients with cloud-fusion ratio `1.0`; accuracy was `0.6335` and loss was
`0.9388`. In the `epsilon_0=4.0` run, the same update point had 77 exhausted
clients, 80 HE clients, cloud-fusion ratio `0.3571`, accuracy `0.4425`, and loss
`2.0155`. This controlled comparison confirms that the late collapse was
caused by the horizon-budget mismatch and its forced mechanism/topology shift,
not by HE ciphertext arithmetic itself.

## Conclusion

Privacy-budget exhaustion is not necessary for the observed accuracy collapse.
It can trigger additional DP-to-HE switching in the original run, but the
collapse reproduced while every client retained a large privacy budget. The
identified trigger was the selector's fixed and massively underestimated
feature-clipping term. The learning rate and extreme non-IID partition can
still affect variance and final accuracy, but they do not explain the sharp
round-11 failure observed in the old run.
