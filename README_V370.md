# V370 candidate admission — opt-in selector patch

This is a selector-only patch against the `dynfed/selection.py` included in the user's 50-round audit bundle. It is **not** a complete AutoDL runner integration and has not been tested against the full project.

## What it does

- Opt-in static client–connected-edge pair trust via `SelectionConfig.strict_pair_admission` and `trusted_client_edge_pairs`.
- FL-first based on candidate resource, memory, privacy, and predicted deadline feasibility.
- LIE: trusted split without fast-client requirement. LIIE: requires predeclared fast-client deadline, independently of LIE.
- Rejects split modes for untrusted client-edge pairs under the simplified trust policy.
- Adds `mode_audit` fields for trust, deadline, FL-first outcome and generation exclusions.
- Preserves existing DP protection checks and the `trusted_lie_joint_sample_dp` calibration incompatibility guard.

## Critical integration requirements

1. Apply patch to the exact V369D `dynfed/selection.py` revision (check `git apply --check` first).
2. Wire the experiment config/CLI to load the explicit trust manifest using `enable_static_pair_admission(config, path)` **before** selection. Without this, `strict_pair_admission` stays false and the old behavior remains.
3. Supply the actual `client.edge_id` from the fixed topology. The patched selector's internal call does this; inspect other callers of `enumerate_candidates`.
4. Make `mode_audit` metadata persistent in desired output CSV/JSON if needed. It currently appears in the in-memory audit dict and does not automatically create new CSV columns.
5. Validate any trust manifest against the actual topology; the example manifest is illustrative only and should not be used as an experimental truth.
6. Run full tests and 1-round smoke in AutoDL before the 5-round trial. Do not disable DP gates.

## Apply

```bash
cd /root/autodl-tmp/DynFL-privacy
git apply --check v370_selection_admission.patch
git apply v370_selection_admission.patch
python -m compileall -q dynfed/selection.py
```

The patch has been tested with 14 isolated selector tests using stubs for missing runtime modules; it does **not** establish a 50-round regression or end-to-end privacy protection.

## Caveat

The `strict_pair_admission` path is intentionally opt-in. No assumption is made that `he-execution profiled` provides encryption. The trust policy cannot substitute for Sample-DP calibration and release protection.
