from __future__ import annotations

import math

import torch

from dynfed.joint_calibration import (
    JointCalibrationEntry,
    JointUpdateCalibrationTable,
    PairedUpdateMomentAccumulator,
)
from dynfed.selection import (
    Candidate,
    SelectionConfig,
    _LearningReplacementStats,
    _joint_calibration_learning_components,
    _sample_dp_event_counts,
)


def _candidate(mode: str) -> Candidate:
    return Candidate(
        mode=mode,
        mechanisms={},
        time=1.0,
        accuracy=0.0,
        risk=0.0,
        epsilon_used=0.0,
        communication_volume=0.0,
        feasible_resource=True,
        feasible_privacy=True,
        feasible_risk=True,
        feasible_time=True,
    )


def test_paired_accumulator_estimates_bias_and_trace_without_linearizing_phi():
    acc = PairedUpdateMomentAccumulator()
    clean = torch.tensor([0.6, 0.2], dtype=torch.float64)
    for diff in (
        torch.tensor([0.00, -0.03], dtype=torch.float64),
        torch.tensor([0.01, -0.02], dtype=torch.float64),
        torch.tensor([0.02, -0.01], dtype=torch.float64),
    ):
        acc.update(clean, clean + diff)

    entry = acc.finalize()
    assert entry.sample_count == 3
    assert torch.allclose(entry.clean_update_mean, clean)
    assert torch.allclose(
        entry.bias_mean,
        torch.tensor([0.01, -0.02], dtype=torch.float64),
    )
    assert math.isclose(entry.variance_trace, 0.0002, rel_tol=0.0, abs_tol=1e-12)


def test_joint_selector_proxy_matches_hand_calculation(tmp_path):
    table = JointUpdateCalibrationTable(
        metadata={"cross_client_rng": "conditionally_independent"}
    )
    table.add_entry(
        mode="LIC",
        client_id=1,
        entry=JointCalibrationEntry(
            clean_update_mean=torch.tensor([0.6, 0.2], dtype=torch.float64),
            bias_mean=torch.tensor([0.01, -0.02], dtype=torch.float64),
            variance_trace=0.003,
            sample_count=10,
        ),
    )
    table.add_entry(
        mode="LIIC",
        client_id=2,
        entry=JointCalibrationEntry(
            clean_update_mean=torch.tensor([0.2, 0.5], dtype=torch.float64),
            bias_mean=torch.tensor([0.03, 0.01], dtype=torch.float64),
            variance_trace=0.004,
            sample_count=10,
        ),
    )
    table.add_entry(
        mode="LIE",
        client_id=3,
        entry=JointCalibrationEntry(
            clean_update_mean=torch.tensor([-0.2, 0.1], dtype=torch.float64),
            bias_mean=torch.tensor([9.0, 9.0], dtype=torch.float64),  # a_3=0, must not enter b_s
            variance_trace=99.0,
            sample_count=10,
        ),
    )
    table.set_ideal_update(torch.tensor([0.2, 0.2], dtype=torch.float64))
    path = table.save(tmp_path / "joint_calibration.pt")

    config = SelectionConfig(
        privacy_unit="sample",
        learning_objective="joint_calibration",
        joint_calibration_path=str(path),
        joint_calibration_e_alg_policy="table",
    )
    profile = {
        1: _candidate("LIC"),
        2: _candidate("LIIC"),
        3: _candidate("LIE"),
    }
    result = _joint_calibration_learning_components(
        config,
        profile,
        {1: 40.0, 2: 30.0, 3: 30.0},
        admitted_client_ids=(1, 2, 3),
    )
    assert result is not None
    assert math.isclose(result.total, 0.07753877551020402, rel_tol=0.0, abs_tol=1e-12)
    assert math.isclose(result.variance_trace, 0.0017142857142857142, abs_tol=1e-12)
    assert result.source == "joint_calibration_table_e_alg"


def test_round_conditioned_table_uses_nearest_calibrated_round(tmp_path):
    table = JointUpdateCalibrationTable(metadata={"cross_client_rng": "independent"})
    table.add_entry(
        mode="LIIC",
        state_key="round:10",
        entry=JointCalibrationEntry(
            clean_update_mean=torch.tensor([1.0]),
            bias_mean=torch.tensor([0.1]),
            variance_trace=0.2,
            sample_count=3,
        ),
    )
    table.set_ideal_update(torch.tensor([0.0]), state_key="round:10")
    path = table.save(tmp_path / "round_table.pt")
    loaded = JointUpdateCalibrationTable.load(path)
    entry = loaded.lookup(mode="LIIC", state_key="round:12")
    assert torch.allclose(entry.clean_update_mean, torch.tensor([1.0], dtype=torch.float64))
    assert torch.allclose(
        loaded.ideal_update(state_key="round:12"),
        torch.tensor([0.0], dtype=torch.float64),
    )


def test_sample_event_counts_follow_runtime_batches_and_hierarchical_retraining():
    config = SelectionConfig(
        privacy_unit="sample",
        split_batch_size=16,
        privacy_local_epochs=1,
        L_block_cycles=5,
        split_end_optimizer_enabled=False,
    )
    assert _sample_dp_event_counts(config, "LIC", 40) == (3, 3, 0)
    assert _sample_dp_event_counts(config, "LIIC", 30) == (0, 0, 2)
    # E_edge_loops=3 in the current runtime and the training block is actually
    # rerun three times, so the Sample-RDP accountant must charge all blocks.
    assert _sample_dp_event_counts(config, "LIEIIIC", 40) == (9, 9, 0)
    assert _sample_dp_event_counts(config, "LIIEIIIC", 30) == (0, 0, 6)



def _write_replacement_table(tmp_path, *, include_ideal: bool = True):
    table = JointUpdateCalibrationTable(
        metadata={"cross_client_rng": "conditionally_independent"}
    )
    values = {
        (1, "LIC"): ([0.6, 0.2], [0.01, -0.02], 0.003),
        (1, "LIE"): ([0.4, 0.15], [0.02, -0.01], 0.002),
        (2, "LIIC"): ([0.2, 0.5], [0.03, 0.01], 0.004),
        (2, "LIE"): ([0.1, 0.4], [0.00, 0.02], 0.005),
        (3, "LIE"): ([-0.2, 0.1], [0.01, 0.00], 0.006),
        (3, "LIC"): ([-0.1, 0.2], [-0.02, 0.01], 0.007),
    }
    for (cid, mode), (clean, bias, var) in values.items():
        table.add_entry(
            mode=mode,
            client_id=cid,
            entry=JointCalibrationEntry(
                clean_update_mean=torch.tensor(clean, dtype=torch.float64),
                bias_mean=torch.tensor(bias, dtype=torch.float64),
                variance_trace=var,
                sample_count=8,
            ),
        )
    if include_ideal:
        table.set_ideal_update(torch.tensor([0.2, 0.2], dtype=torch.float64))
    return table.save(tmp_path / ("replacement_table.pt" if include_ideal else "replacement_zero.pt"))


def test_joint_replacement_stats_matches_full_profile_table_ealg(tmp_path):
    path = _write_replacement_table(tmp_path)
    config = SelectionConfig(
        privacy_unit="sample",
        learning_objective="joint_calibration",
        joint_calibration_path=str(path),
        joint_calibration_e_alg_policy="table",
    )
    profile = {1: _candidate("LIC"), 2: _candidate("LIIC"), 3: _candidate("LIE")}
    masses = {1: 40.0, 2: 30.0, 3: 30.0}
    edges = {1: 0, 2: 0, 3: 1}
    admitted = (1, 2, 3)
    stats = _LearningReplacementStats(config, profile, masses, edges, admitted)

    replacement = _candidate("LIC")
    fast = stats.replacement_cost(3, replacement, admitted)
    trial = dict(profile)
    trial[3] = replacement
    exact = _joint_calibration_learning_components(config, trial, masses, admitted)
    assert exact is not None
    assert fast is not None
    assert math.isclose(fast, exact.total, rel_tol=0.0, abs_tol=1e-12)


def test_joint_replacement_stats_matches_full_profile_zero_ealg(tmp_path):
    path = _write_replacement_table(tmp_path, include_ideal=False)
    config = SelectionConfig(
        privacy_unit="sample",
        learning_objective="joint_calibration",
        joint_calibration_path=str(path),
        joint_calibration_e_alg_policy="zero",
    )
    profile = {1: _candidate("LIC"), 2: _candidate("LIIC"), 3: _candidate("LIE")}
    masses = {1: 40.0, 2: 30.0, 3: 30.0}
    edges = {1: 0, 2: 0, 3: 1}
    admitted = (1, 2, 3)
    stats = _LearningReplacementStats(config, profile, masses, edges, admitted)

    replacement = _candidate("LIE")
    fast = stats.replacement_cost(1, replacement, admitted)
    trial = dict(profile)
    trial[1] = replacement
    exact = _joint_calibration_learning_components(config, trial, masses, admitted)
    assert exact is not None
    assert fast is not None
    assert math.isclose(fast, exact.total, rel_tol=0.0, abs_tol=1e-12)
