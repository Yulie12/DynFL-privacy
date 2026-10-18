"""Removed learning objectives must fail closed; no proxy fallback remains."""
import pytest

from dynfed.selection import (
    Candidate,
    SelectionConfig,
    _evaluate_profile,
    _selection_learning_cost,
)


def _candidate(mode, time, mechanism="dp"):
    link = "L_E_upd" if mode == "LIIE" else "L_C_upd"
    return Candidate(
        mode=mode, mechanisms={"upd": mechanism}, link_mechanisms={link: mechanism},
        time=time, accuracy=0.1, risk=0.1, epsilon_used=1.0,
        communication_volume=1.0, feasible_resource=True,
        feasible_privacy=True, feasible_risk=True, feasible_time=True,
        update_noise_multiplier=0.5,
    )


@pytest.mark.parametrize(
    "objective",
    ["legacy_fusion_dp", "joint_calibration", "online_j_oracle_diagnostic"],
)
def test_removed_learning_objectives_are_rejected(objective):
    with pytest.raises(ValueError, match="online_upper_bound"):
        SelectionConfig(learning_objective=objective)


def test_online_objective_requires_a_manifest():
    with pytest.raises(ValueError, match="online_bound_manifest_path"):
        SelectionConfig(learning_objective="online_upper_bound")


def test_mainline_fusion_overlay_is_rejected():
    with pytest.raises(ValueError, match="mainline_fusion"):
        SelectionConfig(mainline_fusion=True)


def test_profile_objective_fails_closed_without_online_certificate():
    config = SelectionConfig(
        omega_update_clip_norm=0.25, omega_update_dimension=10.0,
    )
    profile = {0: _candidate("LIIC", 3.0), 1: _candidate("LIIE", 1.0, "none")}
    masses = {0: 5.0, 1: 5.0}
    edges = {0: 0, 1: 0}
    with pytest.raises(ValueError, match="online_upper_bound"):
        _evaluate_profile(
            config, profile, masses, edges, {},
            flow_objectives=((0, 1), 4.0),
        )
    with pytest.raises(ValueError, match="online_upper_bound"):
        _selection_learning_cost(config, profile, masses, edges, (0, 1))