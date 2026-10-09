"""Stage 3: CLI must opt into trust and joint Sample-DP together."""
import sys

import pytest

from experiments import run_fmnist_lenet5 as entry
from dynfed.selection import SelectionConfig


def _args(monkeypatch, *flags):
    monkeypatch.setattr(sys, 'argv', ['run_fmnist_lenet5.py', *flags])
    return entry.parse_args()


def test_default_does_not_enable_trust(monkeypatch):
    args = _args(monkeypatch)
    entry._validate_trusted_lie_cli(args)
    assert args.trusted_edge_split_execution is False
    assert args.trusted_lie_joint_sample_dp is False


def test_explicit_trusted_lie_enables_both_flags(monkeypatch):
    args = _args(monkeypatch, '--trusted-edge-split-execution', '--trusted-lie-joint-sample-dp')
    entry._validate_trusted_lie_cli(args)
    config = SelectionConfig(
        privacy_unit=args.privacy_unit,
        learning_objective=args.learning_objective,
        trusted_edge_split_execution=args.trusted_edge_split_execution,
        trusted_lie_joint_sample_dp=args.trusted_lie_joint_sample_dp,
    )
    assert config.trusted_edge_split_execution
    assert config.trusted_lie_joint_sample_dp


@pytest.mark.parametrize('flags, expected', [
    (('--trusted-edge-split-execution',), 'requires explicit'),
    (('--trusted-lie-joint-sample-dp',), 'requires --trusted-edge'),
    (('--trusted-edge-split-execution', '--trusted-lie-joint-sample-dp', '--privacy-unit', 'client'), 'privacy-unit sample'),
    (('--trusted-edge-split-execution', '--trusted-lie-joint-sample-dp', '--learning-objective', 'joint_calibration'), 'new learning-proxy calibration'),
])
def test_rejects_unsafe_cli_combinations(monkeypatch, flags, expected):
    args = _args(monkeypatch, *flags)
    with pytest.raises(ValueError, match=expected):
        entry._validate_trusted_lie_cli(args)
