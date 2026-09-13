import copy
import json
from pathlib import Path

import pytest

from experiments.run_paper_config import ROOT, build_command, validate_config


def config():
    return json.loads((ROOT / "configs/method2_fixed_he_dp_cifar10.json").read_text())


def test_bridge_preserves_horizon_and_real_custody():
    cfg = config()
    validate_config(cfg)
    command = build_command(cfg, seed=42, policies=cfg["policies"], rounds=None, max_new_rounds=2)
    assert command[command.index("--rounds") + 1] == "2"
    assert command[command.index("--privacy-horizon") + 1] == "100"
    assert command[command.index("--he-custody") + 1] == "trusted_edge"


@pytest.mark.parametrize("policy", ["ours", "random", "fixed_he"])
def test_unwired_policies_fail_closed(policy):
    with pytest.raises(ValueError):
        build_command(config(), seed=42, policies=[policy], rounds=None)


def test_no_profiled_fallback_or_resume():
    cfg = config()
    cfg["he"]["execution"] = "profiled"
    with pytest.raises(ValueError):
        validate_config(cfg)
    with pytest.raises(ValueError):
        build_command(config(), seed=42, policies=["fixed_secure_aggregate"],
                      rounds=None, resume_from_run=Path("old"))


def test_resource_policy_bridge():
    cfg = json.loads((ROOT / "configs/method2_resource_placement_cifar10.json").read_text())
    validate_config(cfg)
    command = build_command(cfg, seed=42, policies=["resource_placement"], rounds=None, max_new_rounds=2)
    assert command[command.index("--execution-layout") + 1] == "resource_dynamic"
    assert "--resource-profile" in command
