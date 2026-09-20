import pytest

pytestmark = pytest.mark.skip(reason="legacy Method2/mainline_fusion coverage archived by Q75; not part of the formal DynFL mainline")

import math
import random
from dataclasses import replace

import pytest
import torch

from dynfed.independent_release import IndependentReleaseAccount
from dynfed.selection import SelectionConfig, enumerate_candidates, _candidate_accuracy_jitter
from dynfed.fmnist_lenet5_dynamic import _training_base_state, Lenet5Config
from dynfed.training import MODE_SPECS


class DummyModel(torch.nn.Module):
    def __init__(self, value: float):
        super().__init__()
        self.w = torch.nn.Parameter(torch.tensor([value], dtype=torch.float32))


def _fusion_config():
    return SelectionConfig(
        rounds=5,
        num_clients=2,
        num_edges=1,
        trusted_edge_split_execution=True,
        allow_he=True,
        update_mechanism_options=("dp", "he3", "dp_he3"),
        update_protection_goal="released_model_dp",
        mainline_fusion=True,
    )


def test_mainline_fusion_enumerates_six_modes_without_privacy_choice():
    config = _fusion_config()
    candidates = enumerate_candidates(
        config=config,
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=60,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(42),
        policy="ours",
    )
    modes = {candidate.mode for candidate in candidates}
    assert modes == {"LIE", "LIIE", "LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC"}
    assert all(candidate.global_release_required for candidate in candidates)
    assert len(candidates) == 6
    assert all(candidate.update_dp_events == 0 for candidate in candidates)
    assert all(candidate.feature_dp_events == 0 for candidate in candidates)


def test_release_account_state_round_trip_preserves_next_round_and_epsilon():
    account = IndependentReleaseAccount([3, 7], 0.1, 8.0, 1e-5, 5)
    first_epsilon = account.reserve(0, [0, 1])
    state = account.state_dict()

    restored = IndependentReleaseAccount([3, 7], 0.1, 8.0, 1e-5, 5)
    restored.load_state_dict(state)

    assert restored.next_round == 1
    assert restored.ledger.current_epsilon() == pytest.approx(first_epsilon)
    assert restored.reserve(1, [0, 1]) > first_epsilon


def test_fusion_training_base_ignores_private_returned_state():
    global_end = DummyModel(1.0)
    global_edge = DummyModel(2.0)
    private_state = {
        "end": {"w": torch.tensor([9.0])},
        "edge": {"w": torch.tensor([8.0])},
    }
    candidate = next(
        c for c in enumerate_candidates(
            config=_fusion_config(),
            client_id=0,
            edge_factor=1.0,
            compute_factor=1.0,
            samples=60,
            remaining_epsilon=8.0,
            round_idx=0,
            rng=random.Random(42),
            policy="ours",
        )
        if c.mode == "LIEIIIC"
    )

    base = _training_base_state(
        client_id=0,
        candidate=candidate,
        client_model_states={0: private_state},
        global_end=global_end,
        global_edge=global_edge,
        device=torch.device("cpu"),
        training_stage=1,
        mainline_fusion=True,
    )

    assert base["end"]["w"].item() == pytest.approx(1.0)
    assert base["edge"]["w"].item() == pytest.approx(2.0)


def test_fusion_config_requires_full_participation_and_real_he():
    from dynfed.fmnist_lenet5_dynamic import _validate_mainline_fusion

    selection = _fusion_config()
    training = Lenet5Config(he_backend="seal", he_execution="real", require_real_he=True)
    _validate_mainline_fusion(selection, training)

    with pytest.raises(ValueError, match="aggregation_fraction"):
        _validate_mainline_fusion(
            SelectionConfig(
                mainline_fusion=True,
                trusted_edge_split_execution=True,
                update_protection_goal="released_model_dp",
                aggregation_fraction=0.5,
            ),
            training,
        )
    with pytest.raises(ValueError, match="real HE"):
        _validate_mainline_fusion(selection, Lenet5Config(he_backend="seal", he_execution="profiled"))


def test_v30_paper_config_enables_fusion_contract():
    import json
    from experiments.run_paper_config import ROOT, build_command, validate_config

    config = json.loads((ROOT / "configs/paper_v30_cifar10_resnet18.json").read_text())
    validate_config(config)
    command = build_command(config, seed=42, policies=["ours"], rounds=2)
    assert "--mainline-fusion" in command
    assert command[command.index("--update-protection-goal") + 1] == "released_model_dp"
    assert "--require-real-he" in command


def test_v30_formal_policy_set_contains_no_privacy_bypass_control():
    import json
    from experiments.run_paper_config import ROOT, validate_config

    for filename in (
        "paper_v30_cifar10_resnet18.json",
        "paper_v30_fmnist_lenet5.json",
    ):
        config = json.loads((ROOT / "configs" / filename).read_text())
        validate_config(config)
        assert "no_protection" not in config["policies"]
        assert "fixed_he" not in config["policies"]
        assert "fixed_dp" not in config["policies"]


def test_fusion_omega_uses_common_aggregate_release_cost():
    from dynfed.selection import _fusion_aggregate_noise_cost

    config = _fusion_config()
    candidates = enumerate_candidates(
        config=config, client_id=0, edge_factor=1.0, compute_factor=1.0,
        samples=60, remaining_epsilon=8.0, round_idx=0,
        rng=random.Random(42), policy="ours",
    )
    profile = {0: next(c for c in candidates if c.mode == "LIIC"),
               1: next(c for c in candidates if c.mode == "LIEIIC")}
    cost = _fusion_aggregate_noise_cost(
        config, profile, {0: 0, 1: 0}, (0, 1), {0: 3.0, 1: 7.0}
    )
    assert cost > 0.0
    assert _fusion_aggregate_noise_cost(
        config, profile, {0: 0, 1: 0}, (), {0: 3.0, 1: 7.0}
    ) == 0.0


def test_fusion_privacy_audit_reports_one_global_release_account():
    from dynfed.privacy import privacy_execution_audit

    audit = privacy_execution_audit([
        {
            "mainline_fusion": 1,
            "num_global_update_clients": 2,
            "uniform_update_dp": True,
            "global_release_count": 1,
            "global_release_epsilon": 1.25,
            "he_execution_status": "real",
        }
    ])

    assert audit["accountant_scope"] == "one_global_release_per_round_fixed_public_roster"
    assert audit["global_release_count"] == 1
    assert audit["global_release_epsilon"] == pytest.approx(1.25)
    assert audit["end_to_end_dp_status"] == "not_established"


def test_fusion_release_audit_reports_fixed_dp_plus_he_for_every_mode():
    from dynfed.fmnist_lenet5_dynamic import _reported_release_mechanism

    candidates = enumerate_candidates(
        config=_fusion_config(),
        client_id=0,
        edge_factor=1.0,
        compute_factor=1.0,
        samples=60,
        remaining_epsilon=8.0,
        round_idx=0,
        rng=random.Random(42),
        policy="ours",
    )
    assert {
        _reported_release_mechanism(candidate, mainline_fusion=True)
        for candidate in candidates
    } == {"dp_he3"}


def test_fusion_config_requires_rdp_auto_accounting():
    from dynfed.fmnist_lenet5_dynamic import _validate_mainline_fusion

    selection = replace(_fusion_config(), dp_accounting_mode="rdp_manual")
    with pytest.raises(ValueError, match="rdp_auto"):
        _validate_mainline_fusion(
            selection,
            Lenet5Config(he_backend="seal", he_execution="real", require_real_he=True),
        )


def test_fusion_config_requires_full_update_he_encryption():
    from dynfed.fmnist_lenet5_dynamic import _validate_mainline_fusion

    with pytest.raises(ValueError, match="he_aggregation_size=0"):
        _validate_mainline_fusion(
            _fusion_config(),
            Lenet5Config(he_backend="seal", 
                he_execution="real",
                require_real_he=True,
                he_aggregation_size=128,
            ),
        )


def test_v30_config_rejects_nonautomatic_release_accounting():
    import copy
    import json
    from experiments.run_paper_config import ROOT, validate_config

    config = json.loads((ROOT / "configs/paper_v30_cifar10_resnet18.json").read_text())
    config = copy.deepcopy(config)
    config["privacy"]["accounting_mode"] = "rdp_manual"
    with pytest.raises(ValueError, match="rdp_auto"):
        validate_config(config)


def test_v30_config_rejects_partial_he_aggregation():
    import copy
    import json
    from experiments.run_paper_config import ROOT, validate_config

    config = json.loads((ROOT / "configs/paper_v30_cifar10_resnet18.json").read_text())
    config = copy.deepcopy(config)
    config["he"]["aggregation_size"] = 128
    with pytest.raises(ValueError, match="aggregation_size=0"):
        validate_config(config)


def test_fusion_reporting_scope_describes_global_release_not_per_link_policy():
    from dynfed.fmnist_lenet5_dynamic import _privacy_reporting_scope

    scope = _privacy_reporting_scope(True)
    assert scope == {
        "dp_accountant_scope": "one_global_release_per_round_fixed_public_roster",
        "privacy_policy_scope": "dynamic_mode_fixed_global_release",
        "protected_object": "global_aggregate_release",
        "privacy_mechanism_scope": "fixed_client_dp_plus_he_cloud_confidentiality",
        "privacy_guarantee": "fixed_global_release_dp_plus_he",
    }


def test_legacy_reporting_scope_remains_per_link():
    from dynfed.fmnist_lenet5_dynamic import _privacy_reporting_scope

    scope = _privacy_reporting_scope(False)
    assert scope["dp_accountant_scope"] == "recorded_dp_events_only"
    assert scope["privacy_policy_scope"] == "per_link"
    assert scope["protected_object"] == "cross_domain_model_update"
    assert scope["privacy_mechanism_scope"] == "local_packet_or_secure_aggregate"
    assert scope["privacy_guarantee"] == "legacy_per_link_mechanisms"


def test_global_release_summary_uses_release_account_state():
    from dynfed.fmnist_lenet5_dynamic import _global_release_summary

    account = IndependentReleaseAccount([3, 7], 0.1, 8.0, 1e-5, 5)
    epsilon = account.reserve(0, [0, 1])
    summary = _global_release_summary(account)

    assert summary["global_release_count"] == 1
    assert summary["global_release_epsilon"] == pytest.approx(epsilon)
    assert summary["global_release_target_epsilon"] == pytest.approx(8.0)
    assert summary["global_release_sensitivity"] == pytest.approx(account.sensitivity)
    assert summary["global_release_noise_multiplier"] == pytest.approx(account.multiplier)
    assert summary["global_release_noise_std"] == pytest.approx(account.noise_std)


def test_fusion_privacy_parameters_calibrate_exactly_one_release_per_round():
    from dynfed.privacy import calibrate_gaussian_noise
    from dynfed.selection import resolved_privacy_parameters

    config = _fusion_config()
    params = resolved_privacy_parameters(config)
    assert params["max_update_events_per_round"] == 1
    assert params["update_horizon_events"] == config.rounds
    assert params["update_noise_multiplier"] == pytest.approx(
        calibrate_gaussian_noise(
            float(params["update_budget"]),
            float(params["delta"]),
            config.rounds,
        )
    )


def test_fusion_candidate_utility_does_not_charge_fixed_he_as_mode_penalty():
    config = _fusion_config()
    candidates = enumerate_candidates(
        config=config, client_id=0, edge_factor=1.0, compute_factor=1.0,
        samples=60, remaining_epsilon=8.0, round_idx=0,
        rng=random.Random(42), policy="ours",
    )
    progress = 1.0 / config.rounds
    for candidate in candidates:
        expected = 0.2 + (
            0.83 - MODE_SPECS[candidate.mode].mode_penalty - 0.2
        ) * (1.0 - math.exp(-3.0 * progress))
        expected += _candidate_accuracy_jitter(
            config, 0, 0, candidate.mode, {}
        )
        assert candidate.accuracy == pytest.approx(max(0.0, min(0.95, expected)))


def test_cli_exposes_mainline_fusion_flag(monkeypatch):
    from experiments import run_fmnist_lenet5

    monkeypatch.setattr(
        "sys.argv",
        ["run_fmnist_lenet5.py", "--mainline-fusion"],
    )
    args = run_fmnist_lenet5.parse_args()
    assert args.mainline_fusion is True


def test_runner_fusion_defaults_only_include_result_dp_compatible_policies(monkeypatch):
    import sys
    from experiments.run_fmnist_lenet5 import parse_args, _resolved_policies

    monkeypatch.setattr(sys, "argv", ["run_fmnist_lenet5.py", "--mainline-fusion"])
    args = parse_args()
    policies = _resolved_policies(args)
    assert "ours" in policies
    assert "no_protection" not in policies
    assert "fixed_he" not in policies
    assert "fixed_dp" not in policies
    assert "fixed_fedavg" in policies
    assert "fixed_splitfed" in policies
    assert "fixed_hfl" in policies
    assert "nsga2" in policies


def test_v30_fusion_config_rejects_control_policies_inside_mandatory_release_run():
    import copy
    import json
    from experiments.run_paper_config import ROOT, validate_config

    config = json.loads((ROOT / "configs/paper_v30_cifar10_resnet18.json").read_text())
    config = copy.deepcopy(config)
    config["policies"] = ["ours", "no_protection"]
    with pytest.raises(ValueError, match="incompatible.*mainline fusion"):
        validate_config(config)


def test_fusion_rejects_legacy_dp_vs_he_stability_switching():
    from dynfed.fmnist_lenet5_dynamic import _validate_mainline_fusion

    selection = replace(_fusion_config(), enforce_cloud_dp_stability=True)
    with pytest.raises(ValueError, match="legacy cloud-DP stability"):
        _validate_mainline_fusion(
            selection,
            Lenet5Config(he_backend="seal", he_execution="real", require_real_he=True),
        )


def test_v30_config_disables_legacy_cloud_dp_stability_switching():
    import json
    from experiments.run_paper_config import ROOT

    for filename in ("paper_v30_cifar10_resnet18.json", "paper_v30_fmnist_lenet5.json"):
        config = json.loads((ROOT / "configs" / filename).read_text())
        assert config["system"]["mainline_fusion"] is True
        assert config["privacy"]["enforce_cloud_dp_stability"] is False
