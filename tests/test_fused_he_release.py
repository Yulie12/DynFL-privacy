import pytest

pytestmark = pytest.mark.skip(reason="legacy Method2/mainline_fusion coverage archived by Q75; not part of the formal DynFL mainline")

import numpy as np
import pytest
import torch

from dynfed.fused_he_release import aggregate_fused_release
from dynfed.he_backend import HEOperationMetrics, check_he_backend


def test_real_fused_release_preserves_frozen_backbone_and_public_weights():
    if not check_he_backend("seal").available:
        pytest.skip("SEAL unavailable")
    end = torch.nn.Linear(1, 1, bias=False)
    end.weight.requires_grad_(False)
    frozen = end.weight.detach().clone()
    edge = torch.nn.Linear(1, 1, bias=False)
    initial = edge.weight.detach().clone()
    # Three noisy contributions, two on the first edge; unequal public weights.
    updates = [({"end": {}, "edge": {"weight": torch.tensor([[v]])}},
                1, None, [i]) for i, v in enumerate([0.1, 0.2, -0.3])]
    metrics = HEOperationMetrics("seal")
    audit = aggregate_fused_release(updates, [1, 2, 7], {0: 0, 1: 0, 2: 1},
                                    end, edge, metrics)
    torch.testing.assert_close(end.weight, frozen, rtol=0, atol=0)
    torch.testing.assert_close(edge.weight, initial - 0.16, rtol=0, atol=1e-5)
    assert metrics.encrypted_parameter_values == 2
    assert audit["cloud_pid"] != audit["custodian_pid"]
    assert audit["aggregate_only_decryption_enforced"]
    assert not audit["cloud_secret_key_transmitted"]
    assert audit["max_abs_error"] < 1e-5


def test_formal_method2_command_preserves_tested_scope_and_horizon():
    import json
    from experiments.run_paper_config import DEFAULT_CONFIG, build_command, validate_config
    config = json.loads(DEFAULT_CONFIG.read_text())
    validate_config(config)
    cmd = build_command(config, seed=42, policies=["ours"], rounds=None, max_new_rounds=2)
    assert cmd[cmd.index("--model") + 1] == "resnet18_pretrained_head"
    assert float(cmd[cmd.index("--dp-clip-norm") + 1]) == 0.1
    assert cmd[cmd.index("--rounds") + 1] == "100"
    assert cmd[cmd.index("--local-epochs") + 1] == "3"
    assert cmd[cmd.index("--he-execution") + 1] == "real"
    assert "--equal-optimizer-work-control" not in cmd

    control_cmd = build_command(
        config, seed=42, policies=["ours"], rounds=None,
        equal_optimizer_work_control=True,
    )
    assert "--equal-optimizer-work-control" in control_cmd


def test_multilevel_stages_extend_only_the_clients_own_training():
    from dynfed.fmnist_lenet5_dynamic import _client_epoch_count, Lenet5Config
    from dynfed.selection import SelectionConfig
    from types import SimpleNamespace
    selection = SelectionConfig(mainline_fusion=True)
    training = Lenet5Config(local_epochs=3)
    for mode in ("LIEIIIC", "LIIEIIIC"):
        assert _client_epoch_count(training, selection, SimpleNamespace(mode=mode)) == 9
    assert _client_epoch_count(training, selection, SimpleNamespace(mode="LIIC")) == 3


def test_equal_optimizer_work_control_does_not_change_mainline_default():
    from dynfed.fmnist_lenet5_dynamic import _client_epoch_count, Lenet5Config
    from dynfed.selection import SelectionConfig
    from types import SimpleNamespace
    selection = SelectionConfig(mainline_fusion=True)
    control = Lenet5Config(local_epochs=3, equal_optimizer_work_control=True)
    for mode in ("LIIC", "LIEIIIC", "LIIEIIIC"):
        assert _client_epoch_count(control, selection, SimpleNamespace(mode=mode)) == 3


def test_all_fused_modes_reach_one_release_with_no_repeated_edge_aggregation():
    import random
    from dynfed.selection import SelectionConfig, enumerate_candidates, _profile_flow_result
    from dynfed.flow_executor import execute_mixed_round_flow
    from dynfed.selection import _profile_flow_inputs_by_candidate
    config = SelectionConfig(mainline_fusion=True, trusted_edge_split_execution=True,
                             update_protection_goal="released_model_dp", allow_he=True)
    candidates = enumerate_candidates(config=config, client_id=0, edge_factor=1,
        compute_factor=1, samples=60, remaining_epsilon=8, round_idx=0,
        rng=random.Random(42), policy="ours")
    from dynfed.selection import _local_omega_components
    from dynfed.fmnist_lenet5_dynamic import _candidate_training_mechanisms
    from dynfed.privacy import normalize_mechanism
    for candidate in candidates:
        _local_omega_components(candidate, config)
        for mechanism in _candidate_training_mechanisms(candidate).values():
            normalize_mechanism(mechanism)
    profile = dict(enumerate(candidates))
    samples = {i: 60 for i in profile}
    edges = {i: i % 2 for i in profile}
    inputs = _profile_flow_inputs_by_candidate(config, {i: [c] for i, c in profile.items()},
                                               samples, edges, {}, tuple(profile))
    actual = execute_mixed_round_flow(round_idx=0,
        clients=[next(iter(inputs[i].values())) for i in profile], aggregation_fraction=1)
    estimated = _profile_flow_result(config, profile, samples, edges, {}, inputs)
    assert actual.round_duration == pytest.approx(estimated.round_duration)
    assert set(actual.selected_client_ids) == set(profile)
    assert sum(e["event"] == "global_dp_he_release" for e in actual.flow_events) == 1
    edge_events = [e for e in actual.flow_events if e["event"] == "trusted_noisy_edge_aggregate"]
    assert len(edge_events) == 2
    assert all(e["loop_factor"] == 1 for e in edge_events)
    assert all(c.edge_to_cloud_time > 0 for c in candidates)
