from __future__ import annotations

import copy

import numpy as np
import torch

import dynfed.fmnist_lenet5_dynamic as dynamic
from dynfed.joint_calibration import JointCalibrationCaptureSession


def _nested(value: float):
    return {
        "end": {"w": torch.tensor([value], dtype=torch.float32)},
        "edge": {},
    }


def test_capture_session_builds_client_and_mode_fallback_cells(tmp_path):
    session = JointCalibrationCaptureSession(
        metadata={"cross_client_rng": "conditionally_independent"},
        include_mode_fallback=True,
    )
    clean = _nested(1.0)
    session.add_pair(
        state_key="round:0",
        mode="LIIC",
        client_id=3,
        clean_update=clean,
        private_update=_nested(1.2),
    )
    session.add_pair(
        state_key="round:0",
        mode="LIIC",
        client_id=3,
        clean_update=clean,
        private_update=_nested(0.8),
    )
    session.set_ideal_update(_nested(-0.1), state_key="round:0")

    table = session.build_table(min_trials=2)
    exact = table.lookup(mode="LIIC", state_key="round:0", client_id=3)
    fallback = table.lookup(mode="LIIC", state_key="round:0", client_id=99)
    assert exact.sample_count == 2
    assert fallback.sample_count == 2
    assert torch.allclose(exact.bias_mean, torch.tensor([0.0], dtype=torch.float64), atol=1e-7, rtol=0.0)
    assert torch.allclose(table.ideal_update(state_key="round:0"), torch.tensor([-0.1], dtype=torch.float64))

    path = session.save(tmp_path / "capture.pt", min_trials=2)
    assert path.exists()



def test_mode_fallback_does_not_turn_between_client_bias_into_dp_variance():
    session = JointCalibrationCaptureSession(include_mode_fallback=True)
    for private in (1.1, 0.9):
        session.add_pair(
            state_key="round:0", mode="LIIC", client_id=1,
            clean_update=_nested(1.0), private_update=_nested(private),
        )
    # Same zero within-client variance pattern shifted by a large client bias.
    for private in (3.1, 2.9):
        session.add_pair(
            state_key="round:0", mode="LIIC", client_id=2,
            clean_update=_nested(1.0), private_update=_nested(private),
        )
    table = session.build_table(min_trials=2)
    fallback = table.lookup(mode="LIIC", state_key="round:0", client_id=99)
    c1 = table.lookup(mode="LIIC", state_key="round:0", client_id=1)
    c2 = table.lookup(mode="LIIC", state_key="round:0", client_id=2)
    expected = 0.5 * (c1.variance_trace + c2.variance_trace)
    assert abs(fallback.variance_trace - expected) < 1e-9
    assert abs(float(fallback.bias_mean.item()) - 1.0) < 1e-7

def test_clean_calibration_payload_removes_sample_dp_but_preserves_common_randomness():
    private = {
        "privacy_unit": "sample",
        "mechanisms": {"emb": "dp", "grad": "dp", "upd": "he3"},
        "sample_embedding_noise_multiplier": 2.0,
        "sample_label_grad_noise_multiplier": 2.0,
        "sample_optimizer_noise_multiplier": 2.0,
        "dp_update_mode": "upd_only",
        "dp_seed": 123,
        "training_seed": 456,
        "global_end_state": {"w": torch.tensor([1.0])},
    }
    clean = dynamic._clean_calibration_payload(private)
    assert clean["privacy_unit"] == "client"
    assert all(value == "none" for value in clean["mechanisms"].values())
    assert clean["sample_embedding_noise_multiplier"] is None
    assert clean["sample_label_grad_noise_multiplier"] is None
    assert clean["sample_optimizer_noise_multiplier"] is None
    assert clean["dp_update_mode"] == "off"
    assert clean["training_seed"] == private["training_seed"]
    assert clean["global_end_state"] is private["global_end_state"]


def test_paired_payload_reuses_training_seed_and_varies_only_dp_seed(monkeypatch):
    calls = []

    def fake_worker(payload, model_cache=None):
        calls.append(copy.copy(payload))
        value = 0.0 if payload["privacy_unit"] == "client" else float(payload["dp_seed"] % 97)
        return {"finite": True, "state_diff": _nested(value), "client_id": payload["client_id"]}

    monkeypatch.setattr(dynamic, "_client_train_worker", fake_worker)
    private = {
        "client_id": 7,
        "privacy_unit": "sample",
        "mechanisms": {"emb": "dp", "grad": "dp", "upd": "none"},
        "sample_embedding_noise_multiplier": 1.0,
        "sample_label_grad_noise_multiplier": 1.0,
        "sample_optimizer_noise_multiplier": 1.0,
        "dp_update_mode": "upd_only",
        "dp_seed": 1,
        "training_seed": 321,
    }
    clean, privates = dynamic._run_paired_calibration_payload(
        private,
        trials=3,
        base_seed=11,
        round_idx=2,
        client_id=7,
        mode="LIC",
    )
    assert clean["end"]["w"].item() == 0.0
    assert len(privates) == 3
    assert len(calls) == 4
    assert calls[0]["privacy_unit"] == "client"
    assert all(call["training_seed"] == 321 for call in calls)
    private_seeds = [call["dp_seed"] for call in calls[1:]]
    assert len(set(private_seeds)) == 3


def test_reference_update_is_gradient_step_without_mutating_models():
    end = torch.nn.Linear(2, 2, bias=False)
    edge = torch.nn.Linear(2, 2, bias=False)
    before_end = {k: v.detach().clone() for k, v in end.state_dict().items()}
    before_edge = {k: v.detach().clone() for k, v in edge.state_dict().items()}
    x = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32)
    y = np.asarray([0, 1, 0], dtype=np.int64)
    update = dynamic._calibration_reference_update(
        global_end=end,
        global_edge=edge,
        x=x,
        y=y,
        device=torch.device("cpu"),
        input_shape=(2,),
        learning_rate=0.1,
        batch_size=2,
    )
    assert set(update) == {"end", "edge"}
    assert sum(t.numel() for part in update.values() for t in part.values()) == 8
    assert any(float(torch.linalg.vector_norm(t)) > 0.0 for part in update.values() for t in part.values())
    for key, value in before_end.items():
        assert torch.equal(end.state_dict()[key], value)
    for key, value in before_edge.items():
        assert torch.equal(edge.state_dict()[key], value)


def test_capture_round_resamples_heldout_pool_to_runtime_client_mass(monkeypatch):
    candidate = dynamic.Candidate(
        mode="LIIC",
        mechanisms={"emb": "none", "logits": "none", "grad": "none", "emb_grad": "none", "upd": "he3"},
        time=1.0,
        accuracy=0.0,
        risk=0.0,
        epsilon_used=0.0,
        communication_volume=0.0,
        feasible_resource=True,
        feasible_privacy=True,
        feasible_risk=True,
        feasible_time=True,
        sample_optimizer_noise_multiplier=1.0,
    )
    selection = dynamic.SelectionConfig(
        privacy_unit="sample",
        seed=13,
        L_block_cycles=5,
    )
    train_config = dynamic.Lenet5Config(
        joint_calibration_capture_trials=2,
        joint_calibration_capture_max_clients=1,
        joint_calibration_capture_sample_limit=2,
        joint_calibration_capture_scope="candidate_modes",
    )
    session = JointCalibrationCaptureSession(include_mode_fallback=True)
    observed = {}

    def fake_pairs(payload, **kwargs):
        observed["trajectory_len"] = len(payload["x"])
        return _nested(0.0), [_nested(0.1), _nested(-0.1)]

    def fake_reference(**kwargs):
        observed["reference_len"] = len(kwargs["x"])
        return _nested(-0.01)

    monkeypatch.setattr(dynamic, "_run_paired_calibration_payload", fake_pairs)
    monkeypatch.setattr(dynamic, "_calibration_reference_update", fake_reference)
    monkeypatch.setattr(
        dynamic,
        "resolved_privacy_parameters",
        lambda _selection: {"feature_noise_multiplier": 1.0, "update_noise_multiplier": 1.0},
    )

    end = torch.nn.Linear(1, 1, bias=False)
    edge = torch.nn.Identity()
    x = np.arange(9, dtype=np.float32).reshape(9, 1)
    y = np.zeros(9, dtype=np.int64)
    stats = dynamic._capture_joint_calibration_round(
        session=session,
        selected=[(0, candidate, [candidate], 1.0)],
        train_config=train_config,
        selection=selection,
        global_end=end,
        global_edge=edge,
        client_model_states={},
        train_client_indices=[np.arange(7, dtype=np.int64)],
        client_test_indices=[np.asarray([7, 8], dtype=np.int64)],
        x_train=x,
        y_train=y,
        device=torch.device("cpu"),
        model_name="lenet5",
        input_shape=(1,),
        num_classes=2,
        round_idx=0,
    )
    assert stats["captured_pairs"] == 2
    assert observed["trajectory_len"] == 7
    assert observed["reference_len"] == 7
    table = session.build_table(min_trials=2)
    assert table.lookup(mode="LIIC", state_key="round:0", client_id=0).sample_count == 2


def test_capture_wrapper_builds_dedicated_bootstrap_command(tmp_path):
    from argparse import Namespace
    from pathlib import Path
    from experiments.run_joint_calibration_capture import build_capture_command

    root = Path(__file__).resolve().parents[1]
    args = Namespace(
        config=root / "configs" / "paper_v34_cifar10_resnet18_joint_proxy.json",
        output=tmp_path / "joint.pt",
        seed=41,
        policy="full_dynfl",
        rounds=1,
        trials=2,
        period=1,
        max_clients=1,
        sample_limit=16,
        scope="candidate_modes",
        real_he=False,
        dry_run=True,
    )
    command, config = build_capture_command(args)
    joined = " ".join(command)
    assert "--learning-objective legacy_fusion_dp" in joined
    assert "--joint-calibration-capture-trials 2" in joined
    assert "--joint-calibration-capture-period 1" in joined
    assert "--joint-calibration-capture-max-clients 1" in joined
    assert "--joint-calibration-capture-scope candidate_modes" in joined
    assert config["policies"] == ["full_dynfl"]
    assert config["he"]["execution"] == "profiled"
