"""Predecision simulator-J smoke + fail-closed constraints."""
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from dynfed.online_j_bound import certificate_candidate_key
from dynfed.online_j_oracle_diagnostic import (
    evaluate_oracle_diagnostic,
    full_sgd_update_rms_bound,
    make_oracle_snapshot,
)
from dynfed.selection import SelectionConfig


def candidate(mode, mechanisms):
    return SimpleNamespace(
        mode=mode,
        link_mechanisms=mechanisms,
        mechanisms={},
        dp_execution_plan="",
        feasible=True,
        sample_optimizer_noise_multiplier=0.0,
    )


def _snap(cand, path, *, cid=0):
    data = {
        "schema": "dynfl.online_j.oracle_diagnostic.v1",
        "round_index": 0, "available_before_round": 0,
        "formal_online_j_certified": False,
        "used_raw_client_training_data": True,
        "cloud_step": 1.0,
        "ideal_population_update_norm": 0.5,
        "clients": {str(cid): {"samples": 10, "ideal_client_update_norm": 0.5,
            "reference_deviation_upper": 1.0, "candidates": {
            certificate_candidate_key(cand): {
                "mode": cand.mode, "update_rms_bound": 2.0,
                "he_only_no_extra_release_dp": True,
            }
        }}},
    }
    path.write_text(json.dumps(data), encoding="utf-8")


def test_rms_recurrence_matches_sgd_deterministic_extreme():
    got = full_sgd_update_rms_bound(
        parameter_norm=0, dimension=1, batch_sizes=(10, 10),
        learning_rate=0.1, momentum=0, weight_decay=0,
        gradient_clip=3, noise_multiplier=0, sample_dp=False,
    )
    assert got == pytest.approx(.6)
    assert full_sgd_update_rms_bound(
        parameter_norm=0, dimension=10, batch_sizes=(10, 10),
        learning_rate=0.1, momentum=.9, weight_decay=0,
        gradient_clip=3, noise_multiplier=0, sample_dp=False,
    ) > got
    noisy = full_sgd_update_rms_bound(
        parameter_norm=0, dimension=10, batch_sizes=(10, 10),
        learning_rate=0.1, momentum=0, weight_decay=0,
        gradient_clip=3, noise_multiplier=2, sample_dp=True,
    )
    assert noisy > got


def test_oracle_bound_uses_candidate_envelope_not_legacy_fusion(tmp_path):
    cand = candidate("LIIC", {"L_C_upd": "he3"})
    path = tmp_path / "snapshot.json"
    _snap(cand, path)
    result = evaluate_oracle_diagnostic(
        snapshot_path=str(path), round_index=0,
        profile={0: cand}, client_samples={0: 10},
        admitted_client_ids=(0,), reaches_cloud=lambda _: True,
    )
    assert result.upper == pytest.approx(6.25)  # min(direct,coupled), not Fusion 0.36
    assert "NOT_DEPLOYMENT_CERTIFIED" in result.source
    no_admit = evaluate_oracle_diagnostic(
        snapshot_path=str(path), round_index=0,
        profile={0: cand}, client_samples={0: 10},
        admitted_client_ids=(), reaches_cloud=lambda _: True,
    )
    assert no_admit.upper == pytest.approx(0.25)


def test_missing_candidate_stale_round_and_tampered_attestation_fail_closed(tmp_path):
    cand = candidate("LIIC", {"L_C_upd": "he3"})
    path = tmp_path / "snapshot.json"
    _snap(cand, path)
    kwargs = dict(snapshot_path=str(path), profile={0: cand},
                  client_samples={0: 10}, admitted_client_ids=(0,),
                  reaches_cloud=lambda _: True)
    with pytest.raises(ValueError, match="stale/late"):
        evaluate_oracle_diagnostic(round_index=1, **kwargs)
    with pytest.raises(ValueError, match="no oracle new-J entry"):
        evaluate_oracle_diagnostic(round_index=0, profile={0: candidate("LIIEIIIC", {"E_C_upd": "he3"})},
                                   **{k: v for k, v in kwargs.items() if k != "profile"})
    data = json.loads(path.read_text())
    data["formal_online_j_certified"] = True
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="non-certification"):
        evaluate_oracle_diagnostic(round_index=0, **kwargs)


def test_strict_j_still_requires_manifest():
    with pytest.raises(ValueError, match="requires online_bound_manifest_path"):
        SelectionConfig(learning_objective="online_upper_bound")
    cfg = SelectionConfig(learning_objective="online_j_oracle_diagnostic")
    assert cfg.learning_objective == "online_j_oracle_diagnostic"


def test_predecision_snapshot_actual_full_model_gradient(monkeypatch):
    import torch
    import dynfed.fmnist_lenet5_dynamic as training
    # Small synthetic models are used to make this a fast code path check.
    # The builder delegates the real model reference calculation to the
    # existing exact full-gradient helper, which is tested separately.
    end = torch.nn.Linear(2, 3)
    edge = torch.nn.Linear(3, 2)
    monkeypatch.setattr(training, "_calibration_reference_update",
        lambda **kwargs: {
            "end": {n: torch.ones_like(p) for n, p in end.named_parameters()},
            "edge": {n: torch.ones_like(p) for n, p in edge.named_parameters()},
        })
    cand1 = candidate("LIIC", {"L_C_upd": "he3"})
    cand2 = candidate("LIIEIIIC", {"E_C_upd": "he3", "L_E_upd": "none"})
    data = make_oracle_snapshot(
        round_index=0,
        selected=[(0, cand1, [cand1, cand2], 8.0),
                  (1, cand1, [cand1, cand2], 8.0)],
        client_train_indices={0: np.array([0, 1]), 1: np.array([2, 3])},
        x_train=np.zeros((4, 2), dtype=np.float32),
        y_train=np.zeros(4, dtype=np.int64),
        global_end=end, global_edge=edge, model_name="lenet5",
        input_shape=(1, 1, 2), device=torch.device("cpu"),
        learning_rate=.1, cloud_step=1., weight_decay=0.,
        local_epochs=2, batch_size=2, sample_clip_norm=3.,
        privacy_unit="client", step_cap=None,
    )
    assert data["formal_online_j_certified"] is False
    assert data["used_raw_client_training_data"] is True
    assert len(data["clients"]["0"]["candidates"]) == 2
    # LIIEIIIC includes three actual Edge SGD/retraining loops; LIIC one.
    entries = {row["mode"]: row for row in data["clients"]["0"]["candidates"].values()}
    assert entries["LIIC"]["edge_training_stages"] == 1
    assert entries["LIIEIIIC"]["edge_training_stages"] == 3
    assert entries["LIIEIIIC"]["update_rms_bound"] > entries["LIIC"]["update_rms_bound"]
    assert data["schema"] == "dynfl.online_j.oracle_diagnostic.v2"
    assert len(data["ideal_client_update_gram"]) == 2
    assert data["ideal_population_update_norm"] > 0


def test_simulator_j_end_to_end_two_real_cloud_modes(monkeypatch, tmp_path):
    """Actual candidate enumeration + Pareto + one training round, synthetic data.

    HE availability is mocked ONLY to allow profiled (plaintext) HE mode
    enumeration in CI. Never a real encryption/DP/deployment attestation.
    """
    import csv
    import torch
    from dynfed import fmnist_lenet5_dynamic as f
    from dynfed.he_backend import HEAvailability

    prior_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        rng = np.random.default_rng(2026)
        x = rng.random((60, 1, 28, 28), dtype=np.float32)
        y = np.arange(60, dtype=np.int64) % 10
        monkeypatch.setattr(f, "load_image_dataset_arrays", lambda *a, **k:
                            (x[:40], y[:40], x[40:], y[40:], (1, 28, 28), "synthetic", 10))
        monkeypatch.setattr(f, "check_he_backend", lambda backend, deps:
                            HEAvailability(True, "seal", "MOCK-ONLY profiled HE"))
        selection = SelectionConfig(
            rounds=1, num_clients=2, num_edges=1, seed=2026,
            output_dir=str(tmp_path),
            learning_objective="online_j_oracle_diagnostic",
            uncapped_local_batches=True, fl_first_split_on_demand=True,
            edge_only_requires_fast_deadline=True,
            require_cloud_participation=True, require_feasible=True,
            update_mechanism_options=("he3",), privacy_unit="sample",
            pareto_max_iters=3, pareto_beam_size=2, pareto_archive_size=5,
            omega_update_clip_norm=3.0, initial_epsilon=8.,
        )
        training = f.Lenet5Config(
            local_epochs=1, train_batch_size=256, device="cpu",
            he_backend="seal", he_execution="profiled", model_name="lenet5",
            dp_release_calibration="tex_packet", console_format="quiet",
            dp_sample_optimizer_clip_norm=3.0,
        )
        summary = f.run_fmnist_lenet5_training(
            selection=selection, train_config=training,
            policies=("full_dynfl",), train_limit=40, test_limit=20,
        )
        assert summary["status"] == "completed"
        out = tmp_path / "full_dynfl"
        snap = json.loads((out / "online_j_ORACLE_NONCERTIFIED_round_000.json").read_text())
        assert snap["formal_online_j_certified"] is False
        assert {row["mode"] for row in snap["clients"]["0"]["candidates"].values()} == {
            "LIIC", "LIIEIIIC"
        }
        with (out / "pareto_profile_audit.csv").open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        assert rows
        assert any(int(row["LIIC_count"]) > 0 for row in rows)
        assert all(row["learning_proxy_source"] ==
                   "SIMULATOR_ORACLE_NEW_J_NOT_DEPLOYMENT_CERTIFIED" for row in rows)
        assert any(float(row["online_j_radius"]) > 0 for row in rows)
    finally:
        torch.set_num_threads(prior_threads)


def test_v2_missing_cloud_ideal_contribution_and_actual_sample_weights(tmp_path):
    """LIIE contributes -p_i*v_i, never an imaginary edge->Cloud upload."""
    cloud = candidate("LIIC", {"L_C_upd": "he3"})
    edge = candidate("LIIE", {"L_E_upd": "none"})
    def client(n, mode_candidate, b):
        return {
            "samples": n, "ideal_client_update_norm": 1.0,
            "candidates": ({certificate_candidate_key(mode_candidate): {
                "mode": mode_candidate.mode, "update_rms_bound": b,
                "he_only_no_extra_release_dp": True,
            }} if b is not None else {}),
        }
    data = {
        "schema": "dynfl.online_j.oracle_diagnostic.v2",
        "round_index": 0, "available_before_round": 0,
        "formal_online_j_certified": False, "used_raw_client_training_data": True,
        "cloud_step": 1.0,
        # v0=(1,0), v1=(0,1), population masses 1:3.
        "ideal_population_update_norm": math.sqrt(.625),
        "ideal_gram_client_ids": [0, 1],
        "ideal_client_update_gram": [[1., 0.], [0., 1.]],
        "clients": {"0": client(10, cloud, .5), "1": client(30, edge, None)},
    }
    path = tmp_path / "v2.json"
    path.write_text(json.dumps(data))
    kwargs = dict(snapshot_path=str(path), round_index=0,
        profile={0: cloud, 1: edge}, client_samples={0: 10, 1: 30},
        admitted_client_ids=(0, 1),
        reaches_cloud=lambda c: c.mode == "LIIC")
    result = evaluate_oracle_diagnostic(**kwargs)
    # c=(1,0), p=(.25,.75), A=||(0.75,-0.75)||; r=(.5+1.0).
    expected_a = math.sqrt(2 * .75 ** 2)
    assert result.central_norm == pytest.approx(expected_a)
    assert result.upper == pytest.approx((expected_a + 1.5) ** 2)
    assert result.reference_radius == 0
    assert result.q_aggregate == 0
    # One client disappears from Cloud entirely: A=||-population mean||,
    # even though their private Edge-only training is still performed.
    none = evaluate_oracle_diagnostic(**{**kwargs, "admitted_client_ids": (1,)})
    assert none.central_norm == pytest.approx(math.sqrt(.625))
    assert none.upper == pytest.approx(.625)


def test_v2_rejects_forged_gram_and_illegal_cloud_split(tmp_path):
    cloud = candidate("LIIC", {"L_C_upd": "he3"})
    edge = candidate("LIEIIC", {"L_E_emb": "trusted", "L_E_grad": "trusted", "E_C_upd": "he3"})
    path = tmp_path / "v2.json"
    data = {
        "schema": "dynfl.online_j.oracle_diagnostic.v2",
        "round_index": 0, "available_before_round": 0,
        "formal_online_j_certified": False, "used_raw_client_training_data": True,
        "cloud_step": 1., "ideal_population_update_norm": 1.,
        "ideal_gram_client_ids": [0], "ideal_client_update_gram": [[1.]],
        "clients": {"0": {"samples": 10, "ideal_client_update_norm": 1.,
            "candidates": {certificate_candidate_key(cloud): {
                "mode": "LIIC", "update_rms_bound": 2.,
                "he_only_no_extra_release_dp": True,
            }}}},
    }
    path.write_text(json.dumps(data))
    kwargs = dict(snapshot_path=str(path), round_index=0,
                  profile={0: cloud}, client_samples={0: 10},
                  admitted_client_ids=(0,), reaches_cloud=lambda _: True)
    assert evaluate_oracle_diagnostic(**kwargs).upper == pytest.approx(9.)
    data["ideal_client_update_gram"] = [[2.]]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="diagonal"):
        evaluate_oracle_diagnostic(**kwargs)
    data["ideal_client_update_gram"] = [[1.]]
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="no oracle new-J entry"):
        evaluate_oracle_diagnostic(**{**kwargs, "profile": {0: edge}})


def test_trusted_split_requires_client_unit_and_no_internal_packet_dp(monkeypatch):
    import torch
    import dynfed.fmnist_lenet5_dynamic as training
    end = torch.nn.Linear(2, 3)
    edge_model = torch.nn.Linear(3, 2)
    monkeypatch.setattr(training, "_calibration_reference_update",
        lambda **kwargs: {
            "end": {n: torch.ones_like(p) for n, p in end.named_parameters()},
            "edge": {n: torch.ones_like(p) for n, p in edge_model.named_parameters()},
        })
    trusted = candidate("LIEIIC", {
        "L_E_emb": "trusted", "L_E_grad": "trusted", "E_C_upd": "he3",
    })
    untrusted = candidate("LIEIIC", {
        "L_E_emb": "none", "L_E_grad": "none", "E_C_upd": "he3",
    })
    args = dict(
        round_index=0, client_train_indices={0: np.arange(2)},
        x_train=np.zeros((2, 2), dtype=np.float32),
        y_train=np.zeros(2, dtype=np.int64),
        global_end=end, global_edge=edge_model, model_name="lenet5",
        input_shape=(1, 1, 2), device=torch.device("cpu"),
        learning_rate=.1, cloud_step=1., weight_decay=0.,
        local_epochs=1, batch_size=2, sample_clip_norm=3., step_cap=None,
    )
    out = make_oracle_snapshot(**{**args, "selected": [(0, trusted, [trusted], 8.)],
                                  "privacy_unit": "client"})
    entry = next(iter(out["clients"]["0"]["candidates"].values()))
    assert entry["bound_source"] == "trusted_split_two_clipped_optimizers_5_plus_5"
    assert entry["edge_training_stages"] == 1
    with pytest.raises(ValueError, match="client-unit clean Split"):
        make_oracle_snapshot(**{**args, "selected": [(0, trusted, [trusted], 8.)],
                                "privacy_unit": "sample"})
    with pytest.raises(ValueError, match="requires trusted"):
        make_oracle_snapshot(**{**args, "selected": [(0, untrusted, [untrusted], 8.)],
                                "privacy_unit": "client"})
