"""Opt-in trusted LIE selector -> dispatch -> single-domain DP-SGD accounting."""
from __future__ import annotations

import random

import numpy as np
import pytest
import torch

from dynfed.fmnist_lenet5_dynamic import (
    _assert_sample_worker_accounting_matches,
    _client_train_worker,
    _assert_trusted_lie_accounting_plan,
)
from dynfed.selection import (
    SelectionConfig,
    _sample_dp_event_counts,
    enumerate_candidates,
)
from dynfed.privacy import SamplePrivacyLedger
from dynfed.split_learning import build_split_pair


@pytest.fixture(autouse=True)
def single_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def make_config(**kwargs):
    data = dict(
        privacy_unit="sample", trusted_edge_split_execution=True,
        trusted_lie_joint_sample_dp=True,
        learning_objective="legacy_fusion_dp",
        split_batch_size=64, rounds=2, L_block_cycles=1,
        privacy_local_epochs=1, initial_epsilon=8,
        excluded_modes=("LIC", "LIEIIC", "LIEIIIC", "LIIE", "LIIC", "LIIEIIIC"),
    )
    data.update(kwargs)
    return SelectionConfig(**data)


@pytest.mark.parametrize("override, pattern", [
    ({"privacy_unit": "client"}, "privacy_unit='sample'"),
    ({"trusted_edge_split_execution": False}, "trusted_edge_split_execution"),
    ({"learning_objective": "joint_calibration"}, "new learning-proxy calibration"),
])
def test_requires_explicit_trust_and_no_stale_proxy(override, pattern):
    with pytest.raises(ValueError, match=pattern):
        make_config(**override)


def test_legacy_defaults_preserve_old_link_accounting():
    c = make_config(trusted_lie_joint_sample_dp=False)
    emb, grad, optimizer = _sample_dp_event_counts(c, "LIE", 80)
    assert (emb, grad, optimizer) == (1, 1, 1)


def test_trusted_lie_counts_one_joint_event_and_no_link_dp():
    c = make_config()
    assert _sample_dp_event_counts(c, "LIE", 80) == (0, 0, 1)
    candidates = enumerate_candidates(
        config=c, client_id=0, edge_factor=1., compute_factor=1.,
        samples=80, remaining_epsilon=8, round_idx=0,
        rng=random.Random(7), policy="dynamic",
    )
    assert len(candidates) == 1
    candidate = candidates[0]
    assert candidate.mode == "LIE"
    assert (candidate.sample_embedding_events, candidate.sample_label_grad_events,
            candidate.sample_optimizer_events) == (0, 0, 1)
    assert candidate.sample_embedding_noise_multiplier is None
    assert candidate.sample_label_grad_noise_multiplier is None
    assert candidate.sample_optimizer_noise_multiplier > 0
    assert candidate.link_mechanisms == {
        "L_E_emb": "trusted", "L_E_grad": "trusted"
    }
    ledger = SamplePrivacyLedger(budget=8)
    projection = ledger.project(
        embedding_events=candidate.sample_embedding_events,
        label_grad_events=candidate.sample_label_grad_events,
        optimizer_events=candidate.sample_optimizer_events,
        embedding_noise_multiplier=1., label_grad_noise_multiplier=1.,
        optimizer_noise_multiplier=candidate.sample_optimizer_noise_multiplier,
    )
    assert projection.embedding_events == 0
    assert projection.label_grad_events == 0
    assert projection.optimizer_events == 1


def test_non_lie_modes_still_use_existing_events():
    c = make_config()
    for mode in ("LIC", "LIEIIC", "LIEIIIC"):
        a, b, _ = _sample_dp_event_counts(c, mode, 80)
        assert a > 0 and b > 0


def test_trusted_lie_worker_event_matches_selector_and_redacts_stats():
    torch.manual_seed(13)
    end, edge = build_split_pair("lenet5", torch.device("cpu"))
    payload = dict(
        client_id=0, mode="LIE", global_end_state=end.state_dict(),
        global_edge_state=edge.state_dict(),
        x=np.random.default_rng(2).normal(size=(2, 1, 28, 28)).astype("float32"),
        y=np.asarray([0, 1], dtype="int64"), epochs=1, lr=.01,
        l2=0., local_steps=1, model_name="lenet5", input_shape=(1,28,28),
        num_classes=10, mechanisms={"emb":"trusted","grad":"trusted","upd":"none"},
        privacy_unit="sample", trusted_split_joint_sample_dp=True,
        trusted_edge=True, sample_embedding_noise_multiplier=None,
        sample_label_grad_noise_multiplier=None,
        sample_optimizer_noise_multiplier=1.0, sample_optimizer_clip_norm=1.,
        dp_clip_norm=.25, dp_feature_noise_multiplier=1.,
        dp_update_noise_multiplier=1., dp_update_mode="off", dp_epsilon=4.,
        dp_seed=12, training_seed=17, device="cpu",
    )
    worker = _client_train_worker(payload)
    assert worker["finite"]
    assert worker["feature_dp_release_batches"] == 0
    assert worker["sample_label_grad_dp_release_batches"] == 0
    assert worker["sample_dp_optimizer_steps"] == 1
    assert worker["trusted_split_joint_dp_steps"] == 1
    assert worker["trusted_split_sensitive_gradient_stats_redacted"] == 1
    assert worker["sample_dp_clipped_sample_count"] == 0
    assert worker["sample_dp_max_raw_grad_norm"] == 0
    assert worker["sample_dp_max_clipped_grad_norm"] == 0

    class Precharged:
        sample_embedding_events = 0
        sample_label_grad_events = 0
        sample_optimizer_events = 1

    assert _assert_sample_worker_accounting_matches(
        admitted_client_ids={0}, selected_by_id={0: Precharged()},
        worker_results={0: worker},
    ) == 1
    with pytest.raises(RuntimeError, match="ledger=2"):
        class Wrong(Precharged):
            sample_optimizer_events = 2
        _assert_sample_worker_accounting_matches(
            admitted_client_ids={0}, selected_by_id={0: Wrong()},
            worker_results={0: worker},
        )


def test_preflight_rejects_mismatched_runtime_batch_before_training():
    c = make_config(split_batch_size=128)
    candidate = enumerate_candidates(
        config=c, client_id=0, edge_factor=1., compute_factor=1.,
        samples=80, remaining_epsilon=8, round_idx=0,
        rng=random.Random(7), policy="dynamic",
    )[0]
    with pytest.raises(RuntimeError, match="batch size mismatch"):
        _assert_trusted_lie_accounting_plan(
            selection=c, candidate=candidate,
            model_name="lenet5", worker_device=torch.device("cpu"),
        )
    _assert_trusted_lie_accounting_plan(
        selection=make_config(), candidate=candidate,
        model_name="lenet5", worker_device=torch.device("cpu"),
    )
