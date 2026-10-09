"""Fail-closed per-stage DP parameters (not a mathematical privacy proof)."""
from types import SimpleNamespace

import pytest

from dynfed.sample_stage_event_audit import SampleEventCounts, SampleStageEventAudit


def _candidate(*, split):
    return SimpleNamespace(
        sample_embedding_events=3 if split else 0,
        sample_label_grad_events=3 if split else 0,
        sample_optimizer_events=0 if split else 3,
        sample_embedding_noise_multiplier=1.25 if split else None,
        sample_label_grad_noise_multiplier=1.5 if split else None,
        sample_optimizer_noise_multiplier=None if split else 1.75,
    )


def _audit(*, split):
    candidate = _candidate(split=split)
    return SampleStageEventAudit(
        client_id=7,
        candidate=candidate,
        scheduled_stages={0: SampleEventCounts(3, 3, 0) if split else SampleEventCounts(0, 0, 3)},
        require_runtime_parameters=True,
        feature_clip_norm=0.25,
        optimizer_clip_norm=1.0,
    )


def _worker(*, split):
    result = {
        "feature_dp_release_batches": 3 if split else 0,
        "sample_label_grad_dp_release_batches": 3 if split else 0,
        "sample_dp_optimizer_steps": 0 if split else 3,
    }
    if split:
        result.update(
            executed_sample_embedding_sigma=1.25,
            executed_sample_label_grad_sigma=1.5,
            executed_sample_feature_clip_norm=0.25,
        )
    else:
        result.update(
            executed_sample_optimizer_sigma=1.75,
            executed_sample_optimizer_clip_norm=1.0,
        )
    return result


@pytest.mark.parametrize("split", [False, True])
def test_accepts_matching_runtime_params(split):
    audit = _audit(split=split)
    audit.observe(stage=0, worker=_worker(split=split))
    assert audit.finalize().optimizer == (0 if split else 3)


@pytest.mark.parametrize("split", [False, True])
def test_missing_runtime_param_fails_closed(split):
    audit = _audit(split=split)
    worker = _worker(split=split)
    worker.pop("executed_sample_feature_clip_norm" if split else "executed_sample_optimizer_clip_norm")
    with pytest.raises(RuntimeError, match="missing/invalid runtime"):
        audit.observe(stage=0, worker=worker)


@pytest.mark.parametrize("split", [False, True])
def test_runtime_noise_mismatch_fails_closed(split):
    audit = _audit(split=split)
    worker = _worker(split=split)
    key = "executed_sample_embedding_sigma" if split else "executed_sample_optimizer_sigma"
    worker[key] = 0.5
    with pytest.raises(RuntimeError, match="charged="):
        audit.observe(stage=0, worker=worker)


@pytest.mark.parametrize("invalid", [None, -1, float('nan'), float('inf'), True, 0])
def test_invalid_noise_fails_closed(invalid):
    audit = _audit(split=False)
    worker = _worker(split=False)
    worker["executed_sample_optimizer_sigma"] = invalid
    with pytest.raises(RuntimeError, match="missing/invalid runtime"):
        audit.observe(stage=0, worker=worker)


def test_strict_mode_requires_candidate_noise():
    cand = _candidate(split=False)
    cand.sample_optimizer_noise_multiplier = None
    with pytest.raises(ValueError, match="sample_optimizer_noise_multiplier"):
        SampleStageEventAudit(
            client_id=1, candidate=cand,
            scheduled_stages={0: SampleEventCounts(0, 0, 3)},
            require_runtime_parameters=True,
            optimizer_clip_norm=1.0,
        )


@pytest.mark.parametrize("mode", ["LIE", "LIIE"])
def test_real_training_reports_execution_site_dp_parameters(mode):
    """Tiny actual Sample-DP worker, not a payload-value echo/mocked receipt."""
    import numpy as np
    import torch

    from dynfed.split_learning import build_split_pair, split_local_train_lenet5

    end, edge = build_split_pair(
        "lenet5", torch.device("cpu"), input_channels=1, image_size=28, num_classes=10
    )
    end_state = {k: v.detach().clone() for k, v in end.state_dict().items()}
    edge_state = {k: v.detach().clone() for k, v in edge.state_dict().items()}
    rng = np.random.default_rng(4)
    x = rng.random((2, 1, 28, 28), dtype=np.float32)
    y = np.array([0, 1], dtype=np.int64)
    diagnostics = {}
    split_local_train_lenet5(
        mode=mode,
        global_end_state=end_state,
        global_edge_state=edge_state,
        x=x, y=y, epochs=1, lr=0.01,
        device=torch.device("cpu"), model_name="lenet5",
        input_shape=(1, 28, 28), num_classes=10,
        mechanisms={"emb": "dp", "grad": "dp"},
        dp_clip_norm=0.25, dp_rng=np.random.default_rng(7),
        training_diagnostics=diagnostics, privacy_unit="sample",
        sample_embedding_noise_multiplier=1.25,
        sample_label_grad_noise_multiplier=1.5,
        sample_optimizer_noise_multiplier=1.75,
        sample_optimizer_clip_norm=1.0, local_steps=1,
    )
    assert diagnostics["sample_dp_optimizer_steps"] == 1
    assert diagnostics["executed_sample_optimizer_sigma"] == 1.75
    assert diagnostics["executed_sample_optimizer_clip_norm"] == 1.0
    if mode == "LIE":
        assert diagnostics["feature_dp_release_batches"] == 1
        assert diagnostics["sample_label_grad_dp_release_batches"] == 1
        assert diagnostics["executed_sample_embedding_sigma"] == 1.25
        assert diagnostics["executed_sample_label_grad_sigma"] == 1.5
        assert diagnostics["executed_sample_feature_clip_norm"] == 0.25
    else:
        assert diagnostics["feature_dp_release_batches"] == 0
        assert diagnostics["sample_label_grad_dp_release_batches"] == 0


def test_three_stage_parameter_receipts_are_checked_at_every_stage():
    candidate = _candidate(split=False)
    candidate.sample_optimizer_events = 9
    audit = SampleStageEventAudit(
        client_id=7, candidate=candidate,
        scheduled_stages={stage: SampleEventCounts(0, 0, 3) for stage in range(3)},
        require_runtime_parameters=True, optimizer_clip_norm=1.0,
    )
    audit.observe(stage=0, worker=_worker(split=False))
    with pytest.raises(RuntimeError, match="missing stage"):
        audit.finalize()
    tampered = _worker(split=False)
    tampered["executed_sample_optimizer_clip_norm"] = 2.0
    with pytest.raises(RuntimeError, match="charged="):
        audit.observe(stage=1, worker=tampered)
    audit.observe(stage=1, worker=_worker(split=False))
    audit.observe(stage=2, worker=_worker(split=False))
    assert audit.finalize() == SampleEventCounts(0, 0, 9)


def test_dropped_client_can_only_close_audited_prefix_with_full_charge_retained():
    from dynfed.privacy import SamplePrivacyLedger
    from dynfed.sample_dispatch_accounting import charge_sample_dispatch_before_worker
    candidate = _candidate(split=False)
    candidate.sample_optimizer_events = 9
    ledger = SamplePrivacyLedger(20.0)
    charge_sample_dispatch_before_worker(
        train_tasks=[(7, candidate, None, 0)], privacy_ledgers={7: ledger},
    )
    audit = SampleStageEventAudit(
        client_id=7, candidate=candidate,
        scheduled_stages={stage: SampleEventCounts(0, 0, 3) for stage in range(3)},
        require_runtime_parameters=True, optimizer_clip_norm=1.0,
    )
    audit.observe(stage=0, worker=_worker(split=False))
    assert audit.finalize_executed_prefix(executed_stages=1) == SampleEventCounts(0, 0, 3)
    assert ledger.optimizer_events == 9
    with pytest.raises(RuntimeError, match="missing stage"):
        audit.finalize()
