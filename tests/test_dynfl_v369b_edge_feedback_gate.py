"""v3.6.9b: Edge model feedback must consume fresh, audited stage receipts.

This only verifies the hierarchical release gate. It is not a sample-level
end-to-end differential privacy proof and does not lift the safety preflight.
"""
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynfed.sample_stage_event_audit import SampleStageEventAudit, SampleEventCounts


def make_audit():
    candidate = SimpleNamespace(
        sample_embedding_events=0,
        sample_label_grad_events=0,
        sample_optimizer_events=6,
        sample_optimizer_noise_multiplier=2.0,
    )
    return SampleStageEventAudit(
        client_id=4,
        candidate=candidate,
        scheduled_stages={i: SampleEventCounts(0, 0, 2) for i in range(3)},
        require_runtime_parameters=True,
        optimizer_clip_norm=1.0,
    )


def receipt(*, finite=True, sigma=2.0):
    return dict(
        feature_dp_release_batches=0,
        sample_label_grad_dp_release_batches=0,
        sample_dp_optimizer_steps=2,
        executed_sample_optimizer_sigma=sigma,
        executed_sample_optimizer_clip_norm=1.0,
        finite=finite,
    )


def test_edge_aggregation_rejects_missing_receipt_and_replay():
    a = make_audit()
    with pytest.raises(RuntimeError, match="no audited worker receipt"):
        a.authorize_edge_aggregate(stage=0)
    a.observe(stage=0, worker=receipt())
    a.authorize_edge_aggregate(stage=0)
    with pytest.raises(RuntimeError, match="duplicate Edge aggregation"):
        a.authorize_edge_aggregate(stage=0)
    with pytest.raises(RuntimeError, match="no audited worker receipt"):
        a.authorize_edge_aggregate(stage=1)
    with pytest.raises(RuntimeError, match="do not match"):
        a.finalize_edge_aggregations()


def test_edge_aggregation_rejects_out_of_order_or_invalid_parameter():
    a = make_audit()
    a.observe(stage=0, worker=receipt())
    a.observe(stage=1, worker=receipt())
    with pytest.raises(RuntimeError, match="out of order"):
        a.authorize_edge_aggregate(stage=1)
    with pytest.raises(RuntimeError, match="charged="):
        a.observe(stage=2, worker=receipt(sigma=1.0))


def test_complete_three_stage_feedback_and_final_release_gate():
    a = make_audit()
    for stage in range(3):
        a.observe(stage=stage, worker=receipt())
        a.authorize_edge_aggregate(stage=stage)
    assert a.finalize() == SampleEventCounts(0, 0, 6)
    a.finalize_edge_aggregations()


def test_admitted_client_cannot_finalize_with_missing_edge_feedback():
    a = make_audit()
    for stage in range(3):
        a.observe(stage=stage, worker=receipt())
        if stage != 2:
            a.authorize_edge_aggregate(stage=stage)
    a.finalize()
    with pytest.raises(RuntimeError, match="do not match"):
        a.finalize_edge_aggregations()


def test_runtime_wires_audit_before_edge_and_finalization_before_cloud():
    source = (
        Path(__file__).resolve().parents[1] / "dynfed" / "fmnist_lenet5_dynamic.py"
    ).read_text(encoding="utf-8-sig")
    edge_loop = source.index("        multi_edge_iter = ")
    receipt_observe = source.index("sample_stage_audits[cycle_cid].observe(", edge_loop)
    guard = source.index("sample_stage_audits[cycle_cid].authorize_edge_aggregate(", edge_loop)
    aggregate = source.index("returned_state, used_real_he = _aggregate_returned_client_models(", edge_loop)
    finalize = source.index("sample_stage_audit.finalize_edge_aggregations()", aggregate)
    cloud = source.index("        admitted_updates = [", finalize)
    assert receipt_observe < guard < aggregate < finalize < cloud
    assert 'if not bool(cycle_results[cycle_cid].get("finite", False)):' in source[receipt_observe:guard]
    # The pre-dispatch hierarchical Sample-DP safety restriction remains.
    assert '        _assert_sample_hierarchical_preflight(' in source
