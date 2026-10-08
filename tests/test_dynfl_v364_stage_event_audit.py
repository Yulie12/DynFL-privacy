"""v3.6.4 audit primitive: stage evidence cannot be replaced by predictions."""
from types import SimpleNamespace

import pytest

from dynfed.sample_stage_event_audit import SampleEventCounts as C
from dynfed.sample_stage_event_audit import SampleStageEventAudit


def candidate(z, g, o):
    return SimpleNamespace(
        sample_embedding_events=z,
        sample_label_grad_events=g,
        sample_optimizer_events=o,
    )


def worker(z, g, o):
    return {
        "feature_dp_release_batches": z,
        "sample_label_grad_dp_release_batches": g,
        "sample_dp_optimizer_steps": o,
    }


def test_split_hierarchical_all_stages_reconciled():
    audit = SampleStageEventAudit(
        client_id=44, candidate=candidate(9, 9, 0),
        scheduled_stages={0: C(3, 3, 0), 1: C(3, 3, 0), 2: C(3, 3, 0)},
    )
    for s in range(3):
        audit.observe(stage=s, worker=worker(3, 3, 0))
    assert audit.finalize() == C(9, 9, 0)


def test_full_local_hierarchical_counts_optimizer():
    audit = SampleStageEventAudit(
        client_id=79, candidate=candidate(0, 0, 6),
        scheduled_stages={0: C(0, 0, 2), 1: C(0, 0, 2), 2: C(0, 0, 2)},
    )
    for stage in range(3):
        audit.observe(stage=stage, worker=worker(0, 0, 2))
    assert audit.finalize().optimizer == 6


def test_first_stage_alone_does_not_complete():
    audit = SampleStageEventAudit(
        client_id=1, candidate=candidate(9, 9, 0),
        scheduled_stages={0: C(3, 3, 0), 1: C(3, 3, 0), 2: C(3, 3, 0)},
    )
    audit.observe(stage=0, worker=worker(3, 3, 0))
    with pytest.raises(RuntimeError, match="missing stage"):
        audit.finalize()


def test_predicted_counts_are_not_worker_observations():
    audit = SampleStageEventAudit(
        client_id=1, candidate=candidate(6, 6, 0),
        scheduled_stages={0: C(3, 3, 0), 1: C(3, 3, 0)},
    )
    with pytest.raises(RuntimeError, match="observed"):
        audit.observe(stage=0, worker=worker(6, 6, 0))


def test_cannot_silently_drop_or_duplicate_stage():
    audit = SampleStageEventAudit(
        client_id=1, candidate=candidate(0, 0, 4),
        scheduled_stages={0: C(0, 0, 2), 1: C(0, 0, 2)},
    )
    audit.observe(stage=0, worker=worker(0, 0, 2))
    with pytest.raises(RuntimeError, match="duplicate"):
        audit.observe(stage=0, worker=worker(0, 0, 2))
    with pytest.raises(RuntimeError, match="unknown/retry"):
        audit.observe(stage=2, worker=worker(0, 0, 2))


@pytest.mark.parametrize("bad", [None, True, -1, 1.5, "NaN"])
def test_bad_worker_counter_fails_closed(bad):
    audit = SampleStageEventAudit(
        client_id=1, candidate=candidate(1, 1, 0),
        scheduled_stages={0: C(1, 1, 0)},
    )
    with pytest.raises((ValueError, TypeError)):
        audit.observe(stage=0, worker=worker(bad, 1, 0))


def test_missing_worker_field_fails_closed():
    audit = SampleStageEventAudit(
        client_id=1, candidate=candidate(1, 1, 0),
        scheduled_stages={0: C(1, 1, 0)},
    )
    with pytest.raises(ValueError, match="missing"):
        audit.observe(stage=0, worker={"feature_dp_release_batches": 1})


def test_stage_plan_must_equal_candidate_precharge():
    with pytest.raises(ValueError, match="stage plan"):
        SampleStageEventAudit(
            client_id=1, candidate=candidate(9, 9, 0),
            scheduled_stages={0: C(3, 3, 0)},
        )


def test_noncontiguous_schedule_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        SampleStageEventAudit(
            client_id=1, candidate=candidate(2, 2, 0),
            scheduled_stages={0: C(1, 1, 0), 2: C(1, 1, 0)},
        )


def test_out_of_order_stage_fails_closed():
    audit = SampleStageEventAudit(
        client_id=1, candidate=candidate(2, 2, 0),
        scheduled_stages={0: C(1, 1, 0), 1: C(1, 1, 0)},
    )
    with pytest.raises(RuntimeError, match="out-of-order"):
        audit.observe(stage=1, worker=worker(1, 1, 0))
