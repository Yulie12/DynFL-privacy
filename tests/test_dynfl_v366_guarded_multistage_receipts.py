"""v3.6.6 Stage receipts wired while multi-stage Sample-DP stays blocked."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynfed.sample_stage_event_audit import (
    SampleEventCounts as C,
    SampleStageEventAudit,
    uniform_sample_stage_plan,
)
from dynfed.privacy import SamplePrivacyLedger
from dynfed.sample_dispatch_accounting import charge_sample_dispatch_before_worker


def candidate(z=9, g=9, o=0, sigma=5.0):
    return SimpleNamespace(
        sample_embedding_events=z,
        sample_label_grad_events=g,
        sample_optimizer_events=o,
        sample_embedding_noise_multiplier=sigma,
        sample_label_grad_noise_multiplier=sigma,
        sample_optimizer_noise_multiplier=sigma if o else None,
    )


def worker(z=3, g=3, o=0):
    return dict(feature_dp_release_batches=z,
                sample_label_grad_dp_release_batches=g,
                sample_dp_optimizer_steps=o)


@pytest.mark.parametrize('totals,stages,each', [
    ((9,9,0), 3, (3,3,0)),
    ((0,0,6), 3, (0,0,2)),
    ((3,3,0), 1, (3,3,0)),
    ((0,0,0), 3, (0,0,0)),
])
def test_uniform_schedule_exact_sum(totals, stages, each):
    plan = uniform_sample_stage_plan(candidate=candidate(*totals), stage_count=stages)
    assert len(plan) == stages
    assert list(plan.values()) == [C(*each)] * stages
    assert tuple(sum(getattr(item, field) for item in plan.values())
                 for field in ('embedding','label_grad','optimizer')) == totals


@pytest.mark.parametrize('bad_count', [0, -1, 1.5, True, 'NaN'])
def test_invalid_stage_count_rejected(bad_count):
    with pytest.raises(ValueError):
        uniform_sample_stage_plan(candidate=candidate(), stage_count=bad_count)


def test_uneven_plan_fails_closed_without_floor_division():
    with pytest.raises(ValueError, match='not divisible'):
        uniform_sample_stage_plan(candidate=candidate(z=8,g=9), stage_count=3)


def test_cloud_dropped_client_has_audited_first_stage_and_no_refund():
    spec = candidate()
    ledger = SamplePrivacyLedger(20.0)
    charge_sample_dispatch_before_worker(
        train_tasks=[(7,spec,None,0)], privacy_ledgers={7:ledger})
    audit = SampleStageEventAudit(client_id=7,candidate=spec,
        scheduled_stages=uniform_sample_stage_plan(candidate=spec,stage_count=3))
    audit.observe(stage=0,worker=worker())
    assert audit.finalize_executed_prefix(executed_stages=1) == C(3,3,0)
    assert ledger.embedding_events == 9  # conservatively charged all stages
    with pytest.raises(RuntimeError,match='missing stage'):
        audit.finalize()


def test_admitted_client_must_have_all_stage_receipts():
    spec = candidate()
    audit = SampleStageEventAudit(client_id=7,candidate=spec,
        scheduled_stages=uniform_sample_stage_plan(candidate=spec,stage_count=3))
    audit.observe(stage=0,worker=worker())
    audit.observe(stage=1,worker=worker())
    with pytest.raises(RuntimeError,match='missing stage'):
        audit.finalize()
    audit.observe(stage=2,worker=worker())
    assert audit.finalize() == C(9,9,0)


def test_dropped_prefix_needs_all_expected_receipts():
    spec=candidate()
    audit=SampleStageEventAudit(client_id=7,candidate=spec,
        scheduled_stages=uniform_sample_stage_plan(candidate=spec,stage_count=3))
    with pytest.raises(RuntimeError,match='prefix mismatch'):
        audit.finalize_executed_prefix(executed_stages=1)
    audit.observe(stage=0,worker=worker())
    with pytest.raises(RuntimeError,match='prefix mismatch'):
        audit.finalize_executed_prefix(executed_stages=2)
    with pytest.raises(ValueError,match='range'):
        audit.finalize_executed_prefix(executed_stages=4)


def test_stage_zero_must_be_observed_before_flow_and_later_before_aggregate():
    source=(Path(__file__).resolve().parents[1]/'dynfed/fmnist_lenet5_dynamic.py').read_text(encoding='utf8')
    guard=source.index('        _assert_sample_hierarchical_preflight(\n')
    charge=source.index('            sample_dispatch_charges = charge_sample_dispatch_before_worker(\n')
    stage_zero=source.index('                sample_stage_audit.observe(stage=0, worker=worker_results[sample_cid])')
    flow=source.index('        flow_result = execute_mixed_round_flow(\n')
    later_stage=source.index('                            sample_stage_audits[cycle_cid].observe(\n')
    edge_aggregate=source.index('                returned_state, used_real_he = _aggregate_returned_client_models(\n')
    final_audit=source.index('                    sample_stage_audit.finalize_executed_prefix(executed_stages=1)')
    cloud_updates=source.index('        admitted_updates = [\n')
    assert guard < charge < stage_zero < flow < later_stage < edge_aggregate < final_audit < cloud_updates
    ast.parse(source)
