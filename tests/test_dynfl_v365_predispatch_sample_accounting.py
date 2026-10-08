"""v3.6.5: paid-on-dispatch Sample-DP accounting, NOT seven-mode release proof."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from dynfed.privacy import SamplePrivacyLedger
from dynfed.sample_dispatch_accounting import charge_sample_dispatch_before_worker


def candidate(z=3, g=3, o=0, sigma=5.0):
    return SimpleNamespace(
        sample_embedding_events=z,
        sample_label_grad_events=g,
        sample_optimizer_events=o,
        sample_embedding_noise_multiplier=sigma,
        sample_label_grad_noise_multiplier=sigma,
        sample_optimizer_noise_multiplier=sigma if o else None,
    )


def test_all_dispatched_are_debited_including_client_not_admitted_by_cloud():
    ledgers = {1: SamplePrivacyLedger(20.0), 2: SamplePrivacyLedger(20.0)}
    charges = charge_sample_dispatch_before_worker(
        train_tasks=[(1, candidate(), None, 0), (2, candidate(o=1), None, 1)],
        privacy_ledgers=ledgers,
    )
    assert set(charges) == {1, 2}
    assert ledgers[1].embedding_events == 3
    assert ledgers[2].optimizer_events == 1
    # Simulate client 2 NOT being included in the returned Cloud cohort:
    admitted = {1}
    assert 2 not in admitted
    assert ledgers[2].embedding_events == 3
    assert charges[2].projection.optimizer_events == 1


def test_budget_failure_blocks_all_dispatch_before_any_debit():
    ledgers = {1: SamplePrivacyLedger(20.0), 2: SamplePrivacyLedger(0.1)}
    with pytest.raises(ValueError, match='before worker'):
        charge_sample_dispatch_before_worker(
            train_tasks=[(1, candidate(), None, 0), (2, candidate(), None, 1)],
            privacy_ledgers=ledgers,
        )
    assert ledgers[1].embedding_events == 0
    assert ledgers[2].embedding_events == 0


def test_retry_counts_as_another_execution_when_called_again():
    ledger = SamplePrivacyLedger(20.0)
    params = dict(train_tasks=[(1, candidate(), None, 0)], privacy_ledgers={1: ledger})
    charge_sample_dispatch_before_worker(**params)
    charge_sample_dispatch_before_worker(**params)
    assert ledger.embedding_events == 6
    assert ledger.label_grad_events == 6


def test_duplicate_client_rejected_before_charge():
    ledger = SamplePrivacyLedger(20.0)
    with pytest.raises(RuntimeError, match='duplicate'):
        charge_sample_dispatch_before_worker(
            train_tasks=[(1, candidate(), None, 0), (1, candidate(), None, 1)],
            privacy_ledgers={1: ledger},
        )
    assert ledger.embedding_events == 0


def test_optimizer_release_without_multiplier_rejected():
    ledger = SamplePrivacyLedger(20.0)
    with pytest.raises(ValueError, match='noise multiplier'):
        charge_sample_dispatch_before_worker(
            train_tasks=[(1, candidate(o=2, sigma=None), None, 0)],
            privacy_ledgers={1: ledger},
        )
    assert ledger.optimizer_events == 0


def test_runtime_integration_uses_predispatch_charge_checkpoint_and_all_worker_audit():
    source_path = Path(__file__).resolve().parents[1] / 'dynfed/fmnist_lenet5_dynamic.py'
    text = source_path.read_text(encoding='utf8')
    preflight = text.index('        _assert_sample_hierarchical_preflight(\n')
    debit = text.index('            sample_dispatch_charges = charge_sample_dispatch_before_worker(\n')
    checkpoint = text.index('                _save_policy_checkpoint(\n', debit)
    worker = text.index('        worker_results = _run_client_training_tasks(\n')
    audit = text.index('                admitted_client_ids=first_pass_complete_ids,\n')
    flow = text.index('        flow_result = execute_mixed_round_flow(\n')
    assert preflight < debit < checkpoint < worker < audit < flow
    assert 'projection = charge.projection' in text
    assert 'sample_stage_audit.observe(stage=0, worker=worker_results[sample_cid])' in text
    assert 'first_pass_complete_ids' in text
    assert 'sample_worker_client_audit_count = len(sample_stage_audits)' in text
    assert 'if (client_id in admitted_client_ids or' in text
    assert 'next_round=round_idx,' in text[checkpoint:worker]
    ast.parse(text)
