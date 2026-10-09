"""v3.6.8 attempt-specific randomness and write-ahead ordering contracts."""
import ast
from pathlib import Path

import pytest

from dynfed.sample_dispatch_identity import SampleDispatchAttempt, SampleDispatchIdentity


def test_retry_after_persisted_checkpoint_changes_every_seed_stream():
    identity = SampleDispatchIdentity(run_nonce="e" * 32)
    first = identity.next_attempt()
    saved = identity.state_dict()
    resumed = SampleDispatchIdentity.from_state_dict(saved)
    retry = resumed.next_attempt()
    assert first.seed("worker-dp", 5, 17, 0) != retry.seed("worker-dp", 5, 17, 0)
    assert first.seed("worker-train", 5, 17, 2) != retry.seed("worker-train", 5, 17, 2)
    assert first.seed("aggregate-dp", 5, "edge-1", 0) != retry.seed("aggregate-dp", 5, "edge-1", 0)
    assert first.seed("worker-dp", 5, 17, 0) == first.seed("worker-dp", 5, 17, 0)
    assert resumed.sequence == 2


def test_all_stages_clients_and_purposes_are_separate():
    a = SampleDispatchIdentity(run_nonce="b" * 32).next_attempt()
    seeds = {a.seed(purpose, 1, cid, stage)
             for purpose in ("worker-dp", "worker-train", "aggregate-dp")
             for cid in (2, 3) for stage in range(3)}
    assert len(seeds) == 18


def test_aggregate_release_purposes_are_domain_separated():
    # Client id and edge id may have the same numeric value, but their DP
    # release mechanisms must never share a noise stream within an attempt.
    attempt = SampleDispatchIdentity(run_nonce="c" * 32).next_attempt()
    local_seed = attempt.seed("aggregate-dp:client-local-update", 0, 7, 10_000)
    edge_seed = attempt.seed("aggregate-dp:edge-local-update", 0, 7, 10_000)
    assert local_seed != edge_seed


def test_independent_runs_have_distinct_nonce_and_randomness():
    a = SampleDispatchIdentity().next_attempt()
    b = SampleDispatchIdentity().next_attempt()
    assert a.run_nonce != b.run_nonce
    assert a.seed("worker-dp", 0, 1) != b.seed("worker-dp", 0, 1)


@pytest.mark.parametrize("state", [
    {}, {"version": 2, "run_nonce": "f" * 32, "sequence": 1},
    {"version": 1, "run_nonce": "bad", "sequence": 1},
    {"version": 1, "run_nonce": "f" * 32, "sequence": -1},
    {"version": 1, "run_nonce": "f" * 32, "sequence": True},
    {"version": 1, "run_nonce": "f" * 32, "sequence": "1"},
])
def test_invalid_checkpoint_fails_closed(state):
    with pytest.raises((ValueError, RuntimeError)):
        SampleDispatchIdentity.from_state_dict(state)


def test_private_worker_seed_requires_identity_and_persists_before_worker():
    source = (Path(__file__).resolve().parents[1] / "dynfed/fmnist_lenet5_dynamic.py").read_text()
    charge = source.index("            sample_dispatch_charges = charge_sample_dispatch_before_worker(\n")
    new_attempt = source.index("                sample_dispatch_attempt = sample_dispatch_identity.next_attempt()")
    checkpoint = source.index("                _save_policy_checkpoint(\n", new_attempt)
    worker = source.index("        worker_results = _run_client_training_tasks(\n", checkpoint)
    assert charge < new_attempt < checkpoint < worker
    assert "sample_dispatch_identity=sample_dispatch_identity," in source[checkpoint:worker]
    assert "sample_dispatch_attempt=sample_dispatch_attempt," in source[worker:]
    assert 'checkpoint.get("sample_dispatch_identity")' in source
    ast.parse(source)


def test_checkpoint_persists_attempt_identity_and_model_state(tmp_path):
    import random
    from types import SimpleNamespace
    import numpy as np
    import torch
    from dynfed.fmnist_lenet5_dynamic import _save_policy_checkpoint, _load_policy_checkpoint

    identity = SampleDispatchIdentity(run_nonce="a" * 32)
    identity.next_attempt()
    end = torch.nn.Linear(2, 1)
    edge = torch.nn.Linear(2, 1)
    output = tmp_path / "checkpoint.pt"
    _save_policy_checkpoint(
        output, policy="full_dynfl", next_round=0,
        global_end=end, global_edge=edge, remaining_epsilon={1: 7.0},
        privacy_ledgers={}, previous_choices={}, fixed_mode_assignments={},
        fixed_privacy_profiles={}, round_rows=[], decision_rows=[],
        candidate_mode_audit_rows=[], pareto_profile_audit_rows=[],
        flow_event_rows=[], link_state_rows=[], best_accuracy=0.0,
        logical_time=0.0, rng=random.Random(2), np_rng=np.random.default_rng(2),
        train_config=SimpleNamespace(), selection=SimpleNamespace(),
        real_he_rounds=0, real_he_aggregated_clients=0,
        global_pareto_selection_rounds=0, client_model_states={},
        sample_dispatch_identity=identity,
    )
    assert output.exists()
    assert not output.with_suffix(".pt.tmp").exists()
    state = _load_policy_checkpoint(tmp_path, torch.device("cpu"))
    restored = SampleDispatchIdentity.from_state_dict(state["sample_dispatch_identity"])
    assert restored.sequence == 1
    old_seed = identity.next_attempt().seed("worker-dp", 0, 1)
    new_seed = restored.next_attempt().seed("worker-dp", 0, 1)
    assert old_seed == new_seed  # same identity and sequence are reproducible
    assert restored.sequence == 2
    assert state["next_round"] == 0
    assert torch.equal(state["global_end_state"]["weight"], end.state_dict()["weight"])


def test_empty_sample_worker_dispatch_needs_no_attempt():
    """An empty worker batch must remain a safe no-op."""
    from types import SimpleNamespace
    from dynfed.fmnist_lenet5_dynamic import _run_client_training_tasks
    assert _run_client_training_tasks(
        tasks=[], train_config=SimpleNamespace(), selection=SimpleNamespace(privacy_unit="sample"),
        global_end=None, global_edge=None, client_model_states={}, x_train=None, y_train=None,
        device=None, model_name="", input_shape=(1, 1, 1), num_classes=1,
        np_rng=None, round_idx=0,
    ) == {}


def test_nonempty_sample_worker_dispatch_refuses_missing_attempt():
    """The worker refuses private work lacking a persisted attempt."""
    from types import SimpleNamespace
    import torch
    from dynfed.fmnist_lenet5_dynamic import _run_client_training_tasks
    with pytest.raises(RuntimeError, match="missing persisted attempt identity"):
        _run_client_training_tasks(
            tasks=[(1, None, None, 0)],
            train_config=SimpleNamespace(executor="serial"),
            selection=SimpleNamespace(privacy_unit="sample"),
            global_end=None, global_edge=None, client_model_states={}, x_train=None, y_train=None,
            device=torch.device("cpu"), model_name="", input_shape=(1, 1, 1), num_classes=1,
            np_rng=None, round_idx=0,
        )
