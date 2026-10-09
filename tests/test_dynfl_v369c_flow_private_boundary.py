"""V369C: private worker tensors do not enter the Flow timing simulator."""
from pathlib import Path

import pytest

from dynfed.flow_executor import ClientFlowInput, execute_mixed_round_flow
from dynfed.sample_flow_boundary import resolve_sample_admitted_updates


def _worker(update, *, finite=True):
    return {"finite": finite, "state_diff": update}


def _resolve(*, ids=(9,), diffs=(None,), counts=(3,), workers=None, expected=None):
    workers = {9: _worker({"end": {"weight": object()}})} if workers is None else workers
    expected = {9: 3} if expected is None else expected
    return resolve_sample_admitted_updates(
        selected_client_ids=ids,
        flow_state_diffs=diffs,
        flow_sample_counts=counts,
        worker_results=workers,
        train_sample_counts=expected,
    )


def test_metadata_only_flow_never_needs_private_model_delta():
    flow = execute_mixed_round_flow(
        round_idx=0,
        clients=[ClientFlowInput(
            client_id=9,
            edge_id=0,
            mode="LIC",
            candidate_time=0.1,
            estimated_local_time=1.0,
            measured_local_time=1.0,
            communication_volume=0.5,
            state_diff=None,
            sample_count=3,
        )],
        aggregation_fraction=1.0,
    )
    assert flow.selected_client_ids == [9]
    assert flow.state_diffs == [None]
    private = {"end": {"weight": object()}}
    result = _resolve(ids=flow.selected_client_ids, diffs=flow.state_diffs,
                      counts=flow.sample_counts, workers={9: _worker(private)})
    assert result[0] is private


@pytest.mark.parametrize("kwargs,pattern", [
    ({"diffs": ({"leak": 1},)}, "private state"),
    ({"ids": (9, 9), "diffs": (None, None), "counts": (3, 3)}, "duplicate"),
    ({"ids": (4,), "counts": (3,)}, "unknown/missing"),
    ({"ids": (9,), "counts": (4,)}, "sample-count"),
    ({"diffs": ()}, "result lengths"),
    ({"workers": {9: _worker({}, finite=False)}}, "non-finite"),
    ({"workers": {9: _worker({})}}, "missing/invalid"),
])
def test_flow_private_update_boundary_fails_closed(kwargs, pattern):
    with pytest.raises(RuntimeError, match=pattern):
        _resolve(**kwargs)


def test_runtime_flow_receives_only_metadata_and_resolves_after_admission():
    src = (Path(__file__).resolve().parents[1] /
           "dynfed" / "fmnist_lenet5_dynamic.py").read_text(encoding="utf-8-sig")
    start = src.index("            flow_inputs.append(")
    sim = src.index("        flow_result = execute_mixed_round_flow(", start)
    boundary = src.index("            admitted_flow_state_diffs = resolve_sample_admitted_updates(", sim)
    aggregated = src.index("        initial_admitted_updates = {", boundary)
    assert 'None if effective_selection.privacy_unit == "sample"' in src[start:sim]
    assert start < sim < boundary < aggregated
    assert src.count("                admitted_flow_state_diffs,\n") == 2
    assert "        _assert_sample_hierarchical_preflight(" in src
