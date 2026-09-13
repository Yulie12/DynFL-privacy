import ast
from pathlib import Path

import numpy as np

from dynfed.fmnist_lenet5_dynamic import _partition_clients_lenet5, _split_client_indices


RUNNER = Path(__file__).resolve().parents[1] / "experiments" / "validate_edge_dp_trajectory.py"


def _load_split_builder():
    source = RUNNER.read_text(encoding="utf-8")
    tree = ast.parse(source)
    functions = [
        node for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        and node.name == "build_diagnostic_client_splits"
    ]
    assert len(functions) == 1, "diagnostic runner must define build_diagnostic_client_splits exactly once"
    module = ast.Module(body=functions, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {
        "np": np,
        "_partition_clients_lenet5": _partition_clients_lenet5,
        "_split_client_indices": _split_client_indices,
    }
    exec(compile(module, str(RUNNER), "exec"), namespace)
    return namespace["build_diagnostic_client_splits"]


def test_iid_zero_holdout_preserves_previous_diagnostic_partition():
    build_splits = _load_split_builder()
    y = np.arange(600) % 10
    expected = np.array_split(np.random.default_rng(42).permutation(len(y)), 100)
    full, train, held = build_splits(
        y, num_clients=100, num_edges=10, partition_mode="iid", holdout_ratio=0.0, seed=42
    )
    assert all(np.array_equal(a, b) for a, b in zip(full, expected))
    assert all(np.array_equal(a, b) for a, b in zip(train, expected))
    assert all(len(part) == 0 for part in held)


def test_extreme_partition_holdout_is_disjoint_and_complete():
    build_splits = _load_split_builder()
    y = np.repeat(np.arange(10), 1200)
    full, train, held = build_splits(
        y,
        num_clients=100,
        num_edges=10,
        partition_mode="extreme_edge_label_skew",
        holdout_ratio=0.2,
        seed=42,
    )
    assert sum(map(len, full)) == 12000
    assert sum(map(len, train)) == 9600
    assert sum(map(len, held)) == 2400
    all_indices = np.concatenate(train + held)
    assert np.array_equal(np.sort(all_indices), np.arange(len(y)))
    for client, indices in enumerate(train):
        assert set(y[indices]) == {client % 10}
        assert set(train[client]).isdisjoint(set(held[client]))


def test_runner_uses_holdout_cli_name_and_records_split_counts():
    source = RUNNER.read_text(encoding="utf-8")
    assert '"--client-holdout-ratio"' in source
    assert "args.local_test_ratio" not in source
    assert "client_full_counts=" in source
    assert "client_train_counts=" in source
    assert "client_holdout_counts=" in source
