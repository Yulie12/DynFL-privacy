import numpy as np

from dynfed.fmnist_lenet5_dynamic import _partition_clients_lenet5, _split_client_indices


def test_iid_preserves_previous_diagnostic_partition():
    y = np.arange(600) % 10
    expected = np.array_split(np.random.default_rng(42).permutation(len(y)), 100)
    actual = _partition_clients_lenet5(y, 100, 10, True, "iid", 42)
    assert all(np.array_equal(a, b) for a, b in zip(actual, expected))


def test_extreme_partition_and_holdout_are_disjoint_and_complete():
    y = np.repeat(np.arange(10), 1200)
    parts = _partition_clients_lenet5(y, 100, 10, False, "extreme_edge_label_skew", 42)
    train, held = _split_client_indices(parts, test_ratio=.2, seed=42)
    assert sum(map(len, train)) == 9600
    assert sum(map(len, held)) == 2400
    all_indices = np.concatenate(train + held)
    assert np.array_equal(np.sort(all_indices), np.arange(len(y)))
    for client, indices in enumerate(train):
        assert set(y[indices]) == {client % 10}
