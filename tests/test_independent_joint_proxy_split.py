import numpy as np
import pytest
from experiments.independent_joint_proxy_split import (
    partition_official_test_indices, check_no_overlap, write_local_split_manifest,
)

def test_repeatable_disjoint_and_bounded():
    a = partition_official_test_indices(100, list(range(8)), seed=123, max_per_client=7)
    b = partition_official_test_indices(100, list(range(8)), seed=123, max_per_client=7)
    assert all(np.array_equal(a[k], b[k]) for k in a)
    assert all(0 < len(v) <= 7 for v in a.values())
    assert len(set(np.concatenate(list(a.values())).tolist())) == sum(map(len, a.values()))

def test_guards(tmp_path):
    with pytest.raises(ValueError):
        partition_official_test_indices(2, [0, 1, 2], seed=0)
    with pytest.raises(ValueError):
        check_no_overlap({0: np.array([1]), 1: np.array([1])})
    with pytest.raises(ValueError, match='cannot establish'):
        write_local_split_manifest(tmp_path / 'm.json', {0: np.array([0])}, seed=1,
                                   dataset='cifar10', calibration_dataset_split='unknown')
