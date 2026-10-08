"""Deterministic official-test split for *local-only* joint-proxy evaluation.

This module does not generate paired trajectories, selector scores, or claim
independence from unknown external calibration sets. It only creates an
index-disjoint client allocation from the official test-array index space.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np


def partition_official_test_indices(
    total: int, client_ids: Sequence[int], *, seed: int, max_per_client: int = 64
) -> dict[int, np.ndarray]:
    """Assign unique indices once across clients; reject empty partitions."""
    if total < 1 or not client_ids or max_per_client < 1:
        raise ValueError('total, client_ids and max_per_client must be positive/nonempty')
    ids = [int(c) for c in client_ids]
    if len(ids) != len(set(ids)) or any(c < 0 for c in ids):
        raise ValueError('client IDs must be unique nonnegative integers')
    if total < len(ids):
        raise ValueError('not enough test examples to allocate one per client')
    rng = np.random.default_rng(int(seed))
    permutation = rng.permutation(total)
    counts = [min(max_per_client, (total // len(ids)) + (i < total % len(ids))) for i in range(len(ids))]
    assigned = {}
    offset = 0
    for client_id, count in zip(sorted(ids), counts):
        assigned[client_id] = np.sort(permutation[offset:offset + count])
        offset += count
    check_no_overlap(assigned)
    return assigned


def check_no_overlap(groups: dict[int, np.ndarray]) -> None:
    arrays = [np.asarray(group, dtype=np.int64).reshape(-1) for group in groups.values()]
    all_indices = np.concatenate(arrays) if arrays else np.asarray([], dtype=np.int64)
    if np.any(all_indices < 0) or len(np.unique(all_indices)) != len(all_indices):
        raise ValueError('overlapping or invalid official-test indices')


def write_local_split_manifest(
    output: Path, groups: dict[int, np.ndarray], *, seed: int,
    dataset: str, calibration_dataset_split: str,
) -> None:
    """Write index hashes, not the sensitive data or feature vectors."""
    check_no_overlap(groups)
    if calibration_dataset_split != 'official_train_subset':
        raise ValueError('cannot establish disjointness from a calibration split not verified as official_train_subset')
    payload = {
        'status': 'test_indices_disjoint_from_official_train_calibration',
        'scope': 'sample_indices_only_not_independent_trajectories_or_model_states',
        'dataset': str(dataset), 'evaluation_split': 'official_test',
        'calibration_split': calibration_dataset_split, 'seed': int(seed),
        'clients': {
            str(cid): {'count': len(idx), 'indices_sha256': hashlib.sha256(
                np.asarray(idx, dtype='<i8').tobytes()).hexdigest()}
            for cid, idx in sorted(groups.items())
        },
        'privacy_warning': 'keep paired update trajectories and client-private data in a trusted local environment',
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + '\n', encoding='utf-8')
