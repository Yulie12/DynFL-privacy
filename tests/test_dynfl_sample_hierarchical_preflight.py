"""Fail-closed safety gate: never silently under-audit multi-stage Sample DP."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


def _helper():
    path = Path(__file__).resolve().parents[1] / 'dynfed' / 'fmnist_lenet5_dynamic.py'
    tree = ast.parse(path.read_text(encoding='utf-8-sig'), filename=str(path))
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef)
             and n.name == '_assert_sample_hierarchical_preflight']
    assert len(nodes) == 1
    # Avoid importing training dependencies for this contract test.
    code = compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec')
    ns = {'Any': object}
    exec(code, ns)
    return ns['_assert_sample_hierarchical_preflight']


def test_preflight_fails_before_hierarchical_sample_dispatch():
    gate = _helper()
    specs = {'LIC': SimpleNamespace(E_edge_loops=1),
             'LIEIIIC': SimpleNamespace(E_edge_loops=3),
             'LIIEIIIC': SimpleNamespace(E_edge_loops=3)}
    tasks = [(1, SimpleNamespace(mode='LIC'), object(), 0),
             (2, SimpleNamespace(mode='LIEIIIC'), object(), 1)]
    with pytest.raises(RuntimeError, match='fail closed'):
        gate(privacy_unit='sample', train_tasks=tasks, mode_specs=specs)
    gate(privacy_unit='client', train_tasks=tasks, mode_specs=specs)
    gate(privacy_unit='sample', train_tasks=tasks[:1], mode_specs=specs)


def test_preflight_integrated_before_calibration_capture_and_worker_dispatch():
    path = Path(__file__).resolve().parents[1] / 'dynfed' / 'fmnist_lenet5_dynamic.py'
    source = path.read_text(encoding='utf-8-sig')
    call = source.index('        _assert_sample_hierarchical_preflight(\n')
    capture = source.index('        calibration_capture_stats:')
    dispatch = source.index('        worker_results = _run_client_training_tasks(')
    assert call < capture < dispatch
