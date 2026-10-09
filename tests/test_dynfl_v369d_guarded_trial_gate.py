"""Guarded multi-stage engineering trials remain opt-in and NOT a DP proof."""
import ast
from pathlib import Path
from types import SimpleNamespace

import pytest
from dynfed.sample_stage_event_audit import uniform_sample_stage_plan

SOURCE = (Path(__file__).resolve().parents[1] / 'dynfed' / 'fmnist_lenet5_dynamic.py').read_text()


def _gate():
    tree = ast.parse(SOURCE)
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                and n.name == '_assert_sample_hierarchical_preflight')
    ns = dict(Any=object, EDGE_CLOUD_MODES={'LIEIIIC', 'LIIEIIIC'},
              uniform_sample_stage_plan=uniform_sample_stage_plan)
    exec(compile(ast.Module(body=[node], type_ignores=[]), '<gate>', 'exec'), ns)
    return ns['_assert_sample_hierarchical_preflight']


def _candidate(mode='LIIEIIIC', n=6):
    return SimpleNamespace(mode=mode, sample_embedding_events=n,
                           sample_label_grad_events=n,
                           sample_optimizer_events=0)


def test_default_preflight_still_blocks_multistage():
    gate = _gate()
    specs = {'LIIEIIIC': SimpleNamespace(E_edge_loops=3)}
    with pytest.raises(RuntimeError, match='fail closed'):
        gate(privacy_unit='sample', train_tasks=[(4,_candidate(),None,0)], mode_specs=specs)


def test_opt_in_checks_divisibility_and_known_hierarchy_before_dispatch():
    gate = _gate()
    specs = {'LIIEIIIC': SimpleNamespace(E_edge_loops=3),
             'UNKNOWN': SimpleNamespace(E_edge_loops=3)}
    tasks = [(4,_candidate(),None,0)]
    assert gate(privacy_unit='sample', train_tasks=tasks, mode_specs=specs,
                allow_experimental_guarded_multistage=True) is None
    with pytest.raises(ValueError, match='not divisible'):
        gate(privacy_unit='sample', train_tasks=[(4,_candidate(n=5),None,0)],
             mode_specs=specs, allow_experimental_guarded_multistage=True)
    with pytest.raises(RuntimeError, match='recognized Edge/Cloud hierarchy'):
        gate(privacy_unit='sample', train_tasks=[(4,_candidate(mode='UNKNOWN'),None,0)],
             mode_specs=specs, allow_experimental_guarded_multistage=True)


def test_runtime_scope_is_explicit_and_profiler_only():
    preflight = SOURCE.index('        _assert_sample_hierarchical_preflight(')
    precharge = SOURCE.index('            sample_dispatch_charges = charge_sample_dispatch_before_worker(')
    assert preflight < precharge
    scope = SOURCE[SOURCE.index('        experimental_multistage_opt_in ='):preflight]
    assert 'DYNFL_EXPERIMENTAL_SAMPLE_MULTISTAGE' in scope
    assert 'effective_selection.mainline_fusion' in scope
    assert 'train_config.he_execution != "profiled"' in scope
    assert 'effective_selection.privacy_unit != "sample"' in scope
    assert 'allow_experimental_guarded_multistage=guarded_multistage_trial' in SOURCE
    assert 'end_to_end_dp=not_established' in SOURCE
