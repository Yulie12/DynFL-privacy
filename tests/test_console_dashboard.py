from __future__ import annotations

from io import StringIO
from pathlib import Path

from dynfed.console_output import (
    TrainingConsoleRouter, managed_training_console, format_round_summary, format_run_header,
)


def test_compact_only_shows_training_dashboard(tmp_path: Path):
    terminal, detail = StringIO(), StringIO()
    router = TrainingConsoleRouter(terminal, detail, 'compact')
    router.write('startup private calibration blah\n')
    router.write('[RUN] policy=full_dynfl dataset=fmnist/lenet5 clients=100 ')
    router.write('rounds=101 P=5 lr=0.01 local_epochs=3 step_cap=5 device=cuda HE=profiled\n')
    router.write('lots of privacy noise\n')
    router.write('[ROUND] 001/101 full_dynfl acc=10.10% train_loss=2.3085 ')
    router.write('test_loss=2.3026 steps=500 skip=0 modes=LIIC:100 wall=8.7s privacy=sample HE=profiled\n')
    router.finish()
    value = terminal.getvalue()
    assert 'DynFL' in value and 'fmnist/lenet5' in value
    assert '10.10%' in value and '2.3085' in value and '2.3026' in value
    assert '500' in value and '8.7s' in value
    assert 'privacy noise' not in value and 'HE=profiled' not in value
    assert 'modes=LIIC' not in value
    assert 'lots of privacy noise' in detail.getvalue()
    assert '[ROUND]' in detail.getvalue()
    assert 'rounds=101 P=5' in detail.getvalue()


def test_quiet_does_not_show_rounds_and_logs_all():
    terminal, details = StringIO(), StringIO()
    router = TrainingConsoleRouter(terminal, details, 'quiet')
    router.write('[ROUND] 001/002 full_dynfl acc=10.10% train_loss=2.3000\n')
    router.finish()
    assert terminal.getvalue() == ''
    assert 'acc=10.10%' in details.getvalue()


def test_verbose_preserves_legacy_print():
    terminal, details = StringIO(), StringIO()
    router = TrainingConsoleRouter(terminal, details, 'verbose')
    router.write('HE profiled\n[ROUND] 001/002\n')
    router.finish()
    assert terminal.getvalue() == details.getvalue()


def test_context_routes_stdout_and_keeps_log(tmp_path, capsys):
    p = tmp_path / 'console_details.log'
    with managed_training_console('compact', p):
        print('hidden debug text')
        print(format_run_header(policy='full_dynfl', dataset='fmnist', model='lenet5',
                                clients=100, rounds=6, selection_period=5, local_epochs=3,
                                step_limit=5, learning_rate=0.01, device='cpu',
                                privacy_unit='sample', he_execution='profiled'))
        print(format_round_summary(policy='full_dynfl', round_number=1,
                                   total_rounds=6, accuracy=0.15, train_loss=2.2,
                                   test_loss=2.1, optimizer_steps=500,
                                   skipped_clients=0, modes={'LIIC': 100},
                                   round_wall_time_sec=5.0, privacy_unit='sample',
                                   he_execution='profiled'))
    terminal = capsys.readouterr().out
    assert 'Train loss' in terminal and '15.00%' in terminal
    assert 'hidden debug text' not in terminal
    assert 'hidden debug text' in p.read_text('utf-8')


def test_exception_is_not_swallowed(tmp_path):
    import pytest
    with pytest.raises(RuntimeError, match='fail'):
        with managed_training_console('quiet', tmp_path / 'full.log'):
            print('prior context')
            raise RuntimeError('fail')
    assert 'prior context' in (tmp_path / 'full.log').read_text('utf-8')
