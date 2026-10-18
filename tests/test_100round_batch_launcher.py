"""Launcher validation without GPU or private releases."""
from __future__ import annotations
import csv
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
from experiments import run_100round_batch as batch


class BatchLauncherTests(unittest.TestCase):
    def test_current_both_model_configs_preserve_locked_parameters(self):
        for name, path in batch.CONFIGS.items():
            batch.validate_config(json.loads(path.read_text(encoding='utf-8')), name)

    def test_zero_training_round_fails_closed(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / 'round_metrics.csv'
            with path.open('w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, ('actual_optimizer_steps', 'test_loss', 'test_accuracy'))
                writer.writeheader()
                for step in (2, 0, 3):
                    writer.writerow(dict(actual_optimizer_steps=step, test_loss=2.1,
                                         test_accuracy=.11))
            with self.assertRaisesRegex(RuntimeError, 'EMPTY TRAINING'):
                batch._read_round_metrics(path, target=3, prior=0)
            # Earlier rounds can be skipped when checking a RESUMED chunk,
            # but never silently ignored during the first chunk.
            self.assertEqual(batch._read_round_metrics(path, target=3, prior=2)[0], .11)

    def test_refuses_overwriting_uncommitted_chunk(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            chunk = root / 'runs' / 'lenet5' / 'seed_2026' / 'full_dynfl' / 'rounds_001_010'
            chunk.mkdir(parents=True)
            with self.assertRaisesRegex(RuntimeError, 'uncommitted'):
                batch._run_chunk(model='lenet5', seed=2026, policy='full_dynfl',
                                 state={'jobs': {}}, root=root,
                                 config=batch.CONFIGS['lenet5'], python='python')

    def test_precheck_invocation_has_100_horizon_and_10_chunk(self):
        with tempfile.TemporaryDirectory() as temp:
            observed = []
            def fake_run(argv, *, cwd, check):
                observed.append(argv)
            with mock.patch.object(batch.subprocess, 'run', side_effect=fake_run):
                batch.main(['--output-root', temp, '--seeds', '2026', '--dry-run'])
            self.assertEqual(len(observed), 2)
            for argv in observed:
                self.assertEqual(argv[argv.index('--rounds') + 1], '100')
                self.assertEqual(argv[argv.index('--max-new-rounds') + 1], '10')
                self.assertNotIn('--final-round-diagnostic-extra-epsilon', argv)

if __name__ == '__main__':
    unittest.main()
