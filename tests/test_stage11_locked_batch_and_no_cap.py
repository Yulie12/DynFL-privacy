"""Tests for locked 256-batch / no-hidden-5-step-cap *opt-in* experiment flags."""
import json
import unittest
from pathlib import Path

from dynfed.fmnist_lenet5_dynamic import Lenet5Config, _client_step_limit
from dynfed.selection import SelectionConfig

ROOT = Path(__file__).resolve().parents[1]


class Stage11LockedParameters(unittest.TestCase):
    def test_legacy_cap_unchanged(self):
        self.assertEqual(_client_step_limit(SelectionConfig()), 5)

    def test_new_experiment_uses_all_batches(self):
        self.assertIsNone(_client_step_limit(SelectionConfig(uncapped_local_batches=True)))
        self.assertEqual(Lenet5Config(train_batch_size=256, local_epochs=3).train_batch_size, 256)

    def test_online_staging_configs_lock_fields(self):
        names = (
            'stage11_fmnist_lenet5_online_5r_STAGING.json',
            'stage11_cifar10_resnet18_layer4_online_5r_STAGING.json',
        )
        for name in names:
            with self.subTest(config=name):
                obj = json.loads((ROOT / 'configs' / name).read_text(encoding='utf8'))
                self.assertEqual(obj['training']['rounds'], 5)
                self.assertEqual(obj['training']['batch_size'], 256)
                self.assertEqual(obj['training']['local_epochs'], 3)
                self.assertTrue(obj['training']['uncapped_local_batches'])
                self.assertEqual(obj['privacy']['clip_norm'], 3)
                self.assertEqual(obj['privacy']['sample_optimizer_clip_norm'], 3)
                self.assertEqual(obj['privacy']['accounting_mode'], 'rdp_auto')
                self.assertEqual(obj['he']['execution'], 'profiled')
                self.assertEqual(obj['system']['split_batch_size'], 256)
                self.assertEqual(obj['learning']['objective'], 'online_upper_bound')
                self.assertIn('required', obj['learning']['online_bound_manifest_path'])

    def test_formal_configs_declare_strict_online_objective(self):
        names = (
            'paper_v30_cifar10_resnet18.json',
            'paper_v31_cifar10_resnet18_fast_response.json',
            'paper_v30_fmnist_lenet5.json',
        )
        for name in names:
            with self.subTest(config=name):
                obj = json.loads((ROOT / 'configs' / name).read_text(encoding='utf8'))
                self.assertEqual(obj['learning']['objective'], 'online_upper_bound')
                self.assertIn(
                    'certificates/REAL_',
                    obj['learning']['online_bound_manifest_path'],
                )

    def test_missing_learning_objective_is_rejected(self):
        from experiments.run_paper_config import validate_config
        obj = json.loads(
            (ROOT / 'configs' / 'paper_v30_cifar10_resnet18.json').read_text(encoding='utf8')
        )
        obj.pop('learning')
        with self.assertRaisesRegex(ValueError, 'learning.objective'):
            validate_config(obj)

    def test_runner_cli_requires_explicit_learning_objective(self):
        import sys
        from unittest.mock import patch
        from experiments import run_fmnist_lenet5 as entry
        with patch.object(sys, 'argv', ['run_fmnist_lenet5.py']):
            with self.assertRaises(SystemExit):
                entry.parse_args()


if __name__ == '__main__':
    unittest.main()
