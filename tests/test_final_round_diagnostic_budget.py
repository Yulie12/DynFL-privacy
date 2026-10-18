"""Final-round epsilon extension is explicit, audited, and never erases RDP."""
import math
import unittest

from dynfed.fmnist_lenet5_dynamic import (
    _apply_final_round_diagnostic_sample_budget_extension,
    Lenet5Config,
    _run_lenet5_policy,
)
from dynfed.privacy import SamplePrivacyLedger


class TestFinalRoundDiagnosticBudget(unittest.TestCase):
    def setUp(self):
        self.ledger = SamplePrivacyLedger(budget=8.0, delta=1e-5)
        self.ledgers = {0: self.ledger}
        self.remaining = {0: self.ledger.remaining_budget}

    def change(self, idx, extra=2.0, total=5):
        return _apply_final_round_diagnostic_sample_budget_extension(
            privacy_ledgers=self.ledgers,
            remaining_epsilon=self.remaining,
            base_budget=8.0,
            extra_epsilon=extra,
            round_idx=idx,
            total_rounds=total,
        )

    def test_default_never_changes_budget(self):
        self.assertFalse(self.change(4, extra=0.0))
        self.assertEqual(self.ledger.total.budget, 8.0)

    def test_only_final_round_changes_budget(self):
        self.assertFalse(self.change(3))
        self.assertEqual(self.ledger.total.budget, 8.0)
        self.assertTrue(self.change(4))
        self.assertEqual(self.ledger.total.budget, 10.0)
        self.assertAlmostEqual(self.remaining[0], self.ledger.remaining_budget)

    def test_no_accountant_rollback_and_idempotence(self):
        projection = self.ledger.add(
            embedding_events=0,
            label_grad_events=0,
            optimizer_events=1,
            embedding_noise_multiplier=4.0,
            label_grad_noise_multiplier=4.0,
            optimizer_noise_multiplier=4.0,
        )
        spent = self.ledger.current_epsilon()
        self.assertAlmostEqual(projection.epsilon_after, spent)
        self.assertTrue(self.change(4))
        self.assertAlmostEqual(self.ledger.current_epsilon(), spent)
        self.assertTrue(self.change(4))
        self.assertAlmostEqual(self.ledger.current_epsilon(), spent)

    def test_wrong_budget_refuses_change(self):
        self.ledger.total.budget = 9.0
        with self.assertRaises(RuntimeError):
            self.change(4)

    def test_non_sample_ledger_refused(self):
        self.ledgers[0] = object()
        with self.assertRaises(TypeError):
            self.change(4)

    def test_bad_extension_refused(self):
        for value in (float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                self.change(4, extra=value)

    def test_policy_binds_diagnostic_epsilon_locally(self):
        # Regression: previously _run_lenet5_policy referenced an undefined
        # module global and crashed before the very first round.
        self.assertIn("extra_epsilon", _run_lenet5_policy.__code__.co_varnames)

    def test_config_default_safe(self):
        self.assertEqual(Lenet5Config().final_round_diagnostic_extra_epsilon, 0.0)

if __name__ == "__main__":
    unittest.main()
