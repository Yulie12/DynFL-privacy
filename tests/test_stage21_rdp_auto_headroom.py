"""Regression for last-round Sample-DP SKIP at the 1e-12 RDP boundary."""
from __future__ import annotations

import unittest

from dynfed.privacy import SamplePrivacyLedger
from dynfed.selection import _sample_rdp_auto_noise_with_headroom


class SampleRdpHeadroomTests(unittest.TestCase):
    @staticmethod
    def _project(ledger, count, sigma):
        return ledger.project(
            embedding_events=0,
            label_grad_events=0,
            optimizer_events=count,
            embedding_noise_multiplier=sigma,
            label_grad_noise_multiplier=sigma,
            optimizer_noise_multiplier=sigma,
        )

    @staticmethod
    def _charge(ledger, count, sigma):
        ledger.add(
            embedding_events=0,
            label_grad_events=0,
            optimizer_events=count,
            embedding_noise_multiplier=sigma,
            label_grad_noise_multiplier=sigma,
            optimizer_noise_multiplier=sigma,
        )

    def test_reproduces_original_fifth_round_numeric_skip(self):
        ledger = SamplePrivacyLedger(budget=8.0, delta=1e-5)
        for index in range(5):
            sigma = ledger.minimum_feasible_shared_noise(9 * (5 - index))
            proposal = self._project(ledger, 9, sigma)
            if index < 4:
                self._charge(ledger, 9, sigma)
            else:
                self.assertAlmostEqual(ledger.remaining_budget, 0.9421185448580438)
                self.assertGreater(proposal.epsilon_increment, ledger.remaining_budget + 1e-12)
                self.assertFalse(proposal.epsilon_increment <= ledger.remaining_budget + 1e-12)
                self.assertEqual(proposal.epsilon_after, 8.000000000001)

    def test_5_rounds_all_admitted_with_headroom_and_epsilon_eight(self):
        ledger = SamplePrivacyLedger(budget=8.0, delta=1e-5)
        for index in range(5):
            sigma = _sample_rdp_auto_noise_with_headroom(ledger, 9 * (5 - index))
            proposal = self._project(ledger, 9, sigma)
            self.assertTrue(ledger.can_apply(proposal))
            self.assertLessEqual(proposal.epsilon_increment, ledger.remaining_budget + 1e-12)
            self._charge(ledger, 9, sigma)
        self.assertLess(ledger.current_epsilon(), 8.0)
        self.assertGreater(ledger.remaining_budget, 1e-7)

    def test_dynamic_three_or_nine_steps_can_switch_without_fifth_skip(self):
        ledger = SamplePrivacyLedger(budget=8.0, delta=1e-5)
        modes = [3, 9, 9, 3, 9]
        for index, current_steps in enumerate(modes):
            future_count = current_steps * (len(modes) - index)
            before = ledger.state_dict()
            sigma = _sample_rdp_auto_noise_with_headroom(ledger, future_count)
            self.assertEqual(ledger.state_dict(), before)
            proposal = self._project(ledger, current_steps, sigma)
            self.assertTrue(ledger.can_apply(proposal))
            self.assertLessEqual(proposal.epsilon_increment, ledger.remaining_budget + 1e-12)
            self._charge(ledger, current_steps, sigma)
        self.assertLess(ledger.current_epsilon(), 8.0)

    def test_exhausted_ledger_fails_closed(self):
        ledger = SamplePrivacyLedger(budget=8.0, delta=1e-5)
        ledger.total._rdp_state = {k: float('inf') for k in ledger.total._rdp_state}
        with self.assertRaises(ValueError):
            _sample_rdp_auto_noise_with_headroom(ledger, 9)

    def test_non_positive_events_fail_closed(self):
        ledger = SamplePrivacyLedger(budget=8.0, delta=1e-5)
        with self.assertRaises(ValueError):
            _sample_rdp_auto_noise_with_headroom(ledger, 0)


if __name__ == '__main__':
    unittest.main()
