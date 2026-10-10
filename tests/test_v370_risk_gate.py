"""Additive tests run alongside the existing 14 V370 admission tests."""
import unittest
from dataclasses import replace
import test_v370_admission as admission

m = admission.m

def candidate(**kwargs):
    args=dict(mode='LIIC', mechanisms={}, time=1., accuracy=.5, risk=.58,
              epsilon_used=0., communication_volume=0., feasible_resource=True,
              feasible_privacy=True, feasible_risk=False, feasible_time=True)
    args.update(kwargs)
    return m.Candidate(**args)

class TestV370RiskGate(unittest.TestCase):
    def test_v370_excludes_over_limit_candidate(self):
        c=candidate(risk_hard_gate=True)
        self.assertFalse(c.feasible)
        self.assertTrue(c.feasible_device)

    def test_choose_candidate_skips_risk_only_pool_without_flag(self):
        c=candidate(risk_hard_gate=True)
        chosen=m.choose_candidate([c], policy='ours_time_first', rng=__import__('random').Random(1), require_feasible=False)
        self.assertEqual(chosen.mode,'SKIP')

    def test_legacy_semantics_unchanged(self):
        c=candidate(risk_hard_gate=False)
        self.assertTrue(c.feasible)

    def test_legal_candidate_accepted_under_gate(self):
        c=candidate(risk_hard_gate=True, feasible_risk=True, risk=.1)
        self.assertTrue(c.feasible)

    def test_privacy_and_device_still_required(self):
        for field in ('feasible_resource','feasible_memory','feasible_privacy'):
            self.assertFalse(replace(candidate(risk_hard_gate=True, feasible_risk=True), **{field:False}).feasible)

    def test_choose_candidate_skips_risk_only_pool(self):
        c=candidate(risk_hard_gate=True)
        chosen=m.choose_candidate([c], policy='ours_time_first', rng=__import__('random').Random(1), require_feasible=True)
        self.assertEqual(chosen.mode,'SKIP')

if __name__=='__main__':
    unittest.main()
