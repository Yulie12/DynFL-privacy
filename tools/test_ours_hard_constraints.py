"""Run from DynFL project root after installing patch."""
import random
import sys
from pathlib import Path
from dataclasses import replace
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dynfed.selection import Candidate, choose_candidate

BASE = dict(mode='LIIE', mechanisms={}, time=19.0, accuracy=.5,
            risk=.58, epsilon_used=0., communication_volume=0.,
            feasible_resource=True, feasible_privacy=True,
            feasible_risk=False, feasible_time=True, risk_hard_gate=True)

def candidate(**kwargs):
    return Candidate(**(BASE | kwargs))

def choose(cands):
    return choose_candidate(cands, policy='full_dynfl', rng=random.Random(42), require_feasible=False)

def main():
    bad_risk = candidate()
    assert not bad_risk.feasible
    assert choose([bad_risk]).mode == 'SKIP', 'Risk violation must SKIP'
    good = candidate(risk=.4, feasible_risk=True)
    assert good.feasible
    assert choose([bad_risk, good]).mode == 'LIIE', 'Feasible candidate must win'
    for flag in ('feasible_resource', 'feasible_memory', 'feasible_privacy',
                 'feasible_time', 'feasible_edge', 'feasible_cloud'):
        assert choose([replace(good, **{flag: False})]).mode == 'SKIP', f'{flag} violation must SKIP'
    print('[PASS] Hard risk, resource, memory, privacy, deadline, edge/cloud feasibility: SKIP when violated')
    print('[PASS] With a feasible candidate present, Ours can select it')
    print('[LIMIT] Unit test does not substitute for a full Ours 4-client smoke run')

if __name__ == '__main__':
    main()
