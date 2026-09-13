import pytest

from dynfed.independent_release import IndependentReleaseAccount


def test_fixed_weights_and_release_count():
    account = IndependentReleaseAccount([1, 3], .1, 8, 1e-5, 2)
    assert account.sensitivity == pytest.approx(.15)
    first = account.reserve(0, [0, 1])
    assert account.reserve(1, [1, 0]) > first
    assert account.ledger.current_epsilon() <= 8 + 1e-8
    with pytest.raises(ValueError):
        account.reserve(2, [0, 1])


def test_no_replay_skips_or_cohort_changes():
    account = IndependentReleaseAccount([1, 1], .1, 8, 1e-5, 100)
    for round_index, ids in [(1, [0, 1]), (0, [0]), (0, [0, 0])]:
        with pytest.raises(ValueError):
            account.reserve(round_index, ids)
    assert account.next_round == 0
    account.reserve(0, [0, 1])
    with pytest.raises(ValueError):
        account.reserve(0, [0, 1])
