"""Finite-trial second-moment correction; no tensors leave the local process."""
import copy
import hashlib

import pytest
import torch

from dynfed.joint_calibration import JointCalibrationEntry
from experiments.independent_joint_proxy_capture import monte_carlo_score
from experiments.validate_joint_proxy_ranking import evaluate


def test_two_trial_bias_correction():
    # One client, two symmetric private updates around zero: biased plug-in
    # ||mean||^2 + sample variance is 2, corrected estimate is 1.
    entry = JointCalibrationEntry(torch.tensor([0.]), torch.tensor([0.]), 2., 2)
    out = monte_carlo_score({0: 'LIIC'}, {(0, 'LIIC'): entry},
                            torch.tensor([0.]), {0: 1.})
    assert out['mc_bias_squared'] == pytest.approx(0.)
    assert out['mc_variance_trace'] == pytest.approx(2.)
    assert out['mc_finite_sample_correction'] == pytest.approx(1.)
    assert out['mc_joint_score'] == pytest.approx(1.)


def test_different_sample_counts_corrected_per_client():
    a = JointCalibrationEntry(torch.tensor([1.]), torch.tensor([1.]), 2., 2)
    b = JointCalibrationEntry(torch.tensor([3.]), torch.tensor([0.]), 4., 4)
    out = monte_carlo_score({0: 'LIIC', 1: 'LIE'},
                            {(0, 'LIIC'): a, (1, 'LIE'): b},
                            torch.tensor([1.]), {0: 1., 1: 3.})
    variance = 2/16 + 9*4/16
    correction = (2/16)/2 + (9*4/16)/4
    assert out['mc_finite_sample_correction'] == pytest.approx(correction)
    assert out['mc_joint_score'] == pytest.approx((2.75-1.)**2 + variance - correction)


def _summary_v2():
    return {
        'format': 'dynfl_independent_joint_summary_v2',
        'mc_estimator': 'unbiased_conditional_second_moment_independent_clients_v2',
        'evaluation_data_independent_of_calibration': True,
        'evaluation_data_provenance': 'private_local_only',
        'conditional_cross_client_rng_independence': True,
        'evaluation_protocol': 'independent_official_test_paired_rng_v1',
        'calibration_sha256': hashlib.sha256(b'calibration').hexdigest(),
        'calibration_source_verified_as_train_subset': True,
        'scope': 'calibration-covered_subcohort_not_full_cohort',
        'measurement_boundary': 'global_relative_worker_updates_before_cloud_release_processing',
        'profiles': [
            {'profile_id': str(i), 'selector_joint_score': float(i),
             'mc_joint_score': float(i)+1., 'mc_bias_squared': float(i),
             'mc_variance_trace': 2., 'mc_finite_sample_correction': 1.,
             'client_count': 1, 'trials_per_cell': 2} for i in (1, 2, 3)
        ],
    }


def test_v2_summary_requires_and_checks_correction():
    original = _summary_v2()
    out = evaluate(original)
    assert out['spearman_rho'] == pytest.approx(1.)
    assert out['formula'] == 'finite_sample_corrected_joint_second_moment_v2'
    bad = copy.deepcopy(original)
    bad['profiles'][0].pop('mc_finite_sample_correction')
    with pytest.raises(ValueError, match='missing finite-sample'):
        evaluate(bad)
    bad = copy.deepcopy(original)
    bad['profiles'][0]['mc_finite_sample_correction'] = 0.
    with pytest.raises(ValueError, match='inconsistent'):
        evaluate(bad)


def test_old_v1_explicitly_legacy_and_still_readable():
    old = _summary_v2()
    old['format'] = 'dynfl_independent_joint_summary_v1'
    for row in old['profiles']:
        row.pop('mc_finite_sample_correction')
        row['mc_joint_score'] = row['mc_bias_squared'] + row['mc_variance_trace']
    out = evaluate(old)
    assert 'legacy_v1' in out['formula']
