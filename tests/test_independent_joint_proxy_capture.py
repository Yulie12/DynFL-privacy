import hashlib
import pytest
import torch
from dynfed.joint_calibration import JointCalibrationEntry
from experiments.independent_joint_proxy_capture import monte_carlo_score
from experiments.validate_joint_proxy_ranking import evaluate


def _summary():
    return {
        'format': 'dynfl_independent_joint_summary_v1',
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
             'mc_joint_score': float(i+1), 'mc_bias_squared': float(i),
             'mc_variance_trace': 1.0, 'client_count': 2,
             'trials_per_cell': 2} for i in (1, 2, 3)
        ],
    }


def test_local_summary_spearman():
    result = evaluate(_summary())
    assert result['spearman_rho'] == pytest.approx(1.0)
    assert result['scope'] == 'calibration-covered_subcohort_not_full_cohort'


def test_summary_rejects_inconsistent_values_and_provenance():
    data = _summary()
    data['profiles'][0]['mc_bias_squared'] = -2.0
    with pytest.raises(ValueError, match='nonfinite/negative'):
        evaluate(data)
    data = _summary()
    data['profiles'][0]['mc_joint_score'] = 999.
    with pytest.raises(ValueError, match='inconsistent'):
        evaluate(data)
    data = _summary()
    data['calibration_source_verified_as_train_subset'] = False
    with pytest.raises(ValueError, match='provenance'):
        evaluate(data)


def test_monte_carlo_weighted_moments():
    a = JointCalibrationEntry(torch.tensor([1.]), torch.tensor([1.]), 2., 3)
    b = JointCalibrationEntry(torch.tensor([3.]), torch.tensor([0.]), 4., 3)
    result = monte_carlo_score({0: 'LIIC', 1: 'LIE'}, {(0, 'LIIC'): a, (1, 'LIE'): b},
                               torch.tensor([1.]), {0: 1., 1: 3.})
    assert result['mc_bias_squared'] == pytest.approx((2.75-1.)**2)
    assert result['mc_variance_trace'] == pytest.approx(2/16+9*4/16)
    assert result['mc_finite_sample_correction'] == pytest.approx(result['mc_variance_trace'] / 3)
    assert result['mc_joint_score'] == pytest.approx(
        result['mc_bias_squared'] + result['mc_variance_trace']
        - result['mc_finite_sample_correction'])
