import json
from pathlib import Path

import numpy as np
import pytest

from dynfed.he_backend import check_he_backend


@pytest.fixture
def custodian():
    status = check_he_backend("seal")
    if not status.available:
        pytest.skip(status.detail)
    from experiments.trusted_edge_custodian import TrustedEdgeCustodian
    with TrustedEdgeCustodian([{"edge": 0, "cloud_weight": 0.25},
                               {"edge": 1, "cloud_weight": 0.75}], release_limit=2) as value:
        yield value


def packets(size=4):
    return {0: np.linspace(-0.1, 0.1, size), 1: np.linspace(0.2, -0.1, size)}


def test_separate_cloud_process_real_he_and_replay_rejection(custodian):
    inputs = packets(5130)
    result, metrics = custodian.aggregate(inputs, 0)
    np.testing.assert_allclose(result, inputs[0] * 0.25 + inputs[1] * 0.75, atol=1e-5, rtol=0)
    assert metrics["cloud_pid"] != metrics["custodian_pid"]
    assert not metrics["cloud_secret_key_transmitted"]
    assert metrics["key_isolation_enforced"]
    assert metrics["custodian_can_bypass_policy"]
    assert metrics["encrypted_parameter_values"] == 10260
    with pytest.raises(ValueError, match="already released"):
        custodian.release()
    with pytest.raises(ValueError, match="replayed"):
        custodian.prepare(inputs, 0)


def test_round_and_cohort_cannot_be_redefined_per_request(custodian):
    with pytest.raises(ValueError, match="Cohort"):
        custodian.prepare({0: np.ones(4)}, 0)
    with pytest.raises(ValueError, match="out of order"):
        custodian.prepare(packets(), 1)
    custodian.prepare(packets(), 0)
    with pytest.raises(ValueError, match="pending"):
        custodian.prepare(packets(), 0)


def test_single_ciphertext_substitution_rejected_before_decryption(custodian):
    directory = custodian.prepare(packets(), 0)
    custodian.run_cloud()
    (directory / "output_0.ct").write_bytes((directory / "input_0_0.ct").read_bytes())
    with pytest.raises(ValueError, match="authorized full sum"):
        custodian.release()
    assert custodian.next_round == 0


@pytest.mark.parametrize("field,new", [("round", 5), ("task_id", "other"),
                                       ("authorization", "other"), ("outputs", ["../input.ct"])])
def test_forged_response_metadata_rejected(custodian, field, new):
    directory = custodian.prepare(packets(), 0)
    custodian.run_cloud()
    path = directory / "response.json"
    response = json.loads(path.read_text())
    response[field] = new
    path.write_text(json.dumps(response))
    with pytest.raises(ValueError, match="mismatch"):
        custodian.release()


def test_cloud_manifest_tampering_rejected(custodian):
    directory = custodian.prepare(packets(), 0)
    path = directory / "request.json"
    request = json.loads(path.read_text())
    request["groups"][0][1] = 1.0
    path.write_text(json.dumps(request))
    custodian.run_cloud()
    with pytest.raises(ValueError, match="Manifest"):
        custodian.release()


def test_stale_response_and_release_horizon_rejected(custodian):
    first = custodian.prepare(packets(), 0)
    custodian.run_cloud()
    old_response = (first / "response.json").read_bytes()
    custodian.release()
    second = custodian.prepare(packets(), 1)
    custodian.run_cloud()
    (second / "response.json").write_bytes(old_response)
    with pytest.raises(ValueError, match="mismatch"):
        custodian.release()
    custodian.run_cloud()
    custodian.release()
    with pytest.raises(ValueError, match="release limit"):
        custodian.prepare(packets(), 2)


def test_cloud_worker_has_no_decryption_or_key_loading_code():
    source = (Path(__file__).resolve().parents[1] / "experiments/he_cloud_worker.py").read_text()
    for forbidden in ("SecretKey(", "Decryptor(", "KeyGenerator(", "trusted_edge_custodian"):
        assert forbidden not in source


def test_ciphertext_tampering_is_rejected_by_cloud(custodian):
    import subprocess
    directory = custodian.prepare(packets(), 0)
    (directory / "input_0_0.ct").write_bytes(b"invalid_public_test_ciphertext")
    with pytest.raises(subprocess.CalledProcessError):
        custodian.run_cloud()
    assert custodian.next_round == 0


def test_dimension_change_rejected_between_rounds(custodian):
    custodian.aggregate(packets(), 0)
    with pytest.raises(ValueError, match="layout changed"):
        custodian.prepare(packets(5), 1)


@pytest.mark.parametrize("arguments", [
    ["--dp-release", "edge_local", "--he-custody", "trusted_edge"],
    ["--dp-release", "edge_local", "--methods", "distributed_dp_he"],
    ["--methods", "clip_only", "clip_only"],
])
def test_incompatible_cli_rejected_before_training(arguments):
    import subprocess
    import sys
    script = Path(__file__).resolve().parents[1] / "experiments/validate_edge_dp_trajectory.py"
    result = subprocess.run([sys.executable, str(script), *arguments], capture_output=True, text=True,
                            timeout=30)
    assert result.returncode == 2
    assert "error:" in result.stderr
    assert "Files already downloaded" not in result.stdout
