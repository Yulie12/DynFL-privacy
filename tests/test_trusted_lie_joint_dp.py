"""Reference checks: opt-in trusted LIE; never claims full-system accounting."""
import numpy as np
import pytest
import torch
import torch.nn as nn

from dynfed.sample_dp import clip_and_aggregate_per_sample_grads
from dynfed.split_learning import build_split_pair, split_local_train_lenet5
from dynfed.trusted_split_sample_dp import joint_per_sample_grads


@pytest.fixture(autouse=True)
def one_thread():
    original = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(original)


def make_case():
    device = torch.device("cpu")
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(17)
        end, edge = build_split_pair("lenet5", device)
        states = (
            {k: v.detach().clone() for k, v in end.state_dict().items()},
            {k: v.detach().clone() for k, v in edge.state_dict().items()},
        )
    x = np.random.default_rng(16).normal(size=(2, 1, 28, 28)).astype("float32")
    y = np.array([0, 1], dtype="int64")
    return device, states, x, y


def run(mode="LIE", **overrides):
    device, (end_state, edge_state), x, y = make_case()
    args = dict(
        mode=mode, global_end_state=end_state, global_edge_state=edge_state,
        x=x, y=y, epochs=1, lr=0.015, device=device,
        model_name="lenet5", input_shape=(1, 28, 28), num_classes=10,
        mechanisms={"emb": "trusted", "grad": "trusted"},
        privacy_unit="sample", trusted_split_joint_sample_dp=True,
        trusted_edge=True, sample_optimizer_noise_multiplier=0.75,
        sample_optimizer_clip_norm=1.0, training_seed=42,
        dp_rng=np.random.default_rng(333),
        training_diagnostics={},
    )
    args.update(overrides)
    original_end = {k: v.clone() for k, v in end_state.items()}
    original_edge = {k: v.clone() for k, v in edge_state.items()}
    result = split_local_train_lenet5(**args)
    for key in original_end:
        assert torch.equal(end_state[key], original_end[key])
    for key in original_edge:
        assert torch.equal(edge_state[key], original_edge[key])
    return result, args["training_diagnostics"]


@pytest.mark.parametrize("overrides, error", [
    ({"trusted_edge": False}, "trusted_edge=True"),
    ({"privacy_unit": "client"}, "privacy_unit='sample'"),
    ({"mechanisms": {"emb": "none", "grad": "trusted"}}, "emb=trusted"),
    ({"mechanisms": {"emb": "trusted", "grad": "dp"}}, "grad=trusted"),
    ({"sample_optimizer_noise_multiplier": None}, "positive finite"),
    ({"sample_optimizer_noise_multiplier": 0.0}, "positive finite"),
    ({"sample_optimizer_noise_multiplier": float("nan")}, "positive finite"),
    ({"sample_embedding_noise_multiplier": 0.5}, "must not charge"),
    ({"sample_optimizer_clip_norm": 0.0}, "must be positive"),
])
def test_opt_in_fail_closed(overrides, error):
    with pytest.raises(ValueError, match=error):
        run(**overrides)


def test_trusted_lie_other_modes_rejected():
    with pytest.raises(ValueError, match="LIE only"):
        run(mode="LIC")


def test_trusted_edge_flag_cannot_enable_legacy_path():
    with pytest.raises(ValueError, match="requires trusted_split_joint_sample_dp"):
        run(trusted_split_joint_sample_dp=False)


def test_joint_gradients_match_manual_two_module_autograd_and_shared_clip():
    torch.manual_seed(13)
    end = nn.Linear(3, 4)
    edge = nn.Linear(4, 2)
    x = torch.tensor([[1., 2., 3.], [-2., 1., 0.]])
    y = torch.tensor([0, 1])
    got = joint_per_sample_grads(end, edge, x, y)
    assert set(got) == {"end.weight", "end.bias", "edge.weight", "edge.bias"}
    for i in range(2):
        loss = nn.functional.cross_entropy(edge(end(x[i:i+1])), y[i:i+1])
        grads = torch.autograd.grad(loss, tuple(end.parameters()) + tuple(edge.parameters()))
        for name, tensor in zip(got, grads):
            torch.testing.assert_close(got[name][i], tensor)

    # Check one global sample norm covers BOTH model portions.
    clean, _ = clip_and_aggregate_per_sample_grads(got, clip_norm=1e6, noise_multiplier=0)
    clipped, info = clip_and_aggregate_per_sample_grads(got, clip_norm=0.1, noise_multiplier=0)
    assert info.clipped_sample_count == 2
    ratio = [torch.linalg.vector_norm(clipped[k]) / torch.linalg.vector_norm(clean[k])
             for k in got if torch.linalg.vector_norm(clean[k]) > 1e-9]
    assert any(float(r) < 1 for r in ratio)


def test_joint_path_updates_both_halves_reproducibly_without_link_noise():
    update, diag = run()
    repeated, repeated_diag = run()
    assert diag["actual_local_batches"] == 1
    assert diag["actual_optimizer_steps"] == 1
    assert diag["sample_dp_optimizer_steps"] == 1
    assert diag["trusted_split_joint_dp_steps"] == 1
    assert diag["sample_dp_sample_count"] == 2
    assert diag["feature_dp_release_batches"] == 0
    assert diag["sample_label_grad_dp_release_batches"] == 0
    assert diag["feature_dp_sample_count"] == 0
    assert diag == repeated_diag
    for side in ("end", "edge"):
        assert len(update[side]) > 0
        assert any(torch.count_nonzero(t).item() for t in update[side].values())
        for name in update[side]:
            torch.testing.assert_close(update[side][name], repeated[side][name], rtol=0, atol=0)


def test_training_batchnorm_rejected():
    end = nn.Sequential(nn.Linear(3, 3), nn.BatchNorm1d(3))
    edge = nn.Linear(3, 2)
    with pytest.raises(ValueError, match="BatchNorm"):
        joint_per_sample_grads(end, edge, torch.randn(2, 3), torch.tensor([0, 1]))


def test_one_step_update_equals_single_joint_dp_aggregation():
    # This catches independent clipping/noising of End and Edge even if both
    # halves happen to change during training.
    device, (end_state, edge_state), x, y = make_case()
    end, edge = build_split_pair("lenet5", device)
    end.load_state_dict(end_state)
    edge.load_state_dict(edge_state)
    per = joint_per_sample_grads(
        end, edge, torch.from_numpy(x), torch.from_numpy(y)
    )
    gaussian_seed = int(np.random.default_rng(333).integers(0, np.iinfo(np.int64).max))
    noise = torch.Generator(device=device).manual_seed(gaussian_seed)
    protected, _ = clip_and_aggregate_per_sample_grads(
        per, clip_norm=1.0, noise_multiplier=0.75, generator=noise
    )
    actual, _ = run()
    for side in ("end", "edge"):
        for name, param in (end if side == "end" else edge).named_parameters():
            if not param.requires_grad:
                continue
            expected_delta = -0.015 * protected[f"{side}.{name}"]
            torch.testing.assert_close(
                actual[side][name], expected_delta, atol=2e-7, rtol=1e-4
            )


def test_cached_models_do_not_retain_previous_private_training_state():
    device, (end_state, edge_state), x, y = make_case()
    cache = {}

    def train(labels):
        diag = {}
        return split_local_train_lenet5(
            "LIE", end_state, edge_state, x, np.array(labels),
            1, .015, device, "lenet5", (1, 28, 28), 10,
            mechanisms={"emb": "trusted", "grad": "trusted"},
            privacy_unit="sample", trusted_split_joint_sample_dp=True,
            trusted_edge=True, sample_optimizer_noise_multiplier=.75,
            sample_optimizer_clip_norm=1.0, training_seed=42,
            dp_rng=np.random.default_rng(333), training_diagnostics=diag,
            model_cache=cache,
        )

    baseline = train([0, 1])
    train([8, 9])
    again = train([0, 1])
    for side in ("end", "edge"):
        for key in baseline[side]:
            torch.testing.assert_close(baseline[side][key], again[side][key], rtol=0, atol=0)
