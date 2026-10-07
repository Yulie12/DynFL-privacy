import torch

from dynfed.sample_dp import clip_and_aggregate_per_sample_grads


def test_joint_per_sample_clipping_across_parameters():
    per_sample_grads = {
        "w": torch.tensor([[3.0], [0.0]]),
        "b": torch.tensor([[4.0], [2.0]]),
    }

    aggregated, diagnostics = clip_and_aggregate_per_sample_grads(
        per_sample_grads,
        clip_norm=4.0,
        noise_multiplier=0.0,
    )

    # Sample 0 has joint norm sqrt(3^2 + 4^2) = 5, so both parameter
    # gradients must use the same clipping scale 4/5 = 0.8.
    #
    # sample 0 after clipping: w=2.4, b=3.2
    # sample 1 unchanged:       w=0.0, b=2.0
    #
    # minibatch means:          w=1.2, b=2.6
    torch.testing.assert_close(
        aggregated["w"],
        torch.tensor([1.2]),
        rtol=0.0,
        atol=1e-6,
    )
    torch.testing.assert_close(
        aggregated["b"],
        torch.tensor([2.6]),
        rtol=0.0,
        atol=1e-6,
    )

    assert diagnostics.sample_count == 2
    assert diagnostics.clipped_sample_count == 1
    assert abs(diagnostics.max_raw_grad_norm - 5.0) < 1e-9
    assert abs(diagnostics.max_clipped_grad_norm - 4.0) < 1e-9


def test_no_clipping_no_noise_matches_batch_mean():
    per_sample_grads = {
        "w": torch.tensor(
            [
                [1.0, 2.0],
                [3.0, 4.0],
                [-1.0, 2.0],
            ]
        ),
        "b": torch.tensor(
            [
                [0.5],
                [-0.5],
                [1.0],
            ]
        ),
    }

    aggregated, diagnostics = clip_and_aggregate_per_sample_grads(
        per_sample_grads,
        clip_norm=100.0,
        noise_multiplier=0.0,
    )

    torch.testing.assert_close(
        aggregated["w"],
        per_sample_grads["w"].mean(dim=0),
        rtol=0.0,
        atol=1e-7,
    )
    torch.testing.assert_close(
        aggregated["b"],
        per_sample_grads["b"].mean(dim=0),
        rtol=0.0,
        atol=1e-7,
    )

    assert diagnostics.sample_count == 3
    assert diagnostics.clipped_sample_count == 0


def test_gaussian_noise_is_reproducible_with_fixed_seed():
    per_sample_grads = {
        "w": torch.zeros(4, 3),
    }

    generator_a = torch.Generator().manual_seed(12345)
    generator_b = torch.Generator().manual_seed(12345)

    aggregated_a, _ = clip_and_aggregate_per_sample_grads(
        per_sample_grads,
        clip_norm=2.0,
        noise_multiplier=0.75,
        generator=generator_a,
    )
    aggregated_b, _ = clip_and_aggregate_per_sample_grads(
        per_sample_grads,
        clip_norm=2.0,
        noise_multiplier=0.75,
        generator=generator_b,
    )

    torch.testing.assert_close(
        aggregated_a["w"],
        aggregated_b["w"],
        rtol=0.0,
        atol=0.0,
    )


def test_noise_scaling_matches_replacement_adjacency_formula():
    batch_size = 4
    clip_norm = 2.0
    noise_multiplier = 0.75
    seed = 24680

    per_sample_grads = {
        "w": torch.zeros(batch_size, 3),
    }

    actual_generator = torch.Generator().manual_seed(seed)
    expected_generator = torch.Generator().manual_seed(seed)

    aggregated, _ = clip_and_aggregate_per_sample_grads(
        per_sample_grads,
        clip_norm=clip_norm,
        noise_multiplier=noise_multiplier,
        generator=actual_generator,
    )

    # Replacement adjacency:
    #
    # sensitivity of clipped gradient sum = 2C
    # sum noise std = 2 * C * sigma
    # mean noise std = 2 * C * sigma / B
    expected_output_std = (
        2.0 * clip_norm * noise_multiplier / float(batch_size)
    )

    expected = torch.randn(
        (3,),
        generator=expected_generator,
    ) * expected_output_std

    torch.testing.assert_close(
        aggregated["w"],
        expected,
        rtol=0.0,
        atol=1e-7,
    )

def test_reference_per_sample_grads_match_batch_cross_entropy_gradient():
    from dynfed.sample_dp import per_sample_grads_reference

    torch.manual_seed(13579)

    model = torch.nn.Linear(3, 2, bias=True)

    inputs = torch.tensor(
        [
            [1.0, 0.0, -1.0],
            [0.5, 2.0, 1.0],
            [-1.0, 1.5, 0.25],
            [2.0, -0.5, 0.75],
        ],
        dtype=torch.float32,
    )
    targets = torch.tensor([0, 1, 1, 0], dtype=torch.long)

    per_sample_grads = per_sample_grads_reference(
        model,
        inputs,
        targets,
        torch.nn.functional.cross_entropy,
    )

    aggregated, diagnostics = clip_and_aggregate_per_sample_grads(
        per_sample_grads,
        clip_norm=100.0,
        noise_multiplier=0.0,
    )

    model.zero_grad(set_to_none=True)
    batch_loss = torch.nn.functional.cross_entropy(
        model(inputs),
        targets,
        reduction="mean",
    )
    batch_loss.backward()

    expected = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    assert set(aggregated) == set(expected)

    for name in expected:
        torch.testing.assert_close(
            aggregated[name],
            expected[name],
            rtol=1e-5,
            atol=1e-6,
        )

    assert diagnostics.sample_count == 4
    assert diagnostics.clipped_sample_count == 0

def test_sample_privacy_ledger_composes_all_mechanisms_in_one_rdp_state():
    from dynfed.privacy import (
        SamplePrivacyLedger,
        epsilon_from_rdp,
        gaussian_rdp,
    )

    delta = 1e-5
    ledger = SamplePrivacyLedger(budget=100.0, delta=delta)

    embedding_events = 2
    label_grad_events = 3
    optimizer_events = 5

    embedding_sigma = 3.0
    label_grad_sigma = 4.0
    optimizer_sigma = 5.0

    projection = ledger.project(
        embedding_events=embedding_events,
        label_grad_events=label_grad_events,
        optimizer_events=optimizer_events,
        embedding_noise_multiplier=embedding_sigma,
        label_grad_noise_multiplier=label_grad_sigma,
        optimizer_noise_multiplier=optimizer_sigma,
    )

    embedding_rdp = gaussian_rdp(
        embedding_sigma,
        embedding_events,
    )
    label_grad_rdp = gaussian_rdp(
        label_grad_sigma,
        label_grad_events,
    )
    optimizer_rdp = gaussian_rdp(
        optimizer_sigma,
        optimizer_events,
    )

    expected_rdp = {
        order: (
            embedding_rdp[order]
            + label_grad_rdp[order]
            + optimizer_rdp[order]
        )
        for order in embedding_rdp
    }
    expected_epsilon = epsilon_from_rdp(expected_rdp, delta)

    assert abs(projection.epsilon_before - 0.0) < 1e-12
    assert abs(projection.epsilon_after - expected_epsilon) < 1e-12
    assert projection.embedding_events == embedding_events
    assert projection.label_grad_events == label_grad_events
    assert projection.optimizer_events == optimizer_events


def test_sample_privacy_ledger_project_is_non_mutating_and_add_commits():
    from dynfed.privacy import SamplePrivacyLedger

    ledger = SamplePrivacyLedger(budget=100.0, delta=1e-5)

    projection = ledger.project(
        embedding_events=1,
        label_grad_events=2,
        optimizer_events=3,
        embedding_noise_multiplier=4.0,
        label_grad_noise_multiplier=5.0,
        optimizer_noise_multiplier=6.0,
    )

    # Projection must not mutate the live ledger.
    assert ledger.current_epsilon() == 0.0
    assert ledger.embedding_events == 0
    assert ledger.label_grad_events == 0
    assert ledger.optimizer_events == 0

    committed = ledger.add(
        embedding_events=1,
        label_grad_events=2,
        optimizer_events=3,
        embedding_noise_multiplier=4.0,
        label_grad_noise_multiplier=5.0,
        optimizer_noise_multiplier=6.0,
    )

    assert abs(
        committed.epsilon_after - projection.epsilon_after
    ) < 1e-12
    assert abs(
        ledger.current_epsilon() - projection.epsilon_after
    ) < 1e-12

    assert ledger.embedding_events == 1
    assert ledger.label_grad_events == 2
    assert ledger.optimizer_events == 3


def test_sample_privacy_ledger_state_round_trip_preserves_accounting():
    from dynfed.privacy import SamplePrivacyLedger

    ledger = SamplePrivacyLedger(budget=100.0, delta=1e-5)

    ledger.add(
        embedding_events=2,
        label_grad_events=1,
        optimizer_events=4,
        embedding_noise_multiplier=3.5,
        label_grad_noise_multiplier=4.5,
        optimizer_noise_multiplier=5.5,
    )

    restored = SamplePrivacyLedger.from_state_dict(
        ledger.state_dict()
    )

    assert abs(
        restored.current_epsilon() - ledger.current_epsilon()
    ) < 1e-12
    assert abs(
        restored.remaining_budget - ledger.remaining_budget
    ) < 1e-12

    assert restored.embedding_events == ledger.embedding_events
    assert restored.label_grad_events == ledger.label_grad_events
    assert restored.optimizer_events == ledger.optimizer_events

    assert restored.state_dict() == ledger.state_dict()

def test_sample_privacy_ledger_minimum_shared_noise_matches_empty_calibration():
    from dynfed.privacy import (
        SamplePrivacyLedger,
        calibrate_gaussian_noise,
    )

    budget = 8.0
    delta = 1e-5
    event_count = 100

    ledger = SamplePrivacyLedger(
        budget=budget,
        delta=delta,
    )

    actual = ledger.minimum_feasible_shared_noise(
        event_count
    )
    expected = calibrate_gaussian_noise(
        budget,
        delta,
        event_count,
    )

    assert abs(actual - expected) < 1e-12


def test_sample_privacy_ledger_minimum_shared_noise_is_non_mutating():
    from dynfed.privacy import SamplePrivacyLedger

    ledger = SamplePrivacyLedger(
        budget=8.0,
        delta=1e-5,
    )

    epsilon_before = ledger.current_epsilon()

    sigma = ledger.minimum_feasible_shared_noise(50)

    assert sigma > 0.0
    assert ledger.current_epsilon() == epsilon_before
    assert ledger.embedding_events == 0
    assert ledger.label_grad_events == 0
    assert ledger.optimizer_events == 0


def test_sample_privacy_ledger_spent_budget_requires_more_future_noise():
    from dynfed.privacy import SamplePrivacyLedger

    ledger = SamplePrivacyLedger(
        budget=8.0,
        delta=1e-5,
    )

    future_events = 50
    sigma_before = ledger.minimum_feasible_shared_noise(
        future_events
    )

    ledger.add(
        embedding_events=0,
        label_grad_events=0,
        optimizer_events=1,
        embedding_noise_multiplier=20.0,
        label_grad_noise_multiplier=20.0,
        optimizer_noise_multiplier=20.0,
    )

    sigma_after = ledger.minimum_feasible_shared_noise(
        future_events
    )

    assert sigma_after > sigma_before

    projection = ledger.project(
        embedding_events=0,
        label_grad_events=0,
        optimizer_events=future_events,
        embedding_noise_multiplier=sigma_after,
        label_grad_noise_multiplier=sigma_after,
        optimizer_noise_multiplier=sigma_after,
    )

    assert ledger.can_apply(projection)

def test_full_local_sample_dp_runtime_matches_mean_gradient_without_noise():
    import numpy as np
    import torch
    import torch.nn.functional as F

    from dynfed.split_learning import (
        build_full_model,
        build_split_pair,
        split_local_train_lenet5,
    )

    device = torch.device("cpu")
    torch.manual_seed(123)

    end, edge = build_split_pair(
        "lenet5",
        device,
        input_channels=1,
        image_size=28,
        num_classes=10,
    )

    end_state = {
        name: value.detach().clone()
        for name, value in end.state_dict().items()
    }
    edge_state = {
        name: value.detach().clone()
        for name, value in edge.state_dict().items()
    }
    full_state = {
        **end_state,
        **edge_state,
    }

    x = (
        np.random.default_rng(7)
        .normal(
            0.0,
            0.1,
            size=(2, 1, 28, 28),
        )
        .astype(np.float32)
    )
    y = np.array(
        [1, 4],
        dtype=np.int64,
    )

    expected_model = build_full_model(
        "lenet5",
        device,
        input_channels=1,
        image_size=28,
        num_classes=10,
    )
    expected_model.load_state_dict(full_state)
    expected_model.train()

    expected_optimizer = torch.optim.SGD(
        [
            param
            for param in expected_model.parameters()
            if param.requires_grad
        ],
        lr=0.01,
    )

    expected_optimizer.zero_grad()
    expected_logits = expected_model(
        torch.from_numpy(x).to(device)
    )
    expected_loss = F.cross_entropy(
        expected_logits,
        torch.from_numpy(y).to(device),
        reduction="mean",
    )
    expected_loss.backward()
    expected_optimizer.step()

    expected_diff = {
        name: param.detach() - full_state[name]
        for name, param in expected_model.named_parameters()
        if param.requires_grad
    }

    diagnostics = {}

    actual = split_local_train_lenet5(
        mode="LIIC",
        global_end_state=end_state,
        global_edge_state=edge_state,
        x=x,
        y=y,
        epochs=1,
        lr=0.01,
        device=device,
        model_name="lenet5",
        input_shape=(1, 28, 28),
        num_classes=10,
        mechanisms={"upd": "none"},
        local_steps=1,
        training_seed=17,
        dp_rng=np.random.default_rng(99),
        privacy_unit="sample",
        sample_optimizer_clip_norm=1e9,
        sample_optimizer_noise_multiplier=0.0,
        training_diagnostics=diagnostics,
    )

    actual_diff = {
        **actual["end"],
        **actual["edge"],
    }

    assert set(actual_diff) == set(expected_diff)

    for name in expected_diff:
        torch.testing.assert_close(
            actual_diff[name],
            expected_diff[name],
            rtol=2e-5,
            atol=2e-6,
        )

    assert diagnostics["actual_optimizer_steps"] == 1
    assert diagnostics["sample_dp_optimizer_steps"] == 1
    assert diagnostics["sample_dp_sample_count"] == 2
    assert diagnostics["sample_dp_clipped_sample_count"] == 0



def test_sample_split_runtime_matches_reference_without_sample_noise():
    import numpy as np
    import torch

    from dynfed.split_learning import (
        build_split_pair,
        split_local_train_lenet5,
    )

    device = torch.device("cpu")
    torch.manual_seed(123)

    end, edge = build_split_pair(
        "lenet5",
        device,
        input_channels=1,
        image_size=28,
        num_classes=10,
    )

    end_state = {
        name: value.detach().clone()
        for name, value in end.state_dict().items()
    }
    edge_state = {
        name: value.detach().clone()
        for name, value in edge.state_dict().items()
    }

    rng = np.random.default_rng(7)

    x = rng.normal(
        0.0,
        0.1,
        size=(2, 1, 28, 28),
    ).astype(np.float32)

    y = np.array(
        [1, 4],
        dtype=np.int64,
    )

    # Use actual embedding/label-gradient clipping in both runs.
    # This checks that the Sample End-side VJP includes the exact
    # detached embedding clipping scale.
    feature_clip = 0.25

    reference_diagnostics = {}

    reference = split_local_train_lenet5(
        mode="LIEIIC",
        global_end_state=end_state,
        global_edge_state=edge_state,
        x=x,
        y=y,
        epochs=1,
        lr=0.01,
        device=device,
        model_name="lenet5",
        input_shape=(1, 28, 28),
        num_classes=10,
        mechanisms={
            "emb": "dp",
            "grad": "dp",
            "upd": "none",
        },
        local_steps=1,
        training_seed=17,
        dp_rng=np.random.default_rng(99),
        dp_clip_norm=feature_clip,
        dp_noise_multiplier=0.0,
        privacy_unit="client",
        training_diagnostics=reference_diagnostics,
    )

    sample_diagnostics = {}

    sample = split_local_train_lenet5(
        mode="LIEIIC",
        global_end_state=end_state,
        global_edge_state=edge_state,
        x=x,
        y=y,
        epochs=1,
        lr=0.01,
        device=device,
        model_name="lenet5",
        input_shape=(1, 28, 28),
        num_classes=10,
        mechanisms={
            "emb": "dp",
            "grad": "dp",
            "upd": "none",
        },
        local_steps=1,
        training_seed=17,
        dp_rng=np.random.default_rng(99),
        dp_clip_norm=feature_clip,

        # Deliberately absurd legacy sigma. If Sample split accidentally
        # uses this value instead of the three Sample sigmas below,
        # numerical agreement will fail dramatically.
        dp_noise_multiplier=123.0,

        privacy_unit="sample",
        sample_embedding_noise_multiplier=0.0,
        sample_label_grad_noise_multiplier=0.0,
        sample_optimizer_noise_multiplier=0.0,
        sample_optimizer_clip_norm=1e9,
        training_diagnostics=sample_diagnostics,
    )

    assert set(sample["end"]) == set(reference["end"])
    assert set(sample["edge"]) == set(reference["edge"])

    for name in reference["end"]:
        torch.testing.assert_close(
            sample["end"][name],
            reference["end"][name],
            rtol=2e-5,
            atol=2e-6,
        )

    for name in reference["edge"]:
        torch.testing.assert_close(
            sample["edge"][name],
            reference["edge"][name],
            rtol=2e-5,
            atol=2e-6,
        )

    assert sample_diagnostics["sample_dp_optimizer_steps"] == 1
    assert sample_diagnostics["sample_dp_sample_count"] == 2
    assert sample_diagnostics["sample_dp_clipped_sample_count"] == 0



def test_reference_per_sample_vjp_grads_match_batch_mean_vjp():
    import torch

    from dynfed.sample_dp import (
        clip_and_aggregate_per_sample_grads,
        per_sample_vjp_grads_reference,
    )

    torch.manual_seed(123)

    model = torch.nn.Linear(
        3,
        2,
        bias=True,
    )

    x = torch.tensor(
        [
            [0.2, -0.4, 0.7],
            [1.1, 0.3, -0.5],
            [-0.8, 0.6, 0.9],
        ],
        dtype=torch.float32,
    )

    upstream = torch.tensor(
        [
            [0.5, -0.2],
            [-0.3, 0.8],
            [0.4, 0.1],
        ],
        dtype=torch.float32,
    )

    per_sample = per_sample_vjp_grads_reference(
        model,
        x,
        upstream,
    )

    aggregated, diagnostics = (
        clip_and_aggregate_per_sample_grads(
            per_sample,
            clip_norm=1e9,
            noise_multiplier=0.0,
        )
    )

    model.zero_grad(set_to_none=True)

    outputs = model(x)
    batch_surrogate = (
        outputs * upstream
    ).sum(dim=1).mean()
    batch_surrogate.backward()

    expected = {
        name: param.grad.detach().clone()
        for name, param in model.named_parameters()
        if param.requires_grad
    }

    assert set(aggregated) == set(expected)

    for name in expected:
        torch.testing.assert_close(
            aggregated[name],
            expected[name],
            rtol=1e-6,
            atol=1e-7,
        )

    assert diagnostics.sample_count == 3
    assert diagnostics.clipped_sample_count == 0


def test_reference_per_sample_vjp_rejects_shape_mismatch():
    import pytest
    import torch

    from dynfed.sample_dp import (
        per_sample_vjp_grads_reference,
    )

    model = torch.nn.Linear(3, 2)

    with pytest.raises(
        ValueError,
        match="output shape",
    ):
        per_sample_vjp_grads_reference(
            model,
            torch.zeros(2, 3),
            torch.zeros(2, 4),
        )


def test_vmap_per_sample_grads_match_reference():
    from dynfed.sample_dp import (
        per_sample_grads_reference,
        per_sample_grads_vmap,
    )

    torch.manual_seed(24680)

    model = torch.nn.Sequential(
        torch.nn.Linear(4, 4),
        torch.nn.BatchNorm1d(4),
        torch.nn.Tanh(),
        torch.nn.Linear(4, 3),
    )
    model.eval()

    # Exercise the runtime-relevant case where BatchNorm state is frozen
    # while other model parameters remain trainable.
    for param in model[1].parameters():
        param.requires_grad_(False)

    inputs = torch.randn(6, 4)
    targets = torch.tensor(
        [0, 1, 2, 1, 0, 2],
        dtype=torch.long,
    )

    reference = per_sample_grads_reference(
        model,
        inputs,
        targets,
        torch.nn.functional.cross_entropy,
    )

    vectorized = per_sample_grads_vmap(
        model,
        inputs,
        targets,
        torch.nn.functional.cross_entropy,
    )

    assert set(vectorized) == set(reference)

    for name in reference:
        assert vectorized[name].shape == reference[name].shape
        torch.testing.assert_close(
            vectorized[name],
            reference[name],
            rtol=1e-5,
            atol=1e-6,
        )


def test_vmap_per_sample_vjp_grads_match_reference():
    from dynfed.sample_dp import (
        per_sample_vjp_grads_reference,
        per_sample_vjp_grads_vmap,
    )

    torch.manual_seed(97531)

    model = torch.nn.Sequential(
        torch.nn.Linear(3, 5),
        torch.nn.ReLU(),
        torch.nn.Linear(5, 2),
    )

    inputs = torch.randn(7, 3)
    upstream = torch.randn(7, 2)

    reference = per_sample_vjp_grads_reference(
        model,
        inputs,
        upstream,
    )

    vectorized = per_sample_vjp_grads_vmap(
        model,
        inputs,
        upstream,
    )

    assert set(vectorized) == set(reference)

    for name in reference:
        assert vectorized[name].shape == reference[name].shape
        torch.testing.assert_close(
            vectorized[name],
            reference[name],
            rtol=1e-5,
            atol=1e-6,
        )


def test_vmap_per_sample_vjp_rejects_shape_mismatch():
    import pytest

    from dynfed.sample_dp import per_sample_vjp_grads_vmap

    model = torch.nn.Linear(3, 2)

    with pytest.raises(
        ValueError,
        match="output shape",
    ):
        per_sample_vjp_grads_vmap(
            model,
            torch.zeros(2, 3),
            torch.zeros(2, 4),
        )




def test_sample_optimizer_noise_multiplier_is_optional_until_execution():
    from dynfed.split_learning import (
        _resolved_sample_optimizer_noise_multiplier,
    )

    assert (
        _resolved_sample_optimizer_noise_multiplier(
            None,
            required=False,
        )
        is None
    )

    assert (
        _resolved_sample_optimizer_noise_multiplier(
            2.5,
            required=False,
        )
        == 2.5
    )


def test_sample_optimizer_noise_multiplier_is_required_at_execution():
    import pytest

    from dynfed.split_learning import (
        _resolved_sample_optimizer_noise_multiplier,
    )

    with pytest.raises(
        ValueError,
        match="optimizer DP-SGD is executed",
    ):
        _resolved_sample_optimizer_noise_multiplier(
            None,
            required=True,
        )

    with pytest.raises(
        ValueError,
        match="must be non-negative",
    ):
        _resolved_sample_optimizer_noise_multiplier(
            -0.1,
            required=False,
        )
