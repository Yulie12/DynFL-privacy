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