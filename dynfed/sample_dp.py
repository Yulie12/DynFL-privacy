from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Mapping

import torch


@dataclass(frozen=True)
class SampleDPDiagnostics:
    sample_count: int
    clipped_sample_count: int
    max_raw_grad_norm: float
    max_clipped_grad_norm: float


def clip_and_aggregate_per_sample_grads(
    per_sample_grads: Mapping[str, torch.Tensor],
    *,
    clip_norm: float,
    noise_multiplier: float,
    generator: torch.Generator | None = None,
) -> tuple[dict[str, torch.Tensor], SampleDPDiagnostics]:
    """Reference sample-level DP aggregation for one minibatch.

    Each mapping value has shape ``[B, ...]`` where the leading dimension
    indexes samples and the remaining dimensions match one trainable
    parameter.

    For each sample, gradients across *all* trainable parameters are treated
    as one vector. A single global L2 clipping factor is computed for that
    sample and applied to every parameter gradient belonging to it.

    Clipped per-sample gradients are summed, independent Gaussian noise is
    added to every parameter coordinate, and the result is divided by the
    minibatch size.

    Replacement adjacency is used, so the sensitivity of the clipped
    gradient sum is ``2 * clip_norm``.

    This is intentionally a simple reference implementation. It does not
    perform privacy accounting or subsampling amplification.
    """
    if not per_sample_grads:
        raise ValueError("per_sample_grads must not be empty")
    if clip_norm <= 0.0:
        raise ValueError("clip_norm must be positive")
    if noise_multiplier < 0.0:
        raise ValueError("noise_multiplier must be non-negative")

    items = list(per_sample_grads.items())

    first_name, first_grad = items[0]
    if first_grad.ndim < 1:
        raise ValueError(
            f"per-sample gradient for {first_name!r} must have a batch dimension"
        )

    batch_size = int(first_grad.shape[0])
    if batch_size <= 0:
        raise ValueError("batch size must be positive")

    reference_device = first_grad.device

    squared_norms = torch.zeros(
        batch_size,
        dtype=torch.float64,
        device=reference_device,
    )

    for name, grad in items:
        if grad.ndim < 1:
            raise ValueError(
                f"per-sample gradient for {name!r} must have a batch dimension"
            )
        if int(grad.shape[0]) != batch_size:
            raise ValueError(
                f"inconsistent batch size for {name!r}: "
                f"{int(grad.shape[0])} != {batch_size}"
            )
        if grad.device != reference_device:
            raise ValueError("all per-sample gradients must be on the same device")

        flat = grad.detach().reshape(batch_size, -1).to(dtype=torch.float64)
        squared_norms += torch.sum(flat * flat, dim=1)

    raw_norms = torch.sqrt(squared_norms)

    scales = torch.clamp(
        float(clip_norm) / (raw_norms + 1e-12),
        max=1.0,
    )

    aggregated: dict[str, torch.Tensor] = {}
    noise_std = 2.0 * float(clip_norm) * float(noise_multiplier)

    for name, grad in items:
        scale_shape = (batch_size,) + (1,) * (grad.ndim - 1)
        param_scales = scales.to(dtype=grad.dtype).reshape(scale_shape)

        clipped = grad * param_scales
        grad_sum = clipped.sum(dim=0)

        if noise_std > 0.0:
            noise = torch.randn(
                grad_sum.shape,
                dtype=grad_sum.dtype,
                device=grad_sum.device,
                generator=generator,
            )
            grad_sum = grad_sum + noise * noise_std

        aggregated[name] = grad_sum / float(batch_size)

    clipped_norms = raw_norms * scales

    diagnostics = SampleDPDiagnostics(
        sample_count=batch_size,
        clipped_sample_count=int(
            (raw_norms > float(clip_norm)).sum().item()
        ),
        max_raw_grad_norm=float(raw_norms.max().item()),
        max_clipped_grad_norm=float(clipped_norms.max().item()),
    )

    return aggregated, diagnostics

def per_sample_grads_reference(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Compute exact per-sample gradients with a microbatch-size-1 loop.

    The returned mapping has one entry per trainable parameter. Each tensor
    has shape ``[B, ...]``, where the leading dimension indexes samples and
    the remaining dimensions match the corresponding parameter.

    This helper intentionally performs no clipping, no noise addition, no
    optimizer step, and no privacy accounting. Its output is designed to be
    consumed by ``clip_and_aggregate_per_sample_grads``.

    ``loss_fn`` must return a scalar loss for a single-sample minibatch.
    """
    if inputs.ndim < 1:
        raise ValueError("inputs must have a batch dimension")
    if targets.ndim < 1:
        raise ValueError("targets must have a batch dimension")

    batch_size = int(inputs.shape[0])
    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if int(targets.shape[0]) != batch_size:
        raise ValueError(
            "inputs and targets must have the same leading batch dimension"
        )

    named_params = [
        (name, param)
        for name, param in model.named_parameters()
        if param.requires_grad
    ]
    if not named_params:
        raise ValueError("model has no trainable parameters")

    param_names = [name for name, _ in named_params]
    params = [param for _, param in named_params]

    collected: dict[str, list[torch.Tensor]] = {
        name: [] for name in param_names
    }

    for sample_idx in range(batch_size):
        sample_inputs = inputs[sample_idx : sample_idx + 1]
        sample_targets = targets[sample_idx : sample_idx + 1]

        outputs = model(sample_inputs)
        loss = loss_fn(outputs, sample_targets)

        if loss.numel() != 1:
            raise ValueError(
                "loss_fn must return a scalar loss for each single-sample minibatch"
            )

        grads = torch.autograd.grad(
            loss,
            params,
            allow_unused=True,
        )

        for (name, param), grad in zip(named_params, grads):
            if grad is None:
                sample_grad = torch.zeros_like(param)
            else:
                sample_grad = grad.detach().clone()

            collected[name].append(sample_grad)

    return {
        name: torch.stack(sample_grads, dim=0)
        for name, sample_grads in collected.items()
    }