"""Opt-in *single-process* reference for trusted LIE joint Sample-DP-SGD.

No noisy embedding release takes place: End and Edge are inside one trusted
execution domain. This is NOT a distributed RPC/attestation implementation.
Only the joined optimizer gradient is clipped/noised. The production ledger,
private mode selection, aggregation, and model release boundaries are NOT
connected here; callers MUST NOT treat the returned updates as an end-to-end
privacy proof. Only publish externally after an appropriate accounting audit.
"""
from __future__ import annotations

from collections.abc import Iterable, MutableMapping

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sample_dp import (
    clip_and_aggregate_per_sample_grads,
    per_sample_grads_reference,
)


class JoinedSplit(nn.Module):
    """Joint autograd graph while preserving End/Edge parameter namespaces."""

    def __init__(self, end: nn.Module, edge: nn.Module) -> None:
        super().__init__()
        self.end = end
        self.edge = edge

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Internal trusted-domain activation. Never sent to an untrusted party.
        return self.edge(self.end(x))


def joint_per_sample_grads(
    end: nn.Module,
    edge: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Return joint End+Edge per-example CE gradients, without noise/clipping."""
    model = JoinedSplit(end, edge)
    if any(
        isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training
        for module in model.modules()
    ):
        # A training BatchNorm mixes examples and invalidates naive microbatch
        # per-sample gradients. Fail closed until a supported BN protocol exists.
        raise ValueError("joint Sample-DP requires all BatchNorm layers in eval mode")
    return per_sample_grads_reference(
        model, inputs, targets,
        lambda logits, labels: F.cross_entropy(logits, labels, reduction="mean"),
    )


def train_trusted_lie_joint_dp(
    *,
    end: nn.Module,
    edge: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    global_end_state: dict[str, torch.Tensor],
    global_edge_state: dict[str, torch.Tensor],
    end_optimizer: torch.optim.Optimizer | None,
    edge_optimizer: torch.optim.Optimizer | None,
    clip_norm: float,
    noise_multiplier: float,
    generator: torch.Generator,
    diagnostics: MutableMapping[str, float | int],
) -> dict[str, dict[str, torch.Tensor]]:
    """Update End and Edge using ONE joint DP clip/noise operation per batch.

    Requires trusted handling of activations, labels, gradients, model states,
    optimizer state, and diagnostics. Relies on a separate privacy ledger for
    composition and accounting of any external releases.
    """
    joint = JoinedSplit(end, edge)
    if any(isinstance(layer, nn.modules.batchnorm._BatchNorm) and layer.training
           for layer in joint.modules()):
        raise ValueError("trusted joint Sample-DP cannot train BatchNorm")
    if not (end_optimizer is not None or edge_optimizer is not None):
        raise ValueError("trusted split requires at least one trainable module")
    if not (float(clip_norm) > 0 and float(noise_multiplier) > 0):
        raise ValueError("trusted split requires positive DP clip and noise")

    named = [(name, p) for name, p in joint.named_parameters() if p.requires_grad]
    if not named:
        raise ValueError("trusted split has no trainable parameters")

    for bx, by in batches:
        diagnostics["actual_local_batches"] += 1
        for optimizer in (end_optimizer, edge_optimizer):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)

        per_sample = joint_per_sample_grads(end, edge, bx, by)
        protected, info = clip_and_aggregate_per_sample_grads(
            per_sample,
            clip_norm=clip_norm,
            noise_multiplier=noise_multiplier,
            generator=generator,
        )
        for name, parameter in named:
            if name not in protected:
                raise RuntimeError(f"missing protected joint gradient: {name}")
            parameter.grad = protected[name].detach().to(parameter).clone()
        for optimizer in (end_optimizer, edge_optimizer):
            if optimizer is not None:
                optimizer.step()

        diagnostics["actual_optimizer_steps"] += 1
        diagnostics["sample_dp_optimizer_steps"] += 1
        diagnostics["trusted_split_joint_dp_steps"] += 1
        diagnostics["sample_dp_sample_count"] += info.sample_count
        # Raw/clipped-gradient statistics depend on unprotected private
        # examples. Never include them in a worker result visible outside the
        # trusted domain. Existing numeric diagnostics retain their neutral
        # zero placeholders; the explicit flag distinguishes REDACTED from
        # measured zero.
        diagnostics["trusted_split_sensitive_gradient_stats_redacted"] = 1

    # Only trainable parameter deltas; cached models reload the supplied public
    # starting state on every call (buffers are not released by this helper).
    return {
        "end": {
            name: (param.detach() - global_end_state[name]).clone()
            for name, param in end.named_parameters() if param.requires_grad
        },
        "edge": {
            name: (param.detach() - global_edge_state[name]).clone()
            for name, param in edge.named_parameters() if param.requires_grad
        },
    }
