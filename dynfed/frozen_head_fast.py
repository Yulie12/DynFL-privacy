"""Opt-in exact per-sample CE gradients for a frozen backbone + linear head.

No clipping, noise, optimizer work, privacy accounting or scheduler behavior is
changed here. Only a redundant vmap-through-frozen-backbone computation is
replaced by one batched feature extraction and analytic linear-head gradients.
"""
from __future__ import annotations

from typing import Callable

import torch
import torch.nn as nn


def per_sample_grads_frozen_linear_head(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Per-example gradients of *unweighted* cross entropy for `classifier`.

    Expected model: pretrained ResNet full model with all backbone parameters
    frozen, BatchNorm in eval, no active dropout and `classifier=nn.Linear`.
    `loss_fn` is part of the compatibility interface; the caller must use
    `torch.nn.functional.cross_entropy` without custom weights/smoothing.
    """
    del loss_fn
    classifier = getattr(model, "classifier", None)
    if not isinstance(classifier, nn.Linear) or classifier.bias is None:
        raise RuntimeError("Fast frozen-head path requires a biased Linear classifier")

    trainable_names = {
        name for name, p in model.named_parameters() if p.requires_grad
    }
    if trainable_names != {"classifier.weight", "classifier.bias"}:
        raise RuntimeError(
            "Fast frozen-head path requires exactly the two trainable "
            "classifier parameters; got " + repr(sorted(trainable_names))
        )

    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm) and module.training:
            raise RuntimeError("Fast frozen-head path requires BatchNorm eval mode")
        if isinstance(module, nn.modules.dropout._DropoutNd) and module.training:
            raise RuntimeError("Fast frozen-head path does not support active dropout")

    if targets.ndim != 1 or inputs.shape[0] != targets.shape[0]:
        raise ValueError("Expected one integer class label per input")

    # Capture the last frozen feature vector entering the linear classifier.
    classifier_inputs: list[torch.Tensor] = []

    def _capture(_module: nn.Module, args: tuple[torch.Tensor, ...]) -> None:
        classifier_inputs.append(args[0])

    handle = classifier.register_forward_pre_hook(_capture)
    try:
        with torch.no_grad():
            logits = model(inputs)
    finally:
        handle.remove()

    if len(classifier_inputs) != 1:
        raise RuntimeError("Expected exactly one classifier invocation")
    features = classifier_inputs[0].detach()
    if features.ndim != 2 or logits.ndim != 2:
        raise RuntimeError("Expected [B, feature_dim] and [B, num_classes]")
    if features.shape[1] != classifier.in_features or logits.shape[1] != classifier.out_features:
        raise RuntimeError("Classifier features/logits have unexpected shape")

    # For per-example CE: dL/dlogits_i = softmax(logits_i) - onehot(y_i).
    logit_grads = torch.softmax(logits, dim=1)
    rows = torch.arange(targets.shape[0], device=targets.device)
    logit_grads[rows, targets.long()] -= 1.0

    return {
        "classifier.weight": logit_grads.unsqueeze(2) * features.unsqueeze(1),
        "classifier.bias": logit_grads,
    }
