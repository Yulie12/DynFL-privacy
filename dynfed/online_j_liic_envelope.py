"""LIIC predecision update-norm envelope; NOT a standalone J certificate.

For actual uncapped zero-momentum LeNet5 SGD, with full-parameter clipping
||q_k|| <= C and (on Sample-DP-SGD) fresh independent additive Gaussian noise:

    theta_{k+1} = (1 - eta*wd)*theta_k - eta*(q_k + z_k).

The q_k can depend arbitrarily on *earlier* noise and on training data. The
following conditional L2 bound needs no gradient smoothness or unbiasedness:

    ||theta_K-theta_0||_{L2} <=
      (1-alpha**K)*||theta_0|| + C*sum_j b_j +
      sqrt(d*sum_j (b_j*2*C*sigma/B_j)**2),
      b_j = eta*alpha**(K-1-j), alpha = 1-eta*wd.

Important: these are RMS norm caps on the *local* update only. The ideal
full-gradient target is unbounded without additional evidence of ||grad F||;
therefore these caps MUST NOT be used as formal J bounds on their own.

The module does not access client gradients or change SGD/DP/selection.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Mapping, Sequence


def _nonnegative(value: Any, key: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{key}: boolean is invalid")
    try:
        x = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{key}: expected finite nonnegative number") from exc
    if not math.isfinite(x) or x < 0:
        raise ValueError(f"{key}: expected finite nonnegative number")
    return x


@dataclass(frozen=True)
class LIICUpdateEnvelope:
    bound_rms: float
    decay_part: float
    clipped_gradient_part: float
    local_dp_noise_rms: float
    steps: int
    client_samples: int
    # Explicitly not a proof of the full learning MSE or of privacy.
    reference_gradient_certified: bool = False

    def record(self) -> dict[str, Any]:
        return asdict(self)


def liic_update_envelope(metadata: Mapping[str, Any], *, local_epochs: int) -> LIICUpdateEnvelope:
    """Bound LIIC local update, conditional on exact captured optimizer inputs.

    Excludes later independent packet/aggregate DP, HE arithmetic and network
    transformations. Those must be bounded separately when constructing J.
    """
    if metadata.get("schema") != "dynfl.liic.sgd.v1" or metadata.get("mode") != "LIIC":
        raise ValueError("LIIC SGD runtime metadata required")
    if metadata.get("model") != "lenet5" or metadata.get("optimizer") != "torch.optim.SGD":
        raise ValueError("unsupported model/optimizer")
    if metadata.get("momentum") != 0:
        raise ValueError("momentum SGD needs a separate proof")
    source = metadata.get("source")
    if source not in {"sample_dp_sgd", "clipped_sgd"}:
        raise ValueError("unsupported SGD source")
    eta = _nonnegative(metadata.get("learning_rate"), "learning_rate")
    wd = _nonnegative(metadata.get("weight_decay"), "weight_decay")
    theta0 = _nonnegative(metadata.get("initial_parameter_norm"), "initial_parameter_norm")
    clip = _nonnegative(metadata.get("gradient_clip_norm"), "gradient_clip_norm")
    sigma = _nonnegative(metadata.get("sample_noise_multiplier"), "sample_noise_multiplier")
    if eta <= 0 or clip <= 0:
        raise ValueError("learning_rate and gradient_clip_norm must be positive")
    if source == "clipped_sgd" and (sigma != 0 or clip != 5.0):
        raise ValueError("clean LIIC SGD uses clip=5 and no minibatch DP")
    dim = metadata.get("vector_dimension")
    if isinstance(dim, bool) or not isinstance(dim, int) or dim != 61706:
        raise ValueError("LeNet5 optimizer parameter scope must be 61706")
    batches = metadata.get("batch_sizes")
    if (not isinstance(batches, list) or not batches or len(batches) > 100000
            or any(isinstance(b, bool) or not isinstance(b, int) or b <= 0 for b in batches)):
        raise ValueError("invalid actual minibatch schedule")
    if metadata.get("planned_optimizer_steps") != len(batches):
        raise ValueError("batch schedule disagrees with planned optimizer steps")
    alpha = 1 - eta * wd
    if not 0 <= alpha <= 1:
        raise ValueError("decay recursion requires 0 <= 1-eta*wd <= 1")
    K = len(batches)
    weights = [eta * alpha ** (K - 1 - j) for j in range(K)]
    decay = (1 - alpha**K) * theta0
    clipped = clip * math.fsum(weights)
    noise_rms = 0.0
    if source == "sample_dp_sgd":
        # Real SUM-noise scale 2*C*sigma then divide by actual minibatch size.
        noise_rms = math.sqrt(dim * math.fsum(
            (weight * 2 * clip * sigma / batch)**2
            for weight, batch in zip(weights, batches)
        ))
    B = math.fsum((decay, clipped, noise_rms))
    if not math.isfinite(B):
        raise ValueError("LIIC update envelope is nonfinite")
    # Stage13 Shadow predates storing epochs. An explicit, independently
    # known epoch count is mandatory: repeated batch sizes alone do not
    # uniquely identify dataset size or epoch count.
    if (isinstance(local_epochs, bool) or not isinstance(local_epochs, int)
            or local_epochs <= 0 or K % local_epochs):
        raise ValueError("explicit positive local_epochs must match batch schedule")
    one_epoch = batches[: K // local_epochs]
    if batches != one_epoch * local_epochs:
        raise ValueError("optimizer minibatch plan does not repeat across epochs")
    one_epoch_samples = sum(one_epoch)
    return LIICUpdateEnvelope(
        bound_rms=B, decay_part=decay, clipped_gradient_part=clipped,
        local_dp_noise_rms=noise_rms, steps=K,
        client_samples=one_epoch_samples,
    )


def sample_weighted_liic_envelope(
    selected: Sequence[tuple[int, LIICUpdateEnvelope]],
    *, cloud_step: float = 1.0,
) -> float:
    """||gamma sum_i a_i u_i||_L2 upper bound, if weights really are n_i/sum n.

    This is conditional on the exact runtime cloud decomposition and no
    unmodeled releases, transformations, aggregation errors or updates.
    """
    gamma = _nonnegative(cloud_step, "cloud_step")
    if gamma <= 0 or not selected:
        raise ValueError("positive cloud_step and at least one client required")
    ids = [cid for cid, _ in selected]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate Client ID in Cloud cohort")
    total = math.fsum(entry.client_samples for _, entry in selected)
    if total <= 0:
        raise ValueError("Cloud cohort has zero sample mass")
    return gamma * math.fsum(
        entry.client_samples / total * entry.bound_rms for _, entry in selected
    )



def fmnist_lenet5_gradient_norm_cap(
    *, model_parameter_norm_upper: float,
    normalized_pixel_abs_bound: float = 2.25,
) -> float:
    """Architecture-derived ||grad F(w)|| cap for THIS LeNet5 CE objective.

    Works for normalized 1x28x28 inputs with |each pixel| <= pixel bound,
    one-hot labels, unregularized mean softmax-cross-entropy, EndNet+EdgeNet
    (conv5/avgpool2/conv5/avgpool2/fc400-120-84-10, ReLU). No data access.

    Each weight and bias tensor's Frobenius norm <= full-parameter norm R.
    A stride-1 5x5 convolution has l2 operator norm <= 5*||W||_F,
    a disjoint 2x2 average pool has operator norm <= 1/2, and
    ||softmax(logits)-one_hot||_2 <= sqrt(2). Standard reverse-mode norm
    inequalities then give a cap for the complete gradient vector.

    Does not cover batch normalization, data augmentation, extra loss terms,
    alternate neural nets, or the floating-point error of the entire trainer.
    R must be a valid UPPER bound on the actual parameters, not an estimate.
    """
    R = _nonnegative(model_parameter_norm_upper, "model_parameter_norm_upper")
    pixel = _nonnegative(normalized_pixel_abs_bound, "normalized_pixel_abs_bound")
    if pixel == 0:
        raise ValueError("normalized_pixel_abs_bound must be positive")
    x = 28.0 * pixel  # sqrt(28*28) pixels
    # Forward activation bounds, including replicated conv biases.
    h1 = 0.5 * (5*R*x + 28*R)
    h2 = 0.5 * (5*R*h1 + 10*R)
    h3 = R*h2 + R
    h4 = R*h3 + R
    # Backward signal bounds. ReLU gates are contractions.
    d5 = math.sqrt(2.0)
    d4 = R*d5
    d3 = R*d4
    d2 = 0.5*R*d3  # second avgpool transpose
    d1 = 0.5 * 5*R*d2  # conv2 transpose and first avgpool transpose
    # Joint gradients W+b for each of five affine operators.
    gradient_cap = math.hypot(
        5*d1*x, 28*d1,      # conv1, bias1
        5*d2*h1, 10*d2,    # conv2, bias2
        d3*h2, d3,         # fc1, bias3
        d4*h3, d4,         # fc2, bias4
        d5*h4, d5,         # fc3, bias5
    )
    if not math.isfinite(gradient_cap):
        raise ValueError("architecture gradient norm cap overflow")
    return gradient_cap



_LENET5_SHAPES = {
    "conv1": ((6,1,5,5),(6,)), "conv2": ((16,6,5,5),(16,)),
    "fc1": ((120,400),(120,)), "fc2": ((84,120),(84,)),
    "fc3": ((10,84),(10,)),
}


def lenet5_layer_norm_upper_from_states(end_state, edge_state,
                                         *, relative_inflation=1e-6,
                                         absolute_inflation=1e-9):
    """Validated, safely inflated W/b norm estimates for this repo's LeNet5.

    These margins are engineering allowances, not certified floating-point
    error proofs. Requires exactly the expected layer tensor shapes.
    """
    import torch
    rel = _nonnegative(relative_inflation, "relative_inflation")
    margin = _nonnegative(absolute_inflation, "absolute_inflation")
    output={}
    for layer,(w_shape,b_shape) in _LENET5_SHAPES.items():
        state=end_state if layer.startswith('conv') else edge_state
        if not isinstance(state, Mapping):
            raise ValueError('expected real LeNet5 model state mapping')
        output[layer]={}
        for suffix,shape,out_key in (("weight",w_shape,"weight_fro"),
                                     ("bias",b_shape,"bias_l2")):
            tensor=state.get(f'{layer}.{suffix}')
            if tensor is None or tuple(getattr(tensor,'shape',())) != shape:
                raise ValueError(f'{layer}.{suffix}: incorrect LeNet5 shape')
            if not torch.is_floating_point(tensor) or not bool(torch.isfinite(tensor).all()):
                raise ValueError(f'{layer}.{suffix}: invalid/nonfinite model parameters')
            norm=float(torch.linalg.vector_norm(tensor.detach().double()).item())
            if not math.isfinite(norm):
                raise ValueError('nonfinite layer norm')
            output[layer][out_key]=norm*(1+rel)+margin
    return output


def fmnist_lenet5_gradient_norm_cap_by_layers(*, layer_norm_uppers,
                                               normalized_pixel_abs_bound=2.25):
    """Architecture-conditional ||grad F|| bound using ten W/b tensor norms.

    Applies only to fixed normalized 1x28x28 inputs, this 5-layer AvgPool
    LeNet5, unregularized CE. This is not a complete J error certificate.
    """
    if not isinstance(layer_norm_uppers,Mapping) or set(layer_norm_uppers)!=set(_LENET5_SHAPES):
        raise ValueError('exactly five LeNet5 affine layers required')
    pixel=_nonnegative(normalized_pixel_abs_bound,'normalized_pixel_abs_bound')
    if pixel<=0: raise ValueError('pixel bound must be positive')
    items=[]
    for layer in _LENET5_SHAPES:
        item=layer_norm_uppers[layer]
        if not isinstance(item,Mapping) or set(item)!={'weight_fro','bias_l2'}:
            raise ValueError(f'{layer} needs weight_fro and bias_l2')
        items.append((_nonnegative(item['weight_fro'],f'{layer}.weight_fro'),
                      _nonnegative(item['bias_l2'],f'{layer}.bias_l2')))
    (w1,b1),(w2,b2),(w3,b3),(w4,b4),(w5,b5)=items
    x=28*pixel
    h1=(5*w1*x+28*b1)*.5
    h2=(5*w2*h1+10*b2)*.5
    h3=w3*h2+b3
    h4=w4*h3+b4
    d5=math.sqrt(2)
    d4=w5*d5
    d3=w4*d4
    d2=w3*d3*.5
    d1=5*w2*d2*.5
    bound=math.hypot(5*d1*x,28*d1,5*d2*h1,10*d2,d3*h2,d3,d4*h3,d4,d5*h4,d5)
    if not math.isfinite(bound): raise ValueError('layerwise gradient bound overflow')
    return bound

def conditional_liic_j_upper(
    *, update_envelope: float, ideal_step: float,
    global_gradient_norm_cap: float | None,
    aggregate_noise_rms: float, numeric_rms: float,
    gradient_cap_independently_proved: bool,
    linear_decomposition_independently_proved: bool,
    noise_and_numeric_bounds_independently_proved: bool,
    evidence: str,
) -> float:
    """Rigorous conditional J <= (B + eta*G + Q_rms + N_rms)^2.

    Does NOT assert any of its external conditions on the caller's behalf.
    Uses Minkowski rather than treating correlated/noisy terms as orthogonal.
    """
    B = _nonnegative(update_envelope, "update_envelope")
    eta = _nonnegative(ideal_step, "ideal_step")
    Q = _nonnegative(aggregate_noise_rms, "aggregate_noise_rms")
    N = _nonnegative(numeric_rms, "numeric_rms")
    if global_gradient_norm_cap is None:
        raise ValueError("missing independently proved ||grad F(w_t)|| cap")
    G = _nonnegative(global_gradient_norm_cap, "global_gradient_norm_cap")
    if (gradient_cap_independently_proved is not True
            or linear_decomposition_independently_proved is not True
            or noise_and_numeric_bounds_independently_proved is not True
            or not isinstance(evidence, str) or not evidence.strip()):
        raise ValueError("external reference/decomposition/noise evidence missing")
    upper = math.fsum((B, eta * G, Q, N))**2
    if not math.isfinite(upper):
        raise ValueError("conditional J upper bound is nonfinite")
    return upper
