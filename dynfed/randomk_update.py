"""DATA-INDEPENDENT masks and single trusted-aggregate Gaussian release.

Scope: fixed public roster, fixed public weights, one central release per round.
Raw updates reach a logical trusted curator: NOT a claim of local, packet,
SecAgg, or full DynFL end-to-end DP. All internal metrics are PRIVATE.
"""
from __future__ import annotations
import math
from collections.abc import Mapping, Iterable, Sequence
import numpy as np
import torch

Nested = dict[str, dict[str, torch.Tensor]]


def _leaves(reference: Mapping[str, Mapping[str, torch.Tensor]]):
    return sorted((part, name, value) for part, vals in reference.items()
                  for name, value in vals.items() if torch.is_floating_point(value))


def build_public_mask(reference: Mapping[str, Mapping[str, torch.Tensor]],
                      fraction: float, seed: int, strategy: str = "randomk") -> Nested:
    """Public/data-independent mask; randomk or layerwise_randomk or classifier_only.

    classifier_only selects exactly classifier/fc parameters, ignoring fraction.
    layerwise_randomk selects ceil(q * n) within each trainable tensor.
    """
    if not math.isfinite(float(fraction)) or not 0 < fraction <= 1:
        raise ValueError("fraction must be in (0, 1]")
    if strategy not in {"randomk", "layerwise_randomk", "classifier_only"}:
        raise ValueError("unknown mask strategy")
    leaves = _leaves(reference)
    total = sum(int(t.numel()) for _, _, t in leaves)
    if total == 0:
        raise ValueError("reference has no floating-point coordinates")
    output: Nested = {p: {} for p in reference}
    rng = np.random.default_rng(int(seed))
    if strategy == "randomk":
        chosen = np.sort(rng.choice(total, size=max(1, int(math.ceil(fraction*total))), replace=False))
    else:
        chosen = None
    offset = 0
    selected_count = 0
    for part, name, value in leaves:
        n = int(value.numel())
        if strategy == "randomk":
            start = int(np.searchsorted(chosen, offset, side="left"))
            stop = int(np.searchsorted(chosen, offset+n, side="left"))
            indices = chosen[start:stop] - offset
        elif strategy == "layerwise_randomk":
            indices = rng.choice(n, size=max(1, int(math.ceil(fraction*n))), replace=False)
        else:
            # Classifier head in the repository's Torchvision ResNet and LeNet.
            indices = np.arange(n, dtype=np.int64) if name.startswith(("classifier.", "fc3.")) else np.empty(0, dtype=np.int64)
        m = torch.zeros(n, dtype=torch.bool, device=value.device)
        if len(indices):
            m[torch.as_tensor(indices.copy(), device=value.device, dtype=torch.long)] = True
        selected_count += int(m.sum().item())
        output[part][name] = m.reshape(value.shape)
        offset += n
    if selected_count == 0:
        raise ValueError("strategy selected zero coordinates (classifier not found)")
    return output


def public_randomk_mask(reference: Mapping[str, Mapping[str, torch.Tensor]],
                        fraction: float, seed: int) -> Nested:
    return build_public_mask(reference, fraction, seed, "randomk")


def diff_norm(update: Mapping[str, Mapping[str, torch.Tensor]]) -> float:
    return math.sqrt(sum(float(t.detach().double().square().sum().item())
                         for part in update.values() for t in part.values()
                         if torch.is_floating_point(t)))


def project_update(update: Mapping[str, Mapping[str, torch.Tensor]], mask: Nested) -> Nested:
    result: Nested = {}
    for part, values in update.items():
        result[part] = {}
        for name, value in values.items():
            if not torch.is_floating_point(value):
                result[part][name] = torch.zeros_like(value)
                continue
            if part not in mask or name not in mask[part] or value.shape != mask[part][name].shape:
                raise ValueError(f"mask missing/mismatch for {part}/{name}")
            result[part][name] = torch.where(mask[part][name].to(value.device),
                                              value.detach(), torch.zeros_like(value))
    return result


def release_projected_aggregate(
    updates: Iterable[Nested], counts: Sequence[float], mask: Nested, *,
    clip_norm: float, noise_multiplier: float, seed: int,
) -> tuple[Nested, dict[str, float]]:
    """Consume updates STREAMING, clip projected client contributions, add noise.

    Replacement sensitivity <= 2*C*max(public weight) for fixed roster.
    sigma=0 is allowed ONLY for explicitly labeled NO-DP diagnostic ablations.
    Caller must privately store raw/private diagnostic fields and compose releases.
    """
    if not counts or not all(math.isfinite(float(c)) and float(c) > 0 for c in counts):
        raise ValueError("counts must be nonempty positive PUBLIC weights")
    if not (math.isfinite(clip_norm) and clip_norm > 0 and
            math.isfinite(noise_multiplier) and noise_multiplier >= 0):
        raise ValueError("clip norm must be positive; multiplier nonnegative")
    weights = [float(c)/sum(map(float, counts)) for c in counts]
    clean: Nested | None = None
    clipped = 0
    full_norm_sum = selected_norm_sum = 0.0
    count_seen = 0
    update_iter = iter(updates)
    for weight in weights:
        try:
            original = next(update_iter)
        except StopIteration as exc:
            raise ValueError("updates/counts lengths differ") from exc
        count_seen += 1
        full_norm = diff_norm(original)
        projected = project_update(original, mask)
        selected_norm = diff_norm(projected)
        full_norm_sum += full_norm
        selected_norm_sum += selected_norm
        if clean is None:
            clean = {p: {name: torch.zeros_like(t) for name, t in part.items()}
                     for p, part in projected.items()}
        scale = min(1.0, clip_norm/max(selected_norm, 1e-12))
        clipped += int(scale < 1.0-1e-12)
        for part, values in clean.items():
            for name, value in values.items():
                if torch.is_floating_point(value):
                    value.add_(projected[part][name], alpha=weight*scale)
    if count_seen != len(counts) or clean is None:
        raise ValueError("updates/counts lengths differ or updates empty")
    try:
        next(update_iter)
    except StopIteration:
        pass
    else:
        raise ValueError("updates/counts lengths differ")
    sensitivity = 2.0*clip_norm*max(weights)
    std = noise_multiplier*sensitivity
    rng = np.random.default_rng(int(seed))
    noisy: Nested = {p: {} for p in clean}
    noise_sq = 0.0
    for part, values in clean.items():
        for name, value in values.items():
            if not torch.is_floating_point(value):
                noisy[part][name] = value.clone()
                continue
            active = mask[part][name].to(value.device)
            n = int(active.sum().item())
            perturbation = torch.zeros_like(value)
            if n and std > 0:
                drawn = torch.from_numpy(rng.normal(0.0, std, size=n).astype(np.float32)).to(
                    device=value.device, dtype=value.dtype)
                perturbation[active] = drawn
                noise_sq += float(drawn.double().square().sum().item())
            noisy[part][name] = value + perturbation
    signal = diff_norm(clean)
    noise = math.sqrt(noise_sq)
    selected = sum(int(m.sum()) for part in mask.values() for m in part.values())
    total = sum(int(m.numel()) for part in mask.values() for m in part.values())
    return noisy, {
        "signal_norm": signal, "noise_norm": noise,
        "noise_to_signal": noise/max(signal, 1e-12),
        "sensitivity": sensitivity, "noise_std": std,
        "clipped_clients": float(clipped), "clipping_fraction": clipped/count_seen,
        "selected_coordinates": float(selected), "total_trainable_coordinates": float(total),
        "selected_fraction_actual": selected/total,
        "mean_client_full_update_norm": full_norm_sum/count_seen,
        "mean_client_projected_update_norm": selected_norm_sum/count_seen,
        "mean_client_signal_retention": selected_norm_sum/max(full_norm_sum, 1e-12),
        "expected_noise_norm": std*math.sqrt(selected),
    }
