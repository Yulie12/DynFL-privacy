"""Noise-calibration frontier for untrusted-edge secure aggregation.

Two independently-sampled client-noise schemes are compared:

``edge_only_exact``
    Assumes the edge does not collude with any client.  Each of K clients adds
    N(0, target_std^2 / K), so the released aggregate has exactly the target
    Gaussian variance.  If client noise shares are later revealed through
    collusion, the conditional unknown variance shrinks in proportion to h/K.

``collusion_robust_independent``
    Requires that any set of at least H_min unknown client noise shares retain
    the full target Gaussian variance after conditioning on the others.  Each
    client therefore adds N(0, target_std^2 / H_min).  With all K clients
    participating, the aggregate variance is inflated by K/H_min.

The module intentionally does *not* claim an exact-target, collusion-robust
threshold Gaussian protocol.  Achieving both properties requires a reviewed
joint/MPC/threshold noise-generation construction rather than ordinary
independent shares.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Iterable


@dataclass(frozen=True)
class NoiseFrontierPoint:
    clients: int
    minimum_unknown_clients: int
    target_noise_std: float
    scheme: str
    client_share_std: float
    released_noise_std: float
    std_inflation: float
    energy_inflation: float
    conditional_unknown_std_at_minimum: float
    conditional_unknown_variance_ratio: float
    full_target_dp_after_minimum_collusion_conditioning: bool
    edge_only_exact_target: bool
    implementation_status: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _validate(clients: int, minimum_unknown_clients: int, target_noise_std: float) -> None:
    if type(clients) is not int or clients < 2:
        raise ValueError("clients must be an integer >= 2")
    if (type(minimum_unknown_clients) is not int
            or not 2 <= minimum_unknown_clients <= clients):
        raise ValueError("minimum_unknown_clients must be in [2, clients]")
    if not math.isfinite(target_noise_std) or target_noise_std <= 0.0:
        raise ValueError("target_noise_std must be finite and positive")


def edge_only_exact_point(
    clients: int, minimum_unknown_clients: int, target_noise_std: float
) -> NoiseFrontierPoint:
    """Exact target variance when the edge cannot learn any client noise share."""
    _validate(clients, minimum_unknown_clients, target_noise_std)
    share_std = target_noise_std / math.sqrt(clients)
    conditional_std = share_std * math.sqrt(minimum_unknown_clients)
    conditional_ratio = minimum_unknown_clients / clients
    return NoiseFrontierPoint(
        clients=clients,
        minimum_unknown_clients=minimum_unknown_clients,
        target_noise_std=target_noise_std,
        scheme="edge_only_exact",
        client_share_std=share_std,
        released_noise_std=target_noise_std,
        std_inflation=1.0,
        energy_inflation=1.0,
        conditional_unknown_std_at_minimum=conditional_std,
        conditional_unknown_variance_ratio=conditional_ratio,
        full_target_dp_after_minimum_collusion_conditioning=(
            minimum_unknown_clients == clients
        ),
        edge_only_exact_target=True,
        implementation_status="available_with_existing_independent_client_noise_shares",
    )


def collusion_robust_independent_point(
    clients: int, minimum_unknown_clients: int, target_noise_std: float
) -> NoiseFrontierPoint:
    """Independent-share calibration robust to all but H_min revealed shares."""
    _validate(clients, minimum_unknown_clients, target_noise_std)
    share_std = target_noise_std / math.sqrt(minimum_unknown_clients)
    released_std = share_std * math.sqrt(clients)
    std_inflation = released_std / target_noise_std
    return NoiseFrontierPoint(
        clients=clients,
        minimum_unknown_clients=minimum_unknown_clients,
        target_noise_std=target_noise_std,
        scheme="collusion_robust_independent",
        client_share_std=share_std,
        released_noise_std=released_std,
        std_inflation=std_inflation,
        energy_inflation=std_inflation ** 2,
        conditional_unknown_std_at_minimum=target_noise_std,
        conditional_unknown_variance_ratio=1.0,
        full_target_dp_after_minimum_collusion_conditioning=True,
        edge_only_exact_target=(clients == minimum_unknown_clients),
        implementation_status="available_with_existing_independent_client_noise_shares",
    )


def threshold_joint_noise_target(
    clients: int, minimum_unknown_clients: int, target_noise_std: float
) -> dict[str, object]:
    """Desired third point; deliberately marked unimplemented.

    This describes the target contract only.  It is not a construction or a
    privacy proof and must not be reported as implemented by DynFL.
    """
    _validate(clients, minimum_unknown_clients, target_noise_std)
    return {
        "clients": clients,
        "minimum_unknown_clients": minimum_unknown_clients,
        "target_noise_std": target_noise_std,
        "scheme": "threshold_joint_exact_target",
        "released_noise_std": target_noise_std,
        "std_inflation": 1.0,
        "energy_inflation": 1.0,
        "full_target_dp_after_minimum_collusion_conditioning": True,
        "edge_only_exact_target": True,
        "implementation_status": "NOT_IMPLEMENTED_requires_reviewed_MPC_or_threshold_noise_protocol",
    }


def build_frontier(
    clients_values: Iterable[int],
    minimum_unknown_values: Iterable[int],
    target_noise_std: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in clients_values:
        for h in minimum_unknown_values:
            if h > k:
                continue
            rows.append(edge_only_exact_point(k, h, target_noise_std).to_dict())
            rows.append(collusion_robust_independent_point(k, h, target_noise_std).to_dict())
            rows.append(threshold_joint_noise_target(k, h, target_noise_std))
    return rows
