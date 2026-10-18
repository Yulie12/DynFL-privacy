#!/usr/bin/env python3
"""Standalone mathematical reference for Stage19 LIIC/LIIEIIIC Sample-DP SGD.

No training, no repository patches, no use of private client data.  This is a
conditional *mathematical* upper bound with explicitly checked assumptions,
NOT a validated deployment certificate or a drop-in Pareto objective.

Model: a fixed, pre-noise Cloud cohort, Sample-DP clipped gradients with
independent isotropic Gaussian noise, fresh PyTorch SGD momentum at the start
of EACH Edge stage, fixed sample-weighted Edge/Cloud model averaging, identical
trainable parameter vector across the clients, and no additional Cloud DP,
nonlinear postprocessing, or HE numerical errors.

The actual Torch code clips each per-sample gradient to C and adds noise with
coordinate std 2*C*sigma/batch_size to its *averaged* gradient.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class Step:
    batch_size: int
    clip_norm: float
    noise_multiplier: float

    def __post_init__(self) -> None:
        if type(self.batch_size) is not int or self.batch_size < 1:
            raise ValueError('batch_size must be positive integer')
        for name in ('clip_norm', 'noise_multiplier'):
            v = getattr(self, name)
            if not math.isfinite(v) or v < 0 or (name == 'clip_norm' and v == 0):
                raise ValueError(f'{name} must be finite and nonnegative (clip > 0)')

    @property
    def noise_std(self) -> float:
        return 2.0 * self.clip_norm * self.noise_multiplier / self.batch_size


@dataclass(frozen=True)
class Client:
    client_id: int
    mode: str  # LIIC, LIIEIIIC, LIIE (zero Cloud contribution), SKIP
    samples: int
    edge_id: int
    stages: tuple[tuple[Step, ...], ...]
    lr: float
    momentum: float
    weight_decay: float

    def __post_init__(self) -> None:
        if type(self.client_id) is not int or self.client_id < 0:
            raise ValueError('client_id must be a nonnegative integer')
        if type(self.samples) is not int or self.samples <= 0:
            raise ValueError('samples must be positive')
        if self.mode not in {'LIIC', 'LIIEIIIC', 'LIIE', 'SKIP'}:
            raise ValueError('unsupported mode: only full-model modes covered')
        if type(self.edge_id) is not int or self.edge_id < 0:
            raise ValueError('edge_id must be a nonnegative integer')
        if not (math.isfinite(self.lr) and self.lr >= 0):
            raise ValueError('lr must be finite nonnegative')
        if not (math.isfinite(self.momentum) and 0 <= self.momentum < 1):
            raise ValueError('momentum must be in [0,1)')
        if not (math.isfinite(self.weight_decay) and self.weight_decay >= 0):
            raise ValueError('weight_decay must be finite nonnegative')
        if self.mode == 'LIIC' and len(self.stages) != 1:
            raise ValueError('LIIC must have exactly one optimizer stage')
        if self.mode == 'LIIEIIIC' and len(self.stages) != 3:
            raise ValueError('LIIEIIIC must have exactly three Edge stages')
        if self.mode in {'LIIC','LIIEIIIC'} and any(not stage for stage in self.stages):
            raise ValueError('Cloud-active stages must have at least one step')


@dataclass(frozen=True)
class NoiseEvent:
    client_id: int
    stage: int
    step: int
    coefficient: float
    gradient_clip_norm: float
    noise_std: float


@dataclass(frozen=True)
class TransferBound:
    # Sum_cloud weighted post-training models = (1 + displacement_scale) * theta0 + noise/grad terms.
    displacement_scale: float
    events: tuple[NoiseEvent, ...]
    deterministic_norm: float
    gradient_radius: float
    noise_rms_radius: float
    upper_j: float
    dimension: int


def _mv(m: tuple[tuple[float,float],tuple[float,float]], x: tuple[float,float]) -> tuple[float,float]:
    return (m[0][0]*x[0]+m[0][1]*x[1], m[1][0]*x[0]+m[1][1]*x[1])


def sgd_stage_coefficients(*, steps: int, lr: float, momentum: float, weight_decay: float) -> tuple[float,tuple[float,...]]:
    """Exact theta_K = f*theta_0 + sum beta[k]*(clipped_grad_k + gaussian_k).

    Optimizer momentum starts at zero on each call (PyTorch SGD without
    Nesterov or dampening). Scalar transfer is identical for every parameter.
    """
    if type(steps) is not int or steps <= 0:
        raise ValueError('steps must be positive integer')
    if not all(math.isfinite(v) for v in (lr,momentum,weight_decay)) or lr < 0 or weight_decay < 0 or not 0 <= momentum < 1:
        raise ValueError('invalid optimizer coefficients')
    matrix = ((1-lr*weight_decay,-lr*momentum),(weight_decay,momentum))
    e_theta = (1.,0.)
    for _ in range(steps):
        e_theta = _mv(matrix,e_theta)
    f = e_theta[0]
    # Reverse: for last injected gradient, influence e_theta^T*b.
    injection = (-lr,1.)
    beta_reversed = []
    for _ in range(steps):
        beta_reversed.append(injection[0])
        injection = _mv(matrix,injection)
    betas = tuple(reversed(beta_reversed))
    if not all(math.isfinite(v) for v in (f,*betas)):
        raise ValueError('coefficient overflow')
    return f,betas


def build_transfer_bound(*, clients: Sequence[Client], dimension: int,
                         theta_norm: float, ideal_delta_norm: float,
                         theta_dot_ideal_delta: float | None = None,
                         cloud_step: float = 1.) -> TransferBound:
    """U >= E||Delta_cloud - Delta_ideal||^2, under the module assumptions.

    If <theta0, Delta_ideal> is not known pre-decision, apply a conservative
    triangle bound to the deterministic term.  The ideal gradient remains a
    **central Oracle** in Stage19: this is not a deployable FL certificate.
    """
    if type(dimension) is not int or dimension <= 0:
        raise ValueError('dimension must be positive integer')
    if not clients or len({c.client_id for c in clients}) != len(clients):
        raise ValueError('clients must be nonempty and unique')
    if not all(math.isfinite(v) and v >= 0 for v in (theta_norm, ideal_delta_norm, cloud_step)):
        raise ValueError('invalid norm/step')
    if theta_dot_ideal_delta is not None:
        if not math.isfinite(theta_dot_ideal_delta) or abs(theta_dot_ideal_delta) > theta_norm*ideal_delta_norm + 1e-8:
            raise ValueError('invalid theta dot ideal delta')
    cloud_clients = [c for c in clients if c.mode in {'LIIC','LIIEIIIC'}]
    cloud_mass = math.fsum(c.samples for c in cloud_clients)
    events: list[NoiseEvent] = []
    averaged_model_scale = 0.
    if cloud_mass:
        for c in cloud_clients:
            if c.mode != 'LIIC':
                continue
            f,beta = sgd_stage_coefficients(steps=len(c.stages[0]), lr=c.lr,
                                             momentum=c.momentum,weight_decay=c.weight_decay)
            mass_coeff = c.samples/cloud_mass
            averaged_model_scale += mass_coeff*f
            for idx,(step,b) in enumerate(zip(c.stages[0],beta)):
                events.append(NoiseEvent(c.client_id,0,idx,cloud_step*mass_coeff*b,
                                         step.clip_norm,step.noise_std))
        edge_groups: dict[int,list[Client]] = {}
        for c in cloud_clients:
            if c.mode == 'LIIEIIIC':
                edge_groups.setdefault(c.edge_id,[]).append(c)
        for edge_id, cohort in sorted(edge_groups.items()):
            edge_mass = math.fsum(c.samples for c in cohort)
            # Fixed cohort and fixed sample weights at every stage.
            scales: list[float] = []
            stage_betas: list[list[tuple[Client,tuple[float,...]]]] = []
            for s in range(3):
                f_edge=0.
                entries=[]
                for c in cohort:
                    f,beta=sgd_stage_coefficients(steps=len(c.stages[s]),lr=c.lr,
                                                  momentum=c.momentum,weight_decay=c.weight_decay)
                    f_edge += (c.samples/edge_mass)*f
                    entries.append((c,beta))
                scales.append(f_edge)
                stage_betas.append(entries)
            f_total=math.prod(scales)
            averaged_model_scale += (edge_mass/cloud_mass)*f_total
            for s,entries in enumerate(stage_betas):
                future_factor = math.prod(scales[s+1:])
                for c,beta in entries:
                    coeff = cloud_step*(c.samples/cloud_mass)*future_factor
                    for idx,(step,b) in enumerate(zip(c.stages[s],beta)):
                        events.append(NoiseEvent(c.client_id,s,idx,coeff*b,
                                                 step.clip_norm,step.noise_std))
    # Gamma*sum_i w_i*(theta_final_i - theta_0); if no Cloud contributors, zero.
    displacement_scale = cloud_step*(averaged_model_scale-1.) if cloud_mass else 0.
    if theta_dot_ideal_delta is not None:
        norm_sq=(displacement_scale**2*theta_norm**2 + ideal_delta_norm**2
                 -2*displacement_scale*theta_dot_ideal_delta)
        deterministic_norm=math.sqrt(max(norm_sq,0.))
    else:
        deterministic_norm=abs(displacement_scale)*theta_norm+ideal_delta_norm
    grad_r = math.fsum(abs(event.coefficient)*event.gradient_clip_norm for event in events)
    noise_r = math.sqrt(dimension*math.fsum((event.coefficient*event.noise_std)**2 for event in events))
    upper_j=(deterministic_norm+grad_r+noise_r)**2
    if not math.isfinite(upper_j):
        raise ValueError('bound overflow')
    return TransferBound(displacement_scale,tuple(events),deterministic_norm,grad_r,noise_r,upper_j,dimension)


if __name__ == '__main__':
    C=3.
    def client(cid,mode,edge):
        nstage=3 if mode=='LIIEIIIC' else 1
        return Client(cid,mode,60,edge,tuple(tuple(Step(20,C,4.6) for _ in range(3)) for _ in range(nstage)),.01,.9,.0001)
    for name, clients in (
        ('LIIC', [client(i,'LIIC',i%2) for i in range(4)]),
        ('LIIEIIIC', [client(i,'LIIEIIIC',i%2) for i in range(4)]),
        ('MIXED', [client(i,'LIIC' if i<2 else 'LIIEIIIC',i%2) for i in range(4)]),
    ):
        b=build_transfer_bound(clients=clients,dimension=1000,theta_norm=10.,ideal_delta_norm=.02)
        print(f'{name:9} deterministic={b.deterministic_norm:.6f} gradient={b.gradient_radius:.6f} noise={b.noise_rms_radius:.6f} U={b.upper_j:.6f}')
    print('Toy coefficients only; numbers are NOT from the ResNet18 run.')
