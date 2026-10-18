"""Isolated fixed-LIIC experiment; never modifies the DynFL training code.

Three paths:
- batch_explore: one 101-release-calibrated multiplier used for *each* minibatch.
  This is NOT (epsilon=8,delta)-DP over 101 rounds. We report its actual
  conservative composed epsilon explicitly.
- batch_accounted: conventional non-subsampled Gaussian-RDP accountant with
  actual full-epoch optimizer-step horizon; one global epsilon=8 goal.
- final_update: ordinary local SGD followed by ONE clipped/noised vector
  release per client per round, with replacement sensitivity <=2*S.

The RDP numbers cover only the specifically modeled client message mechanism;
this experiment is not a formal end-to-end guarantee for DynFL selection,
intermediate messages, dataset-dependent roster, metrics, RNG, or HE.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import secrets
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

# Resolve the project, independent of working-directory invocation.
PROJECT = Path(__file__).resolve().parents[1]
if str(PROJECT) not in sys.path:
    sys.path.insert(0, str(PROJECT))

from dynfed.fmnist_lenet5_dynamic import (  # noqa: E402
    _default_data_root, _partition_clients_lenet5, _split_client_indices,
    load_image_dataset_arrays,
)
from dynfed.privacy import (  # noqa: E402
    calibrate_gaussian_noise, epsilon_from_rdp, gaussian_rdp,
)
from dynfed.sample_dp import (  # noqa: E402
    clip_and_aggregate_per_sample_grads, per_sample_grads_vmap,
)
from dynfed.split_learning import build_full_model, _make_optimizer  # noqa: E402

MODES = ("batch_explore", "batch_accounted", "final_update")


def event_count(samples: int, batch: int, epochs: int) -> int:
    if samples <= 0:
        return 0
    return math.ceil(samples / batch) * epochs


def bound_update(delta: dict[str, torch.Tensor], limit: float) -> tuple[dict[str, torch.Tensor], float]:
    """Whole-vector clip. If local SGD diverges, bound the finite projection."""
    if not delta:
        return {}, 0.0
    clean = {k: torch.nan_to_num(v.detach().float().cpu(), nan=0., posinf=0., neginf=0.) for k, v in delta.items()}
    norm = math.sqrt(sum(float((t.double().square()).sum()) for t in clean.values()))
    factor = min(1.0, limit / max(norm, 1e-12))
    return {k: (v * factor) for k, v in clean.items()}, norm


def noisy_final_update(delta: dict[str, torch.Tensor], clip: float, sigma: float,
                       generator: torch.Generator) -> dict[str, torch.Tensor]:
    bounded, _ = bound_update(delta, clip)
    # Replacement adjacency: any two bounded outputs are at most 2*clip apart.
    standard_deviation = 2.0 * clip * sigma
    return {k: (v + torch.randn(v.shape, generator=generator, dtype=v.dtype) * standard_deviation)
            for k, v in bounded.items()}


def sample_dp_batch(model: torch.nn.Module, bx: torch.Tensor, by: torch.Tensor,
                    optimizer: torch.optim.Optimizer, clip: float,
                    sigma: float, generator: torch.Generator) -> None:
    optimizer.zero_grad(set_to_none=True)
    grads = per_sample_grads_vmap(model, bx, by, F.cross_entropy)
    averaged, _ = clip_and_aggregate_per_sample_grads(
        grads, clip_norm=clip, noise_multiplier=sigma, generator=generator)
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parameter.grad = averaged[name].detach().clone()
    optimizer.step()


def local_delta(model: torch.nn.Module, global_state: dict[str, torch.Tensor],
                x: np.ndarray, y: np.ndarray, *, epochs: int, batch: int,
                lr: float, l2: float, train_seed: int, mode: str, clip: float,
                sigma: float, noise_generator: torch.Generator,
                device: torch.device) -> tuple[dict[str, torch.Tensor], int]:
    """Full local epochs: no 5-step limit and no pre-truncation of samples."""
    model.load_state_dict(global_state)
    model.train()
    opt = _make_optimizer(model, lr, 'lenet5', weight_decay=l2)
    if opt is None:
        raise RuntimeError('No trainable parameters')
    shuffle_gen = torch.Generator().manual_seed(train_seed)
    x_tensor = torch.from_numpy(x).float().reshape(-1, 1, 28, 28)
    y_tensor = torch.from_numpy(y).long()
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_tensor, y_tensor),
        batch_size=batch, shuffle=True, generator=shuffle_gen, drop_last=False)
    steps = 0
    for _ in range(epochs):
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            if mode in ('batch_explore', 'batch_accounted'):
                sample_dp_batch(model, bx, by, opt, clip, sigma, noise_generator)
            else:
                opt.zero_grad(set_to_none=True)
                loss = F.cross_entropy(model(bx), by)
                loss.backward()
                # Match the old no-DP training route. This is NOT sample clipping.
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                opt.step()
            steps += 1
    with torch.no_grad():
        delta = {name: (p.detach().cpu() - global_state[name].detach().cpu()).float()
                 for name, p in model.named_parameters() if p.requires_grad}
    if mode == 'final_update':
        delta = noisy_final_update(delta, clip, sigma, noise_generator)
    return delta, steps


@torch.no_grad()
def test_accuracy(model: torch.nn.Module, state: dict[str, torch.Tensor],
                  x: np.ndarray, y: np.ndarray, *, device: torch.device,
                  batch_size: int = 256) -> tuple[float, float]:
    model.load_state_dict(state)
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    for off in range(0, len(y), batch_size):
        bx = torch.from_numpy(x[off:off + batch_size]).float().reshape(-1, 1, 28, 28).to(device)
        by = torch.from_numpy(y[off:off + batch_size]).long().to(device)
        pred = model(bx)
        loss_sum += float(F.cross_entropy(pred, by, reduction='sum'))
        correct += int((pred.argmax(dim=1) == by).sum())
        total += len(by)
    return 100.0 * correct / max(total, 1), loss_sum / max(total, 1)


def main() -> None:
    p = argparse.ArgumentParser(description='Two-route LIIC DP experiment. Fixed clients and ordinary FedAvg.')
    p.add_argument('--mode', choices=MODES, required=True)
    p.add_argument('--rounds', type=int, default=2, help='Actually execute N rounds, from scratch')
    p.add_argument('--horizon', type=int, default=101, help='Full planned release horizon for calibration')
    p.add_argument('--clients', type=int, default=100)
    p.add_argument('--edges', type=int, default=10)
    p.add_argument('--train-limit', type=int, default=24000)
    p.add_argument('--test-limit', type=int, default=2000)
    p.add_argument('--epochs', type=int, default=3)
    p.add_argument('--batch-size', type=int, default=64)
    p.add_argument('--lr', type=float, default=0.05)
    p.add_argument('--weight-decay', type=float, default=1e-4)
    p.add_argument('--grad-clip', type=float, default=1.0, help='For batch DP paths only')
    p.add_argument('--update-clip', type=float, default=0.05, help='For final update path only')
    p.add_argument('--epsilon', type=float, default=8.0)
    p.add_argument('--delta', type=float, default=1e-5)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--device', choices=('cuda', 'cpu'), default='cuda')
    p.add_argument('--output-root', type=Path, default=Path('out/liic_two_dp_routes'))
    p.add_argument('--synthetic-smoke', action='store_true',
                   help='Code-only smoke test with artificial images; no real dataset/DP claim')
    a = p.parse_args()
    if a.rounds <= 0 or a.horizon < a.rounds or a.epochs <= 0 or a.batch_size <= 0:
        p.error('require 0 < rounds <= horizon, epochs>0, batch-size>0')
    if a.clients <= 0 or a.edges <= 0 or a.epsilon <= 0 or not 0 < a.delta < 1:
        p.error('invalid client/edge/privacy target')
    if a.grad_clip <= 0 or a.update_clip <= 0:
        p.error('clip norms must be positive')
    if a.device == 'cuda' and not torch.cuda.is_available():
        p.error('CUDA is not available. Use --device cpu if appropriate.')
    device = torch.device(a.device)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.synthetic_smoke:
        rng = np.random.default_rng(a.seed)
        x_train = rng.normal(size=(a.clients * 8, 28 * 28)).astype(np.float32)
        y_train = rng.integers(0, 10, len(x_train)).astype(np.int64)
        x_test = rng.normal(size=(32, 28 * 28)).astype(np.float32)
        y_test = rng.integers(0, 10, len(x_test)).astype(np.int64)
        parts = [np.arange(i * 8, (i + 1) * 8, dtype=np.int64) for i in range(a.clients)]
        train_parts = parts
    else:
        x_train, y_train, x_test, y_test, shape, _dataset, classes = load_image_dataset_arrays(
            'fmnist', Path(_default_data_root('fmnist')),
            a.train_limit, a.test_limit, a.seed)
        if shape != (1, 28, 28) or classes != 10:
            raise RuntimeError('Expected Fashion-MNIST shape/classes')
        parts = _partition_clients_lenet5(y_train, a.clients, a.edges, False,
                                          'edge_label_skew', a.seed, .5)
        train_parts, _ = _split_client_indices(parts, test_ratio=.2, seed=a.seed)
    counts = [len(idx) for idx in train_parts]
    active_clients = [i for i, n in enumerate(counts) if n > 0]
    if not active_clients:
        raise RuntimeError('No trainable client')
    max_events_in_round = max(event_count(n, a.batch_size, a.epochs) for n in counts)
    planned_batch_events = a.horizon * max_events_in_round
    assumed_events = a.horizon if a.mode != 'batch_accounted' else planned_batch_events
    sigma = calibrate_gaussian_noise(a.epsilon, a.delta, assumed_events)
    # Honest, worst-case non-subsampled accounting across local batches for the
    # explicitly simulated client-message paths; no silent epsilon division.
    true_events = a.horizon if a.mode == 'final_update' else planned_batch_events
    accounted_epsilon = epsilon_from_rdp(gaussian_rdp(sigma, true_events), a.delta)
    target_met = accounted_epsilon <= a.epsilon + 1e-8
    if a.mode == 'batch_explore':
        print('!!! EXPLORATORY ONLY: batch noise uses ONE event per round in calibration')
        print('!!! This DOES NOT meet requested epsilon budget under full DP-SGD composition.')
        print(f'!!! Requested epsilon={a.epsilon:.3f}; conservatively composed epsilon={accounted_epsilon:.3f}')
    else:
        print('Accountant: non-subsampled Gaussian RDP for the modeled client messages only.')
    print(f'Fixed LIIC | {a.mode} | full local epochs={a.epochs} | batch={a.batch_size} | NO 5-step cap')
    print(f'Clients={len(active_clients)}/{a.clients} | horizon={a.horizon} | planned max events/client={true_events}')
    print(f'Privacy target epsilon={a.epsilon:g}, delta={a.delta:g} | sigma={sigma:.6f} | modeled epsilon={accounted_epsilon:.4f} | target_met={target_met}')
    out = a.output_root / a.mode / time.strftime('%Y%m%d_%H%M%S')
    out.mkdir(parents=True, exist_ok=False)
    details = {
        'mode': a.mode, 'privacy_target_met_under_modeled_mechanism': target_met,
        'requested_epsilon': a.epsilon, 'delta': a.delta,
        'sigma': sigma, 'assumed_events_for_calibration': assumed_events,
        'actual_horizon_accounted_events_max_client': true_events,
        'modeled_epsilon_full_horizon': accounted_epsilon,
        'max_batch_events_per_client_per_round': max_events_in_round,
        'rounds_run': a.rounds, 'horizon': a.horizon, 'epochs': a.epochs,
        'batch_size': a.batch_size, 'gradient_clip': a.grad_clip,
        'final_update_clip': a.update_clip, 'learning_rate': a.lr,
        'client_train_counts_public_assumption': counts,
        'dataset': 'synthetic-smoke' if a.synthetic_smoke else 'FashionMNIST',
        'data_partition': 'fixed edge_label_skew, 0.2 local validation held out',
        'runtime_note': 'fixed LIIC FedAvg standalone; not dynamic-selection DynFL main path',
        'dp_scope': 'client released vector only, conditional on public roster/weights; no formal end-to-end audit',
        'rng_note': 'torch pseudo-RNG for research; DP randomness secrecy/security not audited',
        'test_note': 'evaluation uses separate FMNIST test set; private training loss not logged',
        'sampling_amplification': 'none',
        'he': 'not used: individual DP updates visible to aggregator',
    }
    (out / 'mechanism.json').write_text(json.dumps(details, indent=2, ensure_ascii=False), encoding='utf-8')
    model = build_full_model('lenet5', device, 1, 28, 10)
    global_state = {name: t.detach().cpu().clone() for name, t in model.state_dict().items()}
    param_names = [name for name, param in model.named_parameters() if param.requires_grad]
    total_n = sum(counts[i] for i in active_clients)
    # Independent noise per client, not a trusted/cryptographic secure aggregation simulation.
    noise_gen = torch.Generator(device=device if a.mode != 'final_update' else 'cpu')
    noise_gen.manual_seed(secrets.randbits(63))
    metrics = out / 'round_metrics.csv'
    with metrics.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=['round', 'test_accuracy_pct', 'test_loss',
                                                'optimizer_steps', 'active_clients', 'elapsed_seconds'])
        writer.writeheader()
        for round_no in range(1, a.rounds + 1):
            start = time.monotonic()
            aggregated = {name: torch.zeros_like(global_state[name], dtype=torch.float32)
                          for name in param_names}
            total_steps = 0
            for cid in active_clients:
                delta, steps = local_delta(
                    model, global_state, x_train[train_parts[cid]], y_train[train_parts[cid]],
                    epochs=a.epochs, batch=a.batch_size, lr=a.lr, l2=a.weight_decay,
                    train_seed=a.seed + round_no * 1000003 + cid * 97,
                    mode=a.mode, clip=a.update_clip if a.mode == 'final_update' else a.grad_clip,
                    sigma=sigma, noise_generator=noise_gen, device=device)
                total_steps += steps
                weight = counts[cid] / total_n
                for name in param_names:
                    aggregated[name].add_(delta[name], alpha=weight)
            for name in param_names:
                global_state[name] = global_state[name] + aggregated[name]
            accuracy, loss = test_accuracy(model, global_state, x_test, y_test, device=device)
            elapsed = time.monotonic() - start
            row = {'round': round_no, 'test_accuracy_pct': round(accuracy, 4),
                   'test_loss': round(loss, 6), 'optimizer_steps': total_steps,
                   'active_clients': len(active_clients), 'elapsed_seconds': round(elapsed, 2)}
            writer.writerow(row)
            fh.flush()
            print(f"{round_no:03d}/{a.horizon:03d}  Acc={accuracy:6.2f}%  TestLoss={loss:11.5f}  Steps={total_steps:5d}  {elapsed:6.1f}s", flush=True)
            if not math.isfinite(loss) or loss > 100.0:
                print('Test loss exceeded safety threshold or is non-finite: stopping early. Do not extend this run.')
                break
    print('Output:', out)


if __name__ == '__main__':
    main()
