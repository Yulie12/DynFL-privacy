from __future__ import annotations

import time
from collections import OrderedDict
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class LeNet5(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, padding=2)  # 28->28 (with pad)
        self.pool1 = nn.AvgPool2d(2)                 # 28->14
        self.conv2 = nn.Conv2d(6, 16, 5)             # 14->10
        self.pool2 = nn.AvgPool2d(2)                 # 10->5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def _numpy_to_tensor(
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_t = torch.from_numpy(x).float().to(device).view(-1, 1, 28, 28)
    y_t = torch.from_numpy(y).long().to(device)
    return x_t, y_t


def init_model(device: torch.device) -> LeNet5:
    return LeNet5().to(device)


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def flatten_state(model: nn.Module) -> np.ndarray:
    params = []
    for p in model.parameters():
        params.append(p.data.cpu().numpy().ravel())
    return np.concatenate(params)


def local_train_lenet5(
    global_model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    x_t, y_t = _numpy_to_tensor(x, y, device)
    dataset = torch.utils.data.TensorDataset(x_t, y_t)
    loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

    model = LeNet5().to(device)
    model.load_state_dict(global_model.state_dict())
    model.train()

    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)

    for _ in range(epochs):
        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            output = model(batch_x)
            loss = F.cross_entropy(output, batch_y)
            loss.backward()
            optimizer.step()

    state_diff = OrderedDict()
    for (name, param), (_, global_param) in zip(
        model.named_parameters(), global_model.named_parameters()
    ):
        state_diff[name] = param.data - global_param.data

    return state_diff


def apply_privacy_to_state(
    state_diff: dict[str, torch.Tensor],
    mechanism: str,
    clip_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator,
    device: torch.device,
    epsilon: float = 4.0,
) -> dict[str, torch.Tensor]:
    if mechanism == "none":
        return state_diff

    total_norm_sq = sum((diff ** 2).sum().item() for diff in state_diff.values())
    flat_norm = np.sqrt(max(total_norm_sq, 1e-12))
    scale = min(1.0, clip_norm / flat_norm)

    result = {}
    for name, diff in state_diff.items():
        diff = diff * scale
        if mechanism == "dp":
            sigma = noise_multiplier * clip_norm / max(float(epsilon), 1e-6)
            noise = torch.from_numpy(
                rng.normal(0.0, sigma, size=diff.shape).astype(np.float32)
            ).to(device)
            diff = diff + noise
        result[name] = diff

    return result


def fedavg_states(
    state_diffs: list[dict[str, torch.Tensor]],
    sample_counts: list[int],
    global_model: nn.Module,
    device: torch.device,
) -> nn.Module:
    total = max(1, sum(sample_counts))
    agg = OrderedDict()
    for name, param in global_model.named_parameters():
        agg[name] = torch.zeros_like(param.data, device=device)

    for diff, count in zip(state_diffs, sample_counts):
        factor = count / total
        for name in agg:
            agg[name] += factor * diff[name]

    for name, param in global_model.named_parameters():
        param.data += agg[name]

    return global_model


def evaluate_lenet5(
    model: nn.Module,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
) -> tuple[float, float]:
    x_t, y_t = _numpy_to_tensor(x, y, device)
    model.eval()
    with torch.no_grad():
        output = model(x_t)
        loss = F.cross_entropy(output, y_t)
        pred = output.argmax(dim=1)
        acc = (pred == y_t).float().mean().item()
    return float(loss.item()), acc
