from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_torchvision_resnet(version: str, pretrained: bool = True) -> nn.Module:
    try:
        import torchvision.models as models
    except Exception as exc:  # pragma: no cover - depends on local optional package
        raise RuntimeError("torchvision is required for pretrained ResNet models") from exc

    if version == "18":
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        return models.resnet18(weights=weights)
    if version == "50":
        weights = models.ResNet50_Weights.DEFAULT if pretrained else None
        return models.resnet50(weights=weights)
    raise ValueError(f"Unsupported torchvision ResNet version: {version}")


class EndNet(nn.Module):
    """End-side: conv layers, output 'emb' (400-dim)."""
    def __init__(self, input_channels: int = 1, image_size: int = 28) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5, padding=2)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(2)
        conv1_size = image_size
        pool1_size = conv1_size // 2
        conv2_size = pool1_size - 4
        pool2_size = conv2_size // 2
        self.embedding_dim = 16 * pool2_size * pool2_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        return x.view(x.size(0), -1)


class EdgeNet(nn.Module):
    """Edge/cloud-side: fc layers."""
    def __init__(self, embedding_dim: int = 400, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(emb))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class FullNet(nn.Module):
    """Full LeNet5 for non-split modes (flat keys match {end_state, edge_state})."""
    def __init__(self, input_channels: int = 1, image_size: int = 28, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 6, 5, padding=2)
        self.pool1 = nn.AvgPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.AvgPool2d(2)
        conv1_size = image_size
        pool1_size = conv1_size // 2
        conv2_size = pool1_size - 4
        pool2_size = conv2_size // 2
        self.fc1 = nn.Linear(16 * pool2_size * pool2_size, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class SmallCNNEnd(nn.Module):
    def __init__(self, input_channels: int = 1, image_size: int = 28) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.embedding_dim = 32 * (image_size // 4) * (image_size // 4)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        return x.view(x.size(0), -1)


class SmallCNNEdge(nn.Module):
    def __init__(self, embedding_dim: int = 32 * 7 * 7, num_classes: int = 10) -> None:
        super().__init__()
        self.fc1 = nn.Linear(embedding_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.relu(self.fc1(emb)))


class SmallCNNFull(nn.Module):
    def __init__(self, input_channels: int = 1, image_size: int = 28, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(32 * (image_size // 4) * (image_size // 4), 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return self.fc2(F.relu(self.fc1(x)))


class DriftRaceAvgCNNEnd(nn.Module):
    def __init__(self, input_channels: int = 1, image_size: int = 28) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2)
        conv1_size = image_size - 4
        pool1_size = conv1_size // 2
        conv2_size = pool1_size - 4
        pool2_size = conv2_size // 2
        self.fc1 = nn.Linear(64 * pool2_size * pool2_size, 512)
        self.embedding_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        return F.relu(self.fc1(x))


class DriftRaceAvgCNNEdge(nn.Module):
    def __init__(self, embedding_dim: int = 512, num_classes: int = 10) -> None:
        super().__init__()
        self.classifier = nn.Linear(embedding_dim, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.classifier(emb)


class DriftRaceAvgCNNFull(nn.Module):
    def __init__(self, input_channels: int = 1, image_size: int = 28, num_classes: int = 10) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, 5)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.pool2 = nn.MaxPool2d(2)
        conv1_size = image_size - 4
        pool1_size = conv1_size // 2
        conv2_size = pool1_size - 4
        pool2_size = conv2_size // 2
        self.fc1 = nn.Linear(64 * pool2_size * pool2_size, 512)
        self.classifier = nn.Linear(512, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.classifier(x)


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.GroupNorm(4, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


def _group_norm(channels: int) -> nn.GroupNorm:
    for groups in (32, 16, 8, 4, 2):
        if channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class CifarBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False),
                _group_norm(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)))
        out = self.norm2(self.conv2(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CifarBottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        super().__init__()
        expanded = out_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, out_channels, 1, bias=False)
        self.norm1 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, stride=stride, padding=1, bias=False)
        self.norm2 = _group_norm(out_channels)
        self.conv3 = nn.Conv2d(out_channels, expanded, 1, bias=False)
        self.norm3 = _group_norm(expanded)
        if stride != 1 or in_channels != expanded:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, expanded, 1, stride=stride, bias=False),
                _group_norm(expanded),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.norm1(self.conv1(x)))
        out = F.relu(self.norm2(self.conv2(out)))
        out = self.norm3(self.conv3(out))
        out = out + self.shortcut(x)
        return F.relu(out)


class CifarResNetFeatures(nn.Module):
    def __init__(
        self,
        block: type[nn.Module],
        layers: list[int],
        input_channels: int = 3,
        base_channels: int = 64,
    ) -> None:
        super().__init__()
        self.in_channels = base_channels
        self.conv_in = nn.Conv2d(input_channels, base_channels, 3, stride=1, padding=1, bias=False)
        self.norm_in = _group_norm(base_channels)
        self.layer1 = self._make_layer(block, base_channels, layers[0], stride=1)
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.embedding_dim = base_channels * 8 * getattr(block, "expansion", 1)
        self._init_weights()

    def _make_layer(self, block: type[nn.Module], out_channels: int, blocks: int, stride: int) -> nn.Sequential:
        strides = [stride] + [1] * (blocks - 1)
        layers = []
        for block_stride in strides:
            layers.append(block(self.in_channels, out_channels, block_stride))
            self.in_channels = out_channels * getattr(block, "expansion", 1)
        return nn.Sequential(*layers)

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, 0, 0.01)
                nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.norm_in(self.conv_in(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class ResNet18End(CifarResNetFeatures):
    def __init__(self, input_channels: int = 3, image_size: int = 32) -> None:
        super().__init__(CifarBasicBlock, [2, 2, 2, 2], input_channels=input_channels)


class ResNet50End(CifarResNetFeatures):
    def __init__(self, input_channels: int = 3, image_size: int = 32) -> None:
        super().__init__(CifarBottleneck, [3, 4, 6, 3], input_channels=input_channels)


class ResNetEdge(nn.Module):
    def __init__(self, embedding_dim: int = 512, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


class ResNet18Full(ResNet18End):
    def __init__(self, input_channels: int = 3, image_size: int = 32, num_classes: int = 10) -> None:
        super().__init__(input_channels=input_channels, image_size=image_size)
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(super().forward(x))


class ResNet50Full(ResNet50End):
    def __init__(self, input_channels: int = 3, image_size: int = 32, num_classes: int = 10) -> None:
        super().__init__(input_channels=input_channels, image_size=image_size)
        self.fc = nn.Linear(self.embedding_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(super().forward(x))


class TorchvisionResNetEnd(nn.Module):
    version = "18"

    def __init__(self, input_channels: int = 3, image_size: int = 32) -> None:
        super().__init__()
        base = _make_torchvision_resnet(self.version, pretrained=True)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.embedding_dim = 512 if self.version == "18" else 2048

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        return self.layer2(x)


class TorchvisionResNet18End(TorchvisionResNetEnd):
    version = "18"


class TorchvisionResNet50End(TorchvisionResNetEnd):
    version = "50"


class TorchvisionResNetEdge(nn.Module):
    version = "18"

    def __init__(self, embedding_dim: int = 512, num_classes: int = 10) -> None:
        super().__init__()
        base = _make_torchvision_resnet(self.version, pretrained=True)
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.classifier = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        x = self.layer3(emb)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class TorchvisionResNet18Edge(TorchvisionResNetEdge):
    version = "18"


class TorchvisionResNet50Edge(TorchvisionResNetEdge):
    version = "50"


class TorchvisionResNetFull(nn.Module):
    version = "18"

    def __init__(self, input_channels: int = 3, image_size: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        base = _make_torchvision_resnet(self.version, pretrained=True)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool
        self.classifier = nn.Linear(base.fc.in_features, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] == 1:
            x = x.broadcast_to(x.shape[0], 3, *x.shape[2:])
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


class TorchvisionResNet18Full(TorchvisionResNetFull):
    version = "18"


class TorchvisionResNet50Full(TorchvisionResNetFull):
    version = "50"


class TinyResNetEnd(nn.Module):
    def __init__(self, input_channels: int = 1, image_size: int = 28) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn_in = nn.GroupNorm(4, 32)
        self.block1 = ResidualBlock(32)
        self.block2 = ResidualBlock(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        return x.view(x.size(0), -1)


class TinyResNetEdge(nn.Module):
    def __init__(self, embedding_dim: int = 32, num_classes: int = 10) -> None:
        super().__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, emb: torch.Tensor) -> torch.Tensor:
        return self.fc(emb)


class TinyResNetFull(nn.Module):
    def __init__(self, input_channels: int = 1, image_size: int = 28, num_classes: int = 10) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(input_channels, 32, 3, padding=1)
        self.bn_in = nn.GroupNorm(4, 32)
        self.block1 = ResidualBlock(32)
        self.block2 = ResidualBlock(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn_in(self.conv_in(x)))
        x = self.block1(x)
        x = self.block2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


MODEL_BUILDERS = {
    "lenet5": (EndNet, EdgeNet, FullNet),
    "smallcnn": (SmallCNNEnd, SmallCNNEdge, SmallCNNFull),
    "avgcnn": (DriftRaceAvgCNNEnd, DriftRaceAvgCNNEdge, DriftRaceAvgCNNFull),
    "tinyresnet": (TinyResNetEnd, TinyResNetEdge, TinyResNetFull),
    "resnet18": (ResNet18End, ResNetEdge, ResNet18Full),
    "resnet50": (ResNet50End, ResNetEdge, ResNet50Full),
    "resnet18pretrained": (TorchvisionResNet18End, TorchvisionResNet18Edge, TorchvisionResNet18Full),
    "resnet18pretrainedhead": (TorchvisionResNet18End, TorchvisionResNet18Edge, TorchvisionResNet18Full),
    "resnet50pretrained": (TorchvisionResNet50End, TorchvisionResNet50Edge, TorchvisionResNet50Full),
}


def normalize_model_name(model_name: str) -> str:
    value = str(model_name).strip().lower().replace("-", "").replace("_", "")
    aliases = {
        "lenet": "lenet5",
        "lenet5": "lenet5",
        "cnn": "smallcnn",
        "smallcnn": "smallcnn",
        "avgcnn": "avgcnn",
        "fedavgcnn": "avgcnn",
        "driftraceavgcnn": "avgcnn",
        "resnet": "tinyresnet",
        "tinyresnet": "tinyresnet",
        "resnet8": "tinyresnet",
        "res18": "resnet18",
        "resnet18": "resnet18",
        "cifarresnet18": "resnet18",
        "res50": "resnet50",
        "resnet50": "resnet50",
        "cifarresnet50": "resnet50",
        "res18pretrained": "resnet18pretrained",
        "resnet18pretrained": "resnet18pretrained",
        "resnet18pretrainedhead": "resnet18pretrainedhead",
        "torchvisionresnet18": "resnet18pretrained",
        "driftraceres18": "resnet18pretrained",
        "res50pretrained": "resnet50pretrained",
        "resnet50pretrained": "resnet50pretrained",
        "torchvisionresnet50": "resnet50pretrained",
        "driftraceres50": "resnet50pretrained",
    }
    if value not in aliases:
        raise ValueError(f"Unsupported split model {model_name!r}. Choose one of {sorted(MODEL_BUILDERS)}.")
    return aliases[value]


def build_split_models(
    model_name: str,
    device: torch.device,
    input_channels: int = 1,
    image_size: int = 28,
    num_classes: int = 10,
) -> tuple[nn.Module, nn.Module, nn.Module]:
    end_model, edge_model = build_split_pair(
        model_name,
        device,
        input_channels=input_channels,
        image_size=image_size,
        num_classes=num_classes,
    )
    full_model = build_full_model(
        model_name,
        device,
        input_channels=input_channels,
        image_size=image_size,
        num_classes=num_classes,
    )
    return end_model, edge_model, full_model


def build_split_pair(
    model_name: str,
    device: torch.device,
    input_channels: int = 1,
    image_size: int = 28,
    num_classes: int = 10,
) -> tuple[nn.Module, nn.Module]:
    end_cls, edge_cls, _full_cls = MODEL_BUILDERS[normalize_model_name(model_name)]
    end_model = end_cls(input_channels=input_channels, image_size=image_size).to(device)
    edge_model = edge_cls(embedding_dim=getattr(end_model, "embedding_dim", 32), num_classes=num_classes).to(device)
    _prepare_model_for_training(end_model, model_name)
    _prepare_model_for_training(edge_model, model_name)
    return end_model, edge_model


def build_full_model(
    model_name: str,
    device: torch.device,
    input_channels: int = 1,
    image_size: int = 28,
    num_classes: int = 10,
) -> nn.Module:
    _end_cls, _edge_cls, full_cls = MODEL_BUILDERS[normalize_model_name(model_name)]
    model = full_cls(
        input_channels=input_channels,
        image_size=image_size,
        num_classes=num_classes,
    ).to(device)
    _prepare_model_for_training(model, model_name)
    return model


REAL_OBJECT_SIZES = {
    "emb": 400.0,
    "grad": 400.0,
    "label": 1.0,
    "upd": 61706.0,
    "weakemb": 120.0,
    "strongemb": 400.0,
    "pseudo_label": 1.0,
}


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def _is_pretrained_resnet(model_name: str) -> bool:
    return normalize_model_name(model_name) in {"resnet18pretrained", "resnet50pretrained", "resnet18pretrainedhead"}


def _prepare_model_for_training(model: nn.Module, model_name: str) -> None:
    if not _is_pretrained_resnet(model_name):
        return
    for name, param in model.named_parameters():
        trainable = (
            name.startswith("layer3.")
            or name.startswith("layer4.")
            or name.startswith("classifier.")
            or name.startswith("fc.")
        )
        param.requires_grad_(trainable)
        if normalize_model_name(model_name) == "resnet18pretrainedhead":
            param.requires_grad_(name.startswith(("classifier.", "fc.")))
    for module in model.modules():
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            module.eval()
            for param in module.parameters(recurse=False):
                param.requires_grad_(False)


def _make_optimizer(
    model: nn.Module,
    lr: float,
    model_name: str,
    weight_decay: float | None = None,
) -> torch.optim.Optimizer | None:
    trainable = [param for param in model.parameters() if param.requires_grad]
    if not trainable:
        return None
    normalized = normalize_model_name(model_name)
    if normalized in {"resnet18", "resnet50", "resnet18pretrained", "resnet50pretrained", "resnet18pretrainedhead"}:
        decay = 5e-4 if weight_decay is None else float(weight_decay)
        return torch.optim.SGD(trainable, lr=lr, momentum=0.9, weight_decay=decay)
    decay = 0.0 if weight_decay is None else float(weight_decay)
    return torch.optim.SGD(trainable, lr=lr, momentum=0.0, weight_decay=decay)


def _training_batches(
    loader: torch.utils.data.DataLoader,
    epochs: int,
    local_steps: int | None,
):
    if local_steps is None:
        for _ in range(max(0, int(epochs))):
            yield from loader
        return

    step_limit = max(0, int(local_steps))
    completed = 0
    for _ in range(max(0, int(epochs))):
        for batch in loader:
            if completed >= step_limit:
                return
            yield batch
            completed += 1


def split_local_train_lenet5(
    mode: str,
    global_end_state: dict[str, torch.Tensor],
    global_edge_state: dict[str, torch.Tensor],
    x: np.ndarray,
    y: np.ndarray,
    epochs: int,
    lr: float,
    device: torch.device,
    model_name: str = "lenet5",
    input_shape: tuple[int, int, int] = (1, 28, 28),
    num_classes: int = 10,
    mechanisms: dict[str, str] | None = None,
    dp_clip_norm: float = 1.0,
    dp_noise_multiplier: float = 0.0002,
    dp_rng: np.random.Generator | None = None,
    dp_epsilon: float = 1.0,
    l2: float | None = None,
    local_steps: int | None = None,
    training_seed: int | None = None,
    model_cache: dict[str, Any] | None = None,
) -> dict[str, dict[str, torch.Tensor]]:
    """Split learning with labels retained at the client.

    Returns:
        {"end": end_state_diff, "edge": edge_state_diff}
        where each is a state_dict of parameter differences.
        For no-split modes, only "end" contains the full model diff.
    """
    batch_size = 128 if device.type == "cuda" and normalize_model_name(model_name) in {
        "resnet18pretrainedhead",
        "resnet18",
        "resnet50",
        "resnet18pretrained",
        "resnet50pretrained",
    } else 64
    if local_steps is not None and len(x) > 0:
        max_local_samples = max(1, int(local_steps)) * batch_size
        if len(x) > max_local_samples:
            sample_rng = np.random.default_rng(training_seed)
            sample_indices = sample_rng.choice(len(x), size=max_local_samples, replace=False)
            x = x[sample_indices]
            y = y[sample_indices]
    x_t = torch.from_numpy(x).float().to(device).view(-1, *input_shape)
    y_t = torch.from_numpy(y).long().to(device)
    dataset = torch.utils.data.TensorDataset(x_t, y_t)
    loader_generator = None
    if training_seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(training_seed))
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=loader_generator,
    )
    mechanisms = mechanisms or {}
    if dp_rng is None:
        dp_rng = np.random.default_rng()

    # Whether training uses split (end+edge) or full (end only) mode
    is_split = mode in ("LIE", "LIC", "LIEIIC", "LIEIIIC")

    if not is_split:
        # Full model: merge end + edge states
        model = None if model_cache is None else model_cache.get("full")
        if model is None:
            model = build_full_model(
                model_name,
                device,
                input_channels=input_shape[0],
                image_size=input_shape[1],
                num_classes=num_classes,
            )
            if model_cache is not None:
                model_cache["full"] = model
        full_state = {**global_end_state, **global_edge_state}
        model.load_state_dict(full_state)
        model.train()
        _prepare_model_for_training(model, model_name)
        opt = _make_optimizer(model, lr, model_name, weight_decay=l2)

        for bx, by in _training_batches(loader, epochs, local_steps):
            if opt is None:
                continue
            opt.zero_grad()
            loss = F.cross_entropy(model(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0)
            opt.step()

        diff = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                diff[name] = param.data - full_state[name]

        # Split full diff into end and edge parts for uniform aggregation
        end_keys = set(global_end_state.keys())
        end_diff = {k: v for k, v in diff.items() if k in end_keys}
        edge_diff = {k: v for k, v in diff.items() if k not in end_keys}
        return {"end": end_diff, "edge": edge_diff}

    # Split learning: end computes emb, edge continues
    cached_split = None if model_cache is None else model_cache.get("split")
    if cached_split is None:
        end, edge = build_split_pair(
            model_name,
            device,
            input_channels=input_shape[0],
            image_size=input_shape[1],
            num_classes=num_classes,
        )
        if model_cache is not None:
            model_cache["split"] = (end, edge)
    else:
        end, edge = cached_split
    end.load_state_dict(global_end_state)
    edge.load_state_dict(global_edge_state)
    end.train()
    edge.train()
    _prepare_model_for_training(end, model_name)
    _prepare_model_for_training(edge, model_name)

    end_opt = _make_optimizer(end, lr, model_name, weight_decay=l2)
    edge_opt = _make_optimizer(edge, lr, model_name, weight_decay=l2)

    for bx, by in _training_batches(loader, epochs, local_steps):
        if end_opt is not None:
            end_opt.zero_grad()
        if edge_opt is not None:
            edge_opt.zero_grad()

        # The remote side returns logits. The client keeps labels local and sends
        # a protected loss gradient back to the remote model.
        emb = end(bx)
        transmitted_emb = _protect_tensor_dp(
            emb,
            mechanisms.get("emb", "none"),
            dp_clip_norm,
            dp_noise_multiplier,
            dp_rng,
            device,
            dp_epsilon,
        )
        edge_input = transmitted_emb.detach().requires_grad_(True)
        logits = edge(edge_input)
        client_logits = logits.detach().requires_grad_(True)
        client_loss = F.cross_entropy(client_logits, by, reduction="sum")
        label_grad = torch.autograd.grad(client_loss, client_logits)[0]
        protected_label_grad = _protect_batched_average_gradient_dp(
            label_grad,
            mechanisms.get("grad", "none"),
            dp_clip_norm,
            dp_noise_multiplier,
            dp_rng,
            device,
            dp_epsilon,
        )
        logits.backward(protected_label_grad)
        grad_to_end = edge_input.grad.detach()

        edge_trainable = [p for p in edge.parameters() if p.requires_grad]
        if edge_opt is not None and edge_trainable:
            torch.nn.utils.clip_grad_norm_(edge_trainable, max_norm=5.0)
            edge_opt.step()

        end_trainable = [p for p in end.parameters() if p.requires_grad]
        if end_opt is not None and end_trainable and emb.requires_grad:
            transmitted_emb.backward(grad_to_end)
            torch.nn.utils.clip_grad_norm_(end_trainable, max_norm=5.0)
            end_opt.step()

    end_diff = {
        name: param.data - global_end_state[name]
        for name, param in end.named_parameters() if param.requires_grad
    }
    edge_diff = {
        name: param.data - global_edge_state[name]
        for name, param in edge.named_parameters() if param.requires_grad
    }
    return {"end": end_diff, "edge": edge_diff}


def _protect_batched_average_gradient_dp(
    tensor: torch.Tensor,
    mechanism: str,
    clip_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator,
    device: torch.device,
    epsilon: float = 1.0,
) -> torch.Tensor:
    """Release an averaged output gradient, with DP only when requested."""
    batch_size = max(1, int(tensor.shape[0]))
    if mechanism != "dp":
        return tensor / float(batch_size)
    flat = tensor.detach().reshape(batch_size, -1)
    norms = torch.linalg.vector_norm(flat, ord=2, dim=1, keepdim=True)
    scales = torch.clamp(float(clip_norm) / (norms + 1e-12), max=1.0)
    clipped = (flat * scales).reshape_as(tensor) / float(batch_size)
    std = 2.0 * float(clip_norm) * float(noise_multiplier) / float(batch_size)
    noise = rng.normal(0.0, std, size=tuple(tensor.shape)).astype(np.float32)
    return clipped + torch.from_numpy(noise).to(device=device, dtype=tensor.dtype)


def split_evaluate(
    end_model: EndNet,
    edge_model: EdgeNet,
    x: np.ndarray,
    y: np.ndarray,
    device: torch.device,
    input_shape: tuple[int, int, int] = (1, 28, 28),
    batch_size: int = 128,
) -> tuple[float, float]:
    if len(x) == 0:
        return 0.0, 0.0
    end_model.eval()
    edge_model.eval()
    total_loss = 0.0
    total_correct = 0
    total_seen = 0
    with torch.no_grad():
        for start in range(0, len(x), max(1, int(batch_size))):
            stop = min(start + max(1, int(batch_size)), len(x))
            x_t = torch.from_numpy(x[start:stop]).float().to(device).view(-1, *input_shape)
            y_t = torch.from_numpy(y[start:stop]).long().to(device)
            logits = edge_model(end_model(x_t))
            loss = F.cross_entropy(logits, y_t)
            count = int(stop - start)
            total_loss += float(loss.item()) * count
            total_correct += int((logits.argmax(dim=1) == y_t).sum().item())
            total_seen += count
    return total_loss / total_seen, total_correct / total_seen


def apply_unified_dp(
    diff: dict[str, dict[str, torch.Tensor]],
    mechanism: str,
    clip_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator,
    device: torch.device,
    adaptive: bool = False,
) -> dict[str, dict[str, torch.Tensor]]:
    """Apply client replacement DP to one model update release.

    The complete update is globally clipped to ``clip_norm``. Under
    replacement adjacency its sensitivity is ``2 * clip_norm``.
    """
    if mechanism not in {"dp"}:
        return diff
    if adaptive:
        raise ValueError(
            "Per-tensor clipping has no single global replacement sensitivity."
        )

    # Two globally clipped updates can differ by at most 2C under replacement
    # adjacency. The RDP ledger treats noise_multiplier as sigma / sensitivity.
    sigma = noise_multiplier * 2.0 * clip_norm

    total_norm_sq = 0.0
    for part_key in diff:
        for value in diff[part_key].values():
            if not (torch.is_floating_point(value) or torch.is_complex(value)):
                continue
            total_norm_sq += (value ** 2).sum().item()
    flat_norm = np.sqrt(max(total_norm_sq, 1e-12))
    scale = min(1.0, clip_norm / flat_norm)

    for part_key in diff:
        for name in diff[part_key]:
            value = diff[part_key][name]
            if not (torch.is_floating_point(value) or torch.is_complex(value)):
                continue
            protected = value * scale
            noise = torch.from_numpy(
                rng.normal(0.0, sigma, size=value.shape).astype(np.float32)
            ).to(device)
            diff[part_key][name] = protected + noise

    return diff


def clip_state_difference(
    diff: dict[str, dict[str, torch.Tensor]],
    clip_norm: float,
    device: torch.device,
) -> tuple[dict[str, dict[str, torch.Tensor]], float, float]:
    """Globally clip one complete client contribution without adding noise."""
    if not math.isfinite(float(clip_norm)) or float(clip_norm) <= 0.0:
        raise ValueError("clip_norm must be finite and positive")

    total_norm_sq = 0.0
    for values in diff.values():
        for value in values.values():
            if torch.is_floating_point(value) or torch.is_complex(value):
                magnitude = value.detach().abs().to(dtype=torch.float64)
                if not bool(torch.isfinite(magnitude).all()):
                    raise ValueError("Cannot clip a non-finite client update")
                total_norm_sq += float(torch.sum(magnitude.square()).item())
    if not math.isfinite(total_norm_sq):
        raise ValueError("Client update norm overflowed during clipping")
    original_norm = float(np.sqrt(max(total_norm_sq, 0.0)))
    scale = min(1.0, float(clip_norm) / max(original_norm, 1e-12))
    clipped: dict[str, dict[str, torch.Tensor]] = {}
    for part_name, values in diff.items():
        clipped[part_name] = {}
        for name, value in values.items():
            copied = value.detach().to(device).clone()
            if torch.is_floating_point(copied) or torch.is_complex(copied):
                copied.mul_(scale)
            clipped[part_name][name] = copied
    return clipped, original_norm, scale


def gaussian_state_difference(
    reference: dict[str, dict[str, torch.Tensor]],
    standard_deviation: float,
    rng: np.random.Generator,
    device: torch.device,
) -> dict[str, dict[str, torch.Tensor]]:
    """Generate one Gaussian perturbation matching a nested model update."""
    if not math.isfinite(float(standard_deviation)) or standard_deviation < 0.0:
        raise ValueError("standard_deviation must be finite and non-negative")
    noise: dict[str, dict[str, torch.Tensor]] = {}
    for part_name, values in reference.items():
        noise[part_name] = {}
        for name, value in values.items():
            if torch.is_floating_point(value) or torch.is_complex(value):
                array = rng.normal(
                    0.0,
                    float(standard_deviation),
                    size=tuple(value.shape),
                ).astype(np.float32)
                noise[part_name][name] = torch.from_numpy(array).to(
                    device=device,
                    dtype=value.dtype,
                )
            else:
                noise[part_name][name] = torch.zeros_like(value, device=device)
    return noise


def _protect_tensor_dp(
    tensor: torch.Tensor,
    mechanism: str,
    clip_norm: float,
    noise_multiplier: float,
    rng: np.random.Generator,
    device: torch.device,
    epsilon: float,
) -> torch.Tensor:
    if mechanism != "dp":
        return tensor

    if tensor.ndim >= 2:
        flat = tensor.detach().reshape(tensor.shape[0], -1)
        norms = torch.linalg.vector_norm(flat, dim=1).clamp_min(1e-12)
        scales = (float(clip_norm) / norms).clamp(max=1.0)
        view_shape = [tensor.shape[0]] + [1] * (tensor.ndim - 1)
        protected = tensor * scales.reshape(view_shape)
    else:
        norm = torch.linalg.vector_norm(tensor.detach())
        scale = min(1.0, float(clip_norm) / max(float(norm.item()), 1e-12))
        protected = tensor * scale
    # TeX Eq. (feature_dp_noise) uses replacement adjacency, Delta_z = 2C.
    # The formal guarantee is determined by this multiplier and the RDP event
    # ledger. ``epsilon`` remains only for compatibility with older callers and
    # must not be used as a second noise scaling factor.
    _ = epsilon
    sigma = noise_multiplier * 2.0 * clip_norm
    noise = torch.from_numpy(
        rng.normal(0.0, sigma, size=tuple(tensor.shape)).astype(np.float32)
    ).to(device)
    return protected + noise


def fedavg_split(
    state_diffs: list[dict[str, dict[str, torch.Tensor]]],
    sample_counts: list[float],
    global_end: EndNet,
    global_edge: EdgeNet,
    device: torch.device,
) -> tuple[EndNet, EdgeNet]:
    total = max(1, sum(sample_counts))

    # Aggregate end-side updates
    end_agg: dict[str, torch.Tensor] = {}
    for name, param in global_end.named_parameters():
        if param.requires_grad:
            end_agg[name] = torch.zeros_like(param.data, device=device)

    edge_agg: dict[str, torch.Tensor] = {}
    for name, param in global_edge.named_parameters():
        if param.requires_grad:
            edge_agg[name] = torch.zeros_like(param.data, device=device)

    has_edge = False
    for diff, count in zip(state_diffs, sample_counts):
        factor = count / total
        for name in end_agg:
            if name in diff.get("end", {}):
                update = diff["end"][name].to(device=end_agg[name].device, dtype=end_agg[name].dtype)
                end_agg[name] += factor * update
        if "edge" in diff:
            has_edge = True
            for name in edge_agg:
                if name in diff["edge"]:
                    update = diff["edge"][name].to(device=edge_agg[name].device, dtype=edge_agg[name].dtype)
                    edge_agg[name] += factor * update

    for name in end_agg:
        global_end.state_dict()[name].data += end_agg[name]
    if has_edge:
        for name in edge_agg:
            global_edge.state_dict()[name].data += edge_agg[name]

    return global_end, global_edge
