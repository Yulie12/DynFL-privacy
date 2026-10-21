"""Local datasets, fixed partitions and models used by the paper."""
import copy
import gzip
from pathlib import Path
import pickle
import numpy as np
import torch
from torch import nn


class LeNet5(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.Sequential(nn.Conv2d(1, 6, 5, padding=2), nn.Tanh(), nn.AvgPool2d(2),
                                    nn.Conv2d(6, 16, 5), nn.Tanh(), nn.AvgPool2d(2), nn.Flatten(),
                                    nn.Linear(16*5*5, 120), nn.Tanh(), nn.Linear(120, 84), nn.Tanh(),
                                    nn.Linear(84, 10))

    def forward(self, x):
        return self.layers(x)


def load_data(config, seed):
    root = Path(config.data_root)
    if config.dataset == "fmnist":
        def idx(name, offset, shape):
            path = root / name
            raw = path.read_bytes() if path.exists() else gzip.decompress((root / (name + ".gz")).read_bytes())
            return np.frombuffer(raw, dtype=np.uint8, offset=offset).copy().reshape(shape)
        train_x = idx("train-images-idx3-ubyte", 16, (-1, 1, 28, 28))
        train_y = idx("train-labels-idx1-ubyte", 8, (-1,))
        test_x = idx("t10k-images-idx3-ubyte", 16, (-1, 1, 28, 28))
        test_y = idx("t10k-labels-idx1-ubyte", 8, (-1,))
        mean, std = (0.286,), (0.3205,)
    else:
        folder = root / "cifar-10-batches-py"
        def batch(name):
            with (folder / name).open("rb") as handle:
                data = pickle.load(handle, encoding="bytes")
            return data[b"data"].reshape(-1, 3, 32, 32), np.asarray(data[b"labels"])
        parts = [batch(f"data_batch_{i}") for i in range(1, 6)]
        train_x = np.concatenate([part[0] for part in parts])
        train_y = np.concatenate([part[1] for part in parts])
        test_x, test_y = batch("test_batch")
        mean, std = (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
    rng = np.random.default_rng(seed)
    train_idx = rng.permutation(len(train_y))[:config.train_limit]
    test_idx = rng.permutation(len(test_y))[:config.test_limit]
    def tensor(x):
        value = torch.from_numpy(x.copy()).float() / 255
        return (value - torch.tensor(mean)[None, :, None, None]) / torch.tensor(std)[None, :, None, None]
    return (tensor(train_x[train_idx]), torch.tensor(train_y[train_idx]).long(),
            tensor(test_x[test_idx]), torch.tensor(test_y[test_idx]).long())


def partitions(labels, clients, edges, alpha, seed):
    """Edge-level label skew followed by deterministic within-Edge client splits."""
    rng = np.random.default_rng(seed)
    result = [[] for _ in range(clients)]
    edge_members = [[client for client in range(clients) if client % edges == edge] for edge in range(edges)]
    edge_indices = [[] for _ in range(edges)]
    for label in np.unique(labels):
        indices = np.flatnonzero(labels == label)
        rng.shuffle(indices)
        proportions = rng.dirichlet(np.full(edges, alpha))
        chunks = np.split(indices, (np.cumsum(proportions)[:-1] * len(indices)).astype(int))
        for edge, chunk in enumerate(chunks):
            edge_indices[edge].extend(chunk.tolist())
    for edge, indices in enumerate(edge_indices):
        rng.shuffle(indices)
        for client, chunk in zip(edge_members[edge], np.array_split(indices, len(edge_members[edge]))):
            result[client] = chunk.tolist()
    # A deterministic rebalance ensures nonempty clients without dropping data.
    for client in range(clients):
        if not result[client]:
            donor = max(range(clients), key=lambda key: len(result[key]))
            if len(result[donor]) <= 1:
                raise ValueError("not enough training samples for clients")
            result[client].append(result[donor].pop())
    return result


def make_model(config, seed):
    torch.manual_seed(seed)
    if config.model == "lenet5":
        return LeNet5().to(config.device)
    from torchvision.models import resnet18, ResNet18_Weights
    weights = ResNet18_Weights.DEFAULT
    filename = weights.url.rsplit("/", 1)[-1]
    cached = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not cached.is_file():
        raise FileNotFoundError(f"pretrained ResNet-18 weights must exist locally: {cached}")
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(cached, map_location="cpu", weights_only=True))
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.fc = nn.Linear(model.fc.in_features, 10)
    return model.to(config.device)


def vector(model):
    return torch.cat([p.detach().flatten().cpu() for p in model.parameters() if p.requires_grad])


def assign(model, values):
    cursor = 0
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.requires_grad:
                count = parameter.numel()
                parameter.copy_(values[cursor:cursor+count].view_as(parameter).to(parameter))
                cursor += count
    if cursor != values.numel():
        raise ValueError("trainable vector dimension mismatch")


def local_train(model, base, x, y, indices, config, sigma, ledger, seed):
    """Joint per-example gradient clipping, then one Gaussian mechanism per step.

    Replacement adjacency: sensitivity of the summed clipped gradients is 2C;
    noise std is 2*C*sigma. RDP deliberately claims no sampling amplification.
    Frozen feature extractors and their BatchNorm buffers stay in evaluation mode.
    """
    worker = copy.deepcopy(model)
    assign(worker, base)
    worker.eval()
    params = [p for p in worker.parameters() if p.requires_grad]
    rng = np.random.default_rng(seed)
    generator = torch.Generator(device=config.device).manual_seed(seed)
    if ledger.epsilon(config.local_steps, sigma) > ledger.budget + 1e-8:
        raise RuntimeError("local block failed privacy preflight")
    losses = []
    for _ in range(config.local_steps):
        selected = rng.permutation(indices)[:config.batch_size]
        summed = [torch.zeros_like(parameter) for parameter in params]
        for index in selected:
            logits = worker(x[index:index+1].to(config.device))
            loss = nn.functional.cross_entropy(logits, y[index:index+1].to(config.device))
            gradients = torch.autograd.grad(loss, params)
            norm = torch.sqrt(sum(gradient.square().sum() for gradient in gradients))
            scale = (config.clip_norm / (norm + 1e-12)).clamp(max=1.0)
            for accumulator, gradient in zip(summed, gradients):
                accumulator.add_(gradient.detach() * scale)
            losses.append(float(loss.detach()))
        with torch.no_grad():
            for parameter, accumulator in zip(params, summed):
                noise = torch.randn(parameter.shape, device=config.device, generator=generator)
                gradient = (accumulator + noise * (2 * config.clip_norm * sigma)) / len(selected)
                parameter.add_(gradient, alpha=-config.learning_rate)
        ledger.commit(1, sigma)
    result = vector(worker)
    if not torch.isfinite(result).all():
        raise FloatingPointError("nonfinite DP-SGD model")
    return result, sum(losses) / max(len(losses), 1)


@torch.no_grad()
def evaluate(model, x, y, config):
    model.eval()
    correct, loss = 0, 0.0
    for start in range(0, len(y), config.batch_size):
        target = y[start:start+config.batch_size].to(config.device)
        logits = model(x[start:start+config.batch_size].to(config.device))
        correct += int((logits.argmax(1) == target).sum())
        loss += float(nn.functional.cross_entropy(logits, target, reduction="sum"))
    return correct / len(y), loss / len(y)
