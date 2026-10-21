"""Configuration of the sole supported protocol: paper/main.tex."""
from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path

MODES = ("FL_E", "FL_EC", "FL_C")
CORE_POLICIES = ("latency_only", "joint_no_ema", "proposed")
POLICIES = ("proposed", "joint_no_ema", "latency_only", "fixed_fl_e", "fixed_fl_ec",
            "fixed_fl_c", "fixed_mixed", "random_feasible", "exhaustive")


@dataclass(frozen=True)
class PaperConfig:
    dataset: str = "fmnist"
    model: str = "lenet5"
    data_root: str = "experiments/data/fmnist/FashionMNIST/raw"
    device: str = "cpu"
    clients: int = 100
    edges: int = 10
    rounds: int = 100
    seeds: tuple[int, ...] = (42, 43, 44)
    policies: tuple[str, ...] = POLICIES[:8]
    train_limit: int = 60000
    test_limit: int = 10000
    batch_size: int = 64
    local_steps: int = 2
    edge_stages: int = 3
    learning_rate: float = 0.05
    clip_norm: float = 1.0
    epsilon: float = 8.0
    delta: float = 1e-5
    gamma_d: float = 0.9
    gamma_c: float = 0.9
    epsilon_j: float = 1e-12
    selection_period: int = 5
    evaluation_budget: int = 256
    archive_size: int = 16
    shared_init_rounds: int = 2
    protocol: str = "global_barrier_100"
    experiment_version: str = "shared_init_v1"
    cloud_quorum: float = 1.0
    acknowledgement_mb: float = 0.000064
    model_install_sec: float = 0.001
    gradient_backend: str = "vectorized"
    torch_threads: int = 1
    snapshot_interval: int = 1
    exhaustive_limit: int = 100000
    dirichlet_alpha: float = 0.5
    client_heterogeneity: float = 2.0
    network_jitter: float = 0.25
    local_step_sec: float = 0.1
    end_edge_mb_s: float = 5.0
    end_cloud_mb_s: float = 2.2
    edge_cloud_mb_s: float = 8.0
    end_edge_delay: float = 0.015
    end_cloud_delay: float = 0.04
    edge_cloud_delay: float = 0.01
    edge_aggregation_sec: float = 0.02
    cloud_aggregation_sec: float = 0.04
    switch_sec: float = 0.02
    required_memory_mb: float = 16.0
    client_memory_mb: float = 32.0
    required_compute: float = 1.0
    client_compute_capacity: float = 2.0
    edge_group_capacity: int = 100
    output_root: str = "out/paper"

    def __post_init__(self):
        for name in ("clients", "edges", "rounds", "train_limit", "test_limit", "batch_size",
                     "local_steps", "edge_stages", "selection_period", "evaluation_budget",
                     "archive_size", "exhaustive_limit", "edge_group_capacity", "torch_threads", "snapshot_interval"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.edges > self.clients or self.clients > self.train_limit:
            raise ValueError("require edges <= clients <= train_limit")
        for name in ("epsilon", "clip_norm", "learning_rate", "epsilon_j", "dirichlet_alpha",
                     "local_step_sec", "end_edge_mb_s", "end_cloud_mb_s", "edge_cloud_mb_s",
                     "required_memory_mb", "client_memory_mb", "client_heterogeneity",
                     "required_compute", "client_compute_capacity", "acknowledgement_mb"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be finite and positive")
        for name in ("end_edge_delay", "end_cloud_delay", "edge_cloud_delay",
                     "edge_aggregation_sec", "cloud_aggregation_sec", "switch_sec", "model_install_sec"):
            if not math.isfinite(getattr(self, name)) or getattr(self, name) < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if not 0 < self.delta < 1 or not 0 <= self.gamma_d < 1 or not 0 <= self.gamma_c < 1:
            raise ValueError("invalid delta or history smoothing")
        if not 0 <= self.network_jitter < 1 or self.client_heterogeneity < 1:
            raise ValueError("invalid network jitter or heterogeneity")
        if not self.seeds or not self.policies or len(set(self.policies)) != len(self.policies):
            raise ValueError("require seeds and unique policies")
        if set(self.policies) - set(POLICIES):
            raise ValueError("unsupported paper policy")
        if (self.dataset, self.model) not in {("fmnist", "lenet5"), ("cifar10", "resnet18_head")}:
            raise ValueError("supported tasks: fmnist/lenet5, cifar10/resnet18_head")
        if self.device not in {"cpu", "cuda"}:
            raise ValueError("device must be cpu or cuda")
        if self.protocol != "global_barrier_100" or self.cloud_quorum != 1.0:
            raise ValueError("only global_barrier_100 with cloud_quorum=1.0 is implemented; 90% is deferred")
        if self.shared_init_rounds != 2:
            raise ValueError("the registered shared initialization has exactly two charged rounds")
        if self.gradient_backend not in {"vectorized", "reference"}:
            raise ValueError("gradient_backend must be vectorized or reference")

    def to_dict(self):
        return asdict(self)

    @classmethod
    def load(cls, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        unknown = set(data) - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"unknown configuration fields: {sorted(unknown)}")
        for name in ("seeds", "policies"):
            if name in data:
                data[name] = tuple(data[name])
        return cls(**data)


def effective_config(config, policy, seed=None):
    """Explicit, hashable per-arm settings; reference EMA is not disabled."""
    from dataclasses import replace
    if policy not in POLICIES:
        raise ValueError("unsupported policy")
    settings = {"policies": (policy,)}
    if seed is not None:
        settings["seeds"] = (seed,)
    if policy == "joint_no_ema":
        settings.update(gamma_d=0., gamma_c=.9)
    return replace(config, **settings)
