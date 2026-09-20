import pytest

pytestmark = pytest.mark.skip(reason="legacy Method2/mainline_fusion coverage archived by Q75; not part of the formal DynFL mainline")

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from dynfed.fmnist_lenet5_dynamic import (
    Lenet5Config, _run_client_training_tasks, _client_step_limit,
)
from dynfed.selection import SelectionConfig
import dynfed.split_learning as sl


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = torch.nn.Linear(4, 10)

    def forward(self, x):
        return self.fc(x.flatten(1))


@pytest.mark.parametrize("mode,expected_steps", [("LIIC", 9), ("LIEIIIC", 27), ("LIIEIIIC", 27)])
def test_actual_worker_completes_every_batch_of_every_epoch(monkeypatch, mode, expected_steps):
    monkeypatch.setattr(sl, "build_full_model", lambda *a, **kw: Head())
    monkeypatch.setattr(sl, "build_split_pair", lambda *a, **kw: (torch.nn.Identity(), Head()))
    step_calls = []
    original = torch.optim.SGD.step
    def record_step(self, *args, **kwargs):
        step_calls.append(1)
        return original(self, *args, **kwargs)
    monkeypatch.setattr(torch.optim.SGD, "step", record_step)
    selection = SelectionConfig(mainline_fusion=True, trusted_edge_split_execution=True,
                                update_protection_goal="released_model_dp", L_block_cycles=5)
    candidate = SimpleNamespace(mode=mode, global_release_required=True)
    # 130 examples give 3 batches per CPU epoch. The old limit also subsampled
    # large cohorts, so verify complete epochs rather than just a 9-epoch label.
    x = np.random.default_rng(42).normal(size=(130, 1, 2, 2)).astype(np.float32)
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        result = _run_client_training_tasks(
            tasks=[(0, candidate, np.arange(130), 0)],
            train_config=Lenet5Config(local_epochs=3), selection=selection,
            global_end=torch.nn.Identity(), global_edge=Head(), client_model_states={},
            x_train=x, y_train=np.arange(130) % 10, device=torch.device("cpu"),
            model_name="resnet18_pretrained_head", input_shape=(1, 2, 2), num_classes=10,
            np_rng=np.random.default_rng(42), round_idx=0,
        )[0]
    finally:
        torch.set_num_threads(threads)
    assert len(step_calls) == expected_steps
    assert result["actual_local_batches"] == expected_steps
    assert result["actual_optimizer_steps"] == expected_steps
    assert result["finite"]


def test_legacy_step_limit_is_preserved():
    assert _client_step_limit(SelectionConfig(L_block_cycles=5)) == 5
