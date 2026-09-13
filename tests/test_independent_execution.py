import numpy as np
import pytest
import torch
from torchvision.models import resnet18

from dynfed.independent_execution import planned_mode, train_independent
import dynfed.split_learning as sl


def test_public_schedule_and_unsupported_layout():
    assert planned_mode("alternating_audit", 0, 0) == "LIIC"
    assert planned_mode("alternating_audit", 0, 1) == "LIEIIC"
    with pytest.raises(ValueError):
        planned_mode("ours", 0, 0)


def test_split_full_equivalence_and_no_cached_private_state(monkeypatch):
    monkeypatch.setattr(sl, "_make_torchvision_resnet", lambda *a, **kw: resnet18(weights=None))
    threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(42)
            device = torch.device("cpu")
            name = "resnet18_pretrained_head"
            end, edge = sl.build_split_pair(name, device, 3, 32)
            state = {"end": {k: v.clone() for k, v in end.state_dict().items()},
                     "edge": {k: v.clone() for k, v in edge.state_dict().items()}}
            before = {p: {k: v.clone() for k, v in part.items()} for p, part in state.items()}
            x = np.random.default_rng(7).normal(size=(2, 3, 32, 32)).astype(np.float32)
            cache = {}

            def train(mode, labels):
                return train_independent(mode, state, x, np.array(labels), 1, .01,
                                         device, name, (3, 32, 32), 10, seed=42, cache=cache)

            full = train("LIIC", [0, 1])
            split = train("LIEIIC", [0, 1])
            train("LIEIIC", [7, 8])
            repeated = train("LIEIIC", [0, 1])
            for part in full:
                assert full[part].keys() == split[part].keys()
                for key in full[part]:
                    torch.testing.assert_close(full[part][key], split[part][key], atol=1e-6, rtol=1e-4)
                    torch.testing.assert_close(split[part][key], repeated[part][key], atol=0, rtol=0)
            for part in state:
                for key in state[part]:
                    assert torch.equal(state[part][key], before[part][key])
    finally:
        torch.set_num_threads(threads)
