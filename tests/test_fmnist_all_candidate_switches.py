from dataclasses import replace

import torch

from dynfed.paper_config import MODES, PaperConfig
from dynfed.paper_history import History
from dynfed.paper_runner import system_traces
from tools.run_fmnist_all_candidate_switches import forced_selector


def test_forced_selector_uses_p0_prefix_then_requested_candidate():
    config = replace(PaperConfig(), clients=100, edges=10,
                     client_edge_map=tuple(client//10 for client in range(100)),
                     rounds=10, seeds=(42,), policies=("fixed_mixed",),
                     experiment_version="staged_eight_v1", shared_init_rounds=10,
                     selection_period=1, evaluation_budget=8)
    pools = {client: MODES for client in range(100)}
    samples = {client: 600 for client in range(100)}
    history = History(reference=torch.ones(4), deviations={mode: torch.ones(4) for mode in MODES})
    selector = forced_selector("P3")
    _, before = selector(config, "fixed_mixed", pools, history, system_traces(config,42)[4],
                         samples, 1., 0., 4, {}, [])
    _, after = selector(config, "fixed_mixed", pools, history, system_traces(config,42)[5],
                        samples, 1., 0., 5, {}, [])
    assert before["selected_candidate"] == "P0"
    assert after["selected_candidate"] == "P3"
    assert after["applied_round"] == 6
