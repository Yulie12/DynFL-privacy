from dataclasses import replace
import pytest
from dynfed.paper_barrier import VersionBarrier
from dynfed.paper_config import PaperConfig
from dynfed.paper_timeline import plan


def test_missing_last_ack_blocks_next_round():
    barrier = VersionBarrier(range(3))
    barrier.begin(1)
    barrier.acknowledge(0, 1, 2.)
    barrier.acknowledge(1, 1, 3.)
    with pytest.raises(RuntimeError, match="every client"):
        barrier.require_ready(100.)
    barrier.acknowledge(2, 1, 5.)
    with pytest.raises(RuntimeError, match="before the last"):
        barrier.require_ready(4.)
    barrier.require_ready(5.)
    with pytest.raises(RuntimeError, match="stale"):
        barrier.acknowledge(2, 0, 6.)


def test_skipped_client_is_still_part_of_broadcast_barrier():
    config = PaperConfig(clients=2, edges=1, train_limit=2, test_limit=1)
    traces = [{"edge": 0, "step_sec": .1, "le_rate": 100., "lc_rate": 100., "ec_rate": 100.},
              {"edge": 0, "step_sec": .1, "le_rate": 100., "lc_rate": .01, "ec_rate": 100.}]
    result = plan(config, {0:"FL_C", 1:"SKIP"}, traces, {0:1,1:1}, 1., 0., 0)
    assert result.finish == result.acknowledgement_finish[1]
    assert result.acknowledgement_finish[1] > result.acknowledgement_finish[0]
    assert len(result.accepted) == 1


def test_ack_time_is_not_omitted_from_latency():
    config = PaperConfig(clients=1, edges=1, train_limit=1, test_limit=1)
    traces = [{"edge": 0, "step_sec": .1, "le_rate": 1., "lc_rate": 1., "ec_rate": 1.}]
    a = plan(config, {0:"FL_E"}, traces, {0:1}, 1., 0., 0)
    b = plan(replace(config, acknowledgement_mb=1.), {0:"FL_E"}, traces, {0:1}, 1., 0., 0)
    assert b.finish-a.finish == pytest.approx(2*(1-config.acknowledgement_mb))
    assert a.cutoff == b.cutoff


def test_empty_training_round_does_not_create_cloud_update():
    config = PaperConfig(clients=1, edges=1, train_limit=1, test_limit=1)
    traces = [{"edge": 0, "step_sec": .1, "le_rate": 1., "lc_rate": 1., "ec_rate": 1.}]
    result = plan(config, {0:"SKIP"}, traces, {0:1}, 1., 0., 0)
    assert not result.accepted and not result.pending
    assert set(result.acknowledgement_finish) == {0}


def test_runner_does_not_train_next_round_when_ack_is_missing(tmp_path, monkeypatch):
    import json
    import torch
    import dynfed.paper_runner as runner
    config = PaperConfig(clients=2, edges=1, rounds=2, train_limit=4, test_limit=2,
                         batch_size=1, local_steps=1, seeds=(42,), policies=("fixed_fl_c",))
    torch.set_num_threads(1)
    torch.manual_seed(3)
    data = (torch.randn(4,1,28,28), torch.tensor([0,1,0,1]),
            torch.randn(2,1,28,28), torch.tensor([0,1]))
    original = runner.select

    def omit_last_ack(*args, **kwargs):
        chosen, report = original(*args, **kwargs)
        chosen.schedule.events = [event for event in chosen.schedule.events
                                  if not (event["event"] == "model_ack" and event["client"] == 1)]
        return chosen, report

    monkeypatch.setattr(runner, "select", omit_last_ack)
    destination = tmp_path/"run"
    with pytest.raises(RuntimeError, match="every client"):
        runner.run_policy(config, "fixed_fl_c", 42, data, [[0,1],[2,3]],
                          runner.system_traces(config, 42), destination)
    privacy = json.loads((destination/"privacy_state.json").read_text(encoding="utf-8"))
    assert [privacy[str(client)]["events"] for client in range(2)] == [1,1]
    assert not (destination/"summary.json").exists()
