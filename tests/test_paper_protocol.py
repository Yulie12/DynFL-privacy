from dataclasses import replace
import pytest
import torch
from dynfed.paper_config import PaperConfig
from dynfed.paper_history import History
from dynfed.paper_privacy import Ledger
from dynfed.paper_search import select
from dynfed.paper_timeline import Unit, plan
from dynfed.paper_runner import system_traces


def unit(name, mode, value, samples=1, arrival=0):
    return Unit(name, mode, 0, (0,), samples, arrival, torch.tensor([value], dtype=torch.float64))


def test_deviation_ema_uses_frozen_reference_and_not_update_ema():
    h = History(gamma_d=0.5, gamma_c=0.25)
    h.observe([unit("a", "FL_C", 2)])
    assert h.deviations == {}
    h.observe([unit("b", "FL_C", 6)])
    assert h.deviations["FL_C"].item() == 4
    assert h.reference.item() == 5
    h.observe([unit("c", "FL_C", 9)])
    assert h.deviations["FL_C"].item() == 4
    assert h.reference.item() == 8
    predicted, j, status = h.predict({"FL_C": 3})
    assert predicted.item() == 12
    assert j == pytest.approx(16/64)
    assert status == "ready"


def test_cloud_unit_sample_weights_and_mode_deviations():
    h = History(gamma_c=0)
    h.observe([unit("a", "FL_C", 2)])
    report = h.observe([unit("b", "FL_E", 4, 1), unit("c", "FL_E", 8, 3), unit("d", "FL_C", 10, 4)])
    assert report["weights"] == [0.125, 0.375, 0.5]
    assert h.deviations["FL_E"].item() == 5
    assert h.deviations["FL_C"].item() == 8
    assert report["cloud_delta"].item() == 8.5
    assert report["reference_before"].item() == 2
    assert report["observed_j"] == pytest.approx(6.5**2 / 4)


def test_no_cloud_and_missing_history_are_unassigned():
    h = History()
    assert h.predict({})[:2] == (None, None)
    h.observe([unit("a", "FL_C", 2)])
    assert h.predict({"FL_E": 1})[:2] == (None, None)
    state = h.reference.clone()
    assert h.observe([])["observed_j"] is None
    assert torch.equal(h.reference, state)


def fixture_config():
    return PaperConfig(clients=3, edges=1, rounds=5, train_limit=30, test_limit=10,
                       local_steps=1, edge_stages=3, edge_aggregation_sec=0.2,
                       cloud_aggregation_sec=0.3, end_edge_delay=0, end_cloud_delay=0,
                       edge_cloud_delay=0, switch_sec=0)


def traces():
    return [{"edge": 0, "step_sec": 1.0, "le_rate": 1.0, "lc_rate": 1.0, "ec_rate": 1.0} for _ in range(3)]


def test_late_fl_e_does_not_delay_synchronous_round_and_enters_later():
    cfg, trace = fixture_config(), traces()
    trace[0]["step_sec"] = 100
    first = plan(cfg, {0:"FL_E", 1:"FL_C"}, trace, {0:3, 1:1}, 1, 0, 0)
    assert first.cutoff == 2
    assert first.finish == pytest.approx(3.3)
    assert [u.mode for u in first.accepted] == ["FL_C"]
    assert [u.mode for u in first.pending] == ["FL_E"]
    assert first.client_finish[0] == pytest.approx(102.2)
    pending = replace(first.pending[0], update=torch.tensor([7.0], dtype=torch.float64))
    later = plan(cfg, {1:"FL_C"}, trace, {1:1}, 1, 102, 1, pending=[pending])
    assert pending in later.accepted
    assert sum(u.samples for u in later.accepted) == 4
    assert pending.generated_round == 0


def test_early_fl_e_is_accepted_but_adds_no_sync_return_dependency():
    cfg, trace = fixture_config(), traces()
    trace[1]["step_sec"] = 5
    result = plan(cfg, {0:"FL_E", 1:"FL_C"}, trace, {0:1, 1:1}, 1, 0, 0)
    assert len(result.accepted) == 2
    assert result.cutoff == 6
    assert result.finish == pytest.approx(7.3)


def test_fl_ec_stages_follow_sequential_edge_feedback():
    result = plan(fixture_config(), {0:"FL_EC"}, traces(), {0:1}, 1, 0, 0)
    stages = [event for event in result.events if event["event"] == "edge_stage"]
    assert [e["finish"] for e in stages] == pytest.approx([2.2, 5.4, 8.6])
    assert result.cutoff == pytest.approx(9.6)
    assert result.finish == pytest.approx(11.9)
    assert len(result.units) == 1


def test_all_edge_mode_has_explicit_event_trigger():
    result = plan(fixture_config(), {0:"FL_E"}, traces(), {0:1}, 1, 10, 0)
    assert result.synchronous is False
    assert result.cutoff == pytest.approx(13.2)
    assert result.finish == pytest.approx(13.5)


def test_initialization_collects_all_feasible_modes_before_pareto():
    cfg, trace, h = fixture_config(), traces(), History()
    pools = {client: ("FL_E", "FL_EC", "FL_C") for client in range(3)}
    previous, pending, now = {}, [], 0
    phases = []
    for round_index in range(4):
        chosen, audit = select(cfg, "proposed", pools, h, trace, {0:1,1:1,2:1}, 0.01, now, round_index, previous, pending)
        phases.append(audit["phase"])
        if round_index == 0:
            assert chosen.learning is None
        assert audit["evaluated_profiles"] <= cfg.evaluation_budget
        accepted = [replace(u, update=torch.tensor([1+round_index+0.1*client], dtype=torch.float64))
                    for client, u in enumerate(chosen.schedule.accepted)]
        h.observe(accepted)
        pending = [replace(u, update=torch.tensor([1.0], dtype=torch.float64)) for u in chosen.schedule.pending]
        now, previous = chosen.schedule.finish, chosen.profile
    assert set(h.deviations) == {"FL_E", "FL_EC", "FL_C"}
    assert phases[0] == "history_initialization"
    assert "pareto" in phases or "reuse_or_random" in phases


def test_fixed_mode_exclusion_stays_skip():
    chosen, _ = select(fixture_config(), "fixed_fl_e", {0:("FL_C",)}, History(), traces(), {0:1}, 1, 0, 0, {}, [])
    assert chosen.profile[0] == "SKIP"


def test_lifetime_calibration_includes_actual_remaining_events():
    ledger = Ledger(8.0, 1e-5)
    sigma = ledger.calibrate(12)
    assert ledger.epsilon(12, sigma) <= 8
    ledger.commit(3, sigma)
    assert ledger.events == 3
    next_sigma = ledger.calibrate(9)
    ledger.commit(9, next_sigma)
    assert ledger.epsilon() <= 8
    with pytest.raises(RuntimeError):
        ledger.commit(1000, 0.1)


def test_system_traces_are_independent_of_policy():
    cfg = fixture_config()
    assert system_traces(cfg, 42) == system_traces(replace(cfg, policies=("fixed_fl_e",)), 42)


def test_old_config_fields_are_rejected(tmp_path):
    path = tmp_path / "old.json"
    path.write_text('{"learning_objective":"online_upper_bound"}')
    with pytest.raises(ValueError, match="unknown configuration"):
        PaperConfig.load(path)


def test_dp_sgd_clips_each_example_before_adding_replacement_noise():
    from dynfed.paper_data import local_train, vector
    cfg = replace(fixture_config(), batch_size=2, clip_norm=0.2, learning_rate=0.1)
    torch.manual_seed(7)
    model = torch.nn.Linear(2, 2)
    base = vector(model)
    x = torch.tensor([[20., -10.], [-5., 10.]])
    y = torch.tensor([0, 1])
    sigma, seed = 2.0, 23
    accumulators = [torch.zeros_like(p) for p in model.parameters()]
    for index in (0, 1):
        loss = torch.nn.functional.cross_entropy(model(x[index:index+1]), y[index:index+1])
        gradients = torch.autograd.grad(loss, tuple(model.parameters()))
        norm = torch.sqrt(sum(g.square().sum() for g in gradients))
        scale = min(1., float(cfg.clip_norm / norm))
        for total, gradient in zip(accumulators, gradients):
            total.add_(gradient * scale)
    generator = torch.Generator().manual_seed(seed)
    expected_parts = []
    for parameter, accumulator in zip(model.parameters(), accumulators):
        noise = torch.randn(parameter.shape, generator=generator) * (2 * cfg.clip_norm * sigma)
        expected_parts.append((parameter.detach() - cfg.learning_rate * (accumulator + noise)/2).flatten())
    ledger = Ledger(100., 1e-5)
    actual, _ = local_train(model, base, x, y, [0, 1], cfg, sigma, ledger, seed)
    torch.testing.assert_close(actual, torch.cat(expected_parts))
    assert ledger.events == 1
    torch.testing.assert_close(vector(model), base)


def test_exhaustive_search_matches_complete_small_front():
    cfg = replace(fixture_config(), selection_period=1)
    h = History(reference=torch.tensor([1.], dtype=torch.float64),
                deviations={"FL_E": torch.tensor([1.]), "FL_EC": torch.tensor([0.]), "FL_C": torch.tensor([2.])})
    pools = {0:("FL_E", "FL_EC", "FL_C"), 1:("FL_E", "FL_EC", "FL_C")}
    chosen, audit = select(cfg, "exhaustive", pools, h, traces(), {0:1,1:1}, 0.01, 0, 0, {}, [])
    assert audit["evaluated_profiles"] == 9
    assert audit["phase"] == "pareto"
    assert chosen.learning is not None


def test_all_sync_infeasible_but_pending_async_can_still_update_cloud():
    pending = unit("old", "FL_E", 3, arrival=5)
    result = plan(fixture_config(), {0:"SKIP"}, traces(), {0:1}, 1, 2, 1, pending=[pending])
    assert result.cutoff == 5
    assert result.accepted == [pending]
    assert result.units == []
