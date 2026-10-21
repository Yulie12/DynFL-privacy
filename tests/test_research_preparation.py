from dataclasses import replace
import copy
import json
from pathlib import Path
import pytest
import torch
import dynfed.paper_runner as runner
from dynfed.paper_config import PaperConfig, CORE_POLICIES, effective_config
from dynfed.paper_history import History
from dynfed.paper_search import select
from dynfed.paper_acceptance import validate_units
from dynfed.paper_timeline import Unit
from dynfed.paper_storage import load_checkpoint


@pytest.fixture
def toy(monkeypatch):
    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(11)
    data = (torch.randn(9,2,generator=generator), torch.tensor([0,1,0,1,0,1,0,1,0]),
            torch.randn(3,2,generator=generator), torch.tensor([0,1,0]))
    config = PaperConfig(clients=3, edges=1, rounds=4, train_limit=9, test_limit=3,
                         batch_size=2, local_steps=2, policies=CORE_POLICIES, seeds=(42,),
                         selection_period=1, evaluation_budget=20)
    def model(cfg,seed,features=False):
        torch.manual_seed(seed)
        return torch.nn.Linear(2,2)
    monkeypatch.setattr(runner,"make_training_model",model)
    monkeypatch.setattr(runner,"load_data",lambda cfg,seed:data)
    return config,data,[[0,1,2],[3,4,5],[6,7,8]],runner.system_traces(config,42)


def execute(toy, destination, policy="proposed", **kwargs):
    cfg,data,partition,traces = toy
    return runner.run_policy(cfg,policy,42,data,partition,traces,destination,**kwargs)


def test_shared_initialization_is_identical_and_charged(toy,tmp_path):
    states=[]
    for policy in CORE_POLICIES:
        result=execute(toy,tmp_path/policy,policy)
        assert result["initialization_rounds"] == 2
        assert result["initialization_steps"] == 20
        assert result["initialization_max_epsilon"] > 0
        assert result["initialization_system_time"] > 0
        assert result["initialization_system_time"]+result["selection_phase_system_time"] == pytest.approx(result["system_time"])
        states.append(load_checkpoint(tmp_path/policy/"checkpoint_round_002.pt"))
    for state in states[1:]:
        for key in states[0]["model"]:
            torch.testing.assert_close(state["model"][key],states[0]["model"][key],rtol=0,atol=0)
        torch.testing.assert_close(state["history"].reference,states[0]["history"].reference,rtol=0,atol=0)
        assert state["history"].mode_observations == {"FL_E":1,"FL_EC":1,"FL_C":1}
        assert state["ledgers"] == states[0]["ledgers"]
    no_ema=effective_config(toy[0],"joint_no_ema")
    assert (no_ema.gamma_d,no_ema.gamma_c)==(0.,.9)


def test_joint_methods_use_identical_search_given_identical_history(toy):
    cfg,_,partition,traces=toy
    history=History(reference=torch.tensor([1.],dtype=torch.float64),deviations={
        "FL_E":torch.tensor([1.]),"FL_EC":torch.tensor([-1.]),"FL_C":torch.tensor([2.])})
    pools={client:("FL_E","FL_EC","FL_C") for client in range(3)}
    reports=[]
    for policy in ("proposed","joint_no_ema"):
        chosen,report=select(effective_config(cfg,policy),policy,pools,copy.deepcopy(history),traces[2],
                             {client:len(indices) for client,indices in enumerate(partition)},.01,0.,2,{},[])
        assert len(report["candidates"])<=cfg.evaluation_budget
        reports.append((chosen.profile,report["candidates"],report["archive_profiles"]))
    assert reports[0]==reports[1]


def test_round_boundary_recovery_is_exact(toy,tmp_path):
    baseline=execute(toy,tmp_path/"baseline")
    with pytest.raises(InterruptedError):execute(toy,tmp_path/"resumed",stop_after_round=2)
    result=execute(toy,tmp_path/"resumed",resume=True)
    assert result["executed_local_steps"]==baseline["executed_local_steps"]
    assert result["system_time"]==baseline["system_time"]
    for index in range(1,toy[0].rounds+1):
        a=load_checkpoint(tmp_path/"baseline"/f"checkpoint_round_{index:03d}.pt")
        b=load_checkpoint(tmp_path/"resumed"/f"checkpoint_round_{index:03d}.pt")
        for key in a["model"]:torch.testing.assert_close(a["model"][key],b["model"][key],rtol=0,atol=0)
        assert a["ledgers"]==b["ledgers"]
        torch.testing.assert_close(a["history"].reference,b["history"].reference,rtol=0,atol=0)


def test_mid_local_block_recovery_does_not_repeat_durable_dp_steps(toy,tmp_path,monkeypatch):
    baseline=execute(toy,tmp_path/"baseline")
    original=runner.local_train
    counts={"steps":0,"fail":True}
    def interrupted(*args,**kwargs):
        callback=kwargs["progress_callback"]
        def persist(progress):
            callback(progress)
            counts["steps"]+=1
            if counts["fail"]:
                counts["fail"]=False
                raise InterruptedError("simulated crash after committed DP step")
        kwargs["progress_callback"]=persist
        return original(*args,**kwargs)
    monkeypatch.setattr(runner,"local_train",interrupted)
    with pytest.raises(InterruptedError):execute(toy,tmp_path/"resumed")
    result=execute(toy,tmp_path/"resumed",resume=True)
    assert counts["steps"]==result["executed_local_steps"]==baseline["executed_local_steps"]
    a=torch.load(tmp_path/"baseline"/"final_state.pt",weights_only=False)
    b=torch.load(tmp_path/"resumed"/"final_state.pt",weights_only=False)
    for key in a["model"]:torch.testing.assert_close(a["model"][key],b["model"][key],rtol=0,atol=0)


@pytest.mark.parametrize("fault",["duplicate","missing","stale","late"])
def test_receiver_rejects_invalid_updates(fault):
    expected=Unit("r0:FL_C:c0","FL_C",0,(0,),1,1.,generated_round=0)
    actual=replace(expected,update=torch.tensor([1.]),base_source="cloud:v0",base_round=0)
    units=[actual]
    if fault=="duplicate":units.append(actual)
    elif fault=="missing":units=[]
    elif fault=="stale":units=[replace(actual,base_round=1,base_source="cloud:v1")]
    with pytest.raises(RuntimeError):validate_units([expected],units,0,0,.5 if fault=="late" else 1.)


def test_corrupt_checkpoint_and_config_changes_refuse_recovery(toy,tmp_path):
    with pytest.raises(InterruptedError):execute(toy,tmp_path/"run",stop_after_round=2)
    changed=(replace(toy[0],batch_size=1),*toy[1:])
    with pytest.raises(RuntimeError,match="identity mismatch"):execute(changed,tmp_path/"run",resume=True)
    path=tmp_path/"run"/"checkpoint_round_002.pt"
    path.write_bytes(path.read_bytes()+b"tamper")
    with pytest.raises(RuntimeError,match="checksum"):execute(toy,tmp_path/"run",resume=True)


def test_statistics_gate_rejects_incomplete_or_tampered_ack(toy,tmp_path):
    from experiments.audit_paper import audit
    from experiments.summarize_paper import summarize
    cfg=toy[0]
    root=tmp_path/"scenario"
    runner.run(cfg,root)
    assert len(audit(root))==3
    summarize(root,.8)
    timeline=root/"seed_42"/"proposed"/"timeline_round_002.json"
    value=json.loads(timeline.read_text(encoding="utf-8"))
    value=[event for event in value if not(event["event"]=="model_ack" and event["client"]==2)]
    timeline.write_text(json.dumps(value),encoding="utf-8")
    with pytest.raises(AssertionError):summarize(root,.8)
    (root/"seed_42"/"joint_no_ema"/"summary.json").unlink()
    with pytest.raises(ValueError,match="incomplete"):summarize(root,.8)


def test_unjournalled_privacy_spend_is_not_silently_rolled_back(toy,tmp_path,monkeypatch):
    def unsafe(model,base,x,y,indices,cfg,sigma,ledger,seed,**kwargs):
        ledger.commit(1,sigma)
        raise RuntimeError("injected failure before durable checkpoint")
    monkeypatch.setattr(runner,"local_train",unsafe)
    with pytest.raises(RuntimeError,match="injected"):execute(toy,tmp_path/"run")
    assert (tmp_path/"run"/"unsafe_privacy_state.json").exists()
    with pytest.raises(RuntimeError,match="refusing ledger rollback"):execute(toy,tmp_path/"run",resume=True)


def test_infeasible_shared_initialization_does_not_fill_zero_history(toy):
    cfg,_,partition,traces=toy
    pools={0:("FL_E",),1:("FL_EC",),2:("FL_E",)}
    history=History()
    with pytest.raises(RuntimeError,match="infeasible"):
        select(cfg,"latency_only",pools,history,traces[0],{0:3,1:3,2:3},.01,0.,0,{},[])
    assert history.reference is None and not history.deviations


def test_complete_run_resume_is_immutable(toy,tmp_path):
    from dynfed.paper_storage import file_digest
    path=tmp_path/"run"
    execute(toy,path)
    hashes={str(p.relative_to(path)):file_digest(p) for p in path.rglob('*') if p.is_file()}
    execute(toy,path,resume=True)
    assert hashes=={str(p.relative_to(path)):file_digest(p) for p in path.rglob('*') if p.is_file()}


def test_configuration_annotations_have_no_duplicate_fields():
    import ast
    tree=ast.parse((Path(__file__).resolve().parents[1]/"dynfed/paper_config.py").read_text(encoding="utf-8"))
    cls=next(node for node in tree.body if isinstance(node,ast.ClassDef) and node.name=="PaperConfig")
    fields=[node.target.id for node in cls.body if isinstance(node,ast.AnnAssign)]
    assert len(fields)==len(set(fields))


def test_registration_contains_independent_hashed_units(tmp_path,monkeypatch):
    import experiments.prepare_formal as registration
    monkeypatch.setattr(registration,"dataset_files",lambda cfg:{"fixture_data":"test_hash"})
    units=registration.register_matrix(tmp_path/"registered")
    assert len(units)==48
    assert len({entry["output"] for entry in units})==48
    assert not list((tmp_path/"registered").rglob("summary.json"))
    arms={entry["policy"]:entry for entry in units if entry["dataset"]=="fmnist" and entry["seed"]==42}
    assert (arms["joint_no_ema"]["config"]["gamma_d"],arms["joint_no_ema"]["config"]["gamma_c"])==(0.,.9)
    assert arms["joint_no_ema"]["config_hash"]!=arms["proposed"]["config_hash"]
