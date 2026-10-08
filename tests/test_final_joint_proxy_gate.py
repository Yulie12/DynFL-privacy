import json
import pytest
import torch
from experiments.finalize_joint_proxy import finalize, summarize_calibration


def test_summary_rejects_wrong_format(tmp_path):
    target=tmp_path/'bad.pt'
    torch.save({'format':'other','entries':[]},target)
    with pytest.raises(ValueError,match='format'):
        summarize_calibration(target)


def test_no_independent_data_does_not_claim_rho(tmp_path):
    target=tmp_path/'table.pt'
    torch.save({'format':'dynfl_joint_update_calibration_v1','metadata':{'capture_pair_count':5},'entries':[
        {'state_key':'round:0','mode':'LIC','client_id':0,'clean_update_mean':torch.zeros(2),'bias_mean':torch.zeros(2),'variance_trace':0.,'sample_count':5}
    ]},target)
    result=finalize(target,None)
    assert result['spearman_validation']['status']=='pending_independent_heldout_paired_trajectories'
    assert 'spearman_rho' not in result['spearman_validation']
    assert result['end_to_end_sample_dp']=='not_established'


def test_false_independence_rejected(tmp_path):
    target=tmp_path/'table.pt'
    torch.save({'format':'dynfl_joint_update_calibration_v1','metadata':{},'entries':[
        {'state_key':'round:0','mode':'LIC','client_id':0,'clean_update_mean':torch.zeros(2),'bias_mean':torch.zeros(2),'variance_trace':0.,'sample_count':2}
    ]},target)
    evaluation=tmp_path/'eval.json'
    evaluation.write_text(json.dumps({'evaluation_data_independent_of_calibration':False}),encoding='utf-8')
    with pytest.raises(ValueError,match='independence'):
        finalize(target,evaluation)
