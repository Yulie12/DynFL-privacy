"""Local-only independent joint-proxy Monte Carlo ranking capture.

Dedicated round-zero diagnostic. Uses official test data rather than the
training-subset held-out pool used for calibration. Only scalar MC moments and
prediction scores are serialized; individual parameter updates stay local.
Not a differential privacy audit or a cryptographic proof of independence.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import torch

from dynfed.joint_calibration import (
    JointUpdateCalibrationTable, PairedUpdateMomentAccumulator, as_update_vector,
)
from dynfed.selection import evaluate_global_profile
from experiments.independent_joint_proxy_split import (
    partition_official_test_indices, write_local_split_manifest,
)


def build_ranked_profiles(*, selected, selection, client_samples, client_edges,
                          calibration_path: Path, client_limit: int = 3,
                          count: int = 3):
    """Find >=3 calibrated *cloud-reaching* profiles on a fixed subcohort.

    This is explicitly a subcohort diagnostic, NOT the original full-cohort
    100-client decision. Reject profiles with missing table data or admission.
    """
    from dynfed.selection import _candidate_reaches_cloud

    if selection.learning_objective != 'joint_calibration':
        raise ValueError('official joint_calibration objective is required')
    table = JointUpdateCalibrationTable.load(calibration_path)
    raw = table.to_payload()
    allowed = {(int(e['client_id']), str(e['mode'])) for e in raw['entries']
               if e['client_id'] is not None}
    rows = {}
    for cid, chosen, candidates, remaining in selected:
        cloud = {}
        for c in candidates:
            if c.mode == 'SKIP' or not _candidate_reaches_cloud(c):
                continue
            if not c.feasible or c.epsilon_used > remaining + 1e-12:
                continue
            if (cid, c.mode) not in allowed:
                continue
            prev = cloud.get(c.mode)
            if prev is None or (c.time, c.risk) < (prev.time, prev.risk):
                cloud[c.mode] = c
        if cloud:
            rows[int(cid)] = (chosen, cloud, remaining)
    eligible = sorted(rows, key=lambda cid: (-len(rows[cid][1]), cid))[:client_limit]
    if len(eligible) < 1:
        raise ValueError('no client with a feasible, explicitly calibrated cloud-reaching mode')
    chosen_rows = {}
    for cid in eligible:
        chosen, cloud, remaining = rows[cid]
        base = cloud.get(chosen.mode) or min(cloud.values(), key=lambda c: (c.time, c.mode))
        chosen_rows[cid] = (base, cloud, remaining)
    base_modes = {cid: row[0] for cid, row in chosen_rows.items()}
    options = [base_modes]
    for cid in eligible:
        for mode, cand in sorted(chosen_rows[cid][1].items()):
            if mode != base_modes[cid].mode:
                options.append({**base_modes, cid: cand})
    if len(options) < count:
        raise ValueError('fewer than three calibrated profile configurations in selected subcohort')

    state = replace(selection, joint_calibration_state_key='round:0',
                    joint_calibration_missing_policy='error')
    samples = {cid: float(client_samples[cid]) for cid in eligible}
    edges = {cid: int(client_edges[cid]) for cid in eligible}
    ranked = []
    for option in options:
        rows_for_eval = [(cid, option[cid], [option[cid]], chosen_rows[cid][2])
                         for cid in eligible]
        result = evaluate_global_profile(config=state, selected=rows_for_eval,
                                         client_samples=samples, client_edges=edges)
        if result.learning_proxy_source == 'legacy_fusion_dp':
            continue
        cloud = sorted(cid for cid in eligible if cid in result.admitted_client_ids)
        if len(cloud) != len(eligible):
            continue  # Subcohort all admitted condition needed for fixed weights.
        score = float(result.system_learning_error)
        if not math.isfinite(score) or score < 0:
            continue
        signature = tuple((cid, option[cid].mode) for cid in eligible)
        if any(abs(score - previous[2]) < 1e-12 for previous in ranked):
            continue
        ranked.append((signature, option, score))
        if len(ranked) == count:
            break
    if len(ranked) < count:
        raise ValueError('fewer than three distinct calibrated joint scores with all clients admitted; '
                         'adjust client limit or choose another evaluation state')
    return ranked, eligible


def monte_carlo_score(profile: dict[int, str], moments, ideal_update,
                      client_samples: dict[int, float]):
    """Compute exact formula used by legacy raw-vector validator from moments."""
    ideal = as_update_vector(ideal_update)
    mass = sum(float(client_samples[cid]) for cid in profile)
    if mass <= 0:
        raise ValueError('invalid Cloud sample mass')
    mean = torch.zeros_like(ideal)
    variance = 0.0
    finite_sample_correction = 0.0
    for cid, mode in profile.items():
        entry = moments[(cid, mode)]
        if entry.sample_count < 2 or entry.clean_update_mean.numel() != ideal.numel():
            raise ValueError('invalid independent paired cell moments')
        a = float(client_samples[cid]) / mass
        mean += a * (entry.clean_update_mean + entry.bias_mean)
        variance += a * a * entry.variance_trace
        # E[||sample mean error||^2] exceeds ||population mean error||^2
        # by Tr(Cov(sample mean)); subtract that sampling artifact once.
        finite_sample_correction += a * a * entry.variance_trace / entry.sample_count
    err = mean - ideal
    bias_sq = float(torch.dot(err, err).item())
    total = bias_sq + variance - finite_sample_correction
    if not math.isfinite(total) or total < 0:
        raise ValueError('nonfinite Monte Carlo second moment')
    return {'mc_joint_score': total, 'mc_bias_squared': bias_sq,
            'mc_variance_trace': variance,
            'mc_finite_sample_correction': finite_sample_correction}


def capture_independent_evaluation(*, output: Path, calibration: Path,
                                   selected, selection, train_config,
                                   global_end, global_edge, client_model_states,
                                   train_client_indices, x_test, y_test,
                                   device, model_name: str, input_shape,
                                   num_classes: int, round_idx: int,
                                   trials: int, evaluation_seed: int,
                                   client_limit: int, sample_limit: int,
                                   client_edges: dict[int, int]):
    """Run new paired trajectories on official test examples and write scalars."""
    if int(round_idx) != 0:
        raise ValueError('independent ranking capture currently supports round zero only')
    if trials < 2 or client_limit < 1 or sample_limit < 1:
        raise ValueError('trials >=2, client_limit >=1 and sample_limit >=1 required')
    table = JointUpdateCalibrationTable.load(calibration)
    metadata = table.metadata
    if (metadata.get('source') != 'runtime_heldout_validation_capture'
            or 'held-out validation pool' not in str(metadata.get('calibration_workload', ''))):
        raise ValueError('calibration source cannot be verified as official train subset')
    if str(metadata.get('privacy_unit')) != 'sample':
        raise ValueError('Sample-DP calibration required')
    from dynfed.fmnist_lenet5_dynamic import (
        _training_base_state, _state_dict_to_device, _candidate_training_mechanisms,
        _sample_optimizer_clip_norm, _feature_clip_norm, _update_clip_norm,
        _client_epoch_count, _client_step_limit, _dp_noise_seed,
        _client_training_seed, _run_paired_calibration_payload,
        _calibration_reference_update, _state_difference_from_client_update,
    )
    from dynfed.selection import resolved_privacy_parameters

    samples = {cid: len(train_client_indices[cid]) for cid, *_ in selected}
    ranked, clients = build_ranked_profiles(
        selected=selected, selection=selection, client_samples=samples,
        client_edges=client_edges, calibration_path=calibration,
        client_limit=client_limit, count=3)
    split = partition_official_test_indices(
        len(x_test), clients, seed=evaluation_seed, max_per_client=sample_limit)
    manifest = output.with_suffix('.split_manifest.json')
    write_local_split_manifest(manifest, split, seed=evaluation_seed,
                               dataset=str(metadata.get('dataset', '')),
                               calibration_dataset_split='official_train_subset')
    workloads = {}
    for cid in clients:
        rng = np.random.default_rng(
            _dp_noise_seed(evaluation_seed, round_idx, ['independent_workload', int(cid)]))
        indices = rng.choice(split[cid], size=max(1, samples[cid]),
                             replace=(samples[cid] > len(split[cid])))
        workloads[cid] = (x_test[indices], y_test[indices])

    reference_x = np.concatenate([workloads[cid][0] for cid in sorted(clients)], axis=0)
    reference_y = np.concatenate([workloads[cid][1] for cid in sorted(clients)], axis=0)
    ideal = as_update_vector(_calibration_reference_update(
        global_end=global_end, global_edge=global_edge,
        x=reference_x, y=reference_y, device=device,
        input_shape=input_shape, learning_rate=train_config.learning_rate))

    privacy = resolved_privacy_parameters(selection)
    distinct_cells = {(cid, cand.mode): cand for _signature, option, _score in ranked
                      for cid, cand in option.items()}
    model_cache: dict[str, Any] = {}
    moments = {}
    for (cid, mode), candidate in sorted(distinct_cells.items()):
        base = _training_base_state(
            client_id=cid, candidate=candidate, client_model_states=client_model_states,
            global_end=global_end, global_edge=global_edge,
            device=torch.device('cpu'), training_stage=0,
            mainline_fusion=selection.mainline_fusion)
        x, y = workloads[cid]
        payload = {
            'client_id': cid, 'mode': mode,
            'global_end_state': _state_dict_to_device(base['end'], device),
            'global_edge_state': _state_dict_to_device(base['edge'], device),
            'x': x, 'y': y, 'epochs': _client_epoch_count(train_config, selection, candidate),
            'lr': train_config.learning_rate, 'l2': train_config.l2,
            'local_steps': _client_step_limit(selection), 'model_name': model_name,
            'input_shape': input_shape, 'num_classes': num_classes,
            'mechanisms': _candidate_training_mechanisms(candidate, aggregate_cloud_update_dp=True),
            'privacy_unit': selection.privacy_unit,
            'sample_embedding_noise_multiplier': candidate.sample_embedding_noise_multiplier,
            'sample_label_grad_noise_multiplier': candidate.sample_label_grad_noise_multiplier,
            'sample_optimizer_noise_multiplier': candidate.sample_optimizer_noise_multiplier,
            'sample_optimizer_clip_norm': _sample_optimizer_clip_norm(train_config),
            'dp_clip_norm': train_config.dp_clip_norm,
            'dp_feature_clip_norm': _feature_clip_norm(train_config),
            'dp_update_clip_norm': _update_clip_norm(train_config),
            'dp_feature_noise_multiplier': privacy['feature_noise_multiplier'],
            'dp_update_noise_multiplier': float(candidate.update_noise_multiplier
                if candidate.update_noise_multiplier is not None else privacy['update_noise_multiplier']),
            'dp_update_mode': train_config.dp_update_mode,
            'dp_epsilon': max(selection.dp_emb_epsilon, 1e-6),
            'dp_seed': _dp_noise_seed(evaluation_seed, round_idx, ['independent_base', cid, mode]),
            'training_seed': _client_training_seed(selection.seed, round_idx, cid, 0),
            'device': str(device),
        }
        clean, private = _run_paired_calibration_payload(
            payload, trials=trials, base_seed=evaluation_seed,
            round_idx=round_idx, client_id=cid, mode=mode, model_cache=model_cache)
        # Convert worker-local deltas to contributions relative to the
        # fixed global state. Do not add any extra noise/retraining here.
        clean_cloud_relative = _state_difference_from_client_update(
            client_id=cid, state_diff=clean, client_model_states=client_model_states,
            global_end=global_end, global_edge=global_edge, device=torch.device('cpu'))
        accumulator = PairedUpdateMomentAccumulator()
        for update in private:
            private_cloud_relative = _state_difference_from_client_update(
                client_id=cid, state_diff=update, client_model_states=client_model_states,
                global_end=global_end, global_edge=global_edge, device=torch.device('cpu'))
            accumulator.update(clean_cloud_relative, private_cloud_relative)
        moments[(cid, mode)] = accumulator.finalize()
        del clean, private

    rows = []
    for index, (signature, option, prediction) in enumerate(ranked):
        modes = {cid: cand.mode for cid, cand in option.items()}
        rows.append({'profile_id': f'independent_subcohort_{index}',
                     'selector_joint_score': prediction,
                     **monte_carlo_score(modes, moments, ideal, samples),
                     'client_count': len(modes), 'trials_per_cell': trials,
                     'mode_signature': [[int(cid), str(mode)] for cid, mode in signature]})
    result = {
        'format': 'dynfl_independent_joint_summary_v2',
        'evaluation_data_independent_of_calibration': True,
        'evaluation_data_provenance': 'private_local_only',
        'conditional_cross_client_rng_independence': True,
        'evaluation_protocol': 'independent_official_test_paired_rng_v1',
        'mc_estimator': 'unbiased_conditional_second_moment_independent_clients_v2',
        'calibration_sha256': hashlib.sha256(calibration.read_bytes()).hexdigest(),
        'calibration_source_verified_as_train_subset': True,
        'split_manifest': str(manifest),
        'scope': 'calibration-covered_subcohort_not_full_cohort',
        'measurement_boundary': 'global_relative_worker_updates_before_cloud_release_processing',
        'state_key': f'round:{round_idx}',
        'seed': int(evaluation_seed),
        'profiles': rows,
    }
    from experiments.validate_joint_proxy_ranking import evaluate
    validated = evaluate(result)  # fail before writing an unusable evaluation file
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return validated
