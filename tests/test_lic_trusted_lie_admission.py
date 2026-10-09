"""Regression: trusted LIE does not globally ban direct Cloud Split LIC."""
import random

from dynfed.selection import SelectionConfig, enumerate_candidates, resolved_privacy_parameters


def _candidates(cfg):
    audit = {}
    found = enumerate_candidates(
        config=cfg, client_id=0, edge_factor=1.0, compute_factor=1.0,
        samples=80, remaining_epsilon=8.0, round_idx=0,
        rng=random.Random(7), policy="dynamic", mode_audit=audit,
    )
    return found, audit


def test_sample_trusted_lie_and_lic_can_both_enter_candidate_pool():
    cfg = SelectionConfig(
        privacy_unit="sample", trusted_edge_split_execution=True,
        trusted_lie_joint_sample_dp=True, rounds=2, initial_epsilon=8.0,
        L_block_cycles=1, split_batch_size=64,
        excluded_modes=("LIIE", "LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC"),
    )
    found, audit = _candidates(cfg)
    modes = {c.mode for c in found}
    assert "LIE" in modes
    assert "LIC" in modes, audit
    lic = next(c for c in found if c.mode == "LIC")
    assert lic.sample_embedding_events > 0
    assert lic.sample_label_grad_events > 0
    assert lic.link_mechanisms.get("L_C_emb") == "dp"
    assert lic.link_mechanisms.get("L_C_grad") == "dp"


def test_legacy_client_level_trusted_domain_remains_restricted():
    cfg = SelectionConfig(
        privacy_unit="client", trusted_edge_split_execution=True,
        rounds=2, initial_epsilon=8.0,
        excluded_modes=("LIE", "LIIE", "LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC"),
    )
    found, _ = _candidates(cfg)
    assert all(c.mode != "LIC" for c in found)


def test_sample_calibration_includes_lic_events():
    cfg = SelectionConfig(
        privacy_unit="sample", trusted_edge_split_execution=True,
        trusted_lie_joint_sample_dp=True, rounds=2, initial_epsilon=8.0,
        excluded_modes=("LIE", "LIIE", "LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC"),
    )
    params = resolved_privacy_parameters(cfg)
    assert params["feature_horizon_events"] > 0
