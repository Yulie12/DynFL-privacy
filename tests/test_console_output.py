from dynfed.console_output import format_round_summary, format_run_header


def test_compact_summary_has_real_training_signals():
    result = format_round_summary(
        policy="full_dynfl", round_number=6, total_rounds=101, accuracy=0.1765,
        train_loss=2.3016, test_loss=2.30, optimizer_steps=500,
        skipped_clients=0, modes={"LIIC": 100}, round_wall_time_sec=8.7,
        privacy_unit="sample", he_execution="profiled",
    )
    assert "acc=17.65%" in result
    assert "steps=500" in result and "skip=0" in result
    assert "modes=LIIC:100" in result
    assert "HE=profiled" in result


def test_header_warns_about_step_cap():
    result = format_run_header(
        policy="full_dynfl", dataset="fmnist", model="lenet5", clients=100,
        rounds=101, selection_period=5, local_epochs=3, step_limit=5,
        learning_rate=0.01, device="cuda", privacy_unit="sample", he_execution="profiled",
    )
    assert "step_cap=5" in result
    assert "capped" in result
