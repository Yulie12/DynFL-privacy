import csv
import json

from dynfed.paper_config import PaperConfig
from experiments.audit_paper import reconstruct_dispatches


def test_dispatch_reconstruction_uses_start_not_end_of_window(tmp_path):
    config = PaperConfig(clients=1, edges=1, rounds=2, train_limit=2,
                         test_limit=1, local_steps=1)
    folder = tmp_path / "fixed_fl_e"
    folder.mkdir()
    with (folder / "client_decisions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("round", "client", "mode"))
        writer.writeheader()
        writer.writerows({"round": index, "client": 0, "mode": "FL_E"} for index in range(2))
    trace = [{"edge": 0, "step_sec": .1, "le_rate": 100., "lc_rate": 100., "ec_rate": 100.}]
    (tmp_path / "traces.json").write_text(json.dumps([trace, trace]), encoding="utf-8")
    dispatches = reconstruct_dispatches(folder, tmp_path, config.to_dict(), [[0, 1]], .01)
    start, schedule = dispatches[0]
    assert start == 0.
    assert schedule.finish > start
    assert dispatches[1][0] == schedule.finish
