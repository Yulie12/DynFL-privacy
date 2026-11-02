"""Resume the interrupted E/EC diagnostic without changing its source identity."""
from __future__ import annotations

import csv
import json
from pathlib import Path
import shutil
import time

from dynfed.paper_config import PaperConfig
from dynfed.paper_storage import digest, file_digest, load_checkpoint, source_identity
import dynfed.paper_runner as runner
from experiments.audit_paper import audit
from experiments.run_fmnist_fixed_e_ec_50_50 import (
    DEFAULT_CONFIG,
    DEFAULT_OUTPUT,
    diagnostic_rows,
    diagnostic_select,
    final_report,
    make_plots,
    privacy_preview,
    write_csv,
)


def main() -> None:
    output = DEFAULT_OUTPUT.resolve()
    config = PaperConfig.load(DEFAULT_CONFIG)
    folder = output / "seed_42" / "fixed_mixed"
    manifest = json.loads((output / "experiment_manifest.json").read_text(encoding="utf-8"))
    current_source = source_identity()
    if manifest["config_hash"] != digest(config.to_dict()):
        raise RuntimeError("refuse recovery: configuration hash changed")
    if manifest["source_hash"] != current_source["source_hash"]:
        raise RuntimeError("refuse recovery: scientific source identity changed")
    status = json.loads((folder / "run_status.json").read_text(encoding="utf-8"))
    if status.get("status") != "failed" or status.get("completed_rounds") != 14:
        raise RuntimeError("refuse recovery: expected the preserved round-14 failure boundary")
    checkpoint_path = folder / "checkpoint_round_014.pt"
    checkpoint = load_checkpoint(checkpoint_path)
    if checkpoint["next_round"] != 14 or checkpoint["barrier"].version != 14:
        raise RuntimeError("refuse recovery: checkpoint/version boundary mismatch")
    if len(checkpoint["barrier"].acks) != config.clients:
        raise RuntimeError("refuse recovery: incomplete round-14 ACK barrier")

    stale = folder / "privacy_state.json.tmp"
    if stale.exists():
        evidence = folder / "incident_privacy_state_round15_precommit.json"
        if evidence.exists():
            raise FileExistsError(f"refuse to overwrite incident evidence: {evidence}")
        shutil.copyfile(stale, evidence)
        (folder / "incident_privacy_state_round15_precommit.sha256").write_text(
            file_digest(evidence), encoding="ascii")

    original_atomic_json = runner.atomic_json

    def retry_atomic_json(path, value):
        for attempt in range(8):
            try:
                return original_atomic_json(path, value)
            except PermissionError:
                if attempt == 7:
                    raise
                time.sleep(0.05 * (2 ** attempt))

    runner.atomic_json = retry_atomic_json
    runner.select = diagnostic_select
    print(
        f"RESUME checkpoint={checkpoint_path.name} sha256={file_digest(checkpoint_path)} "
        f"source_hash={current_source['source_hash']}", flush=True,
    )
    runner.run(config, output, resume=True)

    audit_result = audit(output)
    runner.write_json(output / "audit.json", audit_result)
    shutil.copyfile(folder / "round_metrics.csv", output / "round_metrics.csv")
    with (folder / "round_metrics.csv").open(encoding="utf-8-sig") as handle:
        metrics = list(csv.DictReader(handle))
    j_rows = diagnostic_rows(folder, config)
    write_csv(output / "J_trajectory.csv", j_rows)
    make_plots(output, metrics, j_rows)
    final_report(output, metrics, j_rows, privacy_preview(config), audit_result)
    if not all(row["status"] == "passed" for row in audit_result):
        raise RuntimeError("final independent audit failed")
    print(f"COMPLETE output={output} audit=passed", flush=True)


if __name__ == "__main__":
    main()
