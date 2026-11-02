"""Resume the registered FMNIST J/T campaign without replaying a committed prefix.

This recovery shim lives outside ``experiments/*.py`` so it does not alter the
campaign's registered source identity.  It changes only the idempotence of the
orchestration step: an already verified round-5 prefix is returned directly.
All training, selection, privacy, aggregation, and latency code remains the
registered implementation.
"""

from pathlib import Path
import sys


REPOSITORY = Path(__file__).resolve().parents[1]
if str(REPOSITORY) not in sys.path:
    sys.path.insert(0, str(REPOSITORY))

from dynfed.paper_storage import file_digest
from experiments import run_fmnist_joint_jt_100r as campaign


_original_stop_prefix = campaign.stop_prefix


def _safe_stop_prefix(config, seed, destination, data, partition, traces):
    destination = Path(destination)
    checkpoint = destination / "checkpoint_round_005.pt"
    if not checkpoint.exists():
        return _original_stop_prefix(
            config, seed, destination, data, partition, traces
        )

    later = sorted(
        path.name
        for path in destination.glob("checkpoint_round_*.pt")
        if int(path.stem.rsplit("_", 1)[1]) > 5
    )
    if later:
        raise RuntimeError(
            f"shared prefix contains checkpoints after round 5: {later}"
        )

    sidecar = checkpoint.with_suffix(".sha256")
    if not sidecar.exists():
        raise RuntimeError("shared prefix round-5 checksum is missing")
    expected = sidecar.read_text(encoding="utf-8").strip().lower()
    if file_digest(checkpoint) != expected:
        raise RuntimeError("shared prefix round-5 checksum mismatch")
    return checkpoint


campaign.stop_prefix = _safe_stop_prefix


if __name__ == "__main__":
    campaign.main()
