"""Atomic run artifacts and canonical identities (no change to FL mathematics)."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import torch
from contextlib import contextmanager


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def tensor_digest(value):
    array = value.detach().cpu().contiguous().numpy()
    return hashlib.sha256(memoryview(array).cast("B")).hexdigest()


def source_identity():
    root = Path(__file__).resolve().parents[1]
    sources = {str(p.relative_to(root)): file_digest(p) for folder in ("dynfed", "experiments")
               for p in sorted((root/folder).glob("*.py"))}
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    return {"git_revision": revision, "sha256": sources, "source_hash": digest(sources)}


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("w",encoding="utf-8") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False))
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def atomic_torch(path, value):
    path = Path(path)
    temporary = path.with_name(path.name+".tmp")
    with temporary.open("wb") as handle:
        torch.save(value, handle)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def load_checkpoint(path):
    sidecar = path.with_suffix(".sha256")
    if sidecar.exists() and sidecar.read_text(encoding="ascii").strip() != file_digest(path):
        raise RuntimeError("checkpoint checksum mismatch; refuse unsafe recovery")
    return torch.load(path, map_location="cpu", weights_only=False)


def save_checkpoint(path, value):
    if path.exists():
        raise RuntimeError("refuse to overwrite a committed round checkpoint")
    atomic_torch(path, value)
    # A crash before this advisory sidecar is safe: the complete atomic .pt is
    # self-contained and still checked against its run identity on recovery.
    temporary = path.with_suffix(".sha256.tmp")
    temporary.write_text(file_digest(path), encoding="ascii")
    os.replace(temporary, path.with_suffix(".sha256"))


@contextmanager
def exclusive_run(destination):
    """Kernel-held byte lock releases on process death; no stale PID deletion."""
    destination=Path(destination)
    destination.parent.mkdir(parents=True,exist_ok=True)
    path=destination.parent/("."+destination.name+".runlock")
    with path.open("a+b") as handle:
        if path.stat().st_size == 0:
            handle.write(b"0")
            handle.flush()
        handle.seek(0)
        try:
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(handle.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(handle.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError as error:
            raise RuntimeError("run is already active; concurrent resume is forbidden") from error
        try:
            yield
        finally:
            handle.seek(0)
            if os.name == "nt":
                msvcrt.locking(handle.fileno(),msvcrt.LK_UNLCK,1)
            else:
                fcntl.flock(handle.fileno(),fcntl.LOCK_UN)
