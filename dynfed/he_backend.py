from __future__ import annotations

import importlib
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class HEAvailability:
    available: bool
    backend: str
    detail: str


def check_he_backend(backend: str, local_deps: str | None = None) -> HEAvailability:
    backend = backend.strip().lower()
    if backend in {"none", "metadata"}:
        return HEAvailability(False, backend, "HE backend is disabled.")
    if backend not in {"seal", "tenseal"}:
        return HEAvailability(False, backend, f"Unsupported HE backend: {backend}")

    if local_deps:
        deps_path = str(Path(local_deps).resolve())
        if deps_path not in sys.path:
            sys.path.insert(0, deps_path)

    if backend == "tenseal":
        try:
            importlib.import_module("tenseal")
        except Exception as exc:
            return HEAvailability(False, "tenseal", f"Python module 'tenseal' is unavailable: {exc}")
        return HEAvailability(True, "tenseal", "Python module 'tenseal' is importable.")

    try:
        importlib.import_module("seal")
    except Exception as exc:
        return HEAvailability(False, "seal", f"Python module 'seal' is unavailable: {exc}")
    return HEAvailability(True, "seal", "Python module 'seal' is importable.")


def has_he_mechanism(mechanisms: dict[str, str]) -> bool:
    return any(value in {"he2", "he3"} for value in mechanisms.values())
