from __future__ import annotations

import ctypes
import importlib
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .privacy import mechanism_uses_he


CKKS_POLY_MODULUS_DEGREE = 8192
CKKS_COEFF_MOD_BIT_SIZES = (50, 40)
CKKS_SCALE_BITS = 40
CKKS_SCALE = 2 ** CKKS_SCALE_BITS


@dataclass(frozen=True)
class HEAvailability:
    available: bool
    backend: str
    detail: str


@dataclass
class HEOperationMetrics:
    """Measured CKKS work performed by one training round."""

    backend: str
    aggregation_calls: int = 0
    encrypted_updates: int = 0
    encrypted_parameter_values: int = 0
    ciphertext_count: int = 0
    ciphertext_bytes: int = 0
    wall_time_sec: float = 0.0
    process_workers: int = 0
    process_tasks: int = 0
    process_fallbacks: int = 0
    process_chunks_per_task: int = 0
    shared_memory_bytes: int = 0
    mapped_update_bytes: int = 0
    key_setup_time_sec: float = 0.0
    encryption_time_sec: float = 0.0
    addition_time_sec: float = 0.0
    decryption_time_sec: float = 0.0
    max_abs_error: float = 0.0
    failures: int = 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "he_backend": self.backend,
            "he_poly_modulus_degree": CKKS_POLY_MODULUS_DEGREE,
            "he_coeff_mod_bit_sizes": ";".join(
                str(value) for value in CKKS_COEFF_MOD_BIT_SIZES
            ),
            "he_scale_bits": CKKS_SCALE_BITS,
            "he_aggregation_calls": self.aggregation_calls,
            "he_encrypted_updates": self.encrypted_updates,
            "he_encrypted_parameter_values": self.encrypted_parameter_values,
            "he_ciphertext_count": self.ciphertext_count,
            "he_ciphertext_bytes": self.ciphertext_bytes,
            "he_ciphertext_bytes_semantics": (
                "serialized_bytes"
                if self.backend == "tenseal"
                else "seal_save_size_upper_bound"
                if self.backend == "seal"
                else "not_applicable"
            ),
            "he_wall_time_sec": self.wall_time_sec,
            "he_worker_cpu_time_sec": (
                self.encryption_time_sec
                + self.addition_time_sec
                + self.decryption_time_sec
            ),
            "he_process_workers": self.process_workers,
            "he_process_tasks": self.process_tasks,
            "he_process_fallbacks": self.process_fallbacks,
            "he_process_chunks_per_task": self.process_chunks_per_task,
            "he_shared_memory_bytes": self.shared_memory_bytes,
            "he_mapped_update_bytes": self.mapped_update_bytes,
            "he_key_setup_time_sec": self.key_setup_time_sec,
            "he_encryption_time_sec": self.encryption_time_sec,
            "he_addition_time_sec": self.addition_time_sec,
            "he_decryption_time_sec": self.decryption_time_sec,
            "he_max_abs_error": self.max_abs_error,
            "he_failures": self.failures,
        }


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
        seal = importlib.import_module("seal")
    except Exception as exc:
        return HEAvailability(False, "seal", f"Python module 'seal' is unavailable: {exc}")

    try:
        max_error = _validate_seal_backend(seal)
    except Exception as exc:
        return HEAvailability(False, "seal", f"SEAL CKKS runtime validation failed: {exc}")
    return HEAvailability(
        True,
        "seal",
        f"SEAL CKKS encrypted-vector validation passed (max error {max_error:.3e}).",
    )


def decode_seal_vector(encoder: Any, plaintext: Any) -> np.ndarray:
    """Decode CKKS slots, including the zero-stride array emitted by PySEAL on pybind11 3."""
    decoded = np.asarray(encoder.decode(plaintext))
    if decoded.ndim != 1:
        raise RuntimeError(f"SEAL CKKS decoder returned {decoded.ndim} dimensions; expected one")
    if decoded.dtype != np.float64:
        raise RuntimeError(f"SEAL CKKS decoder returned dtype {decoded.dtype}; expected float64")

    if decoded.size > 1 and decoded.strides == (0,):
        # Fed3Scale's PySEAL wrapper allocates the full output buffer, but with
        # pybind11 3.x its one-argument array constructor exposes stride zero.
        raw_buffer = (ctypes.c_double * int(decoded.size)).from_address(int(decoded.ctypes.data))
        return np.ctypeslib.as_array(raw_buffer).copy()
    return np.ascontiguousarray(decoded, dtype=np.float64)


def _validate_seal_backend(seal: Any) -> float:
    expected_left = np.array([0.125, -0.25, 0.5, 1.0], dtype=np.float64)
    expected_right = np.array([0.375, 0.5, -0.125, -0.25], dtype=np.float64)

    parms = seal.EncryptionParameters(seal.scheme_type.ckks)
    poly_modulus_degree = CKKS_POLY_MODULUS_DEGREE
    parms.set_poly_modulus_degree(poly_modulus_degree)
    parms.set_coeff_modulus(
        seal.CoeffModulus.Create(poly_modulus_degree, list(CKKS_COEFF_MOD_BIT_SIZES))
    )
    context = seal.SEALContext(parms)
    keygen = seal.KeyGenerator(context)
    public_key = keygen.create_public_key()
    secret_key = keygen.secret_key()
    encoder = seal.CKKSEncoder(context)
    encryptor = seal.Encryptor(context, public_key)
    decryptor = seal.Decryptor(context, secret_key)
    evaluator = seal.Evaluator(context)
    scale = CKKS_SCALE

    encrypted_left = encryptor.encrypt(encoder.encode(expected_left, scale))
    encrypted_right = encryptor.encrypt(encoder.encode(expected_right, scale))
    encrypted_sum = evaluator.add(encrypted_left, encrypted_right)
    actual = decode_seal_vector(encoder, decryptor.decrypt(encrypted_sum))[: expected_left.size]
    expected = expected_left + expected_right
    max_error = float(np.max(np.abs(actual - expected)))
    if not np.isfinite(actual).all() or max_error > 1e-5:
        raise RuntimeError(
            f"encrypted vector addition differs from plaintext addition (max error {max_error:.3e})"
        )
    return max_error


def has_he_mechanism(mechanisms: dict[str, str]) -> bool:
    return any(mechanism_uses_he(str(value)) for value in mechanisms.values())
