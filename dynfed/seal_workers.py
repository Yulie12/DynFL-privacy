"""Torch-free SEAL worker processes for parallel CKKS update aggregation.

These functions run inside ``multiprocessing`` workers on Windows, where each
worker re-imports its module. Keeping them in a module that does NOT import
torch lets worker processes load only numpy + seal + the CKKS backend, so the
parent's large torch/model footprint is not multiplied across every worker.

The parent orchestrates chunked aggregation from ``fmnist_lenet5_dynamic`` and
passes each chunk's ``(update_count x chunk_size)`` float32 column slice as an
argument, so no large shared-memory block is needed.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from .he_backend import (
    CKKS_COEFF_MOD_BIT_SIZES,
    CKKS_POLY_MODULUS_DEGREE,
    CKKS_SCALE,
    decode_seal_vector,
)

_SEAL_PROCESS_RUNTIME: dict[str, Any] | None = None


def _init_seal_process_runtime(
    public_key_path: str,
    secret_key_path: str,
    factors: tuple[float, ...],
    encrypted_mask: tuple[bool, ...],
) -> None:
    global _SEAL_PROCESS_RUNTIME

    import seal

    parms = seal.EncryptionParameters(seal.scheme_type.ckks)
    parms.set_poly_modulus_degree(CKKS_POLY_MODULUS_DEGREE)
    parms.set_coeff_modulus(
        seal.CoeffModulus.Create(
            CKKS_POLY_MODULUS_DEGREE,
            list(CKKS_COEFF_MOD_BIT_SIZES),
        )
    )
    context = seal.SEALContext(parms)
    public_key = seal.PublicKey()
    public_key.load(context, public_key_path)
    secret_key = seal.SecretKey()
    secret_key.load(context, secret_key_path)
    _SEAL_PROCESS_RUNTIME = {
        "factors": factors,
        "encrypted_mask": encrypted_mask,
        "encoder": seal.CKKSEncoder(context),
        "encryptor": seal.Encryptor(context, public_key),
        "decryptor": seal.Decryptor(context, secret_key),
        "evaluator": seal.Evaluator(context),
    }


def _seal_process_aggregate_chunk(
    task: tuple[int, np.ndarray],
) -> tuple[int, np.ndarray, int, int, float, float, float, float]:
    start, columns = task
    if _SEAL_PROCESS_RUNTIME is None:
        raise RuntimeError("SEAL process runtime was not initialized")

    width = int(columns.shape[1])
    runtime = _SEAL_PROCESS_RUNTIME
    encoder = runtime["encoder"]
    encryptor = runtime["encryptor"]
    decryptor = runtime["decryptor"]
    evaluator = runtime["evaluator"]
    encrypted_sum = None
    plaintext_sum = np.zeros(width, dtype=np.float64)
    expected_chunk = np.zeros(width, dtype=np.float64)
    ciphertext_count = 0
    ciphertext_bytes = 0
    encryption_time = 0.0
    addition_time = 0.0

    for row, factor, encrypted in zip(
        columns,
        runtime["factors"],
        runtime["encrypted_mask"],
    ):
        values = np.ascontiguousarray(
            row.astype(np.float64) * factor,
            dtype=np.float64,
        )
        expected_chunk += values
        if not encrypted:
            plaintext_sum += values
            continue
        encryption_started_at = time.perf_counter()
        ciphertext = encryptor.encrypt(encoder.encode(values, CKKS_SCALE))
        encryption_time += time.perf_counter() - encryption_started_at
        ciphertext_count += 1
        ciphertext_bytes += int(ciphertext.save_size())
        if encrypted_sum is None:
            encrypted_sum = ciphertext
        else:
            addition_started_at = time.perf_counter()
            encrypted_sum = evaluator.add(encrypted_sum, ciphertext)
            addition_time += time.perf_counter() - addition_started_at

    if encrypted_sum is None:
        raise ValueError("Encrypted aggregation requires at least one HE protected update")
    addition_started_at = time.perf_counter()
    encrypted_sum = evaluator.add_plain(
        encrypted_sum,
        encoder.encode(np.ascontiguousarray(plaintext_sum), CKKS_SCALE),
    )
    addition_time += time.perf_counter() - addition_started_at
    decryption_started_at = time.perf_counter()
    decoded = decode_seal_vector(encoder, decryptor.decrypt(encrypted_sum))[:width]
    decryption_time = time.perf_counter() - decryption_started_at
    max_abs_error = float(np.max(np.abs(decoded - expected_chunk)))
    return (
        start,
        np.ascontiguousarray(decoded, dtype=np.float32),
        ciphertext_count,
        ciphertext_bytes,
        encryption_time,
        addition_time,
        decryption_time,
        max_abs_error,
    )
