from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.he_backend import decode_seal_vector


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark addition only CKKS parameter chains.")
    parser.add_argument("--local-deps", default=".he_deps")
    parser.add_argument("--encryptions", type=int, default=100)
    parser.add_argument("--payload-mb", type=float, default=4.0)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = str((ROOT / args.local_deps).resolve())
    if deps not in sys.path:
        sys.path.insert(0, deps)
    import seal

    scale = 2**40
    rows: list[dict[str, object]] = []
    parameter_sets = (
        (8192, [40, 40, 40, 40]),
        (8192, [50, 40]),
        (4096, [50, 40]),
    )
    for degree, bits in parameter_sets:
        values = np.linspace(-0.1, 0.1, degree // 2, dtype=np.float64)
        parameters = seal.EncryptionParameters(seal.scheme_type.ckks)
        parameters.set_poly_modulus_degree(degree)
        parameters.set_coeff_modulus(seal.CoeffModulus.Create(degree, bits))
        context = seal.SEALContext(parameters)
        keygen = seal.KeyGenerator(context)
        public_key = keygen.create_public_key()
        secret_key = keygen.secret_key()
        encoder = seal.CKKSEncoder(context)
        encryptor = seal.Encryptor(context, public_key)
        evaluator = seal.Evaluator(context)
        decryptor = seal.Decryptor(context, secret_key)

        started_at = time.perf_counter()
        ciphertexts = [
            encryptor.encrypt(encoder.encode(values / args.encryptions, scale))
            for _ in range(args.encryptions)
        ]
        encryption_time = time.perf_counter() - started_at
        started_at = time.perf_counter()
        encrypted_sum = ciphertexts[0]
        for ciphertext in ciphertexts[1:]:
            evaluator.add_inplace(encrypted_sum, ciphertext)
        addition_time = time.perf_counter() - started_at
        decoded = decode_seal_vector(
            encoder,
            decryptor.decrypt(encrypted_sum),
        )[: values.size]
        payload_value_count = max(1, int(args.payload_mb * 1_000_000 / 4))
        payload_bytes = 0
        payload_max_error = 0.0
        payload_started_at = time.perf_counter()
        for start in range(0, payload_value_count, values.size):
            count = min(values.size, payload_value_count - start)
            payload_values = values[:count]
            payload_ciphertext = encryptor.encrypt(
                encoder.encode(payload_values, scale)
            )
            payload_bytes += int(payload_ciphertext.save_size())
            payload_decoded = decode_seal_vector(
                encoder,
                decryptor.decrypt(payload_ciphertext),
            )[:count]
            payload_max_error = max(
                payload_max_error,
                float(np.max(np.abs(payload_decoded - payload_values))),
            )
        payload_wall_time = time.perf_counter() - payload_started_at
        rows.append(
            {
                "coeff_modulus_bits": bits,
                "poly_modulus_degree": degree,
                "slot_count": int(values.size),
                "encryptions": args.encryptions,
                "encryption_time_sec": encryption_time,
                "addition_time_sec": addition_time,
                "ciphertext_bytes": int(ciphertexts[0].save_size()),
                "encrypted_values_per_sec": (
                    float(args.encryptions * values.size) / max(encryption_time, 1e-12)
                ),
                "max_abs_error": float(np.max(np.abs(decoded - values))),
                "payload_mb": float(args.payload_mb),
                "payload_ciphertext_bytes": payload_bytes,
                "payload_wall_time_sec": payload_wall_time,
                "payload_max_abs_error": payload_max_error,
                "max_coeff_modulus_bits_at_128_security": int(
                    seal.CoeffModulus.MaxBitCount(degree, seal.sec_level_type.tc128)
                ),
            }
        )

    rendered = json.dumps({"results": rows}, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
