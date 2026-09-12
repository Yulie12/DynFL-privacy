"""Ciphertext-only cloud arithmetic worker. No decryption API or private key."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import os
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import seal

from dynfed.he_backend import CKKS_COEFF_MOD_BIT_SIZES, CKKS_POLY_MODULUS_DEGREE


def cloud_sum(directory):
    directory = Path(directory)
    manifest_bytes = (directory / "request.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    params = seal.EncryptionParameters(seal.scheme_type.ckks)
    params.set_poly_modulus_degree(CKKS_POLY_MODULUS_DEGREE)
    params.set_coeff_modulus(seal.CoeffModulus.Create(CKKS_POLY_MODULUS_DEGREE,
                                                   list(CKKS_COEFF_MOD_BIT_SIZES)))
    context = seal.SEALContext(params)
    evaluator = seal.Evaluator(context)
    started = time.perf_counter()
    outputs = []
    for chunk, records in enumerate(manifest["chunks"]):
        accumulator = None
        for position, record in enumerate(records):
            name = f"input_{chunk}_{position}.ct"
            data = (directory / name).read_bytes()
            if hashlib.sha256(data).hexdigest() != record["sha256"]:
                raise ValueError("Input ciphertext digest mismatch")
            value = seal.Ciphertext()
            value.load(context, str(directory / name))
            accumulator = value if accumulator is None else evaluator.add(accumulator, value)
        if accumulator is None:
            raise ValueError("Empty cohort")
        name = f"output_{chunk}.ct"
        accumulator.save(str(directory / name))
        outputs.append(name)
    response = dict(task_id=manifest["task_id"], round=manifest["round"],
                    authorization=manifest["authorization"],
                    manifest_digest=hashlib.sha256(manifest_bytes).hexdigest(), outputs=outputs,
                    cloud_pid=os.getpid(), ciphertext_sum_sec=time.perf_counter() - started,
                    secret_key_received=False, plaintext_updates_received=False,
                    noise_shares_received=False)
    (directory / "response.json").write_text(json.dumps(response), encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    cloud_sum(parser.parse_args().directory)
