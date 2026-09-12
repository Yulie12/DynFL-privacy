"""Designated trusted key holder with an authorized ciphertext-only cloud worker.

Local process separation only. No host sandbox, malicious custodian protection,
network authentication, threshold cryptography, or production DP claim.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import uuid

import numpy as np
import seal

from dynfed.he_backend import (CKKS_COEFF_MOD_BIT_SIZES, CKKS_POLY_MODULUS_DEGREE,
                              CKKS_SCALE, decode_seal_vector)


class TrustedEdgeCustodian:
    def __init__(self, groups, *, custodian_edge=0, release_limit=100):
        setup_started = time.perf_counter()
        self.groups = tuple((g["edge"], float(g["cloud_weight"])) for g in groups)
        ids = [i for i, _ in self.groups]
        weights = [w for _, w in self.groups]
        if (not ids or len(set(ids)) != len(ids) or custodian_edge not in ids
                or any(not math.isfinite(w) or w <= 0 for w in weights)
                or not math.isclose(sum(weights), 1, rel_tol=0, abs_tol=1e-12)):
            raise ValueError("Fixed nonempty normalized cohort including custodian required")
        if type(release_limit) is not int or release_limit < 1:
            raise ValueError("Positive release limit required")
        self.release_limit, self.custodian_edge = release_limit, custodian_edge
        self.task_id = uuid.uuid4().hex
        self.next_round, self.pending, self.dimensions = 0, None, None
        self.storage = tempfile.TemporaryDirectory(prefix="dynfl_custodian_")
        self.root = Path(self.storage.name)
        params = seal.EncryptionParameters(seal.scheme_type.ckks)
        params.set_poly_modulus_degree(CKKS_POLY_MODULUS_DEGREE)
        params.set_coeff_modulus(seal.CoeffModulus.Create(CKKS_POLY_MODULUS_DEGREE,
                                                       list(CKKS_COEFF_MOD_BIT_SIZES)))
        self.context = seal.SEALContext(params)
        generator = seal.KeyGenerator(self.context)
        self.public_key = generator.create_public_key()
        self._secret_key = generator.secret_key()
        self.encoder = seal.CKKSEncoder(self.context)
        self.encryptor = seal.Encryptor(self.context, self.public_key)
        self._decryptor = seal.Decryptor(self.context, self._secret_key)
        self._evaluator = seal.Evaluator(self.context)
        self.key_setup_time_sec = time.perf_counter() - setup_started

    def close(self):
        self.storage.cleanup()

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def prepare(self, packets, round_idx):
        """Trusted-side authorization; must never be exposed as a cloud RPC.

        Inputs are already noisy edge packets. Weighting and encryption are
        simulated in the trusted process, not on independently deployed edges.
        """
        if self.pending is not None or round_idx != self.next_round or round_idx >= self.release_limit:
            raise ValueError("Round pending, replayed, out of order, or beyond release limit")
        if set(packets) != {i for i, _ in self.groups}:
            raise ValueError("Cohort mismatch")
        vectors = [np.asarray(packets[i], dtype=np.float64) for i, _ in self.groups]
        if any(v.ndim != 1 or v.size == 0 or v.shape != vectors[0].shape or not np.isfinite(v).all()
               for v in vectors):
            raise ValueError("Finite same-layout packets required")
        dimensions = vectors[0].size
        if self.dimensions is not None and dimensions != self.dimensions:
            raise ValueError("Model layout changed")
        self.dimensions = dimensions
        authorization = uuid.uuid4().hex
        directory = self.root / authorization
        directory.mkdir()
        manifest = dict(task_id=self.task_id, round=round_idx, authorization=authorization,
                        dimensions=dimensions, groups=self.groups,
                        ckks_degree=CKKS_POLY_MODULUS_DEGREE,
                        coeff_modulus_bits=list(CKKS_COEFF_MOD_BIT_SIZES), chunks=[])
        expected = []
        expected_digests = []
        encryption_sec, verify_sum_sec, ciphertext_bytes = 0.0, 0.0, 0
        chunk_size = CKKS_POLY_MODULUS_DEGREE // 2
        for chunk, start in enumerate(range(0, dimensions, chunk_size)):
            records, accumulator = [], None
            for position, ((edge, weight), vector) in enumerate(zip(self.groups, vectors)):
                began = time.perf_counter()
                values = np.ascontiguousarray(vector[start:start + chunk_size] * weight)
                encrypted = self.encryptor.encrypt(self.encoder.encode(values, CKKS_SCALE))
                encryption_sec += time.perf_counter() - began
                file = directory / f"input_{chunk}_{position}.ct"
                encrypted.save(str(file))
                data = file.read_bytes()
                ciphertext_bytes += len(data)
                records.append(dict(edge=edge, sha256=hashlib.sha256(data).hexdigest()))
                began = time.perf_counter()
                accumulator = encrypted if accumulator is None else self._evaluator.add(accumulator, encrypted)
                verify_sum_sec += time.perf_counter() - began
            manifest["chunks"].append(records)
            # Retain trusted expected sums; never decrypt caller-provided bytes.
            expected.append(accumulator)
            scratch = self.root / "serialization.ct"
            accumulator.save(str(scratch))
            expected_digests.append(hashlib.sha256(scratch.read_bytes()).hexdigest())
        manifest_bytes = json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode()
        (directory / "request.json").write_bytes(manifest_bytes)
        self.pending = dict(directory=directory, manifest=manifest,
                            manifest_digest=hashlib.sha256(manifest_bytes).hexdigest(),
                            expected=expected, expected_digests=expected_digests,
                            encryption_sec=encryption_sec, verify_sum_sec=verify_sum_sec,
                            ciphertext_bytes=ciphertext_bytes)
        return directory

    def run_cloud(self):
        if self.pending is None:
            raise ValueError("No authorized round")
        worker = Path(__file__).with_name("he_cloud_worker.py")
        began = time.perf_counter()
        result = subprocess.run([sys.executable, str(worker), str(self.pending["directory"])],
                                check=True, capture_output=True, text=True, timeout=120,
                                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.pending["cloud_process_sec"] = time.perf_counter() - began
        return result

    def release(self):
        if self.pending is None:
            raise ValueError("No authorized round or already released")
        p = self.pending
        response = json.loads((p["directory"] / "response.json").read_text(encoding="utf-8"))
        for field in ("task_id", "round", "authorization"):
            if response.get(field) != p["manifest"][field]:
                raise ValueError(f"Authorization mismatch: {field}")
        outputs = [f"output_{i}.ct" for i in range(len(p["expected"]))]
        if response.get("manifest_digest") != p["manifest_digest"] or response.get("outputs") != outputs:
            raise ValueError("Manifest or output layout mismatch")
        for name, digest in zip(outputs, p["expected_digests"]):
            if hashlib.sha256((p["directory"] / name).read_bytes()).hexdigest() != digest:
                raise ValueError("Ciphertext is not the authorized full sum")
        # Consume before returning a plaintext; no second decryption for this round.
        self.pending = None
        self.next_round += 1
        began = time.perf_counter()
        result = np.concatenate([decode_seal_vector(self.encoder, self._decryptor.decrypt(value))
                                 for value in p["expected"]])[:self.dimensions]
        if not np.isfinite(result).all():
            raise ValueError("Nonfinite decryption")
        metrics = dict(backend="seal", encrypted_updates=len(self.groups),
                       key_setup_time_sec=self.key_setup_time_sec if p["manifest"]["round"] == 0 else 0.0,
                       encrypted_parameter_values=len(self.groups) * self.dimensions,
                       ciphertext_count=len(self.groups) * len(p["expected"]),
                       ciphertext_bytes=p["ciphertext_bytes"],
                       encryption_time_sec=p["encryption_sec"],
                       trusted_ciphertext_verification_sum_sec=p["verify_sum_sec"],
                       decryption_time_sec=time.perf_counter() - began,
                       cloud_process_sec=p.get("cloud_process_sec"),
                       cloud_pid=response.get("cloud_pid"), custodian_pid=os.getpid(),
                       cloud_secret_key_transmitted=False, key_isolation_enforced=True,
                       aggregate_only_decryption_enforced=True,
                       isolation_scope="protocol_API_and_separate_process_not_OS_sandbox",
                       round_binding_enforced=True, replay_rejection_enforced=True,
                       trusted_full_key_holder=self.custodian_edge,
                       custodian_can_bypass_policy=True, threshold_decryption=False,
                       task_id=self.task_id, released_round=p["manifest"]["round"])
        return result, metrics

    def aggregate(self, packets, round_idx):
        began = time.perf_counter()
        self.prepare(packets, round_idx)
        self.run_cloud()
        result, metrics = self.release()
        metrics["wall_time_sec"] = time.perf_counter() - began
        return result, metrics
