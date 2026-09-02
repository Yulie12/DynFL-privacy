from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.he_backend import CKKS_SCALE
from dynfed.fmnist_lenet5_dynamic import _reset_ckks_runtime, _seal_ckks_runtime


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark parallel SEAL CKKS encryption.")
    parser.add_argument("--local-deps", default=".he_deps")
    parser.add_argument("--encryptions", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    deps = str((ROOT / args.local_deps).resolve())
    if deps not in sys.path:
        sys.path.insert(0, deps)

    import seal

    _reset_ckks_runtime()
    runtime, key_setup_time = _seal_ckks_runtime()
    context = runtime["context"]
    public_key = runtime["public_key"]
    values = np.linspace(-1.0, 1.0, max(1, args.chunk_size), dtype=np.float64)
    local = threading.local()

    def encrypt_once(_index: int) -> int:
        if not hasattr(local, "encoder"):
            local.encoder = seal.CKKSEncoder(context)
            local.encryptor = seal.Encryptor(context, public_key)
        plaintext = local.encoder.encode(values, CKKS_SCALE)
        ciphertext = local.encryptor.encrypt(plaintext)
        return int(ciphertext.save_size())

    encrypt_once(0)
    rows: list[dict[str, float | int]] = []
    for workers in sorted(set(max(1, int(value)) for value in args.workers)):
        started_at = time.perf_counter()
        if workers == 1:
            sizes = [encrypt_once(index) for index in range(args.encryptions)]
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                sizes = list(executor.map(encrypt_once, range(args.encryptions)))
        wall_time = time.perf_counter() - started_at
        rows.append(
            {
                "workers": workers,
                "encryptions": int(args.encryptions),
                "chunk_size": int(values.size),
                "wall_time_sec": wall_time,
                "encryptions_per_sec": float(args.encryptions) / max(wall_time, 1e-12),
                "ciphertext_bytes": int(sizes[0]),
                "key_setup_time_sec": key_setup_time,
            }
        )

    payload = {"results": rows}
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
