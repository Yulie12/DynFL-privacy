from __future__ import annotations

import argparse
import gc
import json
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.he_backend import (
    CKKS_COEFF_MOD_BIT_SIZES,
    CKKS_POLY_MODULUS_DEGREE,
    CKKS_SCALE,
    HEOperationMetrics,
)
from dynfed.fmnist_lenet5_dynamic import (
    _reset_ckks_runtime,
    _seal_ckks_runtime,
    fedavg_split_seal,
)


_PROCESS_ENCODER = None
_PROCESS_ENCRYPTOR = None
_PROCESS_VALUES = None


class _FlatParameterModule(torch.nn.Module):
    def __init__(self, parameter_count: int) -> None:
        super().__init__()
        self.values = torch.nn.Parameter(torch.zeros(parameter_count, dtype=torch.float32))


def _process_tree_rss_bytes() -> int | None:
    try:
        import psutil
    except ImportError:
        return None

    process = psutil.Process()
    total = process.memory_info().rss
    for child in process.children(recursive=True):
        try:
            total += child.memory_info().rss
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return int(total)


def _measure_peak_rss(operation):
    stop = threading.Event()
    peak = _process_tree_rss_bytes()

    def sample() -> None:
        nonlocal peak
        while not stop.wait(0.05):
            current = _process_tree_rss_bytes()
            if current is not None:
                peak = current if peak is None else max(peak, current)

    sampler = threading.Thread(target=sample, name="seal-rss-sampler", daemon=True)
    sampler.start()
    try:
        result = operation()
    finally:
        stop.set()
        sampler.join()
        current = _process_tree_rss_bytes()
        if current is not None:
            peak = current if peak is None else max(peak, current)
    return result, peak


def _full_aggregation_case(
    *,
    parameter_count: int,
    update_count: int,
    workers: int,
) -> dict[str, float | int | str | None]:
    end_model = _FlatParameterModule(parameter_count)
    edge_model = torch.nn.Module()
    update_scale = 1e-4
    state_diffs = [
        {
            "end": {
                "values": torch.full(
                    (parameter_count,),
                    update_scale * (index + 1),
                    dtype=torch.float32,
                )
            },
            "edge": {},
        }
        for index in range(update_count)
    ]
    sample_counts = [1.0] * update_count
    metrics = HEOperationMetrics(backend="seal")
    _reset_ckks_runtime()

    started_at = time.perf_counter()
    (_updated_end, _updated_edge), peak_rss_bytes = _measure_peak_rss(
        lambda: fedavg_split_seal(
            state_diffs,
            sample_counts,
            end_model,
            edge_model,
            torch.device("cpu"),
            encrypted_mask=[True] * update_count,
            he_aggregation_size=0,
            he_workers=workers,
            he_metrics=metrics,
        )
    )
    end_to_end_wall_time_sec = time.perf_counter() - started_at
    expected = update_scale * (update_count + 1) / 2.0
    model_max_abs_error = float(
        torch.max(torch.abs(end_model.values.detach() - expected)).item()
    )
    metric_values = metrics.as_dict()
    return {
        "benchmark": "full_aggregation",
        "workers": workers,
        "updates": update_count,
        "parameter_count": parameter_count,
        "plaintext_update_bytes": parameter_count * np.dtype(np.float32).itemsize,
        "estimated_process_shared_memory_bytes": (
            parameter_count * update_count * np.dtype(np.float32).itemsize
        ),
        "end_to_end_wall_time_sec": end_to_end_wall_time_sec,
        "peak_process_tree_rss_bytes": peak_rss_bytes,
        "model_max_abs_error": model_max_abs_error,
        **metric_values,
    }


def _init_process_worker(local_deps: str, public_key_path: str, chunk_size: int) -> None:
    global _PROCESS_ENCODER, _PROCESS_ENCRYPTOR, _PROCESS_VALUES

    if local_deps not in sys.path:
        sys.path.insert(0, local_deps)
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
    _PROCESS_ENCODER = seal.CKKSEncoder(context)
    _PROCESS_ENCRYPTOR = seal.Encryptor(context, public_key)
    _PROCESS_VALUES = np.linspace(-1.0, 1.0, max(1, chunk_size), dtype=np.float64)


def _process_encrypt_once(_index: int) -> int:
    plaintext = _PROCESS_ENCODER.encode(_PROCESS_VALUES, CKKS_SCALE)
    ciphertext = _PROCESS_ENCRYPTOR.encrypt(plaintext)
    return int(ciphertext.save_size())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark SEAL CKKS encryption and full encrypted FedAvg."
    )
    parser.add_argument("--local-deps", default=".he_deps")
    parser.add_argument("--encryptions", type=int, default=128)
    parser.add_argument("--chunk-size", type=int, default=4096)
    parser.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    parser.add_argument("--skip-processes", action="store_true")
    parser.add_argument(
        "--aggregate-parameter-count",
        type=int,
        default=0,
        help="Run full encrypted FedAvg cases with this many parameters; 0 disables them.",
    )
    parser.add_argument(
        "--aggregate-updates",
        type=int,
        nargs="+",
        default=[10, 20, 40],
        help="Encrypted participant counts for full aggregation cases.",
    )
    parser.add_argument(
        "--max-shared-memory-gb",
        type=float,
        default=8.0,
        help="Skip a full aggregation case when its input matrix exceeds this size.",
    )
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

    representative_ciphertext = runtime["encryptor"].encrypt(
        runtime["encoder"].encode(values, CKKS_SCALE)
    )
    representative_save_size_upper_bound = int(representative_ciphertext.save_size())
    with tempfile.TemporaryDirectory(prefix="dynfl_seal_serialized_") as temp_dir:
        serialized_path = Path(temp_dir) / "ciphertext.bin"
        representative_ciphertext.save(str(serialized_path))
        representative_serialized_bytes = serialized_path.stat().st_size

    encrypt_once(0)
    rows: list[dict[str, float | int | str | None]] = []
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
                "benchmark": "encryption_only",
                "executor": "thread",
                "workers": workers,
                "encryptions": int(args.encryptions),
                "chunk_size": int(values.size),
                "wall_time_sec": wall_time,
                "encryptions_per_sec": float(args.encryptions) / max(wall_time, 1e-12),
                "ciphertext_save_size_upper_bound_bytes": int(sizes[0]),
                "representative_serialized_ciphertext_bytes": representative_serialized_bytes,
                "key_setup_time_sec": key_setup_time,
            }
        )

    if not args.skip_processes:
        with tempfile.TemporaryDirectory(prefix="dynfl_seal_") as temp_dir:
            public_key_path = str(Path(temp_dir) / "public_key.bin")
            public_key.save(public_key_path)
            for workers in sorted(set(max(1, int(value)) for value in args.workers)):
                started_at = time.perf_counter()
                with ProcessPoolExecutor(
                    max_workers=workers,
                    initializer=_init_process_worker,
                    initargs=(deps, public_key_path, int(values.size)),
                ) as executor:
                    sizes = list(executor.map(_process_encrypt_once, range(args.encryptions)))
                wall_time = time.perf_counter() - started_at
                rows.append(
                    {
                        "benchmark": "encryption_only",
                        "executor": "process",
                        "workers": workers,
                        "encryptions": int(args.encryptions),
                        "chunk_size": int(values.size),
                        "wall_time_sec": wall_time,
                        "encryptions_per_sec": float(args.encryptions) / max(wall_time, 1e-12),
                        "ciphertext_save_size_upper_bound_bytes": int(sizes[0]),
                        "representative_serialized_ciphertext_bytes": representative_serialized_bytes,
                        "key_setup_time_sec": key_setup_time,
                    }
                )

    if args.aggregate_parameter_count > 0:
        parameter_count = int(args.aggregate_parameter_count)
        if parameter_count < 1:
            raise ValueError("--aggregate-parameter-count must be positive")
        max_shared_memory_bytes = int(args.max_shared_memory_gb * (1024 ** 3))
        for update_count in sorted(
            set(max(1, int(value)) for value in args.aggregate_updates)
        ):
            required_shared_memory_bytes = (
                parameter_count * update_count * np.dtype(np.float32).itemsize
            )
            for workers in sorted(set(max(1, int(value)) for value in args.workers)):
                if workers > 1 and required_shared_memory_bytes > max_shared_memory_bytes:
                    rows.append(
                        {
                            "benchmark": "full_aggregation",
                            "workers": workers,
                            "updates": update_count,
                            "parameter_count": parameter_count,
                            "estimated_process_shared_memory_bytes": (
                                required_shared_memory_bytes
                            ),
                            "status": "skipped_memory_guard",
                        }
                    )
                    continue
                rows.append(
                    _full_aggregation_case(
                        parameter_count=parameter_count,
                        update_count=update_count,
                        workers=workers,
                    )
                )
                gc.collect()

    payload = {
        "ciphertext_size_note": (
            "save_size is an upper bound; representative_serialized_ciphertext_bytes "
            "is measured by writing one ciphertext with the configured SEAL binding"
        ),
        "representative_save_size_upper_bound_bytes": representative_save_size_upper_bound,
        "results": rows,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
