from __future__ import annotations

import argparse
import functools
import hashlib
import json
import math
import mimetypes
import os
import shutil
import subprocess
import sys
import threading
import time
import csv
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dynfed.version import CURRENT_EXECUTION_REVISION, CURRENT_UPDATE_PARAMETER_SCOPE


TRAINING_PROCESS: subprocess.Popen | None = None
ACTIVE_OUTPUT_ROOT: Path | None = None
LAST_STOP_MESSAGE: str | None = None
TRAINING_STATE_LOCK = threading.RLock()
TRAINING_PYTHON_ENV = "DYNFED_TRAINING_PYTHON"
PRIVACY_SCHEMA_VERSION = "rdp_total_v1"


def _json_response(payload: object) -> bytes:
    return json.dumps(
        _json_safe(payload),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")


def _json_safe(value: object) -> object:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {key: _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _serialized_training_state(func):
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        with TRAINING_STATE_LOCK:
            return func(*args, **kwargs)

    return wrapped


BASE_ARGS = [
    "--clients", "100",
    "--edges", "10",
    "--train-limit", "12000",
    "--test-limit", "2000",
    "--lr", "0.15",
    "--seed", "42",
    "--initial-epsilon", "8.0",
    "--dp-event-epsilon", "0.05",
    "--dp-emb-epsilon", "8.0",
    "--dp-upd-epsilon", "8.0",
    "--dp-clip-norm", "1.0",
    "--dp-noise-multiplier", "0.0002",
    "--dp-accounting-mode", "rdp_auto",
    "--dp-delta", "1e-5",
    "--resource-limit", "1.35",
    "--time-limit", "8.0",
    "--risk-limit", "0.5",
    "--aggregation-fraction", "1.0",
]


DP_PROFILES = {
    "balanced": {
        "label": "Auto RDP (C=1)",
        "slug": "dpbal",
        "dp_event_epsilon": "0.05",
        "dp_clip_norm": "1.0",
        "dp_noise_multiplier": "0.0002",
        "dp_update_mode": "upd_only",
    },
    "strong": {
        "label": "Auto RDP (C=20)",
        "slug": "dpstrong",
        "dp_event_epsilon": "0.005",
        "dp_clip_norm": "20.0",
        "dp_noise_multiplier": "0.002",
        "dp_update_mode": "upd_only",
    },
    "cifar_resnet": {
        "label": "CIFAR Auto RDP (C=1)",
        "slug": "dpcifar02",
        "dp_event_epsilon": "0.02",
        "dp_clip_norm": "1.0",
        "dp_noise_multiplier": "0.0002",
        "dp_update_mode": "upd_only",
    },
    "weak_update": {
        "label": "Auto RDP (C=0.5)",
        "slug": "dpweak",
        "dp_event_epsilon": "0.1",
        "dp_clip_norm": "0.5",
        "dp_noise_multiplier": "0.0001",
        "dp_update_mode": "upd_only",
    },
}


def configured_training_hyperparams(dataset: str, model: str) -> tuple[str, str]:
    local_epochs = "1"
    lr = "0.15"
    if model in {"resnet18_pretrained", "resnet50_pretrained"} and dataset in {"cifar10", "cifar100"}:
        local_epochs = "3"
        lr = "0.01" if model == "resnet18_pretrained" else "0.005"
    elif model in {"resnet18", "resnet50"} and dataset in {"cifar10", "cifar100"}:
        local_epochs = "3"
        lr = "0.03" if model == "resnet18" else "0.02"
    elif dataset == "cifar100":
        lr = "0.005" if model == "tinyresnet" else "0.02"
    elif dataset == "cifar10":
        lr = "0.01" if model == "tinyresnet" else "0.03"
    elif model == "tinyresnet":
        lr = "0.03"
    return local_epochs, lr


def _clamped_learning_rate(value: float | None) -> float | None:
    if value is None:
        return None
    return max(0.000001, min(float(value), 1.0))


def _dp_profile_config(profile_name: str) -> tuple[str, dict[str, str]]:
    if profile_name not in DP_PROFILES:
        profile_name = "balanced"
    return profile_name, DP_PROFILES[profile_name]


PRESETS = {
    "paper100": {
        "label": "100r paper set",
        "output_root": "out/fmnist_lenet5_paper_100r",
        "args": [
            "experiments/run_fmnist_lenet5.py",
            "--rounds", "100",
            "--local-epochs", "1",
            *BASE_ARGS,
            "--output-root", "out/fmnist_lenet5_paper_100r",
            "--policies", "ours", "fixed_dp", "privacy_only", "no_protection", "random",
        ],
    },
    "paper100_core": {
        "label": "100r three-policy diagnostic",
        "output_root": "out/fmnist_lenet5_paper_100r_ep3",
        "args": [
            "experiments/run_fmnist_lenet5.py",
            "--rounds", "100",
            "--local-epochs", "1",
            *BASE_ARGS,
            "--output-root", "out/fmnist_lenet5_paper_100r_ep3",
            "--policies", "ours", "privacy_only", "random",
        ],
    },
    "random100": {
        "label": "100r random only",
        "output_root": "out/fmnist_lenet5_tex_100r_ep3_random",
        "args": [
            "experiments/run_fmnist_lenet5.py",
            "--rounds", "100",
            "--local-epochs", "1",
            *BASE_ARGS,
            "--output-root", "out/fmnist_lenet5_tex_100r_ep3_random",
            "--policies", "random",
        ],
    },
}


def custom_paper_preset(rounds: int) -> dict:
    rounds = max(1, min(int(rounds), 500))
    output_root = f"out/fmnist_lenet5_custom_{rounds}r"
    return {
        "label": f"{rounds}r custom paper set",
        "output_root": output_root,
        "args": [
            "experiments/run_fmnist_lenet5.py",
            "--rounds", str(rounds),
            "--local-epochs", "1",
            *BASE_ARGS,
            "--output-root", output_root,
            "--policies", "ours", "fixed_dp", "privacy_only", "no_protection", "random",
        ],
    }


def configured_preset(
    mode: str,
    rounds: int,
    time_limit: float,
    train_limit: int,
    test_limit: int,
    policies: list[str],
    figure_axis: str,
    partition_mode: str,
    clients: int,
    edges: int,
    seed: int,
    client_heterogeneity: float,
    edge_heterogeneity: float,
    selection_period: int,
    aggregation_fraction: float,
    pareto_archive_size: int,
    pareto_max_iters: int,
    pareto_neighbor_top_k: int,
    pareto_conflict_only: bool,
    cloud_fusion_xi: float,
    cloud_fusion_eps: float,
    min_edge_cloud_fusion_ratio: float,
    resource_limit: float,
    risk_limit: float,
    executor: str,
    executor_workers: int | None,
    local_epochs: int | None,
    learning_rate: float | None,
    initial_epsilon: float,
    dp_emb_epsilon: float,
    dp_upd_epsilon: float,
    he_backend: str,
    require_real_he: bool,
    he_aggregation_size: int,
    dp_profile: str,
    dataset: str,
    model: str,
    device: str,
    resume_from_run: str = "",
) -> dict:
    allowed = {
        "ours",
        "ours_no_omega",
        "ours_fixed_liieiiic",
        "fixed_dp",
        "privacy_only",
        "no_protection",
        "random",
        "fixed_fedavg",
        "fixed_splitfed",
        "fixed_hfl",
        "nsga2",
        "fixed_liieiiic",
        "individual_optimal",
        "performance_only",
        "best_accuracy",
        "accuracy_oracle",
    }
    selected = [policy for policy in policies if policy in allowed]
    if not selected:
        selected = ["ours"]
    rounds = max(1, min(int(rounds), 500))
    train_limit = max(1, min(int(train_limit), 60000))
    test_limit = max(1, min(int(test_limit), 10000))
    seed = max(0, min(int(seed), 999999))
    time_limit = max(0.1, min(float(time_limit), 300.0))
    clients = max(1, min(int(clients), 200))
    edges = max(1, min(int(edges), 20))
    mode = mode if mode in {"rounds", "time"} else "rounds"
    figure_axis = figure_axis if figure_axis in {"round", "time"} else "round"
    partition_mode = partition_mode if partition_mode in {"iid", "client_noniid", "edge_label_skew", "extreme_edge_label_skew"} else "client_noniid"
    dataset = dataset if dataset in {"fmnist", "cifar10", "cifar100"} else "fmnist"
    model = model if model in {
        "lenet5",
        "smallcnn",
        "avgcnn",
        "tinyresnet",
        "resnet18",
        "resnet50",
        "resnet18_pretrained",
        "resnet50_pretrained",
    } else "lenet5"
    client_heterogeneity = max(1.0, min(float(client_heterogeneity), 10.0))
    edge_heterogeneity = max(1.0, min(float(edge_heterogeneity), 10.0))
    selection_period = max(1, min(int(selection_period), 100))
    aggregation_fraction = max(0.1, min(float(aggregation_fraction), 1.0))
    pareto_archive_size = max(2, min(int(pareto_archive_size), 128))
    pareto_max_iters = max(0, min(int(pareto_max_iters), 200))
    pareto_neighbor_top_k = max(0, min(int(pareto_neighbor_top_k), 1000))
    cloud_fusion_xi = max(0.0, min(float(cloud_fusion_xi), 100.0))
    cloud_fusion_eps = max(0.000001, min(float(cloud_fusion_eps), 10.0))
    min_edge_cloud_fusion_ratio = max(
        0.0, min(float(min_edge_cloud_fusion_ratio), 1.0)
    )
    resource_limit = max(0.0, min(float(resource_limit), 100.0))
    risk_limit = max(0.0, min(float(risk_limit), 1.0))
    executor = executor if executor in {"serial", "process_pool"} else "serial"
    effective_executor_workers = (
        max(1, min(int(executor_workers), 64))
        if executor_workers is not None
        else None
    )
    initial_epsilon = max(0.0, min(float(initial_epsilon), 100.0))
    dp_emb_epsilon = max(0.001, min(float(dp_emb_epsilon), 100.0))
    dp_upd_epsilon = max(0.001, min(float(dp_upd_epsilon), 100.0))
    he_backend = he_backend if he_backend in {"none", "seal", "tenseal"} else "none"
    he_aggregation_size = max(0, min(int(he_aggregation_size), 2000000))
    default_local_epochs, default_lr = configured_training_hyperparams(dataset, model)
    effective_local_epochs = max(1, min(int(local_epochs), 20)) if local_epochs is not None else int(default_local_epochs)
    effective_lr = _clamped_learning_rate(learning_rate)
    effective_lr_text = f"{effective_lr:g}" if effective_lr is not None else default_lr
    dp_profile, dp_config = _dp_profile_config(dp_profile)
    device = device if device in {"cpu", "cuda"} else "cpu"
    policy_abbrev = {
        "ours": "ours",
        "ours_no_omega": "noomega",
        "ours_fixed_liieiiic": "fixedmode",
        "individual_optimal": "ind",
        "fixed_dp": "fdp",
        "privacy_only": "priv",
        "no_protection": "nop",
        "random": "rnd",
        "fixed_fedavg": "fedavg",
        "fixed_splitfed": "splitfed",
        "fixed_hfl": "hfl",
        "nsga2": "nsga2",
        "fixed_liieiiic": "hfl",
        "performance_only": "bal",
        "best_accuracy": "util",
        "accuracy_oracle": "oracle",
    }
    selected_slug = "_".join(policy_abbrev.get(policy, policy[:6]) for policy in selected)
    selected_hash = hashlib.sha1(",".join(selected).encode("utf-8")).hexdigest()[:8]
    method_slug = f"{len(selected)}m_{selected_hash}"
    model_slug = {
        "resnet18_pretrained": "r18pt",
        "resnet50_pretrained": "r50pt",
        "tinyresnet": "tinyrn",
        "smallcnn": "scnn",
    }.get(model, model)
    partition_slug = {
        "extreme_edge_label_skew": "xels",
        "edge_label_skew": "els",
        "client_noniid": "cniid",
        "iid": "iid",
    }.get(partition_mode, partition_mode)
    output_descriptor = (
        f"{dataset}_{model_slug}_{partition_slug}_{clients}c_{edges}e_{mode}_{rounds}r_"
        f"seed{seed}_t{str(time_limit).replace('.', 'p')}_"
        f"rl{str(resource_limit).replace('.', 'p')}_risk{str(risk_limit).replace('.', 'p')}_"
        f"ch{str(client_heterogeneity).replace('.', 'p')}_eh{str(edge_heterogeneity).replace('.', 'p')}_"
        f"sp{selection_period}_agg{str(aggregation_fraction).replace('.', 'p')}_"
        f"kp{pareto_archive_size}_im{pareto_max_iters}_"
        f"nk{pareto_neighbor_top_k}_{'conf' if pareto_conflict_only else 'all'}_"
        f"xi{str(cloud_fusion_xi).replace('.', 'p')}_ce{str(cloud_fusion_eps).replace('.', 'p')}_"
        f"rmin{str(min_edge_cloud_fusion_ratio).replace('.', 'p')}_"
        f"ep{effective_local_epochs}_"
        f"lr{effective_lr_text.replace('.', 'p')}_"
        f"eps{str(initial_epsilon).replace('.', 'p')}_"
        f"emb{str(dp_emb_epsilon).replace('.', 'p')}_"
        f"upd{str(dp_upd_epsilon).replace('.', 'p')}_"
        f"he{'full' if he_aggregation_size == 0 else he_aggregation_size}_"
        f"{dp_config['slug']}_"
        f"{executor}{effective_executor_workers or ''}_"
        f"{device}_"
        f"{method_slug}"
    )
    config_hash = hashlib.sha1(output_descriptor.encode("utf-8")).hexdigest()[:10]
    executor_slug = "pool" if executor == "process_pool" else "ser"
    output_root = (
        f"out/{dataset}_{model_slug}_{partition_slug}_{clients}c_{edges}e_"
        f"{mode}{rounds}r_s{seed}_sp{selection_period}_{dp_config['slug']}_"
        f"{executor_slug}_{device}_{method_slug}_{config_hash}"
    )
    args = [
        "experiments/run_fmnist_lenet5.py",
        "--execution-revision", CURRENT_EXECUTION_REVISION,
        "--rounds", str(rounds),
        "--dataset", dataset,
        "--model", model,
        "--local-epochs", str(effective_local_epochs),
        "--lr", effective_lr_text,
        "--clients", str(clients),
        "--edges", str(edges),
        "--train-limit", str(train_limit),
        "--test-limit", str(test_limit),
        "--seed", str(seed),
        "--initial-epsilon", f"{initial_epsilon:g}",
        "--partition-mode", partition_mode,
        "--selection-period", str(selection_period),
        "--aggregation-fraction", str(aggregation_fraction),
        "--pareto-archive-size", str(pareto_archive_size),
        "--pareto-max-iters", str(pareto_max_iters),
        "--pareto-neighbor-top-k", str(pareto_neighbor_top_k),
        "--cloud-fusion-xi", f"{cloud_fusion_xi:g}",
        "--cloud-fusion-eps", f"{cloud_fusion_eps:g}",
        "--edge-aggregation-beta", "0.01",
        "--edge-aggregation-fixed", "0.02",
        "--cloud-aggregation-beta", "0.015",
        "--cloud-aggregation-fixed", "0.04",
        "--min-edge-cloud-fusion-ratio", f"{min_edge_cloud_fusion_ratio:g}",
        "--resource-limit", f"{resource_limit:g}",
        "--memory-limit", f"{resource_limit:g}",
        "--time-limit", f"{time_limit:g}",
        "--risk-limit", f"{risk_limit:g}",
        "--dp-event-epsilon", "0.05",
        "--dp-emb-epsilon", f"{dp_emb_epsilon:g}",
        "--dp-upd-epsilon", f"{dp_upd_epsilon:g}",
        "--dp-feature-epsilon-budget", f"{dp_emb_epsilon:g}",
        "--dp-update-epsilon-budget", f"{dp_upd_epsilon:g}",
        "--dp-accounting-mode", "rdp_auto",
        "--dp-delta", "1e-5",
        "--dp-clip-norm", dp_config["dp_clip_norm"],
        "--dp-noise-multiplier", dp_config["dp_noise_multiplier"],
        "--dp-update-mode", dp_config["dp_update_mode"],
        "--he-backend", he_backend,
        "--he-execution", "real" if require_real_he else "profiled",
        "--he-aggregation-size", str(he_aggregation_size),
        "--executor", executor,
        "--client-heterogeneity", str(client_heterogeneity),
        "--edge-heterogeneity", str(edge_heterogeneity),
        "--device", device,
        "--require-feasible",
        "--require-edge-cloud-coverage",
        "--trusted-edge-split-execution",
        "--enforce-cloud-dp-stability",
        "--cloud-dp-stability-threshold", "1.0",
        "--output-root", output_root,
        "--policies", *selected,
    ]
    duplicate_options = sorted(
        option
        for option in set(arg for arg in args if arg.startswith("--"))
        if args.count(option) > 1
    )
    if duplicate_options:
        raise RuntimeError(
            "Configured command contains duplicate options: "
            + ", ".join(duplicate_options)
        )
    if resume_from_run:
        args.extend(["--resume-from-run", resume_from_run])
    if require_real_he:
        args.append("--require-real-he")
    if effective_executor_workers is not None:
        args.extend(["--executor-workers", str(effective_executor_workers)])
    if pareto_conflict_only:
        args.append("--pareto-conflict-only")
    return {
        "label": f"configured {mode} experiment",
        "output_root": output_root,
        "figure_axis": figure_axis,
        "args": args,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Monitor live LeNet5 experiment progress.")
    parser.add_argument("--output-root", default="out/fmnist_lenet5")
    parser.add_argument("--port", type=int, default=8765)
    return parser.parse_args()


def latest_status_path(output_root: Path) -> Path | None:
    candidates = list(output_root.rglob("live_status.json"))
    if output_root.name == "live_status.json" and output_root.exists():
        candidates.append(output_root)
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def read_status(output_root: Path) -> dict:
    status_path = latest_status_path(output_root)
    if status_path is None:
        if (
            TRAINING_PROCESS is not None
            and TRAINING_PROCESS.poll() is not None
            and ACTIVE_OUTPUT_ROOT is not None
            and output_root.resolve() == ACTIVE_OUTPUT_ROOT.resolve()
        ):
            return {
                "status": "failed",
                "message": (
                    f"Training process exited with code {TRAINING_PROCESS.returncode} before writing live_status.json. "
                    "Check out/configured_train.err.log."
                ),
                "output_root": str(output_root),
                "recent_rounds": [],
                "policy_statuses": [],
                "run_figures": [],
                "paper_experiments": available_paper_experiments(),
                "merged_runs": available_merged_runs(),
                "merge_sources": available_merge_sources(),
                "delete_targets": available_delete_targets(),
            }
        latest_payload = latest_completed_status_payload()
        if latest_payload is not None and not _training_process_is_running():
            return latest_payload
        return {
            "status": "waiting",
            "message": f"Waiting for live_status.json under {output_root}",
            "output_root": str(output_root),
            "recent_rounds": [],
            "policy_statuses": [],
            "run_figures": [],
            "paper_experiments": available_paper_experiments(),
            "merged_runs": available_merged_runs(),
            "merge_sources": available_merge_sources(),
            "delete_targets": available_delete_targets(),
        }
    try:
        with status_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except json.JSONDecodeError:
        payload = {"status": "loading", "message": "Status file is being updated", "recent_rounds": []}
    run_dir = _infer_run_dir(status_path, payload)
    payload["status_path"] = str(status_path)
    payload["run_dir"] = str(run_dir)
    payload["policy_statuses"] = read_policy_statuses(run_dir, payload)
    run_figures = available_run_figures(run_dir)
    if not run_figures and _is_completed_run(run_dir):
        build_current_run_figures(run_dir)
        run_figures = available_run_figures(run_dir)
    payload["run_figures"] = run_figures
    payload["paper_experiments"] = available_paper_experiments()
    payload["merged_runs"] = available_merged_runs()
    payload["merge_sources"] = available_merge_sources()
    payload["delete_targets"] = available_delete_targets()
    return payload


def _training_process_is_running() -> bool:
    return bool(TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None)


def latest_completed_status_payload() -> dict | None:
    latest_path = ROOT / "out" / "_latest_status.json"
    if not latest_path.exists():
        return None
    try:
        with latest_path.open("r", encoding="utf-8-sig") as file:
            payload = json.load(file)
    except (json.JSONDecodeError, OSError):
        return None
    run_dir = Path(str(payload.get("run_dir", "")))
    if not run_dir.is_absolute():
        run_dir = ROOT / run_dir
    if not run_dir.exists():
        status_path = Path(str(payload.get("status_path", "")))
        if not status_path.is_absolute():
            status_path = ROOT / status_path
        if status_path.exists():
            run_dir = _infer_run_dir(status_path, payload)
    if not run_dir.exists():
        return None
    status_path = run_dir / "live_status.json"
    payload["status_path"] = str(status_path if status_path.exists() else latest_path)
    payload["run_dir"] = str(run_dir)
    payload["output_root"] = str(run_dir.parent)
    payload["message"] = payload.get("message") or f"Loaded latest completed run: {run_dir}"
    payload["policy_statuses"] = read_policy_statuses(run_dir, payload)
    run_figures = available_run_figures(run_dir)
    if not run_figures and _is_completed_run(run_dir):
        build_current_run_figures(run_dir)
        run_figures = available_run_figures(run_dir)
    payload["run_figures"] = run_figures
    payload["paper_experiments"] = available_paper_experiments()
    payload["merged_runs"] = available_merged_runs()
    payload["merge_sources"] = available_merge_sources()
    payload["delete_targets"] = available_delete_targets()
    payload["latest_status_fallback"] = True
    return payload


def available_run_figures(run_dir: Path) -> list[dict]:
    figure_dir = run_dir / "figures"
    items = []
    for name, label in [
        ("current_round_accuracy.png", "Current run: Round-Accuracy"),
        ("current_time_accuracy.png", "Current run: Time-Accuracy"),
    ]:
        path = figure_dir / name
        if path.exists():
            items.append({"name": name, "label": label, "mtime": path.stat().st_mtime})
    return items


def available_paper_experiments() -> list[dict]:
    archive_root = ROOT / "out" / "paper_experiments"
    if not archive_root.exists():
        return []
    items = []
    for path in sorted(archive_root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)[:8]:
        if not path.is_dir():
            continue
        figures = []
        for figure in sorted((path / "figures").glob("*.png")) if (path / "figures").exists() else []:
            figures.append({"name": figure.name, "path": str(figure), "label": figure.stem})
        items.append({"name": path.name, "path": str(path), "mtime": path.stat().st_mtime, "figures": figures})
    return items


def available_merged_runs() -> list[dict]:
    merge_root = ROOT / "out" / "merged_runs"
    if not merge_root.exists():
        return []
    items = []
    for path in sorted(merge_root.iterdir(), key=lambda item: item.stat().st_mtime, reverse=True)[:8]:
        if not path.is_dir():
            continue
        figures = []
        figure_dir = path / "figures"
        for figure in sorted(figure_dir.glob("*.png")) if figure_dir.exists() else []:
            figures.append({"name": figure.name, "path": str(figure), "label": figure.stem})
        items.append({"name": path.name, "path": str(path), "mtime": path.stat().st_mtime, "figures": figures})
    return items


def available_merge_sources(limit: int = 80) -> list[dict]:
    items = []
    roots = [ROOT / "out"]
    for out_root in roots:
        if not out_root.exists():
            continue
        for summary_path in _safe_rglob(out_root, "summary.json"):
            policy_dir = summary_path.parent
            run_dir = policy_dir.parent
            if "merged_runs" in run_dir.parts:
                continue
            metrics_path = policy_dir / "round_metrics.csv"
            if not metrics_path.exists():
                continue
            try:
                with summary_path.open("r", encoding="utf-8") as file:
                    summary = json.load(file)
            except json.JSONDecodeError:
                continue
            config = _read_run_config(run_dir)
            if _is_legacy_tinyresnet_config(config):
                continue
            selection = config.get("selection", {}) if config else {}
            training = config.get("training", {}) if config else {}
            policy = str(summary.get("policy") or policy_dir.name)
            item = {
                "id": _merge_source_id(policy_dir),
                "policy": policy,
                "run_name": run_dir.name,
                "run_dir": str(run_dir),
                "policy_dir": str(policy_dir),
                "mtime": summary_path.stat().st_mtime,
                "rounds": summary.get("rounds"),
                "final_test_accuracy": summary.get("final_test_accuracy"),
                "best_test_accuracy": summary.get("best_test_accuracy"),
                "total_logical_time": summary.get("total_logical_time"),
                "per_client_test_mean": summary.get("per_client_test_accuracy_mean"),
                "per_client_test_min": summary.get("per_client_test_accuracy_min"),
                "per_client_test_max": summary.get("per_client_test_accuracy_max"),
                "per_client_test_std": summary.get("per_client_test_accuracy_std"),
                "dataset_name": training.get("dataset_name") or summary.get("dataset"),
                "model_name": training.get("model_name") or summary.get("model"),
                "model_revision": training.get("model_revision"),
                "device": training.get("device"),
                "learning_rate": training.get("learning_rate"),
                "selection_period": training.get("selection_period"),
                "client_heterogeneity": selection.get("client_heterogeneity"),
                "edge_heterogeneity": selection.get("edge_heterogeneity"),
                "num_edges": selection.get("num_edges"),
            }
            item["param_suffix"] = _parameter_suffix(item)
            items.append(item)
    items.sort(key=lambda item: item["mtime"], reverse=True)
    return items[:limit]


def _read_run_config(run_dir: Path) -> dict:
    config_path = run_dir / "config.json"
    if not config_path.exists():
        return {}
    try:
        with config_path.open("r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return {}


def _is_legacy_tinyresnet_config(config: dict) -> bool:
    training = config.get("training", {}) if config else {}
    model = str(training.get("model_name", "")).lower()
    if model != "tinyresnet":
        return False
    return training.get("model_revision") != "groupnorm_v2"


def _safe_rglob(root: Path, pattern: str) -> list[Path]:
    try:
        return list(root.rglob(pattern))
    except (FileNotFoundError, PermissionError, OSError):
        results: list[Path] = []
        if not root.exists():
            return results
        stack = [root]
        while stack:
            current = stack.pop()
            try:
                for child in current.iterdir():
                    if child.is_dir():
                        stack.append(child)
                    elif child.match(pattern):
                        results.append(child)
            except (FileNotFoundError, PermissionError, OSError):
                continue
        return results


def merge_selected_runs(
    source_ids: list[str],
    name: str = "",
    tail_start_round: int | None = None,
    smooth_curves: bool = False,
) -> dict:
    source_map = {item["id"]: item for item in available_merge_sources(limit=500)}
    selected = sorted(
        [source_map[source_id] for source_id in source_ids if source_id in source_map],
        key=lambda item: _policy_sort_key(str(item["policy"])),
    )
    if not selected:
        return {"ok": False, "message": "No valid method runs selected"}

    safe_name = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in (name.strip() or "merged")).strip("_")
    if not safe_name:
        safe_name = "merged"
    merge_root = ROOT / "out" / "merged_runs"
    merge_root.mkdir(parents=True, exist_ok=True)
    merge_dir = merge_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{safe_name}"
    merge_dir.mkdir(parents=True, exist_ok=False)

    summaries = []
    policies = []
    used_labels: set[str] = set()
    selected_counts: dict[str, int] = {}
    for item in selected:
        selected_counts[str(item["policy"])] = selected_counts.get(str(item["policy"]), 0) + 1
    for item in selected:
        policy = _merged_policy_label(item, used_labels, force_suffix=selected_counts.get(str(item["policy"]), 0) > 1)
        used_labels.add(policy)
        policy_source = Path(item["policy_dir"])
        policy_target = merge_dir / policy
        policy_target.mkdir(parents=True, exist_ok=True)
        for filename in ["summary.json", "round_metrics.csv", "client_decisions.csv", "flow_events.csv", "live_status.json"]:
            source = policy_source / filename
            if source.exists():
                shutil.copy2(source, policy_target / filename)
        summary_path = policy_target / "summary.json"
        if summary_path.exists():
            try:
                with summary_path.open("r", encoding="utf-8") as file:
                    summary = json.load(file)
                summary["policy_original"] = summary.get("policy", item["policy"])
                summary["policy"] = policy
                summary["source_run_dir"] = item["run_dir"]
                summary["source_run_name"] = item["run_name"]
                summaries.append(summary)
            except json.JSONDecodeError:
                pass
        policies.append(policy)

    _write_csv(merge_dir / "summary_table.csv", summaries)
    _write_json(
        merge_dir / "config.json",
        {
            "merged": True,
            "name": safe_name,
            "policies": policies,
            "sources": selected,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
    )
    outputs = build_current_run_figures(
        merge_dir,
        tail_start_round=tail_start_round,
        smooth_curves=smooth_curves,
    )
    return {
        "ok": True,
        "message": f"Merged {len(policies)} methods into {merge_dir}",
        "merge_dir": str(merge_dir),
        "figures": [str(path) for path in outputs],
    }


def _merged_policy_label(item: dict, used_labels: set[str], force_suffix: bool = False) -> str:
    base = str(item["policy"])
    if base not in used_labels and not force_suffix:
        return base
    safe_suffix = _parameter_suffix(item) or _time_suffix(item) or str(len(used_labels) + 1)
    label = f"{base}_{safe_suffix}"
    counter = 2
    while label in used_labels:
        label = f"{base}_{safe_suffix}_{counter}"
        counter += 1
    return label


def _parameter_suffix(item: dict) -> str:
    config = _read_run_config(Path(str(item.get("run_dir", ""))))
    if not config:
        return ""
    selection = config.get("selection", {})
    training = config.get("training", {})
    parts = []
    dataset = str(training.get("dataset_name") or item.get("dataset_name") or "").lower()
    if dataset and dataset not in {"fmnist", "fashion-mnist", "fashionmnist"}:
        parts.append(dataset)
    model = str(training.get("model_name") or item.get("model_name") or "").lower()
    if model:
        parts.append(model)
    device = str(training.get("device") or item.get("device") or "").lower()
    if device and device != "cpu":
        parts.append(device)
    if selection.get("num_clients") is not None:
        parts.append(f"{int(selection['num_clients'])}c")
    if selection.get("num_edges") is not None:
        parts.append(f"{int(selection['num_edges'])}e")
    if selection.get("rounds") is not None:
        parts.append(f"{int(selection['rounds'])}r")
    if selection.get("seed") is not None:
        parts.append(f"s{int(selection['seed'])}")
    if training.get("selection_period") is not None:
        parts.append(f"sp{int(training['selection_period'])}")
    if training.get("learning_rate") is not None:
        parts.append(f"lr{_short_num(training['learning_rate'])}")
    if selection.get("client_heterogeneity") is not None:
        parts.append(f"ch{_short_num(selection['client_heterogeneity'])}")
    if selection.get("edge_heterogeneity") is not None:
        parts.append(f"eh{_short_num(selection['edge_heterogeneity'])}")
    feature_epsilon = selection.get("dp_feature_epsilon_budget")
    update_epsilon = selection.get("dp_update_epsilon_budget")
    if feature_epsilon is not None and update_epsilon is not None:
        if abs(float(feature_epsilon) - float(update_epsilon)) <= 1e-12:
            parts.append(f"eps{_short_num(feature_epsilon)}")
        else:
            parts.append(f"emb{_short_num(feature_epsilon)}")
            parts.append(f"upd{_short_num(update_epsilon)}")
    return "_".join(parts)


def _time_suffix(item: dict) -> str:
    run_name = str(item.get("run_name", ""))
    suffix = run_name.replace("2026-07-13_", "").replace("_lenet5_dynamic", "").replace("-", "")
    return "".join(ch for ch in suffix if ch.isalnum())[-8:]


def _short_num(value: object) -> str:
    number = float(value)
    if number.is_integer():
        return str(int(number))
    return str(number).replace(".", "p")


def _merge_source_id(policy_dir: Path) -> str:
    return str(policy_dir.relative_to(ROOT)).replace("\\", "/")


def available_delete_targets(limit: int = 160) -> list[dict]:
    out_root = ROOT / "out"
    if not out_root.exists():
        return []
    targets: dict[str, dict] = {}

    def add_target(path: Path, kind: str) -> None:
        if not path.exists() or not path.is_dir():
            return
        try:
            resolved = path.resolve()
            resolved.relative_to(out_root.resolve())
        except ValueError:
            return
        if resolved == out_root.resolve():
            return
        rel = str(path.relative_to(ROOT)).replace("\\", "/")
        targets[rel] = {
            "id": rel,
            "name": path.name,
            "kind": kind,
            "path": str(path),
            "mtime": path.stat().st_mtime,
        }

    for path in out_root.iterdir():
        if not path.is_dir():
            continue
        if path.name in {"merged_runs", "paper_experiments"}:
            for child in path.iterdir():
                if child.is_dir():
                    add_target(child, "merged" if path.name == "merged_runs" else "paper_archive")
        else:
            add_target(path, "output_root")

    for summary_table in out_root.rglob("summary_table.csv"):
        run_dir = summary_table.parent
        if "merged_runs" in run_dir.parts or "paper_experiments" in run_dir.parts:
            continue
        add_target(run_dir, "run")

    items = sorted(targets.values(), key=lambda item: item["mtime"], reverse=True)
    return items[:limit]


def delete_selected_dirs(target_ids: list[str]) -> dict:
    global ACTIVE_OUTPUT_ROOT
    if TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None:
        return {"ok": False, "message": "Training is running. Stop it before deleting output directories."}
    if not target_ids:
        return {"ok": False, "message": "No directories selected"}

    out_root = (ROOT / "out").resolve()
    paths: list[Path] = []
    for target_id in target_ids:
        candidate = (ROOT / target_id).resolve()
        try:
            candidate.relative_to(out_root)
        except ValueError:
            continue
        if candidate == out_root or not candidate.exists() or not candidate.is_dir():
            continue
        paths.append(candidate)

    if not paths:
        return {"ok": False, "message": "No valid output directories selected"}

    unique_paths = sorted(set(paths), key=lambda path: len(path.parts), reverse=True)
    deleted = []
    errors = []
    for path in unique_paths:
        if not path.exists():
            continue
        try:
            shutil.rmtree(path)
            deleted.append(str(path))
        except OSError as exc:
            errors.append(f"{path}: {exc}")

    if ACTIVE_OUTPUT_ROOT is not None and not ACTIVE_OUTPUT_ROOT.exists():
        ACTIVE_OUTPUT_ROOT = None

    return {
        "ok": not errors,
        "message": f"Deleted {len(deleted)} directories" + (f"; {len(errors)} failed" if errors else ""),
        "deleted": deleted,
        "errors": errors,
    }


def save_current_run_for_paper(
    run_dir: Path,
    tail_start_round: int | None = None,
    smooth_curves: bool = False,
) -> dict:
    if not run_dir.exists():
        return {"ok": False, "message": "Current run directory does not exist"}
    outputs = build_current_run_figures(
        run_dir,
        tail_start_round=tail_start_round,
        smooth_curves=smooth_curves,
    )
    archive_root = ROOT / "out" / "paper_experiments"
    archive_root.mkdir(parents=True, exist_ok=True)
    archive_dir = archive_root / f"{time.strftime('%Y%m%d_%H%M%S')}_{run_dir.name}"
    archive_dir.mkdir(parents=True, exist_ok=False)
    for name in ["config.json", "summary_table.csv", "live_status.json"]:
        source = run_dir / name
        if source.exists():
            shutil.copy2(source, archive_dir / name)
    figures_dir = archive_dir / "figures"
    figures_dir.mkdir(exist_ok=True)
    for output in outputs:
        shutil.copy2(output, figures_dir / output.name)
    for policy_dir in run_dir.iterdir():
        if policy_dir.is_dir() and (policy_dir / "summary.json").exists():
            target = archive_dir / policy_dir.name
            target.mkdir(exist_ok=True)
            for name in ["summary.json", "round_metrics.csv", "client_decisions.csv"]:
                source = policy_dir / name
                if source.exists():
                    shutil.copy2(source, target / name)
    return {
        "ok": True,
        "message": f"Saved paper experiment: {archive_dir}",
        "archive_dir": str(archive_dir),
        "figures": [str(path) for path in outputs],
    }


def build_current_run_figures(
    run_dir: Path,
    smooth_window: int = 9,
    tail_start_round: int | None = None,
    smooth_curves: bool = False,
) -> list[Path]:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    policies = sorted([
        path.name
        for path in run_dir.iterdir()
        if path.is_dir() and (path / "round_metrics.csv").exists()
    ], key=_policy_sort_key)
    if not policies:
        return []
    figure_dir = run_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    policy_rows = {policy: _read_csv_dicts(run_dir / policy / "round_metrics.csv") for policy in policies}
    max_times = [
        max(float(row["logical_time"]) for row in rows)
        for rows in policy_rows.values()
        if rows
    ]
    common_time_max = min(max_times) if max_times else None
    for filename, x_key, x_label, x_max in [
        ("current_round_accuracy.png", "round", "Round", None),
        ("current_time_accuracy.png", "logical_time", "Logical time", None),
        ("current_time_accuracy_common_axis.png", "logical_time", "Logical time (common axis)", common_time_max),
    ]:
        if x_max is not None and x_max <= 0:
            continue
        fig, ax = plt.subplots(figsize=(7.8, 4.8))
        series = []
        for policy in policies:
            rows = policy_rows[policy]
            if x_key == "round":
                xs = [int(row["round"]) + 1 for row in rows]
            else:
                xs = [float(row["logical_time"]) for row in rows]
            ys = [float(row["test_accuracy"]) for row in rows]
            rounds = [int(row["round"]) + 1 for row in rows]
            if smooth_curves:
                plot_ys, spread = _ema_mean_std(ys, smooth_window)
            else:
                plot_ys = ys
                spread = [0.0 for _ in ys]
            if x_max is not None:
                xs, plot_ys, spread, rounds = _clip_curve_to_x_max(xs, plot_ys, spread, rounds, x_max)
                if len(xs) < 2:
                    continue
            (line,) = ax.plot(xs, plot_ys, linewidth=2.1, label=_short_policy_label(policy))
            color = line.get_color()
            if smooth_curves:
                lower = [max(0.0, mean - std) for mean, std in zip(plot_ys, spread)]
                upper = [min(1.0, mean + std) for mean, std in zip(plot_ys, spread)]
                ax.fill_between(xs, lower, upper, color=color, alpha=0.14, linewidth=0)
            series.append({"policy": policy, "xs": xs, "rounds": rounds, "ys": plot_ys, "color": color})
        tail_x_start = None
        tail_x_end = None
        if x_max is not None and x_key == "logical_time" and tail_start_round is not None:
            starts = []
            for item in series:
                start = next(
                    (
                        item["xs"][index]
                        for index, round_number in enumerate(item["rounds"])
                        if int(round_number) >= tail_start_round
                    ),
                    None,
                )
                if start is not None:
                    starts.append(start)
            if starts:
                tail_x_start = max(starts)
                tail_x_end = x_max
        _add_tail_zoom_inset(
            ax,
            series,
            tail_start_round=tail_start_round,
            tail_x_start=tail_x_start,
            tail_x_end=tail_x_end,
        )
        ax.set_xlabel(x_label)
        ax.set_ylabel(f"Test accuracy (EMA, span={smooth_window})" if smooth_curves else "Test accuracy")
        ax.set_ylim(0.0, 0.9)
        if x_max is not None:
            ax.set_xlim(left=0.0, right=x_max)
        ax.grid(True, alpha=0.25)
        ax.legend(
            loc="upper left",
            ncol=2 if len(policies) > 4 else 1,
            frameon=False,
            fontsize=8,
            handlelength=1.6,
            columnspacing=0.8,
            borderaxespad=0.35,
        )
        fig.tight_layout()
        output = figure_dir / filename
        fig.savefig(output, dpi=220)
        plt.close(fig)
        outputs.append(output)
    return outputs


def _short_policy_label(policy: str) -> str:
    policy_labels = {
        "ours": "Ours",
        "ours_no_omega": "No Error Cost Estimate",
        "ours_fixed_liieiiic": "Fixed Mode LIIEIIIC",
        "individual_optimal": "Individual Optimal",
        "fixed_dp": "Fixed DP",
        "privacy_only": "Privacy Only",
        "no_protection": "No Protection",
        "random": "Random",
        "fixed_fedavg": "Fixed FedAvg",
        "fixed_splitfed": "Fixed SplitFed",
        "fixed_hfl": "Fixed HFL",
        "nsga2": "NSGA II",
        "fixed_liieiiic": "Fixed HFL (LIIEIIIC)",
        "performance_only": "Global Balance Upper",
        "best_accuracy": "Global Utility Upper",
        "accuracy_oracle": "Accuracy Oracle Upper",
    }
    if policy in policy_labels:
        return policy_labels[policy]
    sp = _policy_sp_value(policy)
    if sp is not None:
        return f"SP{sp}"
    if len(policy) <= 14:
        return policy
    return policy[:12] + "..."


def _policy_sp_value(policy: str) -> int | None:
    for part in str(policy).split("_"):
        if part.startswith("sp") and part[2:].isdigit():
            return int(part[2:])
    return None


def _policy_sort_key(policy: str) -> tuple[int, int, str]:
    sp = _policy_sp_value(policy)
    if sp is not None:
        return (0, sp, str(policy))
    return (1, 0, str(policy))


def _interpolate_at_x(xs: list[float], ys: list[float], target_x: float) -> float | None:
    if not xs or not ys or len(xs) != len(ys):
        return None
    if target_x < xs[0] or target_x > xs[-1]:
        return None
    for index in range(1, len(xs)):
        left_x = xs[index - 1]
        right_x = xs[index]
        if left_x <= target_x <= right_x:
            left_y = ys[index - 1]
            right_y = ys[index]
            if right_x == left_x:
                return right_y
            ratio = (target_x - left_x) / (right_x - left_x)
            return left_y + (right_y - left_y) * ratio
    return ys[-1] if target_x == xs[-1] else None


def _clip_curve_to_x_max(
    xs: list[float],
    ys: list[float],
    spread: list[float],
    rounds: list[int],
    x_max: float,
) -> tuple[list[float], list[float], list[float], list[int]]:
    keep = [index for index, x_value in enumerate(xs) if x_value <= x_max]
    clipped_xs = [xs[index] for index in keep]
    clipped_ys = [ys[index] for index in keep]
    clipped_spread = [spread[index] for index in keep]
    clipped_rounds = [rounds[index] for index in keep]
    if xs and xs[0] <= x_max <= xs[-1] and (not clipped_xs or clipped_xs[-1] < x_max):
        y_at_max = _interpolate_at_x(xs, ys, x_max)
        spread_at_max = _interpolate_at_x(xs, spread, x_max)
        round_at_max = _interpolate_at_x(xs, [float(item) for item in rounds], x_max)
        if y_at_max is not None and spread_at_max is not None and round_at_max is not None:
            clipped_xs.append(x_max)
            clipped_ys.append(y_at_max)
            clipped_spread.append(spread_at_max)
            clipped_rounds.append(int(round(round_at_max)))
    return clipped_xs, clipped_ys, clipped_spread, clipped_rounds


def _curve_segment_by_x(
    xs: list[float],
    ys: list[float],
    x_start: float,
    x_end: float,
) -> tuple[list[float], list[float]]:
    if x_end <= x_start:
        return [], []
    y_start = _interpolate_at_x(xs, ys, x_start)
    y_end = _interpolate_at_x(xs, ys, x_end)
    if y_start is None or y_end is None:
        return [], []
    middle = [(x, y) for x, y in zip(xs, ys) if x_start < x < x_end]
    segment_xs = [x_start, *[x for x, _ in middle], x_end]
    segment_ys = [y_start, *[y for _, y in middle], y_end]
    return segment_xs, segment_ys


def _add_tail_zoom_inset(
    ax: Any,
    series: list[dict],
    tail_start_round: int | None = None,
    tail_points: int = 51,
    tail_x_start: float | None = None,
    tail_x_end: float | None = None,
) -> None:
    valid = [
        item for item in series
        if item.get("xs") and item.get("ys") and len(item["xs"]) == len(item["ys"])
    ]
    if not valid:
        return
    if tail_x_start is None or tail_x_end is None:
        all_xs = [float(x_value) for item in valid for x_value in item["xs"]]
        if all_xs:
            x_min = min(all_xs)
            x_max = max(all_xs)
            if tail_start_round is None and x_max > x_min:
                tail_x_start = x_min + 0.75 * (x_max - x_min)
                tail_x_end = x_max
    tail_segments = []
    for item in valid:
        if tail_x_start is not None and tail_x_end is not None:
            xs, ys = _curve_segment_by_x(item["xs"], item["ys"], tail_x_start, tail_x_end)
        else:
            rounds = item.get("rounds") or item["xs"]
            if tail_start_round is not None:
                pairs = [
                    (x, y)
                    for x, y, round_number in zip(item["xs"], item["ys"], rounds)
                    if int(round_number) >= tail_start_round
                ]
                if len(pairs) < 2:
                    pairs = list(zip(item["xs"], item["ys"]))[-max(2, tail_points):]
            else:
                pairs = list(zip(item["xs"], item["ys"]))[-max(2, tail_points):]
            xs = [x for x, _ in pairs]
            ys = [y for _, y in pairs]
        if len(xs) >= 2:
            tail_segments.append({"xs": xs, "ys": ys, "color": item["color"]})
    if not tail_segments:
        return
    x0 = min(segment["xs"][0] for segment in tail_segments)
    max_x = max(segment["xs"][-1] for segment in tail_segments)
    tail_values = [y for segment in tail_segments for y in segment["ys"]]
    if len(tail_values) < 2:
        return
    y_min = max(0.0, min(tail_values) - 0.02)
    y_max = min(1.0, max(tail_values) + 0.02)
    if y_max - y_min < 0.05:
        mid = (y_min + y_max) / 2
        y_min = max(0.0, mid - 0.025)
        y_max = min(1.0, mid + 0.025)

    axins = ax.inset_axes([0.58, 0.08, 0.38, 0.32])
    axins.set_facecolor((1.0, 1.0, 1.0, 0.97))
    for segment in tail_segments:
        axins.plot(segment["xs"], segment["ys"], linewidth=1.9, color=segment["color"])
    axins.set_xlim(x0, max_x)
    axins.set_ylim(y_min, y_max)
    axins.grid(True, alpha=0.24, linewidth=0.55)
    axins.tick_params(labelsize=7.5, pad=1.5, length=2.5)
    for spine in axins.spines.values():
        spine.set_edgecolor("#555555")
        spine.set_linewidth(0.9)


def _ema_mean_std(values: list[float], span: int) -> tuple[list[float], list[float]]:
    if not values:
        return [], []
    span = max(1, int(span))
    alpha = 2.0 / (span + 1.0)
    means = [float(values[0])]
    variances = [0.0]
    for value in values[1:]:
        value = float(value)
        previous_mean = means[-1]
        mean = alpha * value + (1.0 - alpha) * previous_mean
        residual = value - previous_mean
        variance = alpha * (residual ** 2) + (1.0 - alpha) * variances[-1]
        means.append(mean)
        variances.append(variance)
    stds = [variance ** 0.5 for variance in variances]
    return means, stds


def _read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    if not fieldnames:
        return
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2, ensure_ascii=False)


def _tail_start_round_from_payload(payload: dict, default: int | None = None) -> int | None:
    raw_value = payload.get("tail_start_round", default)
    if raw_value in ("", None, "auto"):
        return None
    try:
        return max(1, int(raw_value))
    except (TypeError, ValueError):
        return default


def _smooth_curves_from_payload(payload: dict, default: bool = False) -> bool:
    raw_value = payload.get("smooth_curves", default)
    if isinstance(raw_value, bool):
        return raw_value
    if isinstance(raw_value, str):
        return raw_value.strip().lower() not in {"0", "false", "no", "off", "raw"}
    return bool(raw_value)


def _mode_uses_cloud(mode: str) -> bool:
    return mode in {"LIC", "LIIC", "LIEIIC", "LIEIIIC", "LIIEIIIC"}


def _le_only_round_count(policy_dir: Path) -> int | None:
    decisions_path = policy_dir / "client_decisions.csv"
    if not decisions_path.exists():
        return None
    rounds: dict[str, list[str]] = {}
    for row in _read_csv_dicts(decisions_path):
        rounds.setdefault(str(row.get("round", "")), []).append(str(row.get("mode", "")))
    return sum(1 for modes in rounds.values() if modes and not any(_mode_uses_cloud(mode) for mode in modes))


def _infer_run_dir(status_path: Path, payload: dict) -> Path:
    parent = status_path.parent
    known_policies = set(PRESETS["paper100"]["args"][PRESETS["paper100"]["args"].index("--policies") + 1 :])
    active = payload.get("active_policy") or payload.get("policy")
    if active and (parent / str(active)).is_dir():
        return parent
    if any((parent / policy).is_dir() for policy in known_policies):
        return parent
    return parent.parent


def read_policy_statuses(run_dir: Path, current: dict) -> list[dict]:
    policies = []
    if isinstance(current.get("policies"), list):
        policies = [str(item) for item in current.get("policies", []) if item]
    config_path = run_dir / "config.json"
    if not policies and config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
            policies = list(config.get("policies", []))
        except json.JSONDecodeError:
            policies = []
    if not policies:
        policies = _active_training_policies()
    if not policies:
        policies = ["ours", "ours_no_omega", "ours_fixed_liieiiic", "individual_optimal", "fixed_dp", "privacy_only", "no_protection", "random", "accuracy_oracle"]

    status_rows = []
    active = current.get("active_policy") or current.get("policy")
    for policy in policies:
        policy_dir = run_dir / policy
        live_path = policy_dir / "live_status.json"
        summary_path = policy_dir / "summary.json"
        if live_path.exists():
            try:
                with live_path.open("r", encoding="utf-8") as file:
                    item = json.load(file)
            except (json.JSONDecodeError, PermissionError, OSError):
                item = {"status": "loading"}
            if "summary" in item:
                summary = item["summary"]
                item.setdefault("test_accuracy", summary.get("final_test_accuracy"))
                item.setdefault("best_test_accuracy", summary.get("best_test_accuracy"))
                item.setdefault("logical_time", summary.get("total_logical_time"))
                item.setdefault("cumulative_communication_volume", summary.get("total_communication_volume"))
                item.setdefault(
                    "larger_channel_epsilon",
                    summary.get(
                        "larger_channel_epsilon",
                        summary.get("total_epsilon_used"),
                    ),
                )
                item.setdefault("feature_epsilon", summary.get("max_feature_epsilon"))
                item.setdefault("update_epsilon", summary.get("max_update_epsilon"))
                item.setdefault("mode_distribution", summary.get("mode_distribution"))
                item.setdefault("mean_global_update_clients", summary.get("mean_global_update_clients"))
                item.setdefault("per_client_test_mean", summary.get("per_client_test_accuracy_mean"))
                item.setdefault("per_client_test_min", summary.get("per_client_test_accuracy_min"))
                item.setdefault("per_client_test_max", summary.get("per_client_test_accuracy_max"))
                item.setdefault("per_client_test_std", summary.get("per_client_test_accuracy_std"))
                item.setdefault("per_client_test", summary.get("per_client_test"))
        elif summary_path.exists():
            with summary_path.open("r", encoding="utf-8") as file:
                summary = json.load(file)
            item = {
                "status": "completed",
                "policy": policy,
                "round": summary.get("rounds"),
                "rounds": summary.get("rounds"),
                "progress": 1.0,
                "test_accuracy": summary.get("final_test_accuracy"),
                "best_test_accuracy": summary.get("best_test_accuracy"),
                "logical_time": summary.get("total_logical_time"),
                "cumulative_communication_volume": summary.get("total_communication_volume"),
                "larger_channel_epsilon": summary.get(
                    "larger_channel_epsilon",
                    summary.get("total_epsilon_used"),
                ),
                "feature_epsilon": summary.get("max_feature_epsilon"),
                "update_epsilon": summary.get("max_update_epsilon"),
                "mode_distribution": summary.get("mode_distribution"),
                "mean_global_update_clients": summary.get("mean_global_update_clients"),
                "per_client_test_mean": summary.get("per_client_test_accuracy_mean"),
                "per_client_test_min": summary.get("per_client_test_accuracy_min"),
                "per_client_test_max": summary.get("per_client_test_accuracy_max"),
                "per_client_test_std": summary.get("per_client_test_accuracy_std"),
                "per_client_test": summary.get("per_client_test"),
            }
        else:
            item = {"status": "pending", "policy": policy, "round": 0, "rounds": current.get("rounds")}
        if policy == active and current.get("status") == "running":
            item = {**item, **current, "status": "running", "policy": policy}
        metrics_path = policy_dir / "round_metrics.csv"
        item["le_only_rounds"] = _le_only_round_count(policy_dir)
        if metrics_path.exists():
            metric_rows = _read_csv_dicts(metrics_path)
            global_counts = [
                float(row.get("num_global_update_clients", 0) or 0)
                for row in metric_rows
                if "num_global_update_clients" in row
            ]
            if global_counts:
                item["mean_global_update_clients"] = sum(global_counts) / len(global_counts)
            item["curve"] = [
                {
                    "round": int(row["round"]) + 1,
                    "logical_time": float(row["logical_time"]),
                    "test_accuracy": float(row["test_accuracy"]),
                }
                for row in metric_rows[-500:]
            ]
        elif item.get("recent_rounds"):
            item["curve"] = [
                {
                    "round": int(row["round"]) + 1,
                    "logical_time": float(row["logical_time"]),
                    "test_accuracy": float(row["test_accuracy"]),
                }
                for row in item["recent_rounds"]
            ]
        item["policy"] = policy
        status_rows.append(item)
    return status_rows


def _active_training_policies() -> list[str]:
    if TRAINING_PROCESS is None:
        return []
    args = getattr(TRAINING_PROCESS, "args", [])
    if not isinstance(args, (list, tuple)):
        return []
    values = [str(item) for item in args]
    if "--policies" not in values:
        return []
    start = values.index("--policies") + 1
    policies: list[str] = []
    for value in values[start:]:
        if value.startswith("--"):
            break
        policies.append(value)
    return policies


def html_page() -> bytes:
    return HTML.encode("utf-8")


def active_output_root(default: Path) -> Path:
    return ACTIVE_OUTPUT_ROOT or default


def training_python_executable() -> str:
    configured = os.environ.get(TRAINING_PYTHON_ENV, "").strip()
    if configured:
        return configured
    windows_torch_python = Path("D:/soft/Python310/python.exe")
    if windows_torch_python.exists() and Path(sys.executable) != windows_torch_python:
        return str(windows_torch_python)
    return sys.executable


def find_reusable_run(output_root: Path, preset: dict) -> Path | None:
    requested = _preset_policies(preset)
    if not requested or not output_root.exists():
        return None
    candidates = [
        path for path in output_root.iterdir()
        if path.is_dir() and (path / "summary_table.csv").exists()
    ]
    for run_dir in sorted(candidates, key=lambda path: path.stat().st_mtime, reverse=True):
        if _run_has_completed_policies(run_dir, requested):
            return run_dir
    return None


def _preset_policies(preset: dict) -> list[str]:
    args = list(preset.get("args", []))
    if "--policies" not in args:
        return []
    start = args.index("--policies") + 1
    policies = []
    for item in args[start:]:
        if str(item).startswith("--"):
            break
        policies.append(str(item))
    return policies


def _run_has_completed_policies(run_dir: Path, requested: list[str]) -> bool:
    config_path = run_dir / "config.json"
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
            existing = [str(item) for item in config.get("policies", [])]
            if existing and existing != requested:
                return False
        except (json.JSONDecodeError, OSError):
            return False
    for policy in requested:
        policy_dir = run_dir / policy
        if not (policy_dir / "summary.json").exists():
            return False
        if not (policy_dir / "round_metrics.csv").exists():
            return False
    return True


def _preset_arg_map(preset: dict) -> dict[str, object]:
    args = [str(item) for item in preset.get("args", [])]
    values: dict[str, object] = {}
    flags = {"--iid", "--require-feasible", "--require-cloud", "--require-real-he"}
    index = 0
    while index < len(args):
        item = args[index]
        if not item.startswith("--"):
            index += 1
            continue
        if item == "--policies":
            policies = []
            index += 1
            while index < len(args) and not args[index].startswith("--"):
                policies.append(args[index])
                index += 1
            values[item] = policies
            continue
        if item in flags:
            values[item] = True
            index += 1
            continue
        if index + 1 < len(args):
            values[item] = args[index + 1]
            index += 2
        else:
            index += 1
    return values


def _int_arg(args: dict[str, object], key: str, default: int) -> int:
    try:
        return int(args.get(key, default))
    except (TypeError, ValueError):
        return default


def _optional_int_arg(args: dict[str, object], key: str) -> int | None:
    value = args.get(key)
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float_arg(args: dict[str, object], key: str, default: float) -> float:
    try:
        return float(args.get(key, default))
    except (TypeError, ValueError):
        return default


def _str_arg(args: dict[str, object], key: str, default: str) -> str:
    value = args.get(key, default)
    return str(value)


def _expected_config_from_preset(preset: dict) -> dict:
    args = _preset_arg_map(preset)
    iid = bool(args.get("--iid", False))
    partition_mode = "iid" if iid else _str_arg(args, "--partition-mode", "client_noniid")
    return {
        "selection": {
            "rounds": _int_arg(args, "--rounds", 200),
            "num_clients": _int_arg(args, "--clients", 100),
            "num_edges": _int_arg(args, "--edges", 10),
            "seed": _int_arg(args, "--seed", 42),
            "initial_epsilon": _float_arg(args, "--initial-epsilon", 8.0),
            "dp_event_epsilon": _float_arg(args, "--dp-event-epsilon", 0.05),
            "dp_emb_epsilon": _float_arg(args, "--dp-emb-epsilon", 8.0),
            "dp_upd_epsilon": _float_arg(args, "--dp-upd-epsilon", 8.0),
            "dp_accounting_mode": _str_arg(args, "--dp-accounting-mode", "rdp_auto"),
            "dp_delta": _float_arg(args, "--dp-delta", 1e-5),
            "dp_feature_epsilon_budget": _float_arg(
                args,
                "--dp-feature-epsilon-budget",
                _float_arg(args, "--initial-epsilon", 8.0),
            ),
            "dp_update_epsilon_budget": _float_arg(
                args,
                "--dp-update-epsilon-budget",
                _float_arg(args, "--initial-epsilon", 8.0),
            ),
            "resource_limit": _float_arg(args, "--resource-limit", 1.35),
            "memory_limit": _float_arg(args, "--memory-limit", 1.35),
            "time_limit": _float_arg(args, "--time-limit", 300.0),
            "risk_limit": _float_arg(args, "--risk-limit", 0.5),
            "aggregation_fraction": _float_arg(args, "--aggregation-fraction", 1.0),
            "pareto_archive_size": _int_arg(args, "--pareto-archive-size", 16),
            "pareto_max_iters": _int_arg(args, "--pareto-max-iters", 50),
            "pareto_neighbor_top_k": _int_arg(args, "--pareto-neighbor-top-k", 0),
            "pareto_conflict_only": bool(args.get("--pareto-conflict-only", False)),
            "cloud_fusion_xi": _float_arg(args, "--cloud-fusion-xi", 0.2),
            "cloud_fusion_eps": _float_arg(args, "--cloud-fusion-eps", 0.05),
            "edge_aggregation_beta": _float_arg(args, "--edge-aggregation-beta", 0.01),
            "edge_aggregation_fixed": _float_arg(args, "--edge-aggregation-fixed", 0.02),
            "cloud_aggregation_beta": _float_arg(args, "--cloud-aggregation-beta", 0.015),
            "cloud_aggregation_fixed": _float_arg(args, "--cloud-aggregation-fixed", 0.04),
            "min_edge_cloud_fusion_ratio": _float_arg(
                args, "--min-edge-cloud-fusion-ratio", 0.5
            ),
            "client_heterogeneity": _float_arg(args, "--client-heterogeneity", 2.0),
            "edge_heterogeneity": _float_arg(args, "--edge-heterogeneity", 1.5),
            "require_feasible": bool(args.get("--require-feasible", False)),
            "require_cloud_participation": bool(args.get("--require-cloud", False)),
            "require_edge_cloud_coverage": bool(
                args.get("--require-edge-cloud-coverage", False)
            ),
            "enforce_cloud_dp_stability": bool(
                args.get("--enforce-cloud-dp-stability", False)
            ),
            "cloud_dp_stability_threshold": _float_arg(
                args, "--cloud-dp-stability-threshold", 1.0
            ),
            "trusted_edge_split_execution": bool(
                args.get("--trusted-edge-split-execution", False)
            ),
        },
        "training": {
            "dataset_name": _str_arg(args, "--dataset", "cifar10"),
            "model_name": _str_arg(args, "--model", "resnet18_pretrained"),
            "model_revision": "groupnorm_v2",
            "update_parameter_scope": CURRENT_UPDATE_PARAMETER_SCOPE,
            "execution_revision": _str_arg(
                args, "--execution-revision", CURRENT_EXECUTION_REVISION
            ),
            "local_epochs": _int_arg(args, "--local-epochs", 3),
            "learning_rate": _float_arg(args, "--lr", 0.01),
            "iid": iid,
            "partition_mode": partition_mode,
            "selection_period": _int_arg(args, "--selection-period", 1),
            "dp_clip_norm": _float_arg(args, "--dp-clip-norm", 1.0),
            "dp_noise_multiplier": _float_arg(args, "--dp-noise-multiplier", 0.0002),
            "dp_update_mode": _str_arg(args, "--dp-update-mode", "upd_only"),
            "executor": _str_arg(args, "--executor", "serial"),
            "executor_workers": _optional_int_arg(args, "--executor-workers"),
            "device": _str_arg(args, "--device", "cuda"),
            "he_backend": _str_arg(args, "--he-backend", "seal"),
            "he_execution": _str_arg(args, "--he-execution", "profiled"),
            "he_local_deps": _str_arg(args, "--he-local-deps", ".he_deps"),
            "require_real_he": bool(args.get("--require-real-he", False)),
        },
        "train_limit": _int_arg(args, "--train-limit", 12000),
        "test_limit": _int_arg(args, "--test-limit", 2000),
        "policies": [str(item) for item in args.get("--policies", [])],
    }


def _config_value_matches(actual: object, expected: object) -> bool:
    if expected is None:
        return actual is None or str(actual).strip() == ""
    if isinstance(expected, bool):
        return bool(actual) == expected
    if isinstance(expected, int) and not isinstance(expected, bool):
        try:
            return int(actual) == expected
        except (TypeError, ValueError):
            return False
    if isinstance(expected, float):
        try:
            return abs(float(actual) - expected) < 1e-9
        except (TypeError, ValueError):
            return False
    return str(actual) == str(expected)


def _config_matches_resume_target(config: dict, expected: dict) -> bool:
    for group in ["selection", "training"]:
        actual_group = config.get(group, {})
        for key, expected_value in expected[group].items():
            if key in {"rounds"}:
                continue
            if not _config_value_matches(actual_group.get(key), expected_value):
                return False
    for key in ["train_limit", "test_limit"]:
        if not _config_value_matches(config.get(key), expected[key]):
            return False
    return True


def _policy_checkpoint_round(policy_dir: Path) -> int | None:
    if not (policy_dir / "checkpoint.pt").exists():
        return None
    metrics_path = policy_dir / "round_metrics.csv"
    if not metrics_path.exists():
        return None
    rows = _read_csv_dicts(metrics_path)
    if not rows:
        return None
    try:
        return max(int(row["round"]) for row in rows if str(row.get("round", "")).strip() != "") + 1
    except (KeyError, TypeError, ValueError):
        return None


def _resume_round_for_policies(run_dir: Path, requested: list[str]) -> int | None:
    rounds = []
    for policy in requested:
        policy_round = _policy_checkpoint_round(run_dir / policy)
        if policy_round is None:
            return None
        rounds.append(policy_round)
    return min(rounds) if rounds else None


def _resume_he_runtime_is_compatible(
    run_dir: Path,
    requested: list[str],
    expected: dict,
) -> bool:
    training = expected.get("training", {})
    if training.get("he_backend") != "seal" or not training.get("require_real_he"):
        return True
    validation_marker = "SEAL CKKS encrypted-vector validation passed"
    for policy in requested:
        summary_path = run_dir / policy / "summary.json"
        try:
            with summary_path.open("r", encoding="utf-8") as file:
                summary = json.load(file)
        except (json.JSONDecodeError, OSError):
            return False
        if validation_marker not in str(summary.get("he_status", "")):
            return False
    return True


def _add_resume_arg(preset: dict, resume_from_run: Path) -> dict:
    updated = {**preset, "args": list(preset.get("args", []))}
    args = updated["args"]
    if "--resume-from-run" in args:
        index = args.index("--resume-from-run")
        if index + 1 < len(args):
            args[index + 1] = str(resume_from_run)
        else:
            args.append(str(resume_from_run))
    else:
        args.extend(["--resume-from-run", str(resume_from_run)])
    updated["auto_resume_from_run"] = str(resume_from_run)
    return updated


def _tail_text(path: Path, max_chars: int = 4000) -> str:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return ""
    return text[-max_chars:].strip()


def find_auto_resume_run(preset: dict) -> dict | None:
    requested = _preset_policies(preset)
    if not requested:
        return None
    expected = _expected_config_from_preset(preset)
    target_rounds = int(expected["selection"]["rounds"])
    out_root = ROOT / "out"
    if not out_root.exists():
        return None

    best: dict | None = None
    for config_path in out_root.rglob("config.json"):
        if "merged_runs" in config_path.parts or "paper_experiments" in config_path.parts:
            continue
        run_dir = config_path.parent
        try:
            with config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
        except (json.JSONDecodeError, OSError):
            continue
        if not _config_matches_resume_target(config, expected):
            continue
        if not _resume_he_runtime_is_compatible(run_dir, requested, expected):
            continue
        source_rounds = _resume_round_for_policies(run_dir, requested)
        if source_rounds is None or source_rounds <= 0 or source_rounds >= target_rounds:
            continue
        candidate = {
            "run_dir": run_dir,
            "source_rounds": source_rounds,
            "target_rounds": target_rounds,
            "mtime": config_path.stat().st_mtime,
        }
        if best is None:
            best = candidate
            continue
        if source_rounds > best["source_rounds"] or (
            source_rounds == best["source_rounds"] and candidate["mtime"] > best["mtime"]
        ):
            best = candidate
    return best


@_serialized_training_state
def start_training(
    preset_name: str,
    default_output_root: Path,
    rounds: int | None = None,
    mode: str = "rounds",
    time_limit: float = 300.0,
    train_limit: int = 12000,
    test_limit: int = 2000,
    policies: list[str] | None = None,
    figure_axis: str = "round",
    partition_mode: str = "extreme_edge_label_skew",
    clients: int = 100,
    edges: int = 10,
    seed: int = 42,
    client_heterogeneity: float = 2.0,
    edge_heterogeneity: float = 1.5,
    selection_period: int = 1,
    aggregation_fraction: float = 1.0,
    pareto_archive_size: int = 16,
    pareto_max_iters: int = 50,
    pareto_neighbor_top_k: int = 0,
    pareto_conflict_only: bool = False,
    cloud_fusion_xi: float = 0.2,
    cloud_fusion_eps: float = 0.05,
    min_edge_cloud_fusion_ratio: float = 0.5,
    resource_limit: float = 1.35,
    risk_limit: float = 0.5,
    executor: str = "serial",
    executor_workers: int | None = None,
    local_epochs: int | None = None,
    learning_rate: float | None = None,
    initial_epsilon: float = 8.0,
    dp_emb_epsilon: float = 8.0,
    dp_upd_epsilon: float = 8.0,
    he_backend: str = "seal",
    require_real_he: bool = True,
    he_aggregation_size: int = 0,
    dp_profile: str = "cifar_resnet",
    dataset: str = "cifar10",
    model: str = "resnet18_pretrained",
    device: str = "cuda",
    resume_from_run: str = "",
    reuse_completed: bool = False,
) -> dict:
    global ACTIVE_OUTPUT_ROOT, LAST_STOP_MESSAGE, TRAINING_PROCESS
    if preset_name == "paper50":
        preset_name = "paper100"
    if preset_name in {"configured", "custom"}:
        preset = configured_preset(
            mode,
            rounds or 200,
            time_limit,
            train_limit,
            test_limit,
            policies or [],
            figure_axis,
            partition_mode,
            clients,
            edges,
            seed,
            client_heterogeneity,
            edge_heterogeneity,
            selection_period,
            aggregation_fraction,
            pareto_archive_size,
            pareto_max_iters,
            pareto_neighbor_top_k,
            pareto_conflict_only,
            cloud_fusion_xi,
            cloud_fusion_eps,
            min_edge_cloud_fusion_ratio,
            resource_limit,
            risk_limit,
            executor,
            executor_workers,
            local_epochs,
            learning_rate,
            initial_epsilon,
            dp_emb_epsilon,
            dp_upd_epsilon,
            he_backend,
            require_real_he,
            he_aggregation_size,
            dp_profile,
            dataset,
            model,
            device,
            resume_from_run,
        )
        if preset_name == "custom":
            preset["label"] = f"{rounds or 200}r custom paper set"
    else:
        preset = PRESETS.get(preset_name)
    if preset is None:
        return {"ok": False, "message": f"Unknown preset: {preset_name}"}

    output_root = ROOT / preset["output_root"]
    current = read_status(active_output_root(default_output_root))
    process_running = TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None
    if current.get("status") == "running" and process_running:
        return {
            "ok": False,
            "message": f"Training is already running: {current.get('policy')} round {current.get('round')}/{current.get('rounds')}",
        }
    if process_running:
        return {"ok": False, "message": f"Training process is already running, PID {TRAINING_PROCESS.pid}"}

    LAST_STOP_MESSAGE = None

    if reuse_completed:
        reusable_run = find_reusable_run(output_root, preset)
        if reusable_run is not None:
            ACTIVE_OUTPUT_ROOT = output_root
            outputs = build_current_run_figures(reusable_run)
            return {
                "ok": True,
                "reused": True,
                "preset": preset_name,
                "message": f"Reused completed experiment: {reusable_run}",
                "output_root": str(output_root),
                "run_dir": str(reusable_run),
                "figures": [str(path) for path in outputs],
            }

    auto_resume = None
    if not resume_from_run:
        auto_resume = find_auto_resume_run(preset)
        if auto_resume is not None:
            preset = _add_resume_arg(preset, auto_resume["run_dir"])

    log_dir = ROOT / "out"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / f"{preset_name}_train.log"
    stderr_path = log_dir / f"{preset_name}_train.err.log"
    stdout = stdout_path.open("a", encoding="utf-8")
    stderr = stderr_path.open("a", encoding="utf-8")
    stdout.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] start {preset['label']}\n")
    stdout.flush()

    ACTIVE_OUTPUT_ROOT = output_root
    training_python = training_python_executable()
    env = os.environ.copy()
    env.setdefault("PYTHONIOENCODING", "utf-8")
    TRAINING_PROCESS = subprocess.Popen(
        [training_python, "-u", *preset["args"]],
        cwd=ROOT,
        stdout=stdout,
        stderr=stderr,
        env=env,
        creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
    )
    time.sleep(0.5)
    if TRAINING_PROCESS.poll() is not None:
        return_code = TRAINING_PROCESS.returncode
        TRAINING_PROCESS = None
        err_tail = _tail_text(stderr_path)
        message = f"Training failed to start, exit code {return_code}"
        if err_tail:
            message += f": {err_tail.splitlines()[-1]}"
        return {
            "ok": False,
            "preset": preset_name,
            "message": message,
            "output_root": str(output_root),
            "training_python": training_python,
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "stderr_tail": err_tail,
        }
    return {
        "ok": True,
        "pid": TRAINING_PROCESS.pid,
        "preset": preset_name,
        "message": (
            f"Started {preset['label']}, PID {TRAINING_PROCESS.pid}; "
            f"auto-resume from {auto_resume['source_rounds']} to {auto_resume['target_rounds']} rounds: {auto_resume['run_dir']}"
            if auto_resume is not None
            else f"Started {preset['label']}, PID {TRAINING_PROCESS.pid}"
        ),
        "output_root": str(output_root),
        "auto_resume": auto_resume is not None,
        "resume_from_run": str(auto_resume["run_dir"]) if auto_resume is not None else resume_from_run,
        "training_python": training_python,
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def rebuild_figures() -> dict:
    log_dir = ROOT / "out"
    log_dir.mkdir(parents=True, exist_ok=True)
    stdout_path = log_dir / "rebuild_paper_figures.log"
    stderr_path = log_dir / "rebuild_paper_figures.err.log"
    with stdout_path.open("a", encoding="utf-8") as stdout, stderr_path.open("a", encoding="utf-8") as stderr:
        stdout.write(f"\n\n[{time.strftime('%Y-%m-%d %H:%M:%S')}] rebuild paper figures\n")
        stdout.flush()
        completed = subprocess.run(
            [sys.executable, "experiments/rebuild_paper_figures.py", "--fallback"],
            cwd=ROOT,
            stdout=stdout,
            stderr=stderr,
            text=True,
        )
    return {
        "ok": completed.returncode == 0,
        "message": "Rebuilt paper figures" if completed.returncode == 0 else f"Figure rebuild failed: {completed.returncode}",
        "stdout": str(stdout_path),
        "stderr": str(stderr_path),
    }


def _stop_result(message: str, cleanup: dict, stopped: list[str]) -> dict:
    global LAST_STOP_MESSAGE
    LAST_STOP_MESSAGE = message
    return {"ok": True, "message": message, "cleanup": cleanup, "stopped_pids": stopped}


@_serialized_training_state
def stop_training() -> dict:
    global TRAINING_PROCESS, ACTIVE_OUTPUT_ROOT
    cleanup_root = ACTIVE_OUTPUT_ROOT
    stopped: list[str] = []
    if TRAINING_PROCESS is None:
        stopped = stop_all_training_processes()
        cleanup = cleanup_incomplete_active_run(cleanup_root)
        ACTIVE_OUTPUT_ROOT = None if cleanup.get("deleted") else ACTIVE_OUTPUT_ROOT
        suffix = f"; {cleanup['message']}" if cleanup.get("message") else ""
        proc_msg = f"Stopped residual training processes: {', '.join(stopped)}" if stopped else "No tracked training process is running"
        return _stop_result(proc_msg + suffix, cleanup, stopped)
    pid = TRAINING_PROCESS.pid
    if TRAINING_PROCESS.poll() is not None:
        TRAINING_PROCESS = None
        stopped = stop_all_training_processes()
        cleanup = cleanup_incomplete_active_run(cleanup_root)
        ACTIVE_OUTPUT_ROOT = None if cleanup.get("deleted") else ACTIVE_OUTPUT_ROOT
        suffix = f"; {cleanup['message']}" if cleanup.get("message") else ""
        extra = f"; stopped residual training processes: {', '.join(stopped)}" if stopped else ""
        return _stop_result(f"Training process {pid} already stopped" + extra + suffix, cleanup, stopped)
    if os.name == "nt":
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
        )
        try:
            TRAINING_PROCESS.wait(timeout=5)
        except subprocess.TimeoutExpired:
            TRAINING_PROCESS.kill()
            TRAINING_PROCESS.wait(timeout=5)
        message = f"Stopped training process tree {pid}"
    else:
        TRAINING_PROCESS.terminate()
        try:
            TRAINING_PROCESS.wait(timeout=5)
            message = f"Stopped training process {pid}"
        except subprocess.TimeoutExpired:
            TRAINING_PROCESS.kill()
            TRAINING_PROCESS.wait(timeout=5)
            message = f"Killed training process {pid}"
    TRAINING_PROCESS = None
    stopped = stop_all_training_processes()
    cleanup = cleanup_incomplete_active_run(cleanup_root)
    ACTIVE_OUTPUT_ROOT = None if cleanup.get("deleted") else ACTIVE_OUTPUT_ROOT
    suffix = f"; {cleanup['message']}" if cleanup.get("message") else ""
    extra = f"; stopped residual training processes: {', '.join(stopped)}" if stopped else ""
    return _stop_result(message + extra + suffix, cleanup, stopped)


def stop_all_training_processes(exclude_pids: set[int] | None = None) -> list[str]:
    exclude_pids = exclude_pids or set()
    stopped: list[str] = []
    for attempt in range(6):
        try:
            completed = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-Command",
                    (
                        "Get-CimInstance Win32_Process | "
                        "Where-Object { $_.Name -like 'python*.exe' } | "
                        "ForEach-Object { \"$($_.ProcessId)`t$($_.CommandLine)\" }"
                    ),
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
                timeout=10,
                creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
            )
        except (OSError, subprocess.TimeoutExpired):
            return stopped
        pids = []
        current_pid = os.getpid()
        for line in completed.stdout.splitlines():
            if "\t" not in line:
                continue
            pid_text, command_line = line.split("\t", 1)
            pid_text = pid_text.strip()
            if not pid_text.isdigit():
                continue
            proc_id = int(pid_text)
            normalized = command_line.replace("\\", "/").lower()
            if proc_id == current_pid or proc_id in exclude_pids:
                continue
            if "experiments/run_fmnist_lenet5.py" in normalized:
                pids.append(proc_id)
        if not pids:
            break
        for proc_id in sorted(set(pids)):
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {proc_id} -Force -ErrorAction SilentlyContinue"],
                    cwd=ROOT,
                    capture_output=True,
                    text=True,
                    timeout=10,
                    creationflags=subprocess.CREATE_NO_WINDOW if hasattr(subprocess, "CREATE_NO_WINDOW") else 0,
                )
                pid_text = str(proc_id)
                if pid_text not in stopped:
                    stopped.append(pid_text)
            except (OSError, subprocess.TimeoutExpired):
                continue
        if attempt < 5:
            time.sleep(0.5)
    return stopped


def cleanup_incomplete_active_run(output_root: Path | None) -> dict:
    if output_root is None:
        return {"deleted": [], "message": ""}
    try:
        root = output_root.resolve()
        root.relative_to((ROOT / "out").resolve())
    except ValueError:
        return {"deleted": [], "message": "Skipped cleanup outside out/"}
    status_path = latest_status_path(root)
    if status_path is None:
        return {"deleted": [], "message": ""}
    payload = {}
    try:
        with status_path.open("r", encoding="utf-8") as file:
            payload = json.load(file)
    except (json.JSONDecodeError, OSError):
        payload = {}
    run_dir = _infer_run_dir(status_path, payload).resolve()
    try:
        run_dir.relative_to((ROOT / "out").resolve())
    except ValueError:
        return {"deleted": [], "message": "Skipped cleanup outside out/"}
    if _is_completed_run(run_dir):
        return {"deleted": [], "message": "Current run is complete; kept existing results"}
    deleted = []
    try:
        if run_dir.exists() and run_dir != root:
            shutil.rmtree(run_dir)
            deleted.append(str(run_dir))
        parent_live = root / "live_status.json"
        if parent_live.exists():
            parent_live.unlink()
            deleted.append(str(parent_live))
    except OSError as exc:
        return {"deleted": deleted, "message": f"Cleanup failed: {exc}"}
    return {"deleted": deleted, "message": f"Deleted incomplete run artifacts ({len(deleted)} items)" if deleted else ""}


def _is_completed_run(run_dir: Path) -> bool:
    summary_table = run_dir / "summary_table.csv"
    if not summary_table.exists():
        return False
    config_path = run_dir / "config.json"
    policies = []
    if config_path.exists():
        try:
            with config_path.open("r", encoding="utf-8") as file:
                config = json.load(file)
            policies = [str(item) for item in config.get("policies", [])]
        except (json.JSONDecodeError, OSError):
            policies = []
    if not policies:
        policies = [
            path.name for path in run_dir.iterdir()
            if path.is_dir() and (path / "summary.json").exists()
        ]
    return bool(policies) and all(
        (run_dir / policy / "summary.json").exists()
        and (run_dir / policy / "round_metrics.csv").exists()
        for policy in policies
    )


class MonitorHandler(BaseHTTPRequestHandler):
    output_root: Path

    def do_GET(self) -> None:
        route = urlparse(self.path).path
        if route == "/":
            self._send(200, "text/html; charset=utf-8", html_page())
            return
        if route == "/status":
            status = read_status(active_output_root(self.output_root))
            process_running = bool(TRAINING_PROCESS is not None and TRAINING_PROCESS.poll() is None)
            if process_running and status.get("status") not in {"waiting", "loading", "running"}:
                status = {
                    "status": "loading",
                    "message": f"Training process PID {TRAINING_PROCESS.pid} started; waiting for first status update",
                    "progress": 0.0,
                    "output_root": str(active_output_root(self.output_root)),
                }
            elif LAST_STOP_MESSAGE and not process_running and status.get("status") in {"waiting", "loading", "running"}:
                status["status"] = "stopped"
                status["message"] = LAST_STOP_MESSAGE
            elif not process_running and status.get("status") == "running":
                return_code = TRAINING_PROCESS.returncode if TRAINING_PROCESS is not None else None
                status["status"] = "failed"
                status["message"] = (
                    f"Training process exited with code {return_code}. Check out/configured_train.err.log."
                    if return_code is not None
                    else "Training status is stale and no training process is running. Check out/configured_train.err.log."
                )
            status["training_process_running"] = process_running
            status["training_pid"] = TRAINING_PROCESS.pid if process_running else None
            status["available_presets"] = {key: {"label": value["label"], "output_root": value["output_root"]} for key, value in PRESETS.items()}
            payload = _json_response(status)
            self._send(200, "application/json; charset=utf-8", payload)
            return
        if route.startswith("/figures/"):
            name = Path(route.removeprefix("/figures/")).name
            figure_path = ROOT / "Privacy_Utility_Tradeoff__20260525" / "figures" / name
            if figure_path.exists():
                self._send(200, mimetypes.guess_type(figure_path.name)[0] or "application/octet-stream", figure_path.read_bytes())
                return
        if route.startswith("/run-figures/"):
            status = read_status(active_output_root(self.output_root))
            run_dir = Path(status.get("run_dir", ""))
            name = Path(route.removeprefix("/run-figures/")).name
            figure_path = run_dir / "figures" / name
            if figure_path.exists():
                self._send(200, mimetypes.guess_type(figure_path.name)[0] or "application/octet-stream", figure_path.read_bytes())
                return
        if route.startswith("/paper-experiments/"):
            relative = Path(route.removeprefix("/paper-experiments/"))
            if ".." not in relative.parts:
                file_path = ROOT / "out" / "paper_experiments" / relative
                if file_path.exists() and file_path.is_file():
                    self._send(200, mimetypes.guess_type(file_path.name)[0] or "application/octet-stream", file_path.read_bytes())
                    return
        if route.startswith("/merged-runs/"):
            relative = Path(route.removeprefix("/merged-runs/"))
            if ".." not in relative.parts:
                file_path = ROOT / "out" / "merged_runs" / relative
                if file_path.exists() and file_path.is_file():
                    self._send(200, mimetypes.guess_type(file_path.name)[0] or "application/octet-stream", file_path.read_bytes())
                    return
        if route == "/legacy":
            legacy = ROOT / "plot" / "privacy_mode_selection_greedy.gif"
            if legacy.exists():
                self._send(200, mimetypes.guess_type(legacy.name)[0] or "image/gif", legacy.read_bytes())
                return
        self._send(404, "text/plain; charset=utf-8", b"Not found")

    def do_POST(self) -> None:
        parsed = urlparse(self.path)
        route = parsed.path
        if route == "/start":
            query = parse_qs(parsed.query)
            preset = query.get("preset", ["configured"])[0]
            if (
                preset in {"configured", "custom"}
                and query.get("privacy_schema", [""])[0] != PRIVACY_SCHEMA_VERSION
            ):
                payload = json.dumps(
                    {
                        "ok": False,
                        "message": (
                            "The monitor page is outdated. Refresh the page before "
                            "starting; DP fields now represent total RDP epsilon targets."
                        ),
                    },
                    ensure_ascii=False,
                ).encode("utf-8")
                self._send(409, "application/json; charset=utf-8", payload)
                return
            try:
                rounds = int(query.get("rounds", ["200"])[0])
            except ValueError:
                rounds = 200
            try:
                time_limit = float(query.get("time_limit", ["300.0"])[0])
            except ValueError:
                time_limit = 300.0
            try:
                train_limit = int(query.get("train_limit", ["12000"])[0])
            except ValueError:
                train_limit = 12000
            try:
                test_limit = int(query.get("test_limit", ["2000"])[0])
            except ValueError:
                test_limit = 2000
            policies = [
                item
                for value in query.get(
                    "policies",
                    ["ours,individual_optimal,no_protection,random"],
                )
                for item in value.split(",")
                if item
            ]
            mode = query.get("mode", ["rounds"])[0]
            figure_axis = query.get("figure_axis", ["round"])[0]
            partition_mode = query.get("partition_mode", ["extreme_edge_label_skew"])[0]
            try:
                clients = int(query.get("clients", ["100"])[0])
            except ValueError:
                clients = 100
            try:
                edges = int(query.get("edges", ["10"])[0])
            except ValueError:
                edges = 10
            try:
                seed = int(query.get("seed", ["42"])[0])
            except ValueError:
                seed = 42
            try:
                client_heterogeneity = float(query.get("client_heterogeneity", ["2.0"])[0])
            except ValueError:
                client_heterogeneity = 2.0
            try:
                edge_heterogeneity = float(query.get("edge_heterogeneity", ["1.5"])[0])
            except ValueError:
                edge_heterogeneity = 1.5
            try:
                selection_period = int(query.get("selection_period", ["1"])[0])
            except ValueError:
                selection_period = 1
            try:
                aggregation_fraction = float(query.get("aggregation_fraction", ["1.0"])[0])
            except ValueError:
                aggregation_fraction = 1.0
            try:
                pareto_archive_size = int(query.get("pareto_archive_size", ["16"])[0])
            except ValueError:
                pareto_archive_size = 16
            try:
                pareto_max_iters = int(query.get("pareto_max_iters", ["50"])[0])
            except ValueError:
                pareto_max_iters = 50
            try:
                pareto_neighbor_top_k = int(query.get("pareto_neighbor_top_k", ["0"])[0])
            except ValueError:
                pareto_neighbor_top_k = 0
            pareto_conflict_only = query.get("pareto_conflict_only", ["0"])[0] not in {
                "0",
                "false",
                "False",
            }
            try:
                cloud_fusion_xi = float(query.get("cloud_fusion_xi", ["0.2"])[0])
            except ValueError:
                cloud_fusion_xi = 0.2
            try:
                cloud_fusion_eps = float(query.get("cloud_fusion_eps", ["0.05"])[0])
            except ValueError:
                cloud_fusion_eps = 0.05
            try:
                min_edge_cloud_fusion_ratio = float(
                    query.get("min_edge_cloud_fusion_ratio", ["0.5"])[0]
                )
            except ValueError:
                min_edge_cloud_fusion_ratio = 0.5
            try:
                resource_limit = float(query.get("resource_limit", ["1.35"])[0])
            except ValueError:
                resource_limit = 1.35
            try:
                risk_limit = float(query.get("risk_limit", ["0.5"])[0])
            except ValueError:
                risk_limit = 0.5
            executor = query.get("executor", ["serial"])[0]
            executor_workers = None
            executor_workers_text = query.get("executor_workers", [""])[0].strip()
            if executor_workers_text:
                try:
                    executor_workers = int(executor_workers_text)
                except ValueError:
                    executor_workers = None
            local_epochs = None
            local_epochs_text = query.get("local_epochs", [""])[0].strip()
            if local_epochs_text:
                try:
                    local_epochs = int(local_epochs_text)
                except ValueError:
                    local_epochs = None
            learning_rate = None
            learning_rate_text = query.get("learning_rate", [""])[0].strip()
            if learning_rate_text:
                try:
                    learning_rate = float(learning_rate_text)
                except ValueError:
                    learning_rate = None
            try:
                initial_epsilon = float(query.get("initial_epsilon", ["8.0"])[0])
            except ValueError:
                initial_epsilon = 8.0
            try:
                dp_emb_epsilon = float(query.get("dp_emb_epsilon", ["8.0"])[0])
            except ValueError:
                dp_emb_epsilon = 8.0
            try:
                dp_upd_epsilon = float(query.get("dp_upd_epsilon", ["8.0"])[0])
            except ValueError:
                dp_upd_epsilon = 8.0
            he_backend = query.get("he_backend", ["seal"])[0]
            require_real_he = query.get("require_real_he", ["true"])[0].lower() == "true"
            try:
                he_aggregation_size = int(query.get("he_aggregation_size", ["0"])[0])
            except ValueError:
                he_aggregation_size = 0
            dp_profile = query.get("dp_profile", ["cifar_resnet"])[0]
            dataset = query.get("dataset", ["cifar10"])[0]
            model = query.get("model", ["resnet18_pretrained"])[0]
            device = query.get("device", ["cuda"])[0]
            resume_from_run = query.get("resume_from_run", [""])[0].strip()
            reuse_completed = query.get("reuse_completed", ["false"])[0].lower() == "true"
            payload = json.dumps(
                start_training(
                    preset,
                    self.output_root,
                    rounds,
                    mode,
                    time_limit,
                    train_limit,
                    test_limit,
                    policies,
                    figure_axis,
                    partition_mode,
                    clients,
                    edges,
                    seed,
                    client_heterogeneity,
                    edge_heterogeneity,
                    selection_period,
                    aggregation_fraction,
                    pareto_archive_size,
                    pareto_max_iters,
                    pareto_neighbor_top_k,
                    pareto_conflict_only,
                    cloud_fusion_xi,
                    cloud_fusion_eps,
                    min_edge_cloud_fusion_ratio,
                    resource_limit,
                    risk_limit,
                    executor,
                    executor_workers,
                    local_epochs,
                    learning_rate,
                    initial_epsilon,
                    dp_emb_epsilon,
                    dp_upd_epsilon,
                    he_backend,
                    require_real_he,
                    he_aggregation_size,
                    dp_profile,
                    dataset,
                    model,
                    device,
                    resume_from_run,
                    reuse_completed,
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", payload)
            return
        if route == "/stop":
            payload = json.dumps(stop_training(), ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", payload)
            return
        if route == "/rebuild-figures":
            payload = json.dumps(rebuild_figures(), ensure_ascii=False).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", payload)
            return
        if route == "/build-run-figures":
            body = self._read_json_body()
            status = read_status(active_output_root(self.output_root))
            run_dir = Path(status.get("run_dir", ""))
            outputs = build_current_run_figures(
                run_dir,
                tail_start_round=_tail_start_round_from_payload(body),
                smooth_curves=_smooth_curves_from_payload(body),
            ) if run_dir.exists() else []
            payload = json.dumps(
                {"ok": bool(outputs), "message": f"Built {len(outputs)} current run figures", "outputs": [str(path) for path in outputs]},
                ensure_ascii=False,
            ).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", payload)
            return
        if route == "/save-paper-run":
            body = self._read_json_body()
            status = read_status(active_output_root(self.output_root))
            run_dir = Path(status.get("run_dir", ""))
            payload = json.dumps(
                save_current_run_for_paper(
                    run_dir,
                    tail_start_round=_tail_start_round_from_payload(body),
                    smooth_curves=_smooth_curves_from_payload(body),
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", payload)
            return
        if route == "/merge-runs":
            body = self._read_json_body()
            payload = json.dumps(
                merge_selected_runs(
                    [str(item) for item in body.get("source_ids", [])],
                    str(body.get("name", "")),
                    _tail_start_round_from_payload(body),
                    _smooth_curves_from_payload(body),
                ),
                ensure_ascii=False,
            ).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", payload)
            return
        if route == "/delete-dirs":
            body = self._read_json_body()
            payload = json.dumps(
                delete_selected_dirs([str(item) for item in body.get("target_ids", [])]),
                ensure_ascii=False,
            ).encode("utf-8")
            self._send(200, "application/json; charset=utf-8", payload)
            return
        self._send(404, "text/plain; charset=utf-8", b"Not found")

    def log_message(self, fmt: str, *args: object) -> None:
        return

    def _send(self, code: int, content_type: str, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Cache-Control", "no-store")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_json_body(self) -> dict:
        try:
            length = int(self.headers.get("Content-Length", "0") or "0")
        except ValueError:
            length = 0
        if length <= 0:
            return {}
        try:
            return json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>DynFedPrivacy LeNet5 Monitor</title>
<style>
:root {
  color-scheme: dark;
  --bg: #101214;
  --panel: #191d20;
  --panel-2: #20262a;
  --text: #eef2f3;
  --muted: #aab4b8;
  --line: #2e383d;
  --green: #4fd08a;
  --blue: #5ca8ff;
  --amber: #f3c65f;
  --red: #ee6f70;
}
* { box-sizing: border-box; }
body {
  margin: 0;
  font-family: "Segoe UI", Arial, sans-serif;
  background: var(--bg);
  color: var(--text);
}
main { width: min(1320px, calc(100vw - 32px)); margin: 0 auto; padding: 22px 0 32px; }
header { display: flex; justify-content: space-between; gap: 18px; align-items: flex-end; margin-bottom: 18px; }
h1 { margin: 0; font-size: 26px; font-weight: 650; letter-spacing: 0; }
.sub { color: var(--muted); font-size: 13px; margin-top: 6px; overflow-wrap: anywhere; }
.pill { padding: 8px 11px; border: 1px solid var(--line); border-radius: 6px; background: var(--panel); color: var(--muted); font-size: 13px; }
.actions { display: flex; gap: 10px; align-items: center; flex-wrap: wrap; justify-content: flex-end; }
button { border: 1px solid var(--green); border-radius: 6px; background: rgba(79, 208, 138, .12); color: var(--text); padding: 9px 13px; font: inherit; cursor: pointer; }
button:hover { background: rgba(79, 208, 138, .2); }
button:disabled { cursor: not-allowed; opacity: .55; }
.control-panel { display: flex; flex-wrap: wrap; gap: 10px; margin-bottom: 14px; }
.control-panel button { min-width: 148px; }
.control-panel button.secondary { border-color: var(--blue); background: rgba(92, 168, 255, .12); }
.control-panel button.secondary:hover { background: rgba(92, 168, 255, .2); }
.control-panel button.danger { border-color: var(--red); background: rgba(239, 100, 97, .12); }
.control-panel button.danger:hover { background: rgba(239, 100, 97, .2); }
.compact-controls { margin-top: 12px; margin-bottom: 0; }
.experiment-form { display: grid; grid-template-columns: repeat(4, minmax(160px, 1fr)); gap: 12px; align-items: end; margin-bottom: 12px; }
.field { display: grid; gap: 6px; }
.field label { color: var(--muted); font-size: 12px; }
.round-input, select { width: 100%; border: 1px solid var(--line); border-radius: 6px; background: #0f1315; color: var(--text); padding: 9px 10px; font: inherit; }
.round-input:focus, select:focus { outline: 1px solid var(--green); }
.method-list { display: flex; flex-wrap: wrap; gap: 8px; margin: 4px 0 12px; }
.method-list label { border: 1px solid var(--line); border-radius: 999px; padding: 7px 10px; color: var(--muted); cursor: pointer; }
.method-list input { margin-right: 6px; }
.method-list label:has(input:checked) { color: var(--text); border-color: var(--green); background: rgba(79, 208, 138, .10); }
.grid { display: grid; grid-template-columns: 1.2fr .8fr; gap: 14px; }
.cards { display: grid; grid-template-columns: repeat(4, minmax(130px, 1fr)); gap: 10px; margin-bottom: 14px; }
.policy-table-wrap { overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; }
.policy-table { width: 100%; border-collapse: collapse; font-size: 13px; min-width: 760px; }
.policy-table th, .policy-table td { padding: 9px 10px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
.policy-table th:first-child, .policy-table td:first-child,
.policy-table th:nth-child(2), .policy-table td:nth-child(2) { text-align: left; }
.policy-table tr:last-child td { border-bottom: 0; }
.status-tag { font-size: 12px; color: var(--muted); border: 1px solid var(--line); border-radius: 999px; padding: 3px 8px; }
.status-tag.running { color: var(--green); border-color: var(--green); background: rgba(79, 208, 138, .1); }
.status-tag.completed, .status-tag.policy_completed { color: var(--blue); border-color: var(--blue); background: rgba(92, 168, 255, .1); }
.status-tag.failed { color: var(--red); border-color: var(--red); background: rgba(238, 111, 112, .1); }
.figure-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(520px, 1fr)); gap: 14px; }
.figure-grid.large-figures { grid-template-columns: 1fr; }
.figure-card { border: 1px solid var(--line); border-radius: 8px; background: #111518; overflow: hidden; }
.figure-card img { display: block; width: 100%; min-height: 360px; max-height: 76vh; object-fit: contain; background: #fff; }
.figure-caption { padding: 9px 10px; color: var(--muted); font-size: 12px; overflow-wrap: anywhere; border-top: 1px solid var(--line); }
.figure-caption a { color: var(--blue); text-decoration: none; font-weight: 650; margin-left: 8px; }
.merge-list { display: grid; gap: 8px; max-height: 280px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: #111518; margin-bottom: 12px; }
.merge-item { display: grid; grid-template-columns: 24px 200px 1fr 88px 88px 110px; gap: 8px; align-items: center; padding: 8px; border: 1px solid var(--line); border-radius: 6px; color: var(--muted); font-size: 12px; }
.merge-item strong { color: var(--text); font-size: 13px; }
.merge-item .run-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.merge-item:has(input:checked) { border-color: var(--green); background: rgba(79, 208, 138, .08); }
.delete-list { display: grid; gap: 8px; max-height: 260px; overflow: auto; border: 1px solid var(--line); border-radius: 8px; padding: 8px; background: #111518; margin-bottom: 12px; }
.delete-item { display: grid; grid-template-columns: 24px 112px 1fr; gap: 8px; align-items: center; padding: 8px; border: 1px solid var(--line); border-radius: 6px; color: var(--muted); font-size: 12px; }
.delete-item strong { color: var(--text); font-size: 13px; }
.delete-item .dir-name { overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.delete-item:has(input:checked) { border-color: var(--red); background: rgba(239, 100, 97, .08); }
.live-svg-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
.live-svg { width: 100%; height: 340px; display: block; border: 1px solid var(--line); border-radius: 6px; background: #ffffff; }
.live-chart-controls { display: flex; gap: 10px; align-items: center; margin-bottom: 10px; flex-wrap: wrap; }
.mode-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(260px, 1fr)); gap: 12px; }
.mode-card { border: 1px solid var(--line); border-radius: 8px; padding: 12px; background: #111518; }
.mode-title { display: flex; justify-content: space-between; gap: 8px; align-items: center; margin-bottom: 10px; font-weight: 650; }
.mode-bar-row { display: grid; grid-template-columns: 72px 1fr 46px; gap: 8px; align-items: center; margin: 7px 0; font-size: 12px; }
.mode-bar-track { height: 9px; border: 1px solid var(--line); border-radius: 999px; background: #0b0d0f; overflow: hidden; }
.mode-bar-fill { height: 100%; background: var(--green); }
.card, .panel { background: var(--panel); border: 1px solid var(--line); border-radius: 8px; }
.card { padding: 13px 14px; min-height: 82px; }
.label { color: var(--muted); font-size: 12px; }
.value { font-size: 24px; line-height: 1.2; margin-top: 8px; font-weight: 650; }
.bar-wrap { height: 14px; background: #0b0d0f; border: 1px solid var(--line); border-radius: 999px; overflow: hidden; margin: 12px 0 4px; }
.bar { height: 100%; width: 0%; background: linear-gradient(90deg, var(--green), var(--blue)); transition: width .25s ease; }
.panel { padding: 15px; margin-bottom: 14px; }
.panel h2 { margin: 0 0 12px; font-size: 15px; font-weight: 650; }
details.panel { display: block; }
details.panel > summary { cursor: pointer; list-style: none; font-size: 15px; font-weight: 650; margin: 0; }
details.panel > summary::-webkit-details-marker { display: none; }
details.panel > summary::after { content: "Open"; float: right; color: var(--muted); font-size: 12px; font-weight: 500; }
details.panel[open] > summary { margin-bottom: 12px; }
details.panel[open] > summary::after { content: "Close"; }
.section-kicker { color: var(--muted); font-size: 12px; margin: -4px 0 12px; }
.module-grid { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 14px; align-items: start; }
.wide-panel { grid-column: 1 / -1; }
canvas { width: 100%; height: 285px; display: block; background: #111518; border: 1px solid var(--line); border-radius: 6px; }
.viz { height: 285px; position: relative; overflow: hidden; background: #111518; border: 1px solid var(--line); border-radius: 6px; }
.cloud { position: absolute; left: 50%; top: 26px; transform: translateX(-50%); width: 142px; height: 70px; border: 2px solid var(--blue); border-radius: 38px; display: grid; place-items: center; color: var(--blue); font-weight: 650; opacity: .55; }
.cloud.active { opacity: 1; background: rgba(92, 168, 255, .12); box-shadow: 0 0 18px rgba(92, 168, 255, .35); }
.topology-lines { position: absolute; inset: 0; width: 100%; height: 100%; pointer-events: none; }
.topology-lines line { stroke: rgba(170, 180, 184, .32); stroke-width: 1.4; stroke-dasharray: 4 5; }
.edge { position: absolute; width: 112px; height: 52px; border: 2px solid var(--green); border-radius: 7px; display: grid; place-items: center; color: var(--green); bottom: 92px; transform: translateX(-50%); }
.edge.e1 { left: 29%; } .edge.e2 { left: 71%; }
.edge.active { background: rgba(79, 208, 138, .12); box-shadow: 0 0 18px rgba(79, 208, 138, .35); }
.client { position: absolute; width: clamp(32px, 5.2vw, 42px); height: 38px; border-radius: 7px; border: 1px solid var(--amber); bottom: 24px; display: grid; place-items: center; color: var(--amber); font-size: 12px; transform: translateX(-50%); }
.client.active { background: rgba(79, 208, 138, .14); border-color: var(--green); color: var(--green); box-shadow: 0 0 18px rgba(79, 208, 138, .35); }
.packet { position: absolute; min-width: 24px; height: 18px; padding: 0 5px; border-radius: 999px; background: var(--packet-color, var(--amber)); color: #101214; box-shadow: 0 0 16px var(--packet-color, var(--amber)); left: var(--sx); top: var(--sy); transform: translate(-50%, -50%); animation: flyPacket 2.2s linear infinite; animation-delay: var(--delay); display: grid; place-items: center; font-size: 10px; font-weight: 700; }
.packet.local { min-width: 26px; height: 20px; animation: localPulse 1.5s ease-in-out infinite; }
@keyframes flyPacket {
  from { left: var(--sx); top: var(--sy); opacity: .1; transform: translate(-50%, -50%) scale(.8); }
  45% { left: var(--mx); top: var(--my); opacity: 1; transform: translate(-50%, -50%) scale(1); }
  to { left: var(--tx); top: var(--ty); opacity: .22; transform: translate(-50%, -50%) scale(.9); }
}
@keyframes localPulse {
  0% { transform: scale(.45); opacity: .2; }
  50% { transform: scale(1.25); opacity: .95; }
  100% { transform: scale(.45); opacity: .2; }
}
.mode-row { display: grid; grid-template-columns: 86px 1fr 54px; align-items: center; gap: 10px; margin: 9px 0; font-size: 13px; }
.mode-meter { height: 10px; background: #0b0d0f; border-radius: 99px; overflow: hidden; border: 1px solid var(--line); }
.mode-fill { height: 100%; width: 0%; background: var(--green); }
table { width: 100%; border-collapse: collapse; font-size: 12px; }
th, td { padding: 7px 8px; border-bottom: 1px solid var(--line); text-align: right; white-space: nowrap; }
th:first-child, td:first-child { text-align: left; }
th { color: var(--muted); font-weight: 500; }
@media (max-width: 960px) { .grid, .cards, .experiment-form, .live-svg-grid, .module-grid { grid-template-columns: 1fr; } header { align-items: flex-start; flex-direction: column; } }
@media (max-width: 620px) {
  .viz { height: 330px; }
  .client { bottom: 54px; }
  .client[data-client="1"], .client[data-client="3"], .client[data-client="5"], .client[data-client="7"], .client[data-client="9"] { bottom: 12px; }
  .client[data-client="0"] { left: 10%; } .client[data-client="2"] { left: 30%; } .client[data-client="4"] { left: 50%; } .client[data-client="6"] { left: 70%; } .client[data-client="8"] { left: 90%; }
  .client[data-client="1"] { left: 10%; } .client[data-client="3"] { left: 30%; } .client[data-client="5"] { left: 50%; } .client[data-client="7"] { left: 70%; } .client[data-client="9"] { left: 90%; }
}
</style>
</head>
<body>
<main>
  <header>
    <div>
      <h1>DynFedPrivacy LeNet5 Monitor</h1>
      <div id="path" class="sub">waiting for run output</div>
    </div>
    <div class="actions">
      <div id="updated" class="pill">waiting</div>
    </div>
  </header>

  <section class="panel">
    <h2>Main Comparison</h2>
    <div class="section-kicker">Run the primary method comparison. Privacy-Only is available for diagnostic runs but is not selected by default.</div>
    <div class="experiment-form">
      <div class="field">
        <label for="runMode">Run mode</label>
        <select id="runMode">
          <option value="rounds">Round setting</option>
          <option value="time">Time-limit setting</option>
        </select>
      </div>
      <div class="field">
        <label for="roundsInput">Rounds</label>
        <input class="round-input" id="roundsInput" type="number" min="1" max="500" step="1" value="200">
      </div>
      <div class="field">
        <label for="datasetSelect">Dataset</label>
        <select id="datasetSelect">
          <option value="fmnist">Fashion-MNIST</option>
          <option value="cifar10" selected>CIFAR-10</option>
          <option value="cifar100">CIFAR-100</option>
        </select>
      </div>
      <div class="field">
        <label for="modelSelect">Model</label>
        <select id="modelSelect">
          <option value="lenet5">LeNet-5</option>
          <option value="smallcnn">Small CNN</option>
          <option value="avgcnn">DriftRace AvgCNN</option>
          <option value="tinyresnet">Tiny ResNet</option>
          <option value="resnet18">ResNet-18</option>
          <option value="resnet50">ResNet-50</option>
          <option value="resnet18_pretrained" selected>ResNet-18 Pretrained</option>
          <option value="resnet50_pretrained">ResNet-50 Pretrained</option>
        </select>
      </div>
      <div class="field">
        <label for="deviceSelect">Device</label>
        <select id="deviceSelect">
          <option value="cpu">CPU</option>
          <option value="cuda" selected>GPU (CUDA)</option>
        </select>
      </div>
      <div class="field">
        <label for="executorSelect">Executor</label>
        <select id="executorSelect">
          <option value="serial" selected>Serial</option>
          <option value="process_pool">Process pool</option>
        </select>
      </div>
      <div class="field">
        <label for="executorWorkersInput">Executor workers</label>
        <input class="round-input" id="executorWorkersInput" type="number" min="1" max="64" step="1" placeholder="Auto">
      </div>
      <div class="field">
        <label for="clientsInput">Clients</label>
        <input class="round-input" id="clientsInput" type="number" min="1" max="200" step="1" value="100">
      </div>
      <div class="field">
        <label for="edgesInput">Edges</label>
        <input class="round-input" id="edgesInput" type="number" min="1" max="20" step="1" value="10">
      </div>
      <div class="field">
        <label for="seedInput">Seed</label>
        <input class="round-input" id="seedInput" type="number" min="0" max="999999" step="1" value="42">
      </div>
      <div class="field">
        <label for="timeLimitInput">Time limit constraint</label>
        <input class="round-input" id="timeLimitInput" type="number" min="0.1" max="300" step="0.1" value="300">
      </div>
      <div class="field">
        <label for="trainLimitInput">Train sample limit</label>
        <input class="round-input" id="trainLimitInput" type="number" min="1" max="60000" step="100" value="12000">
      </div>
      <div class="field">
        <label for="testLimitInput">Test sample limit</label>
        <input class="round-input" id="testLimitInput" type="number" min="1" max="10000" step="100" value="2000">
      </div>
      <div class="field">
        <label for="figureAxis">Figure axis</label>
        <select id="figureAxis">
          <option value="round">Round-Accuracy</option>
          <option value="time">Time-Accuracy</option>
        </select>
      </div>
      <div class="field">
        <label for="partitionMode">Data distribution</label>
        <select id="partitionMode">
          <option value="extreme_edge_label_skew" selected>Extreme edge label-skew</option>
          <option value="edge_label_skew">Edge label-skew</option>
          <option value="client_noniid">Client non-IID</option>
          <option value="iid">IID</option>
        </select>
      </div>
      <div class="field">
        <label for="clientHetInput">Client heterogeneity</label>
        <input class="round-input" id="clientHetInput" type="number" min="1.0" max="10.0" step="0.1" value="2.0">
      </div>
      <div class="field">
        <label for="edgeHetInput">Edge heterogeneity</label>
        <input class="round-input" id="edgeHetInput" type="number" min="1.0" max="10.0" step="0.1" value="1.5">
      </div>
      <div class="field">
        <label for="selectionPeriodInput">Strategy update period</label>
        <input class="round-input" id="selectionPeriodInput" type="number" min="1" max="100" step="1" value="1">
      </div>
      <div class="field">
        <label for="aggregationFractionInput">Aggregation fraction</label>
        <input class="round-input" id="aggregationFractionInput" type="number" min="0.1" max="1.0" step="0.05" value="1.0">
      </div>
      <div class="field">
        <label for="paretoArchiveSizeInput">Pareto archive K_P</label>
        <input class="round-input" id="paretoArchiveSizeInput" type="number" min="2" max="128" step="1" value="16">
      </div>
      <div class="field">
        <label for="paretoMaxItersInput">Pareto expansions I_max</label>
        <input class="round-input" id="paretoMaxItersInput" type="number" min="0" max="200" step="1" value="50">
      </div>
      <div class="field">
        <label for="cloudFusionXiInput">Cloud-fusion penalty xi</label>
        <input class="round-input" id="cloudFusionXiInput" type="number" min="0" max="100" step="0.01" value="0.2">
      </div>
      <div class="field">
        <label for="cloudFusionEpsInput">Cloud-fusion epsilon</label>
        <input class="round-input" id="cloudFusionEpsInput" type="number" min="0.000001" max="10" step="0.001" value="0.05">
      </div>
      <div class="field">
        <label for="minEdgeCloudFusionInput">Min cloud samples per edge</label>
        <input class="round-input" id="minEdgeCloudFusionInput" type="number" min="0" max="1" step="0.05" value="0.5">
      </div>
      <div class="field">
        <label for="resourceLimitInput">Resource limit</label>
        <input class="round-input" id="resourceLimitInput" type="number" min="0" max="100" step="0.05" value="1.35">
      </div>
      <div class="field">
        <label for="riskLimitInput">Risk limit</label>
        <input class="round-input" id="riskLimitInput" type="number" min="0" max="1" step="0.01" value="0.5">
      </div>
      <div class="field">
        <label for="localEpochsInput">Local epochs</label>
        <input class="round-input" id="localEpochsInput" type="number" min="1" max="20" step="1" placeholder="Auto">
      </div>
      <div class="field">
        <label for="learningRateInput">Learning rate</label>
        <input class="round-input" id="learningRateInput" type="number" min="0.000001" max="1.0" step="0.0001" placeholder="Auto">
      </div>
      <input id="initialEpsilonInput" type="hidden" value="8.0">
      <div class="field">
        <label for="dpEmbEpsilonInput">Feature total epsilon target</label>
        <input class="round-input" id="dpEmbEpsilonInput" type="number" min="0.001" max="100" step="0.1" value="8.0">
      </div>
      <div class="field">
        <label for="dpUpdEpsilonInput">Update total epsilon target</label>
        <input class="round-input" id="dpUpdEpsilonInput" type="number" min="0.001" max="100" step="0.1" value="8.0">
      </div>
      <div class="field">
        <label for="dpProfileSelect">DP profile</label>
        <select id="dpProfileSelect">
          <option value="balanced">Auto RDP (C=1)</option>
          <option value="strong">Auto RDP (C=20)</option>
          <option value="cifar_resnet" selected>CIFAR Auto RDP (C=1)</option>
          <option value="weak_update">Auto RDP (C=0.5)</option>
        </select>
      </div>
      <div class="field">
        <label for="heBackendSelect">HE backend</label>
        <select id="heBackendSelect">
          <option value="none">None</option>
          <option value="seal" selected>SEAL</option>
          <option value="tenseal">TenSEAL</option>
        </select>
      </div>
      <div class="field">
        <label for="requireRealHeInput">Require real HE</label>
        <select id="requireRealHeInput">
          <option value="true" selected>True</option>
          <option value="false">False</option>
        </select>
      </div>
      <div class="field">
        <label for="resumeFromRunInput">Resume from run directory</label>
        <input class="round-input" id="resumeFromRunInput" type="text" placeholder="Optional checkpointed lenet5_dynamic run path">
      </div>
    </div>
    <div class="method-list" id="methodList">
      <label><input type="checkbox" value="ours" checked>DynFedPrivacy</label>
      <label><input type="checkbox" value="individual_optimal" checked>Individual-Optimal</label>
      <label><input type="checkbox" value="fixed_dp">Fixed-DP</label>
      <label><input type="checkbox" value="privacy_only">Privacy-Only</label>
      <label><input type="checkbox" value="no_protection" checked>No-Protection</label>
      <label><input type="checkbox" value="random" checked>Random</label>
      <label><input type="checkbox" value="fixed_fedavg">Fixed FedAvg</label>
      <label><input type="checkbox" value="fixed_splitfed">Fixed SplitFed</label>
      <label><input type="checkbox" value="fixed_hfl">Fixed HFL</label>
      <label><input type="checkbox" value="nsga2">NSGA II</label>
      <label><input type="checkbox" value="fixed_liieiiic">Fixed-HFL (LIIEIIIC)</label>
      <label><input type="checkbox" value="performance_only">Global-Balance Upper</label>
      <label><input type="checkbox" value="best_accuracy">Global-Utility Upper</label>
      <label><input type="checkbox" value="accuracy_oracle">Accuracy-Oracle Upper</label>
    </div>
    <div class="control-panel">
      <button type="button" id="runConfiguredBtn">Start Experiment</button>
    </div>
    <div class="sub" id="controlStatus">Ready</div>
  </section>

  <div class="module-grid">
  <details class="panel">
    <summary>Heterogeneity Ablation</summary>
    <div class="section-kicker">Run the same method set under IID, client non-IID, edge label-skew, and extreme edge label-skew.</div>
    <div class="experiment-form">
      <div class="field">
        <label for="ablRoundsInput">Rounds</label>
        <input class="round-input" id="ablRoundsInput" type="number" min="1" max="500" step="1" value="100">
      </div>
      <div class="field">
        <label for="ablEdgesInput">Edges</label>
        <input class="round-input" id="ablEdgesInput" type="number" min="1" max="5" step="1" value="3">
      </div>
      <div class="field">
        <label for="ablTimeLimitInput">Time limit constraint</label>
        <input class="round-input" id="ablTimeLimitInput" type="number" min="0.1" max="300" step="0.1" value="8.0">
      </div>
      <div class="field">
        <label for="ablFigureAxis">Figure axis</label>
        <select id="ablFigureAxis">
          <option value="round">Round-Accuracy</option>
          <option value="time">Time-Accuracy</option>
        </select>
      </div>
      <div class="field">
        <label for="ablClientHetInput">Client heterogeneity</label>
        <input class="round-input" id="ablClientHetInput" type="number" min="1.0" max="10.0" step="0.1" value="2.0">
      </div>
      <div class="field">
        <label for="ablEdgeHetInput">Edge heterogeneity</label>
        <input class="round-input" id="ablEdgeHetInput" type="number" min="1.0" max="10.0" step="0.1" value="1.5">
      </div>
      <div class="field">
        <label for="ablSelectionPeriodInput">Strategy update period</label>
        <input class="round-input" id="ablSelectionPeriodInput" type="number" min="1" max="100" step="1" value="5">
      </div>
    </div>
    <div class="method-list" id="ablationMethodList">
      <label><input type="checkbox" value="ours" checked>DynFedPrivacy</label>
      <label><input type="checkbox" value="individual_optimal" checked>Individual-Optimal</label>
      <label><input type="checkbox" value="fixed_dp" checked>Fixed-DP</label>
      <label><input type="checkbox" value="privacy_only">Privacy-Only</label>
      <label><input type="checkbox" value="no_protection">No-Protection</label>
      <label><input type="checkbox" value="random" checked>Random</label>
      <label><input type="checkbox" value="fixed_liieiiic">Fixed-HFL (LIIEIIIC)</label>
      <label><input type="checkbox" value="accuracy_oracle">Accuracy-Oracle Upper</label>
    </div>
    <div class="control-panel">
      <button class="secondary ablation-run" type="button" data-partition="iid">Run IID</button>
      <button class="secondary ablation-run" type="button" data-partition="client_noniid">Run Client non-IID</button>
      <button class="secondary ablation-run" type="button" data-partition="edge_label_skew">Run Edge label-skew</button>
      <button type="button" class="ablation-run" data-partition="extreme_edge_label_skew">Run Extreme edge label-skew</button>
    </div>
    <div class="sub" id="ablationStatus">Ready</div>
  </details>

  <details class="panel" open>
    <summary>Algorithm Ablation</summary>
    <div class="section-kicker">All variants inherit the current main experiment settings.</div>
    <div class="method-list">
      <label>Full Method</label>
      <label>Without Global Coordination</label>
      <label>No Error Cost Estimate</label>
      <label>Fixed Mode LIIEIIIC</label>
    </div>
    <div class="control-panel">
      <button type="button" id="runAlgorithmAblationBtn">Run Algorithm Ablation</button>
    </div>
    <div class="sub" id="algorithmAblationStatus">Ready. Completed matching runs will be reused.</div>
  </details>

  <details class="panel" open>
    <summary>Parameter Sweep</summary>
    <div class="section-kicker">Run DynFedPrivacy under multiple strategy update periods using the current main experiment settings.</div>
    <div class="experiment-form">
      <div class="field">
        <label for="sweepPeriodsInput">Strategy update periods</label>
        <input class="round-input" id="sweepPeriodsInput" type="text" value="1,5,10,15,20">
      </div>
      <div class="field">
        <label for="sweepMethodSelect">Method</label>
        <select id="sweepMethodSelect">
          <option value="ours" selected>DynFedPrivacy</option>
          <option value="individual_optimal">Individual-Optimal</option>
          <option value="fixed_dp">Fixed-DP</option>
          <option value="random">Random</option>
          <option value="fixed_liieiiic">Fixed-HFL (LIIEIIIC)</option>
          <option value="accuracy_oracle">Accuracy-Oracle Upper</option>
        </select>
      </div>
    </div>
    <div class="control-panel">
      <button type="button" id="runPeriodSweepBtn">Run Period Sweep</button>
    </div>
    <div class="sub" id="periodSweepStatus">Ready. Results can be combined in Merge Runs after each period finishes.</div>
  </details>

  <details class="panel" open>
    <summary>Privacy Budget Sweep</summary>
    <div class="section-kicker">Vary the feature and update total epsilon targets together while keeping the current main experiment settings.</div>
    <div class="experiment-form">
      <div class="field">
        <label for="privacyBudgetsInput">Total epsilon targets</label>
        <input class="round-input" id="privacyBudgetsInput" type="text" value="1,2,4,8">
      </div>
      <div class="field">
        <label for="privacySweepMethodSelect">Method</label>
        <select id="privacySweepMethodSelect">
          <option value="ours" selected>DynFedPrivacy</option>
          <option value="individual_optimal">Individual Optimal</option>
          <option value="random">Random</option>
        </select>
      </div>
    </div>
    <div class="control-panel">
      <button type="button" id="runPrivacySweepBtn">Run Privacy Budget Sweep</button>
    </div>
    <div class="sub" id="privacySweepStatus">Ready. Each value is applied to both privacy channels.</div>
  </details>

  <details class="panel" open>
    <summary>Heterogeneity × Period Sweep</summary>
    <div class="section-kicker">Validate whether a strategy update period remains strong across multiple heterogeneity settings.</div>
    <div class="experiment-form">
      <div class="field">
        <label for="hetSweepRoundsInput">Rounds</label>
        <input class="round-input" id="hetSweepRoundsInput" type="number" min="1" max="500" step="1" value="200">
      </div>
      <div class="field">
        <label for="hetSweepSeedInput">Seed</label>
        <input class="round-input" id="hetSweepSeedInput" type="number" min="0" max="999999" step="1" value="42">
      </div>
      <div class="field">
        <label for="hetSweepEdgesInput">Edges</label>
        <input class="round-input" id="hetSweepEdgesInput" type="number" min="1" max="5" step="1" value="3">
      </div>
      <div class="field">
        <label for="hetSweepPartitionMode">Data distribution</label>
        <select id="hetSweepPartitionMode">
          <option value="extreme_edge_label_skew" selected>Extreme edge label-skew</option>
          <option value="edge_label_skew">Edge label-skew</option>
          <option value="client_noniid">Client non-IID</option>
          <option value="iid">IID</option>
        </select>
      </div>
      <div class="field">
        <label for="hetSweepTimeLimitInput">Time limit constraint</label>
        <input class="round-input" id="hetSweepTimeLimitInput" type="number" min="0.1" max="300" step="0.1" value="8.0">
      </div>
      <div class="field">
        <label for="hetSweepPeriodsInput">Strategy update periods</label>
        <input class="round-input" id="hetSweepPeriodsInput" type="text" value="5,10,15,20,25,50">
      </div>
      <div class="field">
        <label for="hetSweepScenariosInput">Heterogeneity scenarios</label>
        <input class="round-input" id="hetSweepScenariosInput" type="text" value="2.0/1.5,4.0/3.0,6.0/4.0">
      </div>
      <div class="field">
        <label for="hetSweepMethodSelect">Method</label>
        <select id="hetSweepMethodSelect">
          <option value="ours" selected>DynFedPrivacy</option>
          <option value="individual_optimal">Individual-Optimal</option>
          <option value="fixed_dp">Fixed-DP</option>
          <option value="random">Random</option>
          <option value="fixed_liieiiic">Fixed-HFL (LIIEIIIC)</option>
          <option value="accuracy_oracle">Accuracy-Oracle Upper</option>
        </select>
      </div>
    </div>
    <div class="control-panel">
      <button type="button" id="runHetPeriodSweepBtn">Run Heterogeneity × Period Sweep</button>
    </div>
    <div class="sub" id="hetPeriodSweepStatus">Use scenario format client/edge, separated by commas. Example: 2.0/1.5,4.0/3.0,6.0/4.0.</div>
  </details>

  <details class="panel">
    <summary>Robustness Analysis</summary>
    <div class="section-kicker">One-click robustness runs for strong heterogeneity, strict latency pressure, and multi-seed stability.</div>
    <div class="experiment-form">
      <div class="field">
        <label for="robustRoundsInput">Rounds</label>
        <input class="round-input" id="robustRoundsInput" type="number" min="1" max="500" step="1" value="100">
      </div>
      <div class="field">
        <label for="robustSeedInput">Seed</label>
        <input class="round-input" id="robustSeedInput" type="number" min="0" max="999999" step="1" value="43">
      </div>
      <div class="field">
        <label for="robustEdgesInput">Edges</label>
        <input class="round-input" id="robustEdgesInput" type="number" min="1" max="5" step="1" value="3">
      </div>
      <div class="field">
        <label for="robustPartitionMode">Data distribution</label>
        <select id="robustPartitionMode">
          <option value="extreme_edge_label_skew" selected>Extreme edge label-skew</option>
          <option value="edge_label_skew">Edge label-skew</option>
          <option value="client_noniid">Client non-IID</option>
          <option value="iid">IID</option>
        </select>
      </div>
      <div class="field">
        <label for="robustTimeLimitInput">Time limit constraint</label>
        <input class="round-input" id="robustTimeLimitInput" type="number" min="0.1" max="300" step="0.1" value="8.0">
      </div>
      <div class="field">
        <label for="robustClientHetInput">Client heterogeneity</label>
        <input class="round-input" id="robustClientHetInput" type="number" min="1.0" max="10.0" step="0.1" value="3.0">
      </div>
      <div class="field">
        <label for="robustEdgeHetInput">Edge heterogeneity</label>
        <input class="round-input" id="robustEdgeHetInput" type="number" min="1.0" max="10.0" step="0.1" value="2.5">
      </div>
      <div class="field">
        <label for="robustSelectionPeriodInput">Strategy update period</label>
        <input class="round-input" id="robustSelectionPeriodInput" type="number" min="1" max="100" step="1" value="5">
      </div>
    </div>
    <div class="method-list" id="robustMethodList">
      <label><input type="checkbox" value="ours" checked>DynFedPrivacy</label>
      <label><input type="checkbox" value="individual_optimal" checked>Individual-Optimal</label>
      <label><input type="checkbox" value="fixed_dp" checked>Fixed-DP</label>
      <label><input type="checkbox" value="random" checked>Random</label>
      <label><input type="checkbox" value="fixed_liieiiic">Fixed-HFL (LIIEIIIC)</label>
      <label><input type="checkbox" value="no_protection">No-Protection</label>
      <label><input type="checkbox" value="privacy_only">Privacy-Only</label>
      <label><input type="checkbox" value="accuracy_oracle">Accuracy-Oracle Upper</label>
    </div>
    <div class="control-panel">
      <button type="button" id="runRobustStrongHetBtn">A. Strong Heterogeneity</button>
      <button class="secondary" type="button" id="runRobustLatencyBtn">B. Time-Stress</button>
    </div>
    <div class="sub">A uses extreme edge label-skew with stronger client/edge heterogeneity. B tightens the time limit. C repeats the same setting under different seeds.</div>
    <div class="control-panel">
      <button class="secondary robust-seed-run" type="button" data-seed="42">C. Seed 42</button>
      <button class="secondary robust-seed-run" type="button" data-seed="43">C. Seed 43</button>
      <button class="secondary robust-seed-run" type="button" data-seed="44">C. Seed 44</button>
    </div>
    <div class="sub" id="robustStatus">Ready</div>
  </details>
  </div>

  <section class="panel">
    <h2>Run Monitor</h2>
    <div class="label" id="headline">Waiting for experiment status</div>
    <div class="bar-wrap"><div class="bar" id="progress"></div></div>
    <div class="sub" id="message"></div>
    <div class="control-panel compact-controls">
      <button class="secondary danger" type="button" id="stopRunStatusBtn">Stop Run</button>
    </div>
  </section>

  <div class="cards">
    <div class="card"><div class="label">Policy</div><div class="value" id="policy">-</div></div>
    <div class="card"><div class="label">Round</div><div class="value" id="round">-</div></div>
    <div class="card"><div class="label">Test Accuracy</div><div class="value" id="acc">-</div></div>
    <div class="card"><div class="label">Best Accuracy</div><div class="value" id="best">-</div></div>
    <div class="card"><div class="label">Logical Time</div><div class="value" id="logical">-</div></div>
    <div class="card"><div class="label">Larger Channel Epsilon</div><div class="value" id="eps">-</div></div>
    <div class="card"><div class="label">Feature Epsilon</div><div class="value" id="featureEps">-</div></div>
    <div class="card"><div class="label">Update Epsilon</div><div class="value" id="updateEps">-</div></div>
    <div class="card"><div class="label">Effective Clients</div><div class="value" id="clients">-</div></div>
    <div class="card"><div class="label">Communication</div><div class="value" id="comm">-</div></div>
    <div class="card"><div class="label">Per‑Client Test Acc</div><div class="value" id="perClientAcc" style="font-size:18px">-</div></div>
  </div>

  <section class="panel">
    <h2>Results Table</h2>
    <div id="policyRuns"></div>
  </section>

  <section class="panel">
    <h2>Merge Runs</h2>
    <div class="section-kicker">Select method outputs from different runs and build one combined run directory.</div>
    <div class="experiment-form">
      <div class="field">
        <label for="mergeNameInput">Merged experiment name</label>
        <input class="round-input" id="mergeNameInput" type="text" value="paper_combined">
      </div>
      <div class="field">
        <label for="tailZoomStartInput">Tail zoom start round</label>
        <input class="round-input" id="tailZoomStartInput" type="text" placeholder="Auto: last 25%">
      </div>
      <div class="field">
        <label for="smoothCurvesToggle">Curve smoothing</label>
        <label class="pill"><input id="smoothCurvesToggle" type="checkbox"> Smooth curves</label>
      </div>
    </div>
    <div class="merge-list" id="mergeSourceList"></div>
    <div class="control-panel">
      <button type="button" id="mergeRunsBtn">Merge Selected Methods</button>
    </div>
    <div class="sub" id="mergeStatus">Select method outputs to merge.</div>
    <div class="figure-grid large-figures" id="mergeFigureGrid"></div>
  </section>

  <section class="panel">
    <h2>Clean Output Directories</h2>
    <div class="section-kicker">Select old experiment directories under out/ and delete only the checked items. Deletion is disabled while training is running.</div>
    <div class="delete-list" id="deleteTargetList"></div>
    <div class="control-panel">
      <button class="secondary danger" type="button" id="deleteDirsBtn">Delete Selected Directories</button>
    </div>
    <div class="sub" id="deleteStatus">Select directories to delete.</div>
  </section>

  <section class="panel">
    <h2>Paper Experiment Archive</h2>
    <div class="section-kicker">Save the current run and paper figures. The tail zoom uses the last quarter of the axis by default.</div>
    <div class="control-panel">
      <button type="button" id="savePaperRunBtn">Save Current Run</button>
      <button class="secondary" type="button" id="buildRunFiguresBtn">Build Figures</button>
      <button class="secondary" type="button" id="rebuildBtn">Rebuild Paper Figures</button>
    </div>
    <div class="sub" id="paperArchiveStatus">Ready</div>
    <div class="figure-grid" id="runFigureGrid"></div>
    <div class="figure-grid" id="paperArchiveGrid"></div>
  </section>

  <div class="module-grid">
  <section class="panel">
    <h2>Live Accuracy Curves</h2>
    <div class="live-chart-controls">
      <div class="sub">Round-accuracy and time-accuracy are drawn together. The time axis is clipped to the shortest valid max time among visible curves.</div>
      <label class="pill"><input id="liveSmoothCurvesToggle" type="checkbox"> Smooth live curves</label>
    </div>
    <div class="live-svg-grid">
      <svg class="live-svg" id="liveRoundSvg" viewBox="0 0 760 340" role="img" aria-label="Live round accuracy chart"></svg>
      <svg class="live-svg" id="liveTimeSvg" viewBox="0 0 760 340" role="img" aria-label="Live time accuracy chart"></svg>
    </div>
  </section>

  <section class="panel">
    <h2>Mode Selection Distribution</h2>
    <div class="mode-grid" id="modeGrid"></div>
  </section>
  </div>

  <div class="module-grid">
    <section class="panel">
      <h2>Cloud-Edge-Client Flow</h2>
      <div class="viz">
        <svg class="topology-lines" viewBox="0 0 100 100" preserveAspectRatio="none" aria-hidden="true">
          <line x1="29" y1="66" x2="8" y2="89"></line><line x1="29" y1="66" x2="18" y2="89"></line><line x1="29" y1="66" x2="28" y2="89"></line><line x1="29" y1="66" x2="38" y2="89"></line><line x1="29" y1="66" x2="48" y2="89"></line>
          <line x1="71" y1="66" x2="52" y2="89"></line><line x1="71" y1="66" x2="62" y2="89"></line><line x1="71" y1="66" x2="72" y2="89"></line><line x1="71" y1="66" x2="82" y2="89"></line><line x1="71" y1="66" x2="92" y2="89"></line>
        </svg>
        <div class="cloud" id="cloudNode">Cloud</div>
        <div class="edge e1" data-edge="0">Edge 0</div>
        <div class="edge e2" data-edge="1">Edge 1</div>
        <div class="client" data-client="0" style="left:8%">C0</div><div class="client" data-client="2" style="left:18%">C2</div><div class="client" data-client="4" style="left:28%">C4</div><div class="client" data-client="6" style="left:38%">C6</div><div class="client" data-client="8" style="left:48%">C8</div>
        <div class="client" data-client="1" style="left:52%">C1</div><div class="client" data-client="3" style="left:62%">C3</div><div class="client" data-client="5" style="left:72%">C5</div><div class="client" data-client="7" style="left:82%">C7</div><div class="client" data-client="9" style="left:92%">C9</div>
      </div>
      <div class="sub" id="activeClients" style="margin-top:10px">Active clients: -</div>
      <div class="sub" id="activePath">Active path: waiting</div>
    </section>

    <section class="panel">
      <h2>Recent Rounds</h2>
      <table>
        <thead><tr><th>Round</th><th>Acc</th><th>Best</th><th>Time</th><th>Clients</th></tr></thead>
        <tbody id="roundRows"></tbody>
      </table>
    </section>
  </div>
</main>
<script>
const $ = id => document.getElementById(id);
const fmt = (v, n=3) => Number.isFinite(Number(v)) ? Number(v).toFixed(n) : "-";
let latestStatus = {};
let refreshInFlight = null;
let autoBuiltRunFiguresFor = "";
let runFigureSignature = "";

function tailZoomStartRound() {
  const input = $("tailZoomStartInput");
  const raw = input?.value.trim() || "";
  if (!raw || raw.toLowerCase() === "auto") return "";
  const value = Number(raw);
  if (!Number.isFinite(value) || value < 1) return "";
  return Math.round(value);
}

function smoothCurvesEnabled() {
  return $("smoothCurvesToggle")?.checked !== false;
}

function liveSmoothCurvesEnabled() {
  return $("liveSmoothCurvesToggle")?.checked !== false;
}
let paperArchiveSignature = "";
let mergeSourceSignature = "";
let deleteTargetSignature = "";
let latestMergeFigures = [];
const clientX = {0: 8, 2: 18, 4: 28, 6: 38, 8: 48, 1: 52, 3: 62, 5: 72, 7: 82, 9: 92};
const paperFigures = [
  ["fmnist_lenet5_100r_ep3_accuracy.png", "100r / 3 local epochs accuracy"],
  ["fmnist_lenet5_accuracy_convergence_with_random.png", "100r main accuracy convergence"],
  ["fmnist_lenet5_time_accuracy_with_random.png", "100r main logical time vs accuracy"],
  ["fmnist_lenet5_privacy_timeline_with_random.png", "Privacy budget timeline"],
  ["fmnist_lenet5_mode_distribution_with_random.png", "DynFedPrivacy mode distribution"]
];

function esc(value) {
  return String(value ?? "").replace(/[&<>"']/g, ch => ({
    "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"
  }[ch]));
}

function drawChart(rows) {
  const canvas = $("chart"), ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#111518"; ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.strokeStyle = "#2e383d"; ctx.lineWidth = 1;
  for (let i = 0; i < 6; i++) {
    const y = 34 + i * 62;
    ctx.beginPath(); ctx.moveTo(48, y); ctx.lineTo(canvas.width - 22, y); ctx.stroke();
  }
  if (!rows.length) return;
  const xs = rows.map(r => Number(r.round) + 1);
  const acc = rows.map(r => Number(r.test_accuracy));
  const times = rows.map(r => Number(r.logical_time));
  const minX = Math.min(...xs), maxX = Math.max(...xs);
  const maxTime = Math.max(...times, 1);
  const x = v => 48 + (v - minX) / Math.max(maxX - minX, 1) * (canvas.width - 80);
  const yAcc = v => canvas.height - 34 - v * (canvas.height - 70);
  const yTime = v => canvas.height - 34 - (v / maxTime) * (canvas.height - 70);
  line(xs, acc, x, yAcc, "#4fd08a");
  line(xs, times, x, yTime, "#5ca8ff");
  ctx.fillStyle = "#aab4b8"; ctx.font = "13px Segoe UI";
  ctx.fillText("green: accuracy", 54, 22);
  ctx.fillText("blue: logical time", 178, 22);
}

function line(xs, ys, xfn, yfn, color) {
  const ctx = $("chart").getContext("2d");
  ctx.strokeStyle = color; ctx.lineWidth = 3; ctx.beginPath();
  xs.forEach((xv, i) => i ? ctx.lineTo(xfn(xv), yfn(ys[i])) : ctx.moveTo(xfn(xv), yfn(ys[i])));
  ctx.stroke();
}

function parseDistribution(dist) {
  if (!dist) return {};
  if (typeof dist === "object") return dist;
  const parsed = {};
  String(dist).split(";").forEach(item => {
    const match = item.match(/^([^:]+):(\d+)/);
    if (match) parsed[match[1]] = Number(match[2]);
  });
  return parsed;
}

function renderModes(dist) {
  const entries = Object.entries(parseDistribution(dist));
  const total = entries.reduce((s, [, v]) => s + Number(v), 0) || 1;
  $("modes").innerHTML = entries.length ? entries.map(([k, v]) => {
    const p = Number(v) / total * 100;
    return `<div class="mode-row"><div>${k}</div><div class="mode-meter"><div class="mode-fill" style="width:${p}%"></div></div><div>${p.toFixed(1)}%</div></div>`;
  }).join("") : "<div class='sub'>No mode decisions yet</div>";
}

function renderActiveClients(ids, details) {
  const selected = Array.isArray(details) && details.length ? details : (ids || []).map(v => ({
    client_id: v,
    edge_id: Number(v) % 2,
    mode: ""
  }));
  const active = new Set(selected.map(item => String(item.client_id)));
  const activeEdges = new Set(selected.filter(item => String(item.mode || "").includes("E")).map(item => String(item.edge_id)));
  const cloudActive = selected.some(item => String(item.mode || "").includes("C"));
  document.querySelectorAll(".client").forEach(el => {
    el.classList.toggle("active", active.has(el.dataset.client));
  });
  document.querySelectorAll(".edge").forEach(el => {
    el.classList.toggle("active", activeEdges.has(el.dataset.edge));
  });
  $("cloudNode").classList.toggle("active", cloudActive);
  $("activeClients").textContent = active.size ? `Active clients: ${[...active].map(v => "C" + v).join(", ")}` : "Active clients: waiting";
  const modes = [...new Set(selected.map(item => item.mode).filter(Boolean))];
  const edges = [...activeEdges].map(v => "Edge " + v).join(", ") || "no edge";
  $("activePath").textContent = modes.length ? `Active path: ${modes.join(", ")} | ${edges} | ${cloudActive ? "cloud used" : "cloud not used"}` : "Active path: waiting";
  renderPackets(selected);
}

function renderPackets(selected) {
  const viz = document.querySelector(".viz");
  viz.querySelectorAll(".packet").forEach(el => el.remove());
  selected.slice(0, 10).forEach((item, index) => {
    const mode = String(item.mode || "");
    const clientId = Number(item.client_id);
    const edgeId = Number.isFinite(Number(item.edge_id)) ? Number(item.edge_id) : clientId % 2;
    const narrow = window.matchMedia("(max-width: 620px)").matches;
    const startsBottomRow = narrow && [1, 3, 5, 7, 9].includes(clientId);
    const sx = clientX[clientId] ?? (8 + index * 9);
    const sy = startsBottomRow ? 318 : (narrow ? 276 : 242);
    const usesEdge = mode.includes("E");
    const usesCloud = mode.includes("C");
    const edgeX = edgeId === 1 ? 71 : 29;
    const edgeY = 167;
    const cloudX = 50;
    const cloudY = 61;
    const tx = usesCloud ? cloudX : (usesEdge ? edgeX : sx);
    const ty = usesCloud ? cloudY : (usesEdge ? edgeY : sy - 22);
    const mx = usesEdge ? edgeX : (sx + tx) / 2;
    const my = usesEdge ? edgeY : Math.min(sy - 30, ty + 42);
    const packet = document.createElement("div");
    packet.className = usesEdge || usesCloud ? "packet" : "packet local";
    packet.style.setProperty("--sx", `${sx}%`);
    packet.style.setProperty("--sy", `${sy}px`);
    packet.style.setProperty("--mx", `${mx}%`);
    packet.style.setProperty("--my", `${my}px`);
    packet.style.setProperty("--tx", `${tx}%`);
    packet.style.setProperty("--ty", `${ty}px`);
    packet.style.setProperty("--delay", `${(index % 5) * 0.18}s`);
    packet.style.setProperty("--packet-color", edgeId === 1 ? "#5ca8ff" : "#f3c65f");
    packet.textContent = `C${clientId}`;
    packet.title = `C${clientId} -> ${usesCloud ? "Cloud" : (usesEdge ? "Edge " + edgeId : "Local")} (${mode || "pending"})`;
    viz.appendChild(packet);
  });
}

function policyLabel(policy) {
  return {
    ours: "DynFedPrivacy",
    ours_no_omega: "No Error Cost Estimate",
    ours_fixed_liieiiic: "Fixed Mode LIIEIIIC",
    individual_optimal: "Individual-Optimal",
    fixed_dp: "Fixed-DP",
    privacy_only: "Privacy-Only",
    no_protection: "No-Protection",
    random: "Random",
    fixed_fedavg: "Fixed FedAvg",
    fixed_splitfed: "Fixed SplitFed",
    fixed_hfl: "Fixed HFL",
    nsga2: "NSGA II",
    fixed_liieiiic: "Fixed-HFL (LIIEIIIC)",
    performance_only: "Global-Balance Upper",
    best_accuracy: "Global-Utility Upper",
    accuracy_oracle: "Accuracy-Oracle Upper"
  }[policy] || policy || "-";
}

function renderPolicyRuns(items) {
  const target = $("policyRuns");
  if (!Array.isArray(items) || !items.length) {
    target.innerHTML = "<div class='sub'>No policy runs found yet</div>";
    return;
  }
  const rows = items.map(item => {
    const progress = Math.max(0, Math.min(100, Number(item.progress || 0) * 100));
    const status = item.status || "pending";
    const perC = item.per_client_test_mean != null
      ? `${fmt(item.per_client_test_mean, 4)}±${fmt(item.per_client_test_std, 4)}`
      : "-";
    return `<tr>
      <td>${policyLabel(item.policy)}</td>
      <td><span class="status-tag ${status}">${status}</span></td>
      <td>${item.round || 0}/${item.rounds || "-"}</td>
      <td>${progress.toFixed(0)}%</td>
      <td>${fmt(item.test_accuracy, 4)}</td>
      <td>${fmt(item.best_test_accuracy, 4)}</td>
      <td>${fmt(item.logical_time, 1)}</td>
      <td>${fmt(item.cumulative_communication_volume, 1)}</td>
      <td>${fmt(item.larger_channel_epsilon, 3)}</td>
      <td>${fmt(item.mean_global_update_clients, 2)}</td>
      <td>${perC}</td>
    </tr>`;
  }).join("");
  target.innerHTML = `<div class="policy-table-wrap">
    <table class="policy-table">
      <thead><tr><th>Method</th><th>Status</th><th>Round</th><th>Progress</th><th>Acc</th><th>Best</th><th>Time</th><th>Comm</th><th>Larger channel eps</th><th>Global Clients</th><th>PerC Acc</th></tr></thead>
      <tbody>${rows}</tbody>
    </table>
  </div>`;
}

function renderLiveSvg(items) {
  renderSingleLiveSvg($("liveRoundSvg"), items, "round");
  renderSingleLiveSvg($("liveTimeSvg"), items, "time");
}

function renderModeDistributions(items) {
  const target = $("modeGrid");
  if (!Array.isArray(items) || !items.length) {
    target.innerHTML = "<div class='sub'>No mode distribution data yet</div>";
    return;
  }
  const cards = items.map((item, index) => {
    const dist = parseDistribution(item.mode_distribution);
    const entries = Object.entries(dist);
    const total = entries.reduce((sum, [, value]) => sum + Number(value), 0) || 1;
    const bars = entries.length ? entries
      .sort((a, b) => Number(b[1]) - Number(a[1]))
      .map(([mode, value]) => {
        const pct = Number(value) / total * 100;
        return `<div class="mode-bar-row">
          <div>${mode}</div>
          <div class="mode-bar-track"><div class="mode-bar-fill" style="width:${pct}%;"></div></div>
          <div>${pct.toFixed(1)}%</div>
        </div>`;
      }).join("") : "<div class='sub'>Pending</div>";
    return `<div class="mode-card">
      <div class="mode-title"><span>${policyLabel(item.policy)}</span><span class="status-tag ${item.status || "pending"}">${item.status || "pending"}</span></div>
      ${bars}
    </div>`;
  }).join("");
  target.innerHTML = cards;
}

function renderSingleLiveSvg(svg, items, axis) {
  const colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf"];
  const smooth = liveSmoothCurvesEnabled();
  let series = (items || []).map((item, idx) => ({
    policy: item.policy,
    label: policyLabel(item.policy),
    color: colors[idx % colors.length],
    points: smooth ? smoothCurve(item.curve || [], 9) : rawCurve(item.curve || [])
  })).filter(s => s.points.length);
  let timeClip = null;
  if (axis === "time" && series.length) {
    const maxTimes = series.map(s => Math.max(...s.points.map(p => Number(p.logical_time)).filter(Number.isFinite))).filter(Number.isFinite);
    if (maxTimes.length) {
      timeClip = Math.min(...maxTimes);
      series = series.map(s => ({...s, points: s.points.filter(p => Number(p.logical_time) <= timeClip)})).filter(s => s.points.length);
    }
  }
  const w = 760, h = 340, ml = 58, mr = 18, mt = 24, mb = 48;
  const xValues = series.flatMap(s => s.points.map(p => axis === "time" ? Number(p.logical_time) : Number(p.round)));
  const yValues = series.flatMap(s => s.points.flatMap(p => [Number(p.test_accuracy), Number(p.lower), Number(p.upper)]));
  const xmax = Math.max(...xValues, 1);
  const xmin = Math.min(...xValues, 0);
  const ymax = Math.min(1, Math.max(...yValues, 0.9));
  const ymin = 0;
  const xScale = x => ml + (Number(x) - xmin) / Math.max(xmax - xmin, 1e-9) * (w - ml - mr);
  const yScale = y => h - mb - (Number(y) - ymin) / Math.max(ymax - ymin, 1e-9) * (h - mt - mb);
  const grid = [0, .2, .4, .6, .8, 1.0].map(v => {
    const y = yScale(v);
    return `<line x1="${ml}" y1="${y}" x2="${w - mr}" y2="${y}" stroke="#e5e7eb" stroke-width="1"/><text x="${ml - 10}" y="${y + 4}" text-anchor="end" font-size="11" fill="#555">${v.toFixed(1)}</text>`;
  }).join("");
  const paths = series.map(s => {
    const d = s.points.map((p, i) => {
      const xVal = axis === "time" ? p.logical_time : p.round;
      const cmd = i === 0 ? "M" : "L";
      return `${cmd}${xScale(xVal).toFixed(1)},${yScale(p.test_accuracy).toFixed(1)}`;
    }).join(" ");
    const last = s.points[s.points.length - 1];
    const lx = xScale(axis === "time" ? last.logical_time : last.round);
    const ly = yScale(last.test_accuracy);
    let band = "";
    if (smooth) {
      const upper = s.points.map((p, i) => {
        const xVal = axis === "time" ? p.logical_time : p.round;
        return `${i === 0 ? "M" : "L"}${xScale(xVal).toFixed(1)},${yScale(p.upper).toFixed(1)}`;
      }).join(" ");
      const lower = s.points.slice().reverse().map(p => {
        const xVal = axis === "time" ? p.logical_time : p.round;
        return `L${xScale(xVal).toFixed(1)},${yScale(p.lower).toFixed(1)}`;
      }).join(" ");
      band = `<path d="${upper} ${lower} Z" fill="${s.color}" opacity="0.14" stroke="none"/>`;
    }
    return `${band}<path d="${d}" fill="none" stroke="${s.color}" stroke-width="2.4"/><circle cx="${lx}" cy="${ly}" r="3.5" fill="${s.color}"/>`;
  }).join("");
  const legend = series.map((s, i) => {
    const x = ml + (i % 3) * 180;
    const y = h - 20 + Math.floor(i / 3) * 16;
    return `<rect x="${x}" y="${y - 9}" width="10" height="10" fill="${s.color}"/><text x="${x + 16}" y="${y}" font-size="12" fill="#333">${s.label}</text>`;
  }).join("");
  const xlabel = axis === "time" ? `Logical time (clipped at ${fmt(timeClip, 1)}s)` : "Round";
  const titleSuffix = smooth ? "smoothed" : "raw";
  const title = axis === "time" ? `Time-Accuracy (${titleSuffix})` : `Round-Accuracy (${titleSuffix})`;
  const note = smooth ? "Line: EMA, span=9; band: EMA residual std" : "Line: raw unsmoothed accuracy";
  svg.innerHTML = `
    <rect width="${w}" height="${h}" fill="#fff"/>
    <text x="${w / 2}" y="16" text-anchor="middle" font-size="14" font-weight="700" fill="#222">${title}</text>
    ${grid}
    <line x1="${ml}" y1="${h - mb}" x2="${w - mr}" y2="${h - mb}" stroke="#333"/>
    <line x1="${ml}" y1="${mt}" x2="${ml}" y2="${h - mb}" stroke="#333"/>
    <text x="${(w + ml - mr) / 2}" y="${h - 10}" text-anchor="middle" font-size="13" fill="#333">${xlabel}</text>
    <text x="18" y="${(h - mt - mb) / 2 + mt}" text-anchor="middle" font-size="13" fill="#333" transform="rotate(-90 18 ${(h - mt - mb) / 2 + mt})">Test accuracy</text>
    <text x="${ml}" y="${h - 32}" font-size="11" fill="#666">${note}</text>
    ${paths || `<text x="${w / 2}" y="${h / 2}" text-anchor="middle" font-size="14" fill="#777">No live curve data yet</text>`}
    ${legend}
  `;
}

function rawCurve(points) {
  if (!Array.isArray(points) || !points.length) return [];
  return points.map(point => {
    const value = Number(point.test_accuracy);
    return {
      ...point,
      test_accuracy: value,
      lower: value,
      upper: value
    };
  });
}

function smoothCurve(points, window=9) {
  if (!Array.isArray(points) || !points.length) return [];
  const span = Math.max(1, Number(window) || 1);
  const alpha = 2 / (span + 1);
  let mean = Number(points[0].test_accuracy);
  if (!Number.isFinite(mean)) mean = 0;
  let variance = 0;
  return points.map((point, index) => {
    const value = Number(point.test_accuracy);
    if (index > 0 && Number.isFinite(value)) {
      const previousMean = mean;
      mean = alpha * value + (1 - alpha) * previousMean;
      const residual = value - previousMean;
      variance = alpha * Math.pow(residual, 2) + (1 - alpha) * variance;
    }
    const std = Math.sqrt(Math.max(0, variance));
    return {
      ...point,
      test_accuracy: mean,
      lower: Math.max(0, mean - std),
      upper: Math.min(1, mean + std)
    };
  });
}

function renderFigures() {
  if (!$("figureGrid")) return;
  const stamp = Date.now();
  const axis = $("figureAxis") ? $("figureAxis").value : "round";
  const filtered = axis === "time"
    ? paperFigures.filter(([name]) => name.includes("time_accuracy"))
    : paperFigures.filter(([name]) => name.includes("accuracy_convergence") || name.includes("100r_ep3_accuracy"));
  $("figureGrid").innerHTML = filtered.map(([name, label]) => `
    <div class="figure-card">
      <img src="/figures/${name}?t=${stamp}" alt="${label}" loading="lazy">
      <div class="figure-caption">${label}<br>${name}</div>
    </div>
  `).join("");
}

function renderRunFigures(items) {
  if (!$("runFigureGrid")) return;
  const signature = JSON.stringify((items || []).map(item => [item.name, item.mtime]));
  if (signature === runFigureSignature) return;
  runFigureSignature = signature;
  if (!Array.isArray(items) || !items.length) {
    $("runFigureGrid").innerHTML = "<div class='sub'>No current-run figures yet. Build them after a run has produced round_metrics.csv.</div>";
    return;
  }
  $("runFigureGrid").innerHTML = items.map(item => `
    <div class="figure-card">
      <img src="/run-figures/${item.name}" alt="${item.label}" loading="lazy">
      <div class="figure-caption">${item.label}<br>${item.name}<a href="/run-figures/${item.name}" target="_blank">Open Large</a></div>
    </div>
  `).join("");
}

function renderPaperArchive(items) {
  const target = $("paperArchiveGrid");
  if (!target) return;
  const signature = JSON.stringify((items || []).map(item => [item.name, item.mtime, (item.figures || []).map(fig => fig.name).join(",")]));
  if (signature === paperArchiveSignature) return;
  paperArchiveSignature = signature;
  if (!Array.isArray(items) || !items.length) {
    target.innerHTML = "<div class='sub'>No saved paper experiments yet</div>";
    return;
  }
  target.innerHTML = items.map(item => {
    const figures = (item.figures || []).map(fig => {
      const src = `/paper-experiments/${encodeURIComponent(item.name)}/figures/${encodeURIComponent(fig.name)}`;
      return `<div class="figure-card">
        <img src="${src}" alt="${fig.label}" loading="lazy">
        <div class="figure-caption">${item.name}<br>${fig.name}<a href="${src}" target="_blank">Open Large</a></div>
      </div>`;
    }).join("");
    return figures || `<div class="figure-card"><div class="figure-caption">${item.name}<br>No figures saved</div></div>`;
  }).join("");
}

function renderMergeSources(items) {
  const target = $("mergeSourceList");
  if (!target) return;
  const signature = JSON.stringify((items || []).map(item => [
    item.id, item.policy, item.run_name, item.mtime, item.final_test_accuracy, item.best_test_accuracy, item.per_client_test_mean
  ]));
  if (signature === mergeSourceSignature) return;
  mergeSourceSignature = signature;
  if (!Array.isArray(items) || !items.length) {
    target.innerHTML = "<div class='sub'>No completed method outputs found under out/.</div>";
    return;
  }
  target.innerHTML = items.map(item => {
    const perC = item.per_client_test_mean != null
      ? `${fmt(item.per_client_test_mean, 4)}±${fmt(item.per_client_test_std, 4)}`
      : "-";
    return `<label class="merge-item" title="${esc(item.run_dir)}">
      <input type="checkbox" value="${esc(item.id)}">
      <strong>${esc(policyLabel(item.policy))}${item.param_suffix ? " " + esc(item.param_suffix) : ""}</strong>
      <span class="run-name">${esc(item.run_name)}</span>
      <span>Final ${fmt(item.final_test_accuracy, 4)}</span>
      <span>Best ${fmt(item.best_test_accuracy, 4)}</span>
      <span>PerC ${perC}</span>
    </label>`;
  }).join("");
}

function renderMergeFigures(figures, mergedRuns=[]) {
  const target = $("mergeFigureGrid");
  if (!target) return;
  let items = [];
  if (Array.isArray(figures) && figures.length) {
    items = figures.map(path => ({path, run: "Merged run"}));
  } else if (Array.isArray(mergedRuns) && mergedRuns.length) {
    const latest = mergedRuns[0];
    items = (latest.figures || []).map(fig => ({path: fig.path, run: latest.name}));
  }
  if (!items.length) {
    target.innerHTML = "<div class='sub'>No merged figures yet. Select method outputs and click Merge Selected Methods.</div>";
    return;
  }
  target.innerHTML = items.map(item => {
    const normalized = String(item.path).replaceAll("\\", "/");
    const marker = "/out/merged_runs/";
    const idx = normalized.indexOf(marker);
    const rel = idx >= 0 ? normalized.slice(idx + marker.length) : normalized.split("/merged_runs/").pop();
    const src = `/merged-runs/${rel.split("/").map(encodeURIComponent).join("/")}`;
    const name = rel.split("/").pop();
    return `<div class="figure-card">
      <img src="${src}" alt="${esc(name)}" loading="lazy">
      <div class="figure-caption">${esc(item.run)}<br>${esc(name)}<a href="${src}" target="_blank">Open Large</a></div>
    </div>`;
  }).join("");
}

function renderDeleteTargets(items) {
  const target = $("deleteTargetList");
  if (!target) return;
  const signature = JSON.stringify((items || []).map(item => [item.id, item.kind, item.mtime]));
  if (signature === deleteTargetSignature) return;
  deleteTargetSignature = signature;
  if (!Array.isArray(items) || !items.length) {
    target.innerHTML = "<div class='sub'>No deletable output directories found under out/.</div>";
    return;
  }
  target.innerHTML = items.map(item => `
    <label class="delete-item" title="${esc(item.path)}">
      <input type="checkbox" value="${esc(item.id)}">
      <strong>${esc(item.kind)}</strong>
      <span class="dir-name">${esc(item.id)}</span>
    </label>
  `).join("");
}

async function deleteDirs() {
  const btn = $("deleteDirsBtn");
  const selected = Array.from(document.querySelectorAll("#deleteTargetList input:checked")).map(input => input.value);
  if (!selected.length) {
    $("deleteStatus").textContent = "Select at least one directory first.";
    return;
  }
  const ok = confirm(`Delete ${selected.length} selected output director${selected.length === 1 ? "y" : "ies"}? This cannot be undone.`);
  if (!ok) return;
  btn.disabled = true;
  const oldText = btn.textContent;
  btn.textContent = "Deleting...";
  try {
    const res = await fetch("/delete-dirs", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({target_ids: selected})
    });
    const result = await res.json();
    $("deleteStatus").textContent = result.message || "Delete request finished";
    $("controlStatus").textContent = result.message || "Delete request finished";
    deleteTargetSignature = "";
    mergeSourceSignature = "";
    await refresh();
  } catch (err) {
    $("deleteStatus").textContent = String(err);
  } finally {
    btn.textContent = oldText;
    btn.disabled = latestStatus.status === "running" || Boolean(latestStatus.training_process_running);
  }
}

async function mergeRuns() {
  const btn = $("mergeRunsBtn");
  const selected = Array.from(document.querySelectorAll("#mergeSourceList input:checked")).map(input => input.value);
  if (!selected.length) {
    $("mergeStatus").textContent = "Select at least one method output first.";
    return;
  }
  btn.disabled = true;
  const oldText = btn.textContent;
  btn.textContent = "Merging...";
  try {
    const res = await fetch("/merge-runs", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        source_ids: selected,
        name: $("mergeNameInput").value || "paper_combined",
        tail_start_round: tailZoomStartRound(),
        smooth_curves: smoothCurvesEnabled()
      })
    });
    const result = await res.json();
    $("mergeStatus").textContent = result.message || "Merge finished";
    $("controlStatus").textContent = result.message || "Merge finished";
    latestMergeFigures = result.figures || [];
    renderMergeFigures(latestMergeFigures);
    await refresh();
  } catch (err) {
    $("mergeStatus").textContent = String(err);
  } finally {
    btn.textContent = oldText;
    btn.disabled = latestStatus.status === "running" || Boolean(latestStatus.training_process_running);
  }
}

async function buildRunFigures() {
  const btn = $("buildRunFiguresBtn");
  btn.disabled = true;
  const oldText = btn.textContent;
  btn.textContent = "Building...";
  try {
    const res = await fetch("/build-run-figures", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        tail_start_round: tailZoomStartRound(),
        smooth_curves: smoothCurvesEnabled()
      })
    });
    const result = await res.json();
    $("controlStatus").textContent = result.message || "Build current run figures finished";
    await refresh();
  } catch (err) {
    $("controlStatus").textContent = String(err);
  } finally {
    btn.textContent = oldText;
    btn.disabled = latestStatus.status === "running" || Boolean(latestStatus.training_process_running);
  }
}

function setControlsDisabled(disabled) {
  document.querySelectorAll("[data-start-preset], #startBtn, #runConfiguredBtn, #runAlgorithmAblationBtn, #runPeriodSweepBtn, #runPrivacySweepBtn, #runHetPeriodSweepBtn, .ablation-run, .robust-seed-run, #runRobustLatencyBtn, #runRobustStrongHetBtn, #mergeRunsBtn, #deleteDirsBtn, #savePaperRunBtn, #buildRunFiguresBtn, #rebuildBtn").forEach(btn => {
    btn.disabled = disabled;
  });
  document.querySelectorAll("#stopRunStatusBtn").forEach(btn => {
    btn.disabled = false;
  });
}

async function savePaperRun() {
  const btn = $("savePaperRunBtn");
  btn.disabled = true;
  const oldText = btn.textContent;
  btn.textContent = "Saving...";
  try {
    const res = await fetch("/save-paper-run", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        tail_start_round: tailZoomStartRound(),
        smooth_curves: smoothCurvesEnabled()
      })
    });
    const result = await res.json();
    $("paperArchiveStatus").textContent = result.message || "Save request finished";
    $("controlStatus").textContent = result.message || "Save request finished";
    await refresh();
  } catch (err) {
    $("paperArchiveStatus").textContent = String(err);
  } finally {
    btn.textContent = oldText;
    btn.disabled = latestStatus.status === "running" || Boolean(latestStatus.training_process_running);
  }
}


async function refresh() {
  if (refreshInFlight) return refreshInFlight;
  refreshInFlight = refreshOnce();
  try {
    return await refreshInFlight;
  } finally {
    refreshInFlight = null;
  }
}

async function refreshOnce() {
  const res = await fetch("/status", {cache: "no-store"});
  if (!res.ok) throw new Error(`Status request failed: HTTP ${res.status}`);
  const s = await res.json();
  latestStatus = s;
  const progress = Math.max(0, Math.min(100, Number(s.progress || 0) * 100));
  $("progress").style.width = progress + "%";
  $("headline").textContent = `${s.status || "unknown"} ${progress.toFixed(1)}%`;
  $("message").textContent = s.message || "";
  $("updated").textContent = s.updated_at || "waiting";
  $("path").textContent = s.status_path || s.output_root || "";
  $("policy").textContent = s.active_policy || s.policy || "-";
  $("round").textContent = s.rounds ? `${s.round || 0}/${s.rounds}` : "-";
  $("acc").textContent = fmt(s.test_accuracy, 4);
  $("best").textContent = fmt(s.best_test_accuracy, 4);
  $("logical").textContent = `${fmt(s.logical_time, 1)}s`;
  $("eps").textContent = fmt(s.larger_channel_epsilon ?? s.epsilon_used, 3);
  $("featureEps").textContent = fmt(s.feature_epsilon, 3);
  $("updateEps").textContent = fmt(s.update_epsilon, 3);
  $("clients").textContent = s.effective_clients ?? "-";
  $("comm").textContent = fmt(s.cumulative_communication_volume, 1);
  const activePol = s.active_policy || s.policy;
  const pcStatus = (s.policy_statuses || []).find(p => p.policy === activePol);
  const pcMean = pcStatus?.per_client_test_mean;
  const pcStd = pcStatus?.per_client_test_std;
  $("perClientAcc").textContent = pcMean != null ? `${fmt(pcMean, 4)} ±${fmt(pcStd, 4)}` : "-";
  setControlsDisabled(s.status === "running" || Boolean(s.training_process_running));
  const rows = s.recent_rounds || [];
  renderActiveClients(s.selected_client_ids, s.selected_clients);
  renderPolicyRuns(s.policy_statuses);
  renderLiveSvg(s.policy_statuses);
  renderModeDistributions(s.policy_statuses);
  renderRunFigures(s.run_figures);
  renderMergeSources(s.merge_sources);
  renderMergeFigures(latestMergeFigures, s.merged_runs);
  renderDeleteTargets(s.delete_targets);
  renderPaperArchive(s.paper_experiments);
  $("roundRows").innerHTML = rows.slice().reverse().map(r =>
    `<tr><td>${Number(r.round) + 1}</td><td>${fmt(r.test_accuracy, 4)}</td><td>${fmt(r.best_accuracy, 4)}</td><td>${fmt(r.logical_time, 1)}</td><td>${r.num_effective_clients ?? "-"}</td></tr>`
  ).join("");
}

async function pollStatus() {
  try {
    await refresh();
  } catch (err) {
    $("message").textContent = `Monitor connection error: ${String(err)}`;
  }
  window.setTimeout(pollStatus, document.hidden ? 10000 : 2500);
}

function selectedMethods() {
  return Array.from(document.querySelectorAll("#methodList input:checked")).map(input => input.value);
}

function selectedAblationMethods() {
  return Array.from(document.querySelectorAll("#ablationMethodList input:checked")).map(input => input.value);
}

function selectedRobustMethods() {
  return Array.from(document.querySelectorAll("#robustMethodList input:checked")).map(input => input.value);
}

function currentPrivacyParams() {
  return {
    privacy_schema: "rdp_total_v1",
    initial_epsilon: $("initialEpsilonInput")?.value || "8.0",
    dp_emb_epsilon: $("dpEmbEpsilonInput")?.value || "8.0",
    dp_upd_epsilon: $("dpUpdEpsilonInput")?.value || "8.0",
    dp_profile: $("dpProfileSelect")?.value || "cifar_resnet"
  };
}

function currentExecutorParams() {
  return {
    executor: $("executorSelect")?.value || "serial",
    executor_workers: $("executorWorkersInput")?.value.trim() || ""
  };
}

function currentConstraintParams() {
  return {
    train_limit: $("trainLimitInput")?.value || "12000",
    test_limit: $("testLimitInput")?.value || "2000",
    resource_limit: $("resourceLimitInput")?.value || "1.35",
    risk_limit: $("riskLimitInput")?.value || "0.5",
    min_edge_cloud_fusion_ratio: $("minEdgeCloudFusionInput")?.value || "0.5",
    he_backend: $("heBackendSelect")?.value || "seal",
    require_real_he: $("requireRealHeInput")?.value || "true"
  };
}

function mainExperimentParams(overrides={}) {
  const rounds = Math.max(1, Math.min(500, Number($("roundsInput").value || 200)));
  const clients = Math.max(1, Math.min(200, Number($("clientsInput").value || 100)));
  const edges = Math.max(1, Math.min(20, Number($("edgesInput").value || 10)));
  const seed = Math.max(0, Math.min(999999, Number($("seedInput").value || 42)));
  const timeLimit = Math.max(0.1, Math.min(300, Number($("timeLimitInput").value || 300)));
  const trainLimit = Math.max(1, Math.min(60000, Number($("trainLimitInput").value || 12000)));
  const testLimit = Math.max(1, Math.min(10000, Number($("testLimitInput").value || 2000)));
  const clientHet = Math.max(1.0, Math.min(10.0, Number($("clientHetInput").value || 2.0)));
  const edgeHet = Math.max(1.0, Math.min(10.0, Number($("edgeHetInput").value || 1.5)));
  const selectionPeriod = Math.max(1, Math.min(100, Number($("selectionPeriodInput").value || 1)));
  const aggregationFraction = Math.max(0.1, Math.min(1.0, Number($("aggregationFractionInput").value || 1.0)));
  const paretoArchiveSize = Math.max(2, Math.min(128, Number($("paretoArchiveSizeInput").value || 16)));
  const paretoMaxIters = Math.max(0, Math.min(200, Number($("paretoMaxItersInput").value || 50)));
  const cloudFusionXi = Math.max(0.0, Math.min(100.0, Number($("cloudFusionXiInput").value || 0.2)));
  const cloudFusionEps = Math.max(0.000001, Math.min(10.0, Number($("cloudFusionEpsInput").value || 0.05)));
  const minEdgeCloudFusion = Math.max(0.0, Math.min(1.0, Number($("minEdgeCloudFusionInput").value || 0.5)));
  const resourceLimit = Math.max(0.0, Math.min(100.0, Number($("resourceLimitInput").value || 1.35)));
  const riskLimit = Math.max(0.0, Math.min(1.0, Number($("riskLimitInput").value || 0.5)));
  const initialEpsilon = Math.max(0.0, Math.min(100.0, Number($("initialEpsilonInput").value || 8.0)));
  const dpEmbEpsilon = Math.max(0.001, Math.min(100.0, Number($("dpEmbEpsilonInput").value || 8.0)));
  const dpUpdEpsilon = Math.max(0.001, Math.min(100.0, Number($("dpUpdEpsilonInput").value || 8.0)));
  let localEpochs = "";
  const localEpochsRaw = $("localEpochsInput")?.value.trim() || "";
  if (localEpochsRaw) {
    localEpochs = String(Math.max(1, Math.min(20, Number(localEpochsRaw) || 1)));
  }
  let learningRate = "";
  const learningRateRaw = $("learningRateInput")?.value.trim() || "";
  if (learningRateRaw) {
    learningRate = String(Math.max(0.000001, Math.min(1.0, Number(learningRateRaw) || 0.000001)));
  }
  let executorWorkers = "";
  const executorWorkersRaw = $("executorWorkersInput")?.value.trim() || "";
  if (executorWorkersRaw) {
    executorWorkers = String(Math.max(1, Math.min(64, Number(executorWorkersRaw) || 1)));
  }
  $("roundsInput").value = String(rounds);
  $("clientsInput").value = String(clients);
  $("edgesInput").value = String(edges);
  $("seedInput").value = String(seed);
  $("timeLimitInput").value = String(timeLimit);
  $("trainLimitInput").value = String(trainLimit);
  $("testLimitInput").value = String(testLimit);
  $("clientHetInput").value = String(clientHet);
  $("edgeHetInput").value = String(edgeHet);
  $("selectionPeriodInput").value = String(selectionPeriod);
  $("aggregationFractionInput").value = String(aggregationFraction);
  $("paretoArchiveSizeInput").value = String(paretoArchiveSize);
  $("paretoMaxItersInput").value = String(paretoMaxIters);
  $("cloudFusionXiInput").value = String(cloudFusionXi);
  $("cloudFusionEpsInput").value = String(cloudFusionEps);
  $("minEdgeCloudFusionInput").value = String(minEdgeCloudFusion);
  $("resourceLimitInput").value = String(resourceLimit);
  $("riskLimitInput").value = String(riskLimit);
  $("initialEpsilonInput").value = String(initialEpsilon);
  $("dpEmbEpsilonInput").value = String(dpEmbEpsilon);
  $("dpUpdEpsilonInput").value = String(dpUpdEpsilon);
  if ($("localEpochsInput") && localEpochs) $("localEpochsInput").value = localEpochs;
  if ($("learningRateInput") && learningRate) $("learningRateInput").value = learningRate;
  if ($("executorWorkersInput") && executorWorkers) $("executorWorkersInput").value = executorWorkers;
  return {
    mode: $("runMode").value,
    rounds: String(rounds),
    time_limit: String(timeLimit),
    train_limit: String(trainLimit),
    test_limit: String(testLimit),
    dataset: $("datasetSelect").value,
    model: $("modelSelect").value,
    device: $("deviceSelect").value,
    executor: $("executorSelect")?.value || "serial",
    executor_workers: executorWorkers,
    figure_axis: $("figureAxis").value,
    partition_mode: $("partitionMode").value,
    clients: String(clients),
    edges: String(edges),
    seed: String(seed),
    client_heterogeneity: String(clientHet),
    edge_heterogeneity: String(edgeHet),
    selection_period: String(selectionPeriod),
    aggregation_fraction: String(aggregationFraction),
    pareto_archive_size: String(paretoArchiveSize),
    pareto_max_iters: String(paretoMaxIters),
    pareto_neighbor_top_k: "0",
    pareto_conflict_only: "0",
    cloud_fusion_xi: String(cloudFusionXi),
    cloud_fusion_eps: String(cloudFusionEps),
    min_edge_cloud_fusion_ratio: String(minEdgeCloudFusion),
    resource_limit: String(resourceLimit),
    risk_limit: String(riskLimit),
    local_epochs: localEpochs,
    learning_rate: learningRate,
    initial_epsilon: String(initialEpsilon),
    dp_emb_epsilon: String(dpEmbEpsilon),
    dp_upd_epsilon: String(dpUpdEpsilon),
    he_backend: $("heBackendSelect")?.value || "seal",
    require_real_he: $("requireRealHeInput")?.value || "true",
    ...currentPrivacyParams(),
    resume_from_run: $("resumeFromRunInput")?.value.trim() || "",
    ...overrides
  };
}

async function startTrainingRequest(preset, params={}) {
  const query = new URLSearchParams({preset});
  Object.entries(params).forEach(([key, value]) => query.set(key, value));
  const res = await fetch(`/start?${query.toString()}`, {method: "POST"});
  return await res.json();
}

async function startTraining(preset, btn, params={}) {
  btn.disabled = true;
  const oldText = btn.textContent;
  btn.textContent = "Starting...";
  try {
    const result = await startTrainingRequest(preset, params);
    $("message").textContent = result.message || "Start request finished";
    $("controlStatus").textContent = result.message || "Start request finished";
    await refresh();
  } catch (err) {
    $("message").textContent = String(err);
    $("controlStatus").textContent = String(err);
  } finally {
    btn.textContent = oldText;
    setControlsDisabled(latestStatus.status === "running" || Boolean(latestStatus.training_process_running));
  }
}

async function waitForTrainingToFinish(startResult=null) {
  let sawProcess = Boolean(startResult && startResult.pid);
  for (;;) {
    await new Promise(resolve => setTimeout(resolve, 2500));
    await refresh();
    if (latestStatus.training_process_running) {
      sawProcess = true;
      continue;
    }
    if (sawProcess && latestStatus.status !== "running") return latestStatus;
    if (!sawProcess && latestStatus.status !== "waiting" && latestStatus.status !== "loading") return latestStatus;
  }
}

async function runPeriodSweep() {
  const btn = $("runPeriodSweepBtn");
  const oldText = btn.textContent;
  const periods = $("sweepPeriodsInput").value
    .split(",")
    .map(value => Number(value.trim()))
    .filter(value => Number.isFinite(value) && value >= 1 && value <= 100)
    .map(value => Math.round(value));
  const uniquePeriods = Array.from(new Set(periods));
  if (!uniquePeriods.length) {
    $("periodSweepStatus").textContent = "Enter at least one valid period, for example 1,5,10,15,20.";
    return;
  }
  btn.disabled = true;
  btn.textContent = "Running Sweep...";
  try {
    for (let index = 0; index < uniquePeriods.length; index += 1) {
      const period = uniquePeriods[index];
      $("periodSweepStatus").textContent = `Starting period ${period} (${index + 1}/${uniquePeriods.length})`;
      const result = await startTrainingRequest("configured", mainExperimentParams({
        policies: $("sweepMethodSelect").value,
        selection_period: String(period),
        reuse_completed: "true"
      }));
      $("controlStatus").textContent = result.message || `Period ${period} start request finished`;
      $("message").textContent = result.message || `Period ${period} start request finished`;
      await refresh();
      if (!result.ok) {
        $("periodSweepStatus").textContent = `Stopped sweep at period ${period}: ${result.message || "start failed"}`;
        break;
      }
      if (!result.reused) {
        $("periodSweepStatus").textContent = `Running period ${period} (${index + 1}/${uniquePeriods.length})`;
        const finalStatus = await waitForTrainingToFinish(result);
        if (finalStatus.status === "failed") {
          $("periodSweepStatus").textContent = `Stopped sweep at period ${period}: ${finalStatus.message || "training failed"}`;
          break;
        }
      }
      $("periodSweepStatus").textContent = `Finished period ${period} (${index + 1}/${uniquePeriods.length})`;
    }
    $("periodSweepStatus").textContent += ". Use Merge Runs to combine the completed period runs.";
  } catch (err) {
    $("periodSweepStatus").textContent = String(err);
  } finally {
    btn.textContent = oldText;
    setControlsDisabled(latestStatus.status === "running" || Boolean(latestStatus.training_process_running));
  }
}

function parsePrivacyBudgetList() {
  return Array.from(new Set($("privacyBudgetsInput").value
    .split(",")
    .map(value => Number(value.trim()))
    .filter(value => Number.isFinite(value) && value >= 0.001 && value <= 100)))
    .sort((left, right) => left - right);
}

async function runPrivacyBudgetSweep() {
  const btn = $("runPrivacySweepBtn");
  const oldText = btn.textContent;
  const budgets = parsePrivacyBudgetList();
  if (!budgets.length) {
    $("privacySweepStatus").textContent = "Enter at least one valid epsilon target, for example 1,2,4,8.";
    return;
  }
  btn.disabled = true;
  btn.textContent = "Running Sweep...";
  try {
    for (let index = 0; index < budgets.length; index += 1) {
      const budget = budgets[index];
      $("privacySweepStatus").textContent = `Starting epsilon ${budget} (${index + 1}/${budgets.length})`;
      const result = await startTrainingRequest("configured", mainExperimentParams({
        policies: $("privacySweepMethodSelect").value,
        initial_epsilon: String(budget),
        dp_emb_epsilon: String(budget),
        dp_upd_epsilon: String(budget),
        reuse_completed: "true"
      }));
      $("controlStatus").textContent = result.message || `Epsilon ${budget} start request finished`;
      $("message").textContent = result.message || `Epsilon ${budget} start request finished`;
      await refresh();
      if (!result.ok) {
        $("privacySweepStatus").textContent = `Stopped sweep at epsilon ${budget}: ${result.message || "start failed"}`;
        break;
      }
      if (!result.reused) {
        $("privacySweepStatus").textContent = `Running epsilon ${budget} (${index + 1}/${budgets.length})`;
        const finalStatus = await waitForTrainingToFinish(result);
        if (finalStatus.status === "failed") {
          $("privacySweepStatus").textContent = `Stopped sweep at epsilon ${budget}: ${finalStatus.message || "training failed"}`;
          break;
        }
      }
      $("privacySweepStatus").textContent = `Finished epsilon ${budget} (${index + 1}/${budgets.length})`;
    }
    $("privacySweepStatus").textContent += ". Completed runs are ready for privacy sensitivity aggregation.";
  } catch (err) {
    $("privacySweepStatus").textContent = String(err);
  } finally {
    btn.textContent = oldText;
    setControlsDisabled(latestStatus.status === "running" || Boolean(latestStatus.training_process_running));
  }
}

async function runAlgorithmAblation() {
  const btn = $("runAlgorithmAblationBtn");
  const oldText = btn.textContent;
  const variants = [
    {policy: "ours", label: "Full Method"},
    {policy: "individual_optimal", label: "Without Global Coordination"},
    {policy: "ours_no_omega", label: "No Error Cost Estimate"},
    {policy: "ours_fixed_liieiiic", label: "Fixed Mode LIIEIIIC"}
  ];
  btn.disabled = true;
  btn.textContent = "Running Ablation...";
  try {
    for (let index = 0; index < variants.length; index += 1) {
      const variant = variants[index];
      $("algorithmAblationStatus").textContent = `Starting ${variant.label} (${index + 1}/${variants.length})`;
      const result = await startTrainingRequest("configured", mainExperimentParams({
        policies: variant.policy,
        reuse_completed: "true"
      }));
      $("controlStatus").textContent = result.message || `${variant.label} start request finished`;
      $("message").textContent = result.message || `${variant.label} start request finished`;
      await refresh();
      if (!result.ok) {
        $("algorithmAblationStatus").textContent = `Stopped at ${variant.label}. ${result.message || "Start failed"}`;
        return;
      }
      if (!result.reused) {
        $("algorithmAblationStatus").textContent = `Running ${variant.label} (${index + 1}/${variants.length})`;
        const finalStatus = await waitForTrainingToFinish(result);
        if (finalStatus.status === "failed") {
          $("algorithmAblationStatus").textContent = `Stopped at ${variant.label}. ${finalStatus.message || "Training failed"}`;
          return;
        }
      }
      $("algorithmAblationStatus").textContent = `Finished ${variant.label} (${index + 1}/${variants.length})`;
    }
    $("algorithmAblationStatus").textContent = "Finished 4/4. Select the four completed runs in Merge Runs.";
  } catch (err) {
    $("algorithmAblationStatus").textContent = String(err);
  } finally {
    btn.textContent = oldText;
    setControlsDisabled(latestStatus.status === "running" || Boolean(latestStatus.training_process_running));
  }
}

function parsePeriodList(inputId) {
  return Array.from(new Set($(inputId).value
    .split(",")
    .map(value => Number(value.trim()))
    .filter(value => Number.isFinite(value) && value >= 1 && value <= 100)
    .map(value => Math.round(value))));
}

function parseHeterogeneityScenarios() {
  return $("hetSweepScenariosInput").value
    .split(",")
    .map(value => value.trim())
    .filter(Boolean)
    .map(value => {
      const parts = value.split(/[/:]/).map(item => Number(item.trim()));
      if (parts.length !== 2 || !Number.isFinite(parts[0]) || !Number.isFinite(parts[1])) return null;
      const client = Math.max(1.0, Math.min(10.0, parts[0]));
      const edge = Math.max(1.0, Math.min(10.0, parts[1]));
      return {client, edge, label: `${client}/${edge}`};
    })
    .filter(Boolean);
}

async function runHeterogeneityPeriodSweep() {
  const btn = $("runHetPeriodSweepBtn");
  const oldText = btn.textContent;
  const periods = parsePeriodList("hetSweepPeriodsInput");
  const scenarios = parseHeterogeneityScenarios();
  if (!periods.length) {
    $("hetPeriodSweepStatus").textContent = "Enter at least one valid period, for example 5,10,15,20,25,50.";
    return;
  }
  if (!scenarios.length) {
    $("hetPeriodSweepStatus").textContent = "Enter at least one valid heterogeneity scenario, for example 4.0/3.0.";
    return;
  }

  const rounds = Math.max(1, Math.min(500, Number($("hetSweepRoundsInput").value || 200)));
  const seed = Math.max(0, Math.min(999999, Number($("hetSweepSeedInput").value || 42)));
  const edges = Math.max(1, Math.min(5, Number($("hetSweepEdgesInput").value || 3)));
  const timeLimit = Math.max(0.1, Math.min(300, Number($("hetSweepTimeLimitInput").value || 8)));
  $("hetSweepRoundsInput").value = String(rounds);
  $("hetSweepSeedInput").value = String(seed);
  $("hetSweepEdgesInput").value = String(edges);
  $("hetSweepTimeLimitInput").value = String(timeLimit);

  btn.disabled = true;
  btn.textContent = "Running Matrix...";
  const total = scenarios.length * periods.length;
  let completed = 0;
  try {
    for (const scenario of scenarios) {
      for (const period of periods) {
        const label = `ch${scenario.client} eh${scenario.edge} sp${period}`;
        $("hetPeriodSweepStatus").textContent = `Starting ${label} (${completed + 1}/${total})`;
        const result = await startTrainingRequest("configured", {
          mode: "rounds",
          rounds: String(rounds),
          seed: String(seed),
          time_limit: String(timeLimit),
          policies: $("hetSweepMethodSelect").value,
          figure_axis: "round",
          partition_mode: $("hetSweepPartitionMode").value,
          edges: String(edges),
          client_heterogeneity: String(scenario.client),
          edge_heterogeneity: String(scenario.edge),
          ...currentConstraintParams(),
          ...currentExecutorParams(),
          ...currentPrivacyParams(),
          selection_period: String(period),
          reuse_completed: "true"
        });
        $("controlStatus").textContent = result.message || `${label} start request finished`;
        $("message").textContent = result.message || `${label} start request finished`;
        await refresh();
        if (!result.ok) {
          $("hetPeriodSweepStatus").textContent = `Stopped at ${label}: ${result.message || "start failed"}`;
          return;
        }
        if (!result.reused) {
          $("hetPeriodSweepStatus").textContent = `Running ${label} (${completed + 1}/${total})`;
          await waitForTrainingToFinish(result);
        }
        completed += 1;
        $("hetPeriodSweepStatus").textContent = `Finished ${label} (${completed}/${total})`;
      }
    }
    $("hetPeriodSweepStatus").textContent = `Finished ${completed}/${total}. Use Merge Runs to combine each scenario or all period runs.`;
  } catch (err) {
    $("hetPeriodSweepStatus").textContent = String(err);
  } finally {
    btn.textContent = oldText;
    setControlsDisabled(latestStatus.status === "running" || Boolean(latestStatus.training_process_running));
  }
}

async function stopRun(btn) {
  const oldText = btn.textContent;
  btn.disabled = true;
  btn.textContent = "Stopping...";
  try {
    const res = await fetch("/stop", {method: "POST"});
    const result = await res.json();
    $("message").textContent = result.message || "Stop request finished";
    $("controlStatus").textContent = result.message || "Stop request finished";
    await refresh();
  } catch (err) {
    $("message").textContent = String(err);
    $("controlStatus").textContent = String(err);
  } finally {
    btn.textContent = oldText;
    setControlsDisabled(latestStatus.status === "running" || Boolean(latestStatus.training_process_running));
  }
}

async function rebuildFigures() {
  const btn = $("rebuildBtn");
  btn.disabled = true;
  const oldText = btn.textContent;
  btn.textContent = "Rebuilding...";
  try {
    const res = await fetch("/rebuild-figures", {method: "POST"});
    const result = await res.json();
    $("controlStatus").textContent = result.message || "Rebuild request finished";
    $("message").textContent = result.message || "Rebuild request finished";
    renderFigures();
  } catch (err) {
    $("controlStatus").textContent = String(err);
  } finally {
    btn.textContent = oldText;
    setControlsDisabled(latestStatus.status === "running" || Boolean(latestStatus.training_process_running));
  }
}

$("stopRunStatusBtn").addEventListener("click", () => stopRun($("stopRunStatusBtn")));
$("savePaperRunBtn").addEventListener("click", savePaperRun);
$("runConfiguredBtn").addEventListener("click", () => {
  const methods = selectedMethods();
  startTraining("configured", $("runConfiguredBtn"), mainExperimentParams({
    policies: methods.join(","),
  }));
});
$("runPeriodSweepBtn")?.addEventListener("click", runPeriodSweep);
$("runPrivacySweepBtn")?.addEventListener("click", runPrivacyBudgetSweep);
$("runAlgorithmAblationBtn")?.addEventListener("click", runAlgorithmAblation);
$("runHetPeriodSweepBtn")?.addEventListener("click", runHeterogeneityPeriodSweep);
document.querySelectorAll(".ablation-run").forEach(btn => {
  btn.addEventListener("click", () => {
    const rounds = Math.max(1, Math.min(500, Number($("ablRoundsInput").value || 100)));
    const edges = Math.max(1, Math.min(5, Number($("ablEdgesInput").value || 3)));
    const timeLimit = Math.max(0.1, Math.min(300, Number($("ablTimeLimitInput").value || 8)));
    const clientHet = Math.max(1.0, Math.min(10.0, Number($("ablClientHetInput").value || 2.0)));
    const edgeHet = Math.max(1.0, Math.min(10.0, Number($("ablEdgeHetInput").value || 1.5)));
    const selectionPeriod = Math.max(1, Math.min(100, Number($("ablSelectionPeriodInput").value || 5)));
    const methods = selectedAblationMethods();
    $("ablRoundsInput").value = String(rounds);
    $("ablEdgesInput").value = String(edges);
    $("ablTimeLimitInput").value = String(timeLimit);
    $("ablClientHetInput").value = String(clientHet);
    $("ablEdgeHetInput").value = String(edgeHet);
    $("ablSelectionPeriodInput").value = String(selectionPeriod);
    $("ablationStatus").textContent = `Starting ${btn.dataset.partition}`;
    startTraining("configured", btn, {
      mode: "rounds",
      rounds: String(rounds),
      time_limit: String(timeLimit),
      policies: methods.join(","),
      figure_axis: $("ablFigureAxis").value,
      partition_mode: btn.dataset.partition,
      edges: String(edges),
      client_heterogeneity: String(clientHet),
      edge_heterogeneity: String(edgeHet),
      ...currentConstraintParams(),
      ...currentExecutorParams(),
      ...currentPrivacyParams(),
      selection_period: String(selectionPeriod)
    }).then(() => {
      $("ablationStatus").textContent = $("controlStatus").textContent;
    });
  });
});

function robustParams(overrides={}) {
  const rounds = Math.max(1, Math.min(500, Number($("robustRoundsInput").value || 100)));
  const seed = Math.max(0, Math.min(999999, Number($("robustSeedInput").value || 43)));
  const edges = Math.max(1, Math.min(5, Number($("robustEdgesInput").value || 3)));
  const timeLimit = Math.max(0.1, Math.min(300, Number($("robustTimeLimitInput").value || 8)));
  const clientHet = Math.max(1.0, Math.min(10.0, Number($("robustClientHetInput").value || 3.0)));
  const edgeHet = Math.max(1.0, Math.min(10.0, Number($("robustEdgeHetInput").value || 2.5)));
  const selectionPeriod = Math.max(1, Math.min(100, Number($("robustSelectionPeriodInput").value || 5)));
  $("robustRoundsInput").value = String(rounds);
  $("robustSeedInput").value = String(seed);
  $("robustEdgesInput").value = String(edges);
  $("robustTimeLimitInput").value = String(timeLimit);
  $("robustClientHetInput").value = String(clientHet);
  $("robustEdgeHetInput").value = String(edgeHet);
  $("robustSelectionPeriodInput").value = String(selectionPeriod);
  return {
    mode: "rounds",
    rounds: String(rounds),
    seed: String(seed),
    time_limit: String(timeLimit),
    policies: selectedRobustMethods().join(","),
    figure_axis: "round",
    partition_mode: $("robustPartitionMode").value,
    edges: String(edges),
    client_heterogeneity: String(clientHet),
    edge_heterogeneity: String(edgeHet),
    ...currentConstraintParams(),
    ...currentExecutorParams(),
    ...currentPrivacyParams(),
    selection_period: String(selectionPeriod),
    ...overrides
  };
}

document.querySelectorAll(".robust-seed-run").forEach(btn => {
  btn.addEventListener("click", () => {
    $("robustSeedInput").value = btn.dataset.seed;
    $("robustStatus").textContent = `Starting seed robustness: ${btn.dataset.seed}`;
    startTraining("configured", btn, robustParams({seed: btn.dataset.seed})).then(() => {
      $("robustStatus").textContent = $("controlStatus").textContent;
    });
  });
});
$("runRobustLatencyBtn").addEventListener("click", () => {
  $("robustStatus").textContent = "Starting time-stress robustness";
  startTraining("configured", $("runRobustLatencyBtn"), robustParams({time_limit: "4.0"})).then(() => {
    $("robustStatus").textContent = $("controlStatus").textContent;
  });
});
$("runRobustStrongHetBtn").addEventListener("click", () => {
  $("robustStatus").textContent = "Starting strong heterogeneity robustness";
  startTraining("configured", $("runRobustStrongHetBtn"), robustParams({
    client_heterogeneity: "4.0",
    edge_heterogeneity: "3.0",
    partition_mode: "extreme_edge_label_skew",
    edges: "3"
  })).then(() => {
    $("robustStatus").textContent = $("controlStatus").textContent;
  });
});
$("rebuildBtn")?.addEventListener("click", rebuildFigures);
$("buildRunFiguresBtn")?.addEventListener("click", buildRunFigures);
$("mergeRunsBtn")?.addEventListener("click", mergeRuns);
$("deleteDirsBtn")?.addEventListener("click", deleteDirs);
$("figureAxis").addEventListener("change", () => {
  renderLiveSvg(latestStatus.policy_statuses);
});
$("datasetSelect").addEventListener("change", () => {
  $("dpProfileSelect").value = $("datasetSelect").value === "fmnist"
    ? "balanced"
    : "cifar_resnet";
});
$("liveSmoothCurvesToggle")?.addEventListener("change", () => {
  renderLiveSvg(latestStatus.policy_statuses);
});
$("view100Btn")?.addEventListener("click", () => {
  window.location.href = "/figures/fmnist_lenet5_100r_ep3_accuracy.png";
});
document.addEventListener("visibilitychange", () => {
  if (!document.hidden) refresh().catch(() => {});
});
pollStatus();
</script>
</body>
</html>
"""


def main() -> None:
    args = parse_args()
    output_root = Path(args.output_root)
    if not output_root.is_absolute():
        output_root = ROOT / output_root
    MonitorHandler.output_root = output_root
    server = ThreadingHTTPServer(("127.0.0.1", args.port), MonitorHandler)
    print(f"[OK] monitoring {output_root}")
    print(f"[OK] open http://127.0.0.1:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
