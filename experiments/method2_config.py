"""Explicit launcher bridge for the verified fixed HE + DP workflow."""
import math
from pathlib import Path
import sys

PROTOCOL = "independent_fixed_he_dp_v1"


def validate(config):
    if config.get("execution_protocol") != PROTOCOL:
        raise ValueError("Unknown independent release protocol")
    if config.get("policies") not in (["fixed_secure_aggregate"], ["resource_placement"]):
        raise ValueError("Dynamic policies have not been integrated with this protocol")
    training, privacy, he = config["training"], config["privacy"], config["he"]
    if training["dataset"] != "cifar10" or training["model"] not in {
        "resnet18_pretrained_head", "resnet18_pretrained_adapter"
    }:
        raise ValueError("Only verified CIFAR10 low-dimensional scopes are supported")
    if training.get("executor") != "serial":
        raise ValueError("Only serial independent execution is verified")
    if training.get("execution_layout", "local") not in {"local", "trusted_split", "alternating_audit", "resource_dynamic"}:
        raise ValueError("Unsupported independent execution layout")
    if (config["policies"] == ["resource_placement"]) != (training.get("execution_layout") == "resource_dynamic"):
        raise ValueError("Resource policy and execution layout must agree")
    if training["partition_mode"] not in {"iid", "extreme_edge_label_skew"}:
        raise ValueError("Unsupported partition")
    for key in ("rounds", "train_limit", "test_limit", "local_epochs"):
        if type(training[key]) is not int or training[key] < 1:
            raise ValueError("Positive integer training counts required")
    clients, edges = config["system"]["clients"], config["system"]["edges"]
    if type(clients) is not int or type(edges) is not int or not 1 <= edges <= clients <= training["train_limit"]:
        raise ValueError("Invalid fixed cohort")
    if not 0 <= training["client_holdout_ratio"] < 1:
        raise ValueError("Invalid holdout")
    for value in (training["learning_rate"], privacy["epsilon"], privacy["clip_norm"]):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Positive finite optimization and privacy parameters required")
    if not 0 < privacy["delta"] < 1:
        raise ValueError("Invalid delta")
    if he != {"backend": "seal", "execution": "real", "custody": "trusted_edge"}:
        raise ValueError("Real SEAL and trusted edge custody are mandatory")


def build(config, root: Path, *, seed, policies, rounds=None, max_new_rounds=None, resume_from_run=None):
    validate(config)
    if policies != config["policies"] or resume_from_run is not None:
        raise ValueError("Only fixed_secure_aggregate is wired; resume is not implemented")
    training, privacy = config["training"], config["privacy"]
    horizon = training["rounds"] if rounds is None else rounds
    if type(horizon) is not int or horizon < 1 or (max_new_rounds is not None and max_new_rounds < 1):
        raise ValueError("Positive release counts required")
    values = dict(model=training["model"], rounds=min(horizon, max_new_rounds or horizon),
                  execution_layout=training.get("execution_layout", "local"),
                  privacy_horizon=horizon, seed=seed, train_limit=training["train_limit"],
                  test_limit=training["test_limit"], partition_mode=training["partition_mode"],
                  client_holdout_ratio=training["client_holdout_ratio"], local_epochs=training["local_epochs"],
                  lr=training["learning_rate"], device=training["device"],
                  clients=config["system"]["clients"], edges=config["system"]["edges"],
                  epsilon=privacy["epsilon"], delta=privacy["delta"], clip_norm=privacy["clip_norm"],
                  dp_release="distributed_he", he_custody="trusted_edge", methods="distributed_dp_he",
                  output_root=config["output_root"])
    command = [sys.executable, str(root / "experiments" / "validate_edge_dp_trajectory.py")]
    for key, value in values.items():
        command.extend(["--" + key.replace("_", "-"), str(value)])
    if config.get("resource_profile"):
        command.extend(["--resource-profile", str((root / config["resource_profile"]).resolve())])
    return command
