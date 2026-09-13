"""Independent local/split execution, with no private cross-client feedback."""
from .split_learning import split_local_train_lenet5

LAYOUTS = ("local", "trusted_split", "alternating_audit")


def planned_mode(layout, client, round_index):
    if layout not in LAYOUTS:
        raise ValueError("Unsupported independent execution layout")
    if layout == "alternating_audit":
        # A public test schedule, not a resource optimizer or paper policy.
        return "LIEIIC" if (client + round_index) % 2 else "LIIC"
    return "LIEIIC" if layout == "trusted_split" else "LIIC"


def train_independent(mode, state, x, y, epochs, lr, device, model, shape, classes, *, seed, cache):
    if mode not in {"LIIC", "LIEIIC"}:
        raise ValueError("Mode requires an unimplemented privacy/feedback protocol")
    # The existing trainer strictly reloads both state parts and recreates the
    # optimizer on each call. Only model allocations may be cached across clients.
    return split_local_train_lenet5(
        mode, state["end"], state["edge"], x, y, epochs, lr, device, model, shape, classes,
        training_seed=seed, model_cache=cache, mechanisms={"emb": "trusted", "grad": "trusted"})
