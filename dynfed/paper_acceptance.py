"""Reject duplicate, foreign, incomplete and stale units before Cloud mutation."""
import math


def validate_units(expected, actual, round_index, version, cutoff):
    expected_by_id = {unit.identity: unit for unit in expected}
    if len(expected_by_id) != len(expected) or len({unit.identity for unit in actual}) != len(actual):
        raise RuntimeError("duplicate Cloud unit")
    if set(expected_by_id) != {unit.identity for unit in actual}:
        raise RuntimeError("incomplete or foreign Cloud unit set")
    for unit in actual:
        planned = expected_by_id[unit.identity]
        if unit.generated_round != round_index or unit.base_round != version or unit.base_source != f"cloud:v{version}":
            raise RuntimeError("stale Cloud unit or wrong model version")
        if (unit.clients, unit.samples, unit.mode, unit.edge, unit.arrival) != (
                planned.clients, planned.samples, planned.mode, planned.edge, planned.arrival):
            raise RuntimeError("Cloud unit differs from predetermined membership")
        if not math.isfinite(unit.arrival) or unit.arrival > cutoff+1e-12:
            raise RuntimeError("late Cloud unit after aggregation cutoff")
        if unit.update is None:
            raise RuntimeError("missing trained Cloud update")
