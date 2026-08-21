"""Lightweight force-field parameter-space construction.

This module deliberately has no Torch or BoTorch imports. LAMMPS validation,
sampling utilities, and result processing can therefore inspect parameters
without importing the Bayesian-optimization runtime.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any


def parameter_difference_error(
    resolved_params: Mapping[str, Any],
    differences: object,
) -> str | None:
    """Return the first hard parameter-difference violation, if any.

    ``differences`` is the compiled ``parameter_constraints.differences``
    list.  Values are checked after fixed, tied, and other derived parameters
    have been resolved.  Invalid constraint documents, missing parameter
    values, and non-finite numbers fail closed: none of those cases may reach
    LAMMPS accidentally.

    This function is deliberately pure so candidate generators and execution
    backends can enforce exactly the same rule without importing LAMMPS,
    Torch, or BoTorch.
    """
    if differences is None:
        return None
    if not isinstance(differences, Sequence) or isinstance(
        differences, (str, bytes, bytearray)
    ):
        return (
            "parameter difference constraints malformed: expected a list "
            "of difference rules"
        )
    if not differences:
        return None
    if not isinstance(resolved_params, Mapping):
        return "parameter difference constraints: resolved parameters are not a mapping"

    for index, rule in enumerate(differences, start=1):
        prefix = f"parameter difference constraint #{index}"
        if not isinstance(rule, Mapping):
            return f"{prefix} malformed: expected a mapping"

        parameter = rule.get("parameter")
        if not isinstance(parameter, str) or not parameter.strip():
            return f"{prefix} malformed: 'parameter' must be a non-empty string"
        parameter = parameter.strip()

        type_labels = rule.get("types")
        if (
            not isinstance(type_labels, Sequence)
            or isinstance(type_labels, (str, bytes, bytearray))
            or len(type_labels) != 2
            or any(not isinstance(label, str) or not label.strip()
                   for label in type_labels)
        ):
            return f"{prefix} malformed: 'types' must contain exactly two labels"
        first_label, second_label = (label.strip() for label in type_labels)
        if first_label == second_label:
            return f"{prefix} malformed: the two type labels must be different"

        raw_max_abs = rule.get("max_abs")
        if isinstance(raw_max_abs, bool):
            return f"{prefix} malformed: 'max_abs' must be a finite non-negative number"
        try:
            max_abs = float(raw_max_abs)
        except (TypeError, ValueError):
            return f"{prefix} malformed: 'max_abs' must be a finite non-negative number"
        if not math.isfinite(max_abs) or max_abs < 0.0:
            return f"{prefix} malformed: 'max_abs' must be a finite non-negative number"

        first_key = f"{first_label}_{parameter}"
        second_key = f"{second_label}_{parameter}"
        for key in (first_key, second_key):
            if key not in resolved_params:
                return f"{prefix} cannot be evaluated: missing resolved parameter '{key}'"
            raw_value = resolved_params[key]
            if isinstance(raw_value, bool):
                return f"{prefix} cannot be evaluated: '{key}' is not a finite number"
            try:
                value = float(raw_value)
            except (TypeError, ValueError):
                return f"{prefix} cannot be evaluated: '{key}' is not a finite number"
            if not math.isfinite(value):
                return f"{prefix} cannot be evaluated: '{key}' is not finite"

        first_value = float(resolved_params[first_key])
        second_value = float(resolved_params[second_key])
        difference = abs(first_value - second_value)
        tolerance = 1.0e-12 * max(
            1.0, max_abs, abs(first_value), abs(second_value)
        )
        if difference > max_abs + tolerance:
            return (
                f"{prefix} violated: |{first_key} - {second_key}|="
                f"{difference:.12g} > max_abs {max_abs:.12g}"
            )

    return None


def build_parameter_space(
    config: dict[str, Any],
    *,
    require_nonempty: bool = True,
) -> list[tuple[str, float, float]]:
    """Return ordered independent ``(name, lower, upper)`` dimensions."""
    charge_enabled = bool(config["charge"]["enabled"])
    neutrality = config["charge"]["neutrality_constraint"]
    derived_type = neutrality.get("derive_from_type")
    neutrality_enabled = bool(neutrality.get("enabled", True))
    derived_targets = {
        item["target"]
        for item in config["pair_params"].get("derived_params", [])
    }

    space: list[tuple[str, float, float]] = []
    for atom_type in config["atom_types"]:
        type_id = atom_type["type"]
        label = atom_type["label"]
        for parameter, bounds in atom_type["params"].items():
            if parameter == "charge" and not charge_enabled:
                continue
            if (
                parameter == "charge"
                and neutrality_enabled
                and type_id == derived_type
            ):
                continue
            name = f"{label}_{parameter}"
            if name in derived_targets:
                continue
            if isinstance(bounds, dict) and "min" in bounds and "max" in bounds:
                space.append((name, float(bounds["min"]), float(bounds["max"])))

    if config["pair_params"].get("mixing_rule") == "none":
        for pair in config["pair_params"].get("explicit_pairs", []):
            first, second = pair["types"]
            for parameter in ("epsilon", "sigma"):
                bounds = pair[parameter]
                space.append((
                    f"cross_{first}_{second}_{parameter}",
                    float(bounds["min"]),
                    float(bounds["max"]),
                ))

    if require_nonempty and not space:
        raise ValueError(
            "No free BO parameters found. Check atom_types.params and "
            "pair_params in the generated engine configuration."
        )
    return space
