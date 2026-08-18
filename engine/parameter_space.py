"""Lightweight force-field parameter-space construction.

This module deliberately has no Torch or BoTorch imports. LAMMPS validation,
sampling utilities, and result processing can therefore inspect parameters
without importing the Bayesian-optimization runtime.
"""

from __future__ import annotations

from typing import Any


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
