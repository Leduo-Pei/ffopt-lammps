"""Compile the public FFOpt input model to the validated engine schema."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from engine.config_loader import deep_merge
from engine.resources import resolve_builtin_resource

from .defaults import method_defaults
from .input_file import FFOptInput, InputFileError, PropertySpec
from .lammps_data import inspect_lammps_data

BULK_TARGETS = {"a", "b", "c", "alpha", "beta", "gamma_ang", "density"}
TARGET_ALIASES = {"gamma": "gamma_ang", "sublimation": "esub_proxy", "adsorption": "ead"}
DEFAULT_UNITS = {
    "a": "A", "b": "A", "c": "A",
    "alpha": "degree", "beta": "degree", "gamma_ang": "degree",
    "density": "g/cm3", "esub_proxy": "kJ/mol", "ead": "kcal/mol",
    "surf_energy": "J/m2",
}
SUPPORTED_TARGET_UNITS = {
    "a": {"a": "A", "angstrom": "A", "angstroms": "A"},
    "b": {"a": "A", "angstrom": "A", "angstroms": "A"},
    "c": {"a": "A", "angstrom": "A", "angstroms": "A"},
    "alpha": {"degree": "degree", "degrees": "degree", "deg": "degree"},
    "beta": {"degree": "degree", "degrees": "degree", "deg": "degree"},
    "gamma_ang": {"degree": "degree", "degrees": "degree", "deg": "degree"},
    "density": {"g/cm3": "g/cm3", "g/cm^3": "g/cm3"},
    "esub_proxy": {"kj/mol": "kJ/mol", "kjmol": "kJ/mol"},
    "ead": {"kcal/mol": "kcal/mol", "kcalmol": "kcal/mol"},
    "surf_energy": {"j/m2": "J/m2", "j/m^2": "J/m2"},
}
PROPERTY_OUTPUT_UNITS = {
    "bulk": {
        "a_0K": "A", "a": "A", "b": "A", "c": "A",
        "alpha": "degree", "beta": "degree", "gamma_ang": "degree",
        "density": "g/cm3", "pe": "kcal/mol", "enthalpy": "kcal/mol",
    },
    "sublimation": {
        "esub_proxy": "kJ/mol", "esub_proxy_kcal_mol": "kcal/mol",
        "esub_single_pe": "kcal/mol", "esub_bulk_pe_per_mol": "kcal/mol",
    },
    "adsorption": {
        "ead": "kcal/mol", "E_complex": "kcal/mol",
        "E_slab": "kcal/mol", "E_mol": "kcal/mol",
    },
    "surface": {
        "surf_energy": "J/m2", "E_complete": "kcal/mol",
        "E_split": "kcal/mol", "A_xy": "A2",
    },
    "elasticity": {
        "B": "GPa", "Cprime": "GPa", "C44": "GPa",
        "C11": "GPa", "C12": "GPa", "G_hill": "GPa",
        "E_hill": "GPa", "nu_hill": "dimensionless",
    },
}


@dataclass(frozen=True)
class CompiledInput:
    document: FFOptInput
    config: dict[str, Any]
    project_data: dict[str, Any]
    dimensions: int
    fitted_properties: tuple[str, ...]
    validation_properties: tuple[str, ...]
    material: dict[str, Any]
    crystal: dict[str, Any]
    parameter_constraints: dict[str, list[dict[str, Any]]]


def _resource(relative: str) -> str:
    return str(resolve_builtin_resource(f"builtin:{relative}"))


def _path(document: FFOptInput, value: str) -> str:
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        candidate = document.path.parent / candidate
    return str(candidate.resolve())


def _property(document: FFOptInput, name: str) -> PropertySpec | None:
    return next((item for item in document.properties if item.name == name), None)


def _primary_data(document: FFOptInput) -> Path:
    choices: list[str | None] = []
    bulk = _property(document, "bulk")
    sub = _property(document, "sublimation")
    ads = _property(document, "adsorption")
    surface = _property(document, "surface")
    choices.extend([
        bulk.data_files.get("bulk") if bulk else None,
        sub.data_files.get("bulk") if sub else None,
        ads.data_files.get("molecule") if ads else None,
        ads.data_files.get("complex") if ads else None,
        surface.data_files.get("complete") if surface else None,
    ])
    value = next((item for item in choices if item), None)
    if value is None:
        raise InputFileError(
            document.path, 1,
            "at least one property must provide a data file for parameter/type validation",
        )
    return Path(_path(document, value))


def _param_value(
    document: FFOptInput,
    label: str,
    name: str,
    initial: float,
    *,
    optimize: bool,
) -> float | dict[str, float]:
    params = document.parameters
    if (
        not optimize
        or name in params.fixed
        or (label, name) in params.fixed_by_type
        or (document.material_kind == "elemental" and name == "charge")
    ):
        return float(initial)
    range_spec = params.type_ranges.get((label, name)) or params.default_ranges.get(name)
    if range_spec is None:  # Parser validation normally catches this.
        raise InputFileError(document.path, 1, f"missing range for {label} {name}")
    low, high = range_spec.bounds(initial)
    if name in {"epsilon", "sigma"} and low <= 0.0:
        raise InputFileError(
            document.path, range_spec.line,
            f"{label} {name} range must remain positive, got [{low}, {high}]",
        )
    return {"min": low, "max": high, "init": float(initial)}


def _compile_atom_types(
    document: FFOptInput,
    primary_data: Path,
    *,
    optimize: bool,
) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    if not primary_data.exists():
        raise InputFileError(document.path, 1, f"data file not found: {primary_data}")
    summary = inspect_lammps_data(primary_data)
    by_id = {item.type_id: item for item in summary.atom_types}
    if document.material_kind == "elemental":
        declared_ids = {item.type_id for item in document.parameters.atom_types}
        observed_ids = set(by_id)
        if declared_ids != observed_ids:
            raise InputFileError(
                document.path,
                document.material_line or 1,
                "elemental primary-data atom-type IDs must exactly match the "
                "parameter table; "
                f"declared={sorted(declared_ids)}, observed={sorted(observed_ids)}, "
                f"missing={sorted(observed_ids - declared_ids)}, "
                f"unused={sorted(declared_ids - observed_ids)}",
            )
    documents: list[dict[str, Any]] = []
    dimensions = 0
    for item in document.parameters.atom_types:
        observed = by_id.get(item.type_id)
        if observed is None:
            raise InputFileError(
                document.path, item.line,
                f"type {item.type_id} ({item.label}) is absent from {primary_data.name}",
            )
        params = {
            "epsilon": _param_value(
                document, item.label, "epsilon", item.epsilon, optimize=optimize
            ),
            "sigma": _param_value(
                document, item.label, "sigma", item.sigma, optimize=optimize
            ),
            "charge": _param_value(
                document, item.label, "charge", item.charge, optimize=optimize
            ),
        }
        dimensions += sum(isinstance(value, dict) for value in params.values())
        documents.append({
            "type": item.type_id,
            "label": item.label,
            "mass": observed.mass,
            "params": params,
        })
    derive_label = document.parameters.derive_charge
    if derive_label is not None:
        derived = next(item for item in document.parameters.atom_types if item.label == derive_label)
        if isinstance(documents[document.parameters.atom_types.index(derived)]["params"]["charge"], dict):
            dimensions -= 1
    counts = {
        next(spec.label for spec in document.parameters.atom_types if spec.type_id == item.type_id): int(item.atom_count)
        for item in summary.atom_types
        if any(spec.type_id == item.type_id for spec in document.parameters.atom_types)
    }
    return documents, dimensions, counts


def _compile_parameter_constraints(
    document: FFOptInput,
    atom_types: list[dict[str, Any]],
    dimensions: int,
    *,
    optimize: bool,
) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, str]], int]:
    """Compile the small public parameter graph without hiding its meaning.

    Exact ties map to the legacy engine's ``derived_params`` mechanism, so a
    tied value is sampled once.  Difference limits remain an explicit domain
    constraint for constraint-aware optimizers; retaining the declaration in
    the compiled configuration avoids converting a hard physical bound into a
    penalty or an undocumented range truncation.
    """
    specs = document.parameters.atom_types
    source_by_label = {item.label: item for item in specs}
    compiled_by_label = {item["label"]: item for item in atom_types}
    result: dict[str, list[dict[str, Any]]] = {
        "ties": [],
        "differences": [],
    }
    derived: list[dict[str, str]] = []

    for tie in document.parameters.ties.values():
        labels = [item.label for item in specs]
        initials = [
            float(getattr(source_by_label[label], tie.parameter))
            for label in labels
        ]
        if max(initials) - min(initials) > 1.0e-12:
            raise InputFileError(
                document.path,
                tie.line,
                f"tie {tie.parameter!r} requires identical initial values",
            )
        source_label = labels[0]
        source_key = f"{source_label}_{tie.parameter}"
        compiled_values = [
            compiled_by_label[label]["params"][tie.parameter]
            for label in labels
        ]
        ranged = [value for value in compiled_values if isinstance(value, dict)]
        if ranged:
            if len(ranged) != len(compiled_values):
                raise InputFileError(
                    document.path,
                    tie.line,
                    f"tie {tie.parameter!r} mixes fixed and ranged atom types",
                )
            low = max(float(value["min"]) for value in ranged)
            high = min(float(value["max"]) for value in ranged)
            initial = initials[0]
            if low >= high or not low <= initial <= high:
                raise InputFileError(
                    document.path,
                    tie.line,
                    f"tie {tie.parameter!r} has no common range containing its initial value",
                )
            compiled_by_label[source_label]["params"][tie.parameter] = {
                "min": low,
                "max": high,
                "init": initial,
            }
            if optimize:
                dimensions -= len(labels) - 1
        for label in labels[1:]:
            target = f"{label}_{tie.parameter}"
            derived.append({
                "target": target,
                "expression": source_key,
                "comment": f"tie {tie.parameter} all",
            })
        result["ties"].append({
            "parameter": tie.parameter,
            "types": labels,
            "source": source_label,
        })

    for constraint in document.parameters.differences:
        first_spec = source_by_label[constraint.first]
        second_spec = source_by_label[constraint.second]
        first_initial = float(getattr(first_spec, constraint.parameter))
        second_initial = float(getattr(second_spec, constraint.parameter))
        if abs(first_initial - second_initial) > constraint.maximum + 1.0e-12:
            raise InputFileError(
                document.path,
                constraint.line,
                "initial sigma difference exceeds the declared maximum",
            )
        first_value = compiled_by_label[constraint.first]["params"][constraint.parameter]
        second_value = compiled_by_label[constraint.second]["params"][constraint.parameter]
        if isinstance(first_value, dict) and isinstance(second_value, dict):
            first_low, first_high = float(first_value["min"]), float(first_value["max"])
            second_low, second_high = float(second_value["min"]), float(second_value["max"])
            minimum_gap = max(
                first_low - second_high,
                second_low - first_high,
                0.0,
            )
            if minimum_gap > constraint.maximum:
                raise InputFileError(
                    document.path,
                    constraint.line,
                    "sigma difference constraint is incompatible with the type ranges",
                )
        result["differences"].append({
            "parameter": constraint.parameter,
            "types": [constraint.first, constraint.second],
            "max_abs": constraint.maximum,
            "unit": "A",
        })

    return result, derived, dimensions


def _target_document(document: FFOptInput, prop: PropertySpec) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    allowed = {
        "bulk": BULK_TARGETS,
        "sublimation": {"esub_proxy"},
        "adsorption": {"ead"},
        "surface": {"surf_energy"},
    }[prop.name]
    for target in prop.targets:
        name = TARGET_ALIASES.get(target.name, target.name)
        if name not in allowed:
            raise InputFileError(
                document.path, target.line,
                f"property {prop.name!r} cannot provide target {target.name!r}",
            )
        if name in result:
            raise InputFileError(document.path, target.line, f"duplicate target {name!r}")
        unit = target.unit or DEFAULT_UNITS[name]
        normalized_unit = SUPPORTED_TARGET_UNITS[name].get(unit.lower())
        if normalized_unit is None:
            accepted = ", ".join(sorted(set(SUPPORTED_TARGET_UNITS[name].values())))
            raise InputFileError(
                document.path,
                target.line,
                f"target {target.name!r} unit must be {accepted}; "
                f"automatic unit conversion is not implemented",
            )
        tolerance = target.tolerance
        if tolerance is None:
            tolerance = 10.0 if name == "esub_proxy" else max(abs(target.value) * 0.03, 1.0e-6)
        result[name] = {
            "value": target.value,
            "weight": target.weight,
            "unit": normalized_unit,
            "tolerance": tolerance,
        }
    return result


def _compile_bulk(document: FFOptInput, prop: PropertySpec, config: dict[str, Any]) -> dict[str, Any]:
    value = prop.data_files.get("bulk")
    if not value:
        raise InputFileError(document.path, prop.line, "bulk property requires 'data PATH'")
    is_bcc = document.crystal_family == "bcc"
    if is_bcc and document.material_kind != "elemental":
        raise InputFileError(
            document.path,
            document.material_line or prop.line,
            "the schema-1 BCC bulk protocol currently requires "
            "'material elemental'",
        )
    if is_bcc and "cells" not in prop.settings:
        raise InputFileError(
            document.path,
            prop.line,
            "BCC bulk requires explicit 'cells_in_data NX NY NZ'",
        )
    cells = tuple(int(item) for item in prop.settings.get("cells", (1, 1, 1)))
    if any(int(value) < 1 for value in cells):
        raise InputFileError(
            document.path,
            prop.setting_lines.get("cells", prop.line),
            "bulk cells_in_data values must be positive integers",
        )
    replicate = tuple(
        int(item) for item in prop.settings.get("replicate", (1, 1, 1))
    )
    if any(value < 1 for value in replicate):
        raise InputFileError(
            document.path,
            prop.setting_lines.get("replicate", prop.line),
            "bulk replicate values must be positive integers",
        )
    effective_cells = tuple(
        count * multiplier for count, multiplier in zip(cells, replicate)
    )
    temperature = float(prop.settings.get("temperature", 300.0))
    timestep = float(prop.settings.get("timestep", 1.0))
    cutoff = float(prop.settings.get("cutoff", 8.0))
    equilibration = int(prop.settings.get("equilibration", 20000))
    production = int(prop.settings.get("production", 40000))
    seed = int(prop.settings.get("seed", 101))
    tdamp = float(prop.settings.get("tdamp", 100.0))
    pdamp = float(prop.settings.get("pdamp", 1000.0))
    if temperature <= 0.0:
        raise InputFileError(
            document.path,
            prop.setting_lines.get("temperature", prop.line),
            "bulk temperature must be positive",
        )
    if timestep <= 0.0:
        raise InputFileError(
            document.path,
            prop.setting_lines.get("timestep", prop.line),
            "bulk timestep must be positive",
        )
    if tdamp <= 0.0:
        raise InputFileError(
            document.path,
            prop.setting_lines.get("tdamp", prop.line),
            "bulk tdamp must be positive",
        )
    if pdamp <= 0.0:
        raise InputFileError(
            document.path,
            prop.setting_lines.get("pdamp", prop.line),
            "bulk pdamp must be positive",
        )
    if cutoff <= 0.0:
        raise InputFileError(
            document.path,
            prop.setting_lines.get("cutoff", prop.line),
            "bulk cutoff must be positive",
        )
    if equilibration < 0:
        raise InputFileError(
            document.path,
            prop.setting_lines.get("equilibration", prop.line),
            "bulk equilibration cannot be negative",
        )
    if production < 5000:
        raise InputFileError(
            document.path,
            prop.setting_lines.get("production", prop.line),
            "bulk production must be at least 5000 timesteps for averaging",
        )
    if seed < 1:
        raise InputFileError(
            document.path,
            prop.setting_lines.get("seed", prop.line),
            "bulk seed must be positive",
        )
    config["manifest"]["data_files"]["bulk"] = _path(document, value)
    bulk_config: dict[str, Any] = {
        "nx": int(effective_cells[0]),
        "ny": int(effective_cells[1]),
        "nz": int(effective_cells[2]),
        "npt_seed": seed,
        "equil_steps": equilibration,
        "prod_steps": production,
        "temperature": temperature,
        "pressure": float(prop.settings.get("pressure", 1.0)),
        "protocol": "minimize+npt",
    }
    if is_bcc:
        bulk_config.update({
            "cells_in_data": list(cells),
            "replicate": list(replicate),
            "effective_cells": list(effective_cells),
            "tdamp": tdamp,
            "pdamp": pdamp,
            "runtime_replicate": True,
            "protocol": "box_relax_0k+tri_npt",
        })
    config["lammps"].update({
        "bulk_input": _resource(
            "lammps/inputs/bulk/in.bcc.elemental"
            if is_bcc
            else "lammps/inputs/bulk/in.bulk.mol"
        ),
        "cutoff": cutoff,
        "timestep": timestep,
        "bulk": bulk_config,
    })
    return _target_document(document, prop)


def _compile_sublimation(
    document: FFOptInput,
    prop: PropertySpec,
    config: dict[str, Any],
    bulk_prop: PropertySpec | None,
) -> dict[str, Any]:
    single = prop.data_files.get("single")
    bulk = prop.data_files.get("bulk") or (bulk_prop.data_files.get("bulk") if bulk_prop else None)
    if not single or not bulk:
        raise InputFileError(document.path, prop.line, "sublimation requires bulk and single data files")
    single_path = Path(_path(document, single))
    if not single_path.exists():
        raise InputFileError(document.path, prop.line, f"single-molecule data not found: {single_path}")
    single_summary = inspect_lammps_data(single_path)
    molecule_atoms = int(
        single_summary.declared_counts.get(
            "atoms", sum(item.atom_count for item in single_summary.atom_types)
        )
    )
    target_temperature = float(prop.settings.get("temperature", 298.15))
    single_cutoff = float(prop.settings.get("cutoff", config["lammps"]["cutoff"]))
    if target_temperature <= 0.0:
        raise InputFileError(
            document.path,
            prop.setting_lines.get("temperature", prop.line),
            "sublimation temperature must be positive",
        )
    if single_cutoff <= 0.0:
        raise InputFileError(
            document.path,
            prop.setting_lines.get("cutoff", prop.line),
            "sublimation cutoff must be positive",
        )
    targets = _target_document(document, prop)
    target_value = targets.get("esub_proxy", {}).get("value", float("nan"))
    config["sublimation"] = {
        "enabled": prop.fitted,
        "data_files": {"bulk": _path(document, bulk), "single": str(single_path)},
        "inputs": {
            "single": _resource("lammps/inputs/molecule/in.sublimation.single"),
        },
        "molecule_atoms": molecule_atoms,
        "target_temperature": target_temperature,
        "target_kj_mol": target_value,
        "thermal_correction_kj_mol": 0.0,
        "tolerance_kj_mol": targets.get("esub_proxy", {}).get("tolerance", 10.0),
        "cutoff": single_cutoff,
        "kspace_accuracy": config["charge"]["kspace_accuracy"],
        "protocol": "bulk_npt_plus_single_minimize",
    }
    return targets


def _compile_adsorption(document: FFOptInput, prop: PropertySpec, config: dict[str, Any]) -> dict[str, Any]:
    protocol = str(prop.settings.get("protocol", "")).lower()
    if not protocol:
        raise InputFileError(document.path, prop.line, "adsorption requires 'protocol minimize'")
    if protocol != "minimize":
        raise InputFileError(
            document.path, prop.line,
            "the bundled adsorption evaluator currently supports protocol minimize only",
        )
    missing = [name for name in ("complex", "slab", "molecule") if name not in prop.data_files]
    if missing:
        raise InputFileError(document.path, prop.line, f"adsorption data missing: {missing}")
    cutoff = float(prop.settings.get("cutoff", 7.0))
    if cutoff <= 0.0:
        raise InputFileError(
            document.path,
            prop.setting_lines.get("cutoff", prop.line),
            "adsorption cutoff must be positive",
        )
    data_paths = {
        role: Path(_path(document, prop.data_files[role]))
        for role in ("complex", "slab", "molecule")
    }
    summaries = {}
    for role, data_path in data_paths.items():
        try:
            summaries[role] = inspect_lammps_data(data_path)
        except (OSError, ValueError) as exc:
            raise InputFileError(
                document.path,
                prop.line,
                f"cannot inspect adsorption {role} data {data_path}: {exc}",
            ) from exc

    metal_label = str(prop.settings.get("metal", "Au"))
    metal_line = prop.setting_lines.get("metal", prop.line)
    labels_by_role = {
        role: {item.label: item for item in summary.atom_types}
        for role, summary in summaries.items()
    }
    for role in ("complex", "slab"):
        metal_type = labels_by_role[role].get(metal_label)
        if metal_type is None:
            raise InputFileError(
                document.path,
                metal_line,
                f"adsorption metal label {metal_label!r} is absent from {role} data",
            )
        if not metal_type.charges or any(abs(charge) > 1.0e-8 for charge in metal_type.charges):
            raise InputFileError(
                document.path,
                metal_line,
                f"adsorption metal label {metal_label!r} must be uncharged in {role} data",
            )
    slab_labels = set(labels_by_role["slab"])
    if slab_labels != {metal_label}:
        extras = sorted(slab_labels - {metal_label})
        raise InputFileError(
            document.path,
            metal_line,
            "schema 1 adsorption requires a single fixed substrate type; "
            f"slab contains additional label(s): {extras}",
        )
    molecule_labels = set(labels_by_role["molecule"])
    if metal_label in molecule_labels:
        raise InputFileError(
            document.path,
            metal_line,
            f"adsorption molecule data must not contain metal label {metal_label!r}",
        )
    complex_molecule_labels = set(labels_by_role["complex"]) - {metal_label}
    if complex_molecule_labels != molecule_labels:
        missing = sorted(molecule_labels - complex_molecule_labels)
        extra = sorted(complex_molecule_labels - molecule_labels)
        raise InputFileError(
            document.path,
            prop.line,
            "adsorption complex molecular labels must match molecule data; "
            f"missing={missing}, extra={extra}",
        )
    parameter_labels = {item.label for item in document.parameters.atom_types}
    if molecule_labels != parameter_labels:
        missing = sorted(parameter_labels - molecule_labels)
        extra = sorted(molecule_labels - parameter_labels)
        raise InputFileError(
            document.path,
            prop.line,
            "adsorption molecule labels must match the parameter table; "
            f"missing={missing}, extra={extra}",
        )
    config["adsorption"] = {
        "enabled": prop.fitted,
        "data_files": {
            "complex": str(data_paths["complex"]),
            "slab": str(data_paths["slab"]),
            "mol": str(data_paths["molecule"]),
        },
        "inputs": {
            "complex": _resource("lammps/inputs/adsorption/in.complex"),
            "slab": _resource("lammps/inputs/adsorption/in.slab"),
            "mol": _resource("lammps/inputs/adsorption/in.mol"),
        },
        "metal": {"label": metal_label},
        "md": {
            # Retain inert prerelease defaults in runtime snapshots so a
            # minimize-only adsorption checkpoint keeps its scientific hash.
            # They are not accepted as public ffopt.in settings or consumed by
            # the current evaluator.
            "temp": 300.0,
            "timestep": 1.0,
            "equil_steps": 20000,
            "prod_steps": 20000,
            "seed": 101,
            "cutoff": cutoff,
            "kspace_accuracy": config["charge"]["kspace_accuracy"],
            "protocol": "minimize",
        },
    }
    return _target_document(document, prop)


def _compile_surface(
    document: FFOptInput,
    prop: PropertySpec,
    config: dict[str, Any],
    primary_data: Path,
) -> dict[str, Any]:
    """Compile the bundled, 0 K BCC(110) cleavage calculation.

    The complete and split boxes are an energy difference, so merely having
    the same total atom count is insufficient: both boxes must contain the
    same count of every numbered atom type.  Type comments in the data files
    are also checked against the primary structure to catch accidental type
    reordering before an expensive LAMMPS run.
    """
    facet = prop.settings.get("facet")
    if facet is None:
        raise InputFileError(
            document.path,
            prop.line,
            "surface requires an explicit 'facet 110' declaration",
        )
    if document.crystal_family != "bcc":
        raise InputFileError(
            document.path,
            prop.setting_lines.get("facet", prop.line),
            "the bundled surface evaluator supports BCC(110) only; "
            f"crystal is {document.crystal_family!r}",
        )
    if str(facet) != "110":
        raise InputFileError(
            document.path,
            prop.setting_lines.get("facet", prop.line),
            "the bundled surface evaluator supports facet 110 only",
        )

    missing = [role for role in ("complete", "split") if role not in prop.data_files]
    if missing:
        raise InputFileError(
            document.path,
            prop.line,
            "surface requires complete and split data files; "
            f"missing={missing}",
        )

    replicate = tuple(int(value) for value in prop.settings.get("replicate", (1, 1, 1)))
    if len(replicate) != 3 or any(value < 1 for value in replicate):
        raise InputFileError(
            document.path,
            prop.setting_lines.get("replicate", prop.line),
            "surface replicate values must be three positive integers",
        )

    data_paths = {
        role: Path(_path(document, prop.data_files[role]))
        for role in ("complete", "split")
    }
    summaries = {}
    for role, data_path in data_paths.items():
        try:
            summaries[role] = inspect_lammps_data(data_path)
        except (OSError, ValueError) as exc:
            raise InputFileError(
                document.path,
                prop.line,
                f"cannot inspect surface {role} data {data_path}: {exc}",
            ) from exc
    try:
        primary_summary = inspect_lammps_data(primary_data)
    except (OSError, ValueError) as exc:  # Usually reported by _compile_atom_types first.
        raise InputFileError(
            document.path,
            prop.line,
            f"cannot inspect primary data {primary_data}: {exc}",
        ) from exc

    def atom_count(role: str, summary: Any) -> int:
        parsed = sum(item.atom_count for item in summary.atom_types)
        declared = summary.declared_counts.get("atoms")
        if parsed < 1:
            raise InputFileError(
                document.path,
                prop.line,
                f"surface {role} data contains no readable atoms",
            )
        if declared is not None and int(declared) != parsed:
            raise InputFileError(
                document.path,
                prop.line,
                f"surface {role} atom header ({declared}) does not match "
                f"the parsed Atoms section ({parsed})",
            )
        return parsed

    complete_count = atom_count("complete", summaries["complete"])
    split_count = atom_count("split", summaries["split"])
    if complete_count != split_count:
        raise InputFileError(
            document.path,
            prop.line,
            "BCC(110) cleavage requires equal atom counts: "
            f"complete={complete_count}, split={split_count}",
        )

    expected_ids = {item.type_id for item in document.parameters.atom_types}
    expected_data_labels = {
        item.type_id: item.label for item in primary_summary.atom_types
    }
    counts_by_role: dict[str, dict[int, int]] = {}
    for role, summary in summaries.items():
        observed_ids = {item.type_id for item in summary.atom_types}
        if observed_ids != expected_ids:
            raise InputFileError(
                document.path,
                prop.line,
                f"surface {role} atom-type IDs must match the parameter table; "
                f"expected={sorted(expected_ids)}, observed={sorted(observed_ids)}",
            )
        observed_labels = {item.type_id: item.label for item in summary.atom_types}
        if observed_labels != expected_data_labels:
            raise InputFileError(
                document.path,
                prop.line,
                f"surface {role} type labels must match the primary data by type ID; "
                f"expected={expected_data_labels}, observed={observed_labels}",
            )
        counts_by_role[role] = {
            item.type_id: int(item.atom_count) for item in summary.atom_types
        }
    if counts_by_role["complete"] != counts_by_role["split"]:
        raise InputFileError(
            document.path,
            prop.line,
            "BCC(110) cleavage requires identical per-type composition: "
            f"complete={counts_by_role['complete']}, split={counts_by_role['split']}",
        )

    config["manifest"].update({"surface_facet": "110"})
    config["manifest"]["data_files"].update({
        "surf_complete": str(data_paths["complete"]),
        "surf_split": str(data_paths["split"]),
    })
    config["lammps"].update({
        "surf_input": _resource("lammps/inputs/surface/in.bcc.surface"),
    })
    config["lammps"]["surf"].update({
        "replicate": list(replicate),
        "protocol": "minimize_0k",
    })
    return _target_document(document, prop)


def _compile_elasticity(
    document: FFOptInput,
    prop: PropertySpec,
    config: dict[str, Any],
    bulk_prop: PropertySpec | None,
    surface_prop: PropertySpec | None,
) -> dict[str, Any]:
    """Compile a role-aware cubic-elasticity contract.

    The targets intentionally live below ``config.elasticity`` rather than in
    the legacy ``config.targets`` mapping.  The latter feeds the first-stage
    weighted structural objective, whereas static elasticity is a downstream
    constrained-minimax objective and dynamic elasticity is promotion or
    validation evidence.
    """

    if bulk_prop is None or "bulk" not in bulk_prop.data_files:
        raise InputFileError(
            document.path,
            prop.line,
            "elasticity requires a bulk property with 'data PATH'",
        )

    canonical_bulk_targets = {
        TARGET_ALIASES.get(target.name, target.name) for target in bulk_prop.targets
    }
    surface_targets = (
        {
            TARGET_ALIASES.get(target.name, target.name)
            for target in surface_prop.targets
        }
        if surface_prop is not None
        else set()
    )
    gate_requirements = {
        "lattice": (canonical_bulk_targets, {"a", "b", "c"}, "bulk a/b/c targets"),
        "angles": (
            canonical_bulk_targets,
            {"alpha", "beta", "gamma_ang"},
            "bulk alpha/beta/gamma targets",
        ),
        "density": (canonical_bulk_targets, {"density"}, "a bulk density target"),
        "surface": (surface_targets, {"surf_energy"}, "a surface-energy target"),
    }
    compiled_gates: dict[str, dict[str, Any]] = {}
    for gate_name, (available, required, description) in gate_requirements.items():
        key = f"gate.{gate_name}"
        if key not in prop.settings:
            continue
        if not required <= available:
            missing = sorted(required - available)
            raise InputFileError(
                document.path,
                prop.setting_lines[key],
                f"elasticity {gate_name} gate requires {description}; "
                f"missing={missing}",
            )
        compiled_gates[gate_name] = {
            (
                "maximum_absolute_error_degree"
                if gate_name == "angles"
                else "maximum_relative_error_percent"
            ): float(prop.settings[key]),
            "unit": "degree" if gate_name == "angles" else "percent",
        }

    strain = [float(value) for value in prop.settings.get(
        "strain", (0.002, 0.004, 0.006)
    )]
    replicate = [int(value) for value in prop.settings.get("replicate", (1, 1, 1))]
    minimum_r2 = float(prop.settings.get("minimum_r2", 0.98))
    born_required = bool(prop.settings.get("born", True))
    reporting_tier = float(prop.settings.get("tier", 20.0))

    targets_by_fidelity: dict[str, dict[str, dict[str, Any]]] = {}
    for fidelity in ("static", "dynamic"):
        targets_by_fidelity[fidelity] = {
            name: {
                "value": next(
                    target.value
                    for target in prop.elasticity_targets
                    if target.fidelity == fidelity and target.name == name
                ),
                "unit": "GPa",
            }
            for name in ("B", "Cprime", "C44")
            if fidelity in prop.elasticity_modules
        }

    modules: dict[str, dict[str, Any]] = {}
    static = prop.elasticity_modules["static"]
    modules["static"] = {
        "evaluator": "static_cubic",
        "role": static.role,
        "fidelity": "static_0k",
        "cost_class": "low",
        "targets": targets_by_fidelity["static"],
        "protocol": {
            "method": "symmetric_energy_strain",
            "strain_magnitudes": strain,
            "replicate": replicate,
            "temperature_k": 0.0,
        },
    }

    dynamic = prop.elasticity_modules.get("dynamic")
    if dynamic is not None:
        dynamic_module: dict[str, Any] = {
            "evaluator": "dynamic_cubic",
            "role": dynamic.role,
            "fidelity": "dynamic_300k",
            "cost_class": "high",
            "targets": targets_by_fidelity["dynamic"],
            "protocol": {
                "method": "symmetric_stress_strain",
                "strain_magnitudes": strain,
                "replicate": replicate,
                "temperature_k": float(prop.settings.get("temperature", 300.0)),
                "timestep_fs": float(prop.settings.get("timestep", 1.0)),
                "equilibration_steps": int(
                    prop.settings.get("equilibration", 20000)
                ),
                "production_steps": int(prop.settings.get("production", 40000)),
                "seeds": [int(value) for value in prop.settings.get(
                    "seeds", (101, 202, 303)
                )],
            },
        }
        if "validation_seeds" in prop.settings:
            dynamic_module["validation_protocol"] = {
                "seeds": [
                    int(value) for value in prop.settings["validation_seeds"]
                ],
            }
        modules["dynamic"] = dynamic_module

    config["elasticity"] = {
        "enabled": True,
        "symmetry": "cubic",
        "target_basis": ["B", "Cprime", "C44"],
        "derived_diagnostics_only": ["G_hill", "E_hill", "nu_hill"],
        "selection": {
            "method": "constrained_minimax_relative_error",
            "structural_gates": compiled_gates,
            "fit_quality": {"minimum_r2": minimum_r2},
            "born_stability": {"required": born_required},
        },
        # A tier labels scientific quality; it is not an eligibility gate and
        # must not discard the best feasible compromise when a pair potential
        # cannot reach the requested percentage.
        "reporting": {
            "mechanical_tier_percent": reporting_tier,
            "tier_is_hard_gate": False,
        },
        "modules": modules,
    }
    return {}


def _stage_line(document: FFOptInput, stage: str, *keys: str) -> int:
    lines = document.stage_setting_lines.get(stage, {})
    return next((lines[key] for key in keys if key in lines), 1)


def _apply_stage_settings(document: FFOptInput, config: dict[str, Any], stages: dict[str, Any]) -> None:
    if "screen" in document.workflow:
        stages.setdefault("screen", {})
    if "finalists" in document.workflow:
        stages.setdefault("finalists", {})
    bo_map = {
        "method": "method", "initial_points": "n_initial", "rounds": "n_bo_iterations",
        "max_rounds": "n_bo_iterations", "batch_size": "batch_size",
        "random_seed": "random_seed", "objective": "objective",
    }
    sample_map = {
        "points": "n_points", "centers": "elite_centers", "elite_centers": "elite_centers",
        "seeds": "seeds", "radii": "radii", "global_fraction": "global_fraction",
        "design_seed": "design_seed",
        "center_selection": "center_selection",
        "boundary_fraction": "boundary_fraction",
    }
    nn_map = {
        "model": "model", "method": "model", "ensemble": "ensemble_size",
        "hidden_layers": "hidden_layers", "epochs": "max_epochs", "max_epochs": "max_epochs",
        "batch_size": "batch_size", "learning_rate": "learning_rate",
        "validation_fraction": "val_fraction", "test_fraction": "test_fraction",
    }
    al_map = {
        "rounds": "n_rounds", "candidates": "n_candidates_per_round",
        "candidate_pool": "n_candidate_pool", "uncertainty_threshold": "uncertainty_threshold",
        "sampling_domain": "sampling_domain", "local_radii": "local_radii",
        "static_candidates": "static_points",
        "improvement_fraction": "improvement_fraction",
        "boundary_fraction": "boundary_fraction",
        "global_fraction": "global_fraction",
    }
    audit_map = {"top_k": "top_k", "seeds": "seeds", "max_workers": "max_workers"}
    screen_map = {
        "minimum": "minimum",
        "per_dimension": "per_dimension",
        "maximum": "maximum",
        "core_fraction": "core_fraction",
        "buffer_multiplier": "buffer_multiplier",
        "elite_fraction": "objective_elite_fraction",
    }
    finalists_map = {
        "minimum": "minimum",
        "maximum": "maximum",
        "window": "near_optimal_window_percent",
        "diverse": "diverse_reserve",
    }
    maps = {
        "bo": bo_map,
        "sample": sample_map,
        "nn": nn_map,
        "al": al_map,
        "audit": audit_map,
        "screen": screen_map,
        "finalists": finalists_map,
    }
    destinations = {
        "bo": config["optimization"], "sample": stages.setdefault("sample", {}),
        "nn": config["nn"], "al": config["active_learning"],
        "audit": stages.setdefault(
            "audit", {"top_k": 8, "seeds": [101, 202, 303]}
        ),
        "screen": stages.get("screen", {}),
        "finalists": stages.get("finalists", {}),
    }
    for stage, settings in document.stage_settings.items():
        if stage == "finalize":
            if settings:
                raise InputFileError(
                    document.path,
                    _stage_line(document, stage, next(iter(settings))),
                    "finalize does not accept settings",
                )
            continue
        if stage == "validate":
            allowed = {
                "trajectory", "require_tolerances", "objective_max",
                "max_error_percent",
            }
            unknown = set(settings) - allowed
            if unknown:
                raise InputFileError(
                    document.path,
                    _stage_line(document, stage, *sorted(unknown)),
                    f"unknown validate setting(s): {sorted(unknown)}",
                )
            if "trajectory" in settings and str(settings["trajectory"]).lower() != "final":
                raise InputFileError(
                    document.path,
                    _stage_line(document, stage, "trajectory"),
                    "validate trajectory is a legacy compatibility setting and "
                    "only accepts 'final'",
                )
            if "require_tolerances" in settings and not isinstance(
                settings["require_tolerances"], bool
            ):
                raise InputFileError(
                    document.path,
                    _stage_line(document, stage, "require_tolerances"),
                    "validate require_tolerances must be yes or no",
                )
            for key in ("objective_max", "max_error_percent"):
                if key in settings:
                    value = float(settings[key])
                    if not math.isfinite(value) or value <= 0.0:
                        raise InputFileError(
                            document.path,
                            _stage_line(document, stage, key),
                            f"validate {key} must be positive",
                        )
            config["validation"].update({
                key: value for key, value in settings.items()
                if key != "trajectory"
            })
            continue
        mapping = maps[stage]
        destination = destinations[stage]
        for key, value in settings.items():
            if key == "early_stop" and stage == "bo":
                destination.setdefault("early_stop", {}).update(value)
                continue
            if key == "early_stop" and stage == "al":
                aliases = {"patience": "patience", "improvement": "min_improvement"}
                unknown = set(value) - set(aliases)
                if unknown:
                    raise InputFileError(
                        document.path,
                        _stage_line(
                            document, stage,
                            *(f"early_stop.{item}" for item in sorted(unknown)),
                        ),
                        f"unknown AL early_stop setting(s): {sorted(unknown)}",
                    )
                early = destination.setdefault("early_stop", {})
                early.update({aliases[item]: raw for item, raw in value.items()})
                continue
            if key == "coverage" and stage == "bo":
                values = value if isinstance(value, list) else [value]
                if len(values) % 2:
                    raise InputFileError(
                        document.path,
                        _stage_line(document, stage, key),
                        "BO coverage requires KEY VALUE pairs",
                    )
                aliases = {
                    "archive": "archive_target",
                    "pool": "candidate_pool",
                    "feasible": "feasible_fraction",
                    "boundary": "boundary_fraction",
                    "uncertainty": "uncertainty_fraction",
                    "global": "global_fraction",
                }
                pairs = dict(zip(values[::2], values[1::2]))
                unknown = {str(item).lower() for item in pairs} - set(aliases)
                if unknown:
                    raise InputFileError(
                        document.path,
                        _stage_line(document, stage, key),
                        f"unknown BO coverage setting(s): {sorted(unknown)}",
                    )
                coverage = destination.setdefault("coverage", {})
                for item, raw in pairs.items():
                    coverage[aliases[str(item).lower()]] = raw
                continue
            if key == "stability_audit" and stage == "bo":
                if not isinstance(value, bool):
                    raise InputFileError(
                        document.path,
                        _stage_line(document, stage, key),
                        "BO stability_audit must be on or off",
                    )
                destination.setdefault("stability_audit", {})["enabled"] = value
                continue
            if stage == "bo" and key.startswith("stability_"):
                audit_key = key.removeprefix("stability_")
                audit_mapping = {
                    "top_k": "top_k",
                    "seeds": "seeds",
                    "max_objective_std": "max_objective_std",
                    "max_property_rel_std": "max_property_rel_std",
                    "noise_penalty": "noise_penalty",
                    "failure_penalty": "failure_penalty",
                }
                if audit_key not in audit_mapping:
                    raise InputFileError(
                        document.path,
                        _stage_line(document, stage, key),
                        f"unknown BO setting {key!r}",
                    )
                if audit_key == "seeds" and not isinstance(value, list):
                    value = [value]
                destination.setdefault("stability_audit", {})[
                    audit_mapping[audit_key]
                ] = value
                continue
            if key == "acquisition" and stage == "al":
                acquisition = str(value).lower()
                allowed = (
                    {"uncertainty", "constrained_minimax"}
                    if document.material_kind != "molecular"
                    else {"uncertainty"}
                )
                if acquisition not in allowed:
                    raise InputFileError(
                        document.path,
                        _stage_line(document, stage, key),
                        f"AL acquisition must be one of {sorted(allowed)}",
                    )
                destination["acquisition"] = acquisition
                continue
            if key not in mapping:
                raise InputFileError(
                    document.path,
                    _stage_line(document, stage, key),
                    f"unknown {stage} setting {key!r}",
                )
            mapped = mapping[key]
            list_settings = {
                "sample": {"seeds", "radii"},
                "nn": {"hidden_layers"},
                "al": {"local_radii"},
                "audit": {"seeds"},
            }
            if mapped in list_settings.get(stage, set()) and not isinstance(value, list):
                value = [value]
            if stage == "nn" and mapped == "model" and str(value).lower() == "ann":
                value = "mlp_ensemble"
            destination[mapped] = value

    optimization = config["optimization"]
    method = str(optimization.get("method", "auto")).lower()
    if method not in {"auto", "gp", "turbo", "saasbo"}:
        raise InputFileError(
            document.path,
            _stage_line(document, "bo", "method"),
            "BO method must be auto, gp, turbo, or saasbo",
        )
    optimization["method"] = method
    objective = str(optimization.get("objective", "weighted_rmse")).lower()
    if objective not in {"weighted_rmse", "feasible_coverage"}:
        raise InputFileError(
            document.path,
            _stage_line(document, "bo", "objective"),
            "BO objective must be weighted_rmse or feasible_coverage",
        )
    optimization["objective"] = objective
    if objective == "feasible_coverage" and document.material_kind == "molecular":
        raise InputFileError(
            document.path,
            _stage_line(document, "bo", "objective"),
            "BO feasible_coverage requires an explicit material workflow",
        )
    configured_coverage = optimization.get("coverage", {})
    if configured_coverage and objective != "feasible_coverage":
        raise InputFileError(
            document.path,
            _stage_line(document, "bo", "coverage"),
            "BO coverage settings require 'objective feasible_coverage'",
        )
    if objective == "feasible_coverage":
        if method not in {"auto", "gp"}:
            raise InputFileError(
                document.path,
                _stage_line(document, "bo", "method", "objective"),
                "BO feasible_coverage uses a GP and requires method auto or gp",
            )
        optimization["method"] = "gp"
        coverage = optimization.setdefault("coverage", {})
        coverage.setdefault("archive_target", 96)
        coverage.setdefault("candidate_pool", 16384)
        coverage.setdefault("feasible_fraction", 0.50)
        coverage.setdefault("boundary_fraction", 0.25)
        coverage.setdefault("uncertainty_fraction", 0.15)
        coverage.setdefault("global_fraction", 0.10)
        for key in ("archive_target", "candidate_pool"):
            value = coverage[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise InputFileError(
                    document.path,
                    _stage_line(document, "bo", "coverage"),
                    f"BO coverage {key} must be a positive integer",
                )
        fraction_keys = (
            "feasible_fraction", "boundary_fraction",
            "uncertainty_fraction", "global_fraction",
        )
        if any(isinstance(coverage[key], bool) for key in fraction_keys):
            raise InputFileError(
                document.path,
                _stage_line(document, "bo", "coverage"),
                "BO coverage fractions must be numeric, not booleans",
            )
        try:
            fractions = [float(coverage[key]) for key in fraction_keys]
        except (TypeError, ValueError) as exc:
            raise InputFileError(
                document.path,
                _stage_line(document, "bo", "coverage"),
                "BO coverage fractions must be numeric",
            ) from exc
        if any(not math.isfinite(value) or value < 0.0 for value in fractions):
            raise InputFileError(
                document.path,
                _stage_line(document, "bo", "coverage"),
                "BO coverage fractions must be finite and non-negative",
            )
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise InputFileError(
                document.path,
                _stage_line(document, "bo", "coverage"),
                "BO coverage feasible/boundary/uncertainty/global fractions must sum to 1",
            )
    bo_public_keys = {
        "n_initial": ("initial_points",),
        "n_bo_iterations": ("max_rounds", "rounds"),
        "batch_size": ("batch_size",),
    }
    for key, public_keys in bo_public_keys.items():
        value = optimization.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InputFileError(
                document.path,
                _stage_line(document, "bo", *public_keys),
                f"BO {key} must be a positive integer",
            )
    random_seed = optimization.get("random_seed")
    if isinstance(random_seed, bool) or not isinstance(random_seed, int):
        raise InputFileError(
            document.path,
            _stage_line(document, "bo", "random_seed"),
            "BO random_seed must be an integer",
        )
    early_stop = optimization.get("early_stop", {})
    unknown_early = set(early_stop) - {"enabled", "patience", "min_improvement"}
    if unknown_early:
        raise InputFileError(
            document.path,
            _stage_line(
                document,
                "bo",
                *(f"early_stop.{key}" for key in sorted(unknown_early)),
            ),
            f"unknown BO early_stop setting(s): {sorted(unknown_early)}",
        )
    if not isinstance(early_stop.get("enabled", True), bool):
        raise InputFileError(
            document.path,
            _stage_line(document, "bo", "early_stop.enabled"),
            "BO early_stop enabled must be on or off",
        )
    if int(early_stop.get("patience", 1)) < 1:
        raise InputFileError(
            document.path,
            _stage_line(document, "bo", "early_stop.patience"),
            "BO early_stop patience must be positive",
        )
    min_improvement = float(early_stop.get("min_improvement", 0.0))
    if not math.isfinite(min_improvement) or min_improvement < 0.0:
        raise InputFileError(
            document.path,
            _stage_line(document, "bo", "early_stop.min_improvement"),
            "BO early_stop min_improvement must be finite and non-negative",
        )
    stability = optimization.get("stability_audit", {})
    if not isinstance(stability.get("enabled", True), bool):
        raise InputFileError(
            document.path,
            _stage_line(document, "bo", "stability_audit"),
            "BO stability_audit must be on or off",
        )
    if int(stability.get("top_k", 0)) < 1:
        raise InputFileError(
            document.path,
            _stage_line(document, "bo", "stability_top_k"),
            "BO stability_top_k must be positive",
        )
    stability_seeds = stability.get("seeds", [])
    if not stability_seeds or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 1
        for seed in stability_seeds
    ):
        raise InputFileError(
            document.path,
            _stage_line(document, "bo", "stability_seeds"),
            "BO stability_seeds must be positive integers",
        )
    for key in (
        "max_objective_std",
        "max_property_rel_std",
        "noise_penalty",
        "failure_penalty",
    ):
        value = float(stability.get(key, 0.0))
        if not math.isfinite(value) or value < 0.0:
            raise InputFileError(
                document.path,
                _stage_line(document, "bo", f"stability_{key}"),
                f"BO stability_{key} must be finite and non-negative",
            )

    sample = stages.setdefault("sample", {})
    sample_public_keys = {
        "n_points": ("points",),
        "elite_centers": ("centers", "elite_centers"),
    }
    for key, public_keys in sample_public_keys.items():
        if int(sample.get(key, 0)) < 1:
            raise InputFileError(
                document.path,
                _stage_line(document, "sample", *public_keys),
                f"sample {key} must be positive",
            )
    if int(sample["elite_centers"]) > int(sample["n_points"]):
        raise InputFileError(
            document.path,
            _stage_line(document, "sample", "centers", "elite_centers"),
            "sample centers cannot exceed points",
        )
    if not sample.get("seeds"):
        raise InputFileError(
            document.path,
            _stage_line(document, "sample", "seeds"),
            "sample seeds cannot be empty",
        )
    if any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 1
        for seed in sample["seeds"]
    ):
        raise InputFileError(
            document.path,
            _stage_line(document, "sample", "seeds"),
            "sample seeds must be positive integers",
        )
    design_seed = sample.get("design_seed")
    if isinstance(design_seed, bool) or not isinstance(design_seed, int):
        raise InputFileError(
            document.path,
            _stage_line(document, "sample", "design_seed"),
            "sample design_seed must be an integer",
        )
    radii = [float(value) for value in sample.get("radii", [])]
    if not radii or any(
        not math.isfinite(value) or value <= 0.0 or value > 1.0
        for value in radii
    ):
        raise InputFileError(
            document.path,
            _stage_line(document, "sample", "radii"),
            "sample radii must be in (0, 1]",
        )
    global_fraction = float(sample.get("global_fraction", 0.0))
    if not 0.0 <= global_fraction <= 1.0:
        raise InputFileError(
            document.path,
            _stage_line(document, "sample", "global_fraction"),
            "sample global_fraction must be in [0, 1]",
        )
    boundary_fraction = float(sample.get("boundary_fraction", 0.0))
    if not 0.0 <= boundary_fraction <= 1.0:
        raise InputFileError(
            document.path,
            _stage_line(document, "sample", "boundary_fraction"),
            "sample boundary_fraction must be in [0, 1]",
        )
    if boundary_fraction + global_fraction > 1.0 + 1.0e-12:
        raise InputFileError(
            document.path,
            _stage_line(document, "sample", "boundary_fraction", "global_fraction"),
            "sample boundary_fraction + global_fraction cannot exceed 1",
        )
    if str(sample.get("center_selection", "diverse")) not in {"top", "diverse"}:
        raise InputFileError(
            document.path,
            _stage_line(document, "sample", "center_selection"),
            "sample center_selection must be top or diverse",
        )

    nn = config["nn"]
    supported_models = {
        "mlp_ensemble", "gp", "random_forest", "xgboost", "extra_trees",
        "pair_product_extra_trees", "svr_rbf", "polynomial_ridge",
    }
    if str(nn.get("model", "")).lower() not in supported_models:
        raise InputFileError(
            document.path,
            _stage_line(document, "nn", "method", "model"),
            "NN method must be ann, gp, random_forest, xgboost, extra_trees, "
            "pair_product_extra_trees, svr_rbf, or polynomial_ridge",
        )
    nn_public_keys = {
        "ensemble_size": ("ensemble",),
        "max_epochs": ("epochs", "max_epochs"),
        "batch_size": ("batch_size",),
    }
    for key, public_keys in nn_public_keys.items():
        if int(nn.get(key, 0)) < 1:
            raise InputFileError(
                document.path,
                _stage_line(document, "nn", *public_keys),
                f"NN {key} must be positive",
            )
    hidden_layers = nn.get("hidden_layers", [])
    if not hidden_layers or any(
        isinstance(width, bool) or not isinstance(width, int) or width < 1
        for width in hidden_layers
    ):
        raise InputFileError(
            document.path,
            _stage_line(document, "nn", "hidden_layers"),
            "NN hidden_layers must be positive integers",
        )
    learning_rate = float(nn.get("learning_rate", 0.0))
    if not math.isfinite(learning_rate) or learning_rate <= 0.0:
        raise InputFileError(
            document.path,
            _stage_line(document, "nn", "learning_rate"),
            "NN learning_rate must be positive",
        )
    val_fraction = float(nn.get("val_fraction", 0.0))
    test_fraction = float(nn.get("test_fraction", 0.0))
    if (
        not math.isfinite(val_fraction)
        or not math.isfinite(test_fraction)
        or val_fraction < 0.0
        or test_fraction < 0.0
        or val_fraction + test_fraction >= 1.0
    ):
        raise InputFileError(
            document.path,
            _stage_line(
                document,
                "nn",
                "validation_fraction",
                "test_fraction",
            ),
            "NN validation_fraction and test_fraction must be non-negative and sum to < 1",
        )

    active_learning = config["active_learning"]
    al_public_keys = {
        "n_rounds": ("rounds",),
        "n_candidates_per_round": ("candidates",),
        "n_candidate_pool": ("candidate_pool",),
    }
    for key, public_keys in al_public_keys.items():
        if int(active_learning.get(key, 0)) < 1:
            raise InputFileError(
                document.path,
                _stage_line(document, "al", *public_keys),
                f"AL {key} must be positive",
            )
    if int(active_learning["n_candidate_pool"]) < int(active_learning["n_candidates_per_round"]):
        raise InputFileError(
            document.path,
            _stage_line(document, "al", "candidate_pool"),
            "AL candidate_pool cannot be smaller than candidates",
        )
    uncertainty_threshold = float(
        active_learning.get("uncertainty_threshold", -1.0)
    )
    if not math.isfinite(uncertainty_threshold) or uncertainty_threshold < 0.0:
        raise InputFileError(
            document.path,
            _stage_line(document, "al", "uncertainty_threshold"),
            "AL uncertainty_threshold must be finite and non-negative",
        )
    if str(active_learning.get("sampling_domain", "")) not in {"global", "core_envelope"}:
        raise InputFileError(
            document.path,
            _stage_line(document, "al", "sampling_domain"),
            "AL sampling_domain must be global or core_envelope",
        )
    al_radii = [float(value) for value in active_learning.get("local_radii", [])]
    if not al_radii or any(
        not math.isfinite(value) or value <= 0.0 or value > 1.0
        for value in al_radii
    ):
        raise InputFileError(
            document.path,
            _stage_line(document, "al", "local_radii"),
            "AL local_radii must be in (0, 1]",
        )
    if "static_points" in active_learning and int(active_learning["static_points"]) < 1:
        raise InputFileError(
            document.path,
            _stage_line(document, "al", "static_candidates"),
            "AL static_candidates must be positive",
        )
    if str(active_learning.get("acquisition", "uncertainty")) == "constrained_minimax":
        fractions = [
            float(active_learning.get("improvement_fraction", 0.50)),
            float(active_learning.get("boundary_fraction", 0.30)),
            float(active_learning.get("global_fraction", 0.20)),
        ]
        if any(not math.isfinite(value) or value < 0.0 for value in fractions):
            raise InputFileError(
                document.path,
                _stage_line(document, "al", "improvement_fraction"),
                "AL acquisition fractions must be finite and non-negative",
            )
        if not math.isclose(sum(fractions), 1.0, rel_tol=0.0, abs_tol=1.0e-9):
            raise InputFileError(
                document.path,
                _stage_line(document, "al", "improvement_fraction"),
                "AL improvement/boundary/global fractions must sum to 1",
            )
        al_stop = active_learning.setdefault("early_stop", {})
        al_stop.setdefault("patience", 3)
        al_stop.setdefault("min_improvement", 0.5)
        if int(al_stop["patience"]) < 1:
            raise InputFileError(
                document.path,
                _stage_line(document, "al", "early_stop.patience"),
                "AL early_stop patience must be positive",
            )
        improvement = float(al_stop["min_improvement"])
        if not math.isfinite(improvement) or improvement < 0.0:
            raise InputFileError(
                document.path,
                _stage_line(document, "al", "early_stop.improvement"),
                "AL early_stop improvement must be finite and non-negative",
            )

    audit = stages.setdefault("audit", {"top_k": 8, "seeds": [101, 202, 303]})
    if int(audit.get("top_k", 0)) < 1:
        raise InputFileError(
            document.path,
            _stage_line(document, "audit", "top_k"),
            "audit top_k must be positive",
        )
    if not audit.get("seeds") or any(
        isinstance(seed, bool) or not isinstance(seed, int) or seed < 1
        for seed in audit["seeds"]
    ):
        raise InputFileError(
            document.path,
            _stage_line(document, "audit", "seeds"),
            "audit seeds must be positive integers",
        )

    if "screen" in document.workflow:
        screen = stages.setdefault("screen", {})
        defaults = {
            "minimum": 480,
            "per_dimension": 160,
            "maximum": 1200,
            "core_fraction": 0.75,
            "buffer_multiplier": 1.50,
            "objective_elite_fraction": 0.10,
        }
        for key, value in defaults.items():
            screen.setdefault(key, value)
        for key in ("minimum", "per_dimension", "maximum"):
            value = screen[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise InputFileError(
                    document.path,
                    _stage_line(document, "screen", key),
                    f"screen {key} must be a positive integer",
                )
        if int(screen["minimum"]) > int(screen["maximum"]):
            raise InputFileError(
                document.path,
                _stage_line(document, "screen", "minimum", "maximum"),
                "screen minimum cannot exceed maximum",
            )
        for key in ("core_fraction", "objective_elite_fraction"):
            value = float(screen[key])
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise InputFileError(
                    document.path,
                    _stage_line(document, "screen", key),
                    f"screen {key} must be in [0, 1]",
                )
        if float(screen["buffer_multiplier"]) < 1.0:
            raise InputFileError(
                document.path,
                _stage_line(document, "screen", "buffer_multiplier"),
                "screen buffer_multiplier must be at least 1",
            )

    if "finalists" in document.workflow:
        finalists = stages.setdefault("finalists", {})
        defaults = {
            "minimum": 10,
            "maximum": 20,
            "near_optimal_window_percent": 1.0,
            "diverse_reserve": 1,
        }
        for key, value in defaults.items():
            finalists.setdefault(key, value)
        for key in ("minimum", "maximum", "diverse_reserve"):
            value = finalists[key]
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise InputFileError(
                    document.path,
                    _stage_line(document, "finalists", key),
                    f"finalists {key} must be a non-negative integer",
                )
        if int(finalists["minimum"]) < 1:
            raise InputFileError(
                document.path,
                _stage_line(document, "finalists", "minimum"),
                "finalists minimum must be positive",
            )
        if int(finalists["minimum"]) > int(finalists["maximum"]):
            raise InputFileError(
                document.path,
                _stage_line(document, "finalists", "minimum", "maximum"),
                "finalists minimum cannot exceed maximum",
            )
        window = float(finalists["near_optimal_window_percent"])
        if not math.isfinite(window) or window < 0.0:
            raise InputFileError(
                document.path,
                _stage_line(document, "finalists", "window"),
                "finalists window must be finite and non-negative",
            )


def compile_input(document: FFOptInput) -> CompiledInput:
    """Compile a parsed input file without reading any legacy project YAML."""
    primary = _primary_data(document)
    optimize = "bo" in document.workflow
    atom_types, dimensions, type_counts = _compile_atom_types(
        document, primary, optimize=optimize
    )
    parameter_constraints, derived_params, dimensions = (
        _compile_parameter_constraints(
            document,
            atom_types,
            dimensions,
            optimize=optimize,
        )
    )
    if dimensions < 1 and any(stage in {"bo", "sample", "nn", "al"} for stage in document.workflow):
        raise InputFileError(document.path, 1, "parameter constraints leave no free dimensions")
    params = document.parameters
    if params.mixing_epsilon not in {"default", "geometric"}:
        raise InputFileError(
            document.path, 1,
            "the LAMMPS backend supports geometric epsilon mixing; choose sigma "
            "default, geometric (LAMMPS mix geometric), or arithmetic "
            "(LAMMPS mix arithmetic)",
        )
    derive_type = None
    if params.derive_charge is not None:
        derive_type = next(item.type_id for item in params.atom_types if item.label == params.derive_charge)

    config = deep_merge(method_defaults(), {
        "manifest": {
            "system_name": document.project,
            "crystal_structure": document.crystal_family,
            "surface_facet": "none",
            "data_files": {"bulk": str(primary)},
        },
        "pair_params": {
            "mixing_rule": params.mixing_sigma,
            "explicit_pairs": [],
            "derived_params": derived_params,
        },
        "charge": {
            "enabled": document.material_kind != "elemental",
            "kspace_accuracy": 1.0e-5,
            "neutrality_constraint": {
                "enabled": derive_type is not None,
                "derive_from_type": derive_type,
                "feature_type_counts": type_counts,
            },
            "constraints": {"abs_max": params.charge_abs_max},
        },
        "atom_types": atom_types,
        "property_evaluators": {
            "bulk": {"enabled": False},
            "sublimation": {"enabled": False},
            "adsorption": {"enabled": False},
            "surface": {"enabled": False},
        },
        "targets": {},
        "lammps": {
            "bulk_input": _resource("lammps/inputs/bulk/in.bulk.mol"),
            "surf_input": _resource("lammps/inputs/bulk/in.bulk.mol"),
            "compute_surface": False,
            "cutoff": 8.0,
            "timestep": 1.0,
            "bulk": {
                "nx": 1, "ny": 1, "nz": 1, "npt_seed": 101,
                "equil_steps": 20000, "prod_steps": 40000,
                "temperature": 300.0, "pressure": 1.0,
                "protocol": "minimize+npt",
            },
            "surf": {"npt_seed": 101, "equil_steps": 5000, "prod_steps": 10000},
        },
        "output_mapping": {
            "bulk": {
                "columns": [
                    "a_0K", "a", "b", "c", "alpha", "beta", "gamma_ang",
                    "density", "pe", "enthalpy",
                ],
                "target_keys": {
                    "a": "a", "b": "b", "c": "c", "alpha": "alpha",
                    "beta": "beta", "gamma_ang": "gamma_ang", "density": "density",
                },
            },
            "surf": {"E_slab_key": "E_slab", "A_xy_key": "A_xy"},
        },
        "surf_energy_sanity": {"min": 0.1, "max": 10.0},
        "sublimation": {"enabled": False},
        "adsorption": {"enabled": False},
        "validation": {
            "property_evaluators": {},
            "property_settings": {},
            "property_units": {},
            "trajectory": "final",
        },
    })

    material = {
        "kind": document.material_kind,
        "name": document.material_name or document.project,
    }
    if (
        document.material_kind == "elemental"
        and document.crystal_family == "bcc"
    ):
        material.update({
            "force_field": "lj/cut",
            "pair_style": "lj/cut",
        })
        config["lammps"]["pair_style"] = "lj/cut"
        config["pair_params"]["pair_style"] = "lj/cut"
    crystal = {"family": document.crystal_family}
    material_pipeline = bool(
        document.material_kind != "molecular"
        or document.crystal_family != "molecular"
        or any(stage in {"screen", "finalists"} for stage in document.workflow)
    )
    uses_material_extension = bool(
        document.material_line is not None
        or document.crystal_line is not None
        or parameter_constraints["ties"]
        or parameter_constraints["differences"]
        or material_pipeline
    )
    if uses_material_extension:
        config["material"] = material
        config["crystal"] = crystal
        config["parameter_constraints"] = parameter_constraints

    fitted: list[str] = []
    validation_only: list[str] = []
    bulk_prop = _property(document, "bulk")
    surface_prop = _property(document, "surface")
    for prop in document.properties:
        if prop.name == "bulk":
            targets = _compile_bulk(document, prop, config)
        elif prop.name == "sublimation":
            targets = _compile_sublimation(document, prop, config, bulk_prop)
        elif prop.name == "adsorption":
            targets = _compile_adsorption(document, prop, config)
        elif prop.name == "surface":
            targets = _compile_surface(document, prop, config, primary)
        elif prop.name == "elasticity":
            targets = _compile_elasticity(
                document,
                prop,
                config,
                bulk_prop,
                surface_prop,
            )
        else:  # Parser validation makes this branch defensive only.
            raise InputFileError(
                document.path,
                prop.line,
                f"property input compilation is not implemented for {prop.name!r}",
            )
        config["validation"]["property_units"].update(PROPERTY_OUTPUT_UNITS[prop.name])
        config["validation"]["property_units"].update({
            name: info["unit"] for name, info in targets.items()
        })
        if prop.fitted:
            fitted.append(prop.name)
            config["targets"].update(targets)
            if prop.name != "elasticity":
                config["property_evaluators"][prop.name] = {"enabled": True}
            if prop.name == "sublimation":
                config["sublimation"]["enabled"] = True
                config["property_evaluators"]["bulk"] = {"enabled": True}
            elif prop.name == "adsorption":
                config["adsorption"]["enabled"] = True
            elif prop.name == "surface":
                config["lammps"]["compute_surface"] = True
        else:
            validation_only.append(prop.name)
            config["validation"]["property_evaluators"][prop.name] = {"enabled": True}
            if prop.name == "sublimation":
                config["validation"]["property_evaluators"]["bulk"] = {"enabled": True}
                config["validation"]["property_settings"]["sublimation"] = {"enabled": True}
            elif prop.name == "adsorption":
                config["validation"]["property_settings"]["adsorption"] = {"enabled": True}
            elif prop.name == "surface":
                config["validation"]["property_settings"]["lammps"] = {
                    "compute_surface": True,
                }
    # Every fitted property is also recomputed by final validation.
    for name in fitted:
        if name != "elasticity":
            config["validation"]["property_evaluators"].setdefault(
                name, {"enabled": True}
            )

    stages: dict[str, Any] = {
        "sample": {
            "n_points": min(2500, max(1500, 75 * max(dimensions, 1))),
            "seeds": [101, 202, 303],
            "elite_centers": min(24, max(4, dimensions)),
            "center_selection": "diverse",
            "center_pool_multiplier": 5,
            "radii": [0.01, 0.025, 0.05],
            "global_fraction": 0.10,
            "design_seed": 20260727,
        },
        "audit": {"top_k": 8, "seeds": [101, 202, 303]},
    }
    _apply_stage_settings(document, config, stages)
    config.setdefault("workflow", {})["active_properties"] = list(config["targets"])

    pipeline_stages = list(document.workflow)
    if material_pipeline:
        # ``al`` remains the concise public token.  The material registry expands
        # it into independently restartable constrained-refinement rounds.
        pipeline_stages = [
            "constrained_al" if stage == "al" else stage
            for stage in pipeline_stages
        ]
    pipeline: dict[str, Any] = {
        "stages": pipeline_stages,
        "audit": dict(stages["audit"]),
    }
    if material_pipeline and "screen" in pipeline_stages:
        # The public ``screen`` stage expands to two executable nodes.  Keep
        # the fully defaulted scientific selection contract on both nodes so
        # their signatures and command builders do not depend on knowing that
        # the settings originally came from a synthetic public-stage token.
        screen_settings = dict(stages["screen"])
        pipeline["candidates"] = dict(screen_settings)
        pipeline["static"] = dict(screen_settings)
    if material_pipeline and "constrained_al" in pipeline_stages:
        active_learning = config["active_learning"]
        early_stop = active_learning.get("early_stop", {})
        improvement_fraction = float(
            active_learning.get("improvement_fraction", 0.50)
        )
        maximum_rounds = int(active_learning["n_rounds"])
        pipeline["stage_repetitions"] = {
            "constrained_al": maximum_rounds,
        }
        pipeline["constrained_al"] = {
            "maximum_rounds": maximum_rounds,
            "patience": int(early_stop.get("patience", 3)),
            "minimum_improvement_percent_points": float(
                early_stop.get("min_improvement", 0.5)
            ),
            "structural_proposals_per_round": int(
                active_learning["n_candidates_per_round"]
            ),
            "mechanical_proposals_per_round": int(
                active_learning.get("static_points", 20)
            ),
            "improvement_fraction": improvement_fraction,
            # The prerelease explicit-pool backend calls the improvement
            # tranche ``feasible``.  Preserve the alias until every backend
            # consumes the generic improvement name.
            "feasible_fraction": improvement_fraction,
            "boundary_fraction": float(
                active_learning.get("boundary_fraction", 0.30)
            ),
            "global_fraction": float(
                active_learning.get("global_fraction", 0.20)
            ),
            "candidate_pool": int(active_learning["n_candidate_pool"]),
            "seed": int(config["optimization"]["random_seed"]),
            # Reuse the declared replicated structural-audit seeds.  A
            # single NPT trajectory is not sufficient evidence for promoting
            # a newly acquired candidate to exact static mechanics.
            "structural_seeds": [int(seed) for seed in stages["audit"]["seeds"]],
        }
        pipeline["graph_mode"] = "linear"
    if material_pipeline and "finalists" in pipeline_stages:
        finalist_settings = dict(stages["finalists"])
        pipeline["finalists"] = {
            **finalist_settings,
            "top_n": int(finalist_settings["maximum"]),
            "diversity_slots": int(finalist_settings["diverse_reserve"]),
        }

    project_data = {
        "schema_version": 1,
        "project": {
            "name": document.project,
            "default_machine": "local",
            "run_root": f"runs/{document.project}",
        },
        "stages": stages,
        "pipeline": pipeline,
    }
    if uses_material_extension:
        project_data.update({
            "material": material,
            "crystal": crystal,
            "parameter_constraints": parameter_constraints,
        })
    return CompiledInput(
        document=document,
        config=config,
        project_data=project_data,
        dimensions=dimensions,
        fitted_properties=tuple(fitted),
        validation_properties=tuple(validation_only),
        material=material,
        crystal=crystal,
        parameter_constraints=parameter_constraints,
    )
