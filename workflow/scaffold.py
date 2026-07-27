"""Create a portable FFOpt project from a molecular LAMMPS data file."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re
import shutil
from typing import Any, Iterable

import yaml

from engine.resources import resolve_builtin_resource
from .lammps_data import AtomTypeSummary, LammpsDataSummary, inspect_lammps_data


BULK_TARGETS = {"a", "b", "c", "alpha", "beta", "gamma_ang", "density"}
DEFAULT_UNITS = {
    "a": "A",
    "b": "A",
    "c": "A",
    "alpha": "degree",
    "beta": "degree",
    "gamma_ang": "degree",
    "density": "g/cm3",
    "esub_proxy": "kJ/mol",
}
SUPPORTED_COEFFICIENT_STYLES = {
    "Bond Coeffs": "harmonic",
    "Angle Coeffs": "harmonic",
    "Dihedral Coeffs": "harmonic",
    "Improper Coeffs": "cvff",
}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    value: float
    weight: float = 1.0
    unit: str = ""
    tolerance: float | None = None


@dataclass(frozen=True)
class ScaffoldResult:
    root: Path
    project_file: Path
    dimensions: int
    atom_types: int
    targets: tuple[str, ...]
    warnings: tuple[str, ...]


def parse_target_spec(value: str) -> TargetSpec:
    """Parse ``NAME=VALUE[,WEIGHT[,UNIT]]`` from the CLI."""
    name, separator, payload = value.partition("=")
    name = name.strip()
    if not separator or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError(
            f"Invalid target {value!r}; expected NAME=VALUE[,WEIGHT[,UNIT]]"
        )
    columns = [item.strip() for item in payload.split(",", maxsplit=2)]
    if not columns[0]:
        raise ValueError(f"Target {name!r} has no value")
    target_value = float(columns[0])
    weight = float(columns[1]) if len(columns) > 1 and columns[1] else 1.0
    unit = columns[2] if len(columns) > 2 and columns[2] else DEFAULT_UNITS.get(name, "")
    if weight < 0.0:
        raise ValueError(f"Target {name!r} weight must be non-negative")
    return TargetSpec(name=name, value=target_value, weight=weight, unit=unit)


def _safe_label(raw: str, type_id: int, used: set[str]) -> str:
    label = re.sub(r"[^A-Za-z0-9_]", "_", raw.strip())
    label = re.sub(r"_+", "_", label).strip("_") or f"type_{type_id}"
    if label[0].isdigit():
        label = f"type_{label}"
    base = label
    suffix = 2
    while label in used:
        label = f"{base}_{suffix}"
        suffix += 1
    used.add(label)
    return label


def _single_charge(atom_type: AtomTypeSummary) -> float | None:
    if not atom_type.charges:
        return None
    if len(atom_type.charges) != 1:
        raise ValueError(
            f"Atom type {atom_type.type_id} ({atom_type.label}) has multiple "
            "charges in the data file. FFOpt currently applies one charge per type."
        )
    return float(atom_type.charges[0])


def _range(initial: float, low_scale: float, high_scale: float) -> dict[str, float]:
    low, high = sorted((initial * low_scale, initial * high_scale))
    return {"min": float(low), "max": float(high), "init": float(initial)}


def _charge_range(initial: float, window: float, limit: float) -> dict[str, float]:
    boundary = limit * (1.0 - 1.0e-9)
    return {
        "min": max(-boundary, initial - window),
        "max": min(boundary, initial + window),
        "init": initial,
    }


def _validate_data_contract(summary: LammpsDataSummary, label: str) -> list[str]:
    warnings: list[str] = []
    if summary.atom_style != "full":
        raise ValueError(
            f"{label} uses atom_style {summary.atom_style!r}; the bundled molecular "
            "templates currently require 'Atoms # full'."
        )
    for section, expected in SUPPORTED_COEFFICIENT_STYLES.items():
        if section not in summary.section_styles:
            continue
        observed = summary.section_styles[section].split()[0]
        if observed != expected:
            raise ValueError(
                f"{label} declares '{section} # {summary.section_styles[section]}'; "
                f"the bundled template currently requires {expected!r}."
            )
    for atom_type in summary.atom_types:
        if len(atom_type.pair_coefficients) < 2:
            raise ValueError(
                f"Atom type {atom_type.type_id} ({atom_type.label}) has no usable "
                "Pair Coeffs epsilon/sigma values."
            )
        epsilon, sigma = atom_type.pair_coefficients[:2]
        if epsilon <= 0.0 or sigma <= 0.0:
            raise ValueError(
                f"Atom type {atom_type.type_id} has non-positive epsilon/sigma: "
                f"{epsilon}, {sigma}"
            )
        if len(atom_type.pair_coefficients) > 2:
            warnings.append(
                f"Type {atom_type.type_id} has extra Pair Coeffs columns; only "
                "the first two are interpreted as epsilon and sigma."
            )
        _single_charge(atom_type)
    return warnings


def _copy_file(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def _write_yaml(path: Path, document: dict[str, Any], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(header.rstrip() + "\n\n")
        yaml.safe_dump(document, handle, sort_keys=False, allow_unicode=False)


def _atom_documents(
    summary: LammpsDataSummary,
    *,
    mode: str,
    epsilon_scale: tuple[float, float],
    sigma_scale: tuple[float, float],
    charge_window: float,
    charge_limit: float,
) -> tuple[list[dict[str, Any]], dict[int, str], bool, int]:
    optimize_epsilon = mode in {"full", "fix_sigma", "lj_only"}
    optimize_sigma = mode in {"full", "lj_only"}
    optimize_charge = mode in {"full", "fix_sigma", "charge_only"}
    labels: dict[int, str] = {}
    used: set[str] = set()
    documents = []
    dimensions = 0
    has_charge = any(item.charges for item in summary.atom_types)
    for atom_type in summary.atom_types:
        label = _safe_label(atom_type.label, atom_type.type_id, used)
        labels[atom_type.type_id] = label
        epsilon, sigma = map(float, atom_type.pair_coefficients[:2])
        charge = _single_charge(atom_type)
        charge = float(charge if charge is not None else 0.0)
        if has_charge and abs(charge) >= charge_limit:
            raise ValueError(
                f"Initial charge {charge} for type {atom_type.type_id} reaches "
                f"--charge-limit {charge_limit}; increase the limit explicitly."
            )
        params: dict[str, Any] = {
            "epsilon": (
                _range(epsilon, *epsilon_scale) if optimize_epsilon else epsilon
            ),
            "sigma": _range(sigma, *sigma_scale) if optimize_sigma else sigma,
            "charge": (
                _charge_range(charge, charge_window, charge_limit)
                if optimize_charge and has_charge else charge
            ),
        }
        dimensions += sum(isinstance(value, dict) for value in params.values())
        documents.append({
            "type": atom_type.type_id,
            "label": label,
            "mass": atom_type.mass,
            "params": params,
        })
    return documents, labels, has_charge, dimensions


def create_project(
    *,
    name: str,
    data_file: str | Path,
    destination: str | Path,
    targets: Iterable[TargetSpec],
    single_data: str | Path | None = None,
    mode: str = "full",
    cells: tuple[int, int, int] = (1, 1, 1),
    mixing_rule: str = "geometric",
    derive_charge: str | int = "auto",
    epsilon_scale: tuple[float, float] = (0.5, 2.0),
    sigma_scale: tuple[float, float] = (0.85, 1.15),
    charge_window: float = 0.30,
    charge_limit: float = 2.0,
    force: bool = False,
) -> ScaffoldResult:
    """Create a self-contained scientific project plus builtin method references."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", name):
        raise ValueError("Project name must start with a letter and contain no spaces")
    if mode not in {"full", "fix_sigma", "charge_only", "lj_only"}:
        raise ValueError(f"Unsupported parameter mode: {mode}")
    if mixing_rule not in {"geometric", "arithmetic"}:
        raise ValueError("Scaffold mixing_rule must be geometric or arithmetic")
    if any(int(value) < 1 for value in cells):
        raise ValueError("--cells values must all be positive integers")
    if charge_window <= 0.0 or charge_limit <= 0.0:
        raise ValueError("Charge window and limit must be positive")
    if min(*epsilon_scale, *sigma_scale) <= 0.0:
        raise ValueError("Epsilon and sigma scale factors must be positive")

    target_list = list(targets)
    if not target_list:
        raise ValueError("At least one --target is required")
    duplicate_targets = {item.name for item in target_list if sum(
        other.name == item.name for other in target_list
    ) > 1}
    if duplicate_targets:
        raise ValueError(f"Duplicate targets: {sorted(duplicate_targets)}")
    unsupported = {item.name for item in target_list} - BULK_TARGETS - {"esub_proxy"}
    if unsupported:
        raise ValueError(
            f"The generated bulk/sublimation project cannot provide {sorted(unsupported)}"
        )
    if any(item.name == "esub_proxy" for item in target_list) and not single_data:
        raise ValueError("esub_proxy requires --single-data")

    source = Path(data_file).expanduser().resolve()
    summary = inspect_lammps_data(source)
    warnings = _validate_data_contract(summary, "Bulk data")
    single_source = Path(single_data).expanduser().resolve() if single_data else None
    single_summary = None
    if single_source:
        single_summary = inspect_lammps_data(single_source)
        warnings.extend(_validate_data_contract(single_summary, "Single-molecule data"))
        if len(single_summary.atom_types) != len(summary.atom_types):
            raise ValueError("Bulk and single-molecule data have different atom-type counts")
        bulk_type_ids = {item.type_id for item in summary.atom_types}
        single_type_ids = {item.type_id for item in single_summary.atom_types}
        if bulk_type_ids != single_type_ids:
            raise ValueError("Bulk and single-molecule atom-type IDs do not match")

    root = Path(destination).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(
            f"Destination is not empty: {root}. Pass --force to overwrite generated files."
        )
    root.mkdir(parents=True, exist_ok=True)

    atoms, labels, has_charge, dimensions = _atom_documents(
        summary,
        mode=mode,
        epsilon_scale=epsilon_scale,
        sigma_scale=sigma_scale,
        charge_window=charge_window,
        charge_limit=charge_limit,
    )
    charge_is_optimized = mode in {"full", "fix_sigma", "charge_only"} and has_charge
    derive_type: int | None = None
    if charge_is_optimized and derive_charge != "none":
        if derive_charge == "auto":
            if summary.total_charge is not None and abs(summary.total_charge) <= 1.0e-6:
                derive_type = max(
                    summary.atom_types,
                    key=lambda item: (item.atom_count, -item.type_id),
                ).type_id
        else:
            derive_type = int(derive_charge)
            if derive_type not in labels:
                raise ValueError(f"Derived charge type {derive_type} is not in the data file")
    if derive_type is not None:
        dimensions -= 1
    if dimensions < 1:
        raise ValueError(
            "The selected mode and neutrality constraint leave no free parameters"
        )

    bulk_destination = root / "data" / "bulk" / source.name
    _copy_file(source, bulk_destination)
    single_destination = None
    if single_source:
        single_destination = root / "data" / "molecule" / single_source.name
        _copy_file(single_source, single_destination)

    bulk_input = root / "lammps" / "in.bulk.mol"
    _copy_file(
        resolve_builtin_resource("builtin:lammps/inputs/bulk/in.bulk.mol"),
        bulk_input,
    )
    if single_source:
        _copy_file(
            resolve_builtin_resource(
                "builtin:lammps/inputs/molecule/in.sublimation.bulk"
            ),
            root / "lammps" / "in.sublimation.bulk",
        )
        _copy_file(
            resolve_builtin_resource(
                "builtin:lammps/inputs/molecule/in.sublimation.single"
            ),
            root / "lammps" / "in.sublimation.single",
        )

    feature_counts = {
        labels[item.type_id]: int(item.atom_count) for item in summary.atom_types
    }
    system_document = {
        "manifest": {
            "system_name": name,
            "crystal_structure": "molecular",
            "surface_facet": "none",
            "data_files": {"bulk": str(bulk_destination.relative_to(root)).replace("\\", "/")},
        },
        "pair_params": {
            "mixing_rule": mixing_rule,
            "explicit_pairs": [],
            "derived_params": [],
        },
        "charge": {
            "enabled": has_charge,
            "kspace_accuracy": 1.0e-5,
            "neutrality_constraint": {
                "enabled": derive_type is not None,
                "derive_from_type": derive_type,
                "feature_type_counts": feature_counts,
            },
            "constraints": {"abs_max": charge_limit},
        },
        "atom_types": atoms,
    }
    _write_yaml(
        root / "configs" / "system.yaml",
        system_document,
        "# Generated molecular system. Review every inferred epsilon/sigma/charge range.\n"
        "# Pair Coeffs are interpreted as: type epsilon sigma.",
    )

    bulk_targets = {}
    for target in target_list:
        if target.name not in BULK_TARGETS:
            continue
        bulk_targets[target.name] = {
            "value": target.value,
            "weight": target.weight,
            "unit": target.unit or DEFAULT_UNITS[target.name],
            "tolerance": target.tolerance or max(abs(target.value) * 0.03, 1.0e-6),
        }
    bulk_document = {
        "property_evaluators": {"bulk": {"enabled": True}},
        "targets": bulk_targets,
        "lammps": {
            "bulk_input": "lammps/in.bulk.mol",
            "surf_input": "lammps/in.bulk.mol",
            "compute_surface": False,
            "cutoff": 8.0,
            "timestep": 1.0,
            "bulk": {
                "nx": int(cells[0]),
                "ny": int(cells[1]),
                "nz": int(cells[2]),
                "npt_seed": 101,
                "equil_steps": 20000,
                "prod_steps": 40000,
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
    }
    _write_yaml(
        root / "configs" / "properties" / "bulk.yaml",
        bulk_document,
        "# Bulk NPT evaluator and experimental targets.\n"
        "# nx/ny/nz are the primitive-cell repeats already present in the data file.",
    )

    property_modules = ["configs/properties/bulk.yaml"]
    esub_target = next(
        (target for target in target_list if target.name == "esub_proxy"), None
    )
    if esub_target and single_summary and single_destination:
        molecule_atoms = int(
            single_summary.declared_counts.get(
                "atoms", sum(item.atom_count for item in single_summary.atom_types)
            )
        )
        sublimation_document = {
            "property_evaluators": {"sublimation": {"enabled": True}},
            "targets": {
                "esub_proxy": {
                    "value": esub_target.value,
                    "weight": esub_target.weight,
                    "unit": esub_target.unit or "kJ/mol",
                    "tolerance": esub_target.tolerance or 10.0,
                }
            },
            "sublimation": {
                "enabled": True,
                "data_files": {
                    "bulk": str(bulk_destination.relative_to(root)).replace("\\", "/"),
                    "single": str(single_destination.relative_to(root)).replace("\\", "/"),
                },
                "inputs": {
                    "bulk": "lammps/in.sublimation.bulk",
                    "single": "lammps/in.sublimation.single",
                },
                "molecule_atoms": molecule_atoms,
                "target_temperature": 298.15,
                "target_kj_mol": esub_target.value,
                "thermal_correction_kj_mol": 0.0,
                "tolerance_kj_mol": esub_target.tolerance or 10.0,
                "cutoff": 8.0,
                "kspace_accuracy": 1.0e-5,
            },
        }
        _write_yaml(
            root / "configs" / "properties" / "sublimation.yaml",
            sublimation_document,
            "# Potential-energy sublimation proxy. This is not a full ideal-gas\n"
            "# translational/rotational/vibrational thermochemical correction.",
        )
        property_modules.append("configs/properties/sublimation.yaml")

    sample_points = min(2500, max(800, 50 * max(dimensions, 1)))
    project_document = {
        "schema_version": 1,
        "project": {
            "name": name,
            "default_machine": "local",
            "run_root": f"runs/{name}",
        },
        "modules": {
            "system": "configs/system.yaml",
            "properties": property_modules,
            "methods": {
                "bo": "builtin:configs/methods/bo/portable.yaml",
                "nn": "builtin:configs/methods/nn/portable.yaml",
                "al": "builtin:configs/methods/al/portable.yaml",
            },
            "machines": {
                "local": "builtin:configs/machines/local.yaml",
                "cluster": "builtin:configs/machines/cluster.yaml",
            },
        },
        "stages": {
            "sample": {
                "source": "auto",
                "n_points": sample_points,
                "seeds": [101, 202, 303],
                "elite_centers": min(24, max(4, dimensions)),
                "center_selection": "diverse",
                "center_pool_multiplier": 5,
                "radii": [0.01, 0.025, 0.05],
                "global_fraction": 0.10,
                "design_seed": 20260727,
            }
        },
        "pipeline": {
            "stages": ["bo", "sample", "nn", "al", "audit", "finalize", "validate"],
            "audit": {"top_k": 8, "seeds": [101, 202, 303]},
        },
        "overrides": {
            "workflow": {"active_properties": [item.name for item in target_list]}
        },
    }
    _write_yaml(
        root / "project.yaml",
        project_document,
        "# FFOpt project entry point. Scientific inputs stay local; reusable\n"
        "# methods and machine defaults are loaded from the installed package.",
    )

    readme = f"""# {name}

Generated by `ffopt init` from `{source.name}`.

## Review before running

1. Check every range in `configs/system.yaml`.
2. Confirm `nx`, `ny`, and `nz` in `configs/properties/bulk.yaml`.
3. Confirm that the data file uses the expected Class-I/CVFF bonded styles.
4. Configure the local LAMMPS executable with `ffopt machine configure`.

## Commands

```bash
ffopt show --project project.yaml
ffopt doctor --project project.yaml --machine local
ffopt run --project project.yaml --machine local --dry-run
ffopt run --project project.yaml --machine local --resume
```

Free parameter dimensions generated: **{dimensions}**.
"""
    (root / "README.md").write_text(readme, encoding="utf-8", newline="\n")
    (root / ".gitignore").write_text(
        "runs/\n__pycache__/\n*.py[cod]\n", encoding="utf-8", newline="\n"
    )
    manifest = {
        "generator": "ffopt init",
        "source_data": str(source),
        "single_data": str(single_source) if single_source else None,
        "parameter_mode": mode,
        "dimensions": dimensions,
        "atom_types": len(atoms),
        "targets": [item.name for item in target_list],
        "assumptions": {
            "pair_coeff_columns": ["epsilon", "sigma"],
            "bonded_styles": SUPPORTED_COEFFICIENT_STYLES,
            "mixing_rule": mixing_rule,
        },
        "warnings": warnings,
    }
    (root / "scaffold_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8", newline="\n"
    )
    return ScaffoldResult(
        root=root,
        project_file=root / "project.yaml",
        dimensions=dimensions,
        atom_types=len(atoms),
        targets=tuple(item.name for item in target_list),
        warnings=tuple(warnings),
    )
