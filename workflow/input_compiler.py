"""Compile the public FFOpt input model to the validated engine schema."""

from __future__ import annotations

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
}


@dataclass(frozen=True)
class CompiledInput:
    document: FFOptInput
    config: dict[str, Any]
    project_data: dict[str, Any]
    dimensions: int
    fitted_properties: tuple[str, ...]
    validation_properties: tuple[str, ...]


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
) -> float | dict[str, float]:
    params = document.parameters
    if name in params.fixed or (label, name) in params.fixed_by_type:
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


def _compile_atom_types(document: FFOptInput, primary_data: Path) -> tuple[list[dict[str, Any]], int, dict[str, int]]:
    if not primary_data.exists():
        raise InputFileError(document.path, 1, f"data file not found: {primary_data}")
    summary = inspect_lammps_data(primary_data)
    by_id = {item.type_id: item for item in summary.atom_types}
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
            "epsilon": _param_value(document, item.label, "epsilon", item.epsilon),
            "sigma": _param_value(document, item.label, "sigma", item.sigma),
            "charge": _param_value(document, item.label, "charge", item.charge),
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
    protocol = str(prop.settings.get("protocol", "standard")).lower()
    if protocol not in {"standard", "crystal_npt", "minimize+npt"}:
        raise InputFileError(
            document.path, prop.line,
            "bulk uses the fixed standard minimize + NPT protocol; remove protocol or use 'protocol standard'",
        )
    cells = prop.settings.get("cells", (1, 1, 1))
    if any(int(value) < 1 for value in cells):
        raise InputFileError(document.path, prop.line, "bulk cells must be positive")
    config["manifest"]["data_files"]["bulk"] = _path(document, value)
    config["lammps"].update({
        "bulk_input": _resource("lammps/inputs/bulk/in.bulk.mol"),
        "cutoff": float(prop.settings.get("cutoff", 8.0)),
        "timestep": float(prop.settings.get("timestep", 1.0)),
        "bulk": {
            "nx": int(cells[0]), "ny": int(cells[1]), "nz": int(cells[2]),
            "npt_seed": int(prop.settings.get("seed", 101)),
            "equil_steps": int(prop.settings.get("equilibration", 20000)),
            "prod_steps": int(prop.settings.get("production", 40000)),
            "temperature": float(prop.settings.get("temperature", 300.0)),
            "pressure": float(prop.settings.get("pressure", 1.0)),
            "protocol": "minimize+npt",
        },
    })
    return _target_document(document, prop)


def _compile_sublimation(
    document: FFOptInput,
    prop: PropertySpec,
    config: dict[str, Any],
    bulk_prop: PropertySpec | None,
) -> dict[str, Any]:
    protocol = str(prop.settings.get("protocol", "standard")).lower()
    if protocol not in {"standard", "potential_energy"}:
        raise InputFileError(
            document.path, prop.line,
            "sublimation currently uses bulk NPT mean PE + minimized single-molecule PE",
        )
    single = prop.data_files.get("single")
    bulk = prop.data_files.get("bulk") or (bulk_prop.data_files.get("bulk") if bulk_prop else None)
    if not single or not bulk:
        raise InputFileError(document.path, prop.line, "sublimation requires bulk and single data files")
    single_path = Path(_path(document, single))
    if not single_path.exists():
        raise InputFileError(document.path, prop.line, f"single-molecule data not found: {single_path}")
    single_summary = inspect_lammps_data(single_path)
    molecule_atoms = int(prop.settings.get(
        "molecule_atoms",
        single_summary.declared_counts.get("atoms", sum(item.atom_count for item in single_summary.atom_types)),
    ))
    targets = _target_document(document, prop)
    target_value = targets.get("esub_proxy", {}).get("value", float("nan"))
    config["sublimation"] = {
        "enabled": prop.fitted,
        "data_files": {"bulk": _path(document, bulk), "single": str(single_path)},
        "inputs": {
            "bulk": _resource("lammps/inputs/molecule/in.sublimation.bulk"),
            "single": _resource("lammps/inputs/molecule/in.sublimation.single"),
        },
        "molecule_atoms": molecule_atoms,
        "target_temperature": float(prop.settings.get("temperature", 298.15)),
        "target_kj_mol": target_value,
        "thermal_correction_kj_mol": 0.0,
        "tolerance_kj_mol": targets.get("esub_proxy", {}).get("tolerance", 10.0),
        "cutoff": float(prop.settings.get("cutoff", config["lammps"]["cutoff"])),
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
    config["adsorption"] = {
        "enabled": prop.fitted,
        "data_files": {
            "complex": _path(document, prop.data_files["complex"]),
            "slab": _path(document, prop.data_files["slab"]),
            "mol": _path(document, prop.data_files["molecule"]),
        },
        "inputs": {
            "complex": _resource("lammps/inputs/adsorption/in.complex"),
            "slab": _resource("lammps/inputs/adsorption/in.slab"),
            "mol": _resource("lammps/inputs/adsorption/in.mol"),
        },
        "metal": {"label": prop.settings.get("metal", "Au")},
        "md": {
            "cutoff": float(prop.settings.get("cutoff", 7.0)),
            "timestep": float(prop.settings.get("timestep", 1.0)),
            "temp": float(prop.settings.get("temperature", 300.0)),
            "seed": int(prop.settings.get("seed", 101)),
            "equil_steps": int(prop.settings.get("equilibration", 20000)),
            "prod_steps": int(prop.settings.get("production", 20000)),
            "kspace_accuracy": config["charge"]["kspace_accuracy"],
            "protocol": "minimize",
        },
    }
    return _target_document(document, prop)


def _apply_stage_settings(document: FFOptInput, config: dict[str, Any], stages: dict[str, Any]) -> None:
    bo_map = {
        "method": "method", "initial_points": "n_initial", "rounds": "n_bo_iterations",
        "max_rounds": "n_bo_iterations", "batch_size": "batch_size",
        "random_seed": "random_seed",
    }
    sample_map = {
        "points": "n_points", "centers": "elite_centers", "elite_centers": "elite_centers",
        "seeds": "seeds", "radii": "radii", "global_fraction": "global_fraction",
        "design_seed": "design_seed",
        "center_selection": "center_selection",
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
    }
    maps = {"bo": bo_map, "sample": sample_map, "nn": nn_map, "al": al_map}
    destinations = {
        "bo": config["optimization"], "sample": stages.setdefault("sample", {}),
        "nn": config["nn"], "al": config["active_learning"],
    }
    for stage, settings in document.stage_settings.items():
        if stage == "validate":
            allowed = {
                "trajectory", "require_tolerances", "objective_max",
                "max_error_percent",
            }
            unknown = set(settings) - allowed
            if unknown:
                raise InputFileError(document.path, 1, f"unknown validate setting(s): {sorted(unknown)}")
            if "trajectory" in settings and str(settings["trajectory"]).lower() != "final":
                raise InputFileError(document.path, 1, "validate trajectory currently supports final")
            if "require_tolerances" in settings and not isinstance(
                settings["require_tolerances"], bool
            ):
                raise InputFileError(
                    document.path, 1,
                    "validate require_tolerances must be yes or no",
                )
            for key in ("objective_max", "max_error_percent"):
                if key in settings and float(settings[key]) <= 0.0:
                    raise InputFileError(document.path, 1, f"validate {key} must be positive")
            config["validation"].update(settings)
            continue
        mapping = maps[stage]
        destination = destinations[stage]
        for key, value in settings.items():
            if key == "early_stop" and stage == "bo":
                destination.setdefault("early_stop", {}).update(value)
                continue
            if key == "stability_audit" and stage == "bo":
                if not isinstance(value, bool):
                    raise InputFileError(
                        document.path, 1, "BO stability_audit must be on or off"
                    )
                destination.setdefault("stability_audit", {})["enabled"] = value
                continue
            if key == "acquisition" and stage == "al":
                if str(value).lower() != "uncertainty":
                    raise InputFileError(document.path, 1, "AL acquisition currently supports uncertainty")
                continue
            if key not in mapping:
                raise InputFileError(document.path, 1, f"unknown {stage} setting {key!r}")
            mapped = mapping[key]
            list_settings = {
                "sample": {"seeds", "radii"},
                "nn": {"hidden_layers"},
                "al": {"local_radii"},
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
            document.path, 1,
            "BO method must be auto, gp, turbo, or saasbo",
        )
    for key in ("n_initial", "n_bo_iterations", "batch_size"):
        value = optimization.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise InputFileError(document.path, 1, f"BO {key} must be a positive integer")
    early_stop = optimization.get("early_stop", {})
    unknown_early = set(early_stop) - {"enabled", "patience", "min_improvement"}
    if unknown_early:
        raise InputFileError(
            document.path, 1,
            f"unknown BO early_stop setting(s): {sorted(unknown_early)}",
        )
    if not isinstance(early_stop.get("enabled", True), bool):
        raise InputFileError(document.path, 1, "BO early_stop enabled must be on or off")
    if int(early_stop.get("patience", 1)) < 1:
        raise InputFileError(document.path, 1, "BO early_stop patience must be positive")
    if float(early_stop.get("min_improvement", 0.0)) < 0.0:
        raise InputFileError(document.path, 1, "BO early_stop min_improvement cannot be negative")

    sample = stages.setdefault("sample", {})
    for key in ("n_points", "elite_centers"):
        if int(sample.get(key, 0)) < 1:
            raise InputFileError(document.path, 1, f"sample {key} must be positive")
    if int(sample["elite_centers"]) > int(sample["n_points"]):
        raise InputFileError(document.path, 1, "sample centers cannot exceed points")
    if not sample.get("seeds"):
        raise InputFileError(document.path, 1, "sample seeds cannot be empty")
    radii = [float(value) for value in sample.get("radii", [])]
    if not radii or any(value <= 0.0 or value > 1.0 for value in radii):
        raise InputFileError(document.path, 1, "sample radii must be in (0, 1]")
    global_fraction = float(sample.get("global_fraction", 0.0))
    if not 0.0 <= global_fraction <= 1.0:
        raise InputFileError(document.path, 1, "sample global_fraction must be in [0, 1]")
    if str(sample.get("center_selection", "diverse")) not in {"top", "diverse"}:
        raise InputFileError(document.path, 1, "sample center_selection must be top or diverse")

    nn = config["nn"]
    for key in ("ensemble_size", "max_epochs", "batch_size"):
        if int(nn.get(key, 0)) < 1:
            raise InputFileError(document.path, 1, f"NN {key} must be positive")
    val_fraction = float(nn.get("val_fraction", 0.0))
    test_fraction = float(nn.get("test_fraction", 0.0))
    if val_fraction < 0.0 or test_fraction < 0.0 or val_fraction + test_fraction >= 1.0:
        raise InputFileError(
            document.path, 1,
            "NN validation_fraction and test_fraction must be non-negative and sum to < 1",
        )

    active_learning = config["active_learning"]
    for key in ("n_rounds", "n_candidates_per_round", "n_candidate_pool"):
        if int(active_learning.get(key, 0)) < 1:
            raise InputFileError(document.path, 1, f"AL {key} must be positive")
    if int(active_learning["n_candidate_pool"]) < int(active_learning["n_candidates_per_round"]):
        raise InputFileError(document.path, 1, "AL candidate_pool cannot be smaller than candidates")


def compile_input(document: FFOptInput) -> CompiledInput:
    """Compile a parsed input file without reading any legacy project YAML."""
    primary = _primary_data(document)
    atom_types, dimensions, type_counts = _compile_atom_types(document, primary)
    if dimensions < 1 and any(stage in {"bo", "sample", "nn", "al"} for stage in document.workflow):
        raise InputFileError(document.path, 1, "parameter constraints leave no free dimensions")
    params = document.parameters
    if params.mixing_epsilon != "geometric":
        raise InputFileError(
            document.path, 1,
            "the LAMMPS backend supports geometric epsilon mixing; choose sigma "
            "geometric (LAMMPS mix geometric) or arithmetic (LAMMPS mix arithmetic)",
        )
    derive_type = None
    if params.derive_charge is not None:
        derive_type = next(item.type_id for item in params.atom_types if item.label == params.derive_charge)

    config = deep_merge(method_defaults(), {
        "manifest": {
            "system_name": document.project,
            "crystal_structure": "molecular",
            "surface_facet": "none",
            "data_files": {"bulk": str(primary)},
        },
        "pair_params": {
            "mixing_rule": params.mixing_sigma,
            "explicit_pairs": [],
            "derived_params": [],
        },
        "charge": {
            "enabled": True,
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

    fitted: list[str] = []
    validation_only: list[str] = []
    bulk_prop = _property(document, "bulk")
    for prop in document.properties:
        if prop.name == "bulk":
            targets = _compile_bulk(document, prop, config)
        elif prop.name == "sublimation":
            targets = _compile_sublimation(document, prop, config, bulk_prop)
        elif prop.name == "adsorption":
            targets = _compile_adsorption(document, prop, config)
        else:
            raise InputFileError(document.path, prop.line, "surface input compilation is not implemented yet")
        config["validation"]["property_units"].update(PROPERTY_OUTPUT_UNITS[prop.name])
        config["validation"]["property_units"].update({
            name: info["unit"] for name, info in targets.items()
        })
        if prop.fitted:
            fitted.append(prop.name)
            config["targets"].update(targets)
            config["property_evaluators"][prop.name] = {"enabled": True}
            if prop.name == "sublimation":
                config["sublimation"]["enabled"] = True
                config["property_evaluators"]["bulk"] = {"enabled": True}
            elif prop.name == "adsorption":
                config["adsorption"]["enabled"] = True
        else:
            validation_only.append(prop.name)
            config["validation"]["property_evaluators"][prop.name] = {"enabled": True}
            if prop.name == "sublimation":
                config["validation"]["property_evaluators"]["bulk"] = {"enabled": True}
                config["validation"]["property_settings"]["sublimation"] = {"enabled": True}
            elif prop.name == "adsorption":
                config["validation"]["property_settings"]["adsorption"] = {"enabled": True}
    # Every fitted property is also recomputed by final validation.
    for name in fitted:
        config["validation"]["property_evaluators"].setdefault(name, {"enabled": True})

    stages: dict[str, Any] = {
        "sample": {
            "n_points": min(2500, max(800, 50 * max(dimensions, 1))),
            "seeds": [101, 202, 303],
            "elite_centers": min(24, max(4, dimensions)),
            "center_selection": "diverse",
            "center_pool_multiplier": 5,
            "radii": [0.01, 0.025, 0.05],
            "global_fraction": 0.10,
            "design_seed": 20260727,
        }
    }
    _apply_stage_settings(document, config, stages)
    config.setdefault("workflow", {})["active_properties"] = list(config["targets"])

    project_data = {
        "schema_version": 1,
        "project": {
            "name": document.project,
            "default_machine": "local",
            "run_root": f"runs/{document.project}",
        },
        "stages": stages,
        "pipeline": {
            "stages": list(document.workflow),
            "audit": {"top_k": 8, "seeds": [101, 202, 303]},
        },
    }
    return CompiledInput(
        document=document,
        config=config,
        project_data=project_data,
        dimensions=dimensions,
        fitted_properties=tuple(fitted),
        validation_properties=tuple(validation_only),
    )
