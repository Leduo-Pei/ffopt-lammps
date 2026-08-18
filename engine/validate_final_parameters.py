#!/usr/bin/env python3
"""Run one complete trajectory-producing validation from explicit parameters."""

from __future__ import annotations

import argparse
import json
import math
import shutil
from pathlib import Path

import pandas as pd

from .config_loader import deep_merge, load_config
from .lammps_interface import LAMMPSRunner
from .parameter_space import build_parameter_space


def _load_parameters(path: Path) -> dict[str, float]:
    """Accept final summaries, NN/AL result JSON, or BO text exports."""
    if path.suffix.lower() == ".txt":
        values: dict[str, float] = {}
        for raw in path.read_text(encoding="utf-8-sig").splitlines():
            body = raw.split("#", 1)[0].strip()
            if not body or "=" not in body:
                continue
            name, value = body.split("=", 1)
            values[name.strip()] = float(value.strip())
        return values

    with path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    if "raw_free_parameters" in document:
        raw = document["raw_free_parameters"]
    elif isinstance(document.get("best"), dict):
        raw = document["best"].get("params", document["best"])
    elif isinstance(document.get("best_lammps"), dict):
        raw = document["best_lammps"].get("params", document["best_lammps"])
    else:
        raise ValueError(
            f"Unsupported parameter document {path}; expected raw_free_parameters "
            "or a best.params record"
        )
    return {key: float(value) for key, value in raw.items()}


def _parameter_space(config: dict) -> list[tuple[str, float, float]]:
    """Return the free space, allowing a fully fixed validation-only input."""
    return build_parameter_space(config, require_nonempty=False)


def _initial_parameters(config: dict) -> dict[str, float]:
    """Extract the user-entered initial value for every free parameter."""
    names = {name for name, _, _ in _parameter_space(config)}
    values: dict[str, float] = {}
    for atom_type in config["atom_types"]:
        label = atom_type["label"]
        for parameter, spec in atom_type["params"].items():
            name = f"{label}_{parameter}"
            if name not in names or not isinstance(spec, dict):
                continue
            initial = spec.get("init")
            if initial is None:
                initial = (float(spec["min"]) + float(spec["max"])) / 2.0
            values[name] = float(initial)
    for pair in config.get("pair_params", {}).get("explicit_pairs", []):
        type_1, type_2 = pair["types"]
        for parameter in ("epsilon", "sigma"):
            name = f"cross_{type_1}_{type_2}_{parameter}"
            if name not in names:
                continue
            spec = pair[parameter]
            initial = spec.get("init")
            if initial is None:
                initial = (float(spec["min"]) + float(spec["max"])) / 2.0
            values[name] = float(initial)
    missing = sorted(names - values.keys())
    if missing:
        raise ValueError(f"Initial values are missing from the config: {missing}")
    return values


def _assess_acceptance(
    config: dict,
    *,
    objective: float | None,
    properties: dict[str, float],
    percent_errors: dict[str, float],
) -> dict:
    """Evaluate optional final-validation thresholds without running LAMMPS."""
    def finite(value: object) -> bool:
        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    validation = config.get("validation", {})
    active_targets = {
        name: target
        for name, target in config.get("targets", {}).items()
        if float(target.get("weight", 1.0)) > 0.0
    }
    reasons: list[str] = []
    missing_properties = [
        name
        for name in active_targets
        if name not in properties or not finite(properties[name])
    ]
    if missing_properties:
        reasons.append(
            "missing or non-finite fitted properties: "
            + ", ".join(missing_properties)
        )

    tolerance_checks: dict[str, dict[str, float | bool]] = {}
    missing_tolerances: list[str] = []
    for name, target in active_targets.items():
        if target.get("tolerance") is None:
            missing_tolerances.append(name)
            continue
        if name in missing_properties:
            continue
        absolute_error = abs(float(properties[name]) - float(target["value"]))
        tolerance = float(target["tolerance"])
        tolerance_checks[name] = {
            "absolute_error": absolute_error,
            "tolerance": tolerance,
            "passed": absolute_error <= tolerance,
        }

    require_tolerances = bool(validation.get("require_tolerances", False))
    if require_tolerances:
        if missing_tolerances:
            reasons.append(
                "required target tolerance is not defined: "
                + ", ".join(missing_tolerances)
            )
        failed = [name for name, item in tolerance_checks.items() if not item["passed"]]
        if failed:
            reasons.append("target tolerance exceeded: " + ", ".join(failed))

    objective_max = validation.get("objective_max")
    if objective_max is not None and (
        objective is None
        or not finite(objective)
        or objective > float(objective_max)
    ):
        reasons.append(f"objective {objective} exceeds {float(objective_max):g}")

    max_error_percent = validation.get("max_error_percent")
    missing_percent_errors = [
        name
        for name in active_targets
        if name not in percent_errors
        or not finite(percent_errors[name])
    ]
    finite_percent_errors = [
        float(percent_errors[name])
        for name in active_targets
        if name not in missing_percent_errors
    ]
    observed_max_error = max(finite_percent_errors, default=None)
    if max_error_percent is not None and missing_percent_errors:
        reasons.append(
            "missing or non-finite percent errors: "
            + ", ".join(missing_percent_errors)
        )
    if (
        max_error_percent is not None
        and observed_max_error is not None
        and observed_max_error > float(max_error_percent)
    ):
        reasons.append(
            f"maximum percent error {observed_max_error:.6g} exceeds "
            f"{float(max_error_percent):g}"
        )
    return {
        "passed": not reasons,
        "require_tolerances": require_tolerances,
        "objective_max": objective_max,
        "max_error_percent": max_error_percent,
        "observed_max_error_percent": observed_max_error,
        "tolerances": tolerance_checks,
        "reasons": reasons,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parameters", type=Path)
    source.add_argument(
        "--initial",
        action="store_true",
        help="Validate the initial type values compiled from ffopt.in.",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output exists: {args.output_dir}; pass --overwrite to replace it"
            )
        shutil.rmtree(args.output_dir)
    args.output_dir.mkdir(parents=True)

    config = load_config(args.config)
    validation = config.get("validation", {})
    overrides = {
        "property_evaluators": validation.get("property_evaluators", {}),
    }
    overrides.update(validation.get("property_settings", {}))
    config = deep_merge(config, overrides)
    if args.initial:
        params = _initial_parameters(config)
        parameter_source = "initial force-field values from ffopt.in"
    else:
        params = _load_parameters(args.parameters)
        parameter_source = str(args.parameters.resolve())
    expected = [item[0] for item in _parameter_space(config)]
    missing = [name for name in expected if name not in params]
    if missing:
        raise ValueError(f"Final parameter file is missing: {missing}")

    runner = LAMMPSRunner(config)
    results = runner.evaluate_batch(
        [params], str(args.output_dir), save_traj=True
    )
    result = results[0]
    if not result.success:
        raise RuntimeError(result.error_msg)

    resolved = runner._resolve_params(params)
    atom_rows = []
    for atom_type in config["atom_types"]:
        label = atom_type["label"]
        atom_rows.append({
            "type": int(atom_type["type"]),
            "label": label,
            "epsilon_kcal_mol": float(resolved[f"{label}_epsilon"]),
            "sigma_angstrom": float(resolved[f"{label}_sigma"]),
            "charge_e": float(resolved[f"{label}_charge"]),
        })
    pd.DataFrame(atom_rows).to_csv(
        args.output_dir / "final_atom_parameters.csv", index=False
    )
    generated_pair_file = Path(
        runner._build_pair_coeffs(resolved, str(args.output_dir))
    )
    pair_text = generated_pair_file.read_text(encoding="ascii").replace(
        "# pair_coeffs.lmp -- generated by FFOpt-LAMMPS\n"
        "# DO NOT EDIT: overwritten before each LAMMPS call\n",
        "# Complete FFOpt final parameters; include after read_data.\n",
        1,
    )
    (args.output_dir / "final_parameters.lammps").write_text(
        pair_text, encoding="ascii", newline="\n"
    )
    generated_pair_file.unlink()
    parameter_document = {
        "source": parameter_source,
        "raw_free_parameters": params,
        "resolved_parameters": resolved,
        "atom_types": atom_rows,
    }
    with (args.output_dir / "final_parameters.json").open(
        "w", encoding="utf-8"
    ) as handle:
        json.dump(parameter_document, handle, indent=2)

    rows = []
    for name, value in result.properties.items():
        target = config.get("targets", {}).get(name, {})
        reference = target.get("value")
        tolerance = target.get("tolerance")
        absolute_error = (
            abs(float(value) - float(reference))
            if reference is not None else None
        )
        within_tolerance = (
            absolute_error <= float(tolerance)
            if absolute_error is not None and tolerance is not None else None
        )
        rows.append({
            "property": name,
            "value": float(value),
            "reference": reference,
            "unit": target.get(
                "unit", validation.get("property_units", {}).get(name, "")
            ),
            "error_percent": result.per_property_error.get(name),
            "absolute_error": absolute_error,
            "tolerance": tolerance,
            "within_tolerance": within_tolerance,
        })
    pd.DataFrame(rows).to_csv(
        args.output_dir / "computed_properties.csv", index=False
    )
    objective = float(result.objective) if config.get("targets") else None
    acceptance = _assess_acceptance(
        config,
        objective=objective,
        properties=result.properties,
        percent_errors=result.per_property_error,
    )
    accepted = bool(acceptance["passed"])
    summary = {
        "success": True,
        "objective": objective,
        "config": str(Path(args.config).resolve()),
        "parameters": parameter_source,
        "parameter_artifact": str(
            (args.output_dir / "final_parameters.json").resolve()
        ),
        "properties": result.properties,
        "per_property_error": result.per_property_error,
        "acceptance": acceptance,
    }
    with open(args.output_dir / "validation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"success   : {result.success}")
    print(f"objective : {objective:.9f}" if objective is not None else "objective : n/a (no targets)")
    for row in rows:
        print(
            f"{row['property']:24s} {row['value']: .9f} "
            f"{row['unit']}"
        )
    print(f"output    : {args.output_dir.resolve()}")
    print(f"acceptance: {'PASS' if accepted else 'FAIL'}")
    if not accepted:
        for reason in acceptance["reasons"]:
            print(f"  - {reason}")
        raise RuntimeError("Final validation did not meet the configured acceptance criteria")


if __name__ == "__main__":
    main()
