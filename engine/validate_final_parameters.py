#!/usr/bin/env python3
"""Run one complete trajectory-producing validation from explicit parameters."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .config_loader import deep_merge, load_config
from .lammps_interface import LAMMPSRunner
from .optimizer import ForceFieldOptimizer


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
    try:
        return ForceFieldOptimizer._build_param_space(config)
    except ValueError as exc:
        if "No free BO parameters found" not in str(exc):
            raise
        return []


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

    rows = []
    for name, value in result.properties.items():
        target = config.get("targets", {}).get(name, {})
        reference = target.get("value")
        rows.append({
            "property": name,
            "value": float(value),
            "reference": reference,
            "unit": target.get(
                "unit", validation.get("property_units", {}).get(name, "")
            ),
            "error_percent": result.per_property_error.get(name),
        })
    pd.DataFrame(rows).to_csv(
        args.output_dir / "computed_properties.csv", index=False
    )
    objective = float(result.objective) if config.get("targets") else None
    summary = {
        "success": True,
        "objective": objective,
        "config": str(Path(args.config).resolve()),
        "parameters": parameter_source,
        "properties": result.properties,
        "per_property_error": result.per_property_error,
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


if __name__ == "__main__":
    main()
