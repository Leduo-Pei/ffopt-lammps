#!/usr/bin/env python3
"""Run one complete trajectory-producing validation from explicit parameters."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import pandas as pd

from .config_loader import load_config
from .lammps_interface import LAMMPSRunner
from .optimizer import ForceFieldOptimizer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--parameters", required=True, type=Path)
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
    with open(args.parameters, encoding="utf-8") as fh:
        document = json.load(fh)
    params = {
        key: float(value)
        for key, value in document["raw_free_parameters"].items()
    }
    expected = [
        item[0] for item in ForceFieldOptimizer._build_param_space(config)
    ]
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
            "unit": target.get("unit", ""),
            "error_percent": result.per_property_error.get(name),
        })
    pd.DataFrame(rows).to_csv(
        args.output_dir / "computed_properties.csv", index=False
    )
    summary = {
        "success": True,
        "objective": float(result.objective),
        "config": str(Path(args.config).resolve()),
        "parameters": str(args.parameters.resolve()),
        "properties": result.properties,
        "per_property_error": result.per_property_error,
    }
    with open(args.output_dir / "validation_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)

    print(f"success   : {result.success}")
    print(f"objective : {result.objective:.9f}")
    for row in rows:
        print(
            f"{row['property']:24s} {row['value']: .9f} "
            f"{row['unit']}"
        )
    print(f"output    : {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
