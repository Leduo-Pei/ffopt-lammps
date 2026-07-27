#!/usr/bin/env python3
"""Select the best robust AL audit row and export reproducible artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .config_loader import load_config
from .lammps_interface import LAMMPSRunner
from .optimizer import ForceFieldOptimizer
from utils.objective_rescoring import active_targets, aggregate_replicates


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--audit", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    runner = LAMMPSRunner(config)
    param_names = [
        item[0] for item in ForceFieldOptimizer._build_param_space(config)
    ]

    candidates = []
    targets = active_targets(config)
    for audit in args.audit:
        path = audit / "stable_results.csv"
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        replicate_path = audit / "stability_replicates.csv"
        if replicate_path.exists():
            frame = aggregate_replicates(
                pd.read_csv(replicate_path), targets, param_names,
                source_summary=frame,
            )
        row = frame.sort_values("objective").iloc[0].copy()
        row["audit_directory"] = str(audit.resolve())
        candidates.append(row)
    if not candidates:
        raise RuntimeError("No non-empty stable_results.csv files found")

    ranking = pd.DataFrame(candidates).sort_values("objective").reset_index(drop=True)
    best = ranking.iloc[0]
    raw_params = {name: float(best[name]) for name in param_names}
    resolved = runner._resolve_params(raw_params)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    ranking[[
        "audit_directory", "candidate_id", "center_objective",
        "objective_mean", "objective_std", "objective", "n_success", "n_seeds"
    ]].to_csv(args.output_dir / "audit_comparison.csv", index=False)

    properties = []
    for name, target in config["targets"].items():
        mean = float(best.get(f"calc_{name}", np.nan))
        std = float(best.get(f"std_{name}", np.nan))
        reference = float(target["value"])
        properties.append({
            "property": name,
            "reference": reference,
            "mean": mean,
            "std": std,
            "absolute_error": mean - reference,
            "relative_error_percent": 100.0 * (mean - reference) / abs(reference),
            "unit": target.get("unit", ""),
            "weight": float(target.get("weight", 1.0)),
        })
    pd.DataFrame(properties).to_csv(
        args.output_dir / "final_properties.csv", index=False
    )

    atom_rows = []
    for atom in config["atom_types"]:
        label = atom["label"]
        atom_rows.append({
            "type": int(atom["type"]),
            "label": label,
            "epsilon": float(resolved[f"{label}_epsilon"]),
            "sigma": float(resolved[f"{label}_sigma"]),
            "charge": float(resolved[f"{label}_charge"]),
        })
    atom_frame = pd.DataFrame(atom_rows)
    atom_frame.to_csv(args.output_dir / "final_atom_parameters.csv", index=False)

    with open(args.output_dir / "final_parameters.lammps", "w", encoding="ascii") as fh:
        for row in atom_rows:
            fh.write(
                f"{row['type']:3d}  {row['epsilon']:.10f}  "
                f"{row['sigma']:.10f}  {row['charge']:+.10f}  "
                f"# {row['label']}\n"
            )

    selected_replicates = Path(best["audit_directory"]) / "stability_replicates.csv"
    selected_seeds = []
    if selected_replicates.exists():
        replicate_frame = pd.read_csv(selected_replicates)
        if "candidate_id" in replicate_frame and "seed" in replicate_frame:
            selected_seeds = sorted(
                pd.to_numeric(
                    replicate_frame.loc[
                        replicate_frame["candidate_id"].astype(int).eq(
                            int(best["candidate_id"])
                        ),
                        "seed",
                    ],
                    errors="coerce",
                ).dropna().astype(int).unique().tolist()
            )

    summary = {
        "selection_rule": "minimum robust objective = audit-seed mean + std",
        "source_audit": str(best["audit_directory"]),
        "source_candidate_id": int(best["candidate_id"]),
        "single_seed_center_objective": float(best["center_objective"]),
        "objective_mean": float(best["objective_mean"]),
        "objective_std": float(best["objective_std"]),
        "robust_objective": float(best["objective"]),
        "n_success": int(best["n_success"]),
        "n_seeds": int(best["n_seeds"]),
        "seeds": selected_seeds,
        "raw_free_parameters": raw_params,
        "resolved_parameters": resolved,
    }
    with open(args.output_dir / "final_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
    with open(args.output_dir / "final_parameters.yaml", "w", encoding="utf-8") as fh:
        yaml.safe_dump(
            {"atom_types": atom_rows, "raw_free_parameters": raw_params},
            fh, sort_keys=False
        )

    print("Final robust AL result")
    print(f"  source : {best['audit_directory']}")
    print(f"  mean   : {best['objective_mean']:.9f}")
    print(f"  std    : {best['objective_std']:.9f}")
    print(f"  robust : {best['objective']:.9f}")
    print(f"  output : {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
