#!/usr/bin/env python3
"""Correct stale objectives in saved BO/sampling/AL CSV files without LAMMPS."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from .config_loader import load_config
from .parameter_space import build_parameter_space
from utils.objective_rescoring import (
    active_targets,
    aggregate_replicates,
    rescore_frame,
)


REPLICATE_NAMES = {
    "stability_replicates.csv",
    "local_replicates.csv",
}
RESULT_NAMES = {
    "all_results.csv",
    "local_results.csv",
    "stable_results.csv",
    "stability_replicates.csv",
    "local_replicates.csv",
}


def matching_summary(path: Path) -> Path | None:
    if path.name == "local_replicates.csv":
        candidate = path.with_name("local_results.csv")
    else:
        candidate = path.with_name("stable_results.csv")
    return candidate if candidate.exists() else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    config = load_config(args.config)
    targets = active_targets(config)
    parameter_columns = [
        item[0] for item in build_parameter_space(config)
    ]
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates: list[pd.DataFrame] = []
    exact_summaries: list[pd.DataFrame] = []
    sampling_summary: pd.DataFrame | None = None
    processed: list[dict] = []
    csv_paths = sorted(
        path for path in input_root.rglob("*.csv")
        if path.name in RESULT_NAMES and output_dir not in path.parents
    )
    replicate_paths = [path for path in csv_paths if path.name in REPLICATE_NAMES]

    for path in replicate_paths:
        relative = path.relative_to(input_root)
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = pd.read_csv(path, low_memory=False)
        rescored = rescore_frame(raw, targets)
        rescored.to_csv(destination, index=False)

        summary_path = matching_summary(path)
        source_summary = (
            pd.read_csv(summary_path, low_memory=False)
            if summary_path is not None else None
        )
        summary = aggregate_replicates(
            raw, targets, parameter_columns, source_summary=source_summary
        )
        # A historical stability summary may mark candidates unsuccessful not
        # because LAMMPS failed, but because objective/property seed spread
        # exceeded the configured audit tolerances. Re-aggregation must keep
        # that acceptance decision; otherwise noisy candidates silently become
        # stable merely because all replicate processes exited successfully.
        if path.name == "stability_replicates.csv" and source_summary is not None:
            key = (
                "candidate_id"
                if "candidate_id" in summary.columns
                else "candidate_rank"
            )
            source_key = (
                "candidate_id"
                if "candidate_id" in source_summary.columns
                else "candidate_rank"
            )
            if "stable" in source_summary.columns:
                accepted_mask = source_summary["stable"].astype(str).str.lower().isin(
                    ["true", "1"]
                )
            elif "success" in source_summary.columns:
                accepted_mask = source_summary["success"].astype(str).str.lower().isin(
                    ["true", "1"]
                )
            else:
                accepted_mask = pd.Series(True, index=source_summary.index)
            accepted = set(source_summary.loc[accepted_mask, source_key])
            summary["replicates_complete"] = summary["success"]
            summary["stable"] = summary[key].isin(accepted)
            summary["success"] = summary["success"] & summary["stable"]
        exact_summaries.append(summary.copy())
        if path.name == "local_replicates.csv":
            sampling_summary = summary.copy()
        summary_name = (
            "local_results.csv"
            if path.name == "local_replicates.csv"
            else "stable_results.csv"
        )
        summary_destination = destination.with_name(summary_name)
        summary.to_csv(summary_destination, index=False)

        fully_successful = summary["success"].astype(str).str.lower().isin(
            ["true", "1"]
        )
        ranked = summary.loc[
            fully_successful & np.isfinite(summary["objective"])
        ].copy()
        if not ranked.empty:
            ranked.insert(0, "source_file", str(relative).replace("\\", "/"))
            ranked.insert(1, "source_candidate_key", (
                ranked["candidate_id"]
                if "candidate_id" in ranked.columns else ranked["candidate_rank"]
            ))
            candidates.append(ranked)
        processed.append({
            "source": str(relative).replace("\\", "/"),
            "rows": len(raw),
            "summary_rows": len(summary),
        })

    exact_lookup = pd.concat(exact_summaries, ignore_index=True, sort=False)
    exact_lookup["_parameter_key"] = (
        exact_lookup[parameter_columns].round(12).astype(str).agg("|".join, axis=1)
    )
    exact_lookup = (
        exact_lookup.sort_values("objective")
        .drop_duplicates("_parameter_key")
        .set_index("_parameter_key")
    )
    sampling_lookup = None
    if sampling_summary is not None:
        sampling_summary["_parameter_key"] = (
            sampling_summary[parameter_columns]
            .round(12).astype(str).agg("|".join, axis=1)
        )
        sampling_lookup = sampling_summary.set_index("_parameter_key")

    summary_sources = {
        matching_summary(path).resolve()
        for path in replicate_paths if matching_summary(path) is not None
    }
    for path in csv_paths:
        if path.name in REPLICATE_NAMES or path.resolve() in summary_sources:
            continue
        relative = path.relative_to(input_root)
        destination = output_dir / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        raw = pd.read_csv(path, low_memory=False)
        if all(f"calc_{name}" in raw.columns for name in targets):
            corrected = rescore_frame(raw, targets)
            if (
                path.name in {"stable_results.csv", "local_results.csv"}
                and all(name in corrected.columns for name in parameter_columns)
            ):
                lookup = exact_lookup
                if sampling_lookup is not None and "bo_plus_sampling_1200" in path.parts:
                    lookup = sampling_lookup
                corrected["_parameter_key"] = (
                    corrected[parameter_columns]
                    .round(12).astype(str).agg("|".join, axis=1)
                )
                replacement_columns = [
                    column for column in lookup.columns
                    if column in corrected.columns and column not in parameter_columns
                ]
                matched = corrected["_parameter_key"].isin(lookup.index)
                if matched.any():
                    replacement = lookup.loc[
                        corrected.loc[matched, "_parameter_key"],
                        replacement_columns,
                    ]
                    replacement.index = corrected.index[matched]
                    corrected.loc[matched, replacement_columns] = replacement
                corrected = corrected.drop(columns="_parameter_key")
            corrected.to_csv(destination, index=False)
            processed.append({
                "source": str(relative).replace("\\", "/"),
                "rows": len(raw),
                "summary_rows": None,
            })

    if not candidates:
        raise RuntimeError("No complete replicate datasets were found")

    ranking = pd.concat(candidates, ignore_index=True, sort=False)
    ranking = ranking.sort_values("objective").reset_index(drop=True)
    parameter_key = ranking[parameter_columns].round(12).astype(str).agg("|".join, axis=1)
    ranking = ranking.loc[~parameter_key.duplicated()].reset_index(drop=True)
    ranking.insert(0, "corrected_rank", np.arange(1, len(ranking) + 1))
    ranking["candidate_id"] = np.arange(1, len(ranking) + 1)
    ranking.to_csv(output_dir / "stable_results.csv", index=False)
    ranking.to_csv(output_dir / "robust_candidate_ranking.csv", index=False)

    best = ranking.iloc[0]
    properties = []
    for name, info in targets.items():
        target = float(info["value"])
        mean = float(best[f"calc_{name}"])
        properties.append({
            "property": name,
            "calculated_mean": mean,
            "seed_std": float(best[f"std_{name}"]),
            "target": target,
            "relative_error_percent": 100.0 * (mean - target) / abs(target),
            "weight": float(info.get("weight", 1.0)),
            "unit": info.get("unit", ""),
        })
    pd.DataFrame(properties).to_csv(
        output_dir / "best_corrected_properties.csv", index=False
    )
    best_parameters = pd.DataFrame([
        {"parameter": name, "value": float(best[name])}
        for name in parameter_columns
    ])
    best_parameters.to_csv(output_dir / "best_corrected_parameters.csv", index=False)

    report = {
        "config": str(Path(args.config).resolve()),
        "input_root": str(input_root),
        "target_values": {name: float(info["value"]) for name, info in targets.items()},
        "selection_rule": "minimum three-seed robust objective = mean + population std",
        "best_source_file": str(best["source_file"]),
        "best_source_candidate_key": int(best["source_candidate_key"]),
        "best_objective_mean": float(best["objective_mean"]),
        "best_objective_std": float(best["objective_std"]),
        "best_robust_objective": float(best["objective"]),
        "processed_files": processed,
    }
    (output_dir / "correction_summary.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
