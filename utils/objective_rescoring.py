"""Recompute force-field objectives from saved calculated properties."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd


def objective_provenance(targets: Mapping[str, Mapping]) -> dict[str, float | str]:
    """Columns that make the numerical definition of an objective auditable."""
    values: dict[str, float | str] = {
        "_objective_formula": "weighted_relative_rmse_v1",
    }
    for name, info in targets.items():
        values[f"_objective_target_{name}"] = float(info["value"])
        values[f"_objective_weight_{name}"] = float(info.get("weight", 1.0))
    return values


def validate_objective_provenance(
    frame: pd.DataFrame,
    targets: Mapping[str, Mapping],
    source: str,
) -> None:
    """Reject scored CSVs that explicitly declare a different objective."""
    mismatches: list[str] = []
    for name, info in targets.items():
        for kind, expected in (
            ("target", float(info["value"])),
            ("weight", float(info.get("weight", 1.0))),
        ):
            column = f"_objective_{kind}_{name}"
            if column not in frame.columns:
                continue
            values = pd.to_numeric(frame[column], errors="coerce").dropna().unique()
            if len(values) and not np.allclose(values, expected, rtol=0.0, atol=1.0e-10):
                mismatches.append(
                    f"{name} {kind}: CSV={values.tolist()} config={expected}"
                )
    if mismatches:
        detail = "; ".join(mismatches)
        raise ValueError(
            f"Objective provenance mismatch in {source}: {detail}. "
            "Run rescore_results.py with the intended recipe before NN/AL use."
        )


def active_targets(config: Mapping) -> dict[str, dict]:
    """Return weighted targets that are enabled by the expanded recipe."""
    selected: dict[str, dict] = {}
    for name, info in config.get("targets", {}).items():
        if float(info.get("weight", 1.0)) <= 0.0:
            continue
        if name == "ead" and not config.get("adsorption", {}).get("enabled", False):
            continue
        if name == "esub_proxy" and not config.get("sublimation", {}).get(
            "enabled", False
        ):
            continue
        if name == "surf_energy":
            surface_enabled = config.get("surface", {}).get(
                "enabled",
                config.get("lammps", {}).get(
                    "compute_surface", config.get("compute_surface", False)
                ),
            )
            if not surface_enabled:
                continue
        selected[name] = dict(info)
    return selected


def rescore_frame(frame: pd.DataFrame, targets: Mapping[str, Mapping]) -> pd.DataFrame:
    """Return a copy with errors and weighted-RMSE objective recomputed."""
    result = frame.copy()
    if "objective" in result.columns and "objective_original" not in result.columns:
        result["objective_original"] = result["objective"]

    weighted_sq = np.zeros(len(result), dtype=float)
    valid = np.ones(len(result), dtype=bool)
    weight_sum = 0.0
    for name, info in targets.items():
        column = f"calc_{name}"
        if column not in result.columns:
            valid[:] = False
            continue
        values = pd.to_numeric(result[column], errors="coerce").to_numpy(dtype=float)
        target = float(info["value"])
        weight = float(info.get("weight", 1.0))
        relative = np.abs(values - target) / (abs(target) + 1.0e-10)
        result[f"error_{name}"] = 100.0 * relative
        weighted_sq += weight * relative**2
        weight_sum += weight
        valid &= np.isfinite(values)

    objective = np.full(len(result), np.nan, dtype=float)
    if weight_sum > 0.0:
        objective[valid] = np.sqrt(weighted_sq[valid] / weight_sum)
    if "success" in result.columns:
        success = result["success"].astype(str).str.lower().isin(["true", "1"])
        objective[~success.to_numpy()] = np.nan
    result["objective"] = objective
    for column, value in objective_provenance(targets).items():
        result[column] = value
    return result


def aggregate_replicates(
    replicates: pd.DataFrame,
    targets: Mapping[str, Mapping],
    parameter_columns: Sequence[str],
    source_summary: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build mean+population-std robust summaries from rescored replicates."""
    rescored = rescore_frame(replicates, targets)
    key = "candidate_id" if "candidate_id" in rescored.columns else "candidate_rank"
    rows: list[dict] = []
    source_by_key = None
    if source_summary is not None and key in source_summary.columns:
        source_by_key = source_summary.drop_duplicates(key).set_index(key)

    for candidate, group in rescored.groupby(key, sort=False):
        successful = group.loc[np.isfinite(group["objective"])].copy()
        if source_by_key is not None and candidate in source_by_key.index:
            base = source_by_key.loc[candidate]
            if isinstance(base, pd.DataFrame):
                base = base.iloc[0]
            row = base.to_dict()
        else:
            row = {key: candidate}
            first = group.iloc[0]
            for column in parameter_columns:
                if column in first.index:
                    row[column] = first[column]

        n_seeds = len(group)
        n_success = len(successful)
        row.update(
            {
                key: candidate,
                "success": n_success == n_seeds,
                "n_success": n_success,
                "n_seeds": n_seeds,
                "failure_rate": 1.0 - n_success / max(n_seeds, 1),
            }
        )
        row.update(objective_provenance(targets))
        if n_success:
            values = successful["objective"].to_numpy(dtype=float)
            row["objective_mean"] = float(values.mean())
            row["objective_std"] = float(values.std(ddof=0))
            row["objective"] = row["objective_mean"] + row["objective_std"]
            for name, info in targets.items():
                prop_values = pd.to_numeric(
                    successful[f"calc_{name}"], errors="coerce"
                ).to_numpy(dtype=float)
                row[f"calc_{name}"] = float(np.nanmean(prop_values))
                row[f"std_{name}"] = float(np.nanstd(prop_values, ddof=0))
                row[f"error_{name}"] = 100.0 * abs(
                    row[f"calc_{name}"] - float(info["value"])
                ) / (abs(float(info["value"])) + 1.0e-10)
        else:
            row["objective_mean"] = np.nan
            row["objective_std"] = np.nan
            row["objective"] = np.nan
            for name in targets:
                row[f"calc_{name}"] = np.nan
                row[f"std_{name}"] = np.nan
                row[f"error_{name}"] = np.nan
        rows.append(row)

    return pd.DataFrame(rows).sort_values("objective", na_position="last")
