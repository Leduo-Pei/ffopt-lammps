"""Material-independent structural feasible-region coverage helpers.

The first optimization stage is a satisficing problem: once every declared
property lies inside its accepted band, smaller centre error is not considered
better.  These helpers turn exact property observations into continuous band
ratios and choose a reproducible mixture of feasible-interior, boundary,
surrogate-uncertainty, and global-novelty points.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
import pandas as pd

from .constraint_objectives import assess_constraint, summarize_constraints


_LATTICE = ("a", "b", "c")
_ANGLES = ("alpha", "beta", "gamma_ang")


def _truthy(values: pd.Series) -> pd.Series:
    return values.astype(str).str.strip().str.lower().isin(
        ("true", "1", "yes", "on")
    )


def structural_constraint_specs(config: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Compile exact structural gates from a runtime configuration.

    Material workflows use the role-aware elasticity gate declaration.  A
    target without a group gate falls back to its explicit absolute tolerance,
    which keeps the helper useful for other crystal families.
    """

    targets = config.get("targets", {})
    if not isinstance(targets, Mapping):
        raise ValueError("config.targets must be a mapping")
    elasticity = config.get("elasticity", {})
    selection = elasticity.get("selection", {}) if isinstance(elasticity, Mapping) else {}
    gates = selection.get("structural_gates", {}) if isinstance(selection, Mapping) else {}
    specs: list[dict[str, Any]] = []

    def append(names: Sequence[str], gate: Mapping[str, Any] | None) -> None:
        for name in names:
            raw = targets.get(name)
            if not isinstance(raw, Mapping) or "value" not in raw:
                continue
            if gate and "maximum_relative_error_percent" in gate:
                tolerance = float(gate["maximum_relative_error_percent"])
                mode = "relative_percent"
            elif gate and "maximum_absolute_error_degree" in gate:
                tolerance = float(gate["maximum_absolute_error_degree"])
                mode = "absolute"
            elif raw.get("tolerance") is not None:
                tolerance = float(raw["tolerance"])
                mode = "absolute"
            else:
                continue
            if not math.isfinite(tolerance) or tolerance <= 0.0:
                raise ValueError(f"Structural gate {name!r} requires a positive tolerance")
            specs.append({
                "name": name,
                "column": name,
                "target": float(raw["value"]),
                "tolerance": tolerance,
                "mode": mode,
            })

    append(_LATTICE, gates.get("lattice") if isinstance(gates, Mapping) else None)
    append(_ANGLES, gates.get("angles") if isinstance(gates, Mapping) else None)
    append(("density",), gates.get("density") if isinstance(gates, Mapping) else None)
    append(("surf_energy",), gates.get("surface") if isinstance(gates, Mapping) else None)

    covered = {item["name"] for item in specs}
    append(
        tuple(name for name in targets if name not in covered),
        None,
    )
    if not specs:
        raise ValueError(
            "feasible_coverage requires at least one target with a structural gate "
            "or explicit tolerance"
        )
    return specs


def assess_structural_properties(
    properties: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Return a flat, auditable structural constraint assessment."""

    constraints = []
    for spec in structural_constraint_specs(config):
        constraints.append(assess_constraint(
            spec["name"],
            properties.get(spec["column"]),
            spec["target"],
            spec["tolerance"],
            mode=spec["mode"],
        ))
    summary = summarize_constraints(constraints)
    result: dict[str, Any] = {
        "structural_feasible": bool(summary["feasible"]),
        "structural_constraint_violation": float(summary["constraint_violation"]),
        "structural_band_max_ratio": float(summary["maximum_band_ratio"]),
        "structural_minimum_margin": float(summary["minimum_margin"]),
        "structural_limiting_constraint": str(summary["limiting_constraint"]),
        "structural_failed_constraints": ";".join(summary["failed_constraints"]),
        # Feasible points intentionally tie at zero.  Coverage and diversity,
        # rather than centre error, select among them.
        "acquisition_objective": float(summary["constraint_violation"]),
    }
    for name, item in summary["constraints"].items():
        result[f"structural_observed_{name}"] = float(item["observed"])
        result[f"structural_target_{name}"] = float(item["target"])
        result[f"structural_tolerance_{name}"] = float(item["tolerance"])
        result[f"structural_mode_{name}"] = str(item["mode"])
        result[f"structural_error_{name}"] = float(item["error"])
        result[f"structural_ratio_{name}"] = float(item["ratio"])
        result[f"structural_margin_{name}"] = float(item["margin"])
        result[f"structural_pass_{name}"] = bool(item["passed"])
    return result


def _unit(values: np.ndarray) -> np.ndarray:
    array = np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    if array.size == 0:
        return array
    lo, hi = float(array.min()), float(array.max())
    return (array - lo) / (hi - lo + 1.0e-12)


def coverage_role_counts(
    batch_size: int,
    fractions: Mapping[str, float],
) -> dict[str, int]:
    """Allocate an exact batch size using largest remainders."""

    roles = (
        "feasible_interior", "constraint_boundary",
        "model_uncertainty", "global_novelty",
    )
    if isinstance(batch_size, bool) or not isinstance(batch_size, (int, np.integer)):
        raise ValueError("batch_size must be a positive integer")
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    raw = np.asarray([float(fractions.get(role, 0.0)) for role in roles])
    if np.any(~np.isfinite(raw)) or np.any(raw < 0.0) or raw.sum() <= 0.0:
        raise ValueError("coverage fractions must be finite, non-negative, and non-zero")
    raw /= raw.sum()
    exact = raw * int(batch_size)
    counts = np.floor(exact).astype(int)
    order = np.argsort(-(exact - counts), kind="stable")
    for index in order[: int(batch_size) - int(counts.sum())]:
        counts[index] += 1
    positive = [index for index, value in enumerate(raw) if value > 0.0]
    if batch_size >= len(positive):
        for index in positive:
            if counts[index] > 0:
                continue
            donors = [item for item in positive if counts[item] > 1]
            if donors:
                donor = max(donors, key=lambda item: (counts[item], raw[item]))
                counts[donor] -= 1
                counts[index] += 1
    return dict(zip(roles, counts.tolist()))


def select_coverage_batch(
    pool: np.ndarray,
    *,
    existing: np.ndarray | None,
    feasible_probability: Sequence[float],
    model_uncertainty: Sequence[float],
    batch_size: int,
    fractions: Mapping[str, float],
    minimum_novelty: float = 1.0e-6,
) -> tuple[np.ndarray, list[str]]:
    """Select a deterministic, diverse mixed-role batch in unit coordinates."""

    points = np.asarray(pool, dtype=float)
    if points.ndim != 2 or not len(points):
        raise ValueError("pool must be a non-empty two-dimensional array")
    if np.any(~np.isfinite(points)):
        raise ValueError("pool contains non-finite coordinates")
    if np.any(points < -1.0e-12) or np.any(points > 1.0 + 1.0e-12):
        raise ValueError("pool coordinates must lie in the normalized [0, 1] box")
    probability = np.asarray(feasible_probability, dtype=float)
    uncertainty = np.asarray(model_uncertainty, dtype=float)
    if probability.shape != (len(points),) or uncertainty.shape != (len(points),):
        raise ValueError("probability and uncertainty must contain one value per pool row")
    probability = np.clip(np.nan_to_num(probability, nan=0.0), 0.0, 1.0)
    uncertainty = np.maximum(
        np.nan_to_num(uncertainty, nan=0.0, posinf=0.0, neginf=0.0), 0.0
    )

    prior = np.asarray(existing, dtype=float) if existing is not None else np.empty((0, points.shape[1]))
    if prior.ndim != 2 or prior.shape[1] != points.shape[1]:
        raise ValueError("existing coordinates have the wrong shape")
    if np.any(~np.isfinite(prior)):
        raise ValueError("existing coordinates contain non-finite values")
    if not math.isfinite(float(minimum_novelty)) or minimum_novelty < 0.0:
        raise ValueError("minimum_novelty must be finite and non-negative")
    if len(prior):
        novelty = np.linalg.norm(points[:, None, :] - prior[None, :, :], axis=2).min(axis=1)
    else:
        novelty = np.ones(len(points), dtype=float)
    keep = novelty > float(minimum_novelty)
    points = points[keep]
    probability = probability[keep]
    uncertainty = uncertainty[keep]
    novelty = novelty[keep]
    if len(points):
        # An explicit pool may be assembled from multiple generators.  Keep
        # the first occurrence so no physical coordinate can consume two
        # roles in the same batch.
        _, unique_indices = np.unique(points, axis=0, return_index=True)
        unique_indices = np.sort(unique_indices)
        points = points[unique_indices]
        probability = probability[unique_indices]
        uncertainty = uncertainty[unique_indices]
        novelty = novelty[unique_indices]
    if len(points) < batch_size:
        raise ValueError("coverage pool has fewer novel points than the requested batch")

    diversity = 0.25 + 0.75 * _unit(novelty)
    scores = {
        "feasible_interior": _unit(probability) * diversity,
        "constraint_boundary": _unit(4.0 * probability * (1.0 - probability)) * diversity,
        "model_uncertainty": _unit(uncertainty) * diversity,
        "global_novelty": _unit(novelty),
    }
    counts = coverage_role_counts(batch_size, fractions)
    selected: list[int] = []
    roles: list[str] = []
    available = set(range(len(points)))
    for role, count in counts.items():
        for _ in range(count):
            candidates = np.asarray(sorted(available), dtype=int)
            if not len(candidates):
                break
            if selected:
                distance = np.linalg.norm(
                    points[candidates, None, :] - points[np.asarray(selected)][None, :, :],
                    axis=2,
                ).min(axis=1)
            else:
                distance = novelty[candidates]
            # A classifier may legitimately saturate at zero or one.  Keep a
            # non-zero score floor so diversity remains active instead of
            # degenerating to candidate-pool row order.
            combined = (
                0.05 + 0.95 * _unit(scores[role][candidates])
            ) * (0.20 + 0.80 * _unit(distance))
            chosen = int(candidates[int(np.argmax(combined))])
            selected.append(chosen)
            roles.append(role)
            available.remove(chosen)
    while len(selected) < batch_size:
        candidates = np.asarray(sorted(available), dtype=int)
        chosen = int(candidates[int(np.argmax(novelty[candidates]))])
        selected.append(chosen)
        roles.append("global_novelty")
        available.remove(chosen)
    return points[np.asarray(selected)], roles


def select_coverage_representative(
    frame: pd.DataFrame,
    *,
    parameter_names: Sequence[str] = (),
) -> pd.Series:
    """Choose one deterministic report representative without changing acquisition.

    Exact structural feasibility is the first ordering layer.  Within the
    feasible set, the ordinary fitted-property objective chooses a convenient
    representative for human reporting.  If no feasible row exists, the
    continuous constraint violation is minimized instead.  This avoids the
    arbitrary ``first zero`` produced by a satisficing acquisition objective.
    """

    if frame.empty:
        raise ValueError("coverage representative requires at least one row")
    data = frame.copy()
    success = (
        _truthy(data["success"])
        if "success" in data
        else pd.Series(True, index=data.index)
    )
    data = data.loc[success].copy()
    if data.empty:
        raise ValueError("coverage representative requires a successful row")

    if "structural_feasible" in data:
        feasible = _truthy(data["structural_feasible"])
    elif "structural_band_max_ratio" in data:
        feasible = (
            pd.to_numeric(data["structural_band_max_ratio"], errors="coerce")
            <= 1.0 + 1.0e-12
        ).fillna(False)
    else:
        raise ValueError(
            "coverage representative requires structural_feasible or "
            "structural_band_max_ratio"
        )
    fit_column = "fit_objective" if "fit_objective" in data else "objective"
    fit = (
        pd.to_numeric(data[fit_column], errors="coerce")
        if fit_column in data
        else pd.Series(math.inf, index=data.index)
    ).fillna(math.inf)
    violation = (
        pd.to_numeric(data["structural_constraint_violation"], errors="coerce")
        if "structural_constraint_violation" in data
        else pd.Series(math.inf, index=data.index)
    ).fillna(math.inf)
    band_ratio = (
        pd.to_numeric(data["structural_band_max_ratio"], errors="coerce")
        if "structural_band_max_ratio" in data
        else pd.Series(math.inf, index=data.index)
    ).fillna(math.inf)

    def sortable_float(value: Any) -> float:
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            return math.inf
        return numeric if math.isfinite(numeric) else math.inf

    def key(index: Any) -> tuple[Any, ...]:
        parameters = tuple(
            sortable_float(data.at[index, name]) if name in data else math.inf
            for name in parameter_names
        )
        if bool(feasible.loc[index]):
            return (0, float(fit.loc[index]), parameters, str(index))
        return (
            1,
            float(violation.loc[index]),
            float(band_ratio.loc[index]),
            float(fit.loc[index]),
            parameters,
            str(index),
        )

    selected = min(data.index, key=key)
    return data.loc[selected].copy()


def build_coverage_archives(
    frame: pd.DataFrame,
    *,
    parameter_names: Sequence[str],
    parameter_bounds: np.ndarray,
    archive_target: int,
    anchor_max_band_ratio: float = 3.0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return diverse strict-feasible rows and labelled boundary anchors.

    Rank one is meaningful rather than input-order dependent: the feasible
    archive starts from the best fitted-property representative, while the
    anchor set starts at the measured structural boundary.  Remaining rows
    are deterministic maximin additions in normalized parameter space and
    report their selection distance explicitly.
    """

    if not parameter_names:
        raise ValueError("parameter_names must not be empty")
    required = [*parameter_names, "structural_band_max_ratio"]
    missing = [name for name in required if name not in frame]
    if missing:
        raise ValueError(f"coverage results are missing columns: {missing}")
    if (
        isinstance(archive_target, bool)
        or not isinstance(archive_target, (int, np.integer))
        or archive_target < 1
    ):
        raise ValueError("archive_target must be a positive integer")
    if (
        not math.isfinite(float(anchor_max_band_ratio))
        or anchor_max_band_ratio <= 0.0
    ):
        raise ValueError("anchor_max_band_ratio must be finite and positive")
    data = frame.copy()
    success = (
        _truthy(data["success"])
        if "success" in data else pd.Series(True, index=data.index)
    )
    numeric = data[required].apply(pd.to_numeric, errors="coerce")
    pool = data.loc[success & np.isfinite(numeric.to_numpy(float)).all(axis=1)].copy()
    pool["structural_band_max_ratio"] = pd.to_numeric(
        pool["structural_band_max_ratio"], errors="coerce"
    )
    if "structural_feasible" in pool:
        exact_feasible = _truthy(pool["structural_feasible"])
    else:
        exact_feasible = pd.Series(True, index=pool.index)
    strict = pool.loc[
        exact_feasible
        & (pool["structural_band_max_ratio"] <= 1.0 + 1.0e-12)
    ].copy()
    anchors = pool.loc[
        pool["structural_band_max_ratio"] <= float(anchor_max_band_ratio)
    ].copy()
    if anchors.empty and not pool.empty:
        anchors = pool.nsmallest(
            min(max(1, archive_target // 4), len(pool)),
            "structural_band_max_ratio",
        )

    bounds = np.asarray(parameter_bounds, dtype=float)
    if bounds.shape != (len(parameter_names), 2):
        raise ValueError("parameter_bounds must have shape (n_parameters, 2)")
    if np.any(~np.isfinite(bounds)) or np.any(bounds[:, 1] <= bounds[:, 0]):
        raise ValueError("parameter_bounds must contain finite increasing ranges")
    span = bounds[:, 1] - bounds[:, 0]

    def prepare(rows: pd.DataFrame) -> pd.DataFrame:
        if rows.empty:
            return rows.copy()
        order = rows.copy()
        fit_column = "fit_objective" if "fit_objective" in order else "objective"
        if fit_column in order:
            order["_coverage_fit_sort"] = pd.to_numeric(
                order[fit_column], errors="coerce"
            ).fillna(math.inf)
            order = order.sort_values(
                [*parameter_names, "_coverage_fit_sort"],
                kind="stable",
            )
            order = order.drop_duplicates(list(parameter_names), keep="first")
            return order.drop(columns="_coverage_fit_sort").reset_index(drop=True)
        return order.drop_duplicates(
            list(parameter_names), keep="first"
        ).reset_index(drop=True)

    strict = prepare(strict)
    anchors = prepare(anchors)

    def maximin(
        rows: pd.DataFrame,
        count: int,
        *,
        seed_mode: str,
        role_column: str,
        distance_column: str,
    ) -> pd.DataFrame:
        if rows.empty:
            result = rows.copy()
            result[role_column] = pd.Series(dtype=str)
            result[distance_column] = pd.Series(dtype=float)
            return result
        coords = (rows[list(parameter_names)].to_numpy(float) - bounds[:, 0]) / span
        ratios = rows["structural_band_max_ratio"].to_numpy(float)
        if seed_mode == "fit":
            representative = select_coverage_representative(
                rows, parameter_names=parameter_names
            )
            chosen = [int(rows.index.get_loc(representative.name))]
            seed_role = "best_fit_structural_feasible"
        else:
            chosen = [int(np.argmin(np.abs(ratios - 1.0)))]
            seed_role = "closest_observed_boundary"
        selection_distances = [0.0]
        nearest = np.linalg.norm(coords - coords[chosen[0]], axis=1)
        while len(chosen) < min(int(count), len(rows)):
            nearest[chosen] = -np.inf
            nxt = int(np.argmax(nearest))
            selection_distances.append(float(nearest[nxt]))
            chosen.append(nxt)
            nearest = np.minimum(nearest, np.linalg.norm(coords - coords[nxt], axis=1))
        result = rows.iloc[chosen].copy().reset_index(drop=True)
        result[role_column] = [
            seed_role,
            *("maximin_diversity" for _ in range(len(result) - 1)),
        ]
        result[distance_column] = selection_distances
        return result

    feasible_archive = maximin(
        strict,
        int(archive_target),
        seed_mode="fit",
        role_column="archive_selection_role",
        distance_column="archive_selection_distance_normalized",
    )
    coverage_anchors = maximin(
        anchors,
        int(archive_target),
        seed_mode="boundary",
        role_column="anchor_selection_role",
        distance_column="anchor_selection_distance_normalized",
    )
    feasible_archive.insert(0, "archive_rank", np.arange(1, len(feasible_archive) + 1))
    coverage_anchors.insert(0, "anchor_rank", np.arange(1, len(coverage_anchors) + 1))
    anchor_exact_feasible = (
        _truthy(coverage_anchors["structural_feasible"])
        if "structural_feasible" in coverage_anchors
        else coverage_anchors["structural_band_max_ratio"] <= 1.0 + 1.0e-12
    )
    coverage_anchors["anchor_class"] = np.where(
        anchor_exact_feasible
        & (coverage_anchors["structural_band_max_ratio"] <= 1.0 + 1.0e-12),
        "strict_feasible",
        "near_boundary",
    )
    return feasible_archive, coverage_anchors


__all__ = [
    "assess_structural_properties",
    "build_coverage_archives",
    "coverage_role_counts",
    "select_coverage_batch",
    "select_coverage_representative",
    "structural_constraint_specs",
]
