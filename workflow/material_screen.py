"""Deterministic, gate-aware selection for the initial static elastic screen.

The structural campaign may contain thousands of exact BO/sample/audit rows.
Static elasticity is deliberately calculated only for a bounded design whose
size scales with the free-parameter dimension.  Selection preserves objective
elites and fills the remaining quota by maximin distance, while recording
whether every row came from the strict structural core, the relaxed buffer, or
the global safeguard.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

import numpy as np
import pandas as pd

from engine.cubic_elastic_batch import assess_structural_gates
from engine.parameter_space import build_parameter_space


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _objective(frame: pd.DataFrame) -> np.ndarray:
    for name in ("objective", "objective_mean", "obj_structural"):
        if name in frame:
            values = pd.to_numeric(frame[name], errors="coerce").to_numpy(float)
            return np.where(np.isfinite(values), values, math.inf)
    return np.full(len(frame), math.inf, dtype=float)


def _normalised_coordinates(
    frame: pd.DataFrame,
    parameter_space: Sequence[tuple[str, float, float]],
) -> np.ndarray:
    names = [name for name, _lower, _upper in parameter_space]
    lower = np.asarray([value for _name, value, _upper in parameter_space])
    span = np.asarray([
        max(upper - low, 1.0e-15)
        for _name, low, upper in parameter_space
    ])
    return (frame[names].to_numpy(float) - lower) / span


def _maximin_fill(
    *,
    candidates: Sequence[int],
    count: int,
    coordinates: np.ndarray,
    selected: list[int],
    keys: Sequence[str],
) -> list[int]:
    available = [int(index) for index in candidates if int(index) not in selected]
    chosen: list[int] = []
    while available and len(chosen) < count:
        references = [*selected, *chosen]
        if references:
            distance = np.min(
                np.linalg.norm(
                    coordinates[available, None, :]
                    - coordinates[np.asarray(references), :][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
        else:
            distance = np.ones(len(available), dtype=float)
        position = max(
            range(len(available)),
            key=lambda item: (float(distance[item]), str(keys[available[item]])),
        )
        chosen.append(available.pop(position))
    return chosen


def select_static_screen_candidates(
    frame: pd.DataFrame,
    *,
    config: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> pd.DataFrame:
    """Return the bounded initial static design with an auditable selection role."""

    parameter_space = build_parameter_space(dict(config))
    names = [name for name, _lower, _upper in parameter_space]
    missing = [name for name in names if name not in frame]
    if missing:
        raise ValueError(f"Static-screen evidence is missing parameters: {missing}")
    selected_frame = frame.copy().reset_index(drop=True)
    if "success" in selected_frame:
        selected_frame = selected_frame.loc[
            selected_frame["success"].map(_truthy)
        ].reset_index(drop=True)
    finite = np.isfinite(selected_frame[names].to_numpy(float)).all(axis=1)
    selected_frame = selected_frame.loc[finite].reset_index(drop=True)
    if selected_frame.empty:
        empty = selected_frame.copy()
        empty.insert(0, "screen_selection_rank", pd.Series(dtype=int))
        empty.insert(1, "screen_selection_role", pd.Series(dtype=str))
        empty.insert(2, "screen_region", pd.Series(dtype=str))
        return empty

    gate = [
        assess_structural_gates(row, config)
        for row in selected_frame.to_dict(orient="records")
    ]
    margin = np.asarray([float(item["structural_margin"]) for item in gate])
    strict = np.asarray([bool(item["structural_gate_pass"]) for item in gate])
    buffer_multiplier = float(settings.get("buffer_multiplier", 1.5))
    buffer = (~strict) & (margin >= 1.0 - buffer_multiplier - 1.0e-12)
    regions = np.where(strict, "core", np.where(buffer, "buffer", "global"))
    selected_frame["structural_gate_pass"] = strict
    selected_frame["structural_margin"] = margin
    selected_frame["screen_region"] = regions

    dimensions = len(names)
    minimum = int(settings.get("minimum", 480))
    per_dimension = int(settings.get("per_dimension", 160))
    maximum = int(settings.get("maximum", 1200))
    target = min(len(selected_frame), maximum, max(minimum, dimensions * per_dimension))
    core_fraction = float(settings.get("core_fraction", 0.75))
    elite_fraction = float(settings.get("objective_elite_fraction", 0.10))
    core_quota = min(target, int(round(target * core_fraction)))
    elite_count = min(target, int(math.ceil(target * elite_fraction)))

    objective = _objective(selected_frame)
    keys = list(
        selected_frame.get(
            "parameter_key", pd.Series(selected_frame.index.map(str))
        ).astype(str)
    )
    coordinates = _normalised_coordinates(selected_frame, parameter_space)
    selected: list[int] = []
    roles: dict[int, str] = {}

    # Objective elites are never lost to geometric thinning. Prefer strict
    # core rows, then the relaxed buffer, then the global safeguard.
    elite_order = sorted(
        range(len(selected_frame)),
        key=lambda index: (
            0 if strict[index] else 1 if buffer[index] else 2,
            float(objective[index]),
            keys[index],
        ),
    )[:elite_count]
    selected.extend(elite_order)
    roles.update({index: "objective_elite" for index in elite_order})

    core_indices = np.flatnonzero(strict).tolist()
    needed_core = max(0, core_quota - sum(strict[index] for index in selected))
    chosen = _maximin_fill(
        candidates=core_indices,
        count=needed_core,
        coordinates=coordinates,
        selected=selected,
        keys=keys,
    )
    selected.extend(chosen)
    roles.update({index: "core_diverse" for index in chosen})

    for region, role in (
        (np.flatnonzero(buffer).tolist(), "buffer_diverse"),
        (core_indices, "core_diverse_fill"),
        (np.flatnonzero(~(strict | buffer)).tolist(), "global_diverse"),
    ):
        chosen = _maximin_fill(
            candidates=region,
            count=target - len(selected),
            coordinates=coordinates,
            selected=selected,
            keys=keys,
        )
        selected.extend(chosen)
        roles.update({index: role for index in chosen})
        if len(selected) >= target:
            break

    result = selected_frame.iloc[selected[:target]].copy().reset_index(drop=True)
    result.insert(0, "screen_selection_rank", np.arange(1, len(result) + 1))
    result.insert(
        1,
        "screen_selection_role",
        [roles[index] for index in selected[:target]],
    )
    result["screen_target_count"] = target
    result["screen_core_quota"] = core_quota
    result["screen_buffer_multiplier"] = buffer_multiplier
    return result


__all__ = ["select_static_screen_candidates"]
