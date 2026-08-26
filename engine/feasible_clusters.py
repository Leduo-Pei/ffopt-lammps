"""Deterministic clustering and quota-aware selection in parameter space.

The helpers in this module deliberately know nothing about a material, atom
type, or parameter name.  Candidates are deduplicated by the same exact
IEEE-754 parameter key used by the artifact manifests, normalised with the
declared parameter bounds, and connected by a deterministic minimum spanning
tree.  A pronounced gap in the tree identifies disconnected feasible basins;
otherwise the evidence is treated as one basin rather than inventing clusters.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any

import numpy as np
import pandas as pd

from workflow.artifact_manifest import ArtifactManifestError, canonical_parameter_key


class FeasibleClusterError(ValueError):
    """Raised when feasible-region clustering or selection is ill-defined."""


def _stable_value_token(value: Any) -> str:
    """Return a deterministic scalar token used only to break duplicate ties."""

    try:
        missing = pd.isna(value)
    except (TypeError, ValueError):
        missing = False
    if isinstance(missing, (bool, np.bool_)) and bool(missing):
        return "null"
    if isinstance(value, (bool, np.bool_)):
        return f"bool:{int(value)}"
    if isinstance(value, Real):
        numeric = float(value)
        if math.isnan(numeric):
            return "float:nan"
        if math.isinf(numeric):
            return "float:+inf" if numeric > 0.0 else "float:-inf"
        return f"float:{numeric.hex()}"
    if isinstance(value, str):
        return f"str:{value!r}"
    return f"{type(value).__module__}.{type(value).__qualname__}:{value!r}"


def _row_signature(row: pd.Series, columns: Sequence[Any]) -> str:
    return "|".join(
        f"{column!r}={_stable_value_token(row[column])}"
        for column in sorted(columns, key=str)
    )


def _parameter_bounds(
    parameter_columns: Sequence[str],
    bounds: Mapping[str, Sequence[float]] | Sequence[Sequence[Any]],
) -> tuple[np.ndarray, np.ndarray]:
    names = list(parameter_columns)
    if isinstance(bounds, Mapping):
        missing = [name for name in names if name not in bounds]
        if missing:
            raise FeasibleClusterError(f"parameter bounds are missing: {missing}")
        pairs = [bounds[name] for name in names]
    else:
        raw = list(bounds)
        if len(raw) != len(names):
            raise FeasibleClusterError(
                "parameter bounds must have one entry per parameter column"
            )
        if raw and len(raw[0]) == 3 and isinstance(raw[0][0], str):
            addressed = {str(item[0]): item[1:] for item in raw}
            missing = [name for name in names if name not in addressed]
            if missing:
                raise FeasibleClusterError(f"parameter bounds are missing: {missing}")
            pairs = [addressed[name] for name in names]
        else:
            pairs = raw

    try:
        lower = np.asarray([float(pair[0]) for pair in pairs], dtype=float)
        upper = np.asarray([float(pair[1]) for pair in pairs], dtype=float)
    except (IndexError, TypeError, ValueError) as exc:
        raise FeasibleClusterError("every parameter bound must be a numeric (low, high) pair") from exc
    if not np.isfinite(lower).all() or not np.isfinite(upper).all():
        raise FeasibleClusterError("parameter bounds must be finite")
    invalid = [names[index] for index in np.flatnonzero(upper <= lower)]
    if invalid:
        raise FeasibleClusterError(f"parameter bounds must have positive span: {invalid}")
    return lower, upper


def _quality_rank(
    frame: pd.DataFrame,
    quality_column: str | None,
    *,
    ascending: bool,
) -> np.ndarray:
    if quality_column is None:
        return np.zeros(len(frame), dtype=float)
    if quality_column not in frame:
        raise FeasibleClusterError(f"quality column {quality_column!r} is missing")
    values = pd.to_numeric(frame[quality_column], errors="coerce").to_numpy(float)
    rank = values if ascending else -values
    return np.where(np.isfinite(rank), rank, math.inf)


def _deduplicate_candidates(
    frame: pd.DataFrame,
    *,
    parameter_columns: Sequence[str],
    quality_column: str | None,
    quality_ascending: bool,
) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame):
        raise FeasibleClusterError("candidates must be supplied as a pandas DataFrame")
    if frame.empty:
        raise FeasibleClusterError("no feasible candidates were supplied")
    names = list(parameter_columns)
    if not names or len(set(names)) != len(names):
        raise FeasibleClusterError("parameter_columns must contain unique names")
    missing = [name for name in names if name not in frame]
    if missing:
        raise FeasibleClusterError(f"candidate parameters are missing: {missing}")

    numeric = frame[names].apply(pd.to_numeric, errors="coerce").to_numpy(float)
    if not np.isfinite(numeric).all():
        raise FeasibleClusterError("candidate parameter values must be finite numbers")

    result = frame.copy().reset_index(drop=True)
    keys: list[str] = []
    try:
        for values in numeric:
            keys.append(
                canonical_parameter_key(
                    {name: float(value) for name, value in zip(names, values, strict=True)}
                )
            )
    except ArtifactManifestError as exc:
        raise FeasibleClusterError(f"candidate parameters cannot be canonicalised: {exc}") from exc

    if "parameter_key" in result:
        recorded = result["parameter_key"]
        present = recorded.notna() & recorded.astype(str).ne("")
        mismatch = present & recorded.astype(str).ne(pd.Series(keys, index=result.index))
        if mismatch.any():
            raise FeasibleClusterError(
                "recorded parameter_key does not match the exact parameter values"
            )
    result["parameter_key"] = keys
    result["_cluster_quality_rank"] = _quality_rank(
        result, quality_column, ascending=quality_ascending
    )
    columns = list(result.columns)
    result["_cluster_row_signature"] = [
        _row_signature(row, columns) for _index, row in result.iterrows()
    ]
    result = result.sort_values(
        ["parameter_key", "_cluster_quality_rank", "_cluster_row_signature"],
        kind="mergesort",
    ).drop_duplicates("parameter_key", keep="first")
    return result.sort_values("parameter_key", kind="mergesort").reset_index(drop=True)


def _normalised_coordinates(
    frame: pd.DataFrame,
    parameter_columns: Sequence[str],
    lower: np.ndarray,
    upper: np.ndarray,
) -> np.ndarray:
    values = frame[list(parameter_columns)].apply(pd.to_numeric, errors="raise").to_numpy(float)
    tolerance = 64.0 * np.finfo(float).eps * np.maximum(1.0, np.maximum(abs(lower), abs(upper)))
    if np.any(values < lower - tolerance) or np.any(values > upper + tolerance):
        raise FeasibleClusterError("candidate parameter values fall outside the declared bounds")
    return np.clip((values - lower) / (upper - lower), 0.0, 1.0)


def _minimum_spanning_tree(coordinates: np.ndarray) -> list[tuple[int, int, float]]:
    """Return a deterministic Prim tree for canonical-key-sorted coordinates."""

    count = len(coordinates)
    if count < 2:
        return []
    used = np.zeros(count, dtype=bool)
    used[0] = True
    best = np.linalg.norm(coordinates - coordinates[0], axis=1)
    parent = np.zeros(count, dtype=int)
    best[0] = math.inf
    edges: list[tuple[int, int, float]] = []
    for _step in range(count - 1):
        available = np.flatnonzero(~used)
        distance = float(np.min(best[available]))
        tied = available[best[available] == distance]
        child = int(tied[0])
        source = int(parent[child])
        edges.append((source, child, distance))
        used[child] = True
        candidate_distance = np.linalg.norm(coordinates - coordinates[child], axis=1)
        for index in np.flatnonzero(~used):
            proposed = float(candidate_distance[index])
            if proposed < best[index] or (
                proposed == best[index] and child < parent[index]
            ):
                best[index] = proposed
                parent[index] = child
    return edges


def _cut_tree(
    count: int,
    edges: Sequence[tuple[int, int, float]],
    *,
    maximum_clusters: int,
    separation_ratio: float,
    minimum_separation: float,
) -> tuple[np.ndarray, float | None]:
    cut_indices: set[int] = set()
    threshold: float | None = None
    weights = sorted(float(edge[2]) for edge in edges if edge[2] > 0.0)
    if maximum_clusters > 1 and len(weights) >= 2:
        gaps: list[tuple[float, float, float, int]] = []
        for index, (lower, upper) in enumerate(zip(weights[:-1], weights[1:], strict=True)):
            ratio = upper / max(lower, np.finfo(float).tiny)
            if upper >= minimum_separation:
                gaps.append((ratio, upper - lower, upper, index))
        if gaps:
            ratio, _gap, upper, _index = max(gaps)
            if ratio >= separation_ratio:
                threshold = upper
                candidates = [
                    index for index, edge in enumerate(edges) if float(edge[2]) >= upper
                ]
                candidates.sort(
                    key=lambda index: (
                        -float(edges[index][2]),
                        min(edges[index][0], edges[index][1]),
                        max(edges[index][0], edges[index][1]),
                    )
                )
                cut_indices = set(candidates[: maximum_clusters - 1])

    adjacency: list[list[int]] = [[] for _index in range(count)]
    for edge_index, (left, right, _weight) in enumerate(edges):
        if edge_index in cut_indices:
            continue
        adjacency[left].append(right)
        adjacency[right].append(left)
    components: list[list[int]] = []
    unseen = set(range(count))
    while unseen:
        root = min(unseen)
        stack = [root]
        component: list[int] = []
        unseen.remove(root)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbour in sorted(adjacency[node], reverse=True):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    stack.append(neighbour)
        components.append(sorted(component))
    components.sort(key=lambda component: component[0])
    labels = np.empty(count, dtype=int)
    for label, component in enumerate(components, start=1):
        labels[component] = label
    return labels, threshold


def cluster_feasible_candidates(
    frame: pd.DataFrame,
    *,
    parameter_columns: Sequence[str],
    bounds: Mapping[str, Sequence[float]] | Sequence[Sequence[Any]],
    quality_column: str | None = None,
    quality_ascending: bool = True,
    maximum_clusters: int = 8,
    separation_ratio: float = 3.0,
    minimum_separation: float = 0.05,
) -> pd.DataFrame:
    """Deduplicate candidates and deterministically label separated basins.

    Cluster discovery cuts the largest pronounced gap in a deterministic
    minimum spanning tree.  All tree edges above that gap are cut, up to
    ``maximum_clusters - 1``.  If no gap meets ``separation_ratio`` and
    ``minimum_separation``, every candidate belongs to one cluster.
    """

    if isinstance(maximum_clusters, bool) or int(maximum_clusters) < 1:
        raise FeasibleClusterError("maximum_clusters must be a positive integer")
    maximum_clusters = int(maximum_clusters)
    if not math.isfinite(float(separation_ratio)) or float(separation_ratio) <= 1.0:
        raise FeasibleClusterError("separation_ratio must be finite and greater than one")
    if not math.isfinite(float(minimum_separation)) or float(minimum_separation) < 0.0:
        raise FeasibleClusterError("minimum_separation must be finite and non-negative")

    names = list(parameter_columns)
    lower, upper = _parameter_bounds(names, bounds)
    result = _deduplicate_candidates(
        frame,
        parameter_columns=names,
        quality_column=quality_column,
        quality_ascending=quality_ascending,
    )
    coordinates = _normalised_coordinates(result, names, lower, upper)
    edges = _minimum_spanning_tree(coordinates)
    labels, threshold = _cut_tree(
        len(result),
        edges,
        maximum_clusters=maximum_clusters,
        separation_ratio=float(separation_ratio),
        minimum_separation=float(minimum_separation),
    )
    result["feasible_cluster"] = [f"cluster_{label:03d}" for label in labels]
    sizes = result["feasible_cluster"].value_counts()
    result["feasible_cluster_size"] = result["feasible_cluster"].map(sizes).astype(int)
    result = result.drop(columns=["_cluster_quality_rank", "_cluster_row_signature"])
    result.attrs.update(
        {
            "feasible_cluster_count": int(len(sizes)),
            "feasible_cluster_cut_threshold": threshold,
            "feasible_cluster_maximum": maximum_clusters,
            "feasible_cluster_separation_ratio": float(separation_ratio),
            "feasible_cluster_minimum_separation": float(minimum_separation),
        }
    )
    return result


def select_cluster_balanced_candidates(
    frame: pd.DataFrame,
    *,
    parameter_columns: Sequence[str],
    bounds: Mapping[str, Sequence[float]] | Sequence[Sequence[Any]],
    quality_column: str,
    budget: int,
    quality_ascending: bool = True,
    minimum_per_cluster: int = 1,
    quality_elite_fraction: float = 0.25,
    required_parameter_keys: Sequence[str] = (),
    maximum_clusters: int = 8,
    separation_ratio: float = 3.0,
    minimum_separation: float = 0.05,
) -> pd.DataFrame:
    """Select an exact budget using quality, basin quotas, then maximin fill."""

    clustered = cluster_feasible_candidates(
        frame,
        parameter_columns=parameter_columns,
        bounds=bounds,
        quality_column=quality_column,
        quality_ascending=quality_ascending,
        maximum_clusters=maximum_clusters,
        separation_ratio=separation_ratio,
        minimum_separation=minimum_separation,
    )
    if isinstance(budget, bool) or int(budget) < 1:
        raise FeasibleClusterError("budget must be a positive integer")
    budget = int(budget)
    if budget > len(clustered):
        raise FeasibleClusterError(
            f"budget {budget} exceeds {len(clustered)} unique feasible candidates"
        )
    if isinstance(minimum_per_cluster, bool) or int(minimum_per_cluster) < 0:
        raise FeasibleClusterError("minimum_per_cluster must be a non-negative integer")
    minimum_per_cluster = int(minimum_per_cluster)
    fraction = float(quality_elite_fraction)
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise FeasibleClusterError("quality_elite_fraction must lie in [0, 1]")

    keys = list(clustered["parameter_key"].astype(str))
    key_to_index = {key: index for index, key in enumerate(keys)}
    required = sorted(set(str(key) for key in required_parameter_keys))
    missing_required = [key for key in required if key not in key_to_index]
    if missing_required:
        raise FeasibleClusterError(
            f"required parameter keys are absent from the feasible candidates: {missing_required}"
        )
    if len(required) > budget:
        raise FeasibleClusterError("required candidates exceed the selection budget")

    quality = _quality_rank(clustered, quality_column, ascending=quality_ascending)
    lower, upper = _parameter_bounds(parameter_columns, bounds)
    coordinates = _normalised_coordinates(clustered, parameter_columns, lower, upper)
    selected: list[int] = []
    roles: dict[int, str] = {}
    distances: dict[int, float] = {}

    for key in required:
        index = key_to_index[key]
        selected.append(index)
        roles[index] = "required"
        distances[index] = math.nan

    cluster_indices = {
        cluster: sorted(
            indices,
            key=lambda index: (float(quality[index]), keys[index]),
        )
        for cluster, indices in clustered.groupby("feasible_cluster", sort=True).indices.items()
    }
    quota_shortfall = sum(
        max(
            0,
            min(minimum_per_cluster, len(indices))
            - sum(index in selected for index in indices),
        )
        for indices in cluster_indices.values()
    )
    if len(selected) + quota_shortfall > budget:
        raise FeasibleClusterError(
            "budget is too small to satisfy required candidates and every cluster quota"
        )
    for cluster in sorted(cluster_indices):
        indices = cluster_indices[cluster]
        quota = min(minimum_per_cluster, len(indices))
        already = sum(index in selected for index in indices)
        for index in (item for item in indices if item not in selected):
            if already >= quota:
                break
            selected.append(index)
            roles[index] = "cluster_quota"
            distances[index] = math.nan
            already += 1

    elite_target = int(math.ceil(budget * fraction))
    quality_order = sorted(
        (index for index in range(len(clustered)) if index not in selected),
        key=lambda index: (float(quality[index]), keys[index]),
    )
    while len(selected) < min(budget, elite_target) and quality_order:
        index = quality_order.pop(0)
        selected.append(index)
        roles[index] = "quality_elite"
        distances[index] = math.nan

    while len(selected) < budget:
        available = [index for index in range(len(clustered)) if index not in selected]
        if not available:
            raise FeasibleClusterError("selection exhausted before reaching the requested budget")
        if selected:
            nearest = np.min(
                np.linalg.norm(
                    coordinates[available, None, :]
                    - coordinates[np.asarray(selected), :][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            largest = float(np.max(nearest))
            tied = [
                available[position]
                for position, value in enumerate(nearest)
                if float(value) == largest
            ]
            index = min(tied, key=lambda item: (float(quality[item]), keys[item]))
            distances[index] = largest
        else:
            index = min(available, key=lambda item: (float(quality[item]), keys[item]))
            distances[index] = math.inf
        selected.append(index)
        roles[index] = "maximin_fill"

    result = clustered.iloc[selected].copy().reset_index(drop=True)
    # A later fidelity may cluster a table that already records the previous
    # fidelity's selection audit.  Publish one unambiguous set of columns for
    # the current decision instead of colliding with or silently retaining the
    # upstream ranks.
    result = result.drop(
        columns=[
            name for name in (
                "cluster_selection_rank",
                "cluster_selection_role",
                "cluster_selection_min_distance",
            )
            if name in result
        ]
    )
    result.insert(0, "cluster_selection_rank", np.arange(1, budget + 1, dtype=int))
    result.insert(
        1,
        "cluster_selection_role",
        [roles[index] for index in selected],
    )
    result.insert(
        2,
        "cluster_selection_min_distance",
        [distances[index] for index in selected],
    )
    result.attrs.update(clustered.attrs)
    result.attrs.update(
        {
            "cluster_selection_budget": budget,
            "cluster_selection_minimum_per_cluster": minimum_per_cluster,
            "cluster_selection_quality_elite_fraction": fraction,
        }
    )
    return result


__all__ = [
    "FeasibleClusterError",
    "cluster_feasible_candidates",
    "select_cluster_balanced_candidates",
]
