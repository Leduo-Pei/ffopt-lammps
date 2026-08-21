"""One-round, resumable structure-constrained minimax refinement.

This module is the material-independent scientific core used between cheap
structural evaluation and expensive mechanical validation.  It deliberately
does not discover files, submit jobs, or assume a crystal structure.  Every
evidence file and candidate pool is supplied explicitly by the caller.

The exact ranking is lexicographic:

1. every structural constraint must pass;
2. optional stability and fit-quality gates must pass;
3. minimise the largest relative mechanical error (``M-infinity``);
4. minimise the mechanical relative-error RMSE.

A reporting threshold (for example 20 percent) labels result quality only.  It
never removes the best measured candidate.  Each invocation completes exactly
one round and publishes its files transactionally with a content-addressed
manifest, making an identical retry safe to reuse.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .constraint_objectives import (
    assess_constraint,
    minimax_relative_errors,
    summarize_constraints,
)
from .config_loader import load_config
from .cubic_elasticity import derive_cubic_properties
from .property_contracts import PropertyRole, coerce_property_role
from workflow.artifact_manifest import (
    ArtifactManifestError,
    build_artifact_manifest,
    canonical_parameter_key,
    check_artifact_reuse,
    scientific_config_hash,
    write_artifact_manifest,
)


_EPS = 1.0e-12
_STATE_SCHEMA = "ffopt-constrained-refinement-state-v1"
_SPEC_SCHEMA = "ffopt-constrained-refinement-v1"
_OUTPUT_NAMES = {
    "state": "refinement_state.json",
    "ranking": "exact_structural_mechanical_ranking.csv",
    "assessed": "assessed_candidates.csv",
    "mechanical_proposals": "mechanical_proposals.csv",
    "structural_proposals": "structural_proposals.csv",
    "report": "refinement_report.md",
    "disposition": "stage_disposition.json",
}


class RefinementError(ValueError):
    """Raised when refinement evidence or immutable output is inconsistent."""


def _finite_float(value: Any, *, field: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise RefinementError(f"{field} must be a finite number") from exc
    if not math.isfinite(result):
        raise RefinementError(f"{field} must be a finite number")
    return result


def _positive_int(value: Any, *, field: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        raise RefinementError(f"{field} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise RefinementError(f"{field} must be an integer") from exc
    lower = 0 if allow_zero else 1
    if result < lower or float(result) != float(value):
        qualifier = "non-negative" if allow_zero else "positive"
        raise RefinementError(f"{field} must be a {qualifier} integer")
    return result


@dataclass(frozen=True)
class StructuralConstraintSpec:
    """One exact observed structural/property gate."""

    name: str
    column: str
    target: float
    tolerance: float
    mode: str = "relative_percent"
    role: str = PropertyRole.CONSTRAINT.value

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> StructuralConstraintSpec:
        try:
            name = str(raw["name"])
            column = str(raw["column"])
            target = _finite_float(raw["target"], field=f"constraint {name} target")
            tolerance = _finite_float(
                raw["tolerance"], field=f"constraint {name} tolerance"
            )
        except KeyError as exc:
            raise RefinementError(
                f"Structural constraint is missing {exc.args[0]!r}"
            ) from exc
        if not name or not column:
            raise RefinementError("Constraint names and columns must not be empty")
        if tolerance <= 0.0:
            raise RefinementError(f"constraint {name} tolerance must be positive")
        mode = str(raw.get("mode", "relative_percent"))
        # Delegate the closed vocabulary to the shared constraint primitive.
        assess_constraint(name, target, target, tolerance, mode=mode)
        role = coerce_property_role(raw.get("role", "constraint"))
        if role is not PropertyRole.CONSTRAINT:
            raise RefinementError(
                f"Structural gate {name!r} must use role 'constraint', got {role.value!r}"
            )
        return cls(name, column, target, tolerance, mode, role.value)


@dataclass(frozen=True)
class MechanicalObjectiveSpec:
    """One independent mechanical minimax objective."""

    name: str
    column: str
    target: float
    role: str = PropertyRole.OBJECTIVE.value

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> MechanicalObjectiveSpec:
        try:
            name = str(raw["name"])
            column = str(raw["column"])
            target = _finite_float(raw["target"], field=f"objective {name} target")
        except KeyError as exc:
            raise RefinementError(
                f"Mechanical objective is missing {exc.args[0]!r}"
            ) from exc
        if not name or not column:
            raise RefinementError("Objective names and columns must not be empty")
        if abs(target) <= _EPS:
            raise RefinementError(
                f"objective {name} target must be non-zero for relative scoring"
            )
        role = coerce_property_role(raw.get("role", "objective"))
        if role is not PropertyRole.OBJECTIVE:
            raise RefinementError(
                f"Mechanical target {name!r} must use role 'objective', got {role.value!r}"
            )
        return cls(name, column, target, role.value)


@dataclass(frozen=True)
class RefinementSpec:
    """Validated, JSON-compatible scientific settings for one campaign."""

    parameter_names: tuple[str, ...]
    structural_constraints: tuple[StructuralConstraintSpec, ...]
    mechanical_objectives: tuple[MechanicalObjectiveSpec, ...]
    maximum_rounds: int = 8
    patience: int = 3
    minimum_improvement_percent_points: float = 0.5
    mechanical_proposals_per_round: int = 20
    structural_proposals_per_round: int = 20
    report_threshold_percent: float = 20.0
    require_stability: bool = True
    stability_column: str = "mechanically_stable"
    minimum_fit_quality: float | None = None
    fit_quality_column: str = "minimum_fit_r2"
    derivation: Mapping[str, Any] | None = None

    @classmethod
    def from_mapping(cls, raw: Mapping[str, Any]) -> RefinementSpec:
        schema = raw.get("schema", _SPEC_SCHEMA)
        if schema != _SPEC_SCHEMA:
            raise RefinementError(
                f"Unsupported refinement spec schema {schema!r}; expected {_SPEC_SCHEMA!r}"
            )
        parameter_names = tuple(str(item) for item in raw.get("parameter_names", ()))
        if not parameter_names or any(not item for item in parameter_names):
            raise RefinementError("parameter_names must contain non-empty names")
        if len(set(parameter_names)) != len(parameter_names):
            raise RefinementError("parameter_names must not contain duplicates")
        constraints = tuple(
            StructuralConstraintSpec.from_mapping(item)
            for item in raw.get("structural_constraints", ())
        )
        objectives = tuple(
            MechanicalObjectiveSpec.from_mapping(item)
            for item in raw.get("mechanical_objectives", ())
        )
        if not constraints:
            raise RefinementError("At least one structural constraint is required")
        if not objectives:
            raise RefinementError("At least one mechanical objective is required")
        if len({item.name for item in constraints}) != len(constraints):
            raise RefinementError("Structural constraint names must be unique")
        if len({item.name for item in objectives}) != len(objectives):
            raise RefinementError("Mechanical objective names must be unique")
        minimum_improvement = _finite_float(
            raw.get("minimum_improvement_percent_points", 0.5),
            field="minimum_improvement_percent_points",
        )
        if minimum_improvement < 0.0:
            raise RefinementError("minimum_improvement_percent_points must be non-negative")
        threshold = _finite_float(
            raw.get("report_threshold_percent", 20.0),
            field="report_threshold_percent",
        )
        if threshold <= 0.0:
            raise RefinementError("report_threshold_percent must be positive")
        fit_quality = raw.get("minimum_fit_quality")
        if fit_quality is not None:
            fit_quality = _finite_float(fit_quality, field="minimum_fit_quality")
        derivation = raw.get("derivation")
        if derivation is not None and not isinstance(derivation, Mapping):
            raise RefinementError("derivation must be a mapping")
        return cls(
            parameter_names=parameter_names,
            structural_constraints=constraints,
            mechanical_objectives=objectives,
            maximum_rounds=_positive_int(
                raw.get("maximum_rounds", 8), field="maximum_rounds"
            ),
            patience=_positive_int(raw.get("patience", 3), field="patience"),
            minimum_improvement_percent_points=minimum_improvement,
            mechanical_proposals_per_round=_positive_int(
                raw.get("mechanical_proposals_per_round", 20),
                field="mechanical_proposals_per_round",
                allow_zero=True,
            ),
            structural_proposals_per_round=_positive_int(
                raw.get("structural_proposals_per_round", 20),
                field="structural_proposals_per_round",
                allow_zero=True,
            ),
            report_threshold_percent=threshold,
            require_stability=bool(raw.get("require_stability", True)),
            stability_column=str(raw.get("stability_column", "mechanically_stable")),
            minimum_fit_quality=fit_quality,
            fit_quality_column=str(raw.get("fit_quality_column", "minimum_fit_r2")),
            derivation=dict(derivation) if derivation is not None else None,
        )

    def scientific_identity(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["schema"] = _SPEC_SCHEMA
        return payload


@dataclass(frozen=True)
class AcquisitionResult:
    """Proposals and the capability established for this evidence snapshot.

    ``scientific_convergence_capable`` is deliberately a per-call result.  A
    backend may implement the necessary probabilistic machinery while
    returning ``False`` for a sparse or poorly fitted round.  Diagnostics must
    contain only strict-JSON values because they are persisted in the round
    state for audit and restart.
    """

    proposals: pd.DataFrame
    backend: str
    capability: str
    scientific_convergence_capable: bool
    diagnostics: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class AcquisitionContext:
    """Explicit evidence supplied to a structural acquisition backend."""

    candidate_pool: pd.DataFrame
    structural_observations: pd.DataFrame
    mechanical_observations: pd.DataFrame
    parameter_names: tuple[str, ...]
    round_number: int
    proposal_count: int
    seed: int
    structural_constraints: tuple[StructuralConstraintSpec, ...] = ()
    mechanical_objectives: tuple[MechanicalObjectiveSpec, ...] = ()
    refinement_spec: RefinementSpec | None = None


class StructuralAcquisition(Protocol):
    """Extension point for future GP/TuRBO/SCBO acquisition implementations."""

    name: str
    capability: str
    # Declares that the implementation can establish scientific convergence
    # when its per-round data/model-quality requirements are met.
    scientific_convergence_capable: bool

    def scientific_identity(self) -> Mapping[str, Any]:
        """Return deterministic settings included in the stage fingerprint."""

    def propose(self, context: AcquisitionContext) -> AcquisitionResult:
        """Select unevaluated structural coordinates for exactly one round."""


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _recorded_gate(value: Any) -> bool | None:
    """Return a recorded boolean gate, or None for legacy/missing evidence."""

    if value is None:
        return None
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return _truthy(value)


def _parameter_mapping(row: Mapping[str, Any], names: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in names:
        result[name] = _finite_float(row.get(name), field=f"parameter {name!r}")
    return result


def _with_parameter_keys(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    missing = [name for name in names if name not in frame]
    if missing:
        raise RefinementError(f"Evidence is missing parameter columns: {missing}")
    result = frame.copy()
    result["parameter_key"] = [
        canonical_parameter_key(_parameter_mapping(row, names))
        for row in result.to_dict(orient="records")
    ]
    return result


def _read_evidence(paths: Sequence[Path], names: Sequence[str], *, label: str) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_order, raw_path in enumerate(paths):
        path = Path(raw_path).resolve()
        if not path.is_file():
            raise RefinementError(f"Explicit {label} evidence does not exist: {path}")
        try:
            frame = pd.read_csv(path, float_precision="round_trip")
        except (OSError, ValueError, pd.errors.EmptyDataError) as exc:
            raise RefinementError(f"Cannot read {label} evidence {path}: {exc}") from exc
        if frame.empty:
            continue
        frame = _with_parameter_keys(frame, names)
        frame["_source_order"] = source_order
        frame["_row_order"] = np.arange(len(frame))
        frame["observation_source"] = str(path)
        if "success" in frame:
            frame = frame.loc[frame["success"].map(_truthy)].copy()
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[*names, "parameter_key"])
    combined = pd.concat(frames, ignore_index=True, sort=False)
    audited = (
        combined["audit_certified"].map(_truthy).astype(int)
        if "audit_certified" in combined
        else pd.Series(0, index=combined.index)
    )
    seeds = (
        pd.to_numeric(combined["n_seeds"], errors="coerce").fillna(0)
        if "n_seeds" in combined
        else pd.Series(0, index=combined.index)
    )
    combined["_evidence_priority"] = 10_000 * audited + 100 * seeds
    combined = combined.sort_values(
        ["_evidence_priority", "_source_order", "_row_order"],
        ascending=[False, False, False],
        kind="stable",
    )
    combined = combined.drop_duplicates("parameter_key", keep="first")
    return combined.drop(
        columns=["_evidence_priority", "_source_order", "_row_order"]
    ).reset_index(drop=True)


def _enrich_cubic_mechanics(
    frame: pd.DataFrame, derivation: Mapping[str, Any] | None
) -> pd.DataFrame:
    if derivation is None:
        return frame
    kind = str(derivation.get("kind", ""))
    if kind != "cubic":
        raise RefinementError(f"Unsupported mechanical derivation kind {kind!r}")
    columns = {
        name: str(derivation.get(f"{name}_column", name))
        for name in ("C11", "C12", "C44")
    }
    if frame.empty:
        result = frame.copy()
        for column in (
            "derived_B",
            "derived_Cprime",
            "derived_C44",
            "derived_born_stable",
            "derived_minimum_born_margin_gpa",
            "derived_G_hill",
            "derived_E_hill",
            "derived_nu_hill",
        ):
            result[column] = pd.Series(dtype=float)
        return result
    missing = [column for column in columns.values() if column not in frame]
    if missing:
        raise RefinementError(f"Cubic mechanical evidence is missing columns: {missing}")
    result = frame.copy()
    derived_rows: list[dict[str, Any]] = []
    for row in result.to_dict(orient="records"):
        properties = derive_cubic_properties(
            row[columns["C11"]], row[columns["C12"]], row[columns["C44"]]
        )
        targets = properties["independent_targets_gpa"]
        derived_rows.append({
            "derived_B": targets["B"],
            "derived_Cprime": targets["Cprime"],
            "derived_C44": targets["C44"],
            "derived_born_stable": properties["born_stability"]["stable"],
            "derived_minimum_born_margin_gpa": properties["born_stability"][
                "minimum_margin_gpa"
            ],
            "derived_G_hill": properties["polycrystalline_moduli_gpa"]["G_hill"],
            "derived_E_hill": properties["polycrystalline_moduli_gpa"]["E_hill"],
            "derived_nu_hill": properties["nu_hill"],
        })
    return pd.concat(
        [result.reset_index(drop=True), pd.DataFrame(derived_rows)], axis=1
    )


def _assess_structure(
    row: Mapping[str, Any], constraints: Sequence[StructuralConstraintSpec]
) -> dict[str, Any]:
    assessments = [
        assess_constraint(
            item.name,
            row.get(item.column),
            item.target,
            item.tolerance,
            mode=item.mode,
        )
        for item in constraints
    ]
    return summarize_constraints(assessments)


def _objective_values(
    row: Mapping[str, Any], objectives: Sequence[MechanicalObjectiveSpec]
) -> tuple[dict[str, Any], dict[str, float]]:
    values = {item.name: row.get(item.column) for item in objectives}
    targets = {item.name: item.target for item in objectives}
    return minimax_relative_errors(values, targets), targets


def assess_and_rank_candidates(
    structural: pd.DataFrame,
    mechanical: pd.DataFrame,
    spec: RefinementSpec,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return all assessed matches and the exact eligible minimax ranking."""
    missing_constraints = [
        item.column for item in spec.structural_constraints if item.column not in structural
    ]
    if len(structural) and missing_constraints:
        raise RefinementError(
            f"Structural evidence is missing constraint columns: {missing_constraints}"
        )
    missing_objectives = [
        item.column for item in spec.mechanical_objectives if item.column not in mechanical
    ]
    if len(mechanical) and missing_objectives:
        raise RefinementError(
            f"Mechanical evidence is missing objective columns: {missing_objectives}"
        )
    mechanical_by_key = {
        str(row["parameter_key"]): row
        for row in mechanical.to_dict(orient="records")
    }
    records: list[dict[str, Any]] = []
    for structural_row in structural.to_dict(orient="records"):
        key = str(structural_row["parameter_key"])
        constraint_summary = _assess_structure(
            structural_row, spec.structural_constraints
        )
        audit_gate = _recorded_gate(structural_row.get("audit_certified"))
        if audit_gate is False:
            # Multi-seed exact observations are admissible only when every
            # replicate passed the structural gates.  Legacy evidence without
            # an audit field retains its historical mean-property semantics.
            constraint_summary = {
                **constraint_summary,
                "feasible": False,
                "constraint_violation": math.inf,
                "maximum_band_ratio": math.inf,
                "minimum_margin": -math.inf,
                "limiting_constraint": "replicate_audit",
                "failed_constraints": [
                    *constraint_summary["failed_constraints"],
                    "replicate_audit",
                ],
            }
        result = dict(structural_row)
        result.update({
            "structural_feasible": bool(constraint_summary["feasible"]),
            "structural_constraint_violation": constraint_summary[
                "constraint_violation"
            ],
            "structural_maximum_band_ratio": constraint_summary[
                "maximum_band_ratio"
            ],
            "structural_minimum_margin": constraint_summary["minimum_margin"],
            "structural_limiting_constraint": constraint_summary[
                "limiting_constraint"
            ],
            "structural_failed_constraints": "|".join(
                constraint_summary["failed_constraints"]
            ),
            "mechanical_observed": key in mechanical_by_key,
            "mechanical_eligible": False,
            "within_mechanical_report_threshold": False,
            "mechanical_quality_tier": "not_evaluated",
        })
        mechanical_row = mechanical_by_key.get(key)
        if mechanical_row is None:
            records.append(result)
            continue
        for column, value in mechanical_row.items():
            if column not in {*spec.parameter_names, "parameter_key"}:
                result[f"mechanical__{column}"] = value
        objective_summary, _targets = _objective_values(
            mechanical_row, spec.mechanical_objectives
        )
        result["mechanical_max_error_percent"] = objective_summary[
            "maximum_error_percent"
        ]
        result["mechanical_rmse_percent"] = objective_summary["rmse_percent"]
        result["mechanical_limiting_target"] = objective_summary["limiting_target"]
        for name, error in objective_summary["errors_percent"].items():
            result[f"mechanical_error_{name}_percent"] = error

        stability_ok = True
        if spec.require_stability:
            stability_ok = _truthy(mechanical_row.get(spec.stability_column, False))
        fit_ok = True
        fit_value: float | None = None
        if spec.minimum_fit_quality is not None:
            try:
                fit_value = _finite_float(
                    mechanical_row.get(spec.fit_quality_column),
                    field=spec.fit_quality_column,
                )
            except RefinementError:
                fit_ok = False
            else:
                fit_ok = fit_value + _EPS >= spec.minimum_fit_quality
        finite_score = math.isfinite(float(objective_summary["maximum_error_percent"]))
        eligible = bool(
            constraint_summary["feasible"] and stability_ok and fit_ok and finite_score
        )
        within = bool(
            eligible
            and float(objective_summary["maximum_error_percent"])
            <= spec.report_threshold_percent + _EPS
        )
        result.update({
            "mechanical_stability_gate_pass": stability_ok,
            "mechanical_fit_gate_pass": fit_ok,
            "mechanical_fit_quality": fit_value,
            "mechanical_eligible": eligible,
            "within_mechanical_report_threshold": within,
            "mechanical_quality_tier": (
                "within_report_threshold"
                if within
                else "best_effort_mechanical"
                if eligible
                else "ineligible"
            ),
        })
        records.append(result)

    assessed = pd.DataFrame(records)
    ranked = assessed.loc[assessed.get(
        "mechanical_eligible", pd.Series(False, index=assessed.index)
    ).astype(bool)].copy()
    if not ranked.empty:
        ranked = ranked.sort_values(
            [
                "mechanical_max_error_percent",
                "mechanical_rmse_percent",
                "structural_minimum_margin",
                "parameter_key",
            ],
            ascending=[True, True, False, True],
            kind="stable",
        ).reset_index(drop=True)
        ranked.insert(0, "candidate_rank", np.arange(1, len(ranked) + 1))
    else:
        ranked = pd.DataFrame(columns=[
            "candidate_rank",
            "parameter_key",
            *spec.parameter_names,
            "mechanical_max_error_percent",
            "mechanical_rmse_percent",
            "mechanical_limiting_target",
            "within_mechanical_report_threshold",
            "mechanical_quality_tier",
        ])
    return assessed, ranked


def _normalized_parameters(frame: pd.DataFrame, names: Sequence[str]) -> np.ndarray:
    values = frame[list(names)].to_numpy(float)
    if not len(values):
        return np.empty((0, len(names)), dtype=float)
    lows = np.min(values, axis=0)
    spans = np.maximum(np.max(values, axis=0) - lows, _EPS)
    return (values - lows) / spans


def _minimum_distance(points: np.ndarray, references: np.ndarray) -> np.ndarray:
    if not len(points):
        return np.empty(0, dtype=float)
    if not len(references):
        return np.ones(len(points), dtype=float)
    return np.min(
        np.linalg.norm(points[:, None, :] - references[None, :, :], axis=2),
        axis=1,
    )


def _diverse_select(
    frame: pd.DataFrame,
    names: Sequence[str],
    count: int,
    *,
    score_column: str | None = None,
    excluded_indices: Sequence[int] = (),
) -> list[int]:
    available = [index for index in frame.index if index not in set(excluded_indices)]
    if count <= 0 or not available:
        return []
    coordinates = _normalized_parameters(frame, names)
    if score_column is not None and score_column in frame:
        scores = pd.to_numeric(frame[score_column], errors="coerce").fillna(0.0)
    else:
        scores = pd.Series(0.0, index=frame.index)
    score_values = scores.to_numpy(float)
    score_span = max(float(np.max(score_values) - np.min(score_values)), _EPS)
    normalized_score = (score_values - np.min(score_values)) / score_span
    selected: list[int] = []
    while len(selected) < min(count, len(available)):
        remaining = [index for index in available if index not in selected]
        if selected or excluded_indices:
            references = coordinates[[*excluded_indices, *selected]]
            distance = _minimum_distance(coordinates[remaining], references)
        else:
            distance = np.ones(len(remaining), dtype=float)
        winner_position = max(
            range(len(remaining)),
            key=lambda pos: (
                float(normalized_score[remaining[pos]]) + 0.25 * float(distance[pos]),
                str(frame.loc[remaining[pos], "parameter_key"]),
            ),
        )
        selected.append(remaining[winner_position])
    return selected


class ExplicitPoolAcquisition:
    """Select from an explicitly supplied, externally scored candidate pool.

    Optional columns are ``acquisition_score`` or ``p_structural_feasible``
    for exploitation, ``structural_entropy`` for boundary learning, and
    ``global_distance`` for global coverage.  Missing scores degrade safely to
    deterministic parameter-space diversity.  This backend does not train or
    generate points and therefore cannot, by itself, establish scientific
    convergence.
    """

    name = "explicit_pool"
    capability = "selection_only"
    scientific_convergence_capable = False

    def __init__(
        self,
        *,
        feasible_fraction: float = 0.60,
        boundary_fraction: float = 0.25,
    ) -> None:
        self.feasible_fraction = float(feasible_fraction)
        self.boundary_fraction = float(boundary_fraction)
        if not 0.0 <= self.feasible_fraction <= 1.0:
            raise RefinementError("feasible_fraction must lie in [0, 1]")
        if not 0.0 <= self.boundary_fraction <= 1.0:
            raise RefinementError("boundary_fraction must lie in [0, 1]")
        if self.feasible_fraction + self.boundary_fraction > 1.0 + _EPS:
            raise RefinementError(
                "feasible_fraction + boundary_fraction must not exceed 1"
            )

    def scientific_identity(self) -> Mapping[str, Any]:
        return {
            "backend": self.name,
            "capability": self.capability,
            "scientific_convergence_capable": self.scientific_convergence_capable,
            "feasible_fraction": self.feasible_fraction,
            "boundary_fraction": self.boundary_fraction,
        }

    def propose(self, context: AcquisitionContext) -> AcquisitionResult:
        pool = context.candidate_pool.copy()
        empty_columns = [
            "candidate_id",
            "selection_role",
            "parameter_key",
            *context.parameter_names,
        ]
        if pool.empty or context.proposal_count <= 0:
            return AcquisitionResult(
                pd.DataFrame(columns=empty_columns),
                self.name,
                self.capability,
                self.scientific_convergence_capable,
            )
        observed = set(context.structural_observations["parameter_key"].astype(str))
        pool = pool.loc[~pool["parameter_key"].astype(str).isin(observed)].copy()
        pool = pool.drop_duplicates("parameter_key", keep="first").reset_index(drop=True)
        if pool.empty:
            return AcquisitionResult(
                pd.DataFrame(columns=empty_columns),
                self.name,
                self.capability,
                self.scientific_convergence_capable,
            )

        count = min(context.proposal_count, len(pool))
        exploitation = int(round(count * self.feasible_fraction))
        boundary = int(round(count * self.boundary_fraction))
        if exploitation + boundary > count:
            boundary = count - exploitation
        global_count = count - exploitation - boundary
        exploit_score = (
            "acquisition_score"
            if "acquisition_score" in pool
            else "p_structural_feasible"
            if "p_structural_feasible" in pool
            else None
        )
        selected: list[int] = []
        roles: dict[int, str] = {}
        for role, tranche_count, score in (
            ("predicted_feasible", exploitation, exploit_score),
            ("structural_boundary", boundary, "structural_entropy"),
            ("global_diversity", global_count, "global_distance"),
        ):
            chosen = _diverse_select(
                pool,
                context.parameter_names,
                tranche_count,
                score_column=score,
                excluded_indices=selected,
            )
            selected.extend(chosen)
            roles.update({index: role for index in chosen})
        if len(selected) < count:
            chosen = _diverse_select(
                pool,
                context.parameter_names,
                count - len(selected),
                score_column="global_distance",
                excluded_indices=selected,
            )
            selected.extend(chosen)
            roles.update({index: "batch_fill" for index in chosen})
        proposals = pool.loc[selected].copy().reset_index(drop=True)
        proposals.insert(0, "selection_role", [roles[index] for index in selected])
        proposals.insert(0, "candidate_id", np.arange(1, len(proposals) + 1))
        return AcquisitionResult(
            proposals,
            self.name,
            self.capability,
            self.scientific_convergence_capable,
        )


def _mechanical_proposals(
    assessed: pd.DataFrame, spec: RefinementSpec
) -> pd.DataFrame:
    if assessed.empty:
        return pd.DataFrame(columns=[
            "candidate_id",
            "selection_role",
            "parameter_key",
            *spec.parameter_names,
        ])
    eligible = assessed.loc[
        assessed["structural_feasible"].astype(bool)
        & ~assessed["mechanical_observed"].astype(bool)
    ].copy().reset_index(drop=True)
    if eligible.empty or spec.mechanical_proposals_per_round <= 0:
        return pd.DataFrame(columns=[
            "candidate_id",
            "selection_role",
            "parameter_key",
            *spec.parameter_names,
        ])
    selected = _diverse_select(
        eligible,
        spec.parameter_names,
        spec.mechanical_proposals_per_round,
        score_column="structural_minimum_margin",
    )
    proposals = eligible.loc[selected].copy().reset_index(drop=True)
    # Exact evidence can retain the acquisition metadata that originally led
    # to its evaluation.  The proposal artifact records the decision made in
    # *this* round, so replace those columns rather than creating duplicate
    # labels or leaking a stale candidate id.
    proposals = proposals.drop(
        columns=["candidate_id", "selection_role"], errors="ignore"
    )
    proposals.insert(0, "selection_role", "measured_structural_feasible")
    proposals.insert(0, "candidate_id", np.arange(1, len(proposals) + 1))
    return proposals


def _validate_structural_proposals(
    proposals: Any,
    *,
    parameter_names: Sequence[str],
    maximum_count: int,
) -> pd.DataFrame:
    if not isinstance(proposals, pd.DataFrame):
        raise RefinementError("Structural acquisition proposals must be a DataFrame")
    result = proposals.copy()
    required = {"candidate_id", "selection_role", "parameter_key", *parameter_names}
    if not required.issubset(result.columns):
        missing = sorted(required - set(result.columns))
        raise RefinementError(f"Structural acquisition output is missing columns: {missing}")
    if len(result) > maximum_count:
        raise RefinementError(
            "Structural acquisition returned more proposals than the configured round budget"
        )
    if result["candidate_id"].duplicated().any():
        raise RefinementError("Structural proposal candidate_id values must be unique")
    expected_keys = [
        canonical_parameter_key(_parameter_mapping(row, parameter_names))
        for row in result.to_dict(orient="records")
    ]
    if list(result["parameter_key"].astype(str)) != expected_keys:
        raise RefinementError(
            "Structural proposal parameter_key does not match its parameter values"
        )
    if result["parameter_key"].duplicated().any():
        raise RefinementError("Structural proposal parameter keys must be unique")
    return result


def _read_previous_state(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    source = Path(path).resolve()
    if not source.is_file():
        raise RefinementError(f"Explicit previous state does not exist: {source}")
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise RefinementError(f"Cannot read previous refinement state {source}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema") != _STATE_SCHEMA:
        raise RefinementError(f"Previous state {source} has an unsupported schema")
    if raw.get("status") in {"converged", "budget_exhausted"}:
        raise RefinementError(
            f"Previous refinement is terminal ({raw.get('status')}); no next round is valid"
        )
    return raw


def _progress_state(
    *,
    previous: Mapping[str, Any],
    round_number: int,
    round_best: float | None,
    round_best_key: str | None,
    spec: RefinementSpec,
    convergence_capable: bool,
) -> dict[str, Any]:
    previous_best_raw = previous.get("best_mechanical_max_error_percent")
    previous_best = (
        float(previous_best_raw)
        if previous_best_raw is not None and math.isfinite(float(previous_best_raw))
        else math.inf
    )
    current_best = round_best if round_best is not None else math.inf
    if current_best < previous_best:
        improvement = (
            previous_best - current_best if math.isfinite(previous_best) else None
        )
        best = current_best
        best_key = round_best_key
        stale = 0
    else:
        improvement = 0.0 if math.isfinite(previous_best) else None
        best = previous_best
        best_key = previous.get("best_candidate_parameter_key")
        stale = int(previous.get("no_improvement_rounds", 0)) + 1
    if improvement is not None and improvement < spec.minimum_improvement_percent_points:
        stale = int(previous.get("no_improvement_rounds", 0)) + 1

    incumbent_available = math.isfinite(best)
    patience_reached = incumbent_available and stale >= spec.patience
    if patience_reached and convergence_capable:
        status = "converged"
        convergence_status = "static_search_converged"
        stop_reason = "static_mechanical_minimax_patience"
    elif round_number >= spec.maximum_rounds:
        status = "budget_exhausted"
        convergence_status = "budget_exhausted"
        stop_reason = "maximum_rounds"
    else:
        status = "active"
        convergence_status = (
            "incomplete_capability"
            if not convergence_capable
            else "searching"
        )
        stop_reason = ""
    return {
        "status": status,
        "convergence_status": convergence_status,
        "stop_reason": stop_reason,
        "convergence_scope": "static_structure_constrained_mechanical_search",
        "requires_finalist_validation": True,
        "best_mechanical_max_error_percent": best if incumbent_available else None,
        "best_candidate_parameter_key": best_key if incumbent_available else None,
        "round_improvement_percent_points": improvement,
        "no_improvement_rounds": stale,
    }


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _output_paths(directory: Path) -> dict[str, Path]:
    return {label: directory / name for label, name in _OUTPUT_NAMES.items()}


@dataclass(frozen=True)
class RefinementRoundResult:
    """Published round state and whether an existing manifest was reused."""

    output_dir: Path
    state: Mapping[str, Any]
    reused: bool


def run_refinement_round(
    *,
    spec: RefinementSpec | Mapping[str, Any],
    structural_paths: Sequence[str | os.PathLike[str]],
    mechanical_paths: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    previous_state_path: str | os.PathLike[str] | None = None,
    candidate_pool_path: str | os.PathLike[str] | None = None,
    config_artifact_path: str | os.PathLike[str] | None = None,
    acquisition: StructuralAcquisition | None = None,
    seed: int = 20260820,
    expected_round: int | None = None,
) -> RefinementRoundResult:
    """Plan and publish exactly one immutable refinement round.

    ``structural_paths`` and ``mechanical_paths`` must enumerate the complete
    evidence set intended for this round; no filesystem discovery occurs.
    ``candidate_pool_path`` is also explicit.  The default backend merely
    selects from that pool and is therefore marked incapable of proving
    scientific convergence.  A future surrogate backend can implement
    :class:`StructuralAcquisition` and make a stronger, auditable claim.
    """
    validated = spec if isinstance(spec, RefinementSpec) else RefinementSpec.from_mapping(spec)
    if not structural_paths:
        raise RefinementError("At least one explicit structural evidence path is required")
    structural_files = tuple(Path(item).resolve() for item in structural_paths)
    mechanical_files = tuple(Path(item).resolve() for item in mechanical_paths)
    previous_path = Path(previous_state_path).resolve() if previous_state_path else None
    pool_path = Path(candidate_pool_path).resolve() if candidate_pool_path else None
    previous = _read_previous_state(previous_path)
    round_number = int(previous.get("completed_round", 0)) + 1
    if expected_round is not None and round_number != int(expected_round):
        raise RefinementError(
            f"Explicit round is {expected_round}, but previous state requires round "
            f"{round_number}"
        )
    if round_number > validated.maximum_rounds:
        raise RefinementError(
            f"Round {round_number} exceeds maximum_rounds={validated.maximum_rounds}"
        )
    backend = acquisition or ExplicitPoolAcquisition()
    backend_identity = backend.scientific_identity()
    if not isinstance(backend_identity, Mapping):
        raise RefinementError("Acquisition scientific_identity() must return a mapping")
    declared_convergence_capable = bool(backend.scientific_convergence_capable)
    identity = {
        "backend": str(backend.name),
        "capability": str(backend.capability),
        "scientific_convergence_capable": declared_convergence_capable,
        "settings": dict(backend_identity),
    }
    campaign_fingerprint = scientific_config_hash({
        "refinement": validated.scientific_identity(),
        "acquisition": identity,
    })
    if previous:
        if previous.get("campaign_fingerprint") != campaign_fingerprint:
            raise RefinementError(
                "Previous state belongs to a different refinement/acquisition campaign"
            )
        if int(previous.get("selection_seed", seed)) != int(seed):
            raise RefinementError("selection seed changed within a refinement campaign")
    scientific_config = {
        "refinement": validated.scientific_identity(),
        "acquisition": identity,
        "round_number": round_number,
    }
    inputs: dict[str, Path] = {
        **{f"structural_{index:03d}": path for index, path in enumerate(structural_files)},
        **{f"mechanical_{index:03d}": path for index, path in enumerate(mechanical_files)},
    }
    if previous_path is not None:
        inputs["previous_state"] = previous_path
    if pool_path is not None:
        inputs["candidate_pool"] = pool_path
    if config_artifact_path is not None:
        inputs["config"] = Path(config_artifact_path).resolve()

    destination = Path(output_dir).resolve()
    manifest_path = destination / "artifact_manifest.json"
    identifier = f"constrained_refinement_round_{round_number:04d}"
    if destination.exists():
        paths = _output_paths(destination)
        decision = check_artifact_reuse(
            manifest_path,
            kind="stage",
            identifier=identifier,
            parameters=None,
            seeds=[int(seed)],
            scientific_config=scientific_config,
            input_artifacts=inputs,
            expected_outputs=paths,
        )
        if decision.reusable:
            state = json.loads(paths["state"].read_text(encoding="utf-8"))
            return RefinementRoundResult(destination, state, True)
        raise RefinementError(
            f"Output directory cannot be reused safely: {destination}: {decision.reason}"
        )

    structural = _read_evidence(
        structural_files, validated.parameter_names, label="structural"
    )
    mechanical = _read_evidence(
        mechanical_files, validated.parameter_names, label="mechanical"
    )
    mechanical = _enrich_cubic_mechanics(mechanical, validated.derivation)
    candidate_pool = (
        _read_evidence([pool_path], validated.parameter_names, label="candidate-pool")
        if pool_path is not None
        else pd.DataFrame(columns=[*validated.parameter_names, "parameter_key"])
    )
    assessed, ranking = assess_and_rank_candidates(structural, mechanical, validated)
    mechanical_proposals = _mechanical_proposals(assessed, validated)
    acquisition_result = backend.propose(AcquisitionContext(
        candidate_pool=candidate_pool,
        structural_observations=structural,
        mechanical_observations=mechanical,
        parameter_names=validated.parameter_names,
        round_number=round_number,
        proposal_count=validated.structural_proposals_per_round,
        seed=int(seed),
        structural_constraints=validated.structural_constraints,
        mechanical_objectives=validated.mechanical_objectives,
        refinement_spec=validated,
    ))
    if acquisition_result.backend != backend.name:
        raise RefinementError("Acquisition result backend does not match backend.name")
    if acquisition_result.capability != backend.capability:
        raise RefinementError("Acquisition result capability does not match backend.capability")
    if (
        bool(acquisition_result.scientific_convergence_capable)
        and not declared_convergence_capable
    ):
        raise RefinementError(
            "Acquisition result claims scientific convergence capability that the "
            "backend did not declare"
        )
    if not isinstance(acquisition_result.diagnostics, Mapping):
        raise RefinementError("Acquisition diagnostics must be a mapping")
    acquisition_diagnostics = dict(acquisition_result.diagnostics)
    try:
        json.dumps(acquisition_diagnostics, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise RefinementError(
            "Acquisition diagnostics must contain only finite strict-JSON values"
        ) from exc
    convergence_capable = bool(
        acquisition_result.scientific_convergence_capable
    )
    structural_proposals = _validate_structural_proposals(
        acquisition_result.proposals,
        parameter_names=validated.parameter_names,
        maximum_count=validated.structural_proposals_per_round,
    )

    round_best = (
        float(ranking.iloc[0]["mechanical_max_error_percent"])
        if len(ranking)
        else None
    )
    round_best_key = str(ranking.iloc[0]["parameter_key"]) if len(ranking) else None
    progress = _progress_state(
        previous=previous,
        round_number=round_number,
        round_best=round_best,
        round_best_key=round_best_key,
        spec=validated,
        convergence_capable=convergence_capable,
    )
    within_threshold = int(
        ranking.get(
            "within_mechanical_report_threshold", pd.Series(dtype=bool)
        ).astype(bool).sum()
    )
    state = {
        "schema": _STATE_SCHEMA,
        **progress,
        "campaign_fingerprint": campaign_fingerprint,
        "selection_seed": int(seed),
        "completed_round": round_number,
        "maximum_rounds": validated.maximum_rounds,
        "patience": validated.patience,
        "minimum_improvement_percent_points": (
            validated.minimum_improvement_percent_points
        ),
        "report_threshold_percent": validated.report_threshold_percent,
        "best_effort_retained": bool(progress["best_candidate_parameter_key"]),
        "round_best_mechanical_max_error_percent": round_best,
        "round_best_candidate_parameter_key": round_best_key,
        "structural_observations": int(len(structural)),
        "mechanical_observations": int(len(mechanical)),
        "matched_candidates": int(assessed["mechanical_observed"].sum())
        if len(assessed)
        else 0,
        "structurally_feasible_candidates": int(assessed["structural_feasible"].sum())
        if len(assessed)
        else 0,
        "eligible_mechanical_candidates": int(len(ranking)),
        "within_mechanical_report_threshold": within_threshold,
        "mechanical_proposals": int(len(mechanical_proposals)),
        "structural_proposals": int(len(structural_proposals)),
        "acquisition_backend": acquisition_result.backend,
        "acquisition_capability": acquisition_result.capability,
        "acquisition_scientific_convergence_capable": bool(
            acquisition_result.scientific_convergence_capable
        ),
        "acquisition_diagnostics": acquisition_diagnostics,
        "outputs": dict(_OUTPUT_NAMES),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    ))
    try:
        paths = _output_paths(staging)
        assessed.to_csv(paths["assessed"], index=False)
        ranking.to_csv(paths["ranking"], index=False)
        mechanical_proposals.to_csv(paths["mechanical_proposals"], index=False)
        structural_proposals.to_csv(paths["structural_proposals"], index=False)
        _atomic_json(paths["state"], state)
        report_lines = [
            "# Structure-constrained mechanical minimax round",
            "",
            f"- Status: {state['status']} ({state['convergence_status']})",
            f"- Completed round: {round_number}/{validated.maximum_rounds}",
            f"- Acquisition: {acquisition_result.backend} / {acquisition_result.capability}",
            f"- Exact structurally eligible mechanical candidates: {len(ranking)}",
            f"- Candidates in the {validated.report_threshold_percent:g}% reporting tier: "
            f"{within_threshold}",
            f"- Mechanical proposals: {len(mechanical_proposals)}",
            f"- Structural proposals: {len(structural_proposals)}",
            "",
            (
                f"Best measured M-infinity: {round_best:.6f}% (best effort retained)."
                if round_best is not None
                else "No exact structurally eligible mechanical observation is available yet."
            ),
            "",
            "The reporting threshold is not an eligibility gate. Structural constraints",
            "are exact hard gates; M-infinity and then RMSE rank the remaining candidates.",
        ]
        paths["report"].write_text("\n".join(report_lines) + "\n", encoding="utf-8")
        _atomic_json(paths["disposition"], {
            "schema": "ffopt-stage-disposition-v1",
            "stage": identifier,
            "disposition": "executed",
            "round": round_number,
            "status": state["status"],
            "convergence_status": state["convergence_status"],
        })
        manifest = build_artifact_manifest(
            kind="stage",
            identifier=identifier,
            parameters=None,
            seeds=[int(seed)],
            scientific_config=scientific_config,
            input_artifacts=inputs,
            expected_outputs=paths,
        )
        write_artifact_manifest(staging / "artifact_manifest.json", manifest)
        os.replace(staging, destination)
    except (OSError, ArtifactManifestError, ValueError):
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return RefinementRoundResult(destination, state, False)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Plan exactly one explicit structure-constrained mechanical "
            "minimax refinement round"
        )
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help=(
            "JSON/YAML refinement spec, or a runtime config containing a "
            "constrained_refinement mapping"
        ),
    )
    parser.add_argument(
        "--structural-observations",
        action="append",
        required=True,
        type=Path,
        help="Explicit structural CSV; repeat for cumulative evidence files",
    )
    parser.add_argument(
        "--static-observations",
        "--mechanical-observations",
        dest="mechanical_observations",
        action="append",
        default=[],
        type=Path,
        help="Explicit static/mechanical CSV; repeat for cumulative evidence files",
    )
    parser.add_argument(
        "--candidate-pool",
        type=Path,
        help="Explicit externally generated/scored candidate CSV",
    )
    parser.add_argument(
        "--previous-state",
        type=Path,
        help="refinement_state.json from the immediately preceding round",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--round",
        type=int,
        help="Expected one-based round number; checked against --previous-state",
    )
    parser.add_argument(
        "--max-rounds",
        type=int,
        help="Override maximum_rounds (use the same value for every campaign round)",
    )
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--feasible-fraction", type=float, default=0.60)
    parser.add_argument("--boundary-fraction", type=float, default=0.25)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Module CLI for pipeline integration; it never trains or submits jobs."""
    args = _build_argument_parser().parse_args(argv)
    config_path = args.config.resolve()
    loaded = load_config(config_path)
    raw_spec = loaded.get("constrained_refinement", loaded)
    if not isinstance(raw_spec, Mapping):
        raise RefinementError("constrained_refinement config section must be a mapping")
    spec = RefinementSpec.from_mapping(raw_spec)
    if args.max_rounds is not None:
        spec = replace(
            spec,
            maximum_rounds=_positive_int(args.max_rounds, field="--max-rounds"),
        )
    result = run_refinement_round(
        spec=spec,
        structural_paths=args.structural_observations,
        mechanical_paths=args.mechanical_observations,
        candidate_pool_path=args.candidate_pool,
        previous_state_path=args.previous_state,
        output_dir=args.output_dir,
        config_artifact_path=config_path,
        acquisition=ExplicitPoolAcquisition(
            feasible_fraction=args.feasible_fraction,
            boundary_fraction=args.boundary_fraction,
        ),
        seed=args.seed,
        expected_round=args.round,
    )
    payload = dict(result.state)
    payload["output_dir"] = str(result.output_dir)
    payload["reused"] = result.reused
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


__all__ = [
    "AcquisitionContext",
    "AcquisitionResult",
    "ExplicitPoolAcquisition",
    "MechanicalObjectiveSpec",
    "RefinementError",
    "RefinementRoundResult",
    "RefinementSpec",
    "StructuralAcquisition",
    "StructuralConstraintSpec",
    "assess_and_rank_candidates",
    "run_refinement_round",
]


if __name__ == "__main__":
    raise SystemExit(main())
