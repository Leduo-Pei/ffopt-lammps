"""Candidate-batch and finalist orchestration for cubic elasticity.

This module composes :mod:`engine.cubic_elastic_runner`; it does not duplicate
the static or dynamic elasticity implementation.  Candidate directories are
addressed by the canonical free-parameter key, never by an input rank.  The
selection policy is deliberately lexicographic:

1. measured structural gates;
2. cubic Born stability;
3. minimum fit R2;
4. finite minimax error over independent B/Cprime/C44 targets;
5. relative RMSE;
6. same-element artificial-type contrast; and
7. structural gate margin.

The configured percentage tier labels quality but is not a hard gate.  A
hard-gate-eligible candidate outside that tier remains the best-effort result.
An empty eligible set is a valid scientific outcome and still produces a
complete, manifested stage.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import json
import math
import os
from pathlib import Path
import shlex
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .config_loader import load_config
from .cubic_elastic_runner import (
    DYNAMIC_PROTOCOL,
    STATIC_PROTOCOL,
    ElasticStateBackend,
    LammpsTemplateElasticBackend,
    ResourceBudget,
    SingleCandidateCubicElasticRunner,
)
from .lammps_interface import LAMMPSRunner
from .parameter_space import build_parameter_space
from .resources import resolve_builtin_resource
from workflow.artifact_manifest import (
    build_artifact_manifest,
    canonical_parameter_key,
    check_artifact_reuse,
    scientific_config_hash,
    sha256_file,
    write_artifact_manifest,
)
from workflow.lammps_data import inspect_lammps_data


PROTOCOL_ALIASES = {
    "static": STATIC_PROTOCOL,
    "dynamic": DYNAMIC_PROTOCOL,
    STATIC_PROTOCOL: STATIC_PROTOCOL,
    DYNAMIC_PROTOCOL: DYNAMIC_PROTOCOL,
}
INDEPENDENT_TARGETS = ("B", "Cprime", "C44")


class CubicElasticBatchError(RuntimeError):
    """Base error for candidate-batch orchestration."""


class CandidateParameterError(CubicElasticBatchError):
    """Raised when candidate parameters are missing, invalid, or tampered."""


class BatchArtifactError(CubicElasticBatchError):
    """Raised when an immutable batch or candidate artifact changed."""


@dataclass(frozen=True)
class NestedResourcePlan:
    """Audited outer-candidate and inner-state concurrency allocation."""

    available_cores: int
    cores_per_state: int
    candidate_workers: int
    state_workers: int
    omp_threads_per_rank: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("available_cores", self.available_cores),
            ("cores_per_state", self.cores_per_state),
            ("candidate_workers", self.candidate_workers),
            ("state_workers", self.state_workers),
            ("omp_threads_per_rank", self.omp_threads_per_rank),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.allocated_cores > self.available_cores:
            raise ValueError(
                "Nested cubic-elastic batch would oversubscribe the allocation: "
                f"candidate_workers({self.candidate_workers}) * "
                f"state_workers({self.state_workers}) * "
                f"cores_per_state({self.cores_per_state}) * "
                f"omp_threads_per_rank({self.omp_threads_per_rank}) = "
                f"{self.allocated_cores} "
                f"> available_cores({self.available_cores})"
            )

    @property
    def allocated_cores(self) -> int:
        return (
            self.candidate_workers
            * self.state_workers
            * self.cores_per_state
            * self.omp_threads_per_rank
        )

    def candidate_budget(self) -> ResourceBudget:
        return ResourceBudget(
            max_workers=self.state_workers,
            cores_per_state=self.cores_per_state,
            available_cores=(
                self.state_workers
                * self.cores_per_state
                * self.omp_threads_per_rank
            ),
            omp_threads_per_rank=self.omp_threads_per_rank,
        )

    def to_dict(self) -> dict[str, int]:
        return {
            "available_cores": self.available_cores,
            "cores_per_state": self.cores_per_state,
            "candidate_workers": self.candidate_workers,
            "state_workers": self.state_workers,
            "omp_threads_per_rank": self.omp_threads_per_rank,
            "allocated_cores": self.allocated_cores,
        }


def plan_nested_resources(
    *,
    available_cores: int,
    cores_per_state: int,
    candidate_workers: int,
    state_workers: int | None = None,
    omp_threads_per_rank: int = 1,
) -> NestedResourcePlan:
    """Create an explicit non-oversubscribing nested execution plan."""

    if any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in (
            available_cores,
            cores_per_state,
            candidate_workers,
            omp_threads_per_rank,
        )
    ):
        raise ValueError(
            "available_cores, cores_per_state, candidate_workers and "
            "omp_threads_per_rank must be positive integers"
        )
    capacity = available_cores // (
        cores_per_state * omp_threads_per_rank * candidate_workers
    )
    if capacity < 1:
        raise ValueError(
            "The allocation cannot provide one state worker per candidate worker"
        )
    selected = capacity if state_workers is None else state_workers
    return NestedResourcePlan(
        available_cores=available_cores,
        cores_per_state=cores_per_state,
        candidate_workers=candidate_workers,
        state_workers=selected,
        omp_threads_per_rank=omp_threads_per_rank,
    )


def effective_available_cores(requested: int) -> int:
    """Cap a declared budget by the complete active SLURM CPU allocation."""

    if isinstance(requested, bool) or not isinstance(requested, int) or requested < 1:
        raise ValueError("requested available cores must be a positive integer")
    raw_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    raw_tasks = os.environ.get("SLURM_NTASKS")
    if raw_cpus is None:
        return requested
    try:
        cpus_per_task = int(raw_cpus)
        task_count = int(raw_tasks) if raw_tasks is not None else 1
    except ValueError as exc:
        raise ValueError(
            "SLURM_CPUS_PER_TASK and SLURM_NTASKS must be integers"
        ) from exc
    allocated = cpus_per_task * task_count
    if cpus_per_task < 1 or task_count < 1:
        raise ValueError(
            "SLURM_CPUS_PER_TASK and SLURM_NTASKS must be positive"
        )
    return min(requested, allocated)


@dataclass(frozen=True)
class Candidate:
    parameter_key: str
    raw_parameters: dict[str, float]
    source_row: dict[str, Any]
    source_rank: int

    @property
    def digest(self) -> str:
        return self.parameter_key.rsplit(":", 1)[-1]


@dataclass(frozen=True)
class PreparedCandidate:
    candidate: Candidate
    resolved_parameters: dict[str, float]
    force_field_include: Path
    parameter_contrast: dict[str, float]


BackendFactory = Callable[..., ElasticStateBackend]


def _strict_json_bytes(value: Any) -> bytes:
    try:
        return (
            json.dumps(
                value,
                indent=2,
                sort_keys=True,
                ensure_ascii=True,
                allow_nan=False,
            )
            + "\n"
        ).encode("ascii")
    except (TypeError, ValueError) as exc:
        raise CubicElasticBatchError(f"value is not strict JSON: {exc}") from exc


def _write_immutable_json(path: Path, value: Any) -> None:
    content = _strict_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise BatchArtifactError(f"refusing to overwrite immutable artifact: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != content:
                raise BatchArtifactError(
                    f"refusing to overwrite immutable artifact: {path}"
                )
        except OSError:
            if path.exists():
                if path.read_bytes() != content:
                    raise BatchArtifactError(
                        f"refusing to overwrite immutable artifact: {path}"
                    )
            else:
                os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    content = _strict_json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _scalar(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if pd.isna(value):
        return None
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _candidate_rows(path: Path) -> list[dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        frame = pd.read_csv(
            path,
            sep="\t" if suffix == ".tsv" else ",",
            float_precision="round_trip",
        )
        return [
            {str(key): _scalar(value) for key, value in row.items()}
            for row in frame.to_dict(orient="records")
        ]
    if suffix == ".json":
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise CandidateParameterError(f"cannot read candidate JSON {path}: {exc}") from exc
        if isinstance(value, dict) and "candidates" in value:
            if set(value) != {"candidates"}:
                raise CandidateParameterError(
                    "candidate JSON wrapper may contain only the 'candidates' field"
                )
            value = value["candidates"]
        if isinstance(value, dict):
            value = [value]
        if not isinstance(value, list) or any(not isinstance(row, dict) for row in value):
            raise CandidateParameterError(
                "candidate JSON must be an object, an array of objects, or {'candidates': [...]}"
            )
        return [{str(key): _scalar(item) for key, item in row.items()} for row in value]
    raise CandidateParameterError("candidate file must use .csv, .tsv, or .json")


def load_candidates(
    path: str | os.PathLike[str],
    parameter_space: Sequence[tuple[str, float, float]],
) -> list[Candidate]:
    """Load, validate, key, and deduplicate candidates without using rank identity."""

    source = Path(path).resolve()
    rows = _candidate_rows(source)
    names = [item[0] for item in parameter_space]
    bounds = {name: (float(lower), float(upper)) for name, lower, upper in parameter_space}
    candidates: list[Candidate] = []
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        missing = [name for name in names if name not in row or row[name] is None]
        if missing:
            raise CandidateParameterError(
                f"candidate row {index} is missing free parameters: {missing}"
            )
        raw: dict[str, float] = {}
        for name in names:
            try:
                value = float(row[name])
            except (TypeError, ValueError) as exc:
                raise CandidateParameterError(
                    f"candidate row {index} parameter {name!r} is not numeric"
                ) from exc
            if not math.isfinite(value):
                raise CandidateParameterError(
                    f"candidate row {index} parameter {name!r} is not finite"
                )
            lower, upper = bounds[name]
            tolerance = 1.0e-12 * max(1.0, abs(lower), abs(upper), abs(value))
            if value < lower - tolerance or value > upper + tolerance:
                raise CandidateParameterError(
                    f"candidate row {index} parameter {name!r}={value:.12g} "
                    f"is outside [{lower:.12g}, {upper:.12g}]"
                )
            raw[name] = value
        key = canonical_parameter_key(raw)
        recorded_key = row.get("parameter_key")
        if recorded_key is not None and str(recorded_key) != key:
            raise CandidateParameterError(
                f"candidate row {index} parameter_key does not match its parameter values"
            )
        if key in seen:
            continue
        seen.add(key)
        rank_value = row.get("candidate_rank", row.get("rank", index))
        try:
            source_rank = int(rank_value)
        except (TypeError, ValueError):
            source_rank = index
        candidates.append(Candidate(key, raw, dict(row), source_rank))
    return candidates


def _first_finite(row: Mapping[str, Any], names: Sequence[str]) -> float | None:
    for name in names:
        if name not in row or row[name] is None:
            continue
        try:
            value = float(row[name])
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            return value
    return None


def _property_value(row: Mapping[str, Any], name: str) -> float | None:
    return _first_finite(
        row,
        (
            f"calc_{name}",
            f"mean_{name}",
            name,
        ),
    )


def assess_structural_gates(
    row: Mapping[str, Any], config: Mapping[str, Any]
) -> dict[str, Any]:
    """Evaluate compiled measured-property gates and report a signed margin."""

    elasticity = config.get("elasticity", {})
    selection = elasticity.get("selection", {}) if isinstance(elasticity, Mapping) else {}
    gates = selection.get("structural_gates", {}) if isinstance(selection, Mapping) else {}
    targets = config.get("targets", {})
    if not gates:
        previous = row.get("structural_gate_pass")
        passed = True if previous is None else str(previous).lower() in {"true", "1", "yes"}
        margin = _first_finite(row, ("structural_margin", "structural_gate_margin"))
        return {
            "structural_gate_pass": passed,
            "structural_margin": (
                1.0 if margin is None and passed else -math.inf if margin is None else margin
            ),
            "structural_gate_reason": "" if passed else "precomputed_structural_gate",
            "structural_errors": {},
        }

    group_properties = {
        "lattice": ("a", "b", "c"),
        "angles": ("alpha", "beta", "gamma_ang"),
        "density": ("density",),
        "surface": ("surf_energy",),
    }
    errors: dict[str, float] = {}
    normalized_margins: list[float] = []
    reasons: list[str] = []
    for group, specification in gates.items():
        if group not in group_properties or not isinstance(specification, Mapping):
            reasons.append(f"unsupported_gate:{group}")
            normalized_margins.append(-math.inf)
            continue
        if group == "angles":
            limit = float(specification["maximum_absolute_error_degree"])
        else:
            limit = float(specification["maximum_relative_error_percent"])
        for name in group_properties[group]:
            target_spec = targets.get(name)
            target = (
                float(target_spec["value"])
                if isinstance(target_spec, Mapping) and "value" in target_spec
                else None
            )
            calculated = _property_value(row, name)
            if target is None or calculated is None or not math.isfinite(target):
                errors[name] = math.inf
                normalized_margins.append(-math.inf)
                reasons.append(f"missing_measured_{name}")
                continue
            if group == "angles":
                error = abs(calculated - target)
            else:
                if abs(target) <= 1.0e-15:
                    error = math.inf
                else:
                    error = 100.0 * abs(calculated - target) / abs(target)
            errors[name] = error
            margin = 1.0 - error / limit if math.isfinite(error) else -math.inf
            normalized_margins.append(margin)
            if error > limit + 1.0e-12:
                reasons.append(f"{name}_error>{limit:g}")
    margin = min(normalized_margins) if normalized_margins else 1.0
    return {
        "structural_gate_pass": bool(not reasons and margin >= -1.0e-12),
        "structural_margin": float(margin),
        "structural_gate_reason": ";".join(reasons),
        "structural_errors": errors,
    }


def same_element_parameter_contrast(
    resolved: Mapping[str, float], config: Mapping[str, Any]
) -> dict[str, float]:
    """Return dimensionless epsilon/sigma contrast for artificial elemental types."""

    material = config.get("material", {})
    if not isinstance(material, Mapping) or material.get("kind") != "elemental":
        return {"epsilon_contrast": 0.0, "sigma_contrast": 0.0, "contrast": 0.0}
    labels = [str(item["label"]) for item in config.get("atom_types", [])]

    def contrast(parameter: str) -> float:
        values = [float(resolved[f"{label}_{parameter}"]) for label in labels]
        if len(values) < 2:
            return 0.0
        denominator = float(np.mean(np.abs(values)))
        if denominator <= 1.0e-15:
            return 0.0 if max(values) == min(values) else math.inf
        return (max(values) - min(values)) / denominator

    epsilon = contrast("epsilon")
    sigma = contrast("sigma")
    return {
        "epsilon_contrast": epsilon,
        "sigma_contrast": sigma,
        "contrast": max(epsilon, sigma),
    }


def _mechanical_metrics(
    values: Mapping[str, float], targets: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    signed: dict[str, float] = {}
    absolute: dict[str, float] = {}
    for name in INDEPENDENT_TARGETS:
        target = float(targets[name]["value"])
        value = float(values[name])
        error = 100.0 * (value - target) / abs(target)
        signed[name] = error
        absolute[name] = abs(error)
    vector = np.asarray(list(absolute.values()), dtype=float)
    return {
        "mechanical_signed_errors_percent": signed,
        "mechanical_errors_percent": absolute,
        "mechanical_max_error_percent": float(np.max(vector)),
        "mechanical_rmse_percent": float(math.sqrt(np.mean(vector**2))),
    }


def _base_result(
    candidate: Candidate,
    structural: Mapping[str, Any],
    prepared: PreparedCandidate | None,
) -> dict[str, Any]:
    row = {str(key): _scalar(value) for key, value in candidate.source_row.items()}
    row.update(candidate.raw_parameters)
    row.update({
        "parameter_key": candidate.parameter_key,
        "source_rank": candidate.source_rank,
        "structural_gate_pass": bool(structural["structural_gate_pass"]),
        "structural_margin": float(structural["structural_margin"]),
        "structural_gate_reason": str(structural["structural_gate_reason"]),
    })
    for name, error in structural["structural_errors"].items():
        row[f"structural_error_{name}"] = error
    if prepared is not None:
        row.update({
            "same_element_parameter_contrast": prepared.parameter_contrast["contrast"],
            "epsilon_contrast": prepared.parameter_contrast["epsilon_contrast"],
            "sigma_contrast": prepared.parameter_contrast["sigma_contrast"],
        })
    return row


def summarize_single_elastic_result(
    *,
    candidate: Candidate,
    prepared: PreparedCandidate,
    structural: Mapping[str, Any],
    summary: Mapping[str, Any],
    module: Mapping[str, Any],
    minimum_r2: float,
    born_required: bool,
    quality_tier_percent: float,
) -> dict[str, Any]:
    """Flatten one single-candidate summary into the common ranking contract."""

    row = _base_result(candidate, structural, prepared)
    constants = summary["elastic_constants_gpa"]
    independent = summary["independent_moduli_gpa"]
    diagnostics = summary["derived_diagnostics"]
    fit_quality = summary["fit_quality"]
    fit_r2 = float(
        fit_quality.get("minimum_r2", fit_quality.get("minimum_relevant_r2", math.nan))
    )
    born = bool(summary["born_stability"]["stable"])
    row.update({
        "calculation_status": "completed",
        "candidate_directory": str(summary.get("candidate_directory", "")),
        "candidate_fingerprint": summary["candidate_fingerprint"],
        "C11_gpa": float(constants["C11"]),
        "C12_gpa": float(constants["C12"]),
        "C44_gpa": float(constants["C44"]),
        "B_gpa": float(independent["B"]),
        "Cprime_gpa": float(independent["Cprime"]),
        "G_hill_gpa": diagnostics["G_hill_gpa"],
        "E_hill_gpa": diagnostics["E_hill_gpa"],
        "nu_hill": diagnostics["nu_hill"],
        "minimum_fit_r2": fit_r2,
        "born_stability_pass": born,
        "fit_quality_pass": bool(math.isfinite(fit_r2) and fit_r2 >= minimum_r2),
    })
    values = {
        "B": row["B_gpa"],
        "Cprime": row["Cprime_gpa"],
        "C44": row["C44_gpa"],
    }
    metrics = _mechanical_metrics(values, module["targets"])
    row.update(metrics)
    for name, value in metrics["mechanical_signed_errors_percent"].items():
        row[f"error_{name}_percent"] = value
    finite = all(math.isfinite(float(values[name])) for name in INDEPENDENT_TARGETS)
    row["finite_mechanical_score"] = finite
    row["finalist_eligible"] = bool(
        row["structural_gate_pass"]
        and (born or not born_required)
        and row["fit_quality_pass"]
        and finite
    )
    row["quality_tier_percent"] = quality_tier_percent
    row["within_quality_tier"] = bool(
        row["finalist_eligible"]
        and row["mechanical_max_error_percent"] <= quality_tier_percent + 1.0e-12
    )
    row["quality_status"] = (
        "within_tier"
        if row["within_quality_tier"]
        else "best_effort"
        if row["finalist_eligible"]
        else "rejected"
    )
    reasons = []
    if not row["structural_gate_pass"]:
        reasons.append("structural_gate")
    if born_required and not born:
        reasons.append("born_stability")
    if not row["fit_quality_pass"]:
        reasons.append(f"fit_r2<{minimum_r2:g}")
    if not finite:
        reasons.append("nonfinite_mechanical_score")
    row["finalist_rejection_reason"] = ";".join(reasons)
    return row


def _truth_series(frame: pd.DataFrame, name: str, default: bool = False) -> pd.Series:
    if name not in frame:
        return pd.Series(default, index=frame.index, dtype=bool)
    return frame[name].astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "on"}
    )


def normalize_ranking_contract(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize static-screen or constrained-refinement evidence columns."""

    normalized = frame.copy()
    aliases = {
        "structural_gate_pass": "structural_feasible",
        "structural_margin": "structural_minimum_margin",
        "born_stability_pass": "mechanical_stability_gate_pass",
        "fit_quality_pass": "mechanical_fit_gate_pass",
        "finalist_eligible": "mechanical_eligible",
    }
    for target, source in aliases.items():
        # Constrained-refinement gate columns are newer, explicit decisions;
        # when both are present they supersede any stale evidence copied from
        # the structural/static input table.
        if source in normalized:
            normalized[target] = normalized[source]
    if "finite_mechanical_score" not in normalized:
        maximum = pd.to_numeric(
            normalized.get(
                "mechanical_max_error_percent",
                pd.Series(math.nan, index=normalized.index),
            ),
            errors="coerce",
        )
        rmse = pd.to_numeric(
            normalized.get(
                "mechanical_rmse_percent",
                pd.Series(math.nan, index=normalized.index),
            ),
            errors="coerce",
        )
        normalized["finite_mechanical_score"] = np.isfinite(maximum) & np.isfinite(rmse)
    if "same_element_parameter_contrast" not in normalized:
        normalized["same_element_parameter_contrast"] = 0.0
    return normalized


def rank_elastic_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Apply the auditable gate-first, minimax-second finalist ordering."""

    ranked = normalize_ranking_contract(frame)
    if ranked.empty:
        for name in (
            "structural_gate_pass",
            "born_stability_pass",
            "fit_quality_pass",
            "finite_mechanical_score",
            "finalist_eligible",
        ):
            if name not in ranked:
                ranked[name] = pd.Series(dtype=bool)
        if "result_rank" not in ranked:
            ranked["result_rank"] = pd.Series(dtype=int)
        return ranked
    if "result_rank" in ranked:
        ranked = ranked.drop(columns=["result_rank"])
    ranked["structural_gate_pass"] = _truth_series(ranked, "structural_gate_pass")
    ranked["born_stability_pass"] = _truth_series(ranked, "born_stability_pass")
    ranked["fit_quality_pass"] = _truth_series(ranked, "fit_quality_pass")
    ranked["finite_mechanical_score"] = _truth_series(
        ranked, "finite_mechanical_score"
    )
    # Fail closed if an upstream table contains an inconsistent eligibility
    # flag.  The auditable gate evidence, rather than that convenience flag,
    # decides whether a row may enter finalist selection.
    recorded_eligible = _truth_series(ranked, "finalist_eligible")
    ranked["finalist_eligible"] = (
        recorded_eligible
        & ranked["structural_gate_pass"]
        & ranked["born_stability_pass"]
        & ranked["fit_quality_pass"]
        & ranked["finite_mechanical_score"]
    )
    numeric_defaults = {
        "mechanical_max_error_percent": math.inf,
        "mechanical_rmse_percent": math.inf,
        "same_element_parameter_contrast": math.inf,
        "structural_margin": -math.inf,
    }
    for name, default in numeric_defaults.items():
        values = (
            pd.to_numeric(ranked[name], errors="coerce")
            if name in ranked
            else pd.Series(default, index=ranked.index, dtype=float)
        )
        ranked[name] = values.fillna(default)
    if "parameter_key" not in ranked:
        ranked["parameter_key"] = ranked.index.map(str)
    ranked = ranked.sort_values(
        [
            "structural_gate_pass",
            "born_stability_pass",
            "fit_quality_pass",
            "finite_mechanical_score",
            "mechanical_max_error_percent",
            "mechanical_rmse_percent",
            "same_element_parameter_contrast",
            "structural_margin",
            "parameter_key",
        ],
        ascending=[False, False, False, False, True, True, True, False, True],
        kind="mergesort",
    ).reset_index(drop=True)
    ranked.insert(0, "result_rank", np.arange(1, len(ranked) + 1))
    return ranked


def select_finalists(
    frame: pd.DataFrame,
    *,
    parameter_space: Sequence[tuple[str, float, float]],
    top_n: int,
    near_optimal_window_percent: float,
    diversity_slots: int,
    minimum: int = 1,
) -> pd.DataFrame:
    """Select quality-first finalists, then satisfy a declared minimum floor.

    The near-optimal window protects quality and diversity.  When that window
    contains fewer than ``minimum`` eligible rows, the remaining slots are
    filled in the already deterministic M-infinity/RMSE ranking order.  The
    floor never admits a hard-gate failure and never exceeds ``top_n``.
    """

    if top_n < 1:
        raise ValueError("top_n must be positive")
    if minimum < 1 or minimum > top_n:
        raise ValueError("minimum must lie in [1, top_n]")
    if near_optimal_window_percent < 0.0 or not math.isfinite(near_optimal_window_percent):
        raise ValueError("near_optimal_window_percent must be finite and non-negative")
    if diversity_slots < 0 or diversity_slots >= top_n:
        raise ValueError("diversity_slots must lie in [0, top_n)")
    ranked = rank_elastic_results(frame)
    eligible = ranked.loc[ranked["finalist_eligible"]].reset_index(drop=True)
    if eligible.empty:
        empty = eligible.copy()
        empty["finalist_selection_order"] = pd.Series(dtype=int)
        empty["finalist_selection_role"] = pd.Series(dtype=str)
        return empty
    threshold = (
        float(eligible.loc[0, "mechanical_max_error_percent"])
        + near_optimal_window_percent
    )
    pool = eligible.loc[
        eligible["mechanical_max_error_percent"] <= threshold + 1.0e-12
    ].reset_index(drop=True)
    maximum = min(top_n, len(pool))
    diversity_count = min(diversity_slots, max(0, maximum - 1))
    quality_count = max(1, maximum - diversity_count)
    chosen = list(range(quality_count))
    roles = {index: "near_optimal_quality" for index in chosen}

    names = [name for name, _lower, _upper in parameter_space]
    remaining = set(range(quality_count, len(pool)))
    if diversity_count and remaining and all(name in pool for name in names):
        lows = np.asarray([lower for _name, lower, _upper in parameter_space], dtype=float)
        spans = np.asarray(
            [max(upper - lower, 1.0e-15) for _name, lower, upper in parameter_space],
            dtype=float,
        )
        points = (pool[names].to_numpy(float) - lows) / spans
        minimum_distance = np.full(len(pool), np.inf, dtype=float)
        for index in chosen:
            minimum_distance = np.minimum(
                minimum_distance,
                np.linalg.norm(points - points[index], axis=1),
            )
        while remaining and len(chosen) < maximum:
            index = max(
                remaining,
                key=lambda item: (minimum_distance[item], -item),
            )
            chosen.append(index)
            roles[index] = "near_optimal_diverse"
            remaining.remove(index)
            minimum_distance = np.minimum(
                minimum_distance,
                np.linalg.norm(points - points[index], axis=1),
            )

    selected = pool.iloc[chosen].copy()
    selected["finalist_selection_role"] = [roles[index] for index in chosen]
    if len(selected) < minimum:
        chosen_keys = set(selected["parameter_key"].astype(str))
        supplement = eligible.loc[
            ~eligible["parameter_key"].astype(str).isin(chosen_keys)
        ].head(minimum - len(selected)).copy()
        supplement["finalist_selection_role"] = "minimum_quality_floor"
        selected = pd.concat([selected, supplement], ignore_index=True)
    else:
        selected = selected.reset_index(drop=True)
    selected.insert(0, "finalist_selection_order", np.arange(1, len(selected) + 1))
    selected["near_optimal_threshold_percent"] = threshold
    return selected


def _protocol_module(config: Mapping[str, Any], protocol: str) -> Mapping[str, Any]:
    fidelity = "static" if protocol == STATIC_PROTOCOL else "dynamic"
    elasticity = config.get("elasticity")
    if not isinstance(elasticity, Mapping) or not elasticity.get("enabled", False):
        raise CubicElasticBatchError("compiled config does not enable cubic elasticity")
    modules = elasticity.get("modules", {})
    module = modules.get(fidelity) if isinstance(modules, Mapping) else None
    if not isinstance(module, Mapping):
        raise CubicElasticBatchError(f"compiled config has no elasticity {fidelity} module")
    targets = module.get("targets")
    if not isinstance(targets, Mapping) or set(targets) != set(INDEPENDENT_TARGETS):
        raise CubicElasticBatchError(
            f"elasticity {fidelity} targets must be exactly B, Cprime and C44"
        )
    return module


def _signed_strains(module: Mapping[str, Any]) -> tuple[float, ...]:
    protocol = module.get("protocol", {})
    magnitudes = sorted({float(value) for value in protocol["strain_magnitudes"]})
    if len(magnitudes) < 2 or any(
        not math.isfinite(value) or value <= 0.0 for value in magnitudes
    ):
        raise CubicElasticBatchError(
            "elasticity strain_magnitudes need at least two finite positive values"
        )
    return tuple([-value for value in reversed(magnitudes)] + magnitudes)


def _parameter_directory(output_dir: Path, candidate: Candidate) -> Path:
    return output_dir / "candidate_runs" / candidate.digest


def _copy_or_verify_generated(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise BatchArtifactError(f"immutable generated candidate input changed: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def prepare_candidate(
    candidate: Candidate,
    *,
    output_dir: Path,
    parameter_runner: LAMMPSRunner,
    config: Mapping[str, Any],
) -> PreparedCandidate:
    """Resolve constraints and publish one immutable force-field include."""

    resolved = parameter_runner._resolve_params(candidate.raw_parameters)
    feasibility = parameter_runner._resolved_parameter_feasibility_error(resolved)
    if feasibility:
        raise CandidateParameterError(
            f"candidate {candidate.parameter_key} violates resolved constraints: {feasibility}"
        )
    root = _parameter_directory(output_dir, candidate) / "inputs"
    identity = {
        "schema_version": 1,
        "parameter_key": candidate.parameter_key,
        "raw_free_parameters": candidate.raw_parameters,
        "resolved_parameters": resolved,
    }
    _write_immutable_json(root / "resolved_parameters.json", identity)
    with tempfile.TemporaryDirectory(prefix="ffopt-pair-") as temporary:
        generated = Path(parameter_runner._build_pair_coeffs(resolved, temporary))
        content = generated.read_bytes()
    include = root / "force_field.lmp"
    _copy_or_verify_generated(include, content)
    contrast = same_element_parameter_contrast(resolved, config)
    return PreparedCandidate(candidate, resolved, include, contrast)


def _lammps_command(config: Mapping[str, Any]) -> tuple[str, ...]:
    lammps = config["lammps"]
    parallel = config["parallel"]
    executable = str(lammps["executable"])
    omp_threads = max(1, int(parallel.get("omp_threads_per_worker", 1)))
    scheduler_launcher = str(parallel.get("scheduler_launcher", "")).strip()
    launcher = shlex.split(
        scheduler_launcher or str(lammps.get("mpiexec", "mpiexec"))
    )
    if not launcher:
        raise CubicElasticBatchError("lammps.mpiexec cannot be empty")
    name = Path(launcher[0]).name.lower()
    if name in {"srun", "srun.exe"}:
        return (
            *launcher,
            "--exact",
            "--exclusive",
            "--nodes=1",
            "--ntasks", "{cores}",
            "--cpus-per-task", str(omp_threads),
            executable,
        )
    if not bool(parallel.get("use_mpi", False)):
        return (executable,)
    return (*launcher, "-n", "{cores}", executable)


def _lammps_variables(
    config: Mapping[str, Any], module: Mapping[str, Any], protocol: str
) -> dict[str, Any]:
    bulk_data = Path(config["manifest"]["data_files"]["bulk"])
    atom_style = inspect_lammps_data(bulk_data).atom_style or "atomic"
    settings = module["protocol"]
    replicate = [int(value) for value in settings.get("replicate", (1, 1, 1))]
    if len(replicate) != 3 or any(value < 1 for value in replicate):
        raise CubicElasticBatchError("elasticity replicate must have three positive integers")
    variables: dict[str, Any] = {
        "units": "real",
        "atom_style": atom_style,
        "cutoff": float(config["lammps"]["cutoff"]),
        "replicate_x": replicate[0],
        "replicate_y": replicate[1],
        "replicate_z": replicate[2],
        "min_dmax": float(settings.get("min_dmax", 0.1)),
        "energy_tolerance": float(settings.get("energy_tolerance", 1.0e-10)),
        "force_tolerance": float(settings.get("force_tolerance", 1.0e-10)),
        "max_iterations": int(settings.get("max_iterations", 10000)),
        "max_evaluations": int(settings.get("max_evaluations", 100000)),
        "box_vmax": float(settings.get("box_vmax", 0.001)),
        "target_pressure_atm": 0.0,
    }
    if protocol == DYNAMIC_PROTOCOL:
        timestep = float(settings["timestep_fs"])
        production = int(settings["production_steps"])
        sample_every = int(settings.get("sample_every", 10))
        sample_repeat = int(settings.get("sample_repeat", min(100, production // sample_every)))
        sample_repeat = max(1, sample_repeat)
        frequency = sample_every * sample_repeat
        if production < frequency:
            sample_repeat = max(1, production // sample_every)
            frequency = sample_every * sample_repeat
        variables.update({
            "target_pressure_atm": float(config["lammps"]["bulk"].get("pressure", 1.0)),
            "temperature_k": float(settings["temperature_k"]),
            "tdamp": float(settings.get("tdamp_fs", 100.0 * timestep)),
            "pdamp": float(settings.get("pdamp_fs", 1000.0 * timestep)),
            "nvt_damp": float(settings.get("nvt_damp_fs", 100.0 * timestep)),
            "timestep": timestep,
            "thermo_every": int(settings.get("thermo_every", 1000)),
            "npt_equil_steps": int(settings["equilibration_steps"]),
            "nvt_equil_steps": int(
                settings.get("nvt_equilibration_steps", settings["equilibration_steps"])
            ),
            "nvt_prod_steps": production,
            "sample_every": sample_every,
            "sample_repeat": sample_repeat,
            "sample_frequency": frequency,
        })
    return variables


def default_backend_factory(
    *,
    protocol: str,
    config: Mapping[str, Any],
    module: Mapping[str, Any],
    trajectory_seed: int | None,
    force_field_include: Path,
) -> ElasticStateBackend:
    """Build the production LAMMPS backend for one candidate/seed work unit."""

    del trajectory_seed  # the single-candidate runner passes it to run_state
    prefix = "static" if protocol == STATIC_PROTOCOL else "dynamic"
    return LammpsTemplateElasticBackend(
        protocol=protocol,
        command=_lammps_command(config),
        reference_template=resolve_builtin_resource(
            f"builtin:lammps/inputs/elasticity/in.cubic.{prefix}_reference"
        ),
        strain_template=resolve_builtin_resource(
            f"builtin:lammps/inputs/elasticity/in.cubic.{prefix}_strain"
        ),
        bulk_data=Path(config["manifest"]["data_files"]["bulk"]),
        force_field_include=force_field_include,
        variables=_lammps_variables(config, module, protocol),
        timeout_seconds=float(config["lammps"].get("timeout", 21600)),
        omp_threads_per_rank=max(
            1, int(config.get("parallel", {}).get("omp_threads_per_worker", 1))
        ),
    )


def _single_scientific_config(
    config: Mapping[str, Any], module: Mapping[str, Any], protocol: str
) -> dict[str, Any]:
    selection = config["elasticity"]["selection"]
    reporting = config["elasticity"]["reporting"]
    result = {
        "elasticity_module": module,
        "selection": selection,
        "reporting": reporting,
        "material": config.get("material", {}),
        "crystal": config.get("crystal", {}),
    }
    if protocol == DYNAMIC_PROTOCOL:
        result.update({"stress_convention": "compression_positive", "stress_to_gpa": 0.000101325})
    return result


def _candidate_runner(
    *,
    prepared: PreparedCandidate,
    output_dir: Path,
    protocol: str,
    trajectory_seed: int | None,
    config: Mapping[str, Any],
    module: Mapping[str, Any],
    resources: NestedResourcePlan,
    backend_factory: BackendFactory,
) -> SingleCandidateCubicElasticRunner:
    protocol_dir = "static" if protocol == STATIC_PROTOCOL else f"dynamic/seed_{trajectory_seed}"
    candidate_dir = _parameter_directory(output_dir, prepared.candidate) / protocol_dir
    backend = backend_factory(
        protocol=protocol,
        config=config,
        module=module,
        trajectory_seed=trajectory_seed,
        force_field_include=prepared.force_field_include,
    )
    identifier = (
        f"{prepared.candidate.parameter_key}:{protocol}"
        if trajectory_seed is None
        else f"{prepared.candidate.parameter_key}:{protocol}:seed:{trajectory_seed}"
    )
    return SingleCandidateCubicElasticRunner(
        candidate_dir=candidate_dir,
        candidate_id=identifier,
        protocol=protocol,
        parameters=prepared.candidate.raw_parameters,
        scientific_config=_single_scientific_config(config, module, protocol),
        trajectory_seed=trajectory_seed,
        backend=backend,
        resource_budget=resources.candidate_budget(),
    )


def _failed_row(
    candidate: Candidate,
    structural: Mapping[str, Any],
    message: str,
) -> dict[str, Any]:
    row = _base_result(candidate, structural, None)
    row.update({
        "calculation_status": "failed",
        "calculation_error": message,
        "born_stability_pass": False,
        "fit_quality_pass": False,
        "finite_mechanical_score": False,
        "finalist_eligible": False,
        "mechanical_max_error_percent": math.inf,
        "mechanical_rmse_percent": math.inf,
        "same_element_parameter_contrast": math.inf,
        "within_quality_tier": False,
        "quality_status": "rejected",
        "finalist_rejection_reason": "calculation_failed",
    })
    return row


def aggregate_dynamic_seed_results(
    rows: Sequence[Mapping[str, Any]],
    *,
    candidate: Candidate,
    prepared: PreparedCandidate,
    structural: Mapping[str, Any],
    module: Mapping[str, Any],
    expected_seeds: Sequence[int],
    minimum_r2: float,
    born_required: bool,
    quality_tier_percent: float,
) -> dict[str, Any]:
    """Aggregate truly independent finite-T seed results for one parameter key."""

    successful = [row for row in rows if row.get("calculation_status") == "completed"]
    result = _base_result(candidate, structural, prepared)
    result.update({
        "calculation_status": "completed" if len(successful) == len(expected_seeds) else "partial",
        "expected_seeds": " ".join(str(seed) for seed in expected_seeds),
        "n_seeds_expected": len(expected_seeds),
        "n_seeds_success": len(successful),
        "all_seeds_complete": len(successful) == len(expected_seeds),
    })
    if not successful:
        return _failed_row(candidate, structural, "all finite-temperature seeds failed") | {
            "expected_seeds": result["expected_seeds"],
            "n_seeds_expected": len(expected_seeds),
            "n_seeds_success": 0,
            "all_seeds_complete": False,
        }

    aggregate_columns = (
        "C11_gpa",
        "C12_gpa",
        "C44_gpa",
        "B_gpa",
        "Cprime_gpa",
        "G_hill_gpa",
        "E_hill_gpa",
        "nu_hill",
    )
    for name in aggregate_columns:
        values = np.asarray([float(row[name]) for row in successful], dtype=float)
        result[name] = float(np.mean(values))
        result[f"{name}_mean"] = float(np.mean(values))
        result[f"{name}_std"] = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
    result["minimum_fit_r2"] = min(float(row["minimum_fit_r2"]) for row in successful)
    result["born_stability_pass"] = bool(
        len(successful) == len(expected_seeds)
        and all(bool(row["born_stability_pass"]) for row in successful)
    )
    result["fit_quality_pass"] = bool(
        len(successful) == len(expected_seeds)
        and result["minimum_fit_r2"] >= minimum_r2
    )
    metrics = _mechanical_metrics(
        {name: result[f"{name}_gpa"] for name in INDEPENDENT_TARGETS},
        module["targets"],
    )
    result.update(metrics)
    for name, value in metrics["mechanical_signed_errors_percent"].items():
        result[f"error_{name}_percent"] = value
    result["finite_mechanical_score"] = all(
        math.isfinite(float(result[f"{name}_gpa"])) for name in INDEPENDENT_TARGETS
    )
    result["finalist_eligible"] = bool(
        result["structural_gate_pass"]
        and result["all_seeds_complete"]
        and (result["born_stability_pass"] or not born_required)
        and result["fit_quality_pass"]
        and result["finite_mechanical_score"]
    )
    result["quality_tier_percent"] = quality_tier_percent
    result["within_quality_tier"] = bool(
        result["finalist_eligible"]
        and result["mechanical_max_error_percent"] <= quality_tier_percent + 1.0e-12
    )
    result["quality_status"] = (
        "within_tier"
        if result["within_quality_tier"]
        else "best_effort"
        if result["finalist_eligible"]
        else "rejected"
    )
    reasons = []
    if not result["structural_gate_pass"]:
        reasons.append("structural_gate")
    if not result["all_seeds_complete"]:
        reasons.append("incomplete_seed_set")
    if born_required and not result["born_stability_pass"]:
        reasons.append("born_stability")
    if not result["fit_quality_pass"]:
        reasons.append(f"fit_r2<{minimum_r2:g}")
    result["finalist_rejection_reason"] = ";".join(reasons)
    return result


def _stage_scientific_config(
    config: Mapping[str, Any],
    *,
    protocol: str,
    top_n: int,
    minimum: int,
    near_optimal_window_percent: float,
    diversity_slots: int,
    evaluate_structural_failures: bool,
) -> dict[str, Any]:
    return {
        "protocol": protocol,
        "elasticity": config["elasticity"],
        "material": config.get("material", {}),
        "crystal": config.get("crystal", {}),
        "parameter_space": build_parameter_space(dict(config)),
        "finalist_selection": {
            "top_n": top_n,
            "minimum": minimum,
            "near_optimal_window_percent": near_optimal_window_percent,
            "diversity_slots": diversity_slots,
        },
        "evaluate_structural_failures": bool(evaluate_structural_failures),
    }


def _batch_identity(
    *,
    config_path: Path,
    parameters_path: Path,
    scientific_config: Mapping[str, Any],
) -> dict[str, Any]:
    core = {
        "schema_version": 1,
        "config": sha256_file(config_path).to_dict(),
        "parameters": sha256_file(parameters_path).to_dict(),
        "scientific_config_sha256": scientific_config_hash(scientific_config),
    }
    return {**core, "batch_fingerprint": scientific_config_hash(core)}


def _output_paths(output_dir: Path, protocol: str) -> dict[str, Path]:
    result_name = "static_results.csv" if protocol == STATIC_PROTOCOL else "dynamic_results.csv"
    outputs = {
        "results": output_dir / result_name,
        "finalists": output_dir / "finalists_selected.csv",
        "best_candidate": output_dir / "best_candidate.json",
        "batch_summary": output_dir / "batch_summary.json",
    }
    if protocol == DYNAMIC_PROTOCOL:
        outputs["seed_results"] = output_dir / "dynamic_seed_results.csv"
    return outputs


def _stage_inputs(
    *,
    output_dir: Path,
    config_path: Path,
    parameters_path: Path,
    manifest_paths: Sequence[Path],
) -> dict[str, Path]:
    inputs = {
        "batch_identity": output_dir / "batch_identity.json",
        "config": config_path,
        "parameters": parameters_path,
    }
    for index, path in enumerate(sorted(set(manifest_paths))):
        inputs[f"candidate_manifest_{index:06d}"] = path
    return inputs


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _best_candidate_document(
    ranked: pd.DataFrame,
    parameter_names: Sequence[str],
) -> dict[str, Any]:
    eligible = ranked.loc[ranked["finalist_eligible"]] if not ranked.empty else ranked
    if eligible.empty:
        return {
            "schema_version": 1,
            "status": "zero_hard_gate_eligible",
            "message": "No candidate passed structure, Born, R2, and finite-score gates.",
        }
    row = eligible.iloc[0]
    within = bool(row["within_quality_tier"])
    return {
        "schema_version": 1,
        "status": "within_quality_tier" if within else "best_effort",
        "parameter_key": str(row["parameter_key"]),
        "raw_free_parameters": {name: float(row[name]) for name in parameter_names},
        "selection": {
            "result_rank": int(row["result_rank"]),
            "mechanical_max_error_percent": float(row["mechanical_max_error_percent"]),
            "mechanical_rmse_percent": float(row["mechanical_rmse_percent"]),
            "quality_tier_percent": float(row["quality_tier_percent"]),
            "within_quality_tier": within,
            "best_effort": not within,
            "selection_rule": (
                "structure -> Born -> R2 gates; then M_infinity -> RMSE -> "
                "same-element contrast -> structural margin"
            ),
        },
    }


def run_elasticity_batch(
    *,
    config_path: str | os.PathLike[str],
    parameters_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    protocol: str,
    resources: NestedResourcePlan,
    top_n: int = 20,
    near_optimal_window_percent: float = 5.0,
    diversity_slots: int = 2,
    minimum: int = 1,
    evaluate_structural_failures: bool = False,
    backend_factory: BackendFactory = default_backend_factory,
) -> pd.DataFrame:
    """Run or resume one static or dynamic candidate batch."""

    canonical_protocol = PROTOCOL_ALIASES.get(protocol)
    if canonical_protocol is None:
        raise ValueError("protocol must be static or dynamic")
    config_source = Path(config_path).resolve()
    parameter_source = Path(parameters_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    config = load_config(config_source)
    module = _protocol_module(config, canonical_protocol)
    parameter_space = build_parameter_space(config)
    candidates = load_candidates(parameter_source, parameter_space)
    scientific = _stage_scientific_config(
        config,
        protocol=canonical_protocol,
        top_n=top_n,
        minimum=minimum,
        near_optimal_window_percent=near_optimal_window_percent,
        diversity_slots=diversity_slots,
        evaluate_structural_failures=evaluate_structural_failures,
    )
    _write_immutable_json(
        destination / "batch_identity.json",
        _batch_identity(
            config_path=config_source,
            parameters_path=parameter_source,
            scientific_config=scientific,
        ),
    )

    selection = config["elasticity"]["selection"]
    minimum_r2 = float(selection["fit_quality"]["minimum_r2"])
    born_required = bool(selection["born_stability"]["required"])
    quality_tier = float(config["elasticity"]["reporting"]["mechanical_tier_percent"])
    structural_by_key = {
        candidate.parameter_key: assess_structural_gates(candidate.source_row, config)
        for candidate in candidates
    }

    # Dynamic promotion consumes the ranked static table and only evaluates
    # selected hard-gate finalists. Static screening evaluates every measured
    # structural pass, retaining skipped rows for provenance.
    selected_input = pd.DataFrame()
    if canonical_protocol == DYNAMIC_PROTOCOL:
        input_frame = pd.DataFrame([candidate.source_row for candidate in candidates])
        selected_input = select_finalists(
            input_frame,
            parameter_space=parameter_space,
            top_n=top_n,
            near_optimal_window_percent=near_optimal_window_percent,
            diversity_slots=diversity_slots,
            minimum=minimum,
        )
        selected_keys = set(selected_input["parameter_key"].astype(str))
        candidates = [item for item in candidates if item.parameter_key in selected_keys]

    parameter_runner = LAMMPSRunner(config, start_scheduler_pool=False)
    strains = _signed_strains(module)
    prepared: dict[str, PreparedCandidate] = {}
    initial_rows: list[dict[str, Any]] = []
    for candidate in candidates:
        structural = structural_by_key[candidate.parameter_key]
        if (
            canonical_protocol == STATIC_PROTOCOL
            and not evaluate_structural_failures
            and not structural["structural_gate_pass"]
        ):
            row = _failed_row(candidate, structural, "skipped: structural hard gate failed")
            row["calculation_status"] = "skipped_structural_gate"
            row["finalist_rejection_reason"] = "structural_gate"
            initial_rows.append(row)
            continue
        try:
            prepared[candidate.parameter_key] = prepare_candidate(
                candidate,
                output_dir=destination,
                parameter_runner=parameter_runner,
                config=config,
            )
        except Exception as exc:
            initial_rows.append(_failed_row(candidate, structural, str(exc)))

    seed_values = (
        [int(value) for value in module["protocol"]["seeds"]]
        if canonical_protocol == DYNAMIC_PROTOCOL
        else [None]
    )
    work_units = [
        (prepared_item, seed)
        for prepared_item in prepared.values()
        for seed in seed_values
    ]
    if work_units:
        # The machine profile declares a maximum outer-worker capacity.  A
        # small batch should not strand the remaining allocation: reduce the
        # actual candidate/seed workers to the work-unit count and use the
        # freed CPUs for independent strain states of each active candidate.
        resources = plan_nested_resources(
            available_cores=resources.available_cores,
            cores_per_state=resources.cores_per_state,
            candidate_workers=min(resources.candidate_workers, len(work_units)),
            omp_threads_per_rank=resources.omp_threads_per_rank,
        )

    def run_unit(
        prepared_item: PreparedCandidate,
        seed: int | None,
    ) -> tuple[str, int | None, dict[str, Any], Path]:
        runner = _candidate_runner(
            prepared=prepared_item,
            output_dir=destination,
            protocol=canonical_protocol,
            trajectory_seed=seed,
            config=config,
            module=module,
            resources=resources,
            backend_factory=backend_factory,
        )
        summary = (
            runner.run_static(strains)
            if canonical_protocol == STATIC_PROTOCOL
            else runner.run_dynamic(strains)
        )
        summary = dict(summary)
        summary["candidate_directory"] = str(runner.candidate_dir)
        row = summarize_single_elastic_result(
            candidate=prepared_item.candidate,
            prepared=prepared_item,
            structural=structural_by_key[prepared_item.candidate.parameter_key],
            summary=summary,
            module=module,
            minimum_r2=minimum_r2,
            born_required=born_required,
            quality_tier_percent=quality_tier,
        )
        row["trajectory_seed"] = seed
        return (
            prepared_item.candidate.parameter_key,
            seed,
            row,
            runner.candidate_manifest_path,
        )

    completed: list[tuple[str, int | None, dict[str, Any], Path]] = []
    failed_units: list[dict[str, Any]] = []
    if work_units:
        with ThreadPoolExecutor(max_workers=resources.candidate_workers) as executor:
            futures = {
                executor.submit(run_unit, item, seed): (item, seed)
                for item, seed in work_units
            }
            for future in as_completed(futures):
                item, seed = futures[future]
                try:
                    completed.append(future.result())
                except Exception as exc:
                    failed = _failed_row(
                        item.candidate,
                        structural_by_key[item.candidate.parameter_key],
                        str(exc),
                    )
                    failed["trajectory_seed"] = seed
                    failed_units.append(failed)

    manifest_paths = [item[3] for item in completed]
    if canonical_protocol == STATIC_PROTOCOL:
        result_rows = [*initial_rows, *(item[2] for item in completed), *failed_units]
        seed_frame = pd.DataFrame()
    else:
        seed_rows = [item[2] for item in completed] + failed_units
        seed_frame = pd.DataFrame(seed_rows)
        result_rows = list(initial_rows)
        by_key: dict[str, list[dict[str, Any]]] = {}
        for row in seed_rows:
            by_key.setdefault(str(row["parameter_key"]), []).append(row)
        for candidate in candidates:
            prepared_item = prepared.get(candidate.parameter_key)
            if prepared_item is None:
                continue
            result_rows.append(
                aggregate_dynamic_seed_results(
                    by_key.get(candidate.parameter_key, []),
                    candidate=candidate,
                    prepared=prepared_item,
                    structural=structural_by_key[candidate.parameter_key],
                    module=module,
                    expected_seeds=[int(value) for value in seed_values],
                    minimum_r2=minimum_r2,
                    born_required=born_required,
                    quality_tier_percent=quality_tier,
                )
            )

    ranked = rank_elastic_results(pd.DataFrame(result_rows))
    finalists = select_finalists(
        ranked,
        parameter_space=parameter_space,
        top_n=top_n,
        near_optimal_window_percent=near_optimal_window_percent,
        diversity_slots=diversity_slots,
        minimum=minimum,
    )
    outputs = _output_paths(destination, canonical_protocol)
    _write_frame(outputs["results"], ranked)
    _write_frame(outputs["finalists"], finalists)
    if canonical_protocol == DYNAMIC_PROTOCOL:
        _write_frame(outputs["seed_results"], seed_frame)
    best = _best_candidate_document(
        ranked,
        [name for name, _lower, _upper in parameter_space],
    )
    _write_json(outputs["best_candidate"], best)
    preparation_failures = sum(
        str(row.get("calculation_status")) == "failed" for row in initial_rows
    )
    incomplete_failures = len(failed_units) + preparation_failures
    summary_document = {
        "schema_version": 1,
        "protocol": canonical_protocol,
        "input_candidates": len(candidates),
        "completed_work_units": len(completed),
        "failed_work_units": incomplete_failures,
        "hard_gate_eligible": int(
            ranked["finalist_eligible"].sum() if "finalist_eligible" in ranked else 0
        ),
        "within_quality_tier": int(
            ranked["within_quality_tier"].sum() if "within_quality_tier" in ranked else 0
        ),
        "selected_finalists": len(finalists),
        "scientific_outcome": (
            "incomplete_failures" if incomplete_failures else best["status"]
        ),
        "resources": resources.to_dict(),
    }
    _write_json(outputs["batch_summary"], summary_document)

    stage_inputs = _stage_inputs(
        output_dir=destination,
        config_path=config_source,
        parameters_path=parameter_source,
        manifest_paths=manifest_paths,
    )
    manifest = build_artifact_manifest(
        kind="stage",
        identifier=f"cubic_elastic_batch:{canonical_protocol}",
        parameters=None,
        seeds=[] if canonical_protocol == STATIC_PROTOCOL else seed_values,
        scientific_config=scientific,
        input_artifacts=stage_inputs,
        expected_outputs=outputs,
    )
    stage_manifest = destination / "stage_manifest.json"
    # A scientific zero-eligible outcome is complete. Execution failures are
    # not: leave the stage unmanifested so a later invocation can retry only
    # the missing parameter-key/seed work units.
    if incomplete_failures:
        if stage_manifest.exists():
            raise BatchArtifactError(
                "a previously completed batch now contains an invalid or "
                "unreusable immutable candidate artifact"
            )
        return ranked
    if stage_manifest.exists():
        decision = check_artifact_reuse(
            stage_manifest,
            kind="stage",
            identifier=f"cubic_elastic_batch:{canonical_protocol}",
            parameters=None,
            seeds=[] if canonical_protocol == STATIC_PROTOCOL else seed_values,
            scientific_config=scientific,
            input_artifacts=stage_inputs,
            expected_outputs=outputs,
        )
        if not decision.reusable:
            raise BatchArtifactError(
                f"completed batch is not safely reusable: {decision.reason}"
            )
    else:
        write_artifact_manifest(stage_manifest, manifest)
    return ranked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a resumable cubic-elastic candidate/finalist batch"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--parameters", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--protocol", required=True, choices=("static", "dynamic"))
    parser.add_argument("--available-cores", required=True, type=int)
    parser.add_argument("--cores-per-state", required=True, type=int)
    parser.add_argument("--candidate-workers", type=int, default=1)
    parser.add_argument(
        "--state-workers",
        type=int,
        help="Inner state concurrency; omitted uses the largest non-oversubscribing value",
    )
    parser.add_argument("--omp-threads-per-state", type=int, default=1)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--minimum", type=int, default=1)
    parser.add_argument("--near-optimal-window-percent", type=float, default=5.0)
    parser.add_argument("--diversity-slots", type=int, default=2)
    parser.add_argument(
        "--evaluate-structural-failures",
        action="store_true",
        help=(
            "Still calculate static elasticity for structurally rejected rows; "
            "intended for complete final validation, not candidate screening"
        ),
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    resources = plan_nested_resources(
        available_cores=effective_available_cores(arguments.available_cores),
        cores_per_state=arguments.cores_per_state,
        candidate_workers=arguments.candidate_workers,
        state_workers=arguments.state_workers,
        omp_threads_per_rank=arguments.omp_threads_per_state,
    )
    ranked = run_elasticity_batch(
        config_path=arguments.config,
        parameters_path=arguments.parameters,
        output_dir=arguments.output_dir,
        protocol=arguments.protocol,
        resources=resources,
        top_n=arguments.top_n,
        minimum=arguments.minimum,
        near_optimal_window_percent=arguments.near_optimal_window_percent,
        diversity_slots=arguments.diversity_slots,
        evaluate_structural_failures=arguments.evaluate_structural_failures,
    )
    eligible = int(ranked["finalist_eligible"].sum()) if "finalist_eligible" in ranked else 0
    print(
        json.dumps(
            {
                "status": "completed",
                "protocol": arguments.protocol,
                "rows": len(ranked),
                "hard_gate_eligible": eligible,
                "output_dir": str(arguments.output_dir.resolve()),
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "BatchArtifactError",
    "Candidate",
    "CandidateParameterError",
    "CubicElasticBatchError",
    "NestedResourcePlan",
    "PreparedCandidate",
    "aggregate_dynamic_seed_results",
    "assess_structural_gates",
    "build_parser",
    "default_backend_factory",
    "effective_available_cores",
    "load_candidates",
    "main",
    "normalize_ranking_contract",
    "plan_nested_resources",
    "prepare_candidate",
    "rank_elastic_results",
    "run_elasticity_batch",
    "same_element_parameter_contrast",
    "select_finalists",
    "summarize_single_elastic_result",
]


if __name__ == "__main__":
    raise SystemExit(main())
