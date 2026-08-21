"""One resumable, exact material active-learning round.

This module is the execution bridge between constrained GP acquisition and the
exact LAMMPS/property runners.  A call completes exactly one round:

1. load an explicitly named cumulative evidence set;
2. generate a deterministic, hard-domain candidate pool;
3. acquire coordinates with :class:`GaussianProcessStructuralAcquisition`;
4. evaluate structural properties exactly, keyed and manifested per candidate;
5. promote exact structural passes to the static cubic-elastic batch;
6. merge the new labels and publish the generic constrained-refinement state.

No directory scan, rank-based identity, or modification-time selection is used.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Protocol

import numpy as np
import pandas as pd

from .config_loader import load_config
from .constrained_refinement import (
    AcquisitionContext,
    AcquisitionResult,
    RefinementSpec,
    run_refinement_round,
)
from .cubic_elastic_batch import (
    BackendFactory,
    NestedResourcePlan,
    assess_structural_gates,
    default_backend_factory,
    effective_available_cores,
    plan_nested_resources,
    run_elasticity_batch,
)
from .gp_structural_acquisition import GaussianProcessStructuralAcquisition
from .lammps_interface import EvalResult, LAMMPSRunner
from .parameter_space import build_parameter_space
from workflow.artifact_manifest import (
    build_artifact_manifest,
    canonical_parameter_key,
    check_artifact_reuse,
    sha256_file,
    write_artifact_manifest,
)


_EVIDENCE_SCHEMA = "ffopt-material-al-round-evidence-v1"
_STAGE_IDENTIFIER = "material_al_round"
_CORE_OUTPUTS = {
    "state": "refinement_state.json",
    "assessed": "assessed_candidates.csv",
    "ranking": "exact_structural_mechanical_ranking.csv",
    "structural_proposals": "structural_proposals.csv",
    "mechanical_proposals": "mechanical_proposals.csv",
    "report": "refinement_report.md",
    "disposition": "stage_disposition.json",
    "core_manifest": "artifact_manifest.json",
}
_ROUND_OUTPUTS = {
    **_CORE_OUTPUTS,
    "round_evidence": "round_evidence.json",
    "cumulative_structural": "cumulative_structural_observations.csv",
    "cumulative_static": "cumulative_static_observations.csv",
    "candidate_pool": "candidate_pool.csv",
    "planned_structural": "planned_structural_proposals.csv",
    "new_structural": "new_structural_observations.csv",
    "executed_static": "executed_static_proposals.csv",
    "new_static": "new_static_observations.csv",
    "static_completion": "static_batch_completion.json",
}


class MaterialALRoundError(RuntimeError):
    """Raised when a material AL round cannot be resumed safely."""


class StructuralRunner(Protocol):
    def feasibility_error(self, raw_params: dict[str, float]) -> str | None: ...

    def evaluate_replicates(
        self,
        params: dict[str, float],
        work_dir: str,
        seeds: list[int],
        save_traj: bool = False,
        reuse_existing: bool = False,
    ) -> list[EvalResult]: ...

    # Implementations may expose additional immutable files which affect an
    # exact structural evaluation.  The concrete LAMMPS runner is also
    # inspected for its standard input/data attributes, so this hook is mainly
    # for alternative backends and tests.
    def structural_input_artifacts(
        self,
    ) -> Mapping[str, str | os.PathLike[str]]: ...


StructuralRunnerFactory = Callable[[dict[str, Any]], StructuralRunner]
ElasticityBatchRunner = Callable[..., pd.DataFrame]


@dataclass(frozen=True)
class MaterialALRoundResult:
    output_dir: Path
    state: Mapping[str, Any]
    reused: bool


class _ReplayAcquisition:
    """Replay one GP decision while retaining the GP campaign identity."""

    def __init__(
        self,
        backend: GaussianProcessStructuralAcquisition,
        result: AcquisitionResult,
    ) -> None:
        self._backend = backend
        self._result = result
        self.name = backend.name
        self.capability = backend.capability
        self.scientific_convergence_capable = backend.scientific_convergence_capable

    def scientific_identity(self) -> Mapping[str, Any]:
        return self._backend.scientific_identity()

    def propose(self, _context: AcquisitionContext) -> AcquisitionResult:
        return self._result


def _default_structural_runner_factory(config: dict[str, Any]) -> StructuralRunner:
    return LAMMPSRunner(config)


def _release_structural_scheduler_pool(runner: StructuralRunner) -> None:
    """Release persistent structural workers before nested elastic srun steps."""

    scheduler_pool = getattr(runner, "scheduler_pool", None)
    if scheduler_pool is None:
        return
    try:
        close = getattr(scheduler_pool, "close", None)
        if callable(close):
            close()
    finally:
        # LAMMPSRunner would otherwise keep dispatching through a pool whose
        # worker step has ended, and the worker step would retain every task in
        # the constrained-AL allocation while the elastic batch tries to start.
        setattr(runner, "scheduler_pool", None)


def _json_bytes(value: Any) -> bytes:
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


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _json_bytes(value)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_json(path: Path, value: Any) -> None:
    payload = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise MaterialALRoundError(f"immutable JSON changed: {path}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


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


def _copy_or_verify(source: Path, destination: Path) -> None:
    content = source.read_bytes()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if destination.read_bytes() != content:
            raise MaterialALRoundError(f"immutable round output changed: {destination}")
        return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)


def _parameter_values(row: Mapping[str, Any], names: Sequence[str]) -> dict[str, float]:
    result: dict[str, float] = {}
    for name in names:
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise MaterialALRoundError(
                f"evidence row is missing finite parameter {name!r}"
            ) from exc
        if not math.isfinite(value):
            raise MaterialALRoundError(f"parameter {name!r} must be finite")
        result[name] = value
    return result


def _optional_gate_allows(row: Mapping[str, Any], name: str) -> bool:
    """Treat an absent audit field as legacy evidence, but fail a recorded audit."""

    value = row.get(name)
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return True
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, np.integer)):
        return bool(value)
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _with_parameter_keys(frame: pd.DataFrame, names: Sequence[str]) -> pd.DataFrame:
    missing = [name for name in names if name not in frame]
    if missing:
        raise MaterialALRoundError(f"evidence is missing parameters: {missing}")
    result = frame.copy()
    keys = [
        canonical_parameter_key(_parameter_values(row, names))
        for row in result.to_dict(orient="records")
    ]
    if "parameter_key" in result:
        recorded = list(result["parameter_key"].astype(str))
        if recorded != keys:
            raise MaterialALRoundError(
                "recorded parameter_key does not match exact free parameters"
            )
    result["parameter_key"] = keys
    return result


def _read_frames(paths: Sequence[Path], names: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_order, path in enumerate(paths):
        source = path.resolve()
        if not source.is_file():
            raise MaterialALRoundError(f"explicit evidence does not exist: {source}")
        frame = pd.read_csv(source, float_precision="round_trip")
        if frame.empty:
            continue
        frame = _with_parameter_keys(frame, names)
        frame["_evidence_source_order"] = source_order
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=[*names, "parameter_key"])
    combined = pd.concat(frames, ignore_index=True, sort=False)
    # Later paths are newer exact evidence.  A single path keeps its first
    # canonical row, which makes a repeated invocation deterministic.
    combined = combined.sort_values(
        "_evidence_source_order", kind="stable"
    ).drop_duplicates("parameter_key", keep="last")
    return combined.drop(columns="_evidence_source_order").reset_index(drop=True)


def _relative_record(path: Path, base: Path, *, rows: int | None = None) -> dict[str, Any]:
    digest = sha256_file(path)
    record: dict[str, Any] = {
        "path": os.path.relpath(path, base).replace("\\", "/"),
        "sha256": digest.sha256,
        "size_bytes": digest.size_bytes,
    }
    if rows is not None:
        record["rows"] = int(rows)
    return record


def _resolve_record(record: Mapping[str, Any], base: Path, *, label: str) -> Path:
    raw = record.get("path")
    if not isinstance(raw, str) or not raw:
        raise MaterialALRoundError(f"{label} path is missing")
    path = (base / raw).resolve()
    digest = sha256_file(path)
    if digest.sha256 != record.get("sha256") or digest.size_bytes != record.get(
        "size_bytes"
    ):
        raise MaterialALRoundError(f"{label} hash does not match: {path}")
    return path


def _load_previous_evidence(path: Path) -> tuple[list[Path], list[Path], dict[str, Any]]:
    source = path.resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise MaterialALRoundError(f"cannot read previous evidence {source}: {exc}") from exc
    if not isinstance(document, dict) or document.get("schema") != _EVIDENCE_SCHEMA:
        raise MaterialALRoundError(f"unsupported previous evidence schema: {source}")
    base = source.parent
    structural = _resolve_record(
        document["cumulative_structural"], base, label="cumulative_structural"
    )
    static = _resolve_record(
        document["cumulative_static"], base, label="cumulative_static"
    )
    for index, record in enumerate(document.get("candidate_manifests", [])):
        _resolve_record(record, base, label=f"candidate_manifest[{index}]")
    static_manifest = document.get("static_batch_manifest")
    if isinstance(static_manifest, Mapping):
        _resolve_record(static_manifest, base, label="static_batch_manifest")
    return [structural], [static], document


def _candidate_pool(
    *,
    structural: pd.DataFrame,
    mechanical: pd.DataFrame,
    external_paths: Sequence[Path],
    parameter_space: Sequence[tuple[str, float, float]],
    runner: StructuralRunner,
    pool_size: int,
    global_fraction: float,
    local_radii: Sequence[float],
    seed: int,
    round_number: int,
) -> pd.DataFrame:
    if pool_size < 1:
        raise MaterialALRoundError("candidate_pool size must be positive")
    names = [name for name, _lower, _upper in parameter_space]
    lower = np.asarray([low for _name, low, _upper in parameter_space], dtype=float)
    upper = np.asarray([high for _name, _low, high in parameter_space], dtype=float)
    span = upper - lower
    if np.any(span <= 0.0):
        raise MaterialALRoundError("free-parameter bounds must have positive span")
    observed = set(structural["parameter_key"].astype(str))
    rows: list[dict[str, Any]] = []
    seen = set(observed)

    def accept(values: np.ndarray, source: str) -> None:
        if len(rows) >= pool_size:
            return
        params = {name: float(value) for name, value in zip(names, values)}
        key = canonical_parameter_key(params)
        if key in seen or runner.feasibility_error(params) is not None:
            return
        seen.add(key)
        rows.append({
            "parameter_key": key,
            **params,
            "pool_source": source,
            "evidence_kind": "unobserved_coordinate",
        })

    for source_path in external_paths:
        source = source_path.resolve()
        if not source.is_file():
            raise MaterialALRoundError(f"explicit candidate pool does not exist: {source}")
        frame = pd.read_csv(source, float_precision="round_trip")
        if frame.empty:
            continue
        for row in frame.to_dict(orient="records"):
            accept(
                np.asarray([float(row[name]) for name in names]),
                f"explicit:{source}",
            )

    # Incumbents and exact structural boundary points anchor the local part.
    centres = structural.copy()
    if len(centres):
        if "structural_feasible" in centres:
            feasible = centres["structural_feasible"].astype(str).str.lower().isin(
                {"true", "1", "yes"}
            )
        elif "structural_gate_pass" in centres:
            feasible = centres["structural_gate_pass"].astype(str).str.lower().isin(
                {"true", "1", "yes"}
            )
        else:
            feasible = pd.Series(False, index=centres.index)
        centres["_feasible"] = feasible
        if "structural_minimum_margin" in centres:
            centres["_margin"] = pd.to_numeric(
                centres["structural_minimum_margin"], errors="coerce"
            )
        elif "structural_margin" in centres:
            centres["_margin"] = pd.to_numeric(
                centres["structural_margin"], errors="coerce"
            )
        else:
            centres["_margin"] = -math.inf
        score_by_key: dict[str, float] = {}
        if "mechanical_max_error_percent" in mechanical:
            score_by_key = dict(zip(
                mechanical["parameter_key"].astype(str),
                pd.to_numeric(
                    mechanical["mechanical_max_error_percent"], errors="coerce"
                ).fillna(math.inf),
            ))
        centres["_mechanical"] = centres["parameter_key"].map(
            lambda key: score_by_key.get(str(key), math.inf)
        )
        centres = centres.sort_values(
            ["_feasible", "_mechanical", "_margin", "parameter_key"],
            ascending=[False, True, False, True],
            kind="stable",
        ).head(16)
    centre_values = (
        centres[names].to_numpy(float)
        if len(centres)
        else np.empty((0, len(names)), dtype=float)
    )

    global_target = max(1, int(round(pool_size * global_fraction)))
    local_target = pool_size - global_target
    rng = np.random.default_rng(int(seed) + 1009 * int(round_number))
    try:
        from scipy.stats import qmc
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise MaterialALRoundError("candidate-pool generation requires scipy") from exc
    generated_before = len(rows)
    batch_power = max(6, int(math.ceil(math.log2(max(pool_size * 2, 64)))))
    sobol = qmc.Sobol(
        d=len(names), scramble=True, seed=int(seed) + 7919 * int(round_number)
    )
    for point in sobol.random_base2(batch_power):
        accept(lower + point * span, "global_sobol")
        if len(rows) - generated_before >= global_target:
            break

    local_added = 0
    attempts = 0
    radii = [float(value) for value in local_radii if float(value) > 0.0]
    if not radii:
        radii = [0.01, 0.025, 0.05]
    while local_added < local_target and len(rows) < pool_size and attempts < pool_size * 50:
        attempts += 1
        if len(centre_values):
            centre = centre_values[int(rng.integers(0, len(centre_values)))]
            radius = radii[int(rng.integers(0, len(radii)))]
            values = np.clip(
                centre + rng.uniform(-radius, radius, len(names)) * span,
                lower,
                upper,
            )
            before = len(rows)
            accept(values, "local_exact_anchor")
            local_added += int(len(rows) > before)
        else:
            accept(lower + rng.random(len(names)) * span, "global_no_anchor")

    # Hard-domain rejection can consume many local draws. Fill the declared
    # budget with an independent deterministic global stream, never relaxing a
    # parameter constraint.
    while len(rows) < pool_size and attempts < pool_size * 200:
        attempts += 1
        accept(lower + rng.random(len(names)) * span, "global_constraint_fill")
    if len(rows) < pool_size:
        raise MaterialALRoundError(
            f"only generated {len(rows)}/{pool_size} unique hard-feasible coordinates"
        )
    return pd.DataFrame(rows[:pool_size])


def _gp_backend(config: Mapping[str, Any]) -> GaussianProcessStructuralAcquisition:
    active = config.get("active_learning", {})
    if not isinstance(active, Mapping):
        active = {}
    return GaussianProcessStructuralAcquisition(
        improvement_fraction=float(active.get("improvement_fraction", 0.50)),
        boundary_fraction=float(active.get("boundary_fraction", 0.30)),
        global_fraction=float(active.get("global_fraction", 0.20)),
        trust_region_radius=float(active.get("trust_region_radius", 0.25)),
        minimum_structural_observations=int(
            active.get("minimum_structural_observations", 12)
        ),
        minimum_mechanical_observations=int(
            active.get("minimum_mechanical_observations", 10)
        ),
        minimum_holdout_r2=float(active.get("minimum_holdout_r2", 0.25)),
        maximum_dimensions=int(active.get("maximum_gp_dimensions", 8)),
        diversity_weight=float(active.get("diversity_weight", 0.20)),
        expected_improvement_exploration=float(
            active.get("expected_improvement_exploration", 0.01)
        ),
        maximum_local_anchors=int(active.get("maximum_local_anchors", 8)),
    )


def _structural_scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "schema": "ffopt-exact-structural-candidate-v1",
        "material": config.get("material", {}),
        "crystal": config.get("crystal", {}),
        "targets": config.get("targets", {}),
        "property_evaluators": config.get("property_evaluators", {}),
        "lammps": {
            key: value
            for key, value in config.get("lammps", {}).items()
            if key not in {"executable", "mpiexec", "mpi_flavor"}
        },
        "parameter_constraints": config.get("parameter_constraints", {}),
        "pair_params": config.get("pair_params", {}),
        "elasticity_selection": config.get("elasticity", {}).get(
            "selection", {}
        ),
    }


def _structural_runner_input_artifacts(
    runner: StructuralRunner,
) -> dict[str, Path]:
    """Return every external file that can change an exact structure label."""

    artifacts: dict[str, Path] = {}
    attributes = (
        "bulk_data",
        "bulk_input",
        "surf_input",
        "surf_complete",
        "surf_split",
        "ads_complex_input",
        "ads_slab_input",
        "ads_mol_input",
        "ads_complex",
        "ads_slab",
        "ads_mol",
        "sub_single_input",
        "sub_single_data",
    )
    for name in attributes:
        raw = getattr(runner, name, None)
        if raw is None:
            continue
        path = Path(raw).resolve()
        if not path.is_file():
            raise MaterialALRoundError(
                f"structural runner input artifact is missing: {path}"
            )
        artifacts[name] = path
    provider = getattr(runner, "structural_input_artifacts", None)
    if callable(provider):
        supplied = provider()
        if not isinstance(supplied, Mapping):
            raise MaterialALRoundError(
                "structural_input_artifacts() must return a mapping"
            )
        for raw_name, raw_path in supplied.items():
            name = str(raw_name)
            if not name:
                raise MaterialALRoundError(
                    "structural input artifact labels must be non-empty"
                )
            path = Path(raw_path).resolve()
            if not path.is_file():
                raise MaterialALRoundError(
                    f"structural runner input artifact is missing: {path}"
                )
            artifacts[f"extra_{name}"] = path
    return artifacts


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _evaluate_structural_candidate(
    *,
    proposal: Mapping[str, Any],
    names: Sequence[str],
    targets: Sequence[str],
    seeds: Sequence[int],
    runner: StructuralRunner,
    config_path: Path,
    runner_input_artifacts: Mapping[str, Path],
    config: Mapping[str, Any],
    scientific_config: Mapping[str, Any],
    root: Path,
) -> tuple[dict[str, Any], Path]:
    params = _parameter_values(proposal, names)
    key = canonical_parameter_key(params)
    digest = key.rsplit(":", 1)[-1]
    candidate_dir = root / digest
    input_path = candidate_dir / "candidate_input.json"
    result_json = candidate_dir / "structural_result.json"
    result_csv = candidate_dir / "structural_result.csv"
    manifest_path = candidate_dir / "candidate_manifest.json"
    candidate_input = {
        "schema": "ffopt-exact-structural-candidate-input-v1",
        "parameter_key": key,
        "parameters": params,
        "selection_role": str(proposal.get("selection_role", "")),
    }
    _write_immutable_json(input_path, candidate_input)
    inputs = {
        "config": config_path,
        "candidate_input": input_path,
        **{
            f"runner_{name}": path
            for name, path in sorted(runner_input_artifacts.items())
        },
    }
    outputs = {"result_json": result_json, "result_csv": result_csv}
    if manifest_path.exists():
        decision = check_artifact_reuse(
            manifest_path,
            kind="candidate",
            identifier="exact_structural_properties",
            parameters=params,
            seeds=[int(seed) for seed in seeds],
            scientific_config=scientific_config,
            input_artifacts=inputs,
            expected_outputs=outputs,
        )
        if not decision.reusable:
            raise MaterialALRoundError(
                f"structural candidate {key} is not safely reusable: {decision.reason}"
            )
        frame = pd.read_csv(
            result_csv,
            float_precision="round_trip",
            dtype={
                "parameter_key": str,
                "selection_role": str,
                "evidence_kind": str,
                "structural_seeds": str,
                "error_msg": str,
            },
        )
        if len(frame) != 1:
            raise MaterialALRoundError(f"candidate result must have one row: {result_csv}")
        return frame.iloc[0].to_dict(), manifest_path

    results = runner.evaluate_replicates(
        params,
        str(candidate_dir / "work"),
        [int(seed) for seed in seeds],
        save_traj=False,
        reuse_existing=True,
    )
    successful = [result for result in results if result.success]
    row: dict[str, Any] = {
        "parameter_key": key,
        **params,
        "candidate_id": proposal.get("candidate_id"),
        "selection_role": proposal.get("selection_role", ""),
        "evidence_kind": "exact_lammps_structural",
        "is_exact_observation": True,
        "surrogate_is_exact_observation": True,
        "success": len(successful) == len(seeds),
        "n_success": len(successful),
        "n_seeds": len(seeds),
        "structural_seeds": " ".join(str(seed) for seed in seeds),
        "error_msg": " | ".join(
            result.error_msg for result in results if not result.success
        ),
    }
    objectives = np.asarray(
        [float(result.objective) for result in successful], dtype=float
    )
    row["objective"] = float(np.mean(objectives)) if len(objectives) else math.nan
    row["objective_std"] = float(np.std(objectives)) if len(objectives) else math.nan
    for target in targets:
        values = np.asarray(
            [result.properties.get(target, math.nan) for result in successful],
            dtype=float,
        )
        row[f"calc_{target}"] = (
            float(np.nanmean(values)) if np.isfinite(values).any() else math.nan
        )
        row[f"std_{target}"] = (
            float(np.nanstd(values)) if np.isfinite(values).any() else math.nan
        )
    replicate_gate_results: list[dict[str, Any]] = []
    for result in results:
        if not result.success:
            replicate_gate_results.append({
                "structural_gate_pass": False,
                "structural_margin": -math.inf,
                "structural_gate_reason": result.error_msg or "evaluation_failed",
                "structural_errors": {},
            })
            continue
        replicate_row = {
            f"calc_{target}": result.properties.get(target, math.nan)
            for target in targets
        }
        replicate_gate_results.append(assess_structural_gates(replicate_row, config))
    replicate_all_pass = bool(
        len(replicate_gate_results) == len(seeds)
        and all(item["structural_gate_pass"] for item in replicate_gate_results)
    )
    mean_gate = assess_structural_gates(row, config)
    row["replicate_gate_all_pass"] = replicate_all_pass
    row["replicate_minimum_structural_margin"] = float(
        min(
            (float(item["structural_margin"]) for item in replicate_gate_results),
            default=-math.inf,
        )
    )
    row["audit_certified"] = bool(row["success"] and replicate_all_pass)
    row["structural_gate_pass"] = bool(
        mean_gate["structural_gate_pass"] and row["audit_certified"]
    )
    row["structural_margin"] = min(
        float(mean_gate["structural_margin"]),
        float(row["replicate_minimum_structural_margin"]),
    )
    result_document = {
        "schema": "ffopt-exact-structural-candidate-result-v1",
        "parameter_key": key,
        "parameters": params,
        "seeds": [int(seed) for seed in seeds],
        "summary": _json_safe(row),
        "replicates": [
            {
                "seed": int(seed),
                "success": bool(result.success),
                "objective": _json_safe(float(result.objective)),
                "properties": _json_safe(result.properties),
                "per_property_error": _json_safe(result.per_property_error),
                "error_msg": result.error_msg,
                "structural_gate": _json_safe(gate),
            }
            for seed, result, gate in zip(seeds, results, replicate_gate_results)
        ],
    }
    _write_json(result_json, result_document)
    _write_frame(result_csv, pd.DataFrame([row]))
    manifest = build_artifact_manifest(
        kind="candidate",
        identifier="exact_structural_properties",
        parameters=params,
        seeds=[int(seed) for seed in seeds],
        scientific_config=scientific_config,
        input_artifacts=inputs,
        expected_outputs=outputs,
    )
    write_artifact_manifest(manifest_path, manifest)
    return row, manifest_path


def _evaluate_structural_batch(
    *,
    proposals: pd.DataFrame,
    names: Sequence[str],
    targets: Sequence[str],
    seeds: Sequence[int],
    runner: StructuralRunner,
    config_path: Path,
    runner_input_artifacts: Mapping[str, Path],
    config: Mapping[str, Any],
    scientific_config: Mapping[str, Any],
    root: Path,
    workers: int,
) -> tuple[pd.DataFrame, list[Path]]:
    if proposals.empty:
        columns = [
            "parameter_key", *names, "candidate_id", "selection_role",
            "evidence_kind", "success", "n_success", "n_seeds",
            *(f"calc_{name}" for name in targets),
        ]
        return pd.DataFrame(columns=columns), []
    rows: list[dict[str, Any]] = []
    manifests: list[Path] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(proposals)))) as executor:
        futures = {
            executor.submit(
                _evaluate_structural_candidate,
                proposal=proposal,
                names=names,
                targets=targets,
                seeds=seeds,
                runner=runner,
                config_path=config_path,
                runner_input_artifacts=runner_input_artifacts,
                config=config,
                scientific_config=scientific_config,
                root=root,
            ): index
            for index, proposal in enumerate(proposals.to_dict(orient="records"))
        }
        ordered: dict[int, tuple[dict[str, Any], Path]] = {}
        for future in as_completed(futures):
            ordered[futures[future]] = future.result()
    for index in range(len(proposals)):
        row, manifest = ordered[index]
        rows.append(row)
        manifests.append(manifest)
    return pd.DataFrame(rows), manifests


def _normalised_coordinates(
    frame: pd.DataFrame,
    parameter_space: Sequence[tuple[str, float, float]],
) -> np.ndarray:
    names = [name for name, _lower, _upper in parameter_space]
    lower = np.asarray([low for _name, low, _upper in parameter_space])
    spans = np.asarray([
        max(high - low, 1.0e-15)
        for _name, low, high in parameter_space
    ])
    return (frame[names].to_numpy(float) - lower) / spans


def _select_static_promotions(
    *,
    structural: pd.DataFrame,
    mechanical: pd.DataFrame,
    new_keys: set[str],
    config: Mapping[str, Any],
    parameter_space: Sequence[tuple[str, float, float]],
    maximum: int,
    elite_fraction: float,
) -> pd.DataFrame:
    mechanical_keys = set(mechanical["parameter_key"].astype(str))
    candidates = structural.loc[
        ~structural["parameter_key"].astype(str).isin(mechanical_keys)
    ].copy()
    if candidates.empty or maximum <= 0:
        return candidates.head(0)
    gates = [
        assess_structural_gates(row, config)
        for row in candidates.to_dict(orient="records")
    ]
    candidates["structural_gate_pass"] = [
        bool(item["structural_gate_pass"])
        and _optional_gate_allows(row, "audit_certified")
        for item, row in zip(gates, candidates.to_dict(orient="records"))
    ]
    candidates["structural_margin"] = [float(item["structural_margin"]) for item in gates]
    candidates = candidates.loc[candidates["structural_gate_pass"]].reset_index(drop=True)
    if candidates.empty:
        return candidates
    candidates["promotion_origin"] = np.where(
        candidates["parameter_key"].astype(str).isin(new_keys),
        "current_round_exact_pass",
        "prior_exact_pass_backlog",
    )
    count = min(maximum, len(candidates))
    objective = pd.to_numeric(
        candidates.get("objective", pd.Series(math.inf, index=candidates.index)),
        errors="coerce",
    ).fillna(math.inf)
    elite_count = min(count, max(1, int(math.ceil(count * elite_fraction))))
    elite = sorted(
        range(len(candidates)),
        key=lambda index: (
            0 if candidates.iloc[index]["promotion_origin"] == "current_round_exact_pass" else 1,
            float(objective.iloc[index]),
            str(candidates.iloc[index]["parameter_key"]),
        ),
    )[:elite_count]
    selected = list(elite)
    roles = {index: "structural_objective_elite" for index in elite}
    coordinates = _normalised_coordinates(candidates, parameter_space)
    while len(selected) < count:
        remaining = [index for index in range(len(candidates)) if index not in selected]
        distances = np.min(
            np.linalg.norm(
                coordinates[remaining, None, :] - coordinates[selected][None, :, :],
                axis=2,
            ),
            axis=1,
        )
        position = max(
            range(len(remaining)),
            key=lambda item: (
                float(distances[item]),
                str(candidates.iloc[remaining[item]]["parameter_key"]),
            ),
        )
        winner = remaining[position]
        selected.append(winner)
        roles[winner] = "structural_feasible_diverse"
    result = candidates.iloc[selected].copy().reset_index(drop=True)
    result.insert(0, "static_promotion_rank", np.arange(1, len(result) + 1))
    result.insert(1, "static_promotion_role", [roles[index] for index in selected])
    return result


def _merge_evidence(
    previous: pd.DataFrame,
    current: pd.DataFrame,
    names: Sequence[str],
) -> pd.DataFrame:
    if current.empty:
        return previous.copy().reset_index(drop=True)
    merged = pd.concat([previous, current], ignore_index=True, sort=False)
    merged = _with_parameter_keys(merged, names)
    return merged.drop_duplicates("parameter_key", keep="last").reset_index(drop=True)


def _round_input_artifacts(
    *,
    runtime_config: Path,
    refinement_config: Path,
    structural_paths: Sequence[Path],
    mechanical_paths: Sequence[Path],
    candidate_pool_paths: Sequence[Path],
    previous_state: Path | None,
    previous_evidence: Path | None,
) -> dict[str, Path]:
    inputs: dict[str, Path] = {
        "runtime_config": runtime_config,
        "refinement_config": refinement_config,
    }
    inputs.update({
        f"structural_{index:03d}": path
        for index, path in enumerate(structural_paths)
    })
    inputs.update({
        f"static_{index:03d}": path
        for index, path in enumerate(mechanical_paths)
    })
    inputs.update({
        f"candidate_pool_{index:03d}": path
        for index, path in enumerate(candidate_pool_paths)
    })
    if previous_state is not None:
        inputs["previous_state"] = previous_state
    if previous_evidence is not None:
        inputs["previous_evidence"] = previous_evidence
    return inputs


def _round_scientific_identity(
    *,
    round_number: int,
    maximum_rounds: int,
    seed: int,
    structural_seeds: Sequence[int],
    pool_size: int,
    available_cores: int,
    cores_per_state: int,
    omp_threads_per_rank: int,
) -> dict[str, Any]:
    # Execution parallelism affects provenance/resume but not the scientific
    # result. It is included so an interrupted stage cannot silently change its
    # nested resource plan halfway through the same output directory.
    return {
        "schema": "ffopt-material-al-round-v1",
        "round": int(round_number),
        "maximum_rounds": int(maximum_rounds),
        "selection_seed": int(seed),
        "structural_seeds": [int(value) for value in structural_seeds],
        "candidate_pool_size": int(pool_size),
        "available_cores": int(available_cores),
        "cores_per_state": int(cores_per_state),
        "omp_threads_per_rank": int(omp_threads_per_rank),
        "acquisition_backend": "gp_structural_constraints",
    }


def run_material_al_round(
    *,
    runtime_config_path: str | os.PathLike[str],
    refinement_config_path: str | os.PathLike[str],
    structural_paths: Sequence[str | os.PathLike[str]],
    mechanical_paths: Sequence[str | os.PathLike[str]],
    candidate_pool_paths: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    round_number: int,
    maximum_rounds: int,
    previous_state_path: str | os.PathLike[str] | None = None,
    previous_evidence_path: str | os.PathLike[str] | None = None,
    seed: int = 20260820,
    structural_seeds: Sequence[int] = (101,),
    available_cores: int = 1,
    cores_per_state: int = 2,
    omp_threads_per_rank: int = 1,
    candidate_workers: int | None = None,
    state_workers: int | None = None,
    structural_runner_factory: StructuralRunnerFactory = _default_structural_runner_factory,
    elasticity_batch_runner: ElasticityBatchRunner = run_elasticity_batch,
    elasticity_backend_factory: BackendFactory = default_backend_factory,
    acquisition_backend: GaussianProcessStructuralAcquisition | None = None,
) -> MaterialALRoundResult:
    """Execute or resume one exact, evidence-producing material AL round."""

    runtime_source = Path(runtime_config_path).resolve()
    refinement_source = Path(refinement_config_path).resolve()
    destination = Path(output_dir).resolve()
    previous_state = (
        Path(previous_state_path).resolve() if previous_state_path is not None else None
    )
    previous_evidence = (
        Path(previous_evidence_path).resolve()
        if previous_evidence_path is not None
        else None
    )
    if round_number < 1 or maximum_rounds < round_number:
        raise MaterialALRoundError("round must lie in [1, maximum_rounds]")
    if round_number == 1 and (previous_state is not None or previous_evidence is not None):
        raise MaterialALRoundError("round 1 cannot consume previous state/evidence")
    if round_number > 1 and (previous_state is None or previous_evidence is None):
        raise MaterialALRoundError(
            "rounds after 1 require explicit previous-state and previous-evidence"
        )
    seeds = tuple(int(value) for value in structural_seeds)
    if not seeds or any(value < 1 for value in seeds):
        raise MaterialALRoundError("structural_seeds must contain positive integers")
    available = effective_available_cores(int(available_cores))

    config = load_config(runtime_source)
    refinement_document = load_config(refinement_source)
    raw_spec = refinement_document.get("constrained_refinement", refinement_document)
    if not isinstance(raw_spec, Mapping):
        raise MaterialALRoundError("refinement config must contain a mapping")
    spec = RefinementSpec.from_mapping(raw_spec)
    if spec.maximum_rounds != maximum_rounds:
        spec = replace(spec, maximum_rounds=maximum_rounds)
    parameter_space = build_parameter_space(config)
    names = tuple(name for name, _lower, _upper in parameter_space)

    direct_structural = [Path(path).resolve() for path in structural_paths]
    direct_mechanical = [Path(path).resolve() for path in mechanical_paths]
    previous_document: dict[str, Any] | None = None
    if previous_evidence is not None:
        carried_structural, carried_mechanical, previous_document = (
            _load_previous_evidence(previous_evidence)
        )
        evidence_structural_paths = [*carried_structural, *direct_structural]
        evidence_mechanical_paths = [*carried_mechanical, *direct_mechanical]
    else:
        evidence_structural_paths = direct_structural
        evidence_mechanical_paths = direct_mechanical
    if not evidence_structural_paths:
        raise MaterialALRoundError("at least one explicit structural evidence path is required")
    candidate_sources = [Path(path).resolve() for path in candidate_pool_paths]
    runner = structural_runner_factory(config)
    runner_input_artifacts = _structural_runner_input_artifacts(runner)
    input_artifacts = _round_input_artifacts(
        runtime_config=runtime_source,
        refinement_config=refinement_source,
        structural_paths=evidence_structural_paths,
        mechanical_paths=evidence_mechanical_paths,
        candidate_pool_paths=candidate_sources,
        previous_state=previous_state,
        previous_evidence=previous_evidence,
    )
    input_artifacts.update({
        f"structural_runner_{name}": path
        for name, path in sorted(runner_input_artifacts.items())
    })
    active = config.get("active_learning", {})
    if not isinstance(active, Mapping):
        active = {}
    pool_size = int(active.get("n_candidate_pool", 16384))
    scientific_identity = _round_scientific_identity(
        round_number=round_number,
        maximum_rounds=maximum_rounds,
        seed=seed,
        structural_seeds=seeds,
        pool_size=pool_size,
        available_cores=available,
        cores_per_state=cores_per_state,
        omp_threads_per_rank=omp_threads_per_rank,
    )
    fixed_outputs = {
        label: destination / name for label, name in _ROUND_OUTPUTS.items()
    }
    stage_manifest = destination / "material_al_manifest.json"
    if stage_manifest.exists():
        try:
            decision = check_artifact_reuse(
                stage_manifest,
                kind="stage",
                identifier=f"{_STAGE_IDENTIFIER}:{round_number:04d}",
                parameters=None,
                seeds=seeds,
                scientific_config=scientific_identity,
                input_artifacts=input_artifacts,
                expected_outputs=fixed_outputs,
            )
            if not decision.reusable:
                raise MaterialALRoundError(
                    f"completed material AL round is not reusable: {decision.reason}"
                )
            _load_previous_evidence(fixed_outputs["round_evidence"])
            state = json.loads(fixed_outputs["state"].read_text(encoding="utf-8"))
            return MaterialALRoundResult(destination, state, True)
        finally:
            _release_structural_scheduler_pool(runner)

    destination.mkdir(parents=True, exist_ok=True)
    structural = _read_frames(evidence_structural_paths, names)
    mechanical = _read_frames(evidence_mechanical_paths, names)
    pre_structural_rows = len(structural)
    pre_mechanical_rows = len(mechanical)
    # Add exact gate annotations before choosing local pool anchors. These are
    # measured decisions, never GP predictions.
    if len(structural):
        structural_records = structural.to_dict("records")
        gate = [assess_structural_gates(row, config) for row in structural_records]
        structural["structural_gate_pass"] = [
            bool(item["structural_gate_pass"])
            and _optional_gate_allows(row, "audit_certified")
            for item, row in zip(gate, structural_records)
        ]
        structural["structural_margin"] = [
            min(
                float(item["structural_margin"]),
                float(row.get("replicate_minimum_structural_margin", math.inf))
                if _optional_gate_allows(row, "audit_certified")
                else -math.inf,
            )
            for item, row in zip(gate, structural_records)
        ]
    pool = _candidate_pool(
        structural=structural,
        mechanical=mechanical,
        external_paths=candidate_sources,
        parameter_space=parameter_space,
        runner=runner,
        pool_size=pool_size,
        global_fraction=float(active.get("global_fraction", 0.20)),
        local_radii=active.get("local_radii", [0.01, 0.025, 0.05]),
        seed=seed,
        round_number=round_number,
    )
    _write_frame(fixed_outputs["candidate_pool"], pool)

    backend = acquisition_backend or _gp_backend(config)
    if not isinstance(backend, GaussianProcessStructuralAcquisition):
        raise MaterialALRoundError(
            "material AL requires GaussianProcessStructuralAcquisition"
        )
    acquisition = backend.propose(AcquisitionContext(
        candidate_pool=pool,
        structural_observations=structural,
        mechanical_observations=mechanical,
        parameter_names=spec.parameter_names,
        round_number=round_number,
        proposal_count=spec.structural_proposals_per_round,
        seed=int(seed),
        structural_constraints=spec.structural_constraints,
        mechanical_objectives=spec.mechanical_objectives,
        refinement_spec=spec,
    ))
    planned = acquisition.proposals.copy()
    _write_frame(fixed_outputs["planned_structural"], planned)

    structural_cores = (
        int(config.get("parallel", {}).get("cores_per_worker", 1))
        * int(config.get("parallel", {}).get("omp_threads_per_worker", 1))
    )
    if structural_cores < 1:
        raise MaterialALRoundError("parallel.cores_per_worker must be positive")
    configured_workers = int(config.get("parallel", {}).get("max_workers", 1))
    structural_workers = min(
        max(1, configured_workers), max(1, available // structural_cores)
    )
    targets = tuple(str(name) for name in config.get("targets", {}))
    try:
        new_structural, candidate_manifests = _evaluate_structural_batch(
            proposals=planned,
            names=names,
            targets=targets,
            seeds=seeds,
            runner=runner,
            config_path=runtime_source,
            runner_input_artifacts=runner_input_artifacts,
            config=config,
            scientific_config=_structural_scientific_config(config),
            root=destination / "structural_candidates",
            workers=structural_workers,
        )
    finally:
        _release_structural_scheduler_pool(runner)
    _write_frame(fixed_outputs["new_structural"], new_structural)
    cumulative_structural = _merge_evidence(structural, new_structural, names)
    _write_frame(fixed_outputs["cumulative_structural"], cumulative_structural)

    screen = config.get("material_screen", {})
    if not isinstance(screen, Mapping):
        screen = {}
    static_maximum = int(
        active.get("static_points", spec.mechanical_proposals_per_round)
    )
    static_proposals = _select_static_promotions(
        structural=cumulative_structural,
        mechanical=mechanical,
        new_keys=set(new_structural.get("parameter_key", pd.Series(dtype=str)).astype(str)),
        config=config,
        parameter_space=parameter_space,
        maximum=static_maximum,
        elite_fraction=float(screen.get("objective_elite_fraction", 0.10)),
    )
    _write_frame(fixed_outputs["executed_static"], static_proposals)
    static_dir = destination / "static_batch"
    static_dir.mkdir(parents=True, exist_ok=True)
    if static_proposals.empty:
        new_static = pd.DataFrame(columns=[*names, "parameter_key"])
        _write_frame(static_dir / "static_results.csv", new_static)
        static_manifest_path: Path | None = None
        static_completion = {
            "status": "zero_structurally_eligible",
            "requested": 0,
            "completed": 0,
        }
    else:
        maximum_candidate_workers = max(
            1,
            available
            // (int(cores_per_state) * int(omp_threads_per_rank)),
        )
        selected_candidate_workers = min(
            int(candidate_workers or config.get("parallel", {}).get("max_workers", 1)),
            len(static_proposals),
            maximum_candidate_workers,
        )
        resources: NestedResourcePlan = plan_nested_resources(
            available_cores=available,
            cores_per_state=int(cores_per_state),
            candidate_workers=max(1, selected_candidate_workers),
            state_workers=state_workers,
            omp_threads_per_rank=int(omp_threads_per_rank),
        )
        new_static = elasticity_batch_runner(
            config_path=runtime_source,
            parameters_path=fixed_outputs["executed_static"],
            output_dir=static_dir,
            protocol="static",
            resources=resources,
            top_n=max(1, len(static_proposals)),
            near_optimal_window_percent=5.0,
            diversity_slots=min(2, max(0, len(static_proposals) - 1)),
            evaluate_structural_failures=False,
            backend_factory=elasticity_backend_factory,
        )
        summary_path = static_dir / "batch_summary.json"
        if summary_path.is_file():
            static_summary = json.loads(summary_path.read_text(encoding="utf-8"))
            if int(static_summary.get("failed_work_units", 0)):
                raise MaterialALRoundError(
                    "static elastic batch has incomplete work units; rerun the same round"
                )
        static_manifest_path = static_dir / "stage_manifest.json"
        if not static_manifest_path.is_file():
            raise MaterialALRoundError(
                "completed non-empty static batch did not publish stage_manifest.json"
            )
        static_completion = {
            "status": "completed",
            "requested": len(static_proposals),
            "completed": len(new_static),
            "stage_manifest": str(static_manifest_path),
        }
    _write_frame(fixed_outputs["new_static"], new_static)
    _write_json(fixed_outputs["static_completion"], static_completion)
    cumulative_static = _merge_evidence(mechanical, new_static, names)
    _write_frame(fixed_outputs["cumulative_static"], cumulative_static)

    core_dir = destination / "refinement_core"
    core = run_refinement_round(
        spec=spec,
        structural_paths=[fixed_outputs["cumulative_structural"]],
        mechanical_paths=[fixed_outputs["cumulative_static"]],
        candidate_pool_path=fixed_outputs["candidate_pool"],
        previous_state_path=previous_state,
        output_dir=core_dir,
        config_artifact_path=refinement_source,
        acquisition=_ReplayAcquisition(backend, acquisition),
        seed=seed,
        expected_round=round_number,
    )
    for label, filename in _CORE_OUTPUTS.items():
        _copy_or_verify(core_dir / filename, fixed_outputs[label])

    evidence_document = {
        "schema": _EVIDENCE_SCHEMA,
        "disposition": "executed",
        "round": round_number,
        "previous_round": previous_document.get("round") if previous_document else None,
        "cumulative_structural": _relative_record(
            fixed_outputs["cumulative_structural"], destination, rows=len(cumulative_structural)
        ),
        "cumulative_static": _relative_record(
            fixed_outputs["cumulative_static"], destination, rows=len(cumulative_static)
        ),
        "new_structural": _relative_record(
            fixed_outputs["new_structural"], destination, rows=len(new_structural)
        ),
        "new_static": _relative_record(
            fixed_outputs["new_static"], destination, rows=len(new_static)
        ),
        "candidate_manifests": [
            _relative_record(path, destination) for path in candidate_manifests
        ],
        "static_batch_manifest": (
            _relative_record(static_manifest_path, destination)
            if static_manifest_path is not None
            else None
        ),
        "training_counts": {
            "pre_round_structural": pre_structural_rows,
            "post_round_structural": len(cumulative_structural),
            "pre_round_static": pre_mechanical_rows,
            "post_round_static": len(cumulative_static),
        },
        "acquisition": {
            "backend": acquisition.backend,
            "capability": acquisition.capability,
            "scientific_convergence_capable": (
                acquisition.scientific_convergence_capable
            ),
            "diagnostics": dict(acquisition.diagnostics),
        },
    }
    _write_json(fixed_outputs["round_evidence"], evidence_document)
    manifest = build_artifact_manifest(
        kind="stage",
        identifier=f"{_STAGE_IDENTIFIER}:{round_number:04d}",
        parameters=None,
        seeds=seeds,
        scientific_config=scientific_identity,
        input_artifacts=input_artifacts,
        expected_outputs=fixed_outputs,
    )
    write_artifact_manifest(stage_manifest, manifest)
    return MaterialALRoundResult(destination, core.state, False)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run exactly one resumable GP/exact material AL round"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--refinement-config", required=True, type=Path)
    parser.add_argument("--structural-observations", action="append", type=Path, default=[])
    parser.add_argument("--static-observations", action="append", type=Path, default=[])
    parser.add_argument("--candidate-pool", action="append", type=Path, default=[])
    parser.add_argument("--previous-state", type=Path)
    parser.add_argument("--previous-evidence", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--round", required=True, type=int)
    parser.add_argument("--max-rounds", required=True, type=int)
    parser.add_argument("--seed", type=int, default=20260820)
    parser.add_argument("--structural-seeds", nargs="+", type=int, default=[101])
    parser.add_argument("--available-cores", required=True, type=int)
    parser.add_argument("--cores-per-state", type=int, default=2)
    parser.add_argument("--omp-threads-per-state", type=int, default=1)
    parser.add_argument("--candidate-workers", type=int)
    parser.add_argument("--state-workers", type=int)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = run_material_al_round(
        runtime_config_path=args.config,
        refinement_config_path=args.refinement_config,
        structural_paths=args.structural_observations,
        mechanical_paths=args.static_observations,
        candidate_pool_paths=args.candidate_pool,
        previous_state_path=args.previous_state,
        previous_evidence_path=args.previous_evidence,
        output_dir=args.output_dir,
        round_number=args.round,
        maximum_rounds=args.max_rounds,
        seed=args.seed,
        structural_seeds=args.structural_seeds,
        available_cores=args.available_cores,
        cores_per_state=args.cores_per_state,
        omp_threads_per_rank=args.omp_threads_per_state,
        candidate_workers=args.candidate_workers,
        state_workers=args.state_workers,
    )
    payload = dict(result.state)
    payload.update({"output_dir": str(result.output_dir), "reused": result.reused})
    print(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "MaterialALRoundError",
    "MaterialALRoundResult",
    "build_parser",
    "main",
    "run_material_al_round",
]
