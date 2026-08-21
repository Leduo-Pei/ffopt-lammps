"""Resumable single-candidate runners for cubic elastic properties.

The numerical reductions live in :mod:`engine.cubic_elasticity`.  This module
owns the execution contract around those reductions:

* one equilibrated reference is created per candidate;
* every strain state starts independently from that same reference artifact;
* completed states are reused only after content-addressed manifest checks;
* parameters, scientific configuration, one trajectory seed, and input artifacts form an
  immutable candidate identity; and
* state-level parallelism is rejected before launch when it would oversubscribe
  the explicitly supplied core budget.

The backend protocol is deliberately small.  Unit tests and alternative
simulation engines can implement it without LAMMPS, while
:class:`LammpsTemplateElasticBackend` provides the production subprocess
adapter for the templates installed with FFOpt.
"""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
import csv
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
from typing import Any, Protocol
import uuid

from .cubic_elasticity import fit_cubic_energy_strain, fit_cubic_stress_strain
from workflow.artifact_manifest import (
    build_artifact_manifest,
    canonical_parameter_key,
    check_artifact_reuse,
    scientific_config_hash,
    sha256_file,
    write_artifact_manifest,
)


STATIC_PROTOCOL = "cubic_static_0k"
DYNAMIC_PROTOCOL = "cubic_dynamic_finite_temperature"
STATIC_MODES = ("hydro", "orthorhombic", "shear")
DYNAMIC_DIRECTIONS = ("x", "y", "z", "xy", "xz", "yz")

# LAMMPS ``real`` energy divided by A^3 -> GPa.
KCAL_PER_MOL_ANGSTROM3_TO_GPA = 6.947695457055374
ATM_TO_GPA = 0.000101325


class CubicElasticRunnerError(RuntimeError):
    """Base error for single-candidate elasticity execution."""


class CandidateIdentityError(CubicElasticRunnerError):
    """Raised when a candidate directory belongs to another scientific input."""


class StateArtifactError(CubicElasticRunnerError):
    """Raised when an allegedly completed state fails closed validation."""


@dataclass(frozen=True)
class ElasticState:
    """One reference or independently applied deformation state."""

    protocol: str
    name: str
    role: str
    strain: float = 0.0
    mode: str | None = None
    direction: str | None = None

    def scientific_dict(self) -> dict[str, Any]:
        return {
            "protocol": self.protocol,
            "name": self.name,
            "role": self.role,
            "strain": float(self.strain),
            "mode": self.mode,
            "direction": self.direction,
        }


@dataclass(frozen=True)
class StateExecutionResult:
    """Record and extra output files produced by one backend invocation.

    Artifact paths must be relative to the supplied state work directory.  The
    runner adds ``result.json`` itself and verifies that every declared file is
    a regular file before publishing the state.
    """

    record: Mapping[str, Any]
    artifacts: Mapping[str, str | os.PathLike[str]]


class ElasticStateBackend(Protocol):
    """Backend contract consumed by :class:`SingleCandidateCubicElasticRunner`."""

    def input_artifacts(self) -> Mapping[str, str | os.PathLike[str]]:
        """Return immutable templates/data/force-field files used by the backend."""

    def scientific_identity(self) -> Mapping[str, Any]:
        """Return non-file backend settings that can change scientific results."""

    def expected_artifacts(self, state: ElasticState) -> Mapping[str, str | os.PathLike[str]]:
        """Return label-to-relative-path outputs expected for ``state``."""

    def run_state(
        self,
        state: ElasticState,
        work_dir: Path,
        reference_artifacts: Mapping[str, Path],
        *,
        cores: int,
        trajectory_seed: int | None,
    ) -> StateExecutionResult:
        """Execute one state in ``work_dir`` and return its scientific record."""


@dataclass(frozen=True)
class ResourceBudget:
    """Explicit nested-parallelism budget for strain-state execution."""

    max_workers: int
    cores_per_state: int
    available_cores: int
    omp_threads_per_rank: int = 1

    def __post_init__(self) -> None:
        for name, value in (
            ("max_workers", self.max_workers),
            ("cores_per_state", self.cores_per_state),
            ("available_cores", self.available_cores),
            ("omp_threads_per_rank", self.omp_threads_per_rank),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        requested = (
            self.max_workers
            * self.cores_per_state
            * self.omp_threads_per_rank
        )
        if requested > self.available_cores:
            raise ValueError(
                "Elastic state parallelism would oversubscribe the allocation: "
                f"max_workers({self.max_workers}) * cores_per_state({self.cores_per_state}) "
                f"* omp_threads_per_rank({self.omp_threads_per_rank}) "
                f"= {requested} > available_cores({self.available_cores})"
            )

    def to_dict(self) -> dict[str, int]:
        return {
            "max_workers": self.max_workers,
            "cores_per_state": self.cores_per_state,
            "available_cores": self.available_cores,
            "omp_threads_per_rank": self.omp_threads_per_rank,
        }


def _finite_strains(values: Sequence[float]) -> tuple[float, ...]:
    strains: list[float] = []
    for raw in values:
        value = float(raw)
        if not math.isfinite(value):
            raise ValueError("strain values must be finite")
        if abs(value) <= 1.0e-14:
            continue
        strains.append(value)
    if not strains:
        raise ValueError("at least one non-zero symmetric strain pair is required")
    strains = sorted(set(strains))
    for value in strains:
        if not any(math.isclose(other, -value, rel_tol=1.0e-9, abs_tol=1.0e-14) for other in strains):
            raise ValueError(f"strain values must be symmetric; missing {-value}")
    return tuple(strains)


def _json_bytes(value: Any) -> bytes:
    try:
        text = json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise CubicElasticRunnerError(f"scientific record is not strict JSON: {exc}") from exc
    return (text + "\n").encode("ascii")


def _write_immutable(path: Path, value: Any) -> None:
    """Create a byte-stable JSON file and refuse non-identical replacement."""

    content = _json_bytes(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise CandidateIdentityError(f"refusing to overwrite immutable identity: {path}")
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
                raise CandidateIdentityError(
                    f"refusing to overwrite immutable identity: {path}"
                )
        except OSError:
            if path.exists():
                if path.read_bytes() != content:
                    raise CandidateIdentityError(
                        f"refusing to overwrite immutable identity: {path}"
                    )
            else:
                os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    """Atomically replace a generated, non-identity JSON reduction."""

    path.parent.mkdir(parents=True, exist_ok=True)
    content = _json_bytes(value)
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


def _safe_label(label: str) -> str:
    if not label or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in label):
        raise ValueError(f"artifact label must contain only letters, digits, '_' or '-': {label!r}")
    return label


def _resolve_relative_outputs(
    root: Path,
    paths: Mapping[str, str | os.PathLike[str]],
) -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    resolved_root = root.resolve()
    for raw_label, raw_path in paths.items():
        label = _safe_label(str(raw_label))
        relative = Path(raw_path)
        if relative.is_absolute():
            raise CubicElasticRunnerError(
                f"backend output {label!r} must be relative to its state directory"
            )
        target = (root / relative).resolve()
        try:
            target.relative_to(resolved_root)
        except ValueError as exc:
            raise CubicElasticRunnerError(
                f"backend output {label!r} escapes its state directory: {relative}"
            ) from exc
        outputs[label] = target
    return outputs


def _identity_payload(
    *,
    candidate_id: str,
    protocol: str,
    parameters: Mapping[str, float] | Sequence[float],
    trajectory_seed: int | None,
    scientific_config: Mapping[str, Any],
    input_artifacts: Mapping[str, str | os.PathLike[str]],
) -> dict[str, Any]:
    artifact_hashes = {
        str(label): sha256_file(path).to_dict()
        for label, path in sorted(input_artifacts.items())
    }
    normalized_parameters: dict[str, float] | list[float]
    if isinstance(parameters, Mapping):
        normalized_parameters = {
            str(name): float(value) for name, value in sorted(parameters.items())
        }
    else:
        normalized_parameters = [float(value) for value in parameters]
    core = {
        "schema_version": 1,
        "candidate_id": candidate_id,
        "protocol": protocol,
        "parameters": normalized_parameters,
        "parameter_key": canonical_parameter_key(parameters),
        "trajectory_seed": trajectory_seed,
        "scientific_config": scientific_config,
        "scientific_config_sha256": scientific_config_hash(scientific_config),
        "input_artifacts": artifact_hashes,
    }
    return {
        **core,
        "candidate_fingerprint": scientific_config_hash(core),
    }


def _state_tag(prefix: str, strain: float) -> str:
    sign = "p" if strain >= 0.0 else "m"
    magnitude = f"{abs(strain):.9g}".replace(".", "p").replace("-", "m")
    return f"{prefix}_{sign}{magnitude}"


def static_states(strains: Sequence[float]) -> tuple[ElasticState, tuple[ElasticState, ...]]:
    """Build a 0 K reference and independent three-mode deformation states."""

    values = _finite_strains(strains)
    reference = ElasticState(STATIC_PROTOCOL, "reference", "reference")
    states = tuple(
        ElasticState(
            STATIC_PROTOCOL,
            _state_tag(mode, strain),
            "strain",
            strain=strain,
            mode=mode,
        )
        for mode in STATIC_MODES
        for strain in values
    )
    return reference, states


def dynamic_states(strains: Sequence[float]) -> tuple[ElasticState, tuple[ElasticState, ...]]:
    """Build a finite-T reference, zero state, and six directional strain states."""

    values = _finite_strains(strains)
    reference = ElasticState(DYNAMIC_PROTOCOL, "reference", "reference")
    states = [ElasticState(DYNAMIC_PROTOCOL, "zero", "strain", direction="zero")]
    states.extend(
        ElasticState(
            DYNAMIC_PROTOCOL,
            _state_tag(direction, strain),
            "strain",
            strain=strain,
            direction=direction,
        )
        for direction in DYNAMIC_DIRECTIONS
        for strain in values
    )
    return reference, tuple(states)


class SingleCandidateCubicElasticRunner:
    """Execute and reduce one immutable cubic-elastic candidate directory."""

    def __init__(
        self,
        *,
        candidate_dir: str | os.PathLike[str],
        candidate_id: str,
        protocol: str,
        parameters: Mapping[str, float] | Sequence[float],
        scientific_config: Mapping[str, Any],
        trajectory_seed: int | None,
        backend: ElasticStateBackend,
        resource_budget: ResourceBudget,
    ) -> None:
        if protocol not in {STATIC_PROTOCOL, DYNAMIC_PROTOCOL}:
            raise ValueError(f"unsupported cubic elasticity protocol {protocol!r}")
        if not candidate_id.strip():
            raise ValueError("candidate_id must be non-empty")
        if trajectory_seed is not None and (
            isinstance(trajectory_seed, bool) or not isinstance(trajectory_seed, int)
        ):
            raise ValueError("trajectory_seed must be one integer or None")
        if protocol == DYNAMIC_PROTOCOL and trajectory_seed is None:
            raise ValueError("finite-temperature candidates require exactly one trajectory_seed")
        if protocol == STATIC_PROTOCOL and trajectory_seed is not None:
            raise ValueError("deterministic 0 K candidates require trajectory_seed=None")
        self.candidate_dir = Path(candidate_dir).resolve()
        self.candidate_id = candidate_id
        self.protocol = protocol
        self.parameters = parameters
        self.scientific_config = dict(scientific_config)
        self.trajectory_seed = trajectory_seed
        self.backend = backend
        self.resource_budget = resource_budget
        self._plan_config: dict[str, Any] | None = None
        self.input_artifacts = {
            str(label): Path(path).resolve()
            for label, path in backend.input_artifacts().items()
        }
        if not self.input_artifacts:
            raise ValueError("backend must declare at least one immutable input artifact")
        self.backend_scientific_identity = dict(backend.scientific_identity())

    @property
    def identity_path(self) -> Path:
        return self.candidate_dir / "candidate_identity.json"

    @property
    def candidate_manifest_path(self) -> Path:
        return self.candidate_dir / "artifact_manifest.json"

    def _prepare_identity(self) -> None:
        self.candidate_dir.mkdir(parents=True, exist_ok=True)
        identity = _identity_payload(
            candidate_id=self.candidate_id,
            protocol=self.protocol,
            parameters=self.parameters,
            trajectory_seed=self.trajectory_seed,
            scientific_config=self._effective_config,
            input_artifacts=self.input_artifacts,
        )
        _write_immutable(self.identity_path, identity)

    @property
    def _effective_config(self) -> dict[str, Any]:
        if self._plan_config is None:
            raise CubicElasticRunnerError("elastic strain plan has not been configured")
        return self._plan_config

    def _configure_plan(self, strains: Sequence[float]) -> tuple[float, ...]:
        values = _finite_strains(strains)
        self._plan_config = {
            **self.scientific_config,
            "elastic_protocol": self.protocol,
            "strain_values": list(values),
            "backend_scientific_identity": self.backend_scientific_identity,
        }
        return values

    def _state_config(self, state: ElasticState) -> dict[str, Any]:
        return {
            "candidate_protocol": self.protocol,
            "candidate_scientific_config": self._effective_config,
            "state": state.scientific_dict(),
        }

    def _state_root(self, state: ElasticState) -> Path:
        return self.candidate_dir / "states" / state.name

    def _state_inputs(
        self,
        reference_artifacts: Mapping[str, Path],
    ) -> dict[str, Path]:
        inputs = {"candidate_identity": self.identity_path, **self.input_artifacts}
        inputs.update(
            {f"reference_{label}": path for label, path in reference_artifacts.items()}
        )
        return inputs

    @property
    def _manifest_seeds(self) -> tuple[int, ...]:
        return () if self.trajectory_seed is None else (self.trajectory_seed,)

    def _completed_outputs(self, state: ElasticState) -> dict[str, Path]:
        completed = self._state_root(state) / "completed"
        declared = _resolve_relative_outputs(completed, self.backend.expected_artifacts(state))
        return {"result": completed / "result.json", **declared}

    def _load_reusable_state(
        self,
        state: ElasticState,
        reference_artifacts: Mapping[str, Path],
    ) -> tuple[dict[str, Any], dict[str, Path]] | None:
        state_root = self._state_root(state)
        manifest_path = state_root / "artifact_manifest.json"
        if not manifest_path.exists():
            return None
        outputs = self._completed_outputs(state)
        decision = check_artifact_reuse(
            manifest_path,
            kind="candidate",
            identifier=f"{self.candidate_id}:state:{state.name}",
            parameters=self.parameters,
            seeds=self._manifest_seeds,
            scientific_config=self._state_config(state),
            input_artifacts=self._state_inputs(reference_artifacts),
            expected_outputs=outputs,
        )
        if not decision.reusable:
            raise StateArtifactError(
                f"completed state {state.name!r} is not safely reusable: {decision.reason}"
            )
        try:
            record = json.loads(outputs["result"].read_text(encoding="ascii"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise StateArtifactError(
                f"completed state {state.name!r} has an unreadable result: {exc}"
            ) from exc
        return record, {label: path for label, path in outputs.items() if label != "result"}

    def _publish_attempt(self, state: ElasticState, attempt: Path) -> Path:
        state_root = self._state_root(state)
        completed = state_root / "completed"
        if completed.exists():
            abandoned = state_root / "abandoned" / (
                f"completed_without_manifest_{uuid.uuid4().hex}"
            )
            abandoned.parent.mkdir(parents=True, exist_ok=True)
            os.replace(completed, abandoned)
        os.replace(attempt, completed)
        return completed

    def _run_state(
        self,
        state: ElasticState,
        reference_artifacts: Mapping[str, Path],
    ) -> tuple[dict[str, Any], dict[str, Path]]:
        reusable = self._load_reusable_state(state, reference_artifacts)
        if reusable is not None:
            return reusable

        state_root = self._state_root(state)
        attempts = state_root / "attempts"
        attempts.mkdir(parents=True, exist_ok=True)
        attempt = attempts / uuid.uuid4().hex
        attempt.mkdir()
        result = self.backend.run_state(
            state,
            attempt,
            reference_artifacts,
            cores=self.resource_budget.cores_per_state,
            trajectory_seed=self.trajectory_seed,
        )
        if not isinstance(result, StateExecutionResult):
            raise CubicElasticRunnerError(
                f"backend returned {type(result).__name__}, expected StateExecutionResult"
            )
        declared = _resolve_relative_outputs(attempt, self.backend.expected_artifacts(state))
        returned = _resolve_relative_outputs(attempt, result.artifacts)
        if set(declared) != set(returned):
            raise CubicElasticRunnerError(
                f"backend artifact labels for {state.name!r} differ from the declaration: "
                f"expected {sorted(declared)}, returned {sorted(returned)}"
            )
        for label in declared:
            if declared[label] != returned[label]:
                raise CubicElasticRunnerError(
                    f"backend artifact path mismatch for {state.name!r}/{label}: "
                    f"expected {declared[label]}, returned {returned[label]}"
                )
            if not declared[label].is_file():
                raise CubicElasticRunnerError(
                    f"backend did not produce {state.name!r}/{label}: {declared[label]}"
                )
        _write_json(attempt / "result.json", dict(result.record))
        completed = self._publish_attempt(state, attempt)
        outputs = self._completed_outputs(state)
        manifest = build_artifact_manifest(
            kind="candidate",
            identifier=f"{self.candidate_id}:state:{state.name}",
            parameters=self.parameters,
            seeds=self._manifest_seeds,
            scientific_config=self._state_config(state),
            input_artifacts=self._state_inputs(reference_artifacts),
            expected_outputs=outputs,
        )
        write_artifact_manifest(state_root / "artifact_manifest.json", manifest)
        record = json.loads((completed / "result.json").read_text(encoding="ascii"))
        return record, {label: path for label, path in outputs.items() if label != "result"}

    def _run_independent_states(
        self,
        states: Sequence[ElasticState],
        reference_artifacts: Mapping[str, Path],
    ) -> list[tuple[ElasticState, dict[str, Any]]]:
        completed: list[tuple[ElasticState, dict[str, Any]]] = []
        failures: list[str] = []
        with ThreadPoolExecutor(max_workers=self.resource_budget.max_workers) as executor:
            futures = {
                executor.submit(self._run_state, state, reference_artifacts): state
                for state in states
            }
            for future in as_completed(futures):
                state = futures[future]
                try:
                    record, _outputs = future.result()
                    completed.append((state, record))
                except Exception as exc:  # preserve every independent state result
                    failures.append(f"{state.name}: {exc}")
        if failures:
            raise CubicElasticRunnerError(
                "one or more independent elastic states failed: " + "; ".join(failures)
            )
        order = {state.name: index for index, state in enumerate(states)}
        completed.sort(key=lambda item: order[item[0].name])
        return completed

    def _candidate_outputs(self) -> dict[str, Path]:
        return {
            "summary": self.candidate_dir / "elasticity_summary.json",
            "records_json": self.candidate_dir / "strain_records.json",
            "records_csv": self.candidate_dir / "strain_records.csv",
        }

    def _candidate_inputs(self, states: Sequence[ElasticState]) -> dict[str, Path]:
        inputs = {"candidate_identity": self.identity_path, **self.input_artifacts}
        inputs.update(
            {
                f"state_manifest_{state.name}": self._state_root(state) / "artifact_manifest.json"
                for state in states
            }
        )
        return inputs

    def _reuse_candidate(self, states: Sequence[ElasticState]) -> dict[str, Any] | None:
        if not self.candidate_manifest_path.exists():
            return None
        decision = check_artifact_reuse(
            self.candidate_manifest_path,
            kind="candidate",
            identifier=self.candidate_id,
            parameters=self.parameters,
            seeds=self._manifest_seeds,
            scientific_config=self._effective_config,
            input_artifacts=self._candidate_inputs(states),
            expected_outputs=self._candidate_outputs(),
        )
        if not decision.reusable:
            raise StateArtifactError(
                f"completed candidate {self.candidate_id!r} is not safely reusable: "
                f"{decision.reason}"
            )
        return json.loads(
            self._candidate_outputs()["summary"].read_text(encoding="ascii")
        )

    @staticmethod
    def _flatten_summary(protocol: str, fit: Mapping[str, Any]) -> dict[str, Any]:
        elasticity = dict(fit["elasticity"])
        constants = dict(elasticity["elastic_constants_gpa"])
        independent = dict(elasticity["independent_targets_gpa"])
        poly = dict(elasticity["polycrystalline_moduli_gpa"])
        return {
            "schema_version": 1,
            "protocol": protocol,
            "elastic_constants_gpa": constants,
            "independent_moduli_gpa": independent,
            "derived_diagnostics": {
                "G_hill_gpa": poly["G_hill"],
                "E_hill_gpa": poly["E_hill"],
                "nu_hill": elasticity["nu_hill"],
                "zener_anisotropy": elasticity["zener_anisotropy"],
                "cauchy_pressure_gpa": elasticity["cauchy_pressure_gpa"],
            },
            "born_stability": elasticity["born_stability"],
            "fit_method": fit["method"],
            "fit_quality": fit["fit_quality"],
            "fit_diagnostics": fit["fits"],
            "eligible": elasticity["eligible"],
        }

    def _write_records_csv(self, records: Sequence[Mapping[str, Any]]) -> None:
        path = self._candidate_outputs()["records_csv"]
        keys = sorted({str(key) for record in records for key in record})
        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
        )
        temporary = Path(temporary_name)
        try:
            with os.fdopen(descriptor, "w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=keys)
                writer.writeheader()
                writer.writerows(records)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)

    def _publish_candidate(
        self,
        *,
        all_states: Sequence[ElasticState],
        records: Sequence[Mapping[str, Any]],
        summary: Mapping[str, Any],
    ) -> dict[str, Any]:
        outputs = self._candidate_outputs()
        _write_json(outputs["records_json"], list(records))
        self._write_records_csv(records)
        _write_json(outputs["summary"], dict(summary))
        manifest = build_artifact_manifest(
            kind="candidate",
            identifier=self.candidate_id,
            parameters=self.parameters,
            seeds=self._manifest_seeds,
            scientific_config=self._effective_config,
            input_artifacts=self._candidate_inputs(all_states),
            expected_outputs=outputs,
        )
        write_artifact_manifest(self.candidate_manifest_path, manifest)
        return dict(summary)

    def run_static(self, strains: Sequence[float]) -> dict[str, Any]:
        """Run or safely reuse a 0 K three-mode energy-curvature candidate."""

        if self.protocol != STATIC_PROTOCOL:
            raise ValueError(f"runner protocol is {self.protocol!r}, not {STATIC_PROTOCOL!r}")
        values = self._configure_plan(strains)
        reference, deformation_states = static_states(values)
        all_states = (reference, *deformation_states)
        self._prepare_identity()
        reusable = self._reuse_candidate(all_states)
        if reusable is not None:
            return reusable
        reference_record, reference_artifacts = self._run_state(reference, {})
        try:
            reference_energy = float(reference_record["potential_energy_kcal_mol"])
            reference_volume = float(reference_record["volume_angstrom3"])
        except (KeyError, TypeError, ValueError) as exc:
            raise CubicElasticRunnerError(
                "static reference record needs potential_energy_kcal_mol and volume_angstrom3"
            ) from exc
        if not math.isfinite(reference_energy) or not math.isfinite(reference_volume) or reference_volume <= 0.0:
            raise CubicElasticRunnerError("static reference energy/volume must be finite and volume positive")
        completed = self._run_independent_states(deformation_states, reference_artifacts)
        records: list[dict[str, Any]] = []
        for state, raw in completed:
            try:
                potential = float(raw["potential_energy_kcal_mol"])
                volume = float(raw["volume_angstrom3"])
            except (KeyError, TypeError, ValueError) as exc:
                raise CubicElasticRunnerError(
                    f"static state {state.name!r} needs potential_energy_kcal_mol and volume_angstrom3"
                ) from exc
            records.append(
                {
                    "state": state.name,
                    "mode": state.mode,
                    "strain": state.strain,
                    "potential_energy_kcal_mol": potential,
                    "volume_angstrom3": volume,
                    # Elastic energy is normalized by the common reference
                    # volume, never by each deformed cell volume.
                    "energy_density_gpa": (
                        potential / reference_volume * KCAL_PER_MOL_ANGSTROM3_TO_GPA
                    ),
                }
            )
        reference_density = (
            reference_energy / reference_volume * KCAL_PER_MOL_ANGSTROM3_TO_GPA
        )
        fit = fit_cubic_energy_strain(
            records,
            reference_energy_density_gpa=reference_density,
        )
        summary = self._flatten_summary(self.protocol, fit)
        summary.update(
            {
                "candidate_id": self.candidate_id,
                "candidate_fingerprint": json.loads(
                    self.identity_path.read_text(encoding="ascii")
                )["candidate_fingerprint"],
                "reference": dict(reference_record),
                "strain_values": list(values),
                "resource_budget": self.resource_budget.to_dict(),
                "trajectory_seed": self.trajectory_seed,
            }
        )
        return self._publish_candidate(
            all_states=all_states,
            records=records,
            summary=summary,
        )

    def run_dynamic(self, strains: Sequence[float]) -> dict[str, Any]:
        """Run or safely reuse finite-T directional stress-strain states."""

        if self.protocol != DYNAMIC_PROTOCOL:
            raise ValueError(f"runner protocol is {self.protocol!r}, not {DYNAMIC_PROTOCOL!r}")
        values = self._configure_plan(strains)
        reference, deformation_states = dynamic_states(values)
        all_states = (reference, *deformation_states)
        self._prepare_identity()
        reusable = self._reuse_candidate(all_states)
        if reusable is not None:
            return reusable
        reference_record, reference_artifacts = self._run_state(reference, {})
        completed = self._run_independent_states(deformation_states, reference_artifacts)
        records: list[dict[str, Any]] = []
        stress_fields = ("sxx", "syy", "szz", "sxy", "sxz", "syz")
        for state, raw in completed:
            try:
                row = {field: float(raw[field]) for field in stress_fields}
            except (KeyError, TypeError, ValueError) as exc:
                raise CubicElasticRunnerError(
                    f"dynamic state {state.name!r} must provide {', '.join(stress_fields)}"
                ) from exc
            records.append(
                {
                    "state": state.name,
                    "direction": state.direction,
                    "strain": state.strain,
                    **row,
                    **{
                        key: raw[key]
                        for key in ("temperature_k", "density_g_cm3")
                        if key in raw
                    },
                }
            )
        stress_convention = str(
            self.scientific_config.get("stress_convention", "compression_positive")
        )
        stress_to_gpa = float(
            self.scientific_config.get(
                "stress_to_gpa",
                ATM_TO_GPA if stress_convention == "compression_positive" else 1.0,
            )
        )
        fit = fit_cubic_stress_strain(
            records,
            stress_convention=stress_convention,
            stress_to_gpa=stress_to_gpa,
        )
        summary = self._flatten_summary(self.protocol, fit)
        summary.update(
            {
                "candidate_id": self.candidate_id,
                "candidate_fingerprint": json.loads(
                    self.identity_path.read_text(encoding="ascii")
                )["candidate_fingerprint"],
                "reference": dict(reference_record),
                "strain_values": list(values),
                "stress_convention": stress_convention,
                "stress_to_gpa": stress_to_gpa,
                "resource_budget": self.resource_budget.to_dict(),
                "trajectory_seed": self.trajectory_seed,
            }
        )
        return self._publish_candidate(
            all_states=all_states,
            records=records,
            summary=summary,
        )


@dataclass
class LammpsTemplateElasticBackend:
    """Subprocess backend for the packaged cubic-elastic LAMMPS templates.

    ``command`` is an argv sequence, never a shell string.  The token
    ``{cores}`` is replaced for each state, for example::

        ("srun", "--exclusive", "-n", "{cores}", "lmp")

    The supplied force-field include is already resolved for the candidate;
    parameter-to-coefficient compilation remains outside this single-candidate
    execution layer.
    """

    protocol: str
    command: Sequence[str]
    reference_template: str | os.PathLike[str]
    strain_template: str | os.PathLike[str]
    bulk_data: str | os.PathLike[str]
    force_field_include: str | os.PathLike[str]
    variables: Mapping[str, Any]
    timeout_seconds: float | None = None
    omp_threads_per_rank: int = 1

    def __post_init__(self) -> None:
        if (
            isinstance(self.omp_threads_per_rank, bool)
            or not isinstance(self.omp_threads_per_rank, int)
            or self.omp_threads_per_rank < 1
        ):
            raise ValueError("omp_threads_per_rank must be a positive integer")

    def input_artifacts(self) -> Mapping[str, str | os.PathLike[str]]:
        return {
            "reference_template": self.reference_template,
            "strain_template": self.strain_template,
            "bulk_data": self.bulk_data,
            "force_field_include": self.force_field_include,
        }

    def scientific_identity(self) -> Mapping[str, Any]:
        return {
            "backend": "lammps_template_elasticity",
            "protocol": self.protocol,
            "variables": {str(key): value for key, value in sorted(self.variables.items())},
        }

    def expected_artifacts(self, state: ElasticState) -> Mapping[str, str]:
        outputs = {"stdout": "lammps.stdout", "stderr": "lammps.stderr"}
        if state.role == "reference":
            outputs["restart"] = "elastic_reference.restart"
        return outputs

    @staticmethod
    def _marker_and_fields(state: ElasticState) -> tuple[str, tuple[str, ...]]:
        if state.protocol == STATIC_PROTOCOL and state.role == "reference":
            return "FFOPT_STATIC_REFERENCE:", (
                "potential_energy_kcal_mol",
                "volume_angstrom3",
                "lx_angstrom",
                "ly_angstrom",
                "lz_angstrom",
                "density_g_cm3",
            )
        if state.protocol == STATIC_PROTOCOL:
            return "FFOPT_STATIC_STRAIN:", (
                "potential_energy_kcal_mol",
                "volume_angstrom3",
                "pxx_atm",
                "pyy_atm",
                "pzz_atm",
                "pxy_atm",
                "density_g_cm3",
            )
        if state.protocol == DYNAMIC_PROTOCOL and state.role == "reference":
            return "FFOPT_DYNAMIC_REFERENCE:", (
                "lx_angstrom",
                "ly_angstrom",
                "lz_angstrom",
                "xy_angstrom",
                "xz_angstrom",
                "yz_angstrom",
                "temperature_k",
                "density_g_cm3",
            )
        return "FFOPT_DYNAMIC_STRESS:", (
            "sxx",
            "syy",
            "szz",
            "sxy",
            "sxz",
            "syz",
            "temperature_k",
            "density_g_cm3",
        )

    @staticmethod
    def _copy(source: str | os.PathLike[str], destination: Path) -> None:
        source_path = Path(source).resolve()
        if not source_path.is_file():
            raise CubicElasticRunnerError(f"LAMMPS backend input is missing: {source_path}")
        shutil.copy2(source_path, destination)

    def run_state(
        self,
        state: ElasticState,
        work_dir: Path,
        reference_artifacts: Mapping[str, Path],
        *,
        cores: int,
        trajectory_seed: int | None,
    ) -> StateExecutionResult:
        if state.protocol != self.protocol:
            raise CubicElasticRunnerError(
                f"backend protocol {self.protocol!r} cannot run {state.protocol!r}"
            )
        template = self.reference_template if state.role == "reference" else self.strain_template
        self._copy(template, work_dir / "in.elastic")
        self._copy(self.force_field_include, work_dir / "force_field.lmp")
        variables = {str(key): value for key, value in self.variables.items()}
        variables["force_field_file"] = "force_field.lmp"
        if state.role == "reference":
            self._copy(self.bulk_data, work_dir / "bulk.data")
            variables["bulk_data"] = "bulk.data"
        else:
            restart = reference_artifacts.get("restart")
            if restart is None:
                raise CubicElasticRunnerError(
                    f"state {state.name!r} did not receive the common reference restart"
                )
            self._copy(restart, work_dir / "elastic_reference.restart")
        if state.mode is not None:
            variables["mode"] = state.mode
        if state.direction is not None:
            variables["direction"] = state.direction
        variables["strain"] = format(float(state.strain), ".17g")
        if state.protocol == DYNAMIC_PROTOCOL:
            if trajectory_seed is None:  # guarded by the runner; keep backend safe standalone
                raise CubicElasticRunnerError("dynamic LAMMPS state needs one trajectory_seed")
            variables["npt_seed"] = trajectory_seed

        command = [str(token).replace("{cores}", str(cores)) for token in self.command]
        command.extend(("-in", "in.elastic"))
        for name, value in sorted(variables.items()):
            command.extend(("-var", name, str(value)))
        environment = {
            **os.environ,
            "OMP_NUM_THREADS": str(self.omp_threads_per_rank),
            "MKL_NUM_THREADS": str(self.omp_threads_per_rank),
        }
        completed = subprocess.run(
            command,
            cwd=work_dir,
            stdin=subprocess.DEVNULL,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=self.timeout_seconds,
            check=False,
            env=environment,
        )
        stdout_path = work_dir / "lammps.stdout"
        stderr_path = work_dir / "lammps.stderr"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")
        if completed.returncode != 0:
            raise CubicElasticRunnerError(
                f"LAMMPS state {state.name!r} exited with code {completed.returncode}; "
                f"inspect {stdout_path} and {stderr_path}"
            )

        marker, fields = self._marker_and_fields(state)
        matching = [
            line.strip()[len(marker) :].strip()
            for line in completed.stdout.splitlines()
            if line.strip().startswith(marker)
        ]
        if not matching:
            raise CubicElasticRunnerError(
                f"LAMMPS state {state.name!r} did not print marker {marker}"
            )
        tokens = matching[-1].split()
        if len(tokens) != len(fields):
            raise CubicElasticRunnerError(
                f"LAMMPS marker {marker} returned {len(tokens)} values; expected {len(fields)}"
            )
        try:
            record = {name: float(value) for name, value in zip(fields, tokens, strict=True)}
        except ValueError as exc:
            raise CubicElasticRunnerError(
                f"LAMMPS marker {marker} contains a non-numeric value"
            ) from exc
        artifacts = {"stdout": "lammps.stdout", "stderr": "lammps.stderr"}
        if state.role == "reference":
            restart = work_dir / "elastic_reference.restart"
            if not restart.is_file():
                raise CubicElasticRunnerError(
                    f"LAMMPS reference {state.name!r} did not write {restart.name}"
                )
            artifacts["restart"] = restart.name
        return StateExecutionResult(record=record, artifacts=artifacts)


def _exact_fields(
    value: Any,
    *,
    location: str,
    required: set[str],
    optional: set[str] | None = None,
) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{location} must be a JSON object")
    allowed = required | (optional or set())
    missing = sorted(required - set(value))
    unknown = sorted(set(value) - allowed)
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unknown:
            details.append(f"unknown {unknown}")
        raise ValueError(f"{location} fields are invalid: " + "; ".join(details))
    return value


def _job_relative(base: Path, value: Any, *, field: str) -> Path:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty path string")
    path = Path(value)
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_lammps_candidate_job(
    job_path: str | os.PathLike[str],
) -> tuple[SingleCandidateCubicElasticRunner, tuple[float, ...]]:
    """Compile one strict JSON job into a LAMMPS-backed candidate runner.

    The JSON file is an internal worker contract intended for a future batch or
    pipeline adapter, not a second user-facing project format.  Relative paths
    are resolved against the job file directory.  Runtime resource fields are
    deliberately excluded from the scientific fingerprint.
    """

    source = Path(job_path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read elasticity job {source}: {exc}") from exc
    root = _exact_fields(
        raw,
        location="elasticity job",
        required={
            "schema_version",
            "candidate_id",
            "candidate_dir",
            "protocol",
            "parameters",
            "scientific_config",
            "trajectory_seed",
            "strains",
            "resources",
            "lammps",
        },
    )
    if root["schema_version"] != 1:
        raise ValueError("elasticity job schema_version must be 1")
    if not isinstance(root["scientific_config"], dict):
        raise ValueError("scientific_config must be a JSON object")
    if not isinstance(root["parameters"], (dict, list)):
        raise ValueError("parameters must be a JSON object or array")
    if not isinstance(root["strains"], list):
        raise ValueError("strains must be a JSON array")

    resources = _exact_fields(
        root["resources"],
        location="resources",
        required={"max_workers", "cores_per_state", "available_cores"},
        optional={"omp_threads_per_rank"},
    )
    lammps = _exact_fields(
        root["lammps"],
        location="lammps",
        required={
            "command",
            "reference_template",
            "strain_template",
            "bulk_data",
            "force_field_include",
            "variables",
        },
        optional={"timeout_seconds"},
    )
    command = lammps["command"]
    if (
        not isinstance(command, list)
        or not command
        or any(not isinstance(token, str) or not token for token in command)
    ):
        raise ValueError("lammps.command must be a non-empty JSON array of strings")
    if not isinstance(lammps["variables"], dict):
        raise ValueError("lammps.variables must be a JSON object")

    base = source.parent
    backend = LammpsTemplateElasticBackend(
        protocol=str(root["protocol"]),
        command=tuple(command),
        reference_template=_job_relative(
            base, lammps["reference_template"], field="lammps.reference_template"
        ),
        strain_template=_job_relative(
            base, lammps["strain_template"], field="lammps.strain_template"
        ),
        bulk_data=_job_relative(base, lammps["bulk_data"], field="lammps.bulk_data"),
        force_field_include=_job_relative(
            base, lammps["force_field_include"], field="lammps.force_field_include"
        ),
        variables=lammps["variables"],
        timeout_seconds=(
            None
            if lammps.get("timeout_seconds") is None
            else float(lammps["timeout_seconds"])
        ),
        omp_threads_per_rank=int(resources.get("omp_threads_per_rank", 1)),
    )
    budget = ResourceBudget(
        max_workers=resources["max_workers"],
        cores_per_state=resources["cores_per_state"],
        available_cores=resources["available_cores"],
        omp_threads_per_rank=resources.get("omp_threads_per_rank", 1),
    )
    runner = SingleCandidateCubicElasticRunner(
        candidate_dir=_job_relative(base, root["candidate_dir"], field="candidate_dir"),
        candidate_id=str(root["candidate_id"]),
        protocol=str(root["protocol"]),
        parameters=root["parameters"],
        scientific_config=root["scientific_config"],
        trajectory_seed=root["trajectory_seed"],
        backend=backend,
        resource_budget=budget,
    )
    return runner, _finite_strains(root["strains"])


def run_lammps_candidate_job(job_path: str | os.PathLike[str]) -> dict[str, Any]:
    """Run one strict JSON worker job and return its summary."""

    runner, strains = load_lammps_candidate_job(job_path)
    if runner.protocol == STATIC_PROTOCOL:
        return runner.run_static(strains)
    return runner.run_dynamic(strains)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Run one resumable cubic-elastic candidate job"
    )
    parser.add_argument(
        "--job",
        required=True,
        type=Path,
        help="Strict internal JSON job produced by the FFOpt batch/pipeline layer",
    )
    arguments = parser.parse_args(argv)
    summary = run_lammps_candidate_job(arguments.job)
    print(
        json.dumps(
            {
                "status": "completed",
                "candidate_id": summary["candidate_id"],
                "protocol": summary["protocol"],
                "candidate_fingerprint": summary["candidate_fingerprint"],
            },
            sort_keys=True,
        )
    )
    return 0


__all__ = [
    "ATM_TO_GPA",
    "DYNAMIC_DIRECTIONS",
    "DYNAMIC_PROTOCOL",
    "KCAL_PER_MOL_ANGSTROM3_TO_GPA",
    "STATIC_MODES",
    "STATIC_PROTOCOL",
    "CandidateIdentityError",
    "CubicElasticRunnerError",
    "ElasticState",
    "ElasticStateBackend",
    "LammpsTemplateElasticBackend",
    "ResourceBudget",
    "SingleCandidateCubicElasticRunner",
    "StateArtifactError",
    "StateExecutionResult",
    "dynamic_states",
    "load_lammps_candidate_job",
    "main",
    "run_lammps_candidate_job",
    "static_states",
]


if __name__ == "__main__":
    raise SystemExit(main())
