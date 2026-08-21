"""Restartable BO -> sampling -> NN -> AL workflow orchestration."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import uuid

from ffopt import __version__
from engine.parameter_space import build_parameter_space

from .project import Project, compose_config
from .material_pipeline import (
    MATERIAL_EXECUTABLE_KINDS,
    build_refinement_spec,
    load_refinement_state,
    validate_material_al_stage_outputs,
    write_skipped_refinement_stage,
)
from .stage_registry import (
    LEGACY_PIPELINE_STAGES,
    StageGraph,
    StageNode,
    StageRegistry,
    legacy_stage_registry,
    material_stage_registry,
)
from .state import StageRecord, WorkflowState, canonical_hash

# Public compatibility alias.  The executable definitions, dependencies, command
# tokens, and artifacts live in ``workflow.stage_registry``.
PIPELINE_STAGES = LEGACY_PIPELINE_STAGES


@dataclass(frozen=True)
class StageSpec:
    name: str
    signature: str
    command: list[str]
    output_dir: Path
    artifacts: list[Path]
    kind: str = ""
    command_token: str = ""
    dependencies: tuple[str, ...] = ()


def _serializable_config(config: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def _scientific_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return simulation inputs without host-specific execution settings."""
    excluded = {
        "active_learning",
        "checkpoint",
        "cluster",
        "machine",
        "machine_learning",
        "nn",
        "optimization",
        "parallel",
        "workflow",
    }
    result = {
        key: value
        for key, value in _serializable_config(config).items()
        if key not in excluded
    }
    lammps = dict(result.get("lammps", {}))
    for key in ("executable", "mpiexec", "mpi_flavor", "timeout"):
        lammps.pop(key, None)
    result["lammps"] = lammps
    return result


def _initial_free_parameters(config: dict[str, Any]) -> dict[str, float]:
    """Extract public input initial values for validation-only material runs."""

    names = {name for name, _lower, _upper in build_parameter_space(config)}
    values: dict[str, float] = {}
    for atom_type in config.get("atom_types", []):
        label = str(atom_type["label"])
        for parameter, specification in atom_type.get("params", {}).items():
            name = f"{label}_{parameter}"
            if name not in names or not isinstance(specification, Mapping):
                continue
            initial = specification.get("init")
            if initial is None:
                initial = (
                    float(specification["min"]) + float(specification["max"])
                ) / 2.0
            values[name] = float(initial)
    for pair in config.get("pair_params", {}).get("explicit_pairs", []):
        first, second = pair["types"]
        for parameter in ("epsilon", "sigma"):
            name = f"cross_{first}_{second}_{parameter}"
            if name not in names:
                continue
            specification = pair[parameter]
            initial = specification.get("init")
            if initial is None:
                initial = (
                    float(specification["min"]) + float(specification["max"])
                ) / 2.0
            values[name] = float(initial)
    missing = sorted(names - set(values))
    if missing:
        raise ValueError(f"Initial material parameters are missing: {missing}")
    return {name: values[name] for name, _low, _high in build_parameter_space(config)}


def _command_text(command: Iterable[str]) -> str:
    values = [str(item) for item in command]
    return subprocess.list2cmdline(values) if os.name == "nt" else shlex.join(values)


def _write_immutable(path: Path, text: str) -> None:
    """Create a content-addressed provenance file without partial writes."""
    if path.exists():
        if path.read_text(encoding="utf-8") != text:
            raise RuntimeError(f"Provenance hash collision or corruption: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(text, encoding="utf-8", newline="\n")
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_immutable_bytes(path: Path, payload: bytes) -> None:
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Provenance hash collision or corruption: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(payload)
    try:
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


class PipelineRunner:
    """Build and execute a deterministic, persisted workflow stage graph."""

    def __init__(
        self,
        *,
        project: Project,
        machine: str,
        run_id: str = "default",
        resume: bool = False,
        dry_run: bool = False,
        watch: bool = False,
        poll_seconds: int = 60,
        from_stage: str | None = None,
        until: str | None = None,
        stage_registry: StageRegistry | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise ValueError("run_id may contain only letters, numbers, '.', '_' and '-'")
        self.project = project
        self.machine = machine
        self.config = compose_config(project, machine)
        self.backend = str(
            self.config.get("machine", {}).get(
                "backend", "slurm" if machine == "cluster" else "local"
            )
        )
        if self.backend not in {"local", "slurm"}:
            raise ValueError(f"Unsupported machine backend: {self.backend}")
        self.run_id = run_id
        self.resume = resume
        self.dry_run = dry_run
        self.watch = watch
        self.poll_seconds = max(1, int(poll_seconds))
        self._observed_job_ids: set[str] = set()
        self.root = (project.run_root / "pipelines" / run_id).resolve()
        serialized = _serializable_config(self.config)
        self.config_hash = canonical_hash(serialized)
        self.scientific_hash = canonical_hash(_scientific_config(self.config))
        provenance = self.root / "provenance"
        self.config_path = provenance / f"runtime_{self.config_hash[:16]}.json"
        input_bytes = project.path.read_bytes()
        input_hash = hashlib.sha256(input_bytes).hexdigest()
        self.input_snapshot_path = provenance / f"input_{input_hash[:16]}.in"
        self._config_snapshot = (
            json.dumps(serialized, indent=2, ensure_ascii=False, sort_keys=True) + "\n"
        )
        self._input_snapshot = input_bytes
        environment = {
            "ffopt_lammps": __version__,
            "python": platform.python_version(),
            "python_executable": sys.executable,
            "platform": platform.platform(),
            "machine_profile": machine,
            "backend": self.backend,
            "lammps_executable": self.config.get("lammps", {}).get("executable"),
            "mpi_launcher": self.config.get("lammps", {}).get("mpiexec"),
            "mpi_launcher_flavor": self.config.get("lammps", {}).get("mpi_flavor"),
        }
        environment_text = json.dumps(
            environment, indent=2, ensure_ascii=False, sort_keys=True
        ) + "\n"
        environment_hash = hashlib.sha256(environment_text.encode("utf-8")).hexdigest()
        self.environment_path = provenance / f"environment_{environment_hash[:16]}.json"
        self._environment_snapshot = environment_text
        self.state_path = self.root / "state.sqlite"
        pipeline_cfg = project.data.get("pipeline", {})
        self.pipeline_cfg = pipeline_cfg
        configured_stages = pipeline_cfg.get("stages", PIPELINE_STAGES)
        active_tokens = self._active_stages(list(configured_stages))
        material_config = self.config.get("material")
        compiled_material = bool(
            isinstance(material_config, Mapping)
            and material_config.get("kind") not in {None, "molecular"}
        )
        material_requested = compiled_material or bool(
            set(active_tokens) & ({"screen"} | MATERIAL_EXECUTABLE_KINDS)
        )
        self.stage_registry = stage_registry or (
            material_stage_registry() if material_requested else legacy_stage_registry()
        )
        repetitions = dict(pipeline_cfg.get("stage_repetitions", {}))
        if "constrained_al" in active_tokens and "constrained_al" not in repetitions:
            round_settings = pipeline_cfg.get("constrained_al", {})
            repetitions["constrained_al"] = int(
                round_settings.get(
                    "max_rounds",
                    self.config.get("elasticity", {})
                    .get("refinement", {})
                    .get("maximum_rounds", 8),
                )
            )
        graph_mode = str(pipeline_cfg.get("graph_mode", "linear")).strip().lower()
        if graph_mode not in {"linear", "dag"}:
            raise ValueError("pipeline.graph_mode must be 'linear' or 'dag'")
        self.stage_graph: StageGraph = self.stage_registry.expand(
            active_tokens,
            repetitions=repetitions,
            legacy_linear=graph_mode == "linear",
        )
        self.stage_names = list(self.stage_graph.names)
        self._stage_nodes = {node.name: node for node in self.stage_graph.nodes}
        self.material_workflow = material_requested
        self.refinement_spec_path: Path | None = None
        self._refinement_spec_snapshot: str | None = None
        self.refinement_maximum_rounds: int | None = None
        if self._has_kind("constrained_al"):
            number_of_rounds = sum(
                node.kind == "constrained_al" for node in self.stage_graph.nodes
            )
            refinement_settings = dict(pipeline_cfg.get("constrained_al", {}))
            search_maximum = int(
                refinement_settings.get("maximum_rounds", number_of_rounds)
            )
            if search_maximum < 1 or search_maximum > number_of_rounds:
                raise ValueError(
                    "pipeline.constrained_al.maximum_rounds must lie between 1 "
                    "and the predeclared constrained_al stage repetitions"
                )
            self.refinement_maximum_rounds = search_maximum
            refinement_spec = build_refinement_spec(
                self.config,
                maximum_rounds=search_maximum,
                settings=refinement_settings,
            )
            refinement_text = json.dumps(
                refinement_spec, indent=2, ensure_ascii=False, sort_keys=True
            ) + "\n"
            refinement_hash = hashlib.sha256(
                refinement_text.encode("utf-8")
            ).hexdigest()
            self.refinement_spec_path = (
                provenance / f"refinement_{refinement_hash[:16]}.json"
            )
            self._refinement_spec_snapshot = refinement_text
        self.from_stage = from_stage
        self.until = until
        self._validate_bounds()

    def _write_provenance(self) -> None:
        _write_immutable(self.config_path, self._config_snapshot)
        _write_immutable_bytes(self.input_snapshot_path, self._input_snapshot)
        _write_immutable(self.environment_path, self._environment_snapshot)
        if self.refinement_spec_path is not None:
            assert self._refinement_spec_snapshot is not None
            _write_immutable(
                self.refinement_spec_path, self._refinement_spec_snapshot
            )

    def _active_stages(self, stages: list[str]) -> list[str]:
        if not self.config.get("nn", {}).get("enabled", True):
            stages = [name for name in stages if name not in {"nn", "al"}]
        if not self.config.get("active_learning", {}).get("enabled", True):
            stages = [name for name in stages if name != "al"]
        return stages

    def _validate_bounds(self) -> None:
        for label, value in (("from-stage", self.from_stage), ("until", self.until)):
            if value and value not in self.stage_names:
                raise ValueError(
                    f"--{label}={value!r} is not active; choose from {self.stage_names}"
                )
        if (
            self.from_stage
            and self.until
            and self.stage_names.index(self.from_stage)
            > self.stage_names.index(self.until)
        ):
            raise ValueError("--from-stage must not come after --until")

    def _node(self, name: str) -> StageNode:
        try:
            return self._stage_nodes[name]
        except KeyError as exc:
            raise ValueError(f"Unknown expanded pipeline stage: {name!r}") from exc

    def _has_kind(self, kind: str) -> bool:
        return any(node.kind == kind for node in self.stage_graph.nodes)

    def _selected_names(self) -> list[str]:
        start = self.stage_names.index(self.from_stage) if self.from_stage else 0
        stop = self.stage_names.index(self.until) + 1 if self.until else len(self.stage_names)
        return self.stage_names[start:stop]

    def _stage_settings(self, name: str) -> dict[str, Any]:
        node = self._node(name)
        kind = node.kind
        if kind == "bo":
            return {
                "optimization": self.config.get("optimization", {}),
                "checkpoint": self.config.get("checkpoint", {}),
            }
        if kind == "sample":
            return dict(self.project.data.get("stages", {}).get(name, {}) or
                        self.project.data.get("stages", {}).get(kind, {}))
        if kind == "nn":
            return {"nn": self.config.get("nn", {})}
        if kind == "al":
            return {
                "nn": self.config.get("nn", {}),
                "active_learning": self.config.get("active_learning", {}),
            }
        if kind == "audit":
            configured = self.project.data.get("pipeline", {}).get("audit")
            return dict(
                configured
                or self.config.get("optimization", {}).get("stability_audit", {})
            )
        pipeline = self.project.data.get("pipeline", {})
        stage_blocks = self.project.data.get("stages", {})
        public_stage = (
            "screen" if kind in {"candidates", "static"} else kind
        )
        settings: dict[str, Any] = {}
        if kind == "constrained_al":
            settings.update(self.config.get("active_learning", {}))
        settings.update(stage_blocks.get(public_stage, {}) or {})
        settings.update(pipeline.get(kind, {}) or {})
        settings.update(pipeline.get(name, {}) or {})
        return settings

    def _signature(self, name: str, upstream: str) -> str:
        return canonical_hash({
            "stage": name,
            "scientific_config": _scientific_config(self.config),
            "settings": self._stage_settings(name),
            "upstream": upstream,
        })

    def _python_module(self, module: str, *args: object) -> list[str]:
        return [sys.executable, "-m", module, *[str(item) for item in args]]

    def _sample_command(self, output: Path) -> list[str]:
        settings = self._stage_settings("sample")
        if self.material_workflow:
            # Material sampling has separate, explicit interior and boundary
            # evidence.  Never choose either source by directory age or
            # runtime existence.
            source = self.root / "bo" / "feasible_archive.csv"
            boundary_source = self.root / "bo" / "coverage_anchors.csv"
        else:
            # Preserve the prerelease molecular compatibility behaviour.
            source = self.root / "bo" / "stable_results.csv"
            if not source.exists():
                source = self.root / "bo" / "all_results.csv"
            boundary_source = None
        command = self._python_module(
            "engine.local_sampling",
            "--config", self.config_path,
            "--source", source,
            "--output-dir", output,
            "--n-points", settings.get("n_points", 1200),
            "--elite-centers", settings.get("elite_centers", 8),
            "--center-selection", settings.get("center_selection", "top"),
            "--center-pool-multiplier", settings.get("center_pool_multiplier", 5),
            "--radii", *settings.get("radii", [0.005, 0.015, 0.04]),
            "--global-fraction", settings.get("global_fraction", 0.1),
            "--seeds", *settings.get("seeds", [101, 202, 303]),
            "--design-seed", settings.get("design_seed", 20260709),
        )
        if boundary_source is not None:
            command.extend([
                "--boundary-source", str(boundary_source),
                "--boundary-fraction",
                str(settings.get("boundary_fraction", 0.0)),
            ])
        max_workers = settings.get("max_workers")
        if max_workers:
            command.extend(["--max-workers", str(max_workers)])
        return command

    def _latest_al_data(self) -> Path:
        al_dir = self.root / "al"
        candidates: list[tuple[int, Path]] = []
        for path in al_dir.glob("data_round_*/all_results.csv"):
            match = re.fullmatch(r"data_round_(\d+)", path.parent.name)
            if match:
                candidates.append((int(match.group(1)), path))
        if candidates:
            return max(candidates, key=lambda item: item[0])[1]
        final_round = int(self.config.get("active_learning", {}).get("n_rounds", 1))
        return al_dir / f"data_round_{final_round}" / "all_results.csv"

    def _audit_settings(self) -> tuple[int, list[int], int | None]:
        settings = self._stage_settings("audit")
        top_k = int(settings.get("top_k", 8))
        seeds = [int(seed) for seed in settings.get("seeds", [101, 202, 303])]
        max_workers = settings.get("max_workers")
        return top_k, seeds, int(max_workers) if max_workers else None

    def _material_candidate_sources(self) -> tuple[Path, ...]:
        """Return the declared evidence set; no filesystem discovery is used."""
        sources = [
            self.root / "bo" / "all_results.csv",
            self.root / "bo" / "feasible_archive.csv",
            self.root / "bo" / "coverage_anchors.csv",
        ]
        if self._has_kind("sample"):
            sources.append(self.root / "sample" / "local_results.csv")
        if self._has_kind("audit"):
            sources.append(self.root / "audit" / "stable_results.csv")
        return tuple(sources)

    def _constrained_nodes(self) -> list[StageNode]:
        return [
            node for node in self.stage_graph.nodes if node.kind == "constrained_al"
        ]

    def _constrained_round_number(self, name: str) -> int:
        for index, node in enumerate(self._constrained_nodes(), start=1):
            if node.name == name:
                return index
        raise ValueError(f"Stage {name!r} is not a constrained-refinement round")

    def _previous_constrained_name(self, name: str) -> str | None:
        round_number = self._constrained_round_number(name)
        return (
            None
            if round_number == 1
            else self._constrained_nodes()[round_number - 2].name
        )

    def _elastic_batch_resource_args(self, name: str) -> list[str]:
        profile = self._slurm_profile(name)
        parallel = self.config.get("parallel", {})
        mpi_ranks = max(1, int(parallel.get("cores_per_worker", 1)))
        omp_threads = max(
            1, int(parallel.get("omp_threads_per_worker", 1))
        )
        configured_candidates = max(1, int(parallel.get("max_workers", 1)))
        if self.backend == "slurm":
            available = int(
                profile.get(
                    "cores",
                    int(profile.get("tasks", 1))
                    * int(profile.get("cpus_per_task", 1)),
                )
            )
        else:
            available = configured_candidates * mpi_ranks * omp_threads
        state_cpu_cost = mpi_ranks * omp_threads
        if available < state_cpu_cost:
            raise ValueError(
                f"Machine profile allocates {available} CPUs for {name}, but one "
                f"elastic state needs {mpi_ranks} MPI ranks x {omp_threads} OMP "
                "threads"
            )
        candidate_workers = min(
            configured_candidates, available // state_cpu_cost
        )
        command = [
            "--available-cores", str(available),
            "--cores-per-state", str(mpi_ranks),
            "--omp-threads-per-state", str(omp_threads),
            "--candidate-workers", str(candidate_workers),
        ]
        return command

    def _finalist_selection_args(self, name: str) -> list[str]:
        settings = self._stage_settings(name)
        command: list[str] = []
        for option, keys, converter in (
            ("--minimum", ("minimum",), int),
            ("--top-n", ("top_n", "maximum"), int),
            (
                "--near-optimal-window-percent",
                ("near_optimal_window_percent",),
                float,
            ),
            ("--diversity-slots", ("diversity_slots", "diverse_reserve"), int),
        ):
            value = next(
                (settings[key] for key in keys if settings.get(key) is not None),
                None,
            )
            if value is not None:
                command.extend([option, str(converter(value))])
        return command

    def _stage_command(self, name: str, output: Path) -> list[str]:
        command_token = self._node(name).command_token
        bo_dir = self.root / "bo"
        sample_file = self.root / "sample" / "local_results.csv"
        nn_dir = self.root / "nn"
        if command_token == "bo":
            command = self._python_module(
                "engine.run", "--config", self.config_path, "--output-dir", output
            )
            if self.resume:
                command.append("--resume")
            return command
        if command_token == "sample":
            return self._sample_command(output)
        if command_token == "nn":
            command = self._python_module(
                "engine.nn_surrogate",
                "--config", self.config_path,
                "--bo-dir", bo_dir,
                "--output-dir", output,
            )
            if self._has_kind("sample"):
                command.extend(["--additional-core-file", sample_file])
            return command
        if command_token == "material-nn":
            if self.refinement_spec_path is None:
                raise ValueError("Material surrogate preflight requires a refinement spec")
            return self._python_module(
                "engine.material_surrogate_preflight",
                "--config", self.config_path,
                "--refinement-config", self.refinement_spec_path,
                "--structural-observations",
                self.root / "candidates" / "candidates.csv",
                "--static-observations", self.root / "static" / "static_results.csv",
                "--candidate-pool", self.root / "candidates" / "candidate_pool.csv",
                "--output-dir", output,
                "--seed", self._stage_settings(name).get("seed", 20260820),
            )
        if command_token == "al":
            command = self._python_module(
                "engine.active_learning",
                "--config", self.config_path,
                "--bo-dir", bo_dir,
                "--nn-dir", nn_dir,
                "--output-dir", output,
            )
            if self._has_kind("sample"):
                command.extend(["--additional-core-file", sample_file])
            if self.resume:
                command.append("--resume")
            return command
        if command_token == "audit":
            if self.material_workflow:
                sources = [bo_dir / "all_results.csv"]
                if self._has_kind("sample"):
                    sources.append(sample_file)
            else:
                sources = [
                    self._latest_al_data()
                    if self._has_kind("al")
                    else bo_dir / "all_results.csv"
                ]
            top_k, seeds, max_workers = self._audit_settings()
            command = self._python_module(
                "engine.stability_audit",
                "--config", self.config_path,
                "--output-dir", output,
                "--top-k", top_k,
                "--seeds", *seeds,
            )
            for source in sources:
                command.extend(["--source", str(source)])
            if max_workers:
                command.extend(["--max-workers", str(max_workers)])
            return command
        if command_token == "candidates":
            command = self._python_module(
                "workflow.material_candidates",
                "--config", self.config_path,
            )
            for source in self._material_candidate_sources():
                command.extend(["--source", str(source)])
            screen = self._stage_settings("static")
            for option, key in (
                ("--screen-minimum", "minimum"),
                ("--screen-per-dimension", "per_dimension"),
                ("--screen-maximum", "maximum"),
                ("--screen-core-fraction", "core_fraction"),
                ("--screen-buffer-multiplier", "buffer_multiplier"),
                ("--screen-elite-fraction", "objective_elite_fraction"),
            ):
                if screen.get(key) is not None:
                    command.extend([option, str(screen[key])])
            command.extend(["--output-dir", str(output)])
            return command
        if command_token == "static":
            return self._python_module(
                "engine.cubic_elastic_batch",
                "--config", self.config_path,
                "--parameters",
                self.root / "candidates" / "static_screen_candidates.csv",
                "--output-dir", output,
                "--protocol", "static",
                "--evaluate-structural-failures",
                *self._elastic_batch_resource_args(name),
                *self._finalist_selection_args(name),
            )
        if command_token == "constrained-al":
            if self.refinement_spec_path is None:
                raise ValueError("Constrained AL requires a refinement spec snapshot")
            round_number = self._constrained_round_number(name)
            command = self._python_module(
                "engine.material_al_round",
                "--config", self.config_path,
                "--refinement-config", self.refinement_spec_path,
                "--candidate-pool",
                self.root / "nn" / "candidate_pool.csv",
                "--output-dir", output,
                "--round", round_number,
                "--max-rounds", self.refinement_maximum_rounds,
                "--seed", self._stage_settings(name).get("seed", 20260820),
                "--structural-seeds",
                *self._stage_settings(name).get(
                    "structural_seeds",
                    [self.config.get("lammps", {}).get("bulk", {}).get("npt_seed", 101)],
                ),
                *self._elastic_batch_resource_args(name),
            )
            previous = self._previous_constrained_name(name)
            if previous is None:
                command.extend([
                    "--structural-observations",
                    str(self.root / "candidates" / "candidates.csv"),
                    "--static-observations",
                    str(self.root / "static" / "static_results.csv"),
                ])
            else:
                command.extend([
                    "--previous-state",
                    str(self.root / previous / "refinement_state.json"),
                    "--previous-evidence",
                    str(self.root / previous / "round_evidence.json"),
                ])
            return command
        if command_token == "finalists":
            constrained = self._constrained_nodes()
            if not constrained:
                raise ValueError("Finalists require at least one constrained AL round")
            parameters = (
                self.root
                / constrained[-1].name
                / "exact_structural_mechanical_ranking.csv"
            )
            return self._python_module(
                "engine.cubic_elastic_batch",
                "--config", self.config_path,
                "--parameters", parameters,
                "--output-dir", output,
                "--protocol", "dynamic",
                *self._elastic_batch_resource_args(name),
                *self._finalist_selection_args(name),
            )
        if command_token == "finalize":
            return self._python_module(
                "engine.finalize_al",
                "--config", self.config_path,
                "--audit", self.root / "audit",
                "--output-dir", output,
            )
        if command_token == "material-validate":
            command = self._python_module(
                "engine.material_validation",
                "--config", self.config_path,
                "--output-dir", output,
                *self._elastic_batch_resource_args(name),
            )
            constrained = self._constrained_nodes()
            if self._has_kind("finalists") and constrained:
                finalists = self._stage_settings("finalists")
                top_n = int(finalists.get("maximum", finalists.get("top_n", 20)))
                command.extend([
                    "--parameters", str(self.root / "finalists" / "best_candidate.json"),
                    "--static-ranking",
                    str(
                        self.root
                        / constrained[-1].name
                        / "exact_structural_mechanical_ranking.csv"
                    ),
                    "--dynamic-ranking",
                    str(self.root / "finalists" / "dynamic_results.csv"),
                    "--top-n", str(top_n),
                ])
            elif self._has_kind("bo"):
                command.extend([
                    "--parameters", str(self.root / "bo" / "best_parameters.txt")
                ])
            else:
                for parameter, value in _initial_free_parameters(self.config).items():
                    command.extend(["--parameter", f"{parameter}={value:.17g}"])
            return command
        if command_token == "validate":
            parameters = None
            if self._has_kind("finalists"):
                parameters = self.root / "finalists" / "best_candidate.json"
            elif self._has_kind("finalize"):
                parameters = self.root / "finalize" / "final_summary.json"
            elif self._has_kind("al"):
                parameters = self.root / "al" / "final_parameters.json"
            elif self._has_kind("nn"):
                parameters = self.root / "nn" / "nn_optimize_result.json"
            elif self._has_kind("bo"):
                parameters = self.root / "bo" / "best_parameters.txt"
            command = self._python_module(
                "engine.validate_final_parameters",
                "--config", self.config_path,
                "--output-dir", output,
                "--overwrite",
            )
            if parameters is not None:
                command.extend(["--parameters", str(parameters)])
            else:
                command.append("--initial")
            return command
        raise ValueError(
            f"No command builder is installed for stage {name!r} "
            f"(command token {command_token!r})"
        )

    def _stage_artifacts(self, name: str, output: Path) -> list[Path]:
        return [output / item for item in self._node(name).expected_artifacts]

    def build_specs(self) -> list[StageSpec]:
        specs: list[StageSpec] = []
        signatures: dict[str, str] = {}
        for node in self.stage_graph.nodes:
            name = node.name
            output = self.root / name
            if not node.dependencies:
                upstream = "root"
            elif len(node.dependencies) == 1:
                upstream = signatures[node.dependencies[0]]
            else:
                upstream = canonical_hash({
                    "dependencies": [
                        (dependency, signatures[dependency])
                        for dependency in node.dependencies
                    ]
                })
            signature = self._signature(name, upstream)
            specs.append(StageSpec(
                name=name,
                signature=signature,
                command=self._stage_command(name, output),
                output_dir=output,
                artifacts=self._stage_artifacts(name, output),
                kind=node.kind,
                command_token=node.command_token,
                dependencies=node.dependencies,
            ))
            signatures[name] = signature
        return specs

    def _print_plan(self, specs: list[StageSpec]) -> None:
        print(f"Pipeline : {self.project.name}/{self.run_id}")
        print(f"Backend  : {self.backend}")
        print(f"Run root : {self.root}")
        print("Stages   : " + " -> ".join(spec.name for spec in specs))
        for spec in specs:
            print(f"\n[{spec.name}]\n> {_command_text(spec.command)}")

    def _slurm_profile(self, stage: str) -> dict[str, Any]:
        cluster = self.config.get("cluster", {})
        node = self._node(stage)
        kind = node.kind
        cpu_material = self.material_workflow and node.command_token in {
            "candidates", "static", "constrained-al", "finalists", "material-validate",
        }
        if kind == "candidates":
            fallback = "bo"
        elif cpu_material:
            fallback = "al" if cluster.get("al") else "bo"
        elif node.command_token == "material-nn":
            fallback = "al" if cluster.get("al") else "bo"
        else:
            fallback = {
                "sample": "bo",
                "audit": "bo",
                "validate": "bo",
                "finalize": "nn",
            }.get(kind, kind)
        if node.command_token == "material-validate":
            # The legacy validate profile is intentionally one LAMMPS-sized
            # job.  Material validation composes structural, static and
            # dynamic batches, so it needs the material AL allocation unless
            # the operator supplies a dedicated override.
            profile = dict(
                cluster.get(
                    "material_validate",
                    cluster.get("al", cluster.get("bo", {})),
                )
            )
        elif node.command_token == "material-nn":
            profile = dict(cluster.get(stage, cluster.get(fallback, {})))
        else:
            profile = dict(
                cluster.get(stage, cluster.get(kind, cluster.get(fallback, {})))
            )
        if not profile and fallback != kind:
            profile = dict(cluster.get(fallback, {}))
        if cpu_material or (
            node.command_token == "material-nn"
        ):
            profile["gpu"] = 0
        elastic_material = node.command_token in {
            "static", "constrained-al", "finalists", "material-validate",
        }
        if elastic_material and self.backend == "slurm":
            parallel = self.config.get("parallel", {})
            omp_threads = max(
                1, int(parallel.get("omp_threads_per_worker", 1))
            )
            total_cores = int(
                profile.get(
                    "cores",
                    int(profile.get("tasks", 1))
                    * int(profile.get("cpus_per_task", 1)),
                )
            )
            usable_cores = total_cores - total_cores % omp_threads
            if usable_cores < omp_threads:
                raise ValueError(
                    f"Machine profile does not allocate one OMP rank for {stage}: "
                    f"cores={total_cores}, omp_threads={omp_threads}"
                )
            # These stages are one Python controller launching many exclusive
            # nested srun steps.  Expose the complete CPU allocation as
            # one-rank slots; the scientific candidate/state budget is derived
            # separately from max_workers, MPI ranks and OMP threads.
            tasks = usable_cores // omp_threads
            profile.update({
                "cores": usable_cores,
                "tasks": tasks,
                "cpus_per_task": omp_threads,
                "distributed_steps": True,
            })
            nodes = max(1, int(profile.get("nodes", 1)))
            if tasks % nodes == 0:
                profile["tasks_per_node"] = tasks // nodes
            else:
                profile.pop("tasks_per_node", None)
        if "env_setup" not in profile:
            profile["env_setup"] = cluster.get("env_setup", [])
        return profile

    def _write_slurm_script(self, spec: StageSpec) -> Path:
        jobs = self.root / "jobs"
        logs = self.root / "logs"
        jobs.mkdir(parents=True, exist_ok=True)
        logs.mkdir(parents=True, exist_ok=True)
        profile = self._slurm_profile(spec.name)
        if bool(profile.get("distributed_steps", False)):
            task_directives = [
                f"#SBATCH --ntasks={profile.get('tasks', profile.get('cores', 1))}",
                f"#SBATCH --cpus-per-task={profile.get('cpus_per_task', 1)}",
            ]
            if profile.get("tasks_per_node"):
                task_directives.append(
                    f"#SBATCH --ntasks-per-node={profile['tasks_per_node']}"
                )
        else:
            task_directives = [
                "#SBATCH --ntasks=1",
                f"#SBATCH --cpus-per-task={profile.get('cores', 1)}",
            ]
        directives = [
            f"#SBATCH --job-name=ffopt_{spec.name}",
            f"#SBATCH --output={logs / (spec.name + '_%j.out')}",
            f"#SBATCH --error={logs / (spec.name + '_%j.err')}",
            f"#SBATCH --nodes={profile.get('nodes', 1)}",
            *task_directives,
            f"#SBATCH --time={profile.get('time', '24:00:00')}",
        ]
        if str(profile.get("partition", "")).strip():
            directives.append(f"#SBATCH --partition={profile['partition']}")
        if str(profile.get("qos", "")).strip():
            directives.append(f"#SBATCH --qos={profile['qos']}")
        if str(profile.get("mem", "")).strip() not in {"", "0", "None"}:
            directives.append(f"#SBATCH --mem={profile['mem']}")
        if int(profile.get("gpu", 0)) > 0:
            directives.append(f"#SBATCH --gres=gpu:{int(profile['gpu'])}")
        env_setup = profile.get("env_setup", [])
        if isinstance(env_setup, str):
            env_lines = env_setup.splitlines()
        else:
            env_lines = [str(line) for line in env_setup]
        lines = [
            "#!/bin/bash",
            *directives,
            "",
            "set -euo pipefail",
            "export PYTHONNOUSERSITE=1",
            "export PYTHONUNBUFFERED=1",
            *env_lines,
            f"cd {shlex.quote(str(self.project.root))}",
            _command_text(spec.command),
            "",
        ]
        path = jobs / f"{spec.name}.sh"
        path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        return path

    @staticmethod
    def _scheduler_query(command: list[str]) -> subprocess.CompletedProcess[str] | None:
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=10,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            return None
        return completed if completed.returncode == 0 else None

    @classmethod
    def _slurm_state(cls, job_id: str) -> str:
        successful_queries = 0
        queued = cls._scheduler_query(
            ["squeue", "-h", "-j", job_id, "-o", "%T"]
        )
        if queued is not None:
            successful_queries += 1
            state = queued.stdout.strip().splitlines()
            if state:
                return state[0].strip().upper()
        accounting = cls._scheduler_query([
            "sacct", "-n", "-X", "-j", job_id, "--format", "State", "-P",
        ])
        if accounting is not None:
            successful_queries += 1
            values = [
                line.split("|")[0].strip().upper()
                for line in accounting.stdout.splitlines()
                if line.strip()
            ]
            if values:
                return values[0].split()[0].rstrip("+")
        return "UNKNOWN" if successful_queries else "QUERY_UNAVAILABLE"

    def _refresh_waiting(self, state: WorkflowState, record: StageRecord) -> str:
        if not record.job_id:
            state.transition(record.name, "failed", message="Missing SLURM job id")
            return "failed"
        scheduler_state = self._slurm_state(record.job_id)
        active = {
            "PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED",
        }
        indeterminate = {"UNKNOWN", "QUERY_UNAVAILABLE"}
        if scheduler_state in active:
            state.transition(
                record.name,
                "waiting",
                message=f"SLURM {record.job_id}: {scheduler_state}",
            )
            return "waiting"
        if scheduler_state in indeterminate:
            if WorkflowState.artifacts_exist(record):
                state.transition(
                    record.name,
                    "completed",
                    message=(
                        f"SLURM {record.job_id}: scheduler query inconclusive; "
                        "all artifacts verified"
                    ),
                )
                return "completed"
            state.transition(
                record.name,
                "waiting",
                message=(
                    f"SLURM {record.job_id}: scheduler query inconclusive; "
                    "preserving job state"
                ),
            )
            return "waiting"
        if WorkflowState.artifacts_exist(record):
            state.transition(
                record.name,
                "completed",
                message=f"SLURM {record.job_id}: {scheduler_state}; artifacts verified",
            )
            return "completed"
        state.transition(
            record.name,
            "failed",
            message=f"SLURM {record.job_id}: {scheduler_state}; expected artifacts missing",
        )
        return "failed"

    def _submit_slurm(self, state: WorkflowState, spec: StageSpec) -> None:
        # A retry may use a new machine-profile snapshot. Refresh the persisted
        # command only after the previous scheduler attempt is terminal.
        state.prepare(
            spec.name,
            spec.signature,
            spec.command,
            spec.output_dir,
            spec.artifacts,
        )
        script = self._write_slurm_script(spec)
        try:
            completed = subprocess.run(
                ["sbatch", "--parsable", str(script)],
                capture_output=True,
                text=True,
                check=False,
            )
        except OSError as exc:
            message = f"Could not execute sbatch: {exc}"
            state.transition(spec.name, "failed", message=message)
            raise RuntimeError(message) from exc
        if completed.returncode:
            message = completed.stderr.strip() or completed.stdout.strip()
            state.transition(spec.name, "failed", message=message)
            raise RuntimeError(f"SLURM submission failed for {spec.name}: {message}")
        match = re.search(r"(\d+)", completed.stdout)
        if not match:
            state.transition(spec.name, "failed", message=completed.stdout.strip())
            raise RuntimeError(f"Could not parse SLURM job id: {completed.stdout!r}")
        job_id = match.group(1)
        self._observed_job_ids.add(job_id)
        state.transition(
            spec.name,
            "waiting",
            job_id=job_id,
            increment_attempt=True,
            message=f"Submitted {script.name}",
        )
        print(f"[{spec.name}] submitted as SLURM job {job_id}")

    def _run_local(self, state: WorkflowState, spec: StageSpec) -> None:
        if self.material_workflow and spec.command_token in {
            "candidates",
            "static",
            "material-nn",
            "constrained-al",
            "finalists",
            "material-validate",
        }:
            # Material-stage publishers atomically install their complete
            # directory and therefore own that path themselves.
            spec.output_dir.parent.mkdir(parents=True, exist_ok=True)
        else:
            # Preserve the legacy molecular runner contract: historical stage
            # entry points are handed an output directory that already exists.
            spec.output_dir.mkdir(parents=True, exist_ok=True)
        state.transition(spec.name, "running", increment_attempt=True)
        print(f"\n[{spec.name}] > {_command_text(spec.command)}", flush=True)
        try:
            completed = subprocess.run(
                spec.command, cwd=self.project.root, check=False
            )
        except BaseException as exc:
            state.transition(spec.name, "failed", message=str(exc))
            raise
        if completed.returncode:
            state.transition(
                spec.name,
                "failed",
                message=f"Command exited with status {completed.returncode}",
            )
            raise RuntimeError(
                f"Pipeline stage {spec.name!r} failed with exit code {completed.returncode}"
            )
        record = state.get(spec.name)
        if record is None or not WorkflowState.artifacts_exist(record):
            missing = [str(path) for path in spec.artifacts if not path.exists()]
            state.transition(spec.name, "failed", message=f"Missing artifacts: {missing}")
            raise RuntimeError(f"Stage {spec.name!r} did not create: {missing}")
        state.transition(spec.name, "completed")
        print(f"[{spec.name}] completed")

    def _terminal_refinement_predecessor(
        self, spec: StageSpec
    ) -> tuple[str, dict[str, Any]] | None:
        if spec.kind != "constrained_al":
            return None
        previous = self._previous_constrained_name(spec.name)
        if previous is None:
            return None
        state_path = self.root / previous / "refinement_state.json"
        if not state_path.is_file():
            return None
        previous_state = load_refinement_state(state_path)
        if previous_state["status"] not in {"converged", "budget_exhausted"}:
            return None
        return previous, previous_state

    def _skip_terminal_refinement_round(
        self,
        state: WorkflowState,
        spec: StageSpec,
        previous_name: str,
        previous_state: Mapping[str, Any],
    ) -> None:
        round_number = self._constrained_round_number(spec.name)
        seed = int(self._stage_settings(spec.name).get("seed", 20260820))
        write_skipped_refinement_stage(
            stage_name=spec.name,
            round_number=round_number,
            previous_stage_name=previous_name,
            previous_dir=self.root / previous_name,
            output_dir=spec.output_dir,
            config_path=self.config_path,
            scientific_config={
                "pipeline_stage_signature": spec.signature,
                "scientific_hash": self.scientific_hash,
            },
            seed=seed,
        )
        missing = [str(path) for path in spec.artifacts if not path.is_file()]
        if missing:
            raise RuntimeError(
                f"Skipped refinement stage {spec.name!r} did not create: {missing}"
            )
        reason = f"terminal predecessor {previous_name}: {previous_state['status']}"
        state.transition(spec.name, "skipped", message=reason)
        print(f"[{spec.name}] skipped ({reason})")

    def _run_once(self, state: WorkflowState, specs: list[StageSpec]) -> str:
        selected = set(self._selected_names())
        for spec in specs:
            record = state.prepare(
                spec.name,
                spec.signature,
                spec.command,
                spec.output_dir,
                spec.artifacts,
            )
            if spec.name not in selected:
                if (
                    self.from_stage
                    and self.stage_names.index(spec.name)
                    < self.stage_names.index(self.from_stage)
                    and not state.is_complete(spec.name, spec.signature)
                ):
                    raise RuntimeError(
                        f"Cannot start from {self.from_stage!r}: upstream stage "
                        f"{spec.name!r} is not complete. Run without --from-stage "
                        "or choose an earlier stage."
                    )
                continue
            if state.is_complete(spec.name, spec.signature):
                if spec.command_token == "constrained-al":
                    valid, reason = validate_material_al_stage_outputs(
                        spec.output_dir
                    )
                    if not valid:
                        state.transition(
                            spec.name,
                            "pending",
                            message=f"Material AL manifest verification failed: {reason}",
                        )
                    else:
                        existing = state.get(spec.name)
                        disposition = (
                            "skipped"
                            if existing and existing.status == "skipped"
                            else "complete"
                        )
                        print(
                            f"[{spec.name}] {disposition}; reusing verified artifacts"
                        )
                        continue
                else:
                    existing = state.get(spec.name)
                    disposition = "skipped" if existing and existing.status == "skipped" else "complete"
                    print(f"[{spec.name}] {disposition}; reusing verified artifacts")
                    continue
            terminal = self._terminal_refinement_predecessor(spec)
            if terminal is not None:
                self._skip_terminal_refinement_round(
                    state, spec, terminal[0], terminal[1]
                )
                continue
            if record.status == "completed":
                state.transition(spec.name, "pending", message="Completed artifacts are missing")
                record = state.get(spec.name)  # type: ignore[assignment]
            if record and record.status == "waiting" and self.backend == "slurm":
                refreshed = self._refresh_waiting(state, record)
                if refreshed == "completed":
                    continue
                if refreshed == "waiting":
                    if record.job_id:
                        self._observed_job_ids.add(record.job_id)
                    return "waiting"
                record = state.get(spec.name)
                message = record.message if record else "unknown scheduler failure"
                failed_job_id = record.job_id if record else None
                if not self.resume or failed_job_id in self._observed_job_ids:
                    raise RuntimeError(
                        f"SLURM stage {spec.name!r} failed: {message}. "
                        "Inspect its log, then run the same command again to resume."
                    )
            if record and record.status in {"failed", "running"} and not self.resume:
                raise RuntimeError(
                    f"Stage {spec.name!r} is {record.status}; rerun the same FFOpt command"
                )
            if self.backend == "slurm":
                self._submit_slurm(state, spec)
                return "waiting"
            self._run_local(state, spec)
        return "completed"

    def run(self) -> str:
        specs = self.build_specs()
        if self.dry_run:
            self._print_plan([spec for spec in specs if spec.name in self._selected_names()])
            return "dry-run"
        self.root.mkdir(parents=True, exist_ok=True)
        with WorkflowState(self.state_path) as state:
            previous = state.metadata()
            previous_version = previous.get("ffopt_version")
            if previous_version is None and previous.get("environment"):
                try:
                    previous_environment = json.loads(
                        Path(previous["environment"]).read_text(encoding="utf-8")
                    )
                    previous_version = previous_environment.get("ffopt_lammps")
                except (OSError, ValueError, TypeError):
                    previous_version = None
            if previous_version not in {None, __version__} and state.list():
                raise RuntimeError(
                    f"This pipeline started with FFOpt {previous_version}, but the "
                    f"current version is {__version__}. Resume with the original "
                    "version or use --new to start an independent run."
                )
            if (
                previous.get("scientific_hash") not in {None, self.scientific_hash}
                and state.list()
            ):
                raise RuntimeError(
                    "The scientific input changed after this pipeline started. "
                    "Use --new to create an independent run instead of mixing checkpoints."
                )
            self._write_provenance()
            state.initialize({
                "project": self.project.name,
                "project_file": str(self.project.path),
                "machine": self.machine,
                "backend": self.backend,
                "run_id": self.run_id,
                "config": str(self.config_path),
                "input_snapshot": str(self.input_snapshot_path),
                "environment": str(self.environment_path),
                "ffopt_version": __version__,
                "config_hash": self.config_hash,
                "scientific_hash": self.scientific_hash,
            })
            while True:
                outcome = self._run_once(state, self.build_specs())
                if outcome != "waiting" or not self.watch:
                    return outcome
                print(f"Waiting {self.poll_seconds}s for the active SLURM stage...")
                time.sleep(self.poll_seconds)


def load_pipeline_status(project: Project, run_id: str) -> dict[str, Any] | None:
    path = project.run_root / "pipelines" / run_id / "state.sqlite"
    if not path.exists():
        return None
    with WorkflowState(path) as state:
        return state.to_dict()
