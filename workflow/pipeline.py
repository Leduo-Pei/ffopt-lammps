"""Restartable BO -> sampling -> NN -> AL workflow orchestration."""

from __future__ import annotations

import os
import re
import shlex
import subprocess
import sys
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .project import Project, compose_config
from .state import StageRecord, WorkflowState, canonical_hash

PIPELINE_STAGES = (
    "bo",
    "sample",
    "nn",
    "al",
    "audit",
    "finalize",
    "validate",
)


@dataclass(frozen=True)
class StageSpec:
    name: str
    signature: str
    command: list[str]
    output_dir: Path
    artifacts: list[Path]


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
    for key in ("executable", "mpiexec", "timeout"):
        lammps.pop(key, None)
    result["lammps"] = lammps
    return result


def _command_text(command: Iterable[str]) -> str:
    values = [str(item) for item in command]
    return subprocess.list2cmdline(values) if os.name == "nt" else shlex.join(values)


class PipelineRunner:
    """Build and execute a deterministic, persisted workflow stage graph."""

    def __init__(
        self,
        *,
        project: Project,
        machine: str,
        config_path: str | Path,
        run_id: str = "default",
        resume: bool = False,
        dry_run: bool = False,
        watch: bool = False,
        poll_seconds: int = 60,
        from_stage: str | None = None,
        until: str | None = None,
    ) -> None:
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", run_id):
            raise ValueError("run_id may contain only letters, numbers, '.', '_' and '-'")
        self.project = project
        self.machine = machine
        self.config_path = Path(config_path).resolve()
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
        self.root = (project.run_root / "pipelines" / run_id).resolve()
        self.state_path = self.root / "state.sqlite"
        pipeline_cfg = project.data.get("pipeline", {})
        configured_stages = pipeline_cfg.get("stages", PIPELINE_STAGES)
        unknown = [name for name in configured_stages if name not in PIPELINE_STAGES]
        if unknown:
            raise ValueError(f"Unknown pipeline stages: {unknown}")
        self.stage_names = self._active_stages(list(configured_stages))
        self._validate_stage_dependencies()
        self.from_stage = from_stage
        self.until = until
        self._validate_bounds()

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

    def _validate_stage_dependencies(self) -> None:
        required = {
            "sample": {"bo"},
            "nn": {"bo"},
            "al": {"bo", "nn"},
            "audit": {"bo"},
            "finalize": {"audit"},
        }
        active = set(self.stage_names)
        for stage, dependencies in required.items():
            if stage not in active:
                continue
            missing = dependencies - active
            if missing:
                raise ValueError(
                    f"Pipeline stage {stage!r} requires {sorted(missing)}"
                )

    def _selected_names(self) -> list[str]:
        start = self.stage_names.index(self.from_stage) if self.from_stage else 0
        stop = self.stage_names.index(self.until) + 1 if self.until else len(self.stage_names)
        return self.stage_names[start:stop]

    def _stage_settings(self, name: str) -> dict[str, Any]:
        if name == "bo":
            return {
                "optimization": self.config.get("optimization", {}),
                "checkpoint": self.config.get("checkpoint", {}),
            }
        if name == "sample":
            return dict(self.project.data.get("stages", {}).get("sample", {}))
        if name == "nn":
            return {"nn": self.config.get("nn", {})}
        if name == "al":
            return {
                "nn": self.config.get("nn", {}),
                "active_learning": self.config.get("active_learning", {}),
            }
        if name == "audit":
            configured = self.project.data.get("pipeline", {}).get("audit")
            return dict(
                configured
                or self.config.get("optimization", {}).get("stability_audit", {})
            )
        return dict(self.project.data.get("pipeline", {}).get(name, {}))

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
        source = self.root / "bo" / "stable_results.csv"
        if not source.exists():
            source = self.root / "bo" / "all_results.csv"
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

    def _stage_command(self, name: str, output: Path) -> list[str]:
        bo_dir = self.root / "bo"
        sample_file = self.root / "sample" / "local_results.csv"
        nn_dir = self.root / "nn"
        if name == "bo":
            command = self._python_module(
                "engine.run", "--config", self.config_path, "--output-dir", output
            )
            if self.resume:
                command.append("--resume")
            return command
        if name == "sample":
            return self._sample_command(output)
        if name == "nn":
            command = self._python_module(
                "engine.nn_surrogate",
                "--config", self.config_path,
                "--bo-dir", bo_dir,
                "--output-dir", output,
            )
            if "sample" in self.stage_names:
                command.extend(["--additional-core-file", sample_file])
            return command
        if name == "al":
            command = self._python_module(
                "engine.active_learning",
                "--config", self.config_path,
                "--bo-dir", bo_dir,
                "--nn-dir", nn_dir,
                "--output-dir", output,
            )
            if "sample" in self.stage_names:
                command.extend(["--additional-core-file", sample_file])
            if self.resume:
                command.append("--resume")
            return command
        if name == "audit":
            source = self._latest_al_data() if "al" in self.stage_names else bo_dir / "all_results.csv"
            top_k, seeds, max_workers = self._audit_settings()
            command = self._python_module(
                "engine.stability_audit",
                "--config", self.config_path,
                "--source", source,
                "--output-dir", output,
                "--top-k", top_k,
                "--seeds", *seeds,
            )
            if max_workers:
                command.extend(["--max-workers", str(max_workers)])
            return command
        if name == "finalize":
            return self._python_module(
                "engine.finalize_al",
                "--config", self.config_path,
                "--audit", self.root / "audit",
                "--output-dir", output,
            )
        if name == "validate":
            parameters = None
            if "finalize" in self.stage_names:
                parameters = self.root / "finalize" / "final_summary.json"
            elif "al" in self.stage_names:
                parameters = self.root / "al" / "nn_round_final" / "nn_optimize_result.json"
            elif "nn" in self.stage_names:
                parameters = self.root / "nn" / "nn_optimize_result.json"
            elif "bo" in self.stage_names:
                parameters = self.root / "bo" / "best_parameters.txt"
            command = self._python_module(
                "engine.validate_final_parameters",
                "--config", self.config_path,
                "--output-dir", output,
                "--overwrite",
            )
            if parameters is not None:
                command.extend(["--parameters", parameters])
            else:
                command.append("--initial")
            return command
        raise ValueError(f"Unsupported stage: {name}")

    @staticmethod
    def _stage_artifacts(name: str, output: Path) -> list[Path]:
        names = {
            "bo": ["all_results.csv"],
            "sample": ["local_results.csv", "local_replicates.csv"],
            "nn": ["forward_nn.pt", "nn_optimize_result.json"],
            "al": ["active_learning_history.json", "nn_round_final/nn_optimize_result.json"],
            "audit": ["stable_results.csv", "stability_replicates.csv"],
            "finalize": ["final_summary.json", "final_parameters.lammps"],
            "validate": ["validation_summary.json", "computed_properties.csv"],
        }[name]
        return [output / item for item in names]

    def build_specs(self) -> list[StageSpec]:
        specs: list[StageSpec] = []
        upstream = "root"
        for name in self.stage_names:
            output = self.root / name
            signature = self._signature(name, upstream)
            specs.append(StageSpec(
                name=name,
                signature=signature,
                command=self._stage_command(name, output),
                output_dir=output,
                artifacts=self._stage_artifacts(name, output),
            ))
            upstream = signature
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
        fallback = {
            "sample": "bo",
            "audit": "bo",
            "validate": "bo",
            "finalize": "nn",
        }.get(stage, stage)
        profile = dict(cluster.get(stage, cluster.get(fallback, {})))
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
            *env_lines,
            f"cd {shlex.quote(str(self.project.root))}",
            _command_text(spec.command),
            "",
        ]
        path = jobs / f"{spec.name}.sh"
        path.write_text("\n".join(lines), encoding="utf-8", newline="\n")
        return path

    @staticmethod
    def _slurm_state(job_id: str) -> str:
        queued = subprocess.run(
            ["squeue", "-h", "-j", job_id, "-o", "%T"],
            capture_output=True,
            text=True,
            check=False,
        )
        state = queued.stdout.strip().splitlines()
        if state:
            return state[0].strip().upper()
        accounting = subprocess.run(
            ["sacct", "-n", "-X", "-j", job_id, "--format", "State", "-P"],
            capture_output=True,
            text=True,
            check=False,
        )
        values = [line.split("|")[0].strip().upper() for line in accounting.stdout.splitlines() if line.strip()]
        return values[0].split()[0].rstrip("+") if values else "UNKNOWN"

    def _refresh_waiting(self, state: WorkflowState, record: StageRecord) -> str:
        if not record.job_id:
            state.transition(record.name, "failed", message="Missing SLURM job id")
            return "failed"
        if WorkflowState.artifacts_exist(record):
            state.transition(
                record.name,
                "completed",
                message=f"Recovered artifacts from SLURM job {record.job_id}",
            )
            return "completed"
        scheduler_state = self._slurm_state(record.job_id)
        active = {"PENDING", "RUNNING", "CONFIGURING", "COMPLETING", "SUSPENDED"}
        if scheduler_state in active:
            state.transition(
                record.name,
                "waiting",
                message=f"SLURM {record.job_id}: {scheduler_state}",
            )
            return "waiting"
        if scheduler_state == "COMPLETED" and WorkflowState.artifacts_exist(record):
            state.transition(record.name, "completed", message=f"SLURM {record.job_id} completed")
            return "completed"
        state.transition(
            record.name,
            "failed",
            message=f"SLURM {record.job_id}: {scheduler_state}; expected artifacts missing",
        )
        return "failed"

    def _submit_slurm(self, state: WorkflowState, spec: StageSpec) -> None:
        script = self._write_slurm_script(spec)
        completed = subprocess.run(
            ["sbatch", str(script)], capture_output=True, text=True, check=False
        )
        if completed.returncode:
            message = completed.stderr.strip() or completed.stdout.strip()
            state.transition(spec.name, "failed", message=message)
            raise RuntimeError(f"SLURM submission failed for {spec.name}: {message}")
        match = re.search(r"(\d+)", completed.stdout)
        if not match:
            state.transition(spec.name, "failed", message=completed.stdout.strip())
            raise RuntimeError(f"Could not parse SLURM job id: {completed.stdout!r}")
        job_id = match.group(1)
        state.transition(
            spec.name,
            "waiting",
            job_id=job_id,
            increment_attempt=True,
            message=f"Submitted {script.name}",
        )
        print(f"[{spec.name}] submitted as SLURM job {job_id}")

    def _run_local(self, state: WorkflowState, spec: StageSpec) -> None:
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
                continue
            if state.is_complete(spec.name, spec.signature):
                print(f"[{spec.name}] complete; reusing verified artifacts")
                continue
            if record.status == "completed":
                state.transition(spec.name, "pending", message="Completed artifacts are missing")
                record = state.get(spec.name)  # type: ignore[assignment]
            if record and record.status == "waiting" and self.backend == "slurm":
                refreshed = self._refresh_waiting(state, record)
                if refreshed == "completed":
                    continue
                if refreshed == "waiting":
                    return "waiting"
                record = state.get(spec.name)
            if record and record.status in {"failed", "running"} and not self.resume:
                raise RuntimeError(
                    f"Stage {spec.name!r} is {record.status}; rerun with --resume"
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
            current_hash = canonical_hash(_serializable_config(self.config))
            scientific_hash = canonical_hash(_scientific_config(self.config))
            if (
                previous.get("scientific_hash") not in {None, scientific_hash}
                and state.list()
            ):
                raise RuntimeError(
                    "The scientific input changed after this pipeline started. "
                    "Use --new to create an independent run instead of mixing checkpoints."
                )
            state.initialize({
                "project": self.project.name,
                "project_file": str(self.project.path),
                "machine": self.machine,
                "backend": self.backend,
                "run_id": self.run_id,
                "config": str(self.config_path),
                "config_hash": current_hash,
                "scientific_hash": scientific_hash,
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
