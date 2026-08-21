"""Unified CLI that delegates to the validated BO/NN/AL implementations."""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import site
import shutil
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from ffopt import __version__

from .input_file import PROPERTY_NAMES
from .lammps_data import inspect_lammps_data
from .machine import (
    available_machine_profiles,
    build_machine_profile,
    load_machine_profile,
    machine_path,
    resolve_executable,
    save_machine_profile,
)
from .project import (
    Project,
    compose_config,
    load_project,
    resolve_path,
)

DEFAULT_PROJECT = "ffopt.in"


def _planned_bo_method(
    config: dict,
    dimensions: int,
    *,
    saasbo_available: bool | None = None,
) -> tuple[str, str, str]:
    """Return configured method, effective method, and any automatic fallback note."""
    optimization = config.get("optimization", {})
    configured = str(optimization.get("method", "auto")).lower()
    if configured != "auto":
        return configured, configured, ""

    threshold = 10 if optimization.get("accuracy_priority", False) else 16
    if dimensions < 7:
        preferred = "gp"
    elif dimensions < threshold:
        preferred = "turbo"
    else:
        preferred = "saasbo"
    if preferred != "saasbo":
        return configured, preferred, ""

    if saasbo_available is None:
        saasbo_available = importlib.util.find_spec("pyro") is not None
    if saasbo_available:
        return configured, preferred, ""
    return configured, "turbo", "SAASBO preferred by dimension, but Pyro is unavailable"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _newest(patterns: Iterable[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(Path.cwd().glob(pattern))
    matches = [path for path in matches if path.exists()]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _slurm_job_summary(job_id: str | None) -> tuple[str, str] | None:
    if not job_id:
        return None
    if shutil.which("squeue") is not None:
        try:
            result = subprocess.run(
                ["squeue", "-h", "-j", str(job_id), "-o", "%T|%R"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
        except (OSError, subprocess.TimeoutExpired):
            result = None
        if result is not None:
            lines = [line for line in result.stdout.splitlines() if line.strip()]
            if result.returncode == 0 and lines:
                state, separator, reason = lines[0].partition("|")
                return state.strip().lower(), reason.strip() if separator else ""

    if shutil.which("sacct") is None:
        return None
    try:
        result = subprocess.run(
            [
                "sacct", "-n", "-X", "-j", str(job_id),
                "--format=State,Reason", "--parsable2",
            ],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if result.returncode != 0 or not lines:
        return None
    state, separator, reason = lines[0].partition("|")
    return state.split("+", 1)[0].strip().lower(), reason.strip() if separator else ""


def _bo_checkpoint_summary(output_dir: str | None) -> str | None:
    if not output_dir:
        return None
    path = Path(output_dir) / "checkpoints" / "latest.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return None
    evaluations = len(data.get("all_results", []))
    round_number = int(data.get("round", 0))
    best = data.get("best_valid_obj")
    best_text = f" best={float(best):.6f}" if best is not None else ""
    return f"checkpoint round={round_number} evaluations={evaluations}{best_text}"


def _latest_sample(project: Project) -> Path | None:
    return _newest([
        f"runs/{project.name}/sample_*/local_results.csv",
        f"runs/{project.name}/legacy/**/local_results.csv",
        "local_sampling_*/local_results.csv",
        "sampling_*/local_results.csv",
    ])


def _stage_defaults(project: Project, stage: str) -> dict:
    return dict(project.data.get("stages", {}).get(stage, {}))


def _common(args: argparse.Namespace) -> tuple[Project, str]:
    selector = getattr(args, "input", None) or args.project
    project = load_project(selector)
    os.chdir(project.root)
    machine = args.machine or project.default_machine
    return project, machine


def _execution_backend(project: Project, machine: str) -> str:
    config = compose_config(project, machine)
    declared = config.get("machine", {}).get("backend")
    if declared:
        return str(declared)
    return "slurm" if machine == "cluster" else "local"


def _free_parameter_count(config: dict) -> int:
    """Count independent force-field dimensions in an expanded config."""
    charge_cfg = config["charge"]
    neutrality = charge_cfg.get("neutrality_constraint", {})
    derived_type = neutrality.get("derive_from_type")
    derived_targets = {
        item["target"] for item in config["pair_params"].get("derived_params", [])
    }
    count = 0
    for atom_type in config["atom_types"]:
        for name, bounds in atom_type["params"].items():
            if name == "charge" and not charge_cfg.get("enabled", False):
                continue
            if (
                name == "charge"
                and neutrality.get("enabled", True)
                and atom_type["type"] == derived_type
            ):
                continue
            key = f"{atom_type['label']}_{name}"
            if key in derived_targets:
                continue
            if isinstance(bounds, dict) and "min" in bounds and "max" in bounds:
                count += 1
    if config["pair_params"].get("mixing_rule") == "none":
        count += 2 * len(config["pair_params"].get("explicit_pairs", []))
    return count


def cmd_status(args: argparse.Namespace) -> None:
    project, machine = _common(args)
    run_root = project.run_root
    from .pipeline import load_pipeline_status

    pipeline = load_pipeline_status(project, args.run_id)
    if pipeline and args.machine is None:
        recorded_machine = pipeline.get("metadata", {}).get("machine")
        if recorded_machine:
            machine = str(recorded_machine)
    print(f"Project: {project.name} [{machine}]\nRun root: {run_root}\n")
    if pipeline:
        print(f"Pipeline: {args.run_id}")
        for stage in pipeline["stages"]:
            job_id = stage.get("job_id")
            scheduler = _slurm_job_summary(job_id)
            display_status = scheduler[0] if scheduler else stage["status"]
            if scheduler:
                location_key = (
                    "nodes" if scheduler[0] in {"running", "completing"} else "reason"
                )
                location = (
                    f" {location_key}={scheduler[1]}" if scheduler[1] else ""
                )
                detail = f"job={job_id}{location}"
            elif job_id:
                detail = f"job={job_id}"
            else:
                detail = stage.get("message") or ""
            artifact_paths = [Path(path) for path in stage.get("artifacts", [])]
            artifact_count = sum(path.exists() for path in artifact_paths)
            artifact_detail = (
                f" artifacts={artifact_count}/{len(artifact_paths)}"
                if artifact_paths else ""
            )
            print(
                f"  {stage['name']:10s} {display_status:10s} "
                f"attempt={stage['attempt']:<2d}{artifact_detail} {detail}"
            )
            if stage["name"] == "bo":
                checkpoint = _bo_checkpoint_summary(stage.get("output_dir"))
                if checkpoint:
                    print(f"  {'':10s} {'':10s} {checkpoint}")
        print(f"\nUse 'ffopt results {project.path} --run-id {args.run_id}' for exact paths.")
        return
    # Compatibility view for historical unmanaged runs.  Managed pipelines
    # return above and therefore never mix their state with glob/mtime guesses.
    nn_defaults = _stage_defaults(project, "nn")
    preferred_bo = nn_defaults.get("bo_dir")
    preferred_bo_path = resolve_path(preferred_bo) if preferred_bo else None
    configured_stable = (
        preferred_bo_path / "stable_results.csv"
        if preferred_bo_path and (preferred_bo_path / "stable_results.csv").exists()
        else None
    )
    configured_checkpoint = (
        preferred_bo_path / "checkpoints" / "latest.json"
        if preferred_bo_path
        and (preferred_bo_path / "checkpoints" / "latest.json").exists()
        else None
    )
    artifacts = [
        ("BO stable", configured_stable or _newest([f"runs/{project.name}/bo_*/stable_results.csv", f"runs/{project.name}/legacy/**/stable_results.csv", "bo_*/stable_results.csv"])),
        ("BO checkpoint", configured_checkpoint or _newest([f"runs/{project.name}/bo_*/checkpoints/latest.json", f"runs/{project.name}/legacy/**/checkpoints/latest.json", "bo_*/checkpoints/latest.json"])),
        ("Sampling", _latest_sample(project)),
        ("NN model", _newest([f"runs/{project.name}/nn_*/forward_nn.pt", f"runs/{project.name}/legacy/**/forward_nn.pt", "nn_output_*/forward_nn.pt"])),
        ("AL history", _newest([f"runs/{project.name}/al_*/active_learning_history.json", f"runs/{project.name}/legacy/**/active_learning_history.json", "active_learning_output_*/active_learning_history.json"])),
        ("Validation", _newest([f"runs/{project.name}/validation_*/validation_summary.json", "outputs/**/validation_summary.json"])),
    ]
    for label, path in artifacts:
        print(f"{label:15s}: {path if path else '-'}")


def cmd_results(args: argparse.Namespace) -> None:
    from .artifacts import collect_pipeline_results, format_pipeline_results

    project = load_project(args.input)
    try:
        report = collect_pipeline_results(project, args.run_id)
    except FileNotFoundError as exc:
        raise SystemExit(str(exc)) from exc
    _include_material_top_results(report)
    print(
        json.dumps(report, indent=2, ensure_ascii=False)
        if args.json else format_pipeline_results(report)
    )


def _include_material_top_results(report: dict) -> None:
    """Expose manifest-declared Top-N products through ``ffopt results``.

    Top-N products are conditional: validation-only and BO+validate workflows
    do not create them, so they cannot be unconditional stage-registry
    artifacts.  A full material validation records their exact paths in its
    persisted summary.  Reading only that state-addressed file keeps results
    discovery deterministic and avoids directory glob/mtime heuristics.
    """

    for stage in report.get("stages", []):
        if stage.get("name") != "validate" or not stage.get("output_dir"):
            continue
        summary_path = Path(stage["output_dir"]) / "validation_summary.json"
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, OSError, UnicodeError, json.JSONDecodeError):
            continue
        top = summary.get("top_parameters_report", {})
        if not isinstance(top, dict) or not top.get("enabled"):
            continue
        artifacts = stage.setdefault("artifacts", [])
        recorded = {
            str(item.get("path"))
            for item in artifacts
            if isinstance(item, dict) and item.get("path")
        }
        for key in ("csv", "json", "markdown", "manifest"):
            raw_path = top.get(key)
            if not raw_path or str(raw_path) in recorded:
                continue
            path = Path(raw_path)
            artifacts.append({
                "path": str(path),
                "exists": path.is_file(),
                "role": f"top_parameters_{key}",
            })
            recorded.add(str(path))


def cmd_logs(args: argparse.Namespace) -> None:
    from .artifacts import find_pipeline_logs, pipeline_root, tail_file

    project = load_project(args.input)
    files = find_pipeline_logs(project, args.run_id, args.stage)
    if not files:
        root = pipeline_root(project, args.run_id)
        qualifier = f" for stage {args.stage!r}" if args.stage else ""
        raise SystemExit(f"No SLURM logs found{qualifier} under {root / 'logs'}")
    if args.stage is None:
        latest_stage = files[-1].name.rsplit("_", 1)[0]
        files = find_pipeline_logs(project, args.run_id, latest_stage)
    for path in files:
        print(f"==> {path} <==")
        if not args.paths:
            content = tail_file(path, args.lines)
            if content:
                print(content)


def _resolve_run_stage_bound(
    project: Project,
    value: str | None,
    *,
    upper: bool,
) -> str | None:
    """Map concise public material stages to concrete executable nodes.

    Material command files deliberately expose ``screen`` and ``al`` while the
    recoverable runtime graph expands them to ``candidates``/``static`` and
    numbered constrained-AL rounds.  CLI bounds must preserve that public
    vocabulary: starting from a macro selects its first node and stopping at a
    macro selects its last node.  Exact expanded names remain valid and are
    checked by :class:`workflow.pipeline.PipelineRunner`.
    """

    if value is None:
        return None
    pipeline = project.data.get("pipeline", {})
    stages = [str(stage) for stage in pipeline.get("stages", [])]
    if value == "screen" and "screen" in stages:
        return "static" if upper else "candidates"
    if value == "al" and "constrained_al" in stages:
        repetitions = pipeline.get("stage_repetitions", {})
        count = int(repetitions.get("constrained_al", 1))
        return f"constrained_al_{count if upper else 1:02d}"
    return value


def _public_workflow(project: Project) -> list[str]:
    """Return the user-authored workflow rather than compiler-only tokens."""

    compilation = project.compilation
    document = getattr(compilation, "document", None)
    if document is not None:
        return [str(stage) for stage in document.workflow]
    return [str(stage) for stage in project.data.get("pipeline", {}).get("stages", [])]


def cmd_run(args: argparse.Namespace) -> None:
    project, machine = _common(args)
    from .pipeline import PipelineRunner

    run_id = args.run_id
    if args.new and run_id == "default":
        run_id = _timestamp()
    runner = PipelineRunner(
        project=project,
        machine=machine,
        run_id=run_id,
        resume=not args.new,
        dry_run=args.dry_run,
        watch=args.watch,
        poll_seconds=args.poll_seconds,
        from_stage=_resolve_run_stage_bound(project, args.from_stage, upper=False),
        until=_resolve_run_stage_bound(project, args.until, upper=True),
    )
    outcome = runner.run()
    if outcome == "waiting":
        print("A SLURM stage is active. Run the same command again to continue, or add --watch.")
    elif outcome == "completed":
        print(f"Pipeline completed: {runner.root}")


def cmd_doctor(args: argparse.Namespace) -> None:
    project, machine = _common(args)
    config = compose_config(project, machine)
    checks: list[tuple[str, bool, str]] = []
    warnings: list[tuple[str, str]] = []
    checks.append(("FFOpt version", True, __version__))
    checks.append(("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0]))
    if site.ENABLE_USER_SITE:
        warnings.append((
            "Python user-site",
            f"enabled ({site.getusersitepackages()}); set PYTHONNOUSERSITE=1 for isolation",
        ))
    required_modules = ["yaml", "numpy", "pandas"]
    stages = set(project.data.get("pipeline", {}).get("stages", []))
    if "bo" in stages:
        required_modules.extend(["scipy", "sklearn", "torch", "botorch", "gpytorch"])
    elif stages & {"nn", "al"}:
        required_modules.extend(["scipy", "sklearn", "torch"])
    for module in required_modules:
        installed = importlib.util.find_spec(module) is not None
        checks.append((f"Python module: {module}", installed, "installed" if installed else "missing"))
    optional_modules: list[tuple[str, str]] = []
    if str(config.get("optimization", {}).get("method", "auto")).lower() == "saasbo":
        optional_modules.append(("pyro", "explicit SAASBO"))
    if str(config.get("nn", {}).get("model", "mlp_ensemble")).lower() == "xgboost":
        optional_modules.append(("xgboost", "NN method xgboost"))
    for module, reason in optional_modules:
        installed = importlib.util.find_spec(module) is not None
        checks.append((
            f"Optional module: {module}",
            installed,
            f"installed ({reason})" if installed else f"missing; required by {reason}",
        ))
    if "bo" in stages:
        configured_bo, effective_bo, bo_note = _planned_bo_method(
            config,
            project.compilation.dimensions,
        )
        checks.append((
            "BO method",
            True,
            f"{configured_bo} -> {effective_bo} "
            f"({project.compilation.dimensions} dimensions)",
        ))
        if bo_note:
            warnings.append(("BO method fallback", bo_note))

    backend = _execution_backend(project, machine)

    if backend == "local":
        executable = resolve_executable(config["lammps"]["executable"])
        checks.append((
            "LAMMPS executable", executable is not None,
            executable or str(config["lammps"]["executable"]),
        ))
        if bool(config.get("parallel", {}).get("use_mpi", False)):
            mpi_value = config["lammps"].get("mpiexec", "mpiexec")
            mpiexec = resolve_executable(mpi_value)
            checks.append(("MPI launcher", mpiexec is not None, mpiexec or str(mpi_value)))
    else:
        sbatch = shutil.which("sbatch")
        checks.append(("SLURM sbatch", sbatch is not None, sbatch or "not on this host"))
        lammps_value = config["lammps"]["executable"]
        executable = resolve_executable(lammps_value)
        checks.append((
            "LAMMPS executable", executable is not None,
            executable or str(lammps_value),
        ))
        mpi_name = str(config["lammps"].get("mpiexec", "mpiexec"))
        mpi_launcher = resolve_executable(mpi_name)
        checks.append(("MPI launcher", mpi_launcher is not None, mpi_launcher or mpi_name))
        step_name = str(config.get("parallel", {}).get("scheduler_launcher", "srun"))
        step_launcher = shutil.which(step_name)
        checks.append((
            "SLURM step launcher", step_launcher is not None,
            step_launcher or step_name,
        ))

    def visit_files(value: object, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child in value.items():
                child_path = f"{key}.{child_key}" if key else str(child_key)
                visit_files(child, child_path)
        elif isinstance(value, list):
            for child in value:
                visit_files(child, key)
        elif isinstance(value, str) and (
            "input" in key.lower()
            or "data_file" in key.lower()
            or value.lower().endswith(".data")
        ):
            path = resolve_path(value)
            checks.append((f"Input: {path.name}", path.exists(), str(path)))

    lammps_files = dict(config.get("lammps", {}))
    if not lammps_files.get("compute_surface", False):
        lammps_files.pop("surf_input", None)
    visit_files(lammps_files, "lammps")
    validation = config.get("validation", {})
    validation_settings = validation.get("property_settings", {})
    validation_evaluators = validation.get("property_evaluators", {})

    def property_requested(name: str) -> bool:
        return bool(
            config.get(name, {}).get("enabled", False)
            or validation_settings.get(name, {}).get("enabled", False)
            or validation_evaluators.get(name, {}).get("enabled", False)
        )

    if property_requested("sublimation"):
        visit_files(config["sublimation"], "sublimation")
    if property_requested("adsorption"):
        visit_files(config["adsorption"], "adsorption")

    from .data_contract import check_data_files

    data_roles: dict[str, str] = {}
    manifest_data = config.get("manifest", {}).get("data_files", {})
    if property_requested("bulk") or property_requested("sublimation"):
        bulk_data = manifest_data.get("bulk")
        if bulk_data:
            data_roles["bulk"] = str(bulk_data)
    if property_requested("sublimation"):
        single_data = (
            config.get("sublimation", {}).get("data_files", {}).get("single")
        )
        if single_data:
            data_roles["single"] = str(single_data)
    if property_requested("adsorption"):
        adsorption_data = config.get("adsorption", {}).get("data_files", {})
        for source, role in (
            ("complex", "complex"),
            ("slab", "slab"),
            ("mol", "molecule"),
        ):
            value = adsorption_data.get(source)
            if value:
                data_roles[role] = str(value)
    if data_roles:
        try:
            data_report = check_data_files(**data_roles)
        except (FileNotFoundError, OSError, TypeError, ValueError) as exc:
            checks.append(("LAMMPS data contract", False, str(exc)))
        else:
            checks.append((
                "LAMMPS data contract",
                data_report.ok,
                f"{len(data_report.files)} files; errors={len(data_report.errors)} "
                f"warnings={len(data_report.warnings)}",
            ))
            grouped_findings: dict[tuple[str, str, str], set[str]] = {}
            for finding in data_report.findings:
                key = (finding.severity, finding.code, finding.message)
                grouped_findings.setdefault(key, set()).update(finding.roles)
            for (severity, code, message), roles in grouped_findings.items():
                role_text = f"[{', '.join(sorted(roles))}] " if roles else ""
                if severity == "ERROR":
                    checks.append((f"Data: {code}", False, role_text + message))
                else:
                    warnings.append((f"Data: {code}", role_text + message))

    needs_torch = bool(stages & {"bo", "nn", "al"})
    if backend == "local" and needs_torch:
        try:
            import torch
            gpu = bool(torch.cuda.is_available())
            requested = str(
                config.get("machine_learning", {}).get("device", "auto")
            ).lower()
            detail = (
                torch.cuda.get_device_name(0)
                if gpu else f"torch {torch.__version__}; CPU fallback"
            )
            checks.append((
                "PyTorch device",
                gpu or requested not in {"cuda", "gpu"},
                detail,
            ))
        except (ImportError, OSError, RuntimeError) as exc:
            checks.append(("PyTorch device", False, str(exc)))
    elif backend == "slurm" and stages & {"nn", "al"}:
        nn_resources = config.get("cluster", {}).get("nn", {})
        gpu_request = int(nn_resources.get("gpu", 0))
        slots = int(nn_resources.get("tasks", 1))
        ranks = int(config.get("parallel", {}).get("cores_per_worker", 1))
        device = f"gpu={gpu_request}" if gpu_request else "CPU; no GPU requested"
        detail = f"{device}; LAMMPS validation slots={slots} x {ranks} MPI ranks"
        checks.append(("SLURM NN resources", True, detail))

    failed = False
    print(
        f"Doctor: {project.name} [{machine}]\n"
        "Runtime config: validated in memory; an immutable snapshot is saved on run\n"
    )
    for label, ok, detail in checks:
        failed |= not ok
        print(f"[{'OK' if ok else 'FAIL'}] {label:24s} {detail}")
    for label, detail in warnings:
        print(f"[WARN] {label:24s} {detail}")
    if failed:
        raise SystemExit(1)


def _add_context(parser: argparse.ArgumentParser, *, positional: bool = True) -> None:
    if positional:
        parser.add_argument(
            "input",
            nargs="?",
            help="FFOpt command input (default: ffopt.in).",
        )
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help=argparse.SUPPRESS,
    )
    parser.add_argument(
        "--machine",
        default=None,
        help="Reusable machine profile; defaults to the project setting.",
    )


def cmd_machine(args: argparse.Namespace) -> None:
    if args.action == "probe":
        from .machine import format_machine_probe, probe_machine_environment
        report = probe_machine_environment(args.partition)
        print(json.dumps(report, indent=2) if args.json else format_machine_probe(report))
        return
    if args.action == "list":
        names = available_machine_profiles()
        if "local" in names:
            print("local [user override]")
        else:
            print("local [built-in, serial]")
        for name in names:
            if name != "local":
                print(f"{name} [user]")
        return
    if args.action == "test":
        from .machine import execute_machine_test
        from .defaults import machine_defaults

        profile = load_machine_profile(args.name)
        if profile is None:
            if args.name == "local":
                profile = machine_defaults(args.name)
            else:
                raise SystemExit(
                    f"Machine profile {args.name!r} is not configured in "
                    f"{machine_path(args.name)}"
                )
        result = execute_machine_test(
            args.name, profile, dry_run=args.dry_run, wait=not args.no_wait
        )
        command = " ".join(str(item) for item in result["command"])
        print(f"Machine test : {args.name}")
        print(f"Backend      : {result['backend']}")
        print(f"Command      : {command}")
        if result.get("script"):
            print(f"SLURM script : {result['script']}")
        if result.get("job_id"):
            print(f"Job ID       : {result['job_id']}")
        print(f"Status       : {result['status']}")
        if result.get("output") and result["status"] == "failed":
            print(result["output"][-4000:])
        if result["status"] == "failed":
            raise SystemExit(2)
        return
    if args.action == "show":
        from .defaults import machine_defaults

        profile = load_machine_profile(args.name)
        if profile is None:
            if args.name == "local":
                profile = machine_defaults(args.name)
            else:
                raise SystemExit(
                    f"Machine profile {args.name!r} is not configured in "
                    f"{machine_path(args.name)}"
                )
        print(json.dumps(
            {key: value for key, value in profile.items() if not key.startswith("_")},
            indent=2,
        ))
        return
    backend = args.backend or "local"
    lammps = args.lammps
    mpi = args.mpi
    workers = args.workers
    ranks = args.ranks or 1
    omp_threads = args.omp_threads or 1
    partition = args.partition
    cores = args.cores
    memory_per_node = args.memory_per_node
    if args.auto:
        from .machine import probe_machine_environment

        probe = probe_machine_environment(args.partition)
        backend = args.backend or probe["backend"]
        if not args.lammps and not probe["lammps"].get("available", False):
            raise SystemExit(
                "LAMMPS was not found on PATH. Pass --lammps /absolute/path/to/lmp."
            )
        lammps = lammps or probe["lammps"]["path"]
        recommendation = probe.get("recommendation")
        if backend == "slurm":
            if recommendation is None:
                raise SystemExit(
                    "Automatic SLURM configuration needs a readable sinfo result. "
                    "Pass --partition or configure the profile explicitly."
                )
            partition = partition or recommendation["partition"]
            ranks = args.ranks or recommendation["ranks"]
            omp_threads = args.omp_threads or recommendation["omp_threads"]
            workers = workers or recommendation["workers"] * args.nodes
            cores = cores or recommendation["cores"] * args.nodes
            memory_per_node = memory_per_node or recommendation["memory_per_node"]
        else:
            workers = workers or max(1, probe["logical_cpus"] // (ranks * omp_threads))
        if ranks > 1 and not args.mpi and not probe["mpi"].get("available", False):
            raise SystemExit(
                "MPI was not found on PATH. Pass --mpi /absolute/path/to/mpirun."
            )
        mpi = mpi or probe["mpi"]["path"]
        print("Using conservative values from 'ffopt machine probe'.")
    profile = build_machine_profile(
        name=args.name,
        backend=backend,
        lammps=lammps,
        mpi=mpi,
        workers=workers,
        ranks=ranks,
        omp_threads=omp_threads,
        timeout=args.timeout,
        partition=partition,
        qos=args.qos,
        cores=cores,
        nodes=args.nodes,
        gpus=args.gpus,
        walltime=args.walltime,
        memory_per_node=memory_per_node,
    )
    path = save_machine_profile(args.name, profile, overwrite=args.force)
    print(f"Machine profile saved: {path}")


def cmd_inspect(args: argparse.Namespace) -> None:
    summary = inspect_lammps_data(args.data_file)
    if args.json:
        print(json.dumps(summary.to_dict(), indent=2, ensure_ascii=False))
        return
    print(f"File          : {summary.path}")
    print(f"Title         : {summary.title}")
    print(f"Atom style    : {summary.atom_style or 'not declared'}")
    print(f"Atoms         : {summary.declared_counts.get('atoms', '?')}")
    print(f"Atom types    : {len(summary.atom_types)}")
    print(f"Molecules     : {summary.molecule_count if summary.molecule_count is not None else '?'}")
    charge = "?" if summary.total_charge is None else f"{summary.total_charge:.12g} e"
    print(f"Total charge  : {charge}\n")
    print("Type  Label             Count  Mass          Pair coefficients        Charge range")
    for item in summary.atom_types:
        mass = "-" if item.mass is None else f"{item.mass:.8g}"
        pair = ", ".join(f"{value:.8g}" for value in item.pair_coefficients) or "-"
        if item.charge_min is None:
            charge_range = "-"
        elif item.charge_min == item.charge_max:
            charge_range = f"{item.charge_min:.8g}"
        else:
            charge_range = f"{item.charge_min:.8g} .. {item.charge_max:.8g}"
        print(
            f"{item.type_id:4d}  {item.label:16.16s}  {item.atom_count:5d}  "
            f"{mass:12s}  {pair:23.23s}  {charge_range}"
        )


def cmd_data(args: argparse.Namespace) -> None:
    from .data_contract import check_data_files, format_data_report

    try:
        report = check_data_files(
            bulk=args.bulk,
            single=args.single,
            complex=args.complex,
            slab=args.slab,
            molecule=args.molecule,
        )
    except (FileNotFoundError, TypeError, ValueError) as exc:
        raise SystemExit(f"Data check failed: {exc}") from exc
    if args.json:
        print(json.dumps(report.to_dict(), indent=2, ensure_ascii=False))
    else:
        print(format_data_report(report))
    if not report.ok or (args.strict and report.warnings):
        raise SystemExit(2)


def cmd_check(args: argparse.Namespace) -> None:
    """Parse and semantically validate one public command input."""
    project = load_project(args.input)
    if project.runtime_config is None:
        raise SystemExit("ffopt check expects a .in/.inp command file")
    compilation = project.compilation
    print(f"OK: {project.path}")
    print(f"Project               : {project.name}")
    print(f"Free dimensions       : {compilation.dimensions}")
    print(f"Fitted properties     : {', '.join(compilation.fitted_properties) or '-'}")
    print(f"Validation-only       : {', '.join(compilation.validation_properties) or '-'}")
    print(f"Workflow              : {' -> '.join(_public_workflow(project))}")


def cmd_explain(args: argparse.Namespace) -> None:
    """Explain the expanded parameter/property plan without writing a config."""
    project = load_project(args.input)
    if project.runtime_config is None:
        raise SystemExit("ffopt explain expects a .in/.inp command file")
    compilation = project.compilation
    config = project.runtime_config
    free = []
    fixed = []
    for atom_type in config["atom_types"]:
        for name, value in atom_type["params"].items():
            label = f"{atom_type['label']}_{name}"
            (free if isinstance(value, dict) else fixed).append(label)
    neutrality = config["charge"]["neutrality_constraint"]
    derived = "-"
    if neutrality.get("enabled"):
        type_id = neutrality.get("derive_from_type")
        atom_type = next(item for item in config["atom_types"] if item["type"] == type_id)
        derived = f"{atom_type['label']}_charge"
        free = [name for name in free if name != derived]
    print(f"Input                 : {project.path}")
    print(f"Project               : {project.name}")
    print(f"Workflow              : {' -> '.join(_public_workflow(project))}")
    print(f"Independent dimensions: {compilation.dimensions}")
    print(f"Free parameters       : {len(free)}")
    print(f"Fixed parameters      : {len(fixed)}")
    print(f"Derived parameter     : {derived}")
    mixing = config["pair_params"]["mixing_rule"]
    mixing_detail = (
        "epsilon geometric, sigma geometric"
        if mixing == "geometric"
        else "epsilon geometric, sigma arithmetic"
    )
    print(f"LAMMPS mixing rule    : {mixing} ({mixing_detail})")
    print("\nProperties:")
    for prop in compilation.document.properties:
        role = "fit + final validation" if prop.fitted else "final validation only"
        protocol = prop.settings.get("protocol")
        if prop.name == "bulk":
            protocol = "minimize + NPT"
        elif prop.name == "sublimation":
            protocol = "bulk NPT PE + single minimize PE"
        print(f"  {prop.name:14s} {role:24s} protocol={protocol or 'module default'}")
        for target in prop.targets:
            target_name = (
                "sublimation_enthalpy"
                if target.name == "esub_proxy"
                else target.name
            )
            print(
                f"    target {target_name}={target.value:g} "
                f"{target.unit} weight={target.weight:g} "
                f"tolerance={target.tolerance if target.tolerance is not None else '-'}"
            )
    sample = project.data.get("stages", {}).get("sample", {})
    stages = set(project.data["pipeline"]["stages"])
    if "bo" in stages:
        optimization = config["optimization"]
        early_stop = optimization.get("early_stop", {})
        audit = optimization.get("stability_audit", {})
        warm_start_count = int(
            any(
                isinstance(spec, dict) and "init" in spec
                for atom_type in config.get("atom_types", [])
                for spec in atom_type.get("params", {}).values()
            )
            or bool(optimization.get("seed_params"))
        )
        initial_lhs = int(optimization.get("n_initial", 0))
        rounds = int(optimization.get("n_bo_iterations", 0))
        batch_size = int(optimization.get("batch_size", 0))
        total_evaluations = initial_lhs + warm_start_count + rounds * batch_size
        stability_top_k = int(audit.get("top_k", 0))
        stability_seeds = list(audit.get("seeds", []))
        stability_evaluations = (
            stability_top_k * len(stability_seeds)
            if audit.get("enabled")
            else 0
        )
        configured_bo, effective_bo, bo_note = _planned_bo_method(
            config,
            compilation.dimensions,
        )
        method_display = (
            configured_bo
            if configured_bo == effective_bo
            else f"{configured_bo}->{effective_bo}"
        )
        print(
            "\nBO                    : "
            f"method={method_display} "
            f"lhs={initial_lhs} "
            f"warm_start={warm_start_count} "
            f"initial_total={initial_lhs + warm_start_count} "
            f"rounds={rounds} "
            f"batch={batch_size} "
            f"search_evaluations={total_evaluations} "
            f"seed={optimization.get('random_seed')}"
        )
        if bo_note:
            print(f"BO method note         : {bo_note}")
        print(
            "BO stopping/audit      : "
            f"patience={early_stop.get('patience')} "
            f"min_improvement={early_stop.get('min_improvement')} "
            f"audit={'on' if audit.get('enabled') else 'off'} "
            f"top_k={stability_top_k} seeds={stability_seeds} "
            f"audit_max_evaluations={stability_evaluations} "
            f"stage_max_evaluations={total_evaluations + stability_evaluations}"
        )
    if "sample" in stages:
        seeds = sample.get("seeds", [])
        points = int(sample.get("n_points", 0))
        print(
            f"Sampling              : {points} points x {len(seeds)} seeds; "
            f"centers={sample.get('elite_centers')} radii={sample.get('radii')}"
        )
    if "nn" in stages:
        nn = config["nn"]
        print(
            "ANN                   : "
            f"method={nn.get('model')} ensemble={nn.get('ensemble_size')} "
            f"layers={nn.get('hidden_layers')} epochs={nn.get('max_epochs')}"
        )
    if "al" in stages:
        active_learning = config["active_learning"]
        print(
            "Active learning       : "
            f"rounds={active_learning.get('n_rounds')} "
            f"LAMMPS candidates/round={active_learning.get('n_candidates_per_round')} "
            f"domain={active_learning.get('sampling_domain')}"
        )
    if "audit" in stages:
        final_audit = project.data.get("pipeline", {}).get("audit", {})
        final_seeds = final_audit.get("seeds", [])
        print(
            "Final robust audit    : "
            f"top_k={final_audit.get('top_k')} x {len(final_seeds)} seeds; "
            "rank=mean+std"
        )
    if "validate" in stages:
        validation = config["validation"]
        print(
            "Final acceptance      : "
            f"tolerances={'required' if validation.get('require_tolerances') else 'reported'} "
            f"objective_max={validation.get('objective_max', '-')} "
            f"max_error_percent={validation.get('max_error_percent', '-')}"
        )


def cmd_plugins(args: argparse.Namespace) -> None:
    from engine.property_evaluators import PROPERTY_EVALUATORS

    descriptions = PROPERTY_EVALUATORS.describe()
    for item in descriptions:
        item["ffopt_input"] = item["name"] in PROPERTY_NAMES
    if args.json:
        print(json.dumps(descriptions, indent=2))
        return
    print("Property evaluators (ffopt.in indicates public input support):")
    for item in descriptions:
        dependencies = ",".join(item["dependencies"]) or "-"
        provides = ",".join(item["provides"]) or "dynamic"
        print(
            f"  {item['name']:14s} ffopt.in={'yes' if item['ffopt_input'] else 'no ':3s} "
            f"dependencies={dependencies:12s} "
            f"provides={provides}"
        )
        print(f"  {'':14s} source={item['source']}  class={item['class']}")


def cmd_init(args: argparse.Namespace) -> None:
    from .scaffold import create_project, parse_target_spec

    try:
        targets = [parse_target_spec(value) for value in args.target]
        result = create_project(
            name=args.name,
            data_file=args.data_file,
            single_data=args.single_data,
            complex_data=args.complex_data,
            slab_data=args.slab_data,
            molecule_data=args.molecule_data,
            project_type=args.project_type,
            metal_label=args.metal_label,
            destination=args.destination or args.name,
            targets=targets,
            mode=args.mode,
            cells=tuple(args.cells),
            mixing_rule=args.mixing_rule,
            derive_charge=args.derive_charge,
            epsilon_scale=tuple(args.epsilon_scale),
            sigma_scale=tuple(args.sigma_scale),
            charge_window=args.charge_window,
            charge_limit=args.charge_limit,
            force=args.force,
        )
    except (FileNotFoundError, FileExistsError, TypeError, ValueError) as exc:
        raise SystemExit(f"Project initialization failed: {exc}") from exc

    print(f"Project created : {result.root}")
    print(f"Project file    : {result.project_file}")
    print(f"Atom types      : {result.atom_types}")
    print(f"Free dimensions : {result.dimensions}")
    public_targets = {
        "esub_proxy": "sublimation_enthalpy",
        "ead": "adsorption_energy",
        "gamma_ang": "gamma",
    }
    print(
        "Targets         : "
        + ", ".join(public_targets.get(name, name) for name in result.targets)
    )
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    print("\nNext:")
    print(f"  cd {result.root}")
    print("  ffopt check ffopt.in")
    print("  ffopt explain ffopt.in")
    print("  ffopt doctor ffopt.in --machine local")
    print("  ffopt run ffopt.in --machine local --dry-run")
    print("  ffopt run ffopt.in --machine local")


def cmd_self_test(args: argparse.Namespace) -> None:
    """Prepare and run the packaged BTAH scientific acceptance workflow."""
    from .defaults import machine_defaults
    from .machine import execute_machine_test
    from .pipeline import PipelineRunner
    from .self_test import prepare_self_test_project

    destination = (
        Path(args.workdir).expanduser()
        if args.workdir
        else Path.cwd() / "ffopt-self-test" / args.machine
    )
    try:
        prepared = prepare_self_test_project(destination)
    except FileExistsError as exc:
        raise SystemExit(str(exc)) from exc
    project = load_project(prepared.input_file)
    profile = load_machine_profile(args.machine)
    if profile is None:
        if args.machine == "local":
            profile = machine_defaults(args.machine)
        else:
            raise SystemExit(
                f"Machine profile {args.machine!r} is not configured. Run "
                "'ffopt machine configure' first."
            )

    print(f"Self-test project : {prepared.root}")
    print(f"Machine profile   : {args.machine}")
    print("Acceptance        : objective <= 0.03 and every fitted error <= 3%")
    if args.prepare_only:
        print(f"Prepared only. Review {prepared.input_file}")
        return

    if not args.skip_machine_test and not args.dry_run:
        print("\n[preflight] Running the LAMMPS/MPI machine test...")
        result = execute_machine_test(args.machine, profile, wait=True)
        print(f"Machine test      : {result['status']}")
        if result["status"] != "passed":
            detail = str(
                result.get("output") or result.get("submission_output") or ""
            )
            if detail:
                print(detail[-4000:])
            raise SystemExit(2)

    runner = PipelineRunner(
        project=project,
        machine=args.machine,
        run_id=args.run_id,
        resume=True,
        dry_run=args.dry_run,
        watch=args.watch,
        poll_seconds=args.poll_seconds,
    )
    outcome = runner.run()
    if outcome == "waiting":
        print(
            "A SLURM stage is active. Repeat this exact self-test command to "
            "continue, or use --watch."
        )
    elif outcome == "completed":
        print(f"Scientific acceptance PASS: {runner.root}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffopt",
        description=(
            "One restartable command for force-field optimization and physical validation."
        ),
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {__version__}",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Show a full Python traceback for troubleshooting.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser(
        "init", help="Create a portable molecular project from a LAMMPS data file."
    )
    init.add_argument("name", help="Project and system name.")
    init.add_argument(
        "--bulk-data", dest="data_file",
        help="Bulk molecular-crystal data file.",
    )
    init.add_argument("--data-file", dest="data_file", help=argparse.SUPPRESS)
    init.add_argument(
        "--project-type",
        choices=["auto", "molecular-crystal", "adsorption", "crystal-adsorption"],
        default="auto",
        help="Project template; auto infers it from the supplied data files.",
    )
    init.add_argument(
        "--single-data",
        help="Isolated molecule data file for the sublimation-enthalpy estimate.",
    )
    init.add_argument("--destination", help="Output directory; defaults to NAME.")
    init.add_argument(
        "--target", action="append", default=[],
        help=(
            "Repeat NAME=VALUE[,WEIGHT[,UNIT[,TOLERANCE]]], "
            "e.g. density=1.25,1.0,g/cm3,0.03."
        ),
    )
    init.add_argument("--complex-data", help="Adsorbate+slab LAMMPS data file.")
    init.add_argument("--slab-data", help="Clean slab LAMMPS data file.")
    init.add_argument("--molecule-data", help="Isolated adsorbate LAMMPS data file.")
    init.add_argument("--metal-label", default="Au", help="Metal type label in adsorption data.")
    init.add_argument(
        "--mode", choices=["full", "fix_sigma", "charge_only", "lj_only"],
        default="full",
        help=(
            "Free parameter families: all; epsilon+charge; charge only; or "
            "epsilon+sigma (default: full)."
        ),
    )
    init.add_argument(
        "--cells", type=int, nargs=3, metavar=("NX", "NY", "NZ"),
        default=[1, 1, 1],
        help=(
            "Primitive-cell repeats already present in the bulk data file "
            "(default: 1 1 1)."
        ),
    )
    init.add_argument(
        "--mixing-rule", choices=["geometric", "arithmetic"], default="geometric",
        help=(
            "LAMMPS unlike-pair mixing rule; epsilon remains geometric while "
            "this choice controls sigma (default: geometric)."
        ),
    )
    init.add_argument(
        "--derive-charge", default="auto",
        help=(
            "auto, none, or one atom-type ID/label whose charge is recovered "
            "from exact total neutrality (default: auto)."
        ),
    )
    init.add_argument(
        "--epsilon-scale", type=float, nargs=2, default=[0.5, 2.0],
        metavar=("LOW", "HIGH"),
        help=(
            "Multipliers around each initial epsilon for free epsilon parameters "
            "(default: 0.5 2.0)."
        ),
    )
    init.add_argument(
        "--sigma-scale", type=float, nargs=2, default=[0.85, 1.15],
        metavar=("LOW", "HIGH"),
        help=(
            "Multipliers around each initial sigma for free sigma parameters "
            "(default: 0.85 1.15)."
        ),
    )
    init.add_argument(
        "--charge-window", type=float, default=0.30, metavar="DELTA",
        help=(
            "Symmetric +/- charge search window around each initial charge "
            "(default: 0.30)."
        ),
    )
    init.add_argument(
        "--charge-limit", type=float, default=2.0, metavar="ABS_MAX",
        help="Absolute safety limit applied to every resolved charge (default: 2.0).",
    )
    init.add_argument(
        "--force", action="store_true",
        help="Allow generated files in an existing destination to be replaced.",
    )
    init.set_defaults(function=cmd_init)

    self_test = sub.add_parser(
        "self-test",
        help="Run the packaged BTAH machine and scientific acceptance test.",
    )
    self_test.add_argument(
        "--machine", default="local",
        help="Machine profile to test (default: built-in local).",
    )
    self_test.add_argument(
        "--workdir",
        help="Acceptance project directory; an identical existing project resumes.",
    )
    self_test.add_argument(
        "--run-id", default="acceptance",
        help="Pipeline identifier below the work directory (default: acceptance).",
    )
    self_test.add_argument(
        "--watch", action="store_true",
        help="Poll SLURM and submit subsequent stages until acceptance finishes.",
    )
    self_test.add_argument(
        "--poll-seconds", type=int, default=60,
        help="SLURM polling interval used with --watch (default: 60).",
    )
    self_test.add_argument(
        "--prepare-only", action="store_true",
        help="Install and verify the packaged project without executing it.",
    )
    self_test.add_argument(
        "--skip-machine-test", action="store_true",
        help="Skip the fast LAMMPS/MPI preflight when it has already passed.",
    )
    self_test.add_argument(
        "--dry-run", action="store_true",
        help="Print the scientific stage commands without submitting jobs.",
    )
    self_test.set_defaults(function=cmd_self_test)

    inspect = sub.add_parser("inspect", help="Inspect atom types and parameters in a LAMMPS data file.")
    inspect.add_argument("data_file", help="LAMMPS data file to inspect.")
    inspect.add_argument("--json", action="store_true", help="Emit a machine-readable summary.")
    inspect.set_defaults(function=cmd_inspect)
    data = sub.add_parser(
        "data", help="Validate LAMMPS data files and cross-file compatibility."
    )
    data.add_argument("action", choices=["check"], help="Data operation to perform.")
    data.add_argument("--bulk", help="Periodic molecular-crystal data file.")
    data.add_argument("--single", help="Isolated molecule used with --bulk.")
    data.add_argument("--complex", help="Adsorbate+slab data file.")
    data.add_argument("--slab", help="Clean slab data file.")
    data.add_argument("--molecule", help="Isolated adsorbate data file.")
    data.add_argument("--strict", action="store_true", help="Treat warnings as failures.")
    data.add_argument("--json", action="store_true", help="Emit a machine-readable report.")
    data.set_defaults(function=cmd_data)
    check = sub.add_parser("check", help="Validate one ffopt.in file without running LAMMPS.")
    check.add_argument("input", help="FFOpt command input to validate.")
    check.set_defaults(function=cmd_check)
    explain = sub.add_parser("explain", help="Explain parameters, properties, and workflow in ffopt.in.")
    explain.add_argument("input", help="FFOpt command input to expand and explain.")
    explain.set_defaults(function=cmd_explain)
    plugins = sub.add_parser("plugins", help="List built-in and installed property evaluators.")
    plugins.add_argument("--json", action="store_true", help="Emit machine-readable metadata.")
    plugins.set_defaults(function=cmd_plugins)
    machine = sub.add_parser("machine", help="Configure reusable local or SLURM execution profiles.")
    machine.add_argument(
        "action", choices=["configure", "show", "list", "probe", "test"],
        help="Configure, inspect, discover, or test a reusable machine profile.",
    )
    machine.add_argument(
        "--name", default="local", help="Profile name (default: local)."
    )
    machine.add_argument(
        "--backend", choices=["local", "slurm"],
        help="Execution backend when configuring a profile.",
    )
    machine.add_argument("--lammps", help="LAMMPS executable; auto-detected when omitted.")
    machine.add_argument("--mpi", help="MPI launcher; auto-detected when omitted.")
    machine.add_argument(
        "--workers",
        type=int,
        help="Concurrent, independent LAMMPS evaluations across the allocation.",
    )
    machine.add_argument(
        "--mpi-ranks", dest="ranks", type=int,
        help="MPI ranks used by each independent LAMMPS evaluation.",
    )
    machine.add_argument("--ranks", dest="ranks", type=int, help=argparse.SUPPRESS)
    machine.add_argument(
        "--omp-threads", type=int,
        help="OpenMP threads used by each MPI rank (normally 1).",
    )
    machine.add_argument(
        "--timeout", type=int, default=21600,
        help="Per-evaluation timeout in seconds (default: 21600).",
    )
    machine.add_argument("--partition", help="SLURM partition name.")
    machine.add_argument("--qos", help="Optional SLURM quality-of-service name.")
    machine.add_argument(
        "--total-cores", dest="cores", type=int,
        help="Total CPU cores requested across all SLURM nodes.",
    )
    machine.add_argument("--cores", dest="cores", type=int, help=argparse.SUPPRESS)
    machine.add_argument(
        "--nodes", type=int, default=1,
        help="SLURM nodes used by parallel LAMMPS stages (default: 1).",
    )
    machine.add_argument(
        "--gpus", type=int, default=0,
        help="GPUs requested by NN/AL stages (default: 0).",
    )
    machine.add_argument(
        "--walltime", default="24:00:00",
        help="SLURM time limit, for example 06:00:00 or 14-00:00:00.",
    )
    machine.add_argument(
        "--memory-per-node",
        help=(
            "SLURM memory requested on each node, for example 64G. "
            "When omitted, the cluster default is used."
        ),
    )
    machine.add_argument(
        "--force", action="store_true",
        help="Replace the named profile; other profiles are preserved.",
    )
    machine.add_argument(
        "--auto", action="store_true",
        help="Configure conservative defaults from the current host and sinfo.",
    )
    machine.add_argument("--json", action="store_true", help="Emit machine details as JSON.")
    machine.add_argument(
        "--dry-run", action="store_true",
        help="Prepare and print a machine test without submitting or running it.",
    )
    machine.add_argument(
        "--no-wait", action="store_true",
        help="Submit a SLURM machine test and return without waiting for completion.",
    )
    machine.set_defaults(function=cmd_machine)
    for name, function, help_text in (
        ("status", cmd_status, "Show resumable pipeline state and artifacts."),
        ("doctor", cmd_doctor, "Check Python, LAMMPS, MPI, data, and device setup."),
    ):
        child = sub.add_parser(name, help=help_text)
        _add_context(child)
        if name == "status":
            child.add_argument(
                "--run-id", default="default",
                help="Named pipeline run to display (default: default).",
            )
        child.set_defaults(function=function)

    results = sub.add_parser(
        "results", help="Show exact outputs and expected artifacts for one pipeline run."
    )
    results.add_argument(
        "input", nargs="?", default=DEFAULT_PROJECT,
        help="FFOpt command input (default: ffopt.in).",
    )
    results.add_argument(
        "--run-id", default="default", help="Pipeline identifier (default: default)."
    )
    results.add_argument("--json", action="store_true", help="Emit machine-readable paths.")
    results.set_defaults(function=cmd_results)

    logs = sub.add_parser(
        "logs", help="Show the latest SLURM stdout/stderr for a pipeline stage."
    )
    logs.add_argument(
        "input", nargs="?", default=DEFAULT_PROJECT,
        help="FFOpt command input (default: ffopt.in).",
    )
    logs.add_argument(
        "--run-id", default="default", help="Pipeline identifier (default: default)."
    )
    logs.add_argument(
        "--stage",
        help=(
            "Concrete stage to inspect (for example constrained_al_02); "
            "defaults to the most recent logged stage."
        ),
    )
    logs.add_argument(
        "--lines", type=int, default=80, help="Lines read from each log (default: 80)."
    )
    logs.add_argument("--paths", action="store_true", help="List log paths without file content.")
    logs.set_defaults(function=cmd_logs)

    run = sub.add_parser(
        "run",
        help=(
            "Run the configured molecular or material workflow as one "
            "restartable local/SLURM pipeline."
        ),
    )
    _add_context(run)
    run.add_argument(
        "--run-id", default="default", help="Pipeline identifier (default: default)."
    )
    run.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)
    run.add_argument(
        "--new", action="store_true",
        help="Start an independent timestamped run instead of auto-resuming.",
    )
    run.add_argument(
        "--dry-run", action="store_true",
        help="Compile and print stage commands without executing or submitting them.",
    )
    run.add_argument(
        "--watch", action="store_true",
        help="On SLURM, keep polling and submit each next stage automatically.",
    )
    run.add_argument(
        "--poll-seconds", type=int, default=60,
        help="SLURM polling interval used with --watch (default: 60).",
    )
    run.add_argument(
        "--from-stage",
        help=(
            "Begin at a public stage (including screen/al) or a concrete expanded "
            "stage; completed upstream artifacts are required."
        ),
    )
    run.add_argument(
        "--until",
        help=(
            "Stop after a public stage (including screen/al) or a concrete expanded "
            "stage while retaining the configured full workflow."
        ),
    )
    run.set_defaults(function=cmd_run)
    return parser


def main() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(line_buffering=True)
        except (AttributeError, ValueError):
            pass
    args = build_parser().parse_args()
    try:
        args.function(args)
    except BrokenPipeError:
        # Commands are routinely piped to head/tail on login nodes.  Replace
        # stdout before interpreter shutdown so its final flush stays quiet.
        try:
            descriptor = os.open(os.devnull, os.O_WRONLY)
            os.dup2(descriptor, sys.stdout.fileno())
            os.close(descriptor)
        except (AttributeError, OSError, ValueError):
            pass
        raise SystemExit(0) from None
    except KeyboardInterrupt:
        print(
            "\nFFOpt interrupted. Repeat the same run command to resume from saved state.",
            file=sys.stderr,
        )
        raise SystemExit(130) from None
    except (
        FileNotFoundError,
        FileExistsError,
        OSError,
        RuntimeError,
        TypeError,
        ValueError,
    ) as exc:
        if getattr(args, "debug", False) or os.environ.get("FFOPT_DEBUG") == "1":
            raise
        print(f"FFOpt error: {exc}", file=sys.stderr)
        print("Rerun with 'ffopt --debug ...' to show the full traceback.", file=sys.stderr)
        raise SystemExit(2) from None
