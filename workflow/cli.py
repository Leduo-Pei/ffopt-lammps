"""Unified CLI that delegates to the validated BO/NN/AL implementations."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import os
import shutil
import subprocess
import sys
from collections.abc import Iterable
from datetime import datetime, timezone
from pathlib import Path

from .lammps_data import inspect_lammps_data
from .machine import (
    available_machine_profiles,
    build_machine_profile,
    load_machine_profile,
    machine_path,
    save_machine_profile,
)
from .project import (
    SOURCE_ROOT,
    Project,
    compose_config,
    load_project,
    resolve_path,
    write_generated_config,
)

DEFAULT_PROJECT = "ffopt.in"


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")


def _run(command: list[object]) -> None:
    command = [str(item) for item in command]
    print("\n> " + subprocess.list2cmdline(command))
    completed = subprocess.run(command, cwd=Path.cwd(), check=False)
    if completed.returncode:
        raise SystemExit(completed.returncode)


def _preview(command: list[object]) -> None:
    print("DRY RUN - command not executed:\n> " + subprocess.list2cmdline([str(item) for item in command]))


def _python(script: str, *args: object) -> list[object]:
    return [sys.executable, SOURCE_ROOT / script, *args]


def _module(module: str, *args: object) -> list[object]:
    return [sys.executable, "-m", module, *args]


def _newest(patterns: Iterable[str]) -> Path | None:
    matches: list[Path] = []
    for pattern in patterns:
        matches.extend(Path.cwd().glob(pattern))
    matches = [path for path in matches if path.exists()]
    return max(matches, key=lambda path: path.stat().st_mtime) if matches else None


def _latest_bo(project: Project, stable: bool = True) -> Path | None:
    names = ["stable_results.csv", "all_results.csv"] if stable else ["all_results.csv"]
    patterns: list[str] = []
    for name in names:
        patterns.extend([
            f"runs/{project.name}/bo_*/{name}",
            f"runs/{project.name}/legacy/**/{name}",
            f"bo_*/{name}",
        ])
    result = _newest(patterns)
    return result.parent if result else None


def _latest_nn(project: Project) -> Path | None:
    result = _newest([
        f"runs/{project.name}/nn_*/forward_nn.pt",
        f"runs/{project.name}/nn_*/hybrid_manifest.json",
        f"runs/{project.name}/legacy/**/forward_nn.pt",
        f"runs/{project.name}/legacy/**/hybrid_manifest.json",
        "nn_output_*/forward_nn.pt",
    ])
    return result.parent if result else None


def _latest_sample(project: Project) -> Path | None:
    return _newest([
        f"runs/{project.name}/sample_*/local_results.csv",
        f"runs/{project.name}/legacy/**/local_results.csv",
        "local_sampling_*/local_results.csv",
        "sampling_*/local_results.csv",
    ])


def _stage_defaults(project: Project, stage: str) -> dict:
    return dict(project.data.get("stages", {}).get(stage, {}))


def _validate_nn_inputs(config: dict, bo_dir: Path, core_files: list[Path]) -> None:
    """Fail before SLURM submission when the NN data contract is invalid."""
    raw_path = bo_dir / "all_results.csv"
    if not raw_path.exists():
        raise SystemExit(f"NN preflight failed: missing {raw_path}")
    paths = [raw_path]
    stable_path = bo_dir / "stable_results.csv"
    if stable_path.exists():
        paths.append(stable_path)
    paths.extend(core_files)

    targets = {}
    for name, info in config.get("targets", {}).items():
        if float(info.get("weight", 1.0)) <= 0.0:
            continue
        if name == "ead" and not config.get("adsorption", {}).get("enabled", False):
            continue
        if name == "esub_proxy" and not config.get("sublimation", {}).get("enabled", False):
            continue
        targets[name] = info

    rows_by_path: dict[Path, list[dict[str, str]]] = {}
    headers_by_path: dict[Path, set[str]] = {}
    for path in paths:
        if not path.exists():
            raise SystemExit(f"NN preflight failed: missing {path}")
        with path.open(encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            headers_by_path[path] = set(reader.fieldnames or [])
            rows = list(reader)
            rows_by_path[path] = rows
        if not rows:
            raise SystemExit(f"NN preflight failed: empty CSV {path}")
        provenance_rows = rows[:200]
        for name, info in targets.items():
            for kind, expected in (
                ("target", float(info["value"])),
                ("weight", float(info.get("weight", 1.0))),
            ):
                column = f"_objective_{kind}_{name}"
                values = {
                    float(row[column])
                    for row in provenance_rows
                    if row.get(column) not in (None, "")
                }
                if values and any(abs(value - expected) > 1.0e-10 for value in values):
                    raise SystemExit(
                        f"NN preflight failed: {path} declares {column}={sorted(values)}, "
                        f"but config requires {expected}. Rescore the CSV first."
                    )

    training_cfg = config.get("nn", {}).get("training_data", {})
    if training_cfg.get("mode", "legacy") == "core_buffer":
        required_properties = {f"calc_{name}" for name in targets}
        for path in paths:
            missing = required_properties - headers_by_path[path]
            if missing:
                raise SystemExit(
                    f"NN preflight failed: {path} is missing {sorted(missing)}"
                )

        def successful_objectives(selected_paths: list[Path]) -> list[float]:
            values: list[float] = []
            for selected_path in selected_paths:
                for row in rows_by_path[selected_path]:
                    if str(row.get("success", "true")).lower() not in ("true", "1"):
                        continue
                    try:
                        objective = float(row["objective"])
                    except (KeyError, TypeError, ValueError):
                        continue
                    if math.isfinite(objective):
                        values.append(objective)
            return values

        core_paths = ([stable_path] if stable_path.exists() else []) + core_files
        if not core_paths:
            core_paths = [raw_path]
        core_max = float(training_cfg.get("core_objective_max", float("inf")))
        buffer_max = float(training_cfg.get("buffer_objective_max", float("inf")))
        core_values = successful_objectives(core_paths)
        raw_values = successful_objectives([raw_path])
        n_core = sum(value <= core_max for value in core_values)
        n_buffer = sum(value <= buffer_max for value in raw_values)
        best_core = min(core_values, default=float("nan"))
        best_raw = min(raw_values, default=float("nan"))
        print(
            "NN preflight selection: "
            f"core={n_core} (max={core_max:g}, best={best_core:g}) "
            f"buffer={n_buffer} (max={buffer_max:g}, best={best_raw:g})"
        )
        if n_core < 10:
            raise SystemExit(
                "NN preflight failed: fewer than 10 stable core rows survive "
                f"objective<={core_max:g}; best available objective is "
                f"{best_core:g}. Set project-specific nn.training_data "
                "cutoffs appropriate for this parameter regime."
            )
        if n_buffer < 10:
            raise SystemExit(
                "NN preflight failed: fewer than 10 raw buffer rows survive "
                f"objective<={buffer_max:g}; best available objective is "
                f"{best_raw:g}."
            )

    print(f"NN preflight: {len(paths)} CSV files passed all checks")


def _common(args: argparse.Namespace) -> tuple[Project, str, Path]:
    selector = getattr(args, "input", None) or args.project
    project = load_project(selector)
    os.chdir(project.root)
    machine = args.machine or project.default_machine
    return project, machine, write_generated_config(project, machine)


def _execution_backend(project: Project, machine: str) -> str:
    config = compose_config(project, machine)
    declared = config.get("machine", {}).get("backend")
    if declared:
        return str(declared)
    return "slurm" if machine == "cluster" else "local"


def _free_parameter_count(config: dict) -> int:
    """Count free dimensions without importing the heavy ML runtime."""
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


def cmd_show(args: argparse.Namespace) -> None:
    project, machine, config_path = _common(args)
    config = compose_config(project, machine)
    active = config.get("workflow", {}).get("active_properties", list(config.get("targets", {})))
    print(f"Project       : {project.name}")
    print(f"Project file  : {project.path}")
    print(f"Machine       : {machine}")
    print(f"Backend       : {_execution_backend(project, machine)}")
    print(f"Generated cfg : {config_path}")
    print(f"Run root      : {project.run_root}")
    print(f"Dimensions    : {_free_parameter_count(config)}")
    print(f"Properties    : {', '.join(active)}")
    print(f"BO / NN       : {config['optimization']['method']} / {config['nn']['model']}")
    print(f"AL domain     : {config['active_learning'].get('sampling_domain', 'global')}")


def cmd_config(args: argparse.Namespace) -> None:
    _, _, config_path = _common(args)
    print(config_path)


def cmd_status(args: argparse.Namespace) -> None:
    project, machine, _ = _common(args)
    run_root = project.run_root
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
    print(f"Project: {project.name} [{machine}]\nRun root: {run_root}\n")
    from .pipeline import load_pipeline_status

    pipeline = load_pipeline_status(project, args.run_id)
    if pipeline:
        print(f"Pipeline: {args.run_id}")
        for stage in pipeline["stages"]:
            detail = stage.get("job_id") or stage.get("message") or ""
            print(
                f"  {stage['name']:10s} {stage['status']:10s} "
                f"attempt={stage['attempt']:<2d} {detail}"
            )
        print()
    for label, path in artifacts:
        print(f"{label:15s}: {path if path else '-'}")


def cmd_run(args: argparse.Namespace) -> None:
    project, machine, config_path = _common(args)
    from .pipeline import PipelineRunner

    run_id = args.run_id
    if args.new and run_id == "default":
        run_id = _timestamp()
    auto_resume = project.runtime_config is not None and not args.new

    runner = PipelineRunner(
        project=project,
        machine=machine,
        config_path=config_path,
        run_id=run_id,
        resume=args.resume or auto_resume,
        dry_run=args.dry_run,
        watch=args.watch,
        poll_seconds=args.poll_seconds,
        from_stage=args.from_stage,
        until=args.until,
    )
    outcome = runner.run()
    if outcome == "waiting":
        if project.runtime_config is not None:
            print("A SLURM stage is active. Run the same command again to continue, or add --watch.")
        else:
            print(
                "A SLURM stage is active. Run the same command with --resume to "
                "continue, or add --watch to advance automatically."
            )
    elif outcome == "completed":
        print(f"Pipeline completed: {runner.root}")


def cmd_doctor(args: argparse.Namespace) -> None:
    project, machine, config_path = _common(args)
    config = compose_config(project, machine)
    checks: list[tuple[str, bool, str]] = []
    checks.append(("Python >= 3.10", sys.version_info >= (3, 10), sys.version.split()[0]))
    for module in (
        "yaml", "numpy", "pandas", "torch", "sklearn", "botorch", "gpytorch",
    ):
        installed = importlib.util.find_spec(module) is not None
        checks.append((f"Python module: {module}", installed, "installed" if installed else "missing"))

    backend = _execution_backend(project, machine)
    if backend == "local":
        executable = Path(config["lammps"]["executable"])
        mpiexec = Path(config["lammps"].get("mpiexec", ""))
        checks.append(("LAMMPS executable", executable.exists(), str(executable)))
        checks.append(("MPI launcher", mpiexec.exists(), str(mpiexec)))
    else:
        sbatch = shutil.which("sbatch")
        checks.append(("SLURM sbatch", sbatch is not None, sbatch or "not on this host"))
        executable = Path(config["lammps"]["executable"])
        checks.append(("LAMMPS executable", executable.exists(), str(executable)))
        mpi_name = str(config["lammps"].get("mpiexec", "mpiexec"))
        mpi_launcher = shutil.which(mpi_name)
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

    if backend == "local":
        try:
            import torch
            gpu = bool(torch.cuda.is_available())
            detail = torch.cuda.get_device_name(0) if gpu else f"torch {torch.__version__}, CUDA unavailable"
            checks.append(("PyTorch CUDA", gpu, detail))
        except (ImportError, OSError, RuntimeError) as exc:
            checks.append(("PyTorch CUDA", False, str(exc)))
    else:
        gpu_request = int(config.get("cluster", {}).get("nn", {}).get("gpu", 0))
        detail = f"gpu={gpu_request}" if gpu_request else "CPU profile; no GPU requested"
        checks.append(("SLURM NN resources", True, detail))

    failed = False
    print(f"Doctor: {project.name} [{machine}]\nConfig: {config_path}\n")
    for label, ok, detail in checks:
        failed |= not ok
        print(f"[{'OK' if ok else 'FAIL'}] {label:24s} {detail}")
    if failed:
        raise SystemExit(1)


def cmd_bo(args: argparse.Namespace) -> None:
    project, machine, config_path = _common(args)
    is_slurm = _execution_backend(project, machine) == "slurm"
    if is_slurm:
        command = _python("slurm/submit.py", "bo", "--config", config_path)
    else:
        command = _module("engine.run", "--config", config_path)
    if args.resume:
        command.append("--resume")
    if args.output_dir:
        command.extend(["--output-dir", resolve_path(args.output_dir)])
    if args.smoke:
        if is_slurm:
            raise SystemExit("Cluster BO already performs a pre-flight check; --smoke is local-only.")
        command.append("--dry-run")
    if args.save_traj and args.smoke:
        command.append("--save-traj")
    if args.dry_run:
        if is_slurm:
            command.append("--dry-run")
            _run(command)
        else:
            _preview(command)
        return
    _run(command)


def _sampling_args(project: Project, args: argparse.Namespace) -> tuple[Path, Path, dict]:
    defaults = _stage_defaults(project, "sample")
    source_arg = args.source or defaults.get("source", "auto")
    source = _latest_bo(project) if source_arg == "auto" else resolve_path(source_arg)
    if source is None:
        raise SystemExit("No BO result found. Pass --source explicitly.")
    if source.is_dir():
        stable = source / "stable_results.csv"
        source = stable if stable.exists() else source / "all_results.csv"
    if args.output_dir:
        output = resolve_path(args.output_dir)
    elif defaults.get("output_dir"):
        output = resolve_path(defaults["output_dir"])
    else:
        output = project.run_root / f"sample_{_timestamp()}"
    return source, output, defaults


def cmd_sample(args: argparse.Namespace) -> None:
    project, machine, config_path = _common(args)
    source, output, defaults = _sampling_args(project, args)
    n_points = args.n_points or defaults.get("n_points", 1200)
    elite_centers = args.elite_centers or defaults.get("elite_centers", 8)
    center_selection = args.center_selection or defaults.get(
        "center_selection", "top"
    )
    center_pool_multiplier = (
        args.center_pool_multiplier
        if args.center_pool_multiplier is not None
        else defaults.get("center_pool_multiplier", 5)
    )
    radii = args.radii or defaults.get("radii", [0.005, 0.015, 0.04])
    global_fraction = args.global_fraction if args.global_fraction is not None else defaults.get("global_fraction", 0.1)
    seeds = args.seeds or defaults.get("seeds", [101])
    design_seed = args.design_seed or defaults.get("design_seed")
    is_slurm = _execution_backend(project, machine) == "slurm"
    if is_slurm:
        command = _python(
            "slurm/submit_local_sampling.py", "--config", config_path,
            "--source", source, "--output-dir", output,
            "--n-points", n_points, "--elite-centers", elite_centers,
            "--center-selection", center_selection,
            "--center-pool-multiplier", center_pool_multiplier,
            "--radii", *radii, "--global-fraction", global_fraction,
            "--seeds", *seeds,
        )
    else:
        command = _module(
            "engine.local_sampling", "--config", config_path,
            "--source", source, "--output-dir", output,
            "--n-points", n_points, "--elite-centers", elite_centers,
            "--center-selection", center_selection,
            "--center-pool-multiplier", center_pool_multiplier,
            "--radii", *radii, "--global-fraction", global_fraction,
            "--seeds", *seeds,
        )
    if args.max_workers:
        command.extend(["--max-workers", args.max_workers])
    if design_seed:
        command.extend(["--design-seed", design_seed])
    if is_slurm and args.dependency:
        command.extend(["--dependency", args.dependency])
    if args.dry_run and is_slurm:
        command.append("--dry-run")
    elif args.dry_run:
        _preview(command)
        return
    _run(command)


def cmd_nn(args: argparse.Namespace) -> None:
    project, machine, config_path = _common(args)
    defaults = _stage_defaults(project, "nn")
    preferred_bo = args.bo_dir or defaults.get("bo_dir")
    bo_dir = resolve_path(preferred_bo) if preferred_bo else _latest_bo(project)
    if bo_dir is None:
        raise SystemExit("No BO result found. Pass --bo-dir explicitly.")
    configured_core_files = list(
        args.additional_core_file or defaults.get("additional_core_files", [])
    )
    core_files = [
        resolve_path(item)
        for item in configured_core_files
    ]
    sample_defaults = _stage_defaults(project, "sample")
    if not core_files and sample_defaults:
        sample_file = _latest_sample(project)
        if sample_file is not None:
            core_files.append(sample_file)
            print(f"NN auto-selected sampling data: {sample_file}")
        elif not args.allow_bo_only:
            raise SystemExit(
                "NN preflight failed: this project defines a sample stage, but no "
                "local_results.csv was found. Run 'ffopt sample "
                f"--project {project.path} --machine {machine}' first, or pass "
                "--allow-bo-only to deliberately train from BO data alone."
            )
        else:
            print("WARNING: --allow-bo-only selected; NN will not use independent "
                  "sampling data.")
    config = compose_config(project, machine)
    _validate_nn_inputs(config, bo_dir, core_files)
    output = resolve_path(args.output_dir) if args.output_dir else project.run_root / f"nn_{_timestamp()}"
    is_slurm = _execution_backend(project, machine) == "slurm"
    if is_slurm:
        command = _python(
            "slurm/submit.py", "nn", "--config", config_path,
            "--bo-dir", bo_dir, "--output-dir", output,
        )
    else:
        command = _module(
            "engine.nn_surrogate", "--config", config_path,
            "--bo-dir", bo_dir, "--output-dir", output,
        )
    for path in core_files:
        command.extend(["--additional-core-file", path])
    if args.no_validate:
        command.append("--no-validate")
    if args.save_traj:
        command.append("--save-traj")
    if args.reuse_models_from:
        if is_slurm:
            raise SystemExit("--reuse-models-from is currently a local NN option")
        command.extend([
            "--reuse-models-from", resolve_path(args.reuse_models_from)
        ])
    if args.skip_optimize:
        command.append("--skip-optimize")
    if args.dry_run and is_slurm:
        command.append("--dry-run")
    elif args.dry_run:
        _preview(command)
        return
    _run(command)


def cmd_al(args: argparse.Namespace) -> None:
    project, machine, config_path = _common(args)
    al_defaults = _stage_defaults(project, "al")
    nn_defaults = _stage_defaults(project, "nn")
    preferred_bo = args.bo_dir or al_defaults.get("bo_dir") or nn_defaults.get("bo_dir")
    bo_dir = resolve_path(preferred_bo) if preferred_bo else _latest_bo(project)
    nn_dir = resolve_path(args.nn_dir) if args.nn_dir else _latest_nn(project)
    if bo_dir is None or nn_dir is None:
        raise SystemExit("AL needs BO and NN results. Pass --bo-dir/--nn-dir explicitly.")
    output = resolve_path(args.output_dir) if args.output_dir else project.run_root / f"al_{_timestamp()}"
    is_slurm = _execution_backend(project, machine) == "slurm"
    if is_slurm:
        command = _python(
            "slurm/submit.py", "al", "--config", config_path,
            "--bo-dir", bo_dir, "--nn-dir", nn_dir, "--output-dir", output,
        )
    else:
        command = _module(
            "engine.active_learning", "--config", config_path,
            "--bo-dir", bo_dir, "--nn-dir", nn_dir, "--output-dir", output,
        )
    if args.no_validate:
        command.append("--no-validate")
    for path in args.additional_core_file or []:
        command.extend(["--additional-core-file", resolve_path(path)])
    if args.save_traj:
        command.append("--save-traj")
    if args.resume:
        command.append("--resume")
    if args.dry_run and is_slurm:
        command.append("--dry-run")
    elif args.dry_run:
        _preview(command)
        return
    _run(command)


def cmd_audit(args: argparse.Namespace) -> None:
    project, _, config_path = _common(args)
    source = resolve_path(args.source) if args.source else _latest_bo(project, stable=False)
    if source is None:
        raise SystemExit("No BO result found. Pass --source explicitly.")
    if source.is_dir():
        source = source / "all_results.csv"
    output = resolve_path(args.output_dir) if args.output_dir else project.run_root / f"audit_{_timestamp()}"
    command = _module(
        "engine.stability_audit", "--config", config_path, "--source", source,
        "--output-dir", output, "--top-k", args.top_k, "--seeds", *args.seeds,
    )
    if args.max_workers:
        command.extend(["--max-workers", args.max_workers])
    _run(command)


def cmd_validate(args: argparse.Namespace) -> None:
    project, _, config_path = _common(args)
    output = resolve_path(args.output_dir) if args.output_dir else project.run_root / f"validation_{_timestamp()}"
    command = _module(
        "engine.validate_final_parameters", "--config", config_path,
        "--output-dir", output,
    )
    if args.initial:
        command.append("--initial")
    else:
        parameters = resolve_path(args.parameters) if args.parameters else _newest([
            f"runs/{project.name}/**/final_summary.json",
            f"runs/{project.name}/**/final_parameters.json",
            f"runs/{project.name}/legacy/**/final_summary.json",
            f"runs/{project.name}/legacy/**/final_parameters.json",
        ])
        if parameters is None:
            raise SystemExit(
                "No final parameter JSON found. Pass --parameters or --initial."
            )
        command.extend(["--parameters", parameters])
    if args.overwrite:
        command.append("--overwrite")
    if args.dry_run:
        _preview(command)
        return
    _run(command)


def cmd_finalize(args: argparse.Namespace) -> None:
    project, _, config_path = _common(args)
    audits = [resolve_path(path) for path in args.audit]
    output = (
        resolve_path(args.output_dir)
        if args.output_dir
        else project.run_root / f"final_{_timestamp()}"
    )
    command = _module(
        "engine.finalize_al", "--config", config_path,
        "--output-dir", output,
    )
    for audit in audits:
        command.extend(["--audit", audit])
    _run(command)


def cmd_plot(args: argparse.Namespace) -> None:
    project, _, config_path = _common(args)
    bo_dir = resolve_path(args.bo_dir) if args.bo_dir else _latest_bo(project)
    if bo_dir is None:
        raise SystemExit("No BO result found. Pass --bo-dir explicitly.")
    command = _module("engine.run", "--config", config_path, "--plot", bo_dir)
    if args.nn_dir:
        command.extend(["--nn-dir", resolve_path(args.nn_dir)])
    _run(command)


def _add_context(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project",
        default=DEFAULT_PROJECT,
        help="FFOpt command input (default: ffopt.in).",
    )
    parser.add_argument("--machine", default=None, help="Project or user execution profile; defaults to the project setting.")


def cmd_machine(args: argparse.Namespace) -> None:
    if args.action == "list":
        names = available_machine_profiles()
        print("\n".join(names) if names else "No user machine profiles configured.")
        return
    if args.action == "show":
        profile = load_machine_profile(args.name)
        if profile is None:
            raise SystemExit(f"Machine profile not found: {machine_path(args.name)}")
        print(json.dumps(
            {key: value for key, value in profile.items() if not key.startswith("_")},
            indent=2,
        ))
        return
    profile = build_machine_profile(
        name=args.name,
        backend=args.backend,
        lammps=args.lammps,
        mpi=args.mpi,
        workers=args.workers,
        ranks=args.ranks,
        omp_threads=args.omp_threads,
        timeout=args.timeout,
        partition=args.partition,
        qos=args.qos,
        cores=args.cores,
        nodes=args.nodes,
        gpus=args.gpus,
        walltime=args.walltime,
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
    print(f"Workflow              : {' -> '.join(project.data['pipeline']['stages'])}")


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
    print(f"Workflow              : {' -> '.join(project.data['pipeline']['stages'])}")
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
                f"{target.unit} weight={target.weight:g}"
            )
    sample = project.data.get("stages", {}).get("sample", {})
    if "sample" in project.data["pipeline"]["stages"]:
        seeds = sample.get("seeds", [])
        points = int(sample.get("n_points", 0))
        print(f"\nSampling              : {points} points x {len(seeds)} seeds")


def cmd_plugins(args: argparse.Namespace) -> None:
    from engine.property_evaluators import PROPERTY_EVALUATORS

    descriptions = PROPERTY_EVALUATORS.describe()
    if args.json:
        print(json.dumps(descriptions, indent=2))
        return
    print("Property evaluators:")
    for item in descriptions:
        dependencies = ",".join(item["dependencies"]) or "-"
        provides = ",".join(item["provides"]) or "dynamic"
        print(
            f"  {item['name']:14s} dependencies={dependencies:12s} "
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
    print(f"Targets         : {', '.join(result.targets)}")
    for warning in result.warnings:
        print(f"WARNING: {warning}")
    print("\nNext:")
    print(f"  cd {result.root}")
    print("  ffopt check ffopt.in")
    print("  ffopt explain ffopt.in")
    print("  ffopt run ffopt.in --dry-run")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ffopt",
        description="One command for BO, sampling, surrogate training and active learning.",
    )
    sub = parser.add_subparsers(dest="command", required=True)
    init = sub.add_parser(
        "init", help="Create a portable molecular project from a LAMMPS data file."
    )
    init.add_argument("name", help="Project and system name.")
    init.add_argument("--data-file", required=True, help="Bulk molecular-crystal data file.")
    init.add_argument(
        "--single-data",
        help="Isolated molecule data file for the sublimation-enthalpy estimate.",
    )
    init.add_argument("--destination", help="Output directory; defaults to NAME.")
    init.add_argument(
        "--target", action="append", required=True,
        help="Repeat NAME=VALUE[,WEIGHT[,UNIT]], e.g. a=4.24,1.0,A.",
    )
    init.add_argument(
        "--mode", choices=["full", "fix_sigma", "charge_only", "lj_only"],
        default="full",
    )
    init.add_argument(
        "--cells", type=int, nargs=3, metavar=("NX", "NY", "NZ"),
        default=[1, 1, 1],
        help="Primitive-cell repeats already present in the bulk data file.",
    )
    init.add_argument(
        "--mixing-rule", choices=["geometric", "arithmetic"], default="geometric"
    )
    init.add_argument(
        "--derive-charge", default="auto",
        help="auto, none, or atom type ID used to enforce total neutrality.",
    )
    init.add_argument("--epsilon-scale", type=float, nargs=2, default=[0.5, 2.0])
    init.add_argument("--sigma-scale", type=float, nargs=2, default=[0.85, 1.15])
    init.add_argument("--charge-window", type=float, default=0.30)
    init.add_argument("--charge-limit", type=float, default=2.0)
    init.add_argument("--force", action="store_true")
    init.set_defaults(function=cmd_init)

    inspect = sub.add_parser("inspect", help="Inspect atom types and parameters in a LAMMPS data file.")
    inspect.add_argument("data_file")
    inspect.add_argument("--json", action="store_true", help="Emit a machine-readable summary.")
    inspect.set_defaults(function=cmd_inspect)
    check = sub.add_parser("check", help="Validate one ffopt.in file without running LAMMPS.")
    check.add_argument("input")
    check.set_defaults(function=cmd_check)
    explain = sub.add_parser("explain", help="Explain parameters, properties, and workflow in ffopt.in.")
    explain.add_argument("input")
    explain.set_defaults(function=cmd_explain)
    plugins = sub.add_parser("plugins", help="List built-in and installed property evaluators.")
    plugins.add_argument("--json", action="store_true")
    plugins.set_defaults(function=cmd_plugins)
    machine = sub.add_parser("machine", help="Configure reusable local or SLURM execution profiles.")
    machine.add_argument("action", choices=["configure", "show", "list"])
    machine.add_argument("--name", default="local")
    machine.add_argument("--backend", choices=["local", "slurm"], default="local")
    machine.add_argument("--lammps", help="LAMMPS executable; auto-detected when omitted.")
    machine.add_argument("--mpi", help="MPI launcher; auto-detected when omitted.")
    machine.add_argument("--workers", type=int)
    machine.add_argument("--ranks", type=int, default=1)
    machine.add_argument("--omp-threads", type=int, default=1)
    machine.add_argument("--timeout", type=int, default=21600)
    machine.add_argument("--partition")
    machine.add_argument("--qos")
    machine.add_argument("--cores", type=int)
    machine.add_argument(
        "--nodes", type=int, default=1,
        help="SLURM nodes used by parallel LAMMPS stages (default: 1).",
    )
    machine.add_argument(
        "--gpus", type=int, default=0,
        help="GPUs requested by NN/AL stages (default: 0).",
    )
    machine.add_argument("--walltime", default="24:00:00")
    machine.add_argument("--force", action="store_true")
    machine.set_defaults(function=cmd_machine)
    for name, function in (
        ("show", cmd_show),
        ("status", cmd_status),
        ("config", cmd_config),
        ("doctor", cmd_doctor),
    ):
        child = sub.add_parser(name)
        _add_context(child)
        if name == "status":
            child.add_argument(
                "--run-id", default="default",
                help="Named pipeline run to display (default: default).",
            )
        child.set_defaults(function=function)

    run = sub.add_parser(
        "run", help="Run the restartable BO -> sampling -> NN -> AL pipeline."
    )
    run.add_argument("input", nargs="?", help="Public .in input file (recommended).")
    _add_context(run)
    run.add_argument("--run-id", default="default")
    run.add_argument("--resume", action="store_true")
    run.add_argument(
        "--new", action="store_true",
        help="Start an independent timestamped run instead of auto-resuming.",
    )
    run.add_argument("--dry-run", action="store_true")
    run.add_argument(
        "--watch", action="store_true",
        help="On SLURM, keep polling and submit each next stage automatically.",
    )
    run.add_argument("--poll-seconds", type=int, default=60)
    run.add_argument("--from-stage", choices=[
        "bo", "sample", "nn", "al", "audit", "finalize", "validate"
    ])
    run.add_argument("--until", choices=[
        "bo", "sample", "nn", "al", "audit", "finalize", "validate"
    ])
    run.set_defaults(function=cmd_run)

    bo = sub.add_parser("bo", help="Run or submit Bayesian optimization.")
    _add_context(bo)
    bo.add_argument("--resume", action="store_true")
    bo.add_argument("--output-dir")
    bo.add_argument("--dry-run", action="store_true", help="Preview only; never run or submit work.")
    bo.add_argument("--smoke", action="store_true", help="Local-only: run one real LAMMPS pipeline evaluation.")
    bo.add_argument("--save-traj", action="store_true", help="Save trajectories during --smoke.")
    bo.set_defaults(function=cmd_bo)

    sample = sub.add_parser("sample", help="Generate multi-centre local/global LAMMPS samples.")
    _add_context(sample)
    sample.add_argument("--source")
    sample.add_argument("--output-dir")
    sample.add_argument("--n-points", type=int)
    sample.add_argument("--elite-centers", type=int)
    sample.add_argument("--center-selection", choices=["top", "diverse"])
    sample.add_argument("--center-pool-multiplier", type=int)
    sample.add_argument("--radii", type=float, nargs="+")
    sample.add_argument("--global-fraction", type=float)
    sample.add_argument("--seeds", type=int, nargs="+")
    sample.add_argument("--max-workers", type=int)
    sample.add_argument("--design-seed", type=int)
    sample.add_argument("--dependency")
    sample.add_argument("--dry-run", action="store_true", help="Preview only; never run or submit work.")
    sample.set_defaults(function=cmd_sample)

    for name, function in (("nn", cmd_nn), ("al", cmd_al)):
        child = sub.add_parser(name)
        _add_context(child)
        child.add_argument("--bo-dir")
        child.add_argument("--output-dir")
        child.add_argument("--no-validate", action="store_true")
        child.add_argument("--save-traj", action="store_true")
        child.add_argument("--dry-run", action="store_true", help="Preview only; never run or submit work.")
        if name == "nn":
            child.add_argument("--additional-core-file", action="append")
            child.add_argument(
                "--reuse-models-from",
                help="Reuse unchanged property models from forward_nn.pt.",
            )
            child.add_argument(
                "--skip-optimize",
                action="store_true",
                help="Train and score the surrogate without candidate search.",
            )
            child.add_argument(
                "--allow-bo-only", action="store_true",
                help="Deliberately train without the project's sampling stage.",
            )
        else:
            child.add_argument("--nn-dir")
            child.add_argument("--additional-core-file", action="append")
            child.add_argument(
                "--resume", action="store_true",
                help="Continue after the last fully recorded AL round.",
            )
        child.set_defaults(function=function)

    audit = sub.add_parser("audit", help="Re-evaluate top candidates with independent seeds.")
    _add_context(audit)
    audit.add_argument("--source")
    audit.add_argument("--output-dir")
    audit.add_argument("--top-k", type=int, default=20)
    audit.add_argument("--seeds", type=int, nargs="+", default=[101, 202, 303])
    audit.add_argument("--max-workers", type=int)
    audit.set_defaults(function=cmd_audit)

    finalize = sub.add_parser(
        "finalize", help="Select the best robust audit result and export parameters."
    )
    _add_context(finalize)
    finalize.add_argument("--audit", action="append", required=True)
    finalize.add_argument("--output-dir")
    finalize.set_defaults(function=cmd_finalize)

    validate = sub.add_parser("validate", help="Run final trajectory-producing LAMMPS validation.")
    _add_context(validate)
    validation_source = validate.add_mutually_exclusive_group()
    validation_source.add_argument("--parameters")
    validation_source.add_argument(
        "--initial",
        action="store_true",
        help="Use initial type values from ffopt.in.",
    )
    validate.add_argument("--output-dir")
    validate.add_argument("--overwrite", action="store_true")
    validate.add_argument("--dry-run", action="store_true")
    validate.set_defaults(function=cmd_validate)

    plot = sub.add_parser("plot", help="Generate BO/NN figures.")
    _add_context(plot)
    plot.add_argument("--bo-dir")
    plot.add_argument("--nn-dir")
    plot.set_defaults(function=cmd_plot)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)
