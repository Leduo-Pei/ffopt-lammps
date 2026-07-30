"""User-level execution profiles for local workstations and schedulers."""

from __future__ import annotations

import math
import os
import platform
import shlex
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import yaml

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

import tomli_w


def config_home() -> Path:
    """Return the FFOpt user configuration directory.

    ``FFOPT_CONFIG_DIR`` is useful for tests and managed installations. The
    default is deliberately identical on Windows and Unix so project files and
    documentation remain portable.
    """
    override = os.environ.get("FFOPT_CONFIG_DIR")
    if override:
        return Path(override).expanduser().resolve()
    return Path.home() / ".config" / "ffopt"


def machine_dir() -> Path:
    """Legacy per-profile YAML directory, retained for read compatibility."""
    return config_home() / "machines"


def machine_path(name: str) -> Path:
    if not name or any(char in name for char in "\\/:"):
        raise ValueError(f"Invalid machine profile name: {name!r}")
    return config_home() / "machines.toml"


def _machine_document() -> dict[str, Any]:
    path = config_home() / "machines.toml"
    if not path.exists():
        return {"machines": {}}
    with path.open("rb") as handle:
        data = tomllib.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("machines", {}), dict):
        raise TypeError(f"Invalid FFOpt machine configuration: {path}")
    data.setdefault("machines", {})
    return data


def load_machine_profile(name: str) -> dict[str, Any] | None:
    path = machine_path(name)
    document = _machine_document()
    data = document["machines"].get(name)
    if data is None:
        legacy_path = machine_dir() / f"{name}.yaml"
        if not legacy_path.exists():
            return None
        with legacy_path.open(encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        path = legacy_path
    if not isinstance(data, dict):
        raise TypeError(f"Machine profile must contain a mapping: {path}")
    data = dict(data)
    data.setdefault("machine", {})["name"] = name
    data["_machine_profile_path"] = str(path)
    return data


def available_machine_profiles() -> list[str]:
    names = set(_machine_document()["machines"])
    directory = machine_dir()
    if directory.exists():
        names.update(path.stem for path in directory.glob("*.yaml"))
    return sorted(names)


def _detect_lammps() -> str:
    configured = os.environ.get("LAMMPS_EXECUTABLE")
    if configured:
        return configured
    for candidate in ("lmp", "lmp_mpi", "lmp_serial", "lmp.exe"):
        found = shutil.which(candidate)
        if found:
            return found
    return "lmp"


def _detect_mpi() -> str:
    configured = os.environ.get("MPIEXEC_EXECUTABLE")
    if configured:
        return configured
    for candidate in ("mpiexec", "mpirun", "srun", "mpiexec.exe"):
        found = shutil.which(candidate)
        if found:
            return found
    return "mpiexec"


def _version_line(command: str, arguments: list[str]) -> str:
    executable = shutil.which(command) or command
    try:
        completed = subprocess.run(
            [executable, *arguments], capture_output=True, text=True,
            timeout=10, check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"unavailable: {exc}"
    output = completed.stdout or completed.stderr
    return next((line.strip() for line in output.splitlines() if line.strip()), "no version output")


def probe_machine_environment(partition: str | None = None) -> dict[str, Any]:
    """Inspect executables and scheduler resources without changing config."""
    lammps = _detect_lammps()
    mpi = _detect_mpi()
    lammps_path = shutil.which(lammps) or (str(Path(lammps).resolve()) if Path(lammps).exists() else None)
    mpi_path = shutil.which(mpi) or (str(Path(mpi).resolve()) if Path(mpi).exists() else None)
    sbatch = shutil.which("sbatch")
    sinfo = shutil.which("sinfo")
    backend = "slurm" if sbatch and sinfo else "local"
    partitions: list[dict[str, Any]] = []
    if sinfo:
        completed = subprocess.run(
            [sinfo, "-h", "-o", "%P|%c|%m|%a|%l|%D|%t"],
            capture_output=True, text=True, check=False,
        )
        for raw in completed.stdout.splitlines():
            columns = [item.strip() for item in raw.split("|")]
            if len(columns) != 7:
                continue
            name = columns[0].rstrip("*")
            if partition and name != partition:
                continue
            try:
                cores = int(columns[1])
                memory_mb = int(columns[2])
                nodes = int(columns[5])
            except ValueError:
                continue
            partitions.append({
                "name": name,
                "default": columns[0].endswith("*"),
                "cores_per_node": cores,
                "memory_mb_per_node": memory_mb,
                "availability": columns[3],
                "time_limit": columns[4],
                "nodes": nodes,
                "state": columns[6],
            })
    preferred = next((item for item in partitions if item["default"]), None)
    if preferred is None and partitions:
        preferred = partitions[0]
    recommendation = None
    if preferred:
        ranks = 4 if preferred["cores_per_node"] >= 4 else 1
        workers_per_node = max(1, preferred["cores_per_node"] // ranks)
        safe_memory_gb = max(1, min(64, preferred["memory_mb_per_node"] // 1024 // 2))
        recommendation = {
            "partition": preferred["name"],
            "nodes": 1,
            "cores": preferred["cores_per_node"],
            "workers": workers_per_node,
            "ranks": ranks,
            "omp_threads": 1,
            "memory_per_node": f"{safe_memory_gb}G",
            "note": "Review node sharing and memory policy before production use.",
        }
    gpu_available = False
    gpu_name = None
    try:
        import torch
        gpu_available = bool(torch.cuda.is_available())
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass
    return {
        "host": platform.node(),
        "platform": platform.platform(),
        "backend": backend,
        "logical_cpus": max(1, os.cpu_count() or 1),
        "conda_environment": os.environ.get("CONDA_DEFAULT_ENV"),
        "python": shutil.which("python"),
        "lammps": {
            "path": lammps_path or lammps,
            "available": lammps_path is not None,
            "version": _version_line(lammps, ["-help"]),
        },
        "mpi": {
            "path": mpi_path or mpi,
            "available": mpi_path is not None,
            "version": _version_line(mpi, ["--version"]),
        },
        "slurm": {
            "sbatch": sbatch,
            "srun": shutil.which("srun"),
            "sinfo": sinfo,
            "partitions": partitions,
        },
        "torch_cuda": {"available": gpu_available, "device": gpu_name},
        "recommendation": recommendation,
    }


def format_machine_probe(report: dict[str, Any]) -> str:
    lines = [
        "FFOpt machine probe",
        "",
        f"Host       : {report['host']}",
        f"Backend    : {report['backend']}",
        f"Logical CPU: {report['logical_cpus']}",
        f"Conda env  : {report['conda_environment'] or '-'}",
        f"Python     : {report['python'] or '-'}",
        (
            f"LAMMPS     : {report['lammps']['path']}"
            f"{' [NOT FOUND]' if not report['lammps'].get('available', True) else ''}"
        ),
        f"             {report['lammps']['version']}",
        (
            f"MPI        : {report['mpi']['path']}"
            f"{' [NOT FOUND]' if not report['mpi'].get('available', True) else ''}"
        ),
        f"             {report['mpi']['version']}",
        f"Torch CUDA : {report['torch_cuda']['available']} "
        f"{report['torch_cuda']['device'] or ''}".rstrip(),
    ]
    partitions = report["slurm"]["partitions"]
    if partitions:
        lines.extend(["", "SLURM partitions:"])
        for item in partitions:
            marker = "*" if item["default"] else " "
            lines.append(
                f"  {marker}{item['name']}: nodes={item['nodes']} "
                f"cores/node={item['cores_per_node']} mem/node={item['memory_mb_per_node']}MB "
                f"limit={item['time_limit']} state={item['state']}"
            )
    recommendation = report.get("recommendation")
    if recommendation:
        lines.extend([
            "",
            "Conservative starting point:",
            (
                f"  partition={recommendation['partition']} nodes={recommendation['nodes']} "
                f"cores={recommendation['cores']} workers={recommendation['workers']} "
                f"ranks={recommendation['ranks']} omp={recommendation['omp_threads']} "
                f"memory={recommendation['memory_per_node']}"
            ),
            f"  {recommendation['note']}",
        ])
    return "\n".join(lines)


def _machine_test_input() -> str:
    return """units real
atom_style atomic
boundary p p p
region box block 0 10 0 10 0 10
create_box 1 box
create_atoms 1 single 5 5 5
mass 1 12.011
pair_style lj/cut 8.0
pair_coeff 1 1 0.10 3.40
neighbor 2.0 bin
run 0
print \"FFOPT_MACHINE_TEST_OK\"
"""


def prepare_machine_test(name: str, profile: dict[str, Any]) -> dict[str, Any]:
    """Write a tiny LAMMPS smoke input and return local/SLURM commands."""
    root = config_home() / "machine-tests" / name
    root.mkdir(parents=True, exist_ok=True)
    input_path = root / "in.machine_test"
    input_path.write_text(_machine_test_input(), encoding="ascii", newline="\n")
    lammps = str(profile["lammps"]["executable"])
    mpi = str(profile["lammps"].get("mpiexec", "mpiexec"))
    parallel = profile.get("parallel", {})
    ranks = max(1, int(parallel.get("cores_per_worker", 1)))
    omp_threads = max(1, int(parallel.get("omp_threads_per_worker", 1)))
    backend = str(profile.get("machine", {}).get("backend", "local"))
    if ranks > 1:
        if Path(mpi).name.lower().startswith("srun"):
            command = [mpi, "--ntasks", str(ranks), "--cpus-per-task", str(omp_threads), lammps]
        elif backend == "slurm":
            command = [
                sys.executable,
                "-m", "workflow.mpi_local_exec",
                "--launcher", mpi,
                "--ranks", str(ranks),
                "--slots", "1",
                "--", lammps,
            ]
        else:
            command = [mpi, "-n", str(ranks), lammps]
    else:
        command = [lammps]
    command.extend(["-in", str(input_path)])
    result: dict[str, Any] = {
        "backend": backend,
        "root": root,
        "input": input_path,
        "command": command,
        "omp_threads": omp_threads,
    }
    if backend == "slurm":
        cluster = profile.get("cluster", {})
        resources = cluster.get("validate", cluster.get("bo", {}))
        output_path = root / "slurm_%j.out"
        error_path = root / "slurm_%j.err"
        directives = [
            "#!/bin/bash",
            f"#SBATCH --job-name=ffopt_test_{name}",
            f"#SBATCH --output={output_path}",
            f"#SBATCH --error={error_path}",
            "#SBATCH --nodes=1",
            "#SBATCH --ntasks=1",
            f"#SBATCH --cpus-per-task={ranks * omp_threads}",
            "#SBATCH --time=00:10:00",
        ]
        if str(resources.get("partition", "")).strip():
            directives.append(f"#SBATCH --partition={resources['partition']}")
        if str(resources.get("qos", "")).strip():
            directives.append(f"#SBATCH --qos={resources['qos']}")
        memory = str(resources.get("mem", "")).strip()
        if memory not in {"", "0", "None"}:
            directives.append(f"#SBATCH --mem={memory}")
        script = root / "test.sh"
        quoted = " ".join(shlex.quote(str(item)) for item in command)
        script.write_text(
            "\n".join([
                *directives, "", "set -euo pipefail",
                f"export OMP_NUM_THREADS={omp_threads}",
                f"cd {shlex.quote(str(root))}", quoted, "",
            ]),
            encoding="ascii", newline="\n",
        )
        result["script"] = script
    return result


def execute_machine_test(
    name: str, profile: dict[str, Any], *, dry_run: bool = False, wait: bool = True
) -> dict[str, Any]:
    prepared = prepare_machine_test(name, profile)
    if dry_run:
        return {**prepared, "status": "dry-run"}
    environment = os.environ.copy()
    environment["OMP_NUM_THREADS"] = str(prepared["omp_threads"])
    if prepared["backend"] == "local":
        try:
            completed = subprocess.run(
                prepared["command"], cwd=prepared["root"], env=environment,
                capture_output=True, text=True, timeout=120, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            return {**prepared, "status": "failed", "output": str(exc)}
        output = completed.stdout + completed.stderr
        ok = completed.returncode == 0 and "FFOPT_MACHINE_TEST_OK" in output
        return {**prepared, "status": "passed" if ok else "failed", "output": output}
    submission = ["sbatch"]
    if wait:
        submission.append("--wait")
    submission.append(str(prepared["script"]))
    completed = subprocess.run(
        submission, capture_output=True, text=True, check=False,
    )
    job_id = next((token for token in completed.stdout.split() if token.isdigit()), None)
    result = {
        **prepared,
        "job_id": job_id,
        "submission_output": completed.stdout + completed.stderr,
    }
    output = ""
    if wait and job_id:
        output_path = prepared["root"] / f"slurm_{job_id}.out"
        error_path = prepared["root"] / f"slurm_{job_id}.err"
        output = "\n".join(
            part for part in (
                output_path.read_text(encoding="utf-8", errors="replace")
                if output_path.exists() else "",
                error_path.read_text(encoding="utf-8", errors="replace")
                if error_path.exists() else "",
            )
            if part
        )
        result["output"] = output
    if completed.returncode != 0:
        result["status"] = "failed"
        return result
    if not wait:
        result["status"] = "submitted"
        return result
    result["status"] = "passed" if "FFOPT_MACHINE_TEST_OK" in output else "failed"
    return result


def build_machine_profile(
    *,
    name: str,
    backend: str,
    lammps: str | None = None,
    mpi: str | None = None,
    workers: int | None = None,
    ranks: int = 1,
    omp_threads: int = 1,
    timeout: int = 21600,
    partition: str | None = None,
    qos: str | None = None,
    cores: int | None = None,
    nodes: int = 1,
    gpus: int = 0,
    walltime: str = "24:00:00",
    memory_per_node: str | None = None,
) -> dict[str, Any]:
    if backend not in {"local", "slurm"}:
        raise ValueError("backend must be 'local' or 'slurm'")
    cpu_count = max(1, os.cpu_count() or 1)
    ranks = max(1, int(ranks))
    omp_threads = max(1, int(omp_threads))
    nodes = max(1, int(nodes))
    gpus = max(0, int(gpus))
    memory = str(memory_per_node or "0").strip()
    if not memory:
        memory = "0"
    workers = max(1, int(workers or max(1, cpu_count // (ranks * omp_threads))))
    profile: dict[str, Any] = {
        "machine": {
            "name": name,
            "backend": backend,
            "extends": "cluster" if backend == "slurm" else "local",
        },
        "lammps": {
            "executable": lammps or _detect_lammps(),
            "mpiexec": mpi or _detect_mpi(),
            "timeout": int(timeout),
        },
        "parallel": {
            "max_workers": workers,
            "cores_per_worker": ranks,
            "omp_threads_per_worker": omp_threads,
            "use_mpi": ranks > 1,
            **({
                "scheduler_launcher": "srun",
                "scheduler_nodes": nodes,
                "workers_per_node": math.ceil(workers / nodes),
            } if backend == "slurm" else {}),
        },
        "machine_learning": {"device": "auto"},
    }
    if backend == "slurm":
        required_cpus = workers * ranks * omp_threads
        allocated_cpus = int(cores or required_cpus)
        if allocated_cpus < required_cpus:
            raise ValueError(
                "SLURM cores must cover workers * ranks * omp_threads "
                f"({allocated_cpus} < {required_cpus})"
            )
        if allocated_cpus % omp_threads:
            raise ValueError("SLURM cores must be divisible by omp_threads")
        per_node_cpus = max(1, math.ceil(allocated_cpus / nodes))

        distributed = {
            "partition": partition or "",
            "qos": qos or "",
            "nodes": nodes,
            "cores": allocated_cpus,
            "tasks": workers,
            **({"tasks_per_node": workers // nodes} if workers % nodes == 0 else {}),
            "cpus_per_task": ranks * omp_threads,
            "distributed_steps": True,
            "time": walltime,
            "mem": memory,
        }
        single_lammps = {
            **distributed,
            "nodes": 1,
            "cores": ranks * omp_threads,
            "tasks": 1,
            "tasks_per_node": 1,
            "cpus_per_task": ranks * omp_threads,
        }
        single_python = {
            "partition": partition or "",
            "qos": qos or "",
            "nodes": 1,
            "cores": per_node_cpus,
            "time": walltime,
            "mem": memory,
            **({"gpu": gpus} if gpus else {}),
        }
        profile["cluster"] = {
            "bo": dict(distributed),
            "sample": dict(distributed),
            "nn": dict(single_python),
            "al": {
                **distributed,
                **({"gpu": gpus} if gpus else {}),
            },
            "audit": dict(distributed),
            "validate": dict(single_lammps),
            "finalize": {**single_python, "cores": 1},
        }
    return profile


def save_machine_profile(
    name: str,
    profile: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    path = machine_path(name)
    document = _machine_document()
    machines = document["machines"]
    if name in machines and not overwrite:
        raise FileExistsError(
            f"Machine profile {name!r} already exists in {path}. Pass --force to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value for key, value in profile.items() if not key.startswith("_")
    }
    machines[name] = serializable
    with path.open("wb") as handle:
        handle.write(tomli_w.dumps(document).encode("utf-8"))
    return path
