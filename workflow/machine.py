"""User-level execution profiles for local workstations and schedulers."""

from __future__ import annotations

import math
import os
import shutil
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
        raise ValueError(f"Invalid FFOpt machine configuration: {path}")
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
        raise ValueError(f"Machine profile must contain a mapping: {path}")
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
