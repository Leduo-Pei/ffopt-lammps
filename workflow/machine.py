"""User-level execution profiles for local workstations and schedulers."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import yaml


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
    return config_home() / "machines"


def machine_path(name: str) -> Path:
    if not name or any(char in name for char in "\\/:"):
        raise ValueError(f"Invalid machine profile name: {name!r}")
    return machine_dir() / f"{name}.yaml"


def load_machine_profile(name: str) -> dict[str, Any] | None:
    path = machine_path(name)
    if not path.exists():
        return None
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Machine profile must contain a YAML mapping: {path}")
    data.setdefault("machine", {})["name"] = name
    data["_machine_profile_path"] = str(path)
    return data


def available_machine_profiles() -> list[str]:
    directory = machine_dir()
    if not directory.exists():
        return []
    return sorted(path.stem for path in directory.glob("*.yaml"))


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
    walltime: str = "24:00:00",
) -> dict[str, Any]:
    if backend not in {"local", "slurm"}:
        raise ValueError("backend must be 'local' or 'slurm'")
    cpu_count = max(1, os.cpu_count() or 1)
    ranks = max(1, int(ranks))
    omp_threads = max(1, int(omp_threads))
    workers = max(1, int(workers or max(1, cpu_count // (ranks * omp_threads))))
    profile: dict[str, Any] = {
        "machine": {
            "name": name,
            "backend": backend,
            "extends": "cluster" if backend == "slurm" else "local",
        },
        "lammps": {
            "executable": lammps or _detect_lammps(),
            "mpiexec": mpi or ("srun" if backend == "slurm" else _detect_mpi()),
            "timeout": int(timeout),
        },
        "parallel": {
            "max_workers": workers,
            "cores_per_worker": ranks,
            "omp_threads_per_worker": omp_threads,
            "use_mpi": ranks > 1 or backend == "slurm",
        },
        "machine_learning": {"device": "auto"},
    }
    if backend == "slurm":
        profile["cluster"] = {
            stage: {
                "partition": partition or "",
                "qos": qos or "",
                "nodes": 1,
                "cores": int(cores or workers * ranks),
                "time": walltime,
                "mem": "0",
                **({"gpu": 1} if stage in {"nn", "al"} else {}),
            }
            for stage in ("bo", "nn", "al")
        }
    return profile


def save_machine_profile(
    name: str,
    profile: dict[str, Any],
    *,
    overwrite: bool = False,
) -> Path:
    path = machine_path(name)
    if path.exists() and not overwrite:
        raise FileExistsError(
            f"Machine profile already exists: {path}. Pass --force to replace it."
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value for key, value in profile.items() if not key.startswith("_")
    }
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# User-specific FFOpt execution profile.\n")
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)
    return path

