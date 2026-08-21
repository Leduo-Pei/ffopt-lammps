"""Launch one MPI program locally inside an exclusive single-node SLURM step."""

from __future__ import annotations

import argparse
import contextlib
import os
import socket
import subprocess
import tempfile
import time
from collections.abc import Iterator
from pathlib import Path

try:
    import fcntl
except ImportError:  # pragma: no cover - SLURM workers run on Unix
    fcntl = None


_FLAVOR_ALIASES = {
    "openmpi": "openmpi",
    "open-mpi": "openmpi",
    "ompi": "openmpi",
    "intelmpi": "intelmpi",
    "intel-mpi": "intelmpi",
    "impi": "intelmpi",
    "hydra": "intelmpi",
}


def launcher_flavor_from_version(version_output: str) -> str | None:
    """Parse a probe result without treating an unknown vendor as compatible."""
    value = str(version_output).lower()
    if "open mpi" in value or "openrte" in value or "open rte" in value:
        return "openmpi"
    if "intel(r) mpi" in value or "intel mpi" in value:
        return "intelmpi"
    return None


def resolve_launcher_flavor(
    launcher: str,
    launcher_flavor: str | None = None,
) -> str:
    """Return a deterministic MPI command-line dialect or fail closed.

    Plain ``mpiexec`` and ``mpirun`` names are shared by several incompatible
    MPI implementations.  We therefore accept an explicit profile value and
    only infer a flavor from unambiguous executable/path markers.  In
    particular, this function never shells out to whichever MPI happens to be
    first on ``PATH`` during a resumed run.
    """
    if launcher_flavor is not None and str(launcher_flavor).strip():
        value = str(launcher_flavor).strip().lower()
        try:
            return _FLAVOR_ALIASES[value]
        except KeyError as exc:
            supported = "openmpi, intelmpi"
            raise ValueError(
                f"Unsupported MPI launcher flavor {launcher_flavor!r}; "
                f"choose one of: {supported}"
            ) from exc

    normalized = str(launcher).strip().replace("\\", "/").lower()
    executable = Path(normalized).name
    if executable in {"orterun", "prterun"} or any(
        marker in f"/{normalized.strip('/')}/"
        for marker in ("/openmpi/", "/open-mpi/")
    ):
        return "openmpi"
    if executable in {"mpiexec.hydra", "mpirun.hydra"} or (
        "/intel/" in f"/{normalized.strip('/')}/"
        and "/mpi/" in f"/{normalized.strip('/')}/"
    ) or "/oneapi/mpi/" in f"/{normalized.strip('/')}/":
        return "intelmpi"
    raise ValueError(
        "Cannot determine the MPI launcher flavor from "
        f"{launcher!r}. Set lammps.mpi_flavor to 'openmpi' or 'intelmpi' "
        "in the machine profile. FFOpt will not guess for an ambiguous "
        "mpiexec/mpirun executable."
    )


def build_command(
    launcher: str,
    ranks: int,
    command: list[str],
    *,
    launcher_flavor: str | None = None,
    hostname: str | None = None,
    cpu_list: str | None = None,
) -> list[str]:
    if not command:
        raise ValueError("MPI worker command is empty")
    if ranks < 1:
        raise ValueError("MPI rank count must be positive")
    host = hostname or socket.gethostname()
    application = ["taskset", "-c", cpu_list, *command] if cpu_list else command
    flavor = resolve_launcher_flavor(launcher, launcher_flavor)
    if flavor == "openmpi":
        return [
            launcher,
            "--prtemca", "plm", "ssh",
            "--host", host,
            "-n", str(ranks),
            "--oversubscribe",
            "--bind-to", "none",
            *application,
        ]
    return [
        launcher,
        "-launcher", "fork",
        "-hosts", host,
        "-genv", "I_MPI_PIN", "0",
        "-n", str(ranks),
        *application,
    ]


def build_direct_command(command: list[str], cpu_list: str | None = None) -> list[str]:
    if not command:
        raise ValueError("Direct worker command is empty")
    return ["taskset", "-c", cpu_list, *command] if cpu_list else command


@contextlib.contextmanager
def cpu_slot(slots: int, cpus_per_slot: int) -> Iterator[str | None]:
    """Reserve one node-local CPU slot for the lifetime of an MPI process."""
    if fcntl is None or not hasattr(os, "sched_getaffinity"):
        yield None
        return
    available = sorted(os.sched_getaffinity(0))
    slot_count = min(max(1, slots), max(1, len(available) // cpus_per_slot))
    job = os.environ.get("SLURM_JOB_ID", "manual")
    host = socket.gethostname().split(".", 1)[0]
    lock_root = os.path.join(tempfile.gettempdir(), f"ffopt-{job}-{host}-slots")
    os.makedirs(lock_root, exist_ok=True)
    handles = []
    try:
        while True:
            for index in range(slot_count):
                handle = open(os.path.join(lock_root, f"slot_{index:03d}.lock"), "a+")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                handles.append(handle)
                start = index * cpus_per_slot
                selected = available[start:start + cpus_per_slot]
                yield ",".join(str(cpu) for cpu in selected)
                return
            time.sleep(0.1)
    finally:
        for handle in handles:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
            handle.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher")
    parser.add_argument("--launcher-flavor", choices=("openmpi", "intelmpi"))
    parser.add_argument("--ranks", default=1, type=int)
    parser.add_argument("--direct", action="store_true")
    parser.add_argument("--slots", default=1, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    if not args.direct and not args.launcher:
        parser.error("--launcher is required unless --direct is used")
    omp_threads = max(1, int(os.environ.get("OMP_NUM_THREADS", "1")))
    with cpu_slot(args.slots, args.ranks * omp_threads) as selected_cpus:
        if args.direct:
            worker_command = build_direct_command(command, selected_cpus)
        else:
            worker_command = build_command(
                args.launcher,
                args.ranks,
                command,
                launcher_flavor=args.launcher_flavor,
                cpu_list=selected_cpus,
            )
        result = subprocess.run(worker_command)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
