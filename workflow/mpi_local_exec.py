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

try:
    import fcntl
except ImportError:  # pragma: no cover - SLURM workers run on Unix
    fcntl = None


def build_command(
    launcher: str,
    ranks: int,
    command: list[str],
    *,
    hostname: str | None = None,
    cpu_list: str | None = None,
) -> list[str]:
    if not command:
        raise ValueError("MPI worker command is empty")
    host = hostname or socket.gethostname()
    application = ["taskset", "-c", cpu_list, *command] if cpu_list else command
    return [
        launcher,
        "--prtemca", "plm", "ssh",
        "--host", host,
        "-n", str(ranks),
        "--oversubscribe",
        "--bind-to", "none",
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
                cpu_list=selected_cpus,
            )
        result = subprocess.run(worker_command)
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
