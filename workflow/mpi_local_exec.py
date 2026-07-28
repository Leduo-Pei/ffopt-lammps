"""Launch one MPI program locally inside an exclusive single-node SLURM step."""

from __future__ import annotations

import argparse
import socket
import subprocess


def build_command(
    launcher: str,
    ranks: int,
    command: list[str],
    *,
    hostname: str | None = None,
) -> list[str]:
    if not command:
        raise ValueError("MPI worker command is empty")
    host = hostname or socket.gethostname()
    return [
        launcher,
        "--prtemca", "plm", "ssh",
        "--host", host,
        "-n", str(ranks),
        "--oversubscribe",
        "--bind-to", "none",
        *command,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--launcher", required=True)
    parser.add_argument("--ranks", required=True, type=int)
    parser.add_argument("command", nargs=argparse.REMAINDER)
    args = parser.parse_args()
    command = args.command[1:] if args.command[:1] == ["--"] else args.command
    result = subprocess.run(build_command(args.launcher, args.ranks, command))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
