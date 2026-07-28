"""Worker process used by :mod:`workflow.slurm_pool`."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import uuid
from pathlib import Path
from typing import Any


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    os.replace(temporary, path)


def execute_request(data: dict[str, Any]) -> dict[str, Any]:
    environment = {**os.environ, **{
        str(key): str(value) for key, value in data.get("env", {}).items()
    }}
    try:
        result = subprocess.run(
            [str(item) for item in data["command"]],
            cwd=str(data["cwd"]),
            env=environment,
            capture_output=True,
            text=True,
            timeout=int(data.get("timeout", 21600)),
        )
        return {
            "returncode": result.returncode,
            "stdout": result.stdout,
            "stderr": result.stderr,
        }
    except subprocess.TimeoutExpired as exc:
        return {
            "returncode": 124,
            "stdout": exc.stdout or "",
            "stderr": f"TIMEOUT after {data.get('timeout')}s",
        }
    except Exception as exc:
        return {"returncode": 1, "stdout": "", "stderr": repr(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    rank = int(os.environ.get("SLURM_PROCID", "0"))
    request_dir = args.root / "requests" / f"worker_{rank:04d}"
    request_dir.mkdir(parents=True, exist_ok=True)
    _atomic_json(args.root / "ready" / f"worker_{rank:04d}.json", {
        "rank": rank,
        "hostname": os.uname().nodename,
        "cpus": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else [],
    })
    stop_path = args.root / "STOP"
    while not stop_path.exists():
        requests = sorted(request_dir.glob("*.json"))
        if not requests:
            time.sleep(0.05)
            continue
        request_path = requests[0]
        running_path = request_path.with_suffix(".running")
        try:
            os.replace(request_path, running_path)
        except FileNotFoundError:
            continue
        data = json.loads(running_path.read_text(encoding="utf-8"))
        response = execute_request(data)
        _atomic_json(args.root / "responses" / f"{data['id']}.json", response)
        running_path.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
