"""Persistent SLURM worker pool for concurrent external commands."""

from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import fcntl
except ImportError:  # pragma: no cover - SLURM is Unix-only
    fcntl = None


@dataclass(frozen=True)
class CommandResult:
    returncode: int
    stdout: str
    stderr: str


def _atomic_json(path: Path, data: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + f".{uuid.uuid4().hex}.tmp")
    temporary.write_text(json.dumps(data), encoding="utf-8")
    os.replace(temporary, path)


class SlurmCommandPool:
    """Dispatch commands to one persistent worker per allocated task."""

    def __init__(
        self,
        *,
        launcher: str,
        root: Path,
        workers: int,
        nodes: int,
        cpus_per_worker: int,
        startup_timeout: int = 120,
    ) -> None:
        self.launcher = launcher
        self.root = root
        self.workers = max(1, int(workers))
        self.nodes = max(1, int(nodes))
        self.cpus_per_worker = max(1, int(cpus_per_worker))
        self.startup_timeout = max(10, int(startup_timeout))
        self._process: subprocess.Popen[str] | None = None
        self._stdout_handle = None
        self._stderr_handle = None

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "SlurmCommandPool | None":
        parallel = config.get("parallel", {})
        launcher = parallel.get("scheduler_launcher")
        job_id = os.environ.get("SLURM_JOB_ID")
        if not launcher or not job_id:
            return None
        if fcntl is None:
            raise RuntimeError("SLURM command pools require Unix file locking")
        workers = int(os.environ.get(
            "SLURM_NTASKS", parallel.get("max_workers", 1)
        ))
        nodes = int(os.environ.get(
            "SLURM_JOB_NUM_NODES", parallel.get("scheduler_nodes", 1)
        ))
        cpus = int(parallel.get("cores_per_worker", 1)) * int(
            parallel.get("omp_threads_per_worker", 1)
        )
        run_root = Path(config.get("workflow", {}).get("run_root", Path.cwd()))
        pool = cls(
            launcher=str(launcher),
            root=run_root / "_scheduler" / f"job_{job_id}",
            workers=workers,
            nodes=nodes,
            cpus_per_worker=cpus,
        )
        pool.start()
        return pool

    def __getstate__(self) -> dict[str, Any]:
        state = dict(self.__dict__)
        state["_process"] = None
        state["_stdout_handle"] = None
        state["_stderr_handle"] = None
        return state

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        for name in ("ready", "requests", "responses", "slots"):
            (self.root / name).mkdir(exist_ok=True)
        stop_path = self.root / "STOP"
        if stop_path.exists():
            stop_path.unlink()
        for rank in range(self.workers):
            (self.root / "requests" / f"worker_{rank:04d}").mkdir(exist_ok=True)

        self._stdout_handle = (self.root / "workers.out").open("a", encoding="utf-8")
        self._stderr_handle = (self.root / "workers.err").open("a", encoding="utf-8")
        command = [
            self.launcher,
            "--exact",
            f"--nodes={self.nodes}",
            f"--ntasks={self.workers}",
            f"--cpus-per-task={self.cpus_per_worker}",
            "--distribution=block:block",
            sys.executable,
            "-m", "workflow.slurm_worker",
            "--root", str(self.root),
        ]
        self._process = subprocess.Popen(
            command,
            stdout=self._stdout_handle,
            stderr=self._stderr_handle,
            text=True,
        )
        deadline = time.monotonic() + self.startup_timeout
        while time.monotonic() < deadline:
            ready = list((self.root / "ready").glob("worker_*.json"))
            if len(ready) >= self.workers:
                atexit.register(self.close)
                return
            if self._process.poll() is not None:
                break
            time.sleep(0.1)
        self.close(force=True)
        error_path = self.root / "workers.err"
        detail = error_path.read_text(encoding="utf-8", errors="replace")[-4000:]
        raise RuntimeError(
            f"SLURM command workers did not become ready: {detail.strip()}"
        )

    def _acquire_slot(self, deadline: float):
        while time.monotonic() < deadline:
            for rank in range(self.workers):
                path = self.root / "slots" / f"worker_{rank:04d}.lock"
                handle = path.open("a+")
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError:
                    handle.close()
                    continue
                return rank, handle
            time.sleep(0.05)
        raise TimeoutError("Timed out waiting for a free SLURM command worker")

    def run(
        self,
        command: list[str],
        *,
        cwd: str,
        env: dict[str, str],
        timeout: int,
    ) -> CommandResult:
        deadline = time.monotonic() + max(1, int(timeout))
        rank, lock_handle = self._acquire_slot(deadline)
        request_id = uuid.uuid4().hex
        request_path = (
            self.root / "requests" / f"worker_{rank:04d}" / f"{request_id}.json"
        )
        response_path = self.root / "responses" / f"{request_id}.json"
        try:
            _atomic_json(request_path, {
                "id": request_id,
                "command": [str(item) for item in command],
                "cwd": str(cwd),
                "env": {str(key): str(value) for key, value in env.items()},
                "timeout": int(timeout),
            })
            while time.monotonic() < deadline:
                if response_path.exists():
                    data = json.loads(response_path.read_text(encoding="utf-8"))
                    response_path.unlink()
                    return CommandResult(
                        int(data.get("returncode", 1)),
                        str(data.get("stdout", "")),
                        str(data.get("stderr", data.get("error", ""))),
                    )
                time.sleep(0.05)
            return CommandResult(124, "", f"TIMEOUT after {timeout}s")
        finally:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
            lock_handle.close()

    def close(self, *, force: bool = False) -> None:
        if self._process is None:
            return
        (self.root / "STOP").touch()
        try:
            self._process.wait(timeout=2 if force else 30)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
        self._process = None
        for handle_name in ("_stdout_handle", "_stderr_handle"):
            handle = getattr(self, handle_name)
            if handle is not None:
                handle.close()
                setattr(self, handle_name, None)
