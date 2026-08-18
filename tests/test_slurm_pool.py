import sys

from workflow.slurm_pool import SlurmCommandPool
from workflow.slurm_worker import execute_request, request_command


def test_worker_executes_command_in_requested_directory(tmp_path):
    result = execute_request({
        "command": [sys.executable, "-c", "print('worker-ok')"],
        "cwd": str(tmp_path),
        "env": {"OMP_NUM_THREADS": "4"},
        "timeout": 10,
    })
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "worker-ok"


def test_pool_pickle_state_does_not_include_owner_process(tmp_path):
    pool = SlurmCommandPool(
        launcher="srun", root=tmp_path, workers=24, nodes=2,
        cpus_per_worker=4,
    )
    pool._process = object()
    state = pool.__getstate__()
    assert state["_process"] is None
    assert state["workers"] == 24


def test_worker_wraps_mpi_inside_its_allocated_node(monkeypatch):
    monkeypatch.setattr("workflow.slurm_worker.socket.gethostname", lambda: "node20")
    monkeypatch.setattr(
        "workflow.slurm_worker.os.sched_getaffinity",
        lambda _pid: {4, 5, 6, 7},
        raising=False,
    )
    command = request_command({
        "command": ["lmp", "-in", "in.bulk"],
        "mpi_launcher": "/opt/mpi/mpirun",
        "mpi_ranks": 4,
    })
    assert command[:8] == [
        "/opt/mpi/mpirun", "--prtemca", "plm", "ssh", "--host", "node20",
        "-n", "4",
    ]
    assert command[-6:] == [
        "taskset", "-c", "4,5,6,7", "lmp", "-in", "in.bulk",
    ]
