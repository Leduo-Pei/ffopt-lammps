import json
import sys

from workflow.slurm_pool import SlurmCommandPool
from workflow.slurm_worker import execute_request


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
