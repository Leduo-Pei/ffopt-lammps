import sys
import json

from workflow.slurm_pool import SlurmCommandPool, _write_shared_json
from workflow.slurm_worker import execute_request, request_command
from workflow.machine_test_runner import audit_worker_affinities


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


def test_persistent_worker_step_binds_each_slurm_task_to_cores(tmp_path):
    pool = SlurmCommandPool(
        launcher="srun", root=tmp_path, workers=38, nodes=1,
        cpus_per_worker=2,
    )
    command = pool._worker_launch_command()
    assert "--ntasks=38" in command
    assert "--cpus-per-task=2" in command
    assert "--cpu-bind=cores" in command
    assert "--ntasks-per-node=38" in command


def test_machine_acceptance_allows_disjoint_worker_affinities():
    records = [
        {"rank": 0, "hostname": "node20", "cpus": [0, 1]},
        {"rank": 1, "hostname": "node20", "cpus": [2, 3]},
        {"rank": 2, "hostname": "node21", "cpus": [0, 1]},
    ]
    assert audit_worker_affinities(
        records, expected_workers=3, cpus_per_worker=2
    ) == []


def test_machine_acceptance_rejects_overlapping_or_small_affinities():
    records = [
        {"rank": 0, "hostname": "node20", "cpus": [0, 1]},
        {"rank": 1, "hostname": "node20", "cpus": [1]},
    ]
    errors = audit_worker_affinities(
        records, expected_workers=3, cpus_per_worker=2
    )
    assert any("expected 3 ready workers" in error for error in errors)
    assert any("requires at least 2" in error for error in errors)
    assert any("share CPUs [1]" in error for error in errors)


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
        "mpi_launcher_flavor": "openmpi",
        "mpi_ranks": 4,
    })
    assert command[:8] == [
        "/opt/mpi/mpirun", "--prtemca", "plm", "ssh", "--host", "node20",
        "-n", "4",
    ]
    assert command[-6:] == [
        "taskset", "-c", "4,5,6,7", "lmp", "-in", "in.bulk",
    ]


def test_worker_uses_intel_hydra_dialect_when_profile_selects_it(monkeypatch):
    monkeypatch.setattr("workflow.slurm_worker.socket.gethostname", lambda: "node20")
    monkeypatch.setattr(
        "workflow.slurm_worker.os.sched_getaffinity",
        lambda _pid: {4, 5},
        raising=False,
    )
    command = request_command({
        "command": ["lmp", "-in", "in.bulk"],
        "mpi_launcher": "/opt/intel/oneapi/mpi/latest/bin/mpiexec",
        "mpi_launcher_flavor": "intelmpi",
        "mpi_ranks": 2,
    })
    assert command[:10] == [
        "/opt/intel/oneapi/mpi/latest/bin/mpiexec",
        "-launcher", "fork", "-hosts", "node20",
        "-genv", "I_MPI_PIN", "0", "-n", "2",
    ]
    assert "--prtemca" not in command


def test_shared_json_publishes_the_final_name_without_a_rename(tmp_path, monkeypatch):
    destination = tmp_path / "request.json"
    monkeypatch.setattr(
        "workflow.slurm_pool.os.replace",
        lambda *_: (_ for _ in ()).throw(AssertionError("rename is not NFS-safe")),
    )

    _write_shared_json(destination, {"id": "request-1", "value": 3})

    assert json.loads(destination.read_text(encoding="utf-8")) == {
        "id": "request-1",
        "value": 3,
    }
    assert list(tmp_path.glob("*.tmp")) == []
