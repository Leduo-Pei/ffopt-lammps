import sys
from concurrent.futures import Future

import engine.lammps_interface as lammps_interface
from engine.lammps_interface import EvalResult, LAMMPSRunner


def _runner(
    launcher: str,
    omp_threads: int = 1,
    scheduler_launcher: str | None = None,
) -> LAMMPSRunner:
    runner = object.__new__(LAMMPSRunner)
    runner.mpiexec = launcher
    runner.omp_threads = omp_threads
    runner.scheduler_launcher = scheduler_launcher
    runner.scheduler_node_count = 1
    runner.workers_per_node = 12
    return runner


def test_slurm_launcher_wraps_local_mpi_in_an_exact_single_node_step():
    prefix = _runner("/opt/mpi/mpirun", scheduler_launcher="srun")._mpi_prefix(4)
    assert prefix[:8] == [
        "srun", "--overlap", "--exact", "--nodes=1", "--ntasks=1",
        "--cpus-per-task", "4", sys.executable,
    ]
    assert prefix[8:] == [
        "-m", "workflow.mpi_local_exec", "--launcher", "/opt/mpi/mpirun",
        "--ranks", "4", "--slots", "12", "--",
    ]


def test_slurm_launcher_selects_node_from_evaluation_index():
    runner = _runner("/opt/mpi/mpirun", scheduler_launcher="srun")
    runner.scheduler_node_count = 2
    prefix = runner._mpi_prefix(4, "/run/round_1/eval_0003/bulk")
    assert "--relative=1" in prefix


def test_slurm_openmp_worker_skips_nested_mpi():
    runner = _runner(
        "/opt/mpi/mpirun", omp_threads=4, scheduler_launcher="srun"
    )
    runner.scheduler_node_count = 2
    prefix = runner._mpi_prefix(1, "/run/eval_0002/bulk")
    assert "--direct" in prefix
    assert "--launcher" not in prefix
    assert "--cpus-per-task" in prefix
    assert prefix[prefix.index("--cpus-per-task") + 1] == "4"


def test_regular_mpi_launcher_keeps_n_flag():
    assert _runner("/opt/mpi/bin/mpiexec")._mpi_prefix(4) == [
        "/opt/mpi/bin/mpiexec", "-n", "4",
    ]


def test_python311_linux_batch_uses_dynamic_fresh_processes(
    tmp_path, monkeypatch, capsys
):
    created = []

    class RecordingExecutor:
        def __init__(self, *, max_workers, **options):
            self.max_workers = max_workers
            self.options = options
            self.submissions = []
            created.append(self)

        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def submit(self, function, *args):
            self.submissions.append(args)
            future = Future()
            try:
                future.set_result(function(*args))
            except Exception as exc:
                future.set_exception(exc)
            return future

    runner = object.__new__(LAMMPSRunner)
    runner.ads_enabled = False
    runner._slab_pe = None
    runner.max_workers = 2
    runner._run_single = lambda params, *_: EvalResult(
        params=params,
        properties={"value": params["value"]},
        objective=float(params["value"]),
        success=True,
    )
    monkeypatch.setattr(lammps_interface.os, "name", "posix")
    monkeypatch.setattr(lammps_interface.sys, "version_info", (3, 11, 0))
    monkeypatch.setattr(lammps_interface, "ProcessPoolExecutor", RecordingExecutor)

    results = runner.evaluate_batch(
        [{"value": 1.0}, {"value": 2.0}, {"value": 3.0}], str(tmp_path)
    )

    assert [result.objective for result in results] == [1.0, 2.0, 3.0]
    assert len(created) == 1
    assert created[0].max_workers == 2
    assert created[0].options == {"max_tasks_per_child": 1}
    assert len(created[0].submissions) == 3
    output = capsys.readouterr().out
    assert "LAMMPS progress: 3/3 finished" in output
    assert "objective=3.000000" in output
