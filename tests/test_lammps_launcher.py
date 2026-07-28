import sys

from engine.lammps_interface import LAMMPSRunner


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
