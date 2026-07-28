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
    return runner


def test_slurm_launcher_wraps_local_mpi_in_an_exact_single_node_step():
    prefix = _runner("/opt/mpi/mpirun", scheduler_launcher="srun")._mpi_prefix(4)
    assert prefix[:7] == [
        "srun", "--exact", "--nodes=1", "--ntasks=1",
        "--cpus-per-task", "4", sys.executable,
    ]
    assert prefix[7:] == [
        "-m", "workflow.mpi_local_exec", "--launcher", "/opt/mpi/mpirun",
        "--ranks", "4", "--",
    ]


def test_regular_mpi_launcher_keeps_n_flag():
    assert _runner("/opt/mpi/bin/mpiexec")._mpi_prefix(4) == [
        "/opt/mpi/bin/mpiexec", "-n", "4",
    ]
