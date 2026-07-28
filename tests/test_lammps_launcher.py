from engine.lammps_interface import LAMMPSRunner


def _runner(launcher: str, omp_threads: int = 1) -> LAMMPSRunner:
    runner = object.__new__(LAMMPSRunner)
    runner.mpiexec = launcher
    runner.omp_threads = omp_threads
    return runner


def test_srun_launcher_reserves_an_exclusive_single_node_step():
    prefix = _runner("srun")._mpi_prefix(4)
    assert prefix == [
        "srun", "--exclusive", "--nodes=1", "--ntasks", "4",
        "--cpus-per-task", "1",
    ]


def test_regular_mpi_launcher_keeps_n_flag():
    assert _runner("/opt/mpi/bin/mpiexec")._mpi_prefix(4) == [
        "/opt/mpi/bin/mpiexec", "-n", "4",
    ]
