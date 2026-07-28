from workflow.mpi_local_exec import build_command, build_direct_command


def test_local_mpi_worker_forces_ranks_onto_its_slurm_node():
    command = build_command(
        "/opt/mpi/mpirun", 4, ["/opt/lammps/lmp", "-in", "in.bulk"],
        hostname="node20", cpu_list="8,9,10,11",
    )
    assert command == [
        "/opt/mpi/mpirun", "--prtemca", "plm", "ssh", "--host", "node20",
        "-n", "4", "--oversubscribe", "--bind-to", "none",
        "taskset", "-c", "8,9,10,11", "/opt/lammps/lmp", "-in", "in.bulk",
    ]


def test_direct_worker_applies_its_cpu_slot_without_mpi():
    assert build_direct_command(["lmp", "-sf", "omp"], "4,5,6,7") == [
        "taskset", "-c", "4,5,6,7", "lmp", "-sf", "omp",
    ]
