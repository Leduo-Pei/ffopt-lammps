from workflow.mpi_local_exec import build_command


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
