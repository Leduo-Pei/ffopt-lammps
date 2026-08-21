import pytest

from workflow.mpi_local_exec import (
    build_command,
    build_direct_command,
    launcher_flavor_from_version,
    resolve_launcher_flavor,
)


def test_local_mpi_worker_forces_ranks_onto_its_slurm_node():
    command = build_command(
        "/opt/mpi/mpirun", 4, ["/opt/lammps/lmp", "-in", "in.bulk"],
        launcher_flavor="openmpi",
        hostname="node20", cpu_list="8,9,10,11",
    )
    assert command == [
        "/opt/mpi/mpirun", "--prtemca", "plm", "ssh", "--host", "node20",
        "-n", "4", "--oversubscribe", "--bind-to", "none",
        "taskset", "-c", "8,9,10,11", "/opt/lammps/lmp", "-in", "in.bulk",
    ]


def test_intel_mpi_worker_uses_hydra_local_fork_without_openmpi_flags():
    command = build_command(
        "/opt/intel/oneapi/mpi/latest/bin/mpiexec", 2,
        ["/opt/lammps/lmp", "-in", "in.bulk"],
        launcher_flavor="intelmpi", hostname="node20", cpu_list="8,9",
    )
    assert command == [
        "/opt/intel/oneapi/mpi/latest/bin/mpiexec",
        "-launcher", "fork", "-hosts", "node20",
        "-genv", "I_MPI_PIN", "0", "-n", "2",
        "taskset", "-c", "8,9", "/opt/lammps/lmp", "-in", "in.bulk",
    ]
    assert "--prtemca" not in command
    assert "--oversubscribe" not in command


def test_ambiguous_launcher_fails_closed_without_explicit_flavor():
    with pytest.raises(ValueError, match="will not guess"):
        resolve_launcher_flavor("/usr/bin/mpiexec")


@pytest.mark.parametrize(
    ("launcher", "expected"),
    [
        ("/opt/openmpi/bin/mpirun", "openmpi"),
        ("/opt/intel/oneapi/mpi/latest/bin/mpiexec", "intelmpi"),
        ("mpiexec.hydra", "intelmpi"),
    ],
)
def test_unambiguous_launcher_paths_are_reproducibly_inferred(launcher, expected):
    assert resolve_launcher_flavor(launcher) == expected


@pytest.mark.parametrize(
    ("output", "expected"),
    [
        ("mpiexec (OpenRTE) 4.1.6", "openmpi"),
        ("Intel(R) MPI Library for Linux* OS, Version 2021.15", "intelmpi"),
        ("HYDRA build details: MPICH Version 4.2", None),
    ],
)
def test_version_probe_only_accepts_known_compatible_vendors(output, expected):
    assert launcher_flavor_from_version(output) == expected


def test_direct_worker_applies_its_cpu_slot_without_mpi():
    assert build_direct_command(["lmp", "-sf", "omp"], "4,5,6,7") == [
        "taskset", "-c", "4,5,6,7", "lmp", "-sf", "omp",
    ]
