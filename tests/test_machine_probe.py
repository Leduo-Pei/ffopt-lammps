from workflow.machine import (
    build_machine_profile,
    execute_machine_test,
    format_machine_probe,
    prepare_machine_test,
)


def test_format_machine_probe_with_slurm_recommendation():
    text = format_machine_probe({
        "host": "node",
        "backend": "slurm",
        "logical_cpus": 48,
        "conda_environment": "ffopt-cpu",
        "python": "/env/bin/python",
        "lammps": {"path": "/env/bin/lmp", "version": "LAMMPS test"},
        "mpi": {"path": "/env/bin/mpirun", "version": "Open MPI test"},
        "torch_cuda": {"available": False, "device": None},
        "slurm": {"partitions": [{
            "name": "CPU", "default": True, "cores_per_node": 48,
            "memory_mb_per_node": 380000, "availability": "up",
            "time_limit": "14-00:00:00", "nodes": 25, "state": "mix",
        }]},
        "recommendation": {
            "partition": "CPU", "nodes": 1, "cores": 48,
            "workers": 12, "ranks": 4, "omp_threads": 1,
            "memory_per_node": "64G", "note": "Review first.",
        },
    })
    assert "SLURM partitions" in text
    assert "workers=12" in text


def test_prepare_slurm_machine_test_uses_profile_resources(tmp_path, monkeypatch):
    monkeypatch.setenv("FFOPT_CONFIG_DIR", str(tmp_path / "config"))
    profile = build_machine_profile(
        name="cluster", backend="slurm", lammps="/env/bin/lmp",
        mpi="/env/bin/mpirun", workers=4, ranks=4, nodes=2,
        cores=16, partition="CPU", memory_per_node="4G",
    )
    result = prepare_machine_test("cluster", profile)
    script = result["script"].read_text(encoding="ascii")
    assert "#SBATCH --partition=CPU" in script
    assert "#SBATCH --mem=4G" in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "mpirun" in script
    assert "workflow.mpi_local_exec" in script
    assert "--slots 1" in script
    assert "FFOPT_MACHINE_TEST_OK" in result["input"].read_text(encoding="ascii")


def test_missing_local_lammps_is_reported_as_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("FFOPT_CONFIG_DIR", str(tmp_path / "config"))
    profile = build_machine_profile(
        name="missing", backend="local", lammps=str(tmp_path / "missing-lmp")
    )
    result = execute_machine_test("missing", profile)
    assert result["status"] == "failed"
    assert result["output"]
