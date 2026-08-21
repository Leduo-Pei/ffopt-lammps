import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from workflow.cli import build_parser
from workflow.machine import (
    build_machine_profile,
    execute_machine_test,
    format_machine_probe,
    load_machine_profile,
    prepare_machine_test,
    probe_machine_environment,
    save_machine_profile,
)


def test_builtin_local_profile_is_listed_and_showable(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("FFOPT_CONFIG_DIR", str(tmp_path / "config"))
    parser = build_parser()

    args = parser.parse_args(["machine", "list"])
    args.function(args)
    assert "local [built-in, serial]" in capsys.readouterr().out

    args = parser.parse_args(["machine", "show", "--name", "local"])
    args.function(args)
    output = capsys.readouterr().out
    assert '"backend": "local"' in output
    assert '"max_workers": 1' in output


def test_machine_configure_rejects_ambiguous_parallel_mpi_before_saving(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FFOPT_CONFIG_DIR", str(tmp_path / "config"))
    args = build_parser().parse_args([
        "machine", "configure", "--name", "ambiguous", "--backend", "local",
        "--lammps", "/env/bin/lmp", "--mpi", "/env/bin/mpiexec",
        "--workers", "1", "--mpi-ranks", "2",
    ])
    with pytest.raises(SystemExit, match="--mpi-flavor"):
        args.function(args)
    assert not (tmp_path / "config" / "machines.toml").exists()


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


def test_slurm_probe_timeout_becomes_a_warning(monkeypatch):
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False)),
    )
    monkeypatch.setattr("workflow.machine._detect_lammps", lambda: "/env/bin/lmp")
    monkeypatch.setattr("workflow.machine._detect_mpi", lambda: "/env/bin/mpirun")
    monkeypatch.setattr(
        "workflow.machine.resolve_executable", lambda value: str(value)
    )
    monkeypatch.setattr("workflow.machine._version_line", lambda *_args: "test version")
    commands = {
        "sbatch": "/usr/bin/sbatch",
        "sinfo": "/usr/bin/sinfo",
        "srun": "/usr/bin/srun",
    }
    monkeypatch.setattr(
        "workflow.machine.shutil.which", lambda value: commands.get(value)
    )
    original_run = subprocess.run

    def timeout_sinfo(command, **kwargs):
        if command[0] == "/usr/bin/sinfo":
            raise subprocess.TimeoutExpired(command, 10)
        return original_run(command, **kwargs)

    monkeypatch.setattr("workflow.machine.subprocess.run", timeout_sinfo)

    report = probe_machine_environment()
    assert report["backend"] == "slurm"
    assert report["slurm"]["partitions"] == []
    assert "timed out" in report["slurm"]["probe_error"]
    assert "SLURM probe warning" in format_machine_probe(report)


def test_prepare_slurm_machine_test_uses_profile_resources(tmp_path, monkeypatch):
    monkeypatch.setenv("FFOPT_CONFIG_DIR", str(tmp_path / "config"))
    profile = build_machine_profile(
        name="cluster", backend="slurm", lammps="/env/bin/lmp",
        mpi="/env/bin/mpirun", mpi_flavor="openmpi",
        workers=4, ranks=4, nodes=2,
        cores=16, partition="CPU", memory_per_node="4G",
    )
    result = prepare_machine_test("cluster", profile)
    script = result["script"].read_text(encoding="ascii")
    assert "#SBATCH --partition=CPU" in script
    assert "#SBATCH --mem=4G" in script
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks=4" in script
    assert "#SBATCH --ntasks-per-node=2" in script
    assert result["command"][2:4] == ["workflow.machine_test_runner", "--root"]
    assert "#SBATCH --cpus-per-task=4" in script
    assert "export PYTHONNOUSERSITE=1" in script
    assert "workflow.machine_test_runner" in script
    assert "--workers 4" in script
    assert "--nodes 2" in script
    assert "--mpi-flavor openmpi" in script
    assert "FFOPT_MACHINE_TEST_OK" in result["input"].read_text(encoding="ascii")


def test_prepare_machine_test_propagates_explicit_intel_mpi_flavor(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FFOPT_CONFIG_DIR", str(tmp_path / "config"))
    profile = build_machine_profile(
        name="intel-cluster", backend="slurm", lammps="/env/bin/lmp",
        mpi="/env/bin/mpiexec", mpi_flavor="intelmpi",
        workers=1, ranks=2, nodes=1, cores=2,
    )
    result = prepare_machine_test("intel-cluster", profile)
    assert profile["lammps"]["mpi_flavor"] == "intelmpi"
    assert result["command"][3:7] == [
        "--launcher", "/env/bin/mpiexec",
        "--launcher-flavor", "intelmpi",
    ]


def test_single_node_multiworker_machine_test_exercises_full_pool(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FFOPT_CONFIG_DIR", str(tmp_path / "config"))
    profile = build_machine_profile(
        name="intel-76", backend="slurm", lammps="/env/bin/lmp",
        mpi="/env/bin/mpiexec", mpi_flavor="intelmpi",
        workers=38, ranks=2, nodes=1, cores=76, partition="CPU",
    )
    result = prepare_machine_test("intel-76", profile)
    script = result["script"].read_text(encoding="ascii")
    assert result["command"][2] == "workflow.machine_test_runner"
    assert "--workers" in result["command"]
    assert result["command"][result["command"].index("--workers") + 1] == "38"
    assert "--nodes" in result["command"]
    assert result["command"][result["command"].index("--nodes") + 1] == "1"
    assert "--mpi-flavor" in result["command"]
    assert result["command"][result["command"].index("--mpi-flavor") + 1] == "intelmpi"
    assert "#SBATCH --nodes=1" in script
    assert "#SBATCH --ntasks=38" in script
    assert "#SBATCH --ntasks-per-node=38" in script
    assert "#SBATCH --cpus-per-task=2" in script


def test_prepare_machine_test_rejects_ambiguous_mpi_without_flavor(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("FFOPT_CONFIG_DIR", str(tmp_path / "config"))
    profile = build_machine_profile(
        name="ambiguous-cluster", backend="slurm", lammps="/env/bin/lmp",
        mpi="/env/bin/mpiexec", workers=1, ranks=2, nodes=1, cores=2,
    )
    with pytest.raises(ValueError, match="will not guess"):
        prepare_machine_test("ambiguous-cluster", profile)


def test_missing_local_lammps_is_reported_as_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("FFOPT_CONFIG_DIR", str(tmp_path / "config"))
    profile = build_machine_profile(
        name="missing", backend="local", lammps=str(tmp_path / "missing-lmp")
    )
    result = execute_machine_test("missing", profile)
    assert result["status"] == "failed"
    assert result["output"]


def test_missing_sbatch_is_reported_as_failed(tmp_path, monkeypatch):
    monkeypatch.setenv("FFOPT_CONFIG_DIR", str(tmp_path / "config"))
    profile = build_machine_profile(
        name="cluster", backend="slurm", workers=1, ranks=1, cores=1,
    )
    monkeypatch.setattr(
        "workflow.machine.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("sbatch")),
    )
    result = execute_machine_test("cluster", profile)
    assert result["status"] == "failed"
    assert "sbatch" in result["output"]


def test_machine_profile_write_is_atomic(tmp_path, monkeypatch):
    monkeypatch.setenv("FFOPT_CONFIG_DIR", str(tmp_path / "config"))
    original = build_machine_profile(
        name="workstation", backend="local", workers=2,
    )
    path = save_machine_profile("workstation", original)
    original_bytes = path.read_bytes()
    updated = build_machine_profile(
        name="workstation", backend="local", workers=8,
    )

    monkeypatch.setattr(
        os,
        "replace",
        lambda *_args: (_ for _ in ()).throw(OSError("simulated interruption")),
    )
    with pytest.raises(OSError, match="simulated interruption"):
        save_machine_profile("workstation", updated, overwrite=True)

    assert path.read_bytes() == original_bytes
    assert load_machine_profile("workstation")["parallel"]["max_workers"] == 2
    assert not list(path.parent.glob(".machines.toml.*.tmp"))


def test_malformed_machine_toml_has_recovery_hint(tmp_path, monkeypatch):
    config = tmp_path / "config"
    config.mkdir()
    (config / "machines.toml").write_text("[machines.local\n", encoding="utf-8")
    monkeypatch.setenv("FFOPT_CONFIG_DIR", str(config))

    with pytest.raises(ValueError, match="Repair or move this file"):
        load_machine_profile("local")


def test_slurm_evaluation_timeout_must_fit_inside_walltime():
    with pytest.raises(ValueError, match="timeout must be shorter"):
        build_machine_profile(
            name="bad-timeout",
            backend="slurm",
            workers=1,
            ranks=1,
            nodes=1,
            cores=1,
            walltime="06:00:00",
            timeout=216000,
        )


def test_slurm_profile_rejects_fewer_workers_than_nodes():
    with pytest.raises(ValueError, match="workers must be at least the node count"):
        build_machine_profile(
            name="idle-node",
            backend="slurm",
            workers=1,
            ranks=4,
            nodes=2,
            cores=8,
        )


@pytest.mark.parametrize("name", ["..", ".", "two nodes", "node/one", "1node"])
def test_machine_profile_names_are_portable(name):
    with pytest.raises(ValueError, match="Machine profile name"):
        build_machine_profile(name=name, backend="local")
