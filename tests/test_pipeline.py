from pathlib import Path

from workflow.pipeline import PipelineRunner
from workflow.project import Project


def _project(tmp_path: Path) -> Project:
    path = tmp_path / "project.yaml"
    path.write_text("project: {name: demo}\nmodules: {}\n", encoding="utf-8")
    return Project(path, {
        "project": {"name": "demo", "run_root": "runs/demo"},
        "modules": {},
        "pipeline": {"stages": ["bo", "sample"]},
        "stages": {
            "sample": {
                "n_points": 12,
                "elite_centers": 2,
                "radii": [0.01],
                "global_fraction": 0.0,
                "seeds": [101],
            }
        },
    })


def _config():
    return {
        "machine": {"backend": "local"},
        "manifest": {"system_name": "demo"},
        "lammps": {"executable": "lmp", "mpiexec": "mpiexec"},
        "optimization": {"method": "turbo"},
        "checkpoint": {"enabled": True},
        "nn": {"enabled": True},
        "active_learning": {"enabled": True, "n_rounds": 2},
    }


def test_pipeline_paths_and_commands_are_deterministic(tmp_path, monkeypatch):
    project = _project(tmp_path)
    config_path = tmp_path / "expanded.yaml"
    config_path.write_text("manifest: {system_name: demo}\n", encoding="utf-8")
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: _config())

    runner = PipelineRunner(
        project=project, machine="local", config_path=config_path,
        run_id="trial", dry_run=True,
    )
    specs = runner.build_specs()
    assert [spec.name for spec in specs] == ["bo", "sample"]
    assert specs[0].output_dir == project.run_root / "pipelines" / "trial" / "bo"
    assert "--n-points" in specs[1].command
    assert specs[1].command[specs[1].command.index("--n-points") + 1] == "12"


def test_pipeline_reuses_complete_stage_and_invalidates_downstream_only(
    tmp_path, monkeypatch
):
    project = _project(tmp_path)
    config_path = tmp_path / "expanded.yaml"
    config_path.write_text("manifest: {system_name: demo}\n", encoding="utf-8")
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: _config())
    calls = []

    def fake_run_local(self, state, spec):
        calls.append(spec.name)
        state.transition(spec.name, "running", increment_attempt=True)
        for artifact in spec.artifacts:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("test\n", encoding="utf-8")
        state.transition(spec.name, "completed")

    monkeypatch.setattr(PipelineRunner, "_run_local", fake_run_local)
    first = PipelineRunner(
        project=project, machine="local", config_path=config_path, run_id="trial"
    )
    assert first.run() == "completed"
    assert calls == ["bo", "sample"]

    calls.clear()
    second = PipelineRunner(
        project=project, machine="local", config_path=config_path,
        run_id="trial", resume=True,
    )
    assert second.run() == "completed"
    assert calls == []

    project.data["stages"]["sample"]["n_points"] = 24
    third = PipelineRunner(
        project=project, machine="local", config_path=config_path,
        run_id="trial", resume=True,
    )
    assert third.run() == "completed"
    assert calls == ["sample"]


def test_slurm_script_omits_empty_optional_directives(tmp_path, monkeypatch):
    project = _project(tmp_path)
    config_path = tmp_path / "expanded.yaml"
    config_path.write_text("manifest: {system_name: demo}\n", encoding="utf-8")
    config = _config()
    config["machine"]["backend"] = "slurm"
    config["cluster"] = {
        "env_setup": ["source ~/.bashrc"],
        "bo": {
            "partition": "",
            "qos": "",
            "nodes": 1,
            "cores": 4,
            "time": "12:00:00",
            "mem": "0",
        },
    }
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(
        project=project, machine="cluster", config_path=config_path,
        run_id="trial", dry_run=True,
    )
    script = runner._write_slurm_script(runner.build_specs()[0]).read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=" not in script
    assert "#SBATCH --qos=" not in script
    assert "#SBATCH --mem=0" not in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "source ~/.bashrc" in script
