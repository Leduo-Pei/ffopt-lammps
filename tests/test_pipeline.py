import json
from pathlib import Path

import pytest

from workflow.pipeline import PipelineRunner
from workflow.project import Project
from workflow.state import WorkflowState


def _project(tmp_path: Path) -> Project:
    path = tmp_path / "ffopt.in"
    path.write_text("ffopt 1\nproject demo\n", encoding="utf-8")
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
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: _config())

    runner = PipelineRunner(
        project=project, machine="local", run_id="trial", dry_run=True,
    )
    specs = runner.build_specs()
    assert [spec.name for spec in specs] == ["bo", "sample"]
    assert specs[0].output_dir == project.run_root / "pipelines" / "trial" / "bo"
    assert "--n-points" in specs[1].command
    assert specs[1].command[specs[1].command.index("--n-points") + 1] == "12"
    assert runner.config_path.parent == runner.root / "provenance"
    assert runner.run() == "dry-run"
    assert not runner.root.exists()


def test_real_run_writes_immutable_provenance(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: _config())

    def fake_run_local(self, state, spec):
        state.transition(spec.name, "running", increment_attempt=True)
        for artifact in spec.artifacts:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("test\n", encoding="utf-8")
        state.transition(spec.name, "completed")

    monkeypatch.setattr(PipelineRunner, "_run_local", fake_run_local)
    runner = PipelineRunner(project=project, machine="local", run_id="provenance")
    assert runner.run() == "completed"
    assert runner.config_path.exists()
    assert runner.input_snapshot_path.read_bytes() == project.path.read_bytes()
    environment = json.loads(runner.environment_path.read_text(encoding="utf-8"))
    assert environment["machine_profile"] == "local"


def test_validate_only_uses_initial_input_parameters(tmp_path, monkeypatch):
    project = _project(tmp_path)
    project.data["pipeline"]["stages"] = ["validate"]
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: _config())

    runner = PipelineRunner(
        project=project,
        machine="local",
        run_id="initial_validation",
        dry_run=True,
    )
    command = runner.build_specs()[0].command
    assert "--initial" in command
    assert "--parameters" not in command


def test_pipeline_reuses_complete_stage_and_invalidates_downstream_only(
    tmp_path, monkeypatch
):
    project = _project(tmp_path)
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
        project=project, machine="local", run_id="trial"
    )
    assert first.run() == "completed"
    assert calls == ["bo", "sample"]

    calls.clear()
    second = PipelineRunner(
        project=project, machine="local", run_id="trial", resume=True,
    )
    assert second.run() == "completed"
    assert calls == []

    project.data["stages"]["sample"]["n_points"] = 24
    third = PipelineRunner(
        project=project, machine="local", run_id="trial", resume=True,
    )
    assert third.run() == "completed"
    assert calls == ["sample"]


def test_pipeline_refuses_cross_version_resume_and_snapshots_each_environment(
    tmp_path, monkeypatch
):
    project = _project(tmp_path)
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: _config())

    def fake_run_local(self, state, spec):
        state.transition(spec.name, "running", increment_attempt=True)
        for artifact in spec.artifacts:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("test\n", encoding="utf-8")
        state.transition(spec.name, "completed")

    monkeypatch.setattr(PipelineRunner, "_run_local", fake_run_local)
    first = PipelineRunner(project=project, machine="local", run_id="versioned")
    assert first.run() == "completed"

    monkeypatch.setattr("workflow.pipeline.__version__", "99.0.0")
    upgraded = PipelineRunner(
        project=project, machine="local", run_id="versioned", resume=True
    )
    assert upgraded.environment_path != first.environment_path
    with pytest.raises(RuntimeError, match="started with FFOpt.*use --new"):
        upgraded.run()
    assert not upgraded.environment_path.exists()


def test_from_stage_requires_completed_upstream_artifacts(tmp_path, monkeypatch):
    project = _project(tmp_path)
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: _config())
    runner = PipelineRunner(
        project=project,
        machine="local",
        run_id="jump",
        from_stage="sample",
    )
    with pytest.raises(RuntimeError, match="upstream stage 'bo' is not complete"):
        runner.run()


def test_from_stage_continues_after_operational_pause(tmp_path, monkeypatch):
    project = _project(tmp_path)
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
    assert PipelineRunner(
        project=project, machine="local", run_id="pause", until="bo"
    ).run() == "completed"
    assert calls == ["bo"]

    assert PipelineRunner(
        project=project,
        machine="local",
        run_id="pause",
        from_stage="sample",
        resume=True,
    ).run() == "completed"
    assert calls == ["bo", "sample"]


def test_slurm_script_omits_empty_optional_directives(tmp_path, monkeypatch):
    project = _project(tmp_path)
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
        project=project, machine="cluster", run_id="trial", dry_run=True,
    )
    script = runner._write_slurm_script(runner.build_specs()[0]).read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --partition=" not in script
    assert "#SBATCH --qos=" not in script
    assert "#SBATCH --mem=0" not in script
    assert "#SBATCH --cpus-per-task=4" in script
    assert "export PYTHONNOUSERSITE=1" in script
    assert "export PYTHONUNBUFFERED=1" in script
    assert "source ~/.bashrc" in script


def test_slurm_script_allocates_distributed_candidate_steps(tmp_path, monkeypatch):
    project = _project(tmp_path)
    config = _config()
    config["machine"]["backend"] = "slurm"
    config["cluster"] = {
        "bo": {
            "partition": "CPU",
            "nodes": 2,
            "cores": 96,
            "tasks": 96,
            "tasks_per_node": 48,
            "cpus_per_task": 1,
            "distributed_steps": True,
            "time": "14-00:00:00",
            "mem": "0",
        },
    }
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(
        project=project, machine="ccelab-2node", run_id="distributed", dry_run=True,
    )
    script = runner._write_slurm_script(runner.build_specs()[0]).read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --nodes=2" in script
    assert "#SBATCH --ntasks=96" in script
    assert "#SBATCH --ntasks-per-node=48" in script
    assert "#SBATCH --cpus-per-task=1" in script


def test_slurm_script_requests_explicit_memory_per_node(tmp_path, monkeypatch):
    project = _project(tmp_path)
    config = _config()
    config["machine"]["backend"] = "slurm"
    config["cluster"] = {
        "bo": {
            "partition": "CPU",
            "nodes": 2,
            "cores": 16,
            "tasks": 4,
            "tasks_per_node": 2,
            "cpus_per_task": 4,
            "distributed_steps": True,
            "time": "02:00:00",
            "mem": "64G",
        },
    }
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(
        project=project, machine="ccelab-smoke", run_id="memory", dry_run=True,
    )
    script = runner._write_slurm_script(runner.build_specs()[0]).read_text(
        encoding="utf-8"
    )
    assert "#SBATCH --mem=64G" in script


def test_running_slurm_stage_is_not_completed_by_partial_artifacts(
    tmp_path, monkeypatch
):
    project = _project(tmp_path)
    config = _config()
    config["machine"]["backend"] = "slurm"
    config["cluster"] = {"bo": {}}
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(
        project=project, machine="cluster", run_id="partial", resume=True,
    )
    spec = runner.build_specs()[0]
    for artifact in spec.artifacts:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("partial\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_slurm_state", lambda _: "RUNNING")

    with WorkflowState(runner.state_path) as state:
        state.prepare(
            spec.name, spec.signature, spec.command, spec.output_dir, spec.artifacts
        )
        record = state.transition(spec.name, "waiting", job_id="123")
        assert runner._refresh_waiting(state, record) == "waiting"
        assert state.get(spec.name).status == "waiting"


@pytest.mark.parametrize("scheduler_state", ["UNKNOWN", "QUERY_UNAVAILABLE"])
def test_inconclusive_scheduler_query_never_resubmits(
    tmp_path, monkeypatch, scheduler_state
):
    project = _project(tmp_path)
    config = _config()
    config["machine"]["backend"] = "slurm"
    config["cluster"] = {"bo": {}}
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(
        project=project, machine="cluster", run_id="query-gap", resume=True,
    )
    spec = runner.build_specs()[0]
    monkeypatch.setattr(runner, "_slurm_state", lambda _: scheduler_state)

    with WorkflowState(runner.state_path) as state:
        state.prepare(
            spec.name, spec.signature, spec.command, spec.output_dir, spec.artifacts
        )
        record = state.transition(spec.name, "waiting", job_id="123")
        assert runner._refresh_waiting(state, record) == "waiting"
        refreshed = state.get(spec.name)
        assert refreshed.status == "waiting"
        assert "preserving job state" in refreshed.message


def test_inconclusive_scheduler_query_accepts_complete_artifacts(
    tmp_path, monkeypatch
):
    project = _project(tmp_path)
    config = _config()
    config["machine"]["backend"] = "slurm"
    config["cluster"] = {"bo": {}}
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(
        project=project, machine="cluster", run_id="query-complete", resume=True,
    )
    spec = runner.build_specs()[0]
    for artifact in spec.artifacts:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text("complete\n", encoding="utf-8")
    monkeypatch.setattr(runner, "_slurm_state", lambda _: "UNKNOWN")

    with WorkflowState(runner.state_path) as state:
        state.prepare(
            spec.name, spec.signature, spec.command, spec.output_dir, spec.artifacts
        )
        record = state.transition(spec.name, "waiting", job_id="123")
        assert runner._refresh_waiting(state, record) == "completed"
        assert state.get(spec.name).status == "completed"
        assert "all artifacts verified" in state.get(spec.name).message


def test_scheduler_query_timeout_is_reported_as_unavailable(monkeypatch):
    def timeout(command, **_kwargs):
        raise __import__("subprocess").TimeoutExpired(command, 10)

    monkeypatch.setattr("workflow.pipeline.subprocess.run", timeout)
    assert PipelineRunner._slurm_state("123") == "QUERY_UNAVAILABLE"


def test_scheduler_query_falls_back_to_accounting(monkeypatch):
    responses = iter([
        type("Result", (), {"returncode": 0, "stdout": ""})(),
        type("Result", (), {"returncode": 0, "stdout": "COMPLETED|\n"})(),
    ])

    def completed(_command, **kwargs):
        assert kwargs["timeout"] == 10
        return next(responses)

    monkeypatch.setattr("workflow.pipeline.subprocess.run", completed)
    assert PipelineRunner._slurm_state("123") == "COMPLETED"


def test_missing_sbatch_marks_stage_failed(tmp_path, monkeypatch):
    project = _project(tmp_path)
    project.data["pipeline"]["stages"] = ["bo"]
    config = _config()
    config["machine"]["backend"] = "slurm"
    config["cluster"] = {"bo": {}}
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(
        project=project, machine="cluster", run_id="missing-sbatch", resume=True,
    )
    spec = runner.build_specs()[0]
    monkeypatch.setattr(
        "workflow.pipeline.subprocess.run",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(FileNotFoundError("sbatch")),
    )

    with WorkflowState(runner.state_path) as state:
        state.prepare(
            spec.name, spec.signature, spec.command, spec.output_dir, spec.artifacts
        )
        with pytest.raises(RuntimeError, match="Could not execute sbatch"):
            runner._submit_slurm(state, spec)
        record = state.get(spec.name)
        assert record.status == "failed"
        assert "sbatch" in record.message


def test_watch_stops_after_terminal_slurm_failure(tmp_path, monkeypatch):
    project = _project(tmp_path)
    config = _config()
    config["machine"]["backend"] = "slurm"
    config["cluster"] = {"bo": {}}
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(
        project=project, machine="cluster", run_id="failed", resume=True,
        watch=True, poll_seconds=1,
    )
    spec = runner.build_specs()[0]
    runner.root.mkdir(parents=True, exist_ok=True)
    with WorkflowState(runner.state_path) as state:
        state.initialize({"scientific_hash": None})
        state.prepare(
            spec.name, spec.signature, spec.command, spec.output_dir, spec.artifacts
        )
        state.transition(spec.name, "waiting", job_id="456")
    scheduler_states = iter(["RUNNING", "FAILED"])
    monkeypatch.setattr(runner, "_slurm_state", lambda _: next(scheduler_states))
    monkeypatch.setattr("workflow.pipeline.time.sleep", lambda _: None)
    submit_calls = []
    monkeypatch.setattr(
        runner, "_submit_slurm", lambda *_: submit_calls.append("submitted")
    )

    with pytest.raises(RuntimeError, match="run the same command again"):
        runner.run()
    assert submit_calls == []


def test_new_invocation_resubmits_already_failed_slurm_stage(
    tmp_path, monkeypatch
):
    project = _project(tmp_path)
    project.data["pipeline"]["stages"] = ["bo"]
    config = _config()
    config["machine"]["backend"] = "slurm"
    config["cluster"] = {"bo": {}}
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
    runner = PipelineRunner(
        project=project, machine="cluster", run_id="resume", resume=True,
    )
    spec = runner.build_specs()[0]
    runner.root.mkdir(parents=True, exist_ok=True)
    with WorkflowState(runner.state_path) as state:
        state.initialize({"scientific_hash": None})
        state.prepare(
            spec.name, spec.signature, spec.command, spec.output_dir, spec.artifacts
        )
        state.transition(spec.name, "waiting", job_id="old-job")
    monkeypatch.setattr(runner, "_slurm_state", lambda _: "TIMEOUT")
    submit_calls = []

    def fake_submit(state, stage):
        submit_calls.append(stage.name)
        state.transition(stage.name, "waiting", job_id="new-job")

    monkeypatch.setattr(runner, "_submit_slurm", fake_submit)
    assert runner.run() == "waiting"
    assert submit_calls == ["bo"]


def test_pipeline_allows_method_change_and_reuses_upstream_bo(tmp_path, monkeypatch):
    project = _project(tmp_path)
    project.data["pipeline"]["stages"] = ["bo", "nn"]
    config = _config()
    config["nn"] = {"enabled": True, "model": "mlp_ensemble"}
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)
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
        project=project, machine="local", run_id="trial"
    )
    assert first.run() == "completed"
    assert calls == ["bo", "nn"]

    calls.clear()
    config["nn"]["model"] = "random_forest"
    second = PipelineRunner(
        project=project,
        machine="local",
        run_id="trial",
        resume=True,
    )
    assert second.run() == "completed"
    assert calls == ["nn"]


def test_pipeline_rejects_scientific_change_in_same_run(tmp_path, monkeypatch):
    project = _project(tmp_path)
    project.data["pipeline"]["stages"] = ["bo"]
    config = _config()
    config["targets"] = {"density": {"value": 1.0, "weight": 1.0}}
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)

    def fake_run_local(self, state, spec):
        state.transition(spec.name, "running", increment_attempt=True)
        for artifact in spec.artifacts:
            artifact.parent.mkdir(parents=True, exist_ok=True)
            artifact.write_text("test\n", encoding="utf-8")
        state.transition(spec.name, "completed")

    monkeypatch.setattr(PipelineRunner, "_run_local", fake_run_local)
    first = PipelineRunner(
        project=project, machine="local", run_id="trial"
    )
    assert first.run() == "completed"

    config["targets"]["density"]["value"] = 1.1
    second = PipelineRunner(
        project=project,
        machine="local",
        run_id="trial",
        resume=True,
    )
    with pytest.raises(RuntimeError, match="scientific input changed"):
        second.run()
