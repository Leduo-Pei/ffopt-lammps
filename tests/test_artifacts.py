from pathlib import Path
import json
from types import SimpleNamespace

from engine.config_loader import save_config_snapshot
from workflow.artifacts import (
    collect_pipeline_results,
    find_pipeline_logs,
    format_pipeline_results,
    tail_file,
)
from workflow.cli import _bo_checkpoint_summary, _slurm_job_summary, cmd_status
from workflow.project import Project
from workflow.state import WorkflowState


def _project(tmp_path: Path) -> Project:
    path = tmp_path / "ffopt.in"
    path.write_text("test\n", encoding="ascii")
    return Project(path, {
        "project": {"name": "demo", "run_root": "runs/demo"},
        "pipeline": {"stages": ["bo"]},
    })


def test_results_use_persisted_pipeline_artifacts(tmp_path):
    project = _project(tmp_path)
    root = project.run_root / "pipelines" / "trial"
    artifact = root / "bo" / "all_results.csv"
    artifact.parent.mkdir(parents=True)
    artifact.write_text("objective\n0.1\n", encoding="ascii")
    with WorkflowState(root / "state.sqlite") as state:
        state.prepare("bo", "sig", ["python", "bo"], artifact.parent, [artifact])
        state.transition("bo", "running", increment_attempt=True)
        state.transition("bo", "completed")

    report = collect_pipeline_results(project, "trial")
    assert report["stages"][0]["artifacts"][0]["exists"] is True
    assert "all_results.csv" in format_pipeline_results(report)


def test_stage_log_discovery_and_tail(tmp_path):
    project = _project(tmp_path)
    log_dir = project.run_root / "pipelines" / "trial" / "logs"
    log_dir.mkdir(parents=True)
    output = log_dir / "bo_123.out"
    output.write_text("one\ntwo\nthree\n", encoding="ascii")
    (log_dir / "nn_124.out").write_text("nn\n", encoding="ascii")
    assert find_pipeline_logs(project, "trial", "bo") == [output]
    assert tail_file(output, 2) == "two\nthree"


def test_pipeline_config_snapshot_is_not_duplicated_per_stage(tmp_path):
    provenance = tmp_path / "provenance"
    provenance.mkdir()
    source = provenance / "runtime_deadbeef.json"
    source.write_text("{}\n", encoding="ascii")
    output = tmp_path / "bo"

    result = save_config_snapshot({"_config_path": str(source)}, output)

    assert result == source.resolve()
    assert not (output / "config_used.yaml").exists()


def test_bo_checkpoint_summary_reports_resumable_progress(tmp_path):
    checkpoint = tmp_path / "checkpoints" / "latest.json"
    checkpoint.parent.mkdir()
    checkpoint.write_text(
        json.dumps({"round": 7, "all_results": [{}, {}, {}], "best_valid_obj": 0.0123}),
        encoding="ascii",
    )

    assert _bo_checkpoint_summary(str(tmp_path)) == (
        "checkpoint round=7 evaluations=3 best=0.012300"
    )


def test_status_uses_recorded_machine_when_no_profile_is_requested(
    tmp_path, monkeypatch, capsys
):
    project = _project(tmp_path)
    root = project.run_root / "pipelines" / "trial"
    with WorkflowState(root / "state.sqlite") as state:
        state.initialize({"machine": "cluster-recorded"})

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "workflow.cli._common", lambda _args: (project, "local")
    )
    cmd_status(SimpleNamespace(machine=None, run_id="trial"))

    assert "Project: demo [cluster-recorded]" in capsys.readouterr().out


def test_slurm_job_summary_reports_state_and_reason(monkeypatch):
    monkeypatch.setattr("workflow.cli.shutil.which", lambda _: "/usr/bin/squeue")
    monkeypatch.setattr(
        "workflow.cli.subprocess.run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=0, stdout="PENDING|Resources\n"
        ),
    )

    assert _slurm_job_summary("123") == ("pending", "Resources")


def test_slurm_job_summary_falls_back_to_accounting(monkeypatch):
    monkeypatch.setattr(
        "workflow.cli.shutil.which",
        lambda name: f"/usr/bin/{name}" if name in {"squeue", "sacct"} else None,
    )
    responses = iter([
        SimpleNamespace(returncode=0, stdout=""),
        SimpleNamespace(returncode=0, stdout="TIMEOUT|TimeLimit\n"),
    ])
    monkeypatch.setattr(
        "workflow.cli.subprocess.run", lambda *_args, **_kwargs: next(responses)
    )

    assert _slurm_job_summary("456") == ("timeout", "TimeLimit")
