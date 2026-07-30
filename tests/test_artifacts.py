from pathlib import Path

from workflow.artifacts import (
    collect_pipeline_results,
    find_pipeline_logs,
    format_pipeline_results,
    tail_file,
)
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
