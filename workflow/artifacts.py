"""Locate pipeline products and scheduler logs without directory hunting."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .project import Project
from .state import WorkflowState


def pipeline_root(project: Project, run_id: str) -> Path:
    return (project.run_root / "pipelines" / run_id).resolve()


def available_pipeline_runs(project: Project) -> list[str]:
    parent = project.run_root / "pipelines"
    if not parent.exists():
        return []
    return sorted(
        path.name for path in parent.iterdir()
        if path.is_dir() and (path / "state.sqlite").exists()
    )


def collect_pipeline_results(project: Project, run_id: str) -> dict[str, Any]:
    root = pipeline_root(project, run_id)
    database = root / "state.sqlite"
    if not database.exists():
        choices = available_pipeline_runs(project)
        suffix = f" Available run IDs: {', '.join(choices)}" if choices else ""
        raise FileNotFoundError(f"Pipeline run not found: {root}.{suffix}")
    with WorkflowState(database) as state:
        records = state.list()
    stages = []
    for record in records:
        artifacts = [
            {"path": str(Path(path)), "exists": Path(path).exists()}
            for path in record.artifacts
        ]
        stages.append({
            "name": record.name,
            "status": record.status,
            "attempt": record.attempt,
            "job_id": record.job_id,
            "message": record.message,
            "output_dir": record.output_dir,
            "artifacts": artifacts,
        })
    return {
        "project": project.name,
        "run_id": run_id,
        "root": str(root),
        "stages": stages,
    }


def format_pipeline_results(report: dict[str, Any]) -> str:
    lines = [
        f"FFOpt results: {report['project']}/{report['run_id']}",
        f"Run root     : {report['root']}",
        "",
    ]
    for stage in report["stages"]:
        detail = f" job={stage['job_id']}" if stage.get("job_id") else ""
        lines.append(
            f"[{stage['name']}] {stage['status']} attempt={stage['attempt']}{detail}"
        )
        lines.append(f"  output: {stage['output_dir']}")
        for artifact in stage["artifacts"]:
            marker = "OK" if artifact["exists"] else "MISSING"
            lines.append(f"  {marker:7s} {artifact['path']}")
        if stage.get("message"):
            lines.append(f"  note: {stage['message']}")
    return "\n".join(lines)


def find_pipeline_logs(
    project: Project, run_id: str, stage: str | None = None
) -> list[Path]:
    root = pipeline_root(project, run_id)
    log_dir = root / "logs"
    if not log_dir.exists():
        return []
    pattern = f"{stage}_*" if stage else "*"
    return sorted(
        (
            path for path in log_dir.glob(pattern)
            if path.is_file() and path.suffix.lower() in {".out", ".err"}
        ),
        key=lambda path: (path.stat().st_mtime, path.name),
    )


def tail_file(path: str | Path, lines: int = 80) -> str:
    path = Path(path)
    count = max(1, int(lines))
    content = path.read_text(encoding="utf-8", errors="replace").splitlines()
    return "\n".join(content[-count:])
