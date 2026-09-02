import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from workflow.cli import (
    _include_material_top_results,
    _public_workflow,
    _resolve_run_stage_bound,
    build_parser,
    cmd_promote,
    cmd_status,
)
from workflow.project import Project
from workflow.state import WorkflowState


def _project(tmp_path: Path) -> Project:
    path = tmp_path / "ffopt.in"
    path.write_text("ffopt 1\nproject fe\n", encoding="utf-8")
    document = SimpleNamespace(
        workflow=[
            "bo",
            "sample",
            "audit",
            "screen",
            "nn",
            "al",
            "finalists",
            "validate",
        ]
    )
    return Project(
        path,
        {
            "project": {"name": "fe"},
            "pipeline": {
                "stages": [
                    "bo",
                    "sample",
                    "audit",
                    "screen",
                    "nn",
                    "constrained_al",
                    "finalists",
                    "validate",
                ],
                "stage_repetitions": {"constrained_al": 4},
            },
        },
        compilation=SimpleNamespace(document=document),
    )


def test_material_public_stage_bounds_expand_to_runtime_edges(tmp_path: Path):
    project = _project(tmp_path)

    assert _resolve_run_stage_bound(project, "screen", upper=False) == "candidates"
    assert _resolve_run_stage_bound(project, "screen", upper=True) == "static"
    assert _resolve_run_stage_bound(project, "al", upper=False) == "constrained_al_01"
    assert _resolve_run_stage_bound(project, "al", upper=True) == "constrained_al_04"
    assert (
        _resolve_run_stage_bound(project, "constrained_al_03", upper=True)
        == "constrained_al_03"
    )
    assert _resolve_run_stage_bound(project, "finalists", upper=True) == "finalists"


def test_legacy_al_stage_is_not_remapped(tmp_path: Path):
    project = _project(tmp_path)
    project.data["pipeline"] = {"stages": ["bo", "nn", "al", "validate"]}

    assert _resolve_run_stage_bound(project, "al", upper=False) == "al"
    assert _resolve_run_stage_bound(project, "al", upper=True) == "al"


def test_check_and_explain_use_public_workflow_vocabulary(tmp_path: Path):
    project = _project(tmp_path)

    assert _public_workflow(project) == [
        "bo",
        "sample",
        "audit",
        "screen",
        "nn",
        "al",
        "finalists",
        "validate",
    ]


def test_run_parser_accepts_public_and_expanded_material_stage_names():
    parser = build_parser()

    public = parser.parse_args([
        "run",
        "ffopt.in",
        "--from-stage",
        "screen",
        "--until",
        "al",
    ])
    expanded = parser.parse_args([
        "run",
        "ffopt.in",
        "--from-stage",
        "constrained_al_02",
        "--until",
        "finalists",
    ])

    assert (public.from_stage, public.until) == ("screen", "al")
    assert (expanded.from_stage, expanded.until) == (
        "constrained_al_02",
        "finalists",
    )


def test_promote_parser_requires_explicit_source_identity_and_destination():
    parser = build_parser()

    arguments = parser.parse_args([
        "promote",
        "ffopt.in",
        "--run-id",
        "production-a12",
        "--canonical-root",
        "published/fe",
        "--allow-best-effort",
        "--replace-legacy",
        "--force-downgrade",
        "--dry-run",
        "--json",
    ])

    assert arguments.input == "ffopt.in"
    assert arguments.run_id == "production-a12"
    assert arguments.canonical_root == "published/fe"
    assert arguments.allow_best_effort is True
    assert arguments.replace_legacy is True
    assert arguments.force_downgrade is True
    assert arguments.dry_run is True
    assert arguments.json is True


def test_blocked_promotion_dry_run_has_nonzero_cli_status(
    tmp_path: Path, monkeypatch, capsys
):
    project = _project(tmp_path)
    result = {
        "status": "blocked",
        "canonical_root": str(tmp_path / "canonical" / "fe"),
        "comparison": "ineligible",
        "comparison_reason": "best_effort_not_explicitly_allowed",
        "current_best_will_change": False,
        "blocked_reason": (
            "legacy_current_baseline_would_be_removed_without_replacement"
        ),
    }
    monkeypatch.setattr("workflow.cli.load_project", lambda _path: project)
    monkeypatch.setattr(
        "workflow.promotion.promote_validation",
        lambda **_kwargs: result,
    )
    arguments = SimpleNamespace(
        input="ffopt.in",
        run_id="production-a12",
        canonical_root=result["canonical_root"],
        allow_best_effort=False,
        replace_legacy=True,
        force_downgrade=False,
        dry_run=True,
        json=True,
    )

    with pytest.raises(SystemExit) as exc:
        cmd_promote(arguments)

    assert exc.value.code == 2
    assert json.loads(capsys.readouterr().out)["status"] == "blocked"


def test_results_expose_only_explicitly_declared_material_top_outputs(
    tmp_path: Path,
):
    output = tmp_path / "validate"
    output.mkdir()
    top_csv = output / "TOP_PARAMETERS.csv"
    top_csv.write_text("rank\n1\n", encoding="utf-8")
    (output / "validation_summary.json").write_text(
        json.dumps({
            "top_parameters_report": {
                "enabled": True,
                "csv": str(top_csv),
                "json": str(output / "TOP_PARAMETERS.json"),
            }
        }),
        encoding="utf-8",
    )
    # An unrelated newest-looking file must never be discovered.
    (output / "TOP_PARAMETERS-newer.csv").write_text("bad\n", encoding="utf-8")
    report = {
        "stages": [{
            "name": "validate",
            "output_dir": str(output),
            "artifacts": [{
                "path": str(output / "validation_summary.json"),
                "exists": True,
            }],
        }]
    }

    _include_material_top_results(report)

    artifacts = report["stages"][0]["artifacts"]
    assert [item.get("role") for item in artifacts[1:]] == [
        "top_parameters_csv",
        "top_parameters_json",
    ]
    assert artifacts[1]["exists"] is True
    assert artifacts[2]["exists"] is False
    assert all("newer" not in item["path"] for item in artifacts)


def test_managed_status_never_falls_back_to_glob_or_mtime(
    tmp_path: Path, monkeypatch, capsys
):
    project = _project(tmp_path)
    root = project.run_root / "pipelines" / "trial"
    with WorkflowState(root / "state.sqlite") as state:
        state.initialize({"machine": "cluster"})

    monkeypatch.setattr("workflow.cli._common", lambda _args: (project, "local"))
    monkeypatch.setattr(
        "workflow.cli._newest",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("managed status must not scan legacy outputs")
        ),
    )

    cmd_status(SimpleNamespace(machine=None, run_id="trial"))

    assert "Pipeline: trial" in capsys.readouterr().out
