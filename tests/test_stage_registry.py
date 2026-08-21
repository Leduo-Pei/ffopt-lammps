from pathlib import Path

import pytest

from workflow.pipeline import PipelineRunner, _scientific_config
from workflow.project import Project
from workflow.stage_registry import (
    LEGACY_PIPELINE_STAGES,
    StageDefinition,
    StageRegistry,
    legacy_stage_registry,
    material_stage_registry,
)
from workflow.state import canonical_hash


def _project(tmp_path: Path) -> Project:
    path = tmp_path / "ffopt.in"
    path.write_text("ffopt 1\nproject demo\n", encoding="utf-8")
    return Project(path, {
        "project": {"name": "demo", "run_root": "runs/demo"},
        "modules": {},
        "pipeline": {"stages": ["bo", "sample"]},
        "stages": {"sample": {"n_points": 12}},
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


def test_legacy_registry_declares_commands_artifacts_and_linear_adapter():
    graph = legacy_stage_registry().expand(LEGACY_PIPELINE_STAGES, legacy_linear=True)

    assert graph.names == LEGACY_PIPELINE_STAGES
    assert graph.node("bo").command_token == "bo"
    assert graph.node("bo").expected_artifacts == ("all_results.csv",)
    assert graph.node("sample").dependencies == ("bo",)
    assert graph.node("nn").dependencies == ("sample",)
    assert graph.node("validate").dependencies == ("finalize",)


def test_pipeline_legacy_stage_signatures_remain_compatible(tmp_path, monkeypatch):
    project = _project(tmp_path)
    config = _config()
    monkeypatch.setattr("workflow.pipeline.compose_config", lambda *_: config)

    runner = PipelineRunner(
        project=project, machine="local", run_id="signature", dry_run=True,
    )
    specs = runner.build_specs()
    bo_signature = canonical_hash({
        "stage": "bo",
        "scientific_config": _scientific_config(config),
        "settings": {
            "optimization": config["optimization"],
            "checkpoint": config["checkpoint"],
        },
        "upstream": "root",
    })
    sample_signature = canonical_hash({
        "stage": "sample",
        "scientific_config": _scientific_config(config),
        "settings": {"n_points": 12},
        "upstream": bo_signature,
    })

    assert [spec.signature for spec in specs] == [bo_signature, sample_signature]
    assert specs[1].kind == "sample"
    assert specs[1].command_token == "sample"
    assert specs[1].dependencies == ("bo",)


def test_material_graph_is_opt_in_and_expands_screen_and_al_rounds():
    registry = material_stage_registry()
    graph = registry.expand(
        [
            "bo", "sample", "audit", "screen", "nn",
            "constrained_al", "finalists",
        ],
        repetitions={"constrained_al": 3},
    )

    assert graph.names == (
        "bo",
        "sample",
        "audit",
        "candidates",
        "static",
        "nn",
        "constrained_al_01",
        "constrained_al_02",
        "constrained_al_03",
        "finalists",
    )
    assert graph.node("static").dependencies == ("candidates",)
    assert set(graph.node("constrained_al_01").dependencies) == {"nn", "static"}
    assert graph.node("constrained_al_02").dependencies == (
        "constrained_al_01", "nn", "static",
    )
    assert graph.node("finalists").dependencies == ("constrained_al_03",)
    assert graph.node("constrained_al_01").command_token == "constrained-al"
    assert graph.node("finalists").expected_artifacts == (
        "dynamic_results.csv",
        "dynamic_seed_results.csv",
        "finalists_selected.csv",
        "best_candidate.json",
        "batch_summary.json",
        "stage_manifest.json",
    )
    assert graph.node("bo").expected_artifacts == (
        "all_results.csv",
        "feasible_archive.csv",
        "coverage_anchors.csv",
        "coverage_summary.json",
        "best_parameters.txt",
    )

    with pytest.raises(ValueError, match="Unknown pipeline stage"):
        legacy_stage_registry().expand(["bo", "screen"])


@pytest.mark.parametrize(
    ("tokens", "message"),
    [
        (["missing"], "Unknown pipeline stage"),
        (["bo", "bo"], "Duplicate workflow stages"),
        (["sample"], "requires missing stage 'bo'"),
    ],
)
def test_registry_rejects_unknown_duplicate_and_missing_stages(tokens, message):
    with pytest.raises(ValueError, match=message):
        legacy_stage_registry().expand(tokens)


def test_registry_rejects_dependency_and_expansion_cycles():
    dependencies = StageRegistry([
        StageDefinition("one", "one", requires=("two",)),
        StageDefinition("two", "two", requires=("one",)),
    ])
    with pytest.raises(ValueError, match="dependency cycle"):
        dependencies.expand(["one", "two"])

    expansions = StageRegistry([
        StageDefinition("one", expands_to=("two",)),
        StageDefinition("two", expands_to=("one",)),
    ])
    with pytest.raises(ValueError, match="expansion cycle"):
        expansions.expand(["one"])


def test_registry_rejects_duplicate_concrete_stage_after_macro_expansion():
    registry = material_stage_registry()
    with pytest.raises(ValueError, match="Duplicate concrete pipeline stages"):
        registry.expand(["bo", "audit", "candidates", "screen"])


def test_repeat_count_is_validated():
    registry = material_stage_registry()
    with pytest.raises(ValueError, match="positive integer"):
        registry.expand(
            ["bo", "audit", "screen", "nn", "constrained_al"],
            repetitions={"constrained_al": 0},
        )
    with pytest.raises(ValueError, match="not repeatable"):
        registry.expand(["bo"], repetitions={"bo": 2})
