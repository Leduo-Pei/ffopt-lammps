from __future__ import annotations

import json
import os
from pathlib import Path
import shutil

import pytest

from workflow.artifact_manifest import (
    build_artifact_manifest,
    canonical_parameter_key,
    sha256_file,
    write_artifact_manifest,
)
from workflow.project import Project
from workflow.promotion import PromotionError, promote_validation
from workflow.state import WorkflowState
from engine.cubic_elastic_runner import STATIC_PROTOCOL


def _write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def _project(tmp_path: Path) -> Project:
    project_file = tmp_path / "ffopt.in"
    project_file.write_text("ffopt 1\nproject fe\n", encoding="utf-8")
    return Project(project_file, {
        "project": {"name": "fe", "run_root": "runs/fe"},
        "pipeline": {"stages": ["validate"]},
    })


def _completed_validation(
    project: Project,
    run_id: str,
    *,
    status: str = "accepted",
    score: float = 10.0,
    dynamic_score: float | None = None,
    protocol: str = "protocol-a",
    output_override: Path | None = None,
) -> Path:
    root = project.run_root / "pipelines" / run_id
    output = root / "validate"
    output.mkdir(parents=True)
    parameters = {"epsilon": 5.0, "sigma": 2.0 + score / 1000.0}
    key = canonical_parameter_key(parameters)
    summary = output / "validation_summary.json"
    final_parameters = output / "final_parameters.json"
    model_adequacy = output / "model_adequacy.json"
    _write_json(model_adequacy, {
        "schema_version": 2,
        "material": {
            "kind": "elemental",
            "atom_type_count": 2,
            "same_element_representation": "permanent_ordered_sublattice",
        },
        "physical_transferability": {"status": "requires_validation"},
    })
    _write_json(summary, {
        "schema_version": 1,
        "status": status,
        "hard_gate_pass": True,
        "execution_complete": True,
        "within_mechanical_quality_tier": status == "accepted",
        "mechanical_max_error_percent": score,
        "dynamic_elastic_evidence": {
            "required": True,
            "row": {
                "mechanical_max_error_percent": (
                    score if dynamic_score is None else dynamic_score
                )
            },
        },
        "parameter_key": key,
        "model_adequacy": {
            "report": str(model_adequacy),
            "physical_transferability_status": "requires_validation",
        },
    })
    _write_json(final_parameters, {
        "schema_version": 1,
        "status": status,
        "parameter_key": key,
        "raw_free_parameters": parameters,
    })
    manifest = build_artifact_manifest(
        kind="candidate",
        identifier=f"material_validation:{key}",
        parameters=parameters,
        seeds=[404, 505, 606],
        scientific_config={"protocol": protocol},
        input_artifacts={},
        expected_outputs={
            "validation_summary": summary,
            "final_parameters": final_parameters,
            "model_adequacy": model_adequacy,
        },
    )
    stage_manifest = output / "stage_manifest.json"
    write_artifact_manifest(stage_manifest, manifest)
    with WorkflowState(root / "state.sqlite") as state:
        state.initialize({"run_id": run_id, "project": project.name})
        state.prepare(
            "validate",
            "signature",
            ["python", "validate"],
            output_override or output,
            [summary, final_parameters, stage_manifest],
        )
        state.transition("validate", "running", increment_attempt=True)
        state.transition("validate", "completed")
    return output


def _rewrite_primary_manifest(output: Path, *, protocol: str = "protocol-a") -> None:
    parameters = json.loads((output / "final_parameters.json").read_text())
    manifest = build_artifact_manifest(
        kind="candidate",
        identifier=f"material_validation:{parameters['parameter_key']}",
        parameters=parameters["raw_free_parameters"],
        seeds=[404, 505, 606],
        scientific_config={"protocol": protocol},
        input_artifacts={},
        expected_outputs={
            "validation_summary": output / "validation_summary.json",
            "final_parameters": output / "final_parameters.json",
            "model_adequacy": output / "model_adequacy.json",
        },
    )
    (output / "stage_manifest.json").unlink(missing_ok=True)
    write_artifact_manifest(output / "stage_manifest.json", manifest)


def test_best_effort_requires_explicit_permission_and_dry_run_is_read_only(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    _completed_validation(project, "trial", status="best_effort", score=22.0)
    canonical = tmp_path / "published" / "fe"

    unapproved = promote_validation(
        project=project, run_id="trial", canonical_root=canonical
    )
    assert unapproved["status"] == "published_not_current"
    assert unapproved["comparison"] == "ineligible"
    assert (canonical / "latest_validation.json").is_file()
    assert not (canonical / "current_best.json").exists()
    assert not (canonical / "final_parameters.json").exists()

    dry_root = tmp_path / "dry" / "fe"
    result = promote_validation(
        project=project,
        run_id="trial",
        canonical_root=dry_root,
        allow_best_effort=True,
        dry_run=True,
    )
    assert result["status"] == "dry_run"
    assert result["current_best_will_change"] is True
    assert not dry_root.exists()


@pytest.mark.parametrize(
    ("status", "hard_gate_pass", "within_tier", "message"),
    [
        ("accepted", "false", True, "must be a JSON boolean"),
        ("accepted", True, "false", "must be a JSON boolean"),
        ("accepted", True, False, "status 'accepted' requires"),
        ("accepted", False, True, "status 'accepted' requires"),
        ("best_effort", True, True, "status 'best_effort' requires"),
        ("best_effort", False, False, "status 'best_effort' requires"),
        ("rejected", True, False, "status 'rejected' requires"),
    ],
)
def test_promotion_rejects_non_boolean_or_inconsistent_quality_fields(
    tmp_path: Path,
    status: str,
    hard_gate_pass: object,
    within_tier: object,
    message: str,
) -> None:
    project = _project(tmp_path)
    output = _completed_validation(project, "forged-quality", status=status)
    summary_path = output / "validation_summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["hard_gate_pass"] = hard_gate_pass
    summary["within_mechanical_quality_tier"] = within_tier
    _write_json(summary_path, summary)
    _rewrite_primary_manifest(output)

    canonical = tmp_path / "canonical" / "fe"
    with pytest.raises(PromotionError, match=message):
        promote_validation(
            project=project,
            run_id="forged-quality",
            canonical_root=canonical,
            allow_best_effort=True,
        )

    assert not canonical.exists()


def test_promotion_archives_legacy_and_publishes_immutable_snapshot(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    source = _completed_validation(project, "run-a", score=12.0)
    candidate_file = source / "candidate_runs" / "large" / "trajectory.dump"
    candidate_file.parent.mkdir(parents=True)
    candidate_file.write_bytes(b"large campaign-only evidence")
    source_before = {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    }
    canonical = tmp_path / "canonical" / "fe"
    canonical.mkdir(parents=True)
    (canonical / "final_parameters.json").write_text(
        '{"legacy": true}\n', encoding="utf-8"
    )
    (canonical / "FINAL_RESULTS_A9.md").write_text("old\n", encoding="utf-8")

    with pytest.raises(PromotionError, match="--replace-legacy"):
        promote_validation(
            project=project, run_id="run-a", canonical_root=canonical
        )

    result = promote_validation(
        project=project,
        run_id="run-a",
        canonical_root=canonical,
        replace_legacy=True,
    )

    assert result["status"] == "promoted"
    current = json.loads((canonical / "current_best.json").read_text())
    latest = json.loads((canonical / "latest_attempt.json").read_text())
    assert current["snapshot"] == latest["snapshot"]
    snapshot = canonical / current["snapshot"]
    assert (snapshot / "PUBLICATION.json").is_file()
    assert (snapshot / "SNAPSHOT_MANIFEST.json").is_file()
    assert not (snapshot / "candidate_runs").exists()
    assert (canonical / "final_parameters.json").is_file()
    assert (canonical / "latest_validation.json").is_file()
    archive = Path(result["legacy_archive"])
    assert (archive / "final_parameters.json").is_file()
    assert (archive / "FINAL_RESULTS_A9.md").is_file()
    assert {
        path.relative_to(source).as_posix(): path.read_bytes()
        for path in source.rglob("*")
        if path.is_file()
    } == source_before


def test_latest_attempt_does_not_downgrade_current_best_without_force(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    _completed_validation(project, "better", score=10.0)
    _completed_validation(project, "worse", score=18.0)
    canonical = tmp_path / "canonical" / "fe"

    first = promote_validation(
        project=project, run_id="better", canonical_root=canonical
    )
    second = promote_validation(
        project=project, run_id="worse", canonical_root=canonical
    )
    current = json.loads((canonical / "current_best.json").read_text())
    latest = json.loads((canonical / "latest_attempt.json").read_text())

    assert first["status"] == "promoted"
    assert second["status"] == "published_not_current"
    assert second["comparison"] == "downgrade"
    assert current["publication_id"] == first["publication_id"]
    assert latest["publication_id"] == second["publication_id"]

    forced = promote_validation(
        project=project,
        run_id="worse",
        canonical_root=canonical,
        force_downgrade=True,
    )
    current = json.loads((canonical / "current_best.json").read_text())
    assert forced["status"] == "promoted"
    assert current["publication_id"] == second["publication_id"]


def test_protocol_change_is_published_but_not_made_current(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _completed_validation(project, "baseline", score=15.0, protocol="a")
    _completed_validation(project, "changed", score=5.0, protocol="b")
    canonical = tmp_path / "canonical" / "fe"
    baseline = promote_validation(
        project=project, run_id="baseline", canonical_root=canonical
    )

    changed = promote_validation(
        project=project, run_id="changed", canonical_root=canonical
    )
    current = json.loads((canonical / "current_best.json").read_text())
    latest = json.loads((canonical / "latest_attempt.json").read_text())
    assert changed["comparison"] == "incomparable"
    assert changed["status"] == "published_not_current"
    assert current["publication_id"] == baseline["publication_id"]
    assert latest["publication_id"] == changed["publication_id"]


def test_state_output_dir_must_be_the_managed_validate_directory(tmp_path: Path) -> None:
    project = _project(tmp_path)
    wrong = tmp_path / "elsewhere"
    wrong.mkdir()
    _completed_validation(project, "trial", output_override=wrong)

    with pytest.raises(PromotionError, match="managed path"):
        promote_validation(
            project=project,
            run_id="trial",
            canonical_root=tmp_path / "canonical" / "fe",
        )


def test_dynamic_validation_score_is_the_current_best_comparison_metric(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    _completed_validation(
        project,
        "trial",
        score=35.7,
        dynamic_score=22.26,
    )

    result = promote_validation(
        project=project,
        run_id="trial",
        canonical_root=tmp_path / "canonical" / "fe",
    )

    assert result["quality"]["mechanical_max_error_percent"] == pytest.approx(22.26)
    assert result["quality"]["mechanical_score_basis"].startswith(
        "dynamic_elastic_evidence"
    )
    assert result["applicability"] == {
        "scope": "ordered_sublattice_bulk",
        "physical_transferability_status": "requires_validation",
        "formal_invariance_status": "not_evaluated",
        "empirical_transfer_status": "not_evaluated",
        "elemental_transferability_claim": "not_established",
    }
    publication = json.loads(
        (Path(result["snapshot"]) / "PUBLICATION.json").read_text()
    )
    pointer = json.loads(
        (tmp_path / "canonical" / "fe" / "current_best.json").read_text()
    )
    assert publication["applicability"] == result["applicability"]
    assert pointer["applicability"] == result["applicability"]
    assert pointer["validation_summary"].endswith("/validation_summary.json")


def test_summary_cannot_forge_supported_transferability_against_report(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    output = _completed_validation(project, "forged", score=10.0)
    adequacy_path = output / "model_adequacy.json"
    adequacy = json.loads(adequacy_path.read_text())
    adequacy.update({
        "formal_invariance_status": "pass",
        "empirical_transfer_status": "attested_pass",
        "elemental_transferability_claim": "not_established",
    })
    adequacy["physical_transferability"].update({
        "formal_invariance_status": "pass",
        "empirical_transfer_status": "attested_pass",
        "elemental_transferability_claim": "not_established",
    })
    _write_json(adequacy_path, adequacy)
    summary_path = output / "validation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["model_adequacy"].update({
        "formal_invariance_status": "pass",
        "empirical_transfer_status": "pass",
        "elemental_transferability_claim": "supported",
    })
    _write_json(summary_path, summary)
    parameters = json.loads((output / "final_parameters.json").read_text())
    manifest = build_artifact_manifest(
        kind="candidate",
        identifier=f"material_validation:{parameters['parameter_key']}",
        parameters=parameters["raw_free_parameters"],
        seeds=[404, 505, 606],
        scientific_config={"protocol": "protocol-a"},
        input_artifacts={},
        expected_outputs={
            "validation_summary": summary_path,
            "final_parameters": output / "final_parameters.json",
            "model_adequacy": adequacy_path,
        },
    )
    (output / "stage_manifest.json").unlink()
    write_artifact_manifest(output / "stage_manifest.json", manifest)
    canonical = tmp_path / "canonical" / "fe"

    with pytest.raises(PromotionError, match="model_adequacy conflicts"):
        promote_validation(
            project=project,
            run_id="forged",
            canonical_root=canonical,
        )

    assert not canonical.exists()


def test_completed_rejected_attempt_is_snapshotted_but_never_current(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    output = _completed_validation(project, "rejected", status="rejected", score=80.0)
    summary = json.loads((output / "validation_summary.json").read_text())
    summary["hard_gate_pass"] = False
    _write_json(output / "validation_summary.json", summary)
    # Rewrite the immutable validation manifest after constructing this
    # intentionally rejected but execution-complete fixture.
    parameters = json.loads((output / "final_parameters.json").read_text())
    manifest = build_artifact_manifest(
        kind="candidate",
        identifier=f"material_validation:{parameters['parameter_key']}",
        parameters=parameters["raw_free_parameters"],
        seeds=[404, 505, 606],
        scientific_config={"protocol": "protocol-a"},
        input_artifacts={},
        expected_outputs={
            "validation_summary": output / "validation_summary.json",
            "final_parameters": output / "final_parameters.json",
            "model_adequacy": output / "model_adequacy.json",
        },
    )
    (output / "stage_manifest.json").unlink()
    write_artifact_manifest(output / "stage_manifest.json", manifest)

    canonical = tmp_path / "canonical" / "fe"
    result = promote_validation(
        project=project, run_id="rejected", canonical_root=canonical
    )

    assert result["status"] == "published_not_current"
    assert result["comparison"] == "ineligible"
    assert (canonical / "latest_validation.json").is_file()
    assert not (canonical / "current_best.json").exists()


def test_nested_elastic_input_tampering_is_rejected(tmp_path: Path) -> None:
    project = _project(tmp_path)
    output = _completed_validation(project, "nested", score=12.0)
    config = tmp_path / "runtime.json"
    parameter_source = tmp_path / "selected.json"
    config.write_text("{}\n", encoding="utf-8")
    parameter_source.write_text("{}\n", encoding="utf-8")
    candidate = output / "elastic_candidate.csv"
    candidate.write_text("epsilon,sigma\n5,2\n", encoding="utf-8")

    static = output / "elasticity_static"
    static.mkdir()
    batch_identity = static / "batch_identity.json"
    batch_identity.write_text('{"batch": 1}\n', encoding="utf-8")
    static_results = static / "static_results.csv"
    static_results.write_text("mechanical_max_error_percent\n12\n", encoding="utf-8")
    batch_summary = static / "batch_summary.json"
    best = static / "best_candidate.json"
    finalists = static / "finalists_selected.csv"
    batch_summary.write_text("{}\n", encoding="utf-8")
    best.write_text("{}\n", encoding="utf-8")
    finalists.write_text("rank\n1\n", encoding="utf-8")
    candidate_dir = static / "candidate_runs" / "one" / "static"
    candidate_dir.mkdir(parents=True)
    candidate_result = candidate_dir / "candidate_result.json"
    candidate_result.write_text("{}\n", encoding="utf-8")
    candidate_manifest = build_artifact_manifest(
        kind="candidate",
        identifier="elastic_candidate:test",
        parameters={"epsilon": 5.0, "sigma": 2.0},
        seeds=[],
        scientific_config={"protocol": "static"},
        input_artifacts={},
        expected_outputs={"result": candidate_result},
    )
    write_artifact_manifest(candidate_dir / "artifact_manifest.json", candidate_manifest)
    static_manifest = build_artifact_manifest(
        kind="stage",
        identifier=f"cubic_elastic_batch:{STATIC_PROTOCOL}",
        parameters=None,
        seeds=[],
        scientific_config={"protocol": "static"},
        input_artifacts={
            "batch_identity": batch_identity,
            "candidate_manifest_000000": candidate_dir / "artifact_manifest.json",
            "config": config,
            "parameters": candidate,
        },
        expected_outputs={
            "batch_summary": batch_summary,
            "best_candidate": best,
            "finalists": finalists,
            "results": static_results,
        },
    )
    write_artifact_manifest(static / "stage_manifest.json", static_manifest)

    summary_path = output / "validation_summary.json"
    final_path = output / "final_parameters.json"
    summary = json.loads(summary_path.read_text())
    summary.update({
        "config": str(config),
        "parameter_source": {"path": str(parameter_source)},
        "static_elastic_evidence": {
            "manifest": str(static / "stage_manifest.json"),
            "results": str(static_results),
            "row": {"mechanical_max_error_percent": 12.0},
        },
    })
    _write_json(summary_path, summary)
    parameters = json.loads(final_path.read_text())
    main_manifest = build_artifact_manifest(
        kind="candidate",
        identifier=f"material_validation:{parameters['parameter_key']}",
        parameters=parameters["raw_free_parameters"],
        seeds=[404, 505, 606],
        scientific_config={"protocol": "protocol-a"},
        input_artifacts={
            "compiled_config": config,
            "parameter_source": parameter_source,
            "elastic_candidate": candidate,
            "static_manifest": static / "stage_manifest.json",
            "static_results": static_results,
        },
        expected_outputs={
            "validation_summary": summary_path,
            "final_parameters": final_path,
            "model_adequacy": output / "model_adequacy.json",
        },
    )
    (output / "stage_manifest.json").unlink()
    write_artifact_manifest(output / "stage_manifest.json", main_manifest)

    # Top-level outputs and the static stage outputs remain unchanged.  Only a
    # declared static input is altered; promotion must still fail closed.
    batch_identity.write_text('{"batch": "tampered"}\n', encoding="utf-8")
    with pytest.raises(PromotionError, match="static elasticity input hash mismatch"):
        promote_validation(
            project=project,
            run_id="nested",
            canonical_root=tmp_path / "canonical" / "fe",
        )


def test_replace_legacy_migrates_one_off_current_symlink_layout(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    output = _completed_validation(project, "trial", score=12.0)
    _completed_validation(
        project, "blocked", status="best_effort", score=22.0
    )
    canonical = tmp_path / "canonical" / "fe"
    result_bundle = canonical / "results_a11"
    result_bundle.mkdir(parents=True)
    for name in (
        "validation_summary.json",
        "final_parameters.json",
        "model_adequacy.json",
        "stage_manifest.json",
    ):
        shutil.copy2(output / name, result_bundle / name)
    shutil.copy2(
        output / "validation_summary.json", result_bundle / "latest_validation.json"
    )
    oneoff_parameters = json.loads((output / "final_parameters.json").read_text())
    _write_json(result_bundle / "current_best.json", {
        "schema_version": 1,
        "kind": "current_best",
        "classification": "current_best_baseline",
        "quality": "best_effort",
        "parameter_key": oneoff_parameters["parameter_key"],
        "scope": "ordered_sublattice_bulk",
        "elemental_transferability": "requires_validation",
        "snapshot": str(result_bundle.resolve()),
        "validation_summary": str(
            (result_bundle / "validation_summary.json").resolve()
        ),
        "final_parameters": str((result_bundle / "final_parameters.json").resolve()),
    })
    _write_json(result_bundle / "PROMOTION_MANIFEST.json", {
        "schema_version": 1,
        "kind": "ffopt_baseline_promotion",
        "classification": "current_best_baseline",
        "quality": "best_effort",
        "parameter_key": oneoff_parameters["parameter_key"],
        "scope": "ordered_sublattice_bulk",
        "elemental_transferability": "requires_validation",
    })
    manifest_lines = []
    for path in sorted(result_bundle.rglob("*")):
        if path.is_file() and path.name != "MANIFEST.sha256":
            manifest_lines.append(
                f"{sha256_file(path).sha256}  {path.relative_to(result_bundle).as_posix()}"
            )
    (result_bundle / "MANIFEST.sha256").write_text(
        "\n".join(manifest_lines) + "\n", encoding="ascii"
    )
    try:
        os.symlink(result_bundle, canonical / "current", target_is_directory=True)
        os.symlink(result_bundle / "current_best.json", canonical / "current_best.json")
        os.symlink(result_bundle / "final_parameters.json", canonical / "final_parameters.json")
        os.symlink(
            result_bundle / "latest_validation.json",
            canonical / "latest_validation.json",
        )
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(
        PromotionError,
        match="pointer schema|Legacy canonical|regular non-symlink",
    ):
        promote_validation(
            project=project, run_id="trial", canonical_root=canonical
        )

    before_names = sorted(path.name for path in canonical.iterdir())
    before_links = {
        path.name: os.readlink(path)
        for path in canonical.iterdir()
        if path.is_symlink()
    }
    before_bundle = {
        path.name: path.read_bytes()
        for path in result_bundle.iterdir()
        if path.is_file()
    }
    blocked_dry = promote_validation(
        project=project,
        run_id="blocked",
        canonical_root=canonical,
        replace_legacy=True,
        dry_run=True,
    )
    assert blocked_dry["status"] == "blocked"
    assert blocked_dry["blocked_reason"] == (
        "legacy_current_baseline_would_be_removed_without_replacement"
    )
    with pytest.raises(PromotionError, match="Legacy current baseline"):
        promote_validation(
            project=project,
            run_id="blocked",
            canonical_root=canonical,
            replace_legacy=True,
        )
    assert sorted(path.name for path in canonical.iterdir()) == before_names
    assert not (canonical / "published_validations").exists()
    assert not (canonical / "promotion_journal").exists()
    dry = promote_validation(
        project=project,
        run_id="trial",
        canonical_root=canonical,
        replace_legacy=True,
        dry_run=True,
    )
    assert dry["status"] == "dry_run"
    assert sorted(path.name for path in canonical.iterdir()) == before_names
    assert {
        path.name: os.readlink(path)
        for path in canonical.iterdir()
        if path.is_symlink()
    } == before_links
    assert {
        path.name: path.read_bytes()
        for path in result_bundle.iterdir()
        if path.is_file()
    } == before_bundle

    result = promote_validation(
        project=project,
        run_id="trial",
        canonical_root=canonical,
        replace_legacy=True,
    )

    assert result["status"] == "promoted"
    assert not (canonical / "current").exists()
    assert not (canonical / "current_best.json").is_symlink()
    assert not (canonical / "final_parameters.json").is_symlink()
    archive = Path(result["legacy_archive"])
    archive_manifest = json.loads((archive / "ARCHIVE_MANIFEST.json").read_text())
    assert archive_manifest["entries"]["current"]["link_target"].endswith("results_a11")
    assert (archive / "current_target" / "validation_summary.json").is_file()
    assert archive_manifest["current_target_files"]["validation_summary.json"]


def test_corrupt_latest_never_forgets_a_valid_managed_current(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    _completed_validation(project, "better", score=10.0)
    _completed_validation(project, "worse", score=18.0)
    canonical = tmp_path / "canonical" / "fe"
    promote_validation(
        project=project, run_id="better", canonical_root=canonical
    )
    current_before = (canonical / "current_best.json").read_bytes()
    parameters_before = (canonical / "final_parameters.json").read_bytes()
    snapshots_before = sorted(
        path.name for path in (canonical / "published_validations").iterdir()
    )
    (canonical / "latest_attempt.json").write_text(
        '{"schema": "corrupt-latest"}\n', encoding="utf-8"
    )

    with pytest.raises(PromotionError, match="pointer schema"):
        promote_validation(
            project=project,
            run_id="worse",
            canonical_root=canonical,
            replace_legacy=True,
        )

    assert (canonical / "current_best.json").read_bytes() == current_before
    assert (canonical / "final_parameters.json").read_bytes() == parameters_before
    assert sorted(
        path.name for path in (canonical / "published_validations").iterdir()
    ) == snapshots_before


def test_corrupt_managed_current_fails_closed_even_with_replace_legacy(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    _completed_validation(project, "baseline", score=10.0)
    _completed_validation(project, "candidate", score=5.0)
    canonical = tmp_path / "canonical" / "fe"
    promote_validation(
        project=project, run_id="baseline", canonical_root=canonical
    )
    latest_before = (canonical / "latest_attempt.json").read_bytes()
    parameters_before = (canonical / "final_parameters.json").read_bytes()
    snapshots_before = sorted(
        path.name for path in (canonical / "published_validations").iterdir()
    )
    corrupt = b'{"schema": "corrupt-current"}\n'
    (canonical / "current_best.json").write_bytes(corrupt)

    with pytest.raises(PromotionError, match="pointer schema"):
        promote_validation(
            project=project,
            run_id="candidate",
            canonical_root=canonical,
            replace_legacy=True,
        )

    assert (canonical / "current_best.json").read_bytes() == corrupt
    assert (canonical / "latest_attempt.json").read_bytes() == latest_before
    assert (canonical / "final_parameters.json").read_bytes() == parameters_before
    assert sorted(
        path.name for path in (canonical / "published_validations").iterdir()
    ) == snapshots_before


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("publication_id", "forged-publication"),
        ("parameter_key", "named:sha256:forged"),
        ("quality", {"status": "accepted"}),
        ("protocol", {"validation_schema_version": 999}),
        ("applicability", {"scope": "forged"}),
        ("validation_summary", "published_validations/forged/validation_summary.json"),
    ],
)
def test_managed_current_pointer_metadata_must_match_sealed_publication(
    tmp_path: Path,
    field: str,
    replacement: object,
) -> None:
    project = _project(tmp_path)
    _completed_validation(project, "baseline", score=10.0)
    _completed_validation(project, "candidate", score=5.0)
    canonical = tmp_path / "canonical" / "fe"
    promote_validation(
        project=project, run_id="baseline", canonical_root=canonical
    )
    pointer_path = canonical / "current_best.json"
    pointer = json.loads(pointer_path.read_text())
    pointer[field] = replacement
    _write_json(pointer_path, pointer)
    latest_before = (canonical / "latest_attempt.json").read_bytes()
    parameters_before = (canonical / "final_parameters.json").read_bytes()
    snapshots_before = sorted(
        path.name for path in (canonical / "published_validations").iterdir()
    )

    with pytest.raises(PromotionError, match="Managed pointer"):
        promote_validation(
            project=project,
            run_id="candidate",
            canonical_root=canonical,
            replace_legacy=True,
        )

    assert (canonical / "latest_attempt.json").read_bytes() == latest_before
    assert (canonical / "final_parameters.json").read_bytes() == parameters_before
    assert sorted(
        path.name for path in (canonical / "published_validations").iterdir()
    ) == snapshots_before


def test_managed_pointer_symlink_is_never_accepted(tmp_path: Path) -> None:
    project = _project(tmp_path)
    _completed_validation(project, "baseline", score=10.0)
    _completed_validation(project, "candidate", score=5.0)
    canonical = tmp_path / "canonical" / "fe"
    promote_validation(
        project=project, run_id="baseline", canonical_root=canonical
    )
    pointer = canonical / "current_best.json"
    detached = canonical / "detached_current_best.json"
    pointer.replace(detached)
    try:
        os.symlink(detached, pointer)
    except OSError as exc:
        pytest.skip(f"file symlinks are unavailable: {exc}")

    with pytest.raises(PromotionError, match="regular non-symlink"):
        promote_validation(
            project=project,
            run_id="candidate",
            canonical_root=canonical,
            replace_legacy=True,
        )


def test_normal_python_exception_releases_only_its_own_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    _completed_validation(project, "trial", score=10.0)
    canonical = tmp_path / "canonical" / "fe"

    def fail_snapshot(*_args, **_kwargs):
        raise RuntimeError("injected snapshot failure")

    monkeypatch.setattr("workflow.promotion._create_snapshot", fail_snapshot)
    with pytest.raises(RuntimeError, match="injected snapshot failure"):
        promote_validation(
            project=project,
            run_id="trial",
            canonical_root=canonical,
        )

    assert not (canonical / ".promotion.lock").exists()
    journals = list((canonical / "promotion_journal").glob("*.json"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_text())["phase"] == "started"


def test_keyboard_interrupt_retains_lock_for_deliberate_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = _project(tmp_path)
    _completed_validation(project, "trial", score=10.0)
    canonical = tmp_path / "canonical" / "fe"

    def interrupt_snapshot(*_args, **_kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr("workflow.promotion._create_snapshot", interrupt_snapshot)
    with pytest.raises(KeyboardInterrupt):
        promote_validation(
            project=project,
            run_id="trial",
            canonical_root=canonical,
        )

    lock = canonical / ".promotion.lock"
    assert lock.is_file()
    journals = list((canonical / "promotion_journal").glob("*.json"))
    assert len(journals) == 1
    assert json.loads(journals[0].read_text())["phase"] == "started"


@pytest.mark.parametrize(
    "run_id",
    ["", ".", "..", "../escape", "nested/run", "nested\\run", "C:\\escape"],
)
def test_run_id_cannot_escape_the_managed_pipelines_directory(
    tmp_path: Path,
    run_id: str,
) -> None:
    project = _project(tmp_path)
    canonical = tmp_path / "canonical" / "fe"

    with pytest.raises(PromotionError, match="run_id"):
        promote_validation(
            project=project,
            run_id=run_id,
            canonical_root=canonical,
            dry_run=True,
        )

    assert not canonical.exists()


@pytest.mark.parametrize(
    "managed_name",
    ["published_validations", "archived_legacy", "promotion_journal"],
)
def test_managed_directory_symlink_escape_is_rejected_without_external_writes(
    tmp_path: Path,
    managed_name: str,
) -> None:
    project = _project(tmp_path)
    _completed_validation(project, "trial", score=10.0)
    canonical = tmp_path / "canonical" / "fe"
    canonical.mkdir(parents=True)
    external = tmp_path / f"external-{managed_name}"
    external.mkdir()
    sentinel = external / "sentinel.txt"
    sentinel.write_text("unchanged\n", encoding="utf-8")
    try:
        os.symlink(external, canonical / managed_name, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(PromotionError, match="Managed directory must not be a symlink"):
        promote_validation(
            project=project,
            run_id="trial",
            canonical_root=canonical,
            dry_run=True,
        )

    assert sentinel.read_text(encoding="utf-8") == "unchanged\n"
    assert sorted(path.name for path in external.iterdir()) == ["sentinel.txt"]


def test_source_bundle_mutation_after_preflight_cannot_be_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from workflow import promotion as promotion_module

    project = _project(tmp_path)
    output = _completed_validation(project, "trial", score=10.0)
    canonical = tmp_path / "canonical" / "fe"
    original_create_snapshot = promotion_module._create_snapshot

    def mutate_then_snapshot(*args, **kwargs):
        parameters = output / "final_parameters.json"
        parameters.write_text(
            parameters.read_text(encoding="utf-8") + "\n",
            encoding="utf-8",
        )
        return original_create_snapshot(*args, **kwargs)

    monkeypatch.setattr(
        promotion_module,
        "_create_snapshot",
        mutate_then_snapshot,
    )
    with pytest.raises(PromotionError, match="changed after promotion preflight"):
        promote_validation(
            project=project,
            run_id="trial",
            canonical_root=canonical,
        )

    assert not (canonical / "latest_attempt.json").exists()
    assert not (canonical / "latest_validation.json").exists()
    assert not (canonical / "current_best.json").exists()
    assert not (canonical / "final_parameters.json").exists()
    published = canonical / "published_validations"
    assert published.is_dir()
    assert list(published.iterdir()) == []


def test_existing_snapshot_cannot_inject_self_sealed_publication_metadata(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    _completed_validation(project, "trial", score=10.0)
    canonical = tmp_path / "canonical" / "fe"
    first = promote_validation(
        project=project,
        run_id="trial",
        canonical_root=canonical,
    )
    snapshot = Path(first["snapshot"])
    for name in (
        "current_best.json",
        "final_parameters.json",
        "latest_attempt.json",
        "latest_validation.json",
    ):
        (canonical / name).unlink()
    publication_path = snapshot / "PUBLICATION.json"
    publication = json.loads(publication_path.read_text())
    publication["applicability"] = {"scope": "forged"}
    publication["source_bundle_files"] = {"forged": {"sha256": "0" * 64}}
    _write_json(publication_path, publication)
    files = {
        path.relative_to(snapshot).as_posix(): sha256_file(path).to_dict()
        for path in snapshot.rglob("*")
        if path.is_file() and path.name != "SNAPSHOT_MANIFEST.json"
    }
    _write_json(snapshot / "SNAPSHOT_MANIFEST.json", {
        "schema": "ffopt-published-validation-files-v1",
        "created_at": "forged-but-self-sealed",
        "files": files,
    })

    with pytest.raises(PromotionError, match="Existing publication field"):
        promote_validation(
            project=project,
            run_id="trial",
            canonical_root=canonical,
        )

    assert not (canonical / "current_best.json").exists()
    assert not (canonical / "latest_attempt.json").exists()


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("formal_invariance_status", "invented_formal"),
        ("empirical_transfer_status", "invented_empirical"),
        ("elemental_transferability_claim", "invented_claim"),
        ("physical_transferability_status", "invented_physical"),
        ("applicability_scope", "invented_scope"),
    ],
)
def test_unknown_applicability_enumerations_fail_closed(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    project = _project(tmp_path)
    output = _completed_validation(project, "trial", score=10.0)
    adequacy_path = output / "model_adequacy.json"
    adequacy = json.loads(adequacy_path.read_text())
    summary_path = output / "validation_summary.json"
    summary = json.loads(summary_path.read_text())
    if field == "physical_transferability_status":
        adequacy["physical_transferability"]["status"] = value
        summary["model_adequacy"][field] = value
    elif field == "applicability_scope":
        adequacy[field] = value
        summary["model_adequacy"][field] = value
    else:
        adequacy[field] = value
        adequacy["physical_transferability"][field] = value
        summary["model_adequacy"][field] = value
    _write_json(adequacy_path, adequacy)
    _write_json(summary_path, summary)
    _rewrite_primary_manifest(output)
    canonical = tmp_path / "canonical" / "fe"

    with pytest.raises(PromotionError, match="unknown"):
        promote_validation(
            project=project,
            run_id="trial",
            canonical_root=canonical,
        )

    assert not canonical.exists()


def test_formal_pass_preserves_requires_empirical_validation_claim(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    output = _completed_validation(project, "trial", score=10.0)
    adequacy_path = output / "model_adequacy.json"
    adequacy = json.loads(adequacy_path.read_text())
    adequacy.update({
        "formal_invariance_status": "pass",
        "elemental_transferability_claim": "requires_empirical_validation",
    })
    adequacy["physical_transferability"].update({
        "formal_invariance_status": "pass",
        "elemental_transferability_claim": "requires_empirical_validation",
    })
    _write_json(adequacy_path, adequacy)
    summary_path = output / "validation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["model_adequacy"].update({
        "formal_invariance_status": "pass",
        "elemental_transferability_claim": "requires_empirical_validation",
    })
    _write_json(summary_path, summary)
    _rewrite_primary_manifest(output)

    result = promote_validation(
        project=project,
        run_id="trial",
        canonical_root=tmp_path / "canonical" / "fe",
    )

    assert result["applicability"]["formal_invariance_status"] == "pass"
    assert result["applicability"]["empirical_transfer_status"] == "not_evaluated"
    assert result["applicability"]["elemental_transferability_claim"] == (
        "requires_empirical_validation"
    )


def test_non_elemental_unknown_transferability_is_preserved(
    tmp_path: Path,
) -> None:
    project = _project(tmp_path)
    output = _completed_validation(project, "trial", score=10.0)
    adequacy_path = output / "model_adequacy.json"
    adequacy = json.loads(adequacy_path.read_text())
    adequacy["material"] = {"kind": "molecular", "atom_type_count": 2}
    adequacy.update({
        "formal_invariance_status": "not_applicable",
        "empirical_transfer_status": "not_evaluated",
        "elemental_transferability_claim": "unknown",
    })
    adequacy["physical_transferability"].update({
        "status": "unknown",
        "formal_invariance_status": "not_applicable",
        "empirical_transfer_status": "not_evaluated",
        "elemental_transferability_claim": "unknown",
    })
    _write_json(adequacy_path, adequacy)
    summary_path = output / "validation_summary.json"
    summary = json.loads(summary_path.read_text())
    summary["model_adequacy"].update({
        "physical_transferability_status": "unknown",
        "formal_invariance_status": "not_applicable",
        "empirical_transfer_status": "not_evaluated",
        "elemental_transferability_claim": "unknown",
    })
    _write_json(summary_path, summary)
    _rewrite_primary_manifest(output)

    result = promote_validation(
        project=project,
        run_id="trial",
        canonical_root=tmp_path / "canonical" / "fe",
    )

    assert result["applicability"] == {
        "scope": "validated_material_domain",
        "physical_transferability_status": "unknown",
        "formal_invariance_status": "not_applicable",
        "empirical_transfer_status": "not_evaluated",
        "elemental_transferability_claim": "unknown",
    }
