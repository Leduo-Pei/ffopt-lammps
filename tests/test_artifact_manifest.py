import json

import pytest

from workflow.artifact_manifest import (
    ArtifactManifestError,
    build_artifact_manifest,
    canonical_parameter_key,
    check_artifact_reuse,
    load_artifact_manifest,
    scientific_config_hash,
    validate_artifact_manifest,
    write_artifact_manifest,
)


def _files(tmp_path):
    input_file = tmp_path / "bulk.data"
    output_file = tmp_path / "result.json"
    input_file.write_bytes(b"input-v1\n")
    output_file.write_bytes(b'{"objective": 0.1}\n')
    return {"bulk": input_file}, {"result": output_file}


def _manifest(tmp_path, *, parameters=None, seeds=(202, 101, 101), config=None):
    inputs, outputs = _files(tmp_path)
    manifest = build_artifact_manifest(
        kind="candidate",
        identifier="candidate-0007",
        parameters=parameters or {"sigma": 2.3, "epsilon": 0.2},
        seeds=seeds,
        scientific_config=config or {"cutoff": 12.0, "targets": [2.8665, 7.874]},
        input_artifacts=inputs,
        expected_outputs=outputs,
    )
    path = tmp_path / "artifact_manifest.json"
    write_artifact_manifest(path, manifest)
    return path, inputs, outputs, manifest


def test_parameter_keys_are_stable_and_preserve_parameter_identity():
    first = canonical_parameter_key({"sigma": 2.3, "epsilon": 0.2})
    reordered = canonical_parameter_key({"epsilon": 0.2, "sigma": 2.3})
    assert first == reordered
    assert first.startswith("named:sha256:")
    assert canonical_parameter_key([0.2, 2.3]) != canonical_parameter_key([2.3, 0.2])
    assert canonical_parameter_key([0.0]) == canonical_parameter_key([-0.0])


def test_parameter_and_config_hashes_reject_non_finite_values():
    with pytest.raises(ArtifactManifestError, match="finite"):
        canonical_parameter_key({"epsilon": float("nan")})
    with pytest.raises(ArtifactManifestError, match="finite"):
        scientific_config_hash({"target": float("inf")})


def test_manifest_is_deterministic_sorted_and_idempotently_writable(tmp_path):
    path, inputs, outputs, manifest = _manifest(tmp_path)
    loaded = load_artifact_manifest(path)
    assert loaded == manifest
    assert loaded.seeds == (101, 202)
    assert write_artifact_manifest(path, manifest) == path
    text = path.read_text(encoding="ascii")
    assert text.endswith("\n")
    assert list(json.loads(text)["input_artifacts"]) == ["bulk"]
    assert validate_artifact_manifest(
        path, input_artifacts=inputs, expected_outputs=outputs
    ).valid


def test_immutable_manifest_refuses_different_content(tmp_path):
    path, _inputs, _outputs, _manifest_value = _manifest(tmp_path)
    original = path.read_bytes()
    path.write_text("corrupt but pre-existing\n", encoding="ascii")
    with pytest.raises(ArtifactManifestError, match="immutable"):
        write_artifact_manifest(path, _manifest_value)
    assert path.read_bytes() != original
    assert path.read_text(encoding="ascii") == "corrupt but pre-existing\n"


def test_changed_output_fails_validation_with_specific_reason(tmp_path):
    path, inputs, outputs, _ = _manifest(tmp_path)
    outputs["result"].write_bytes(b'{"objective": 999.9}\n')

    result = validate_artifact_manifest(
        path, input_artifacts=inputs, expected_outputs=outputs
    )
    assert not result.valid
    assert result.issues[0].code == "output_size_mismatch"
    assert "expected" in result.reason


def test_changed_same_size_output_is_detected_by_hash(tmp_path):
    path, inputs, outputs, _ = _manifest(tmp_path)
    original = outputs["result"].read_bytes()
    outputs["result"].write_bytes(b"x" * len(original))

    result = validate_artifact_manifest(
        path, input_artifacts=inputs, expected_outputs=outputs
    )
    assert not result.valid
    assert result.issues[0].code == "output_hash_mismatch"


def test_reuse_checks_parameters_seeds_config_inputs_and_outputs(tmp_path):
    path, inputs, outputs, _ = _manifest(tmp_path)
    common = dict(
        path=path,
        kind="candidate",
        identifier="candidate-0007",
        parameters={"epsilon": 0.2, "sigma": 2.3},
        seeds=[101, 202],
        scientific_config={"targets": [2.8665, 7.874], "cutoff": 12.0},
        input_artifacts=inputs,
        expected_outputs=outputs,
    )
    accepted = check_artifact_reuse(**common)
    assert accepted.reusable
    assert "verified" in accepted.reason

    changed = check_artifact_reuse(**{**common, "parameters": {"epsilon": 0.3, "sigma": 2.3}})
    assert not changed.reusable
    assert any(issue.code == "parameter_key_mismatch" for issue in changed.issues)

    changed = check_artifact_reuse(**{**common, "seeds": [101, 303]})
    assert any(issue.code == "seeds_mismatch" for issue in changed.issues)

    changed = check_artifact_reuse(**{**common, "scientific_config": {"cutoff": 10.0}})
    assert any(issue.code == "scientific_config_mismatch" for issue in changed.issues)

    inputs["bulk"].write_bytes(b"input-v2\n")
    changed = check_artifact_reuse(**common)
    assert any(issue.code.startswith("input_") for issue in changed.issues)


@pytest.mark.parametrize(
    "mutation, expected",
    [
        (lambda raw: raw.update({"unknown": True}), "unknown fields"),
        (lambda raw: raw.update({"manifest_fingerprint": "0" * 64}), "fingerprint"),
        (lambda raw: raw.update({"seeds": [202, 101]}), "sorted and unique"),
        (lambda raw: raw.update({"schema_version": True}), "unsupported manifest schema"),
        (lambda raw: raw.update({"kind": []}), "invalid manifest kind"),
    ],
)
def test_corrupt_or_noncanonical_manifest_fails_closed(tmp_path, mutation, expected):
    path, inputs, outputs, _ = _manifest(tmp_path)
    raw = json.loads(path.read_text(encoding="ascii"))
    mutation(raw)
    path.write_text(json.dumps(raw), encoding="ascii")

    result = validate_artifact_manifest(
        path, input_artifacts=inputs, expected_outputs=outputs
    )
    assert not result.valid
    assert result.manifest is None
    assert result.issues[0].code == "manifest_invalid"
    assert expected in result.reason


def test_truncated_and_duplicate_key_json_fail_closed(tmp_path):
    path, inputs, outputs, _ = _manifest(tmp_path)
    path.write_text('{"schema_version": 1', encoding="ascii")
    result = validate_artifact_manifest(
        path, input_artifacts=inputs, expected_outputs=outputs
    )
    assert not result.valid
    assert "invalid JSON" in result.reason

    path.write_text('{"schema_version": 1, "schema_version": 1}', encoding="ascii")
    result = validate_artifact_manifest(
        path, input_artifacts=inputs, expected_outputs=outputs
    )
    assert not result.valid
    assert "duplicate JSON field" in result.reason


def test_missing_output_means_recompute_instead_of_exception(tmp_path):
    path, inputs, outputs, _ = _manifest(tmp_path)
    outputs["result"].unlink()
    decision = check_artifact_reuse(
        path,
        kind="candidate",
        identifier="candidate-0007",
        parameters={"epsilon": 0.2, "sigma": 2.3},
        seeds=[101, 202],
        scientific_config={"cutoff": 12.0, "targets": [2.8665, 7.874]},
        input_artifacts=inputs,
        expected_outputs=outputs,
    )
    assert not decision.reusable
    assert any(issue.code == "output_unavailable" for issue in decision.issues)


def test_stage_manifest_may_have_no_parameter_vector(tmp_path):
    inputs, outputs = _files(tmp_path)
    manifest = build_artifact_manifest(
        kind="stage",
        identifier="validate",
        parameters=None,
        seeds=[],
        scientific_config={"properties": ["bulk"]},
        input_artifacts=inputs,
        expected_outputs=outputs,
    )
    assert manifest.parameter_key is None
