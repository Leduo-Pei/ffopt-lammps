"""Explicit, provenance-preserving publication of completed validations.

Promotion is deliberately separate from validation.  A source validation is
addressed only by a project and managed pipeline run ID, and publishing never
modifies that pipeline.  Every execution-complete, verified attempt is copied
into an immutable, content-checked snapshot; ``latest_attempt.json`` records
what was most recently considered while ``current_best.json`` changes only
after the comparison policy permits it.
"""

from __future__ import annotations

from contextlib import AbstractContextManager
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import sqlite3
import tempfile
from typing import Any, Mapping
import uuid

from .artifact_manifest import ArtifactManifest, load_artifact_manifest, sha256_file
from .artifacts import pipeline_root
from .material_pipeline import validate_material_stage_outputs
from .project import Project


PROMOTION_SCHEMA_VERSION = 1
POINTER_SCHEMA = "ffopt-promotion-pointer-v1"
PUBLICATION_SCHEMA = "ffopt-published-validation-v1"
SNAPSHOT_MANIFEST_SCHEMA = "ffopt-published-validation-files-v1"

_LEGACY_NAMES = (
    "current_best.json",
    "current",
    "final_parameters.json",
    "final_parameters.lammps",
    "latest_validation.json",
    "FINAL_RESULTS.md",
    "FINAL_VALIDATION.md",
    "TOP_PARAMETERS.csv",
    "TOP_PARAMETERS.json",
    "TOP_PARAMETERS.md",
)
_PUBLICATION_ALLOWLIST = (
    "computed_properties.csv",
    "dynamic_validation_config.json",
    "elastic_candidate.csv",
    "elasticity_dynamic/batch_identity.json",
    "elasticity_dynamic/batch_summary.json",
    "elasticity_dynamic/best_candidate.json",
    "elasticity_dynamic/dynamic_results.csv",
    "elasticity_dynamic/dynamic_seed_results.csv",
    "elasticity_dynamic/finalists_selected.csv",
    "elasticity_dynamic/stage_manifest.json",
    "elasticity_static/batch_identity.json",
    "elasticity_static/batch_summary.json",
    "elasticity_static/best_candidate.json",
    "elasticity_static/finalists_selected.csv",
    "elasticity_static/stage_manifest.json",
    "elasticity_static/static_results.csv",
    "final_atom_parameters.csv",
    "final_parameters.json",
    "final_parameters.lammps",
    "model_adequacy.json",
    "stage_manifest.json",
    "TOP_PARAMETERS.csv",
    "TOP_PARAMETERS.json",
    "top_parameters_manifest.json",
    "TOP_PARAMETERS.md",
    "validation_summary.json",
)
_SAFE_TOKEN = re.compile(r"[^A-Za-z0-9_.-]+")
_MANAGED_DIRECTORIES = (
    "published_validations",
    "archived_legacy",
    "promotion_journal",
)
_APPLICABILITY_SCOPES = {"ordered_sublattice_bulk", "validated_material_domain"}
_FORMAL_STATUSES = {"not_applicable", "not_evaluated", "fail", "pass"}
_EMPIRICAL_STATUSES = {
    "not_applicable",
    "not_evaluated",
    "fail",
    "error",
    "attested_pass",
    "pass",
}
_TRANSFERABILITY_CLAIMS = {
    "unknown",
    "not_applicable",
    "not_established",
    "requires_empirical_validation",
    "not_supported",
    "supported",
}
_PHYSICAL_STATUSES = {
    "unknown",
    "ordered_sublattice_only",
    "not_transferable",
    "incomplete",
    "requires_verified_runner",
    "requires_validation",
    "validated",
}


class PromotionError(ValueError):
    """Raised when a validation cannot be safely published."""


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _timestamp() -> str:
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(value, stream, indent=2, sort_keys=True, ensure_ascii=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _atomic_copy(source: Path, destination: Path) -> None:
    """Atomically refresh one compatibility cache from a published snapshot."""

    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with source.open("rb") as reader, os.fdopen(handle, "wb") as writer:
            shutil.copyfileobj(reader, writer)
            writer.flush()
            os.fsync(writer.fileno())
        if sha256_file(source).to_dict() != sha256_file(temporary).to_dict():
            raise PromotionError(f"Atomic compatibility copy failed verification: {source}")
        os.replace(temporary, destination)
    except Exception:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise PromotionError(f"Cannot read {label} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise PromotionError(f"{label} must contain a JSON object: {path}")
    return value


def _required_json_boolean(document: Mapping[str, Any], field: str) -> bool:
    """Return a schema boolean without accepting truthy JSON lookalikes."""

    value = document.get(field)
    if not isinstance(value, bool):
        raise PromotionError(
            f"Validation summary field {field!r} must be a JSON boolean"
        )
    return value


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
    except ValueError:
        return False
    return True


def _managed_directory(root: Path, name: str, *, create: bool) -> Path:
    """Return one real managed directory without following directory links."""

    if name not in _MANAGED_DIRECTORIES:
        raise PromotionError(f"Unknown managed publication directory: {name}")
    address = root / name
    if address.is_symlink():
        raise PromotionError(f"Managed directory must not be a symlink: {address}")
    if not address.exists():
        if not create:
            return address
        address.mkdir(parents=False)
    if not address.is_dir():
        raise PromotionError(f"Managed publication path is not a directory: {address}")
    resolved = address.resolve(strict=True)
    if resolved.parent != root.resolve():
        raise PromotionError(f"Managed directory escapes canonical root: {address}")
    return resolved


def _validate_managed_directories(root: Path) -> None:
    for name in _MANAGED_DIRECTORIES:
        _managed_directory(root, name, create=False)


def _mapping_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, os.PathLike)):
        raise PromotionError(f"Validation summary does not address input {label!r}")
    return Path(value).expanduser().resolve()


def _nested_mapping(value: Any, *keys: str) -> Any:
    current = value
    for key in keys:
        if not isinstance(current, Mapping):
            return None
        current = current.get(key)
    return current


def _verify_manifest_inputs(
    manifest: ArtifactManifest,
    paths: Mapping[str, Path],
    *,
    label: str,
) -> None:
    """Verify every declared input against one explicitly addressed file."""

    expected = dict(manifest.input_artifacts)
    missing = sorted(set(expected) - set(paths))
    unknown = sorted(set(paths) - set(expected))
    if missing:
        raise PromotionError(f"{label} input path mapping is incomplete: {missing}")
    if unknown:
        raise PromotionError(f"{label} input path mapping has undeclared labels: {unknown}")
    for name, path in paths.items():
        if not path.is_file() or path.is_symlink():
            raise PromotionError(f"{label} input {name!r} is not a regular file: {path}")
        if sha256_file(path) != expected[name]:
            raise PromotionError(f"{label} input hash mismatch for {name!r}: {path}")


def _candidate_manifest_inputs(
    root: Path,
    manifest: ArtifactManifest,
    *,
    protocol: str,
) -> dict[str, Path]:
    labels = sorted(
        name for name, _digest in manifest.input_artifacts
        if name.startswith("candidate_manifest_")
    )
    if not labels:
        return {}
    pattern = (
        "candidate_runs/*/static/artifact_manifest.json"
        if protocol == "static"
        else "candidate_runs/*/dynamic/seed_*/artifact_manifest.json"
    )
    candidates = sorted(path.resolve() for path in root.glob(pattern) if path.is_file())
    expected = dict(manifest.input_artifacts)
    resolved: dict[str, Path] = {}
    unused = set(candidates)
    for name in labels:
        matches = [path for path in unused if sha256_file(path) == expected[name]]
        if len(matches) != 1:
            raise PromotionError(
                f"{root.name} input {name!r} has {len(matches)} matching candidate manifests"
            )
        load_artifact_manifest(matches[0])
        resolved[name] = matches[0]
        unused.remove(matches[0])
    return resolved


def _verify_nested_validation_inputs(
    root: Path,
    summary: Mapping[str, Any],
    manifest: ArtifactManifest,
) -> None:
    """Close the hash chain from final validation to every nested evidence input."""

    parameter_source = _nested_mapping(summary, "parameter_source", "path")
    top = summary.get("top_parameters_report")
    expected_main = dict(manifest.input_artifacts)
    main_paths: dict[str, Path] = {}
    if "compiled_config" in expected_main:
        main_paths["compiled_config"] = _mapping_path(
            summary.get("config"), label="compiled_config"
        )
    if "parameter_source" in expected_main:
        main_paths["parameter_source"] = _mapping_path(
            parameter_source, label="parameter_source"
        )
    fixed_main: dict[str, Path] = {
        "elastic_candidate": root / "elastic_candidate.csv",
        "static_manifest": root / "elasticity_static" / "stage_manifest.json",
        "static_results": root / "elasticity_static" / "static_results.csv",
        "dynamic_validation_config": root / "dynamic_validation_config.json",
        "dynamic_manifest": root / "elasticity_dynamic" / "stage_manifest.json",
        "dynamic_results": root / "elasticity_dynamic" / "dynamic_results.csv",
        "dynamic_seed_results": root / "elasticity_dynamic" / "dynamic_seed_results.csv",
    }
    for name, path in fixed_main.items():
        if name in expected_main:
            main_paths[name] = path.resolve()
    structure_result = _nested_mapping(summary, "structural_evidence", "result")
    structure_manifest = _nested_mapping(summary, "structural_evidence", "manifest")
    if "structure_result" in expected_main:
        path = _mapping_path(structure_result, label="structure_result")
        if not _is_relative_to(path, (root / "candidate_runs").resolve()):
            raise PromotionError("Structural result path escapes validate/candidate_runs")
        main_paths["structure_result"] = path
    if "structure_manifest" in expected_main:
        path = _mapping_path(structure_manifest, label="structure_manifest")
        if not _is_relative_to(path, (root / "candidate_runs").resolve()):
            raise PromotionError("Structural manifest path escapes validate/candidate_runs")
        main_paths["structure_manifest"] = path
    if "top_static_ranking" in expected_main:
        main_paths["top_static_ranking"] = _mapping_path(
            _nested_mapping(top, "static_ranking"), label="top_static_ranking"
        )
    if "top_dynamic_ranking" in expected_main:
        main_paths["top_dynamic_ranking"] = _mapping_path(
            _nested_mapping(top, "dynamic_ranking"), label="top_dynamic_ranking"
        )
    _verify_manifest_inputs(manifest, main_paths, label="material validation")

    static_root = root / "elasticity_static"
    if static_root.exists():
        static_manifest = load_artifact_manifest(static_root / "stage_manifest.json")
        static_paths: dict[str, Path] = {
            "batch_identity": static_root / "batch_identity.json",
            "config": main_paths["compiled_config"],
            "parameters": root / "elastic_candidate.csv",
        }
        static_paths.update(
            _candidate_manifest_inputs(static_root, static_manifest, protocol="static")
        )
        _verify_manifest_inputs(static_manifest, static_paths, label="static elasticity")

    dynamic_root = root / "elasticity_dynamic"
    if dynamic_root.exists():
        dynamic_manifest = load_artifact_manifest(dynamic_root / "stage_manifest.json")
        dynamic_paths: dict[str, Path] = {
            "batch_identity": dynamic_root / "batch_identity.json",
            "config": root / "dynamic_validation_config.json",
            "parameters": static_root / "static_results.csv",
        }
        dynamic_paths.update(
            _candidate_manifest_inputs(dynamic_root, dynamic_manifest, protocol="dynamic")
        )
        _verify_manifest_inputs(dynamic_manifest, dynamic_paths, label="dynamic elasticity")


class _PromotionLock(AbstractContextManager["_PromotionLock"]):
    """Fail-closed inter-process lock for one canonical publication root."""

    def __init__(self, root: Path) -> None:
        self.path = root / ".promotion.lock"
        self.token = uuid.uuid4().hex

    def __enter__(self) -> "_PromotionLock":
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps({
            "token": self.token,
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "created_at": _utc_now(),
        }, sort_keys=True).encode("utf-8")
        try:
            descriptor = os.open(
                self.path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600
            )
        except FileExistsError as exc:
            raise PromotionError(
                f"Another promotion holds the lock {self.path}; inspect the "
                "promotion journal before removing a stale lock"
            ) from exc
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        # Ordinary handled Python errors release this invocation's own lock so
        # the journal can be inspected and the explicit command retried.  A
        # process-level interruption (Ctrl-C/SystemExit) retains the token just
        # like a crash, forcing deliberate stale-lock review before takeover.
        if exc_type is not None and not issubclass(exc_type, Exception):
            return
        try:
            value = _load_json(self.path, label="promotion lock")
        except PromotionError:
            return
        if value.get("token") == self.token:
            self.path.unlink(missing_ok=True)


def _normalized_applicability(
    summary: Mapping[str, Any],
    adequacy: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize old/new adequacy fields without trusting summary-only claims."""

    summary_adequacy = summary.get("model_adequacy")
    if not isinstance(summary_adequacy, Mapping):
        summary_adequacy = {}
    physical = adequacy.get("physical_transferability")
    if not isinstance(physical, Mapping):
        physical = {}
    material = adequacy.get("material")
    if not isinstance(material, Mapping):
        material = {}

    missing = object()

    def report_value(
        field: str,
        *locations: tuple[str, Mapping[str, Any], str],
    ) -> Any:
        values = [
            (label, mapping[key])
            for label, mapping, key in locations
            if key in mapping
        ]
        if values and any(value != values[0][1] for _label, value in values[1:]):
            labels = ", ".join(label for label, _value in values)
            raise PromotionError(
                f"model_adequacy.json has conflicting {field!r} fields: {labels}"
            )
        return values[0][1] if values else missing

    report_fields = {
        "formal_invariance_status": report_value(
            "formal_invariance_status",
            ("root", adequacy, "formal_invariance_status"),
            ("physical_transferability", physical, "formal_invariance_status"),
        ),
        "empirical_transfer_status": report_value(
            "empirical_transfer_status",
            ("root", adequacy, "empirical_transfer_status"),
            ("physical_transferability", physical, "empirical_transfer_status"),
        ),
        "elemental_transferability_claim": report_value(
            "elemental_transferability_claim",
            ("root", adequacy, "elemental_transferability_claim"),
            (
                "physical_transferability",
                physical,
                "elemental_transferability_claim",
            ),
        ),
        "physical_transferability_status": report_value(
            "physical_transferability_status",
            ("root", adequacy, "physical_transferability_status"),
            ("physical_transferability", physical, "status"),
        ),
    }
    for field, report_status in report_fields.items():
        if (
            field in summary_adequacy
            and report_status is not missing
            and summary_adequacy[field] != report_status
        ):
            raise PromotionError(
                "validation_summary.json model_adequacy conflicts with "
                f"model_adequacy.json for {field!r}"
            )

    def conservative_status(
        field: str,
        default: str,
        allowed: set[str],
    ) -> str:
        value = report_fields[field]
        if value is missing or value is None or value == "":
            return default
        if not isinstance(value, str):
            raise PromotionError(
                f"model_adequacy.json field {field!r} must be a string"
            )
        if value not in allowed:
            raise PromotionError(
                f"model_adequacy.json field {field!r} has unknown status {value!r}"
            )
        return value

    formal = conservative_status(
        "formal_invariance_status", "not_evaluated", _FORMAL_STATUSES
    )
    empirical = conservative_status(
        "empirical_transfer_status", "not_evaluated", _EMPIRICAL_STATUSES
    )
    raw_claim = conservative_status(
        "elemental_transferability_claim",
        "not_established",
        _TRANSFERABILITY_CLAIMS,
    )
    if formal == "fail" or empirical == "fail" or raw_claim == "not_supported":
        claim = "not_supported"
    elif raw_claim == "supported" and formal == "pass" and empirical == "pass":
        claim = "supported"
    elif (
        raw_claim == "requires_empirical_validation"
        and formal == "pass"
        and empirical in {"not_evaluated", "not_applicable"}
    ):
        claim = "requires_empirical_validation"
    elif raw_claim == "not_applicable":
        claim = "not_applicable"
    elif raw_claim == "unknown" and str(material.get("kind", "")) != "elemental":
        claim = "unknown"
    else:
        claim = "not_established"

    physical_status = conservative_status(
        "physical_transferability_status",
        "requires_validation",
        _PHYSICAL_STATUSES,
    )
    if physical_status == "validated" and claim != "supported":
        physical_status = "requires_validation"

    summary_scope = summary_adequacy.get("applicability_scope")
    report_scope = adequacy.get("applicability_scope")
    if summary_scope is not None and report_scope is not None and summary_scope != report_scope:
        raise PromotionError(
            "validation_summary.json model_adequacy conflicts with "
            "model_adequacy.json for 'applicability_scope'"
        )
    explicit_scope = report_scope
    if explicit_scope is not None:
        if not isinstance(explicit_scope, str):
            raise PromotionError(
                "model_adequacy.json applicability_scope must be a string"
            )
        if explicit_scope not in _APPLICABILITY_SCOPES:
            raise PromotionError(
                "model_adequacy.json has unknown applicability_scope "
                f"{explicit_scope!r}"
            )
    try:
        atom_type_count = int(material.get("atom_type_count", 0) or 0)
    except (TypeError, ValueError) as exc:
        raise PromotionError(
            "model_adequacy.json material.atom_type_count must be an integer"
        ) from exc
    elemental_multitype = (
        str(material.get("kind", "")) == "elemental"
        and atom_type_count > 1
    )
    scope = str(
        explicit_scope
        or ("ordered_sublattice_bulk" if elemental_multitype else "validated_material_domain")
    )
    return {
        "scope": scope,
        "physical_transferability_status": physical_status,
        "formal_invariance_status": formal,
        "empirical_transfer_status": empirical,
        "elemental_transferability_claim": claim,
    }


def _source_validation(project: Project, run_id: str) -> dict[str, Any]:
    """Resolve and validate exactly ``pipeline_root(project, run_id)/validate``."""

    root = pipeline_root(project, run_id)
    pipelines = (project.run_root / "pipelines").resolve()
    if root.parent != pipelines:
        raise PromotionError(
            f"Managed pipeline run must resolve as a direct child of {pipelines}: {root}"
        )
    database = root / "state.sqlite"
    if database.is_symlink() or not database.is_file():
        raise PromotionError(
            f"Managed pipeline state must be a regular non-symlink file: {database}"
        )
    # WorkflowState is intentionally not used here: its constructor enables
    # WAL and creates schemas.  Promotion audits the completed source strictly
    # read-only, including during --dry-run.
    try:
        connection = sqlite3.connect(
            f"{database.as_uri()}?mode=ro", uri=True, timeout=10.0
        )
        connection.row_factory = sqlite3.Row
        try:
            metadata_rows = connection.execute(
                "SELECT key, value FROM metadata"
            ).fetchall()
            row = connection.execute(
                "SELECT status, attempt, output_dir, artifacts_json, finished_at "
                "FROM stages WHERE name=?",
                ("validate",),
            ).fetchone()
        finally:
            connection.close()
    except (sqlite3.Error, OSError) as exc:
        raise PromotionError(f"Cannot audit pipeline state read-only: {database}: {exc}") from exc
    try:
        metadata = {
            str(item["key"]): json.loads(item["value"])
            for item in metadata_rows
        }
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise PromotionError(f"Pipeline metadata is malformed: {database}") from exc
    if metadata.get("run_id") != run_id:
        raise PromotionError(
            f"Pipeline state run_id {metadata.get('run_id')!r} does not match {run_id!r}"
        )
    if metadata.get("project") != project.name:
        raise PromotionError(
            f"Pipeline state project {metadata.get('project')!r} does not match "
            f"{project.name!r}"
        )
    if row is None:
        raise PromotionError("Pipeline state has no validate stage")
    if row["status"] != "completed" or not row["finished_at"]:
        raise PromotionError(
            "Validate stage must be completed with a persisted finished_at timestamp"
        )
    expected_address = root / "validate"
    if expected_address.is_symlink() or not expected_address.is_dir():
        raise PromotionError(
            f"Managed validate output must be a real directory, not a symlink: "
            f"{expected_address}"
        )
    expected = expected_address.resolve()
    try:
        recorded = Path(row["output_dir"]).resolve(strict=True)
    except (OSError, RuntimeError) as exc:
        raise PromotionError(f"Validate output_dir cannot be resolved: {row['output_dir']}") from exc
    if recorded != expected:
        raise PromotionError(
            f"Validate output_dir must equal the managed path {expected}, got {recorded}"
        )
    try:
        expected_artifacts = json.loads(row["artifacts_json"])
    except (TypeError, json.JSONDecodeError) as exc:
        raise PromotionError("Validate stage artifacts_json is malformed") from exc
    if not isinstance(expected_artifacts, list):
        raise PromotionError("Validate stage artifacts_json must contain a list")
    valid, reason = validate_material_stage_outputs(
        expected,
        command_token="material-validate",
        expected_artifacts=expected_artifacts,
    )
    if not valid:
        raise PromotionError(f"Validate artifact contract failed: {reason}")

    for subdirectory, token in (
        (expected / "elasticity_static", "static"),
        (expected / "elasticity_dynamic", "finalists"),
    ):
        if not subdirectory.exists():
            continue
        if subdirectory.is_symlink() or not subdirectory.is_dir():
            raise PromotionError(
                f"Validation substage must be a real directory: {subdirectory}"
            )
        sub_valid, sub_reason = validate_material_stage_outputs(
            subdirectory,
            command_token=token,
            expected_artifacts=[subdirectory / "stage_manifest.json"],
        )
        if not sub_valid:
            raise PromotionError(
                f"Validation substage artifact contract failed for "
                f"{subdirectory.name}: {sub_reason}"
            )

    summary = _load_json(expected / "validation_summary.json", label="validation summary")
    parameters = _load_json(expected / "final_parameters.json", label="final parameters")
    adequacy_path = expected / "model_adequacy.json"
    adequacy = (
        _load_json(adequacy_path, label="model adequacy")
        if adequacy_path.is_file()
        else {}
    )
    manifest = load_artifact_manifest(expected / "stage_manifest.json")
    _verify_nested_validation_inputs(expected, summary, manifest)
    if summary.get("execution_complete") is not True:
        raise PromotionError("Validation summary is not execution_complete")
    status = str(summary.get("status", ""))
    if status not in {"accepted", "best_effort", "rejected"}:
        raise PromotionError(f"Validation has an unknown scientific status: {status!r}")
    if summary.get("parameter_key") != parameters.get("parameter_key"):
        raise PromotionError("Validation summary and final parameters disagree on parameter_key")
    if manifest.parameter_key != summary.get("parameter_key"):
        raise PromotionError("Validation manifest and summary disagree on parameter_key")
    dynamic = summary.get("dynamic_elastic_evidence")
    dynamic_row = dynamic.get("row") if isinstance(dynamic, Mapping) else None
    score_basis = "dynamic_elastic_evidence.row.mechanical_max_error_percent"
    raw_score = (
        dynamic_row.get("mechanical_max_error_percent")
        if isinstance(dynamic_row, Mapping)
        else None
    )
    if raw_score is None:
        raw_score = summary.get("mechanical_max_error_percent")
        score_basis = "validation_summary.mechanical_max_error_percent"
    try:
        score = float(raw_score) if raw_score is not None else None
    except (TypeError, ValueError) as exc:
        raise PromotionError("Validation mechanical comparison score is not numeric") from exc
    if score is not None and not math.isfinite(score):
        raise PromotionError("Validation mechanical comparison score must be finite")
    if status in {"accepted", "best_effort"} and score is None:
        raise PromotionError("Eligible validation has no finite mechanical comparison score")

    protocol = {
        "validation_schema_version": summary.get("schema_version"),
        "scientific_config_sha256": manifest.scientific_config_sha256,
        "validation_seeds": list(manifest.seeds),
    }
    hard_gate_pass = _required_json_boolean(summary, "hard_gate_pass")
    within_tier = _required_json_boolean(
        summary, "within_mechanical_quality_tier"
    )
    if status == "accepted" and (not hard_gate_pass or not within_tier):
        raise PromotionError(
            "Validation status 'accepted' requires hard_gate_pass=true and "
            "within_mechanical_quality_tier=true"
        )
    if status == "best_effort" and (not hard_gate_pass or within_tier):
        raise PromotionError(
            "Validation status 'best_effort' requires hard_gate_pass=true and "
            "within_mechanical_quality_tier=false"
        )
    if status == "rejected" and hard_gate_pass:
        raise PromotionError(
            "Validation status 'rejected' requires hard_gate_pass=false"
        )

    quality = {
        "status": status,
        "hard_gate_pass": hard_gate_pass,
        "within_mechanical_quality_tier": within_tier,
        "mechanical_max_error_percent": score,
        "mechanical_score_basis": score_basis if score is not None else None,
    }
    publication_bundle_files = _publication_bundle_inventory(expected)
    return {
        "pipeline_root": root,
        "validate_dir": expected,
        "state_database": database,
        "state_finished_at": row["finished_at"],
        "state_attempt": int(row["attempt"]),
        "manifest_fingerprint": manifest.manifest_fingerprint,
        "parameter_key": summary.get("parameter_key"),
        "quality": quality,
        "protocol": protocol,
        "applicability": _normalized_applicability(summary, adequacy),
        "publication_bundle_files": publication_bundle_files,
    }


def _legacy_files(
    root: Path,
    *,
    managed_latest: bool,
    managed_current: bool,
) -> list[Path]:
    managed = set()
    if managed_latest:
        managed.add("latest_validation.json")
    if managed_current:
        managed.update({"current_best.json", "final_parameters.json"})
    paths = [root / name for name in _LEGACY_NAMES if name not in managed]
    if root.is_dir():
        paths.extend(sorted(root.glob("FINAL_RESULTS*.md")))
    unique: dict[str, Path] = {}
    for path in paths:
        if path.exists() or path.is_symlink():
            if not path.is_file() and not path.is_symlink():
                raise PromotionError(f"Legacy canonical artifact is not a regular file: {path}")
            unique[path.name] = path
    return [unique[name] for name in sorted(unique)]


def _validate_legacy_symlinks(
    root: Path,
    files: list[Path],
    *,
    recognized_a11: bool,
) -> None:
    for path in files:
        if not path.is_symlink():
            continue
        if not recognized_a11:
            raise PromotionError(
                "Legacy symlinks are accepted only for the strictly sealed A11 "
                f"one-off layout: {path}"
            )
        try:
            target = path.resolve(strict=True)
        except (OSError, RuntimeError) as exc:
            raise PromotionError(f"Legacy symlink is broken: {path}") from exc
        if not _is_relative_to(target, root.resolve()):
            raise PromotionError(f"Legacy symlink escapes canonical root: {path}")


def _file_inventory(root: Path, *, exclude: set[str] | None = None) -> dict[str, Any]:
    excluded = exclude or set()
    inventory: dict[str, Any] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise PromotionError(f"Snapshot contains a symlink: {path}")
        if path.is_dir():
            continue
        relative = path.relative_to(root).as_posix()
        if relative in excluded:
            continue
        if not path.is_file():
            raise PromotionError(f"Snapshot contains a non-regular file: {path}")
        inventory[relative] = sha256_file(path).to_dict()
    return inventory


def _publication_bundle_inventory(source: Path) -> dict[str, Any]:
    """Freeze hashes for the fixed compact source bundle during preflight."""

    inventory: dict[str, Any] = {}
    for relative in _PUBLICATION_ALLOWLIST:
        origin = source / Path(relative)
        if not origin.exists():
            continue
        if not origin.is_file() or origin.is_symlink():
            raise PromotionError(f"Publication source is not a regular file: {origin}")
        inventory[relative] = sha256_file(origin).to_dict()
    required = {
        "validation_summary.json",
        "final_parameters.json",
        "stage_manifest.json",
    }
    missing = sorted(required - set(inventory))
    if missing:
        raise PromotionError(f"Publication bundle is missing required files: {missing}")
    return inventory


def _copy_publication_bundle(
    source: Path,
    destination: Path,
    *,
    expected_inventory: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Copy the compact bundle and bind it to preflight hashes when supplied."""

    source_inventory = _publication_bundle_inventory(source)
    if expected_inventory is not None and source_inventory != dict(expected_inventory):
        raise PromotionError(
            "Publication source bundle changed after promotion preflight"
        )
    copied: dict[str, Any] = {}
    for relative, source_hash in source_inventory.items():
        origin = source / Path(relative)
        target = destination / Path(relative)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, target)
        target_hash = sha256_file(target).to_dict()
        if target_hash != source_hash:
            raise PromotionError(f"Publication copy hash mismatch: {origin}")
        copied[relative] = source_hash
    if (destination / "candidate_runs").exists():
        raise PromotionError("candidate_runs must never enter a published validation bundle")
    return copied


def _verify_inventory(root: Path, document: Mapping[str, Any]) -> None:
    if document.get("schema") != SNAPSHOT_MANIFEST_SCHEMA:
        raise PromotionError(f"Unknown snapshot manifest schema under {root}")
    files = document.get("files")
    if not isinstance(files, dict):
        raise PromotionError(f"Snapshot manifest has no files mapping under {root}")
    actual_names = set(_file_inventory(root, exclude={"SNAPSHOT_MANIFEST.json"}))
    if actual_names != set(files):
        raise PromotionError(f"Published snapshot file set changed: {root}")
    for relative, expected in files.items():
        digest = sha256_file(root / relative).to_dict()
        if digest != expected:
            raise PromotionError(f"Published snapshot artifact changed: {root / relative}")


def _publication_from_pointer(root: Path, pointer_name: str) -> dict[str, Any] | None:
    pointer_path = root / pointer_name
    if not pointer_path.exists() and not pointer_path.is_symlink():
        return None
    if pointer_path.is_symlink() or not pointer_path.is_file():
        raise PromotionError(
            f"Managed pointer must be a regular non-symlink file: {pointer_path}"
        )
    pointer = _load_json(pointer_path, label=pointer_name)
    if pointer.get("schema") != POINTER_SCHEMA:
        raise PromotionError(f"Unknown pointer schema in {pointer_path}")
    relative = pointer.get("snapshot")
    if not isinstance(relative, str) or not relative:
        raise PromotionError(f"Pointer has no snapshot path: {pointer_path}")
    relative_path = Path(relative)
    if relative_path.is_absolute():
        raise PromotionError(f"Pointer snapshot path must be relative: {pointer_path}")
    snapshot_address = root / relative_path
    if snapshot_address.is_symlink() or not snapshot_address.is_dir():
        raise PromotionError(
            f"Pointer snapshot must be a real published directory: {pointer_path}"
        )
    snapshot = snapshot_address.resolve(strict=True)
    published = _managed_directory(root, "published_validations", create=False)
    if not published.exists():
        raise PromotionError(
            f"Managed pointer has no published_validations directory: {pointer_path}"
        )
    if not _is_relative_to(snapshot, published):
        raise PromotionError(f"Pointer escapes published_validations: {pointer_path}")
    canonical_relative = snapshot.relative_to(root.resolve()).as_posix()
    if relative != canonical_relative:
        raise PromotionError(f"Pointer snapshot path is not canonical: {pointer_path}")
    inventory = _load_json(
        snapshot / "SNAPSHOT_MANIFEST.json", label="snapshot manifest"
    )
    _verify_inventory(snapshot, inventory)
    publication = _load_json(snapshot / "PUBLICATION.json", label="publication")
    if publication.get("schema") != PUBLICATION_SCHEMA:
        raise PromotionError(f"Unknown publication schema under {snapshot}")
    if publication.get("publication_id") != snapshot.name:
        raise PromotionError(
            f"Publication ID does not match its immutable directory: {snapshot}"
        )
    published_bundle = publication.get("source_bundle_files")
    if not isinstance(published_bundle, Mapping):
        raise PromotionError(f"Publication has no source bundle inventory: {snapshot}")
    if dict(published_bundle) != _publication_bundle_inventory(snapshot):
        raise PromotionError(
            f"Publication source bundle inventory does not match its snapshot: {snapshot}"
        )
    for field in (
        "publication_id",
        "parameter_key",
        "quality",
        "protocol",
        "applicability",
    ):
        if pointer.get(field) != publication.get(field):
            raise PromotionError(
                f"Managed pointer field {field!r} does not match its sealed "
                f"publication: {pointer_path}"
            )
    expected_validation = f"{relative}/validation_summary.json"
    if pointer.get("validation_summary") != expected_validation:
        raise PromotionError(
            f"Managed pointer validation_summary does not address its sealed "
            f"snapshot: {pointer_path}"
        )
    validation_path = snapshot / "validation_summary.json"
    if validation_path.is_symlink() or not validation_path.is_file():
        raise PromotionError(
            f"Managed pointer snapshot has no regular validation summary: {pointer_path}"
        )
    publication["_snapshot"] = str(snapshot)
    return publication


def _is_recognized_a11_oneoff(root: Path) -> bool:
    """Recognize only the sealed A11 symlink-based publication layout."""

    current_link = root / "current"
    current_pointer = root / "current_best.json"
    final_parameters = root / "final_parameters.json"
    latest_validation = root / "latest_validation.json"
    if not all(path.is_symlink() for path in (
        current_link,
        current_pointer,
        final_parameters,
        latest_validation,
    )):
        return False
    try:
        target = current_link.resolve(strict=True)
        if not target.is_dir() or not _is_relative_to(target, root.resolve()):
            return False
        pointer_target = current_pointer.resolve(strict=True)
        final_target = final_parameters.resolve(strict=True)
        latest_target = latest_validation.resolve(strict=True)
        if not all(_is_relative_to(path, target) for path in (
            pointer_target,
            final_target,
            latest_target,
        )):
            return False
        document = _load_json(current_pointer, label="A11 one-off current best")
    except (OSError, RuntimeError, PromotionError):
        return False
    if document.get("schema_version") != 1 or document.get("kind") != "current_best":
        return False
    if document.get("classification") != "current_best_baseline":
        return False
    if document.get("quality") not in {"accepted", "best_effort"}:
        return False
    parameter_key = document.get("parameter_key")
    if not isinstance(parameter_key, str) or not parameter_key:
        return False
    try:
        declared_snapshot = Path(str(document["snapshot"])).expanduser().resolve()
        declared_parameters = Path(
            str(document["final_parameters"])
        ).expanduser().resolve()
        declared_validation = Path(
            str(document["validation_summary"])
        ).expanduser().resolve()
    except (KeyError, OSError, RuntimeError):
        return False
    fields_valid = (
        declared_snapshot == target
        and declared_parameters == final_target
        and declared_validation.is_file()
        and _is_relative_to(declared_validation, target)
    )
    if not fields_valid:
        return False
    try:
        promotion = _load_json(
            target / "PROMOTION_MANIFEST.json", label="A11 promotion manifest"
        )
        if (
            promotion.get("schema_version") != 1
            or promotion.get("kind") != "ffopt_baseline_promotion"
            or promotion.get("classification") != document.get("classification")
            or promotion.get("quality") != document.get("quality")
            or promotion.get("parameter_key") != parameter_key
            or promotion.get("scope") != "ordered_sublattice_bulk"
            or document.get("scope") != promotion.get("scope")
            or promotion.get("elemental_transferability") != "requires_validation"
            or document.get("elemental_transferability")
            != promotion.get("elemental_transferability")
        ):
            return False
        manifest_path = target / "MANIFEST.sha256"
        listed: dict[str, str] = {}
        for raw_line in manifest_path.read_text(encoding="ascii").splitlines():
            digest, separator, relative = raw_line.partition("  ")
            if (
                not separator
                or not re.fullmatch(r"[0-9a-f]{64}", digest)
                or not relative
            ):
                return False
            address = target / Path(relative)
            if address.is_symlink():
                return False
            candidate = address.resolve(strict=True)
            if not _is_relative_to(candidate, target):
                return False
            if sha256_file(candidate).sha256 != digest:
                return False
            listed[Path(relative).as_posix()] = digest
        actual: set[str] = set()
        for directory, names, files in os.walk(target, followlinks=False):
            directory_path = Path(directory)
            if any((directory_path / name).is_symlink() for name in names):
                return False
            for name in files:
                path = directory_path / name
                if path.is_symlink():
                    return False
                relative = path.relative_to(target).as_posix()
                if relative != "MANIFEST.sha256":
                    actual.add(relative)
        return actual == set(listed)
    except (OSError, RuntimeError, PromotionError, ValueError):
        return False


def _read_canonical_pointers(
    root: Path,
    *,
    replace_legacy: bool,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, bool]:
    """Read current/latest independently; only known A11 legacy is replaceable."""

    legacy_a11 = False
    try:
        current = _publication_from_pointer(root, "current_best.json")
    except PromotionError:
        if not replace_legacy or not _is_recognized_a11_oneoff(root):
            raise
        current = None
        legacy_a11 = True

    # Never let a bad latest-attempt pointer erase a valid current baseline.
    # A11 had no latest_attempt.json, so there is no legacy exception here.
    latest = _publication_from_pointer(root, "latest_attempt.json")
    return current, latest, legacy_a11


def _comparison(
    candidate: Mapping[str, Any], current: Mapping[str, Any] | None
) -> tuple[str, str]:
    if current is None:
        return "promote", "no_current_best"
    if candidate["protocol"] != current.get("protocol"):
        return "incomparable", "scientific_validation_protocol_changed"
    candidate_quality = candidate["quality"]
    current_quality = current.get("quality", {})
    rank = {"best_effort": 1, "accepted": 2}
    candidate_rank = rank.get(str(candidate_quality.get("status")), 0)
    current_rank = rank.get(str(current_quality.get("status")), 0)
    if candidate_rank > current_rank:
        return "promote", "higher_quality_status"
    if candidate_rank < current_rank:
        return "downgrade", "lower_quality_status"
    try:
        candidate_score = float(candidate_quality["mechanical_max_error_percent"])
        current_score = float(current_quality["mechanical_max_error_percent"])
    except (KeyError, TypeError, ValueError):
        return "incomparable", "missing_comparable_mechanical_score"
    tolerance = 1.0e-12 * max(1.0, abs(candidate_score), abs(current_score))
    if candidate_score < current_score - tolerance:
        return "promote", "lower_mechanical_max_error"
    if candidate_score > current_score + tolerance:
        return "downgrade", "higher_mechanical_max_error"
    if candidate.get("parameter_key") == current.get("parameter_key"):
        return "equivalent", "same_parameter_and_quality"
    return "equivalent", "equal_quality_score"


def _pointer(publication: Mapping[str, Any], relative_snapshot: str) -> dict[str, Any]:
    return {
        "schema": POINTER_SCHEMA,
        "updated_at": _utc_now(),
        "publication_id": publication["publication_id"],
        "parameter_key": publication["parameter_key"],
        "snapshot": relative_snapshot,
        "quality": publication["quality"],
        "protocol": publication["protocol"],
        "applicability": publication["applicability"],
        "validation_summary": f"{relative_snapshot}/validation_summary.json",
    }


def _legacy_identity(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        target_text = os.readlink(path)
        resolved = path.resolve(strict=True)
        identity: dict[str, Any] = {
            "kind": "symlink",
            "link_target": target_text,
            "resolved_path": str(resolved),
            "resolved_kind": "directory" if resolved.is_dir() else "file",
        }
        if resolved.is_file():
            identity["resolved_artifact"] = sha256_file(resolved).to_dict()
        return identity
    if not path.is_file():
        raise PromotionError(f"Legacy canonical entry is not a regular file: {path}")
    return {"kind": "file", "artifact": sha256_file(path).to_dict()}


def _archive_legacy(root: Path, files: list[Path], transaction_id: str) -> Path | None:
    if not files:
        return None
    parent = _managed_directory(root, "archived_legacy", create=True)
    destination = parent / transaction_id
    if destination.exists():
        raise PromotionError(f"Legacy archive already exists: {destination}")
    staging = Path(tempfile.mkdtemp(prefix=".legacy-", dir=parent))
    try:
        expected: dict[str, Any] = {}
        current_target_files: dict[str, Any] | None = None
        for source in files:
            identity = _legacy_identity(source)
            expected[source.name] = identity
            if identity["kind"] == "file" or identity.get("resolved_kind") == "file":
                target = staging / source.name
                shutil.copy2(source.resolve(strict=True), target)
                expected_hash = (
                    identity["artifact"]
                    if identity["kind"] == "file"
                    else identity["resolved_artifact"]
                )
                if sha256_file(target).to_dict() != expected_hash:
                    raise PromotionError(f"Legacy archive copy verification failed: {source}")
            elif source.name == "current":
                # Do not recursively follow a directory symlink.  Copy only
                # the same fixed, compact scientific publication allowlist.
                current_target_files = _copy_publication_bundle(
                    source.resolve(strict=True), staging / "current_target"
                )
            else:
                raise PromotionError(
                    f"Unsupported legacy directory symlink cannot be archived: {source}"
                )
        _atomic_json(staging / "ARCHIVE_MANIFEST.json", {
            "schema": "ffopt-legacy-canonical-archive-v1",
            "created_at": _utc_now(),
            "entries": expected,
            "current_target_files": current_target_files,
        })
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination


def _create_snapshot(
    root: Path,
    source: Mapping[str, Any],
    project: Project,
    run_id: str,
    transaction_id: str,
) -> tuple[Path, dict[str, Any]]:
    parent = _managed_directory(root, "published_validations", create=True)
    safe_run = _SAFE_TOKEN.sub("_", run_id).strip("._") or "run"
    publication_id = f"{safe_run}-{source['manifest_fingerprint'][:16]}"
    destination = parent / publication_id
    publication_identity = {
        "schema": PUBLICATION_SCHEMA,
        "schema_version": PROMOTION_SCHEMA_VERSION,
        "publication_id": publication_id,
        "project": project.name,
        "project_file": str(project.path),
        "run_id": run_id,
        "source_pipeline_root": str(source["pipeline_root"]),
        "source_validate_dir": str(source["validate_dir"]),
        "source_state_database": str(source["state_database"]),
        "source_validate_finished_at": source["state_finished_at"],
        "source_validate_attempt": source["state_attempt"],
        "source_manifest_fingerprint": source["manifest_fingerprint"],
        "parameter_key": source["parameter_key"],
        "quality": source["quality"],
        "protocol": source["protocol"],
        "applicability": source["applicability"],
        "source_bundle_files": source["publication_bundle_files"],
    }
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise PromotionError(
                f"Existing publication must be a real immutable directory: {destination}"
            )
        publication = _load_json(destination / "PUBLICATION.json", label="publication")
        inventory = _load_json(
            destination / "SNAPSHOT_MANIFEST.json", label="snapshot manifest"
        )
        _verify_inventory(destination, inventory)
        for field, expected in publication_identity.items():
            if publication.get(field) != expected:
                raise PromotionError(
                    f"Existing publication field {field!r} does not match the "
                    f"current source identity: {destination}"
                )
        return destination, publication

    staging = Path(tempfile.mkdtemp(prefix=".publish-", dir=parent))
    try:
        # Copy only the compact, fixed scientific publication bundle.  The
        # campaign retains candidate_runs and other bulky replay evidence.
        copied_source_files = _copy_publication_bundle(
            Path(source["validate_dir"]),
            staging,
            expected_inventory=source["publication_bundle_files"],
        )
        publication = {
            **publication_identity,
            "published_at": _utc_now(),
            "transaction_id": transaction_id,
            "source_bundle_files": copied_source_files,
        }
        _atomic_json(staging / "PUBLICATION.json", publication)
        inventory = _file_inventory(staging)
        _atomic_json(staging / "SNAPSHOT_MANIFEST.json", {
            "schema": SNAPSHOT_MANIFEST_SCHEMA,
            "created_at": _utc_now(),
            "files": inventory,
        })
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return destination, publication


def promote_validation(
    *,
    project: Project,
    run_id: str,
    canonical_root: str | os.PathLike[str],
    allow_best_effort: bool = False,
    replace_legacy: bool = False,
    force_downgrade: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Publish one managed validation and, when allowed, make it current best."""

    if (
        not run_id
        or run_id in {".", ".."}
        or Path(run_id).name != run_id
        or _SAFE_TOKEN.search(run_id)
    ):
        raise PromotionError(
            "run_id must be one non-empty ASCII path component containing only "
            "letters, digits, dot, underscore, or hyphen"
        )
    source = _source_validation(project, run_id)
    root = Path(canonical_root).expanduser().resolve()
    if root.name != project.name:
        raise PromotionError(
            f"canonical_root must end with the exact project name {project.name!r}: {root}"
        )
    if root == Path(root.anchor) or root == Path.home().resolve():
        raise PromotionError(f"canonical_root is too broad: {root}")
    if _is_relative_to(root, source["validate_dir"]):
        raise PromotionError("canonical_root cannot be inside the source validation directory")
    if root.exists():
        if root.is_symlink() or not root.is_dir():
            raise PromotionError(f"canonical_root must be a real directory: {root}")
        _validate_managed_directories(root)
        current, latest, _legacy_a11 = _read_canonical_pointers(
            root, replace_legacy=replace_legacy
        )
    else:
        current, latest, _legacy_a11 = None, None, False
    legacy = _legacy_files(
        root,
        managed_latest=latest is not None,
        managed_current=current is not None,
    )
    _validate_legacy_symlinks(root, legacy, recognized_a11=_legacy_a11)
    if legacy and not replace_legacy:
        raise PromotionError(
            "Legacy canonical artifacts exist; use --replace-legacy to archive them: "
            + ", ".join(path.name for path in legacy)
        )

    candidate_view = {
        "parameter_key": source["parameter_key"],
        "quality": source["quality"],
        "protocol": source["protocol"],
        "applicability": source["applicability"],
    }

    def promotion_eligibility() -> tuple[bool, str]:
        quality = source["quality"]
        if not quality["hard_gate_pass"] or quality["status"] == "rejected":
            return False, "scientific_hard_gate_rejected"
        if quality["status"] == "best_effort" and not allow_best_effort:
            return False, "best_effort_not_explicitly_allowed"
        return True, "eligible_for_current_best_comparison"

    eligible, eligibility_reason = promotion_eligibility()

    def decide(current_publication: Mapping[str, Any] | None) -> tuple[str, str, bool]:
        if not eligible:
            return "ineligible", eligibility_reason, False
        comparison, comparison_reason = _comparison(candidate_view, current_publication)
        change = comparison == "promote" or (
            comparison in {"downgrade", "incomparable"} and force_downgrade
        )
        return comparison, comparison_reason, change

    decision, reason, update_current = decide(current)
    legacy_has_current_baseline = any(
        path.name in {"current", "current_best.json", "final_parameters.json"}
        for path in legacy
    )
    takeover_blocked = legacy_has_current_baseline and not update_current
    planned = {
        "status": "blocked" if dry_run and takeover_blocked else "dry_run" if dry_run else "pending",
        "project": project.name,
        "run_id": run_id,
        "source_validate_dir": str(source["validate_dir"]),
        "canonical_root": str(root),
        "parameter_key": source["parameter_key"],
        "quality": source["quality"],
        "protocol": source["protocol"],
        "applicability": source["applicability"],
        "comparison": decision,
        "comparison_reason": reason,
        "current_best_eligible": eligible,
        "current_best_eligibility_reason": eligibility_reason,
        "current_best_will_change": update_current,
        "legacy_to_archive": [path.name for path in legacy],
        "legacy_a11_oneoff": _legacy_a11,
        "legacy_current_baseline": legacy_has_current_baseline,
        "blocked_reason": (
            "legacy_current_baseline_would_be_removed_without_replacement"
            if takeover_blocked else None
        ),
    }
    if dry_run:
        return planned
    if takeover_blocked:
        raise PromotionError(
            "Legacy current baseline cannot be replaced by an ineligible, "
            "unapproved best-effort, worse, or incomparable attempt"
        )

    transaction_id = f"{_timestamp()}-{uuid.uuid4().hex[:8]}"
    journal_path = root / "promotion_journal" / f"{transaction_id}.json"
    with _PromotionLock(root):
        _validate_managed_directories(root)
        journal_dir = _managed_directory(root, "promotion_journal", create=True)
        journal_path = journal_dir / f"{transaction_id}.json"
        # Re-read mutable canonical state while holding the lock.
        current, latest, _legacy_a11 = _read_canonical_pointers(
            root, replace_legacy=replace_legacy
        )
        legacy = _legacy_files(
            root,
            managed_latest=latest is not None,
            managed_current=current is not None,
        )
        _validate_legacy_symlinks(root, legacy, recognized_a11=_legacy_a11)
        if legacy and not replace_legacy:
            raise PromotionError("Legacy canonical artifacts appeared before lock acquisition")
        decision, reason, update_current = decide(current)
        legacy_has_current_baseline = any(
            path.name in {"current", "current_best.json", "final_parameters.json"}
            for path in legacy
        )
        if legacy_has_current_baseline and not update_current:
            raise PromotionError(
                "Legacy current baseline cannot be removed without an eligible replacement"
            )
        journal = {
            "schema": "ffopt-promotion-transaction-v1",
            "transaction_id": transaction_id,
            "started_at": _utc_now(),
            "phase": "started",
            "project": project.name,
            "run_id": run_id,
            "source_manifest_fingerprint": source["manifest_fingerprint"],
            "comparison": decision,
            "comparison_reason": reason,
            "force_downgrade": force_downgrade,
            "legacy_a11_oneoff": _legacy_a11,
            "previous_current_publication_id": (
                current.get("publication_id") if current else None
            ),
        }
        _atomic_json(journal_path, journal)
        snapshot, publication = _create_snapshot(
            root, source, project, run_id, transaction_id
        )
        relative_snapshot = snapshot.relative_to(root).as_posix()
        journal.update({"phase": "snapshot_published", "snapshot": relative_snapshot})
        _atomic_json(journal_path, journal)

        archive = _archive_legacy(root, legacy, transaction_id)
        journal.update({
            "phase": "legacy_archived",
            "legacy_archive": str(archive) if archive else None,
        })
        _atomic_json(journal_path, journal)

        # Remove only byte-identical loose legacy caches after a durable
        # archive exists, before writing the new managed compatibility caches.
        if archive is not None:
            archive_manifest = _load_json(
                archive / "ARCHIVE_MANIFEST.json", label="legacy archive manifest"
            )
            archived_entries = archive_manifest.get("entries", {})
            for path in legacy:
                if _legacy_identity(path) != archived_entries.get(path.name):
                    raise PromotionError(
                        f"Legacy artifact changed during promotion; retained in place: {path}"
                    )
            for path in legacy:
                path.unlink()

        # latest_validation is the compatibility cache for the latest valid
        # attempt, independently of whether it becomes current best.  The
        # pointer retains its immutable snapshot provenance.
        _atomic_copy(
            snapshot / "validation_summary.json", root / "latest_validation.json"
        )
        _atomic_json(
            root / "latest_attempt.json", _pointer(publication, relative_snapshot)
        )
        journal["phase"] = "latest_attempt_updated"
        _atomic_json(journal_path, journal)

        if update_current:
            _atomic_copy(
                snapshot / "final_parameters.json", root / "final_parameters.json"
            )
            _atomic_json(
                root / "current_best.json", _pointer(publication, relative_snapshot)
            )
        elif current is not None:
            # Repair/preserve the compatibility cache from the authoritative
            # current-best snapshot; a worse latest attempt must never replace it.
            current_snapshot = Path(str(current["_snapshot"]))
            _atomic_copy(
                current_snapshot / "final_parameters.json",
                root / "final_parameters.json",
            )
        journal.update({
            "phase": "current_best_updated" if update_current else "current_best_retained",
            "current_best_updated": update_current,
        })
        _atomic_json(journal_path, journal)

        journal.update({"phase": "completed", "finished_at": _utc_now()})
        _atomic_json(journal_path, journal)

    return {
        **planned,
        "status": "promoted" if update_current else "published_not_current",
        "publication_id": publication["publication_id"],
        "snapshot": str(snapshot),
        "legacy_archive": str(archive) if archive else None,
        "transaction_journal": str(journal_path),
        "current_best_changed": update_current,
        "current_best_will_change": update_current,
        "comparison": decision,
        "comparison_reason": reason,
    }


__all__ = ["PromotionError", "promote_validation"]
