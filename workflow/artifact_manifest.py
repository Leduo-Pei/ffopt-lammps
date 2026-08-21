"""Content-addressed manifests for resumable FFOpt work units.

This module deliberately does not know about pipeline stages, schedulers, or a
particular material.  A caller supplies a scientific identity (parameters,
seeds, configuration, and input files) plus the files produced by a completed
work unit.  The resulting JSON manifest can then be used to decide whether a
candidate or stage is safe to reuse.

Manifests are deterministic, immutable once written, and fail closed: malformed
JSON, unknown fields, changed inputs, or changed outputs all make a work unit
non-reusable.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
import tempfile
from typing import Any, Iterable, Literal, Mapping, Sequence


MANIFEST_SCHEMA_VERSION = 1
_MANIFEST_KINDS = frozenset({"candidate", "stage"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_PARAMETER_KEY_RE = re.compile(r"^(named|sequence):sha256:[0-9a-f]{64}$")
_TOP_LEVEL_FIELDS = frozenset({
    "schema_version",
    "kind",
    "identifier",
    "parameter_key",
    "seeds",
    "scientific_config_sha256",
    "input_artifacts",
    "expected_outputs",
    "execution_fingerprint",
    "manifest_fingerprint",
})

ParameterValues = Mapping[str, Real] | Sequence[Real]
ArtifactPaths = Mapping[str, str | os.PathLike[str]]


class ArtifactManifestError(ValueError):
    """Raised when a manifest cannot be constructed, parsed, or written safely."""


@dataclass(frozen=True)
class ArtifactHash:
    """Cryptographic identity of one regular file."""

    sha256: str
    size_bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True)
class ArtifactManifest:
    """Immutable in-memory representation of a completed work unit."""

    schema_version: int
    kind: Literal["candidate", "stage"]
    identifier: str
    parameter_key: str | None
    seeds: tuple[int, ...]
    scientific_config_sha256: str
    input_artifacts: tuple[tuple[str, ArtifactHash], ...]
    expected_outputs: tuple[tuple[str, ArtifactHash], ...]
    execution_fingerprint: str
    manifest_fingerprint: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "kind": self.kind,
            "identifier": self.identifier,
            "parameter_key": self.parameter_key,
            "seeds": list(self.seeds),
            "scientific_config_sha256": self.scientific_config_sha256,
            "input_artifacts": {
                label: digest.to_dict() for label, digest in self.input_artifacts
            },
            "expected_outputs": {
                label: digest.to_dict() for label, digest in self.expected_outputs
            },
            "execution_fingerprint": self.execution_fingerprint,
            "manifest_fingerprint": self.manifest_fingerprint,
        }


@dataclass(frozen=True)
class ManifestIssue:
    """One reason why a manifest or its artifacts cannot be reused."""

    code: str
    message: str


@dataclass(frozen=True)
class ManifestValidation:
    """Strict validation result; any issue means fail closed."""

    valid: bool
    issues: tuple[ManifestIssue, ...]
    manifest: ArtifactManifest | None = None

    @property
    def reason(self) -> str:
        if not self.issues:
            return "manifest and artifacts verified"
        return "; ".join(f"{item.code}: {item.message}" for item in self.issues)


@dataclass(frozen=True)
class ReuseDecision:
    """Decision returned before reusing a candidate or stage."""

    reusable: bool
    issues: tuple[ManifestIssue, ...]
    manifest: ArtifactManifest | None = None

    @property
    def reason(self) -> str:
        if not self.issues:
            return "scientific identity and output hashes verified"
        return "; ".join(f"{item.code}: {item.message}" for item in self.issues)


def _float_token(value: Real, *, location: str) -> str:
    number = float(value)
    if not math.isfinite(number):
        raise ArtifactManifestError(f"{location} must be finite, got {number!r}")
    # A signed zero is not a distinct force-field parameter.
    if number == 0.0:
        number = 0.0
    return number.hex()


def _canonical_bytes(value: Any, *, location: str = "value") -> bytes:
    """Encode supported data without relying on JSON float formatting.

    The encoding is internal and length-prefixed, so user strings cannot be
    confused with type markers.  Lists and tuples intentionally have the same
    identity because configuration loaders may choose either representation.
    """

    if value is None:
        return b"n"
    if isinstance(value, bool):
        return b"b1" if value else b"b0"
    if isinstance(value, Integral):
        payload = str(int(value)).encode("ascii")
        return b"i" + str(len(payload)).encode("ascii") + b":" + payload
    if isinstance(value, Real):
        payload = _float_token(value, location=location).encode("ascii")
        return b"f" + str(len(payload)).encode("ascii") + b":" + payload
    if isinstance(value, os.PathLike):
        value = os.fspath(value)
    if isinstance(value, str):
        payload = value.encode("utf-8")
        return b"s" + str(len(payload)).encode("ascii") + b":" + payload
    if isinstance(value, Mapping):
        items: list[tuple[str, Any]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise ArtifactManifestError(
                    f"{location} mapping keys must be strings, got {type(key).__name__}"
                )
            items.append((key, item))
        items.sort(key=lambda pair: pair[0])
        result = [b"m", str(len(items)).encode("ascii"), b":"]
        for key, item in items:
            result.append(_canonical_bytes(key, location=f"{location} key"))
            result.append(_canonical_bytes(item, location=f"{location}.{key}"))
        return b"".join(result)
    if isinstance(value, (list, tuple)):
        result = [b"l", str(len(value)).encode("ascii"), b":"]
        for index, item in enumerate(value):
            result.append(_canonical_bytes(item, location=f"{location}[{index}]"))
        return b"".join(result)
    raise ArtifactManifestError(
        f"{location} has unsupported type {type(value).__name__}; "
        "use strings, finite numbers, booleans, null, lists, or string-keyed mappings"
    )


def _sha256_value(value: Any) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def canonical_parameter_key(parameters: ParameterValues) -> str:
    """Return a stable key for ordered or name-addressed parameter values.

    Named parameters are independent of mapping insertion order.  Ordered
    parameters preserve sequence order.  Every value is encoded using the exact
    finite IEEE-754 representation used by Python, avoiding locale and display
    precision changes.
    """

    if isinstance(parameters, Mapping):
        canonical: list[tuple[str, str]] = []
        for name, value in parameters.items():
            if not isinstance(name, str) or not name:
                raise ArtifactManifestError("parameter names must be non-empty strings")
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ArtifactManifestError(
                    f"parameter {name!r} must be a real number, got {type(value).__name__}"
                )
            canonical.append((name, _float_token(value, location=f"parameter {name!r}")))
        if not canonical:
            raise ArtifactManifestError("at least one named parameter is required")
        canonical.sort(key=lambda pair: pair[0])
        return f"named:sha256:{_sha256_value(canonical)}"

    if isinstance(parameters, (str, bytes)) or not isinstance(parameters, Sequence):
        raise ArtifactManifestError("parameters must be a mapping or an ordered sequence")
    canonical_sequence: list[str] = []
    for index, value in enumerate(parameters):
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ArtifactManifestError(
                f"parameter {index} must be a real number, got {type(value).__name__}"
            )
        canonical_sequence.append(_float_token(value, location=f"parameter {index}"))
    if not canonical_sequence:
        raise ArtifactManifestError("at least one ordered parameter is required")
    return f"sequence:sha256:{_sha256_value(canonical_sequence)}"


def scientific_config_hash(config: Any) -> str:
    """Hash a scientific configuration using deterministic typed encoding."""

    return _sha256_value(config)


def sha256_file(path: str | os.PathLike[str], *, chunk_size: int = 1024 * 1024) -> ArtifactHash:
    """Hash one regular file and record its size."""

    candidate = Path(path)
    if not candidate.exists():
        raise ArtifactManifestError(f"artifact does not exist: {candidate}")
    if not candidate.is_file():
        raise ArtifactManifestError(f"artifact is not a regular file: {candidate}")
    digest = hashlib.sha256()
    size = 0
    try:
        with candidate.open("rb") as handle:
            while True:
                chunk = handle.read(chunk_size)
                if not chunk:
                    break
                size += len(chunk)
                digest.update(chunk)
    except OSError as exc:
        raise ArtifactManifestError(f"cannot read artifact {candidate}: {exc}") from exc
    return ArtifactHash(sha256=digest.hexdigest(), size_bytes=size)


def _normalize_seeds(seeds: Iterable[int]) -> tuple[int, ...]:
    try:
        raw = list(seeds)
    except TypeError as exc:
        raise ArtifactManifestError("seeds must be an iterable of integers") from exc
    normalized: set[int] = set()
    for seed in raw:
        if isinstance(seed, bool) or not isinstance(seed, Integral):
            raise ArtifactManifestError(
                f"seeds must contain only integers, got {type(seed).__name__}"
            )
        normalized.add(int(seed))
    return tuple(sorted(normalized))


def _normalize_artifact_paths(paths: ArtifactPaths, *, role: str) -> dict[str, Path]:
    if not isinstance(paths, Mapping):
        raise ArtifactManifestError(f"{role} artifacts must be a label-to-path mapping")
    normalized: dict[str, Path] = {}
    for label, path in paths.items():
        if not isinstance(label, str) or not label:
            raise ArtifactManifestError(f"{role} artifact labels must be non-empty strings")
        try:
            normalized[label] = Path(path)
        except TypeError as exc:
            raise ArtifactManifestError(
                f"{role} artifact {label!r} has an invalid path: {path!r}"
            ) from exc
    return normalized


def _hash_artifacts(paths: ArtifactPaths, *, role: str) -> tuple[tuple[str, ArtifactHash], ...]:
    normalized = _normalize_artifact_paths(paths, role=role)
    return tuple(
        (label, sha256_file(path)) for label, path in sorted(normalized.items())
    )


def _artifact_dict(items: tuple[tuple[str, ArtifactHash], ...]) -> dict[str, dict[str, Any]]:
    return {label: digest.to_dict() for label, digest in items}


def _execution_payload(
    *,
    kind: str,
    identifier: str,
    parameter_key: str | None,
    seeds: tuple[int, ...],
    scientific_config_sha256: str,
    input_artifacts: tuple[tuple[str, ArtifactHash], ...],
    output_labels: Iterable[str],
) -> dict[str, Any]:
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "kind": kind,
        "identifier": identifier,
        "parameter_key": parameter_key,
        "seeds": list(seeds),
        "scientific_config_sha256": scientific_config_sha256,
        "input_artifacts": _artifact_dict(input_artifacts),
        "expected_output_labels": sorted(output_labels),
    }


def _manifest_payload_without_fingerprint(manifest: ArtifactManifest) -> dict[str, Any]:
    payload = manifest.to_dict()
    payload.pop("manifest_fingerprint")
    return payload


def build_artifact_manifest(
    *,
    kind: Literal["candidate", "stage"],
    identifier: str,
    parameters: ParameterValues | None,
    seeds: Iterable[int],
    scientific_config: Any,
    input_artifacts: ArtifactPaths,
    expected_outputs: ArtifactPaths,
) -> ArtifactManifest:
    """Build a deterministic manifest after all expected outputs exist."""

    if kind not in _MANIFEST_KINDS:
        raise ArtifactManifestError(f"kind must be 'candidate' or 'stage', got {kind!r}")
    if not isinstance(identifier, str) or not identifier.strip():
        raise ArtifactManifestError("identifier must be a non-empty string")
    if kind == "candidate" and parameters is None:
        raise ArtifactManifestError("candidate manifests require parameters")
    parameter_key = None if parameters is None else canonical_parameter_key(parameters)
    normalized_seeds = _normalize_seeds(seeds)
    config_sha256 = scientific_config_hash(scientific_config)
    inputs = _hash_artifacts(input_artifacts, role="input")
    outputs = _hash_artifacts(expected_outputs, role="output")
    if not outputs:
        raise ArtifactManifestError("at least one expected output is required")

    execution_fingerprint = _sha256_value(_execution_payload(
        kind=kind,
        identifier=identifier,
        parameter_key=parameter_key,
        seeds=normalized_seeds,
        scientific_config_sha256=config_sha256,
        input_artifacts=inputs,
        output_labels=(label for label, _digest in outputs),
    ))
    provisional = ArtifactManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        kind=kind,
        identifier=identifier,
        parameter_key=parameter_key,
        seeds=normalized_seeds,
        scientific_config_sha256=config_sha256,
        input_artifacts=inputs,
        expected_outputs=outputs,
        execution_fingerprint=execution_fingerprint,
        manifest_fingerprint="",
    )
    manifest_fingerprint = _sha256_value(_manifest_payload_without_fingerprint(provisional))
    return ArtifactManifest(
        **{
            **provisional.__dict__,
            "manifest_fingerprint": manifest_fingerprint,
        }
    )


def _json_bytes(manifest: ArtifactManifest) -> bytes:
    return (
        json.dumps(
            manifest.to_dict(),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def write_artifact_manifest(
    path: str | os.PathLike[str], manifest: ArtifactManifest
) -> Path:
    """Atomically create an immutable manifest, allowing idempotent rewrites.

    An existing byte-identical manifest is accepted.  Any different existing
    content, including a corrupt partial file, is preserved and raises instead
    of being silently replaced.
    """

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    content = _json_bytes(manifest)
    if destination.exists():
        try:
            existing = destination.read_bytes()
        except OSError as exc:
            raise ArtifactManifestError(f"cannot read existing manifest {destination}: {exc}") from exc
        if existing == content:
            return destination
        raise ArtifactManifestError(f"refusing to overwrite immutable manifest: {destination}")

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            # Linking a fully flushed temporary file gives create-if-absent
            # semantics, unlike os.replace(), which could overwrite a racing
            # writer's immutable manifest.
            os.link(temporary, destination)
        except FileExistsError:
            if destination.read_bytes() != content:
                raise ArtifactManifestError(
                    f"refusing to overwrite immutable manifest: {destination}"
                )
        except OSError:
            # Some filesystems do not permit hard links.  Preserve atomic
            # visibility with replace after one final existence check.
            if destination.exists():
                if destination.read_bytes() != content:
                    raise ArtifactManifestError(
                        f"refusing to overwrite immutable manifest: {destination}"
                    )
            else:
                os.replace(temporary, destination)
        return destination
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass


def _unique_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ArtifactManifestError(f"duplicate JSON field {key!r}")
        result[key] = value
    return result


def _require_sha256(value: Any, *, field: str) -> str:
    if not isinstance(value, str) or not _SHA256_RE.fullmatch(value):
        raise ArtifactManifestError(f"{field} must be a lowercase SHA-256 digest")
    return value


def _parse_artifact_hashes(value: Any, *, field: str) -> tuple[tuple[str, ArtifactHash], ...]:
    if not isinstance(value, dict):
        raise ArtifactManifestError(f"{field} must be an object")
    result: list[tuple[str, ArtifactHash]] = []
    for label, raw in value.items():
        if not isinstance(label, str) or not label:
            raise ArtifactManifestError(f"{field} labels must be non-empty strings")
        if not isinstance(raw, dict) or set(raw) != {"sha256", "size_bytes"}:
            raise ArtifactManifestError(
                f"{field}.{label} must contain exactly sha256 and size_bytes"
            )
        digest = _require_sha256(raw["sha256"], field=f"{field}.{label}.sha256")
        size = raw["size_bytes"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 0:
            raise ArtifactManifestError(f"{field}.{label}.size_bytes must be a non-negative integer")
        result.append((label, ArtifactHash(digest, size)))
    result.sort(key=lambda pair: pair[0])
    return tuple(result)


def _parse_manifest(raw: Any) -> ArtifactManifest:
    if not isinstance(raw, dict):
        raise ArtifactManifestError("manifest root must be a JSON object")
    fields = set(raw)
    missing = sorted(_TOP_LEVEL_FIELDS - fields)
    extra = sorted(fields - _TOP_LEVEL_FIELDS)
    if missing or extra:
        details = []
        if missing:
            details.append(f"missing fields {missing}")
        if extra:
            details.append(f"unknown fields {extra}")
        raise ArtifactManifestError("manifest schema mismatch: " + ", ".join(details))
    schema_version = raw["schema_version"]
    if (
        isinstance(schema_version, bool)
        or not isinstance(schema_version, int)
        or schema_version != MANIFEST_SCHEMA_VERSION
    ):
        raise ArtifactManifestError(
            f"unsupported manifest schema {schema_version!r}; "
            f"expected {MANIFEST_SCHEMA_VERSION}"
        )
    kind = raw["kind"]
    if not isinstance(kind, str) or kind not in _MANIFEST_KINDS:
        raise ArtifactManifestError(f"invalid manifest kind {kind!r}")
    identifier = raw["identifier"]
    if not isinstance(identifier, str) or not identifier.strip():
        raise ArtifactManifestError("identifier must be a non-empty string")
    parameter_key = raw["parameter_key"]
    if parameter_key is not None and (
        not isinstance(parameter_key, str) or not _PARAMETER_KEY_RE.fullmatch(parameter_key)
    ):
        raise ArtifactManifestError("parameter_key has an invalid canonical-key format")
    if kind == "candidate" and parameter_key is None:
        raise ArtifactManifestError("candidate manifests require parameter_key")
    seeds_raw = raw["seeds"]
    if not isinstance(seeds_raw, list):
        raise ArtifactManifestError("seeds must be a JSON array")
    seeds = _normalize_seeds(seeds_raw)
    if list(seeds) != seeds_raw:
        raise ArtifactManifestError("seeds must be sorted and unique")
    config_sha256 = _require_sha256(
        raw["scientific_config_sha256"], field="scientific_config_sha256"
    )
    inputs = _parse_artifact_hashes(raw["input_artifacts"], field="input_artifacts")
    outputs = _parse_artifact_hashes(raw["expected_outputs"], field="expected_outputs")
    if not outputs:
        raise ArtifactManifestError("expected_outputs must not be empty")
    execution_fingerprint = _require_sha256(
        raw["execution_fingerprint"], field="execution_fingerprint"
    )
    manifest_fingerprint = _require_sha256(
        raw["manifest_fingerprint"], field="manifest_fingerprint"
    )
    manifest = ArtifactManifest(
        schema_version=MANIFEST_SCHEMA_VERSION,
        kind=kind,
        identifier=identifier,
        parameter_key=parameter_key,
        seeds=seeds,
        scientific_config_sha256=config_sha256,
        input_artifacts=inputs,
        expected_outputs=outputs,
        execution_fingerprint=execution_fingerprint,
        manifest_fingerprint=manifest_fingerprint,
    )
    expected_execution = _sha256_value(_execution_payload(
        kind=kind,
        identifier=identifier,
        parameter_key=parameter_key,
        seeds=seeds,
        scientific_config_sha256=config_sha256,
        input_artifacts=inputs,
        output_labels=(label for label, _digest in outputs),
    ))
    if execution_fingerprint != expected_execution:
        raise ArtifactManifestError("execution_fingerprint does not match manifest content")
    expected_manifest = _sha256_value(_manifest_payload_without_fingerprint(manifest))
    if manifest_fingerprint != expected_manifest:
        raise ArtifactManifestError("manifest_fingerprint does not match manifest content")
    return manifest


def load_artifact_manifest(path: str | os.PathLike[str]) -> ArtifactManifest:
    """Load and intrinsically validate a manifest."""

    source = Path(path)
    try:
        text = source.read_text(encoding="ascii")
    except (OSError, UnicodeError) as exc:
        raise ArtifactManifestError(f"cannot read manifest {source}: {exc}") from exc
    try:
        raw = json.loads(text, object_pairs_hook=_unique_json_object)
    except ArtifactManifestError:
        raise
    except (json.JSONDecodeError, ValueError) as exc:
        raise ArtifactManifestError(f"invalid JSON manifest {source}: {exc}") from exc
    return _parse_manifest(raw)


def _verify_artifacts(
    recorded: tuple[tuple[str, ArtifactHash], ...],
    actual_paths: ArtifactPaths,
    *,
    role: str,
) -> list[ManifestIssue]:
    issues: list[ManifestIssue] = []
    try:
        normalized = _normalize_artifact_paths(actual_paths, role=role)
    except ArtifactManifestError as exc:
        return [ManifestIssue(f"{role}_mapping_invalid", str(exc))]
    recorded_map = dict(recorded)
    if set(normalized) != set(recorded_map):
        missing_labels = sorted(set(recorded_map) - set(normalized))
        extra_labels = sorted(set(normalized) - set(recorded_map))
        if missing_labels:
            issues.append(ManifestIssue(
                f"{role}_labels_missing", f"paths not supplied for {missing_labels}"
            ))
        if extra_labels:
            issues.append(ManifestIssue(
                f"{role}_labels_unexpected", f"unrecorded labels supplied: {extra_labels}"
            ))
    for label in sorted(set(normalized) & set(recorded_map)):
        try:
            actual = sha256_file(normalized[label])
        except ArtifactManifestError as exc:
            issues.append(ManifestIssue(f"{role}_unavailable", f"{label}: {exc}"))
            continue
        expected = recorded_map[label]
        if actual.size_bytes != expected.size_bytes:
            issues.append(ManifestIssue(
                f"{role}_size_mismatch",
                f"{label}: expected {expected.size_bytes} bytes, found {actual.size_bytes}",
            ))
        elif actual.sha256 != expected.sha256:
            issues.append(ManifestIssue(
                f"{role}_hash_mismatch",
                f"{label}: expected {expected.sha256}, found {actual.sha256}",
            ))
    return issues


def validate_artifact_manifest(
    path: str | os.PathLike[str],
    *,
    input_artifacts: ArtifactPaths,
    expected_outputs: ArtifactPaths,
) -> ManifestValidation:
    """Validate schema, fingerprints, and all referenced file contents."""

    try:
        manifest = load_artifact_manifest(path)
    except ArtifactManifestError as exc:
        issue = ManifestIssue("manifest_invalid", str(exc))
        return ManifestValidation(False, (issue,), None)
    issues = [
        *_verify_artifacts(manifest.input_artifacts, input_artifacts, role="input"),
        *_verify_artifacts(manifest.expected_outputs, expected_outputs, role="output"),
    ]
    return ManifestValidation(not issues, tuple(issues), manifest)


def check_artifact_reuse(
    path: str | os.PathLike[str],
    *,
    kind: Literal["candidate", "stage"],
    identifier: str,
    parameters: ParameterValues | None,
    seeds: Iterable[int],
    scientific_config: Any,
    input_artifacts: ArtifactPaths,
    expected_outputs: ArtifactPaths,
) -> ReuseDecision:
    """Return whether a work unit exactly matches current scientific intent.

    This function never treats a parsing or hashing failure as reusable.  It
    reports all identity mismatches it can establish so callers can explain why
    a candidate will be recomputed.
    """

    try:
        manifest = load_artifact_manifest(path)
    except ArtifactManifestError as exc:
        issue = ManifestIssue("manifest_invalid", str(exc))
        return ReuseDecision(False, (issue,), None)

    issues: list[ManifestIssue] = []
    if manifest.kind != kind:
        issues.append(ManifestIssue("kind_mismatch", f"expected {kind}, found {manifest.kind}"))
    if manifest.identifier != identifier:
        issues.append(ManifestIssue(
            "identifier_mismatch", f"expected {identifier!r}, found {manifest.identifier!r}"
        ))
    try:
        parameter_key = None if parameters is None else canonical_parameter_key(parameters)
    except ArtifactManifestError as exc:
        issues.append(ManifestIssue("parameters_invalid", str(exc)))
        parameter_key = None
    if manifest.parameter_key != parameter_key:
        issues.append(ManifestIssue(
            "parameter_key_mismatch",
            f"expected {parameter_key!r}, found {manifest.parameter_key!r}",
        ))
    try:
        normalized_seeds = _normalize_seeds(seeds)
    except ArtifactManifestError as exc:
        issues.append(ManifestIssue("seeds_invalid", str(exc)))
        normalized_seeds = ()
    if manifest.seeds != normalized_seeds:
        issues.append(ManifestIssue(
            "seeds_mismatch", f"expected {list(normalized_seeds)}, found {list(manifest.seeds)}"
        ))
    try:
        config_sha256 = scientific_config_hash(scientific_config)
    except ArtifactManifestError as exc:
        issues.append(ManifestIssue("scientific_config_invalid", str(exc)))
        config_sha256 = ""
    if manifest.scientific_config_sha256 != config_sha256:
        issues.append(ManifestIssue(
            "scientific_config_mismatch",
            f"expected {config_sha256}, found {manifest.scientific_config_sha256}",
        ))

    issues.extend(_verify_artifacts(manifest.input_artifacts, input_artifacts, role="input"))
    issues.extend(_verify_artifacts(manifest.expected_outputs, expected_outputs, role="output"))
    return ReuseDecision(not issues, tuple(issues), manifest)
