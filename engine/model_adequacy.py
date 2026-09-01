"""Read-only diagnostics for force-field model form and applicability.

The optimizer answers a computational question: *can this parameterization
reproduce the requested observations?*  That is not the same as the physical
question: *will the fitted model transfer to configurations that were not in
the fit?*  This module keeps those conclusions separate.

``assess_model_adequacy`` is deliberately a pure function.  It does not read a
LAMMPS input, launch a calculation, reject a candidate, or mutate the compiled
configuration. Missing model-form information is ``"unknown"``; a declared
transferability experiment that has not run is ``"not_evaluated"``. In
particular, an elemental material represented by several permanent LAMMPS atom
types is flagged as an ordered-sublattice surrogate, but that warning is not a
proof that a fit is impossible or that its ordered-sublattice bulk validation
failed.

Optional elasticity input may be either a direct zero-kelvin mapping::

    {"C12": 138.1, "C44": 121.9}

or a mapping with ``zero_k`` and/or ``finite_temperature`` records.  A record
may place the constants directly in the record or below
``elastic_constants_gpa``.  Finite-temperature Cauchy pressure is always
labelled ``diagnostic_only`` because thermal, stress-fluctuation, and kinetic
contributions invalidate a literal zero-kelvin central-force interpretation.

The legacy optional label-swap result accepts ``passed`` (a boolean), or numerical
``reference_value``/``swapped_value`` values and an optional
``relative_tolerance``.  Numerical inputs are required to be finite; the
returned report contains no NaN or infinity and is JSON-compatible. A positive
``elemental_transferability_claim`` additionally requires a future trusted
runner to read, schema-check, and hash the referenced artifacts. This module
can structurally validate a provenance *attestation*, but it cannot verify file
content; therefore attested ``pass`` records never establish transferability.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
import re
from typing import Any


REPORT_SCHEMA_VERSION = 2
TRANSFERABILITY_EVIDENCE_SCHEMA_VERSION = 1
_PARAMETERS = ("epsilon", "sigma", "charge")
_ZERO_K_KEYS = ("zero_k", "zero_kelvin", "0k", "0_K")
_FINITE_T_KEYS = ("finite_temperature", "finite_t", "300k", "300_K")
_EPS = 1.0e-15
_RELABEL_RTOL = 1.0e-12
_RELABEL_ATOL = 1.0e-12
_TRANSFERABILITY_TESTS = (
    "label_swap",
    "surface_registry",
    "vacancy",
    "migration_path_sensitivity",
)
_TRANSFERABILITY_MINIMUM_CASES = {
    "label_swap": 3,
    "surface_registry": 2,
    "vacancy": 2,
    "migration_path_sensitivity": 2,
}
_SUPPORTED_FORMAL_LJ_STYLES = {"lj/cut"}
_SUPPORTED_PROVENANCE_PRODUCERS = {"ffopt-transfer-audit-v1"}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_TRANSFERABILITY_STATUSES = {
    "pass",
    "fail",
    "error",
    "not_evaluated",
    "not_applicable",
}
_TRANSFERABILITY_ALIASES = {
    "surface_terminations": "surface_registry",
    "diffusion": "migration_path_sensitivity",
}


class ModelAdequacyError(ValueError):
    """Raised when supplied diagnostic evidence is malformed or non-finite."""


def _mapping(value: Any, location: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ModelAdequacyError(f"{location} must be a mapping")
    return value


def _optional_mapping(value: Any, location: str) -> Mapping[str, Any]:
    if value is None:
        return {}
    return _mapping(value, location)


def _finite(value: Any, location: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ModelAdequacyError(f"{location} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ModelAdequacyError(f"{location} must be a finite number")
    return number


def _optional_finite(record: Mapping[str, Any], keys: Sequence[str], location: str) -> float | None:
    for key in keys:
        if key in record:
            return _finite(record[key], f"{location}.{key}")
    return None


def _clean_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    result = value.strip()
    return result or None


def _atom_types(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    raw = config.get("atom_types")
    if raw is None:
        return []
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        raise ModelAdequacyError("config.atom_types must be a sequence")
    result: list[Mapping[str, Any]] = []
    for index, item in enumerate(raw):
        result.append(_mapping(item, f"config.atom_types[{index}]"))
    return result


def _type_label(item: Mapping[str, Any], index: int) -> str:
    label = _clean_text(item.get("label"))
    if label is not None:
        return label
    type_id = item.get("type")
    if isinstance(type_id, int) and not isinstance(type_id, bool):
        return f"type_{type_id}"
    return f"type_{index + 1}"


def _parameter_value(item: Mapping[str, Any], parameter: str, location: str) -> dict[str, Any]:
    raw_params = item.get("params")
    if raw_params is None:
        return {"mode": "unknown", "initial": None}
    params = _mapping(raw_params, f"{location}.params")
    if parameter not in params:
        return {"mode": "unknown", "initial": None}
    value = params[parameter]
    if isinstance(value, Mapping):
        initial = _optional_finite(value, ("init",), f"{location}.params.{parameter}")
        low = _optional_finite(value, ("min",), f"{location}.params.{parameter}")
        high = _optional_finite(value, ("max",), f"{location}.params.{parameter}")
        if low is not None and high is not None and low > high:
            raise ModelAdequacyError(
                f"{location}.params.{parameter} has min greater than max"
            )
        if initial is not None and low is not None and initial < low:
            raise ModelAdequacyError(
                f"{location}.params.{parameter} init lies below min"
            )
        if initial is not None and high is not None and initial > high:
            raise ModelAdequacyError(
                f"{location}.params.{parameter} init lies above max"
            )
        return {
            "mode": "variable",
            "initial": initial,
            "minimum": low,
            "maximum": high,
        }
    return {"mode": "fixed", "initial": _finite(value, f"{location}.params.{parameter}")}


def _constraints(config: Mapping[str, Any]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    raw = config.get("parameter_constraints")
    if raw is None:
        return [], []
    constraints = _mapping(raw, "config.parameter_constraints")

    def records(name: str) -> list[Mapping[str, Any]]:
        value = constraints.get(name, [])
        if isinstance(value, (str, bytes, bytearray)) or not isinstance(value, Sequence):
            raise ModelAdequacyError(
                f"config.parameter_constraints.{name} must be a sequence"
            )
        return [
            _mapping(item, f"config.parameter_constraints.{name}[{index}]")
            for index, item in enumerate(value)
        ]

    return records("ties"), records("differences")


def _labels_from_rule(rule: Mapping[str, Any]) -> list[str]:
    raw = rule.get("types")
    if isinstance(raw, (str, bytes, bytearray)) or not isinstance(raw, Sequence):
        return []
    return [value for value in (_clean_text(item) for item in raw) if value is not None]


def _parameter_symmetry(
    config: Mapping[str, Any],
    atom_types: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    labels = [_type_label(item, index) for index, item in enumerate(atom_types)]
    ties, differences = _constraints(config)
    rows: list[dict[str, Any]] = []

    for parameter in _PARAMETERS:
        values = [
            _parameter_value(item, parameter, f"config.atom_types[{index}]")
            for index, item in enumerate(atom_types)
        ]
        tied_rules = [rule for rule in ties if rule.get("parameter") == parameter]
        difference_rules = [
            rule for rule in differences if rule.get("parameter") == parameter
        ]

        tie_labels = sorted({label for rule in tied_rules for label in _labels_from_rule(rule)})
        limits: list[dict[str, Any]] = []
        for index, rule in enumerate(difference_rules):
            maximum = _optional_finite(
                rule,
                ("max_abs",),
                f"config.parameter_constraints.differences[{index}]",
            )
            if maximum is not None and maximum < 0.0:
                raise ModelAdequacyError(
                    f"difference limit for {parameter} must be non-negative"
                )
            limits.append(
                {
                    "types": _labels_from_rule(rule),
                    "max_abs": maximum,
                    "unit": _clean_text(rule.get("unit")),
                }
            )

        if len(atom_types) <= 1:
            relation = "single_type"
            can_differ: bool | None = False
        elif tied_rules:
            relation = "tied"
            can_differ = False
        elif difference_rules:
            relation = "difference_limited"
            can_differ = True
        elif any(value["mode"] == "unknown" for value in values):
            relation = "unknown"
            can_differ = None
        elif all(value["mode"] == "fixed" for value in values):
            initials = [value["initial"] for value in values]
            if all(math.isclose(initials[0], item, rel_tol=0.0, abs_tol=1.0e-15) for item in initials[1:]):
                relation = "fixed_equal"
                can_differ = False
            else:
                relation = "fixed_different"
                can_differ = True
        else:
            relation = "independent"
            can_differ = True

        rows.append(
            {
                "parameter": parameter,
                "relation": relation,
                "can_differ_between_same_element_types": can_differ,
                "type_values": [
                    {"type": label, **value} for label, value in zip(labels, values)
                ],
                "tied_types": tie_labels,
                "difference_limits": limits,
            }
        )

    return {
        "same_element_type_labels": labels,
        "parameters": rows,
        "parameters_that_can_differ": [
            row["parameter"]
            for row in rows
            if row["can_differ_between_same_element_types"] is True
        ],
        "parameters_with_unknown_relation": [
            row["parameter"]
            for row in rows
            if row["can_differ_between_same_element_types"] is None
        ],
    }


def _pair_model(config: Mapping[str, Any]) -> dict[str, Any]:
    pair_params = _optional_mapping(config.get("pair_params"), "config.pair_params")
    lammps = _optional_mapping(config.get("lammps"), "config.lammps")
    material = _optional_mapping(config.get("material"), "config.material")

    candidates: list[tuple[str, str]] = []
    for source, value in (
        ("pair_params.pair_style", pair_params.get("pair_style")),
        ("lammps.pair_style", lammps.get("pair_style")),
        ("material.force_field", material.get("force_field")),
    ):
        text = _clean_text(value)
        if text is not None:
            candidates.append((source, text.lower().replace("_", "/")))

    unique_styles = sorted({style for _, style in candidates})
    if len(unique_styles) == 1:
        pair_style: str | None = unique_styles[0]
        pair_style_source = ",".join(source for source, _ in candidates)
    else:
        pair_style = None
        pair_style_source = "conflicting_declarations" if unique_styles else "unknown"

    if pair_style is None:
        central_lj: bool | None = None
        family = "unknown"
    elif pair_style == "lj" or pair_style.startswith("lj/"):
        central_lj = True
        family = "central_lj_pair"
    else:
        central_lj = False
        family = "other_or_composite"

    configured_mixing = _clean_text(pair_params.get("mixing_rule"))
    if configured_mixing is not None:
        configured_mixing = configured_mixing.lower()
    resolved_mixing: str | None
    if configured_mixing == "default":
        persisted = _clean_text(config.get("physics_feature_mixing_rule"))
        if persisted is not None:
            resolved_mixing = persisted.lower()
        elif central_lj is True:
            # Mirrors the runtime feature resolver for the supported LJ styles.
            resolved_mixing = "geometric"
        else:
            resolved_mixing = None
    elif configured_mixing in {"geometric", "arithmetic", "sixthpower", "none"}:
        resolved_mixing = configured_mixing
    else:
        resolved_mixing = None

    return {
        "pair_style": pair_style,
        "pair_style_source": pair_style_source,
        "pair_style_declarations": [
            {"source": source, "value": style} for source, style in candidates
        ],
        "family": family,
        "central_lj_pair_style": central_lj,
        "mixing": {
            "configured": configured_mixing,
            "uses_lammps_native_default": configured_mixing == "default",
            "resolved_for_diagnostics": resolved_mixing,
            "resolution_status": "known" if resolved_mixing is not None else "unknown",
        },
    }


def _elastic_constants(record: Mapping[str, Any], location: str) -> tuple[float | None, float | None]:
    nested = record.get("elastic_constants_gpa")
    values = _mapping(nested, f"{location}.elastic_constants_gpa") if nested is not None else record
    c12 = _optional_finite(values, ("C12", "c12", "C12_gpa", "c12_gpa"), location)
    c44 = _optional_finite(values, ("C44", "c44", "C44_gpa", "c44_gpa"), location)
    return c12, c44


def _elastic_record(
    record: Mapping[str, Any] | None,
    *,
    location: str,
    interpretation: str,
    assumed_temperature_k: float | None,
) -> dict[str, Any]:
    if record is None:
        return {
            "status": "unknown",
            "interpretation": interpretation,
            "temperature_k": assumed_temperature_k,
            "C12_gpa": None,
            "C44_gpa": None,
            "cauchy_delta_gpa": None,
            "normalized_cauchy_delta": None,
            "normalization_scale_gpa": None,
        }

    c12, c44 = _elastic_constants(record, location)
    temperature = _optional_finite(record, ("temperature_k", "temperature"), location)
    if temperature is None:
        temperature = assumed_temperature_k
    if temperature is not None and temperature < 0.0:
        raise ModelAdequacyError(f"{location}.temperature_k must be non-negative")
    if c12 is None or c44 is None:
        return {
            "status": "unknown",
            "interpretation": interpretation,
            "temperature_k": temperature,
            "C12_gpa": c12,
            "C44_gpa": c44,
            "cauchy_delta_gpa": None,
            "normalized_cauchy_delta": None,
            "normalization_scale_gpa": None,
        }

    delta = c12 - c44
    scale = max(abs(c12), abs(c44))
    normalized = None if scale <= _EPS else delta / scale
    return {
        "status": "available",
        "interpretation": interpretation,
        "temperature_k": temperature,
        "C12_gpa": c12,
        "C44_gpa": c44,
        "cauchy_delta_gpa": delta,
        "normalized_cauchy_delta": normalized,
        "normalization_scale_gpa": scale,
    }


def _elasticity_diagnostics(results: Mapping[str, Any] | None) -> dict[str, Any]:
    if results is None:
        zero_record = None
        finite_record = None
        direct_assumption = None
    else:
        source = _mapping(results, "elasticity_results")
        zero_key = next((key for key in _ZERO_K_KEYS if key in source), None)
        finite_key = next((key for key in _FINITE_T_KEYS if key in source), None)
        is_nested = zero_key is not None or finite_key is not None
        if is_nested:
            zero_value = source.get(zero_key) if zero_key is not None else None
            finite_value = source.get(finite_key) if finite_key is not None else None
            zero_record = (
                None
                if zero_value is None
                else _mapping(zero_value, f"elasticity_results.{zero_key}")
            )
            finite_record = (
                None
                if finite_value is None
                else _mapping(finite_value, f"elasticity_results.{finite_key}")
            )
            direct_assumption = None
        else:
            zero_record = source
            finite_record = None
            direct_assumption = "direct elasticity input is interpreted as zero kelvin"

    return {
        "cauchy_definition": "C12 - C44",
        "normalization": "(C12 - C44) / max(abs(C12), abs(C44))",
        "zero_kelvin": {
            **_elastic_record(
                zero_record,
                location="elasticity_results.zero_k",
                interpretation="model_form_diagnostic",
                assumed_temperature_k=0.0,
            ),
            "assumption": direct_assumption,
            "scope": "simple central pair model at zero pressure",
        },
        "finite_temperature": {
            **_elastic_record(
                finite_record,
                location="elasticity_results.finite_temperature",
                interpretation="diagnostic_only",
                assumed_temperature_k=None,
            ),
            "scope": (
                "thermal, stress-fluctuation, and kinetic contributions prevent a "
                "literal zero-kelvin Cauchy test"
            ),
        },
    }


def _label_swap_diagnostic(result: Mapping[str, Any] | None) -> dict[str, Any]:
    if result is None:
        return {
            "status": "unknown",
            "reference_value": None,
            "swapped_value": None,
            "absolute_change": None,
            "normalized_absolute_change": None,
            "relative_tolerance": None,
            "interpretation": "transferability_diagnostic",
        }
    record = _mapping(result, "label_swap_result")
    passed = record.get("passed")
    if passed is not None and not isinstance(passed, bool):
        raise ModelAdequacyError("label_swap_result.passed must be boolean")
    explicit_status = _clean_text(record.get("status"))
    if explicit_status is not None:
        explicit_status = explicit_status.lower()
        if explicit_status not in {"pass", "fail", "unknown", "measured"}:
            raise ModelAdequacyError(
                "label_swap_result.status must be pass, fail, measured, or unknown"
            )

    reference = _optional_finite(
        record,
        ("reference_value", "original", "reference"),
        "label_swap_result",
    )
    swapped = _optional_finite(
        record,
        ("swapped_value", "swapped"),
        "label_swap_result",
    )
    tolerance = _optional_finite(
        record,
        ("relative_tolerance",),
        "label_swap_result",
    )
    if tolerance is not None and tolerance < 0.0:
        raise ModelAdequacyError("label_swap_result.relative_tolerance must be non-negative")

    absolute_change: float | None = None
    normalized: float | None = None
    if reference is not None and swapped is not None:
        absolute_change = abs(swapped - reference)
        scale = max(abs(reference), abs(swapped))
        normalized = None if scale <= _EPS else absolute_change / scale

    if passed is not None:
        status = "pass" if passed else "fail"
    elif explicit_status is not None:
        status = explicit_status
    elif normalized is not None and tolerance is not None:
        status = "pass" if normalized <= tolerance else "fail"
    elif absolute_change is not None:
        status = "measured"
    else:
        status = "unknown"

    return {
        "status": status,
        "reference_value": reference,
        "swapped_value": swapped,
        "absolute_change": absolute_change,
        "normalized_absolute_change": normalized,
        "relative_tolerance": tolerance,
        "interpretation": "transferability_diagnostic",
    }


def _transferability_status(value: Any, location: str) -> str:
    status = _clean_text(value)
    if status is None:
        raise ModelAdequacyError(f"{location} must be a transferability status")
    normalized = status.lower()
    if normalized not in _TRANSFERABILITY_STATUSES:
        choices = ", ".join(sorted(_TRANSFERABILITY_STATUSES))
        raise ModelAdequacyError(f"{location} must be one of: {choices}")
    return normalized


def _aggregate_case_statuses(statuses: Sequence[str]) -> str:
    """Aggregate independent cases with fail-closed precedence."""
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "error" for status in statuses):
        return "error"
    if any(status == "not_evaluated" for status in statuses):
        return "not_evaluated"
    if statuses and all(status == "not_applicable" for status in statuses):
        return "not_applicable"
    return "pass" if statuses else "not_evaluated"


def _provenance_attestation(
    value: Any,
    *,
    location: str,
) -> tuple[dict[str, Any] | None, list[str]]:
    """Validate attestation shape; this does not read or verify artifacts."""
    if value is None:
        return None, [f"{location} is required for a pass"]
    record = _mapping(value, location)
    errors: list[str] = []
    schema_version = record.get("schema_version")
    if schema_version != 1:
        errors.append(f"{location}.schema_version must equal 1")
    audited = record.get("content_audit_attested")
    if audited is not True:
        errors.append(f"{location}.content_audit_attested must be true")
    producer = _clean_text(record.get("producer"))
    if producer not in _SUPPORTED_PROVENANCE_PRODUCERS:
        choices = ", ".join(sorted(_SUPPORTED_PROVENANCE_PRODUCERS))
        errors.append(f"{location}.producer must be one of: {choices}")
    hashes: dict[str, str | None] = {}
    for key in ("manifest_sha256", "protocol_sha256", "metrics_sha256"):
        digest = _clean_text(record.get(key))
        normalized = digest.lower() if digest is not None else None
        if normalized is None or _SHA256_PATTERN.fullmatch(normalized) is None:
            errors.append(f"{location}.{key} must be a 64-character SHA-256")
        hashes[key] = normalized
    if errors:
        return None, errors
    return {
        "schema_version": 1,
        "content_audit_attested": True,
        "producer": producer,
        **hashes,
    }, []


def _normalize_transferability_record(
    test_id: str,
    value: Any,
) -> dict[str, Any]:
    if value is None:
        source: Mapping[str, Any] = {}
        explicit_status = None
    elif isinstance(value, str):
        source = {}
        explicit_status = _transferability_status(
            value, f"transferability_evidence.{test_id}"
        )
    else:
        source = _mapping(value, f"transferability_evidence.{test_id}")
        explicit = source.get("status")
        explicit_status = (
            None
            if explicit is None
            else _transferability_status(
                explicit, f"transferability_evidence.{test_id}.status"
            )
        )

    raw_cases = source.get("cases", [])
    if isinstance(raw_cases, (str, bytes, bytearray)) or not isinstance(
        raw_cases, Sequence
    ):
        raise ModelAdequacyError(
            f"transferability_evidence.{test_id}.cases must be a sequence"
        )
    cases: list[dict[str, Any]] = []
    for index, raw_case in enumerate(raw_cases):
        location = f"transferability_evidence.{test_id}.cases[{index}]"
        if isinstance(raw_case, str):
            case = {"status": _transferability_status(raw_case, location)}
        else:
            case_source = _mapping(raw_case, location)
            case = {
                "status": _transferability_status(
                    case_source.get("status"), f"{location}.status"
                )
            }
            for key in ("id", "evidence", "message"):
                text = _clean_text(case_source.get(key))
                if text is not None:
                    case[key] = text
        cases.append(case)

    case_status = _aggregate_case_statuses([case["status"] for case in cases])
    if explicit_status is not None and cases and explicit_status != case_status:
        raise ModelAdequacyError(
            f"transferability_evidence.{test_id}.status conflicts with its cases"
        )
    status = explicit_status or case_status
    minimum_cases = _TRANSFERABILITY_MINIMUM_CASES[test_id]
    applicable_cases = [
        case for case in cases
        if case["status"] not in {"not_evaluated", "not_applicable"}
    ]
    verification_errors: list[str] = []
    attestation: dict[str, Any] | None = None
    if status == "pass":
        if len(applicable_cases) < minimum_cases:
            verification_errors.append(
                f"{test_id} requires at least {minimum_cases} applicable cases"
            )
        case_ids = [case.get("id") for case in applicable_cases]
        if any(case_id is None for case_id in case_ids):
            verification_errors.append(
                f"every applicable {test_id} case must have a non-empty id"
            )
        elif len(set(case_ids)) != len(case_ids):
            verification_errors.append(f"{test_id} case ids must be unique")
        attestation, provenance_errors = _provenance_attestation(
            source.get("provenance_attestation"),
            location=f"transferability_evidence.{test_id}.provenance_attestation",
        )
        verification_errors.extend(provenance_errors)
        if verification_errors:
            status = "error"

    effective_status = "attested_pass" if status == "pass" else status
    result: dict[str, Any] = {
        "id": test_id,
        "status": status,
        "effective_status": effective_status,
        "claimed_status": explicit_status or case_status,
        "cases": cases,
        "case_count": len(cases),
        "applicable_case_count": len(applicable_cases),
        "minimum_cases": minimum_cases,
        "verification_status": (
            "attested_not_content_verified"
            if status == "pass"
            else "rejected_unverified_pass"
            if verification_errors
            else "not_required_for_nonpass"
        ),
        "provenance_attestation": attestation,
        "verification_errors": verification_errors,
    }
    for key in ("evidence", "message"):
        text = _clean_text(source.get(key))
        if text is not None:
            result[key] = text
    return result


def _aggregate_required_test_statuses(statuses: Sequence[str]) -> str:
    """Aggregate required evidence without upgrading attestations to verification."""
    if any(status == "fail" for status in statuses):
        return "fail"
    if any(status == "error" for status in statuses):
        return "error"
    if any(status in {"not_evaluated", "not_applicable"} for status in statuses):
        return "not_evaluated"
    if statuses and all(status == "attested_pass" for status in statuses):
        return "attested_pass"
    return "pass" if statuses and all(status == "pass" for status in statuses) else "error"


def aggregate_transferability_evidence(
    evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize and aggregate the independent elemental transferability gate.

    This evidence is deliberately separate from ordered-sublattice bulk
    validation.  A failed or missing transferability experiment limits the
    claim that may be made; it does not retroactively invalidate properties
    calculated for the original permanent type assignment.
    """
    source = _optional_mapping(evidence, "transferability_evidence")
    normalized_source: dict[str, Any] = {}
    for raw_id, value in source.items():
        test_id = _TRANSFERABILITY_ALIASES.get(str(raw_id), str(raw_id))
        if test_id not in _TRANSFERABILITY_TESTS:
            raise ModelAdequacyError(
                f"unsupported transferability evidence id: {raw_id}"
            )
        if test_id in normalized_source:
            raise ModelAdequacyError(
                f"duplicate transferability evidence for {test_id}"
            )
        normalized_source[test_id] = value

    tests = {
        test_id: _normalize_transferability_record(
            test_id, normalized_source.get(test_id)
        )
        for test_id in _TRANSFERABILITY_TESTS
    }
    statuses = [
        tests[test_id]["effective_status"] for test_id in _TRANSFERABILITY_TESTS
    ]
    aggregate = _aggregate_required_test_statuses(statuses)
    return {
        "schema_version": TRANSFERABILITY_EVIDENCE_SCHEMA_VERSION,
        "status": aggregate,
        "required_tests": list(_TRANSFERABILITY_TESTS),
        "complete": aggregate in {"pass", "fail"},
        "counts": {
            status: statuses.count(status)
            for status in sorted(_TRANSFERABILITY_STATUSES | {"attested_pass"})
        },
        "tests": tests,
        "scope": "elemental_transferability_only",
        "ordered_sublattice_bulk_validation_affected": False,
    }


def _resolved_type_value(
    resolved: Mapping[str, Any],
    label: str,
    parameter: str,
) -> float | None:
    key = f"{label}_{parameter}"
    if key not in resolved:
        return None
    return _finite(resolved[key], f"resolved_parameters.{key}")


def _mixed_lj_coefficients(
    first: tuple[float, float],
    second: tuple[float, float],
    rule: str,
) -> tuple[float, float]:
    epsilon_i, sigma_i = first
    epsilon_j, sigma_j = second
    if min(epsilon_i, epsilon_j, sigma_i, sigma_j) < 0.0:
        raise ModelAdequacyError("resolved LJ epsilon and sigma must be non-negative")
    if rule == "geometric":
        return math.sqrt(epsilon_i * epsilon_j), math.sqrt(sigma_i * sigma_j)
    if rule == "arithmetic":
        return math.sqrt(epsilon_i * epsilon_j), 0.5 * (sigma_i + sigma_j)
    if rule == "sixthpower":
        denominator = sigma_i**6 + sigma_j**6
        if denominator <= _EPS:
            return math.sqrt(epsilon_i * epsilon_j), 0.0
        sigma = (0.5 * denominator) ** (1.0 / 6.0)
        epsilon = (
            2.0
            * math.sqrt(epsilon_i * epsilon_j)
            * sigma_i**3
            * sigma_j**3
            / denominator
        )
        return epsilon, sigma
    raise ModelAdequacyError(f"unsupported resolved LJ mixing rule: {rule}")


def _explicit_cross_coefficients(
    config: Mapping[str, Any],
    resolved: Mapping[str, Any],
) -> dict[tuple[int, int], tuple[float, float]]:
    pair_params = _optional_mapping(config.get("pair_params"), "config.pair_params")
    result: dict[tuple[int, int], tuple[float, float]] = {}
    for index, raw in enumerate(pair_params.get("explicit_pairs", [])):
        pair = _mapping(raw, f"config.pair_params.explicit_pairs[{index}]")
        raw_types = pair.get("types")
        if (
            isinstance(raw_types, (str, bytes, bytearray))
            or not isinstance(raw_types, Sequence)
            or len(raw_types) != 2
            or any(isinstance(item, bool) or not isinstance(item, int) for item in raw_types)
        ):
            raise ModelAdequacyError(
                f"config.pair_params.explicit_pairs[{index}].types must contain two integer ids"
            )
        first, second = int(raw_types[0]), int(raw_types[1])
        epsilon_key = f"cross_{first}_{second}_epsilon"
        sigma_key = f"cross_{first}_{second}_sigma"
        reverse_epsilon_key = f"cross_{second}_{first}_epsilon"
        reverse_sigma_key = f"cross_{second}_{first}_sigma"
        if epsilon_key in resolved and sigma_key in resolved:
            epsilon = _finite(resolved[epsilon_key], f"resolved_parameters.{epsilon_key}")
            sigma = _finite(resolved[sigma_key], f"resolved_parameters.{sigma_key}")
        elif reverse_epsilon_key in resolved and reverse_sigma_key in resolved:
            epsilon = _finite(
                resolved[reverse_epsilon_key],
                f"resolved_parameters.{reverse_epsilon_key}",
            )
            sigma = _finite(
                resolved[reverse_sigma_key],
                f"resolved_parameters.{reverse_sigma_key}",
            )
        else:
            continue
        result[tuple(sorted((first, second)))] = (epsilon, sigma)
    return result


def assess_same_element_relabeling_invariance(
    compiled_config: Mapping[str, Any],
    *,
    resolved_parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Test invariance to arbitrary same-element type reassignment.

    A simultaneous permutation ``P.T @ M @ P`` merely renames a parameter
    matrix and is therefore not this test.  Arbitrary reassignment is
    invariant only when every complete pair-coefficient row and column is the
    same (and type-local mass/charge values are the same).  The full ``i,j``
    matrix is expanded using the resolved LAMMPS mixing rule before testing.
    """
    config = _mapping(compiled_config, "compiled_config")
    material = _optional_mapping(config.get("material"), "config.material")
    atom_types = _atom_types(config)
    elemental = (_clean_text(material.get("kind")) or "unknown").lower() == "elemental"
    if not elemental or len(atom_types) <= 1:
        return {
            "status": "not_applicable",
            "formal_invariance_status": "not_applicable",
            "reason": "requires an elemental material represented by multiple atom types",
            "criterion": "all complete pair-coefficient rows/columns and type-local attributes are identical",
            "pair_matrix_complete": False,
            "pair_coefficients": [],
            "violations": [],
        }
    pair_model = _pair_model(config)
    pair_style = pair_model["pair_style"]
    if pair_style not in _SUPPORTED_FORMAL_LJ_STYLES:
        return {
            "status": "not_evaluated",
            "formal_invariance_status": "not_evaluated",
            "reason": (
                "formal relabeling is implemented only for a single, fully "
                "resolved supported LJ pair style"
            ),
            "pair_style": pair_style,
            "pair_style_source": pair_model["pair_style_source"],
            "supported_pair_styles": sorted(_SUPPORTED_FORMAL_LJ_STYLES),
            "criterion": "all complete pair-coefficient rows/columns and type-local attributes are identical",
            "pair_matrix_complete": False,
            "pair_coefficients": [],
            "violations": [],
        }
    if resolved_parameters is None:
        return {
            "status": "not_evaluated",
            "formal_invariance_status": "not_evaluated",
            "reason": "final resolved parameters were not supplied",
            "criterion": "all complete pair-coefficient rows/columns and type-local attributes are identical",
            "pair_matrix_complete": False,
            "pair_coefficients": [],
            "violations": [],
        }
    resolved = _mapping(resolved_parameters, "resolved_parameters")
    mixing = pair_model["mixing"]["resolved_for_diagnostics"]
    configured_mixing = pair_model["mixing"]["configured"]
    charge_config = _optional_mapping(config.get("charge"), "config.charge")
    charge_enabled = charge_config.get("enabled")
    if charge_enabled is not None and not isinstance(charge_enabled, bool):
        raise ModelAdequacyError("config.charge.enabled must be boolean")

    type_records: list[dict[str, Any]] = []
    self_coefficients: dict[int, tuple[float, float]] = {}
    missing: list[str] = []
    missing_type_attributes: list[str] = []
    assumptions: list[str] = []
    for index, item in enumerate(atom_types):
        type_id = item.get("type")
        if isinstance(type_id, bool) or not isinstance(type_id, int):
            raise ModelAdequacyError(f"config.atom_types[{index}].type must be an integer")
        label = _type_label(item, index)
        epsilon = _resolved_type_value(resolved, label, "epsilon")
        sigma = _resolved_type_value(resolved, label, "sigma")
        if epsilon is None:
            missing.append(f"{label}_epsilon")
        if sigma is None:
            missing.append(f"{label}_sigma")
        if epsilon is not None and sigma is not None:
            self_coefficients[type_id] = (epsilon, sigma)
        charge = _resolved_type_value(resolved, label, "charge")
        if charge is None:
            if charge_enabled is False:
                charge = 0.0
                assumptions.append(
                    f"{label}_charge was absent and is treated as 0.0 because config.charge.enabled is false"
                )
            else:
                missing.append(f"{label}_charge")
        mass = _optional_finite(item, ("mass",), f"config.atom_types[{index}]")
        if mass is None:
            missing_type_attributes.append(f"{label}.mass")
        type_records.append({
            "type": type_id,
            "label": label,
            "mass": mass,
            "charge": charge,
        })

    if missing:
        return {
            "status": "not_evaluated",
            "formal_invariance_status": "not_evaluated",
            "reason": "final resolved force-field parameters are incomplete",
            "missing_parameters": sorted(missing),
            "charge_enabled": charge_enabled,
            "criterion": "all complete pair-coefficient rows/columns and type-local attributes are identical",
            "pair_matrix_complete": False,
            "pair_coefficients": [],
            "violations": [],
        }
    if missing_type_attributes:
        return {
            "status": "not_evaluated",
            "formal_invariance_status": "not_evaluated",
            "reason": "type-local attributes needed for arbitrary relabeling are incomplete",
            "missing_type_attributes": sorted(missing_type_attributes),
            "assumptions": assumptions,
            "criterion": "all complete pair-coefficient rows/columns and type-local attributes are identical",
            "pair_matrix_complete": False,
            "pair_coefficients": [],
            "violations": [],
        }

    explicit = _explicit_cross_coefficients(config, resolved)
    ids = [record["type"] for record in type_records]
    matrix: dict[tuple[int, int], tuple[float, float]] = {}
    missing_pairs: list[str] = []
    for first in ids:
        for second in ids:
            key = tuple(sorted((first, second)))
            if first == second:
                coefficients = self_coefficients[first]
            elif configured_mixing == "none":
                coefficients = explicit.get(key)
                if coefficients is None:
                    missing_pairs.append(f"{key[0]}-{key[1]}")
                    continue
            elif mixing in {"geometric", "arithmetic", "sixthpower"}:
                coefficients = _mixed_lj_coefficients(
                    self_coefficients[first], self_coefficients[second], mixing
                )
            else:
                missing_pairs.append(f"{key[0]}-{key[1]}")
                continue
            matrix[(first, second)] = coefficients

    pair_rows = [
        {
            "type_i": first,
            "type_j": second,
            "epsilon": matrix[(first, second)][0],
            "sigma": matrix[(first, second)][1],
        }
        for first in ids
        for second in ids
        if (first, second) in matrix
    ]
    if missing_pairs:
        return {
            "status": "not_evaluated",
            "formal_invariance_status": "not_evaluated",
            "reason": "complete pair-coefficient rows/columns could not be resolved",
            "missing_pairs": sorted(set(missing_pairs)),
            "criterion": "all complete pair-coefficient rows/columns and type-local attributes are identical",
            "mixing_rule": mixing,
            "pair_matrix_complete": False,
            "pair_coefficients": pair_rows,
            "violations": [],
        }

    reference = matrix[(ids[0], ids[0])]
    violations: list[dict[str, Any]] = []
    for first in ids:
        for second in ids:
            epsilon, sigma = matrix[(first, second)]
            for parameter, observed, expected in (
                ("epsilon", epsilon, reference[0]),
                ("sigma", sigma, reference[1]),
            ):
                if not math.isclose(
                    observed,
                    expected,
                    rel_tol=_RELABEL_RTOL,
                    abs_tol=_RELABEL_ATOL,
                ):
                    violations.append({
                        "kind": "pair_coefficient",
                        "type_i": first,
                        "type_j": second,
                        "parameter": parameter,
                        "value": observed,
                        "reference_value": expected,
                        "absolute_difference": abs(observed - expected),
                    })
    for attribute in ("mass", "charge"):
        values = [record[attribute] for record in type_records]
        if values[0] is None:
            continue
        for record, value in zip(type_records[1:], values[1:]):
            if value is None or not math.isclose(
                float(value),
                float(values[0]),
                rel_tol=_RELABEL_RTOL,
                abs_tol=_RELABEL_ATOL,
            ):
                violations.append({
                    "kind": "type_local_attribute",
                    "type": record["type"],
                    "parameter": attribute,
                    "value": value,
                    "reference_value": values[0],
                    "absolute_difference": (
                        None if value is None else abs(float(value) - float(values[0]))
                    ),
                })

    status = "fail" if violations else "pass"
    return {
        "status": status,
        "formal_invariance_status": status,
        "reason": (
            "one or more complete pair-coefficient rows/columns or type-local attributes differ"
            if violations
            else "all complete pair-coefficient rows/columns and type-local attributes are identical"
        ),
        "criterion": "all complete pair-coefficient rows/columns and type-local attributes are identical",
        "not_a_matrix_permutation_test": True,
        "mixing_rule": mixing,
        "charge_enabled": charge_enabled,
        "pair_matrix_complete": True,
        "pair_coefficients": pair_rows,
        "type_local_attributes": type_records,
        "assumptions": assumptions,
        "violations": violations,
    }


def assess_model_adequacy(
    compiled_config: Mapping[str, Any],
    *,
    elasticity_results: Mapping[str, Any] | None = None,
    label_swap_result: Mapping[str, Any] | None = None,
    resolved_parameters: Mapping[str, Any] | None = None,
    transferability_evidence: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return a JSON-compatible, read-only model-form applicability report.

    A warning changes neither optimizer eligibility nor candidate acceptance.
    Downstream policy may require the listed validations, but this function
    intentionally does not turn missing evidence into a claim of impossibility.
    """
    config = _mapping(compiled_config, "compiled_config")
    material = _optional_mapping(config.get("material"), "config.material")
    material_kind = _clean_text(material.get("kind")) or "unknown"
    material_name = _clean_text(material.get("name"))
    atom_types = _atom_types(config)
    type_count = len(atom_types)
    elemental = material_kind.lower() == "elemental"

    if not elemental:
        representation = "not_elemental" if material_kind != "unknown" else "unknown"
    elif type_count == 1:
        representation = "single_type_elemental"
    elif type_count > 1:
        representation = "permanent_ordered_sublattice_surrogate"
    else:
        representation = "unknown"

    pair_model = _pair_model(config)
    symmetry = _parameter_symmetry(config, atom_types)
    label_swap = _label_swap_diagnostic(label_swap_result)
    elasticity = _elasticity_diagnostics(elasticity_results)
    formal_invariance = assess_same_element_relabeling_invariance(
        config, resolved_parameters=resolved_parameters
    )
    empirical_input = dict(
        _optional_mapping(transferability_evidence, "transferability_evidence")
    )
    if "label_swap" not in empirical_input and label_swap["status"] in {"pass", "fail"}:
        empirical_input["label_swap"] = {"status": label_swap["status"]}
    empirical_evidence = aggregate_transferability_evidence(empirical_input)

    warnings: list[dict[str, Any]] = []
    required_validations: list[dict[str, Any]] = []
    if elemental and type_count > 1:
        warnings.append(
            {
                "code": "ordered_sublattice_surrogate",
                "severity": "warning",
                "scope": "physical_transferability",
                "message": (
                    "One elemental species is represented by multiple permanent atom "
                    "types. This is an ordered-sublattice surrogate and must not be "
                    "interpreted as two chemical species."
                ),
                "implies_fit_is_impossible": False,
            }
        )
        required_validations = [
            {
                "id": test_id,
                "status": empirical_evidence["tests"][test_id]["effective_status"],
                "minimum_cases": _TRANSFERABILITY_MINIMUM_CASES[test_id],
            }
            for test_id in _TRANSFERABILITY_TESTS
        ]
        if label_swap["status"] == "fail":
            warnings.append(
                {
                    "code": "label_swap_sensitivity",
                    "severity": "warning",
                    "scope": "physical_transferability",
                    "message": (
                        "The supplied label-swap diagnostic failed; predictions depend "
                        "on the permanent same-element type assignment."
                    ),
                    "implies_fit_is_impossible": False,
                }
            )
        if formal_invariance["status"] == "fail":
            warnings.append(
                {
                    "code": "formal_same_element_relabeling_noninvariance",
                    "severity": "warning",
                    "scope": "elemental_transferability",
                    "message": (
                        "The final complete pair-coefficient rows/columns are not "
                        "identical across same-element atom types. The result is valid "
                        "for the fitted ordered sublattice but is not invariant to "
                        "arbitrary same-element type reassignment."
                    ),
                    "implies_fit_is_impossible": False,
                    "affects_ordered_sublattice_bulk_validation": False,
                }
            )

    mixing_status = pair_model["mixing"]["resolution_status"]
    central_lj = pair_model["central_lj_pair_style"]
    if central_lj is None or mixing_status == "unknown":
        composability_status = "unknown"
    elif central_lj is True:
        composability_status = "computationally_composable"
    else:
        composability_status = "model_specific"

    if elemental and type_count > 1:
        formal_status = formal_invariance["status"]
        empirical_status = empirical_evidence["status"]
        if formal_status == "fail":
            transferability_status = "ordered_sublattice_only"
            transferability_claim = "not_supported"
        elif empirical_status == "fail":
            transferability_status = "not_transferable"
            transferability_claim = "not_supported"
        elif empirical_status == "error":
            transferability_status = "incomplete"
            transferability_claim = "not_established"
        elif empirical_status == "attested_pass":
            transferability_status = "requires_verified_runner"
            transferability_claim = "not_established"
        elif empirical_status == "pass":
            # Reserved for a future trusted runner. This pure module cannot
            # verify artifact bytes, so it must not grant the positive claim.
            transferability_status = "requires_verified_runner"
            transferability_claim = "not_established"
        else:
            transferability_status = "requires_validation"
            transferability_claim = (
                "requires_empirical_validation"
                if formal_status == "pass"
                else "not_established"
            )
    else:
        transferability_status = "unknown"
        transferability_claim = "not_applicable" if elemental and type_count == 1 else "unknown"

    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "assessment_kind": "model_form_and_applicability",
        "read_only": True,
        "material": {
            "kind": material_kind,
            "name": material_name,
            "atom_type_count": type_count,
            "same_element_representation": representation,
        },
        "model_form": pair_model,
        "parameter_symmetry": symmetry,
        "elasticity_diagnostics": elasticity,
        "label_swap_diagnostic": label_swap,
        "same_element_relabeling": formal_invariance,
        "transferability_evidence": empirical_evidence,
        "formal_invariance_status": formal_invariance["status"],
        "empirical_transfer_status": empirical_evidence["status"],
        "elemental_transferability_claim": transferability_claim,
        "computational_composability": {
            "status": composability_status,
            "basis": (
                "declared pair style and mixing semantics only; runtime cross-system "
                "compatibility remains an integration test"
            ),
        },
        "physical_transferability": {
            "status": transferability_status,
            "formal_invariance_status": formal_invariance["status"],
            "empirical_transfer_status": empirical_evidence["status"],
            "elemental_transferability_claim": transferability_claim,
            "required_validations": required_validations,
            "warning_is_not_impossibility_proof": True,
            "ordered_sublattice_bulk_validation_affected": False,
        },
        "warnings": warnings,
    }


__all__ = [
    "ModelAdequacyError",
    "REPORT_SCHEMA_VERSION",
    "TRANSFERABILITY_EVIDENCE_SCHEMA_VERSION",
    "aggregate_transferability_evidence",
    "assess_same_element_relabeling_invariance",
    "assess_model_adequacy",
]
