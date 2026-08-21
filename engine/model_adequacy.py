"""Read-only diagnostics for force-field model form and applicability.

The optimizer answers a computational question: *can this parameterization
reproduce the requested observations?*  That is not the same as the physical
question: *will the fitted model transfer to configurations that were not in
the fit?*  This module keeps those conclusions separate.

``assess_model_adequacy`` is deliberately a pure function.  It does not read a
LAMMPS input, launch a calculation, reject a candidate, or mutate the compiled
configuration.  Missing evidence is reported as ``"unknown"``.  In
particular, an elemental material represented by several permanent LAMMPS atom
types is flagged as an ordered-sublattice surrogate, but that warning is not a
proof that a fit is impossible.

Optional elasticity input may be either a direct zero-kelvin mapping::

    {"C12": 138.1, "C44": 121.9}

or a mapping with ``zero_k`` and/or ``finite_temperature`` records.  A record
may place the constants directly in the record or below
``elastic_constants_gpa``.  Finite-temperature Cauchy pressure is always
labelled ``diagnostic_only`` because thermal, stress-fluctuation, and kinetic
contributions invalidate a literal zero-kelvin central-force interpretation.

The optional label-swap result accepts ``passed`` (a boolean), or numerical
``reference_value``/``swapped_value`` values and an optional
``relative_tolerance``.  Numerical inputs are required to be finite; the
returned report contains no NaN or infinity and is JSON-compatible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from numbers import Real
from typing import Any


REPORT_SCHEMA_VERSION = 1
_PARAMETERS = ("epsilon", "sigma", "charge")
_ZERO_K_KEYS = ("zero_k", "zero_kelvin", "0k", "0_K")
_FINITE_T_KEYS = ("finite_temperature", "finite_t", "300k", "300_K")
_EPS = 1.0e-15


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


def assess_model_adequacy(
    compiled_config: Mapping[str, Any],
    *,
    elasticity_results: Mapping[str, Any] | None = None,
    label_swap_result: Mapping[str, Any] | None = None,
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
            {"id": "label_swap", "status": label_swap["status"]},
            {"id": "surface_terminations", "status": "unknown", "minimum_cases": 2},
            {"id": "vacancy", "status": "unknown"},
            {"id": "diffusion", "status": "unknown"},
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

    mixing_status = pair_model["mixing"]["resolution_status"]
    central_lj = pair_model["central_lj_pair_style"]
    if central_lj is None or mixing_status == "unknown":
        composability_status = "unknown"
    elif central_lj is True:
        composability_status = "computationally_composable"
    else:
        composability_status = "model_specific"

    if elemental and type_count > 1:
        transferability_status = "requires_validation"
    else:
        transferability_status = "unknown"

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
        "computational_composability": {
            "status": composability_status,
            "basis": (
                "declared pair style and mixing semantics only; runtime cross-system "
                "compatibility remains an integration test"
            ),
        },
        "physical_transferability": {
            "status": transferability_status,
            "required_validations": required_validations,
            "warning_is_not_impossibility_proof": True,
        },
        "warnings": warnings,
    }


__all__ = [
    "ModelAdequacyError",
    "REPORT_SCHEMA_VERSION",
    "assess_model_adequacy",
]
