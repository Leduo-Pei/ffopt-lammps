"""Content-addressed final validation for material force fields.

The final validation deliberately reruns the complete fitted structural and
surface property plan, then composes the existing cubic-elastic batch runner
for 0 K evidence and finite-temperature evidence.  Explicit validation seeds
form an independent holdout set; legacy configs that reuse promotion seeds are
labelled as such rather than being described as independent.
Mechanical target bands are reporting tiers, not acceptance gates: a
candidate that passes structure, Born stability, and fit quality remains a
``best_effort`` final result when its elastic error exceeds the tier.

No stage is discovered by directory timestamps or globs.  The caller supplies
the compiled configuration, an explicit parameter document (or explicit
``name=value`` mapping), and an independent output directory.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Mapping, Sequence
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np
import pandas as pd

from .config_loader import deep_merge, load_config
from .cubic_elastic_batch import (
    Candidate,
    NestedResourcePlan,
    assess_structural_gates,
    default_backend_factory,
    effective_available_cores,
    plan_nested_resources,
    prepare_candidate,
    run_elasticity_batch,
)
from .lammps_interface import LAMMPSRunner
from .model_adequacy import assess_model_adequacy
from .parameter_space import build_parameter_space
from .validate_final_parameters import _load_parameters
from workflow.artifact_manifest import (
    build_artifact_manifest,
    canonical_parameter_key,
    check_artifact_reuse,
    sha256_file,
    validate_artifact_manifest,
    write_artifact_manifest,
)


class MaterialValidationError(RuntimeError):
    """Base error for final material validation."""


class MaterialValidationArtifactError(MaterialValidationError):
    """Raised when a completed validation artifact no longer verifies."""


RunnerFactory = Callable[[dict[str, Any]], LAMMPSRunner]
ElasticBatchRunner = Callable[..., pd.DataFrame]


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return str(value)


def _json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            _json_safe(value),
            indent=2,
            sort_keys=True,
            ensure_ascii=True,
            allow_nan=False,
        )
        + "\n"
    ).encode("ascii")


def _atomic_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _atomic_bytes(path, _json_bytes(value))


def _write_immutable_bytes(path: Path, content: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != content:
            raise MaterialValidationArtifactError(
                f"refusing to overwrite immutable validation input: {path}"
            )
        return
    _atomic_bytes(path, content)


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    text = frame.to_csv(index=False, lineterminator="\n")
    _atomic_bytes(path, text.encode("utf-8"))


def _truth(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _finite(value: Any) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _validation_config(config: dict[str, Any]) -> dict[str, Any]:
    validation = config.get("validation", {})
    overrides: dict[str, Any] = {
        "property_evaluators": validation.get("property_evaluators", {}),
    }
    property_settings = validation.get("property_settings", {})
    if isinstance(property_settings, Mapping):
        overrides.update(property_settings)
    return deep_merge(config, overrides)


def _parse_assignments(assignments: Sequence[str]) -> dict[str, float]:
    parameters: dict[str, float] = {}
    for assignment in assignments:
        if "=" not in assignment:
            raise ValueError(
                f"explicit parameter must use NAME=VALUE syntax: {assignment!r}"
            )
        name, raw_value = assignment.split("=", 1)
        name = name.strip()
        if not name or name in parameters:
            raise ValueError(f"invalid or duplicate explicit parameter: {name!r}")
        try:
            value = float(raw_value)
        except ValueError as exc:
            raise ValueError(f"parameter {name!r} is not numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"parameter {name!r} must be finite")
        parameters[name] = value
    return parameters


def _validated_candidate(
    parameters: Mapping[str, float],
    config: Mapping[str, Any],
) -> Candidate:
    space = build_parameter_space(dict(config))
    expected = [name for name, _lower, _upper in space]
    missing = sorted(set(expected) - set(parameters))
    unexpected = sorted(set(parameters) - set(expected))
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing {missing}")
        if unexpected:
            details.append(f"unexpected {unexpected}")
        raise ValueError("final free parameters do not match compiled space: " + "; ".join(details))
    bounds = {name: (float(lower), float(upper)) for name, lower, upper in space}
    raw: dict[str, float] = {}
    for name in expected:
        value = float(parameters[name])
        if not math.isfinite(value):
            raise ValueError(f"final parameter {name!r} must be finite")
        lower, upper = bounds[name]
        tolerance = 1.0e-12 * max(1.0, abs(value), abs(lower), abs(upper))
        if value < lower - tolerance or value > upper + tolerance:
            raise ValueError(
                f"final parameter {name!r}={value:.12g} is outside "
                f"[{lower:.12g}, {upper:.12g}]"
            )
        raw[name] = value
    key = canonical_parameter_key(raw)
    return Candidate(key, raw, {"parameter_key": key, **raw}, 1)


def _load_parameter_source(
    *,
    output_dir: Path,
    parameters_path: str | os.PathLike[str] | None,
    parameters: Mapping[str, float] | None,
) -> tuple[dict[str, float], Path, dict[str, Any]]:
    if (parameters_path is None) == (parameters is None):
        raise ValueError("provide exactly one of parameters_path or parameters")
    if parameters_path is not None:
        source = Path(parameters_path).resolve()
        raw = _load_parameters(source)
        provenance = {
            "kind": "parameter_document",
            "path": str(source),
            "artifact": sha256_file(source).to_dict(),
        }
        return raw, source, provenance
    raw = {str(name): float(value) for name, value in parameters.items()}
    source = output_dir / "explicit_parameter_input.json"
    document = {"raw_free_parameters": raw}
    _write_immutable_bytes(source, _json_bytes(document))
    provenance = {
        "kind": "explicit_name_value_mapping",
        "path": str(source.resolve()),
        "artifact": sha256_file(source).to_dict(),
    }
    return raw, source, provenance


def _runner_input_artifacts(
    runner: LAMMPSRunner,
    *,
    config_path: Path,
    parameter_source: Path,
    resolved_parameters: Path,
    force_field_include: Path,
) -> dict[str, Path]:
    artifacts = {
        "compiled_config": config_path,
        "parameter_source": parameter_source,
        "resolved_parameters": resolved_parameters,
        "force_field_include": force_field_include,
    }
    attributes = (
        "bulk_data",
        "bulk_input",
        "surf_input",
        "surf_complete",
        "surf_split",
        "ads_complex_input",
        "ads_slab_input",
        "ads_mol_input",
        "ads_complex",
        "ads_slab",
        "ads_mol",
        "sub_single_input",
        "sub_single_data",
    )
    for name in attributes:
        raw = getattr(runner, name, None)
        if raw is None:
            continue
        path = Path(raw)
        if path.is_file():
            artifacts[f"runner_{name}"] = path
    provider = getattr(runner, "validation_input_artifacts", None)
    if callable(provider):
        for name, raw in provider().items():
            path = Path(raw)
            if not path.is_file():
                raise FileNotFoundError(f"validation input artifact is missing: {path}")
            artifacts[f"runner_extra_{name}"] = path
    return artifacts


def _structure_scientific_config(config: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "material": config.get("material", {}),
        "crystal": config.get("crystal", {}),
        "targets": config.get("targets", {}),
        "lammps": config.get("lammps", {}),
        "property_evaluators": config.get("property_evaluators", {}),
        "validation": config.get("validation", {}),
    }


def _run_or_reuse_structure(
    *,
    runner: LAMMPSRunner,
    candidate: Candidate,
    work_dir: Path,
    config: Mapping[str, Any],
    input_artifacts: Mapping[str, Path],
) -> tuple[dict[str, Any], Path | None]:
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = work_dir / "structure_result.json"
    manifest_path = work_dir / "stage_manifest.json"
    outputs = {"structure_result": result_path}
    scientific = _structure_scientific_config(config)
    identifier = f"material_validation:structure:{candidate.parameter_key}"
    if manifest_path.exists():
        decision = check_artifact_reuse(
            manifest_path,
            kind="candidate",
            identifier=identifier,
            parameters=candidate.raw_parameters,
            seeds=[],
            scientific_config=scientific,
            input_artifacts=input_artifacts,
            expected_outputs=outputs,
        )
        if not decision.reusable:
            raise MaterialValidationArtifactError(
                f"completed structural validation is not reusable: {decision.reason}"
            )
        return json.loads(result_path.read_text(encoding="ascii")), manifest_path

    try:
        evaluated = runner.evaluate_batch(
            [candidate.raw_parameters],
            str(work_dir / "lammps"),
            save_traj=True,
        )[0]
        document = {
            "success": bool(evaluated.success),
            "objective": float(evaluated.objective),
            "properties": dict(evaluated.properties),
            "per_property_error_percent": dict(evaluated.per_property_error),
            "error": str(evaluated.error_msg),
        }
    except Exception as exc:
        document = {
            "success": False,
            "objective": None,
            "properties": {},
            "per_property_error_percent": {},
            "error": str(exc),
        }
    _write_json(result_path, document)
    if not document["success"]:
        return document, None
    manifest = build_artifact_manifest(
        kind="candidate",
        identifier=identifier,
        parameters=candidate.raw_parameters,
        seeds=[],
        scientific_config=scientific,
        input_artifacts=input_artifacts,
        expected_outputs=outputs,
    )
    write_artifact_manifest(manifest_path, manifest)
    return document, manifest_path


def _candidate_csv(
    path: Path,
    candidate: Candidate,
    properties: Mapping[str, Any],
) -> None:
    row: dict[str, Any] = {"parameter_key": candidate.parameter_key}
    row.update(candidate.raw_parameters)
    row.update({f"calc_{name}": properties[name] for name in sorted(properties)})
    content = pd.DataFrame([row]).to_csv(index=False, lineterminator="\n").encode("utf-8")
    _write_immutable_bytes(path, content)


def _dynamic_required(config: Mapping[str, Any]) -> bool:
    elasticity = config.get("elasticity", {})
    modules = elasticity.get("modules", {}) if isinstance(elasticity, Mapping) else {}
    return bool(
        isinstance(elasticity, Mapping)
        and elasticity.get("enabled", False)
        and isinstance(modules, Mapping)
        and isinstance(modules.get("dynamic"), Mapping)
    )


def _terminal_parameter_outcome(
    parameters_path: str | os.PathLike[str] | None,
) -> dict[str, Any] | None:
    """Recognize a completed upstream scientific outcome with no candidate."""

    if parameters_path is None:
        return None
    source = Path(parameters_path).resolve()
    if source.suffix.lower() != ".json":
        return None
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    if not isinstance(document, Mapping):
        return None
    if (
        document.get("status") == "zero_hard_gate_eligible"
        and not document.get("raw_free_parameters")
    ):
        return {str(key): value for key, value in document.items()}
    return None


def _dynamic_validation_seed_plan(
    config: Mapping[str, Any],
    *,
    destination: Path,
    config_source: Path,
) -> tuple[Path, dict[str, Any]]:
    """Create an explicit holdout-seed config or report legacy seed reuse."""

    if not _dynamic_required(config):
        return config_source, {
            "status": "not_applicable",
            "promotion_seeds": [],
            "validation_seeds": [],
            "warnings": [],
        }
    dynamic = config["elasticity"]["modules"]["dynamic"]
    promotion_protocol = dict(dynamic.get("protocol", {}))
    promotion_seeds = [int(value) for value in promotion_protocol.get("seeds", [])]
    validation_protocol = dynamic.get("validation_protocol")
    holdout_seeds = (
        [int(value) for value in validation_protocol.get("seeds", [])]
        if isinstance(validation_protocol, Mapping)
        else []
    )
    if not holdout_seeds:
        return config_source, {
            "status": "reused_promotion_seed_set",
            "promotion_seeds": promotion_seeds,
            "validation_seeds": promotion_seeds,
            "warnings": [{
                "code": "reused_promotion_seed_set",
                "message": (
                    "No dynamic validation_seeds were compiled; final validation "
                    "reuses the deterministic promotion seeds and is not an "
                    "independent finite-temperature replicate."
                ),
            }],
        }

    runtime_protocol = deep_merge(promotion_protocol, dict(validation_protocol))
    runtime_protocol["seeds"] = holdout_seeds
    validation_config = deep_merge(
        dict(config),
        {"elasticity": {"modules": {"dynamic": {"protocol": runtime_protocol}}}},
    )
    validation_source = destination / "dynamic_validation_config.json"
    _write_immutable_bytes(validation_source, _json_bytes(validation_config))
    return validation_source, {
        "status": "independent_validation_seed_set",
        "promotion_seeds": promotion_seeds,
        "validation_seeds": holdout_seeds,
        "warnings": [],
    }


def _row_for_key(frame: pd.DataFrame, key: str) -> dict[str, Any] | None:
    if frame.empty or "parameter_key" not in frame:
        return None
    selected = frame.loc[frame["parameter_key"].astype(str) == key]
    if selected.empty:
        return None
    return {str(name): value for name, value in selected.iloc[0].items()}


def assess_material_validation(
    *,
    structural_result: Mapping[str, Any],
    structural_gate: Mapping[str, Any],
    static_row: Mapping[str, Any] | None,
    dynamic_row: Mapping[str, Any] | None,
    dynamic_required: bool,
    quality_tier_percent: float,
) -> dict[str, Any]:
    """Return accepted/best_effort/rejected without treating tier as a gate."""

    hard_gate_reasons: list[str] = []
    if not _truth(structural_result.get("success", False)):
        hard_gate_reasons.append("structural_calculation_failed")
    if not _truth(structural_gate.get("structural_gate_pass", False)):
        reason = str(structural_gate.get("structural_gate_reason", ""))
        hard_gate_reasons.append("structural_gate" + (f":{reason}" if reason else ""))

    def assess_elastic(row: Mapping[str, Any] | None, fidelity: str) -> None:
        if row is None:
            hard_gate_reasons.append(f"{fidelity}_elasticity_missing")
            return
        if str(row.get("calculation_status", "")) != "completed":
            hard_gate_reasons.append(f"{fidelity}_elasticity_incomplete")
        if not _truth(row.get("born_stability_pass", False)):
            hard_gate_reasons.append(f"{fidelity}_born_stability")
        if not _truth(row.get("fit_quality_pass", False)):
            hard_gate_reasons.append(f"{fidelity}_fit_quality")
        if not _truth(row.get("finite_mechanical_score", False)):
            hard_gate_reasons.append(f"{fidelity}_nonfinite_mechanics")
        if fidelity == "dynamic" and not _truth(row.get("all_seeds_complete", False)):
            hard_gate_reasons.append("dynamic_incomplete_seed_set")

    assess_elastic(static_row, "static")
    if dynamic_required:
        # A missing dynamic row after a static hard-gate rejection is expected,
        # but the final status remains rejected and records the missing evidence.
        assess_elastic(dynamic_row, "dynamic")

    evidence_rows = [row for row in (static_row, dynamic_row if dynamic_required else None) if row]
    mechanical_errors = [
        float(row["mechanical_max_error_percent"])
        for row in evidence_rows
        if _finite(row.get("mechanical_max_error_percent"))
    ]
    within_tier = bool(
        evidence_rows
        and len(mechanical_errors) == len(evidence_rows)
        and all(value <= quality_tier_percent + 1.0e-12 for value in mechanical_errors)
    )
    unique_reasons = list(dict.fromkeys(hard_gate_reasons))
    if unique_reasons:
        status = "rejected"
    elif within_tier:
        status = "accepted"
    else:
        status = "best_effort"
    return {
        "status": status,
        "hard_gate_pass": not unique_reasons,
        "hard_gate_reasons": unique_reasons,
        "quality_tier_percent": quality_tier_percent,
        "within_mechanical_quality_tier": within_tier,
        "mechanical_max_error_percent": max(mechanical_errors, default=None),
        "mechanical_quality": (
            "within_tier" if status == "accepted" else "best_effort" if status == "best_effort" else "ineligible"
        ),
    }


def _resolved_atom_rows(
    config: Mapping[str, Any], resolved: Mapping[str, float]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for atom_type in config.get("atom_types", []):
        label = str(atom_type["label"])
        rows.append({
            "type": int(atom_type["type"]),
            "label": label,
            "mass": float(atom_type["mass"]),
            "epsilon_kcal_mol": float(resolved[f"{label}_epsilon"]),
            "sigma_angstrom": float(resolved[f"{label}_sigma"]),
            "charge_e": float(resolved.get(f"{label}_charge", 0.0)),
        })
    return rows


def _mechanical_property_rows(
    *,
    fidelity: str,
    row: Mapping[str, Any] | None,
    module: Mapping[str, Any],
    evidence_file: Path,
) -> list[dict[str, Any]]:
    if row is None:
        return []
    target_names = {"B_gpa": "B", "Cprime_gpa": "Cprime", "C44_gpa": "C44"}
    properties = (
        "B_gpa",
        "Cprime_gpa",
        "C44_gpa",
        "C11_gpa",
        "C12_gpa",
        "G_hill_gpa",
        "E_hill_gpa",
        "nu_hill",
        "minimum_fit_r2",
    )
    result: list[dict[str, Any]] = []
    for name in properties:
        if not _finite(row.get(name)):
            continue
        target_name = target_names.get(name)
        target_spec = module.get("targets", {}).get(target_name, {}) if target_name else {}
        reference = target_spec.get("value") if isinstance(target_spec, Mapping) else None
        value = float(row[name])
        result.append({
            "fidelity": fidelity,
            "property": target_name or name.removesuffix("_gpa"),
            "value": value,
            "reference": reference,
            "unit": "" if name in {"nu_hill", "minimum_fit_r2"} else "GPa",
            "error_percent": (
                abs(float(row.get(f"error_{target_name}_percent")))
                if target_name and _finite(row.get(f"error_{target_name}_percent"))
                else None
            ),
            "absolute_error": (
                abs(value - float(reference)) if reference is not None else None
            ),
            "tolerance": None,
            "hard_gate_pass": _truth(row.get("finalist_eligible", False)),
            "within_quality_tier": _truth(row.get("within_quality_tier", False)),
            "evidence_file": str(evidence_file.resolve()),
        })
    return result


def _computed_property_rows(
    *,
    config: Mapping[str, Any],
    structural_result: Mapping[str, Any],
    structural_gate: Mapping[str, Any],
    static_row: Mapping[str, Any] | None,
    dynamic_row: Mapping[str, Any] | None,
    structure_results: Path,
    static_results: Path,
    dynamic_results: Path | None,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    errors = structural_result.get("per_property_error_percent", {})
    structural_properties = structural_result.get("properties", {})
    for name in sorted(structural_properties):
        raw_value = structural_properties[name]
        if not _finite(raw_value):
            continue
        target = config.get("targets", {}).get(name, {})
        reference = target.get("value") if isinstance(target, Mapping) else None
        tolerance = target.get("tolerance") if isinstance(target, Mapping) else None
        value = float(raw_value)
        rows.append({
            "fidelity": "structure_surface",
            "property": name,
            "value": value,
            "reference": reference,
            "unit": target.get("unit", "") if isinstance(target, Mapping) else "",
            "error_percent": errors.get(name),
            "absolute_error": abs(value - float(reference)) if reference is not None else None,
            "tolerance": tolerance,
            "hard_gate_pass": _truth(structural_gate["structural_gate_pass"]),
            "within_quality_tier": None,
            "evidence_file": str(structure_results.resolve()),
        })
    modules = config["elasticity"]["modules"]
    rows.extend(_mechanical_property_rows(
        fidelity="static_0k",
        row=static_row,
        module=modules["static"],
        evidence_file=static_results,
    ))
    if dynamic_results is not None and isinstance(modules.get("dynamic"), Mapping):
        rows.extend(_mechanical_property_rows(
            fidelity="dynamic_finite_temperature",
            row=dynamic_row,
            module=modules["dynamic"],
            evidence_file=dynamic_results,
        ))
    return rows


def _final_paths(output_dir: Path, *, include_top_report: bool) -> dict[str, Path]:
    outputs = {
        "validation_summary": output_dir / "validation_summary.json",
        "computed_properties": output_dir / "computed_properties.csv",
        "final_atom_parameters": output_dir / "final_atom_parameters.csv",
        "final_parameters_lammps": output_dir / "final_parameters.lammps",
        "final_parameters": output_dir / "final_parameters.json",
        "model_adequacy": output_dir / "model_adequacy.json",
    }
    if include_top_report:
        outputs.update({
            "top_parameters_csv": output_dir / "TOP_PARAMETERS.csv",
            "top_parameters_json": output_dir / "TOP_PARAMETERS.json",
            "top_parameters_markdown": output_dir / "TOP_PARAMETERS.md",
            "top_parameters_manifest": output_dir / "top_parameters_manifest.json",
        })
    return outputs


def _model_adequacy_document(
    *,
    config: Mapping[str, Any],
    candidate: Candidate,
    static_row: Mapping[str, Any] | None,
    dynamic_row: Mapping[str, Any] | None,
    static_results: Path,
    dynamic_results: Path | None,
) -> dict[str, Any]:
    elasticity: dict[str, Any] = {}
    if (
        static_row is not None
        and _finite(static_row.get("C12_gpa"))
        and _finite(static_row.get("C44_gpa"))
    ):
        elasticity["zero_k"] = {
            "C12_gpa": static_row.get("C12_gpa"),
            "C44_gpa": static_row.get("C44_gpa"),
            "temperature_k": 0.0,
        }
    if (
        dynamic_row is not None
        and _finite(dynamic_row.get("C12_gpa"))
        and _finite(dynamic_row.get("C44_gpa"))
    ):
        elasticity["finite_temperature"] = {
            "C12_gpa": dynamic_row.get("C12_gpa"),
            "C44_gpa": dynamic_row.get("C44_gpa"),
            "temperature_k": config["elasticity"]["modules"]["dynamic"][
                "protocol"
            ].get("temperature_k"),
        }
    report = assess_model_adequacy(
        config,
        elasticity_results=elasticity or None,
        label_swap_result={"status": "unknown"},
    )
    # The ordinary validation runner evaluates the supplied type assignment;
    # it cannot safely exchange permanent atom-type identities in every input
    # data file.  Record that missing experiment explicitly rather than
    # treating an unmeasured swap as evidence of transferability.
    label_swap = report["label_swap_diagnostic"]
    label_swap.update({
        "status": "not_evaluated",
        "reason": (
            "The standard LAMMPSRunner does not construct a content-audited "
            "same-element atom-type label-swapped data set."
        ),
        "required_followup": (
            "Generate an explicit label-swapped bulk/surface data set, rerun the "
            "same validation protocol, and compare energies and fitted properties."
        ),
    })
    for required in report.get("physical_transferability", {}).get(
        "required_validations", []
    ):
        if required.get("id") == "label_swap":
            required["status"] = "not_evaluated"
    report["validation_evidence"] = {
        "parameter_key": candidate.parameter_key,
        "static_results": str(static_results.resolve()),
        "dynamic_results": (
            str(dynamic_results.resolve()) if dynamic_results is not None else None
        ),
        "label_swap": "not_evaluated",
    }
    return report


def _final_scientific_config(
    config: Mapping[str, Any],
    *,
    include_top_report: bool,
    top_n: int,
    dynamic_seed_policy: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "material": config.get("material", {}),
        "crystal": config.get("crystal", {}),
        "targets": config.get("targets", {}),
        "elasticity": config.get("elasticity", {}),
        "validation": config.get("validation", {}),
        "top_parameters_report": {
            "enabled": include_top_report,
            "top_n": top_n if include_top_report else None,
        },
        "dynamic_seed_policy": dynamic_seed_policy,
    }


_COMPUTED_PROPERTY_COLUMNS = (
    "fidelity",
    "property",
    "value",
    "reference",
    "unit",
    "error_percent",
    "absolute_error",
    "tolerance",
    "hard_gate_pass",
    "within_quality_tier",
    "evidence_file",
)


def _publish_zero_eligible_validation(
    *,
    config_source: Path,
    config: Mapping[str, Any],
    parameter_source: Path,
    source_outcome: Mapping[str, Any],
    provenance: Mapping[str, Any],
    destination: Path,
    static_ranking_source: Path | None,
    dynamic_ranking_source: Path | None,
    top_n: int,
) -> dict[str, Any]:
    """Publish a successful pipeline terminal for a scientifically empty set."""

    include_top_report = static_ranking_source is not None
    outputs = _final_paths(destination, include_top_report=include_top_report)
    manifest_path = destination / "stage_manifest.json"
    seed_policy = {
        "status": "not_executed_no_hard_gate_eligible_candidate",
        "promotion_seeds": [],
        "validation_seeds": [],
        "warnings": [],
    }
    scientific = {
        **_final_scientific_config(
            config,
            include_top_report=include_top_report,
            top_n=top_n,
            dynamic_seed_policy=seed_policy,
        ),
        "terminal_upstream_outcome": "zero_hard_gate_eligible",
    }
    inputs: dict[str, Path] = {
        "compiled_config": config_source,
        "parameter_source": parameter_source,
    }
    if include_top_report:
        assert static_ranking_source is not None
        assert dynamic_ranking_source is not None
        inputs.update({
            "top_static_ranking": static_ranking_source,
            "top_dynamic_ranking": dynamic_ranking_source,
        })
    identifier = "material_validation:zero_hard_gate_eligible"
    if manifest_path.exists():
        decision = check_artifact_reuse(
            manifest_path,
            kind="stage",
            identifier=identifier,
            parameters=None,
            seeds=[],
            scientific_config=scientific,
            input_artifacts=inputs,
            expected_outputs=outputs,
        )
        if not decision.reusable:
            raise MaterialValidationArtifactError(
                "completed zero-eligible validation is not reusable: "
                f"{decision.reason}"
            )
        return json.loads(outputs["validation_summary"].read_text(encoding="ascii"))

    top_summary: dict[str, Any] = {"enabled": False}
    if include_top_report:
        from .material_results_report import write_top_parameters_report

        assert static_ranking_source is not None
        assert dynamic_ranking_source is not None
        top_frame = write_top_parameters_report(
            config_path=config_source,
            static_ranking_path=static_ranking_source,
            dynamic_ranking_path=dynamic_ranking_source,
            output_dir=destination,
            top_n=top_n,
            manifest_name="top_parameters_manifest.json",
        )
        top_summary = {
            "enabled": True,
            "reported": len(top_frame),
            "requested_top_n": top_n,
            "static_ranking": str(static_ranking_source),
            "dynamic_ranking": str(dynamic_ranking_source),
            "csv": str(outputs["top_parameters_csv"].resolve()),
            "json": str(outputs["top_parameters_json"].resolve()),
            "markdown": str(outputs["top_parameters_markdown"].resolve()),
            "manifest": str(outputs["top_parameters_manifest"].resolve()),
        }

    atom_rows = [
        {
            "type": int(atom_type["type"]),
            "label": str(atom_type["label"]),
            "mass": float(atom_type["mass"]),
            "epsilon_kcal_mol": None,
            "sigma_angstrom": None,
            "charge_e": None,
            "parameter_status": "unavailable_no_eligible_candidate",
        }
        for atom_type in config.get("atom_types", [])
    ]
    _write_frame(outputs["final_atom_parameters"], pd.DataFrame(atom_rows))
    _atomic_bytes(
        outputs["final_parameters_lammps"],
        (
            "# FFOpt did not select a hard-gate-eligible force field.\n"
            "# Intentionally contains no pair_coeff commands; do not use for simulation.\n"
        ).encode("ascii"),
    )
    _write_json(outputs["final_parameters"], {
        "schema_version": 1,
        "status": "rejected",
        "model_status": "model_inadequate",
        "parameter_availability": "none",
        "parameter_key": None,
        "source": provenance,
        "upstream_outcome": source_outcome,
        "raw_free_parameters": {},
        "resolved_parameters": {},
        "atom_types": atom_rows,
        "force_field_artifact": None,
    })
    _write_frame(
        outputs["computed_properties"],
        pd.DataFrame(columns=_COMPUTED_PROPERTY_COLUMNS),
    )
    adequacy = assess_model_adequacy(
        config,
        elasticity_results=None,
        label_swap_result={"status": "unknown"},
    )
    adequacy["candidate_selection"] = {
        "status": "model_inadequate",
        "reason": "zero_hard_gate_eligible",
        "message": source_outcome.get("message"),
        "parameters_available": False,
    }
    adequacy["validation_evidence"] = {
        "parameter_key": None,
        "static_results": None,
        "dynamic_results": None,
        "label_swap": "not_evaluated_no_candidate",
    }
    _write_json(outputs["model_adequacy"], adequacy)
    summary = {
        "schema_version": 1,
        "status": "rejected",
        "model_status": "model_inadequate",
        "hard_gate_pass": False,
        "hard_gate_reasons": ["zero_hard_gate_eligible_finalists"],
        "quality_tier_percent": float(
            config["elasticity"]["reporting"]["mechanical_tier_percent"]
        ),
        "within_mechanical_quality_tier": False,
        "mechanical_max_error_percent": None,
        "mechanical_quality": "ineligible",
        "parameter_key": None,
        "config": str(config_source),
        "parameter_source": provenance,
        "upstream_outcome": source_outcome,
        "structural_evidence": {"status": "not_executed_no_candidate"},
        "static_elastic_evidence": {"status": "not_executed_no_candidate"},
        "dynamic_elastic_evidence": {
            "required": _dynamic_required(config),
            "status": "not_executed_no_candidate",
            "seed_policy": seed_policy,
        },
        "model_adequacy": {
            "status": "model_inadequate",
            "report": str(outputs["model_adequacy"].resolve()),
            "physical_transferability_status": adequacy[
                "physical_transferability"
            ]["status"],
            "label_swap_status": "not_evaluated_no_candidate",
        },
        "top_parameters_report": top_summary,
        "artifacts": {name: str(path.resolve()) for name, path in outputs.items()},
        "execution_complete": True,
    }
    _write_json(outputs["validation_summary"], summary)
    manifest = build_artifact_manifest(
        kind="stage",
        identifier=identifier,
        parameters=None,
        seeds=[],
        scientific_config=scientific,
        input_artifacts=inputs,
        expected_outputs=outputs,
    )
    write_artifact_manifest(manifest_path, manifest)
    return _json_safe(summary)


def run_material_validation(
    *,
    config_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    resources: NestedResourcePlan,
    parameters_path: str | os.PathLike[str] | None = None,
    parameters: Mapping[str, float] | None = None,
    static_ranking_path: str | os.PathLike[str] | None = None,
    dynamic_ranking_path: str | os.PathLike[str] | None = None,
    top_n: int = 20,
    runner_factory: RunnerFactory = LAMMPSRunner,
    elastic_batch_runner: ElasticBatchRunner = run_elasticity_batch,
    elastic_backend_factory: Callable[..., Any] = default_backend_factory,
) -> dict[str, Any]:
    """Run or safely resume complete validation for one explicit parameter set."""

    config_source = Path(config_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    raw_config = load_config(config_source)
    config = _validation_config(raw_config)
    if (parameters_path is None) == (parameters is None):
        raise ValueError("provide exactly one of parameters_path or parameters")
    if (static_ranking_path is None) != (dynamic_ranking_path is None):
        raise ValueError(
            "static_ranking_path and dynamic_ranking_path must be supplied together"
        )
    include_top_report = static_ranking_path is not None
    static_ranking_source = (
        Path(static_ranking_path).resolve() if static_ranking_path is not None else None
    )
    dynamic_ranking_source = (
        Path(dynamic_ranking_path).resolve() if dynamic_ranking_path is not None else None
    )
    if include_top_report and top_n < 1:
        raise ValueError("top_n must be positive")
    terminal_outcome = _terminal_parameter_outcome(parameters_path)
    if terminal_outcome is not None:
        assert parameters_path is not None
        parameter_source = Path(parameters_path).resolve()
        provenance = {
            "kind": "upstream_scientific_outcome",
            "path": str(parameter_source),
            "artifact": sha256_file(parameter_source).to_dict(),
            "status": "zero_hard_gate_eligible",
        }
        return _publish_zero_eligible_validation(
            config_source=config_source,
            config=config,
            parameter_source=parameter_source,
            source_outcome=terminal_outcome,
            provenance=provenance,
            destination=destination,
            static_ranking_source=static_ranking_source,
            dynamic_ranking_source=dynamic_ranking_source,
            top_n=top_n,
        )

    raw_parameters, parameter_source, provenance = _load_parameter_source(
        output_dir=destination,
        parameters_path=parameters_path,
        parameters=parameters,
    )
    candidate = _validated_candidate(raw_parameters, config)
    dynamic_needed = _dynamic_required(config)
    final_outputs = _final_paths(
        destination, include_top_report=include_top_report
    )
    final_manifest = destination / "stage_manifest.json"
    dynamic_config_source, dynamic_seed_policy = _dynamic_validation_seed_plan(
        config,
        destination=destination,
        config_source=config_source,
    )

    # If a completed stage exists, reject any changed top-level artifact before
    # rewriting reports.  Substage runners below then perform their deeper
    # scientific identity and state-manifest checks.
    static_dir = destination / "elasticity_static"
    dynamic_dir = destination / "elasticity_dynamic"
    structure_dir = destination / "candidate_runs" / candidate.digest / "final_validation" / "structure"
    candidate_csv = destination / "elastic_candidate.csv"

    def final_inputs() -> dict[str, Path]:
        inputs = {
            "compiled_config": config_source,
            "parameter_source": parameter_source,
            "elastic_candidate": candidate_csv,
            "structure_result": structure_dir / "structure_result.json",
            "structure_manifest": structure_dir / "stage_manifest.json",
            "static_results": static_dir / "static_results.csv",
            "static_manifest": static_dir / "stage_manifest.json",
        }
        if dynamic_needed:
            inputs.update({
                "dynamic_validation_config": dynamic_config_source,
                "dynamic_results": dynamic_dir / "dynamic_results.csv",
                "dynamic_seed_results": dynamic_dir / "dynamic_seed_results.csv",
                "dynamic_manifest": dynamic_dir / "stage_manifest.json",
            })
        if include_top_report:
            assert static_ranking_source is not None
            assert dynamic_ranking_source is not None
            inputs.update({
                "top_static_ranking": static_ranking_source,
                "top_dynamic_ranking": dynamic_ranking_source,
            })
        return inputs

    if final_manifest.exists():
        validation = validate_artifact_manifest(
            final_manifest,
            input_artifacts=final_inputs(),
            expected_outputs=final_outputs,
        )
        if not validation.valid:
            raise MaterialValidationArtifactError(
                f"completed final validation artifacts changed: {validation.reason}"
            )

    runner = runner_factory(config)
    prepared = prepare_candidate(
        candidate,
        output_dir=destination,
        parameter_runner=runner,
        config=config,
    )
    resolved_identity = (
        destination / "candidate_runs" / candidate.digest / "inputs" / "resolved_parameters.json"
    )
    structural_inputs = _runner_input_artifacts(
        runner,
        config_path=config_source,
        parameter_source=parameter_source,
        resolved_parameters=resolved_identity,
        force_field_include=prepared.force_field_include,
    )
    structural_result, structural_manifest = _run_or_reuse_structure(
        runner=runner,
        candidate=candidate,
        work_dir=structure_dir,
        config=config,
        input_artifacts=structural_inputs,
    )
    structural_gate = assess_structural_gates(
        structural_result.get("properties", {}), config
    )
    _candidate_csv(candidate_csv, candidate, structural_result.get("properties", {}))

    static_frame = elastic_batch_runner(
        config_path=config_source,
        parameters_path=candidate_csv,
        output_dir=static_dir,
        protocol="static",
        resources=resources,
        top_n=1,
        near_optimal_window_percent=0.0,
        diversity_slots=0,
        evaluate_structural_failures=True,
        backend_factory=elastic_backend_factory,
    )
    static_row = _row_for_key(static_frame, candidate.parameter_key)

    dynamic_frame = pd.DataFrame()
    if dynamic_needed:
        dynamic_frame = elastic_batch_runner(
            config_path=dynamic_config_source,
            parameters_path=static_dir / "static_results.csv",
            output_dir=dynamic_dir,
            protocol="dynamic",
            resources=resources,
            top_n=1,
            near_optimal_window_percent=0.0,
            diversity_slots=0,
            backend_factory=elastic_backend_factory,
        )
    dynamic_row = _row_for_key(dynamic_frame, candidate.parameter_key)
    quality_tier = float(config["elasticity"]["reporting"]["mechanical_tier_percent"])
    assessment = assess_material_validation(
        structural_result=structural_result,
        structural_gate=structural_gate,
        static_row=static_row,
        dynamic_row=dynamic_row,
        dynamic_required=dynamic_needed,
        quality_tier_percent=quality_tier,
    )

    atom_rows = _resolved_atom_rows(config, prepared.resolved_parameters)
    _write_frame(final_outputs["final_atom_parameters"], pd.DataFrame(atom_rows))
    pair_text = prepared.force_field_include.read_text(encoding="ascii").replace(
        "# pair_coeffs.lmp -- generated by FFOpt-LAMMPS\n"
        "# DO NOT EDIT: overwritten before each LAMMPS call\n",
        "# Complete FFOpt final parameters; include after read_data.\n",
        1,
    )
    _atomic_bytes(final_outputs["final_parameters_lammps"], pair_text.encode("ascii"))

    parameter_document = {
        "schema_version": 1,
        "status": assessment["status"],
        "parameter_key": candidate.parameter_key,
        "source": provenance,
        "raw_free_parameters": candidate.raw_parameters,
        "resolved_parameters": prepared.resolved_parameters,
        "atom_types": atom_rows,
        "force_field_artifact": sha256_file(prepared.force_field_include).to_dict(),
    }
    _write_json(final_outputs["final_parameters"], parameter_document)

    static_results = static_dir / "static_results.csv"
    dynamic_results = dynamic_dir / "dynamic_results.csv" if dynamic_needed else None
    computed_rows = _computed_property_rows(
        config=config,
        structural_result=structural_result,
        structural_gate=structural_gate,
        static_row=static_row,
        dynamic_row=dynamic_row,
        structure_results=structure_dir / "structure_result.json",
        static_results=static_results,
        dynamic_results=dynamic_results,
    )
    _write_frame(final_outputs["computed_properties"], pd.DataFrame(computed_rows))
    adequacy = _model_adequacy_document(
        config=config,
        candidate=candidate,
        static_row=static_row,
        dynamic_row=dynamic_row,
        static_results=static_results,
        dynamic_results=dynamic_results,
    )
    _write_json(final_outputs["model_adequacy"], adequacy)
    top_report_summary: dict[str, Any] = {"enabled": False}
    if include_top_report:
        # Local import avoids a module cycle: the standalone report reuses the
        # atomic report writers defined above.
        from .material_results_report import write_top_parameters_report

        assert static_ranking_source is not None
        assert dynamic_ranking_source is not None
        top_frame = write_top_parameters_report(
            config_path=config_source,
            static_ranking_path=static_ranking_source,
            dynamic_ranking_path=dynamic_ranking_source,
            output_dir=destination,
            top_n=top_n,
            manifest_name="top_parameters_manifest.json",
        )
        top_report_summary = {
            "enabled": True,
            "reported": len(top_frame),
            "requested_top_n": top_n,
            "static_ranking": str(static_ranking_source),
            "dynamic_ranking": str(dynamic_ranking_source),
            "csv": str(final_outputs["top_parameters_csv"].resolve()),
            "json": str(final_outputs["top_parameters_json"].resolve()),
            "markdown": str(final_outputs["top_parameters_markdown"].resolve()),
            "manifest": str(final_outputs["top_parameters_manifest"].resolve()),
        }

    summary = {
        "schema_version": 1,
        **assessment,
        "parameter_key": candidate.parameter_key,
        "config": str(config_source),
        "parameter_source": provenance,
        "structural_evidence": {
            "result": str((structure_dir / "structure_result.json").resolve()),
            "manifest": str(structural_manifest.resolve()) if structural_manifest else None,
            "objective": structural_result.get("objective"),
            "gate": structural_gate,
        },
        "static_elastic_evidence": {
            "results": str(static_results.resolve()),
            "manifest": str((static_dir / "stage_manifest.json").resolve()),
            "row": static_row,
        },
        "dynamic_elastic_evidence": (
            {
                "required": True,
                "results": str(dynamic_results.resolve()),
                "seed_results": str((dynamic_dir / "dynamic_seed_results.csv").resolve()),
                "manifest": str((dynamic_dir / "stage_manifest.json").resolve()),
                "row": dynamic_row,
                "seed_policy": dynamic_seed_policy,
            }
            if dynamic_needed
            else {"required": False, "seed_policy": dynamic_seed_policy}
        ),
        "model_adequacy": {
            "report": str(final_outputs["model_adequacy"].resolve()),
            "physical_transferability_status": adequacy[
                "physical_transferability"
            ]["status"],
            "label_swap_status": adequacy["label_swap_diagnostic"]["status"],
        },
        "top_parameters_report": top_report_summary,
        "artifacts": {name: str(path.resolve()) for name, path in final_outputs.items()},
    }
    _write_json(final_outputs["validation_summary"], summary)

    required_manifests = [
        structural_manifest,
        static_dir / "stage_manifest.json",
        dynamic_dir / "stage_manifest.json" if dynamic_needed else None,
    ]
    execution_complete = all(path is not None and Path(path).is_file() for path in required_manifests if path is not None) and structural_manifest is not None
    summary["execution_complete"] = execution_complete
    # Rewrite once with the final execution-completeness field included.
    _write_json(final_outputs["validation_summary"], summary)
    if not execution_complete:
        return _json_safe(summary)

    seeds = list(dynamic_seed_policy["validation_seeds"])
    scientific = _final_scientific_config(
        config,
        include_top_report=include_top_report,
        top_n=top_n,
        dynamic_seed_policy=dynamic_seed_policy,
    )
    manifest = build_artifact_manifest(
        kind="candidate",
        identifier=f"material_validation:{candidate.parameter_key}",
        parameters=candidate.raw_parameters,
        seeds=seeds,
        scientific_config=scientific,
        input_artifacts=final_inputs(),
        expected_outputs=final_outputs,
    )
    if final_manifest.exists():
        decision = check_artifact_reuse(
            final_manifest,
            kind="candidate",
            identifier=f"material_validation:{candidate.parameter_key}",
            parameters=candidate.raw_parameters,
            seeds=seeds,
            scientific_config=scientific,
            input_artifacts=final_inputs(),
            expected_outputs=final_outputs,
        )
        if not decision.reusable:
            raise MaterialValidationArtifactError(
                f"completed final validation is not reusable: {decision.reason}"
            )
    else:
        write_artifact_manifest(final_manifest, manifest)
    return _json_safe(summary)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run complete structure/surface and cubic final validation"
    )
    parser.add_argument("--config", required=True, type=Path)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--parameters", type=Path)
    source.add_argument(
        "--parameter",
        action="append",
        default=[],
        metavar="NAME=VALUE",
        help="Explicit free parameter; repeat once per compiled free dimension",
    )
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--static-ranking", type=Path)
    parser.add_argument("--dynamic-ranking", type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    parser.add_argument("--available-cores", required=True, type=int)
    parser.add_argument("--cores-per-state", required=True, type=int)
    parser.add_argument("--candidate-workers", type=int, default=1)
    parser.add_argument("--state-workers", type=int)
    parser.add_argument("--omp-threads-per-state", type=int, default=1)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    explicit = _parse_assignments(arguments.parameter) if arguments.parameter else None
    resources = plan_nested_resources(
        available_cores=effective_available_cores(arguments.available_cores),
        cores_per_state=arguments.cores_per_state,
        candidate_workers=arguments.candidate_workers,
        state_workers=arguments.state_workers,
        omp_threads_per_rank=arguments.omp_threads_per_state,
    )
    summary = run_material_validation(
        config_path=arguments.config,
        parameters_path=arguments.parameters,
        parameters=explicit,
        static_ranking_path=arguments.static_ranking,
        dynamic_ranking_path=arguments.dynamic_ranking,
        top_n=arguments.top_n,
        output_dir=arguments.output_dir,
        resources=resources,
    )
    print(json.dumps({
        "status": summary["status"],
        "hard_gate_pass": summary["hard_gate_pass"],
        "execution_complete": summary.get("execution_complete", False),
        "output_dir": str(arguments.output_dir.resolve()),
    }, sort_keys=True))
    return 0 if summary.get("execution_complete", False) else 2


__all__ = [
    "MaterialValidationArtifactError",
    "MaterialValidationError",
    "assess_material_validation",
    "build_parser",
    "main",
    "run_material_validation",
]


if __name__ == "__main__":
    raise SystemExit(main())
