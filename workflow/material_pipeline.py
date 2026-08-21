"""Material-workflow adapters shared by the pipeline runner.

The functions here are intentionally deterministic and path-explicit.  They
derive the internal constrained-refinement contract from the compiled material
configuration and publish traceable skipped AL rounds after an earlier round
has reached a terminal scientific state.  They do not discover run folders or
submit jobs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Mapping

import pandas as pd

from engine.parameter_space import build_parameter_space
from workflow.artifact_manifest import (
    ArtifactManifestError,
    build_artifact_manifest,
    check_artifact_reuse,
    load_artifact_manifest,
    sha256_file,
    write_artifact_manifest,
)


MATERIAL_EXECUTABLE_KINDS = frozenset({
    "candidates",
    "static",
    "constrained_al",
    "finalists",
})
TERMINAL_REFINEMENT_STATUSES = frozenset({"converged", "budget_exhausted"})
REFINEMENT_OUTPUTS = {
    "state": "refinement_state.json",
    "ranking": "exact_structural_mechanical_ranking.csv",
    "structural_proposals": "structural_proposals.csv",
    "mechanical_proposals": "mechanical_proposals.csv",
    "report": "refinement_report.md",
    "disposition": "stage_disposition.json",
    "round_evidence": "round_evidence.json",
}
_MATERIAL_AL_OUTPUT_FILES = {
    **REFINEMENT_OUTPUTS,
    "assessed": "assessed_candidates.csv",
    "core_manifest": "artifact_manifest.json",
    "cumulative_structural": "cumulative_structural_observations.csv",
    "cumulative_static": "cumulative_static_observations.csv",
    "candidate_pool": "candidate_pool.csv",
    "planned_structural": "planned_structural_proposals.csv",
    "new_structural": "new_structural_observations.csv",
    "executed_static": "executed_static_proposals.csv",
    "new_static": "new_static_observations.csv",
    "static_completion": "static_batch_completion.json",
}


def validate_material_al_stage_outputs(
    directory: str | os.PathLike[str],
) -> tuple[bool, str]:
    """Validate the outer AL manifest and every output hash it declares."""

    root = Path(directory).resolve()
    manifest_path = root / "material_al_manifest.json"
    try:
        manifest = load_artifact_manifest(manifest_path)
    except ArtifactManifestError as exc:
        return False, str(exc)
    if manifest.kind != "stage" or not manifest.identifier.startswith(
        "material_al_round:"
    ):
        return False, "material AL manifest has the wrong stage identity"
    for label, expected in manifest.expected_outputs:
        filename = _MATERIAL_AL_OUTPUT_FILES.get(label)
        if filename is None:
            return False, f"material AL manifest declares unknown output {label!r}"
        try:
            actual = sha256_file(root / filename)
        except ArtifactManifestError as exc:
            return False, str(exc)
        if actual != expected:
            return False, f"material AL output hash mismatch for {filename}"
    return True, "material AL manifest and output hashes verified"


def _target_value(config: Mapping[str, Any], name: str) -> float:
    try:
        raw = config["targets"][name]
    except (KeyError, TypeError) as exc:
        raise ValueError(
            f"Elasticity structural gate needs compiled target {name!r}"
        ) from exc
    value = raw.get("value") if isinstance(raw, Mapping) else raw
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Compiled target {name!r} must be numeric") from exc


def build_refinement_spec(
    config: Mapping[str, Any],
    *,
    maximum_rounds: int,
    settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Translate compiled elasticity metadata into the strict round spec."""
    elasticity = config.get("elasticity")
    if not isinstance(elasticity, Mapping) or not elasticity.get("enabled"):
        raise ValueError("Material constrained refinement requires enabled elasticity")
    selection = elasticity.get("selection", {})
    if not isinstance(selection, Mapping):
        raise ValueError("elasticity.selection must be a mapping")
    gates = selection.get("structural_gates", {})
    if not isinstance(gates, Mapping):
        raise ValueError("elasticity.selection.structural_gates must be a mapping")
    constraints: list[dict[str, Any]] = []

    def add_relative(group: str, names: tuple[str, ...]) -> None:
        gate = gates.get(group)
        if gate is None:
            return
        if not isinstance(gate, Mapping):
            raise ValueError(f"elasticity structural gate {group!r} must be a mapping")
        tolerance = float(gate["maximum_relative_error_percent"])
        for name in names:
            constraints.append({
                "name": name,
                "column": f"calc_{name}",
                "target": _target_value(config, name),
                "tolerance": tolerance,
                "mode": "relative_percent",
                "role": "constraint",
            })

    add_relative("lattice", ("a", "b", "c"))
    angle_gate = gates.get("angles")
    if angle_gate is not None:
        if not isinstance(angle_gate, Mapping):
            raise ValueError("elasticity structural gate 'angles' must be a mapping")
        tolerance = float(angle_gate["maximum_absolute_error_degree"])
        for name in ("alpha", "beta", "gamma_ang"):
            constraints.append({
                "name": name,
                "column": f"calc_{name}",
                "target": _target_value(config, name),
                "tolerance": tolerance,
                "mode": "absolute",
                "role": "constraint",
            })
    add_relative("density", ("density",))
    add_relative("surface", ("surf_energy",))
    if not constraints:
        raise ValueError(
            "At least one elasticity structural gate is required for constrained refinement"
        )

    modules = elasticity.get("modules", {})
    static = modules.get("static") if isinstance(modules, Mapping) else None
    if not isinstance(static, Mapping):
        raise ValueError("elasticity.modules.static is required")
    targets = static.get("targets", {})
    if not isinstance(targets, Mapping):
        raise ValueError("elasticity.modules.static.targets must be a mapping")
    objectives = []
    for name in ("B", "Cprime", "C44"):
        try:
            raw = targets[name]
        except KeyError as exc:
            raise ValueError(f"Static elasticity target {name!r} is required") from exc
        value = raw.get("value") if isinstance(raw, Mapping) else raw
        objectives.append({
            "name": name,
            "column": f"{name}_gpa",
            "target": float(value),
            "role": "objective",
        })

    active_learning = config.get("active_learning", {})
    if not isinstance(active_learning, Mapping):
        active_learning = {}
    early_stop = active_learning.get("early_stop", {})
    if not isinstance(early_stop, Mapping):
        early_stop = {}
    runtime = {
        "patience": early_stop.get("patience", 3),
        "minimum_improvement_percent_points": early_stop.get(
            "min_improvement", 0.5
        ),
        "mechanical_proposals_per_round": active_learning.get(
            "static_points", 20
        ),
        "structural_proposals_per_round": active_learning.get(
            "n_candidates_per_round", 20
        ),
        **dict(settings or {}),
    }
    fit_quality = selection.get("fit_quality", {})
    born = selection.get("born_stability", {})
    reporting = elasticity.get("reporting", {})
    parameter_names = [item[0] for item in build_parameter_space(dict(config))]
    return {
        "schema": "ffopt-constrained-refinement-v1",
        "parameter_names": parameter_names,
        "structural_constraints": constraints,
        "mechanical_objectives": objectives,
        "maximum_rounds": int(maximum_rounds),
        "patience": int(runtime.get("patience", 3)),
        "minimum_improvement_percent_points": float(
            runtime.get("minimum_improvement_percent_points", 0.5)
        ),
        "mechanical_proposals_per_round": int(
            runtime.get("mechanical_proposals_per_round", 20)
        ),
        "structural_proposals_per_round": int(
            runtime.get("structural_proposals_per_round", 20)
        ),
        "report_threshold_percent": float(
            reporting.get("mechanical_tier_percent", 20.0)
        ),
        "require_stability": bool(
            born.get("required", True) if isinstance(born, Mapping) else True
        ),
        "stability_column": "born_stability_pass",
        "minimum_fit_quality": float(
            fit_quality.get("minimum_r2", 0.98)
            if isinstance(fit_quality, Mapping)
            else 0.98
        ),
        "fit_quality_column": "minimum_fit_r2",
    }


def load_refinement_state(path: str | os.PathLike[str]) -> dict[str, Any]:
    """Read one explicitly named refinement state and validate its status."""
    source = Path(path).resolve()
    try:
        raw = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read refinement state {source}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError(f"Refinement state {source} must be a JSON object")
    status = raw.get("status")
    if status not in {"active", *TERMINAL_REFINEMENT_STATUSES}:
        raise ValueError(f"Refinement state {source} has invalid status {status!r}")
    return raw


def refinement_is_terminal(path: str | os.PathLike[str]) -> bool:
    return load_refinement_state(path).get("status") in TERMINAL_REFINEMENT_STATUSES


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _paths(directory: Path) -> dict[str, Path]:
    return {label: directory / name for label, name in REFINEMENT_OUTPUTS.items()}


def write_skipped_refinement_stage(
    *,
    stage_name: str,
    round_number: int,
    previous_stage_name: str,
    previous_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    config_path: str | os.PathLike[str],
    scientific_config: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    """Publish an immutable skipped round chained to a terminal predecessor."""
    previous = Path(previous_dir).resolve()
    destination = Path(output_dir).resolve()
    prior_paths = _paths(previous)
    prior_manifest = previous / "artifact_manifest.json"
    required_inputs = {
        "config": Path(config_path).resolve(),
        "previous_state": prior_paths["state"],
        "previous_ranking": prior_paths["ranking"],
        "previous_round_evidence": prior_paths["round_evidence"],
        "previous_manifest": prior_manifest,
    }
    previous_state = load_refinement_state(prior_paths["state"])
    previous_status = str(previous_state["status"])
    if previous_status not in TERMINAL_REFINEMENT_STATUSES:
        raise ValueError(
            f"Cannot skip {stage_name}: predecessor status is {previous_status!r}"
        )
    identity = {
        "action": "skip_terminal_refinement_round",
        "stage": stage_name,
        "round": int(round_number),
        "previous_stage": previous_stage_name,
        "terminal_status": previous_status,
        "scientific_config": dict(scientific_config),
    }
    outputs = _paths(destination)
    manifest_path = destination / "artifact_manifest.json"
    material_manifest_path = destination / "material_al_manifest.json"
    if destination.exists():
        decision = check_artifact_reuse(
            material_manifest_path,
            kind="stage",
            identifier=f"material_al_round:{round_number:04d}",
            parameters=None,
            seeds=[int(seed)],
            scientific_config=identity,
            input_artifacts=required_inputs,
            expected_outputs={**outputs, "core_manifest": manifest_path},
        )
        if decision.reusable:
            return load_refinement_state(outputs["state"])
        raise RuntimeError(
            f"Skipped stage output cannot be reused safely: {destination}: "
            f"{decision.reason}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    ))
    try:
        staged = _paths(staging)
        state = {
            **previous_state,
            "pipeline_stage": stage_name,
            "stage_disposition": "skipped",
            "skipped_round": int(round_number),
            "skipped_from_stage": previous_stage_name,
            "skip_reason": f"previous refinement status is {previous_status}",
            "terminal_state_source": str(prior_paths["state"]),
            "terminal_ranking_source": str(prior_paths["ranking"]),
        }
        _write_json(staged["state"], state)
        shutil.copyfile(prior_paths["ranking"], staged["ranking"])
        for label in ("structural_proposals", "mechanical_proposals"):
            prior = pd.read_csv(prior_paths[label], nrows=0)
            prior.to_csv(staged[label], index=False)
        staged["report"].write_text(
            "# Skipped constrained-refinement round\n\n"
            f"- Stage: {stage_name}\n"
            f"- Disposition: skipped\n"
            f"- Reason: previous stage `{previous_stage_name}` ended with "
            f"`{previous_status}`.\n"
            f"- Terminal state: `{prior_paths['state']}`\n",
            encoding="utf-8",
        )
        _write_json(staged["disposition"], {
            "schema": "ffopt-stage-disposition-v1",
            "stage": stage_name,
            "disposition": "skipped",
            "round": int(round_number),
            "previous_stage": previous_stage_name,
            "reason": previous_status,
        })
        previous_evidence = json.loads(
            prior_paths["round_evidence"].read_text(encoding="utf-8")
        )
        if not isinstance(previous_evidence, dict):
            raise ValueError("Previous round_evidence.json must be an object")
        if previous_evidence.get("schema") != "ffopt-material-al-round-evidence-v1":
            raise ValueError("Previous round_evidence.json has an unsupported schema")

        def relink(record: Any) -> Any:
            if not isinstance(record, dict) or not isinstance(record.get("path"), str):
                return record
            source_path = (previous / record["path"]).resolve()
            return {
                **record,
                "path": os.path.relpath(source_path, destination).replace("\\", "/"),
            }

        carried = dict(previous_evidence)
        carried.update({
            "disposition": "skipped",
            "round": int(round_number),
            "previous_round": previous_evidence.get("round"),
            "skip_reason": previous_status,
        })
        for key in (
            "cumulative_structural",
            "cumulative_static",
            "new_structural",
            "new_static",
            "static_batch_manifest",
        ):
            if carried.get(key) is not None:
                carried[key] = relink(carried[key])
        carried["candidate_manifests"] = [
            relink(record) for record in carried.get("candidate_manifests", [])
        ]
        _write_json(staged["round_evidence"], carried)
        manifest = build_artifact_manifest(
            kind="stage",
            identifier=stage_name,
            parameters=None,
            seeds=[int(seed)],
            scientific_config=identity,
            input_artifacts=required_inputs,
            expected_outputs=staged,
        )
        write_artifact_manifest(staging / "artifact_manifest.json", manifest)
        material_manifest = build_artifact_manifest(
            kind="stage",
            identifier=f"material_al_round:{round_number:04d}",
            parameters=None,
            seeds=[int(seed)],
            scientific_config=identity,
            input_artifacts=required_inputs,
            expected_outputs={
                **staged,
                "core_manifest": staging / "artifact_manifest.json",
            },
        )
        write_artifact_manifest(
            staging / "material_al_manifest.json", material_manifest
        )
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return state


__all__ = [
    "MATERIAL_EXECUTABLE_KINDS",
    "REFINEMENT_OUTPUTS",
    "TERMINAL_REFINEMENT_STATUSES",
    "build_refinement_spec",
    "load_refinement_state",
    "refinement_is_terminal",
    "validate_material_al_stage_outputs",
    "write_skipped_refinement_stage",
]
