"""CPU-only constrained-GP preflight for material workflows."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any

from .config_loader import load_config
from .constrained_refinement import AcquisitionContext, RefinementSpec
from .cubic_elastic_batch import assess_structural_gates
from .lammps_interface import LAMMPSRunner
from .material_al_round import (
    _candidate_pool,
    _gp_backend,
    _read_frames,
)
from .parameter_space import build_parameter_space
from workflow.artifact_manifest import (
    build_artifact_manifest,
    check_artifact_reuse,
    write_artifact_manifest,
)


_OUTPUTS = {
    "diagnostics": "surrogate_diagnostics.json",
    "proposals": "initial_proposals.csv",
    "candidate_pool": "candidate_pool.csv",
}


def run_material_surrogate_preflight(
    *,
    runtime_config_path: str | os.PathLike[str],
    refinement_config_path: str | os.PathLike[str],
    structural_paths: Sequence[str | os.PathLike[str]],
    mechanical_paths: Sequence[str | os.PathLike[str]],
    candidate_pool_paths: Sequence[str | os.PathLike[str]],
    output_dir: str | os.PathLike[str],
    seed: int = 20260820,
) -> dict[str, Any]:
    """Fit/probe the exact-evidence GP without claiming or running AL labels."""

    runtime_source = Path(runtime_config_path).resolve()
    refinement_source = Path(refinement_config_path).resolve()
    structural_sources = tuple(Path(path).resolve() for path in structural_paths)
    mechanical_sources = tuple(Path(path).resolve() for path in mechanical_paths)
    external_pools = tuple(Path(path).resolve() for path in candidate_pool_paths)
    if not structural_sources:
        raise ValueError("surrogate preflight requires explicit structural evidence")
    config = load_config(runtime_source)
    refinement_document = load_config(refinement_source)
    raw_spec = refinement_document.get("constrained_refinement", refinement_document)
    if not isinstance(raw_spec, Mapping):
        raise ValueError("refinement config must contain a mapping")
    spec = RefinementSpec.from_mapping(raw_spec)
    parameter_space = build_parameter_space(config)
    names = tuple(name for name, _lower, _upper in parameter_space)
    structural = _read_frames(structural_sources, names)
    mechanical = _read_frames(mechanical_sources, names)
    if len(structural):
        gates = [
            assess_structural_gates(row, config)
            for row in structural.to_dict(orient="records")
        ]
        structural["structural_gate_pass"] = [
            item["structural_gate_pass"] for item in gates
        ]
        structural["structural_margin"] = [item["structural_margin"] for item in gates]
    active = config.get("active_learning", {})
    if not isinstance(active, Mapping):
        active = {}
    runner = LAMMPSRunner(config, start_scheduler_pool=False)
    pool = _candidate_pool(
        structural=structural,
        mechanical=mechanical,
        external_paths=external_pools,
        parameter_space=parameter_space,
        runner=runner,
        pool_size=int(active.get("n_candidate_pool", 16384)),
        global_fraction=float(active.get("global_fraction", 0.20)),
        local_radii=active.get("local_radii", [0.01, 0.025, 0.05]),
        seed=int(seed),
        round_number=1,
    )
    backend = _gp_backend(config)
    acquisition = backend.propose(AcquisitionContext(
        candidate_pool=pool,
        structural_observations=structural,
        mechanical_observations=mechanical,
        parameter_names=spec.parameter_names,
        round_number=1,
        proposal_count=spec.structural_proposals_per_round,
        seed=int(seed),
        structural_constraints=spec.structural_constraints,
        mechanical_objectives=spec.mechanical_objectives,
        refinement_spec=spec,
    ))
    diagnostics = {
        "schema": "ffopt-material-surrogate-preflight-v1",
        "status": "ready",
        "backend": acquisition.backend,
        "capability": acquisition.capability,
        "scientific_convergence_capable": (
            acquisition.scientific_convergence_capable
        ),
        "mode": acquisition.diagnostics.get("mode", "coverage_fallback"),
        "structural_training_rows": len(structural),
        "static_training_rows": len(mechanical),
        "candidate_pool_rows": len(pool),
        "proposal_rows": len(acquisition.proposals),
        "diagnostics": dict(acquisition.diagnostics),
    }
    scientific = {
        "schema": "ffopt-material-surrogate-preflight-v1",
        "seed": int(seed),
        "refinement": spec.scientific_identity(),
        "acquisition": dict(backend.scientific_identity()),
        "candidate_pool_size": int(active.get("n_candidate_pool", 16384)),
    }
    inputs: dict[str, Path] = {
        "runtime_config": runtime_source,
        "refinement_config": refinement_source,
        **{
            f"structural_{index:03d}": path
            for index, path in enumerate(structural_sources)
        },
        **{
            f"static_{index:03d}": path
            for index, path in enumerate(mechanical_sources)
        },
        **{
            f"external_pool_{index:03d}": path
            for index, path in enumerate(external_pools)
        },
    }
    destination = Path(output_dir).resolve()
    outputs = {label: destination / name for label, name in _OUTPUTS.items()}
    manifest_path = destination / "stage_manifest.json"
    if destination.exists():
        decision = check_artifact_reuse(
            manifest_path,
            kind="stage",
            identifier="material_surrogate_preflight",
            parameters=None,
            seeds=[int(seed)],
            scientific_config=scientific,
            input_artifacts=inputs,
            expected_outputs=outputs,
        )
        if decision.reusable:
            return json.loads(outputs["diagnostics"].read_text(encoding="utf-8"))
        raise RuntimeError(
            f"surrogate preflight output is not safely reusable: {decision.reason}"
        )

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    ))
    try:
        staged = {label: staging / name for label, name in _OUTPUTS.items()}
        pool.to_csv(staged["candidate_pool"], index=False)
        acquisition.proposals.to_csv(staged["proposals"], index=False)
        staged["diagnostics"].write_text(
            json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        manifest = build_artifact_manifest(
            kind="stage",
            identifier="material_surrogate_preflight",
            parameters=None,
            seeds=[int(seed)],
            scientific_config=scientific,
            input_artifacts=inputs,
            expected_outputs=staged,
        )
        write_artifact_manifest(staging / "stage_manifest.json", manifest)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return diagnostics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fit/probe the material constrained GP on exact evidence"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--refinement-config", required=True, type=Path)
    parser.add_argument("--structural-observations", action="append", required=True, type=Path)
    parser.add_argument("--static-observations", action="append", default=[], type=Path)
    parser.add_argument("--candidate-pool", action="append", default=[], type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=20260820)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    diagnostics = run_material_surrogate_preflight(
        runtime_config_path=args.config,
        refinement_config_path=args.refinement_config,
        structural_paths=args.structural_observations,
        mechanical_paths=args.static_observations,
        candidate_pool_paths=args.candidate_pool,
        output_dir=args.output_dir,
        seed=args.seed,
    )
    print(json.dumps(diagnostics, indent=2, sort_keys=True, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "run_material_surrogate_preflight"]
