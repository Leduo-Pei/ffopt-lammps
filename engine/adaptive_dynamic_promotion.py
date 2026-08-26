"""Single-node, two-batch finite-temperature finalist promotion.

The first batch evaluates a broad, cluster-balanced design with one trajectory
seed.  Its exact 300 K evidence then decides which smaller set receives the
remaining seeds.  Only candidates with the complete declared seed set may be
ranked as final results.  No historical directories are discovered: an optional
incumbent is the initial point explicitly present in the current input file and
all of its evidence is recomputed under the current protocol.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import pandas as pd

from .config_loader import load_config
from .cubic_elastic_batch import (
    DYNAMIC_PROTOCOL,
    Candidate,
    NestedResourcePlan,
    PreparedCandidate,
    _best_candidate_document,
    _protocol_module,
    aggregate_dynamic_seed_results,
    assess_structural_gates,
    effective_available_cores,
    load_candidates,
    plan_nested_resources,
    rank_elastic_results,
    run_elasticity_batch,
    same_element_parameter_contrast,
    select_finalists,
)
from .feasible_clusters import select_cluster_balanced_candidates
from .lammps_interface import LAMMPSRunner
from .parameter_space import build_parameter_space, initial_parameter_values
from workflow.artifact_manifest import (
    build_artifact_manifest,
    canonical_parameter_key,
    check_artifact_reuse,
    write_artifact_manifest,
)


IDENTIFIER = "adaptive_dynamic_promotion:v1"
_OUTPUT_NAMES = {
    "results": "dynamic_results.csv",
    "seed_results": "dynamic_seed_results.csv",
    "finalists": "finalists_selected.csv",
    "best_candidate": "best_candidate.json",
    "batch_summary": "batch_summary.json",
    "triage_selected": "dynamic_triage_selected.csv",
    "triage_seed_results": "dynamic_triage_seed_results.csv",
    "triage_results": "dynamic_triage_results.csv",
    "confirmation_selected": "dynamic_confirmation_selected.csv",
    "racing_state": "dynamic_racing_state.json",
}


class AdaptivePromotionError(RuntimeError):
    """Raised when the adaptive cascade cannot publish complete evidence."""


def _truth_series(frame: pd.DataFrame, name: str) -> pd.Series:
    if name not in frame:
        return pd.Series(False, index=frame.index, dtype=bool)
    return frame[name].astype(str).str.strip().str.lower().isin(
        {"true", "1", "yes", "on"}
    )


def _write_frame(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        os.close(descriptor)
        frame.to_csv(temporary, index=False)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    ).encode("utf-8")
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _selection_input(frame: pd.DataFrame) -> pd.DataFrame:
    ranked = rank_elastic_results(frame)
    return ranked.loc[ranked["finalist_eligible"]].reset_index(drop=True)


def _select(
    frame: pd.DataFrame,
    *,
    parameter_space: Sequence[tuple[str, float, float]],
    budget: int,
    cluster_mode: str,
    maximum_clusters: int,
    minimum_per_cluster: int,
    required_keys: Sequence[str],
) -> pd.DataFrame:
    eligible = _selection_input(frame)
    if eligible.empty:
        return eligible
    names = [name for name, _lower, _upper in parameter_space]
    bounds = {name: (lower, upper) for name, lower, upper in parameter_space}
    available_keys = set(eligible["parameter_key"].astype(str))
    missing = sorted(set(required_keys) - available_keys)
    if missing:
        raise AdaptivePromotionError(
            "A requested incumbent is not eligible under the current protocol: "
            + ", ".join(missing)
        )
    return select_cluster_balanced_candidates(
        eligible,
        parameter_columns=names,
        bounds=bounds,
        quality_column="mechanical_max_error_percent",
        budget=min(int(budget), len(eligible)),
        quality_ascending=True,
        minimum_per_cluster=(
            int(minimum_per_cluster) if cluster_mode == "auto" else 0
        ),
        required_parameter_keys=tuple(required_keys),
        maximum_clusters=(int(maximum_clusters) if cluster_mode == "auto" else 1),
    )


def _prepared_for_aggregation(
    candidate: Candidate,
    *,
    runner: LAMMPSRunner,
    config: Mapping[str, Any],
) -> PreparedCandidate:
    resolved = runner._resolve_params(candidate.raw_parameters)
    return PreparedCandidate(
        candidate=candidate,
        resolved_parameters=resolved,
        force_field_include=Path(),
        parameter_contrast=same_element_parameter_contrast(resolved, config),
    )


def _aggregate_all_triage_candidates(
    *,
    triage_path: Path,
    seed_frame: pd.DataFrame,
    confirmation_keys: set[str],
    config: dict[str, Any],
    all_seeds: Sequence[int],
) -> pd.DataFrame:
    parameter_space = build_parameter_space(config)
    candidates = load_candidates(triage_path, parameter_space)
    module = _protocol_module(config, DYNAMIC_PROTOCOL)
    selection = config["elasticity"]["selection"]
    minimum_r2 = float(selection["fit_quality"]["minimum_r2"])
    born_required = bool(selection["born_stability"]["required"])
    quality_tier = float(config["elasticity"]["reporting"]["mechanical_tier_percent"])
    runner = LAMMPSRunner(config, start_scheduler_pool=False)
    rows_by_key: dict[str, list[dict[str, Any]]] = {}
    for row in seed_frame.to_dict(orient="records"):
        rows_by_key.setdefault(str(row["parameter_key"]), []).append(row)
    aggregated: list[dict[str, Any]] = []
    for candidate in candidates:
        row = aggregate_dynamic_seed_results(
            rows_by_key.get(candidate.parameter_key, []),
            candidate=candidate,
            prepared=_prepared_for_aggregation(candidate, runner=runner, config=config),
            structural=assess_structural_gates(candidate.source_row, config),
            module=module,
            expected_seeds=all_seeds,
            minimum_r2=minimum_r2,
            born_required=born_required,
            quality_tier_percent=quality_tier,
        )
        row["racing_confirmation_status"] = (
            "confirmed" if candidate.parameter_key in confirmation_keys else "triage_only"
        )
        aggregated.append(row)
    return rank_elastic_results(pd.DataFrame(aggregated))


def run_adaptive_dynamic_promotion(
    *,
    config_path: str | os.PathLike[str],
    parameters_path: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    resources: NestedResourcePlan,
    screen_candidates: int,
    confirm_candidates: int,
    triage_seed: int,
    cluster_mode: str = "auto",
    maximum_clusters: int = 8,
    minimum_per_cluster: int = 1,
    incumbent: str = "off",
    top_n: int = 10,
    near_optimal_window_percent: float = 1.0,
    diversity_slots: int = 2,
    minimum: int = 1,
    require_minimum: bool = False,
    backend_factory=None,
) -> pd.DataFrame:
    """Run or resume the broad-one-seed then narrow-remaining-seeds cascade."""

    if screen_candidates < 1 or confirm_candidates < 1:
        raise ValueError("screen_candidates and confirm_candidates must be positive")
    if confirm_candidates > screen_candidates:
        raise ValueError("confirm_candidates cannot exceed screen_candidates")
    if cluster_mode not in {"auto", "off"}:
        raise ValueError("cluster_mode must be auto or off")
    if incumbent not in {"off", "initial"}:
        raise ValueError("incumbent must be off or initial")

    config_source = Path(config_path).resolve()
    parameter_source = Path(parameters_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    config = load_config(config_source)
    parameter_space = build_parameter_space(config)
    module = _protocol_module(config, DYNAMIC_PROTOCOL)
    all_seeds = [int(seed) for seed in module["protocol"]["seeds"]]
    if triage_seed not in all_seeds:
        raise ValueError("triage_seed is not declared by the dynamic elasticity protocol")
    confirmation_seeds = [seed for seed in all_seeds if seed != triage_seed]
    if not confirmation_seeds:
        raise ValueError("adaptive promotion requires at least one confirmation seed")
    required_keys = (
        [canonical_parameter_key(initial_parameter_values(config))]
        if incumbent == "initial"
        else []
    )
    outputs = {label: destination / name for label, name in _OUTPUT_NAMES.items()}
    scientific = {
        "schema": "ffopt-adaptive-dynamic-promotion-v1",
        "parameter_space": parameter_space,
        "elasticity": config["elasticity"],
        "screen_candidates": int(screen_candidates),
        "confirm_candidates": int(confirm_candidates),
        "triage_seeds": [int(triage_seed)],
        "confirmation_seeds": confirmation_seeds,
        "cluster_mode": cluster_mode,
        "maximum_clusters": int(maximum_clusters),
        "minimum_per_cluster": int(minimum_per_cluster),
        "incumbent": incumbent,
        "incumbent_parameter_key": required_keys[0] if required_keys else None,
        "final_selection": {
            "top_n": int(top_n),
            "minimum": int(minimum),
            "require_minimum": bool(require_minimum),
            "near_optimal_window_percent": float(near_optimal_window_percent),
            "diversity_slots": int(diversity_slots),
        },
    }

    triage_dir = destination / "triage_batch"
    confirmation_dir = destination / "confirmation_batch"
    root_manifest = destination / "stage_manifest.json"
    if root_manifest.is_file() and (triage_dir / "stage_manifest.json").is_file() and (
        confirmation_dir / "stage_manifest.json"
    ).is_file():
        inputs = {
            "config": config_source,
            "parameters": parameter_source,
            "triage_manifest": triage_dir / "stage_manifest.json",
            "confirmation_manifest": confirmation_dir / "stage_manifest.json",
        }
        decision = check_artifact_reuse(
            root_manifest,
            kind="stage",
            identifier=IDENTIFIER,
            parameters=None,
            seeds=all_seeds,
            scientific_config=scientific,
            input_artifacts=inputs,
            expected_outputs=outputs,
        )
        if decision.reusable:
            return pd.read_csv(outputs["results"], float_precision="round_trip")
        raise AdaptivePromotionError(
            f"Completed adaptive finalist stage is not safely reusable: {decision.reason}"
        )

    static_frame = pd.read_csv(parameter_source, float_precision="round_trip")
    triage = _select(
        static_frame,
        parameter_space=parameter_space,
        budget=screen_candidates,
        cluster_mode=cluster_mode,
        maximum_clusters=maximum_clusters,
        minimum_per_cluster=minimum_per_cluster,
        required_keys=required_keys,
    )
    if len(triage) < confirm_candidates:
        raise AdaptivePromotionError(
            f"Only {len(triage)} current-protocol static candidates are eligible; "
            f"at least {confirm_candidates} are required for confirmation"
        )
    _write_frame(outputs["triage_selected"], triage)
    batch_kwargs: dict[str, Any] = {}
    if backend_factory is not None:
        batch_kwargs["backend_factory"] = backend_factory
    run_elasticity_batch(
        config_path=config_source,
        parameters_path=outputs["triage_selected"],
        output_dir=triage_dir,
        protocol="dynamic",
        resources=resources,
        top_n=len(triage),
        minimum=len(triage),
        require_minimum=True,
        near_optimal_window_percent=0.0,
        diversity_slots=0,
        dynamic_seeds=[triage_seed],
        **batch_kwargs,
    )
    if not (triage_dir / "stage_manifest.json").is_file():
        raise AdaptivePromotionError("Triage batch is incomplete; rerun to resume failed units")
    triage_seed_frame = pd.read_csv(
        triage_dir / "dynamic_seed_results.csv", float_precision="round_trip"
    )
    triage_results = pd.read_csv(
        triage_dir / "dynamic_results.csv", float_precision="round_trip"
    )
    _write_frame(outputs["triage_seed_results"], triage_seed_frame)
    _write_frame(outputs["triage_results"], triage_results)

    triage_eligible_keys = set(
        triage_results.loc[
            _truth_series(triage_results, "finalist_eligible"), "parameter_key"
        ].astype(str)
    )
    confirm_required = [key for key in required_keys if key in triage_eligible_keys]
    confirmation = _select(
        triage_results,
        parameter_space=parameter_space,
        budget=confirm_candidates,
        cluster_mode=cluster_mode,
        maximum_clusters=maximum_clusters,
        minimum_per_cluster=minimum_per_cluster,
        required_keys=confirm_required,
    )
    if require_minimum and len(confirmation) < minimum:
        raise AdaptivePromotionError(
            f"Only {len(confirmation)} triage candidates pass current 300 K gates; "
            f"the declared minimum is {minimum}"
        )
    if confirmation.empty:
        raise AdaptivePromotionError("No triage candidate is eligible for confirmation")
    _write_frame(outputs["confirmation_selected"], confirmation)
    run_elasticity_batch(
        config_path=config_source,
        parameters_path=outputs["confirmation_selected"],
        output_dir=confirmation_dir,
        protocol="dynamic",
        resources=resources,
        top_n=len(confirmation),
        minimum=len(confirmation),
        require_minimum=True,
        near_optimal_window_percent=0.0,
        diversity_slots=0,
        dynamic_seeds=confirmation_seeds,
        **batch_kwargs,
    )
    if not (confirmation_dir / "stage_manifest.json").is_file():
        raise AdaptivePromotionError(
            "Confirmation batch is incomplete; rerun to resume failed units"
        )
    confirmation_seed_frame = pd.read_csv(
        confirmation_dir / "dynamic_seed_results.csv", float_precision="round_trip"
    )
    seed_frame = pd.concat(
        [triage_seed_frame, confirmation_seed_frame], ignore_index=True, sort=False
    ).sort_values(["parameter_key", "trajectory_seed"], kind="stable")
    _write_frame(outputs["seed_results"], seed_frame)
    confirmation_keys = set(confirmation["parameter_key"].astype(str))
    ranked = _aggregate_all_triage_candidates(
        triage_path=outputs["triage_selected"],
        seed_frame=seed_frame,
        confirmation_keys=confirmation_keys,
        config=config,
        all_seeds=all_seeds,
    )
    finalists = select_finalists(
        ranked,
        parameter_space=parameter_space,
        top_n=min(int(top_n), max(1, len(confirmation))),
        near_optimal_window_percent=near_optimal_window_percent,
        diversity_slots=min(diversity_slots, max(0, min(top_n, len(confirmation)) - 1)),
        minimum=min(minimum, max(1, len(confirmation))),
    )
    if require_minimum and len(finalists) < minimum:
        raise AdaptivePromotionError(
            f"Only {len(finalists)} candidates completed all seeds and hard gates; "
            f"the declared minimum is {minimum}"
        )
    _write_frame(outputs["results"], ranked)
    _write_frame(outputs["finalists"], finalists)
    best = _best_candidate_document(
        ranked, [name for name, _lower, _upper in parameter_space]
    )
    _write_json(outputs["best_candidate"], best)
    racing_state = {
        "schema_version": 1,
        "status": "completed",
        "triage": {
            "requested_candidates": int(screen_candidates),
            "selected_candidates": len(triage),
            "seeds": [triage_seed],
            "work_units": len(triage),
        },
        "confirmation": {
            "requested_candidates": int(confirm_candidates),
            "selected_candidates": len(confirmation),
            "seeds": confirmation_seeds,
            "work_units": len(confirmation) * len(confirmation_seeds),
        },
        "total_dynamic_work_units": len(seed_frame),
        "complete_seed_candidates": int(_truth_series(ranked, "all_seeds_complete").sum()),
        "incumbent": {
            "mode": incumbent,
            "parameter_key": required_keys[0] if required_keys else None,
            "triaged": bool(required_keys and required_keys[0] in set(triage["parameter_key"])),
            "confirmed": bool(required_keys and required_keys[0] in confirmation_keys),
        },
    }
    _write_json(outputs["racing_state"], racing_state)
    summary = {
        "schema_version": 1,
        "protocol": DYNAMIC_PROTOCOL,
        "mode": "adaptive_two_batch",
        "input_candidates": len(static_frame),
        "triage_candidates": len(triage),
        "confirmation_candidates": len(confirmation),
        "completed_work_units": len(seed_frame),
        "expected_work_units": len(triage) + len(confirmation) * len(confirmation_seeds),
        "hard_gate_eligible": int(_truth_series(ranked, "finalist_eligible").sum()),
        "selected_finalists": len(finalists),
        "required_finalists": int(minimum),
        "require_minimum": bool(require_minimum),
        "scientific_outcome": best["status"],
        "resources": resources.to_dict(),
    }
    _write_json(outputs["batch_summary"], summary)
    inputs = {
        "config": config_source,
        "parameters": parameter_source,
        "triage_manifest": triage_dir / "stage_manifest.json",
        "confirmation_manifest": confirmation_dir / "stage_manifest.json",
    }
    manifest = build_artifact_manifest(
        kind="stage",
        identifier=IDENTIFIER,
        parameters=None,
        seeds=all_seeds,
        scientific_config=scientific,
        input_artifacts=inputs,
        expected_outputs=outputs,
    )
    write_artifact_manifest(root_manifest, manifest)
    return ranked


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run adaptive one-seed triage then multi-seed confirmation"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--parameters", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--available-cores", required=True, type=int)
    parser.add_argument("--cores-per-state", required=True, type=int)
    parser.add_argument("--candidate-workers", type=int, default=1)
    parser.add_argument("--state-workers", type=int)
    parser.add_argument("--omp-threads-per-state", type=int, default=1)
    parser.add_argument("--screen-candidates", required=True, type=int)
    parser.add_argument("--confirm-candidates", required=True, type=int)
    parser.add_argument("--triage-seed", required=True, type=int)
    parser.add_argument("--cluster-mode", choices=("auto", "off"), default="auto")
    parser.add_argument("--maximum-clusters", type=int, default=8)
    parser.add_argument("--minimum-per-cluster", type=int, default=1)
    parser.add_argument("--incumbent", choices=("off", "initial"), default="off")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--minimum", type=int, default=1)
    parser.add_argument("--require-minimum", action="store_true")
    parser.add_argument("--near-optimal-window-percent", type=float, default=1.0)
    parser.add_argument("--diversity-slots", type=int, default=2)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    resources = plan_nested_resources(
        available_cores=effective_available_cores(arguments.available_cores),
        cores_per_state=arguments.cores_per_state,
        candidate_workers=arguments.candidate_workers,
        state_workers=arguments.state_workers,
        omp_threads_per_rank=arguments.omp_threads_per_state,
    )
    ranked = run_adaptive_dynamic_promotion(
        config_path=arguments.config,
        parameters_path=arguments.parameters,
        output_dir=arguments.output_dir,
        resources=resources,
        screen_candidates=arguments.screen_candidates,
        confirm_candidates=arguments.confirm_candidates,
        triage_seed=arguments.triage_seed,
        cluster_mode=arguments.cluster_mode,
        maximum_clusters=arguments.maximum_clusters,
        minimum_per_cluster=arguments.minimum_per_cluster,
        incumbent=arguments.incumbent,
        top_n=arguments.top_n,
        minimum=arguments.minimum,
        require_minimum=arguments.require_minimum,
        near_optimal_window_percent=arguments.near_optimal_window_percent,
        diversity_slots=arguments.diversity_slots,
    )
    print(json.dumps({
        "status": "completed",
        "rows": len(ranked),
        "output_dir": str(Path(arguments.output_dir).resolve()),
    }, sort_keys=True))
    return 0


__all__ = [
    "AdaptivePromotionError",
    "IDENTIFIER",
    "build_parser",
    "main",
    "run_adaptive_dynamic_promotion",
]


if __name__ == "__main__":
    raise SystemExit(main())
