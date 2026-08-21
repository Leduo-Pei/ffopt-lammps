"""Collect explicit measured evidence and NN proposals for material screening."""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shutil
import tempfile
from typing import Any, Sequence

import numpy as np
import pandas as pd

from engine.config_loader import load_config
from engine.parameter_space import build_parameter_space
from workflow.artifact_manifest import (
    build_artifact_manifest,
    canonical_parameter_key,
    check_artifact_reuse,
    write_artifact_manifest,
)
from workflow.material_screen import select_static_screen_candidates


_OUTPUT_NAMES = {
    "candidates": "candidates.csv",
    "candidate_pool": "candidate_pool.csv",
    "static_screen": "static_screen_candidates.csv",
    "summary": "candidates_summary.json",
}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"true", "1", "yes", "on"}


def _parameter_key(row: dict[str, Any], names: Sequence[str]) -> str:
    values: dict[str, float] = {}
    for name in names:
        try:
            value = float(row[name])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"Candidate row is missing finite parameter {name!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"Candidate parameter {name!r} must be finite")
        values[name] = value
    return canonical_parameter_key(values)


def _read_sources(paths: Sequence[Path], names: Sequence[str]) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for source_index, path in enumerate(paths):
        source = path.resolve()
        if not source.is_file():
            raise ValueError(f"Explicit candidate source does not exist: {source}")
        frame = pd.read_csv(source, float_precision="round_trip")
        if frame.empty:
            continue
        missing = [name for name in names if name not in frame]
        if missing:
            raise ValueError(f"Candidate source {source} is missing parameters: {missing}")
        frame = frame.copy()
        frame["parameter_key"] = [
            _parameter_key(row, names) for row in frame.to_dict(orient="records")
        ]
        frame["candidate_source"] = str(source)
        frame["_source_index"] = source_index
        frame["_row_index"] = np.arange(len(frame))
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["parameter_key", *names])
    combined = pd.concat(frames, ignore_index=True, sort=False)
    audited = (
        combined["audit_certified"].map(_truthy).astype(int)
        if "audit_certified" in combined
        else pd.Series(0, index=combined.index)
    )
    seeds = (
        pd.to_numeric(combined["n_seeds"], errors="coerce").fillna(0)
        if "n_seeds" in combined
        else pd.Series(0, index=combined.index)
    )
    combined["_priority"] = audited * 10000 + seeds * 100
    combined = combined.sort_values(
        ["_priority", "_source_index", "_row_index"],
        ascending=[False, False, False],
        kind="stable",
    ).drop_duplicates("parameter_key", keep="first")
    return combined.drop(
        columns=["_priority", "_source_index", "_row_index"]
    ).reset_index(drop=True)


def _nn_rows(path: Path, names: Sequence[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = path.resolve()
    try:
        document = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read explicit NN result {source}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError("NN result root must be a JSON object")

    pool_rows: list[dict[str, Any]] = []
    for rank, item in enumerate(document.get("all_candidates", []), start=1):
        if not isinstance(item, dict) or not isinstance(item.get("params"), dict):
            raise ValueError("NN all_candidates entries must contain a params mapping")
        row = {name: item["params"].get(name) for name in names}
        predicted = float(item.get("predicted_obj", math.inf))
        row.update({
            "predicted_objective": predicted,
            "acquisition_score": (
                1.0 / (1.0 + max(predicted, 0.0)) if math.isfinite(predicted) else 0.0
            ),
            "nn_rank": rank,
            "candidate_source": str(source),
        })
        row["parameter_key"] = _parameter_key(row, names)
        pool_rows.append(row)

    measured_rows: list[dict[str, Any]] = []
    for item in document.get("all_lammps", []):
        if not isinstance(item, dict) or not item.get("lammps_success"):
            continue
        params = item.get("params")
        properties = item.get("lammps_properties")
        if not isinstance(params, dict) or not isinstance(properties, dict):
            raise ValueError(
                "Successful NN LAMMPS entries need params and lammps_properties mappings"
            )
        row = {name: params.get(name) for name in names}
        row.update({f"calc_{name}": value for name, value in properties.items()})
        row.update({
            "objective": item.get("lammps_obj"),
            "success": True,
            "candidate_source": str(source),
        })
        row["parameter_key"] = _parameter_key(row, names)
        measured_rows.append(row)
    return pd.DataFrame(measured_rows), pd.DataFrame(pool_rows)


def collect_material_candidates(
    *,
    config_path: str | os.PathLike[str],
    source_paths: Sequence[str | os.PathLike[str]],
    nn_result_path: str | os.PathLike[str] | None,
    output_dir: str | os.PathLike[str],
    screen_settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Publish one immutable, explicitly sourced candidate collection."""
    config_source = Path(config_path).resolve()
    sources = tuple(Path(item).resolve() for item in source_paths)
    nn_source = Path(nn_result_path).resolve() if nn_result_path is not None else None
    config = load_config(config_source)
    names = [item[0] for item in build_parameter_space(config)]
    scientific_config = {
        "schema": "ffopt-material-candidates-v1",
        "parameter_names": names,
        "source_order": [str(path) for path in sources],
        "nn_result": str(nn_source) if nn_source is not None else None,
        "screen_settings": dict(screen_settings or {}),
    }
    inputs = {
        "config": config_source,
        **{f"source_{index:03d}": path for index, path in enumerate(sources)},
    }
    if nn_source is not None:
        inputs["nn_result"] = nn_source
    destination = Path(output_dir).resolve()
    output_paths = {label: destination / name for label, name in _OUTPUT_NAMES.items()}
    manifest_path = destination / "stage_manifest.json"
    if destination.exists():
        decision = check_artifact_reuse(
            manifest_path,
            kind="stage",
            identifier="material_candidates",
            parameters=None,
            seeds=[],
            scientific_config=scientific_config,
            input_artifacts=inputs,
            expected_outputs=output_paths,
        )
        if decision.reusable:
            return json.loads(output_paths["summary"].read_text(encoding="utf-8"))
        raise RuntimeError(
            f"Candidate output cannot be reused safely: {destination}: {decision.reason}"
        )

    observed = _read_sources(sources, names)
    if nn_source is None:
        nn_measured = pd.DataFrame()
        pool = pd.DataFrame(columns=[
            "parameter_key", *names, "predicted_objective", "acquisition_score"
        ])
    else:
        nn_measured, pool = _nn_rows(nn_source, names)
    if not nn_measured.empty:
        # Audited/replicated evidence from the declared CSV sources has higher
        # authority than a single NN-validation evaluation.  NN measurements
        # therefore only fill parameter keys that are not already represented;
        # they must never silently replace an audited row during concatenation.
        observed_keys_before_nn = set(observed["parameter_key"].astype(str))
        novel_nn = nn_measured.loc[
            ~nn_measured["parameter_key"].astype(str).isin(observed_keys_before_nn)
        ]
        observed = pd.concat([observed, novel_nn], ignore_index=True, sort=False)
        observed = observed.drop_duplicates("parameter_key", keep="first")
    observed_keys = set(observed["parameter_key"].astype(str))
    if not pool.empty:
        pool = pool.loc[~pool["parameter_key"].astype(str).isin(observed_keys)]
        pool = pool.drop_duplicates("parameter_key", keep="first").reset_index(drop=True)
    else:
        pool = pd.DataFrame(columns=[
            "parameter_key", *names, "predicted_objective", "acquisition_score"
        ])
    static_screen = (
        select_static_screen_candidates(
            observed,
            config=config,
            settings=screen_settings,
        )
        if screen_settings is not None
        else observed.copy()
    )
    summary = {
        "schema": "ffopt-material-candidates-v1",
        "parameter_names": names,
        "explicit_sources": [str(path) for path in sources],
        "measured_candidates": int(len(observed)),
        "unevaluated_candidate_pool": int(len(pool)),
        "static_screen_candidates": int(len(static_screen)),
        "static_screen_core": int(
            static_screen.get("screen_region", pd.Series(dtype=str)).eq("core").sum()
        ),
        "static_screen_buffer": int(
            static_screen.get("screen_region", pd.Series(dtype=str)).eq("buffer").sum()
        ),
    }

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(
        prefix=f".{destination.name}.tmp-", dir=destination.parent
    ))
    try:
        staged = {label: staging / name for label, name in _OUTPUT_NAMES.items()}
        observed.to_csv(staged["candidates"], index=False)
        pool.to_csv(staged["candidate_pool"], index=False)
        static_screen.to_csv(staged["static_screen"], index=False)
        staged["summary"].write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        manifest = build_artifact_manifest(
            kind="stage",
            identifier="material_candidates",
            parameters=None,
            seeds=[],
            scientific_config=scientific_config,
            input_artifacts=inputs,
            expected_outputs=staged,
        )
        write_artifact_manifest(staging / "stage_manifest.json", manifest)
        os.replace(staging, destination)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Collect explicit measured and NN material candidates"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--source", action="append", required=True, type=Path)
    parser.add_argument("--nn-result", type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--screen-minimum", type=int)
    parser.add_argument("--screen-per-dimension", type=int)
    parser.add_argument("--screen-maximum", type=int)
    parser.add_argument("--screen-core-fraction", type=float)
    parser.add_argument("--screen-buffer-multiplier", type=float)
    parser.add_argument("--screen-elite-fraction", type=float)
    args = parser.parse_args(argv)
    screen_values = {
        "minimum": args.screen_minimum,
        "per_dimension": args.screen_per_dimension,
        "maximum": args.screen_maximum,
        "core_fraction": args.screen_core_fraction,
        "buffer_multiplier": args.screen_buffer_multiplier,
        "objective_elite_fraction": args.screen_elite_fraction,
    }
    screen_settings = (
        {key: value for key, value in screen_values.items() if value is not None}
        if any(value is not None for value in screen_values.values())
        else None
    )
    summary = collect_material_candidates(
        config_path=args.config,
        source_paths=args.source,
        nn_result_path=args.nn_result,
        output_dir=args.output_dir,
        screen_settings=screen_settings,
    )
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
