"""Explicit, provenance-rich Top-N report for static and dynamic rankings."""

from __future__ import annotations

import argparse
from collections.abc import Mapping, Sequence
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

from .config_loader import load_config
from .cubic_elastic_batch import rank_elastic_results
from .material_validation import (
    MaterialValidationArtifactError,
    _atomic_bytes,
    _json_safe,
    _write_frame,
    _write_json,
)
from .parameter_space import build_parameter_space
from workflow.artifact_manifest import (
    build_artifact_manifest,
    canonical_parameter_key,
    check_artifact_reuse,
    sha256_file,
    write_artifact_manifest,
)


_EVIDENCE_COLUMNS = (
    "mechanical_max_error_percent",
    "mechanical_rmse_percent",
    "structural_margin",
    "minimum_fit_r2",
    "same_element_parameter_contrast",
    "structural_gate_pass",
    "born_stability_pass",
    "fit_quality_pass",
    "finalist_eligible",
    "within_quality_tier",
    "quality_status",
)


def _validated_ranking(
    path: Path,
    parameter_space: Sequence[tuple[str, float, float]],
) -> pd.DataFrame:
    frame = pd.read_csv(path)
    names = [name for name, _lower, _upper in parameter_space]
    if frame.empty:
        for name in ("parameter_key", *names):
            if name not in frame:
                frame[name] = pd.Series(dtype=str if name == "parameter_key" else float)
        return rank_elastic_results(frame)
    missing = [name for name in ("parameter_key", *names) if name not in frame]
    if missing:
        raise ValueError(f"ranking {path} is missing columns: {missing}")
    bounds = {name: (float(low), float(high)) for name, low, high in parameter_space}
    for index, row in frame.iterrows():
        parameters: dict[str, float] = {}
        for name in names:
            try:
                value = float(row[name])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"ranking {path} row {index + 1} parameter {name!r} is not numeric"
                ) from exc
            if not math.isfinite(value):
                raise ValueError(
                    f"ranking {path} row {index + 1} parameter {name!r} is not finite"
                )
            lower, upper = bounds[name]
            tolerance = 1.0e-12 * max(1.0, abs(value), abs(lower), abs(upper))
            if value < lower - tolerance or value > upper + tolerance:
                raise ValueError(
                    f"ranking {path} row {index + 1} parameter {name!r} is out of bounds"
                )
            parameters[name] = value
        expected_key = canonical_parameter_key(parameters)
        if str(row["parameter_key"]) != expected_key:
            raise ValueError(
                f"ranking {path} row {index + 1} parameter_key does not match values"
            )
    ranked = rank_elastic_results(frame)
    return ranked.drop_duplicates("parameter_key", keep="first").reset_index(drop=True)


def _optional_value(row: Mapping[str, Any] | None, name: str) -> Any:
    if row is None or name not in row:
        return None
    return _json_safe(row[name])


def _ranking_map(frame: pd.DataFrame) -> dict[str, dict[str, Any]]:
    return {
        str(row["parameter_key"]): {str(name): value for name, value in row.items()}
        for _index, row in frame.iterrows()
    }


def _same_parameters(
    static_row: Mapping[str, Any],
    dynamic_row: Mapping[str, Any],
    names: Sequence[str],
) -> bool:
    return all(
        math.isclose(
            float(static_row[name]),
            float(dynamic_row[name]),
            rel_tol=0.0,
            abs_tol=1.0e-12 * max(1.0, abs(float(static_row[name]))),
        )
        for name in names
    )


def _build_records(
    static: pd.DataFrame,
    dynamic: pd.DataFrame,
    *,
    parameter_names: Sequence[str],
    top_n: int,
    static_source: Path,
    dynamic_source: Path,
) -> list[dict[str, Any]]:
    if top_n < 1:
        raise ValueError("top_n must be positive")
    static_map = _ranking_map(static)
    dynamic_map = _ranking_map(dynamic)
    dynamic_keys = [
        str(row["parameter_key"])
        for _index, row in dynamic.iterrows()
        if bool(row["finalist_eligible"])
    ]
    # Static-only candidates are visibly lower-evidence fallbacks.  A candidate
    # actually evaluated and rejected dynamically is never resurrected from its
    # static rank.
    static_only_keys = [
        str(row["parameter_key"])
        for _index, row in static.iterrows()
        if bool(row["finalist_eligible"])
        and str(row["parameter_key"]) not in dynamic_map
    ]
    ordered_keys = [*dynamic_keys, *static_only_keys][:top_n]
    records: list[dict[str, Any]] = []
    for selection_rank, key in enumerate(ordered_keys, start=1):
        static_row = static_map.get(key)
        dynamic_row = dynamic_map.get(key)
        if static_row is not None and dynamic_row is not None and not _same_parameters(
            static_row, dynamic_row, parameter_names
        ):
            raise ValueError(f"static/dynamic parameter mismatch for {key}")
        source_row = dynamic_row or static_row
        assert source_row is not None
        evidence = (
            "static+dynamic"
            if static_row is not None and dynamic_row is not None
            else "dynamic_only"
            if dynamic_row is not None
            else "static_only"
        )
        record: dict[str, Any] = {
            "selection_rank": selection_rank,
            "parameter_key": key,
            "static_rank": (
                int(static_row["result_rank"]) if static_row is not None else None
            ),
            "dynamic_rank": (
                int(dynamic_row["result_rank"]) if dynamic_row is not None else None
            ),
            "evidence": evidence,
            "selection_basis": "dynamic" if dynamic_row is not None else "static",
            "static_source": str(static_source.resolve()),
            "dynamic_source": str(dynamic_source.resolve()),
        }
        record.update({name: float(source_row[name]) for name in parameter_names})
        for fidelity, row in (("static", static_row), ("dynamic", dynamic_row)):
            for name in _EVIDENCE_COLUMNS:
                record[f"{fidelity}_{name}"] = _optional_value(row, name)
        records.append(record)
    return records


def _markdown(
    records: Sequence[Mapping[str, Any]], parameter_names: Sequence[str]
) -> str:
    columns = [
        "selection_rank",
        "parameter_key",
        *parameter_names,
        "static_rank",
        "dynamic_rank",
        "evidence",
        "static_mechanical_max_error_percent",
        "dynamic_mechanical_max_error_percent",
    ]
    labels = [
        "Top",
        "Parameter key",
        *parameter_names,
        "Static rank",
        "Dynamic rank",
        "Evidence",
        "Static M∞ (%)",
        "Dynamic M∞ (%)",
    ]
    lines = [
        "# Top force-field parameters",
        "",
        "Dynamic evidence is ranked first. Static-only rows are retained only as "
        "explicit lower-evidence fallbacks; a dynamically rejected candidate is not "
        "resurrected by its static rank.",
        "",
        "| " + " | ".join(labels) + " |",
        "| " + " | ".join("---" for _ in labels) + " |",
    ]
    for record in records:
        values = []
        for name in columns:
            value = record.get(name)
            if name == "parameter_key" and value is not None:
                value = str(value).rsplit(":", 1)[-1][:12]
            elif isinstance(value, float):
                value = f"{value:.8g}"
            values.append("" if value is None else str(value))
        lines.append("| " + " | ".join(values) + " |")
    if not records:
        lines.extend(["", "No hard-gate-eligible parameter set was available."])
    return "\n".join(lines) + "\n"


def write_top_parameters_report(
    *,
    config_path: str | Path,
    static_ranking_path: str | Path,
    dynamic_ranking_path: str | Path,
    output_dir: str | Path,
    top_n: int = 20,
    manifest_name: str = "stage_manifest.json",
) -> pd.DataFrame:
    """Create CSV/JSON/Markdown Top-N artifacts from explicitly named rankings."""

    config_source = Path(config_path).resolve()
    static_source = Path(static_ranking_path).resolve()
    dynamic_source = Path(dynamic_ranking_path).resolve()
    destination = Path(output_dir).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    if not manifest_name or Path(manifest_name).name != manifest_name:
        raise ValueError("manifest_name must be one plain file name")
    config = load_config(config_source)
    parameter_space = build_parameter_space(config)
    parameter_names = [name for name, _lower, _upper in parameter_space]
    outputs = {
        "top_csv": destination / "TOP_PARAMETERS.csv",
        "top_json": destination / "TOP_PARAMETERS.json",
        "top_markdown": destination / "TOP_PARAMETERS.md",
    }
    inputs = {
        "compiled_config": config_source,
        "static_ranking": static_source,
        "dynamic_ranking": dynamic_source,
    }
    scientific = {
        "schema_version": 1,
        "top_n": top_n,
        "parameter_space": parameter_space,
        "selection_policy": (
            "dynamic eligible by dynamic rank, followed by unevaluated static-only "
            "eligible candidates; dynamically rejected candidates excluded"
        ),
    }
    manifest_path = destination / manifest_name
    if manifest_path.exists():
        decision = check_artifact_reuse(
            manifest_path,
            kind="stage",
            identifier="material_top_parameters",
            parameters=None,
            seeds=[],
            scientific_config=scientific,
            input_artifacts=inputs,
            expected_outputs=outputs,
        )
        if not decision.reusable:
            raise MaterialValidationArtifactError(
                f"completed Top-N report is not reusable: {decision.reason}"
            )
        return pd.read_csv(outputs["top_csv"])

    static = _validated_ranking(static_source, parameter_space)
    dynamic = _validated_ranking(dynamic_source, parameter_space)
    records = _build_records(
        static,
        dynamic,
        parameter_names=parameter_names,
        top_n=top_n,
        static_source=static_source,
        dynamic_source=dynamic_source,
    )
    report_columns = [
        "selection_rank",
        "parameter_key",
        "static_rank",
        "dynamic_rank",
        "evidence",
        "selection_basis",
        "static_source",
        "dynamic_source",
        *parameter_names,
        *(
            f"{fidelity}_{name}"
            for fidelity in ("static", "dynamic")
            for name in _EVIDENCE_COLUMNS
        ),
    ]
    frame = pd.DataFrame(records, columns=report_columns)
    _write_frame(outputs["top_csv"], frame)
    document = {
        "schema_version": 1,
        "provenance": {
            "compiled_config": {
                "path": str(config_source),
                "artifact": sha256_file(config_source).to_dict(),
            },
            "static_ranking": {
                "path": str(static_source),
                "artifact": sha256_file(static_source).to_dict(),
            },
            "dynamic_ranking": {
                "path": str(dynamic_source),
                "artifact": sha256_file(dynamic_source).to_dict(),
            },
        },
        "selection_policy": scientific["selection_policy"],
        "requested_top_n": top_n,
        "reported": len(records),
        "parameter_names": parameter_names,
        "records": records,
    }
    _write_json(outputs["top_json"], document)
    _atomic_bytes(
        outputs["top_markdown"],
        _markdown(records, parameter_names).encode("utf-8"),
    )
    manifest = build_artifact_manifest(
        kind="stage",
        identifier="material_top_parameters",
        parameters=None,
        seeds=[],
        scientific_config=scientific,
        input_artifacts=inputs,
        expected_outputs=outputs,
    )
    write_artifact_manifest(manifest_path, manifest)
    return frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate provenance-rich Top-N static/dynamic parameter reports"
    )
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--static-ranking", required=True, type=Path)
    parser.add_argument("--dynamic-ranking", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = build_parser().parse_args(argv)
    frame = write_top_parameters_report(
        config_path=arguments.config,
        static_ranking_path=arguments.static_ranking,
        dynamic_ranking_path=arguments.dynamic_ranking,
        output_dir=arguments.output_dir,
        top_n=arguments.top_n,
    )
    print(json.dumps({
        "status": "completed",
        "reported": len(frame),
        "output_dir": str(arguments.output_dir.resolve()),
    }, sort_keys=True))
    return 0


__all__ = [
    "build_parser",
    "main",
    "write_top_parameters_report",
]


if __name__ == "__main__":
    raise SystemExit(main())
