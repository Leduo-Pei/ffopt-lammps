from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

import engine.local_sampling as local_sampling
import engine.stability_audit as stability_audit
from engine.local_sampling import (
    completed_candidate_ids,
    generate_design,
    read_candidate_source,
    sampling_allocation,
)
from engine.stability_audit import load_audit_sources, unique_top


class _FeasibleRunner:
    @staticmethod
    def feasibility_error(_parameters: dict[str, float]) -> None:
        return None


def _source(rows: list[tuple[float, float, float]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {"x": x, "y": y, "objective": objective, "success": True}
            for x, y, objective in rows
        ]
    )


def test_material_sampling_honors_local_boundary_global_quotas() -> None:
    interior = _source([
        (0.25, 0.25, 0.01),
        (0.75, 0.75, 0.02),
    ])
    boundary = _source([
        (0.05, 0.50, 0.03),
        (0.95, 0.50, 0.04),
    ])
    boundary["anchor_class"] = "near_boundary"
    first = generate_design(
        interior,
        [("x", 0.0, 1.0), ("y", 0.0, 1.0)],
        _FeasibleRunner(),
        10,
        2,
        [0.02, 0.05],
        0.20,
        20260821,
        "diverse",
        5,
        boundary,
        0.30,
    )
    second = generate_design(
        interior,
        [("x", 0.0, 1.0), ("y", 0.0, 1.0)],
        _FeasibleRunner(),
        10,
        2,
        [0.02, 0.05],
        0.20,
        20260821,
        "diverse",
        5,
        boundary,
        0.30,
    )
    assert sampling_allocation(10, 0.20, 0.30) == {
        "local_elite": 5,
        "boundary_local": 3,
        "global_sobol": 2,
    }
    assert first["sampling_mode"].value_counts().to_dict() == {
        "local_elite": 5,
        "boundary_local": 3,
        "global_sobol": 2,
    }
    pd.testing.assert_frame_equal(first, second)


def test_missing_boundary_centers_are_explicitly_reallocated() -> None:
    design = generate_design(
        _source([(0.25, 0.25, 0.01), (0.75, 0.75, 0.02)]),
        [("x", 0.0, 1.0), ("y", 0.0, 1.0)],
        _FeasibleRunner(),
        10,
        2,
        [0.02],
        0.20,
        17,
        boundary_source=pd.DataFrame(
            columns=["x", "y", "objective", "success"]
        ),
        boundary_fraction=0.30,
    )
    assert design["sampling_mode"].value_counts().to_dict() == {
        "local_elite": 8,
        "global_sobol": 2,
    }


def test_audit_unions_sources_by_parameters_and_preserves_provenance(
    tmp_path: Path,
) -> None:
    bo = tmp_path / "bo.csv"
    sample = tmp_path / "sample.csv"
    pd.DataFrame([
        {"candidate_id": 7, "x": 0.2, "y": 0.3, "objective": 0.04,
         "success": True},
        {"candidate_id": 8, "x": 0.8, "y": 0.7, "objective": 0.03,
         "success": True},
    ]).to_csv(bo, index=False)
    pd.DataFrame([
        {"candidate_id": 2, "x": 0.2, "y": 0.3, "objective": 0.01,
         "success": True},
        {"candidate_id": 3, "x": 0.5, "y": 0.6, "objective": 0.02,
         "success": True},
    ]).to_csv(sample, index=False)

    combined = load_audit_sources([bo, sample], ["x", "y"])
    design = unique_top(combined, ["x", "y"], 3)

    assert list(design[["x", "y"]].itertuples(index=False, name=None)) == [
        (0.2, 0.3),
        (0.5, 0.6),
        (0.8, 0.7),
    ]
    assert design.loc[0, "source_path"] == str(sample.resolve())
    assert int(design.loc[0, "source_candidate_id"]) == 2
    assert design["candidate_id"].tolist() == [1, 2, 3]


def test_empty_candidate_archive_is_a_valid_empty_source(tmp_path: Path) -> None:
    path = tmp_path / "empty.csv"
    path.write_bytes(b"")
    frame = read_candidate_source(path, ["x", "y"])
    assert frame.empty
    assert {"x", "y", "objective"}.issubset(frame.columns)


def test_completed_candidates_require_the_exact_seed_set() -> None:
    replicates = pd.DataFrame([
        {"candidate_id": 1, "seed": 101},
        {"candidate_id": 1, "seed": 202},
        {"candidate_id": 2, "seed": 404},
        {"candidate_id": 2, "seed": 505},
    ])
    assert completed_candidate_ids(replicates, [101, 202]) == {1}
    assert completed_candidate_ids(replicates, [404, 505]) == {2}
    assert completed_candidate_ids(replicates, [101, 303]) == set()


def _patch_design_only_runtime(monkeypatch, module) -> None:
    monkeypatch.setattr(
        module,
        "load_config",
        lambda _path: {"targets": {}, "parallel": {"max_workers": 1}},
    )
    monkeypatch.setattr(module, "LAMMPSRunner", lambda _config: _FeasibleRunner())
    monkeypatch.setattr(
        module,
        "build_parameter_space",
        lambda _config: [("x", 0.0, 1.0), ("y", 0.0, 1.0)],
    )


def test_sampling_resume_rejects_changed_source_content(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_design_only_runtime(monkeypatch, local_sampling)
    config = tmp_path / "runtime.json"
    config.write_text("{}", encoding="utf-8")
    source = tmp_path / "feasible.csv"
    boundary = tmp_path / "boundary.csv"
    output = tmp_path / "sample"
    _source([(0.25, 0.25, 0.01), (0.75, 0.75, 0.02)]).to_csv(
        source, index=False
    )
    boundary_frame = _source([(0.05, 0.5, 0.03), (0.95, 0.5, 0.04)])
    boundary_frame["anchor_class"] = "near_boundary"
    boundary_frame.to_csv(boundary, index=False)
    arguments = [
        "local_sampling",
        "--config", str(config),
        "--source", str(source),
        "--boundary-source", str(boundary),
        "--output-dir", str(output),
        "--n-points", "4",
        "--elite-centers", "2",
        "--radii", "0.02",
        "--global-fraction", "0.25",
        "--boundary-fraction", "0.25",
        "--design-seed", "9",
        "--design-only",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    local_sampling.main()
    local_sampling.main()

    changed = pd.read_csv(source)
    changed.loc[0, "objective"] = 0.009
    changed.to_csv(source, index=False)
    with pytest.raises(ValueError, match="does not match the current config"):
        local_sampling.main()


def test_audit_resume_allows_short_top_k_but_rejects_changed_sources(
    tmp_path: Path, monkeypatch
) -> None:
    _patch_design_only_runtime(monkeypatch, stability_audit)
    config = tmp_path / "runtime.json"
    config.write_text("{}", encoding="utf-8")
    source = tmp_path / "source.csv"
    output = tmp_path / "audit"
    _source([(0.2, 0.3, 0.01), (0.8, 0.7, 0.02)]).to_csv(
        source, index=False
    )
    arguments = [
        "stability_audit",
        "--config", str(config),
        "--source", str(source),
        "--output-dir", str(output),
        "--top-k", "4",
        "--design-only",
    ]
    monkeypatch.setattr(sys, "argv", arguments)
    stability_audit.main()
    stability_audit.main()
    assert len(pd.read_csv(output / "design.csv")) == 2

    changed = pd.read_csv(source)
    changed.loc[0, "objective"] = 0.009
    changed.to_csv(source, index=False)
    with pytest.raises(ValueError, match="does not match the current config"):
        stability_audit.main()
