from __future__ import annotations

from pathlib import Path

import pandas as pd

from engine.local_sampling import generate_design, sampling_allocation
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
