from __future__ import annotations

import math
from pathlib import Path

import pytest

from workflow.lammps_data import inspect_lammps_data


ROOT = Path(__file__).resolve().parents[1]


def test_inspect_btah_bulk_data() -> None:
    summary = inspect_lammps_data(ROOT / "data/bulk/BTAH_822_bulk.data")
    assert summary.atom_style == "full"
    assert summary.declared_counts["atoms"] == 4480
    assert summary.declared_counts["atom_types"] == 14
    assert summary.molecule_count == 320
    assert len(summary.atom_types) == 14
    assert all(item.atom_count == 320 for item in summary.atom_types)
    assert summary.atom_types[0].label == "bhN1"
    assert math.isclose(summary.atom_types[0].pair_coefficients[0], 0.2, abs_tol=2.0e-8)
    assert math.isclose(summary.atom_types[0].pair_coefficients[1], 3.2963252376)
    assert math.isclose(summary.total_charge or 0.0, 0.0, abs_tol=1.0e-10)
    assert summary.box is not None
    assert summary.box.representation == "restricted_triclinic"
    expected_vectors = (
        (33.9376, 0.0, 0.0),
        (2.344301974, 35.576845901, 0.0),
        (2.056973237, 12.241887901, 39.463678766),
    )
    assert summary.box_vectors is not None
    for observed, expected in zip(summary.box_vectors, expected_vectors):
        assert observed == pytest.approx(expected)
    assert all(height > 0.0 for height in summary.periodic_face_heights())


def _box_data(header: str) -> str:
    return f"""LAMMPS box test

1 atoms
1 atom types

{header}

Masses

1 1.0 # X

Atoms # atomic

1 1 0.0 0.0 0.0
"""


def test_orthogonal_box_vectors_and_replicated_face_heights(tmp_path: Path) -> None:
    path = tmp_path / "orthogonal.data"
    path.write_text(
        _box_data(
            "0.0 10.0 xlo xhi\n"
            "-1.0 19.0 ylo yhi\n"
            "2.0 32.0 zlo zhi"
        ),
        encoding="utf-8",
    )

    summary = inspect_lammps_data(path)

    assert summary.box is not None
    assert summary.box.representation == "orthogonal"
    assert summary.box.origin == (0.0, -1.0, 2.0)
    assert summary.box_vectors == (
        (10.0, 0.0, 0.0),
        (0.0, 20.0, 0.0),
        (0.0, 0.0, 30.0),
    )
    assert summary.box.volume == pytest.approx(6000.0)
    assert summary.periodic_face_heights() == pytest.approx((10.0, 20.0, 30.0))
    assert summary.replicated_box_vectors((2, 3, 4)) == (
        (20.0, 0.0, 0.0),
        (0.0, 60.0, 0.0),
        (0.0, 0.0, 120.0),
    )
    assert summary.periodic_face_heights((2, 3, 4)) == pytest.approx(
        (20.0, 60.0, 120.0)
    )
    document = summary.to_dict()
    assert document["box"]["periodic_face_heights"] == pytest.approx(
        [10.0, 20.0, 30.0]
    )


def test_restricted_triclinic_face_heights_scale_by_replicate(tmp_path: Path) -> None:
    path = tmp_path / "triclinic.data"
    path.write_text(
        _box_data(
            "0.0 10.0 xlo xhi\n"
            "0.0 8.0 ylo yhi\n"
            "0.0 6.0 zlo zhi\n"
            "2.0 -1.0 1.5 xy xz yz"
        ),
        encoding="utf-8",
    )

    summary = inspect_lammps_data(path)

    assert summary.box is not None
    assert summary.box.representation == "restricted_triclinic"
    assert summary.box_vectors == (
        (10.0, 0.0, 0.0),
        (2.0, 8.0, 0.0),
        (-1.0, 1.5, 6.0),
    )
    assert summary.box.volume == pytest.approx(480.0)
    base_heights = (
        480.0 / math.sqrt(2569.0),
        480.0 / math.sqrt(3825.0),
        6.0,
    )
    assert summary.periodic_face_heights() == pytest.approx(base_heights)
    assert summary.periodic_face_heights((2, 3, 4)) == pytest.approx(
        tuple(scale * height for scale, height in zip((2, 3, 4), base_heights))
    )


@pytest.mark.parametrize(
    "replicate", [(0, 1, 1), (1, 2), (1.0, 1, 1), (True, 1, 1)]
)
def test_periodic_face_heights_reject_invalid_replicates(
    tmp_path: Path, replicate: tuple[object, ...]
) -> None:
    path = tmp_path / "box.data"
    path.write_text(
        _box_data(
            "0.0 10.0 xlo xhi\n"
            "0.0 10.0 ylo yhi\n"
            "0.0 10.0 zlo zhi"
        ),
        encoding="utf-8",
    )
    summary = inspect_lammps_data(path)

    with pytest.raises(ValueError, match="three positive integers"):
        summary.periodic_face_heights(replicate)  # type: ignore[arg-type]


def test_incomplete_box_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "incomplete.data"
    path.write_text(
        _box_data("0.0 10.0 xlo xhi\n0.0 10.0 ylo yhi"),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="incomplete LAMMPS box.*z"):
        inspect_lammps_data(path)


def test_legacy_summary_without_box_remains_readable(tmp_path: Path) -> None:
    path = tmp_path / "legacy-no-box.data"
    path.write_text(_box_data(""), encoding="utf-8")

    summary = inspect_lammps_data(path)

    assert summary.box is None
    assert summary.box_vectors is None
    assert summary.to_dict()["box"] is None
    with pytest.raises(ValueError, match="contains no complete box"):
        summary.periodic_face_heights()
