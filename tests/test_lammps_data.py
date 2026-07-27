from __future__ import annotations

import math
from pathlib import Path

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
