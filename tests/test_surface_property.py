from __future__ import annotations

import math
from pathlib import Path

import pytest

from engine.config_loader import deep_merge
from engine.property_contracts import PropertyCostClass, PropertyRole
from engine.property_evaluators import (
    PROPERTY_EVALUATORS,
    PropertyEvaluationContext,
    SurfaceEvaluator,
)
from workflow.input_compiler import compile_input
from workflow.input_file import InputFileError, parse_input_file


def _data_text(
    *,
    atom_count: int = 4,
    first_count: int = 2,
    first_label: str = "Fe_c",
    second_label: str = "Fe_b",
) -> str:
    second_count = atom_count - first_count
    atoms = []
    atom_id = 1
    for type_id, count, z_offset in ((1, first_count, 0.0), (2, second_count, 0.5)):
        for index in range(count):
            atoms.append(
                f"{atom_id} 1 {type_id} 0.0 {index * 0.5:.3f} 0.0 "
                f"{z_offset:.3f} 0 0 0"
            )
            atom_id += 1
    return f"""BCC Fe test slab

{atom_count} atoms
2 atom types

0.0 10.0 xlo xhi
0.0 10.0 ylo yhi
0.0 20.0 zlo zhi

Masses

1 55.845 # {first_label}
2 55.845 # {second_label}

Pair Coeffs # lj/cut

1 1.0 2.0 # {first_label}
2 1.0 2.0 # {second_label}

Atoms # full

{chr(10).join(atoms)}
"""


def _write_surface_project(
    tmp_path: Path,
    *,
    target: bool = True,
    facet: str = "110",
    complete_text: str | None = None,
    split_text: str | None = None,
) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "complete.data").write_text(
        complete_text or _data_text(), encoding="utf-8"
    )
    (data_dir / "split.data").write_text(
        split_text or _data_text(), encoding="utf-8"
    )
    workflow = "bo validate" if target else "validate"
    ranges = (
        "    range epsilon absolute 0.001 10.0\n"
        "    range sigma absolute 0.001 5.0\n"
        if target
        else ""
    )
    target_line = (
        "    target 2.340 J/m2 weight 1.0 tolerance 0.117\n"
        if target
        else ""
    )
    source = f"""ffopt 1
project fe_surface
material elemental Fe
crystal bcc
workflow {workflow}

parameters
{ranges}    mixing default
    type 1 Fe_corner 6.5 2.3
    type 2 Fe_body 6.5 2.3
end

property surface
    complete data/complete.data
    split data/split.data
    facet {facet}
    replicate 2 2 1
{target_line}end
"""
    path = tmp_path / "ffopt.in"
    path.write_text(source, encoding="utf-8")
    return path


def test_public_bcc110_surface_compiles_as_fitted_constraint(tmp_path: Path) -> None:
    compiled = compile_input(parse_input_file(_write_surface_project(tmp_path)))
    config = compiled.config

    assert compiled.fitted_properties == ("surface",)
    assert compiled.validation_properties == ()
    assert config["manifest"]["surface_facet"] == "110"
    assert Path(config["manifest"]["data_files"]["surf_complete"]).is_file()
    assert Path(config["manifest"]["data_files"]["surf_split"]).is_file()
    assert config["lammps"]["compute_surface"] is True
    assert config["lammps"]["surf"]["replicate"] == [2, 2, 1]
    assert config["lammps"]["surf"]["protocol"] == "minimize_0k"
    surf_input = Path(config["lammps"]["surf_input"])
    assert surf_input.name == "in.bcc.surface"
    assert "replicate ${replicate_x} ${replicate_y} ${replicate_z}" in (
        surf_input.read_text(encoding="utf-8")
    )
    assert config["targets"]["surf_energy"] == {
        "value": pytest.approx(2.34),
        "weight": pytest.approx(1.0),
        "unit": "J/m2",
        "tolerance": pytest.approx(0.117),
    }


def test_targetless_surface_is_enabled_only_by_validation_overlay(tmp_path: Path) -> None:
    compiled = compile_input(
        parse_input_file(_write_surface_project(tmp_path, target=False))
    )
    config = compiled.config

    assert compiled.fitted_properties == ()
    assert compiled.validation_properties == ("surface",)
    assert config["lammps"]["compute_surface"] is False
    assert config["property_evaluators"]["surface"]["enabled"] is False
    assert config["validation"]["property_evaluators"]["surface"]["enabled"] is True
    assert config["validation"]["property_settings"]["lammps"] == {
        "compute_surface": True,
    }
    assert "surf_energy" not in config["targets"]

    validation = config["validation"]
    overrides = {
        "property_evaluators": validation["property_evaluators"],
        **validation["property_settings"],
    }
    validation_config = deep_merge(config, overrides)
    assert validation_config["lammps"]["compute_surface"] is True
    assert Path(validation_config["lammps"]["surf_input"]).is_file()
    assert validation_config["manifest"]["data_files"]["surf_complete"].endswith(
        "complete.data"
    )


@pytest.mark.parametrize("facet", ["100", "111", "fcc110"])
def test_surface_rejects_every_facet_except_bcc110(
    tmp_path: Path, facet: str
) -> None:
    path = _write_surface_project(tmp_path, facet=facet)
    with pytest.raises(InputFileError, match="supports facet 110 only"):
        compile_input(parse_input_file(path))


def test_surface_rejects_unequal_atom_counts(tmp_path: Path) -> None:
    path = _write_surface_project(
        tmp_path,
        split_text=_data_text(atom_count=6, first_count=3),
    )
    with pytest.raises(InputFileError, match="equal atom counts"):
        compile_input(parse_input_file(path))


def test_surface_rejects_internally_inconsistent_atom_header(tmp_path: Path) -> None:
    malformed = _data_text().replace("4 atoms", "5 atoms", 1)
    path = _write_surface_project(tmp_path, split_text=malformed)
    with pytest.raises(InputFileError, match="atom header .* does not match"):
        compile_input(parse_input_file(path))


def test_surface_rejects_equal_total_with_different_type_composition(
    tmp_path: Path,
) -> None:
    path = _write_surface_project(
        tmp_path,
        split_text=_data_text(atom_count=4, first_count=3),
    )
    with pytest.raises(InputFileError, match="identical per-type composition"):
        compile_input(parse_input_file(path))


def test_surface_rejects_type_label_reordering(tmp_path: Path) -> None:
    path = _write_surface_project(
        tmp_path,
        split_text=_data_text(first_label="Fe_b", second_label="Fe_c"),
    )
    with pytest.raises(InputFileError, match="type labels must match"):
        compile_input(parse_input_file(path))


def test_surface_evaluator_contract_and_replicate_variables() -> None:
    contract = PROPERTY_EVALUATORS.contract("surface", discover=False)
    assert contract.roles == frozenset({
        PropertyRole.CONSTRAINT,
        PropertyRole.VALIDATION,
    })
    assert contract.fidelity == "structural"
    assert contract.cost_class is PropertyCostClass.LOW

    class Runner:
        cutoff = 8.0
        surf_replicate = (2, 3, 1)
        use_charge = False
        surf_complete = "complete.data"
        surf_split = "split.data"
        surf_E_key = "energy"
        surf_A_key = "area"
        surf_energy_min = 0.1
        surf_energy_max = 10.0

        def __init__(self) -> None:
            self.variables: dict[str, dict[str, int | float]] = {}

        def _run_surf(self, resolved, eval_dir, tag, path, variables):
            self.variables[tag] = dict(variables)
            if tag == "complete":
                return {"energy": 0.0, "area": 10.0}
            return {"energy": 10.0, "area": 10.0}

    runner = Runner()
    result = SurfaceEvaluator().evaluate(
        runner,
        PropertyEvaluationContext({}, "evaluation"),
        {},
    )
    assert result.success
    assert math.isclose(result.properties["surf_energy"], 0.347385)
    assert runner.variables["complete"] == {
        "cutoff": 8.0,
        "replicate_x": 2,
        "replicate_y": 3,
        "replicate_z": 1,
        "save_traj": 0,
    }
    assert runner.variables["split"] == runner.variables["complete"]
