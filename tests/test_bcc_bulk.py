from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from engine.config_loader import deep_merge
from engine.lammps_interface import LAMMPSRunner
from engine.model_adequacy import assess_model_adequacy
from workflow.artifact_manifest import scientific_config_hash
from workflow.defaults import machine_defaults
from workflow.input_compiler import compile_input
from workflow.input_file import InputFileError, parse_input_file
from workflow.pipeline import _scientific_config


def _data_text(type_count: int = 2) -> str:
    second_mass = "2 55.845 # Fe_body\n" if type_count >= 2 else ""
    second_atom = (
        "2 1 2 0.0 1.43325 1.43325 1.43325\n"
        if type_count >= 2
        else ""
    )
    third_mass = "3 55.845 # Fe_extra\n" if type_count == 3 else ""
    third_atom = "3 1 3 0.0 0.5 0.5 0.5\n" if type_count == 3 else ""
    atom_count = type_count
    return f"""BCC Fe

{atom_count} atoms
{type_count} atom types

0 14.3325 xlo xhi
0 14.3325 ylo yhi
0 14.3325 zlo zhi

Masses

1 55.845 # Fe_corner
{second_mass}{third_mass}
Atoms # full

1 1 1 0.0 0.0 0.0 0.0
{second_atom}{third_atom}"""


def _source(
    *,
    cells: str | None = "5 5 5",
    replicate: str | None = "2 2 2",
    tdamp: float = 75.0,
    pdamp: float = 750.0,
) -> str:
    cells_line = f"    cells_in_data {cells}\n" if cells is not None else ""
    replicate_line = f"    replicate {replicate}\n" if replicate is not None else ""
    return f"""ffopt 1
project fe_bulk
material elemental Fe
crystal bcc
workflow bo validate

parameters
    range epsilon absolute 0.001 10
    range sigma absolute 0.001 5
    mixing default
    tie epsilon all
    type 1 Fe_corner 6.0 2.3
    type 2 Fe_body 6.0 2.3
end

property bulk
    data data/fe.data
{cells_line}{replicate_line}    temperature 300 K
    pressure 1 atm
    timestep 1 fs
    tdamp {tdamp} fs
    pdamp {pdamp} fs
    equilibration 20000
    production 40000
    seed 101
    target a 2.8665 A tolerance 0.03
    target b 2.8665 A tolerance 0.03
    target c 2.8665 A tolerance 0.03
    target alpha 90 degree tolerance 1
    target beta 90 degree tolerance 1
    target gamma 90 degree tolerance 1
    target density 7.874 g/cm3 tolerance 0.08
end
"""


def _write(tmp_path: Path, source: str, *, type_count: int = 2) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir(exist_ok=True)
    (data_dir / "fe.data").write_text(
        _data_text(type_count), encoding="utf-8"
    )
    path = tmp_path / "ffopt.in"
    path.write_text(source, encoding="utf-8")
    return path


def _compiled(tmp_path: Path, source: str | None = None):
    return compile_input(parse_input_file(_write(tmp_path, source or _source())))


def _runtime_config(config: dict[str, Any]) -> dict[str, Any]:
    return deep_merge(config, machine_defaults("local"))


def _capture_bulk_run(runner: LAMMPSRunner, tmp_path: Path) -> dict[str, Any]:
    captured: dict[str, Any] = {}

    def fake_coefficients(_resolved: dict[str, float], directory: str) -> str:
        path = Path(directory) / "pair_coeffs.lmp"
        path.write_text("pair_coeff 1 1 6.0 2.3\n", encoding="utf-8")
        return str(path)

    def fake_run_one(**kwargs: Any) -> dict[str, float]:
        captured.update(kwargs)
        return {"a": 2.8665}

    runner._build_pair_coeffs = fake_coefficients  # type: ignore[method-assign]
    runner._run_one = fake_run_one  # type: ignore[method-assign]
    result = runner._run_bulk({}, str(tmp_path / "evaluation"))
    assert result == {"a": 2.8665}
    return captured


def test_bcc_bulk_compiles_distinct_input_and_runtime_geometry(
    tmp_path: Path,
) -> None:
    compiled = _compiled(tmp_path)
    config = compiled.config
    bulk = config["lammps"]["bulk"]

    assert bulk == {
        "nx": 10,
        "ny": 10,
        "nz": 10,
        "npt_seed": 101,
        "equil_steps": 20000,
        "prod_steps": 40000,
        "temperature": 300.0,
        "pressure": 1.0,
        "protocol": "box_relax_0k+tri_npt",
        "cells_in_data": [5, 5, 5],
        "replicate": [2, 2, 2],
        "effective_cells": [10, 10, 10],
        "tdamp": 75.0,
        "pdamp": 750.0,
        "runtime_replicate": True,
    }
    bulk_input = Path(config["lammps"]["bulk_input"])
    assert bulk_input.name == "in.bcc.elemental"
    executable_lines = [
        line.strip()
        for line in bulk_input.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    assert executable_lines.count(
        "replicate       ${replicate_x} ${replicate_y} ${replicate_z}"
    ) == 1
    assert "variable inv_nx equal 1.0/${nx}" in executable_lines

    assert compiled.material["force_field"] == "lj/cut"
    assert compiled.material["pair_style"] == "lj/cut"
    assert config["lammps"]["pair_style"] == "lj/cut"
    assert config["pair_params"]["pair_style"] == "lj/cut"
    adequacy = assess_model_adequacy(config)
    assert adequacy["model_form"]["pair_style"] == "lj/cut"
    assert adequacy["model_form"]["central_lj_pair_style"] is True
    assert adequacy["model_form"]["pair_style_source"] == (
        "pair_params.pair_style,lammps.pair_style,material.force_field"
    )


def test_bcc_runner_passes_one_replicate_and_effective_divisors(
    tmp_path: Path,
) -> None:
    compiled = _compiled(tmp_path)
    runner = LAMMPSRunner(
        _runtime_config(compiled.config), start_scheduler_pool=False
    )
    captured = _capture_bulk_run(runner, tmp_path)
    variables = captured["extra_vars"]

    assert runner.bulk_cells_in_data == (5, 5, 5)
    assert runner.bulk_replicate == (2, 2, 2)
    assert runner.bulk_effective_cells == (10, 10, 10)
    assert (variables["nx"], variables["ny"], variables["nz"]) == (10, 10, 10)
    assert (
        variables["replicate_x"],
        variables["replicate_y"],
        variables["replicate_z"],
    ) == (2, 2, 2)
    assert variables["tdamp"] == 75.0
    assert variables["pdamp"] == 750.0

    # Replication belongs to the LAMMPS template. The copied input structure
    # remains the original two-atom file, so Python cannot pre-replicate it a
    # second time.
    copied_data = tmp_path / "evaluation" / "bulk" / "fe.data"
    assert "2 atoms" in copied_data.read_text(encoding="utf-8")


def test_bcc_runtime_replicate_is_optional_and_defaults_to_identity(
    tmp_path: Path,
) -> None:
    compiled = _compiled(tmp_path, _source(replicate=None))
    bulk = compiled.config["lammps"]["bulk"]

    assert bulk["cells_in_data"] == [5, 5, 5]
    assert bulk["replicate"] == [1, 1, 1]
    assert bulk["effective_cells"] == [5, 5, 5]
    assert (bulk["nx"], bulk["ny"], bulk["nz"]) == (5, 5, 5)


def test_bcc_geometry_and_damping_are_part_of_the_scientific_hash(
    tmp_path: Path,
) -> None:
    path = _write(tmp_path, _source())
    baseline = compile_input(parse_input_file(path))
    baseline_hash = scientific_config_hash(_scientific_config(baseline.config))

    path.write_text(_source(replicate="1 2 2", tdamp=80.0), encoding="utf-8")
    changed = compile_input(parse_input_file(path))
    changed_hash = scientific_config_hash(_scientific_config(changed.config))

    assert changed_hash != baseline_hash


def test_bcc_bulk_requires_explicit_cells_in_data(tmp_path: Path) -> None:
    path = _write(tmp_path, _source(cells=None))
    property_line = next(
        index
        for index, line in enumerate(path.read_text().splitlines(), 1)
        if line.strip() == "property bulk"
    )

    with pytest.raises(InputFileError, match="BCC bulk requires explicit") as exc:
        parse_input_file(path)
    assert exc.value.line == property_line


@pytest.mark.parametrize(
    ("key", "source", "message"),
    [
        ("replicate", _source(replicate="2 0 2"), "replicate values must be positive"),
        ("tdamp", _source(tdamp=0.0), "tdamp must be positive"),
        ("pdamp", _source(pdamp=-1.0), "pdamp must be positive"),
    ],
)
def test_bcc_bulk_geometry_and_damping_fail_on_the_declaration_line(
    tmp_path: Path,
    key: str,
    source: str,
    message: str,
) -> None:
    path = _write(tmp_path, source)
    expected_line = next(
        index
        for index, line in enumerate(path.read_text().splitlines(), 1)
        if line.strip().startswith(key)
    )

    with pytest.raises(InputFileError, match=message) as exc:
        compile_input(parse_input_file(path))
    assert exc.value.line == expected_line


@pytest.mark.parametrize("type_count", [1, 3])
def test_elemental_primary_data_type_ids_must_exactly_match_parameter_table(
    tmp_path: Path,
    type_count: int,
) -> None:
    path = _write(tmp_path, _source(), type_count=type_count)
    material_line = next(
        index
        for index, line in enumerate(path.read_text().splitlines(), 1)
        if line.startswith("material ")
    )

    with pytest.raises(InputFileError, match="must exactly match") as exc:
        compile_input(parse_input_file(path))
    assert exc.value.line == material_line


def test_runner_rejects_tampered_effective_cell_metadata(tmp_path: Path) -> None:
    compiled = _compiled(tmp_path)
    config = _runtime_config(compiled.config)
    config["lammps"]["bulk"]["effective_cells"] = [5, 5, 5]

    with pytest.raises(ValueError, match=r"cells_in_data \* replicate"):
        LAMMPSRunner(config, start_scheduler_pool=False)


def test_legacy_molecular_bulk_shape_and_runner_variables_are_unchanged(
    tmp_path: Path,
) -> None:
    root = Path(__file__).resolve().parents[1]
    compiled = compile_input(
        parse_input_file(root / "examples" / "btah" / "charge_only.in")
    )
    bulk = compiled.config["lammps"]["bulk"]
    assert bulk == {
        "nx": 8,
        "ny": 2,
        "nz": 2,
        "npt_seed": 101,
        "equil_steps": 20000,
        "prod_steps": 40000,
        "temperature": 300.0,
        "pressure": 1.0,
        "protocol": "minimize+npt",
    }
    assert Path(compiled.config["lammps"]["bulk_input"]).name == "in.bulk.mol"
    assert "pair_style" not in compiled.config["lammps"]
    assert "pair_style" not in compiled.config["pair_params"]
    assert "material" not in compiled.config

    runner = LAMMPSRunner(
        _runtime_config(compiled.config), start_scheduler_pool=False
    )
    captured = _capture_bulk_run(runner, tmp_path)
    variables = captured["extra_vars"]
    assert runner.bulk_runtime_replicate is False
    assert runner.bulk_effective_cells == (8, 2, 2)
    assert not {
        "replicate_x", "replicate_y", "replicate_z", "tdamp", "pdamp"
    } & set(variables)
