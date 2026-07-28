from pathlib import Path

import pytest

from engine.validate_final_parameters import _initial_parameters
from workflow.input_compiler import compile_input
from workflow.input_file import InputFileError, parse_input_file
from workflow.project import compose_config, load_project

ROOT = Path(__file__).resolve().parents[1]


def test_btah_charge_only_and_full_dimensions():
    charge = compile_input(parse_input_file(ROOT / "examples" / "btah" / "charge_only.in"))
    full = compile_input(parse_input_file(ROOT / "examples" / "btah" / "full.in"))
    assert charge.dimensions == 13
    assert full.dimensions == 41
    assert charge.fitted_properties == ("bulk", "sublimation")
    assert charge.validation_properties == ("adsorption",)


def test_targetless_adsorption_is_validation_only():
    project = load_project(ROOT / "examples" / "btah" / "charge_only.in")
    config = compose_config(project, "local")
    assert config["adsorption"]["enabled"] is False
    assert config["property_evaluators"]["adsorption"]["enabled"] is False
    assert config["validation"]["property_settings"]["adsorption"]["enabled"] is True
    assert config["validation"]["property_evaluators"]["adsorption"]["enabled"] is True
    assert "ead" not in config["targets"]


def test_bulk_protocol_is_the_fixed_standard(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "bad.in"
    path.write_text(source.replace("property bulk\n", "property bulk\n    protocol nvt\n"))
    with pytest.raises(InputFileError, match="fixed standard minimize.*NPT"):
        compile_input(parse_input_file(path))


def test_missing_free_range_has_source_error(tmp_path):
    source = (ROOT / "examples" / "btah" / "full.in").read_text()
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "bad.in"
    path.write_text(source.replace("    range sigma   factor 0.85 1.15\n", ""))
    with pytest.raises(InputFileError, match="free parameter 'sigma' has no range"):
        parse_input_file(path)


def test_rejects_workflow_stages_out_of_order(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace(
        "workflow bo sample nn al validate",
        "workflow bo nn sample al validate",
    )
    path = tmp_path / "bad_order.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(InputFileError, match="workflow stages must follow"):
        parse_input_file(path)


def test_arithmetic_sigma_maps_to_lammps_arithmetic_mixing(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace("mixing geometric", "mixing arithmetic")
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "arithmetic.in"
    path.write_text(source, encoding="utf-8")

    compiled = compile_input(parse_input_file(path))
    assert compiled.config["pair_params"]["mixing_rule"] == "arithmetic"


def test_charge_units_are_optional_and_limit_is_absolute():
    document = parse_input_file(ROOT / "examples" / "btah" / "charge_only.in")
    charge_range = document.parameters.default_ranges["charge"]
    assert (charge_range.low, charge_range.high) == (-0.30, 0.30)
    assert document.parameters.charge_abs_max == 1.0


def test_schema_header_is_required(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    path = tmp_path / "missing_header.in"
    path.write_text(source.replace("ffopt 1\n", ""), encoding="utf-8")

    with pytest.raises(InputFileError, match="missing 'ffopt 1'"):
        parse_input_file(path)


def test_charge_limit_rejects_unknown_unit(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    path = tmp_path / "bad_charge_unit.in"
    path.write_text(
        source.replace("charge_limit 1.0", "charge_limit 1.0 volt"),
        encoding="utf-8",
    )

    with pytest.raises(InputFileError, match="optional unit must be e"):
        parse_input_file(path)


def test_initial_validation_extracts_only_independent_parameters():
    compiled = compile_input(
        parse_input_file(ROOT / "examples" / "btah" / "charge_only.in")
    )
    initial = _initial_parameters(compiled.config)
    assert len(initial) == 13
    assert "bhN1_charge" not in initial
    assert initial["bhHn_charge"] == pytest.approx(0.430)
