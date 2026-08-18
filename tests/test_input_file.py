from pathlib import Path
import re

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


def _replace_adsorption_path(source: str, role: str, path: Path) -> str:
    pattern = rf"(?m)^\s*data\s+{role}\s+.+$"
    return re.sub(pattern, f"    data {role} \"{path.as_posix()}\"", source)


def _set_first_atom_charge(source: str, charge: float) -> str:
    lines = source.splitlines()
    in_atoms = False
    for index, line in enumerate(lines):
        payload = line.split("#", 1)[0].strip()
        if payload == "Atoms":
            in_atoms = True
            continue
        if not in_atoms or not payload:
            continue
        fields = payload.split()
        if len(fields) >= 7 and fields[0].isdigit():
            fields[3] = str(charge)
            comment = f" # {line.split('#', 1)[1]}" if "#" in line else ""
            lines[index] = " ".join(fields) + comment
            break
    return "\n".join(lines) + "\n"


def test_targetless_adsorption_is_validation_only():
    project = load_project(ROOT / "examples" / "btah" / "charge_only.in")
    config = compose_config(project, "local")
    assert config["adsorption"]["enabled"] is False
    assert config["property_evaluators"]["adsorption"]["enabled"] is False
    assert config["validation"]["property_settings"]["adsorption"]["enabled"] is True
    assert config["validation"]["property_evaluators"]["adsorption"]["enabled"] is True
    assert "ead" not in config["targets"]


def test_adsorption_rejects_unknown_metal_label_during_check(tmp_path):
    source = (ROOT / "examples" / "btah" / "acceptance.in").read_text()
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    source = source.replace("    metal Au", "    metal Pt")
    path = tmp_path / "unknown_metal.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(InputFileError, match="metal label 'Pt' is absent from complex data"):
        compile_input(parse_input_file(path))


def test_adsorption_rejects_charged_fixed_metal_during_check(tmp_path):
    slab = ROOT / "data" / "adsorption" / "ad_slab.data"
    charged_slab = tmp_path / "charged_slab.data"
    charged_slab.write_text(
        _set_first_atom_charge(slab.read_text(encoding="utf-8"), 0.1),
        encoding="utf-8",
    )
    source = (ROOT / "examples" / "btah" / "acceptance.in").read_text()
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    source = _replace_adsorption_path(source, "slab", charged_slab)
    path = tmp_path / "charged_metal.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(InputFileError, match="metal label 'Au' must be uncharged in slab data"):
        compile_input(parse_input_file(path))


def test_adsorption_rejects_mismatched_molecular_labels_during_check(tmp_path):
    molecule = ROOT / "data" / "adsorption" / "ad_mol.data"
    mismatched_molecule = tmp_path / "mismatched_molecule.data"
    mismatched_molecule.write_text(
        molecule.read_text(encoding="utf-8").replace("# bhHn", "# otherH"),
        encoding="utf-8",
    )
    source = (ROOT / "examples" / "btah" / "acceptance.in").read_text()
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    source = _replace_adsorption_path(source, "molecule", mismatched_molecule)
    path = tmp_path / "mismatched_labels.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(
        InputFileError,
        match="complex molecular labels must match molecule data",
    ):
        compile_input(parse_input_file(path))


def test_bulk_protocol_is_the_fixed_standard(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "bad.in"
    path.write_text(source.replace("property bulk\n", "property bulk\n    protocol nvt\n"))
    with pytest.raises(InputFileError, match="does not use setting 'protocol'"):
        compile_input(parse_input_file(path))


def test_missing_free_range_has_source_error(tmp_path):
    source = (ROOT / "examples" / "btah" / "full.in").read_text()
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "bad.in"
    path.write_text(source.replace("    range sigma   factor 0.85 1.15\n", ""))
    with pytest.raises(InputFileError, match="free parameter 'sigma' has no range"):
        parse_input_file(path)


def test_validate_only_does_not_require_parameter_ranges(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace(
        "workflow bo sample nn al audit finalize validate", "workflow validate"
    )
    source = source.replace("    range charge  delta  0.30\n", "")
    source = source.replace("    fix epsilon sigma\n", "")
    for block in ("bo", "sample", "nn", "al", "audit"):
        source = re.sub(rf"\n{block}\n.*?\nend\n", "\n", source, flags=re.DOTALL)
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "validate_only.in"
    path.write_text(source, encoding="utf-8")

    compiled = compile_input(parse_input_file(path))
    assert compiled.dimensions == 0
    assert all(
        isinstance(value, float)
        for atom_type in compiled.config["atom_types"]
        for value in atom_type["params"].values()
    )


def test_rejects_workflow_stages_out_of_order(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace(
        "workflow bo sample nn al audit finalize validate",
        "workflow bo nn sample al audit finalize validate",
    )
    path = tmp_path / "bad_order.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(InputFileError, match="workflow stages must follow"):
        parse_input_file(path)


def test_arithmetic_sigma_maps_to_lammps_arithmetic_mixing(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace("mixing sigma geometric", "mixing sigma arithmetic")
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "arithmetic.in"
    path.write_text(source, encoding="utf-8")

    compiled = compile_input(parse_input_file(path))
    assert compiled.config["pair_params"]["mixing_rule"] == "arithmetic"


def test_charge_units_are_optional_and_limit_is_absolute():
    document = parse_input_file(ROOT / "examples" / "btah" / "charge_only.in")
    assert set(document.parameters.default_ranges) == {"charge"}
    charge_range = document.parameters.default_ranges["charge"]
    assert (charge_range.low, charge_range.high) == (-0.30, 0.30)
    assert document.parameters.charge_abs_max == 1.0


def test_schema_header_is_required(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    path = tmp_path / "missing_header.in"
    path.write_text(source.replace("ffopt 1\n", ""), encoding="utf-8")

    with pytest.raises(InputFileError, match="missing 'ffopt 1'"):
        parse_input_file(path)


def test_workflow_declaration_is_required(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = re.sub(r"^workflow .*\n", "", source, flags=re.MULTILINE)
    path = tmp_path / "missing_workflow.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(InputFileError, match="missing 'workflow STAGE"):
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


def test_rejects_nonfinite_force_field_values(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace("-0.609", "nan", 1)
    path = tmp_path / "nonfinite.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(InputFileError, match="charge must be finite"):
        parse_input_file(path)


def test_rejects_nonportable_type_label(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace("bhHn", "bh-Hn")
    path = tmp_path / "bad_label.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(InputFileError, match="type label must start with a letter"):
        parse_input_file(path)


def test_initial_validation_extracts_only_independent_parameters():
    compiled = compile_input(
        parse_input_file(ROOT / "examples" / "btah" / "charge_only.in")
    )
    initial = _initial_parameters(compiled.config)
    assert len(initial) == 13
    assert "bhN1_charge" not in initial
    assert initial["bhHn_charge"] == pytest.approx(0.430)


def test_bo_stability_audit_can_be_disabled_for_smoke_runs(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace("    stability_audit on\n", "    stability_audit off\n")
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "smoke.in"
    path.write_text(source, encoding="utf-8")

    compiled = compile_input(parse_input_file(path))
    assert compiled.config["optimization"]["stability_audit"]["enabled"] is False


def test_bo_stability_audit_budget_is_compiled_from_input():
    compiled = compile_input(
        parse_input_file(ROOT / "examples" / "btah" / "charge_only.in")
    )
    audit = compiled.config["optimization"]["stability_audit"]
    assert audit["enabled"] is True
    assert audit["top_k"] == 20
    assert audit["seeds"] == [101, 202, 303]


def test_single_sample_seed_is_normalized_to_a_list():
    compiled = compile_input(
        parse_input_file(ROOT / "examples" / "btah" / "acceptance.in")
    )
    assert compiled.project_data["stages"]["sample"]["seeds"] == [101]


def test_bo_batch_size_is_scientific_input_not_machine_parallelism():
    compiled = compile_input(
        parse_input_file(ROOT / "examples" / "btah" / "charge_only.in")
    )
    assert compiled.config["optimization"]["batch_size"] == 48


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("temperature 300 K", "temperature 300 C", "temperature unit"),
        ("cells_in_data 8 2 2", "replicate 8 2 2", "does not use setting 'replicate'"),
        ("temperature 298.15 K", "bulk_protocol npt", "does not use setting"),
        (
            "temperature 298.15 K",
            "temperature 298.15 K\n    molecule_atoms 14",
            "does not use setting 'molecule_atoms'",
        ),
        (
            "protocol minimize",
            "protocol minimize\n    temperature 300 K",
            "property 'adsorption' does not use setting 'temperature'",
        ),
    ],
)
def test_rejects_ambiguous_or_unused_property_settings(tmp_path, old, new, message):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace(old, new, 1)
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "bad_setting.in"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(InputFileError, match=message):
        parse_input_file(path)


def test_adsorption_runtime_keeps_inert_defaults_for_checkpoint_compatibility():
    compilation = compile_input(
        parse_input_file(ROOT / "examples" / "btah" / "acceptance.in")
    )
    md = compilation.config["adsorption"]["md"]
    assert {
        key: md[key]
        for key in ("temp", "timestep", "equil_steps", "prod_steps", "seed")
    } == {
        "temp": 300.0,
        "timestep": 1.0,
        "equil_steps": 20000,
        "prod_steps": 20000,
        "seed": 101,
    }


def test_accepts_legacy_validation_trajectory_switch_for_resume(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace(
        "\nvalidate\n",
        "\nvalidate\n    trajectory final\n",
        1,
    )
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "obsolete_trajectory.in"
    path.write_text(source, encoding="utf-8")

    compilation = compile_input(parse_input_file(path))
    assert compilation.config["validation"]["trajectory"] == "final"


def test_rejects_unknown_legacy_validation_trajectory_value(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace(
        "\nvalidate\n",
        "\nvalidate\n    trajectory none\n",
        1,
    )
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "bad_legacy_trajectory.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(InputFileError, match="legacy compatibility setting"):
        compile_input(parse_input_file(path))


def test_rejects_duplicate_property_setting(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace("    pressure 1 atm\n", "    pressure 1 atm\n    pressure 2 atm\n")
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "duplicate.in"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(InputFileError, match="duplicate 'pressure' setting"):
        parse_input_file(path)


@pytest.mark.parametrize(
    ("insertion", "message"),
    [
        ("    range charge delta 0.20\n", "duplicate default range"),
        ("    range type typo charge delta 0.10\n", "unknown type label 'typo'"),
        (
            "    range type bhN1 epsilon factor 0.9 1.1\n",
            "cannot be both fixed and ranged",
        ),
        ("    fix type typo charge\n", "fix refers to unknown type label 'typo'"),
        ("    charge_limit 1.1\n", "duplicate charge_limit setting"),
        ("    mixing sigma arithmetic\n", "duplicate mixing rule for 'sigma'"),
    ],
)
def test_rejects_ambiguous_parameter_overrides(tmp_path, insertion, message):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace("parameters\n", "parameters\n" + insertion, 1)
    path = tmp_path / "ambiguous_parameters.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(InputFileError, match=message):
        parse_input_file(path)


def test_duplicate_type_reports_the_second_source_line(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    duplicate = "    type  1  copied  0.2  3.2  0.0\n"
    source = source.replace("    type  1  bhN1", duplicate + "    type  1  bhN1", 1)
    path = tmp_path / "duplicate_type.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(InputFileError, match=r"duplicate type ID 1 \(first set on line"):
        parse_input_file(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("method ann", "method typo", "NN method must be"),
        ("ensemble 8", "ensemble 0", "NN ensemble_size must be positive"),
        (
            "    seeds 101 202 303",
            "    seeds 101.5 202",
            "sample seeds must be",
        ),
        (
            "acquisition uncertainty",
            "acquisition uncertainty\n    sampling_domain moon",
            "AL sampling_domain must be",
        ),
    ],
)
def test_rejects_invalid_stage_values_before_execution(tmp_path, old, new, message):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace(old, new)
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "bad_stage.in"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(InputFileError, match=message):
        compile_input(parse_input_file(path))


def test_stage_validation_reports_the_exact_input_line(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace("ensemble 8", "ensemble 0", 1)
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    expected_line = next(
        index
        for index, text in enumerate(source.splitlines(), start=1)
        if "ensemble 0" in text
    )
    path = tmp_path / "bad_ensemble.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(InputFileError) as captured:
        compile_input(parse_input_file(path))
    assert captured.value.line == expected_line


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("production 40000", "production 4999", "at least 5000"),
        ("pressure 1 atm", "pressure 1 atm\n    cutoff 0 A", "bulk cutoff"),
        ("method ann", "method ann\n    learning_rate nan", "learning_rate"),
        (
            "protocol minimize",
            "protocol minimize\n    cutoff 0 A",
            "adsorption cutoff",
        ),
    ],
)
def test_rejects_invalid_physical_run_values(tmp_path, old, new, message):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace(old, new, 1)
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "invalid_physical_value.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(InputFileError, match=message):
        compile_input(parse_input_file(path))


def test_optimization_rejects_all_zero_target_weights(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace("weight 1.0", "weight 0")
    source = source.replace("weight 0.5", "weight 0")
    source = source.replace("weight 0.3", "weight 0")
    path = tmp_path / "zero_weights.in"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(InputFileError, match="target with positive weight"):
        parse_input_file(path)


def test_relative_objective_rejects_zero_active_target(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace("target 98.5 kJ/mol", "target 0 kJ/mol")
    path = tmp_path / "zero_target.in"
    path.write_text(source, encoding="utf-8")
    with pytest.raises(InputFileError, match="nonzero active target"):
        parse_input_file(path)


def test_rejects_target_units_without_implemented_conversion(tmp_path):
    source = (ROOT / "examples" / "btah" / "charge_only.in").read_text()
    source = source.replace("target 98.5 kJ/mol", "target 1.02 eV")
    source = source.replace("../../data/", (ROOT / "data").as_posix() + "/")
    path = tmp_path / "bad_target_unit.in"
    path.write_text(source, encoding="utf-8")

    with pytest.raises(InputFileError, match="automatic unit conversion is not implemented"):
        compile_input(parse_input_file(path))
