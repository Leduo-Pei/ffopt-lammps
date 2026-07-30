from pathlib import Path

from workflow.cli import _free_parameter_count
from workflow.project import compose_config, load_project
from workflow.scaffold import TargetSpec, create_project, parse_target_spec

ROOT = Path(__file__).resolve().parents[1]


def test_target_cli_format():
    target = parse_target_spec("density=1.3285,0.5,g/cm3")
    assert target.name == "density"
    assert target.value == 1.3285
    assert target.weight == 0.5
    assert target.unit == "g/cm3"


def test_sublimation_target_alias():
    target = parse_target_spec("sublimation=98.5,0.3,kJ/mol")
    assert target.name == "esub_proxy"


def test_charge_only_btah_scaffold_is_external_and_13_dimensional(tmp_path):
    output = tmp_path / "molecule_project"
    result = create_project(
        name="btah_external",
        data_file=ROOT / "data" / "bulk" / "BTAH_822_bulk.data",
        destination=output,
        targets=[TargetSpec("density", 1.3285, unit="g/cm3")],
        mode="charge_only",
        cells=(8, 2, 2),
    )
    assert result.dimensions == 13
    assert (output / "data" / "bulk" / "BTAH_822_bulk.data").exists()
    assert (output / "ffopt.in").exists()
    assert not (output / "configs").exists()
    assert not (output / "project.yaml").exists()

    project = load_project(output / "ffopt.in")
    config = compose_config(project, "local")
    assert _free_parameter_count(config) == 13
    assert config["optimization"]["method"] == "auto"
    assert config["optimization"]["stability_audit"].get("max_workers") is None
    assert config["nn"]["exclude_properties"] == []
    assert config["targets"]["density"]["value"] == 1.3285
    assert config["charge"]["neutrality_constraint"]["enabled"]
    assert config["lammps"]["bulk"]["nx"] == 8
    assert isinstance(config["atom_types"][0]["params"]["epsilon"], float)
    assert isinstance(config["atom_types"][0]["params"]["sigma"], float)
    assert isinstance(config["atom_types"][1]["params"]["charge"], dict)


def test_combined_scaffold_adds_targetless_adsorption_validation(tmp_path):
    output = tmp_path / "combined"
    result = create_project(
        name="btah_combined",
        data_file=ROOT / "data" / "bulk" / "BTAH_822_bulk.data",
        single_data=ROOT / "data" / "molecule" / "BTAH_822_single.data",
        complex_data=ROOT / "data" / "adsorption" / "ad_complex.data",
        slab_data=ROOT / "data" / "adsorption" / "ad_slab.data",
        molecule_data=ROOT / "data" / "adsorption" / "ad_mol.data",
        destination=output,
        targets=[TargetSpec("density", 1.3285, unit="g/cm3")],
        project_type="auto",
        mode="charge_only",
    )
    project = load_project(result.project_file)
    config = compose_config(project, "local")
    assert result.dimensions == 13
    assert project.compilation.fitted_properties == ("bulk",)
    assert project.compilation.validation_properties == ("sublimation", "adsorption")
    assert config["validation"]["property_evaluators"]["adsorption"]["enabled"]
    assert (output / "data" / "adsorption" / "ad_complex.data").exists()


def test_targetless_adsorption_scaffold_creates_validate_only_workflow(tmp_path):
    output = tmp_path / "adsorption"
    result = create_project(
        name="btah_adsorption",
        complex_data=ROOT / "data" / "adsorption" / "ad_complex.data",
        slab_data=ROOT / "data" / "adsorption" / "ad_slab.data",
        molecule_data=ROOT / "data" / "adsorption" / "ad_mol.data",
        destination=output,
        targets=[],
        project_type="adsorption",
        mode="charge_only",
    )
    project = load_project(result.project_file)
    assert project.data["pipeline"]["stages"] == ["validate"]
    assert project.compilation.fitted_properties == ()
    assert project.compilation.validation_properties == ("adsorption",)
