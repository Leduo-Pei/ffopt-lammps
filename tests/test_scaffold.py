from pathlib import Path

import yaml

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
    assert (output / "lammps" / "in.bulk.mol").exists()

    project = load_project(output / "project.yaml")
    config = compose_config(project, "local")
    assert _free_parameter_count(config) == 13
    assert config["optimization"]["method"] == "auto"
    assert config["optimization"]["stability_audit"].get("max_workers") is None
    assert config["nn"]["exclude_properties"] == []
    assert config["targets"]["density"]["value"] == 1.3285
    assert config["charge"]["neutrality_constraint"]["enabled"]
    assert config["lammps"]["bulk"]["nx"] == 8

    system = yaml.safe_load((output / "configs" / "system.yaml").read_text())
    assert isinstance(system["atom_types"][0]["params"]["epsilon"], float)
    assert isinstance(system["atom_types"][0]["params"]["sigma"], float)
    assert isinstance(system["atom_types"][1]["params"]["charge"], dict)
