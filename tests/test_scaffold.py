import json
from pathlib import Path

from workflow.cli import _free_parameter_count, _planned_bo_method, build_parser
from workflow.project import compose_config, load_project
from workflow.scaffold import TargetSpec, create_project, parse_target_spec

ROOT = Path(__file__).resolve().parents[1]


def test_target_cli_format():
    target = parse_target_spec("density=1.3285,0.5,g/cm3,0.03")
    assert target.name == "density"
    assert target.value == 1.3285
    assert target.weight == 0.5
    assert target.unit == "g/cm3"
    assert target.tolerance == 0.03


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
    input_text = (output / "ffopt.in").read_text(encoding="ascii")
    assert "range charge" in input_text
    assert "range epsilon" not in input_text
    assert "range sigma" not in input_text
    assert "target density   1.3285 g/cm3 weight 1 tolerance 0.039855" in input_text
    assert "points 1500" in input_text

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


def test_scaffold_accepts_derived_charge_type_label(tmp_path):
    result = create_project(
        name="btah_label",
        data_file=ROOT / "data" / "bulk" / "BTAH_822_bulk.data",
        destination=tmp_path / "label_project",
        targets=[TargetSpec("density", 1.3285, unit="g/cm3")],
        mode="charge_only",
        derive_charge="bhN1",
    )
    assert result.dimensions == 13
    assert "neutrality derive bhN1" in result.project_file.read_text(encoding="ascii")


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
    input_text = result.project_file.read_text(encoding="utf-8")
    assert "E_sub estimate = E_single,min - <PE_bulk,NPT> / number_of_molecules" in input_text


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
    input_text = result.project_file.read_text(encoding="ascii")
    assert result.dimensions == 0
    assert project.data["pipeline"]["stages"] == ["validate"]
    assert project.compilation.fitted_properties == ()
    assert project.compilation.validation_properties == ("adsorption",)
    assert "range epsilon" not in input_text
    assert "range sigma" not in input_text
    assert "range charge" not in input_text
    assert "fix epsilon" not in input_text
    assert "Validation-only workflow" in input_text
    manifest = json.loads(
        (output / "scaffold_manifest.json").read_text(encoding="ascii")
    )
    assert manifest["parameter_mode"] == "validation_only"


def test_scaffold_quotes_data_paths_with_spaces_and_hashes(tmp_path):
    source_dir = tmp_path / "source files"
    source_dir.mkdir()
    bulk_name = "\u82ef bulk #1.data"
    bulk_source = source_dir / bulk_name
    single_source = source_dir / "BTAH single #1.data"
    bulk_source.write_bytes(
        (ROOT / "data" / "bulk" / "BTAH_822_bulk.data").read_bytes()
    )
    single_source.write_bytes(
        (ROOT / "data" / "molecule" / "BTAH_822_single.data").read_bytes()
    )

    result = create_project(
        name="quoted_paths",
        data_file=bulk_source,
        single_data=single_source,
        destination=tmp_path / "quoted_project",
        targets=[TargetSpec("esub_proxy", 98.5, unit="kJ/mol")],
        mode="charge_only",
    )

    input_text = result.project_file.read_text(encoding="utf-8")
    assert f'data "data/bulk/{bulk_name}"' in input_text
    assert f'bulk "data/bulk/{bulk_name}"' in input_text
    assert 'single "data/molecule/BTAH single #1.data"' in input_text
    project = load_project(result.project_file)
    assert project.compilation.config["manifest"]["data_files"]["bulk"].endswith(
        bulk_name
    )
    assert project.compilation.config["sublimation"]["data_files"]["single"].endswith(
        "BTAH single #1.data"
    )


def test_explain_reports_effective_pipeline_settings(capsys):
    parser = build_parser()
    args = parser.parse_args([
        "explain", str(ROOT / "examples" / "btah" / "acceptance.in")
    ])
    args.function(args)
    output = capsys.readouterr().out
    assert (
        "BO                    : method=turbo lhs=24 warm_start=1 "
        "initial_total=25 rounds=1 batch=24 search_evaluations=49"
    ) in output
    assert "audit=off top_k=20 seeds=[101, 202, 303] audit_max_evaluations=0" in output
    assert "Sampling              : 48 points x 1 seeds" in output
    assert "Surrogate             : method=mlp_ensemble ensemble=2" in output
    assert "Active learning       : rounds=1 LAMMPS candidates/round=4" in output
    assert "Final acceptance      : tolerances=required objective_max=0.03" in output

    args = parser.parse_args([
        "explain", str(ROOT / "examples" / "btah" / "charge_only.in")
    ])
    args.function(args)
    output = capsys.readouterr().out
    assert "Final robust audit    : top_k=8 x 3 seeds; rank=mean+std" in output


def test_auto_bo_method_reports_dimension_choice_and_pyro_fallback():
    config = {"optimization": {"method": "auto", "accuracy_priority": False}}
    assert _planned_bo_method(config, 5, saasbo_available=False) == (
        "auto", "gp", ""
    )
    assert _planned_bo_method(config, 13, saasbo_available=False) == (
        "auto", "turbo", ""
    )
    configured, effective, note = _planned_bo_method(
        config, 41, saasbo_available=False
    )
    assert (configured, effective) == ("auto", "turbo")
    assert "Pyro is unavailable" in note
    assert _planned_bo_method(config, 41, saasbo_available=True) == (
        "auto", "saasbo", ""
    )
