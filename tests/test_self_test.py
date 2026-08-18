import json

import pytest

from workflow.input_compiler import compile_input
from workflow.input_file import parse_input_file
from workflow.self_test import prepare_self_test_project


def test_packaged_acceptance_project_is_reusable_and_machine_independent(tmp_path):
    result = prepare_self_test_project(tmp_path / "acceptance")
    second = prepare_self_test_project(tmp_path / "acceptance")
    assert second == result
    assert result.manifest.exists()
    assert (result.root / "data/bulk/BTAH_822_bulk.data").exists()
    manifest = json.loads(result.manifest.read_text(encoding="utf-8"))
    assert manifest["schema"] == 1
    assert len(manifest["definition_sha256"]) == 64
    assert set(manifest["files_sha256"]) == {
        "ffopt.in",
        "data/adsorption/ad_complex.data",
        "data/adsorption/ad_mol.data",
        "data/adsorption/ad_slab.data",
        "data/bulk/BTAH_822_bulk.data",
        "data/molecule/BTAH_822_single.data",
    }
    compiled = compile_input(parse_input_file(result.input_file))
    assert compiled.dimensions == 13
    assert compiled.project_data["pipeline"]["stages"] == [
        "bo",
        "sample",
        "nn",
        "al",
        "audit",
        "finalize",
        "validate",
    ]
    assert compiled.project_data["pipeline"]["audit"] == {
        "top_k": 4,
        "seeds": [101, 202],
    }
    assert compiled.config["optimization"]["batch_size"] == 24
    assert compiled.project_data["stages"]["sample"]["n_points"] == 48
    assert compiled.config["nn"]["hidden_layers"] == [64, 32]
    assert compiled.config["nn"]["max_epochs"] == 40
    assert compiled.config["active_learning"]["n_candidates_per_round"] == 4
    assert compiled.config["validation"]["objective_max"] == 0.03
    assert compiled.config["adsorption"]["enabled"] is False
    assert compiled.config["validation"]["property_settings"]["adsorption"] == {
        "enabled": True
    }
    assert "ead" not in compiled.config["targets"]


def test_acceptance_project_refuses_unmarked_nonempty_directory(tmp_path):
    destination = tmp_path / "occupied"
    destination.mkdir()
    (destination / "user-file.txt").write_text("keep", encoding="ascii")
    try:
        prepare_self_test_project(destination)
    except FileExistsError as exc:
        assert "not empty" in str(exc)
    else:
        raise AssertionError("unmarked directory was unexpectedly reused")


@pytest.mark.parametrize(
    "relative",
    [
        "ffopt.in",
        "data/adsorption/ad_complex.data",
        "data/adsorption/ad_mol.data",
        "data/adsorption/ad_slab.data",
        "data/bulk/BTAH_822_bulk.data",
        "data/molecule/BTAH_822_single.data",
    ],
)
def test_acceptance_project_refuses_modified_or_missing_definition(tmp_path, relative):
    destination = tmp_path / "acceptance"
    prepare_self_test_project(destination)
    path = destination / relative
    if relative == "ffopt.in":
        path.write_text(path.read_text(encoding="ascii") + "# changed\n", encoding="ascii")
    else:
        path.unlink()

    with pytest.raises(FileExistsError, match="differ from the installed acceptance"):
        prepare_self_test_project(destination)


def test_acceptance_project_refuses_another_ffopt_version(tmp_path):
    destination = tmp_path / "acceptance"
    result = prepare_self_test_project(destination)
    manifest = json.loads(result.manifest.read_text(encoding="ascii"))
    manifest["ffopt_version"] = "0.0.0"
    result.manifest.write_text(json.dumps(manifest), encoding="ascii")

    with pytest.raises(FileExistsError, match="different FFOpt version"):
        prepare_self_test_project(destination)
