from workflow.input_compiler import compile_input
from workflow.input_file import parse_input_file
from workflow.self_test import prepare_self_test_project


def test_packaged_acceptance_project_is_reusable_and_machine_independent(tmp_path):
    result = prepare_self_test_project(tmp_path / "acceptance")
    second = prepare_self_test_project(tmp_path / "acceptance")
    assert second == result
    assert result.manifest.exists()
    assert (result.root / "data/bulk/BTAH_822_bulk.data").exists()
    compiled = compile_input(parse_input_file(result.input_file))
    assert compiled.dimensions == 13
    assert compiled.project_data["pipeline"]["stages"] == [
        "bo", "sample", "nn", "al", "validate"
    ]
    assert compiled.config["optimization"]["batch_size"] == 24
    assert compiled.config["validation"]["objective_max"] == 0.03


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
