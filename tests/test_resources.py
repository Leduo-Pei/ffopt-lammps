from engine.resources import resolve_builtin_resource
from workflow.defaults import method_defaults


def test_typed_method_defaults_and_builtin_lammps_resources():
    config = method_defaults()
    assert config["optimization"]["method"] == "auto"
    assert config["optimization"]["turbo"]["length_init"] == 0.8
    assert config["nn"]["exclude_properties"] == []
    assert config["nn"]["training_data"]["core_objective_max"] == 1.0
    bulk = resolve_builtin_resource("builtin:lammps/inputs/bulk/in.bulk.mol")
    adsorption = resolve_builtin_resource("builtin:lammps/inputs/adsorption/in.complex")
    assert bulk.name == "in.bulk.mol"
    assert adsorption.name == "in.complex"


def test_builtin_resource_rejects_parent_traversal():
    try:
        resolve_builtin_resource("builtin:../pyproject.toml")
    except ValueError as exc:
        assert "Invalid builtin resource" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("parent traversal unexpectedly resolved")
