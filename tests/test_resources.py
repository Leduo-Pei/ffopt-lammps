from engine.config_loader import load_config
from engine.resources import resolve_builtin_resource


def test_builtin_method_resource_resolves_and_expands_includes():
    path = resolve_builtin_resource("builtin:configs/methods/bo/auto.yaml")
    assert path.name == "auto.yaml"
    config = load_config(path)
    assert config["optimization"]["method"] == "auto"
    assert config["optimization"]["turbo"]["length_init"] == 0.8

    portable = load_config(resolve_builtin_resource(
        "builtin:configs/methods/nn/portable.yaml"
    ))
    assert portable["nn"]["exclude_properties"] == []
    assert portable["nn"]["training_data"]["core_objective_max"] == 1.0


def test_builtin_resource_rejects_parent_traversal():
    try:
        resolve_builtin_resource("builtin:../pyproject.toml")
    except ValueError as exc:
        assert "Invalid builtin resource" in str(exc)
    else:  # pragma: no cover
        raise AssertionError("parent traversal unexpectedly resolved")
