from __future__ import annotations

import json
from pathlib import Path

import pytest

from engine.config_loader import ConfigIncludeError, load_config
from workflow.pipeline import _scientific_config
from workflow.state import canonical_hash


def _serializable(config: dict) -> dict:
    return {key: value for key, value in config.items() if not key.startswith("_")}


def test_json_runtime_preserves_exponent_number_type_and_identity(tmp_path: Path):
    runtime = tmp_path / "runtime.json"
    runtime.write_text(
        '{"charge":{"enabled":false,"kspace_accuracy":1e-05}}\n',
        encoding="utf-8",
    )

    loaded = load_config(runtime)

    assert loaded["charge"]["kspace_accuracy"] == pytest.approx(1.0e-5)
    assert isinstance(loaded["charge"]["kspace_accuracy"], float)
    assert _serializable(loaded) == json.loads(runtime.read_text(encoding="utf-8"))
    assert canonical_hash(_scientific_config(loaded)) == canonical_hash(
        _scientific_config(json.loads(runtime.read_text(encoding="utf-8")))
    )
    assert loaded["_config_path"] == str(runtime.resolve())
    assert loaded["_config_sources"] == [str(runtime.resolve())]


def test_legacy_yaml_retains_yaml_scalar_and_include_semantics(tmp_path: Path):
    child = tmp_path / "child.yaml"
    child.write_text(
        "charge:\n  enabled: false\n  kspace_accuracy: 1e-05\n",
        encoding="utf-8",
    )
    root = tmp_path / "root.yaml"
    root.write_text(
        "include: child.yaml\ncharge:\n  enabled: true\n",
        encoding="utf-8",
    )

    loaded = load_config(root)

    assert loaded["charge"] == {
        "enabled": True,
        "kspace_accuracy": "1e-05",
    }
    assert loaded["_config_sources"] == [
        str(child.resolve()),
        str(root.resolve()),
    ]


def test_json_can_include_legacy_yaml_without_changing_each_parser(tmp_path: Path):
    child = tmp_path / "child.yaml"
    child.write_text("legacy: 1e-05\n", encoding="utf-8")
    root = tmp_path / "runtime.json"
    root.write_text(
        '{"include":["child.yaml"],"runtime":1e-05}\n',
        encoding="utf-8",
    )

    loaded = load_config(root)

    assert loaded["runtime"] == pytest.approx(1.0e-5)
    assert isinstance(loaded["runtime"], float)
    assert loaded["legacy"] == "1e-05"


def test_legacy_yaml_can_include_strict_json(tmp_path: Path):
    child = tmp_path / "runtime.JSON"
    child.write_text('{"runtime":1e-05}\n', encoding="utf-8")
    root = tmp_path / "legacy.yaml"
    root.write_text("include: runtime.JSON\nlegacy: 1e-05\n", encoding="utf-8")

    loaded = load_config(root)

    assert isinstance(loaded["runtime"], float)
    assert loaded["legacy"] == "1e-05"


@pytest.mark.parametrize(
    "content, message",
    [
        ('{"value":1,"value":2}\n', "duplicate JSON key"),
        ('{"outer":{"value":1,"value":2}}\n', "duplicate JSON key"),
        ('{"value":NaN}\n', "non-standard JSON number"),
        ('{"value":Infinity}\n', "non-standard JSON number"),
        ('{"value":1e9999}\n', "non-finite JSON number"),
        ('{"value":-1e9999}\n', "non-finite JSON number"),
        ('{"value":1,}\n', "invalid JSON config"),
    ],
)
def test_json_runtime_rejects_noncanonical_input(
    tmp_path: Path,
    content: str,
    message: str,
):
    runtime = tmp_path / "runtime.json"
    runtime.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigIncludeError, match=message):
        load_config(runtime)


def test_config_root_must_be_mapping_for_both_formats(
    tmp_path: Path,
):
    for name, content in (
        ("runtime.json", "[]\n"),
        ("runtime_null.json", "null\n"),
        ("legacy.yaml", "[]\n"),
    ):
        path = tmp_path / name
        path.write_text(content, encoding="utf-8")
        with pytest.raises(ConfigIncludeError, match="top-level config"):
            load_config(path)


def test_empty_legacy_yaml_is_compatible_but_empty_json_is_invalid(tmp_path: Path):
    legacy = tmp_path / "legacy.yaml"
    legacy.write_text("", encoding="utf-8")
    runtime = tmp_path / "runtime.json"
    runtime.write_text("", encoding="utf-8")

    assert _serializable(load_config(legacy)) == {}
    with pytest.raises(ConfigIncludeError, match="invalid JSON config"):
        load_config(runtime)
