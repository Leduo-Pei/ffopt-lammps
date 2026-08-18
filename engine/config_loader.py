#!/usr/bin/env python3
"""
Internal runtime-config loader for the force-field workflow.

The public ``ffopt.in`` file is compiled to an immutable JSON snapshot before
engine stages start. JSON is valid YAML, so ``yaml.safe_load`` reads that
snapshot while retaining compatibility with prerelease YAML checkpoints and
their optional ``include`` lists. Includes are an internal compatibility API,
not part of the public one-input workflow.
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List

import yaml

from .resources import resolve_config_reference


class ConfigIncludeError(RuntimeError):
    """Raised when a composable config cannot be expanded."""


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    """
    Load an internal JSON/YAML config, recursively expanding ``include`` files.

    Parameters
    ----------
    path
        Path to an immutable runtime snapshot or a prerelease YAML config.

    Returns
    -------
    dict
        Expanded runtime mapping consumed by the BO/NN/AL engines.
    """
    root = Path(path).expanduser().resolve()
    sources: List[str] = []
    config = _load_one(root, stack=[], sources=sources)
    config["_config_path"] = str(root)
    config["_config_sources"] = sources
    return config


def _load_one(path: Path, stack: List[Path], sources: List[str]) -> Dict[str, Any]:
    if path in stack:
        chain = " -> ".join(str(p) for p in [*stack, path])
        raise ConfigIncludeError(f"recursive include detected: {chain}")
    if not path.exists():
        raise FileNotFoundError(f"config file not found: {path}")

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ConfigIncludeError(f"top-level YAML must be a mapping: {path}")

    merged: Dict[str, Any] = {}
    includes = data.get("include", [])
    if isinstance(includes, (str, os.PathLike)):
        includes = [includes]
    if not isinstance(includes, list):
        raise ConfigIncludeError(f"'include' must be a string or list: {path}")

    for item in includes:
        include_path = _resolve_include(path.parent, item)
        child = _load_one(include_path, [*stack, path], sources)
        merged = deep_merge(merged, child)

    current = {k: v for k, v in data.items() if k != "include"}
    merged = deep_merge(merged, current)
    sources.append(str(path))
    return merged


def _resolve_include(base_dir: Path, item: Any) -> Path:
    """
    Resolve one include entry.

    Accepted forms:
      - "../machines/local.yaml"
      - {path: "../machines/local.yaml"}
    """
    if isinstance(item, dict):
        if "path" not in item:
            raise ConfigIncludeError(f"include mapping must contain 'path': {item}")
        item = item["path"]
    return resolve_config_reference(str(item), base=base_dir)


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge two config dictionaries without mutating inputs."""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if (
            key in out
            and isinstance(out[key], dict)
            and isinstance(value, dict)
        ):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def require_sections(config: Dict[str, Any], sections: Iterable[str]) -> None:
    """Small helper for scripts that need a clear missing-section error."""
    missing = [s for s in sections if s not in config]
    if missing:
        raise ValueError(f"Config missing required sections: {missing}")


def save_config_snapshot(
    config: Dict[str, Any], output_dir: str | os.PathLike[str]
) -> Path:
    """Save config for direct engine runs; pipelines already have provenance."""
    source = config.get("_config_path")
    if source:
        source_path = Path(source).expanduser().resolve()
        if source_path.parent.name == "provenance" and source_path.is_file():
            return source_path
    destination = Path(output_dir) / "config_used.yaml"
    destination.parent.mkdir(parents=True, exist_ok=True)
    serializable = {
        key: value for key, value in config.items() if not key.startswith("_")
    }
    with destination.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Expanded runtime configuration snapshot.\n")
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)
    return destination
