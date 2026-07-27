"""Resolve portable resources shipped with FFOpt-LAMMPS."""

from __future__ import annotations

import os
from pathlib import Path
import sysconfig
from typing import Iterable


BUILTIN_PREFIX = "builtin:"


def builtin_resource_roots() -> Iterable[Path]:
    """Yield source-checkout and installed-wheel resource roots."""
    override = os.environ.get("FFOPT_RESOURCE_DIR")
    if override:
        yield Path(override).expanduser().resolve()

    source_root = Path(__file__).resolve().parents[1]
    if (source_root / "configs").is_dir():
        yield source_root

    data_root = Path(sysconfig.get_path("data")).resolve()
    yield data_root / "share" / "ffopt"


def resolve_builtin_resource(value: str | os.PathLike[str]) -> Path:
    """Resolve ``builtin:path`` without allowing traversal outside the bundle."""
    text = str(value)
    if not text.startswith(BUILTIN_PREFIX):
        raise ValueError(f"Not a builtin resource reference: {value!r}")
    relative_text = text[len(BUILTIN_PREFIX):].replace("\\", "/").lstrip("/")
    relative = Path(relative_text)
    if not relative_text or relative.is_absolute() or ".." in relative.parts:
        raise ValueError(f"Invalid builtin resource path: {value!r}")

    searched = []
    for root in builtin_resource_roots():
        root = root.resolve()
        candidate = (root / relative).resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"Builtin resource escapes its root: {value!r}")
        searched.append(candidate)
        if candidate.exists():
            return candidate
    locations = ", ".join(str(path) for path in searched)
    raise FileNotFoundError(
        f"Builtin FFOpt resource not found: {value!r}; searched {locations}"
    )


def resolve_config_reference(
    value: str | os.PathLike[str], *, base: str | os.PathLike[str]
) -> Path:
    """Resolve a builtin or filesystem config reference."""
    if str(value).startswith(BUILTIN_PREFIX):
        return resolve_builtin_resource(value)
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = Path(base) / path
    return path.resolve()

