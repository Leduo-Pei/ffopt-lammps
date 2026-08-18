"""Install the packaged BTAH scientific acceptance project."""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from engine.resources import resolve_builtin_resource


@dataclass(frozen=True)
class SelfTestProject:
    root: Path
    input_file: Path
    manifest: Path


def prepare_self_test_project(destination: str | Path) -> SelfTestProject:
    """Create, or reuse, a marked BTAH acceptance project."""
    root = Path(destination).expanduser().resolve()
    input_file = root / "ffopt.in"
    manifest = root / "self_test_manifest.json"
    if root.exists() and any(root.iterdir()):
        if input_file.exists() and manifest.exists():
            return SelfTestProject(root, input_file, manifest)
        raise FileExistsError(
            f"Self-test destination is not empty: {root}. Choose another --workdir."
        )

    root.mkdir(parents=True, exist_ok=True)
    source_input = resolve_builtin_resource("builtin:examples/btah/acceptance.in")
    text = source_input.read_text(encoding="utf-8").replace("../../data/", "data/")
    input_file.write_text(text, encoding="ascii", newline="\n")

    resources = {
        "bulk": "data/bulk/BTAH_822_bulk.data",
        "single": "data/molecule/BTAH_822_single.data",
    }
    for relative in resources.values():
        source = resolve_builtin_resource(f"builtin:{relative}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    manifest.write_text(
        json.dumps(
            {
                "kind": "ffopt-btah-scientific-acceptance",
                "input": "ffopt.in",
                "resources": resources,
                "note": (
                    "The validated solution is used as a warm-start centre. "
                    "This checks software and machine equivalence; it is not "
                    "an independent force-field result."
                ),
            },
            indent=2,
        )
        + "\n",
        encoding="ascii",
        newline="\n",
    )
    (root / ".gitignore").write_text(
        "runs/\n__pycache__/\n*.py[cod]\n", encoding="ascii", newline="\n"
    )
    return SelfTestProject(root, input_file, manifest)
