"""Install the packaged BTAH scientific acceptance project."""

from __future__ import annotations

import hashlib
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

from engine.resources import resolve_builtin_resource
from ffopt import __version__


@dataclass(frozen=True)
class SelfTestProject:
    root: Path
    input_file: Path
    manifest: Path


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _definition_hash(files: dict[str, bytes]) -> str:
    digest = hashlib.sha256()
    for relative, payload in sorted(files.items()):
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
    return digest.hexdigest()


def prepare_self_test_project(destination: str | Path) -> SelfTestProject:
    """Create, or safely reuse, the packaged BTAH acceptance project."""
    root = Path(destination).expanduser().resolve()
    input_file = root / "ffopt.in"
    manifest = root / "self_test_manifest.json"

    source_input = resolve_builtin_resource("builtin:examples/btah/acceptance.in")
    input_payload = source_input.read_text(encoding="utf-8").replace(
        "../../data/", "data/"
    ).encode("ascii")
    resources = {
        "bulk": "data/bulk/BTAH_822_bulk.data",
        "single": "data/molecule/BTAH_822_single.data",
        "adsorption_complex": "data/adsorption/ad_complex.data",
        "adsorption_slab": "data/adsorption/ad_slab.data",
        "adsorption_molecule": "data/adsorption/ad_mol.data",
    }
    expected_files = {"ffopt.in": input_payload}
    for relative in resources.values():
        expected_files[relative] = resolve_builtin_resource(
            f"builtin:{relative}"
        ).read_bytes()
    expected_definition = _definition_hash(expected_files)

    if root.exists() and any(root.iterdir()):
        try:
            metadata = json.loads(manifest.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise FileExistsError(
                f"Self-test destination is not empty and is not a valid marked project: {root}. "
                "Choose another --workdir or preserve its runs and remove it."
            ) from exc
        if metadata.get("kind") != "ffopt-btah-scientific-acceptance":
            raise FileExistsError(
                f"Self-test destination has an unrecognized manifest: {root}. "
                "Choose another --workdir."
            )
        if (
            metadata.get("schema") != 1
            or metadata.get("ffopt_version") != __version__
            or metadata.get("definition_sha256") != expected_definition
        ):
            raise FileExistsError(
                "Existing self-test directory belongs to a different FFOpt "
                f"version or acceptance definition: {root}. Preserve its runs "
                "and choose a new --workdir."
            )
        mismatches = [
            relative
            for relative, expected in expected_files.items()
            if not (root / relative).is_file()
            or (root / relative).read_bytes() != expected
        ]
        if mismatches:
            raise FileExistsError(
                "Existing self-test files differ from the installed acceptance "
                f"definition ({', '.join(mismatches)}): {root}. Preserve any runs, "
                "then use a new --workdir or remove this directory."
            )
        return SelfTestProject(root, input_file, manifest)

    root.mkdir(parents=True, exist_ok=True)
    input_file.write_bytes(input_payload)

    for relative in resources.values():
        source = resolve_builtin_resource(f"builtin:{relative}")
        target = root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    manifest.write_text(
        json.dumps(
            {
                "kind": "ffopt-btah-scientific-acceptance",
                "schema": 1,
                "ffopt_version": __version__,
                "input": "ffopt.in",
                "resources": resources,
                "definition_sha256": expected_definition,
                "files_sha256": {
                    relative: _sha256(payload)
                    for relative, payload in sorted(expected_files.items())
                },
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
