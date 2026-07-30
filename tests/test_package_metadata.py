from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from ffopt import __version__

ROOT = Path(__file__).resolve().parents[1]


def test_public_version_matches_package_metadata():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    assert __version__ == metadata["project"]["version"]
