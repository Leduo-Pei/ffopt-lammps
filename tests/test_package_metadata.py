from pathlib import Path
from types import SimpleNamespace

import pytest
import workflow.cli as cli

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib

from ffopt import __version__
from workflow.cli import build_parser

ROOT = Path(__file__).resolve().parents[1]


def test_public_version_matches_package_metadata():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        metadata = tomllib.load(handle)
    assert __version__ == metadata["project"]["version"]


def test_citation_version_matches_package_metadata():
    citation = (ROOT / "CITATION.cff").read_text(encoding="utf-8")
    assert f"version: {__version__}" in citation
    changelog = (ROOT / "CHANGELOG.md").read_text(encoding="utf-8")
    assert f"## {__version__} -" in changelog


def test_cli_reports_installed_version(capsys):
    with pytest.raises(SystemExit) as exc:
        build_parser().parse_args(["--version"])
    assert exc.value.code == 0
    assert capsys.readouterr().out.strip() == f"ffopt {__version__}"


def test_cli_treats_a_closed_output_pipe_as_success(monkeypatch):
    def closed_pipe(_args):
        raise BrokenPipeError

    parser = SimpleNamespace(
        parse_args=lambda: SimpleNamespace(function=closed_pipe)
    )
    monkeypatch.setattr(cli, "build_parser", lambda: parser)
    monkeypatch.setattr(
        cli.os, "open", lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError())
    )

    with pytest.raises(SystemExit) as exc:
        cli.main()

    assert exc.value.code == 0


@pytest.mark.parametrize(
    "relative",
    [
        "README.md",
        "docs/tutorials/quickstart.md",
        "docs/tutorials/mag1-slurm.md",
        "docs/zh_CN/USER_GUIDE.md",
    ],
)
def test_installation_guides_pin_the_current_release(relative):
    text = (ROOT / relative).read_text(encoding="utf-8")
    expected = f"/v{__version__}/ffopt_lammps-{__version__}-py3-none-any.whl"
    assert expected in text


def test_optional_runtime_extras_are_self_contained():
    with (ROOT / "pyproject.toml").open("rb") as handle:
        project = tomllib.load(handle)["project"]

    core = " ".join(project["dependencies"]).lower()
    assert "torch" not in core
    assert "scipy" not in core
    assert "scikit-learn" not in core

    extras = project["optional-dependencies"]
    for name in ("bo", "full", "saasbo", "all-models"):
        joined = " ".join(extras[name]).lower()
        assert "torch" in joined
        assert "botorch" in joined
        assert "gpytorch" in joined
        assert "scipy" in joined
        assert "scikit-learn" in joined
    assert "pyro-ppl" in " ".join(extras["saasbo"]).lower()
    assert "xgboost" in " ".join(extras["xgboost"]).lower()
    assert "torch" in " ".join(extras["xgboost"]).lower()
