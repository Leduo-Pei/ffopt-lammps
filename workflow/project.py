"""Load a public command input and compose its host-specific runtime config."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from engine.config_loader import deep_merge

from .machine import load_machine_profile


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class Project:
    def __init__(
        self,
        path: Path,
        data: dict[str, Any],
        runtime_config: dict[str, Any] | None = None,
        compilation: Any | None = None,
    ) -> None:
        self.path = path.resolve()
        self.data = data
        self.runtime_config = runtime_config
        self.compilation = compilation

    @property
    def name(self) -> str:
        return str(self.data.get("project", {}).get("name", self.path.stem))

    @property
    def default_machine(self) -> str:
        return "local"

    @property
    def root(self) -> Path:
        return self.path.parent.resolve()

    def resolve(self, value: str | Path) -> Path:
        return resolve_path(value, self.root)

    @property
    def run_root(self) -> Path:
        value = self.data.get("project", {}).get("run_root", f"runs/{self.name}")
        return self.resolve(value)


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((base or Path.cwd()) / path).resolve()


def load_project(path: str | Path) -> Project:
    project_path = resolve_path(path)
    if not project_path.exists():
        raise FileNotFoundError(f"FFOpt input file not found: {project_path}")
    if project_path.suffix.lower() not in {".in", ".inp"}:
        raise ValueError(
            f"FFOpt projects now use .in/.inp command files, got: {project_path}"
        )
    from .input_compiler import compile_input
    from .input_file import parse_input_file

    compilation = compile_input(parse_input_file(project_path))
    return Project(
        project_path,
        compilation.project_data,
        runtime_config=compilation.config,
        compilation=compilation,
    )


def compose_config(project: Project, machine: str) -> dict[str, Any]:
    if project.runtime_config is None:
        raise ValueError("Project has no compiled ffopt.in runtime configuration")
    from .defaults import machine_defaults

    try:
        config = machine_defaults(machine)
    except ValueError:
        profile = load_machine_profile(machine)
        if profile is None:
            raise ValueError(
                f"Unknown machine profile {machine!r}. Configure it with "
                "'ffopt machine configure'."
            )
        backend = str(profile.get("machine", {}).get("backend", "local"))
        config = machine_defaults("cluster" if backend == "slurm" else "local")
    profile = load_machine_profile(machine)
    if profile is not None:
        config = deep_merge(
            config,
            {key: value for key, value in profile.items() if not key.startswith("_")},
        )
    config = deep_merge(config, project.runtime_config)
    config.setdefault("manifest", {})["project_name"] = project.name
    config.setdefault("workflow", {})["run_root"] = str(project.run_root)
    config["workflow"]["project_file"] = str(project.path)
    config["workflow"]["project_root"] = str(project.root)
    config["_project_path"] = str(project.path)
    config["_project_machine"] = machine
    return config


def write_generated_config(project: Project, machine: str) -> Path:
    """Write the expanded internal engine config for provenance and execution."""
    output_dir = project.run_root / "_configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{project.name}_{machine}.json"
    config = compose_config(project, machine)
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    output_path.write_text(
        json.dumps(serializable, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return output_path
