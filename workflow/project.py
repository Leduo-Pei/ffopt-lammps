"""Load a compact project file and compose the legacy workflow config."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from engine.config_loader import deep_merge, load_config
from .machine import load_machine_profile


SOURCE_ROOT = Path(__file__).resolve().parents[1]


class Project:
    def __init__(self, path: Path, data: dict[str, Any]) -> None:
        self.path = path
        self.data = data

    @property
    def name(self) -> str:
        return str(self.data.get("project", {}).get("name", self.path.stem))

    @property
    def default_machine(self) -> str:
        return str(self.data.get("project", {}).get("default_machine", "local"))

    @property
    def root(self) -> Path:
        """Directory against which project assets and outputs are resolved."""
        declared = self.data.get("project", {}).get("root")
        if declared:
            candidate = Path(str(declared)).expanduser()
            if not candidate.is_absolute():
                candidate = self.path.parent / candidate
            return candidate.resolve()
        # Compatibility for the historical repository layout where selectors
        # live in <root>/projects and reference <root>/configs and <root>/data.
        if self.path.parent.name == "projects":
            candidate = self.path.parent.parent
            if (candidate / "configs").is_dir():
                return candidate.resolve()
        return self.path.parent.resolve()

    def resolve(self, value: str | Path) -> Path:
        path = Path(value).expanduser()
        return path.resolve() if path.is_absolute() else (self.root / path).resolve()

    @property
    def run_root(self) -> Path:
        value = self.data.get("project", {}).get("run_root", f"runs/{self.name}")
        return self.resolve(value)


def resolve_path(value: str | Path, base: Path | None = None) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve()
    return ((base or Path.cwd()) / path).resolve()


def resolve_project_path(value: str | Path) -> Path:
    """Resolve a project selector from the caller's working directory."""
    path = resolve_path(value)
    if path.exists():
        return path
    # Source checkouts retain projects/project.yaml as a convenient default.
    fallback = SOURCE_ROOT / "projects" / Path(value).name
    return fallback.resolve() if fallback.exists() else path


def load_project(path: str | Path) -> Project:
    project_path = resolve_project_path(path)
    if not project_path.exists():
        raise FileNotFoundError(f"Project file not found: {project_path}")
    data = {
        key: value
        for key, value in load_config(project_path).items()
        if not key.startswith("_")
    }
    if not isinstance(data, dict):
        raise ValueError(f"Project YAML must contain a mapping: {project_path}")
    if "modules" not in data:
        raise ValueError(f"Project is missing the 'modules' section: {project_path}")
    return Project(project_path, data)


def _base_machine_name(project: Project, machine: str) -> str:
    machines = project.data["modules"].get("machines", {})
    if machine in machines:
        return machine
    profile = load_machine_profile(machine)
    if profile is None:
        choices = sorted(set(machines) | set([machine]))
        raise ValueError(
            f"Unknown machine '{machine}'. Project profiles: {', '.join(choices)}. "
            "Create a user profile with 'ffopt machine configure'."
        )
    machine_cfg = profile.get("machine", {})
    return str(machine_cfg.get("extends") or (
        "cluster" if machine_cfg.get("backend") == "slurm" else "local"
    ))


def module_paths(project: Project, machine: str) -> list[Path]:
    modules = project.data["modules"]
    machines = modules.get("machines", {})
    base_machine = _base_machine_name(project, machine)
    if base_machine not in machines:
        choices = ", ".join(sorted(machines))
        raise ValueError(
            f"Machine '{machine}' extends missing project profile "
            f"'{base_machine}'. Available: {choices}"
        )

    selected: list[str] = [machines[base_machine], modules["system"]]
    selected.extend(modules.get("properties", []))
    methods = modules.get("methods", {})
    selected.extend(methods[key] for key in ("bo", "nn", "al") if key in methods)
    return [project.resolve(item) for item in selected]


def compose_config(project: Project, machine: str) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for path in module_paths(project, machine):
        config = deep_merge(config, load_config(path))

    user_machine = load_machine_profile(machine)
    if user_machine is not None:
        config = deep_merge(
            config,
            {key: value for key, value in user_machine.items()
             if not key.startswith("_")},
        )

    config = deep_merge(config, project.data.get("overrides", {}))
    config.setdefault("manifest", {})["project_name"] = project.name
    config.setdefault("workflow", {})["run_root"] = str(project.run_root)
    config["workflow"]["project_file"] = str(project.path)
    config["workflow"]["project_root"] = str(project.root)
    config["_project_path"] = str(project.path)
    config["_project_machine"] = machine
    return config


def write_generated_config(project: Project, machine: str) -> Path:
    """Write the expanded config used by legacy scripts for reproducibility."""
    output_dir = project.run_root / "_configs"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / f"{project.name}_{machine}.yaml"
    config = compose_config(project, machine)
    serializable = {key: value for key, value in config.items() if not key.startswith("_")}
    with output_path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# Auto-generated by ffopt.py. Edit the project/modules, not this file.\n")
        yaml.safe_dump(serializable, handle, sort_keys=False, allow_unicode=True)
    return output_path
