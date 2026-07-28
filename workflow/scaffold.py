"""Create a portable one-input-file FFOpt molecular project."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .lammps_data import AtomTypeSummary, LammpsDataSummary, inspect_lammps_data

BULK_TARGETS = {"a", "b", "c", "alpha", "beta", "gamma_ang", "density"}
DEFAULT_UNITS = {
    "a": "A", "b": "A", "c": "A",
    "alpha": "degree", "beta": "degree", "gamma_ang": "degree",
    "density": "g/cm3", "esub_proxy": "kJ/mol",
}
TARGET_ALIASES = {
    "gamma": "gamma_ang",
    "sublimation": "esub_proxy",
}
SUPPORTED_COEFFICIENT_STYLES = {
    "Bond Coeffs": "harmonic",
    "Angle Coeffs": "harmonic",
    "Dihedral Coeffs": "harmonic",
    "Improper Coeffs": "cvff",
}


@dataclass(frozen=True)
class TargetSpec:
    name: str
    value: float
    weight: float = 1.0
    unit: str = ""
    tolerance: float | None = None


@dataclass(frozen=True)
class ScaffoldResult:
    root: Path
    project_file: Path
    dimensions: int
    atom_types: int
    targets: tuple[str, ...]
    warnings: tuple[str, ...]


def parse_target_spec(value: str) -> TargetSpec:
    """Parse ``NAME=VALUE[,WEIGHT[,UNIT]]`` from the CLI."""
    name, separator, payload = value.partition("=")
    name = name.strip()
    if not separator or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError(
            f"Invalid target {value!r}; expected NAME=VALUE[,WEIGHT[,UNIT]]"
        )
    name = TARGET_ALIASES.get(name, name)
    columns = [item.strip() for item in payload.split(",", maxsplit=2)]
    if not columns[0]:
        raise ValueError(f"Target {name!r} has no value")
    target_value = float(columns[0])
    weight = float(columns[1]) if len(columns) > 1 and columns[1] else 1.0
    unit = columns[2] if len(columns) > 2 and columns[2] else DEFAULT_UNITS.get(name, "")
    if weight < 0.0:
        raise ValueError(f"Target {name!r} weight must be non-negative")
    return TargetSpec(name=name, value=target_value, weight=weight, unit=unit)


def _safe_label(raw: str, type_id: int, used: set[str]) -> str:
    label = re.sub(r"[^A-Za-z0-9_]", "_", raw.strip())
    label = re.sub(r"_+", "_", label).strip("_") or f"type_{type_id}"
    if label[0].isdigit():
        label = f"type_{label}"
    base = label
    suffix = 2
    while label in used:
        label = f"{base}_{suffix}"
        suffix += 1
    used.add(label)
    return label


def _single_charge(atom_type: AtomTypeSummary) -> float | None:
    if not atom_type.charges:
        return None
    if len(atom_type.charges) != 1:
        raise ValueError(
            f"Atom type {atom_type.type_id} ({atom_type.label}) has multiple "
            "charges. FFOpt currently applies one charge per type."
        )
    return float(atom_type.charges[0])


def _validate_data_contract(summary: LammpsDataSummary, label: str) -> list[str]:
    warnings: list[str] = []
    if summary.atom_style != "full":
        raise ValueError(
            f"{label} uses atom_style {summary.atom_style!r}; bundled molecular "
            "templates require 'Atoms # full'."
        )
    for section, expected in SUPPORTED_COEFFICIENT_STYLES.items():
        if section not in summary.section_styles:
            continue
        observed = summary.section_styles[section].split()[0]
        if observed != expected:
            raise ValueError(
                f"{label} declares '{section} # {summary.section_styles[section]}'; "
                f"the bundled template requires {expected!r}."
            )
    for atom_type in summary.atom_types:
        if len(atom_type.pair_coefficients) < 2:
            raise ValueError(
                f"Atom type {atom_type.type_id} ({atom_type.label}) has no usable "
                "Pair Coeffs epsilon/sigma values."
            )
        epsilon, sigma = atom_type.pair_coefficients[:2]
        if epsilon <= 0.0 or sigma <= 0.0:
            raise ValueError(
                f"Atom type {atom_type.type_id} has non-positive epsilon/sigma: "
                f"{epsilon}, {sigma}"
            )
        if len(atom_type.pair_coefficients) > 2:
            warnings.append(
                f"Type {atom_type.type_id} has extra Pair Coeffs columns; only "
                "the first two are interpreted as epsilon and sigma."
            )
        _single_charge(atom_type)
    return warnings


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def _target_lines(targets: list[TargetSpec]) -> list[str]:
    lines = []
    for target in targets:
        name = "gamma" if target.name == "gamma_ang" else target.name
        tolerance = (
            f" tolerance {target.tolerance:g}" if target.tolerance is not None else ""
        )
        lines.append(
            f"    target {name:<9s} {target.value:g} "
            f"{target.unit or DEFAULT_UNITS[target.name]} "
            f"weight {target.weight:g}{tolerance}"
        )
    return lines


def create_project(
    *,
    name: str,
    data_file: str | Path,
    destination: str | Path,
    targets: Iterable[TargetSpec],
    single_data: str | Path | None = None,
    mode: str = "full",
    cells: tuple[int, int, int] = (1, 1, 1),
    mixing_rule: str = "geometric",
    derive_charge: str | int = "auto",
    epsilon_scale: tuple[float, float] = (0.5, 2.0),
    sigma_scale: tuple[float, float] = (0.85, 1.15),
    charge_window: float = 0.30,
    charge_limit: float = 2.0,
    force: bool = False,
) -> ScaffoldResult:
    """Generate ``ffopt.in`` with explicit, reviewable initial parameters."""
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", name):
        raise ValueError("Project name must start with a letter and contain no spaces")
    if mode not in {"full", "fix_sigma", "charge_only", "lj_only"}:
        raise ValueError(f"Unsupported parameter mode: {mode}")
    if mixing_rule not in {"geometric", "arithmetic"}:
        raise ValueError("mixing_rule must be geometric or arithmetic")
    if any(int(value) < 1 for value in cells):
        raise ValueError("--cells values must all be positive integers")
    if min(*epsilon_scale, *sigma_scale) <= 0.0:
        raise ValueError("epsilon/sigma scale factors must be positive")
    if charge_window <= 0.0 or charge_limit <= 0.0:
        raise ValueError("charge window and limit must be positive")

    target_list = list(targets)
    if not target_list:
        raise ValueError("At least one --target is required")
    names = [item.name for item in target_list]
    if len(names) != len(set(names)):
        raise ValueError("Targets cannot repeat")
    unsupported = set(names) - BULK_TARGETS - {"esub_proxy"}
    if unsupported:
        raise ValueError(f"Scaffold cannot provide targets: {sorted(unsupported)}")
    if "esub_proxy" in names and not single_data:
        raise ValueError("esub_proxy requires --single-data")

    source = Path(data_file).expanduser().resolve()
    summary = inspect_lammps_data(source)
    warnings = _validate_data_contract(summary, "Bulk data")
    single_source = Path(single_data).expanduser().resolve() if single_data else None
    if single_source:
        warnings.extend(
            _validate_data_contract(inspect_lammps_data(single_source), "Single-molecule data")
        )

    root = Path(destination).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(
            f"Destination is not empty: {root}. Pass --force to replace generated files."
        )
    root.mkdir(parents=True, exist_ok=True)
    bulk_destination = root / "data" / "bulk" / source.name
    _copy(source, bulk_destination)
    single_destination = None
    if single_source:
        single_destination = root / "data" / "molecule" / single_source.name
        _copy(single_source, single_destination)

    labels: dict[int, str] = {}
    used: set[str] = set()
    rows = []
    for atom_type in summary.atom_types:
        label = _safe_label(atom_type.label, atom_type.type_id, used)
        labels[atom_type.type_id] = label
        epsilon, sigma = map(float, atom_type.pair_coefficients[:2])
        charge = float(_single_charge(atom_type) or 0.0)
        if abs(charge) >= charge_limit:
            raise ValueError(
                f"Initial charge {charge} for type {atom_type.type_id} reaches "
                f"--charge-limit {charge_limit}"
            )
        rows.append(
            f"    type {atom_type.type_id:3d}  {label:<12s} "
            f"{epsilon: .10f}  {sigma: .10f}  {charge: .10f}"
        )

    derive_label = None
    if mode in {"full", "fix_sigma", "charge_only"} and derive_charge != "none":
        if derive_charge == "auto":
            if summary.total_charge is not None and abs(summary.total_charge) <= 1.0e-6:
                derive_id = max(
                    summary.atom_types,
                    key=lambda item: (item.atom_count, -item.type_id),
                ).type_id
                derive_label = labels[derive_id]
        else:
            derive_id = int(derive_charge)
            if derive_id not in labels:
                raise ValueError(f"Derived charge type {derive_id} is not in the data file")
            derive_label = labels[derive_id]

    fixed = {
        "full": "", "fix_sigma": "    fix sigma\n",
        "charge_only": "    fix epsilon sigma\n", "lj_only": "    fix charge\n",
    }[mode]
    optimized_per_type = {"full": 3, "fix_sigma": 2, "charge_only": 1, "lj_only": 2}[mode]
    estimated_dimensions = len(rows) * optimized_per_type
    if derive_label and mode != "lj_only":
        estimated_dimensions -= 1
    sample_points = min(2500, max(800, 50 * max(estimated_dimensions, 1)))
    neutrality = (
        f"    neutrality derive {derive_label}\n" if derive_label else "    neutrality none\n"
    )

    bulk_targets = [item for item in target_list if item.name in BULK_TARGETS]
    sub_target = next((item for item in target_list if item.name == "esub_proxy"), None)
    sub_block = ""
    if sub_target and single_destination:
        tolerance = (
            f" tolerance {sub_target.tolerance:g}" if sub_target.tolerance is not None else ""
        )
        sub_block = f"""
property sublimation
    bulk data/bulk/{source.name}
    single data/molecule/{single_destination.name}
    temperature 298.15 K
    target {sub_target.value:g} {sub_target.unit or 'kJ/mol'} weight {sub_target.weight:g}{tolerance}
end
"""

    text = f"""# Generated by ffopt init. Review every explicit parameter and range.
ffopt 1
project {name}
workflow bo sample nn al validate

parameters
    range epsilon factor {epsilon_scale[0]:g} {epsilon_scale[1]:g}
    range sigma   factor {sigma_scale[0]:g} {sigma_scale[1]:g}
    range charge  delta  {charge_window:g}
    charge_limit {charge_limit:g}
    mixing {mixing_rule}
{neutrality}{fixed}
    # id  label        epsilon(kcal/mol)  sigma(A)       charge(e)
{chr(10).join(rows)}
end

property bulk
    data data/bulk/{source.name}
    cells_in_data {cells[0]} {cells[1]} {cells[2]}
    temperature 300 K
    pressure 1 atm
    equilibration 20000
    production 40000
{chr(10).join(_target_lines(bulk_targets))}
end
{sub_block}
bo
    method auto
    initial_points 48
    max_rounds 200
    early_stop patience 30
end

sample
    points {sample_points}
    centers {min(24, max(4, estimated_dimensions))}
    center_selection diverse
    radii 0.01 0.025 0.05
    global_fraction 0.10
    seeds 101 202 303
end

nn
    method ann
    ensemble 8
end

al
    acquisition uncertainty
    rounds 2
    candidates 20
end
"""
    input_path = root / "ffopt.in"
    input_path.write_text(text, encoding="ascii", newline="\n")

    from .input_compiler import compile_input
    from .input_file import parse_input_file

    compilation = compile_input(parse_input_file(input_path))
    (root / "README.md").write_text(
        f"# {name}\n\nReview `ffopt.in`, then run:\n\n"
        "```bash\nffopt check ffopt.in\nffopt explain ffopt.in\n"
        "ffopt run ffopt.in\n```\n",
        encoding="ascii", newline="\n",
    )
    (root / ".gitignore").write_text(
        "runs/\n__pycache__/\n*.py[cod]\n", encoding="ascii", newline="\n"
    )
    manifest = {
        "generator": "ffopt init",
        "source_data": str(source),
        "single_data": str(single_source) if single_source else None,
        "parameter_mode": mode,
        "dimensions": compilation.dimensions,
        "atom_types": len(rows),
        "targets": names,
        "warnings": warnings,
    }
    (root / "scaffold_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="ascii", newline="\n"
    )
    return ScaffoldResult(
        root=root,
        project_file=input_path,
        dimensions=compilation.dimensions,
        atom_types=len(rows),
        targets=tuple(names),
        warnings=tuple(warnings),
    )
