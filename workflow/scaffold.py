"""Create a portable one-input-file FFOpt molecular project."""

from __future__ import annotations

import json
import re
import shutil
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from .data_contract import check_data_files
from .lammps_data import AtomTypeSummary

BULK_TARGETS = {"a", "b", "c", "alpha", "beta", "gamma_ang", "density"}
DEFAULT_UNITS = {
    "a": "A", "b": "A", "c": "A",
    "alpha": "degree", "beta": "degree", "gamma_ang": "degree",
    "density": "g/cm3", "esub_proxy": "kJ/mol", "ead": "kcal/mol",
}
TARGET_ALIASES = {
    "gamma": "gamma_ang",
    "sublimation": "esub_proxy",
    "adsorption": "ead",
    "adsorption_energy": "ead",
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
    """Parse ``NAME=VALUE[,WEIGHT[,UNIT[,TOLERANCE]]]`` from the CLI."""
    name, separator, payload = value.partition("=")
    name = name.strip()
    if not separator or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name):
        raise ValueError(
            "Invalid target "
            f"{value!r}; expected NAME=VALUE[,WEIGHT[,UNIT[,TOLERANCE]]]"
        )
    name = TARGET_ALIASES.get(name, name)
    columns = [item.strip() for item in payload.split(",", maxsplit=3)]
    if not columns[0]:
        raise ValueError(f"Target {name!r} has no value")
    target_value = float(columns[0])
    weight = float(columns[1]) if len(columns) > 1 and columns[1] else 1.0
    unit = columns[2] if len(columns) > 2 and columns[2] else DEFAULT_UNITS.get(name, "")
    tolerance = float(columns[3]) if len(columns) > 3 and columns[3] else None
    if weight < 0.0:
        raise ValueError(f"Target {name!r} weight must be non-negative")
    if tolerance is not None and tolerance <= 0.0:
        raise ValueError(f"Target {name!r} tolerance must be positive")
    return TargetSpec(
        name=name,
        value=target_value,
        weight=weight,
        unit=unit,
        tolerance=tolerance,
    )


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


def _copy(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)


def _input_token(value: str) -> str:
    """Quote a generated input token when shlex would split or truncate it."""
    if re.fullmatch(r"[^\s#\"']+", value):
        return value
    if '"' not in value:
        return f'"{value}"'
    if "'" not in value:
        return f"'{value}'"
    raise ValueError(
        f"Cannot represent generated input path {value!r}; rename the file so its "
        "name does not contain both single and double quotes"
    )


def _target_lines(targets: list[TargetSpec]) -> list[str]:
    lines = []
    for target in targets:
        name = "gamma" if target.name == "gamma_ang" else target.name
        tolerance_value = (
            target.tolerance
            if target.tolerance is not None
            else max(abs(target.value) * 0.03, 1.0e-6)
        )
        lines.append(
            f"    target {name:<9s} {target.value:g} "
            f"{target.unit or DEFAULT_UNITS[target.name]} "
            f"weight {target.weight:g} tolerance {tolerance_value:g}"
        )
    return lines


def create_project(
    *,
    name: str,
    data_file: str | Path | None = None,
    destination: str | Path,
    targets: Iterable[TargetSpec],
    single_data: str | Path | None = None,
    complex_data: str | Path | None = None,
    slab_data: str | Path | None = None,
    molecule_data: str | Path | None = None,
    project_type: str = "auto",
    metal_label: str = "Au",
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
    if project_type not in {
        "auto", "molecular-crystal", "adsorption", "crystal-adsorption"
    }:
        raise ValueError(f"Unsupported project type: {project_type}")
    if mixing_rule not in {"geometric", "arithmetic"}:
        raise ValueError("mixing_rule must be geometric or arithmetic")
    if any(int(value) < 1 for value in cells):
        raise ValueError("--cells values must all be positive integers")
    if min(*epsilon_scale, *sigma_scale) <= 0.0:
        raise ValueError("epsilon/sigma scale factors must be positive")
    if charge_window <= 0.0 or charge_limit <= 0.0:
        raise ValueError("charge window and limit must be positive")

    target_list = list(targets)
    names = [item.name for item in target_list]
    if len(names) != len(set(names)):
        raise ValueError("Targets cannot repeat")
    unsupported = set(names) - BULK_TARGETS - {"esub_proxy", "ead"}
    if unsupported:
        raise ValueError(f"Scaffold cannot provide targets: {sorted(unsupported)}")
    if "esub_proxy" in names and not single_data:
        raise ValueError("esub_proxy requires --single-data")
    adsorption_values = (complex_data, slab_data, molecule_data)
    has_adsorption = any(value is not None for value in adsorption_values)
    if has_adsorption and not all(value is not None for value in adsorption_values):
        raise ValueError(
            "Adsorption requires --complex-data, --slab-data, and --molecule-data together"
        )
    if "ead" in names and not has_adsorption:
        raise ValueError("ead requires all three adsorption data files")
    if project_type == "auto":
        project_type = (
            "crystal-adsorption" if data_file and has_adsorption
            else "adsorption" if has_adsorption
            else "molecular-crystal"
        )
    requires_bulk = project_type in {"molecular-crystal", "crystal-adsorption"}
    requires_adsorption = project_type in {"adsorption", "crystal-adsorption"}
    if requires_bulk and not data_file:
        raise ValueError(f"{project_type} requires --data-file/--bulk-data")
    if requires_adsorption and not has_adsorption:
        raise ValueError(f"{project_type} requires all three adsorption data files")
    if has_adsorption and not requires_adsorption:
        raise ValueError(
            "Adsorption data require project type crystal-adsorption (or use --project-type auto)"
        )
    if not requires_bulk and data_file:
        raise ValueError("adsorption-only projects do not use --data-file/--bulk-data")
    if not requires_bulk and single_data:
        raise ValueError("--single-data requires a molecular-crystal project")
    if "esub_proxy" in names and not requires_bulk:
        raise ValueError("sublimation requires a bulk molecular-crystal data file")
    if set(names) & BULK_TARGETS and not requires_bulk:
        raise ValueError("bulk targets require a molecular-crystal project")

    source = Path(data_file).expanduser().resolve() if data_file else None
    single_source = Path(single_data).expanduser().resolve() if single_data else None
    complex_source = Path(complex_data).expanduser().resolve() if complex_data else None
    slab_source = Path(slab_data).expanduser().resolve() if slab_data else None
    molecule_source = Path(molecule_data).expanduser().resolve() if molecule_data else None
    report = check_data_files(
        bulk=source,
        single=single_source,
        complex=complex_source,
        slab=slab_source,
        molecule=molecule_source,
    )
    if not report.ok:
        raise ValueError("; ".join(item.message for item in report.errors))
    warnings = [item.message for item in report.warnings]
    summary = report.files["bulk"] if source else report.files["molecule"]

    root = Path(destination).expanduser().resolve()
    if root.exists() and any(root.iterdir()) and not force:
        raise FileExistsError(
            f"Destination is not empty: {root}. Pass --force to replace generated files."
        )
    root.mkdir(parents=True, exist_ok=True)
    bulk_destination = None
    if source:
        bulk_destination = root / "data" / "bulk" / source.name
        _copy(source, bulk_destination)
    single_destination = None
    if single_source:
        single_destination = root / "data" / "molecule" / single_source.name
        _copy(single_source, single_destination)
    adsorption_destinations: dict[str, Path] = {}
    for role, role_source in (
        ("complex", complex_source),
        ("slab", slab_source),
        ("molecule", molecule_source),
    ):
        if role_source:
            destination_path = root / "data" / "adsorption" / role_source.name
            _copy(role_source, destination_path)
            adsorption_destinations[role] = destination_path

    labels: dict[int, str] = {}
    used: set[str] = set()
    rows = []
    for atom_type in summary.atom_types:
        label = _safe_label(atom_type.label, atom_type.type_id, used)
        labels[atom_type.type_id] = label
        epsilon, sigma = map(float, atom_type.pair_coefficients[:2])
        charge = float(_single_charge(atom_type) or 0.0)
        if abs(charge) > charge_limit:
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
            requested = str(derive_charge)
            try:
                derive_id = int(requested)
            except ValueError:
                matches = [
                    item.type_id
                    for item in summary.atom_types
                    if item.label == requested or labels[item.type_id] == requested
                ]
                if not matches:
                    available = ", ".join(labels.values())
                    raise ValueError(
                        f"Derived charge type {requested!r} is not in the data file; "
                        f"choose an ID or label from: {available}"
                    ) from None
                derive_id = matches[0]
            if derive_id not in labels:
                raise ValueError(f"Derived charge type {derive_id} is not in the data file")
            derive_label = labels[derive_id]

    fixed = {
        "full": "", "fix_sigma": "    fix sigma\n",
        "charge_only": "    fix epsilon sigma\n", "lj_only": "    fix charge\n",
    }[mode] if target_list else ""
    range_lines: list[str] = []
    if target_list and mode in {"full", "fix_sigma", "lj_only"}:
        range_lines.append(
            f"    range epsilon factor {epsilon_scale[0]:g} {epsilon_scale[1]:g}"
        )
    if target_list and mode in {"full", "lj_only"}:
        range_lines.append(
            f"    range sigma   factor {sigma_scale[0]:g} {sigma_scale[1]:g}"
        )
    if target_list and mode in {"full", "fix_sigma", "charge_only"}:
        range_lines.append(f"    range charge  delta  {charge_window:g}")
    range_block = "\n".join(range_lines)
    optimized_per_type = {
        "full": 3, "fix_sigma": 2, "charge_only": 1, "lj_only": 2
    }[mode]
    estimated_dimensions = len(rows) * optimized_per_type if target_list else 0
    if target_list and derive_label and mode != "lj_only":
        estimated_dimensions -= 1
    sample_points = min(2500, max(1500, 75 * max(estimated_dimensions, 1)))
    neutrality = (
        f"    neutrality derive {derive_label}\n" if derive_label else "    neutrality none\n"
    )

    bulk_targets = [item for item in target_list if item.name in BULK_TARGETS]
    sub_target = next((item for item in target_list if item.name == "esub_proxy"), None)
    adsorption_target = next((item for item in target_list if item.name == "ead"), None)
    property_blocks: list[str] = []
    if bulk_destination and source:
        bulk_path = _input_token(f"data/bulk/{source.name}")
        property_blocks.append(f"""property bulk
    # Standard protocol is fixed: energy minimization, then 300 K NPT.
    data {bulk_path}
    cells_in_data {cells[0]} {cells[1]} {cells[2]}
    temperature 300 K
    pressure 1 atm
    equilibration 20000
    production 40000
{chr(10).join(_target_lines(bulk_targets))}
end""")
    sub_block = ""
    if single_destination and source:
        bulk_path = _input_token(f"data/bulk/{source.name}")
        single_path = _input_token(f"data/molecule/{single_destination.name}")
        target_line = ""
        if sub_target:
            tolerance = sub_target.tolerance if sub_target.tolerance is not None else 10.0
            target_line = (
                f"    target {sub_target.value:g} {sub_target.unit or 'kJ/mol'} "
                f"weight {sub_target.weight:g} tolerance {tolerance:g}\n"
            )
        sub_block = f"""
property sublimation
    # E_sub estimate = E_single,min - <PE_bulk,NPT> / number_of_molecules.
    bulk {bulk_path}
    single {single_path}
    temperature 298.15 K
{target_line.rstrip()}
end
""".strip()
        property_blocks.append(sub_block)
    if adsorption_destinations:
        complex_path = _input_token(
            f"data/adsorption/{adsorption_destinations['complex'].name}"
        )
        slab_path = _input_token(
            f"data/adsorption/{adsorption_destinations['slab'].name}"
        )
        molecule_path = _input_token(
            f"data/adsorption/{adsorption_destinations['molecule'].name}"
        )
        target_line = ""
        if adsorption_target:
            tolerance = (
                adsorption_target.tolerance
                if adsorption_target.tolerance is not None
                else max(abs(adsorption_target.value) * 0.03, 1.0e-6)
            )
            target_line = (
                f"    target {adsorption_target.value:g} "
                f"{adsorption_target.unit or 'kcal/mol'} "
                f"weight {adsorption_target.weight:g} tolerance {tolerance:g}\n"
            )
        property_blocks.append(f"""property adsorption
    # Without a target, adsorption is calculated only during final validation.
    data complex {complex_path}
    data slab {slab_path}
    data molecule {molecule_path}
    protocol minimize
    metal {metal_label}
{target_line.rstrip()}
end""")

    workflow = (
        "bo sample nn al audit finalize validate" if target_list else "validate"
    )
    optimization_blocks = ""
    if target_list:
        optimization_blocks = f"""
bo
    # auto selects a BO method from the number of free dimensions.
    method auto
    initial_points 48
    # Scientific candidates per BO round; independent of machine workers.
    batch_size 48
    max_rounds 200
    # Stable BO centers used by focused sampling: top_k x number of seeds.
    stability_audit on
    stability_top_k 20
    stability_seeds 101 202 303
    early_stop patience 30
end

sample
    # Independent local points used to train the surrogate after BO.
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

audit
    # Re-evaluate the best accumulated candidates before final selection.
    top_k 8
    seeds 101 202 303
end
"""

    range_notes = (
        "    # Only free parameters need ranges. Fixed values remain explicit below.\n"
        "    # LJ uses multiplicative factors; charge uses an absolute +/- window."
        if target_list
        else "    # Validation-only workflow: no parameter ranges or fix commands are needed."
    )
    text = f"""# Generated by ffopt init. Paths are relative to this file.
# Review the force-field values, ranges, targets, and run lengths before production.
ffopt 1
project {name}
workflow {workflow}

parameters
{range_notes}
{range_block}
    # Absolute safety bound applied after the per-type charge range.
    charge_limit {charge_limit:g}
    # LAMMPS always uses geometric epsilon mixing; sigma may be geometric or arithmetic.
    mixing epsilon geometric
    mixing sigma {mixing_rule}
{neutrality}{fixed}
    # id  label        epsilon(kcal/mol)  sigma(A)       charge(e)
{chr(10).join(rows)}
end

{chr(10).join(property_blocks)}
{optimization_blocks}
validate
    # Final validation always saves structures and trajectories.
    require_tolerances {"yes" if target_list else "no"}
end
"""
    input_path = root / "ffopt.in"
    input_path.write_text(text, encoding="utf-8", newline="\n")

    from .input_compiler import compile_input
    from .input_file import parse_input_file

    compilation = compile_input(parse_input_file(input_path))
    (root / "README.md").write_text(
        f"# {name}\n\n"
        f"Generated FFOpt project type: `{project_type}`; parameter mode: "
        f"`{mode if target_list else 'validation_only'}`.\n\n"
        "Review every type, any ranges and targets, tolerances, and run lengths in "
        "`ffopt.in`, then validate it:\n\n"
        "```bash\n"
        "ffopt check ffopt.in\n"
        "ffopt explain ffopt.in\n"
        "```\n\n"
        "On a local workstation (direct execution, no scheduler):\n\n"
        "```bash\n"
        "ffopt doctor ffopt.in --machine local\n"
        "ffopt run ffopt.in --machine local --dry-run\n"
        "ffopt run ffopt.in --machine local\n"
        "```\n\n"
        "On a SLURM cluster, choose a configured profile instead of `local`:\n\n"
        "```bash\n"
        "ffopt machine list\n"
        "ffopt doctor ffopt.in --machine PROFILE\n"
        "ffopt run ffopt.in --machine PROFILE --dry-run\n"
        "ffopt run ffopt.in --machine PROFILE --watch\n"
        "```\n",
        encoding="ascii", newline="\n",
    )
    (root / ".gitignore").write_text(
        "runs/\n__pycache__/\n*.py[cod]\n", encoding="ascii", newline="\n"
    )
    manifest = {
        "generator": "ffopt init",
        "project_type": project_type,
        "source_data": str(source) if source else None,
        "single_data": str(single_source) if single_source else None,
        "adsorption_data": {
            "complex": str(complex_source) if complex_source else None,
            "slab": str(slab_source) if slab_source else None,
            "molecule": str(molecule_source) if molecule_source else None,
        },
        "parameter_mode": mode if target_list else "validation_only",
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
