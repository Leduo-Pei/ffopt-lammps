"""Parser and typed model for the public FFOpt command input format."""

from __future__ import annotations

import math
import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PARAMETER_NAMES = {"epsilon", "sigma", "charge"}
LEGACY_PIPELINE_STAGE_ORDER = (
    "bo", "sample", "nn", "al", "audit", "finalize", "validate",
)
MATERIAL_PIPELINE_STAGE_ORDER = (
    "bo", "sample", "audit", "screen", "nn", "al", "finalists", "validate",
)
# Keep the public vocabulary independent of one workflow shape.  The applicable
# ordering and dependency contract is selected only after the complete input is
# parsed, so a user may place ``material`` before or after ``workflow``.
PIPELINE_STAGE_ORDER = LEGACY_PIPELINE_STAGE_ORDER
PIPELINE_STAGES = set(LEGACY_PIPELINE_STAGE_ORDER) | set(MATERIAL_PIPELINE_STAGE_ORDER)
PROPERTY_NAMES = {"bulk", "sublimation", "adsorption", "surface", "elasticity"}
MATERIAL_KINDS = {"elemental", "molecular", "multicomponent"}
CRYSTAL_FAMILIES = {"bcc", "molecular"}


class InputFileError(ValueError):
    """An input error carrying a precise source location."""

    def __init__(self, path: Path, line: int, message: str):
        super().__init__(f"{path}:{line}: {message}")
        self.path = path
        self.line = line
        self.message = message


@dataclass(frozen=True)
class RangeSpec:
    mode: str
    low: float
    high: float
    line: int

    def bounds(self, initial: float) -> tuple[float, float]:
        if self.mode == "factor":
            values = (initial * self.low, initial * self.high)
        elif self.mode == "delta":
            values = (initial + self.low, initial + self.high)
        else:
            values = (self.low, self.high)
        return tuple(sorted(float(value) for value in values))


@dataclass(frozen=True)
class AtomTypeSpec:
    type_id: int
    label: str
    epsilon: float
    sigma: float
    charge: float
    charge_explicit: bool
    line: int


@dataclass(frozen=True)
class TieSpec:
    parameter: str
    labels: str | tuple[str, ...]
    line: int


@dataclass(frozen=True)
class DifferenceSpec:
    parameter: str
    first: str
    second: str
    maximum: float
    line: int


@dataclass
class ParameterSpec:
    atom_types: list[AtomTypeSpec] = field(default_factory=list)
    default_ranges: dict[str, RangeSpec] = field(default_factory=dict)
    type_ranges: dict[tuple[str, str], RangeSpec] = field(default_factory=dict)
    fixed: set[str] = field(default_factory=set)
    fixed_by_type: set[tuple[str, str]] = field(default_factory=set)
    fixed_lines: dict[str, int] = field(default_factory=dict)
    fixed_by_type_lines: dict[tuple[str, str], int] = field(default_factory=dict)
    ties: dict[str, TieSpec] = field(default_factory=dict)
    differences: list[DifferenceSpec] = field(default_factory=list)
    derive_charge: str | None = None
    mixing_epsilon: str = "geometric"
    mixing_sigma: str = "geometric"
    charge_abs_max: float = 1.0
    setting_lines: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True)
class TargetSpec:
    name: str
    value: float
    unit: str
    weight: float
    tolerance: float | None
    line: int


@dataclass(frozen=True)
class ElasticityModuleSpec:
    """One fidelity-specific cubic-elasticity module declaration."""

    fidelity: str
    role: str
    line: int


@dataclass(frozen=True)
class ElasticityTargetSpec:
    """An independent cubic target at one explicitly named fidelity."""

    fidelity: str
    name: str
    value: float
    unit: str
    line: int


@dataclass
class PropertySpec:
    name: str
    line: int
    data_files: dict[str, str] = field(default_factory=dict)
    targets: list[TargetSpec] = field(default_factory=list)
    elasticity_modules: dict[str, ElasticityModuleSpec] = field(default_factory=dict)
    elasticity_targets: list[ElasticityTargetSpec] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    setting_lines: dict[str, int] = field(default_factory=dict)

    @property
    def fitted(self) -> bool:
        return bool(self.targets or self.elasticity_targets)


@dataclass
class FFOptInput:
    path: Path
    version: int | None = None
    project: str = ""
    material_kind: str = "molecular"
    material_name: str | None = None
    material_line: int | None = None
    crystal_family: str = "molecular"
    crystal_line: int | None = None
    workflow: list[str] = field(default_factory=list)
    parameters: ParameterSpec = field(default_factory=ParameterSpec)
    properties: list[PropertySpec] = field(default_factory=list)
    stage_settings: dict[str, dict[str, Any]] = field(default_factory=dict)
    stage_setting_lines: dict[str, dict[str, int]] = field(default_factory=dict)


def _tokens(path: Path, line_number: int, raw: str) -> list[str]:
    lexer = shlex.shlex(raw, posix=True)
    lexer.whitespace_split = True
    lexer.commenters = "#"
    lexer.escape = ""
    try:
        return list(lexer)
    except ValueError as exc:
        raise InputFileError(path, line_number, str(exc)) from exc


def _float(path: Path, line: int, value: str, label: str) -> float:
    try:
        result = float(value)
    except ValueError as exc:
        raise InputFileError(path, line, f"{label} must be numeric, got {value!r}") from exc
    if not math.isfinite(result):
        raise InputFileError(path, line, f"{label} must be finite, got {value!r}")
    return result


def _int(path: Path, line: int, value: str, label: str) -> int:
    try:
        result = int(value)
    except ValueError as exc:
        raise InputFileError(path, line, f"{label} must be an integer, got {value!r}") from exc
    return result


def _scalar(value: str) -> Any:
    lowered = value.lower()
    if lowered in {"yes", "true", "on"}:
        return True
    if lowered in {"no", "false", "off"}:
        return False
    try:
        return int(value)
    except ValueError:
        try:
            return float(value)
        except ValueError:
            return value


def _range_spec(path: Path, line: int, tokens: list[str]) -> RangeSpec:
    if not tokens:
        raise InputFileError(path, line, "range requires factor, delta, or absolute")
    mode = tokens[0].lower()
    if mode == "factor":
        if len(tokens) != 3:
            raise InputFileError(path, line, "factor range requires LOW HIGH")
        low = _float(path, line, tokens[1], "factor lower bound")
        high = _float(path, line, tokens[2], "factor upper bound")
        if low <= 0.0 or high <= 0.0:
            raise InputFileError(path, line, "factor bounds must be positive")
    elif mode == "delta":
        values = tokens[1:]
        if values and values[-1].lower() in {"e", "angstrom", "a", "kcal/mol"}:
            values = values[:-1]
        if len(values) == 1:
            magnitude = _float(path, line, values[0], "delta")
            if magnitude <= 0.0:
                raise InputFileError(path, line, "delta must be positive")
            low, high = -magnitude, magnitude
        elif len(values) == 2:
            low = _float(path, line, values[0], "delta lower bound")
            high = _float(path, line, values[1], "delta upper bound")
        else:
            raise InputFileError(path, line, "delta range requires SIZE or LOW HIGH")
    elif mode in {"absolute", "range"}:
        if len(tokens) != 3:
            raise InputFileError(path, line, "absolute range requires LOW HIGH")
        mode = "absolute"
        low = _float(path, line, tokens[1], "absolute lower bound")
        high = _float(path, line, tokens[2], "absolute upper bound")
    else:
        raise InputFileError(path, line, f"unknown range mode {tokens[0]!r}")
    if low >= high:
        raise InputFileError(path, line, "range lower bound must be smaller than upper bound")
    return RangeSpec(mode, float(low), float(high), line)


def _parse_target(path: Path, prop: PropertySpec, line: int, args: list[str]) -> TargetSpec:
    defaults = {
        "sublimation": "esub_proxy",
        "adsorption": "ead",
        "surface": "surf_energy",
    }
    if not args:
        raise InputFileError(path, line, "target requires NAME VALUE or VALUE")
    try:
        value = float(args[0])
    except ValueError:
        name = args[0]
        if len(args) < 2:
            raise InputFileError(path, line, "target NAME requires a numeric VALUE")
        value = _float(path, line, args[1], "target value")
        rest = args[2:]
    else:
        if prop.name not in defaults:
            raise InputFileError(path, line, "bulk target requires an explicit property name")
        name = defaults[prop.name]
        rest = args[1:]

    unit = ""
    weight = 1.0
    tolerance = None
    index = 0
    option_names = {"weight", "tolerance"}
    if index < len(rest) and rest[index].lower() not in option_names:
        unit = rest[index]
        index += 1
    while index < len(rest):
        key = rest[index].lower()
        if key not in option_names or index + 1 >= len(rest):
            raise InputFileError(path, line, f"invalid target option near {rest[index:]}")
        number = _float(path, line, rest[index + 1], key)
        if key == "weight":
            if number < 0.0:
                raise InputFileError(path, line, "target weight cannot be negative")
            weight = number
        else:
            if number <= 0.0:
                raise InputFileError(path, line, "target tolerance must be positive")
            tolerance = number
        index += 2
    return TargetSpec(name, value, unit, weight, tolerance, line)


def _parse_elasticity_line(
    document: FFOptInput,
    prop: PropertySpec,
    line: int,
    tokens: list[str],
) -> None:
    """Parse the strict cubic-elasticity sub-language.

    Elasticity deliberately does not reuse the legacy weighted-target grammar:
    its three independent moduli are ranked by constrained minimax error, not
    by a weighted RMSE. Keeping the syntax separate prevents a seemingly
    harmless ``weight`` option from silently changing the scientific problem.
    """

    path = document.path
    command = tokens[0].lower()
    args = tokens[1:]

    def set_setting(key: str, value: Any) -> None:
        if key in prop.settings:
            previous = prop.setting_lines[key]
            raise InputFileError(
                path,
                line,
                f"duplicate elasticity {key!r} setting "
                f"(first set on line {previous})",
            )
        prop.settings[key] = value
        prop.setting_lines[key] = line

    if command == "module":
        if len(args) != 2:
            raise InputFileError(
                path,
                line,
                "elasticity module syntax: module static objective | "
                "module dynamic promotion|validation",
            )
        fidelity, role = args[0].lower(), args[1].lower()
        if fidelity not in {"static", "dynamic"}:
            raise InputFileError(
                path,
                line,
                "elasticity module must be static or dynamic",
            )
        allowed_roles = {
            "static": {"objective"},
            "dynamic": {"promotion", "validation"},
        }[fidelity]
        if role not in allowed_roles:
            choices = " or ".join(sorted(allowed_roles))
            raise InputFileError(
                path,
                line,
                f"elasticity {fidelity} module role must be {choices}",
            )
        if fidelity in prop.elasticity_modules:
            previous = prop.elasticity_modules[fidelity]
            raise InputFileError(
                path,
                line,
                f"duplicate elasticity {fidelity!r} module "
                f"(first set on line {previous.line})",
            )
        prop.elasticity_modules[fidelity] = ElasticityModuleSpec(
            fidelity=fidelity,
            role=role,
            line=line,
        )
        return

    if command == "target":
        if len(args) != 4:
            raise InputFileError(
                path,
                line,
                "elasticity target syntax: "
                "target static|dynamic B|Cprime|C44 VALUE GPa",
            )
        fidelity, name, raw_value, unit = args
        fidelity = fidelity.lower()
        if fidelity not in {"static", "dynamic"}:
            raise InputFileError(
                path,
                line,
                "elasticity target fidelity must be static or dynamic",
            )
        if name not in {"B", "Cprime", "C44"}:
            raise InputFileError(
                path,
                line,
                "cubic elasticity fit targets must be exactly B, Cprime, and C44; "
                "K, G, E, and nu are not independent fit targets",
            )
        if unit.lower() != "gpa":
            raise InputFileError(
                path,
                line,
                f"elasticity target {name} unit must be GPa, got {unit!r}",
            )
        value = _float(path, line, raw_value, f"elasticity target {name}")
        if value <= 0.0:
            raise InputFileError(
                path,
                line,
                f"elasticity target {name} must be positive",
            )
        duplicate = next(
            (
                target
                for target in prop.elasticity_targets
                if target.fidelity == fidelity and target.name == name
            ),
            None,
        )
        if duplicate is not None:
            raise InputFileError(
                path,
                line,
                f"duplicate elasticity {fidelity} target {name!r} "
                f"(first set on line {duplicate.line})",
            )
        prop.elasticity_targets.append(
            ElasticityTargetSpec(fidelity, name, value, "GPa", line)
        )
        return

    if command == "gate":
        if len(args) != 3:
            raise InputFileError(
                path,
                line,
                "elasticity gate syntax: gate lattice|density|surface VALUE "
                "percent | gate angles VALUE degree",
            )
        name = args[0].lower()
        expected_units = {
            "lattice": {"percent", "%"},
            "density": {"percent", "%"},
            "surface": {"percent", "%"},
            "angles": {"degree", "degrees", "deg"},
        }
        if name not in expected_units:
            raise InputFileError(
                path,
                line,
                "elasticity gate must be lattice, angles, density, or surface",
            )
        if args[2].lower() not in expected_units[name]:
            unit = "degree" if name == "angles" else "percent"
            raise InputFileError(
                path,
                line,
                f"elasticity {name} gate unit must be {unit}",
            )
        value = _float(path, line, args[1], f"elasticity {name} gate")
        if value <= 0.0:
            raise InputFileError(
                path,
                line,
                f"elasticity {name} gate must be positive",
            )
        set_setting(f"gate.{name}", value)
        return

    if command == "tier":
        if len(args) != 2 or args[1].lower() not in {"percent", "%"}:
            raise InputFileError(
                path,
                line,
                "elasticity tier syntax: tier VALUE percent",
            )
        value = _float(path, line, args[0], "elasticity reporting tier")
        if value <= 0.0:
            raise InputFileError(
                path,
                line,
                "elasticity reporting tier must be positive",
            )
        set_setting("tier", value)
        return

    if command == "born":
        if len(args) != 1 or args[0].lower() not in {"required", "off"}:
            raise InputFileError(
                path,
                line,
                "elasticity born syntax: born required|off",
            )
        set_setting("born", args[0].lower() == "required")
        return

    if command == "r2":
        if len(args) != 1:
            raise InputFileError(path, line, "elasticity r2 requires one value")
        value = _float(path, line, args[0], "elasticity minimum R2")
        if not 0.0 < value <= 1.0:
            raise InputFileError(
                path,
                line,
                "elasticity r2 must be in (0, 1]",
            )
        set_setting("minimum_r2", value)
        return

    if command == "strain":
        if len(args) < 2:
            raise InputFileError(
                path,
                line,
                "elasticity strain requires at least two positive magnitudes",
            )
        values = tuple(
            _float(path, line, value, "elasticity strain magnitude")
            for value in args
        )
        if any(value <= 0.0 or value > 0.05 for value in values):
            raise InputFileError(
                path,
                line,
                "elasticity strain magnitudes must be in (0, 0.05]",
            )
        if any(right <= left for left, right in zip(values, values[1:])):
            raise InputFileError(
                path,
                line,
                "elasticity strain magnitudes must be unique and strictly increasing",
            )
        set_setting("strain", values)
        return

    if command == "replicate":
        if len(args) != 3:
            raise InputFileError(
                path,
                line,
                "elasticity replicate requires NX NY NZ",
            )
        values = tuple(
            _int(path, line, value, "elasticity replicate") for value in args
        )
        if any(value < 1 for value in values):
            raise InputFileError(
                path,
                line,
                "elasticity replicate values must be positive integers",
            )
        set_setting("replicate", values)
        return

    if command in {"temperature", "timestep"}:
        accepted_units = {
            "temperature": {"k"},
            "timestep": {"fs"},
        }[command]
        if len(args) != 2 or args[1].lower() not in accepted_units:
            unit = "K" if command == "temperature" else "fs"
            raise InputFileError(
                path,
                line,
                f"elasticity {command} syntax: {command} VALUE {unit}",
            )
        value = _float(path, line, args[0], f"elasticity {command}")
        if value <= 0.0:
            raise InputFileError(
                path,
                line,
                f"elasticity {command} must be positive",
            )
        set_setting(command, value)
        return

    if command in {"equilibration", "production"}:
        if len(args) != 1:
            raise InputFileError(
                path,
                line,
                f"elasticity {command} requires one integer value",
            )
        value = _int(path, line, args[0], f"elasticity {command}")
        minimum = 0 if command == "equilibration" else 1
        if value < minimum:
            qualifier = "non-negative" if command == "equilibration" else "positive"
            raise InputFileError(
                path,
                line,
                f"elasticity {command} must be {qualifier}",
            )
        set_setting(command, value)
        return

    if command in {"seeds", "validation_seeds"}:
        label = (
            "seeds" if command == "seeds" else "validation_seeds"
        )
        if not args:
            raise InputFileError(
                path,
                line,
                f"elasticity {label} requires one or more values",
            )
        values = tuple(
            _int(path, line, value, f"elasticity {label} value")
            for value in args
        )
        if any(value < 1 for value in values):
            raise InputFileError(
                path,
                line,
                f"elasticity {label} must be positive integers",
            )
        if len(set(values)) != len(values):
            raise InputFileError(
                path, line, f"elasticity {label} must be unique"
            )
        set_setting(command, values)
        return

    raise InputFileError(
        path,
        line,
        f"unknown elasticity property command {tokens[0]!r}",
    )


def _parse_parameter_line(document: FFOptInput, line: int, tokens: list[str]) -> None:
    path = document.path
    params = document.parameters
    command = tokens[0].lower()
    args = tokens[1:]
    if command == "type":
        if len(args) not in {4, 5}:
            raise InputFileError(
                path, line,
                "type requires ID LABEL EPSILON SIGMA [CHARGE]",
            )
        atom_type = AtomTypeSpec(
            _int(path, line, args[0], "type ID"),
            args[1],
            _float(path, line, args[2], "epsilon"),
            _float(path, line, args[3], "sigma"),
            _float(path, line, args[4], "charge") if len(args) == 5 else 0.0,
            len(args) == 5,
            line,
        )
        for previous in params.atom_types:
            if atom_type.type_id == previous.type_id:
                raise InputFileError(
                    path,
                    line,
                    f"duplicate type ID {atom_type.type_id} "
                    f"(first set on line {previous.line})",
                )
            if atom_type.label == previous.label:
                raise InputFileError(
                    path,
                    line,
                    f"duplicate type label {atom_type.label!r} "
                    f"(first set on line {previous.line})",
                )
        params.atom_types.append(atom_type)
        return
    if command == "fix":
        if not args:
            raise InputFileError(path, line, "fix requires one or more parameter names")
        if args[0].lower() == "type":
            if len(args) < 3:
                raise InputFileError(path, line, "fix type requires LABEL PARAMETER...")
            label = args[1]
            names = {name.lower() for name in args[2:]}
            unknown = names - PARAMETER_NAMES
            if unknown:
                raise InputFileError(path, line, f"unknown parameter(s): {sorted(unknown)}")
            for name in names:
                key = (label, name)
                if key in params.fixed_by_type:
                    previous = params.fixed_by_type_lines[key]
                    raise InputFileError(
                        path,
                        line,
                        f"duplicate fix for type {label!r} parameter {name!r} "
                        f"(first set on line {previous})",
                    )
                params.fixed_by_type.add(key)
                params.fixed_by_type_lines[key] = line
        else:
            names = {name.lower() for name in args}
            unknown = names - PARAMETER_NAMES
            if unknown:
                raise InputFileError(path, line, f"unknown parameter(s): {sorted(unknown)}")
            for name in names:
                if name in params.fixed:
                    previous = params.fixed_lines[name]
                    raise InputFileError(
                        path,
                        line,
                        f"duplicate fix for parameter {name!r} "
                        f"(first set on line {previous})",
                    )
                params.fixed.add(name)
                params.fixed_lines[name] = line
        return
    if command == "range":
        if len(args) < 2:
            raise InputFileError(path, line, "range requires PARAMETER and range definition")
        if args[0].lower() == "type":
            if len(args) < 4:
                raise InputFileError(path, line, "range type requires LABEL PARAMETER MODE...")
            label, name = args[1], args[2].lower()
            if name not in PARAMETER_NAMES:
                raise InputFileError(path, line, f"unknown parameter {name!r}")
            key = (label, name)
            if key in params.type_ranges:
                previous = params.type_ranges[key].line
                raise InputFileError(
                    path,
                    line,
                    f"duplicate range for type {label!r} parameter {name!r} "
                    f"(first set on line {previous})",
                )
            params.type_ranges[key] = _range_spec(path, line, args[3:])
        else:
            name = args[0].lower()
            if name not in PARAMETER_NAMES:
                raise InputFileError(path, line, f"unknown parameter {name!r}")
            if name in params.default_ranges:
                previous = params.default_ranges[name].line
                raise InputFileError(
                    path,
                    line,
                    f"duplicate default range for parameter {name!r} "
                    f"(first set on line {previous})",
                )
            params.default_ranges[name] = _range_spec(path, line, args[1:])
        return
    if command == "tie":
        if len(args) != 2 or args[0].lower() not in PARAMETER_NAMES:
            raise InputFileError(path, line, "tie syntax: tie PARAMETER all")
        parameter = args[0].lower()
        if args[1].lower() != "all":
            raise InputFileError(path, line, "tie currently requires the group 'all'")
        if parameter in params.ties:
            previous = params.ties[parameter].line
            raise InputFileError(
                path,
                line,
                f"duplicate tie for parameter {parameter!r} "
                f"(first set on line {previous})",
            )
        params.ties[parameter] = TieSpec(parameter, "all", line)
        return
    if command == "difference":
        if len(args) not in {5, 6} or args[3].lower() != "max":
            raise InputFileError(
                path,
                line,
                "difference syntax: difference sigma LABEL1 LABEL2 max VALUE [A]",
            )
        parameter = args[0].lower()
        if parameter != "sigma":
            raise InputFileError(path, line, "difference currently supports sigma only")
        if len(args) == 6 and args[5].lower() not in {"a", "angstrom", "angstroms"}:
            raise InputFileError(path, line, "sigma difference unit must be A/angstrom")
        first, second = args[1], args[2]
        if first == second:
            raise InputFileError(path, line, "difference requires two distinct type labels")
        maximum = _float(path, line, args[4], "maximum sigma difference")
        if maximum <= 0.0:
            raise InputFileError(path, line, "maximum sigma difference must be positive")
        for previous in params.differences:
            if previous.parameter == parameter and {
                previous.first, previous.second
            } == {first, second}:
                raise InputFileError(
                    path,
                    line,
                    f"duplicate difference for {first!r}/{second!r} "
                    f"(first set on line {previous.line})",
                )
        params.differences.append(
            DifferenceSpec(parameter, first, second, maximum, line)
        )
        return
    if command == "neutrality":
        if "neutrality" in params.setting_lines:
            previous = params.setting_lines["neutrality"]
            raise InputFileError(
                path,
                line,
                f"duplicate neutrality setting (first set on line {previous})",
            )
        if len(args) == 1 and args[0].lower() in {"none", "off"}:
            params.derive_charge = None
        elif len(args) == 2 and args[0].lower() == "derive":
            params.derive_charge = args[1]
        else:
            raise InputFileError(path, line, "neutrality syntax: derive LABEL | none")
        params.setting_lines["neutrality"] = line
        return
    if command == "mixing":
        if len(args) == 1:
            rule = args[0].lower()
            if rule not in {"default", "geometric", "arithmetic"}:
                raise InputFileError(
                    path,
                    line,
                    "mixing rule must be default, geometric, or arithmetic",
                )
            # LAMMPS mix arithmetic still uses geometric epsilon mixing.  The
            # explicit ``default`` value is retained for the backend so it can
            # omit pair_modify and let the selected pair style decide.
            for parameter in ("epsilon", "sigma"):
                setting = f"mixing_{parameter}"
                if setting in params.setting_lines:
                    previous = params.setting_lines[setting]
                    raise InputFileError(
                        path,
                        line,
                        f"duplicate mixing rule for {parameter!r} "
                        f"(first set on line {previous})",
                    )
                params.setting_lines[setting] = line
            params.mixing_epsilon = "default" if rule == "default" else "geometric"
            params.mixing_sigma = rule
        elif len(args) == 2 and args[0].lower() in {"epsilon", "sigma"}:
            rule = args[1].lower()
            if rule not in {"geometric", "arithmetic"}:
                raise InputFileError(
                    path, line, "mixing rule must be geometric or arithmetic"
                )
            parameter = args[0].lower()
            if parameter == "epsilon" and rule != "geometric":
                raise InputFileError(
                    path, line,
                    "the current LAMMPS backend requires geometric epsilon mixing",
                )
            setting = f"mixing_{parameter}"
            if setting in params.setting_lines:
                previous = params.setting_lines[setting]
                raise InputFileError(
                    path,
                    line,
                    f"duplicate mixing rule for {parameter!r} "
                    f"(first set on line {previous})",
                )
            params.setting_lines[setting] = line
            setattr(params, f"mixing_{parameter}", rule)
        else:
            raise InputFileError(
                path,
                line,
                "mixing syntax: mixing default|geometric|arithmetic",
            )
        return
    if command == "charge_limit":
        if "charge_limit" in params.setting_lines:
            previous = params.setting_lines["charge_limit"]
            raise InputFileError(
                path,
                line,
                f"duplicate charge_limit setting (first set on line {previous})",
            )
        if len(args) not in {1, 2}:
            raise InputFileError(path, line, "charge_limit requires VALUE [e]")
        if len(args) == 2 and args[1].lower() != "e":
            raise InputFileError(path, line, "charge_limit optional unit must be e")
        params.charge_abs_max = _float(path, line, args[0], "charge limit")
        if params.charge_abs_max <= 0.0:
            raise InputFileError(path, line, "charge_limit must be positive")
        params.setting_lines["charge_limit"] = line
        return
    raise InputFileError(path, line, f"unknown parameters command {tokens[0]!r}")


def _parse_property_line(document: FFOptInput, prop: PropertySpec, line: int, tokens: list[str]) -> None:
    path = document.path
    command = tokens[0].lower()
    args = tokens[1:]
    if prop.name == "elasticity":
        _parse_elasticity_line(document, prop, line, tokens)
        return
    if command == "target":
        prop.targets.append(_parse_target(path, prop, line, args))
        return
    data_roles = {
        "bulk": {"data", "bulk"},
        "sublimation": {"bulk", "single"},
        "adsorption": {"complex", "slab", "molecule", "mol"},
        "surface": {"complete", "split"},
    }[prop.name]
    if command == "data":
        if prop.name == "bulk" and len(args) == 1:
            role, value = "bulk", args[0]
        elif len(args) == 2:
            role, value = args[0].lower(), args[1]
        else:
            raise InputFileError(path, line, "data requires PATH or ROLE PATH")
        role = "molecule" if role == "mol" else role
        valid_roles = {"bulk" if item == "data" else item for item in data_roles}
        valid_roles.discard("mol")
        if role not in valid_roles:
            raise InputFileError(
                path, line,
                f"property {prop.name!r} does not accept data role {role!r}",
            )
        if role in prop.data_files:
            raise InputFileError(path, line, f"duplicate data role {role!r}")
        prop.data_files[role] = value
        return
    if command in data_roles and len(args) == 1:
        role = "molecule" if command == "mol" else command
        role = "bulk" if role == "data" else role
        if role in prop.data_files:
            raise InputFileError(path, line, f"duplicate data role {role!r}")
        prop.data_files[role] = args[0]
        return
    if not args:
        raise InputFileError(path, line, f"{command} requires a value")

    bulk_settings = {
        "cells_in_data", "temperature", "pressure", "timestep", "cutoff",
        "equilibration", "production", "seed",
    }
    if document.crystal_family == "bcc":
        bulk_settings.update({"replicate", "tdamp", "pdamp"})
    allowed_settings = {
        "bulk": bulk_settings,
        "sublimation": {"temperature", "cutoff"},
        "adsorption": {"cutoff", "protocol", "metal"},
        "surface": {"facet", "replicate"},
    }[prop.name]
    if command not in allowed_settings:
        raise InputFileError(
            path,
            line,
            f"property {prop.name!r} does not use setting {command!r}",
        )

    def set_setting(key: str, value: Any) -> None:
        if key in prop.settings:
            previous = prop.setting_lines[key]
            raise InputFileError(
                path, line, f"duplicate {key!r} setting (first set on line {previous})"
            )
        prop.settings[key] = value
        prop.setting_lines[key] = line

    if command == "cells_in_data":
        if len(args) != 3:
            raise InputFileError(path, line, "cells_in_data requires NX NY NZ")
        set_setting(
            "cells",
            tuple(_int(path, line, value, command) for value in args),
        )
        return
    if command == "replicate":
        if len(args) != 3:
            raise InputFileError(path, line, "replicate requires NX NY NZ")
        set_setting(
            "replicate",
            tuple(_int(path, line, value, command) for value in args),
        )
        return
    if command in {"temperature", "pressure", "timestep", "cutoff", "tdamp", "pdamp"}:
        units = {
            "temperature": {"k"},
            "pressure": {"atm"},
            "timestep": {"fs"},
            "cutoff": {"a", "angstrom", "angstroms"},
            "tdamp": {"fs"},
            "pdamp": {"fs"},
        }[command]
        if len(args) not in {1, 2}:
            raise InputFileError(path, line, f"{command} requires VALUE [UNIT]")
        if len(args) == 2 and args[1].lower() not in units:
            expected = "/".join(sorted(units))
            raise InputFileError(
                path, line, f"{command} unit must be {expected}, got {args[1]!r}"
            )
        set_setting(command, _float(path, line, args[0], command))
        return
    if command in {"equilibration", "production", "seed"}:
        if len(args) != 1:
            raise InputFileError(path, line, f"{command} requires one integer value")
        set_setting(command, _int(path, line, args[0], command))
        return
    if command in {"protocol", "metal"}:
        if len(args) != 1:
            raise InputFileError(path, line, f"{command} requires one value")
        set_setting(command, args[0].lower() if command != "metal" else args[0])
        return
    if command == "facet":
        if len(args) != 1:
            raise InputFileError(path, line, "facet requires one Miller-index value")
        set_setting("facet", args[0])
        return
    raise InputFileError(path, line, f"unknown {prop.name} property command {tokens[0]!r}")


def _parse_stage_line(document: FFOptInput, stage: str, line: int, tokens: list[str]) -> None:
    if len(tokens) < 2:
        raise InputFileError(document.path, line, f"{tokens[0]} requires a value")
    settings = document.stage_settings.setdefault(stage, {})
    setting_lines = document.stage_setting_lines.setdefault(stage, {})
    key = tokens[0].lower()
    values = [_scalar(value) for value in tokens[1:]]
    if key == "early_stop":
        if len(tokens) != 3:
            raise InputFileError(document.path, line, "early_stop requires KEY VALUE")
        child = tokens[1].lower()
        early_stop = settings.setdefault("early_stop", {})
        if child in early_stop:
            raise InputFileError(document.path, line, f"duplicate early_stop {child!r}")
        early_stop[child] = _scalar(tokens[2])
        setting_lines[f"early_stop.{child}"] = line
        return
    if key in settings:
        raise InputFileError(document.path, line, f"duplicate {stage} setting {key!r}")
    settings[key] = values[0] if len(values) == 1 else values
    setting_lines[key] = line


def _validate_elasticity_property(
    document: FFOptInput,
    prop: PropertySpec,
) -> None:
    """Validate cross-line invariants of one cubic-elasticity block."""

    path = document.path
    static = prop.elasticity_modules.get("static")
    if static is None:
        raise InputFileError(
            path,
            prop.line,
            "elasticity requires 'module static objective'",
        )
    if document.crystal_family != "bcc":
        line = document.crystal_line or prop.line
        raise InputFileError(
            path,
            line,
            "the schema-1 cubic elasticity contract requires 'crystal bcc'",
        )

    required_targets = {"B", "Cprime", "C44"}
    for fidelity, module in prop.elasticity_modules.items():
        names = {
            target.name
            for target in prop.elasticity_targets
            if target.fidelity == fidelity
        }
        if names != required_targets:
            missing = sorted(required_targets - names)
            extra = sorted(names - required_targets)
            details = []
            if missing:
                details.append(f"missing={missing}")
            if extra:
                details.append(f"extra={extra}")
            raise InputFileError(
                path,
                module.line,
                f"elasticity {fidelity} module requires exactly independent "
                f"B/Cprime/C44 targets ({', '.join(details)})",
            )

    for target in prop.elasticity_targets:
        if target.fidelity not in prop.elasticity_modules:
            raise InputFileError(
                path,
                target.line,
                f"elasticity {target.fidelity} target requires "
                f"'module {target.fidelity} ...'",
            )

    dynamic_only_settings = {
        "temperature",
        "timestep",
        "equilibration",
        "production",
        "seeds",
        "validation_seeds",
    }
    if "dynamic" not in prop.elasticity_modules:
        unused = sorted(dynamic_only_settings & set(prop.settings))
        if unused:
            key = unused[0]
            raise InputFileError(
                path,
                prop.setting_lines[key],
                f"elasticity {key} requires a dynamic module",
            )
    if "validation_seeds" in prop.settings:
        promotion_seeds = set(prop.settings.get("seeds", (101, 202, 303)))
        holdout_seeds = set(prop.settings["validation_seeds"])
        overlap = sorted(promotion_seeds & holdout_seeds)
        if overlap:
            raise InputFileError(
                path,
                prop.setting_lines["validation_seeds"],
                "elasticity validation_seeds must be disjoint from promotion "
                f"seeds; overlap={overlap}",
            )


def _validate(document: FFOptInput) -> None:
    path = document.path
    if document.version is None:
        raise InputFileError(path, 1, "missing 'ffopt 1' schema declaration")
    if not document.project:
        raise InputFileError(path, 1, "missing 'project NAME'")
    if not document.workflow:
        raise InputFileError(path, 1, "missing 'workflow STAGE ...' declaration")
    if not document.parameters.atom_types:
        raise InputFileError(path, 1, "parameters block contains no atom types")
    ids = [item.type_id for item in document.parameters.atom_types]
    labels = [item.label for item in document.parameters.atom_types]
    if len(ids) != len(set(ids)):
        raise InputFileError(path, 1, "atom type IDs must be unique")
    if len(labels) != len(set(labels)):
        raise InputFileError(path, 1, "atom type labels must be unique")
    if min(ids) < 1:
        raise InputFileError(path, 1, "atom type IDs must be positive")
    for item in document.parameters.atom_types:
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", item.label):
            raise InputFileError(
                path,
                item.line,
                "type label must start with a letter and contain only letters, "
                "numbers, or underscores",
            )
        if item.epsilon <= 0.0:
            raise InputFileError(path, item.line, "initial epsilon must be positive")
        if item.sigma <= 0.0:
            raise InputFileError(path, item.line, "initial sigma must be positive")
        if abs(item.charge) > document.parameters.charge_abs_max:
            raise InputFileError(
                path,
                item.line,
                f"initial charge {item.charge:g} exceeds charge_limit "
                f"{document.parameters.charge_abs_max:g}",
            )
        if document.material_kind != "elemental" and not item.charge_explicit:
            raise InputFileError(
                path,
                item.line,
                "charge may be omitted only for an elemental material",
            )
        if document.material_kind == "elemental" and abs(item.charge) > 1.0e-12:
            raise InputFileError(
                path,
                item.line,
                "elemental atom types must have zero charge",
            )
    if document.material_kind == "elemental" and document.parameters.derive_charge is not None:
        line = document.parameters.setting_lines.get("neutrality", 1)
        raise InputFileError(path, line, "elemental materials do not use charge neutrality")
    if document.parameters.derive_charge not in {None, *labels}:
        raise InputFileError(path, 1, f"neutrality label {document.parameters.derive_charge!r} is unknown")
    known_labels = set(labels)
    for (label, name), spec in document.parameters.type_ranges.items():
        if label not in known_labels:
            raise InputFileError(
                path,
                spec.line,
                f"range refers to unknown type label {label!r}",
            )
        if (
            name in document.parameters.fixed
            or (label, name) in document.parameters.fixed_by_type
            or (document.material_kind == "elemental" and name == "charge")
        ):
            raise InputFileError(
                path,
                spec.line,
                f"type {label!r} parameter {name!r} cannot be both fixed and ranged",
            )
    for key, line in document.parameters.fixed_by_type_lines.items():
        label, _ = key
        if label not in known_labels:
            raise InputFileError(
                path,
                line,
                f"fix refers to unknown type label {label!r}",
            )
    for name, spec in document.parameters.default_ranges.items():
        if name in document.parameters.fixed or (
            document.material_kind == "elemental" and name == "charge"
        ):
            raise InputFileError(
                path,
                spec.line,
                f"parameter {name!r} cannot be both globally fixed and ranged",
            )
    params = document.parameters
    for tie in params.ties.values():
        if len(known_labels) < 2:
            raise InputFileError(
                path,
                tie.line,
                f"tie {tie.parameter!r} requires at least two atom types",
            )
        typed_fixed = [
            label for label in labels
            if (label, tie.parameter) in params.fixed_by_type
        ]
        if typed_fixed:
            raise InputFileError(
                path,
                tie.line,
                f"tie {tie.parameter!r} overlaps type-specific fixes for: "
                + ", ".join(typed_fixed),
            )
    seen_differences: set[tuple[str, frozenset[str]]] = set()
    for constraint in params.differences:
        unknown = {constraint.first, constraint.second} - known_labels
        if unknown:
            raise InputFileError(
                path,
                constraint.line,
                f"difference refers to unknown type label(s): {sorted(unknown)}",
            )
        key = (constraint.parameter, frozenset({constraint.first, constraint.second}))
        if key in seen_differences:  # Defensive: parser normally catches this.
            raise InputFileError(path, constraint.line, "duplicate difference constraint")
        seen_differences.add(key)
        if constraint.parameter in params.ties:
            raise InputFileError(
                path,
                constraint.line,
                f"parameter {constraint.parameter!r} cannot be both tied and difference-limited",
            )
    if "bo" in document.workflow:
        implicit_fixed = set(document.parameters.fixed)
        if document.material_kind == "elemental":
            implicit_fixed.add("charge")
        for name in PARAMETER_NAMES - implicit_fixed:
            missing = [
                item.label for item in document.parameters.atom_types
                if (item.label, name) not in document.parameters.fixed_by_type
                and (item.label, name) not in document.parameters.type_ranges
                and name not in document.parameters.default_ranges
            ]
            if missing:
                raise InputFileError(
                    path, 1,
                    f"free parameter {name!r} has no range for type(s): {', '.join(missing)}",
                )
    prop_names = [item.name for item in document.properties]
    if len(prop_names) != len(set(prop_names)):
        raise InputFileError(path, 1, "each property block may appear only once")
    for prop in document.properties:
        if prop.name == "elasticity":
            _validate_elasticity_property(document, prop)
        if (
            prop.name == "bulk"
            and document.crystal_family == "bcc"
            and "cells" not in prop.settings
        ):
            raise InputFileError(
                path,
                prop.line,
                "BCC bulk requires explicit 'cells_in_data NX NY NZ'",
            )
    if (
        any(stage in {"bo", "sample", "nn", "al"} for stage in document.workflow)
        and not any(prop.fitted for prop in document.properties)
    ):
        raise InputFileError(path, 1, "optimization workflow requires at least one target")
    active_targets = [
        target
        for prop in document.properties
        for target in prop.targets
        if target.weight > 0.0
    ]
    if any(stage in {"bo", "sample", "nn", "al"} for stage in document.workflow):
        if not active_targets:
            if any(prop.elasticity_targets for prop in document.properties):
                raise InputFileError(
                    path,
                    1,
                    "structural BO/NN/AL requires at least one non-elastic target "
                    "with positive weight; static elasticity is a downstream "
                    "constrained objective",
                )
            raise InputFileError(
                path, 1, "optimization workflow requires a target with positive weight"
            )
        zero_targets = [target.name for target in active_targets if target.value == 0.0]
        if zero_targets:
            raise InputFileError(
                path, 1,
                "weighted relative RMSE requires nonzero active target values: "
                + ", ".join(zero_targets),
            )
    material_workflow = (
        document.material_kind != "molecular"
        or document.crystal_family != "molecular"
        or any(stage in {"screen", "finalists"} for stage in document.workflow)
    )
    stage_order = (
        MATERIAL_PIPELINE_STAGE_ORDER
        if material_workflow
        else LEGACY_PIPELINE_STAGE_ORDER
    )
    unsupported = set(document.workflow) - set(stage_order)
    if unsupported:
        kind = "material" if material_workflow else "molecular"
        raise InputFileError(
            path,
            1,
            f"{kind} workflow does not support stage(s): {sorted(unsupported)}; "
            f"use {' -> '.join(stage_order)}",
        )
    positions = [stage_order.index(stage) for stage in document.workflow]
    if positions != sorted(positions):
        raise InputFileError(
            path,
            1,
            "workflow stages must follow: " + " -> ".join(stage_order),
        )

    unused_blocks = set(document.stage_settings) - set(document.workflow)
    if unused_blocks:
        raise InputFileError(
            path, 1,
            "method block(s) are not present in workflow: "
            + ", ".join(sorted(unused_blocks)),
        )
    required = (
        {
            "sample": {"bo"},
            "audit": {"bo", "sample"},
            "screen": {"audit"},
            "nn": {"screen"},
            "al": {"nn", "screen"},
            "finalists": {"al"},
        }
        if material_workflow
        else {
            "sample": {"bo"},
            "nn": {"bo"},
            "al": {"bo", "nn"},
            "audit": {"bo"},
            "finalize": {"audit"},
        }
    )
    active = set(document.workflow)
    for stage, dependencies in required.items():
        if stage in active and not dependencies <= active:
            missing = sorted(dependencies - active)
            raise InputFileError(
                path, 1, f"workflow stage {stage!r} requires {missing}"
            )


def parse_input_file(value: str | Path) -> FFOptInput:
    """Parse one FFOpt command file without executing or touching LAMMPS."""
    path = Path(value).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"FFOpt input file not found: {path}")
    document = FFOptInput(path=path)
    block: tuple[str, Any] | None = None
    seen_top_level: set[str] = set()
    for line_number, raw in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        tokens = _tokens(path, line_number, raw)
        if not tokens:
            continue
        command = tokens[0].lower()
        if command == "end":
            if len(tokens) != 1 or block is None:
                raise InputFileError(path, line_number, "unexpected end")
            block = None
            continue
        if block is not None:
            kind, payload = block
            if kind == "parameters":
                _parse_parameter_line(document, line_number, tokens)
            elif kind == "property":
                _parse_property_line(document, payload, line_number, tokens)
            else:
                _parse_stage_line(document, payload, line_number, tokens)
            continue
        if command == "ffopt":
            if command in seen_top_level:
                raise InputFileError(path, line_number, "duplicate ffopt declaration")
            seen_top_level.add(command)
            if len(tokens) != 2:
                raise InputFileError(path, line_number, "ffopt requires schema version 1")
            document.version = _int(path, line_number, tokens[1], "schema version")
            if document.version != 1:
                raise InputFileError(path, line_number, f"unsupported schema version {document.version}")
        elif command == "project":
            if command in seen_top_level:
                raise InputFileError(path, line_number, "duplicate project declaration")
            seen_top_level.add(command)
            if len(tokens) != 2 or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_.-]*", tokens[1]):
                raise InputFileError(path, line_number, "project requires a portable NAME")
            document.project = tokens[1]
        elif command == "material":
            if command in seen_top_level:
                raise InputFileError(path, line_number, "duplicate material declaration")
            seen_top_level.add(command)
            if len(tokens) not in {2, 3} or tokens[1].lower() not in MATERIAL_KINDS:
                raise InputFileError(
                    path,
                    line_number,
                    "material syntax: material elemental|molecular|multicomponent [NAME]",
                )
            if len(tokens) == 3 and not re.fullmatch(
                r"[A-Za-z][A-Za-z0-9_.-]*", tokens[2]
            ):
                raise InputFileError(path, line_number, "material NAME must be portable")
            document.material_kind = tokens[1].lower()
            document.material_name = tokens[2] if len(tokens) == 3 else None
            document.material_line = line_number
        elif command == "crystal":
            if command in seen_top_level:
                raise InputFileError(path, line_number, "duplicate crystal declaration")
            seen_top_level.add(command)
            if len(tokens) != 2 or tokens[1].lower() not in CRYSTAL_FAMILIES:
                raise InputFileError(
                    path,
                    line_number,
                    "crystal must be bcc or molecular",
                )
            document.crystal_family = tokens[1].lower()
            document.crystal_line = line_number
        elif command == "workflow":
            if command in seen_top_level:
                raise InputFileError(path, line_number, "duplicate workflow declaration")
            seen_top_level.add(command)
            if len(tokens) < 2:
                raise InputFileError(path, line_number, "workflow requires at least one stage")
            stages = [stage.lower() for stage in tokens[1:]]
            unknown = set(stages) - PIPELINE_STAGES
            if unknown:
                raise InputFileError(path, line_number, f"unknown workflow stages: {sorted(unknown)}")
            if len(stages) != len(set(stages)):
                raise InputFileError(path, line_number, "workflow stages cannot repeat")
            document.workflow = stages
        elif command == "parameters":
            if command in seen_top_level:
                raise InputFileError(path, line_number, "duplicate parameters block")
            seen_top_level.add(command)
            if len(tokens) != 1:
                raise InputFileError(path, line_number, "parameters takes no arguments")
            block = ("parameters", None)
        elif command == "property":
            if len(tokens) != 2 or tokens[1].lower() not in PROPERTY_NAMES:
                raise InputFileError(path, line_number, f"property must be one of {sorted(PROPERTY_NAMES)}")
            prop = PropertySpec(tokens[1].lower(), line_number)
            document.properties.append(prop)
            block = ("property", prop)
        elif command in PIPELINE_STAGES:
            if command in seen_top_level:
                raise InputFileError(path, line_number, f"duplicate {command} block")
            seen_top_level.add(command)
            if len(tokens) != 1:
                raise InputFileError(path, line_number, f"{command} block takes no arguments")
            block = ("stage", command)
        else:
            raise InputFileError(path, line_number, f"unknown top-level command {tokens[0]!r}")
    if block is not None:
        raise InputFileError(path, line_number if 'line_number' in locals() else 1, f"unterminated {block[0]} block")
    _validate(document)
    return document
