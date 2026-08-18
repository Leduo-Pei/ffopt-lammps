"""Parser and typed model for the public FFOpt command input format."""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

PARAMETER_NAMES = {"epsilon", "sigma", "charge"}
PIPELINE_STAGE_ORDER = ("bo", "sample", "nn", "al", "audit", "finalize", "validate")
PIPELINE_STAGES = set(PIPELINE_STAGE_ORDER)
PROPERTY_NAMES = {"bulk", "sublimation", "adsorption"}


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
    line: int


@dataclass
class ParameterSpec:
    atom_types: list[AtomTypeSpec] = field(default_factory=list)
    default_ranges: dict[str, RangeSpec] = field(default_factory=dict)
    type_ranges: dict[tuple[str, str], RangeSpec] = field(default_factory=dict)
    fixed: set[str] = field(default_factory=set)
    fixed_by_type: set[tuple[str, str]] = field(default_factory=set)
    derive_charge: str | None = None
    mixing_epsilon: str = "geometric"
    mixing_sigma: str = "geometric"
    charge_abs_max: float = 1.0


@dataclass(frozen=True)
class TargetSpec:
    name: str
    value: float
    unit: str
    weight: float
    tolerance: float | None
    line: int


@dataclass
class PropertySpec:
    name: str
    line: int
    data_files: dict[str, str] = field(default_factory=dict)
    targets: list[TargetSpec] = field(default_factory=list)
    settings: dict[str, Any] = field(default_factory=dict)
    setting_lines: dict[str, int] = field(default_factory=dict)

    @property
    def fitted(self) -> bool:
        return bool(self.targets)


@dataclass
class FFOptInput:
    path: Path
    version: int | None = None
    project: str = ""
    workflow: list[str] = field(default_factory=lambda: [
        "bo", "sample", "nn", "al", "validate"
    ])
    parameters: ParameterSpec = field(default_factory=ParameterSpec)
    properties: list[PropertySpec] = field(default_factory=list)
    stage_settings: dict[str, dict[str, Any]] = field(default_factory=dict)


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
        return float(value)
    except ValueError as exc:
        raise InputFileError(path, line, f"{label} must be numeric, got {value!r}") from exc


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
    if low == high:
        raise InputFileError(path, line, "range bounds must differ")
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


def _parse_parameter_line(document: FFOptInput, line: int, tokens: list[str]) -> None:
    path = document.path
    params = document.parameters
    command = tokens[0].lower()
    args = tokens[1:]
    if command == "type":
        if len(args) != 5:
            raise InputFileError(
                path, line,
                "type requires ID LABEL EPSILON SIGMA CHARGE",
            )
        atom_type = AtomTypeSpec(
            _int(path, line, args[0], "type ID"),
            args[1],
            _float(path, line, args[2], "epsilon"),
            _float(path, line, args[3], "sigma"),
            _float(path, line, args[4], "charge"),
            line,
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
            params.fixed_by_type.update((label, name) for name in names)
        else:
            names = {name.lower() for name in args}
            unknown = names - PARAMETER_NAMES
            if unknown:
                raise InputFileError(path, line, f"unknown parameter(s): {sorted(unknown)}")
            params.fixed.update(names)
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
            params.type_ranges[(label, name)] = _range_spec(path, line, args[3:])
        else:
            name = args[0].lower()
            if name not in PARAMETER_NAMES:
                raise InputFileError(path, line, f"unknown parameter {name!r}")
            params.default_ranges[name] = _range_spec(path, line, args[1:])
        return
    if command == "neutrality":
        if len(args) == 1 and args[0].lower() in {"none", "off"}:
            params.derive_charge = None
        elif len(args) == 2 and args[0].lower() == "derive":
            params.derive_charge = args[1]
        else:
            raise InputFileError(path, line, "neutrality syntax: derive LABEL | none")
        return
    if command == "mixing":
        if len(args) == 1:
            rule = args[0].lower()
            if rule not in {"geometric", "arithmetic"}:
                raise InputFileError(
                    path, line, "mixing rule must be geometric or arithmetic"
                )
            # LAMMPS mix arithmetic still uses geometric epsilon mixing.
            params.mixing_epsilon = "geometric"
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
            setattr(params, f"mixing_{parameter}", rule)
        else:
            raise InputFileError(
                path,
                line,
                "mixing syntax: mixing geometric|arithmetic",
            )
        return
    if command == "charge_limit":
        if len(args) not in {1, 2}:
            raise InputFileError(path, line, "charge_limit requires VALUE [e]")
        if len(args) == 2 and args[1].lower() != "e":
            raise InputFileError(path, line, "charge_limit optional unit must be e")
        params.charge_abs_max = _float(path, line, args[0], "charge limit")
        if params.charge_abs_max <= 0.0:
            raise InputFileError(path, line, "charge_limit must be positive")
        return
    raise InputFileError(path, line, f"unknown parameters command {tokens[0]!r}")


def _parse_property_line(document: FFOptInput, prop: PropertySpec, line: int, tokens: list[str]) -> None:
    path = document.path
    command = tokens[0].lower()
    args = tokens[1:]
    if command == "target":
        prop.targets.append(_parse_target(path, prop, line, args))
        return
    data_roles = {
        "bulk": {"data", "bulk"},
        "sublimation": {"bulk", "single"},
        "adsorption": {"complex", "slab", "molecule", "mol"},
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
    if command in {"temperature", "pressure", "timestep", "cutoff"}:
        units = {
            "temperature": {"k"},
            "pressure": {"atm"},
            "timestep": {"fs"},
            "cutoff": {"a", "angstrom", "angstroms"},
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
    if command in {"equilibration", "production", "seed", "molecule_atoms"}:
        if len(args) != 1:
            raise InputFileError(path, line, f"{command} requires one integer value")
        set_setting(command, _int(path, line, args[0], command))
        return
    if command in {"protocol", "metal"}:
        if len(args) != 1:
            raise InputFileError(path, line, f"{command} requires one value")
        set_setting(command, args[0].lower() if command != "metal" else args[0])
        return
    raise InputFileError(path, line, f"unknown {prop.name} property command {tokens[0]!r}")


def _parse_stage_line(document: FFOptInput, stage: str, line: int, tokens: list[str]) -> None:
    if len(tokens) < 2:
        raise InputFileError(document.path, line, f"{tokens[0]} requires a value")
    settings = document.stage_settings.setdefault(stage, {})
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
        return
    if key in settings:
        raise InputFileError(document.path, line, f"duplicate {stage} setting {key!r}")
    settings[key] = values[0] if len(values) == 1 else values


def _validate(document: FFOptInput) -> None:
    path = document.path
    if document.version is None:
        raise InputFileError(path, 1, "missing 'ffopt 1' schema declaration")
    if not document.project:
        raise InputFileError(path, 1, "missing 'project NAME'")
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
    if document.parameters.derive_charge not in {None, *labels}:
        raise InputFileError(path, 1, f"neutrality label {document.parameters.derive_charge!r} is unknown")
    for name in PARAMETER_NAMES - document.parameters.fixed:
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
    if (
        any(stage in {"bo", "sample", "nn", "al"} for stage in document.workflow)
        and not any(prop.fitted for prop in document.properties)
    ):
        raise InputFileError(path, 1, "optimization workflow requires at least one target")
    unused_blocks = set(document.stage_settings) - set(document.workflow)
    if unused_blocks:
        raise InputFileError(
            path, 1,
            "method block(s) are not present in workflow: "
            + ", ".join(sorted(unused_blocks)),
        )
    required = {
        "sample": {"bo"},
        "nn": {"bo"},
        "al": {"bo", "nn"},
        "audit": {"bo"},
        "finalize": {"audit"},
    }
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
            positions = [PIPELINE_STAGE_ORDER.index(stage) for stage in stages]
            if positions != sorted(positions):
                raise InputFileError(
                    path,
                    line_number,
                    "workflow stages must follow: " + " -> ".join(PIPELINE_STAGE_ORDER),
                )
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
        elif command in {"bo", "sample", "nn", "al", "validate"}:
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
