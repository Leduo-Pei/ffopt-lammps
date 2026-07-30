"""Validate one or more molecular LAMMPS data files as a coherent project."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

from .lammps_data import AtomTypeSummary, LammpsDataSummary, inspect_lammps_data

SUPPORTED_COEFFICIENT_STYLES = {
    "Bond Coeffs": "harmonic",
    "Angle Coeffs": "harmonic",
    "Dihedral Coeffs": "harmonic",
    "Improper Coeffs": "cvff",
}


@dataclass(frozen=True)
class DataFinding:
    severity: str
    code: str
    message: str
    roles: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class DataContractReport:
    files: dict[str, LammpsDataSummary]
    findings: tuple[DataFinding, ...]

    @property
    def errors(self) -> tuple[DataFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "ERROR")

    @property
    def warnings(self) -> tuple[DataFinding, ...]:
        return tuple(item for item in self.findings if item.severity == "WARNING")

    @property
    def ok(self) -> bool:
        return not self.errors

    def to_dict(self) -> dict:
        return {
            "ok": self.ok,
            "files": {role: summary.to_dict() for role, summary in self.files.items()},
            "findings": [item.to_dict() for item in self.findings],
        }


def validate_molecular_data_file(
    summary: LammpsDataSummary, *, role: str
) -> list[DataFinding]:
    """Validate the contract required by the bundled molecular templates."""
    findings: list[DataFinding] = []
    if summary.atom_style != "full":
        findings.append(DataFinding(
            "ERROR", "atom_style",
            f"{role} uses atom_style {summary.atom_style!r}; expected 'full'.",
            (role,),
        ))
    parsed_atoms = sum(item.atom_count for item in summary.atom_types)
    declared_atoms = summary.declared_counts.get("atoms")
    if declared_atoms is not None and parsed_atoms != declared_atoms:
        findings.append(DataFinding(
            "ERROR", "atom_count",
            f"{role} declares {declared_atoms} atoms but {parsed_atoms} were parsed.",
            (role,),
        ))
    labels: dict[str, list[int]] = {}
    for atom_type in summary.atom_types:
        labels.setdefault(atom_type.label, []).append(atom_type.type_id)
        if atom_type.mass is None or atom_type.mass <= 0.0:
            findings.append(DataFinding(
                "ERROR", "mass",
                f"{role} type {atom_type.type_id} ({atom_type.label}) has no positive mass.",
                (role,),
            ))
        if len(atom_type.pair_coefficients) < 2:
            findings.append(DataFinding(
                "ERROR", "pair_coeffs",
                f"{role} type {atom_type.type_id} ({atom_type.label}) has no epsilon/sigma pair.",
                (role,),
            ))
        else:
            epsilon, sigma = atom_type.pair_coefficients[:2]
            if epsilon <= 0.0 or sigma <= 0.0:
                findings.append(DataFinding(
                    "ERROR", "pair_coeffs",
                    f"{role} type {atom_type.type_id} ({atom_type.label}) has "
                    f"epsilon={epsilon:g}, sigma={sigma:g}; both must be positive.",
                    (role,),
                ))
            if len(atom_type.pair_coefficients) > 2:
                findings.append(DataFinding(
                    "WARNING", "pair_coeffs_extra",
                    f"{role} type {atom_type.type_id} ({atom_type.label}) has extra Pair "
                    "Coeffs columns; only epsilon and sigma are used.",
                    (role,),
                ))
        if len(atom_type.charges) > 1:
            findings.append(DataFinding(
                "ERROR", "multiple_type_charges",
                f"{role} type {atom_type.type_id} ({atom_type.label}) has multiple charges; "
                "FFOpt currently assigns one charge per type.",
                (role,),
            ))
    for label, type_ids in labels.items():
        if len(type_ids) > 1:
            findings.append(DataFinding(
                "ERROR", "duplicate_label",
                f"{role} label {label!r} is used by type IDs {type_ids}; labels must be unique.",
                (role,),
            ))
    for section, expected in SUPPORTED_COEFFICIENT_STYLES.items():
        declared = summary.section_styles.get(section)
        if declared and declared.split()[0].lower() != expected:
            findings.append(DataFinding(
                "ERROR", "coefficient_style",
                f"{role} declares '{section} # {declared}'; expected {expected!r}.",
                (role,),
            ))
    if summary.total_charge is not None and abs(summary.total_charge) > 1.0e-6:
        findings.append(DataFinding(
            "WARNING", "net_charge",
            f"{role} initial total charge is {summary.total_charge:.8g} e, not neutral.",
            (role,),
        ))
    return findings


def _types_by_label(summary: LammpsDataSummary) -> dict[str, AtomTypeSummary]:
    return {item.label: item for item in summary.atom_types}


def _different(left: float | None, right: float | None, tolerance: float) -> bool:
    if left is None or right is None:
        return left != right
    return abs(float(left) - float(right)) > tolerance


def _charge(atom_type: AtomTypeSummary) -> float | None:
    return atom_type.charges[0] if len(atom_type.charges) == 1 else None


def _compare_type_sets(
    reference_role: str,
    reference: LammpsDataSummary,
    target_role: str,
    target: LammpsDataSummary,
    *,
    require_same_ids: bool,
    require_all_reference: bool = True,
) -> list[DataFinding]:
    findings: list[DataFinding] = []
    reference_types = _types_by_label(reference)
    target_types = _types_by_label(target)
    missing = sorted(set(reference_types) - set(target_types))
    if missing and require_all_reference:
        findings.append(DataFinding(
            "ERROR", "missing_labels",
            f"{target_role} is missing labels required by {reference_role}: {missing}.",
            (reference_role, target_role),
        ))
    mass_differences: list[str] = []
    id_differences: list[str] = []
    lj_differences: list[str] = []
    charge_differences: list[str] = []
    for label in sorted(set(reference_types) & set(target_types)):
        left, right = reference_types[label], target_types[label]
        if _different(left.mass, right.mass, 1.0e-6):
            mass_differences.append(label)
        if require_same_ids and left.type_id != right.type_id:
            id_differences.append(f"{label}:{left.type_id}->{right.type_id}")
        if (
            len(left.pair_coefficients) >= 2
            and len(right.pair_coefficients) >= 2
            and any(
                _different(a, b, 1.0e-8)
                for a, b in zip(left.pair_coefficients[:2], right.pair_coefficients[:2])
            )
        ):
            lj_differences.append(label)
        if _different(_charge(left), _charge(right), 1.0e-8):
            charge_differences.append(label)
    roles = (reference_role, target_role)
    if mass_differences:
        findings.append(DataFinding(
            "ERROR", "mass_mismatch",
            f"Mass differs for shared labels: {mass_differences}.", roles,
        ))
    if id_differences:
        findings.append(DataFinding(
            "ERROR", "type_id_mismatch",
            f"{reference_role} and {target_role} must use identical type IDs: {id_differences}.",
            roles,
        ))
    if lj_differences:
        findings.append(DataFinding(
            "WARNING", "initial_lj_mismatch",
            f"Initial epsilon/sigma differ for labels {lj_differences}; ffopt.in values "
            "will override molecular Pair Coeffs during evaluation.", roles,
        ))
    if charge_differences:
        findings.append(DataFinding(
            "WARNING", "initial_charge_mismatch",
            f"Initial charges differ for labels {charge_differences}; ffopt.in values "
            "will override molecular charges during evaluation.", roles,
        ))
    return findings


def check_data_files(**paths: str | Path | None) -> DataContractReport:
    """Check role-specific files and compatibility across one FFOpt project."""
    selected = {role: Path(path) for role, path in paths.items() if path is not None}
    if not selected:
        raise ValueError("Provide at least one data file role")
    summaries = {
        role: inspect_lammps_data(path) for role, path in selected.items()
    }
    findings: list[DataFinding] = []
    for role, summary in summaries.items():
        findings.extend(validate_molecular_data_file(summary, role=role))

    primary_role = "bulk" if "bulk" in summaries else (
        "molecule" if "molecule" in summaries else "single" if "single" in summaries else None
    )
    if primary_role:
        primary = summaries[primary_role]
        if "single" in summaries and primary_role != "single":
            findings.extend(_compare_type_sets(
                primary_role, primary, "single", summaries["single"],
                require_same_ids=True,
            ))
        if "molecule" in summaries and primary_role != "molecule":
            findings.extend(_compare_type_sets(
                primary_role, primary, "molecule", summaries["molecule"],
                require_same_ids=False,
            ))
        if "complex" in summaries:
            findings.extend(_compare_type_sets(
                primary_role, primary, "complex", summaries["complex"],
                require_same_ids=False,
            ))
    if "slab" in summaries and "complex" in summaries:
        findings.extend(_compare_type_sets(
            "slab", summaries["slab"], "complex", summaries["complex"],
            require_same_ids=False,
        ))
    if "molecule" in summaries and "complex" in summaries:
        findings.extend(_compare_type_sets(
            "molecule", summaries["molecule"], "complex", summaries["complex"],
            require_same_ids=False,
        ))
    return DataContractReport(summaries, tuple(findings))


def format_data_report(report: DataContractReport) -> str:
    lines = ["LAMMPS data contract", ""]
    for role, summary in report.files.items():
        lines.append(
            f"[{role}] {summary.path}\n"
            f"  style={summary.atom_style or '?'} atoms={summary.declared_counts.get('atoms', '?')} "
            f"types={len(summary.atom_types)} molecules={summary.molecule_count or '?'} "
            f"charge={summary.total_charge if summary.total_charge is not None else '?'}"
        )
    lines.append("")
    if report.findings:
        for finding in report.findings:
            roles = f" ({', '.join(finding.roles)})" if finding.roles else ""
            lines.append(
                f"{finding.severity:7s} {finding.code}{roles}: {finding.message}"
            )
    else:
        lines.append("No compatibility findings.")
    lines.append("")
    lines.append(
        f"Result: {'PASS' if report.ok else 'FAIL'} | "
        f"errors={len(report.errors)} warnings={len(report.warnings)}"
    )
    return "\n".join(lines)
