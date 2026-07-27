"""Lightweight, dependency-free inspection of LAMMPS data files."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any


SECTION_NAMES = {
    "Masses",
    "Pair Coeffs",
    "PairIJ Coeffs",
    "Bond Coeffs",
    "Angle Coeffs",
    "Dihedral Coeffs",
    "Improper Coeffs",
    "Atoms",
    "Velocities",
    "Bonds",
    "Angles",
    "Dihedrals",
    "Impropers",
}


@dataclass(frozen=True)
class AtomTypeSummary:
    type_id: int
    label: str
    mass: float | None
    pair_coefficients: list[float]
    atom_count: int
    charges: list[float]

    @property
    def charge_min(self) -> float | None:
        return min(self.charges) if self.charges else None

    @property
    def charge_max(self) -> float | None:
        return max(self.charges) if self.charges else None

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["charge_min"] = self.charge_min
        result["charge_max"] = self.charge_max
        result.pop("charges")
        return result


@dataclass(frozen=True)
class LammpsDataSummary:
    path: str
    title: str
    atom_style: str | None
    declared_counts: dict[str, int]
    molecule_count: int | None
    total_charge: float | None
    atom_types: list[AtomTypeSummary]
    section_styles: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "title": self.title,
            "atom_style": self.atom_style,
            "declared_counts": self.declared_counts,
            "molecule_count": self.molecule_count,
            "total_charge": self.total_charge,
            "section_styles": self.section_styles,
            "atom_types": [item.to_dict() for item in self.atom_types],
        }


def _section_header(line: str) -> tuple[str | None, str | None]:
    payload, _, comment = line.partition("#")
    name = payload.strip()
    if name in SECTION_NAMES:
        return name, comment.strip() or None
    return None, None


def _numeric_payload(line: str) -> tuple[list[str], str]:
    payload, _, comment = line.partition("#")
    return payload.split(), comment.strip()


def inspect_lammps_data(path: str | Path) -> LammpsDataSummary:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"LAMMPS data file not found: {source}")
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"LAMMPS data file is empty: {source}")

    declared: dict[str, int] = {}
    for line in lines[:80]:
        match = re.match(
            r"^\s*(\d+)\s+(atoms|bonds|angles|dihedrals|impropers|"
            r"atom types|bond types|angle types|dihedral types|improper types)\s*$",
            line,
            flags=re.IGNORECASE,
        )
        if match:
            declared[match.group(2).lower().replace(" ", "_")] = int(match.group(1))

    masses: dict[int, float] = {}
    pair_coeffs: dict[int, list[float]] = {}
    labels: dict[int, str] = {}
    type_counts: Counter[int] = Counter()
    charges: dict[int, list[float]] = defaultdict(list)
    molecules: set[int] = set()
    atom_ids: set[int] = set()
    bond_edges: list[tuple[int, int]] = []
    atom_style: str | None = None
    section_styles: dict[str, str] = {}
    section: str | None = None

    for line in lines:
        header, qualifier = _section_header(line)
        if header:
            section = header
            if header == "Atoms":
                atom_style = qualifier.lower() if qualifier else None
            elif qualifier:
                section_styles[header] = qualifier.lower()
            continue
        columns, comment = _numeric_payload(line)
        if not columns or not columns[0].lstrip("+-").isdigit():
            continue

        if section == "Masses" and len(columns) >= 2:
            type_id = int(columns[0])
            masses[type_id] = float(columns[1])
            if comment:
                labels.setdefault(type_id, comment.split()[0])
        elif section == "Pair Coeffs" and len(columns) >= 2:
            type_id = int(columns[0])
            pair_coeffs[type_id] = [float(value) for value in columns[1:]]
            if comment:
                labels[type_id] = comment.split()[0]
        elif section == "Atoms":
            style = atom_style or ""
            atom_id = int(columns[0])
            atom_ids.add(atom_id)
            if style == "full" and len(columns) >= 7:
                molecule_id, type_id, charge = int(columns[1]), int(columns[2]), float(columns[3])
                molecules.add(molecule_id)
            elif style == "charge" and len(columns) >= 6:
                type_id, charge = int(columns[1]), float(columns[2])
            elif style in {"molecular", "bond", "angle"} and len(columns) >= 6:
                molecule_id, type_id, charge = int(columns[1]), int(columns[2]), None
                molecules.add(molecule_id)
            elif style == "atomic" and len(columns) >= 5:
                type_id, charge = int(columns[1]), None
            elif len(columns) >= 7:
                molecule_id, type_id, charge = int(columns[1]), int(columns[2]), float(columns[3])
                molecules.add(molecule_id)
            elif len(columns) >= 5:
                type_id, charge = int(columns[1]), None
            else:
                continue
            type_counts[type_id] += 1
            if charge is not None:
                charges[type_id].append(charge)
        elif section == "Bonds" and len(columns) >= 4:
            bond_edges.append((int(columns[2]), int(columns[3])))

    bonded_component_count: int | None = None
    if atom_ids and bond_edges:
        parent = {atom_id: atom_id for atom_id in atom_ids}

        def find(atom_id: int) -> int:
            while parent[atom_id] != atom_id:
                parent[atom_id] = parent[parent[atom_id]]
                atom_id = parent[atom_id]
            return atom_id

        for left, right in bond_edges:
            if left not in parent or right not in parent:
                continue
            left_root, right_root = find(left), find(right)
            if left_root != right_root:
                parent[right_root] = left_root
        bonded_component_count = len({find(atom_id) for atom_id in atom_ids})

    declared_type_count = declared.get("atom_types", 0)
    type_ids = sorted(
        set(range(1, declared_type_count + 1))
        | set(masses)
        | set(pair_coeffs)
        | set(type_counts)
    )
    summaries = [
        AtomTypeSummary(
            type_id=type_id,
            label=labels.get(type_id, f"type_{type_id}"),
            mass=masses.get(type_id),
            pair_coefficients=pair_coeffs.get(type_id, []),
            atom_count=type_counts[type_id],
            charges=sorted(set(charges.get(type_id, []))),
        )
        for type_id in type_ids
    ]
    charge_values = [charge for values in charges.values() for charge in values]
    return LammpsDataSummary(
        path=str(source),
        title=lines[0].strip(),
        atom_style=atom_style,
        declared_counts=declared,
        molecule_count=(
            len(molecules)
            if len(molecules) > 1
            else bonded_component_count or (len(molecules) if molecules else None)
        ),
        total_charge=sum(charge_values) if charge_values else None,
        atom_types=summaries,
        section_styles=section_styles,
    )
