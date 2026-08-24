"""Lightweight, dependency-free inspection of LAMMPS data files."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
import math
import operator
from pathlib import Path
import re
from typing import Any, Sequence


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


Vector3 = tuple[float, float, float]
BoxBounds = tuple[tuple[float, float], tuple[float, float], tuple[float, float]]


def _cross(left: Vector3, right: Vector3) -> Vector3:
    return (
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    )


def _dot(left: Vector3, right: Vector3) -> float:
    return sum(a * b for a, b in zip(left, right))


def _norm(vector: Vector3) -> float:
    return math.sqrt(_dot(vector, vector))


def _replicate_factors(values: Sequence[int]) -> tuple[int, int, int]:
    try:
        raw = tuple(values)
    except TypeError as exc:
        raise ValueError("replicate must contain three positive integers") from exc
    if len(raw) != 3 or any(isinstance(value, bool) for value in raw):
        raise ValueError("replicate must contain three positive integers")
    try:
        factors = tuple(operator.index(value) for value in raw)
    except TypeError as exc:
        raise ValueError("replicate must contain three positive integers") from exc
    if any(value < 1 for value in factors):
        raise ValueError("replicate must contain three positive integers")
    return factors  # type: ignore[return-value]


@dataclass(frozen=True)
class LammpsBoxSummary:
    """One orthogonal or restricted-triclinic LAMMPS simulation box.

    ``vectors`` are the three edge vectors ``a``, ``b`` and ``c``. The
    perpendicular face heights are the minimum-image dimensions relevant to
    cutoff validation; unlike ``lx``, ``ly`` and ``lz``, they remain correct
    for tilted boxes.
    """

    origin: Vector3
    vectors: tuple[Vector3, Vector3, Vector3]
    bounds: BoxBounds
    tilt_factors: Vector3
    representation: str

    def __post_init__(self) -> None:
        values = (
            *self.origin,
            *(component for vector in self.vectors for component in vector),
            *(component for bound in self.bounds for component in bound),
            *self.tilt_factors,
        )
        if any(not math.isfinite(float(value)) for value in values):
            raise ValueError("LAMMPS box values must be finite")
        if self.representation not in {"orthogonal", "restricted_triclinic"}:
            raise ValueError(
                "LAMMPS box representation must be orthogonal or restricted_triclinic"
            )
        if self.volume <= 0.0:
            raise ValueError("LAMMPS box vectors must form a right-handed volume")

    @property
    def volume(self) -> float:
        a, b, c = self.vectors
        return _dot(a, _cross(b, c))

    def replicated_vectors(
        self, replicate: Sequence[int] = (1, 1, 1)
    ) -> tuple[Vector3, Vector3, Vector3]:
        factors = _replicate_factors(replicate)
        return tuple(
            tuple(float(factor) * component for component in vector)
            for vector, factor in zip(self.vectors, factors)
        )  # type: ignore[return-value]

    def periodic_face_heights(
        self, replicate: Sequence[int] = (1, 1, 1)
    ) -> Vector3:
        """Return heights normal to the ``bc``, ``ca`` and ``ab`` faces."""

        a, b, c = self.replicated_vectors(replicate)
        volume = _dot(a, _cross(b, c))
        face_areas = (
            _norm(_cross(b, c)),
            _norm(_cross(c, a)),
            _norm(_cross(a, b)),
        )
        if volume <= 0.0 or any(area <= 0.0 for area in face_areas):
            raise ValueError("LAMMPS box vectors do not form a valid periodic volume")
        return tuple(volume / area for area in face_areas)  # type: ignore[return-value]

    @property
    def face_heights(self) -> Vector3:
        return self.periodic_face_heights()

    def to_dict(self) -> dict[str, Any]:
        return {
            "representation": self.representation,
            "origin": list(self.origin),
            "bounds": [list(bound) for bound in self.bounds],
            "tilt_factors": list(self.tilt_factors),
            "vectors": [list(vector) for vector in self.vectors],
            "volume": self.volume,
            "periodic_face_heights": list(self.face_heights),
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
    box: LammpsBoxSummary | None = None

    @property
    def box_vectors(self) -> tuple[Vector3, Vector3, Vector3] | None:
        return self.box.vectors if self.box is not None else None

    def replicated_box_vectors(
        self, replicate: Sequence[int] = (1, 1, 1)
    ) -> tuple[Vector3, Vector3, Vector3]:
        if self.box is None:
            raise ValueError(f"LAMMPS data file contains no complete box: {self.path}")
        return self.box.replicated_vectors(replicate)

    def periodic_face_heights(
        self, replicate: Sequence[int] = (1, 1, 1)
    ) -> Vector3:
        if self.box is None:
            raise ValueError(f"LAMMPS data file contains no complete box: {self.path}")
        return self.box.periodic_face_heights(replicate)

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
            "box": self.box.to_dict() if self.box is not None else None,
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


def _box_number(source: Path, line: int, value: str, label: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise ValueError(
            f"invalid {label} in LAMMPS data file {source}:{line}: {value!r}"
        ) from exc
    if not math.isfinite(number):
        raise ValueError(
            f"non-finite {label} in LAMMPS data file {source}:{line}: {value!r}"
        )
    return number


def _inspect_box(source: Path, lines: Sequence[str]) -> LammpsBoxSummary | None:
    axis_labels = {
        ("xlo", "xhi"): "x",
        ("ylo", "yhi"): "y",
        ("zlo", "zhi"): "z",
    }
    bounds: dict[str, tuple[float, float]] = {}
    tilt_factors: Vector3 | None = None

    for line_number, line in enumerate(lines, start=1):
        header, _qualifier = _section_header(line)
        if header is not None:
            break
        columns, _comment = _numeric_payload(line)
        labels = (
            (columns[-2].lower(), columns[-1].lower())
            if len(columns) == 4
            else None
        )
        if labels in axis_labels:
            axis = axis_labels[labels]
            if axis in bounds:
                raise ValueError(
                    f"duplicate {axis} box bounds in LAMMPS data file "
                    f"{source}:{line_number}"
                )
            lower = _box_number(source, line_number, columns[0], f"{axis}lo")
            upper = _box_number(source, line_number, columns[1], f"{axis}hi")
            if upper <= lower:
                raise ValueError(
                    f"LAMMPS {axis}hi must exceed {axis}lo in {source}:{line_number}"
                )
            bounds[axis] = (lower, upper)
            continue
        if (
            len(columns) == 6
            and tuple(value.lower() for value in columns[-3:])
            == ("xy", "xz", "yz")
        ):
            if tilt_factors is not None:
                raise ValueError(
                    f"duplicate triclinic tilt factors in LAMMPS data file "
                    f"{source}:{line_number}"
                )
            tilt_factors = tuple(
                _box_number(source, line_number, value, label)
                for value, label in zip(columns[:3], ("xy", "xz", "yz"))
            )  # type: ignore[assignment]

    if not bounds and tilt_factors is None:
        return None
    if set(bounds) != {"x", "y", "z"}:
        missing = sorted({"x", "y", "z"} - set(bounds))
        raise ValueError(
            f"incomplete LAMMPS box in {source}; missing bounds for {missing}"
        )
    xy, xz, yz = tilt_factors or (0.0, 0.0, 0.0)
    x, y, z = bounds["x"], bounds["y"], bounds["z"]
    lx, ly, lz = x[1] - x[0], y[1] - y[0], z[1] - z[0]
    representation = (
        "restricted_triclinic" if tilt_factors is not None else "orthogonal"
    )
    return LammpsBoxSummary(
        origin=(x[0], y[0], z[0]),
        vectors=((lx, 0.0, 0.0), (xy, ly, 0.0), (xz, yz, lz)),
        bounds=(x, y, z),
        tilt_factors=(xy, xz, yz),
        representation=representation,
    )


def inspect_lammps_data(path: str | Path) -> LammpsDataSummary:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"LAMMPS data file not found: {source}")
    lines = source.read_text(encoding="utf-8", errors="replace").splitlines()
    if not lines:
        raise ValueError(f"LAMMPS data file is empty: {source}")

    box = _inspect_box(source, lines)

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
        box=box,
    )
