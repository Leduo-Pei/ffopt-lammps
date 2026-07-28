"""CI smoke check for non-Python resources installed through the wheel."""

from pathlib import Path
import sys
import zipfile


wheel = Path(sys.argv[1])
required_suffixes = (
    "share/ffopt/lammps/inputs/bulk/in.bulk.mol",
    "share/ffopt/lammps/inputs/molecule/in.sublimation.bulk",
    "share/ffopt/lammps/inputs/molecule/in.sublimation.single",
    "share/ffopt/lammps/inputs/adsorption/in.complex",
    "share/ffopt/lammps/inputs/adsorption/in.slab",
    "share/ffopt/lammps/inputs/adsorption/in.mol",
)
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
missing = [
    suffix for suffix in required_suffixes
    if not any(name.endswith(suffix) for name in names)
]
if missing:
    raise SystemExit(f"Wheel is missing FFOpt resources: {missing}")
print(f"Wheel resources verified: {wheel}")
