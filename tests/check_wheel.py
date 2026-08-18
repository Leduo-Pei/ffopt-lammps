"""CI smoke check for non-Python resources installed through the wheel."""

from pathlib import Path
import sys
import zipfile


wheel = Path(sys.argv[1])
required_suffixes = (
    "engine/parameter_space.py",
    "workflow/machine_test_runner.py",
    "share/ffopt/lammps/inputs/bulk/in.bulk.mol",
    "share/ffopt/lammps/inputs/molecule/in.sublimation.single",
    "share/ffopt/lammps/inputs/adsorption/in.complex",
    "share/ffopt/lammps/inputs/adsorption/in.slab",
    "share/ffopt/lammps/inputs/adsorption/in.mol",
    "share/ffopt/examples/btah/acceptance.in",
    "share/ffopt/data/bulk/BTAH_822_bulk.data",
    "share/ffopt/data/molecule/BTAH_822_single.data",
    "share/ffopt/data/adsorption/ad_complex.data",
    "share/ffopt/data/adsorption/ad_slab.data",
    "share/ffopt/data/adsorption/ad_mol.data",
)
forbidden_members = {
    "slurm/__init__.py",
    "slurm/submit.py",
    "slurm/submit_local_sampling.py",
    "utils/replicate_learnability.py",
    "utils/sensitivity.py",
}
with zipfile.ZipFile(wheel) as archive:
    names = archive.namelist()
missing = [
    suffix for suffix in required_suffixes
    if not any(name.endswith(suffix) for name in names)
]
if missing:
    raise SystemExit(f"Wheel is missing FFOpt resources: {missing}")
unexpected = sorted(forbidden_members.intersection(names))
if unexpected:
    raise SystemExit(f"Wheel contains removed legacy modules: {unexpected}")
print(f"Wheel resources verified: {wheel}")
