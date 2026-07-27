"""CI smoke check for non-Python resources installed through the wheel."""

from pathlib import Path
import sys
import zipfile


wheel = Path(sys.argv[1])
required_suffixes = (
    "share/ffopt/configs/machines/local.yaml",
    "share/ffopt/configs/machines/cluster.yaml",
    "share/ffopt/configs/methods/bo/auto.yaml",
    "share/ffopt/configs/methods/bo/portable.yaml",
    "share/ffopt/configs/methods/nn/mlp_ensemble.yaml",
    "share/ffopt/configs/methods/nn/portable.yaml",
    "share/ffopt/configs/methods/al/uncertainty_sampling.yaml",
    "share/ffopt/configs/methods/al/portable.yaml",
    "share/ffopt/lammps/inputs/bulk/in.bulk.mol",
    "share/ffopt/lammps/inputs/molecule/in.sublimation.single",
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
