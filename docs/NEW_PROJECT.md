# New molecular projects

## Generated layout

`ffopt init` creates:

```text
project.yaml
configs/system.yaml
configs/properties/bulk.yaml
configs/properties/sublimation.yaml  # when requested
data/bulk/
data/molecule/                       # when requested
lammps/
runs/                                # created during execution, git-ignored
```

The source data files are copied, never modified. `scaffold_manifest.json`
records the source paths, inferred dimensions, targets, assumptions, and
warnings.

Generated projects load `portable.yaml` BO/ANN/AL defaults from the installed
wheel. These use broad, molecule-independent objective cutoffs. They are not
the BTAH-tuned model settings; narrow them only after examining the new
system's stable sampling distribution.

## Required data contract

The current bundled molecular templates deliberately accept a narrow,
testable contract:

- `Atoms # full` data files.
- One charge value per atom type.
- `Pair Coeffs` columns begin with epsilon and sigma in LAMMPS `real` units.
- Class-I bonded terms: harmonic bonds, harmonic angles, harmonic dihedrals,
  and CVFF impropers when those sections are present.
- LJ mixing is `geometric` or `arithmetic`.

The command rejects incompatible files instead of guessing a potential style.
Other atom styles and bonded forms should be added as explicit input-template
families in a later package version.

## Parameter modes

- `full`: optimize epsilon, sigma, and charge.
- `fix_sigma`: optimize epsilon and charge.
- `charge_only`: fix LJ and optimize charge.
- `lj_only`: optimize epsilon and sigma while retaining data-file charges.

Default ranges are epsilon `0.5x to 2.0x`, sigma `0.85x to 1.15x`, and charge
`q0 +/- 0.30 e`. They are starting ranges, not chemical truth. Review every
entry in `configs/system.yaml` before BO.

For a neutral charged system, `--derive-charge auto` derives the most populous
type from total neutrality. Use `--derive-charge none` to disable this, or pass
an explicit type ID. The generated BTAH charge-only regression therefore has
14 charge types and 13 independent dimensions.

## Targets

Repeat `--target NAME=VALUE[,WEIGHT[,UNIT]]`. The generator currently wires
bulk cell lengths, cell angles, density, and the potential-energy sublimation
proxy. `esub_proxy` requires `--single-data`.

`--cells NX NY NZ` describes repeats already present in the bulk data file; it
controls conversion from supercell dimensions to reported unit-cell lengths.
An incorrect value creates systematically wrong targets even when LAMMPS runs
successfully.

## Preflight

Always run:

```bash
ffopt show --project project.yaml
ffopt doctor --project project.yaml --machine local
ffopt run --project project.yaml --machine local --dry-run
```

Only after these pass should the expensive BO stage begin.
