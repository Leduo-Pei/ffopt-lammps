# New molecular projects

## Generated layout

`ffopt init` creates:

```text
ffopt.in
data/bulk/
data/molecule/          # when requested
scaffold_manifest.json
README.md
```

Source data files are copied and never modified. No editable YAML recipes or
LAMMPS templates are generated. Built-in templates are resolved from the
installed FFOpt package, while every scientific value remains explicit in
`ffopt.in`.

## Required data contract

The bundled molecular templates currently require:

- `Atoms # full` data files.
- One charge value per atom type.
- `Pair Coeffs` beginning with epsilon and sigma in LAMMPS `real` units.
- Class-I harmonic bonds, angles, and dihedrals plus CVFF impropers when those
  sections are present.
- A single geometric or arithmetic LJ mixing rule.

The initializer rejects incompatible files instead of guessing.

## Initialization

```bash
ffopt init my_crystal \
  --data-file crystal.data \
  --single-data molecule.data \
  --cells 2 2 2 \
  --mode charge_only \
  --target density=1.25,1.0,g/cm3 \
  --target sublimation=80.0,0.3,kJ/mol
```

Modes only determine the generated `fix` command:

- `full`: no `fix`; optimize epsilon, sigma, and charge.
- `fix_sigma`: write `fix sigma`.
- `charge_only`: write `fix epsilon sigma`.
- `lj_only`: write `fix charge`.

Default ranges are epsilon `0.5x to 2.0x`, sigma `0.85x to 1.15x`, and charge
`q0 +/- 0.30 e`. These are draft bounds, not chemical truth. Review every type
line and range before running.

`--cells NX NY NZ` describes repeats already present in the bulk data file. It
does not replicate the box. Incorrect values systematically distort reported
unit-cell lengths.

## Preflight

```bash
ffopt check ffopt.in
ffopt explain ffopt.in
ffopt run ffopt.in --dry-run
```

Only after all three agree with the intended parameter space and properties
should the expensive workflow begin.
