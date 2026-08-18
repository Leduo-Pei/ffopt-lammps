# New molecular projects

## Generated layout

`ffopt init` creates:

```text
ffopt.in
data/bulk/
data/molecule/          # when requested
data/adsorption/        # complex, slab, and molecule when requested
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

The initializer rejects incompatible files instead of guessing. Numeric type
IDs must match between bulk and isolated-single files. Adsorption files may
reorder type IDs; their molecular types are matched by the unique labels after
`#` in `Masses` or `Pair Coeffs`.

Check files before creating a project:

```bash
ffopt data check --bulk crystal.data --single molecule.data
```

## Initialization

```bash
ffopt init my_crystal \
  --bulk-data crystal.data \
  --single-data molecule.data \
  --cells 2 2 2 \
  --mode charge_only \
  --target density=1.25,1.0,g/cm3,0.04 \
  --target sublimation=80.0,0.3,kJ/mol,8.0
```

Combined crystal and adsorption fitting or validation:

```bash
ffopt init my_combined \
  --bulk-data crystal.data --single-data molecule.data \
  --complex-data complex.data --slab-data slab.data \
  --molecule-data adsorbate.data \
  --mode full \
  --target density=1.25,1.0,g/cm3 \
  --target sublimation=80.0,0.3,kJ/mol
```

No adsorption target was supplied here, so adsorption is evaluated only in
the final validation. An adsorption-only validation project can omit every
target; `ffopt init` then writes `workflow validate` automatically.

Modes only determine the generated `fix` command:

- `full`: no `fix`; optimize epsilon, sigma, and charge.
- `fix_sigma`: write `fix sigma`.
- `charge_only`: write `fix epsilon sigma`.
- `lj_only`: write `fix charge`.

Default ranges are epsilon `0.5x to 2.0x`, sigma `0.85x to 1.15x`, and charge
`q0 +/- 0.30 e`. These are draft bounds, not chemical truth. Review every type
line and range before running. `ffopt init` also writes an independent absolute
charge safety gate of `2.0 e`; reduce it when the chemistry justifies a tighter
limit, for example `1.0 e` for many neutral organic force fields.

The optional fourth target field is the absolute final-validation tolerance:
`NAME=VALUE,WEIGHT,UNIT,TOLERANCE`. When it is omitted, the initializer writes
the effective default explicitly so it cannot remain a hidden assumption.

`--cells NX NY NZ` describes repeats already present in the bulk data file. It
does not replicate the box. Incorrect values systematically distort reported
unit-cell lengths.

## Preflight

```bash
ffopt check ffopt.in
ffopt explain ffopt.in
ffopt machine test --name PROFILE
ffopt doctor ffopt.in --machine PROFILE
ffopt run ffopt.in --machine PROFILE --dry-run
```

Only after all checks agree with the intended parameter space and properties
should the expensive workflow begin.

## Results and continuation

The calculation is restartable by run ID. Repeating the same command resumes a
compatible run; a changed scientific input requires `--new` or a different
run ID.

```bash
ffopt run ffopt.in --machine PROFILE --run-id production --watch
ffopt status ffopt.in --machine PROFILE --run-id production
ffopt results ffopt.in --run-id production
ffopt logs ffopt.in --run-id production --stage sample --lines 100
```
