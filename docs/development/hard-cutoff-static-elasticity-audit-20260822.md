# Hard-cutoff static-elasticity audit (2026-08-22)

## Executive conclusion

The large difference between the legacy Fe result and the `0.3.0a5` result is
not caused by a different LJ cutoff, a different parameter vector, or a
different deformation convention. Both calculations use the same bulk data,
the same legacy-final parameters, `lj/cut 12.5`, LAMMPS default mixing, and an
unshifted potential. The regression is the change of the canonical 0 K
estimator:

- the legacy workflow ranks candidates with three-mode **stress slopes**;
- `0.3.0a5` ranks candidates with three-mode **energy curvature**.

The energy-curvature formulas are algebraically correct for a smooth potential,
but their smoothness assumption is false for `lj/cut` with `pair_modify shift
no`. In this Fe cell, a BCC same-sublattice shell lies only 0.023662 A below the
12.5 A cutoff. The configured strains move thousands of pairs through the hard
cutoff, causing discrete potential-energy jumps. A quartic fit then interprets
those jumps as zero-strain curvature.

Re-fitting the existing `0.3.0a5` LAMMPS pressures, without re-running any
state, gives `B/Cprime/C44 = 233.2103/65.3948/188.9012 GPa`. This reproduces the
legacy result `233.2321/65.3992/189.2011 GPa` to at most 0.159%. The
`0.3.0a5` energy result `342.4838/37.9539/407.1450 GPa` is therefore a cutoff
artifact and must not be used to rank or reject candidates.

## Scope and audit conditions

This was a read-only audit of the completed server artifacts. No server file
was modified and no SLURM job was cancelled or altered.

The candidate was the legacy final vector:

| Parameter | Value |
| --- | ---: |
| epsilon | 5.946462614081354 kcal/mol |
| sigma_mean | 2.3079703513522443 A |
| sigma_difference | -0.6065744766779244 A |
| Fe_corner sigma | 2.0046831130132823 A |
| Fe_body sigma | 2.6112575896912062 A |

The two bulk data files are byte-identical:

```text
e9ce2d5599f18381dc72629ec21ba91ca1c8692c8a1249e1583de2adacd09582
```

Paths:

- legacy data:
  `/data/home/chengzhu02/LAMMPS/calc/BCC/Fe/data/Fe_a_2type_555.data`
- paired-audit data:
  `/data/home/chengzhu02/LAMMPS/calc/FFOPT_PARITY/Fe_cutoff_ab_20260822_a5/data/Fe_a_2type_555.data`

## Implementations and raw evidence

### Legacy stress-slope implementation

- implementation:
  `/data/home/chengzhu02/LAMMPS/calc/BCC/Fe/engine/validate_static_elastic.py`
  (`fit_static_stress`, lines 128-177)
- deformation template:
  `/data/home/chengzhu02/LAMMPS/calc/BCC/Fe/lammps/inputs/mechanical/in.bcc.static_elastic_strain`
- completed validation:
  `/data/home/chengzhu02/LAMMPS/calc/BCC/Fe/runs/fe_bcc/validation_20260821_001108/elastic_static_cubic`
- raw strain, energy, and pressure records:
  `energy_strain.csv`
- fitted slopes:
  `static_elastic_fits.csv`
- reported result and reference state:
  `static_elastic_summary.json`
- expanded runtime configuration:
  `config_used.yaml`

The expanded configuration records `lammps.cutoff: 12.5`. The copied force
field files record `pair_modify shift no`. LAMMPS reports the default geometric
mixing rule.

### `0.3.0a5` energy-curvature implementation

- installed fitter:
  `/data/home/chengzhu02/LAMMPS/ffopt-envs/0.3.0a5-15d4cdf/lib/python3.12/site-packages/engine/cubic_elasticity.py`
  (`fit_cubic_energy_strain`, lines 155-252)
- installed runner:
  `/data/home/chengzhu02/LAMMPS/ffopt-envs/0.3.0a5-15d4cdf/lib/python3.12/site-packages/engine/cubic_elastic_runner.py`
  (`run_static`, lines 712-765)
- installed deformation template:
  `/data/home/chengzhu02/LAMMPS/ffopt-envs/0.3.0a5-15d4cdf/share/ffopt/lammps/inputs/elasticity/in.cubic.static_strain`
- paired-audit candidate directory:
  `/data/home/chengzhu02/LAMMPS/calc/FFOPT_PARITY/Fe_cutoff_ab_20260822_a5/results/cutoff12p5/static/candidate_runs/beac532af78310391b1eeac4c2241d36e7453d7f73f1c23c7204742a15ae8154/static`
- raw energy table:
  `strain_records.csv`
- raw pressure records:
  `states/<state>/completed/result.json` and
  `states/<state>/completed/lammps.stdout`
- reported result:
  `elasticity_summary.json`

The generated force-field include records `pair_modify shift no tail no`.
LAMMPS reports `master list distance cutoff = 14.5`, i.e. the 12.5 A pair
cutoff plus the configured 2.0 A neighbor skin, and the default geometric
mixing rule.

### Reference-state parity

| Quantity | Legacy | `0.3.0a5` | Relative difference |
| --- | ---: | ---: | ---: |
| potential energy (kcal/mol) | -98400.1790691585 | -98400.1791231825 | 5.49e-8% |
| volume (A^3) | 23449.0876302794 | 23449.3616342081 | 0.0011685% |
| box length (A) | 28.6225715625 | 28.6226830475 | 0.0003895% |

The references are not bit-identical because their zero-pressure relaxation
implementations differ slightly, but they represent the same local state. The
near-exact agreement of the independently re-fitted stress moduli below shows
that this small reference difference is not the source of the reported
mechanical discrepancy.

The old and new strain templates use the same three deformation gradients and
the same six non-zero strains, `+/-0.002`, `+/-0.004`, and `+/-0.006`:

```text
hydro:        F = diag(1+d, 1+d, 1+d)
orthorhombic: F = diag(1+d, 1-d, 1/(1-d^2))
shear:        Fxy = d (engineering shear gamma_xy=d)
```

## The two estimators

LAMMPS `real` pressure is positive in compression and is converted with
`1 atm = 0.000101325 GPa`. If `m` is a fitted pressure slope versus `d`, the
legacy estimator uses

```text
mean(Pxx,Pyy,Pzz) = P0 - 3 B d       -> B      = -m_hydro * conversion / 3
Pxx - Pyy         =      -4 C' d     -> C'     = -m_ortho * conversion / 4
Pxy               =        -C44 d    -> C44    = -m_shear * conversion
```

The `0.3.0a5` estimator pair-averages the energy density,

```text
u_even(d) = [u(+d) + u(-d)]/2 - u(0),
u_even(d) = A d^2 + Q d^4,
```

then applies the correct smooth-potential normalizations

```text
A_hydro = 9 B / 2,  A_orthorhombic = 2 C',  A_shear = C44 / 2.
```

Thus, there is no missing factor of two in the `0.3.0a5` equations. The failure
is the use of a smooth energy expansion across discontinuous hard-cutoff events.

## Independent numerical recomputation

Both methods were recomputed directly from each completed set of 18 raw states.
No values in this table come from substituting one workflow's published
summary for another workflow's fit.

| Raw states | Recomputed estimator | B (GPa) | Cprime (GPa) | C44 (GPa) |
| --- | --- | ---: | ---: | ---: |
| legacy | stress slopes | 233.2321406 | 65.3992083 | 189.2010664 |
| `0.3.0a5` | stress slopes | 233.2103205 | 65.3947968 | 188.9011691 |
| legacy | energy curvature | 342.5075712 | 37.9573249 | 134.7233347 |
| `0.3.0a5` | energy curvature | 342.4837563 | 37.9538848 | 407.1449617 |

Relative change from the legacy stress result to the `0.3.0a5` stress re-fit:

| Modulus | Relative change |
| --- | ---: |
| B | -0.00936% |
| Cprime | -0.00675% |
| C44 | -0.15851% |

Relative discrepancy between the two estimators on the same `0.3.0a5` states:

| Modulus | Energy versus stress |
| --- | ---: |
| B | +46.86% |
| Cprime | -41.96% |
| C44 | +115.53% |

The `0.3.0a5` stress-fit quality is high:

| Mode | Slope (atm) | R2 |
| --- | ---: | ---: |
| hydro | -6904820.7406 | 0.99807925 |
| orthorhombic | -2581585.8580 | 0.99999805 |
| shear | -1864309.5891 | 0.99999827 |

The erroneous `0.3.0a5` energy fits nevertheless have R2 values of
0.98798205, 0.99888449, and 0.98079525. All pass the configured minimum of
0.98. R2 is therefore not a sufficient hard-cutoff contamination test.

The apparent quadratic coefficient `u_even(d)/d^2` is more revealing:

| Mode | abs(d)=0.002 | abs(d)=0.004 | abs(d)=0.006 | Unit |
| --- | ---: | ---: | ---: | --- |
| hydro | 2107.353 | 1311.352 | 1165.473 | GPa |
| orthorhombic | 130.921 | 130.992 | 209.198 | GPa |
| shear | 94.802 | 182.996 | 133.780 | GPa |

For comparison, the `0.3.0a5` stress slopes imply smooth-limit coefficients of
approximately `1049.446`, `130.790`, and `94.451 GPa`, respectively. The
energy sequence contains discrete offsets, not a well-resolved quartic
correction.

## Direct identification of the cutoff-crossing shell

The `0.3.0a5` relaxed 10x10x10 BCC box gives

```text
a = 28.62268304752711 / 10 = 2.862268304752711 A
rc/a = 4.3671657123
```

Same-sublattice BCC displacement vectors formed by all sign and permutation
variants of `(3,3,1)` have `n^2=19`. There are 24 directed vectors per atom,
and their unstrained radius is

```text
r19 = sqrt(19) * a = 12.4763382897 A,
rc - r19 = 0.0236617103 A.
```

This shell contains same-type pairs only. With 1000 atoms of each type, all 24
directions correspond to 12000 unique Fe_corner pairs and 12000 unique Fe_body
pairs.

For `pair_modify shift no`, the left-hand LJ energies at 12.5 A are

| Pair | U(rc-) (kcal/mol per pair) |
| --- | ---: |
| Fe_corner--Fe_corner | -0.0004046929220 |
| Fe_body--Fe_body | -0.0019766091583 |

When a pair leaves the cutoff, its energy jumps from this negative value to
zero. The predicted jumps are therefore

| Directed shell vectors leaving | Unique pairs per type | Predicted positive energy jump (kcal/mol) |
| ---: | ---: | ---: |
| 4 | 2000 | 4.7626042 |
| 8 | 4000 | 9.5252083 |
| 24 | 12000 | 28.5756250 |

The actual deformation geometry gives:

| Mode and strain | Shell event |
| --- | --- |
| hydro `+0.002` and larger | all 24 directions leave the cutoff |
| orthorhombic `+/-0.006` | 8 directions leave the cutoff |
| shear `+/-0.004` in `0.3.0a5` | 4 directions leave the cutoff |
| shear `+/-0.006` | 4 directions leave the cutoff |

The tiny reference-box difference makes the shear `abs(d)=0.004` state an
especially strong controlled demonstration:

| Reference | Maximum n^2=19 distance at abs(d)=0.004 | Directions inside |
| --- | ---: | ---: |
| legacy | 12.4999538166 A | 24 |
| `0.3.0a5` | 12.5000025040 A | 20 |

Accordingly, the `0.3.0a5` shear energy at `+0.004` is 4.7622423 kcal/mol
higher than the legacy energy. The hard-cutoff prediction is 4.7626042
kcal/mol; the remaining 0.0003619 kcal/mol is explained by the small smooth
geometry difference. This quantitative match identifies the cutoff crossing,
not merely correlation with it.

The same accounting predicts 9.5252 kcal/mol for the orthorhombic
`abs(d)=0.006` event. Extrapolating the uncontaminated `0.002/0.004` points
leaves an observed excess of about 9.488 kcal/mol, again consistent after the
smooth deformation contribution is included.

## Scientific interpretation

For an unshifted hard cutoff,

```text
U(r) = 4 epsilon [(sigma/r)^12 - (sigma/r)^6], r < rc
U(r) = 0,                                      r >= rc
```

and `U(rc-)` is non-zero. A finite-strain energy curve that changes the set of
interacting pairs is discontinuous. Pair-averaging removes odd smooth terms;
it cannot remove an even cutoff jump. Adding a quartic basis also cannot repair
the dataset: with only three magnitudes, it turns the discrete offsets into a
misleading extrapolated quadratic coefficient.

The endpoint virial pressures remain smooth enough here because the force at
12.5 A is small. Their central slopes agree across strain magnitudes and
between the two reference cells. Those slopes measure the local tangent
response of the actual LAMMPS force field used by the rest of the workflow.

This does not prove that energy curvature is universally unusable. It proves
that it cannot be the canonical estimator for a general workflow that permits
finite-range, unshifted pair potentials unless cutoff-clearance and
energy-work consistency are demonstrated first.

## Required `0.3.0a6` disposition

1. Make three-mode static **stress slopes** the canonical 0 K values used for
   ranking, Born stability, mechanical error, and promotion. Keep the existing
   hydro/orthorhombic/shear deformations and LAMMPS pressure convention.
2. Preserve energy curvature only as a diagnostic. It must not override or
   average with the stress result when hard-cutoff contamination is detected.
3. Persist every raw pressure component already emitted by
   `FFOPT_STATIC_STRAIN`, plus per-mode slope, intercept, R2, RMSE, and the
   central-difference modulus at each strain magnitude.
4. Add a fail-closed consistency audit comparing stress-derived moduli with
   energy-derived moduli and/or integrated stress work. A high energy-fit R2
   must not be treated as evidence of consistency.
5. Add an adaptive-strain retry when stress central differences are not stable.
   Shrinking strains can help avoid a nearby shell crossing, but it is a retry,
   not a reason to make energy curvature canonical again.
6. Add a regression fixture from these exact legacy-final raw records. It must
   reproduce approximately `233.21/65.395/188.90 GPa` from the `0.3.0a5`
   pressures and must flag the energy result as inconsistent.
7. Version the static scientific protocol and invalidate all current
   `0.3.0a5` static/finalist/final-validation artifacts. Structural BO work is
   not made physically invalid by this estimator bug because the Fe BO stage
   evaluates structure and surface only; reuse, if supported, must be an
   explicit provenance-preserving import rather than silent checkpoint reuse.
8. Do not change the user's force-field semantics to hide this issue. Default
   LAMMPS mixing and `shift no` remain valid project choices; the validator must
   correctly measure the chosen model.

Until this disposition is implemented, no `0.3.0a5` energy-curvature static
ranking should be presented as the scientific successor to the legacy Fe
result.
