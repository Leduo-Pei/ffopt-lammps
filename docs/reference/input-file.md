# `ffopt.in` reference

`ffopt.in` is the only user-edited scientific control file. It is a
line-oriented format: commands are case-insensitive, values are separated by
spaces, `#` starts a comment, and each block ends with `end`. Paths are
resolved relative to the input file. Files are UTF-8; quote paths containing
spaces or `#`. `ffopt init` adds those quotes automatically.

Run `ffopt check ffopt.in` after every edit. Unknown commands, duplicate
settings, unsupported units, missing files, incompatible types, impossible
ranges, and invalid workflow dependencies are rejected before LAMMPS starts.

## Minimal structure

```text
ffopt 1
project my_crystal
workflow bo sample nn al audit finalize validate

parameters
    ...
end

property bulk
    ...
end
```

| Command | Meaning |
|---|---|
| `ffopt 1` | Input schema version. It is not an on/off switch. |
| `project NAME` | Portable result identifier; letters first, then letters, numbers, `.`, `_`, or `-`. |
| `workflow ...` | Ordered stages to execute. Common values are `bo` or `bo sample nn al audit finalize validate`. |

Results are written below `runs/<project>/`. The project name has no
molecule-specific meaning.

## Parameter block

```text
parameters
    range charge  delta  0.30
    charge_limit 1.0
    neutrality derive N1
    mixing epsilon geometric
    mixing sigma geometric
    fix epsilon sigma

    # id label epsilon(kcal/mol) sigma(A) charge(e)
    type 1 N1 0.2000 3.2963 -0.5000
    type 2 H1 0.0300 2.4200  0.5000
end
```

### `type`

```text
type ID LABEL EPSILON SIGMA CHARGE
```

There must be exactly one row for every atom type in the primary LAMMPS data
file. IDs are positive and unique. Labels are portable FFOpt names and must be
unique: start with a letter and use only ASCII letters, numbers, and
underscores. Epsilon is in kcal/mol, sigma in angstrom, and charge in
elementary charge. Epsilon and sigma must be positive.

One parameter value applies to every atom of that LAMMPS type. If a data file
uses multiple charges within one atom type, split the atom type before using
FFOpt.

### `range`

Global defaults:

```text
range epsilon factor LOW HIGH
range sigma   factor LOW HIGH
range charge  delta SIZE
range charge  absolute LOW HIGH
```

Per-type override:

```text
range type LABEL charge delta 0.10
range type LABEL sigma absolute 3.0 3.5
```

`factor 0.5 2.0` gives `[0.5 * initial, 2.0 * initial]`. `delta 0.3`
gives `[initial - 0.3, initial + 0.3]`. `absolute` gives the stated bounds.
Bounds are inclusive and reordered if written high-to-low. Epsilon and sigma
bounds must remain positive. The optional trailing unit on a delta is accepted
for readability but is not required.

Every free parameter needs either a global or per-type range. Fixed parameters
still need an initial value in each `type` row but do not need a range. A
per-type range overrides the global range for that label and parameter.

### `fix`

```text
fix epsilon sigma
fix type N1 charge
```

Without `fix`, epsilon, sigma, and charge are optimized wherever a range is
defined. Common modes are:

| Desired optimization | Line |
|---|---|
| All epsilon, sigma, and charge | no `fix` line |
| Epsilon and charge | `fix sigma` |
| Charge only | `fix epsilon sigma` |
| LJ only | `fix charge` |

A per-type `fix` overrides optimization only for that label.

### Charge neutrality

```text
neutrality derive N1
```

The named type charge is removed from the independent input vector and
recovered at every evaluation so the complete system is neutral:

```text
q_N1 = -sum(count_i * q_i, i != N1) / count_N1
```

The recovered value is still included in LAMMPS and in ANN physics features;
it is simply not an independent optimization coordinate. Use `neutrality none`
only for intentionally charged systems.

```text
charge_limit 1.0
```

`charge_limit` is an absolute safety gate: every resolved charge must satisfy
`|q| <= 1.0 e`. It is independent of the local `range charge` window.

### Mixing rules

```text
mixing epsilon geometric
mixing sigma geometric
```

The current backend always uses geometric epsilon mixing. Sigma may use
`geometric` or `arithmetic`, matching LAMMPS `pair_modify mix geometric` or
`pair_modify mix arithmetic`. For unlike types `i,j`:

```text
epsilon_ij = sqrt(epsilon_i * epsilon_j)
sigma_ij   = sqrt(sigma_i * sigma_j)       # geometric
sigma_ij   = (sigma_i + sigma_j) / 2       # arithmetic
```

Choose the rule used by the source force field. Mixing is applied by LAMMPS;
cross terms do not add independent optimization dimensions.

## Bulk property

```text
property bulk
    data data/bulk/crystal.data
    cells_in_data 8 2 2
    temperature 300 K
    pressure 1 atm
    equilibration 20000
    production 40000
    target a        4.2422 A      weight 1.0 tolerance 0.15
    target alpha   72.6300 degree weight 0.5 tolerance 1.50
    target density  1.3285 g/cm3  weight 1.0 tolerance 0.03
end
```

The protocol is fixed: fixed-box conjugate-gradient minimization, velocity
initialization, fully flexible triclinic NPT equilibration, and NPT production.
`equilibration` and `production` are timesteps. `cells_in_data NX NY NZ`
states how many crystallographic unit cells are already present in the data
file; it does not replicate the file. Reported `a`, `b`, and `c` are divided by
these values.

| Setting | Meaning | Default |
|---|---|---|
| `cells_in_data` | Unit-cell repeats already contained in the data file | `1 1 1` |
| `temperature` | NPT temperature | `300 K` |
| `pressure` | NPT pressure | `1 atm` |
| `timestep` | LAMMPS timestep | `1 fs` |
| `cutoff` | LJ and real-space Coulomb cutoff | `8 A` |
| `equilibration` | Discarded NPT timesteps | `20000` |
| `production` | Averaged NPT timesteps | `40000` |
| `seed` | Velocity-initialization seed | `101` |

There is no bulk `protocol` switch in schema 1: minimization followed by NPT
is the fixed, tested workflow. `production` must be at least 5000 timesteps,
the fixed averaging-output interval.

Supported targets and units are:

| Target | Unit |
|---|---|
| `a`, `b`, `c` | `A` |
| `alpha`, `beta`, `gamma` | `degree` |
| `density` | `g/cm3` |

## Sublimation property

```text
property sublimation
    bulk data/bulk/crystal.data
    single data/molecule/single.data
    temperature 298.15 K
    target 98.5 kJ/mol weight 0.3 tolerance 5.0
end
```

The current observable is the potential-energy estimate

```text
E_sub,estimate = E_single,min - <PE_bulk,NPT> / N_molecules
```

reported in kJ/mol. It uses the same bulk NPT calculation when a bulk property
is active, plus an isolated-molecule minimization. The experimental target is
a finite-temperature sublimation enthalpy, while this estimator does not add
ideal-gas translation, rotation, vibration, `pV`, or standard-state thermal
corrections. The validation manifest records this definition.

`temperature` records the experimental target temperature (default
`298.15 K`); the bulk simulation temperature is controlled by the bulk block
and defaults to `300 K`. `cutoff` optionally overrides only the isolated-
molecule minimization cutoff and otherwise inherits the bulk cutoff. The
number of atoms per molecule is read from the required single-molecule data
file and is not a user parameter.

## Adsorption property

```text
property adsorption
    data complex data/adsorption/complex.data
    data slab data/adsorption/slab.data
    data molecule data/adsorption/molecule.data
    protocol minimize
    metal Au
    target -3.5 kcal/mol weight 1.0 tolerance 0.5
end
```

All three data files must have compatible non-metal type IDs, labels, masses,
pair coefficients, and charges. The supported protocol is deterministic
minimization. The energy is

```text
E_ads = E_complex - E_slab - E_molecule
```

in kcal/mol. Omit `target` when no experimental reference exists; FFOpt then
computes adsorption only during final validation and does not train or
optimize against it.

The only optional adsorption settings are `metal LABEL` (default `Au`) and
`cutoff VALUE A` (default `7 A`). Because schema 1 adsorption is deterministic
minimization, temperature, timestep, random seed, equilibration, and production
settings are rejected instead of being accepted and ignored.

Schema 1 assumes that `metal LABEL` identifies one fixed, uncharged substrate
type whose LJ parameters remain in each data file. FFOpt updates the molecular
types shared by the complex and isolated-molecule files. A charged,
multicomponent, or independently optimized substrate requires a future property
contract and must not be represented by relabeling it as this fixed metal.

## Targets, weights, and tolerances

General form:

```text
target NAME VALUE UNIT weight W tolerance T
```

Sublimation and adsorption infer the target name, so `NAME` is omitted.
Target units are checked, not converted. `weight 0` disables that target in
the objective. Positive weights enter the dimensionless weighted relative
RMSE:

```text
r_i = abs(calculated_i - target_i) / abs(target_i)
J   = sqrt(sum(weight_i * r_i^2) / sum(weight_i))
```

`tolerance` is an absolute property-unit threshold used by final validation;
it does not change the optimization objective. If omitted, the compiler uses
3% of the absolute target for bulk properties and 10 kJ/mol for sublimation.
An optimization workflow needs at least one positive-weight, nonzero target;
the relative objective is not defined for a zero reference value.

## BO block

```text
bo
    method auto
    initial_points 48
    batch_size 48
    max_rounds 200
    random_seed 42
    stability_audit on
    stability_top_k 20
    stability_seeds 101 202 303
    early_stop enabled yes
    early_stop patience 30
    early_stop min_improvement 0.001
end
```

| Setting | Meaning | Default |
|---|---|---|
| `method` | `auto`, `gp`, `turbo`, or `saasbo` | `auto` |
| `initial_points` | Latin-hypercube points in the initial design | 48 |
| `batch_size` | Scientific candidates per BO round | 48 |
| `max_rounds` | Maximum BO rounds | 200 |
| `random_seed` | Deterministic design/model seed | 42 |
| `stability_audit` | Audit top BO candidates with independent seeds | on |
| `stability_top_k` | Distinct BO candidates sent to the stability audit | 20 |
| `stability_seeds` | Independent LAMMPS seeds used for each audited candidate | `101 202 303` |
| `early_stop patience` | Stop after this many non-improving rounds | 30 |
| `early_stop min_improvement` | Minimum objective decrease counted as improvement | 0.001 |

The explicit initial parameter vector from the `type` rows is evaluated once
as a warm-start centre in addition to the Latin-hypercube points. Therefore,
with `initial_points 48` the initial stage contains 49 LAMMPS evaluations.
`ffopt explain` reports the LHS count, warm-start count, initial total, and
complete BO evaluation budget separately.

The stability audit adds at most `stability_top_k * number_of_seeds` LAMMPS
evaluations after the search. `ffopt explain` reports this separately and also
prints the maximum complete BO-stage workload. Advanced optional controls are
`stability_max_objective_std`, `stability_max_property_rel_std`,
`stability_noise_penalty`, and `stability_failure_penalty`; their defaults are
`0.05`, `0.05`, `1.0`, and `10.0`, respectively. Keep the defaults unless the
noise model and failure policy have been justified for the material.

`batch_size` is independent of machine `workers`. One and two nodes therefore
perform the same scientific search and differ only in how many candidates run
concurrently.

The BO stability audit and the post-AL `audit` stage have different jobs. The
first supplies repeatable BO centers to focused sampling. The second
re-evaluates the best candidates after all learning rounds and determines the
parameter set eligible for final selection.

With `method auto`, FFOpt uses GP below 7 dimensions, TuRBO from 7 to 15
dimensions, and SAASBO from 16 dimensions upward under the default accuracy
policy. If the optional SAASBO runtime is unavailable, it reports a fallback
to TuRBO. An explicit `method saasbo` never changes method silently; install
`ffopt-lammps[saasbo]` or `ffopt-lammps[all-models]` first.

## Sampling block

```text
sample
    points 2000
    centers 24
    center_selection diverse
    radii 0.01 0.025 0.05
    global_fraction 0.10
    seeds 101 202 303
    design_seed 20260727
end
```

`points` is the number of distinct parameter vectors, not the total number of
LAMMPS jobs. The total is `points * number_of_seeds`. Centers are selected from
successful BO data; `diverse` avoids choosing only near-duplicates. Radii are
fractions of each full parameter bound, not charge units. `global_fraction`
reserves part of the design for the full configured bounds. Three seeds allow
the training table to include mean response and simulation variability.

## ANN block

```text
nn
    method ann
    ensemble 8
    hidden_layers 256 128 128 64
    epochs 1200
    batch_size 128
    learning_rate 0.0005
    validation_fraction 0.15
    test_fraction 0.10
end
```

`method ann` selects the MLP ensemble. `ensemble 8` means eight independently
initialized ANN members are trained on the same split; their mean is the
property prediction and their disagreement contributes model uncertainty.
The test set is held out from fitting and early stopping. Metrics are reported
per property; a single high aggregate score must not hide a poorly learned
property.

The backend also accepts `gp`, `random_forest`, `extra_trees`,
`pair_product_extra_trees`, `svr_rbf`, `polynomial_ridge`, and `xgboost` for
controlled method comparisons. XGBoost is intentionally optional; install it with
`python -m pip install "ffopt-lammps[xgboost]"` before selecting that method.
ANN remains the generated default and packaged acceptance path.

## Active-learning block

```text
al
    acquisition uncertainty
    rounds 2
    candidates 20
    candidate_pool 16384
    uncertainty_threshold 0.03
    sampling_domain core_envelope
    local_radii 0.01 0.025 0.05
end
```

Each round generates a candidate pool inside the learned stable domain,
ranks candidates using predicted objective and ensemble uncertainty, evaluates
the selected candidates with LAMMPS, and adds those labels to the training
data. `candidates` is the number of expensive LAMMPS points per round.

## Robust audit and finalization

```text
audit
    top_k 8
    seeds 101 202 303
end
```

`top_k` is the number of distinct, lowest-objective accumulated candidates
sent back to LAMMPS. Each is evaluated with every listed seed, so this example
runs 24 evaluations. The robust score is `mean objective + objective standard
deviation`; failed replicates remain visible in the audit tables.

`finalize` is a workflow stage, not a settings block. It selects the minimum
robust score from `audit`, resolves fixed and neutrality-derived parameters,
and writes a complete per-type parameter table plus an include-ready LAMMPS
file. Production workflows should keep `audit finalize` immediately before
`validate`.

## Final validation block

```text
validate
    require_tolerances yes
    objective_max 0.03
    max_error_percent 3.0
end
```

Final validation always saves final structures and bulk/adsorption trajectories
when those properties are present; no trajectory setting is required. The old
`trajectory final` spelling is read only so prerelease checkpoints can resume;
new inputs should omit it.
`require_tolerances yes` fails the stage if any fitted property exceeds its
absolute target tolerance. `objective_max` and `max_error_percent` add
independent acceptance gates. Omit a gate when it is not scientifically
justified for the material.

## Common workflow forms

```text
workflow validate                 # evaluate initial type values
workflow bo                       # Bayesian optimization only
workflow bo validate              # BO then physical validation
workflow bo sample nn al audit finalize validate # production workflow
```

Use the workflow line for persistent stage selection. Use `ffopt run ...
--until bo` only as a temporary operational pause.

A validate-only input does not need parameter `range` lines: no search space
is constructed, and every `type` value is passed directly to LAMMPS. Ranges
become mandatory as soon as `bo` is present in the workflow.
