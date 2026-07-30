# FFOpt-LAMMPS

FFOpt-LAMMPS fits molecular force-field parameters to experimental properties
with LAMMPS, Bayesian optimization, stability-aware sampling, ANN surrogates,
active learning, and final trajectory-producing validation.

The public interface is one command file. A calculation project contains only
the input and its LAMMPS data files:

```text
my_project/
|-- ffopt.in
|-- data/
`-- runs/                 # generated automatically
```

## Install

Install the current tagged alpha directly from GitHub:

```bash
python -m pip install \
  "ffopt-lammps[full] @ git+https://github.com/Leduo-Pei/ffopt-lammps.git@v0.2.0a2"
```

For development, clone the repository and install it in editable mode:

```bash
git clone https://github.com/Leduo-Pei/ffopt-lammps.git
cd ffopt-lammps
python -m pip install -e ".[full]"
```

LAMMPS is an external dependency. Install a CUDA build of PyTorch separately
when GPU ANN/AL training is required.

Once installed, `ffopt` can be called from any project directory. The program
source and the user's calculation project do not need to be the same folder.

## First molecular release

The current alpha targets molecular force fields:

- isolated molecules evaluated as part of sublimation or adsorption;
- molecular crystals fitted to bulk and sublimation properties;
- molecular adsorption fitted jointly with, or validated after, crystal fitting.

BTAH is the regression system and exercises a difficult triclinic molecular
crystal plus adsorption. Elemental solids, alloys, multicomponent crystals,
polymorph searches, reactive potentials, and arbitrary LAMMPS styles are not
yet claimed as supported. They can be added through the evaluator registry in
later releases without changing the public workflow.

## Configure the machine once

Machine settings do not belong in the scientific input:

```bash
ffopt machine probe
ffopt machine configure --auto --name local --backend local \
  --lammps /path/to/lmp --force
ffopt machine show --name local
ffopt machine test --name local
```

Profiles are stored in `~/.config/ffopt/machines.toml`. The same `ffopt.in`
can therefore run locally or with a named SLURM profile.

For SLURM, `probe` reads `sinfo` and `--auto` writes a conservative starting
profile. It cannot know a site's QOS, account, node-sharing, or memory policy;
review the generated profile before production.

## Check the data first

FFOpt accepts ordinary text LAMMPS data files; they do not need to come from a
particular builder. MSI2LMP output is supported when it satisfies the molecular
data contract. Type labels must appear after `#` in `Masses` or `Pair Coeffs`,
because labels provide stable cross-file mapping even when adsorption files use
different numeric type IDs.

```bash
ffopt data check \
  --bulk crystal.data --single molecule.data \
  --complex complex.data --slab slab.data --molecule adsorbate.data
```

Structural incompatibilities are errors. Initial epsilon/sigma/charge
differences between compatible files are warnings because `ffopt.in` overrides
them during evaluation. Add `--strict` in CI when warnings should also fail.

## Create a project

```bash
ffopt init my_crystal \
  --bulk-data crystal.data \
  --single-data molecule.data \
  --cells 2 2 2 \
  --mode full \
  --target a=10.1,1.0,A \
  --target density=1.25,1.0,g/cm3 \
  --target sublimation=80.0,0.3,kJ/mol
```

For a combined crystal and adsorption project, add `--complex-data`,
`--slab-data`, and `--molecule-data`. `--project-type auto` is the default and
infers the template from the files. Omitting all targets creates a
validation-only input instead of inventing experimental values.

`ffopt init` writes every type, epsilon, sigma, charge, range, target, and
workflow stage explicitly into `ffopt.in`. It does not create a tree of YAML
recipes. The inferred values are a draft and must be reviewed by the user.

```bash
cd my_crystal
ffopt check ffopt.in
ffopt explain ffopt.in
ffopt run ffopt.in --dry-run
ffopt run ffopt.in
ffopt results ffopt.in
ffopt logs ffopt.in --stage bo --lines 100
```

Running the same command again automatically resumes the latest compatible
pipeline. If the scientific input changed, FFOpt refuses to mix checkpoints;
start an independent run with `ffopt run ffopt.in --new`.

For a direct calculation at the user-entered type values, set
`workflow validate`. No BO result is required; all requested properties are
reported and trajectories are saved under the validation run directory.
The equivalent one-off command is `ffopt validate --project ffopt.in --initial`.

## Parameters

```text
parameters
    range epsilon factor 0.50 2.00
    range sigma   factor 0.85 1.15
    range charge  delta  0.30
    charge_limit 1.0
    neutrality derive bhN1
    mixing epsilon geometric
    mixing sigma geometric

    # Add this one line for charge-only optimization.
    fix epsilon sigma

    # id label epsilon sigma charge
    type 1 bhN1 0.2000 3.2963 -0.609
    type 2 bhHn 0.0474 0.3981  0.430
end
```

No `fix` command means all ranged epsilon, sigma, and charge values are free.
Per-type ranges and fixes are also supported.

LAMMPS always mixes epsilon geometrically in this backend. Sigma may be
geometric or arithmetic, selected independently as shown above.

## Properties and workflow

A property with at least one `target` participates in BO, sampling, ANN, AL,
and final validation. A property without a target is skipped during fitting and
computed only by the `validate` stage. Final validation saves trajectories by
default; optimization batches do not.

```text
workflow bo sample nn al validate

property bulk
    data data/crystal.data
    cells_in_data 2 2 2
    target density 1.25 g/cm3 weight 1.0
end

property adsorption
    data complex data/complex.data
    data slab data/slab.data
    data molecule data/molecule.data
    protocol minimize
end
```

The bulk evaluator always follows the validated standard sequence: 0 K
minimization, then finite-temperature NPT equilibration and production. The
current sublimation estimator uses the bulk NPT mean potential energy and a
minimized isolated-molecule potential energy.

See [the BTAH charge-only example](examples/btah/charge_only.in),
[the full 41-dimensional example](examples/btah/full.in), and
[the input reference](docs/INPUT_FILE.md). For a clean installation on a new
CPU SLURM cluster, follow the [beginner server guide](docs/NEW_SERVER_GUIDE.md)
and run the [small full-pipeline BTAH test](examples/btah/cluster_smoke.in)
before submitting a production fit.

## Development status

This is an alpha research package. BTAH remains the numerical regression
system. Bulk, sublimation, adsorption, and surface evaluators use a plugin
registry; see [property plugins](docs/PROPERTY_PLUGINS.md).
