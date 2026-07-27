# FFOpt-LAMMPS

FFOpt-LAMMPS is a restartable workflow for fitting molecular force-field
parameters to experimental observables. It combines LAMMPS evaluations,
Bayesian optimization, stability-aware sampling, surrogate modelling, active
learning, and final trajectory-producing validation.

The current alpha release uses the BTAH workflow as its regression example.
The migration is deliberately preserving the validated BO/NN/AL numerical
implementations while their project, machine, property, and scheduler
boundaries are made molecule-independent.

## Install from a clone

```bash
python -m pip install -e ".[full]"
ffopt show --project projects/project.yaml
ffopt doctor --project projects/project.yaml --machine local
```

CUDA-enabled PyTorch should be installed for the user's CUDA version before
installing the package extras. LAMMPS is an external dependency and is never
bundled with FFOpt.

## Configure a machine once

```bash
ffopt machine configure --name local --backend local \
  --lammps /path/to/lmp --mpi /path/to/mpiexec
ffopt machine show --name local
```

Machine profiles are stored outside the repository under
`~/.config/ffopt/machines/`. Projects contain scientific inputs only.

## Start a molecular project

`ffopt init` inspects atom types, masses, Pair Coeffs, and per-type charges,
then creates a portable project outside the FFOpt source checkout:

```bash
ffopt init my_crystal \
  --data-file crystal.data \
  --single-data molecule.data \
  --cells 2 2 2 \
  --mode full \
  --target a=10.1,1.0,A \
  --target b=12.3,1.0,A \
  --target density=1.25,1.0,g/cm3 \
  --target esub_proxy=80.0,0.3,kJ/mol

cd my_crystal
ffopt doctor --project project.yaml --machine local
ffopt run --project project.yaml --machine local --dry-run
```

The generated project owns its data, LAMMPS inputs, system ranges, and target
modules. Reusable BO/NN/AL and machine defaults use `builtin:` resources shipped
inside the wheel. See [New molecular projects](docs/NEW_PROJECT.md) for current
force-field assumptions and range rules.

## Run the complete workflow

The normal interface is one command. Outputs are deterministic under
`runs/<project>/pipelines/<run-id>/`, and `state.sqlite` records each stage,
attempt, artifact set, and SLURM job ID.

```bash
# Preview every stage without running LAMMPS.
ffopt run --project projects/btah_charge_only.yaml --machine local --dry-run

# Run locally through final trajectory-producing validation.
ffopt run --project projects/btah_charge_only.yaml --machine local

# Continue after interruption without repeating verified completed stages.
ffopt run --project projects/btah_charge_only.yaml --machine local --resume

# Submit to SLURM and keep advancing when each stage finishes.
ffopt run --project projects/btah_charge_only.yaml --machine cluster --watch

ffopt status --project projects/btah_charge_only.yaml --run-id default
```

Use `--until nn` to stop at a chosen stage, or `--from-stage al --resume` to
continue a deliberately staged campaign. On SLURM without `--watch`, one job is
submitted at a time; rerunning the same command with `--resume` checks its Job
ID and submits the next missing stage.

## Individual stages

```bash
ffopt show --project projects/btah_full.yaml
ffopt bo --project projects/btah_full.yaml --machine local
ffopt sample --project projects/btah_full.yaml --machine cluster
ffopt nn --project projects/btah_full.yaml --machine local
ffopt al --project projects/btah_full.yaml --machine local
ffopt audit --project projects/btah_full.yaml --source path/to/all_results.csv
ffopt finalize --project projects/btah_full.yaml --audit path/to/audit
ffopt validate --project projects/btah_full.yaml --machine local
```

`python -m ffopt ...` is equivalent to the installed `ffopt ...` command. See
[README_BTAH.md](README_BTAH.md) for the BTAH-specific history.

## Development status

This is an alpha research package. BTAH regression tests are authoritative.
Bulk, sublimation, adsorption, and surface calculations now implement the
property-evaluator interface. Third-party packages can add properties through
the `ffopt.property_evaluators` entry-point group; see
[Property plugins](docs/PROPERTY_PLUGINS.md).
