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

The legacy `python ffopt.py ...` entry point remains available during the
migration. See [README_BTAH.md](README_BTAH.md) for the BTAH-specific history.

## Development status

This is an alpha research package. BTAH regression tests are authoritative.
Bulk, sublimation, adsorption, and surface calculations now implement the
property-evaluator interface; the next boundary is a public plugin registry for
third-party property modules.
