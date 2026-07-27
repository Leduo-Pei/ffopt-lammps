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

## Current workflow commands

```bash
ffopt show --project projects/btah_full.yaml
ffopt bo --project projects/btah_full.yaml --machine local
ffopt sample --project projects/btah_full.yaml --machine cluster
ffopt nn --project projects/btah_full.yaml --machine local
ffopt al --project projects/btah_full.yaml --machine local
ffopt validate --project projects/btah_full.yaml --machine local
```

The legacy `python ffopt.py ...` entry point remains available during the
migration. See [README_BTAH.md](README_BTAH.md) for the BTAH-specific history.

## Development status

This is an alpha research package. BTAH regression tests are authoritative.
The next package boundary replaces the monolithic LAMMPS runner with property
plugins and a persistent execution graph.

