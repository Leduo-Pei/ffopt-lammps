# Changelog

All notable changes are recorded here. The format follows Keep a Changelog;
versions use Semantic Versioning while the public API remains in alpha.

## 0.3.0a1 - 2026-08-18

### Added

- Add packaged BTAH machine and scientific acceptance testing with explicit
  objective, relative-error, and absolute-tolerance gates.
- Add a Diataxis-style documentation site, complete Chinese user manual,
  machine-profile reference, output contract, contribution guide, citation
  metadata, and structured issue templates.
- Add strict target-unit validation and richer semantic checks for parameter,
  property, BO, sampling, ANN, AL, and validation settings.
- Add final validation acceptance status to JSON and CSV outputs.

### Changed

- Decouple scientific BO `batch_size` from machine `workers`; one- and
  two-node profiles now evaluate the same candidate budget.
- Reduce the public CLI to one restartable pipeline command plus creation,
  checking, machine, monitoring, and result commands.
- Treat CPU-only PyTorch as valid unless a CUDA device is explicitly required.
- Organize documentation by tutorial, how-to, reference, explanation, and
  development history.

### Removed

- Remove legacy per-stage SLURM submission scripts and obsolete standalone
  sensitivity/replicate utilities from the installed package.

## 0.2.0a3 - 2026-07-30

- Make the SLURM machine test use the same node-local MPI launcher as
  production LAMMPS evaluations.
- Include both scheduler stdout and stderr when a machine test fails.

## 0.2.0a2 - 2026-07-30

- Add molecular LAMMPS data contract and cross-file compatibility checks.
- Add local/SLURM environment probing and executable smoke tests.
- Generate crystal, adsorption, and combined projects from `ffopt init`.
- Support targetless validation-only project generation.
- Add pipeline artifact and SLURM log navigation commands.
- Expand beginner documentation and BTAH regression coverage.

## 0.2.0a1

- Introduce the one-file `ffopt.in` interface, restartable pipeline state, and
  reusable `machines.toml` execution profiles.
