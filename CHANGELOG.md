# Changelog

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
