# Modular Configuration Guide

The workflow now supports composable YAML configs. A recipe under
`configs/recipes/` is a short list of modules to load. Included files are
merged in order: later files override earlier files.

## Where To Edit

- `configs/systems/`: chemistry, data identity, atom types, LJ/charge ranges.
- `configs/properties/`: property targets and the LAMMPS inputs/data needed for each property.
- `configs/machines/`: local or cluster executables, MPI layout, timeouts, SLURM resources.
- `configs/methods/bo/`: BO algorithm and checkpoint/stability settings.
- `configs/methods/nn/`: surrogate model choice and hyperparameters.
- `configs/methods/al/`: active-learning strategy or disable switch.
- `configs/recipes/`: task-level combinations. Usually edit these least.

To switch methods, replace one include in a recipe. For example:

```yaml
include:
  - ../methods/bo/saasbo.yaml
  - ../methods/nn/random_forest.yaml
```

BO is user-selected by default: whichever BO module is included sets
`optimization.method`. Only `configs/methods/bo/auto.yaml` asks the code to pick
the method automatically from dimensionality.

## Common Runs

Local dry-run:

```powershell
.\local\run_dry.ps1
```

Local BO only:

```powershell
.\local\run_bo.ps1
```

Resume BO:

```powershell
& "D:\anaconda3\envs\ffopt-gpu\python.exe" run.py --config configs/recipes/local_full.yaml --resume
```

Run NN after BO:

```powershell
.\local\run_nn.ps1
```

Run AL after NN:

```powershell
.\local\run_al.ps1
```

Cluster BO:

```bash
python slurm/submit.py bo --config configs/recipes/cluster_full.yaml
```

Cluster BO resume:

```bash
python slurm/submit.py bo --config configs/recipes/cluster_full.yaml --resume
```

## Interface-Only Objective

Use `configs/recipes/local_interface_only.yaml` or
`configs/recipes/cluster_interface_only.yaml`. These recipes do not include
`bulk_targets.yaml`, so the objective only scores adsorption energy. The current
runner still performs bulk NPT as a runtime prerequisite; a later property-graph
runner can remove that prerequisite completely.

Quick local wiring test for the interface-only objective:

```powershell
& "D:\anaconda3\envs\ffopt-gpu\python.exe" run.py --config configs/recipes/local_interface_smoke.yaml --dry-run --no-mpi
```
