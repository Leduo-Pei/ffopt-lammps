# First-run setup and acceptance

This tutorial verifies an FFOpt installation before you use your own force
field. The environment and machine preflight take only a few minutes; the
complete BTAH scientific acceptance performs real NPT calculations and can
take hours. It does not require you to edit a configuration file.

## 1. Create an isolated environment

Python 3.11 is the recommended production version.

```bash
conda create -n ffopt python=3.11 -y
conda activate ffopt
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate ffopt
conda install -c conda-forge "lammps=*=*openmpi*" openmpi -y
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install \
  "ffopt-lammps[full] @ https://github.com/Leduo-Pei/ffopt-lammps/releases/download/v0.3.0a4/ffopt_lammps-0.3.0a4-py3-none-any.whl"
```

The command above deliberately installs CPU PyTorch. On a GPU workstation,
install the PyTorch build matching that machine's CUDA driver from the official
PyTorch selector before installing FFOpt; `[full]` will reuse it.

Confirm that all three executables come from the intended environment:

```bash
which python
which ffopt
which lmp
which mpirun
ffopt --version
python -c "import site; print(site.ENABLE_USER_SITE)"
lmp -help | head
```

The user-site check should print `False`, preventing packages in `~/.local`
from silently leaking into the new environment.

Manual GitHub Release downloads can be verified against the accompanying
`SHA256SUMS.txt` before installation.

`torch.cuda.is_available() == False` is expected in a CPU environment. It is
not an FFOpt or LAMMPS failure.

The complete target syntax used by `ffopt init` is
`NAME=VALUE[,WEIGHT[,UNIT[,TOLERANCE]]]`. Generated tolerance values are
explicit in `ffopt.in` and must be reviewed against experimental uncertainty.
`weight` controls a property's relative contribution to the objective.
`tolerance` is a same-unit absolute error threshold used only by final
validation. For example, `a=4.2422,1.0,A,0.15` requires
`|a_calc - 4.2422| <= 0.15 A`. If omitted, `ffopt init` writes
`max(3% of |target|, 1e-6)` explicitly.

## 2. Configure one machine profile

Inspect the host first:

```bash
ffopt machine probe
ffopt machine probe --partition YOUR_PARTITION
```

`YOUR_PARTITION` is the site's SLURM partition name, not a compute-node name.
Use `sinfo` to discover it, or omit the filter to list every visible partition.

For a local workstation:

If both `lmp` and its dependencies are already on `PATH`, first-time users can
skip configuration and use `--machine local`. The built-in profile runs one
serial evaluation at a time. `local` means direct execution on the host where
the command is entered; it does not submit through a scheduler. Do not use it
for production on a cluster login node. Configure a named profile to use
explicit paths or more CPU cores:

```bash
ffopt machine configure \
  --name local-workstation \
  --backend local \
  --lammps "$(which lmp)" \
  --mpi "$(which mpirun)" \
  --mpi-flavor openmpi \
  --workers 4 \
  --mpi-ranks 4 \
  --omp-threads 1 \
  --force
```

For one 48-core SLURM node:

```bash
ffopt machine configure \
  --name cluster-1node \
  --backend slurm \
  --lammps "$(which lmp)" \
  --mpi "$(which mpirun)" \
  --mpi-flavor openmpi \
  --partition YOUR_PARTITION \
  --nodes 1 \
  --total-cores 48 \
  --workers 12 \
  --mpi-ranks 4 \
  --omp-threads 1 \
  --memory-per-node 64G \
  --walltime 06:00:00 \
  --timeout 7200 \
  --force
```

`workers` is the number of independent LAMMPS evaluations that can run at
once. `mpi-ranks` and `omp-threads` belong to each evaluation. Therefore the
minimum CPU allocation is `workers * mpi-ranks * omp-threads`.
Use a realistic short wall time for acceptance so SLURM can backfill it;
production profiles can use a longer limit.
The per-evaluation timeout must remain shorter than the stage wall time so one
pathological parameter point cannot consume the whole allocation.

## 3. Test the installation

The machine test launches a tiny LAMMPS calculation:

```bash
ffopt machine test --name cluster-1node
```

This smoke test should finish in minutes. For a multi-node profile it exercises
the configured nodes, workers, and MPI ranks concurrently and verifies that all
requested nodes participate. It checks executable, MPI, scheduler, and shared-
filesystem queue wiring, but it does not test force-field accuracy.

The scientific self-test runs the packaged BTAH workflow and checks the final
properties, objective, and tolerances:

```bash
ffopt self-test --machine cluster-1node --watch
```

This is a real end-to-end calculation. Its runtime depends on available nodes
and profile concurrency; monitor it with `squeue`, `ffopt status`, and `ffopt
logs` rather than expecting the command to finish in five minutes.

The self-test is deliberately a warm-start software benchmark. Passing it
proves that the parser, scheduler, LAMMPS, BO, sampling, ANN, AL, robust
audit/finalization, validation-only adsorption, restart state, and final
validation work together. It does not independently validate a new material.

## 4. Start your own molecular project

First verify the packaged BTAH data without locating any repository files:

```bash
ffopt inspect builtin:data/bulk/BTAH_822_bulk.data
ffopt data check \
  --bulk builtin:data/bulk/BTAH_822_bulk.data \
  --single builtin:data/molecule/BTAH_822_single.data \
  --strict
```

To create a visible, editable copy of the complete BTAH example instead:

```bash
ffopt self-test --prepare-only --workdir ./ffopt-btah-example
cd ./ffopt-btah-example
```

Ordinary relative paths passed to `inspect`, `data check`, or `init` are
resolved from the shell's current directory. Relative paths written inside
`ffopt.in` are resolved from the directory containing that input file.

For your own data:

```bash
ffopt data check --bulk crystal.data --single molecule.data

ffopt init my_crystal \
  --bulk-data crystal.data \
  --single-data molecule.data \
  --cells 2 2 2 \
  --mode charge_only \
  --target a=10.1,1.0,A \
  --target density=1.25,1.0,g/cm3 \
  --target sublimation=80.0,0.3,kJ/mol

cd my_crystal
ffopt check ffopt.in
ffopt explain ffopt.in
ffopt doctor ffopt.in --machine cluster-1node
ffopt run ffopt.in --machine cluster-1node --dry-run
ffopt run ffopt.in --machine cluster-1node --watch
```

Review every initial parameter, bound, target, and data path in `ffopt.in`
before the production command. FFOpt can validate syntax and file contracts;
it cannot decide whether a guessed force-field range is chemically sensible.

## 5. Resume and inspect

The default run is restartable. Repeat the same command after a wall-time,
logout, or node failure:

```bash
ffopt run ffopt.in --machine cluster-1node --watch
ffopt status ffopt.in --machine cluster-1node
ffopt logs ffopt.in --stage bo --lines 100
ffopt results ffopt.in
```

Use `--new` only when you intentionally want an independent campaign.
