# Five-minute first run

This tutorial verifies an FFOpt installation before you use your own force
field. It does not require you to edit a configuration file.

## 1. Create an isolated environment

Python 3.11 is the recommended production version.

```bash
conda create -n ffopt python=3.11 -y
conda activate ffopt
conda install -c conda-forge lammps openmpi -y
python -m pip install \
  "ffopt-lammps[full] @ git+https://github.com/Leduo-Pei/ffopt-lammps.git@v0.3.0a1"
```

Confirm that all three executables come from the intended environment:

```bash
which python
which ffopt
which lmp
lmp -help | head
```

`torch.cuda.is_available() == False` is expected in a CPU environment. It is
not an FFOpt or LAMMPS failure.

## 2. Configure one machine profile

Inspect the host first:

```bash
ffopt machine probe
```

For a local workstation:

```bash
ffopt machine configure \
  --name local-workstation \
  --backend local \
  --lammps "$(which lmp)" \
  --mpi "$(which mpirun)" \
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
  --partition CPU \
  --nodes 1 \
  --total-cores 48 \
  --workers 12 \
  --mpi-ranks 4 \
  --omp-threads 1 \
  --memory-per-node 64G \
  --walltime 14-00:00:00 \
  --timeout 216000 \
  --force
```

`workers` is the number of independent LAMMPS evaluations that can run at
once. `mpi-ranks` and `omp-threads` belong to each evaluation. Therefore the
minimum CPU allocation is `workers * mpi-ranks * omp-threads`.

## 3. Test the installation

The machine test launches a tiny LAMMPS calculation:

```bash
ffopt machine test --name cluster-1node
```

The scientific self-test runs the packaged BTAH workflow and checks the final
properties, objective, and tolerances:

```bash
ffopt self-test --machine cluster-1node --watch
```

The self-test is deliberately a warm-start software benchmark. Passing it
proves that the parser, scheduler, LAMMPS, BO, sampling, ANN, AL, restart
state, and final validation work together. It does not independently validate
a new material.

## 4. Start your own molecular project

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
