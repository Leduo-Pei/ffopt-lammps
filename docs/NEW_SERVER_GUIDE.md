# New SLURM server: beginner installation and BTAH verification

This guide installs FFOpt without modifying the server's base Conda environment. The concrete
paths below are for `peizy@mag1` and the working directory
`/storage/home/liguoling/peizy/BTAH-workflow`.

## 1. Log in and enter the workspace

```bash
ssh peizy@8.134.109.71 -p 25570
cd /storage/home/liguoling/peizy/BTAH-workflow
pwd
```

The final directory layout is:

```text
BTAH-workflow/
|-- ffopt-lammps/       # program source; update with git pull
|-- btah-validation/    # independent BTAH calculation project
`-- vasp_CPU.sh         # pre-existing file; FFOpt does not modify it
```

## 2. Download or update FFOpt

First installation:

```bash
git clone https://github.com/Leduo-Pei/ffopt-lammps.git
```

Later updates:

```bash
cd /storage/home/liguoling/peizy/BTAH-workflow/ffopt-lammps
git pull --ff-only
```

## 3. Create an isolated CPU environment

The non-interactive shell on this server does not initialize Conda automatically, so first load
its shell helper explicitly:

```bash
source /storage/home/liguoling/peizy/software/anaconda3/etc/profile.d/conda.sh
conda create -y -n ffopt-cpu -c conda-forge \
  --solver libmamba python=3.11 "lammps=*=*openmpi*" openmpi pip
conda activate ffopt-cpu
```

Install CPU PyTorch first. This avoids downloading CUDA libraries on a cluster whose GPU
partition is currently unavailable:

```bash
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
cd /storage/home/liguoling/peizy/BTAH-workflow/ffopt-lammps
python -m pip install -e ".[full,dev]"
```

Verify the executables:

```bash
which python
which ffopt
which lmp
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
lmp -help | head
```

Expected: all three executables are inside `.../envs/ffopt-cpu/bin`; PyTorch reports `False`
for CUDA on this CPU profile; LAMMPS prints its version and help text.

## 4. Configure this machine once

This server has two 48-core idle nodes. The profile below runs 24 independent LAMMPS candidates
at once, with 4 MPI ranks per candidate and one OpenMP thread per rank:

```bash
ffopt machine configure \
  --name ccelab-2node \
  --backend slurm \
  --lammps /storage/home/liguoling/peizy/software/anaconda3/envs/ffopt-cpu/bin/lmp \
  --mpi srun \
  --partition CPU \
  --nodes 2 \
  --cores 96 \
  --workers 24 \
  --ranks 4 \
  --omp-threads 1 \
  --walltime 14-00:00:00 \
  --timeout 216000 \
  --force

ffopt machine show --name ccelab-2node
```

BO, sampling, audit, and AL spread independent four-rank LAMMPS evaluations across both nodes.
ANN training is one Python process and therefore uses one 48-core node; it is not incorrectly
split across two nodes. No GPU resource is requested.

The reusable machine profile is stored in `~/.config/ffopt/machines.toml`. Scientific inputs do
not contain server paths, core counts, partitions, or MPI commands.

## 5. Create an independent BTAH verification project

```bash
cd /storage/home/liguoling/peizy/BTAH-workflow
mkdir -p btah-validation
cp ffopt-lammps/examples/btah/cluster_smoke.in btah-validation/ffopt.in
cp -a ffopt-lammps/data btah-validation/data
sed -i 's#../../data/#data/#g' btah-validation/ffopt.in
cd btah-validation
```

`ffopt.in` is the only scientific control file. The supplied smoke input still uses the standard
bulk minimization plus 300 K NPT protocol, but its MD, BO, sampling, ANN, and AL budgets are
deliberately short. It verifies the whole software path; it is not a production fit.

## 6. Check before submitting

```bash
ffopt check ffopt.in
ffopt explain ffopt.in
ffopt doctor --project ffopt.in --machine ccelab-2node
ffopt run ffopt.in --machine ccelab-2node --run-id install-smoke --dry-run
```

Do not submit if `check` or `doctor` reports a failure. `--dry-run` creates no SLURM job and shows
the exact stage commands and output directory.

## 7. Run the complete verification workflow

Interactive automatic progression, while the SSH terminal remains connected:

```bash
ffopt run ffopt.in \
  --machine ccelab-2node \
  --run-id install-smoke \
  --watch \
  --poll-seconds 60
```

Safer manual progression after disconnecting: run the following same command whenever the active
SLURM stage finishes. FFOpt reads `state.sqlite`, keeps completed stages, and submits only the next
stage:

```bash
ffopt run ffopt.in --machine ccelab-2node --run-id install-smoke
```

Monitor without changing the calculation:

```bash
squeue -u peizy
ffopt status --project ffopt.in --machine ccelab-2node --run-id install-smoke
```

The full stage order is `BO -> sample -> ANN -> AL -> validate`. Logs, checkpoints, trained ANN,
optimized parameters, computed properties, and final trajectories are written below:

```text
btah-validation/runs/btah_cluster_smoke/pipelines/install-smoke/
```

If a time limit kills a stage, use the same `ffopt run` command. Do not delete the run directory
or change `--run-id`. If the scientific input must change, use a new run ID or `--new`; FFOpt will
not silently combine incompatible checkpoints.

## 8. Start a production BTAH calculation only after the smoke test

```bash
cd /storage/home/liguoling/peizy/BTAH-workflow
mkdir -p btah-production
cp ffopt-lammps/examples/btah/charge_only.in btah-production/ffopt.in
cp -a ffopt-lammps/data btah-production/data
sed -i 's#../../data/#data/#g' btah-production/ffopt.in
cd btah-production
ffopt check ffopt.in
ffopt doctor --project ffopt.in --machine ccelab-2node
ffopt run ffopt.in --machine ccelab-2node --run-id production
```

The production input restores 20,000 equilibration steps, 40,000 production steps, 2,000 local
parameter points with three seeds, eight ANN members, and the full AL budget. Review every target,
type parameter, range, fixed parameter, data path, and workflow stage in `ffopt.in` before this
submission.
