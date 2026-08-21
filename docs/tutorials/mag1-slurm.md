# Clean installation and acceptance on `mag1`

This is the verified site guide for `peizy@mag1`. It keeps the pre-existing
`vasp_CPU.sh` untouched and installs FFOpt in an isolated Conda environment.

## Site facts

```text
Home       /storage/home/liguoling/peizy
Workspace  /storage/home/liguoling/peizy/BTAH-workflow
Partition  CPU
Node CPU   48 cores
Conda      /storage/home/liguoling/peizy/software/anaconda3
```

## Choose first install or replacement

For a first installation, do not delete anything. Confirm that the intended
environment name is unused, then continue with **Install the release**:

```bash
source /storage/home/liguoling/peizy/software/anaconda3/etc/profile.d/conda.sh
conda env list
```

Use the replacement procedure below only when deliberately replacing an old
environment named exactly `ffopt` or `ffopt-cpu`. First confirm that no FFOpt
job from those environments is running, then back up the small machine-profile
directory before removing only those named Conda environments:

```bash
cd /storage/home/liguoling/peizy/BTAH-workflow
mkdir -p _preflight_backup
cp -a "$HOME/.config/ffopt" \
  "_preflight_backup/ffopt-config-$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true

source /storage/home/liguoling/peizy/software/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda env list
squeue -u "$USER"
conda env remove -n ffopt -y 2>/dev/null || true
conda env remove -n ffopt-cpu -y 2>/dev/null || true
conda env list
```

Do not remove unrelated Conda environments, project results, machine profiles,
or scheduler jobs. A release installation does not need a source checkout. If
an old checkout named exactly `ffopt-lammps` is confirmed and must be cleared,
archive it instead of deleting it:

```bash
cd /storage/home/liguoling/peizy/BTAH-workflow
realpath ffopt-lammps
mv ffopt-lammps \
  "_preflight_backup/ffopt-lammps-source-$(date +%Y%m%d-%H%M%S)"
```

## Install the release

Install the tagged release. A normal user does not need to clone the source
repository; the `ffopt` command can be called from any project directory:

```bash
cd /storage/home/liguoling/peizy/BTAH-workflow
conda create -y -n ffopt -c conda-forge --solver libmamba \
  python=3.11 "lammps=*=*openmpi*" openmpi pip
conda activate ffopt
conda env config vars set PYTHONNOUSERSITE=1
conda deactivate
conda activate ffopt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install \
  "ffopt-lammps[full] @ https://github.com/Leduo-Pei/ffopt-lammps/releases/download/v0.3.0a4/ffopt_lammps-0.3.0a4-py3-none-any.whl"
```

FFOpt also exports `PYTHONNOUSERSITE=1` in generated SLURM scripts so packages
under `~/.local/lib/python*` cannot leak into compute-node Python processes.

Verify paths and versions:

```bash
which python
which ffopt
which lmp
which mpirun
ffopt --version
python -c "import site; print(site.ENABLE_USER_SITE)"
python -c "import ffopt, torch; print(ffopt.__version__); print(torch.__version__, torch.cuda.is_available())"
python -m pip check
lmp -help | head
```

Expected: executable paths contain `/envs/ffopt/`, FFOpt reports `0.3.0a4`,
user-site reports `False`, `pip check` reports no broken requirements, and CPU
PyTorch reports CUDA `False`.

## Configure one- and two-node profiles

```bash
LMP=/storage/home/liguoling/peizy/software/anaconda3/envs/ffopt/bin/lmp
MPI=/storage/home/liguoling/peizy/software/anaconda3/envs/ffopt/bin/mpirun

ffopt machine configure \
  --name mag1-1node --backend slurm \
  --lammps "$LMP" --mpi "$MPI" --partition CPU \
  --nodes 1 --total-cores 40 --workers 10 \
  --mpi-ranks 4 --omp-threads 1 \
  --memory-per-node 64G --walltime 06:00:00 \
  --timeout 7200 --force

ffopt machine configure \
  --name mag1-2node --backend slurm \
  --lammps "$LMP" --mpi "$MPI" --partition CPU \
  --nodes 2 --total-cores 80 --workers 20 \
  --mpi-ranks 4 --omp-threads 1 \
  --memory-per-node 64G --walltime 06:00:00 \
  --timeout 7200 --force
```

The full two-node profile needs 40 free cores on each node. When that request
cannot backfill promptly, create an optional acceptance profile with four
independent LAMMPS workers on each node:

```bash
ffopt machine configure \
  --name mag1-2node-backfill --backend slurm \
  --lammps "$LMP" --mpi "$MPI" --partition CPU \
  --nodes 2 --total-cores 32 --workers 8 \
  --mpi-ranks 4 --omp-threads 1 \
  --memory-per-node 16G --walltime 06:00:00 \
  --timeout 7200 --force
```

The acceptance profiles leave eight cores per node available on this shared
partition, which also makes short jobs easier to backfill. On exclusive idle
nodes, production profiles may instead use 48/96 total cores and 12/24
workers. Both profiles use the same packaged design: 24 LHS candidates plus
one warm-start centre, followed by one 24-candidate BO round. Thus the
two-node run changes concurrency only, not the 49-evaluation scientific
budget.

## Preflight

```bash
ffopt machine probe --partition CPU
ffopt machine show --name mag1-1node
ffopt machine show --name mag1-2node
ffopt machine test --name mag1-1node
ffopt machine test --name mag1-2node
```

The two-node test must report both allocated hostnames. It uses the full
20-worker/4-rank topology with a zero-step LAMMPS input, so it checks the
cross-node command queue in minutes before the scientific acceptance begins.
Use `mag1-2node-backfill` in both commands when testing that profile.

## Scientific acceptance

Run the same packaged workflow in two separate marked directories:

```bash
mkdir -p /storage/home/liguoling/peizy/BTAH-workflow/acceptance

ffopt self-test --machine mag1-1node \
  --workdir /storage/home/liguoling/peizy/BTAH-workflow/acceptance/one-node \
  --watch --poll-seconds 60

ffopt self-test --machine mag1-2node \
  --workdir /storage/home/liguoling/peizy/BTAH-workflow/acceptance/two-node \
  --watch --poll-seconds 60
```

If the terminal disconnects or a wall-time expires, repeat the identical
command. Do not add `--new`; the self-test resumes its `acceptance` run ID.
The manifest fingerprints `ffopt.in` and all packaged bulk, molecule, and
adsorption data files, so an edited or stale acceptance directory is rejected
rather than silently mixed with the installed release.

Monitor from another shell:

```bash
squeue -u peizy
ffopt status \
  /storage/home/liguoling/peizy/BTAH-workflow/acceptance/one-node/ffopt.in \
  --machine mag1-1node --run-id acceptance
ffopt logs \
  /storage/home/liguoling/peizy/BTAH-workflow/acceptance/one-node/ffopt.in \
  --run-id acceptance --stage bo --lines 100
```

Acceptance requires final LAMMPS validation to satisfy:

```text
objective <= 0.03
maximum fitted-property error <= 3%
every configured absolute tolerance passes
```

The one- and two-node outputs should agree within normal stochastic and
floating-point variation. Compare the validation tables and elapsed stage
times, not just SLURM completion state.

The completed `0.3.0a3` installation, job IDs, timings, exact artifact hashes,
and final property table are recorded in the
[versioned acceptance report](../reference/acceptance-v0.3.0a3.md).

## Start a real project

After both profiles pass, create a separate project directory; do not edit the
packaged acceptance input:

```bash
cd /storage/home/liguoling/peizy/BTAH-workflow
ffopt init my_crystal --bulk-data /path/to/crystal.data ...
cd my_crystal
ffopt check ffopt.in
ffopt explain ffopt.in
ffopt doctor ffopt.in --machine mag1-2node
ffopt run ffopt.in --machine mag1-2node --dry-run
ffopt run ffopt.in --machine mag1-2node --watch
```
