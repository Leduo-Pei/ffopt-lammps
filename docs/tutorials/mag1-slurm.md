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

## Clean reinstall

Back up only the small user machine configuration, then remove the old FFOpt
environment and source checkout:

```bash
cd /storage/home/liguoling/peizy/BTAH-workflow
mkdir -p _preflight_backup
cp -a "$HOME/.config/ffopt" \
  "_preflight_backup/ffopt-config-$(date +%Y%m%d-%H%M%S)" 2>/dev/null || true

source /storage/home/liguoling/peizy/software/anaconda3/etc/profile.d/conda.sh
conda deactivate 2>/dev/null || true
conda env remove -n ffopt-cpu -y
rm -rf /storage/home/liguoling/peizy/BTAH-workflow/ffopt-lammps
rm -rf "$HOME/.config/ffopt"
```

The paths above must be checked with `pwd` and `realpath` before deletion.
Do not remove unrelated Conda environments or scheduler jobs.

Install the tagged release:

```bash
cd /storage/home/liguoling/peizy/BTAH-workflow
git clone --branch v0.3.0a2 --depth 1 \
  https://github.com/Leduo-Pei/ffopt-lammps.git

conda create -y -n ffopt -c conda-forge --solver libmamba \
  python=3.11 "lammps=*=*openmpi*" openmpi pip
conda activate ffopt
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install \
  "ffopt-lammps[full] @ git+https://github.com/Leduo-Pei/ffopt-lammps.git@v0.3.0a2"
```

Verify paths and versions:

```bash
which python
which ffopt
which lmp
which mpirun
python -c "import ffopt, torch; print(ffopt.__version__); print(torch.__version__, torch.cuda.is_available())"
lmp -help | head
```

Expected: executable paths contain `/envs/ffopt/`, FFOpt reports `0.3.0a2`,
and CPU PyTorch reports CUDA `False`.

## Configure one- and two-node profiles

```bash
LMP=/storage/home/liguoling/peizy/software/anaconda3/envs/ffopt/bin/lmp
MPI=/storage/home/liguoling/peizy/software/anaconda3/envs/ffopt/bin/mpirun

ffopt machine configure \
  --name mag1-1node --backend slurm \
  --lammps "$LMP" --mpi "$MPI" --partition CPU \
  --nodes 1 --total-cores 48 --workers 12 \
  --mpi-ranks 4 --omp-threads 1 \
  --memory-per-node 64G --walltime 14-00:00:00 \
  --timeout 216000 --force

ffopt machine configure \
  --name mag1-2node --backend slurm \
  --lammps "$LMP" --mpi "$MPI" --partition CPU \
  --nodes 2 --total-cores 96 --workers 24 \
  --mpi-ranks 4 --omp-threads 1 \
  --memory-per-node 64G --walltime 14-00:00:00 \
  --timeout 216000 --force
```

Twelve four-rank evaluations fill one 48-core node. Twenty-four fill two
nodes. Both profiles use the same 24-candidate packaged acceptance input, so
the two-node run changes concurrency only.

## Preflight

```bash
ffopt machine probe --partition CPU
ffopt machine show --name mag1-1node
ffopt machine show --name mag1-2node
ffopt machine test --name mag1-1node
ffopt machine test --name mag1-2node
```

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
