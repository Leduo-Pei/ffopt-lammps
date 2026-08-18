# Configure execution machines

Machine settings are stored outside projects in
`~/.config/ffopt/machines.toml`. A project can therefore move between a local
workstation and a cluster without changing its scientific input.

## Inspect the host

```bash
ffopt machine probe
ffopt machine probe --partition CPU
```

The probe is advisory. Always verify the scheduler's CPU, memory, MPI, and GPU
policies with the cluster administrator.

## Configure profiles

If `lmp` is on `PATH`, the built-in `local` profile works without creating a
configuration:

```bash
ffopt doctor ffopt.in --machine local
ffopt run ffopt.in --machine local
```

That zero-configuration profile deliberately uses one serial worker. Create a
named profile when executable paths are not on `PATH` or when you want
controlled MPI and parallel-worker settings.

The built-in profile is visible and testable:

```bash
ffopt machine list
ffopt machine show --name local
ffopt machine test --name local
```

Local example:

```bash
ffopt machine configure --name local-workstation \
  --backend local --lammps /path/to/lmp --mpi /path/to/mpirun \
  --workers 4 --mpi-ranks 4 --omp-threads 1 --force
```

SLURM examples for 48-core nodes:

```bash
ffopt machine configure --name cluster-1node \
  --backend slurm --lammps /path/to/lmp --mpi /path/to/mpirun \
  --partition CPU --nodes 1 --total-cores 48 \
  --workers 12 --mpi-ranks 4 --omp-threads 1 \
  --memory-per-node 64G --walltime 14-00:00:00 \
  --timeout 216000 --force

ffopt machine configure --name cluster-2node \
  --backend slurm --lammps /path/to/lmp --mpi /path/to/mpirun \
  --partition CPU --nodes 2 --total-cores 96 \
  --workers 24 --mpi-ranks 4 --omp-threads 1 \
  --memory-per-node 64G --walltime 14-00:00:00 \
  --timeout 216000 --force
```

The two profiles use the same scientific `batch_size` from `ffopt.in`.
Changing `workers` changes concurrency and elapsed time, not BO candidates per
round or sampling point count.

## Meaning of CPU controls

| Option | Meaning | Scientific budget? |
|---|---|---|
| `--workers` | Independent LAMMPS evaluations running concurrently | No |
| `--mpi-ranks` | MPI processes inside each LAMMPS evaluation | No |
| `--omp-threads` | Threads used by each MPI process | No |
| `--total-cores` | Total CPUs requested from SLURM | No |
| `bo batch_size` | Candidates evaluated per BO round | Yes |
| `sample points` | Independent parameter vectors generated | Yes |
| `sample seeds` | LAMMPS replicas per parameter vector | Yes |

For a full allocation, require:

```text
total-cores >= workers * mpi-ranks * omp-threads
```

The per-evaluation `--timeout` must be shorter than `--walltime`. This lets
FFOpt penalize and move past one pathological parameter point while the stage
still has time to save checkpoints and finish other candidates.

`omp-threads 1` is intentional when each LAMMPS job already uses MPI ranks.
It prevents hidden thread oversubscription.

## Re-running configure

Profiles have unique names. Re-running without `--force` refuses to overwrite
an existing name. Re-running with `--force` replaces only that named table and
preserves all other profiles in `machines.toml`.

```bash
ffopt machine list
ffopt machine show --name cluster-1node
ffopt machine test --name cluster-1node
```

For a multi-node profile, `machine test` allocates its configured node and
worker topology and runs one tiny LAMMPS calculation in every worker slot. A
passing result therefore verifies cross-node dispatch rather than only a
single validation process.

## GPU behavior

LAMMPS stages use the executable and resources declared by the profile. ANN
and AL use `machine_learning.device = auto`; a CUDA-enabled PyTorch build uses
an available GPU, while CPU-only PyTorch is valid and falls back to CPU. A
SLURM profile requests GPUs only when `--gpus` is greater than zero.
