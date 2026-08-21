# Machine-profile reference

FFOpt writes profiles to `~/.config/ffopt/machines.toml`. Users should normally
create them with `ffopt machine configure`; direct TOML editing is supported
for inspection and site-specific adjustments.

Only `local` has a built-in zero-configuration profile. Every scheduler must
use a configured name. Names start with an ASCII letter and then use only
letters, numbers, `.`, `_`, or `-`, for example `cluster-2node`.

## Example

```toml
[machines.cluster-2node]
format = 2
backend = "slurm"

[machines.cluster-2node.lammps]
executable = "/absolute/path/to/lmp"
mpiexec = "/absolute/path/to/mpirun"
mpi_flavor = "openmpi"
timeout = 216000

[machines.cluster-2node.parallel]
workers = 24
mpi_ranks = 4
omp_threads = 1

[machines.cluster-2node.slurm]
partition = "YOUR_PARTITION"
nodes = 2
total_cores = 96
walltime = "14-00:00:00"
memory_per_node = "64G"
```

The compact format stores shared resources once. At runtime FFOpt expands the
profile so `bo`, `sample`, `al`, and `audit` can distribute independent LAMMPS
evaluations over all nodes. `nn` keeps its ANN controller on one node while
retaining worker slots for post-training LAMMPS validation. `validate` requests
one LAMMPS-sized allocation. Existing expanded profiles from earlier releases
remain readable.

## Field mapping

| TOML field | Configure option | Meaning |
|---|---|---|
| `backend` | `--backend` | `local` or `slurm` |
| `lammps.executable` | `--lammps` | Absolute LAMMPS path or command on `PATH` |
| `lammps.mpiexec` | `--mpi` | MPI launcher used inside allocated nodes |
| `lammps.mpi_flavor` | `--mpi-flavor` | Explicit launcher dialect: `openmpi` or `intelmpi` |
| `lammps.timeout` | `--timeout` | Seconds allowed for one LAMMPS evaluation |
| `parallel.workers` | `--workers` | Concurrent independent parameter evaluations |
| `parallel.mpi_ranks` | `--mpi-ranks` | MPI ranks per evaluation |
| `parallel.omp_threads` | `--omp-threads` | Threads per MPI rank |
| `slurm.nodes` | `--nodes` | Scheduler nodes |
| `slurm.total_cores` | `--total-cores` | Total allocation sanity value |
| `slurm.partition` | `--partition` | Site-specific SLURM partition, not a node name |
| `slurm.qos` | `--qos` | Optional SLURM QOS |
| `slurm.walltime` | `--walltime` | Job wall-time request |
| `slurm.memory_per_node` | `--memory-per-node` | SLURM memory request per node |
| `slurm.gpus` | `--gpus` | GPUs requested for ML-capable stages |

Machine fields do not alter parameter bounds, random seeds, BO batch size,
sampling count, targets, weights, or LAMMPS protocol. Those belong to
`ffopt.in` and are included in scientific provenance.

When more than one MPI rank is used, set `mpi_flavor` explicitly. Open MPI
and Intel MPI/Hydra both commonly install executables named `mpiexec` or
`mpirun`, but their node-placement and binding flags are incompatible. FFOpt
only infers the flavor from unambiguous paths such as `.../openmpi/...` or
`.../intel/.../mpi/...`; an ambiguous launcher fails before LAMMPS starts.
The choice is stored in the machine profile so resumed runs do not depend on
whichever MPI module happens to be first on `PATH`.

## Environment setup

The generated profile assumes the Python executable used to launch `ffopt` is
available on compute nodes. If a site requires modules or environment setup,
add an `env_setup` array under the profile's `slurm` table:

```toml
[machines.cluster-2node.slurm]
env_setup = [
  "source /path/to/conda.sh",
  "conda activate ffopt",
]
```

Rare stage-specific scheduler changes belong in an override table, so the
common profile stays short:

```toml
[machines.cluster-2node.stages.nn]
walltime = "02:00:00"
gpu = 1
```

Prefer absolute executable paths. Do not store passwords, tokens, or SSH keys
in `machines.toml`.

`ffopt machine configure` updates this file atomically and preserves all other
named profiles. If the file is malformed, FFOpt stops with its exact path and
asks you to repair or move it instead of silently replacing the configuration.
