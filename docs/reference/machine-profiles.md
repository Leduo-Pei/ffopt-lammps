# Machine-profile reference

FFOpt writes profiles to `~/.config/ffopt/machines.toml`. Users should normally
create them with `ffopt machine configure`; direct TOML editing is supported
for inspection and site-specific adjustments.

Only `local` has a built-in zero-configuration profile. Every scheduler must
use a configured name. Names start with an ASCII letter and then use only
letters, numbers, `.`, `_`, or `-`, for example `cluster-2node`.

## Example

```toml
[machines.cluster-2node.machine]
name = "cluster-2node"
backend = "slurm"
extends = "cluster"

[machines.cluster-2node.lammps]
executable = "/absolute/path/to/lmp"
mpiexec = "/absolute/path/to/mpirun"
mpi_flavor = "openmpi"
timeout = 216000

[machines.cluster-2node.parallel]
max_workers = 24
cores_per_worker = 4
omp_threads_per_worker = 1
use_mpi = true
scheduler_launcher = "srun"
scheduler_nodes = 2
workers_per_node = 12
```

Generated `cluster` tables hold stage-specific SLURM directives. `bo`,
`sample`, `al`, and `audit` can distribute independent LAMMPS evaluations over
all nodes. `nn` keeps its ANN training controller on one node and may request a
GPU; that same allocation exposes several worker slots for the stage's
post-training LAMMPS candidate validation. `validate` runs one final parameter
set and therefore requests one LAMMPS-sized allocation.

## Field mapping

| TOML field | Configure option | Meaning |
|---|---|---|
| `machine.backend` | `--backend` | `local` or `slurm` |
| `lammps.executable` | `--lammps` | Absolute LAMMPS path or command on `PATH` |
| `lammps.mpiexec` | `--mpi` | MPI launcher used inside allocated nodes |
| `lammps.mpi_flavor` | `--mpi-flavor` | Explicit launcher dialect: `openmpi` or `intelmpi` |
| `lammps.timeout` | `--timeout` | Seconds allowed for one LAMMPS evaluation |
| `parallel.max_workers` | `--workers` | Concurrent independent parameter evaluations |
| `parallel.cores_per_worker` | `--mpi-ranks` | MPI ranks per evaluation |
| `parallel.omp_threads_per_worker` | `--omp-threads` | Threads per MPI rank |
| `cluster.*.nodes` | `--nodes` | Scheduler nodes |
| `cluster.*.cores` | `--total-cores` | Total allocation sanity value |
| `cluster.*.partition` | `--partition` | SLURM partition |
| `cluster.*.qos` | `--qos` | Optional SLURM QOS |
| `cluster.*.time` | `--walltime` | Job wall-time request |
| `cluster.*.mem` | `--memory-per-node` | SLURM memory request per node |
| `cluster.nn/al.gpu` | `--gpus` | GPUs requested for ML stages |

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
add an `env_setup` array under the profile's `cluster` table:

```toml
[machines.cluster-2node.cluster]
env_setup = [
  "source /path/to/conda.sh",
  "conda activate ffopt",
]
```

Prefer absolute executable paths. Do not store passwords, tokens, or SSH keys
in `machines.toml`.

`ffopt machine configure` updates this file atomically and preserves all other
named profiles. If the file is malformed, FFOpt stops with its exact path and
asks you to repair or move it instead of silently replacing the configuration.
