# FFOpt-LAMMPS architecture

## Design rule

The reusable runtime must not contain molecule names, atom-type labels, target
values, or machine paths. BTAH is a regression project composed from the same
interfaces that future molecular systems use.

## Configuration layers

FFOpt composes configuration in this order:

1. Portable machine defaults from the project.
2. System definition and parameter space.
3. Selected property modules.
4. BO, surrogate, and active-learning method modules.
5. A user machine profile from `~/.config/ffopt/machines/`.
6. Explicit project overrides.

Scientific project files can therefore be committed and shared without local
LAMMPS paths, SLURM partitions, or conda activation commands.

All project-relative paths are resolved from the directory containing
`project.yaml`. A project can be cloned or moved without editing its paths.
References beginning with `builtin:` resolve from the installed wheel's
`share/ffopt/` data directory, allowing external projects to reuse versioned
method modules without depending on the source checkout.

## Runtime boundary

The current alpha keeps the validated numerical implementation in the
`engine`, `utils`, and `viz` compatibility packages. The installed `ffopt`
entry point delegates to that implementation, so migration does not change BO,
NN, AL, or objective values.

`engine.lammps_interface.LAMMPSRunner` supplies validated LAMMPS primitives.
Bulk, sublimation, adsorption, and surface evaluators declare dependencies and
compose only the tasks selected by a project. Sublimation, for example,
declares bulk as a prerequisite and reuses its molecular-crystal energy.
The public registry discovers third-party evaluators through Python package
entry points and validates their names, declared outputs, and dependencies
before any LAMMPS evaluation starts.

## Persistent execution graph

Each named run stores stage records in `state.sqlite`. A stage identity hashes
its scientific configuration, stage settings, and upstream identity. Verified
completed artifacts are reused; changing sampling settings invalidates sampling
and downstream stages while preserving BO. BO checkpoints, sampling replicate
CSVs, and final audit replicate CSVs provide finer-grained continuation inside
the expensive LAMMPS stages.

The graph is:

```text
BO -> focused sampling -> surrogate -> AL -> final audit -> export
   -> trajectory-producing validation
```

Local and SLURM backends consume the same stage records. `ffopt run --resume`
runs only missing or retryable stages. SLURM mode records Job IDs and can either
return after each submission or remain attached with `--watch`.

## Public artifacts

Each pipeline run contains a state database, BO and sampling tables, trained
surrogate, AL history, robust audit, exported parameters, trajectories, final
structures, and a validation report. Expanded configuration snapshots retain
the scientific and execution inputs used by the run.

## Acceptance criteria

- BTAH 41D, 27D, and 13D project composition remains unchanged numerically.
- A project outside the source checkout resolves every path from its own root.
- Machine-specific values never need to enter a project or Git history.
- A new LAMMPS data file can be inspected without molecule-specific code.
- Killing any stage and rerunning with `--resume` never repeats completed work.
- Adding a new system requires data, YAML, and optional LAMMPS
  templates, but no edits under `src/ffopt` or `engine`.
