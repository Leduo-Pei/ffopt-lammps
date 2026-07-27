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

## Runtime boundary

The current alpha keeps the validated numerical implementation in the
`engine`, `utils`, and `viz` compatibility packages. The installed `ffopt`
entry point delegates to that implementation, so migration does not change BO,
NN, AL, or objective values.

The next runtime boundary replaces `engine.lammps_interface.LAMMPSRunner` with
property evaluators. Each evaluator will declare required files, construct
LAMMPS tasks, parse typed results, and apply property-specific sanity checks.
The workflow will execute only the evaluators selected by a project.

## Persistent execution graph

The target workflow stores every task in `state.sqlite`. A task identity is a
content hash of the project schema, resolved parameter vector, random seed,
property evaluator version, and input assets. Completed task identities are
immutable and reusable.

The graph is:

```text
preflight -> BO -> stability audit -> focused sampling -> surrogate
          -> AL proposal -> LAMMPS validation -> final audit -> export
```

Local and SLURM schedulers consume the same task records. `ffopt run --resume`
will submit only missing or retryable tasks, including after a wall-time kill.

## Public artifacts

Each run will contain a manifest, state database, tabular dataset, optimized
parameters, patched LAMMPS data files, trajectories, structures, figures, and a
validation report. Every artifact records configuration and code provenance.

## Acceptance criteria

- BTAH 41D, 27D, and 13D project composition remains unchanged numerically.
- A project outside the source checkout resolves every path from its own root.
- Machine-specific values never need to enter a project or Git history.
- A new LAMMPS data file can be inspected without molecule-specific code.
- Killing any stage and rerunning with `--resume` never repeats completed work.
- Adding a new system eventually requires data, YAML, and optional LAMMPS
  templates, but no edits under `src/ffopt` or `engine`.

