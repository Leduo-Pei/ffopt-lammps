# Resume and control a workflow

## Normal restart

Use one stable run ID and repeat the same command:

```bash
ffopt run ffopt.in --machine cluster-1node --watch
```

FFOpt stores stage state in
`runs/<project>/pipelines/default/state.sqlite`. A stage is complete only when
its command succeeded and all required artifacts exist. A SLURM job killed by
wall time is marked incomplete and resubmitted; BO and AL also use their own
checkpoints where available.

## Stop after one stage

The recommended way to run only BO is to say so in the input:

```text
workflow bo
```

For a temporary pause without editing the input:

```bash
ffopt run ffopt.in --machine cluster-1node --until bo --watch
```

Continue later:

```bash
ffopt run ffopt.in --machine cluster-1node --from-stage sample --watch
```

The upstream artifacts and scientific signature must still match.

## Start an independent campaign

```bash
ffopt run ffopt.in --machine cluster-1node --new --watch
```

`--new` creates a timestamped run. Do not use it for an ordinary restart,
because it deliberately does not reuse the default pipeline state.

## Diagnose state

```bash
ffopt status ffopt.in --machine cluster-1node
ffopt logs ffopt.in --stage sample --lines 120
ffopt results ffopt.in --json
squeue -u "$USER"
```

If `ffopt.in` changes, FFOpt hashes the scientific settings and invalidates
the affected stage and downstream stages. Machine concurrency is excluded
from the scientific hash so the same run can move between compatible one- and
two-node profiles.
