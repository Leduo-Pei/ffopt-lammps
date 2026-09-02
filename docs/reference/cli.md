# Command-line reference

The normal user path has one command style:

```bash
ffopt run ffopt.in --machine PROFILE
```

Stage selection belongs in `workflow` inside `ffopt.in`. Repeating `run`
resumes the compatible default campaign.

## Project creation and checks

| Command | Purpose |
|---|---|
| `ffopt init NAME ...` | Create a project and explicit `ffopt.in` from LAMMPS data. |
| `ffopt inspect FILE.data` | Show type IDs, labels, counts, masses, LJ terms, and charge ranges. |
| `ffopt data check ...` | Validate one or more data files and their cross-file contracts. |
| `ffopt check ffopt.in` | Parse and semantically validate without running LAMMPS. |
| `ffopt explain ffopt.in` | Print dimensions, fixed/free/derived parameters, protocols, scientific budgets, and the effective BO method. |
| `ffopt doctor ffopt.in --machine NAME` | Check the complete runtime environment and report any automatic BO-method fallback. |

Use `ffopt <command> --help` for every option.
`doctor` also checks the cross-file LAMMPS data contract for every active
property. Use `ffopt --debug COMMAND ...` to retain a full traceback while
diagnosing an unexpected native-installation problem.

`init`, `inspect`, and `data check` accept either ordinary paths or packaged
references such as `builtin:data/bulk/BTAH_822_bulk.data`. Ordinary relative
paths start from the current shell directory. Relative paths inside
`ffopt.in` start from the directory containing that input file. Prepare a
visible copy of all BTAH files with:

```bash
ffopt self-test --prepare-only --workdir ./ffopt-btah-example
```

## Machine profiles

```bash
ffopt machine probe
ffopt machine configure --name NAME ...
ffopt machine list
ffopt machine show --name NAME
ffopt machine test --name NAME
```

`configure --force` replaces the named profile only. Profiles are stored in
`~/.config/ffopt/machines.toml`. `local` is the only built-in profile; a
SLURM machine must be configured under an explicit name. `local` means direct
execution on the current host without scheduler submission; do not use it for
production on a cluster login node. A `machine probe --partition NAME` value
is a site-specific SLURM partition reported by `sinfo`, not a node name.

## Execution

```bash
ffopt run ffopt.in --machine NAME --dry-run
ffopt run ffopt.in --machine NAME
ffopt run ffopt.in --machine NAME --watch
```

| Option | Meaning |
|---|---|
| `--dry-run` | Read-only preview of stage commands and SLURM plan. |
| `--watch` | Poll SLURM and automatically submit the next completed dependency. |
| `--poll-seconds N` | Scheduler polling interval. |
| `--until STAGE` | Temporarily stop after a stage. |
| `--from-stage STAGE` | Continue from an active stage after prior artifacts exist. |
| `--run-id NAME` | Use a named restart state instead of `default`. |
| `--new` | Start an independent timestamped campaign. |

## Monitoring and results

```bash
ffopt status ffopt.in --machine NAME
ffopt logs ffopt.in --stage bo --lines 100
ffopt logs ffopt.in --paths
ffopt results ffopt.in
ffopt results ffopt.in --json
```

`status` reports persisted stage state and verifies artifact presence. On a
SLURM login host it also queries the live scheduler state/reason and summarizes
an in-progress BO checkpoint by round, evaluation count, and best objective.
When `--machine` is omitted for an existing pipeline, it displays the profile
recorded when that pipeline was run.
`logs` reads scheduler stdout/stderr. `results` resolves exact paths for
parameters, properties, metrics, histories, and trajectories.

## Promote a validated baseline

Promotion is explicit and is not an automatic pipeline stage:

```bash
ffopt promote ffopt.in \
  --run-id RUN_ID \
  --canonical-root runs/PROJECT \
  --dry-run

ffopt promote ffopt.in \
  --run-id RUN_ID \
  --canonical-root runs/PROJECT \
  --allow-best-effort \
  --replace-legacy
```

The source is always derived as
`runs/<project>/pipelines/<run-id>/validate`; arbitrary source directories are
not accepted. FFOpt requires a completed state record, a matching output
directory, a finished timestamp, and a valid material-validation manifest
whose declared files still match their SHA-256 digests.

| Option | Meaning |
|---|---|
| `--canonical-root PATH` | Required operational publication target; it is deliberately not part of `ffopt.in`, and its final path component must exactly equal the project name. |
| `--allow-best-effort` | Permit a complete hard-gate-passing result that missed its requested mechanical tier to compete for current best; an otherwise publishable attempt is still snapshotted without this flag. |
| `--replace-legacy` | Durably archive an unmanaged root canonical before replacing it on the first managed promotion. An interruption during legacy removal requires journal/archive inspection; deleting a stale lock alone is not recovery. |
| `--force-downgrade` | Permit an otherwise valid lower/incomparable result; never bypasses completeness, hard gates, hashes, or best-effort permission. |
| `--dry-run` | Validate source, target, permission, and comparison without writing. A blocked plan exits with status 2 after printing its result. |
| `--json` | Emit machine-readable promotion outcome and artifact paths. |

Accepted results rank above best-effort results. Results in the same tier and
with the same validation-protocol fingerprint compare their independent
finite-temperature maximum mechanical error, lower first. Different protocols
are incomparable by default. Rejected or incomplete validation attempts can
never become current-best.

## Packaged acceptance test

```bash
ffopt self-test --machine NAME --watch
```

Useful options are `--prepare-only`, `--workdir PATH`, `--dry-run`, and
`--skip-machine-test`. Reusing the same self-test work directory resumes its
marked acceptance project only when the input and packaged data fingerprints
and FFOpt version match. FFOpt refuses to reuse modified, incomplete, stale,
cross-version, or unrelated directories.

## Property plugins

```bash
ffopt plugins
ffopt plugins --json
```

This lists built-in evaluators and Python entry-point plugins. Plugin authoring
is an advanced extension path and is not required for bulk, sublimation, or
adsorption projects.
