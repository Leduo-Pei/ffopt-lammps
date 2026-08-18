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
SLURM machine must be configured under an explicit name.

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
