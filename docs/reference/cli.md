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
| `ffopt explain ffopt.in` | Print dimensions, fixed/free/derived parameters, properties, and protocols. |
| `ffopt doctor ffopt.in --machine NAME` | Check the complete runtime environment. |

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
`~/.config/ffopt/machines.toml`.

## Execution

```bash
ffopt run ffopt.in --machine NAME --dry-run
ffopt run ffopt.in --machine NAME
ffopt run ffopt.in --machine NAME --watch
```

| Option | Meaning |
|---|---|
| `--dry-run` | Print stage commands and SLURM plan without execution. |
| `--watch` | Poll SLURM and automatically submit the next completed dependency. |
| `--poll-seconds N` | Scheduler polling interval. |
| `--until STAGE` | Temporarily stop after a stage. |
| `--from-stage STAGE` | Continue from an active stage after prior artifacts exist. |
| `--run-id NAME` | Use a named restart state instead of `default`. |
| `--new` | Start an independent timestamped campaign. |
| `--resume` | Explicit resume flag; `.in` projects already auto-resume by default. |

## Monitoring and results

```bash
ffopt status ffopt.in --machine NAME
ffopt logs ffopt.in --stage bo --lines 100
ffopt logs ffopt.in --paths
ffopt results ffopt.in
ffopt results ffopt.in --json
```

`status` reports persisted stage state and verifies artifact presence. `logs`
reads scheduler stdout/stderr. `results` resolves exact paths for parameters,
properties, metrics, histories, and trajectories.

## Packaged acceptance test

```bash
ffopt self-test --machine NAME --watch
```

Useful options are `--prepare-only`, `--workdir PATH`, `--dry-run`, and
`--skip-machine-test`. Reusing the same self-test work directory resumes its
marked acceptance project; FFOpt refuses to overwrite an unrelated non-empty
directory.

## Property plugins

```bash
ffopt plugins
ffopt plugins --json
```

This lists built-in evaluators and Python entry-point plugins. Plugin authoring
is an advanced extension path and is not required for bulk, sublimation, or
adsorption projects.
