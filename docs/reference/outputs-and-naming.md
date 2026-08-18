# Outputs and naming

## Project layout

```text
project/
|-- ffopt.in
|-- data/
|   |-- bulk/
|   |-- molecule/
|   `-- adsorption/
`-- runs/
    `-- <project>/
        `-- pipelines/
            `-- <run-id>/
```

Recommended names use lowercase ASCII letters, numbers, `_`, and `-`. Data
filenames should describe role, not a temporary calculation, for example
`benzene_bulk.data`, `benzene_single.data`, `benzene_Au111_complex.data`.

## Pipeline directory

```text
<run-id>/
|-- state.sqlite
|-- jobs/                 # generated SLURM scripts
|-- logs/                 # scheduler stdout/stderr
|-- bo/
|-- sample/
|-- nn/
|-- al/
`-- validate/
```

The default run ID is `default`. Timestamped directories appear only after
`--new` or an explicit `--run-id`.

## Important artifacts

| Stage | Artifact | Purpose |
|---|---|---|
| BO | `all_results.csv` | Every parameter vector, success flag, objective, and calculated property |
| BO | `stable_results.csv` | Multi-seed robust ranking when BO audit is enabled |
| BO | `best_parameters.txt` | Best BO parameter set in engine format |
| Sample | `design.csv` | Deterministic parameter design before LAMMPS |
| Sample | `local_replicates.csv` | One row per parameter vector and seed |
| Sample | `local_results.csv` | Aggregated training rows |
| NN | `forward_nn.pt` | ANN ensemble and feature metadata |
| NN | `parity_data.csv` | Held-out true/predicted values used for parity plots |
| NN | `nn_optimize_result.json` | Per-property test metrics and surrogate/LAMMPS candidates |
| AL | `active_learning_history.json` | Round-by-round predictions and LAMMPS outcomes |
| AL | `final_parameters.json` | Best accumulated AL parameter set |
| Validate | `validation_summary.json` | Final objective, properties, source, and pass/fail gates |
| Validate | `computed_properties.csv` | Calculated/target errors and tolerances |
| Validate | `bulk/` | Final bulk data, logs, averages, and trajectory |
| Validate | `adsorption/` | Component energies, structures, and trajectories when enabled |

Exact availability is reported by:

```bash
ffopt results ffopt.in
```

## Provenance

Generated runtime JSON, stage signatures, input paths, objective targets and
weights, and property definitions are retained with the run. Do not edit
generated JSON or CSV files to continue a campaign. Edit `ffopt.in`, rerun
`ffopt check`, and let FFOpt invalidate affected downstream stages.

## Archiving

Keep `ffopt.in`, all source data files, the complete selected pipeline
directory, the FFOpt version, LAMMPS version, and `machines.toml` profile used
for publication. Large transient per-candidate directories can be compressed
after the final tables and trajectories have been verified.
