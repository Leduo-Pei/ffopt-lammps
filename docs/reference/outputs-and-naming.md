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
Internal parameter columns are generated as `TYPELABEL_epsilon`,
`TYPELABEL_sigma`, and `TYPELABEL_charge`; users name the type once in
`ffopt.in` and do not maintain a second parameter-name list.

## Pipeline directory

```text
<run-id>/
|-- state.sqlite
|-- provenance/           # immutable input, runtime config, and environment
|-- jobs/                 # generated SLURM scripts
|-- logs/                 # scheduler stdout/stderr
|-- bo/
|-- sample/
|-- nn/
|-- al/
|-- audit/
|-- finalize/
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
| Audit | `stability_replicates.csv` | One final-audit row per candidate and seed |
| Audit | `stable_results.csv` | Robust `mean + standard deviation` candidate ranking |
| Finalize | `final_summary.json` | Selected audit source, score, and resolved parameters |
| Finalize | `final_parameters.lammps` | Include-ready complete robust parameter set |
| Validate | `validation_summary.json` | Final objective, properties, source, and pass/fail gates |
| Validate | `computed_properties.csv` | Calculated/target errors and tolerances |
| Validate | `final_atom_parameters.csv` | Complete resolved type/label/epsilon/sigma/charge table |
| Validate | `final_parameters.lammps` | Include-ready `pair_modify`, `pair_coeff`, and `set type ... charge` commands |
| Validate | `final_parameters.json` | Free, derived, fixed, and per-type final parameter provenance |
| Validate | `eval_0000/bulk/` | Final bulk data, logs, averages, structure, and trajectory |
| Validate | `eval_0000/adsorption/` | Component energies, structures, and trajectories when enabled |

Exact availability is reported by:

```bash
ffopt results ffopt.in
```

## Provenance

Content-addressed snapshots of the exact input text, expanded runtime JSON,
FFOpt/Python versions, executable paths, stage signatures, targets, weights,
and property definitions are retained with the run. A pending SLURM job keeps
the immutable configuration named in its submitted script. Do not edit
generated JSON or CSV files to continue a campaign. Method-stage changes in
`ffopt.in` invalidate the affected stage and its downstream stages. Changes to
the physical system, targets, parameter space, or property protocol require a
new run because they change the scientific identity of the campaign.

## Archiving

Keep `ffopt.in`, all source data files, the complete selected pipeline
directory, the FFOpt version, LAMMPS version, and `machines.toml` profile used
for publication. Large transient per-candidate directories can be compressed
after the final tables and trajectories have been verified.
