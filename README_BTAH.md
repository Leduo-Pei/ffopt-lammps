# BTAH LJ/Charge Inverse Workflow

This repository is a modular BO -> NN -> AL workflow for inferring LJ and
charge parameters from experimental or reference properties. BTAH is the current
worked example; the layout is intended to be reused for elements, compounds,
molecular crystals, and interface systems.

## One User Entry Point

For normal work, edit only the `include` in `projects/project.yaml` and use one
command style:

```powershell
python ffopt.py show
python ffopt.py status
python ffopt.py doctor
python ffopt.py bo --dry-run          # preview only; no calculation
python ffopt.py bo --smoke            # one real local LAMMPS check
python ffopt.py bo --machine local
python ffopt.py bo --machine cluster --resume
python ffopt.py sample --machine cluster
python ffopt.py nn --machine local --no-validate
python ffopt.py al --machine local
python ffopt.py audit --machine local
python ffopt.py validate --machine local
python ffopt.py plot
```

The project file selects the system, property modules, BO/NN/AL methods and
machine profile. Expanded configs and new outputs are placed under
`runs/<project>/`. Direct stage scripts are internal implementation details.

## Main Ideas

- `projects/*.yaml` are the only user-facing task recipes.
- `configs/properties/*.yaml` define independent property modules such as bulk
  NPT, adsorption energy, and sublimation proxy.
- `configs/methods/*/*.yaml` define selectable BO, NN, and AL methods.
- `configs/machines/*.yaml` isolate local/cluster paths and resources.
- `engine/` contains implementation code and is not normally edited.
- `analysis/` contains offline diagnostics and model comparisons.
- `data/` and `lammps/inputs/` are organized by property type.
- `archive/` contains old commands/configs; `runs/*/legacy/` contains completed
  historical calculations retained for reproducibility.

## Stage Control

The stages are intentionally independent. Completing BO does not automatically
start NN, and completing NN does not automatically start AL. The user inspects
`python ffopt.py status` and explicitly starts the next stage.

The default ANN module uses a low-noise core plus a wider BO trend buffer:

- stable candidate means: `objective <= 0.03`
- raw BO rows: `objective <= 0.075`
- stable-mean weight: `6`
- group-safe splitting: identical parameter sets never cross train/validation/test

Additional audited or sampled CSV files are listed once under
`stages.nn.additional_core_files` in the project file. The same data contract
is then reused by every `ffopt.py nn` invocation.

## Current Property Modules

- `bulk_runtime.yaml`: runs and parses bulk NPT outputs (`a/b/c/angles/density/pe/enthalpy`).
- `bulk_targets.yaml`: adds bulk structural targets to the objective.
- `adsorption_energy.yaml`: adds `ead` and its Au/BTAH three-box setup.
- `sublimation_proxy.yaml`: adds `esub_proxy = E_single_min - <PE_bulk_NPT>/Nmol`.

For BTAH_v2.1, `sublimation_proxy.yaml` fits this proxy directly to
`98.5 kJ/mol`; `thermal_correction_kj_mol` must remain `0.0` unless a new,
explicitly justified thermochemical target is introduced. If target values or
weights change after simulations have completed, run the internal rescoring tool
before NN filtering, AL ranking, or final selection so stale CSV objectives are
not reused.

```powershell
python -m engine.rescore_results `
  --config runs/btah/_configs/btah_local.yaml `
  --input-root runs/btah/legacy/training_runs/BTAH_822_20260709_remote `
  --output-dir runs/btah/legacy/training_runs/BTAH_822_20260709_remote/corrected_target_98p5
```

New BO/sampling/AL rows record `_objective_target_*` and
`_objective_weight_*` provenance columns. NN and AL reject an explicitly
mismatched scored CSV and direct the user to the command above.

If a recipe omits `bulk_targets.yaml`, bulk structural properties do not enter
the objective. If it also disables sublimation, the runner skips bulk NPT
entirely and can evaluate adsorption alone.

## BO Method Selection

BO is user-selected by the included method module:

- `configs/methods/bo/turbo.yaml`: explicitly uses TuRBO.
- `configs/methods/bo/gp.yaml`: explicitly uses GP.
- `configs/methods/bo/saasbo.yaml`: explicitly uses SAASBO.
- `configs/methods/bo/auto.yaml`: lets the code choose by parameter dimension.

The selected BO method is the path under `modules.methods.bo` in the project
file. Change that one line to switch methods; the command remains unchanged.

During BO, every completed round reports the current global-best calculated
properties beside their targets and relative errors. The same information is
available while the job is running in the BO output directory:

- `progress_latest.csv`: current global best, one row per active property.
- `progress_history.csv`: global-best objective and property errors by round.

Set `optimization.progress_report.property_table_every` in the selected BO
method module to change the terminal reporting interval.

## Data Layout

- `data/bulk/BTAH_822_bulk.data`
- `data/molecule/BTAH_822_single.data`
- `data/adsorption/ad_complex.data`
- `data/adsorption/ad_slab.data`
- `data/adsorption/ad_mol.data`

LAMMPS input files live under `lammps/inputs/` with matching property folders.
