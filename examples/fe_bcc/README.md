# Fe BCC material-workflow example

Copy `ffopt.in` into a project directory, then place the three LAMMPS data
files named by the input under `data/`.  The input is the only scientific file
the user edits; `runs/` is created automatically.

The supplied Fe values are a single known structural-feasible warm start.  They
do not narrow the declared parameter ranges and do not import any old trajectory
or property result; BO, Sample, Audit, elasticity, AL and validation are all run
again under the current campaign provenance.

The `parameters` block explicitly declares `cutoff 12.5 A`. This is the one
cutoff inherited by bulk, surface, both elasticity fidelities, and final
validation; there is no property-local or hidden default. The value is exactly
`2.5 * 5.0 A`, where `5.0 A` is the largest allowed sigma in this example.
Every elemental-BCC input must satisfy `cutoff >= 2.5 * sigma_max` over its full
declared sigma domain.
Each periodic property must simultaneously satisfy `cutoff <= 0.45 * h_min`
after replication, using triclinic-safe face heights. The production replicas
meet both bounds; FFOpt rejects an undersized cell before submitting LAMMPS.

Check the input and print the complete expanded job graph without running:

```bash
ffopt check ffopt.in
ffopt run ffopt.in --machine cluster --dry-run
```

Run the managed campaign:

```bash
ffopt run ffopt.in --machine cluster --watch
```

`al rounds 8` is a budget, not eight user commands.  The pipeline expands the
rounds into separately restartable jobs and advances until scientific
patience, the target budget, or a real failure is recorded.  If the controller
is interrupted, issue the same `ffopt run ... --watch` command; verified
candidate and stage manifests are reused.

Finite-temperature screening is deliberately asymmetric in cost. Exactly 20
hard-gate candidates enter the quick three-seed 300 K promotion protocol; only
the promoted winner enters three independent long validation trajectories. The
`finalists` block uses `minimum 20`, `maximum 20`, and `require_minimum yes`, so
an undersized eligible pool stops before any promotion LAMMPS work instead of
silently continuing with fewer candidates.

This two-type model is an ordered-sublattice LJ surrogate.  Passing the fit
does not by itself establish transferable elemental Fe physics.  The final
report must retain label-swap, surface-termination, defect, and finite-
temperature diagnostics, and should be compared with an EAM/Finnis--Sinclair
baseline when transferability matters.
