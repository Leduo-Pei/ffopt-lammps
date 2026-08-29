# Fe BCC material-workflow example

Copy `ffopt.in` into a project directory, then place the three LAMMPS data
files named by the input under `data/`.  The input is the only scientific file
the user edits; `runs/` is created automatically.

The supplied Fe values are an explicit same-material incumbent coordinate.
They do not narrow the declared parameter ranges and do not import any old
trajectory, property value, or rank; BO, Sample, Audit, elasticity, AL and
validation are all run again under the current campaign provenance. For a new
material, enter a physical initial guess and change `incumbent initial` to
`incumbent off`.

The `parameters` block explicitly declares `cutoff 12.5 A`. This is the one
cutoff inherited by bulk, surface, both elasticity fidelities, and final
validation; there is no property-local or hidden default. The value is exactly
`2.5 * 5.0 A`, where `5.0 A` is the largest allowed sigma in this example.
Every elemental-BCC input must satisfy `cutoff >= 2.5 * sigma_max` over its full
declared sigma domain.
Each periodic property must simultaneously satisfy `cutoff <= 0.45 * h_min`
after replication, using triclinic-safe face heights. The production replicas
meet both bounds; FFOpt rejects an undersized cell before submitting LAMMPS.

The static and dynamic elasticity strain windows are intentionally separate.
The deterministic 0 K screen uses `static_strain 0.0005 0.001 0.002` and
extrapolates symmetric pressure slopes to zero strain. The 300 K calculation
uses `dynamic_strain 0.002 0.004 0.006` for a larger signal-to-noise ratio.
Potential-energy curvature is recorded only as a diagnostic: with the global
`shift no` hard cutoff, neighbour shells can cross the cutoff and create
discrete energy jumps that are not elastic curvature. `static_drift 5 percent`
is a hard eligibility gate, independent of `r2 0.98`: it rejects any candidate
whose canonical `B/Cprime/C44` zero-strain intercept changes by more than 5% in
the outer-shell/full-window audit. The third static magnitude supplies that
independent window check.

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

Finite-temperature screening is deliberately asymmetric in cost. One
single-node job evaluates 38 cluster-balanced hard-gate candidates with seed
101, reranks exact 300 K evidence, then evaluates only 10 candidates with seeds
202/303. Only a candidate completing all three seeds may win; the winner enters
three independent long validation trajectories. `minimum 10` and
`require_minimum yes` prevent publication from an undersized confirmation set.

The Fe selection basis is `B/G/E/nu`. Structure, density, angles, surface
energy, Born stability, fit quality, and static drift remain hard gates. Inside
that feasible set, maximum relative `B/G/E/nu` error is primary, RMSE is
secondary, and replicate standard error breaks an otherwise equal dynamic
score. `Cprime/C44` remain visible diagnostics and stability evidence.
The broad seed-101 triage has no replicate SEM and therefore ranks by exact
one-seed error plus the declared cluster/diversity policy. SEM enters only
after the confirmation set has completed all declared seeds; a one-seed row
cannot become the published winner.

This two-type model is an ordered-sublattice LJ surrogate.  Passing the fit
does not by itself establish transferable elemental Fe physics.  The final
report must retain label-swap, surface-termination, defect, and finite-
temperature diagnostics, and should be compared with an EAM/Finnis--Sinclair
baseline when transferability matters.
