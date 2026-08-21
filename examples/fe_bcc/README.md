# Fe BCC material-workflow example

Copy `ffopt.in` into a project directory, then place the three LAMMPS data
files named by the input under `data/`.  The input is the only scientific file
the user edits; `runs/` is created automatically.

The supplied Fe values are a single known structural-feasible warm start.  They
do not narrow the declared parameter ranges and do not import any old trajectory
or property result; BO, Sample, Audit, elasticity, AL and validation are all run
again under the current campaign provenance.

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

This two-type model is an ordered-sublattice LJ surrogate.  Passing the fit
does not by itself establish transferable elemental Fe physics.  The final
report must retain label-swap, surface-termination, defect, and finite-
temperature diagnostics, and should be compared with an EAM/Finnis--Sinclair
baseline when transferability matters.
