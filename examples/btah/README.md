# BTAH regression workflow

BTAH is the reference system used to verify the general FFOpt-LAMMPS
workflow. The current public examples are:

- `examples/btah/charge_only.in`: fixed epsilon/sigma, 13 independent charges.
- `examples/btah/full.in`: 14 epsilon, 14 sigma, and 13 independent charges.
- `examples/btah/acceptance.in`: reduced-budget warm-start workflow used by
  `ffopt self-test` to verify a machine and the installed release.

The two production examples fit bulk crystal properties and the 98.5 kJ/mol sublimation target.
Adsorption has no experimental target in these examples, so it is calculated
only during final validation and does not affect BO, ANN, or AL.

## Run

```bash
ffopt check examples/btah/charge_only.in
ffopt explain examples/btah/charge_only.in
ffopt run examples/btah/charge_only.in --dry-run
ffopt run examples/btah/charge_only.in
```

Repeating the final command automatically resumes the compatible pipeline.
Use `--new` only when starting an independent campaign.

## Physical evaluation

Bulk follows the validated fixed sequence:

1. Fixed-box 0 K conjugate-gradient minimization.
2. Velocity initialization at the requested temperature.
3. Fully flexible triclinic NPT equilibration.
4. NPT production with averaged cell lengths, angles, density, potential
   energy, and enthalpy.

The current sublimation estimator is

```text
E_sub = E_single,min - <PE_bulk,NPT> / N_molecules
```

It is fitted to the selected 98.5 kJ/mol finite-temperature experimental
reference without translational, rotational, vibrational, or standard-state
corrections. The definition is explicit in the generated run manifest and
must not be silently relabeled as a rigorous thermochemical calculation.

## Parameter modes

The input table always states all initial epsilon, sigma, and charge values.
The line below creates charge-only optimization:

```text
fix epsilon sigma
```

Deleting that line creates the 41-dimensional full optimization. Charge
neutrality removes `bhN1_charge` from the independent space and recovers it at
every evaluation.

## Outputs

Each pipeline stores its SQLite state, generated runtime manifest, BO results,
stability audit, local sampling replicates, ANN model and metrics, AL history,
post-AL audit, robust final parameters, computed properties, trajectories, and
final structures under `runs/<project>/pipelines/<run-id>/`.

Historical campaign analysis is retained in the Git history and the baseline
audit under `docs/`; it is not part of the user-facing configuration surface.
The packaged workflow's actual one/two-node results are in the
[0.3.0a3 acceptance record](../../docs/reference/acceptance-v0.3.0a3.md).
