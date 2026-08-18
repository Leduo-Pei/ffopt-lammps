# Workflow and accuracy model

## What FFOpt learns

For one material and one force-field topology, the surrogate learns

```text
(free epsilon, sigma, charge coordinates) -> LAMMPS-calculated properties
```

The derived neutral charge and optional physics features are reconstructed
before training. Atom-type identity is encoded by stable parameter columns;
the model is not a transferable graph neural network across unrelated
materials. A new material requires its own data, targets, ranges, and fit.

## Why BO comes first

LAMMPS can finish for parameter sets that are physically noisy, seed-sensitive,
or close to structural failure. Those points are poor labels even when their
process return code is zero. BO searches broad configured bounds and finds
lower-objective, feasible regions. Its stability audit identifies candidates
whose properties repeat under independent velocity seeds.

BO is not merely an initializer for an unconstrained ANN. It establishes the
domain in which the parameter-to-property response is learnable.

## Why focused sampling follows BO

ANN accuracy is local to the sampled domain. Focused sampling uses several
diverse BO centers and several normalized radii, plus an optional global
fraction. Multiple centers reduce dependence on a single local basin;
multiple radii provide both fine gradients and neighborhood coverage. Seed
replicates expose aleatoric simulation variability.

The desired training set is not simply the largest possible CSV. It should
contain successful, repeatable, property-informative parameter vectors near
regions that AL is permitted to explore.

## ANN ensemble

Each ANN member is initialized independently. The ensemble mean estimates a
property; disagreement estimates epistemic uncertainty within the training
domain. Per-property held-out R2, RMSE, MAE, residual plots, and train/test
coverage must be inspected. High R2 alone does not prove that the eventual
low-objective region is sampled densely enough.

Extrapolation remains unsafe. An ensemble can agree while all members are
wrong outside the domain they learned.

## Active learning

AL generates candidates within the learned stable envelope, balances predicted
objective and uncertainty, validates selected points with LAMMPS, and retrains
on the new labels. Improvement is measured using LAMMPS objective values, not
surrogate predictions alone.

If AL stalls, repeating the same acquisition is rarely sufficient. Diagnose:

1. Whether the best region lies at a parameter bound.
2. Whether successful data cover more than one local basin.
3. Whether one property dominates residual error.
4. Whether seed variability is larger than the desired improvement.
5. Whether the fixed parameter subset can physically reach the targets.

## Robust selection after AL

The lowest single-seed objective is not automatically the final force field.
The audit stage selects the best distinct candidates accumulated through AL,
repeats their LAMMPS evaluations with configured independent seeds, and ranks
them by `mean objective + standard deviation`. Finalization then exports the
best robust candidate with every fixed and neutrality-derived parameter
resolved. This separates surrogate-guided proposal from physical acceptance.

## Final acceptance

The final parameter set is re-evaluated by LAMMPS with production protocols
and saved trajectories. Acceptance can require all three independent gates:

```text
weighted relative RMSE <= objective_max
every relative property error <= max_error_percent
every absolute property error <= its tolerance
```

One- and two-node profiles should use identical scientific inputs and random
seeds. Their final tables should agree within normal floating-point and
stochastic simulation variation; node count is a performance setting, not a
different optimization protocol.
