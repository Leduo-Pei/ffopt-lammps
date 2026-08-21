# Constrained refinement core

`engine.constrained_refinement` is the scheduler-independent core for the
post-BO refinement loop.  It is deliberately a one-round function: a workflow
controller supplies explicit evidence paths, evaluates the returned queues,
and calls the next round with the new cumulative evidence.  The core never
uses `glob`, modification time, or "latest directory" discovery.

## Scientific order

Every measured candidate is handled in the following order:

1. evaluate the exact structural/property constraints with
   `engine.constraint_objectives`;
2. reject structural failures before looking at mechanics;
3. apply the configured mechanical stability and fit-quality gates;
4. rank eligible candidates by maximum relative mechanical error, then by
   relative-error RMSE;
5. label (but do not filter) candidates using the requested reporting
   threshold.

Consequently, the best structurally valid measured candidate is retained even
when no candidate is within a nominal 20% band.  A round with zero eligible
candidates is a valid result, not an exception.

For cubic materials, the optional `derivation: {kind: cubic}` adapter derives
`B`, `Cprime`, `C44`, Born stability, and diagnostic Hill properties from
measured `C11`, `C12`, and `C44` through `engine.cubic_elasticity`.  Other
materials may supply independent objective columns directly.

## Input and output contracts

`run_refinement_round(...)` requires:

- a validated `RefinementSpec` (or its mapping representation);
- one or more explicit structural CSV paths;
- zero or more explicit mechanical CSV paths (an empty first round is valid);
- an explicit output directory;
- optionally, an explicit previous-state path and candidate-pool path.

All evidence rows must contain every configured parameter column.  Rows are
joined with the exact content-addressed `parameter_key`; decimal rounding is
not used.  If multiple files contain the same key, an audited/multi-seed row is
preferred, followed deterministically by the later explicitly listed source.

Each successful round publishes these files transactionally:

- `refinement_state.json`: machine-readable `active`, `converged`, or
  `budget_exhausted` state;
- `exact_structural_mechanical_ranking.csv`: exact eligible best-effort
  ranking;
- `assessed_candidates.csv`: all structural observations and gate diagnostics;
- `mechanical_proposals.csv`: measured structural passes that still need a
  mechanical label;
- `structural_proposals.csv`: unevaluated coordinates selected by the active
  acquisition backend;
- `refinement_report.md`: concise human-readable summary;
- `artifact_manifest.json`: immutable scientific/input/output fingerprint.

`structural_proposals.csv` always contains at least:

```text
candidate_id,selection_role,parameter_key,<configured parameter columns...>
```

An exact retry verifies the manifest and reuses the round.  Changed inputs,
settings, seeds, or outputs fail closed rather than silently mixing runs.

## Acquisition boundary

`StructuralAcquisition` is the plug-in protocol for candidate generation and
selection.  The included `ExplicitPoolAcquisition` only selects from an
explicitly supplied pool.  It can use optional `acquisition_score` (or
`p_structural_feasible`), `structural_entropy`, and `global_distance` columns;
otherwise it falls back to deterministic parameter-space diversity.

This default backend has capability `selection_only`.  It does **not** train a
surrogate or generate fresh coordinates, so patience cannot be reported as
scientific convergence.  The state remains `active` with
`convergence_status: incomplete_capability` until a scientifically capable
backend is used or the budget is exhausted.

`engine.gp_structural_acquisition.GaussianProcessStructuralAcquisition` is the
material-independent low-dimensional probabilistic backend.  It:

- fits one GP to the continuous band ratio for every configured structural
  gate and uses the bottleneck marginal probability as a conservative joint
  feasibility signal;
- fits a separate GP to measured mechanical M-infinity, using only rows whose
  exact structural observation passed all gates (plus configured stability and
  fit-quality gates);
- allocates an explicit-pool batch among local constrained expected
  improvement, local structural-boundary entropy, and a small global coverage
  safeguard, with greedy within-batch diversity;
- writes only columns prefixed as `surrogate_` on proposals and records the
  exact-observed incumbent separately in `acquisition_diagnostics`;
- degrades to coverage when observations are sparse, dimensionality is too
  high, or held-out GP quality is inadequate.

The backend-level capability declaration means that the implementation can
support scientific convergence.  The per-round result is stricter: it is true
only when every structural model and the mechanical minimax model pass their
data/holdout checks, an exact eligible incumbent exists, and exact finite
structural observations bracket both sides of the gate.  This distinction is
persisted in `refinement_state.json`; a fallback round cannot become
scientifically converged merely because patience elapsed.

The GP backend requires the `bo` optional dependency set and receives all
constraint/objective definitions through `AcquisitionContext`; it contains no
material names, `Cij` assumptions, filesystem discovery, or scheduler logic.
Its candidate pool must contain coordinates rather than exact property values.
Job submission, evaluator execution, and multi-round orchestration remain
pipeline responsibilities rather than hidden work in the acquisition layer.

## Pipeline command

There is one module command and no hidden `train` mode. It plans one round
from already available evidence:

```bash
python -m engine.constrained_refinement \
  --config refinement.json \
  --structural-observations runs/evidence/structural.csv \
  --static-observations runs/evidence/static.csv \
  --candidate-pool runs/round_02/candidate_pool.csv \
  --previous-state runs/round_01/refinement_state.json \
  --output-dir runs/round_02 \
  --round 2 \
  --max-rounds 8
```

Repeat the two observation options when evidence remains split across explicit
files. They must collectively describe the cumulative evidence intended for
the round. `--round` is checked against the previous state, and every round
uses a distinct immutable output directory. Evaluating
`structural_proposals.csv` and `mechanical_proposals.csv` is a separate
pipeline action; this command never launches LAMMPS or SLURM and never searches
for the resulting files.
